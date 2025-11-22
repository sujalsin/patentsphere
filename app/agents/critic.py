from __future__ import annotations

import json
import logging
import re
import statistics
from datetime import datetime
from typing import Dict, Any, List, Set

import psycopg
from psycopg.types.json import Jsonb

from app.agents.base import AgentResult, BaseAgent
from app.services import LLMService, LLMServiceError
from app.services.llm import LLMRequest
from config.settings import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a patent analysis quality judge. Rate the coherence, "
    "fluency, and usefulness of patent analysis responses on a scale of 0.0 to 1.0. "
    "Return only a single floating-point number between 0.0 and 1.0."
)

FLUENCY_PROMPT_TEMPLATE = """Rate the coherence and fluency of this patent analysis response on a scale of 0.0 to 1.0.

Response:
{response}

Return ONLY a single number between 0.0 and 1.0 (e.g., 0.85)."""


class CriticAgent(BaseAgent):
    name = "critic"

    def __init__(self, settings=None, weights: Dict[str, float] | None = None) -> None:
        super().__init__(settings)
        self.settings = settings
        self.weights = weights or {
            "citation_overlap": 0.4,
            "cpc_relevance": 0.3,
            "temporal_diversity": 0.2,
            "llm_fluency": 0.1,
        }
        self.llm = LLMService(settings) if settings else None
        # Default max_hops from config.yaml (citations.max_hops: 2)
        self.max_hops = 2
        # Try to get from settings if available (may not be in Settings model yet)
        try:
            if settings and hasattr(settings, "data") and hasattr(settings.data, "citations"):
                self.max_hops = getattr(settings.data.citations, "max_hops", 2)
        except AttributeError:
            pass  # Use default

    async def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        claims_analysis: Dict[str, Any],
        synthesis_output: Dict[str, Any],
    ) -> AgentResult:
        """Compute reward scores for all components."""
        scores = {}

        # Citation overlap (0.4 weight)
        try:
            scores["citation_overlap"] = await self._compute_citation_overlap(retrieved_chunks)
        except Exception as exc:
            logger.warning("Citation overlap calculation failed: %s", exc)
            scores["citation_overlap"] = 0.0

        # CPC relevance (0.3 weight)
        try:
            scores["cpc_relevance"] = self._compute_cpc_relevance(
                claims_analysis, retrieved_chunks
            )
        except Exception as exc:
            logger.warning("CPC relevance calculation failed: %s", exc)
            scores["cpc_relevance"] = 0.0

        # Temporal diversity (0.2 weight)
        try:
            scores["temporal_diversity"] = self._compute_temporal_diversity(retrieved_chunks)
        except Exception as exc:
            logger.warning("Temporal diversity calculation failed: %s", exc)
            scores["temporal_diversity"] = 0.0

        # LLM fluency (0.1 weight)
        try:
            scores["llm_fluency"] = await self._compute_llm_fluency(synthesis_output)
        except Exception as exc:
            logger.warning("LLM fluency calculation failed: %s", exc)
            scores["llm_fluency"] = 0.5  # Default to neutral if LLM fails

        # Compute weighted total
        total = sum(self.weights[k] * scores[k] for k in scores)

        return AgentResult(
            agent=self.name,
            success=True,
            data={"score": total, "components": scores, "weights": self.weights},
        )

    async def _compute_citation_overlap(
        self, retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Compute citation overlap score using 1-hop and 2-hop citation traversal.

        Returns normalized score 0.0-1.0 based on how many retrieved patents
        are connected via citation graph.
        """
        if not retrieved_chunks or not self.settings:
            return 0.0

        # Extract unique patent IDs from retrieved chunks
        retrieved_patent_ids = {
            chunk.get("patent_id") for chunk in retrieved_chunks if chunk.get("patent_id")
        }
        retrieved_patent_ids = {pid for pid in retrieved_patent_ids if pid}

        if not retrieved_patent_ids:
            return 0.0

        # Get database connection
        pg_cfg = getattr(self.settings, "database", None)
        if not pg_cfg:
            return 0.0

        conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"

        try:
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    # Find all patents cited by retrieved patents (1-hop)
                    cur.execute(
                        """
                        SELECT DISTINCT cited_patent_id
                        FROM patent_citations
                        WHERE citing_patent_id = ANY(%s)
                        """,
                        (list(retrieved_patent_ids),),
                    )
                    one_hop_cited = {row[0] for row in cur.fetchall()}

                    # Find all patents that cite retrieved patents (reverse 1-hop)
                    cur.execute(
                        """
                        SELECT DISTINCT citing_patent_id
                        FROM patent_citations
                        WHERE cited_patent_id = ANY(%s)
                        """,
                        (list(retrieved_patent_ids),),
                    )
                    one_hop_citing = {row[0] for row in cur.fetchall()}

                    # Combine 1-hop neighbors
                    one_hop_neighbors = one_hop_cited | one_hop_citing

                    # For 2-hop, find citations of 1-hop neighbors
                    two_hop_neighbors = set()
                    if self.max_hops >= 2 and one_hop_neighbors:
                        cur.execute(
                            """
                            SELECT DISTINCT cited_patent_id
                            FROM patent_citations
                            WHERE citing_patent_id = ANY(%s)
                            """,
                            (list(one_hop_neighbors),),
                        )
                        two_hop_cited = {row[0] for row in cur.fetchall()}

                        cur.execute(
                            """
                            SELECT DISTINCT citing_patent_id
                            FROM patent_citations
                            WHERE cited_patent_id = ANY(%s)
                            """,
                            (list(one_hop_neighbors),),
                        )
                        two_hop_citing = {row[0] for row in cur.fetchall()}

                        two_hop_neighbors = two_hop_cited | two_hop_citing

                    # Compute overlap: how many retrieved patents are in the citation network?
                    all_neighbors = one_hop_neighbors | two_hop_neighbors
                    overlap_count = len(retrieved_patent_ids & all_neighbors)

                    # Also check if retrieved patents cite each other
                    if len(retrieved_patent_ids) > 1:
                        cur.execute(
                            """
                            SELECT COUNT(*)
                            FROM patent_citations
                            WHERE citing_patent_id = ANY(%s)
                            AND cited_patent_id = ANY(%s)
                            """,
                            (list(retrieved_patent_ids), list(retrieved_patent_ids)),
                        )
                        internal_citations = cur.fetchone()[0]
                    else:
                        internal_citations = 0

                    # Score: overlap ratio + internal citation bonus
                    base_score = overlap_count / len(retrieved_patent_ids) if retrieved_patent_ids else 0.0
                    internal_bonus = min(internal_citations / len(retrieved_patent_ids), 0.3) if retrieved_patent_ids else 0.0

                    return min(base_score + internal_bonus, 1.0)

        except Exception as exc:
            logger.error("Database error in citation overlap: %s", exc)
            return 0.0

    def _compute_cpc_relevance(
        self, claims_analysis: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Compute CPC relevance score by comparing query CPC codes with retrieved patent CPC codes.

        Uses hierarchical matching (G06N3/063 matches G06N3/04 at level 2).
        """
        # Extract CPC codes from claims analysis
        query_cpc_codes = []
        if claims_analysis and "cpc_codes" in claims_analysis:
            cpc_list = claims_analysis["cpc_codes"]
            if isinstance(cpc_list, list):
                for cpc_entry in cpc_list:
                    if isinstance(cpc_entry, dict):
                        code = cpc_entry.get("code")
                        if code:
                            query_cpc_codes.append(code)
                    elif isinstance(cpc_entry, str):
                        query_cpc_codes.append(cpc_entry)

        if not query_cpc_codes:
            return 0.5  # Neutral if no CPC codes available

        # Extract CPC codes from retrieved chunks
        retrieved_cpc_sets = []
        for chunk in retrieved_chunks:
            cpc_data = chunk.get("cpc_codes")
            if not cpc_data:
                continue

            # Handle JSONB/string/list formats
            if isinstance(cpc_data, str):
                try:
                    cpc_data = json.loads(cpc_data)
                except json.JSONDecodeError:
                    continue

            if isinstance(cpc_data, list):
                chunk_cpcs = []
                for item in cpc_data:
                    if isinstance(item, dict):
                        code = item.get("code")
                        if code:
                            chunk_cpcs.append(code)
                    elif isinstance(item, str):
                        chunk_cpcs.append(item)
                if chunk_cpcs:
                    retrieved_cpc_sets.append(set(chunk_cpcs))

        if not retrieved_cpc_sets:
            return 0.0

        # Compute hierarchical similarity
        query_cpc_set = set(query_cpc_codes)
        matches = 0
        total_chunks = len(retrieved_cpc_sets)

        for chunk_cpcs in retrieved_cpc_sets:
            # Check for exact matches
            if query_cpc_set & chunk_cpcs:
                matches += 1
                continue

            # Check for hierarchical matches
            for query_cpc in query_cpc_set:
                for chunk_cpc in chunk_cpcs:
                    if self._cpc_hierarchical_match(query_cpc, chunk_cpc):
                        matches += 1
                        break
                else:
                    continue
                break

        return matches / total_chunks if total_chunks > 0 else 0.0

    def _cpc_hierarchical_match(self, cpc1: str, cpc2: str) -> bool:
        """Check if two CPC codes match at any hierarchical level."""
        # Normalize codes (remove spaces, handle different formats)
        cpc1 = cpc1.replace(" ", "").upper()
        cpc2 = cpc2.replace(" ", "").upper()

        # CPC format: G06N3/063 means Section+Class+Subclass (G06N), Main Group (3), Subgroup (063)
        # Split by / to separate main part from subgroup
        parts1 = cpc1.split("/")
        parts2 = cpc2.split("/")

        # Extract section+class+subclass and main group from first part (e.g., "G06N3" -> "G06N" and "3")
        if len(parts1) > 0 and len(parts2) > 0:
            main_part1 = parts1[0]  # e.g., "G06N3"
            main_part2 = parts2[0]  # e.g., "G06N3"
            
            # Extract section+class+subclass (letters+digits+letter) and main group (digits at end)
            import re
            # Pattern: letters, digits, letter, then digits (main group)
            match1 = re.match(r"^([A-Z]\d+[A-Z])(\d+)$", main_part1)
            match2 = re.match(r"^([A-Z]\d+[A-Z])(\d+)$", main_part2)
            
            if match1 and match2:
                section_class_subclass1 = match1.group(1)  # e.g., "G06N"
                main_group1 = match1.group(2)            # e.g., "3"
                section_class_subclass2 = match2.group(1)
                main_group2 = match2.group(2)
                
                # Section+class+subclass must match
                if section_class_subclass1 != section_class_subclass2:
                    return False
                
                # Main group must also match for hierarchical match
                # (e.g., G06N3/063 matches G06N3/04 because both have main group 3)
                if main_group1 == main_group2:
                    return True
                else:
                    return False  # Different main groups don't match

        return False

    def _compute_temporal_diversity(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Compute temporal diversity score based on publication date spread.

        Returns normalized score 0.0-1.0 based on how spread out the dates are.
        """
        if not retrieved_chunks:
            return 0.0

        # Extract publication dates
        years = []
        for chunk in retrieved_chunks:
            date_val = chunk.get("publication_date")
            if not date_val:
                continue

            # Parse date string (YYYY-MM-DD) or integer (YYYYMMDD)
            year = None
            if isinstance(date_val, str):
                # Try to extract year from YYYY-MM-DD format
                parts = date_val.split("-")
                if len(parts) >= 1:
                    try:
                        year = int(parts[0])
                    except ValueError:
                        continue
            elif isinstance(date_val, int):
                # Extract year from YYYYMMDD format
                year_str = str(date_val)
                if len(year_str) >= 4:
                    try:
                        year = int(year_str[:4])
                    except ValueError:
                        continue

            if year:
                years.append(year)

        if not years:
            return 0.0

        if len(years) == 1:
            return 0.0  # No diversity if only one year

        # Compute temporal spread
        min_year = min(years)
        max_year = max(years)
        year_range = max_year - min_year

        if year_range == 0:
            return 0.0

        # Compute standard deviation of years
        try:
            std_dev = statistics.stdev(years)
        except statistics.StatisticsError:
            std_dev = 0.0

        # Normalize: std_dev / range gives diversity measure
        diversity = std_dev / year_range if year_range > 0 else 0.0

        # Cap at 1.0 and ensure minimum score for any spread
        return min(diversity * 2.0, 1.0)  # Scale up for better scores

    async def _compute_llm_fluency(self, synthesis_output: Dict[str, Any]) -> float:
        """
        Compute LLM fluency score by asking LLM to judge response quality.

        Returns normalized score 0.0-1.0.
        """
        if not self.llm:
            return 0.5  # Neutral if LLM unavailable

        # Extract synthesis text
        synthesis_text = synthesis_output.get("executive_summary", "")
        if not synthesis_text:
            # Fallback to action items or notes
            action_items = synthesis_output.get("action_items", [])
            if action_items:
                synthesis_text = json.dumps(action_items, ensure_ascii=False)
            else:
                synthesis_text = str(synthesis_output)

        if not synthesis_text or len(synthesis_text) < 10:
            return 0.3  # Low score for very short responses

        try:
            prompt = FLUENCY_PROMPT_TEMPLATE.format(response=synthesis_text[:2000])  # Limit length

            llm_request = LLMRequest(
                agent="critic",
                user_prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=50,  # Just need a number
            )

            response = await self.llm.generate(llm_request)

            # Try to extract a float from the response
            response = response.strip()
            # Remove any non-numeric characters except decimal point
            cleaned = "".join(c for c in response if c.isdigit() or c == ".")
            if cleaned:
                try:
                    score = float(cleaned)
                    # Clamp to [0.0, 1.0]
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass

            # Fallback: try to find a number in the response
            numbers = re.findall(r"\d+\.?\d*", response)
            if numbers:
                try:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score / 10.0 if score > 1.0 else score))
                except ValueError:
                    pass

            return 0.5  # Default neutral score

        except LLMServiceError:
            logger.warning("LLM service unavailable for fluency scoring")
            return 0.5
        except Exception as exc:
            logger.error("Error computing LLM fluency: %s", exc)
            return 0.5

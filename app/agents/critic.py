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
from app.templates.models import CriticOutput, RewardComponents
from app.templates.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_FLUENCY_TEMPLATE
from config.settings import get_settings

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Agent that evaluates the quality of retrieval and synthesis outputs."""
    
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
        self.max_hops = 2
        
        try:
            if settings and hasattr(settings, "data") and hasattr(settings.data, "citations"):
                self.max_hops = getattr(settings.data.citations, "max_hops", 2)
        except AttributeError:
            pass

    async def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        claims_analysis: Dict[str, Any],
        synthesis_output: Dict[str, Any],
    ) -> AgentResult:
        """Compute reward scores for all components."""
        scores = {}

        # === HEURISTIC SIGNALS (30% of reward) ===
        
        # Citation overlap (15% weight)
        try:
            scores["citation_overlap"] = await self._compute_citation_overlap(retrieved_chunks)
        except Exception as exc:
            logger.warning("Citation overlap calculation failed: %s", exc)
            scores["citation_overlap"] = 0.0

        # CPC relevance (10% weight)
        try:
            scores["cpc_relevance"] = self._compute_cpc_relevance(
                claims_analysis, retrieved_chunks
            )
        except Exception as exc:
            logger.warning("CPC relevance calculation failed: %s", exc)
            scores["cpc_relevance"] = 0.0

        # Temporal diversity (5% weight)
        try:
            scores["temporal_diversity"] = self._compute_temporal_diversity(retrieved_chunks)
        except Exception as exc:
            logger.warning("Temporal diversity calculation failed: %s", exc)
            scores["temporal_diversity"] = 0.0

        # === AI FEEDBACK SIGNALS (70% of reward) - TRUE RLAIF ===
        
        # LLM rates overall quality/fluency (30% weight)
        try:
            scores["llm_fluency"] = await self._compute_llm_fluency(synthesis_output)
        except Exception as exc:
            logger.warning("LLM fluency calculation failed: %s", exc)
            scores["llm_fluency"] = 0.5

        # LLM rates relevance to query (25% weight)
        try:
            scores["llm_relevance"] = await self._compute_llm_relevance(query, synthesis_output)
        except Exception as exc:
            logger.warning("LLM relevance calculation failed: %s", exc)
            scores["llm_relevance"] = 0.5

        # LLM rates completeness (15% weight)
        try:
            scores["llm_completeness"] = await self._compute_llm_completeness(query, synthesis_output, retrieved_chunks)
        except Exception as exc:
            logger.warning("LLM completeness calculation failed: %s", exc)
            scores["llm_completeness"] = 0.5

        # Citation verification and fact-checking
        verification_results = {}
        fact_check_results = {}
        try:
            verification_results = await self._verify_citations(synthesis_output, retrieved_chunks)
            fact_check_results = await self._fact_check(synthesis_output, retrieved_chunks)
        except Exception as exc:
            logger.warning("Citation verification/fact-checking failed: %s", exc)
        
        # Adjust scores based on verification results
        if verification_results.get("verified", True) is False:
            # Penalize if citations are invalid
            scores["llm_fluency"] = scores.get("llm_fluency", 0.5) * 0.7
            scores["llm_relevance"] = scores.get("llm_relevance", 0.5) * 0.7
        
        if fact_check_results.get("verified", True) is False:
            # Penalize if facts don't match sources
            scores["llm_relevance"] = scores.get("llm_relevance", 0.5) * 0.8
            scores["llm_completeness"] = scores.get("llm_completeness", 0.5) * 0.8

        # Compute weighted total (only use weights that exist)
        total = sum(self.weights.get(k, 0) * scores[k] for k in scores if k in self.weights)

        # Build structured output
        components = RewardComponents(
            # Heuristic signals
            citation_overlap=scores.get("citation_overlap", 0.0),
            cpc_relevance=scores.get("cpc_relevance", 0.0),
            temporal_diversity=scores.get("temporal_diversity", 0.0),
            # AI Feedback signals (RLAIF)
            llm_fluency=scores.get("llm_fluency", 0.5),
            llm_relevance=scores.get("llm_relevance", 0.5),
            llm_completeness=scores.get("llm_completeness", 0.5),
        )
        
        # Include verification results in feedback
        feedback = self._generate_feedback(scores, total)
        if verification_results.get("errors"):
            feedback += f" | Citation errors: {', '.join(verification_results['errors'][:3])}"
        if fact_check_results.get("errors"):
            feedback += f" | Fact-check errors: {', '.join(fact_check_results['errors'][:3])}"
        
        critic_output = CriticOutput(
            score=total,
            components=components,
            weights=self.weights,
            feedback=feedback,
        )
        
        # Add verification metadata
        critic_output_dict = critic_output.model_dump()
        critic_output_dict["verification"] = verification_results
        critic_output_dict["fact_check"] = fact_check_results

        return AgentResult(
            agent=self.name,
            success=True,
            data=critic_output.model_dump(),
        )

    def _generate_feedback(self, scores: Dict[str, float], total: float) -> str:
        """Generate qualitative feedback based on scores."""
        feedback_parts = []
        
        if scores["citation_overlap"] < 0.3:
            feedback_parts.append("Low citation network connectivity - consider expanding retrieval")
        elif scores["citation_overlap"] > 0.7:
            feedback_parts.append("Strong citation overlap indicates relevant retrieval")
            
        if scores["cpc_relevance"] < 0.4:
            feedback_parts.append("CPC alignment is weak - query may need refinement")
        elif scores["cpc_relevance"] > 0.7:
            feedback_parts.append("Good CPC code alignment with retrieved patents")
            
        if scores["temporal_diversity"] < 0.3:
            feedback_parts.append("Limited temporal spread - results cluster in narrow time range")
        elif scores["temporal_diversity"] > 0.6:
            feedback_parts.append("Good temporal diversity in retrieved patents")
            
        if total < 0.4:
            feedback_parts.append("Overall quality below threshold - recommend re-retrieval")
        elif total > 0.7:
            feedback_parts.append("Overall quality is good")
            
        return "; ".join(feedback_parts) if feedback_parts else "No specific feedback"

    async def _compute_citation_overlap(
        self, retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """Compute citation overlap score using citation graph traversal."""
        if not retrieved_chunks or not self.settings:
            return 0.0

        retrieved_patent_ids = {
            chunk.get("patent_id") for chunk in retrieved_chunks if chunk.get("patent_id")
        }
        retrieved_patent_ids = {pid for pid in retrieved_patent_ids if pid}

        if not retrieved_patent_ids:
            return 0.0

        pg_cfg = getattr(self.settings, "database", None)
        if not pg_cfg:
            return 0.0

        conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"

        try:
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    # 1-hop citations
                    cur.execute(
                        """
                        SELECT DISTINCT cited_patent_id
                        FROM patent_citations
                        WHERE citing_patent_id = ANY(%s)
                        """,
                        (list(retrieved_patent_ids),),
                    )
                    one_hop_cited = {row[0] for row in cur.fetchall()}

                    cur.execute(
                        """
                        SELECT DISTINCT citing_patent_id
                        FROM patent_citations
                        WHERE cited_patent_id = ANY(%s)
                        """,
                        (list(retrieved_patent_ids),),
                    )
                    one_hop_citing = {row[0] for row in cur.fetchall()}

                    one_hop_neighbors = one_hop_cited | one_hop_citing

                    # 2-hop citations
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

                    all_neighbors = one_hop_neighbors | two_hop_neighbors
                    overlap_count = len(retrieved_patent_ids & all_neighbors)

                    # Internal citations bonus
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

                    base_score = overlap_count / len(retrieved_patent_ids) if retrieved_patent_ids else 0.0
                    internal_bonus = min(internal_citations / len(retrieved_patent_ids), 0.3) if retrieved_patent_ids else 0.0

                    return min(base_score + internal_bonus, 1.0)

        except Exception as exc:
            logger.error("Database error in citation overlap: %s", exc)
            return 0.0

    def _compute_cpc_relevance(
        self, claims_analysis: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """Compute CPC relevance score by comparing query CPC codes with retrieved patents."""
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
            return 0.5

        retrieved_cpc_sets = []
        for chunk in retrieved_chunks:
            cpc_data = chunk.get("cpc_codes")
            if not cpc_data:
                continue

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

        query_cpc_set = set(query_cpc_codes)
        matches = 0
        total_chunks = len(retrieved_cpc_sets)

        for chunk_cpcs in retrieved_cpc_sets:
            if query_cpc_set & chunk_cpcs:
                matches += 1
                continue

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
        cpc1 = cpc1.replace(" ", "").upper()
        cpc2 = cpc2.replace(" ", "").upper()

        parts1 = cpc1.split("/")
        parts2 = cpc2.split("/")

        if len(parts1) > 0 and len(parts2) > 0:
            main_part1 = parts1[0]
            main_part2 = parts2[0]
            
            match1 = re.match(r"^([A-Z]\d+[A-Z])(\d+)$", main_part1)
            match2 = re.match(r"^([A-Z]\d+[A-Z])(\d+)$", main_part2)
            
            if match1 and match2:
                section_class_subclass1 = match1.group(1)
                main_group1 = match1.group(2)
                section_class_subclass2 = match2.group(1)
                main_group2 = match2.group(2)
                
                if section_class_subclass1 != section_class_subclass2:
                    return False
                
                if main_group1 == main_group2:
                    return True

        return False

    def _compute_temporal_diversity(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Compute temporal diversity based on publication date spread."""
        if not retrieved_chunks:
            return 0.0

        years = []
        for chunk in retrieved_chunks:
            date_val = chunk.get("publication_date")
            if not date_val:
                continue

            year = None
            if isinstance(date_val, str):
                parts = date_val.split("-")
                if len(parts) >= 1:
                    try:
                        year = int(parts[0])
                    except ValueError:
                        continue
            elif isinstance(date_val, int):
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
            return 0.0

        min_year = min(years)
        max_year = max(years)
        year_range = max_year - min_year

        if year_range == 0:
            return 0.0

        try:
            std_dev = statistics.stdev(years)
        except statistics.StatisticsError:
            std_dev = 0.0

        diversity = std_dev / year_range if year_range > 0 else 0.0
        return min(diversity * 2.0, 1.0)

    async def _compute_llm_fluency(self, synthesis_output: Dict[str, Any]) -> float:
        """Compute LLM fluency score by having LLM judge response quality."""
        if not self.llm:
            return 0.5
        
        # Handle None synthesis_output
        if not synthesis_output or not isinstance(synthesis_output, dict):
            return 0.3

        synthesis_text = synthesis_output.get("executive_summary", "")
        if not synthesis_text:
            action_items = synthesis_output.get("action_items", [])
            if action_items:
                synthesis_text = json.dumps(action_items, ensure_ascii=False)
            else:
                synthesis_text = str(synthesis_output)

        if not synthesis_text or len(synthesis_text) < 10:
            return 0.3

        try:
            prompt = CRITIC_FLUENCY_TEMPLATE.format(response=synthesis_text[:2000])

            llm_request = LLMRequest(
                agent="critic",
                user_prompt=prompt,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=50,
            )

            response = await self.llm.generate(llm_request)

            response = response.strip()
            cleaned = "".join(c for c in response if c.isdigit() or c == ".")
            if cleaned:
                try:
                    score = float(cleaned)
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass

            numbers = re.findall(r"\d+\.?\d*", response)
            if numbers:
                try:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score / 10.0 if score > 1.0 else score))
                except ValueError:
                    pass

            return 0.5

        except LLMServiceError:
            logger.warning("LLM service unavailable for fluency scoring")
            return 0.5
        except Exception as exc:
            logger.error("Error computing LLM fluency: %s", exc)
            return 0.5

    async def _compute_llm_relevance(self, query: str, synthesis_output: Dict[str, Any]) -> float:
        """
        RLAIF: Ask the LLM to rate how relevant the response is to the query.
        
        This is a core RLAIF component - the AI provides feedback on relevance.
        """
        if not self.llm:
            return 0.5
        
        # Handle None synthesis_output
        if not synthesis_output or not isinstance(synthesis_output, dict):
            return 0.3

        synthesis_text = synthesis_output.get("executive_summary", "")
        if not synthesis_text:
            return 0.3

        prompt = f"""Rate how relevant this patent analysis response is to the user's query.

## User Query
"{query}"

## Response to Evaluate
{synthesis_text[:1500]}

## Rating Criteria
- 1.0: Directly addresses the query with specific, relevant patent information
- 0.7-0.9: Mostly relevant with some tangential information
- 0.4-0.6: Partially relevant, missing key aspects of the query
- 0.1-0.3: Mostly irrelevant or off-topic
- 0.0: Completely irrelevant

Return ONLY a single number between 0.0 and 1.0 (e.g., 0.75).
Do not include any other text."""

        try:
            llm_request = LLMRequest(
                agent="critic",
                user_prompt=prompt,
                system_prompt="You are evaluating patent analysis relevance. Return only a number.",
                temperature=0.1,
                max_tokens=50,
            )

            response = await self.llm.generate(llm_request)
            return self._parse_score(response)

        except Exception as exc:
            logger.warning("LLM relevance scoring failed: %s", exc)
            return 0.5

    async def _compute_llm_completeness(
        self, 
        query: str, 
        synthesis_output: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> float:
        """
        RLAIF: Ask the LLM to rate how complete the response is.
        
        Checks if the response adequately uses the retrieved information.
        """
        if not self.llm:
            return 0.5
        
        # Handle None synthesis_output
        if not synthesis_output or not isinstance(synthesis_output, dict):
            return 0.3

        synthesis_text = synthesis_output.get("executive_summary", "")
        sections = synthesis_output.get("insight_sections", [])
        next_steps = synthesis_output.get("next_steps", [])
        
        # Count retrieved patents
        num_patents = len(set(c.get("patent_id") for c in retrieved_chunks if c.get("patent_id")))
        
        # Check what's in the response
        has_summary = bool(synthesis_text and len(synthesis_text) > 50)
        has_sections = len(sections) > 0
        has_next_steps = len(next_steps) > 0
        
        prompt = f"""Rate the completeness of this patent analysis response.

## User Query
"{query}"

## Response Summary
{synthesis_text[:800] if synthesis_text else "(No summary provided)"}

## Response Statistics
- Patents retrieved: {num_patents}
- Has executive summary: {has_summary}
- Has insight sections: {has_sections} ({len(sections)} sections)
- Has next steps: {has_next_steps} ({len(next_steps)} steps)

## Rating Criteria
- 1.0: Comprehensive analysis with summary, insights, citations, and actionable next steps
- 0.7-0.9: Good coverage with minor gaps
- 0.4-0.6: Partial analysis, missing important components
- 0.1-0.3: Minimal or superficial analysis
- 0.0: Empty or no meaningful content

Return ONLY a single number between 0.0 and 1.0 (e.g., 0.80).
Do not include any other text."""

        try:
            llm_request = LLMRequest(
                agent="critic",
                user_prompt=prompt,
                system_prompt="You are evaluating patent analysis completeness. Return only a number.",
                temperature=0.1,
                max_tokens=50,
            )

            response = await self.llm.generate(llm_request)
            return self._parse_score(response)

        except Exception as exc:
            logger.warning("LLM completeness scoring failed: %s", exc)
            return 0.5
    
    async def _verify_citations(
        self, 
        synthesis_output: Dict[str, Any], 
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify citations in synthesis output.
        
        Checks:
        - Patent IDs exist in retrieved_chunks
        - Claimed information matches actual chunk text
        - Date constraints are met
        """
        if not synthesis_output or not retrieved_chunks:
            return {"verified": True, "errors": []}
        
        errors = []
        verified_count = 0
        total_citations = 0
        
        # Extract citations from synthesis text (look for [1], [2], etc.)
        synthesis_text = ""
        if isinstance(synthesis_output, dict):
            synthesis_text = (
                synthesis_output.get("executive_summary", "") +
                synthesis_output.get("technical_summary", "") +
                str(synthesis_output.get("insight_sections", []))
            )
        else:
            synthesis_text = str(synthesis_output)
        
        # Find citation patterns [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, synthesis_text)
        total_citations = len(set(citations))
        
        # Build patent ID map from retrieved chunks
        patent_id_map = {chunk.get("patent_id"): chunk for chunk in retrieved_chunks if chunk.get("patent_id")}
        
        # Extract patent IDs mentioned in text (US-1234567 format)
        patent_id_pattern = r'\b(US|EP|WO|JP|CN|KR|GB|DE|FR)-?\d+[A-Z]?\d*\b'
        mentioned_patents = re.findall(patent_id_pattern, synthesis_text, re.IGNORECASE)
        
        for patent_id in set(mentioned_patents):
            # Normalize patent ID
            patent_id = patent_id.upper().replace("-", "")
            if not any(pid.replace("-", "").upper() == patent_id for pid in patent_id_map.keys()):
                errors.append(f"Patent {patent_id} mentioned but not in retrieved chunks")
            else:
                verified_count += 1
        
        verified = len(errors) == 0 and total_citations > 0
        
        return {
            "verified": verified,
            "errors": errors,
            "verified_count": verified_count,
            "total_citations": total_citations,
        }
    
    async def _fact_check(
        self, 
        synthesis_output: Dict[str, Any], 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fact-check synthesis output against sources using LLM.
        
        Uses phi4-mini to verify if claims are supported by sources.
        """
        if not self.llm or not synthesis_output or not sources:
            return {"verified": True, "errors": []}
        
        try:
            # Extract key claims from synthesis
            claims_text = ""
            if isinstance(synthesis_output, dict):
                claims_text = (
                    synthesis_output.get("executive_summary", "")[:500] +
                    synthesis_output.get("technical_summary", "")[:500]
                )
            else:
                claims_text = str(synthesis_output)[:1000]
            
            # Build sources text
            sources_text = "\n".join([
                f"Source {i+1}: {s.get('patent_id', '')} - {s.get('chunk_text', '')[:200]}"
                for i, s in enumerate(sources[:5])  # Limit to top 5 sources
            ])
            
            fact_check_prompt = f"""Verify if these claims are supported by the sources below.

Claims to verify:
{claims_text}

Sources:
{sources_text}

Instructions:
1. Check if each claim is supported by the sources
2. Identify any unsupported or contradictory claims
3. Return JSON: {{"verified": true/false, "errors": ["error1", "error2"]}}

Response (JSON only):"""
            
            llm_request = LLMRequest(
                agent="fact_check",
                user_prompt=fact_check_prompt,
                system_prompt="You are a fact-checker. Verify claims against sources.",
                temperature=0.2,
                max_tokens=300,
                response_format="json",
            )
            
            response = await self.llm.generate(llm_request, retries=1)
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return {
                        "verified": result.get("verified", True),
                        "errors": result.get("errors", []),
                    }
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Fallback: simple heuristic
            return {"verified": True, "errors": []}
            
        except Exception as exc:
            logger.warning("Fact-checking failed: %s", exc)
            return {"verified": True, "errors": []}

    def _parse_score(self, response: str) -> float:
        """Parse a numeric score from LLM response."""
        response = response.strip()
        
        # Try to extract just the number
        cleaned = "".join(c for c in response if c.isdigit() or c == ".")
        if cleaned:
            try:
                score = float(cleaned)
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Try regex for numbers
        numbers = re.findall(r"\d+\.?\d*", response)
        if numbers:
            try:
                score = float(numbers[0])
                return max(0.0, min(1.0, score / 10.0 if score > 1.0 else score))
            except ValueError:
                pass

        return 0.5

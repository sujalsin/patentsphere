from __future__ import annotations

import json
import logging
import textwrap
import time
from typing import Any, Dict, List

from app.agents.base import AgentResult, BaseAgent
from app.services import LLMService, LLMServiceError
from app.services.llm import LLMRequest, LLMValidationError
from app.templates.models import (
    SynthesisOutput,
    InsightSection,
    BulletPoint,
    NextStep,
    CitationReference,
)
from app.templates.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    get_synthesis_prompt,
    get_synthesis_prompt_with_sources,
)

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """Agent that synthesizes outputs from other agents into executive briefings."""
    
    name = "synthesis"

    def __init__(self, settings=None, llm_service: LLMService | None = None):
        super().__init__(settings)
        self._context: Dict[str, Any] = {}
        self._source_index: str = ""  # Pre-built source index for [1], [2] citations
        self.settings = settings
        self.llm = llm_service or LLMService(settings)

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the context from other agents' outputs."""
        self._context = context
        # Build source index for citations
        self._source_index = self._build_source_index(context)
    
    def _build_source_index(self, context: Dict[str, Any]) -> str:
        """Build a source index string for the synthesis prompt."""
        lines = []
        index = 1
        
        # Index patents from citation results
        citation_data = context.get("citation_mapper", {})
        results = citation_data.get("results", [])
        seen_patents = set()
        
        for result in results[:15]:  # Limit to top 15 patents
            patent_id = result.get("patent_id", "")
            if not patent_id or patent_id in seen_patents:
                continue
            seen_patents.add(patent_id)
            
            assignee = result.get("assignee", "Unknown")
            chunk_type = result.get("chunk_type", "")
            score = result.get("score", 0)
            snippet = result.get("chunk_text", "")[:150]
            
            lines.append(
                f"[{index}] {patent_id} ({assignee}) - {chunk_type} - Score: {score:.2f}"
            )
            if snippet:
                lines.append(f"    Snippet: \"{snippet}...\"")
            index += 1
        
        # Index litigation cases
        litigation_data = context.get("litigation_scout") or {}
        cases = litigation_data.get("cases", litigation_data.get("results", []))
        if not isinstance(cases, list):
            cases = []

        for case in cases[:5]:  # Limit to top 5 cases
            case_id = case.get("case_id", case.get("case_number", ""))
            case_name = case.get("case_name", case.get("title", ""))
            outcome = case.get("outcome", "")
            
            if case_name:
                lines.append(
                    f"[{index}] {case_name} (Litigation) - {outcome if outcome else 'Pending'}"
                )
                index += 1
        
        return "\n".join(lines) if lines else "No sources available"

    async def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        success = True
        error_message: str | None = None
        used_fallback = False

        try:
            # Prepare the prompt with all agent data
            prompt_payload = self._prepare_prompt(query)
            
            llm_request = LLMRequest(
                agent=self.name,
                user_prompt=prompt_payload,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.4,
                max_tokens=1800,
                response_format="json",
            )
            
            # Use structured generation with validation
            synthesis, used_fallback = await self.llm.generate_with_fallback(
                request=llm_request,
                output_model=SynthesisOutput,
                fallback_factory=lambda: self._fallback_package(query),
                retries=2,
            )
            
            if used_fallback:
                success = False
                error_message = "LLM generation failed, used heuristic fallback"
                
        except (LLMServiceError, LLMValidationError) as exc:
            logger.warning("SynthesisAgent fallback triggered: %s", exc)
            synthesis = self._fallback_package(query)
            used_fallback = True
            success = False
            error_message = str(exc)
        except Exception as exc:
            logger.exception("SynthesisAgent unexpected failure: %s", exc)
            synthesis = self._fallback_package(query)
            used_fallback = True
            success = False
            error_message = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000
        
        # Convert Pydantic model to dict
        data = synthesis.model_dump() if hasattr(synthesis, 'model_dump') else synthesis.__dict__
        data.update({
            "latency_ms": latency_ms,
            "source": "llm" if not used_fallback else "heuristic",
            "used_fallback": used_fallback,
        })
        
        return AgentResult(agent=self.name, success=success, data=data, error=error_message)
    
    def _join_patents_with_litigation(
        self, 
        retrieved_chunks: List[Dict[str, Any]], 
        litigation_data: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """
        Perform in-memory join: match retrieved patent IDs with litigation case details.
        
        Returns a dict with matched_cases and total_matches.
        """
        if not litigation_data:
            return {"matched_cases": [], "total_matches": 0}
        
        # Extract patent IDs from retrieved chunks
        patent_ids = {chunk.get("patent_id") for chunk in retrieved_chunks if chunk.get("patent_id")}
        
        if not patent_ids:
            return {"matched_cases": [], "total_matches": 0}
        
        # Get case details from litigation data
        case_details = litigation_data.get("case_details", [])
        if not case_details:
            return {"matched_cases": [], "total_matches": 0}
        
        # Perform join: find cases where patent_id matches
        matched_cases = []
        for case in case_details:
            case_patent_id = case.get("patent_id")
            if case_patent_id and case_patent_id in patent_ids:
                matched_cases.append({
                    "patent_id": case_patent_id,
                    "case_number": case.get("case_number", ""),
                    "case_name": case.get("case_name", case.get("title", "")),
                    "outcome": case.get("outcome", "Pending"),
                    "filing_date": case.get("filing_date", ""),
                    "plaintiff_name": case.get("plaintiff_name", ""),
                    "defendant_name": case.get("defendant_name", ""),
                    "risk_level": "high",  # If patent appears in litigation, it's high risk
                })
        
        return {
            "matched_cases": matched_cases,
            "total_matches": len(matched_cases),
        }

    def _prepare_prompt(self, query: str) -> str:
        """Prepare the synthesis prompt with all agent context."""
        claims_data = self._context.get("claims_analyzer", {})
        citation_data = self._context.get("citation_mapper", {})
        litigation_data = self._context.get("litigation_scout", {})
        intent_explanation = self._summarize_intent(query)
        
        # Perform in-memory join: match patents with litigation cases
        retrieved_chunks = citation_data.get("results", [])
        if not retrieved_chunks:
            # Try alternative path (from adaptive_retrieval)
            retrieved_chunks = self._context.get("retrieved_chunks", [])
        
        join_results = self._join_patents_with_litigation(retrieved_chunks, litigation_data)
        
        # Add join results to litigation_data for prompt
        if litigation_data:
            litigation_data = litigation_data.copy()
            litigation_data["matched_cases"] = join_results.get("matched_cases", [])
            litigation_data["total_matches"] = join_results.get("total_matches", 0)
        
        # Use source-indexed prompt if we have sources
        if self._source_index and self._source_index != "No sources available":
            return get_synthesis_prompt_with_sources(
                query=query,
                intent_explanation=intent_explanation,
                claims_data=claims_data,
                citation_data=citation_data,
                litigation_data=litigation_data,
                source_index=self._source_index,
            )
        
        return get_synthesis_prompt(
            query=query,
            intent_explanation=intent_explanation,
            claims_data=claims_data,
            citation_data=citation_data,
            litigation_data=litigation_data,
        )

    def _fallback_package(self, query: str) -> SynthesisOutput:
        """Generate a fallback synthesis when LLM is unavailable."""
        claims = self._context.get("claims_analyzer") or {}
        if isinstance(claims, dict):
            summary = claims.get("summary") or f"Unable to synthesize LLM insight for query: {query}"
        else:
            summary = f"Unable to synthesize LLM insight for query: {query}"

        insight_sections = [
            InsightSection(
                title="Key Signals (Fallback)",
                bullets=[
                    BulletPoint(
                        headline="Claims analyzer summary",
                        details=[summary],
                        citations=[],
                    )
                ],
            )
        ]
        
        next_steps = [
            NextStep(
                priority="medium",
                recommendation="Review claims analysis output manually.",
                rationale="Synthesis fallback triggered; ensure accuracy before acting.",
            )
        ]

        return SynthesisOutput(
            executive_summary=summary,
            insight_sections=insight_sections,
            next_steps=next_steps,
            citations=[],
            risk_score=55,
        )

    def _summarize_intent(self, query: str) -> str:
        """Generate intent explanation based on claims analysis."""
        claims = self._context.get("claims_analyzer") or {}
        detected_type = (claims.get("query_type") if isinstance(claims, dict) else "other") or "other"
        detected_type = detected_type.lower()

        label_map = {
            "emergence": "emerging-tech scouting and blindspot detection",
            "litigation": "litigation risk or defensive positioning",
            "portfolio": "portfolio fit / acquisition targeting",
            "research": "general research or landscaping",
        }
        intent_label = label_map.get(detected_type, "mixed intent research")

        ql = query.lower()
        if any(word in ql for word in ("litigation", "lawsuit", "infringement")):
            intent_label = "litigation risk intelligence"
        elif any(word in ql for word in ("acquisition", "m&a", "takeover")):
            intent_label = "strategic investment / acquisition targeting"
        elif any(word in ql for word in ("funding", "startup", "venture")):
            intent_label = "competitive startup scouting"

        return textwrap.dedent(
            f"""Intent: {intent_label}.
Relevant cues from query: {query[:300]}"""
        ).strip()

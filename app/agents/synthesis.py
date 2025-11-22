from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from app.agents.base import AgentResult, BaseAgent
from app.services import LLMService, LLMServiceError
from app.services.llm import LLMRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are PatentSphere's senior patent analyst. Combine agent outputs "
    "(claims, citations, litigation) into a crisp recommendation with explicit citations."
)

PROMPT_TEMPLATE = """You are given structured agent outputs plus the user's query.

<query>
{query}
</query>

<claims>
{claims}
</claims>

<citations>
{citations}
</citations>

<litigation>
{litigation}
</litigation>

Produce a JSON object with:
{{
  "executive_summary": "3-4 sentences focusing on risk, novelty, key findings.",
  "action_items": [
    {{"priority": "high|medium|low", "recommendation": "string", "rationale": "string"}}
  ],
  "citations": [
    {{"patent_id": "string", "reason": "string"}}
  ],
  "risk_score": 0-100 integer (litigation risk or uncertainty),
  "notes": ["string" ...]
}}

If data is missing, call it out explicitly. Cite patents by ID in the text.
Return ONLY the JSON."""


@dataclass
class SynthesisPackage:
    executive_summary: str
    action_items: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    risk_score: int
    notes: List[str]
    used_fallback: bool = False


class SynthesisAgent(BaseAgent):
    name = "synthesis"

    def __init__(self, settings=None, llm_service: LLMService | None = None):
        super().__init__(settings)
        self._context: Dict[str, Any] = {}
        self.settings = settings
        self.llm = llm_service or LLMService(settings)

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context = context

    async def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        payload = self._prepare_payload(query)
        synthesis: SynthesisPackage
        success = True
        error_message: str | None = None

        try:
            llm_request = LLMRequest(
                agent=self.name,
                user_prompt=payload,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.4,
                max_tokens=1500,
            )
            raw = await self.llm.generate(llm_request)
            synthesis = self._parse_response(raw, fallback_query=query)
        except (LLMServiceError, json.JSONDecodeError) as exc:
            logger.warning("SynthesisAgent fallback triggered: %s", exc)
            synthesis = self._fallback_package(query)
            success = False
            error_message = str(exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("SynthesisAgent unexpected failure: %s", exc)
            synthesis = self._fallback_package(query)
            success = False
            error_message = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000
        data = {
            **synthesis.__dict__,
            "latency_ms": latency_ms,
            "source": "llm" if not synthesis.used_fallback else "heuristic",
        }
        return AgentResult(agent=self.name, success=success, data=data, error=error_message)

    def _prepare_payload(self, query: str) -> str:
        claims = json.dumps(self._context.get("claims_analyzer", {}), ensure_ascii=False)
        citations = json.dumps(self._context.get("citation_mapper", {}), ensure_ascii=False)
        litigation = json.dumps(self._context.get("litigation_scout", {}), ensure_ascii=False)
        return PROMPT_TEMPLATE.format(
            query=query,
            claims=claims,
            citations=citations,
            litigation=litigation,
        )

    def _parse_response(self, raw: str, fallback_query: str) -> SynthesisPackage:
        data = self._coerce_json(raw)
        action_items = data.get("action_items") or []
        citations = data.get("citations") or []
        notes = data.get("notes") or []
        risk = int(data.get("risk_score") or 50)
        risk = max(0, min(100, risk))
        summary = data.get("executive_summary") or f"Summary unavailable for query: {fallback_query}"

        return SynthesisPackage(
            executive_summary=summary.strip(),
            action_items=action_items,
            citations=citations,
            risk_score=risk,
            notes=notes,
        )

    def _coerce_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _fallback_package(self, query: str) -> SynthesisPackage:
        claims = self._context.get("claims_analyzer", {})
        summary = claims.get("summary") or f"Unable to synthesize LLM insight for query: {query}"
        cpc = claims.get("cpc_codes", [])
        notes = ["LLM synthesis unavailable; showing claims summary only."]
        if not cpc:
            notes.append("No CPC predictions available.")

        return SynthesisPackage(
            executive_summary=summary,
            action_items=[
                {
                    "priority": "medium",
                    "recommendation": "Review claims analysis output manually.",
                    "rationale": "Fallback triggered; ensure accuracy before acting.",
                }
            ],
            citations=[],
            risk_score=55,
            notes=notes,
            used_fallback=True,
        )



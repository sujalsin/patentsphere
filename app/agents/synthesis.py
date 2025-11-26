from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from app.agents.base import AgentResult, BaseAgent
from app.services import LLMService, LLMServiceError
from app.services.llm import LLMRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are PatentSphere's senior patent analyst. Combine agent outputs "
    "(claims, citations, litigation) into an executive-ready briefing. "
    "Vary section titles and emphasis based on the user's intent."
)

PROMPT_TEMPLATE = """You are given structured agent outputs plus the user's query.

# User query
{query}

# Detected intent and cues
{intent_explanation}

# Claims analyzer data
{claims}

# Citation mapper / retrieval data
{citations}

# Litigation scout data
{litigation}

Produce JSON with this structure:
{{
  "executive_summary": "2-4 sentences synthesizing the most important insight. Cite patents inline (US123...).",
  "insight_sections": [
    {{
      "title": "Dynamic section title chosen to match the query focus",
      "bullets": [
        {{
          "headline": "Entity or signal with concise descriptor",
          "details": ["Short bullet strings with funding, patent stats, citation velocity, IBM overlap, etc."],
          "citations": ["US1234567A", "WO2023123456"]
        }}
      ]
    }}
  ],
  "next_steps": [
    {{"priority": "high|medium|low", "recommendation": "Actionable step", "rationale": "Why now"}}
  ],
  "citations": [
    {{"patent_id": "string", "reason": "Why referenced"}}
  ],
  "risk_score": 0-100 integer (overall threat/uncertainty score)
}}

Guidelines:
- Choose section titles that suit the query intent (e.g., Emerging Threats, Litigation Blindspots, Portfolio Fit, Strategic Partnerships). Invent new titles when needed.
- Use the provided agent data verbatim when available. If a metric is missing, state that the data was not supplied—do not hallucinate.
- Prefer 2-3 sections with 1-3 bullets each. Each bullet should have 1-3 detail strings.
- Mention IBM overlap or lack thereof when deducible from the data.
- ALWAYS return valid JSON. Do not include commentary outside the JSON object."""


@dataclass
class SynthesisPackage:
    executive_summary: str
    insight_sections: List[Dict[str, Any]]
    next_steps: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    risk_score: int
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
                max_tokens=1800,
                response_format="json",
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
        intent_explanation = self._summarize_intent(query)
        return PROMPT_TEMPLATE.format(
            query=query,
            intent_explanation=intent_explanation,
            claims=claims,
            citations=citations,
            litigation=litigation,
        )

    def _parse_response(self, raw: str, fallback_query: str) -> SynthesisPackage:
        data = self._coerce_json(raw)
        insight_sections = data.get("insight_sections") or []
        next_steps = data.get("next_steps") or data.get("action_items") or []
        citations = data.get("citations") or []
        risk = int(data.get("risk_score") or 50)
        risk = max(0, min(100, risk))
        summary = data.get("executive_summary") or f"Summary unavailable for query: {fallback_query}"

        return SynthesisPackage(
            executive_summary=summary.strip(),
            insight_sections=insight_sections,
            next_steps=next_steps,
            citations=citations,
            risk_score=risk,
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

        insight_sections = [
            {
                "title": "Key Signals (Fallback)",
                "bullets": [
                    {
                        "headline": "Claims analyzer summary",
                        "details": [summary],
                        "citations": [],
                    }
                ],
            }
        ]
        next_steps = [
            {
                "priority": "medium",
                "recommendation": "Review claims analysis output manually.",
                "rationale": "Synthesis fallback triggered; ensure accuracy before acting.",
            }
        ]

        return SynthesisPackage(
            executive_summary=summary,
            insight_sections=insight_sections,
            next_steps=next_steps,
            citations=[],
            risk_score=55,
            used_fallback=True,
        )

    # --------------------------------------------------------------------- helpers

    def _summarize_intent(self, query: str) -> str:
        claims = self._context.get("claims_analyzer", {})
        detected_type = (claims.get("query_type") or "other").lower()

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

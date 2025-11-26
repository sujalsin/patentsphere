from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psycopg
from psycopg.types.json import Jsonb

from app.agents.base import AgentResult, BaseAgent
from app.services import LLMService, LLMServiceError
from app.services.llm import LLMRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are PatentSphere's Claims Analyzer. You translate product specs or "
    "business questions into CPC classifications and concise technical features. "
    "Return well grounded analyses strictly as JSON."
)

PROMPT_TEMPLATE = """Analyze the following patent or product query and extract:
- 2-4 critical technical features with short evidence notes
- Likely CPC codes with confidence (0-1) and justification
- Overall query intent category: one of ["emergence","litigation","portfolio","research","other"]
- Short summary + explicit assumptions

Return ONLY valid JSON with this schema:
{{
  "summary": "string",
  "query_type": "string",
  "features": [
    {{"name": "string", "insight": "string", "evidence": "string"}}
  ],
  "cpc_codes": [
    {{"code": "string", "title": "string", "confidence": 0.0, "justification": "string"}}
  ],
  "assumptions": ["string"],
  "confidence": 0.0
}}

Query:
\"\"\"{query}\"\"\""""


@dataclass
class ClaimsAnalysis:
    summary: str
    query_type: str
    features: List[Dict[str, Any]]
    cpc_codes: List[Dict[str, Any]]
    assumptions: List[str]
    confidence: float
    used_fallback: bool = False


class ClaimsAnalyzerAgent(BaseAgent):
    name = "claims_analyzer"

    def __init__(self, settings=None, llm_service: LLMService | None = None):
        super().__init__(settings)
        self.settings = settings
        self.llm = llm_service or LLMService(settings)
        self.agent_cfg = settings.claims_analyzer if settings else None
        self.bad_json_log_path = Path("logs/claims_analyzer_bad_json.log")

    async def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        query_text = query.strip()
        analysis: ClaimsAnalysis
        success = True
        error_message: str | None = None

        try:
            llm_request = LLMRequest(
                agent=self.name,
                user_prompt=PROMPT_TEMPLATE.format(query=query_text),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=768,  # Slightly higher to avoid truncation
                response_format="json",
            )
            # Add timeout wrapper to fail fast if LLM hangs
            import asyncio
            raw_response = await asyncio.wait_for(
                self.llm.generate(llm_request, retries=2),  # Reduced retries
                timeout=120  # 2 minute timeout for LLM call
            )
            analysis = await self._parse_or_repair_response(
                raw_response, fallback_query=query_text
            )
        except asyncio.TimeoutError:
            logger.warning("ClaimsAnalyzer LLM call timed out, using fallback")
            analysis = self._fallback_analysis(query_text)
            success = False
            error_message = "LLM call timed out after 120s"
        except (LLMServiceError, json.JSONDecodeError) as exc:
            logger.warning("ClaimsAnalyzer LLM fallback activated: %s", exc)
            analysis = self._fallback_analysis(query_text)
            success = False
            error_message = str(exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("ClaimsAnalyzer unexpected failure: %s", exc)
            analysis = self._fallback_analysis(query_text)
            success = False
            error_message = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000
        result_payload = asdict(analysis)
        result_payload.update(
            {
                "latency_ms": latency_ms,
                "source": "llm" if not analysis.used_fallback else "heuristic",
            }
        )

        self._persist_analysis(query_text, analysis, latency_ms)

        return AgentResult(
            agent=self.name,
            success=success,
            data=result_payload,
            error=error_message,
        )

    async def _parse_or_repair_response(
        self, response_text: str, fallback_query: str
    ) -> ClaimsAnalysis:
        try:
            return self._parse_response(response_text, fallback_query)
        except json.JSONDecodeError as exc:
            self._log_bad_response(response_text, exc, stage="initial")
            repaired = await self._repair_json_via_llm(response_text, fallback_query)
            if repaired:
                try:
                    return self._parse_response(repaired, fallback_query)
                except json.JSONDecodeError as repair_exc:
                    self._log_bad_response(repaired, repair_exc, stage="repair")
            raise

    def _parse_response(self, response_text: str, fallback_query: str) -> ClaimsAnalysis:
        data = self._coerce_json(response_text)
        features = self._normalize_list_of_dicts(data.get("features"), ["name", "insight"])
        cpc_codes = self._normalize_list_of_dicts(data.get("cpc_codes"), ["code"])

        summary = data.get("summary") or f"Patent-focused paraphrase of: {fallback_query[:120]}"
        query_type = (data.get("query_type") or "other").lower()
        assumptions = data.get("assumptions") or []
        confidence = float(data.get("confidence") or 0.6)
        confidence = min(max(confidence, 0.0), 1.0)

        return ClaimsAnalysis(
            summary=summary.strip(),
            query_type=query_type,
            features=features,
            cpc_codes=cpc_codes,
            assumptions=assumptions,
            confidence=confidence,
        )

    def _coerce_json(self, text: str) -> Dict[str, Any]:
        """Attempt to coerce arbitrary model output into JSON."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    async def _repair_json_via_llm(
        self, response_text: str, fallback_query: str
    ) -> str | None:
        """Ask the LLM to repair malformed JSON output."""
        repair_prompt = (
            "The previous response was supposed to be valid JSON following this schema:\n"
            '{\n'
            '  "summary": "string",\n'
            '  "query_type": "string",\n'
            '  "features": [{"name": "string", "insight": "string", "evidence": "string"}],\n'
            '  "cpc_codes": [{"code": "string", "title": "string", "confidence": 0-1, "justification": "string"}],\n'
            '  "assumptions": ["string"],\n'
            '  "confidence": 0-1\n'
            '}\n\n'
            "Please fix the JSON for the query below. Return ONLY valid JSON with the same information.\n"
            f"Query: {fallback_query}\n"
            "Original malformed response:\n"
            f"{response_text}\n"
        )

        llm_request = LLMRequest(
            agent=f"{self.name}_json_repair",
            user_prompt=repair_prompt,
            system_prompt="You convert malformed JSON into valid JSON without adding commentary.",
            temperature=0.0,
            max_tokens=512,
            response_format="json",
        )

        try:
            import asyncio

            repaired = await asyncio.wait_for(
                self.llm.generate(llm_request, retries=1),
                timeout=45,
            )
            return repaired.strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ClaimsAnalyzer JSON repair failed: %s", exc)
            return None

    def _log_bad_response(
        self, response_text: str, exc: Exception, stage: str = "initial"
    ) -> None:
        """Persist malformed responses for offline inspection."""
        try:
            self.bad_json_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.bad_json_log_path.open("a", encoding="utf-8") as log_file:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "stage": stage,
                    "error": str(exc),
                    "response_preview": response_text.strip(),
                }
                log_file.write(json.dumps(log_entry) + "\n")
        except Exception as log_exc:  # pylint: disable=broad-except
            logger.debug("Failed to log malformed JSON response: %s", log_exc)

    def _normalize_list_of_dicts(
        self, value: Any, required_keys: List[str]
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, dict) and all(k in entry for k in required_keys):
                normalized.append(entry)
        return normalized

    def _fallback_analysis(self, query: str) -> ClaimsAnalysis:
        tokens = [token.strip(",.") for token in query.split()[:5]]
        features = [
            {
                "name": "keyword_projection",
                "insight": "Heuristic keywords extracted due to LLM fallback.",
                "evidence": ", ".join(tokens),
            }
        ]
        cpc_codes = [
            {
                "code": "G06F17/30",
                "title": "Digital computing or data processing equipment",
                "confidence": 0.35,
                "justification": "Default fallback CPC for software-heavy queries.",
            }
        ]
        return ClaimsAnalysis(
            summary=f"Heuristic summary for query: {query[:160]}",
            query_type="other",
            features=features,
            cpc_codes=cpc_codes,
            assumptions=["LLM unavailable, used keyword fallback."],
            confidence=0.35,
            used_fallback=True,
        )

    def _persist_analysis(
        self, query: str, analysis: ClaimsAnalysis, latency_ms: float
    ) -> None:
        if not self.settings:
            return
        pg_cfg = getattr(self.settings, "database", None)
        if not pg_cfg:
            return

        conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
        try:
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS query_claims_analysis (
                            id SERIAL PRIMARY KEY,
                            query_text TEXT NOT NULL,
                            summary TEXT,
                            query_type TEXT,
                            cpc_codes JSONB,
                            features JSONB,
                            assumptions JSONB,
                            confidence DOUBLE PRECISION,
                            used_fallback BOOLEAN DEFAULT FALSE,
                            latency_ms DOUBLE PRECISION,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        """
                        INSERT INTO query_claims_analysis (
                            query_text, summary, query_type, cpc_codes, features,
                            assumptions, confidence, used_fallback, latency_ms
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query,
                            analysis.summary,
                            analysis.query_type,
                            Jsonb(analysis.cpc_codes),
                            Jsonb(analysis.features),
                            Jsonb(analysis.assumptions),
                            analysis.confidence,
                            analysis.used_fallback,
                            latency_ms,
                        ),
                    )
                conn.commit()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to persist claims analysis: %s", exc)


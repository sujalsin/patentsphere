from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psycopg
from psycopg.types.json import Jsonb

from app.agents.base import AgentResult, BaseAgent
from app.services import LLMService, LLMServiceError
from app.services.llm import LLMRequest, LLMValidationError
from app.templates.models import ClaimsOutput, FeatureItem, CPCCode, RiskEntities, MetadataFilters
from app.templates.prompts import CLAIMS_SYSTEM_PROMPT, get_claims_prompt

logger = logging.getLogger(__name__)


class ClaimsAnalyzerAgent(BaseAgent):
    """Agent that analyzes queries to extract patent-relevant features and CPC codes."""
    
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
        success = True
        error_message: str | None = None
        used_fallback = False

        try:
            llm_request = LLMRequest(
                agent=self.name,
                user_prompt=get_claims_prompt(query_text),
                system_prompt=CLAIMS_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=768,
                response_format="json",
            )
            
            # Use structured generation with validation
            analysis, used_fallback = await self.llm.generate_with_fallback(
                request=llm_request,
                output_model=ClaimsOutput,
                fallback_factory=lambda: self._fallback_analysis(query_text),
                retries=2,
            )
            
            if used_fallback:
                success = False
                error_message = "LLM generation failed, used heuristic fallback"
                
        except asyncio.TimeoutError:
            logger.warning("ClaimsAnalyzer LLM call timed out, using fallback")
            analysis = self._fallback_analysis(query_text)
            used_fallback = True
            success = False
            error_message = "LLM call timed out after 120s"
        except (LLMServiceError, LLMValidationError) as exc:
            logger.warning("ClaimsAnalyzer LLM fallback activated: %s", exc)
            analysis = self._fallback_analysis(query_text)
            used_fallback = True
            success = False
            error_message = str(exc)
        except Exception as exc:
            logger.exception("ClaimsAnalyzer unexpected failure: %s", exc)
            analysis = self._fallback_analysis(query_text)
            used_fallback = True
            success = False
            error_message = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000
        
        # Convert Pydantic model to dict for result payload
        result_payload = analysis.model_dump() if hasattr(analysis, 'model_dump') else asdict(analysis)
        result_payload.update({
            "latency_ms": latency_ms,
            "source": "llm" if not used_fallback else "heuristic",
            "used_fallback": used_fallback,
        })

        self._persist_analysis(query_text, analysis, latency_ms, used_fallback)

        return AgentResult(
            agent=self.name,
            success=success,
            data=result_payload,
            error=error_message,
        )

    def _fallback_analysis(self, query: str) -> ClaimsOutput:
        """Generate a heuristic fallback analysis when LLM is unavailable."""
        tokens = [token.strip(",.") for token in query.split()[:5]]
        
        features = [
            FeatureItem(
                name="keyword_projection",
                insight="Heuristic keywords extracted due to LLM fallback.",
                evidence=", ".join(tokens),
            )
        ]
        
        cpc_codes = [
            CPCCode(
                code="G06F17/30",
                title="Digital computing or data processing equipment",
                confidence=0.35,
                justification="Default fallback CPC for software-heavy queries.",
            )
        ]
        
        # Extract basic risk entities from query (simple heuristic)
        risk_entities = RiskEntities(
            assignees=[],  # Would need NER to extract properly
            topics=tokens[:3],  # Use first few tokens as topics
            search_query=query[:200],  # Use query as-is
        )
        
        metadata_filters = MetadataFilters(
            date_range=None,  # No date inference in fallback
            cpc_codes=[c.code for c in cpc_codes],
        )
        
        return ClaimsOutput(
            summary=f"Heuristic summary for query: {query[:160]}",
            query_type="other",
            features=features,
            cpc_codes=cpc_codes,
            assumptions=["LLM unavailable, used keyword fallback."],
            confidence=0.35,
            risk_entities=risk_entities,
            metadata_filters=metadata_filters,
        )

    def _persist_analysis(
        self, query: str, analysis: ClaimsOutput, latency_ms: float, used_fallback: bool
    ) -> None:
        """Persist analysis to database for telemetry."""
        if not self.settings:
            return
        pg_cfg = getattr(self.settings, "database", None)
        if not pg_cfg:
            return

        conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
        
        # Convert model to dicts for JSONB storage
        cpc_codes_data = [c.model_dump() for c in analysis.cpc_codes]
        features_data = [f.model_dump() for f in analysis.features]
        
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
                            Jsonb(cpc_codes_data),
                            Jsonb(features_data),
                            Jsonb(analysis.assumptions),
                            analysis.confidence,
                            used_fallback,
                            latency_ms,
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to persist claims analysis: %s", exc)

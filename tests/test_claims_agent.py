from __future__ import annotations

import json

import pytest

from app.agents.claims import ClaimsAnalyzerAgent
from app.services.llm import LLMRequest
from config.settings import get_settings


class DummyLLMService:
    def __init__(self, payload: dict[str, object]):
        self.payload = json.dumps(payload)

    async def generate(self, _: LLMRequest, retries: int = 0) -> str:  # noqa: ARG002
        return self.payload


class BrokenLLMService:
    async def generate(self, _: LLMRequest, retries: int = 0) -> str:  # noqa: ARG002
        return "not-json"


@pytest.mark.asyncio
async def test_claims_agent_parses_llm_response():
    settings = get_settings()
    llm_output = {
        "summary": "Key advances in chiplet-based AI accelerators.",
        "query_type": "emergence",
        "features": [
            {
                "name": "chiplet topology",
                "insight": "Uses modular die-to-die interconnects.",
                "evidence": "Query mentions chiplet-based packaging.",
            }
        ],
        "cpc_codes": [
            {
                "code": "G06F17/30",
                "title": "Digital computing or data processing equipment",
                "confidence": 0.82,
                "justification": "Matches AI workload orchestration.",
            }
        ],
        "assumptions": ["Customer targets hyperscaler inference workloads."],
        "confidence": 0.8,
    }

    agent = ClaimsAnalyzerAgent(settings=settings, llm_service=DummyLLMService(llm_output))
    result = await agent.run("chiplet-based AI accelerator for datacenters")

    assert result.success is True
    assert result.data["query_type"] == "emergence"
    assert result.data["cpc_codes"][0]["code"] == "G06F17/30"
    assert result.data["source"] == "llm"


@pytest.mark.asyncio
async def test_claims_agent_falls_back_on_invalid_json():
    settings = get_settings()
    agent = ClaimsAnalyzerAgent(settings=settings, llm_service=BrokenLLMService())
    result = await agent.run("battery recycling process for EV packs")

    assert result.success is False
    assert result.data["source"] == "heuristic"
    assert result.data["cpc_codes"][0]["code"] == "G06F17/30"



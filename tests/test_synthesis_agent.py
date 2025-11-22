from __future__ import annotations

import json

import pytest

from app.agents.synthesis import SynthesisAgent
from app.services.llm import LLMRequest
from config.settings import get_settings


class DummyLLM:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload)

    async def generate(self, _: LLMRequest, retries: int = 0) -> str:  # noqa: ARG002
        return self.payload


class BrokenLLM:
    async def generate(self, _: LLMRequest, retries: int = 0) -> str:  # noqa: ARG002
        return "oops"


@pytest.mark.asyncio
async def test_synthesis_agent_parses_llm_output():
    settings = get_settings()
    payload = {
        "executive_summary": "Patent landscape shows strong modular packaging play.",
        "action_items": [
            {
                "priority": "high",
                "recommendation": "File continuations on chiplet interconnect IP.",
                "rationale": "High novelty vs. citations.",
            }
        ],
        "citations": [
            {"patent_id": "US12345A", "reason": "Closest prior art"},
            {"patent_id": "US67890B", "reason": "Litigated precedent"},
        ],
        "risk_score": 42,
        "notes": ["Consider cross-licensing with foundry partners."],
    }

    agent = SynthesisAgent(settings=settings, llm_service=DummyLLM(payload))
    agent.set_context(
        {
            "claims_analyzer": {"summary": "Claims summary"},
            "citation_mapper": {"results": [{"patent_id": "US12345A"}]},
            "litigation_scout": {"cases": 0},
        }
    )
    result = await agent.run("chiplet interconnect query")

    assert result.success is True
    assert result.data["executive_summary"].startswith("Patent landscape")
    assert result.data["risk_score"] == 42
    assert len(result.data["action_items"]) == 1
    assert result.data["source"] == "llm"


@pytest.mark.asyncio
async def test_synthesis_agent_fallback_on_bad_json():
    settings = get_settings()
    agent = SynthesisAgent(settings=settings, llm_service=BrokenLLM())
    agent.set_context({"claims_analyzer": {"summary": "Claims summary"}})
    result = await agent.run("battery recycling query")

    assert result.success is False
    assert result.data["source"] == "heuristic"
    assert result.data["risk_score"] == 55
    assert "summary" in result.data["executive_summary"].lower()



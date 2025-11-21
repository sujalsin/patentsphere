from __future__ import annotations

import asyncio
import time
from typing import Dict, List

from app.agents.base import AgentResult, BaseAgent
from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.agents.litigation import LitigationScoutAgent
from app.agents.synthesis import SynthesisAgent
from config.settings import get_settings


class Orchestrator:
    def __init__(self) -> None:
        settings = get_settings()
        self.agents: Dict[str, BaseAgent] = {
            "claims": ClaimsAnalyzerAgent(settings=settings),
            "citation": CitationMapperAgent(settings=settings),
            "litigation": LitigationScoutAgent(settings=settings),
            "synthesis": SynthesisAgent(settings=settings),
        }

    async def run_all(self, query: str) -> Dict[str, AgentResult]:
        async def run_agent(name: str, agent: BaseAgent) -> AgentResult:
            start = time.time()
            try:
                result = await agent.run(query)
                result.data["latency_ms"] = (time.time() - start) * 1000
                return result
            except Exception as exc:
                return AgentResult(
                    agent=name,
                    success=False,
                    data={},
                    error=str(exc),
                )

        tasks = [
            asyncio.create_task(run_agent(name, agent))
            for name, agent in self.agents.items()
        ]
        results = await asyncio.gather(*tasks)
        return {res.agent: res for res in results}


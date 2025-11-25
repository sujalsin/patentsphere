from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

import psycopg
from psycopg.types.json import Jsonb

from app.agents.base import AgentResult, BaseAgent
from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.agents.critic import CriticAgent
from app.agents.litigation import LitigationScoutAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from config.settings import get_settings

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self) -> None:
        settings = get_settings()
        self.agents: Dict[str, BaseAgent] = {
            "claims": ClaimsAnalyzerAgent(settings=settings),
            "citation": CitationMapperAgent(settings=settings),
            "citation_mapper": CitationMapperAgent(settings=settings),  # Alias for compatibility
            "litigation": LitigationScoutAgent(settings=settings),
            "litigation_scout": LitigationScoutAgent(settings=settings),  # Alias for compatibility
            "synthesis": SynthesisAgent(settings=settings),
        }
        
        # Add AdaptiveRetrievalAgent if enabled in config
        if settings.adaptive_retrieval.enabled:
            self.agents["adaptive_retrieval"] = AdaptiveRetrievalAgent(settings=settings)
        
        # Add AdaptiveRetrievalAgent if enabled in config
        if settings.adaptive_retrieval.enabled:
            self.agents["adaptive_retrieval"] = AdaptiveRetrievalAgent(settings=settings)
        
        # Add CriticAgent if enabled in config
        if settings.critic.enabled:
            critic_weights = settings.critic.reward_weights
            self.agents["critic"] = CriticAgent(settings=settings, weights=critic_weights)
        
        self.latest_context: Dict[str, Dict[str, Any]] = {}

    async def run_all(self, query: str) -> Dict[str, AgentResult]:
        async def run_agent(name: str, agent: BaseAgent) -> AgentResult:
            start = time.time()
            try:
                # LitigationScoutAgent has a different signature
                if name == "litigation_scout" and hasattr(agent, "run"):
                    # Will be called again later with patent IDs, so just run basic version
                    result = await agent.run(query, retrieved_patent_ids=None)
                else:
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

        # Run non-synthesis and non-critic agents in parallel to build context
        # Exclude duplicates (citation_mapper, litigation_scout are aliases)
        # If AdaptiveRetrievalAgent is enabled, it will replace CitationMapperAgent
        settings = get_settings()
        adaptive_enabled = settings.adaptive_retrieval.enabled if hasattr(settings, 'adaptive_retrieval') else False
        
        parallel_agents = {}
        for name, agent in self.agents.items():
            if name in ("synthesis", "critic", "citation_mapper", "litigation_scout"):
                continue
            # If adaptive_retrieval is enabled, skip citation (it will be handled by adaptive)
            if adaptive_enabled and name == "citation":
                continue
            parallel_agents[name] = agent
        
        # Add timeouts to prevent hanging
        settings = get_settings()
        # Use shorter timeout for individual agents to prevent hanging
        # LLM agents have their own timeouts, but we cap at orchestrator timeout
        agent_timeout = min(settings.orchestrator.timeout, 180)  # Max 3 minutes per agent
        
        async def run_with_timeout(name: str, agent: BaseAgent) -> AgentResult:
            try:
                return await asyncio.wait_for(
                    run_agent(name, agent),
                    timeout=agent_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Agent %s timed out after %ss", name, agent_timeout)
                return AgentResult(
                    agent=name,
                    success=False,
                    data={},
                    error=f"Timeout after {agent_timeout}s",
                )
        
        tasks = [
            asyncio.create_task(run_with_timeout(name, agent))
            for name, agent in parallel_agents.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions from gather
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Agent execution raised exception: %s", result)
                processed_results.append(AgentResult(
                    agent="unknown",
                    success=False,
                    data={},
                    error=str(result),
                ))
            else:
                processed_results.append(result)
        results = processed_results
        context = {res.agent: res.data for res in results if res.success}
        self.latest_context = context
        
        # Run AdaptiveRetrievalAgent if enabled (replaces CitationMapperAgent)
        adaptive_agent = self.agents.get("adaptive_retrieval")
        if adaptive_enabled and adaptive_agent:
            try:
                # Get query_type from claims_analyzer
                claims_data = context.get("claims_analyzer", {})
                query_type = claims_data.get("query_type", "other")
                
                # Run adaptive retrieval
                adaptive_result = await asyncio.wait_for(
                    adaptive_agent.run(query=query, query_type=query_type),
                    timeout=agent_timeout
                )
                
                if adaptive_result.success:
                    # Store adaptive retrieval results
                    context["adaptive_retrieval"] = adaptive_result.data
                    context["citation_mapper"] = adaptive_result.data  # For compatibility
                    
                    # Update results list - replace citation if exists, otherwise append
                    citation_found = False
                    for i, res in enumerate(results):
                        if res.agent == "citation":
                            results[i] = adaptive_result
                            citation_found = True
                            break
                    if not citation_found:
                        results.append(adaptive_result)
                else:
                    logger.warning("AdaptiveRetrievalAgent failed, falling back to CitationMapperAgent: %s", adaptive_result.error)
                    # Fallback to regular citation mapper
                    citation_agent = self.agents.get("citation")
                    if citation_agent:
                        citation_result = await asyncio.wait_for(
                            citation_agent.run(query),
                            timeout=agent_timeout
                        )
                        if citation_result.success:
                            context["citation_mapper"] = citation_result.data
                            # Update or add citation result
                            citation_found = False
                            for i, res in enumerate(results):
                                if res.agent == "citation":
                                    results[i] = citation_result
                                    citation_found = True
                                    break
                            if not citation_found:
                                results.append(citation_result)
            except asyncio.TimeoutError:
                logger.warning("AdaptiveRetrievalAgent timed out, falling back to CitationMapperAgent")
                # Fallback
                citation_agent = self.agents.get("citation")
                if citation_agent:
                    citation_result = await asyncio.wait_for(
                        citation_agent.run(query),
                        timeout=agent_timeout
                    )
                    if citation_result.success:
                        context["citation_mapper"] = citation_result.data
                        results.append(citation_result)
            except Exception as exc:
                logger.warning("AdaptiveRetrievalAgent error: %s, falling back", exc)
                # Fallback
                citation_agent = self.agents.get("citation")
                if citation_agent:
                    citation_result = await asyncio.wait_for(
                        citation_agent.run(query),
                        timeout=agent_timeout
                    )
                    if citation_result.success:
                        context["citation_mapper"] = citation_result.data
                        results.append(citation_result)
        
        # Pass retrieved patent IDs to LitigationScoutAgent if available
        litigation_agent = self.agents.get("litigation")
        if litigation_agent and hasattr(litigation_agent, "run"):
            # Extract patent IDs from citation results
            citation_data = context.get("citation_mapper", {})
            retrieved_patent_ids = []
            if citation_data and "results" in citation_data:
                retrieved_patent_ids = [
                    r.get("patent_id") for r in citation_data["results"]
                    if r.get("patent_id")
                ]
            
            # Re-run litigation agent with patent IDs
            if retrieved_patent_ids:
                try:
                    litigation_result = await litigation_agent.run(
                        query=query,
                        retrieved_patent_ids=retrieved_patent_ids,
                    )
                    if litigation_result.success:
                        # Update context with litigation data
                        context["litigation_scout"] = litigation_result.data
                        # Replace the result in results list
                        for i, res in enumerate(results):
                            if res.agent == "litigation_scout":
                                results[i] = litigation_result
                                break
                        else:
                            results.append(litigation_result)
                except Exception as exc:
                    logger.warning("LitigationScoutAgent re-run failed: %s", exc)

        # Run synthesis last so it can see other agent outputs
        synthesis_result: AgentResult | None = None
        synthesis_agent = self.agents.get("synthesis")
        if synthesis_agent and hasattr(synthesis_agent, "set_context"):
            getattr(synthesis_agent, "set_context")(context)
        if synthesis_agent:
            try:
                synthesis_result = await asyncio.wait_for(
                    run_agent("synthesis", synthesis_agent),
                    timeout=agent_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("SynthesisAgent timed out after %ss", agent_timeout)
                synthesis_result = AgentResult(
                    agent="synthesis",
                    success=False,
                    data={},
                    error=f"Timeout after {agent_timeout}s",
                )

        if synthesis_result:
            results.append(synthesis_result)
            context["synthesis"] = synthesis_result.data

        # Run CriticAgent if enabled (after all other agents complete)
        critic_result: AgentResult | None = None
        critic_agent = self.agents.get("critic")
        if critic_agent:
            try:
                # Extract data for critic
                retrieved_chunks = []
                if "citation_mapper" in context:
                    citation_data = context["citation_mapper"]
                    retrieved_chunks = citation_data.get("results", [])
                
                claims_analysis = context.get("claims_analyzer", {})
                synthesis_output = context.get("synthesis", {})
                
                # Run critic with proper parameters
                start = time.time()
                try:
                    critic_result = await asyncio.wait_for(
                        critic_agent.run(
                            query=query,
                            retrieved_chunks=retrieved_chunks,
                            claims_analysis=claims_analysis,
                            synthesis_output=synthesis_output,
                        ),
                        timeout=agent_timeout
                    )
                    critic_result.data["latency_ms"] = (time.time() - start) * 1000
                    
                    if critic_result.success:
                        # Log reward to database
                        await self._log_reward(
                            query,
                            claims_analysis.get("query_type"),
                            retrieved_chunks,
                            critic_result.data,
                            context,
                        )
                        results.append(critic_result)
                        context["critic"] = critic_result.data
                except asyncio.TimeoutError:
                    logger.warning("CriticAgent timed out after %ss", agent_timeout)
                    critic_result = AgentResult(
                        agent="critic",
                        success=False,
                        data={},
                        error=f"Timeout after {agent_timeout}s",
                    )
                except Exception as exc:
                    logger.warning("CriticAgent execution failed: %s", exc)
                    critic_result = AgentResult(
                        agent="critic",
                        success=False,
                        data={},
                        error=str(exc),
                    )
            except Exception as exc:
                logger.warning("CriticAgent setup failed: %s", exc)

        return {res.agent: res for res in results}

    async def _log_reward(
        self,
        query: str,
        query_type: str | None,
        retrieved_chunks: List[Dict[str, Any]],
        critic_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Log reward to rl_experiences table in PostgreSQL."""
        settings = get_settings()
        pg_cfg = getattr(settings, "database", None)
        if not pg_cfg:
            return

        try:
            # Extract patent IDs from retrieved chunks
            patent_ids = [
                chunk.get("patent_id")
                for chunk in retrieved_chunks
                if chunk.get("patent_id")
            ]

            conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rl_experiences (
                            query_text, query_type, retrieved_patent_ids,
                            total_reward, reward_components, agent_outputs
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query,
                            query_type,
                            patent_ids,
                            critic_data.get("score"),
                            Jsonb(critic_data.get("components", {})),
                            Jsonb(context),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to log reward: %s", exc)


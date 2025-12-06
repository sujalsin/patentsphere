"""PatentSphere Agent Orchestrator.

This module provides the main orchestration layer for patent queries,
using LangGraph for adaptive agent coordination.

The orchestrator supports two modes:
1. LangGraph mode (default): Uses adaptive graph-based orchestration
2. Legacy mode: Uses the original asyncio-based parallel execution
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.types.json import Jsonb

from app.agents.base import AgentResult, BaseAgent
from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.agents.critic import CriticAgent
from app.agents.litigation import LitigationScoutAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from app.graph.graph import PatentGraph, create_patent_graph
from app.graph.state import PatentQueryState
from config.settings import get_settings

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for patent queries.
    
    Supports both LangGraph-based adaptive orchestration and legacy
    parallel execution mode.
    """
    
    def __init__(self, use_langgraph: bool = True) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            use_langgraph: If True, use LangGraph orchestration.
                          If False, use legacy parallel mode.
        """
        self.settings = get_settings()
        self.use_langgraph = use_langgraph
        self._graph: Optional[PatentGraph] = None
        
        # Initialize agents for legacy mode
        self.agents: Dict[str, BaseAgent] = {
            "claims": ClaimsAnalyzerAgent(settings=self.settings),
            "citation": CitationMapperAgent(settings=self.settings),
            "citation_mapper": CitationMapperAgent(settings=self.settings),
            "litigation": LitigationScoutAgent(settings=self.settings),
            "litigation_scout": LitigationScoutAgent(settings=self.settings),
            "synthesis": SynthesisAgent(settings=self.settings),
        }
        
        if self.settings.adaptive_retrieval.enabled:
            self.agents["adaptive_retrieval"] = AdaptiveRetrievalAgent(settings=self.settings)
        
        if self.settings.critic.enabled:
            critic_weights = self.settings.critic.reward_weights
            self.agents["critic"] = CriticAgent(settings=self.settings, weights=critic_weights)
        
        self.latest_context: Dict[str, Dict[str, Any]] = {}
    
    @property
    def graph(self) -> PatentGraph:
        """Get or create the LangGraph instance."""
        if self._graph is None:
            self._graph = create_patent_graph(settings=self.settings)
        return self._graph
    
    async def run_all(self, query: str) -> Dict[str, AgentResult]:
        """
        Run all agents for a query.
        
        Uses LangGraph orchestration if enabled, otherwise falls back
        to legacy parallel execution.
        
        Args:
            query: The user's patent query
        
        Returns:
            Dictionary mapping agent names to their results
        """
        # Input validation
        validated_query = self._validate_query(query)
        if validated_query is None:
            return self._create_error_response("Invalid query: Please provide a non-empty search query.")
        
        if self.use_langgraph:
            return await self._run_langgraph(validated_query)
        else:
            return await self._run_legacy(validated_query)
    
    def _validate_query(self, query: str) -> Optional[str]:
        """
        Validate and sanitize the input query.
        
        Returns:
            Sanitized query string or None if invalid
        """
        if not query:
            return None
        
        # Strip whitespace and normalize
        query = query.strip()
        
        if not query:
            return None
        
        # Check minimum length
        if len(query) < 3:
            return None
        
        # Check maximum length (prevent extremely long queries)
        if len(query) > 2000:
            query = query[:2000]
            logger.warning("Query truncated to 2000 characters")
        
        return query
    
    def _create_error_response(self, error_message: str) -> Dict[str, AgentResult]:
        """Create a standardized error response."""
        return {
            "error": AgentResult(
                agent="orchestrator",
                success=False,
                data={"error": error_message},
                error=error_message,
            ),
            "final": AgentResult(
                agent="final",
                success=False,
                data={
                    "executive_summary": f"Analysis could not be completed: {error_message}",
                    "error": error_message,
                    "api_response": {
                        "response_id": "error",
                        "query": "",
                        "sources_header": "",
                        "answer_section": {
                            "technical_summary": f"Error: {error_message}",
                            "legal_summary": None,
                            "novelty_assessment": None,
                        },
                        "sources": [],
                        "risk_score": 0,
                        "quality_score": 0.0,
                        "metadata": {"error": error_message},
                    },
                },
                error=error_message,
            ),
        }
    
    async def _run_langgraph(self, query: str) -> Dict[str, AgentResult]:
        """Run query using LangGraph orchestration."""
        logger.info("Running query with LangGraph orchestration")
        
        try:
            # Execute the graph with timeout
            # Reduced timeout for faster feedback - if agents are slow, they should timeout individually
            final_state = await asyncio.wait_for(
                self.graph.invoke(
                    query=query,
                    max_iterations=2,  # Reduced iterations for faster results
                ),
                timeout=180,  # 3 minute overall timeout
            )
            
            # Validate final state has required data
            if not final_state or not final_state.get("agent_outputs"):
                logger.warning("LangGraph returned empty state, using fallback")
                return self._create_fallback_response(query, "Analysis returned empty results")
            
            # Convert state to AgentResult format for compatibility
            results = self._state_to_results(final_state)
            
            # Validate we have at least some successful results
            successful_agents = [name for name, r in results.items() if r.success]
            if not successful_agents:
                logger.warning("No agents succeeded, creating fallback response")
                return self._create_fallback_response(query, "No agents completed successfully")
            
            # Update latest context for compatibility
            self.latest_context = {
                name: result.data
                for name, result in results.items()
                if result.success
            }
            
            return results
            
        except asyncio.TimeoutError:
            logger.error("LangGraph execution timed out after 3 minutes")
            
            # Create a timeout response with helpful guidance
            timeout_msg = (
                "Analysis timed out after 3 minutes. "
                "This may happen with complex queries. "
                "Suggestions: (1) Try a more specific query, (2) Reduce query scope, "
                "(3) Check if Ollama models are loaded and responsive."
            )
            return self._create_error_response(timeout_msg)
            
        except Exception as exc:
            logger.exception("LangGraph execution failed: %s", exc)
            # Fall back to legacy mode on failure
            logger.info("Falling back to legacy orchestration")
            try:
                return await self._run_legacy(query)
            except Exception as legacy_exc:
                logger.exception("Legacy mode also failed: %s", legacy_exc)
                return self._create_error_response(f"Analysis failed: {str(exc)}")
    
    def _create_fallback_response(self, query: str, reason: str) -> Dict[str, AgentResult]:
        """Create a fallback response when analysis partially fails."""
        return {
            "final": AgentResult(
                agent="final",
                success=True,
                data={
                    "executive_summary": f"Analysis completed with limited results. {reason}",
                    "query": query,
                    "api_response": {
                        "response_id": f"fallback_{uuid.uuid4().hex[:8]}",
                        "query": query,
                        "sources_header": "Limited results available",
                        "answer_section": {
                            "technical_summary": f"The analysis encountered some issues: {reason}. Please try refining your query or try again later.",
                            "legal_summary": None,
                            "novelty_assessment": None,
                        },
                        "sources": [],
                        "risk_score": 50,
                        "quality_score": 0.3,
                        "metadata": {"fallback_reason": reason},
                    },
                },
            ),
        }
    
    def _state_to_results(self, state: PatentQueryState) -> Dict[str, AgentResult]:
        """Convert LangGraph state to AgentResult dictionary."""
        results: Dict[str, AgentResult] = {}
        
        agent_outputs = state.get("agent_outputs", {})
        for agent_name, output in agent_outputs.items():
            results[agent_name] = AgentResult(
                agent=output.get("agent", agent_name),
                success=output.get("success", False),
                data=output.get("data", {}),
                error=output.get("error"),
            )
        
        # Add final response as a pseudo-result for compatibility
        final_response = state.get("final_response")
        if final_response:
            results["final"] = AgentResult(
                agent="final",
                success=True,
                data=final_response,
            )
        
        return results
    
    async def _run_legacy(self, query: str) -> Dict[str, AgentResult]:
        """Run query using legacy parallel orchestration."""
        logger.info("Running query with legacy parallel orchestration")
        
        async def run_agent(name: str, agent: BaseAgent) -> AgentResult:
            start = time.time()
            try:
                if name == "litigation_scout" and hasattr(agent, "run"):
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

        # Run non-synthesis and non-critic agents in parallel
        adaptive_enabled = self.settings.adaptive_retrieval.enabled
        
        parallel_agents = {}
        for name, agent in self.agents.items():
            if name in ("synthesis", "critic", "citation_mapper", "litigation_scout"):
                continue
            if adaptive_enabled and name == "citation":
                continue
            parallel_agents[name] = agent
        
        agent_timeout = min(self.settings.orchestrator.timeout, 180)
        
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
        
        context = {res.agent: res.data for res in processed_results if res.success}
        self.latest_context = context
        
        # Run adaptive retrieval if enabled
        adaptive_agent = self.agents.get("adaptive_retrieval")
        if adaptive_enabled and adaptive_agent:
            try:
                claims_data = context.get("claims_analyzer", {})
                query_type = claims_data.get("query_type", "other")
                
                adaptive_result = await asyncio.wait_for(
                    adaptive_agent.run(query=query, query_type=query_type),
                    timeout=agent_timeout
                )
                
                if adaptive_result.success:
                    context["adaptive_retrieval"] = adaptive_result.data
                    context["citation_mapper"] = adaptive_result.data
                    processed_results.append(adaptive_result)
            except Exception as exc:
                logger.warning("AdaptiveRetrievalAgent failed: %s", exc)
        
        # Run litigation with patent IDs
        litigation_agent = self.agents.get("litigation")
        if litigation_agent:
            citation_data = context.get("citation_mapper", {})
            retrieved_patent_ids = [
                r.get("patent_id") for r in citation_data.get("results", [])
                if r.get("patent_id")
            ]
            
            if retrieved_patent_ids:
                try:
                    litigation_result = await litigation_agent.run(
                        query=query,
                        retrieved_patent_ids=retrieved_patent_ids,
                    )
                    if litigation_result.success:
                        context["litigation_scout"] = litigation_result.data
                        processed_results.append(litigation_result)
                except Exception as exc:
                    logger.warning("LitigationScoutAgent failed: %s", exc)
        
        # Run synthesis
        synthesis_agent = self.agents.get("synthesis")
        if synthesis_agent:
            if hasattr(synthesis_agent, "set_context"):
                synthesis_agent.set_context(context)
            try:
                synthesis_result = await asyncio.wait_for(
                    run_agent("synthesis", synthesis_agent),
                    timeout=agent_timeout
                )
                processed_results.append(synthesis_result)
                context["synthesis"] = synthesis_result.data
            except asyncio.TimeoutError:
                logger.warning("SynthesisAgent timed out")
        
        # Run critic
        critic_agent = self.agents.get("critic")
        if critic_agent:
            try:
                retrieved_chunks = context.get("citation_mapper", {}).get("results", [])
                claims_analysis = context.get("claims_analyzer", {})
                synthesis_output = context.get("synthesis", {})
                
                critic_result = await asyncio.wait_for(
                    critic_agent.run(
                        query=query,
                        retrieved_chunks=retrieved_chunks,
                        claims_analysis=claims_analysis,
                        synthesis_output=synthesis_output,
                    ),
                    timeout=agent_timeout
                )
                
                if critic_result.success:
                    processed_results.append(critic_result)
                    context["critic"] = critic_result.data
                    
                    # Log reward
                    await self._log_reward(
                        query=query,
                        query_type=claims_analysis.get("query_type"),
                        retrieved_chunks=retrieved_chunks,
                        critic_data=critic_result.data,
                        context=context,
                    )
            except Exception as exc:
                logger.warning("CriticAgent failed: %s", exc)
        
        return {res.agent: res for res in processed_results}
    
    async def stream_query(self, query: str):
        """
        Stream query execution, yielding state updates.
        
        Only available in LangGraph mode.
        
        Args:
            query: The user's patent query
        
        Yields:
            State updates from each node
        """
        if not self.use_langgraph:
            raise NotImplementedError("Streaming only available in LangGraph mode")
        
        async for state_update in self.graph.stream(query, max_iterations=3):
            yield state_update
    
    async def _log_reward(
        self,
        query: str,
        query_type: str | None,
        retrieved_chunks: List[Dict[str, Any]],
        critic_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Log reward to database for RLAIF training."""
        pg_cfg = getattr(self.settings, "database", None)
        if not pg_cfg:
            return

        try:
            patent_ids = [
                chunk.get("patent_id")
                for chunk in retrieved_chunks
                if chunk.get("patent_id")
            ]
            
            chunk_payload = self._prepare_chunk_payload(retrieved_chunks)

            conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rl_experiences (
                            query_text,
                            query_type,
                            retrieved_patent_ids,
                            retrieved_chunks,
                            total_reward,
                            reward_components,
                            agent_outputs
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query,
                            query_type,
                            patent_ids,
                            Jsonb(chunk_payload),
                            critic_data.get("score"),
                            Jsonb(critic_data.get("components", {})),
                            Jsonb(context),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.debug("Failed to log reward: %s", exc)
    
    def _prepare_chunk_payload(
        self, chunks: List[Dict[str, Any]], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Prepare chunk payload for database storage."""
        payload: List[Dict[str, Any]] = []
        for entry in chunks[:limit]:
            payload.append({
                "chunk_id": entry.get("chunk_id"),
                "patent_id": entry.get("patent_id"),
                "chunk_type": entry.get("chunk_type"),
                "chunk_order": entry.get("chunk_order") or entry.get("order"),
                "score": entry.get("score"),
            })
        return payload


# Convenience function for quick queries
async def run_query(query: str, use_langgraph: bool = True) -> Dict[str, AgentResult]:
    """
    Run a patent query through the orchestrator.
    
    Args:
        query: The user's patent query
        use_langgraph: Whether to use LangGraph orchestration
    
    Returns:
        Dictionary mapping agent names to their results
    """
    orchestrator = Orchestrator(use_langgraph=use_langgraph)
    return await orchestrator.run_all(query)

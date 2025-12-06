"""Agent wrapper nodes for the LangGraph patent query graph.

Each node wraps an agent and transforms its output into state updates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from app.graph.state import PatentQueryState, AgentOutput
from app.agents.base import AgentResult
from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from app.agents.litigation import LitigationScoutAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.critic import CriticAgent
from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


class AgentNodes:
    """Container for all agent node functions."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._agents: Dict[str, Any] = {}
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize all agents once."""
        self._agents = {
            "claims_analyzer": ClaimsAnalyzerAgent(settings=self.settings),
            "citation_mapper": CitationMapperAgent(settings=self.settings),
            "synthesis": SynthesisAgent(settings=self.settings),
        }
        
        # Optional agents based on config
        if self.settings.adaptive_retrieval.enabled:
            self._agents["adaptive_retrieval"] = AdaptiveRetrievalAgent(settings=self.settings)
        
        if self.settings.critic.enabled:
            self._agents["critic"] = CriticAgent(
                settings=self.settings,
                weights=self.settings.critic.reward_weights,
            )
        
        # Litigation scout
        self._agents["litigation_scout"] = LitigationScoutAgent(settings=self.settings)
    
    def _result_to_output(self, result: AgentResult, latency_ms: float) -> AgentOutput:
        """Convert AgentResult to AgentOutput."""
        return AgentOutput(
            agent=result.agent,
            success=result.success,
            data=result.data,
            error=result.error,
            latency_ms=latency_ms,
        )
    
    def _get_agent_timeout(self, agent_name: str, default: float = 60.0) -> float:
        """Get timeout for an agent from settings, with fallback."""
        try:
            if agent_name == "claims_analyzer" and hasattr(self.settings, "claims_analyzer"):
                return getattr(self.settings.claims_analyzer, "timeout", default)
            elif agent_name == "synthesis" and hasattr(self.settings, "synthesis"):
                return getattr(self.settings.synthesis, "timeout", default)
            elif agent_name == "critic" and hasattr(self.settings, "critic"):
                return getattr(self.settings.critic, "timeout", default)
            elif agent_name == "litigation_scout" and hasattr(self.settings, "litigation_scout"):
                return getattr(self.settings.litigation_scout, "timeout", default)
            elif agent_name == "citation_mapper" and hasattr(self.settings, "citation_mapper"):
                return getattr(self.settings.citation_mapper, "timeout", default)
        except AttributeError:
            pass
        return default
    
    async def claims_analyzer_node(self, state: PatentQueryState) -> Dict[str, Any]:
        """
        Claims analyzer node - analyzes query to extract features and CPC codes.
        
        This is typically the first node in the graph, providing routing hints
        for subsequent nodes.
        """
        start = time.perf_counter()
        query = state["query"]
        
        agent = self._agents.get("claims_analyzer")
        if not agent:
            return {
                "errors": [*state.get("errors", []), "Claims analyzer not initialized"],
                "messages": [*state.get("messages", []), "Claims analyzer skipped - not initialized"],
            }
        
        timeout = self._get_agent_timeout("claims_analyzer", 30.0)
        
        try:
            result = await asyncio.wait_for(
                agent.run(query),
                timeout=timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            output = self._result_to_output(result, latency_ms)
            
            # Extract query type for routing
            query_type = result.data.get("query_type", "other") if result.success else "other"
            
            return {
                "agent_outputs": {result.agent: output},
                "claims_analysis": result.data if result.success else None,
                "query_type": query_type,
                "messages": [f"Claims analysis completed in {latency_ms:.0f}ms"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("Claims analyzer timed out after %.0fs, using fallback", timeout)
            
            # Graceful degradation: return minimal fallback
            fallback_data = {
                "summary": f"Query analysis timed out: {query[:100]}",
                "query_type": "research",  # Safe default
                "features": [],
                "cpc_codes": [],
                "assumptions": ["Analysis timed out - using default classification"],
                "confidence": 0.3,
            }
            
            fallback_result = AgentResult(
                agent="claims_analyzer",
                success=False,
                data=fallback_data,
                error=f"Timeout after {timeout}s",
            )
            
            output = self._result_to_output(fallback_result, latency_ms)
            
            return {
                "agent_outputs": {"claims_analyzer": output},
                "claims_analysis": fallback_data,
                "query_type": "research",
                "messages": [f"Claims analyzer timed out after {timeout}s - using fallback"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
        except Exception as exc:
            logger.exception("Claims analyzer failed: %s", exc)
            return {
                "errors": [*state.get("errors", []), str(exc)],
                "messages": [f"Claims analyzer error: {exc}"],
            }
    
    async def retrieval_node(self, state: PatentQueryState) -> Dict[str, Any]:
        """
        Retrieval node - performs citation/patent retrieval with hybrid search.
        
        Uses adaptive retrieval if enabled, otherwise falls back to citation mapper.
        Both now support hybrid search (vector + BM25 + CPC filtering).
        """
        start = time.perf_counter()
        query = state["query"]
        query_type = state.get("query_type", "other")
        claims_analysis = state.get("claims_analysis", {})
        
        # Prefer adaptive retrieval if enabled
        if "adaptive_retrieval" in self._agents:
            agent = self._agents["adaptive_retrieval"]
            try:
                result = await asyncio.wait_for(
                    agent.run(query=query, query_type=query_type, claims_analysis=claims_analysis),
                    timeout=180,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                
                if result.success:
                    output = self._result_to_output(result, latency_ms)
                    retrieved_chunks = result.data.get("results", [])
                    patent_ids = list({
                        c.get("patent_id") for c in retrieved_chunks if c.get("patent_id")
                    })
                    
                    return {
                        "agent_outputs": {result.agent: output},
                        "citation_results": result.data,
                        "retrieved_chunks": retrieved_chunks,
                        "retrieved_patent_ids": patent_ids,
                        "messages": [f"Adaptive retrieval (hybrid search) completed in {latency_ms:.0f}ms, {len(retrieved_chunks)} chunks"],
                        "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
                    }
            except Exception as exc:
                logger.warning("Adaptive retrieval failed, falling back to citation mapper: %s", exc)
        
        # Fallback to citation mapper with hybrid search
        agent = self._agents.get("citation_mapper")
        if not agent:
            return {
                "errors": [*state.get("errors", []), "No retrieval agent available"],
                "messages": ["Retrieval skipped - no agent available"],
            }
        
        try:
            result = await asyncio.wait_for(
                agent.run(query, claims_analysis=claims_analysis),
                timeout=120,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            output = self._result_to_output(result, latency_ms)
            retrieved_chunks = result.data.get("results", []) if result.success else []
            patent_ids = list({
                c.get("patent_id") for c in retrieved_chunks if c.get("patent_id")
            })
            
            return {
                "agent_outputs": {result.agent: output},
                "citation_results": result.data if result.success else None,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_patent_ids": patent_ids,
                "messages": [f"Citation mapping (hybrid search) completed in {latency_ms:.0f}ms, {len(retrieved_chunks)} chunks"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "errors": [*state.get("errors", []), "Retrieval timed out"],
                "messages": [f"Retrieval timed out after {latency_ms:.0f}ms"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            return {
                "errors": [*state.get("errors", []), str(exc)],
                "messages": [f"Retrieval error: {exc}"],
            }
    
    async def litigation_node(self, state: PatentQueryState) -> Dict[str, Any]:
        """
        Litigation scout node - checks for litigation signals using general mode.
        
        Uses Risk Entities from ClaimsAnalyzer to search for general litigation patterns.
        Runs in parallel with AdaptiveRetrieval.
        Only executes if the query requires litigation check (query_type=litigation or mentions litigation terms).
        """
        from app.graph.router import should_run_litigation
        
        # Check if litigation should run
        if not should_run_litigation(state):
            logger.info("Litigation scout skipped - not needed for this query type")
            return {
                "messages": ["Litigation scout skipped - query does not require litigation check"],
                "litigation_data": None,
            }
        
        start = time.perf_counter()
        query = state["query"]
        claims_analysis = state.get("claims_analysis", {})
        
        # Extract risk entities from claims analysis
        risk_entities = None
        if isinstance(claims_analysis, dict):
            risk_entities = claims_analysis.get("risk_entities")
            if risk_entities and isinstance(risk_entities, dict):
                # Ensure it's a dict (might be a Pydantic model)
                if hasattr(risk_entities, "model_dump"):
                    risk_entities = risk_entities.model_dump()
        
        agent = self._agents.get("litigation_scout")
        if not agent:
            return {
                "messages": ["Litigation scout not initialized - skipping"],
            }
        
        try:
            # Use general mode for parallel execution
            result = await asyncio.wait_for(
                agent.run_general_mode(query=query, risk_entities=risk_entities),
                timeout=90,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            output = self._result_to_output(result, latency_ms)
            
            return {
                "agent_outputs": {result.agent: output},
                "litigation_data": result.data if result.success else None,
                "litigation_general_data": result.data if result.success else None,  # Alias for compatibility
                "messages": [f"Litigation scout (general mode) completed in {latency_ms:.0f}ms"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "messages": [f"Litigation scout timed out after {latency_ms:.0f}ms"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
        except Exception as exc:
            logger.warning("Litigation scout failed: %s", exc)
            return {
                "messages": [f"Litigation scout error (non-critical): {exc}"],
            }
    
    async def synthesis_node(self, state: PatentQueryState) -> Dict[str, Any]:
        """
        Synthesis node - combines all agent outputs into executive briefing.
        
        This node requires claims analysis and retrieval to have completed.
        """
        start = time.perf_counter()
        query = state["query"]
        
        agent = self._agents.get("synthesis")
        if not agent:
            return {
                "errors": [*state.get("errors", []), "Synthesis agent not initialized"],
                "messages": ["Synthesis skipped - not initialized"],
            }
        
        # Build context from previous agents
        context = {
            "claims_analyzer": state.get("claims_analysis", {}),
            "citation_mapper": state.get("citation_results", {}),
            "litigation_scout": state.get("litigation_data", {}) or state.get("litigation_general_data", {}),
            "retrieved_chunks": state.get("retrieved_chunks", []),  # For in-memory join
        }
        agent.set_context(context)
        
        timeout = self._get_agent_timeout("synthesis", 45.0)
        
        try:
            result = await asyncio.wait_for(
                agent.run(query),
                timeout=timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            output = self._result_to_output(result, latency_ms)
            
            return {
                "agent_outputs": {result.agent: output},
                "synthesis_output": result.data if result.success else None,
                "messages": [f"Synthesis completed in {latency_ms:.0f}ms"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("Synthesis timed out after %.0fs, using fallback", timeout)
            
            # Graceful degradation: create minimal synthesis from available data
            retrieved_chunks = state.get("retrieved_chunks", [])
            claims = state.get("claims_analysis", {}) or {}
            
            fallback_summary = f"Analysis for '{query[:80]}' timed out. Retrieved {len(retrieved_chunks)} patents."
            if isinstance(claims, dict) and claims.get("summary"):
                fallback_summary = claims.get("summary", fallback_summary)
            
            fallback_data = {
                "executive_summary": fallback_summary,
                "insight_sections": [],
                "next_steps": [{
                    "priority": "medium",
                    "recommendation": "Query timed out - try a more specific query",
                    "rationale": "Synthesis agent exceeded timeout limit"
                }],
                "citations": [],
                "risk_score": 50,
            }
            
            fallback_result = AgentResult(
                agent="synthesis",
                success=False,
                data=fallback_data,
                error=f"Timeout after {timeout}s",
            )
            
            output = self._result_to_output(fallback_result, latency_ms)
            
            return {
                "agent_outputs": {"synthesis": output},
                "synthesis_output": fallback_data,
                "messages": [f"Synthesis timed out after {timeout}s - using fallback"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
        except Exception as exc:
            logger.exception("Synthesis failed: %s", exc)
            return {
                "errors": [*state.get("errors", []), str(exc)],
                "messages": [f"Synthesis error: {exc}"],
            }
    
    async def critic_node(self, state: PatentQueryState) -> Dict[str, Any]:
        """
        Critic node - evaluates the quality of the synthesis output.
        
        Provides feedback that can trigger re-retrieval or synthesis refinement.
        """
        start = time.perf_counter()
        query = state["query"]
        
        agent = self._agents.get("critic")
        if not agent:
            return {
                "messages": ["Critic agent not enabled - skipping evaluation"],
                "quality_threshold_met": True,  # Assume OK if no critic
            }
        
        retrieved_chunks = state.get("retrieved_chunks", [])
        claims_analysis = state.get("claims_analysis", {})
        synthesis_output = state.get("synthesis_output", {})
        
        try:
            result = await asyncio.wait_for(
                agent.run(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    claims_analysis=claims_analysis,
                    synthesis_output=synthesis_output,
                ),
                timeout=90,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Add latency to the data dict for display
            result_data = result.data.copy() if result.data else {}
            result_data["latency_ms"] = latency_ms
            
            output = AgentOutput(
                name="critic",
                success=result.success,
                data=result_data,
                latency_ms=latency_ms,
            )
            score = result.data.get("score", 0.0) if result.success else 0.0
            
            # Determine if quality threshold is met (e.g., > 0.6)
            quality_threshold = 0.6
            quality_met = score >= quality_threshold
            
            # Determine if more retrieval is needed based on component scores
            components = result.data.get("components", {})
            needs_more = (
                components.get("citation_overlap", 1.0) < 0.3 or
                components.get("cpc_relevance", 1.0) < 0.3
            )
            
            return {
                "agent_outputs": {**state.get("agent_outputs", {}), "critic": output},
                "critic_score": score,
                "quality_threshold_met": quality_met,
                "needs_more_retrieval": needs_more and not quality_met,
                "messages": [f"Critic evaluation: {score:.2f} (threshold met: {quality_met})"],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "messages": [f"Critic timed out after {latency_ms:.0f}ms"],
                "quality_threshold_met": True,  # Assume OK on timeout
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }
        except Exception as exc:
            logger.warning("Critic failed: %s", exc)
            return {
                "messages": [f"Critic error (non-critical): {exc}"],
                "quality_threshold_met": True,  # Assume OK on error
            }
    
    async def finalize_node(self, state: PatentQueryState) -> Dict[str, Any]:
        """
        Finalize node - assembles the final response with indexed citations.
        
        Uses the enhanced citation manager to build clickable source references
        and formats the response according to the API schema.
        """
        start = time.perf_counter()
        
        import uuid
        from app.templates.models import (
            FinalResponse,
            APIResponse,
            AnswerSectionModel,
            SourceItem,
        )
        from app.templates.citations import (
            CitationIndexManager,
            extract_citations_from_chunks,
            extract_citations_from_litigation,
            PatentURLBuilder,
        )
        from app.services.llm import LLMRequest
        from app.templates.prompts import (
            FINAL_OUTPUT_SYSTEM_PROMPT,
            get_enhanced_final_prompt,
        )
        
        query = state["query"]
        claims = state.get("claims_analysis", {})
        synthesis = state.get("synthesis_output", {})
        critic_score = state.get("critic_score", 0.0)
        retrieved_chunks = state.get("retrieved_chunks", [])
        litigation = state.get("litigation_data") or {}
        
        # Build citation index from retrieved chunks and litigation
        citation_manager = CitationIndexManager()
        
        # Index patent sources
        for chunk in retrieved_chunks[:20]:
            patent_id = chunk.get("patent_id")
            if not patent_id:
                continue
            citation_manager.add_patent(
                patent_id=patent_id,
                title=chunk.get("title", f"Patent {patent_id}"),
                assignee=chunk.get("assignee"),
                snippet=chunk.get("chunk_text", "")[:300],
                score=chunk.get("score"),
                section_type=chunk.get("chunk_type"),
                chunk_id=chunk.get("chunk_id"),
            )
        
        # Index litigation sources
        if litigation:
            extract_citations_from_litigation(litigation, citation_manager)
        
        # Generate sources header
        sources_header = citation_manager.generate_sources_header()
        
        # Build source index for LLM prompt
        source_lines = []
        for source in citation_manager.get_sources():
            if source.type == "patent":
                source_lines.append(
                    f"[{source.index}] {source.id} ({source.assignee or 'Unknown'}) - {source.section_type or 'patent'}"
                )
                if source.snippet:
                    source_lines.append(f"    Key passage: \"{source.snippet[:200]}...\"")
            else:
                source_lines.append(
                    f"[{source.index}] {source.title} (Litigation) - {source.outcome or 'Pending'}"
                )
                if source.snippet:
                    source_lines.append(f"    Key passage: \"{source.snippet[:200]}...\"")
        source_index_str = "\n".join(source_lines) if source_lines else "No sources indexed"
        
        # Ensure synthesis is a dict
        if synthesis is None:
            synthesis = {}
        
        # Try LLM-powered final formatting
        technical_summary = ""
        legal_summary = None
        novelty_assessment = None
        risk_score = synthesis.get("risk_score", 50) if isinstance(synthesis, dict) else 50
        
        try:
            llm = self._agents.get("synthesis")
            if llm and hasattr(llm, "llm"):
                prompt = get_enhanced_final_prompt(
                    query=query,
                    source_index=source_index_str,
                    synthesis_output=synthesis,
                    litigation_data=litigation,
                )
                
                llm_request = LLMRequest(
                    agent="finalize",
                    user_prompt=prompt,
                    system_prompt=FINAL_OUTPUT_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_tokens=1200,
                    response_format="json",
                )
                
                raw_response = await llm.llm.generate(llm_request, retries=1)
                
                # Parse response
                import json
                try:
                    final_data = json.loads(llm.llm._extract_json(raw_response))
                    technical_summary = final_data.get("technical_summary", "")
                    legal_summary = final_data.get("legal_summary")
                    novelty_assessment = final_data.get("novelty_assessment")
                    if final_data.get("risk_score"):
                        risk_score = final_data.get("risk_score")
                except (json.JSONDecodeError, AttributeError):
                    pass
        except Exception as exc:
            logger.warning("LLM finalize failed, using synthesis output: %s", exc)
        
        # Fallback to synthesis output if LLM failed
        if not technical_summary:
            technical_summary = synthesis.get(
                "technical_summary",
                synthesis.get("executive_summary", "Analysis not available")
            )
            legal_summary = synthesis.get("legal_summary")
        
        # Build API response sources list with expandable UI state
        api_sources = []
        for source in citation_manager.get_sources():
            primary_url = source.urls.get("google_patents") or source.urls.get("justia") or ""
            
            # Get full chunk text for expandable drawer
            full_chunk_text = ""
            if source.type == "patent":
                # Find the full chunk text from retrieved_chunks
                for chunk in retrieved_chunks:
                    if chunk.get("patent_id") == source.id and chunk.get("chunk_id") == source.chunk_id:
                        full_chunk_text = chunk.get("chunk_text", "")
                        break
                # Fallback to snippet if not found
                if not full_chunk_text:
                    full_chunk_text = source.snippet
            
            # Build UI state for clickable/expandable citations
            ui_state = {
                "expandable": True,
                "chunk_text": full_chunk_text or source.snippet,
                "chunk_id": source.chunk_id,
            }
            
            api_sources.append(SourceItem(
                index=source.index,
                type=source.type,
                title=source.title,
                assignee=source.assignee,
                url=primary_url,
                snippet=source.snippet,
                score=source.score,
                outcome=source.outcome,
                risk_level=source.risk_level,
                ui_state=ui_state,
            ).model_dump())
        
        # Build the enhanced API response
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        
        api_response = {
            "response_id": response_id,
            "query": query,
            "sources_header": sources_header,
            "answer_section": {
                "technical_summary": technical_summary,
                "legal_summary": legal_summary,
                "novelty_assessment": novelty_assessment,
            },
            "sources": api_sources,
            "risk_score": risk_score if isinstance(risk_score, int) else int(risk_score),
            "quality_score": critic_score or 0.0,
            "metadata": {
                "total_latency_ms": state.get("total_latency_ms", 0),
                "iterations": state.get("iteration", 0) + 1,
                "agents_executed": list(state.get("agent_outputs", {}).keys()),
                "errors": state.get("errors", []),
            },
        }
        
        # Also include legacy format for backwards compatibility
        final = {
            "query": query,
            "executive_summary": synthesis.get("executive_summary", technical_summary),
            "query_analysis": claims,
            "synthesis": synthesis,
            "quality_score": critic_score or 0.0,
            "retrieved_patents": retrieved_chunks[:20],
            "litigation_data": litigation,
            "api_response": api_response,  # Enhanced format
            "metadata": {
                "total_latency_ms": state.get("total_latency_ms", 0),
                "agents_executed": list(state.get("agent_outputs", {}).keys()),
                "errors": state.get("errors", []),
                "messages": state.get("messages", []),
                "iteration": state.get("iteration", 0),
            },
        }
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Add final node output to agent_outputs
        final_output = AgentOutput(
            name="final",
            success=True,
            data={**final, "latency_ms": latency_ms},
            latency_ms=latency_ms,
        )
        
        return {
            "final_response": final,
            "current_phase": "complete",
            "messages": ["Final response assembled with indexed citations"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "final": final_output,
            },
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }


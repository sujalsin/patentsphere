"""Main LangGraph StateGraph definition for PatentSphere.

This module creates the adaptive agent orchestration graph using LangGraph.
The graph supports:
- Conditional routing based on query analysis
- Adaptive re-retrieval based on quality scores
- Dynamic litigation check inclusion
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END

from app.graph.state import PatentQueryState, create_initial_state
from app.graph.nodes import AgentNodes
from app.graph.router import (
    route_after_claims,
    route_after_retrieval,
    route_after_litigation,
    route_after_synthesis,
    route_after_critic,
    increment_iteration,
)
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class PatentGraph:
    """
    LangGraph-based patent query orchestration.
    
    This class encapsulates the StateGraph and provides methods to
    compile and invoke it.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.nodes = AgentNodes(settings=self.settings)
        self._graph: Optional[StateGraph] = None
        self._compiled = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph with all nodes and edges."""
        
        # Create the graph with our state type
        graph = StateGraph(PatentQueryState)
        
        # Add nodes
        graph.add_node("claims_analyzer", self.nodes.claims_analyzer_node)
        graph.add_node("adaptive_retrieval", self.nodes.retrieval_node)  # Renamed from "retrieval" for clarity
        graph.add_node("litigation", self.nodes.litigation_node)
        graph.add_node("synthesis", self.nodes.synthesis_node)
        graph.add_node("critic", self.nodes.critic_node)
        graph.add_node("finalize", self.nodes.finalize_node)
        graph.add_node("increment_iteration", increment_iteration)
        
        # Set entry point
        graph.set_entry_point("claims_analyzer")
        
        # Add conditional edges
        
        # After claims analysis -> parallel execution (litigation and adaptive_retrieval)
        # Both will execute in parallel automatically
        graph.add_edge("claims_analyzer", "litigation")
        graph.add_edge("claims_analyzer", "adaptive_retrieval")
        
        # After parallel execution -> synthesis (both litigation and adaptive_retrieval route here)
        # LangGraph will wait for both to complete before executing synthesis
        graph.add_edge("litigation", "synthesis")
        graph.add_edge("adaptive_retrieval", "synthesis")
        
        # After synthesis -> critic
        graph.add_conditional_edges(
            "synthesis",
            route_after_synthesis,
            {
                "critic": "critic",
            }
        )
        
        # After critic -> finalize or retry
        graph.add_conditional_edges(
            "critic",
            route_after_critic,
            {
                "finalize": "finalize",
                "retry_retrieval": "increment_iteration",
            }
        )
        
        # After increment -> adaptive_retrieval (retry loop)
        graph.add_edge("increment_iteration", "adaptive_retrieval")
        
        # Finalize -> END
        graph.add_edge("finalize", END)
        
        return graph
    
    def compile(self):
        """Compile the graph for execution."""
        if self._compiled is None:
            self._graph = self._build_graph()
            self._compiled = self._graph.compile()
        return self._compiled
    
    async def invoke(
        self,
        query: str,
        max_iterations: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> PatentQueryState:
        """
        Invoke the graph with a query.
        
        Args:
            query: The user's patent query
            max_iterations: Maximum number of retrieval iterations
            config: Optional LangGraph config
        
        Returns:
            Final state after graph execution
        """
        compiled = self.compile()
        initial_state = create_initial_state(query, max_iterations=max_iterations)
        
        logger.info("Starting patent graph execution for query: %s...", query[:100])
        
        # Invoke the graph
        final_state = await compiled.ainvoke(initial_state, config=config)
        
        logger.info(
            "Graph execution complete. Latency: %.0fms, Iterations: %d",
            final_state.get("total_latency_ms", 0),
            final_state.get("iteration", 0) + 1,
        )
        
        return final_state
    
    async def stream(
        self,
        query: str,
        max_iterations: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Stream graph execution, yielding state updates.
        
        Args:
            query: The user's patent query
            max_iterations: Maximum number of retrieval iterations
            config: Optional LangGraph config
        
        Yields:
            State updates from each node
        """
        compiled = self.compile()
        initial_state = create_initial_state(query, max_iterations=max_iterations)
        
        logger.info("Starting streamed patent graph execution for query: %s...", query[:100])
        
        async for state_update in compiled.astream(initial_state, config=config):
            yield state_update
    
    def get_graph_diagram(self) -> str:
        """
        Get a Mermaid diagram of the graph structure.
        
        Returns:
            Mermaid diagram string
        """
        if self._compiled is None:
            self.compile()
        
        try:
            return self._compiled.get_graph().draw_mermaid()
        except Exception:
            # Fallback to manual diagram if draw_mermaid not available
            return """
graph TD
    START --> claims_analyzer
    claims_analyzer --> litigation
    claims_analyzer --> adaptive_retrieval
    litigation --> synthesis
    adaptive_retrieval --> synthesis
    synthesis --> critic
    critic --> finalize
    critic --> increment_iteration
    increment_iteration --> adaptive_retrieval
    finalize --> END
"""


def create_patent_graph(settings: Optional[Settings] = None) -> PatentGraph:
    """
    Factory function to create a PatentGraph instance.
    
    Args:
        settings: Optional settings override
    
    Returns:
        Configured PatentGraph instance
    """
    return PatentGraph(settings=settings)


# Convenience function for direct invocation
async def run_patent_query(
    query: str,
    settings: Optional[Settings] = None,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Run a patent query through the graph and return results.
    
    Args:
        query: The user's patent query
        settings: Optional settings override
        max_iterations: Maximum retrieval iterations
    
    Returns:
        Final response dictionary
    """
    graph = create_patent_graph(settings=settings)
    final_state = await graph.invoke(query, max_iterations=max_iterations)
    return final_state.get("final_response", {})


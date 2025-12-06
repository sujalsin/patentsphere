"""State definitions for the LangGraph patent query graph.

The state is the shared context that flows through the graph, accumulating
outputs from each agent and informing routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from operator import add


# =============================================================================
# Agent Output Types
# =============================================================================

class AgentOutput(TypedDict, total=False):
    """Output from a single agent execution."""
    agent: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str]
    latency_ms: float


class RouterDecision(TypedDict):
    """Decision from the router about which agents to invoke next."""
    next_agents: List[str]
    skip_agents: List[str]
    reasoning: str
    requires_litigation: bool
    requires_deep_retrieval: bool


# =============================================================================
# Graph State Definition
# =============================================================================

def merge_agent_outputs(
    existing: Dict[str, AgentOutput],
    new: Dict[str, AgentOutput]
) -> Dict[str, AgentOutput]:
    """Merge agent outputs, with new outputs taking precedence."""
    result = existing.copy()
    result.update(new)
    return result


def merge_messages(existing: List[str], new: List[str]) -> List[str]:
    """Append new messages to existing."""
    return existing + new


class PatentQueryState(TypedDict, total=False):
    """
    State that flows through the LangGraph patent query graph.
    
    This state accumulates outputs from each agent and tracks
    routing decisions and execution metadata.
    """
    
    # Input
    query: str
    
    # Routing & Control
    query_type: Literal["emergence", "litigation", "portfolio", "research", "other"]
    router_decision: RouterDecision
    current_phase: Literal["analysis", "retrieval", "synthesis", "evaluation", "complete"]
    iteration: int
    max_iterations: int
    
    # Agent Outputs (accumulated)
    agent_outputs: Annotated[Dict[str, AgentOutput], merge_agent_outputs]
    
    # Specific agent data (for easy access)
    claims_analysis: Optional[Dict[str, Any]]
    citation_results: Optional[Dict[str, Any]]
    litigation_data: Optional[Dict[str, Any]]
    synthesis_output: Optional[Dict[str, Any]]
    critic_score: Optional[float]
    
    # Retrieved data
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_patent_ids: List[str]
    
    # Execution metadata
    messages: Annotated[List[str], merge_messages]
    errors: List[str]
    total_latency_ms: Annotated[float, add]  # Use add reducer for parallel updates
    
    # Final output
    final_response: Optional[Dict[str, Any]]
    
    # Adaptive routing signals
    needs_more_retrieval: bool
    needs_litigation_check: bool
    quality_threshold_met: bool


# =============================================================================
# State Initialization & Utilities
# =============================================================================

def create_initial_state(query: str, max_iterations: int = 3) -> PatentQueryState:
    """Create an initial state for a new query."""
    return PatentQueryState(
        query=query,
        query_type="other",
        router_decision=RouterDecision(
            next_agents=["claims_analyzer"],
            skip_agents=[],
            reasoning="Starting with claims analysis",
            requires_litigation=False,
            requires_deep_retrieval=False,
        ),
        current_phase="analysis",
        iteration=0,
        max_iterations=max_iterations,
        agent_outputs={},
        claims_analysis=None,
        citation_results=None,
        litigation_data=None,
        synthesis_output=None,
        critic_score=None,
        retrieved_chunks=[],
        retrieved_patent_ids=[],
        messages=[],
        errors=[],
        total_latency_ms=0.0,
        final_response=None,
        needs_more_retrieval=False,
        needs_litigation_check=False,
        quality_threshold_met=False,
    )


def state_has_agent_output(state: PatentQueryState, agent_name: str) -> bool:
    """Check if an agent has produced output in this state."""
    return agent_name in state.get("agent_outputs", {})


def get_agent_data(state: PatentQueryState, agent_name: str) -> Optional[Dict[str, Any]]:
    """Get the data from a specific agent's output."""
    outputs = state.get("agent_outputs", {})
    if agent_name in outputs:
        return outputs[agent_name].get("data")
    return None


def is_agent_successful(state: PatentQueryState, agent_name: str) -> bool:
    """Check if an agent completed successfully."""
    outputs = state.get("agent_outputs", {})
    if agent_name in outputs:
        return outputs[agent_name].get("success", False)
    return False


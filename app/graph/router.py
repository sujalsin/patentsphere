"""Adaptive routing logic for the LangGraph patent query graph.

The router makes decisions about which agents to invoke based on:
1. Query type detected by claims analyzer
2. Current state of agent outputs
3. Quality scores from critic
4. Iteration count and adaptive signals
"""

from __future__ import annotations

import logging
from typing import Literal, List

from app.graph.state import PatentQueryState, RouterDecision

logger = logging.getLogger(__name__)


# Route type literals for LangGraph conditional edges
RouteType = Literal[
    "retrieval",
    "adaptive_retrieval",
    "litigation",
    "synthesis",
    "critic",
    "finalize",
    "retry_retrieval",
    "end",
]


def route_after_claims(state: PatentQueryState) -> RouteType:
    """
    Route after claims analysis completes.
    
    Decision logic:
    - Routes to parallel execution (litigation and adaptive_retrieval)
    - This function is kept for compatibility but parallel edges are used instead
    """
    query_type = state.get("query_type", "other")
    claims = state.get("claims_analysis")
    
    if not claims:
        logger.warning("Claims analysis failed, proceeding anyway")
    
    logger.info(
        "Claims analysis complete: query_type=%s, starting parallel execution",
        query_type,
    )
    
    # Note: Parallel execution is handled via direct edges in graph
    # This function is not used in the new parallel architecture
    return "adaptive_retrieval"


def route_after_retrieval(state: PatentQueryState) -> RouteType:
    """
    Route after retrieval completes.
    
    Decision logic:
    - If query type is litigation or query mentions litigation -> litigation
    - Otherwise -> synthesis
    """
    query_type = state.get("query_type", "other")
    needs_litigation = state.get("needs_litigation_check", False)
    
    # Check if we should run litigation scout
    if query_type == "litigation" or needs_litigation or _query_mentions_litigation(state):
        logger.info("Routing to litigation scout")
        return "litigation"
    
    logger.info("Routing to synthesis (skipping litigation)")
    return "synthesis"


def route_after_litigation(state: PatentQueryState) -> RouteType:
    """Route after litigation scout - always proceed to synthesis."""
    return "synthesis"


def route_after_synthesis(state: PatentQueryState) -> RouteType:
    """
    Route after synthesis completes.
    
    Decision logic:
    - If critic is enabled -> critic
    - Otherwise -> finalize
    """
    # Check if we have a critic agent configured
    agent_outputs = state.get("agent_outputs", {})
    
    # We'll check in the graph if critic is enabled
    # For now, always route to critic (it will skip if not enabled)
    return "critic"


def route_after_critic(state: PatentQueryState) -> RouteType:
    """
    Route after critic evaluation.
    
    Decision logic:
    - If quality threshold met -> finalize
    - If quality low and iterations remain -> retry_retrieval
    - Otherwise -> finalize
    """
    quality_met = state.get("quality_threshold_met", True)
    needs_more = state.get("needs_more_retrieval", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if quality_met:
        logger.info("Quality threshold met, finalizing")
        return "finalize"
    
    if needs_more and iteration < max_iterations - 1:
        logger.info(
            "Quality below threshold (iteration %d/%d), retrying retrieval",
            iteration + 1,
            max_iterations,
        )
        return "retry_retrieval"
    
    logger.info("Max iterations reached or no retry needed, finalizing")
    return "finalize"


def should_run_litigation(state: PatentQueryState) -> bool:
    """Determine if litigation scout should run."""
    query_type = state.get("query_type", "other")
    
    # Explicit litigation query type
    if query_type == "litigation":
        return True
    
    # Check for litigation-related keywords in query
    if _query_mentions_litigation(state):
        return True
    
    # Check router decision
    router = state.get("router_decision", {})
    if router.get("requires_litigation", False):
        return True
    
    return False


def should_retry_retrieval(state: PatentQueryState) -> bool:
    """Determine if retrieval should be retried."""
    quality_met = state.get("quality_threshold_met", True)
    needs_more = state.get("needs_more_retrieval", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    return not quality_met and needs_more and iteration < max_iterations - 1


def increment_iteration(state: PatentQueryState) -> dict:
    """Increment iteration counter for retry loops."""
    return {
        "iteration": state.get("iteration", 0) + 1,
        "messages": [f"Starting iteration {state.get('iteration', 0) + 2}"],
    }


def make_routing_decision(state: PatentQueryState) -> RouterDecision:
    """
    Make a comprehensive routing decision based on current state.
    
    This function analyzes the state and determines:
    - Which agents should run next
    - Which agents should be skipped
    - Special requirements (litigation, deep retrieval)
    """
    query_type = state.get("query_type", "other")
    agent_outputs = state.get("agent_outputs", {})
    quality_met = state.get("quality_threshold_met", True)
    
    next_agents: List[str] = []
    skip_agents: List[str] = []
    reasoning_parts: List[str] = []
    
    # Determine phase based on what's completed
    has_claims = "claims_analyzer" in agent_outputs
    has_retrieval = "citation_mapper" in agent_outputs or "adaptive_retrieval" in agent_outputs
    has_synthesis = "synthesis" in agent_outputs
    has_critic = "critic" in agent_outputs
    
    if not has_claims:
        next_agents.append("claims_analyzer")
        reasoning_parts.append("Starting with claims analysis")
    elif not has_retrieval:
        next_agents.append("retrieval")
        reasoning_parts.append("Proceeding to retrieval")
        
        # Check if litigation is needed
        if query_type == "litigation" or _query_mentions_litigation(state):
            next_agents.append("litigation_scout")
            reasoning_parts.append("Litigation query detected")
    elif not has_synthesis:
        next_agents.append("synthesis")
        reasoning_parts.append("Combining results in synthesis")
    elif not has_critic:
        next_agents.append("critic")
        reasoning_parts.append("Evaluating output quality")
    else:
        next_agents.append("finalize")
        reasoning_parts.append("All agents complete")
    
    # Skip decisions
    if query_type not in ("litigation",) and not _query_mentions_litigation(state):
        skip_agents.append("litigation_scout")
    
    return RouterDecision(
        next_agents=next_agents,
        skip_agents=skip_agents,
        reasoning="; ".join(reasoning_parts),
        requires_litigation=query_type == "litigation" or _query_mentions_litigation(state),
        requires_deep_retrieval=not quality_met,
    )


def _query_mentions_litigation(state: PatentQueryState) -> bool:
    """Check if the query mentions litigation-related terms."""
    query = state.get("query", "").lower()
    litigation_terms = [
        "litigation",
        "lawsuit",
        "infringement",
        "sue",
        "court",
        "legal action",
        "patent troll",
        "injunction",
        "damages",
        "settlement",
        "defendant",
        "plaintiff",
    ]
    return any(term in query for term in litigation_terms)


def get_current_phase(state: PatentQueryState) -> str:
    """Determine the current phase of execution."""
    agent_outputs = state.get("agent_outputs", {})
    
    if "synthesis" in agent_outputs:
        if "critic" in agent_outputs:
            return "complete"
        return "evaluation"
    
    if "citation_mapper" in agent_outputs or "adaptive_retrieval" in agent_outputs:
        return "synthesis"
    
    if "claims_analyzer" in agent_outputs:
        return "retrieval"
    
    return "analysis"


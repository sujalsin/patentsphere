"""LangGraph orchestrator for the PatentSphere workflow."""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import operator

from graph.nodes import router_node, extractor_node, synthesizer_node, critic_node
from graph.tools import retrieval_node, enrich_documents_with_metadata


class AgentState(TypedDict):
    """State schema for the agent workflow."""
    query: str
    intent: Optional[str]
    # Technical vs. legal separation
    technical_keywords: Optional[List[str]]
    legal_entities: Optional[List[str]]
    patent_ids: Optional[List[str]]
    # Backward compatibility; may be unused by downstream nodes
    keywords: Optional[List[str]]
    date_range: Optional[Dict[str, str]]
    documents: Annotated[List[Dict[str, Any]], operator.add]
    litigation_context: Annotated[List[Dict[str, Any]], operator.add]
    draft: Optional[str]
    critique: Optional[Dict[str, Any]]
    retry_count: int
    final_response: Optional[str]


def should_retry(state: AgentState) -> str:
    """Conditional edge: decide whether to retry or end."""
    critique = state.get("critique", {})
    retry_count = state.get("retry_count", 0)
    
    status = critique.get("status", "PASS") if isinstance(critique, dict) else "PASS"
    
    if status == "FAIL" and retry_count < 3:
        return "retry"
    return "end"


async def synthesizer_wrapper(state: AgentState) -> AgentState:
    """Wrapper to handle async generator from synthesizer_node."""
    draft_text = state.get("draft", "")
    
    # Collect all updates from the async generator
    async for update in synthesizer_node(state):
        if "draft" in update:
            draft_text = update["draft"]
    
    return {"draft": draft_text}


async def retrieval_with_enrichment(state: AgentState) -> AgentState:
    """Wrapper that delegates to retrieval_node (fan-out already handled there)."""
    return await retrieval_node(state)


def increment_retry_count(state: AgentState) -> AgentState:
    """Increment retry count before retrying."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def finalize_response(state: AgentState) -> AgentState:
    """Finalize the response."""
    return {"final_response": state.get("draft", "")}


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retrieval", retrieval_with_enrichment)
    workflow.add_node("synthesizer", synthesizer_wrapper)
    workflow.add_node("critic", critic_node)
    workflow.add_node("increment_retry", increment_retry_count)
    workflow.add_node("finalize", finalize_response)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add edges
    workflow.add_edge("router", "extractor")
    workflow.add_edge("extractor", "retrieval")
    workflow.add_edge("retrieval", "synthesizer")
    workflow.add_edge("synthesizer", "critic")
    
    # Conditional edge from critic
    workflow.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "increment_retry",
            "end": "finalize",
        },
    )
    
    # Retry loop
    workflow.add_edge("increment_retry", "synthesizer")
    
    # Final edge
    workflow.add_edge("finalize", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Global workflow instance
workflow = create_workflow()


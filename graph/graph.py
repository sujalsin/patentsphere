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
    """Retrieval node with document enrichment."""
    # Run retrieval
    retrieval_result = await retrieval_node(state)
    
    # Enrich documents with metadata
    documents = retrieval_result.get("documents", [])
    enriched_docs = await enrich_documents_with_metadata(documents)
    
    # Update litigation context based on enriched documents
    patent_ids = [doc.get("patent_id", "") for doc in enriched_docs if doc.get("patent_id")]
    litigation_context = retrieval_result.get("litigation_context", [])
    
    # If we have patent IDs, fetch additional litigation data
    if patent_ids and state.get("intent") in ["LEGAL", "BOTH"]:
        from db.postgres_client import postgres_client
        try:
            # Ensure connection is established
            if not postgres_client.pool:
                await postgres_client.connect()
            
            # Normalize patent IDs for database lookup (handle format variations)
            # Database might store IDs in different format (with/without dashes, US prefix, etc.)
            normalized_patent_ids = []
            for pid in patent_ids:
                if pid:
                    # Add original format
                    normalized_patent_ids.append(pid)
                    # Add variations: with/without dashes, with/without US prefix
                    pid_upper = pid.upper()
                    normalized_patent_ids.append(pid_upper)
                    normalized_patent_ids.append(pid_upper.replace("-", ""))
                    normalized_patent_ids.append(pid_upper.replace("/", ""))
                    if pid_upper.startswith("US"):
                        normalized_patent_ids.append(pid_upper[2:])
                        normalized_patent_ids.append(pid_upper[2:].replace("-", ""))
                    else:
                        normalized_patent_ids.append(f"US{pid_upper}")
                        normalized_patent_ids.append(f"US-{pid_upper}")
            
            # Remove duplicates
            unique_patent_ids = list(set([pid for pid in normalized_patent_ids if pid]))
            
            # Fetch litigation for all variations
            additional_litigation = await postgres_client.get_litigation_by_patents(unique_patent_ids)
            
            # Also try with just the original patent IDs
            if not additional_litigation:
                additional_litigation = await postgres_client.get_litigation_by_patents(patent_ids)
            
            # Merge and deduplicate
            seen_cases = {case.get("case_number", "") for case in litigation_context}
            for case in additional_litigation:
                if case.get("case_number", "") not in seen_cases:
                    litigation_context.append(case)
                    seen_cases.add(case.get("case_number", ""))
            
            # Debug: Log if we found litigation
            if litigation_context:
                print(f"DEBUG: Found {len(litigation_context)} litigation case(s) for {len(patent_ids)} patent(s)")
            else:
                print(f"DEBUG: No litigation found for {len(patent_ids)} patent(s) (intent: {state.get('intent')})")
        except Exception as e:
            # PostgreSQL unavailable - continue without additional litigation data
            print(f"Warning: Could not fetch additional litigation data: {e}")
            import traceback
            traceback.print_exc()
    
    return {
        "documents": enriched_docs,
        "litigation_context": litigation_context,
    }


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


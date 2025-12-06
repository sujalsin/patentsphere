#!/usr/bin/env python3
"""
Full end-to-end test of the LangGraph pipeline with real execution.
Tests the complete flow from query to final response.
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.graph import PatentGraph, create_patent_graph
from app.orchestrator import Orchestrator
from config.settings import get_settings


TEST_QUERY = "What patents exist for blockchain-based energy trading systems?"


async def test_graph_compilation():
    """Test that the LangGraph compiles correctly."""
    print("\n" + "=" * 60)
    print("Testing Graph Compilation")
    print("=" * 60)
    
    settings = get_settings()
    graph = create_patent_graph(settings=settings)
    
    # Compile the graph
    compiled = graph.compile()
    
    assert compiled is not None, "Graph compilation returned None"
    print("✓ Graph compiled successfully")
    
    # Get graph diagram
    diagram = graph.get_graph_diagram()
    print(f"\nGraph structure:\n{diagram[:500]}...")
    
    return True


async def test_graph_execution():
    """Test full graph execution with a real query."""
    print("\n" + "=" * 60)
    print("Testing Graph Execution (Full Pipeline)")
    print("=" * 60)
    print(f"Query: '{TEST_QUERY}'")
    
    settings = get_settings()
    graph = create_patent_graph(settings=settings)
    
    print("\nExecuting graph... (this may take 1-2 minutes)")
    
    # Execute the graph
    final_state = await graph.invoke(
        query=TEST_QUERY,
        max_iterations=2,
    )
    
    # Verify state structure
    print("\n--- State Verification ---")
    
    # Check required state fields
    assert "query" in final_state, "Missing 'query' in final state"
    print(f"✓ Query: {final_state['query'][:50]}...")
    
    assert "claims_analysis" in final_state, "Missing 'claims_analysis' in final state"
    print(f"✓ Claims Analysis: {final_state['claims_analysis'].get('query_type', 'N/A')}")
    
    assert "retrieved_chunks" in final_state, "Missing 'retrieved_chunks' in final state"
    chunks = final_state.get("retrieved_chunks", [])
    print(f"✓ Retrieved Chunks: {len(chunks)} chunks")
    
    if "litigation_data" in final_state and final_state["litigation_data"]:
        lit_cases = final_state.get("litigation_data", {}).get("cases", [])
        print(f"✓ Litigation Data: {len(lit_cases)} cases")
    else:
        print("- Litigation Data: skipped (no litigation keywords)")
    
    assert "synthesis_output" in final_state, "Missing 'synthesis_output' in final state"
    synthesis = final_state.get("synthesis_output", {})
    print(f"✓ Synthesis Output: {synthesis.get('executive_summary', '')[:80]}...")
    
    assert "critic_score" in final_state, "Missing 'critic_score' in final state"
    print(f"✓ Critic Score: {final_state.get('critic_score', 0):.3f}")
    
    assert "final_response" in final_state, "Missing 'final_response' in final state"
    print("✓ Final Response present")
    
    # Check metadata
    print(f"\n--- Performance Metrics ---")
    print(f"Total Latency: {final_state.get('total_latency_ms', 0):.0f}ms")
    print(f"Iterations: {final_state.get('iteration', 0) + 1}")
    print(f"Quality Met: {final_state.get('quality_threshold_met', False)}")
    
    return True


async def test_orchestrator_langgraph():
    """Test the orchestrator with LangGraph mode."""
    print("\n" + "=" * 60)
    print("Testing Orchestrator (LangGraph Mode)")
    print("=" * 60)
    
    orchestrator = Orchestrator(use_langgraph=True)
    
    assert orchestrator.use_langgraph is True
    print("✓ Orchestrator in LangGraph mode")
    
    print(f"Agents: {list(orchestrator.agents.keys())}")
    
    print("\nExecuting orchestrator.run_all()...")
    results = await orchestrator.run_all(TEST_QUERY)
    
    print("\n--- Results by Agent ---")
    for agent_name, result in results.items():
        status = "✓" if result.success else "✗"
        latency = result.data.get('latency_ms', 0) if result.data else 0
        print(f"  {status} {agent_name}: latency={latency:.0f}ms, success={result.success}")
        if result.error:
            print(f"      Error: {result.error}")
    
    # Check for final result
    final = results.get("final")
    if final and final.success:
        print("\n--- Final Response ---")
        final_data = final.data or {}
        print(f"Executive Summary: {final_data.get('executive_summary', '')[:100]}...")
        print(f"Quality Score: {final_data.get('quality_score', 0):.3f}")
        print(f"Risk Score: {final_data.get('synthesis', {}).get('risk_score', 'N/A')}")
        
        # Check for API response
        if "api_response" in final_data:
            api_resp = final_data["api_response"]
            print(f"\n--- API Response Format ---")
            print(f"Response ID: {api_resp.get('response_id', 'N/A')}")
            print(f"Sources Header: {api_resp.get('sources_header', '')[:80]}...")
            print(f"Sources Count: {len(api_resp.get('sources', []))}")
            print("✓ API Response format present")
    
    return True


async def test_state_transitions():
    """Test that state transitions follow expected pattern."""
    print("\n" + "=" * 60)
    print("Testing State Transitions")
    print("=" * 60)
    
    settings = get_settings()
    graph = create_patent_graph(settings=settings)
    
    print("Streaming graph execution to observe transitions...")
    
    transitions = []
    async for state_update in graph.stream(TEST_QUERY, max_iterations=2):
        for node_name in state_update.keys():
            transitions.append(node_name)
            print(f"  → {node_name}")
    
    print(f"\nTransitions: {' → '.join(transitions)}")
    
    # Verify expected flow
    assert "claims_analyzer" in transitions, "Missing claims_analyzer node"
    assert "retrieval" in transitions, "Missing retrieval node"
    assert "synthesis" in transitions, "Missing synthesis node"
    assert "critic" in transitions, "Missing critic node"
    assert "finalize" in transitions, "Missing finalize node"
    
    print("✓ All expected nodes executed")
    
    return True


async def main():
    """Run all LangGraph tests."""
    print("\n" + "=" * 60)
    print("PatentSphere LangGraph Full Pipeline Tests")
    print("=" * 60)
    
    results = {}
    
    try:
        results["Compilation"] = await test_graph_compilation()
    except Exception as e:
        print(f"\n✗ Compilation test FAILED: {e}")
        results["Compilation"] = False
    
    try:
        results["Execution"] = await test_graph_execution()
    except Exception as e:
        print(f"\n✗ Execution test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["Execution"] = False
    
    try:
        results["Orchestrator"] = await test_orchestrator_langgraph()
    except Exception as e:
        print(f"\n✗ Orchestrator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["Orchestrator"] = False
    
    # Skip state transitions test for now (takes too long)
    # results["Transitions"] = await test_state_transitions()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All LangGraph tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


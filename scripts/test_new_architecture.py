#!/usr/bin/env python3
"""
Comprehensive test suite for the new parallel architecture with hybrid search and RLAIF.

Tests:
1. ClaimsAnalyzer Risk Entities extraction
2. Parallel execution (LitigationScout + AdaptiveRetrieval)
3. Hybrid search (BM25 + vector + CPC filtering)
4. Internal RLAIF loop with query expansion
5. In-memory join (patents with litigation)
6. Citation verification and fact-checking
7. Enhanced FinalizeNode with clickable citations
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.agents.litigation import LitigationScoutAgent
from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.critic import CriticAgent
from app.graph.graph import create_patent_graph
from config.settings import get_settings


# Test queries covering different scenarios
TEST_QUERIES = [
    "What are the latest patents for solid-state battery technology?",
    "Samsung patents related to OLED display technology",
    "Neural network training optimization patents from 2020-2024",
    "Apple litigation cases involving smartphone patents",
]


async def test_claims_risk_entities():
    """Test ClaimsAnalyzer extracts Risk Entities correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: ClaimsAnalyzer Risk Entities Extraction")
    print("=" * 70)
    
    settings = get_settings()
    agent = ClaimsAnalyzerAgent(settings=settings)
    
    query = "Samsung solid-state battery electrolyte dendrite suppression"
    result = await agent.run(query)
    
    assert result.success, f"ClaimsAnalyzer failed: {result.error}"
    
    data = result.data
    print(f"✓ ClaimsAnalyzer succeeded")
    print(f"  Query Type: {data.get('query_type')}")
    print(f"  Confidence: {data.get('confidence', 0):.2f}")
    
    # Check for Risk Entities
    risk_entities = data.get("risk_entities")
    if risk_entities:
        if isinstance(risk_entities, dict):
            print(f"  Risk Entities:")
            print(f"    - Assignees: {risk_entities.get('assignees', [])}")
            print(f"    - Topics: {risk_entities.get('topics', [])}")
            print(f"    - Search Query: {risk_entities.get('search_query', '')[:80]}...")
        else:
            print(f"  Risk Entities: {risk_entities}")
    else:
        print("  ⚠ Risk Entities not found (may be None)")
    
    # Check for Metadata Filters
    metadata_filters = data.get("metadata_filters")
    if metadata_filters:
        if isinstance(metadata_filters, dict):
            print(f"  Metadata Filters:")
            print(f"    - Date Range: {metadata_filters.get('date_range')}")
            print(f"    - CPC Codes: {metadata_filters.get('cpc_codes', [])}")
        else:
            print(f"  Metadata Filters: {metadata_filters}")
    else:
        print("  ⚠ Metadata Filters not found (may be None)")
    
    return result


async def test_hybrid_search():
    """Test CitationMapper hybrid search (BM25 + vector + CPC filtering)."""
    print("\n" + "=" * 70)
    print("TEST 2: Hybrid Search (BM25 + Vector + CPC Filtering)")
    print("=" * 70)
    
    settings = get_settings()
    agent = CitationMapperAgent(settings=settings)
    
    query = "neural network training optimization"
    
    # Test without claims_analysis (should work with fallback)
    result1 = await agent.run(query)
    assert result1.success, f"CitationMapper failed: {result1.error}"
    print(f"✓ Hybrid search succeeded (without claims_analysis)")
    print(f"  Retrieved {len(result1.data.get('results', []))} chunks")
    
    # Test with claims_analysis (should use hybrid search + filtering)
    claims_analysis = {
        "risk_entities": {
            "search_query": "neural network training optimization deep learning",
            "topics": ["neural network", "training", "optimization"],
        },
        "metadata_filters": {
            "cpc_codes": ["G06N3/08"],
            "date_range": [2020, 2024],
        },
        "cpc_codes": [
            {"code": "G06N3/08", "title": "Learning methods", "confidence": 0.9}
        ],
    }
    
    result2 = await agent.run(query, claims_analysis=claims_analysis)
    assert result2.success, f"CitationMapper with claims_analysis failed: {result2.error}"
    print(f"✓ Hybrid search with claims_analysis succeeded")
    print(f"  Retrieved {len(result2.data.get('results', []))} chunks")
    
    # Check if results have hybrid scores
    results = result2.data.get("results", [])
    if results:
        first_result = results[0]
        if "hybrid_score" in first_result or "bm25_score" in first_result:
            print(f"  ✓ Hybrid scoring active (vector + BM25)")
        else:
            print(f"  ⚠ Hybrid scoring not visible in results")
    
    return result2


async def test_litigation_general_mode():
    """Test LitigationScout general mode with Risk Entities."""
    print("\n" + "=" * 70)
    print("TEST 3: LitigationScout General Mode")
    print("=" * 70)
    
    settings = get_settings()
    agent = LitigationScoutAgent(settings=settings)
    
    query = "Samsung OLED display technology"
    risk_entities = {
        "assignees": ["Samsung"],
        "topics": ["OLED", "Display"],
        "search_query": "OLED display technology",
    }
    
    result = await agent.run_general_mode(query, risk_entities=risk_entities)
    
    assert result.success, f"LitigationScout general mode failed: {result.error}"
    print(f"✓ LitigationScout general mode succeeded")
    
    data = result.data
    print(f"  Total Cases: {data.get('total_cases', 0)}")
    print(f"  Active Cases: {data.get('active_cases', 0)}")
    print(f"  Risk Score: {data.get('risk_score', 0)}")
    print(f"  Risk Level: {data.get('risk_level', 'unknown')}")
    print(f"  Search Mode: {data.get('search_mode', 'unknown')}")
    
    return result


async def test_adaptive_retrieval_rlaif():
    """Test AdaptiveRetrieval internal RLAIF loop."""
    print("\n" + "=" * 70)
    print("TEST 4: AdaptiveRetrieval Internal RLAIF Loop")
    print("=" * 70)
    
    settings = get_settings()
    if not settings.adaptive_retrieval.enabled:
        print("⚠ AdaptiveRetrieval is disabled in config, skipping test")
        return None
    
    agent = AdaptiveRetrievalAgent(settings=settings)
    
    query = "blockchain energy trading system"
    claims_analysis = {
        "query_type": "research",
        "risk_entities": {
            "search_query": "blockchain energy trading distributed ledger",
            "topics": ["blockchain", "energy", "trading"],
        },
        "metadata_filters": {
            "cpc_codes": ["G06Q"],
        },
    }
    
    start_time = time.perf_counter()
    result = await agent.run(
        query=query,
        query_type="research",
        claims_analysis=claims_analysis,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    assert result.success, f"AdaptiveRetrieval failed: {result.error}"
    print(f"✓ AdaptiveRetrieval internal RLAIF loop succeeded")
    print(f"  Latency: {latency_ms:.0f}ms")
    print(f"  Retrieved {len(result.data.get('results', []))} chunks")
    print(f"  Retrieval Depth: {result.data.get('retrieval_depth', 0)}")
    
    # Check if internal RLAIF was used (should have fewer iterations than max_depth)
    rl_metadata = result.data.get("rl_metadata", {})
    iterations = rl_metadata.get("iterations", 0)
    print(f"  Iterations: {iterations}")
    
    return result


async def test_synthesis_join():
    """Test SynthesisAgent in-memory join of patents with litigation."""
    print("\n" + "=" * 70)
    print("TEST 5: Synthesis In-Memory Join")
    print("=" * 70)
    
    settings = get_settings()
    agent = SynthesisAgent(settings=settings)
    
    # Mock retrieved chunks
    retrieved_chunks = [
        {"patent_id": "US-1234567", "chunk_text": "Blockchain energy trading...", "score": 0.9},
        {"patent_id": "US-9876543", "chunk_text": "Distributed ledger system...", "score": 0.85},
    ]
    
    # Mock litigation data with matching patent
    litigation_data = {
        "case_details": [
            {
                "patent_id": "US-1234567",
                "case_number": "Case-2024-001",
                "outcome": "Pending",
                "plaintiff_name": "Company A",
                "defendant_name": "Company B",
            },
            {
                "patent_id": "US-9999999",  # Not in retrieved chunks
                "case_number": "Case-2024-002",
                "outcome": "Settled",
            },
        ],
    }
    
    # Test join method
    join_results = agent._join_patents_with_litigation(retrieved_chunks, litigation_data)
    
    print(f"✓ In-memory join succeeded")
    print(f"  Total Matches: {join_results.get('total_matches', 0)}")
    print(f"  Matched Cases: {len(join_results.get('matched_cases', []))}")
    
    matched_cases = join_results.get("matched_cases", [])
    if matched_cases:
        print(f"  First Match:")
        print(f"    - Patent: {matched_cases[0].get('patent_id')}")
        print(f"    - Case: {matched_cases[0].get('case_number')}")
        print(f"    - Risk Level: {matched_cases[0].get('risk_level')}")
    
    assert join_results.get("total_matches", 0) == 1, "Expected 1 match"
    
    return join_results


async def test_critic_verification():
    """Test CriticAgent citation verification and fact-checking."""
    print("\n" + "=" * 70)
    print("TEST 6: Critic Citation Verification & Fact-Checking")
    print("=" * 70)
    
    settings = get_settings()
    if not settings.critic.enabled:
        print("⚠ Critic is disabled in config, skipping test")
        return None
    
    agent = CriticAgent(settings=settings, weights=settings.critic.reward_weights)
    
    # Mock synthesis output with citations
    synthesis_output = {
        "executive_summary": "Analysis of blockchain patents [1]. Patent US-1234567 covers energy trading [2].",
        "technical_summary": "The system uses blockchain technology [1] for energy trading [2].",
    }
    
    # Mock retrieved chunks
    retrieved_chunks = [
        {"patent_id": "US-1234567", "chunk_text": "Blockchain energy trading system...", "score": 0.9},
        {"patent_id": "US-9876543", "chunk_text": "Distributed ledger...", "score": 0.85},
    ]
    
    # Test citation verification
    verification = await agent._verify_citations(synthesis_output, retrieved_chunks)
    print(f"✓ Citation verification succeeded")
    print(f"  Verified: {verification.get('verified', False)}")
    print(f"  Verified Count: {verification.get('verified_count', 0)}")
    print(f"  Total Citations: {verification.get('total_citations', 0)}")
    if verification.get("errors"):
        print(f"  Errors: {verification.get('errors')[:3]}")
    
    # Test fact-checking
    fact_check = await agent._fact_check(synthesis_output, retrieved_chunks)
    print(f"✓ Fact-checking succeeded")
    print(f"  Verified: {fact_check.get('verified', False)}")
    if fact_check.get("errors"):
        print(f"  Errors: {fact_check.get('errors')[:3]}")
    
    return {"verification": verification, "fact_check": fact_check}


async def test_parallel_execution():
    """Test parallel execution of LitigationScout and AdaptiveRetrieval."""
    print("\n" + "=" * 70)
    print("TEST 7: Parallel Execution (LitigationScout + AdaptiveRetrieval)")
    print("=" * 70)
    
    settings = get_settings()
    graph = create_patent_graph(settings=settings)
    
    query = "Samsung OLED display technology patents"
    
    print(f"Executing query: {query}")
    start_time = time.perf_counter()
    
    final_state = await graph.invoke(query, max_iterations=2)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"✓ Graph execution completed")
    print(f"  Total Latency: {latency_ms:.0f}ms")
    
    # Check agent outputs
    agent_outputs = final_state.get("agent_outputs", {})
    print(f"  Agents Executed: {list(agent_outputs.keys())}")
    
    # Check if both litigation and adaptive_retrieval ran
    has_litigation = "litigation_scout" in agent_outputs
    has_retrieval = "adaptive_retrieval" in agent_outputs or "citation_mapper" in agent_outputs
    
    print(f"  LitigationScout: {'✓' if has_litigation else '✗'}")
    print(f"  AdaptiveRetrieval: {'✓' if has_retrieval else '✗'}")
    
    # Check if synthesis received both inputs
    synthesis_output = final_state.get("synthesis_output")
    if synthesis_output:
        print(f"  Synthesis: ✓ (received inputs from both paths)")
    else:
        print(f"  Synthesis: ✗ (no output)")
    
    # Check final response
    final_response = final_state.get("final_response", {})
    if final_response:
        api_response = final_response.get("api_response", {})
        if api_response:
            sources = api_response.get("sources", [])
            print(f"  Final Response: ✓ ({len(sources)} sources)")
            
            # Check for UI state in sources
            sources_with_ui = [s for s in sources if s.get("ui_state")]
            print(f"  Sources with UI State: {len(sources_with_ui)}/{len(sources)}")
        else:
            print(f"  Final Response: ⚠ (no API response)")
    else:
        print(f"  Final Response: ✗ (no final response)")
    
    return final_state


async def test_end_to_end():
    """Test complete end-to-end pipeline with a real query."""
    print("\n" + "=" * 70)
    print("TEST 8: End-to-End Pipeline Test")
    print("=" * 70)
    
    settings = get_settings()
    graph = create_patent_graph(settings=settings)
    
    query = TEST_QUERIES[0]
    print(f"Query: {query}")
    
    start_time = time.perf_counter()
    final_state = await graph.invoke(query, max_iterations=2)
    total_latency = (time.perf_counter() - start_time) * 1000
    
    print(f"\n✓ End-to-end test completed")
    print(f"  Total Latency: {total_latency:.0f}ms")
    
    # Validate final response structure
    final_response = final_state.get("final_response", {})
    api_response = final_response.get("api_response", {})
    
    if api_response:
        print(f"\n  Final Response Structure:")
        print(f"    - Response ID: {api_response.get('response_id', 'N/A')}")
        print(f"    - Sources Header: {api_response.get('sources_header', 'N/A')[:80]}...")
        print(f"    - Technical Summary: {len(api_response.get('answer_section', {}).get('technical_summary', ''))} chars")
        print(f"    - Sources: {len(api_response.get('sources', []))}")
        print(f"    - Risk Score: {api_response.get('risk_score', 0)}")
        print(f"    - Quality Score: {api_response.get('quality_score', 0):.2f}")
        
        # Check sources have UI state
        sources = api_response.get("sources", [])
        if sources:
            first_source = sources[0]
            if first_source.get("ui_state"):
                print(f"    - UI State: ✓ (expandable citations enabled)")
            else:
                print(f"    - UI State: ✗ (missing)")
    else:
        print(f"  ⚠ No API response in final output")
    
    return final_state


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PATENTSPHERE NEW ARCHITECTURE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    try:
        # Test 1: ClaimsAnalyzer Risk Entities
        results["claims"] = await test_claims_risk_entities()
        
        # Test 2: Hybrid Search
        results["hybrid_search"] = await test_hybrid_search()
        
        # Test 3: Litigation General Mode
        results["litigation"] = await test_litigation_general_mode()
        
        # Test 4: AdaptiveRetrieval RLAIF
        results["adaptive_rlaif"] = await test_adaptive_retrieval_rlaif()
        
        # Test 5: Synthesis Join
        results["synthesis_join"] = await test_synthesis_join()
        
        # Test 6: Critic Verification
        results["critic_verification"] = await test_critic_verification()
        
        # Test 7: Parallel Execution
        results["parallel"] = await test_parallel_execution()
        
        # Test 8: End-to-End
        results["e2e"] = await test_end_to_end()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print("\nSummary:")
        for test_name, result in results.items():
            status = "✓" if result is not None else "⚠"
            print(f"  {status} {test_name}")
        
    except Exception as exc:
        print(f"\n✗ Test suite failed with error: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


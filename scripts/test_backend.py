"""Comprehensive backend testing script."""
import asyncio
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.postgres_client import postgres_client
from db.qdrant_client import qdrant_client
from graph.graph import workflow, AgentState
from config import settings


async def test_database_connections():
    """Test database connections."""
    print("=" * 60)
    print("🔌 Testing Database Connections")
    print("=" * 60)
    
    # Test PostgreSQL
    try:
        await postgres_client.connect()
        result = await postgres_client.fetch_one("SELECT 1 as test")
        assert result["test"] == 1
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False
    
    # Test Qdrant
    try:
        await qdrant_client.create_collection_if_not_exists()
        print("✅ Qdrant connection successful")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False
    
    print()
    return True


async def test_data_ingestion():
    """Test that data was ingested properly."""
    print("=" * 60)
    print("📊 Testing Data Ingestion")
    print("=" * 60)
    
    # Check PostgreSQL
    try:
        patent_count = await postgres_client.fetch_one(
            "SELECT COUNT(*) as count FROM patents_metadata"
        )
        patent_count = patent_count["count"] if patent_count else 0
        print(f"✅ PostgreSQL: {patent_count} patents in database")
        
        litigation_count = await postgres_client.fetch_one(
            "SELECT COUNT(*) as count FROM litigation_cases"
        )
        litigation_count = litigation_count["count"] if litigation_count else 0
        print(f"✅ PostgreSQL: {litigation_count} litigation cases in database")
        
        if patent_count == 0:
            print("⚠️  Warning: No patents found. Run ingestion scripts first.")
            return False
    except Exception as e:
        print(f"❌ PostgreSQL data check failed: {e}")
        return False
    
    # Check Qdrant
    try:
        collections = await qdrant_client.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if settings.qdrant_collection in collection_names:
            collection_info = await qdrant_client.client.get_collection(
                settings.qdrant_collection
            )
            points_count = collection_info.points_count
            print(f"✅ Qdrant: {points_count} points in collection")
        else:
            print("⚠️  Warning: Qdrant collection not found. Run ingestion scripts first.")
            return False
    except Exception as e:
        print(f"❌ Qdrant data check failed: {e}")
        return False
    
    print()
    return True


async def test_workflow_query(query: str, query_type: str, expected_intent: str = None):
    """Test a single query through the workflow."""
    print(f"\n{'=' * 60}")
    print(f"🔍 Testing Query ({query_type}): {query[:60]}...")
    print("=" * 60)
    
    initial_state: AgentState = {
        "query": query,
        "intent": None,
        "keywords": None,
        "date_range": None,
        "documents": [],
        "litigation_context": [],
        "draft": None,
        "critique": None,
        "retry_count": 0,
        "final_response": None,
    }
    
    config = {"configurable": {"thread_id": f"test_{query_type}"}}
    
    try:
        # Run workflow
        final_state = None
        async for state in workflow.astream(initial_state, config=config, stream_mode="values"):
            final_state = state
        
        if not final_state:
            print("❌ Workflow returned no state")
            return False
        
        # Check intent
        intent = final_state.get("intent")
        if expected_intent:
            if intent != expected_intent:
                print(f"⚠️  Intent mismatch: expected {expected_intent}, got {intent}")
            else:
                print(f"✅ Intent correctly identified: {intent}")
        else:
            print(f"✅ Intent: {intent}")
        
        # Check keywords
        keywords = final_state.get("keywords", [])
        print(f"✅ Extracted {len(keywords)} search variations:")
        for i, kw in enumerate(keywords[:5], 1):
            print(f"   {i}. {kw}")
        
        # Check documents
        documents = final_state.get("documents", [])
        print(f"✅ Retrieved {len(documents)} documents")
        
        # Check litigation context
        litigation = final_state.get("litigation_context", [])
        if litigation:
            print(f"✅ Found {len(litigation)} litigation cases")
        
        # Check final response
        final_response = final_state.get("final_response", "")
        if final_response:
            print(f"✅ Generated response ({len(final_response)} characters)")
            print(f"\n📄 Response Preview:")
            print("-" * 60)
            print(final_response[:500] + "..." if len(final_response) > 500 else final_response)
            print("-" * 60)
            
            # Check for citations
            import re
            citations = re.findall(r'\[\[([^\]]+)\]\]', final_response)
            if citations:
                print(f"✅ Found {len(citations)} citations in response")
                for citation in citations[:3]:
                    print(f"   - {citation}")
            else:
                print("⚠️  No citations found in response")
        else:
            print("❌ No final response generated")
            return False
        
        # Check retry count
        retry_count = final_state.get("retry_count", 0)
        if retry_count > 0:
            print(f"ℹ️  Workflow retried {retry_count} time(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_different_query_types():
    """Test different types of queries."""
    print("\n" + "=" * 60)
    print("🧪 Testing Different Query Types")
    print("=" * 60)
    
    test_queries = [
        {
            "query": "Has Apple sued anyone over touchscreens?",
            "type": "LEGAL",
            "expected_intent": "LEGAL",
        },
        {
            "query": "Find prior art for transformer neural networks",
            "type": "TECHNICAL",
            "expected_intent": "TECHNICAL",
        },
        {
            "query": "What patents cover self-driving car sensors and have they been litigated?",
            "type": "BOTH",
            "expected_intent": "BOTH",
        },
        {
            "query": "car suspension systems",
            "type": "TECHNICAL",
            "expected_intent": "TECHNICAL",
        },
    ]
    
    results = []
    for test_case in test_queries:
        success = await test_workflow_query(
            test_case["query"],
            test_case["type"],
            test_case["expected_intent"],
        )
        results.append((test_case["type"], success))
        await asyncio.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    for query_type, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {query_type} query")
    
    all_passed = all(success for _, success in results)
    return all_passed


async def test_hybrid_search():
    """Test hybrid search functionality."""
    print("\n" + "=" * 60)
    print("🔍 Testing Hybrid Search")
    print("=" * 60)
    
    try:
        # Test dense search
        results = await qdrant_client.hybrid_search("transformer neural network", limit=5)
        print(f"✅ Hybrid search returned {len(results)} results")
        
        if results:
            print("\nTop result:")
            top = results[0]
            print(f"  Patent ID: {top.get('patent_id', 'N/A')}")
            print(f"  Title: {top.get('title', 'N/A')[:80]}...")
            print(f"  Score: {top.get('score', 0):.4f}")
            print(f"  Source: {top.get('source', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_citation_verification():
    """Test citation verification in critic."""
    print("\n" + "=" * 60)
    print("✅ Testing Citation Verification")
    print("=" * 60)
    
    from graph.nodes import critic_node
    
    # Test with valid citation
    state_valid = {
        "draft": "The system is described in [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1).",
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Test Patent",
                "abstract": "Test",
            }
        ],
    }
    
    result = await critic_node(state_valid)
    critique = result.get("critique", {})
    status = critique.get("status", "")
    
    if status == "PASS":
        print("✅ Valid citation passed verification")
    else:
        print(f"⚠️  Valid citation got status: {status}")
    
    # Test with invalid citation
    state_invalid = {
        "draft": "The system is described in [[US9999999]](url).",
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Test Patent",
                "abstract": "Test",
            }
        ],
    }
    
    result2 = await critic_node(state_invalid)
    critique2 = result2.get("critique", {})
    status2 = critique2.get("status", "")
    
    if status2 == "FAIL":
        print("✅ Invalid citation correctly rejected")
    else:
        print(f"⚠️  Invalid citation got status: {status2}")
    
    return True


async def run_all_tests():
    """Run all backend tests."""
    print("\n" + "🚀 Starting Backend Tests" + "\n")
    
    tests = [
        ("Database Connections", test_database_connections),
        ("Data Ingestion", test_data_ingestion),
        ("Hybrid Search", test_hybrid_search),
        ("Citation Verification", test_citation_verification),
        ("Different Query Types", test_different_query_types),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Cleanup
    await postgres_client.close()
    await qdrant_client.close()
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)



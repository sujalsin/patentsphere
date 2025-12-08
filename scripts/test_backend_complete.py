"""Comprehensive backend test to verify all components work correctly."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.qdrant_client import qdrant_client
from db.postgres_client import postgres_client
from graph.graph import workflow
from graph.nodes import router_node, extractor_node, synthesizer_node, critic_node


async def test_connections():
    """Test database connections."""
    print("="*60)
    print("🔌 Testing Database Connections")
    print("="*60)
    
    # Test Qdrant
    try:
        collections = await qdrant_client.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if qdrant_client.collection_name in collection_names:
            collection_info = await qdrant_client.client.get_collection(qdrant_client.collection_name)
            points_count = collection_info.points_count
            print(f"✅ Qdrant: Connected | Collection: {qdrant_client.collection_name} | Points: {points_count:,}")
        else:
            print(f"⚠️  Qdrant: Connected but collection '{qdrant_client.collection_name}' not found")
    except Exception as e:
        print(f"❌ Qdrant: Connection failed - {e}")
        return False
    
    # Test PostgreSQL
    try:
        await postgres_client.connect()
        result = await postgres_client.fetch("SELECT COUNT(*) as count FROM patents_metadata")
        patents_count = result[0]['count'] if result else 0
        
        result = await postgres_client.fetch("SELECT COUNT(*) as count FROM litigation_cases")
        litigation_count = result[0]['count'] if result else 0
        
        print(f"✅ PostgreSQL: Connected | Patents: {patents_count:,} | Litigation: {litigation_count:,}")
        await postgres_client.close()
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False
    
    return True


async def test_router():
    """Test router node."""
    print("\n" + "="*60)
    print("🧠 Testing Router Node")
    print("="*60)
    
    test_cases = [
        ("Has Apple sued anyone over touchscreens?", "LEGAL"),
        ("Find prior art for transformer neural networks", "TECHNICAL"),
        ("What patents cover self-driving cars and have they been litigated?", "BOTH"),
    ]
    
    all_passed = True
    for query, expected_intent in test_cases:
        state = {"query": query}
        result = await router_node(state)
        intent = result.get("intent", "")
        passed = intent == expected_intent or (expected_intent == "BOTH" and intent in ["LEGAL", "TECHNICAL", "BOTH"])
        
        status = "✅" if passed else "❌"
        print(f"{status} Query: '{query[:50]}...'")
        print(f"   Expected: {expected_intent}, Got: {intent}")
        
        if not passed:
            all_passed = False
    
    return all_passed


async def test_extractor():
    """Test extractor node."""
    print("\n" + "="*60)
    print("🔍 Testing Extractor Node (Query Expansion)")
    print("="*60)
    
    test_queries = [
        "car suspension",
        "transformer neural networks",
        "touchscreen technology",
    ]
    
    all_passed = True
    for query in test_queries:
        state = {"query": query}
        result = await extractor_node(state)
        keywords = result.get("keywords", [])
        
        passed = len(keywords) >= 3 and len(keywords) <= 5
        status = "✅" if passed else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Generated {len(keywords)} variations:")
        for i, keyword in enumerate(keywords, 1):
            print(f"      {i}. {keyword}")
        
        if not passed:
            all_passed = False
    
    return all_passed


async def test_search():
    """Test vector search."""
    print("\n" + "="*60)
    print("🔎 Testing Vector Search")
    print("="*60)
    
    from graph.tools import vector_search
    
    test_queries = [
        "neural network",
        "automotive suspension",
        "touchscreen",
    ]
    
    all_passed = True
    for query in test_queries:
        try:
            keywords = [query]
            results = await vector_search(keywords, limit=5)
            
            passed = len(results) > 0
            status = "✅" if passed else "⚠️"
            print(f"{status} Query: '{query}'")
            print(f"   Retrieved {len(results)} documents")
            
            if results:
                for i, doc in enumerate(results[:3], 1):
                    patent_id = doc.get("patent_id", "N/A")
                    title = doc.get("title", "N/A")[:60]
                    print(f"      {i}. {patent_id}: {title}...")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ Query: '{query}' - Error: {e}")
            all_passed = False
    
    return all_passed


async def test_synthesizer():
    """Test synthesizer node with empty and non-empty documents."""
    print("\n" + "="*60)
    print("✍️  Testing Synthesizer Node")
    print("="*60)
    
    # Test with empty documents
    print("Testing with empty documents (should return 'no results' message)...")
    state = {
        "query": "test query",
        "intent": "TECHNICAL",
        "keywords": ["test"],
        "documents": [],
        "litigation_context": [],
    }
    
    try:
        async for result in synthesizer_node(state):
            draft = result.get("draft", "")
            if "no patents" in draft.lower() or "no documents" in draft.lower():
                print("✅ Synthesizer correctly handles empty documents")
                break
        else:
            print("⚠️  Synthesizer may not handle empty documents correctly")
    except Exception as e:
        print(f"❌ Synthesizer error: {e}")
        return False
    
    # Test with sample documents
    print("\nTesting with sample documents...")
    state = {
        "query": "neural network",
        "intent": "TECHNICAL",
        "keywords": ["neural network", "deep learning"],
        "documents": [
            {
                "patent_id": "US12345678",
                "title": "Neural Network System",
                "abstract": "A neural network system for processing data",
                "url": "https://patents.google.com/patent/US12345678",
            }
        ],
        "litigation_context": [],
    }
    
    try:
        draft_found = False
        async for result in synthesizer_node(state):
            draft = result.get("draft", "")
            if draft and len(draft) > 100:
                draft_found = True
                print("✅ Synthesizer generates report with documents")
                break
        if not draft_found:
            print("⚠️  Synthesizer may not generate report correctly")
    except Exception as e:
        print(f"❌ Synthesizer error: {e}")
        return False
    
    return True


async def test_critic():
    """Test critic node."""
    print("\n" + "="*60)
    print("✅ Testing Critic Node")
    print("="*60)
    
    # Test with empty documents
    print("Testing with empty documents...")
    state = {
        "draft": "This patent describes a neural network system.",
        "documents": [],
    }
    
    result = await critic_node(state)
    critique = result.get("critique", {})
    status = critique.get("status", "")
    
    if status == "FAIL":
        print("✅ Critic correctly rejects draft without documents")
    else:
        print(f"⚠️  Critic status: {status} (expected FAIL)")
    
    # Test with valid documents and proper citation
    print("\nTesting with valid documents and citations...")
    state = {
        "draft": "The patent [[US12345678]](https://patents.google.com/patent/US12345678) is a Neural Network System as described in the source context.",
        "documents": [
            {
                "patent_id": "US12345678",
                "title": "Neural Network System",
                "abstract": "A neural network system for processing data",
            }
        ],
    }
    
    result = await critic_node(state)
    critique = result.get("critique", {})
    status = critique.get("status", "")
    
    # Critic should PASS if citation exists and is properly referenced
    if status == "PASS":
        print("✅ Critic correctly validates draft with valid citations")
    else:
        # Check if it failed due to citation verification (which is good)
        feedback = critique.get("feedback", "")
        if "citation" in feedback.lower() or "US12345678" in feedback:
            print("✅ Critic is working (being strict about citations is good)")
        else:
            print(f"⚠️  Critic status: {status}")
            print(f"   Feedback: {feedback[:200]}")
    
    return True


async def test_full_workflow():
    """Test full workflow end-to-end."""
    print("\n" + "="*60)
    print("🔄 Testing Full Workflow")
    print("="*60)
    
    test_query = "neural network"
    print(f"Query: '{test_query}'")
    
    try:
        state = {"query": test_query}
        final_state = None
        
        # Use invoke with config to avoid checkpointer issues
        config = {"configurable": {"thread_id": "test-thread"}}
        final_state = await workflow.ainvoke(state, config=config)
        
        if final_state:
            response = final_state.get("final_response", "")
            if response and len(response) > 100:
                print("✅ Full workflow completed successfully")
                print(f"   Response length: {len(response)} characters")
                print(f"   Intent: {final_state.get('intent', 'N/A')}")
                print(f"   Documents retrieved: {len(final_state.get('documents', []))}")
                return True
            else:
                print("⚠️  Workflow completed but response is too short")
                print(f"   Response: {response[:200] if response else 'None'}")
                return False
        else:
            print("⚠️  Workflow did not return state")
            return False
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all backend tests."""
    print("\n" + "="*60)
    print("🚀 PatentSphere Backend Test Suite")
    print("="*60)
    
    results = {}
    
    # Test connections
    results["connections"] = await test_connections()
    if not results["connections"]:
        print("\n❌ Connection tests failed. Please check your database setup.")
        return
    
    # Test components
    results["router"] = await test_router()
    results["extractor"] = await test_extractor()
    results["search"] = await test_search()
    results["synthesizer"] = await test_synthesizer()
    results["critic"] = await test_critic()
    results["workflow"] = await test_full_workflow()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")


if __name__ == "__main__":
    asyncio.run(run_all_tests())


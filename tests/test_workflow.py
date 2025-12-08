"""Test the PatentSphere workflow."""
import asyncio
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph import workflow, AgentState
from graph.nodes import router_node, extractor_node, critic_node


@pytest.mark.asyncio
async def test_routing():
    """Test that routing correctly identifies LEGAL queries."""
    print("🧪 Testing Router Node...")
    
    state = {"query": "Has Apple sued anyone over touchscreens?"}
    result = await router_node(state)
    
    intent = result.get("intent", "")
    assert intent == "LEGAL", f"Expected LEGAL, got {intent}"
    print(f"  ✅ Router correctly identified intent: {intent}")


@pytest.mark.asyncio
async def test_query_expansion():
    """Test that extractor generates 3-5 query variations."""
    print("🧪 Testing Extractor Node (Query Expansion)...")
    
    state = {"query": "car suspension"}
    result = await extractor_node(state)
    
    keywords = result.get("keywords", [])
    assert len(keywords) >= 3, f"Expected at least 3 keywords, got {len(keywords)}"
    assert len(keywords) <= 5, f"Expected at most 5 keywords, got {len(keywords)}"
    
    print(f"  ✅ Extractor generated {len(keywords)} variations:")
    for i, keyword in enumerate(keywords, 1):
        print(f"     {i}. {keyword}")


@pytest.mark.asyncio
async def test_citation_verification():
    """Test that critic rejects drafts with hallucinated citations."""
    print("🧪 Testing Critic Node (Citation Verification)...")
    
    # Create a draft with a citation that doesn't exist in context
    state = {
        "draft": "The transformer architecture is described in [[US9999999]](https://patents.google.com/patent/US9999999).",
        "documents": [
            {
                "patent_id": "US1234567",
                "title": "Some other patent",
                "abstract": "Different content",
            }
        ],
    }
    
    result = await critic_node(state)
    critique = result.get("critique", {})
    status = critique.get("status", "")
    
    # Should fail because US9999999 is not in context
    assert status == "FAIL", f"Expected FAIL for hallucinated citation, got {status}"
    print(f"  ✅ Critic correctly rejected draft with missing citation")
    print(f"     Feedback: {critique.get('feedback', '')[:100]}...")


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow since it makes real LLM calls
@pytest.mark.isolated  # Run this test in isolation to avoid event loop conflicts
async def test_rlaf_loop():
    """Test that RLAIF loop works correctly.
    
    Note: This test makes real LLM calls and may have event loop issues when run
    with other tests. Run it separately with: pytest tests/test_workflow.py::test_rlaf_loop
    
    WARNING: This test requires Ollama to be running and may take 30-60 seconds.
    """
    import asyncio
    
    print("🧪 Testing RLAIF Loop...")
    print("   ⚠️  This test makes real LLM calls - may take 30-60 seconds...")
    
    initial_state: AgentState = {
        "query": "test query",
        "intent": None,
        "keywords": None,
        "date_range": None,
        "documents": [
            {
                "patent_id": "US1234567",
                "title": "Test Patent",
                "abstract": "Test abstract",
                "url": "https://patents.google.com/patent/US1234567",
            }
        ],
        "litigation_context": [],
        "draft": "This is a test with citation [[US9999999]](url).",  # Bad citation
        "critique": None,
        "retry_count": 0,
        "final_response": None,
    }
    
    try:
        # First critique should fail - add timeout
        print("   📝 Running first critique (expecting FAIL)...")
        try:
            result = await asyncio.wait_for(critic_node(initial_state), timeout=60.0)
        except asyncio.TimeoutError:
            pytest.skip("LLM call timed out - Ollama may be slow or unresponsive")
        
        critique = result.get("critique", {})
        status = critique.get("status", "")
        
        assert status == "FAIL", "First critique should fail"
        print(f"  ✅ First critique failed as expected (retry_count: 0)")
        
        # Simulate retry
        initial_state["retry_count"] = 1
        initial_state["critique"] = critique
        initial_state["draft"] = "This is a corrected test with citation [[US1234567]](https://patents.google.com/patent/US1234567)."
        
        print("   📝 Running second critique (expecting PASS)...")
        try:
            result2 = await asyncio.wait_for(critic_node(initial_state), timeout=60.0)
        except asyncio.TimeoutError:
            pytest.skip("LLM call timed out - Ollama may be slow or unresponsive")
        
        critique2 = result2.get("critique", {})
        status2 = critique2.get("status", "")
        
        assert status2 == "PASS", "Second critique should pass with correct citation"
        print(f"  ✅ Second critique passed after correction (retry_count: 1)")
    except Exception as e:
        # Log the error for debugging
        print(f"  ❌ Test error: {type(e).__name__}: {e}")
        raise
    finally:
        # Give async HTTP connections time to close properly
        # This prevents "Event loop is closed" errors
        try:
            await asyncio.sleep(0.2)
            # Cancel any pending tasks to ensure clean shutdown
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if pending:
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    # Wait briefly for cancellations
                    await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            # Ignore cleanup errors
            pass


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("🧪 PatentSphere Workflow Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Routing", test_routing),
        ("Query Expansion", test_query_expansion),
        ("Citation Verification", test_citation_verification),
        ("RLAIF Loop", test_rlaf_loop),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ❌ Test failed: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test error: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


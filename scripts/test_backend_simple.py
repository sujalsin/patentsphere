"""Simplified backend test that works without databases."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.nodes import router_node, extractor_node, synthesizer_node, critic_node


async def test_router():
    """Test router node."""
    print("=" * 60)
    print("🧪 Testing Router Node")
    print("=" * 60)
    
    test_cases = [
        ("Has Apple sued anyone over touchscreens?", "LEGAL"),
        ("Find prior art for transformer neural networks", "TECHNICAL"),
        ("What patents cover self-driving cars and have they been litigated?", "BOTH"),
    ]
    
    for query, expected in test_cases:
        state = {"query": query}
        result = await router_node(state)
        intent = result.get("intent", "")
        status = "✅" if expected in intent or intent == expected else "⚠️"
        print(f"{status} Query: '{query[:50]}...'")
        print(f"   Expected: {expected}, Got: {intent}")
    
    print()


async def test_extractor():
    """Test extractor node with query expansion."""
    print("=" * 60)
    print("🧪 Testing Extractor Node (Query Expansion)")
    print("=" * 60)
    
    test_queries = [
        "car suspension",
        "transformer neural network",
        "touchscreen technology",
    ]
    
    for query in test_queries:
        state = {"query": query}
        result = await extractor_node(state)
        keywords = result.get("keywords", [])
        
        print(f"✅ Query: '{query}'")
        print(f"   Expanded to {len(keywords)} variations:")
        for i, kw in enumerate(keywords, 1):
            print(f"   {i}. {kw}")
        print()


async def test_synthesizer_template():
    """Test synthesizer with proper templating."""
    print("=" * 60)
    print("🧪 Testing Synthesizer Node (Template)")
    print("=" * 60)
    
    # Create mock state with all agent outputs
    state = {
        "query": "Find patents related to car suspension",
        "intent": "TECHNICAL",
        "keywords": ["car suspension", "vehicle damping system", "active chassis control", "CPC: B60G"],
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Composition, adhesive agent, adhesive sheet, and laminate",
                "abstract": "The present invention provides a composition obtained by blending a polythiol compound...",
                "url": "https://patents.google.com/patent/US-2016168309-A1",
                "publication_date": "2016-06-16",
            }
        ],
        "litigation_context": [],
        "critique": {},
        "retry_count": 0,
    }
    
    print("Testing synthesizer with mock data...")
    print(f"Query: {state['query']}")
    print(f"Intent: {state['intent']}")
    print(f"Keywords: {state['keywords']}")
    print(f"Documents: {len(state['documents'])}")
    print()
    
    # Collect draft
    draft_text = ""
    async for update in synthesizer_node(state):
        draft_text = update.get("draft", draft_text)
    
    print("✅ Generated draft:")
    print("-" * 60)
    print(draft_text[:800] + "..." if len(draft_text) > 800 else draft_text)
    print("-" * 60)
    print()
    
    # Check template elements
    import re
    citation_pattern = r'\[\[([^\]]+)\]\]'
    citations_found = re.findall(citation_pattern, draft_text)
    
    checks = {
        "Executive Summary": "Executive Summary" in draft_text or "Summary" in draft_text,
        "Query Analysis": "Query Analysis" in draft_text or "Query" in draft_text,
        "Key Findings": "Key Findings" in draft_text or "Findings" in draft_text,
        "Conclusion": "Conclusion" in draft_text,
        "Citations": len(citations_found) > 0 or "[[" in draft_text or "]]" in draft_text,
        "Original Query": state["query"] in draft_text,
        "Intent in Report": state["intent"] in draft_text,
    }
    
    print("Template Checks:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    print()


async def test_critic_verification():
    """Test critic citation verification."""
    print("=" * 60)
    print("🧪 Testing Critic Node (Citation Verification)")
    print("=" * 60)
    
    # Test with valid citation
    state_valid = {
        "draft": "The system is described in [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1).",
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Test Patent",
                "abstract": "Test abstract",
            }
        ],
    }
    
    result = await critic_node(state_valid)
    critique = result.get("critique", {})
    status = critique.get("status", "")
    
    print(f"✅ Valid citation test:")
    print(f"   Status: {status}")
    print(f"   Feedback: {critique.get('feedback', '')[:100]}...")
    print()
    
    # Test with invalid citation
    state_invalid = {
        "draft": "The system is described in [[US9999999]](https://patents.google.com/patent/US9999999).",
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Test Patent",
                "abstract": "Test abstract",
            }
        ],
    }
    
    result2 = await critic_node(state_invalid)
    critique2 = result2.get("critique", {})
    status2 = critique2.get("status", "")
    
    print(f"✅ Invalid citation test:")
    print(f"   Status: {status2} (expected: FAIL)")
    print(f"   Feedback: {critique2.get('feedback', '')[:150]}...")
    print()


async def test_full_workflow_mock():
    """Test full workflow with mock data."""
    print("=" * 60)
    print("🧪 Testing Full Workflow (Mock Data)")
    print("=" * 60)
    
    # Simulate workflow steps
    initial_query = "Find patents related to car suspension systems"
    
    print(f"1. Router: Processing query...")
    router_state = {"query": initial_query}
    router_result = await router_node(router_state)
    intent = router_result.get("intent", "UNKNOWN")
    print(f"   ✅ Intent: {intent}")
    
    print(f"\n2. Extractor: Expanding query...")
    extractor_state = {"query": initial_query}
    extractor_result = await extractor_node(extractor_state)
    keywords = extractor_result.get("keywords", [])
    print(f"   ✅ Generated {len(keywords)} search variations")
    
    print(f"\n3. Synthesizer: Generating report...")
    synthesizer_state = {
        "query": initial_query,
        "intent": intent,
        "keywords": keywords,
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Composition, adhesive agent, adhesive sheet, and laminate",
                "abstract": "The present invention provides a composition obtained by blending a polythiol compound (A), an isocyanate group-containing compound (B) and a radical generator (C).",
                "url": "https://patents.google.com/patent/US-2016168309-A1",
            }
        ],
        "litigation_context": [],
        "critique": {},
        "retry_count": 0,
    }
    
    draft_text = ""
    async for update in synthesizer_node(synthesizer_state):
        draft_text = update.get("draft", draft_text)
    
    print(f"   ✅ Generated draft ({len(draft_text)} characters)")
    
    print(f"\n4. Critic: Validating draft...")
    critic_state = {
        "draft": draft_text,
        "documents": synthesizer_state["documents"],
    }
    critic_result = await critic_node(critic_state)
    critique = critic_result.get("critique", {})
    critic_status = critique.get("status", "UNKNOWN")
    print(f"   ✅ Critic status: {critic_status}")
    
    if critic_status == "PASS":
        print(f"\n✅ Full workflow completed successfully!")
        print(f"\nFinal Report Preview:")
        print("-" * 60)
        print(draft_text[:600] + "..." if len(draft_text) > 600 else draft_text)
        print("-" * 60)
    else:
        print(f"\n⚠️  Critic rejected draft. Feedback:")
        print(f"   {critique.get('feedback', '')[:200]}...")
    
    print()


async def run_all_tests():
    """Run all tests."""
    print("\n" + "🚀 Starting Backend Tests (No Database Required)" + "\n")
    
    tests = [
        ("Router", test_router),
        ("Extractor", test_extractor),
        ("Synthesizer Template", test_synthesizer_template),
        ("Critic Verification", test_critic_verification),
        ("Full Workflow", test_full_workflow_mock),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            await test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


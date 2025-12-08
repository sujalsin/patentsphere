"""Comprehensive backend test with different query types and full validation."""
import asyncio
import sys
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph import workflow, AgentState


async def test_query_type(query: str, query_type: str, description: str):
    """Test a specific query type through the full workflow."""
    print(f"\n{'=' * 70}")
    print(f"🔍 Testing {query_type} Query: {description}")
    print(f"{'=' * 70}")
    print(f"Query: {query}")
    print()
    
    initial_state: AgentState = {
        "query": query,
        "intent": None,
        "keywords": None,
        "date_range": None,
        "documents": [
            {
                "patent_id": "US-2016168309-A1",
                "title": "Composition, adhesive agent, adhesive sheet, and laminate",
                "abstract": "The present invention provides a composition obtained by blending a polythiol compound (A), an isocyanate group-containing compound (B) and a radical generator (C), wherein the polythiol compound (A) is at least one compound selected from an aliphatic polythiol and an aromatic polythiol each of which has a thiol group binding to a primary carbon atom and may contain a hetero atom.",
                "url": "https://patents.google.com/patent/US-2016168309-A1",
                "publication_date": "2016-06-16",
            },
            {
                "patent_id": "US-2016168354-A1",
                "title": "Fluoroelastomer composition",
                "abstract": "The invention pertains to a fluoroelastomer composition comprising at least one fluoroelastomer, from 0.1 to 15 weight parts of meta-divinylbenzene, and from 0.1 to 10 weight parts of at least one peroxide.",
                "url": "https://patents.google.com/patent/US-2016168354-A1",
                "publication_date": "2016-06-16",
            }
        ],
        "litigation_context": [
            {
                "case_number": "1:00-cv-00037",
                "case_name": "Williams, et al v. General Surgical Inc, et al",
                "court_name": "E.D.Tex.",
                "filing_date": "2000-01-13",
                "case_status": "active",
                "plaintiff_name": "Randy Williams",
                "defendant_name": "Tyco International, Ltd",
                "patent_id": "US5655545",
                "outcome": None,
            }
        ] if query_type in ["LEGAL", "BOTH"] else [],
        "draft": None,
        "critique": None,
        "retry_count": 0,
        "final_response": None,
    }
    
    config = {"configurable": {"thread_id": f"test_{query_type}_{hash(query)}"}}
    
    try:
        final_state = None
        async for state in workflow.astream(initial_state, config=config, stream_mode="values"):
            final_state = state
        
        if not final_state:
            print("❌ Workflow returned no state")
            return False
        
        # Validate agent outputs
        print("📊 Agent Outputs:")
        print(f"   ✅ Router Intent: {final_state.get('intent', 'N/A')}")
        print(f"   ✅ Extractor Keywords: {len(final_state.get('keywords', []))} variations")
        print(f"   ✅ Documents Retrieved: {len(final_state.get('documents', []))}")
        print(f"   ✅ Litigation Cases: {len(final_state.get('litigation_context', []))}")
        print(f"   ✅ Retry Count: {final_state.get('retry_count', 0)}")
        
        # Validate final response
        final_response = final_state.get("final_response", "")
        if not final_response:
            print("❌ No final response generated")
            return False
        
        print(f"\n📄 Final Response Analysis:")
        print(f"   Length: {len(final_response)} characters")
        
        # Check template sections
        sections = {
            "Executive Summary": "Executive Summary" in final_response or "Summary" in final_response,
            "Query Analysis": "Query Analysis" in final_response,
            "Key Findings": "Key Findings" in final_response or "Findings" in final_response,
            "Conclusion": "Conclusion" in final_response,
        }
        
        if query_type in ["LEGAL", "BOTH"] and final_state.get("litigation_context"):
            sections["Legal Risks"] = "Legal Risks" in final_response or "⚠️" in final_response
        
        print("\n   Template Sections:")
        all_sections_present = True
        for section, present in sections.items():
            status = "✅" if present else "❌"
            print(f"   {status} {section}")
            if not present:
                all_sections_present = False
        
        # Check citations
        citation_pattern = r'\[\[([^\]]+)\]\]\(([^\)]+)\)'
        citations = re.findall(citation_pattern, final_response)
        
        print(f"\n   Citations: {len(citations)} found")
        if citations:
            for i, (patent_id, url) in enumerate(citations[:3], 1):
                print(f"   {i}. [[{patent_id}]]({url[:50]}...)")
        else:
            print("   ⚠️  No properly formatted citations found")
        
        # Check query information in response
        query_in_response = query.lower() in final_response.lower() or any(
            kw.lower() in final_response.lower() for kw in final_state.get("keywords", [])
        )
        print(f"\n   Query Context: {'✅ Present' if query_in_response else '⚠️ Missing'}")
        
        # Check intent in response
        intent_in_response = final_state.get("intent", "").upper() in final_response.upper()
        print(f"   Intent Info: {'✅ Present' if intent_in_response else '⚠️ Missing'}")
        
        # Display response preview
        print(f"\n📝 Response Preview:")
        print("-" * 70)
        print(final_response[:800] + "..." if len(final_response) > 800 else final_response)
        print("-" * 70)
        
        # Overall validation
        is_valid = (
            all_sections_present and
            len(citations) > 0 and
            query_in_response and
            len(final_response) > 500  # Minimum length check
        )
        
        status = "✅ PASS" if is_valid else "⚠️ PARTIAL"
        print(f"\n{status} - {query_type} Query Test")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_comprehensive_tests():
    """Run comprehensive tests for all query types."""
    print("\n" + "🚀 Comprehensive Backend Testing" + "\n")
    
    test_cases = [
        {
            "query": "Has Apple sued anyone over touchscreens?",
            "type": "LEGAL",
            "description": "Legal query about litigation",
        },
        {
            "query": "Find prior art for transformer neural networks",
            "type": "TECHNICAL",
            "description": "Technical query about prior art",
        },
        {
            "query": "What patents cover self-driving car sensors and have they been litigated?",
            "type": "BOTH",
            "description": "Combined legal and technical query",
        },
        {
            "query": "car suspension systems with active damping",
            "type": "TECHNICAL",
            "description": "Technical query with specific terms",
        },
    ]
    
    results = []
    for test_case in test_cases:
        result = await test_query_type(
            test_case["query"],
            test_case["type"],
            test_case["description"],
        )
        results.append((test_case["type"], result))
        await asyncio.sleep(2)  # Delay between tests
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for query_type, result in results:
        status = "✅ PASS" if result else "⚠️ PARTIAL"
        print(f"{status} - {query_type} Query")
    
    print(f"\nTotal: {passed}/{total} tests fully passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)



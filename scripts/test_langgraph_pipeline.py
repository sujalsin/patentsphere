#!/usr/bin/env python3
"""
Test script for the LangGraph-based patent analysis pipeline.

This script validates:
1. LangGraph orchestration works
2. Citation indexing produces proper output format
3. Final response matches the API schema
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import Orchestrator
from app.templates.citations import CitationIndexManager
from config.settings import get_settings


async def test_citation_format():
    """Test that citation indexing produces the correct format."""
    print("=" * 60)
    print("Testing Citation Index Format")
    print("=" * 60)
    
    manager = CitationIndexManager()
    
    # Add sample patents
    manager.add_patent(
        patent_id="US-1123456",
        title="Method for Decentralized Grid Management",
        assignee="IBM",
        snippet="A plurality of nodes negotiating energy transfer rates via a shared immutable ledger...",
        score=0.92,
        section_type="claims",
    )
    manager.add_patent(
        patent_id="US-9876543",
        title="IoT Integration for Smart Grids",
        assignee="Siemens",
        snippet="Integrating smart meters to validate energy production in real-time...",
        score=0.88,
        section_type="abstract",
    )
    
    # Add sample litigation
    manager.add_litigation(
        case_id="1:16-cv-0023",
        title="SolarCity Corp v. SunPower Corp",
        outcome="Infringement Found",
        risk_level="high",
        snippet="Defendant's use of a peer-to-peer matching algorithm infringes...",
    )
    
    # Check sources header
    header = manager.generate_sources_header()
    print(f"\nSources Header:\n{header}")
    
    assert "[1] US-1123456 (IBM)" in header
    assert "[2] US-9876543 (Siemens)" in header
    assert "[3] SolarCity Corp v. SunPower Corp (Litigation)" in header
    
    # Check sources
    sources = manager.get_sources()
    assert len(sources) == 3
    
    # Check URLs
    s1 = sources[0]
    assert "google_patents" in s1.urls
    assert s1.score == 0.92
    
    print("\n✓ Citation format test PASSED")
    return True


async def test_langgraph_setup():
    """Test that LangGraph orchestration is properly set up."""
    print("\n" + "=" * 60)
    print("Testing LangGraph Setup")
    print("=" * 60)
    
    settings = get_settings()
    orchestrator = Orchestrator(use_langgraph=True)
    
    assert orchestrator.use_langgraph is True
    assert orchestrator.graph is not None
    
    # Compile graph
    compiled = orchestrator.graph.compile()
    assert compiled is not None
    
    print(f"\nAgents initialized: {list(orchestrator.agents.keys())}")
    print(f"LangGraph mode: {orchestrator.use_langgraph}")
    
    print("\n✓ LangGraph setup test PASSED")
    return True


async def test_api_response_schema():
    """Test that the API response schema is valid."""
    print("\n" + "=" * 60)
    print("Testing API Response Schema")
    print("=" * 60)
    
    from app.templates.models import APIResponse, SourceItem, AnswerSectionModel
    
    # Create sample API response
    response = APIResponse(
        response_id="resp_test123",
        query="Analyze the patentability of a blockchain-based energy trading system",
        sources_header="[1] US-1123456 (IBM) · [2] US-9876543 (Siemens)",
        answer_section=AnswerSectionModel(
            technical_summary="The concept of distributed energy trading on blockchain has significant prior art coverage. US-1123456 (IBM) discloses peer-to-peer energy transactions [1].",
            legal_summary="Medium-High litigation risk in this domain.",
            novelty_assessment="Zero-knowledge proofs for privacy appears less crowded.",
        ),
        sources=[
            SourceItem(
                index=1,
                type="patent",
                title="US-1123456: Method for Decentralized Grid Management",
                assignee="IBM",
                url="https://patents.google.com/patent/US1123456",
                snippet="A plurality of nodes negotiating energy transfer rates...",
                score=0.92,
            ),
            SourceItem(
                index=2,
                type="patent",
                title="US-9876543: IoT Integration for Smart Grids",
                assignee="Siemens",
                url="https://patents.google.com/patent/US9876543",
                snippet="Integrating smart meters to validate energy production...",
                score=0.88,
            ),
        ],
        risk_score=65,
        quality_score=0.78,
        metadata={"total_latency_ms": 2500, "iterations": 1},
    )
    
    # Validate model
    response_dict = response.model_dump()
    
    print(f"\nResponse ID: {response_dict['response_id']}")
    print(f"Sources Header: {response_dict['sources_header']}")
    print(f"Sources Count: {len(response_dict['sources'])}")
    print(f"Risk Score: {response_dict['risk_score']}")
    
    # Verify structure
    assert "answer_section" in response_dict
    assert "technical_summary" in response_dict["answer_section"]
    assert "sources" in response_dict
    assert len(response_dict["sources"]) == 2
    
    print("\n✓ API Response Schema test PASSED")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PatentSphere LangGraph Pipeline Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        await test_citation_format()
    except Exception as e:
        print(f"\n✗ Citation format test FAILED: {e}")
        all_passed = False
    
    try:
        await test_langgraph_setup()
    except Exception as e:
        print(f"\n✗ LangGraph setup test FAILED: {e}")
        all_passed = False
    
    try:
        await test_api_response_schema()
    except Exception as e:
        print(f"\n✗ API Response Schema test FAILED: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

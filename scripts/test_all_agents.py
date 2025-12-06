#!/usr/bin/env python3
"""
Comprehensive test script for all PatentSphere agents.
Tests each agent individually with real LLM calls and validates output templates.
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.agents.litigation import LitigationScoutAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.critic import CriticAgent
from app.templates.models import ClaimsOutput, SynthesisOutput
from config.settings import get_settings


TEST_QUERY = "blockchain energy trading peer-to-peer"


async def test_claims_analyzer():
    """Test ClaimsAnalyzerAgent."""
    print("\n" + "=" * 60)
    print("Testing ClaimsAnalyzerAgent")
    print("=" * 60)
    
    settings = get_settings()
    agent = ClaimsAnalyzerAgent(settings=settings)
    
    result = await agent.run(TEST_QUERY)
    
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Latency: {result.data.get('latency_ms', 0):.0f}ms")
    
    if result.success:
        data = result.data
        print(f"Query Type: {data.get('query_type')}")
        print(f"Technical Keywords: {data.get('technical_keywords', [])[:5]}")
        print(f"CPC Codes: {data.get('cpc_codes', [])[:3]}")
        print(f"Confidence: {data.get('confidence_score', 0):.2f}")
        
        # Validate against template
        try:
            validated = ClaimsOutput.model_validate(data)
            print("✓ Output validates against ClaimsOutput template")
        except Exception as e:
            print(f"✗ Template validation failed: {e}")
            return False
    else:
        print(f"✗ Agent failed: {result.error}")
        return False
    
    return True


async def test_citation_mapper():
    """Test CitationMapperAgent."""
    print("\n" + "=" * 60)
    print("Testing CitationMapperAgent")
    print("=" * 60)
    
    settings = get_settings()
    agent = CitationMapperAgent(settings=settings)
    
    result = await agent.run(TEST_QUERY)
    
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Latency: {result.data.get('latency_ms', 0):.0f}ms")
    
    if result.success:
        data = result.data
        chunks = data.get('results', data.get('chunks', []))
        print(f"Retrieved chunks: {len(chunks)}")
        if chunks:
            print(f"Sample chunk patent_id: {chunks[0].get('patent_id', 'N/A')}")
            print(f"Sample chunk score: {chunks[0].get('score', 0):.3f}")
        print("✓ Citation retrieval working")
    else:
        print(f"✗ Agent failed: {result.error}")
        return False
    
    return True


async def test_litigation_scout():
    """Test LitigationScoutAgent."""
    print("\n" + "=" * 60)
    print("Testing LitigationScoutAgent")
    print("=" * 60)
    
    settings = get_settings()
    agent = LitigationScoutAgent(settings=settings)
    
    result = await agent.run(TEST_QUERY)
    
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Latency: {result.data.get('latency_ms', 0):.0f}ms")
    
    if result.success:
        data = result.data
        cases = data.get('cases', data.get('litigation_cases', []))
        print(f"Found litigation cases: {len(cases)}")
        print(f"Risk Level: {data.get('risk_level', 'N/A')}")
        print("✓ Litigation search working")
    else:
        print(f"✗ Agent failed: {result.error}")
        return False
    
    return True


async def test_synthesis_agent():
    """Test SynthesisAgent with mock context."""
    print("\n" + "=" * 60)
    print("Testing SynthesisAgent")
    print("=" * 60)
    
    settings = get_settings()
    agent = SynthesisAgent(settings=settings)
    
    # Set up mock context (normally comes from other agents)
    agent.context = {
        "claims_analysis": {
            "query_type": "prior_art_search",
            "technical_keywords": ["blockchain", "energy", "trading"],
            "cpc_codes": ["H04L9/00", "G06Q40/00"],
            "confidence_score": 0.85,
        },
        "citation_data": {
            "results": [
                {
                    "patent_id": "US-10123456",
                    "title": "Blockchain Energy Trading",
                    "score": 0.92,
                    "assignee": "IBM",
                }
            ]
        },
        "litigation_data": {
            "cases": [],
            "risk_level": "low",
        }
    }
    
    result = await agent.run(TEST_QUERY)
    
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Latency: {result.data.get('latency_ms', 0):.0f}ms")
    
    if result.success:
        data = result.data
        print(f"Executive Summary: {data.get('executive_summary', '')[:100]}...")
        print(f"Risk Score: {data.get('risk_score', 'N/A')}")
        print(f"Sections: {len(data.get('insight_sections', []))}")
        print(f"Next Steps: {len(data.get('next_steps', []))}")
        
        # Validate against template
        try:
            validated = SynthesisOutput.model_validate(data)
            print("✓ Output validates against SynthesisOutput template")
        except Exception as e:
            print(f"✗ Template validation failed: {e}")
            return False
    else:
        print(f"✗ Agent failed: {result.error}")
        return False
    
    return True


async def test_critic_agent():
    """Test CriticAgent."""
    print("\n" + "=" * 60)
    print("Testing CriticAgent")
    print("=" * 60)
    
    settings = get_settings()
    weights = settings.critic.reward_weights
    agent = CriticAgent(settings=settings, weights=weights)
    
    # Mock data for critic (normally comes from other agents)
    mock_chunks = [
        {
            "patent_id": "US-10123456",
            "title": "Blockchain Energy Trading",
            "score": 0.92,
            "publication_date": "2021-05-15",
            "cpc_codes": [{"code": "G06Q40/00"}],
        },
        {
            "patent_id": "US-10654321",
            "title": "Smart Grid Controller",
            "score": 0.85,
            "publication_date": "2019-08-20",
            "cpc_codes": [{"code": "H02J13/00"}],
        }
    ]
    
    mock_claims = {
        "query_type": "prior_art_search",
        "technical_keywords": ["blockchain", "energy", "trading"],
        "cpc_codes": [{"code": "G06Q40/00"}, {"code": "H02J13/00"}],
    }
    
    mock_synthesis = {
        "executive_summary": "The blockchain energy trading domain shows significant patent activity. "
                           "US-10123456 (IBM) discloses peer-to-peer energy transactions using distributed ledgers. "
                           "Key technical areas include smart contracts, grid integration, and real-time settlement.",
        "risk_score": 60,
    }
    
    result = await agent.run(
        query=TEST_QUERY,
        retrieved_chunks=mock_chunks,
        claims_analysis=mock_claims,
        synthesis_output=mock_synthesis,
    )
    
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    
    if result.success:
        data = result.data
        print(f"Quality Score: {data.get('score', 0):.3f}")
        print(f"Citation Overlap: {data.get('components', {}).get('citation_overlap', 0):.3f}")
        print(f"CPC Relevance: {data.get('components', {}).get('cpc_relevance', 0):.3f}")
        print(f"Temporal Diversity: {data.get('components', {}).get('temporal_diversity', 0):.3f}")
        print(f"LLM Fluency: {data.get('components', {}).get('llm_fluency', 0):.3f}")
        print(f"Feedback: {data.get('feedback', 'N/A')}")
        print("✓ Critic evaluation working")
    else:
        print(f"✗ Agent failed: {result.error}")
        return False
    
    return True


async def main():
    """Run all agent tests."""
    print("\n" + "=" * 60)
    print("PatentSphere Agent Tests")
    print("=" * 60)
    print(f"Test Query: '{TEST_QUERY}'")
    
    results = {}
    
    # Test each agent
    results["ClaimsAnalyzer"] = await test_claims_analyzer()
    results["CitationMapper"] = await test_citation_mapper()
    results["LitigationScout"] = await test_litigation_scout()
    results["Synthesis"] = await test_synthesis_agent()
    results["Critic"] = await test_critic_agent()
    
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
        print("All agent tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


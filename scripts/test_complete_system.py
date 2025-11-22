#!/usr/bin/env python3
"""
Complete system test for PatentSphere.
Tests all components including FastAPI endpoints.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from app.orchestrator import Orchestrator
from config.settings import get_settings


async def test_database_connectivity():
    """Test database connections."""
    print("=== Testing Database Connectivity ===\n")
    
    settings = get_settings()
    
    # Test PostgreSQL
    try:
        import psycopg
        conn_str = f"postgresql://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{settings.database.database}"
        with psycopg.connect(conn_str, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM patent_chunks")
                count = cur.fetchone()[0]
                print(f"✓ PostgreSQL: Connected ({count:,} patent chunks)")
                
                cur.execute("SELECT COUNT(*) FROM patent_litigation")
                lit_count = cur.fetchone()[0]
                print(f"✓ PostgreSQL: {lit_count:,} litigation cases")
    except Exception as e:
        print(f"✗ PostgreSQL: {e}")
        return False
    
    # Test Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
        collections = client.get_collections()
        print(f"✓ Qdrant: Connected ({len(collections.collections)} collections)")
    except Exception as e:
        print(f"✗ Qdrant: {e}")
        return False
    
    return True


async def test_ollama_connectivity():
    """Test Ollama LLM service."""
    print("\n=== Testing Ollama Connectivity ===\n")
    
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.llm.ollama_host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", [])]
                print(f"✓ Ollama: Connected")
                print(f"✓ Available models: {', '.join(models)}")
                return True
            else:
                print(f"✗ Ollama: Status {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Ollama: {e}")
        return False


async def test_agents_individually():
    """Test each agent individually."""
    print("\n=== Testing Agents Individually ===\n")
    
    orch = Orchestrator()
    query = "machine learning"
    
    results = {}
    
    # Test Citation Mapper (fast, no LLM)
    print("1. CitationMapperAgent...")
    try:
        citation_agent = orch.agents.get("citation")
        if citation_agent:
            result = await asyncio.wait_for(
                citation_agent.run(query),
                timeout=30
            )
            results["citation"] = result.success
            print(f"   {'✓' if result.success else '✗'} Success: {result.success}")
            if result.success:
                print(f"   Results: {len(result.data.get('results', []))} patents")
        else:
            print("   ✗ Agent not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["citation"] = False
    
    # Test Litigation Scout (fast, no LLM)
    print("\n2. LitigationScoutAgent...")
    try:
        litigation_agent = orch.agents.get("litigation")
        if litigation_agent:
            result = await asyncio.wait_for(
                litigation_agent.run(query, retrieved_patent_ids=None),
                timeout=10
            )
            results["litigation"] = result.success
            print(f"   {'✓' if result.success else '✗'} Success: {result.success}")
            if result.success:
                print(f"   Total cases: {result.data.get('total_cases', 0)}")
        else:
            print("   ✗ Agent not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["litigation"] = False
    
    # Test Claims Analyzer (uses LLM - may take time)
    print("\n3. ClaimsAnalyzerAgent (LLM - may take 2-3 minutes)...")
    try:
        claims_agent = orch.agents.get("claims")
        if claims_agent:
            result = await asyncio.wait_for(
                claims_agent.run(query),
                timeout=300
            )
            results["claims"] = result.success
            print(f"   {'✓' if result.success else '✗'} Success: {result.success}")
            if result.success:
                print(f"   Query type: {result.data.get('query_type', 'N/A')}")
                print(f"   Confidence: {result.data.get('confidence', 0):.2f}")
                print(f"   Source: {result.data.get('source', 'N/A')}")
        else:
            print("   ✗ Agent not found")
    except asyncio.TimeoutError:
        print("   ⚠ Timed out (this is normal for first LLM call)")
        results["claims"] = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["claims"] = False
    
    return results


async def test_fastapi_endpoints():
    """Test FastAPI endpoints."""
    print("\n=== Testing FastAPI Endpoints ===\n")
    
    base_url = "http://localhost:8000"
    results = {}
    
    # Test health
    print("1. GET /health...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                results["health"] = True
                print(f"   ✓ Status: {response.status_code}")
                print(f"   ✓ Profile: {data.get('profile')}")
            else:
                results["health"] = False
                print(f"   ✗ Status: {response.status_code}")
    except Exception as e:
        results["health"] = False
        print(f"   ✗ Error: {e}")
        return results
    
    # Test whoami
    print("\n2. GET /whoami...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/whoami")
            if response.status_code == 200:
                data = response.json()
                results["whoami"] = True
                print(f"   ✓ Status: {response.status_code}")
                print(f"   ✓ Profile: {data.get('profile')}")
            else:
                results["whoami"] = False
                print(f"   ✗ Status: {response.status_code}")
    except Exception as e:
        results["whoami"] = False
        print(f"   ✗ Error: {e}")
    
    # Test retrieve
    print("\n3. GET /retrieve?query=machine+learning...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            start = time.time()
            response = await client.get(f"{base_url}/retrieve", params={"query": "machine learning"})
            elapsed = time.time() - start
            if response.status_code == 200:
                data = response.json()
                results["retrieve"] = True
                print(f"   ✓ Status: {response.status_code}")
                print(f"   ✓ Time: {elapsed:.1f}s")
                if "results" in data:
                    print(f"   ✓ Results: {len(data['results'])} patents")
            else:
                results["retrieve"] = False
                print(f"   ✗ Status: {response.status_code}")
    except Exception as e:
        results["retrieve"] = False
        print(f"   ✗ Error: {e}")
    
    # Test query (may timeout - that's OK)
    print("\n4. GET /query?query=AI+patents...")
    print("   Note: This endpoint may timeout due to LLM calls")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            start = time.time()
            response = await client.get(f"{base_url}/query", params={"query": "AI patents"})
            elapsed = time.time() - start
            if response.status_code == 200:
                data = response.json()
                results["query"] = True
                print(f"   ✓ Status: {response.status_code}")
                print(f"   ✓ Time: {elapsed:.1f}s")
                print(f"   ✓ Agents: {list(data.keys())}")
            else:
                results["query"] = False
                print(f"   ✗ Status: {response.status_code}")
    except httpx.TimeoutException:
        results["query"] = "timeout"
        print("   ⚠ Timed out (expected for long-running queries)")
    except Exception as e:
        results["query"] = False
        print(f"   ✗ Error: {e}")
    
    return results


async def main():
    """Run all tests."""
    print("=" * 60)
    print("PatentSphere Complete System Test")
    print("=" * 60)
    
    # Test database connectivity
    db_ok = await test_database_connectivity()
    
    # Test Ollama
    ollama_ok = await test_ollama_connectivity()
    
    # Test agents
    agent_results = await test_agents_individually()
    
    # Test FastAPI
    api_results = await test_fastapi_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Database: {'✓' if db_ok else '✗'}")
    print(f"Ollama: {'✓' if ollama_ok else '✗'}")
    print(f"\nAgents:")
    for agent, success in agent_results.items():
        status = "✓" if success else "✗"
        print(f"  {agent}: {status}")
    print(f"\nFastAPI Endpoints:")
    for endpoint, result in api_results.items():
        if result is True:
            status = "✓"
        elif result == "timeout":
            status = "⚠"
        else:
            status = "✗"
        print(f"  {endpoint}: {status}")
    
    print("\n" + "=" * 60)
    
    # Overall status
    all_critical = db_ok and ollama_ok and api_results.get("health") and api_results.get("retrieve")
    if all_critical:
        print("✓ All critical components are working!")
    else:
        print("⚠ Some components need attention (see details above)")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


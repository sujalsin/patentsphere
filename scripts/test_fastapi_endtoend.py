#!/usr/bin/env python3
"""
End-to-end testing script for PatentSphere FastAPI system.

This script performs comprehensive testing:
1. Verifies Docker services are running
2. Checks database connectivity and data
3. Tests all FastAPI endpoints
4. Validates all agents are working
5. Measures performance metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import httpx
import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from app.orchestrator import Orchestrator


def check_docker_services() -> bool:
    """Check if Docker services are running."""
    print("=== Checking Docker Services ===")
    try:
        result = subprocess.run(
            ["docker", "compose", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "patentsphere-postgres" in result.stdout and "patentsphere-qdrant" in result.stdout:
            print("✓ PostgreSQL and Qdrant containers are running")
            return True
        else:
            print("✗ Docker services not running")
            print("  Run: docker compose up -d postgres qdrant")
            return False
    except Exception as e:
        print(f"✗ Error checking Docker: {e}")
        return False


def check_database_connectivity(settings) -> bool:
    """Check database connectivity and data."""
    print("\n=== Checking Database Connectivity ===")
    try:
        pg_cfg = settings.database
        conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
        
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Check patent_chunks
                cur.execute("SELECT COUNT(*) FROM patent_chunks")
                chunk_count = cur.fetchone()[0]
                print(f"✓ patent_chunks: {chunk_count:,} records")
                
                # Check patent_citations
                cur.execute("SELECT COUNT(*) FROM patent_citations")
                citation_count = cur.fetchone()[0]
                print(f"✓ patent_citations: {citation_count:,} records")
                
                # Check patent_litigation
                cur.execute("SELECT COUNT(*) FROM patent_litigation")
                litigation_count = cur.fetchone()[0]
                print(f"✓ patent_litigation: {litigation_count:,} records")
                
                # Check rl_experiences
                cur.execute("SELECT COUNT(*) FROM rl_experiences")
                rl_count = cur.fetchone()[0]
                print(f"✓ rl_experiences: {rl_count:,} records")
        
        return True
    except Exception as e:
        print(f"✗ Database connectivity error: {e}")
        return False


def check_qdrant_connectivity(settings) -> bool:
    """Check Qdrant connectivity."""
    print("\n=== Checking Qdrant Connectivity ===")
    try:
        from qdrant_client import QdrantClient
        
        q_cfg = settings.qdrant
        client = QdrantClient(
            host=q_cfg.host,
            port=q_cfg.port,
            https=False,
        )
        
        collections = client.get_collections()
        if q_cfg.collection_name in [c.name for c in collections.collections]:
            # Get collection info
            collection_info = client.get_collection(q_cfg.collection_name)
            print(f"✓ Collection '{q_cfg.collection_name}' exists")
            print(f"  Points: {collection_info.points_count:,}")
            return True
        else:
            print(f"✗ Collection '{q_cfg.collection_name}' not found")
            return False
    except Exception as e:
        print(f"✗ Qdrant connectivity error: {e}")
        return False


async def test_fastapi_endpoints(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test all FastAPI endpoints."""
    print("\n=== Testing FastAPI Endpoints ===")
    results = {}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test /health
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ /health: {data}")
                results["health"] = {"status": "success", "data": data}
            else:
                print(f"✗ /health: Status {response.status_code}")
                results["health"] = {"status": "failed", "status_code": response.status_code}
        except Exception as e:
            print(f"✗ /health: {e}")
            results["health"] = {"status": "error", "error": str(e)}
        
        # Test /whoami
        try:
            response = await client.get(f"{base_url}/whoami")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ /whoami: {data}")
                results["whoami"] = {"status": "success", "data": data}
            else:
                print(f"✗ /whoami: Status {response.status_code}")
                results["whoami"] = {"status": "failed", "status_code": response.status_code}
        except Exception as e:
            print(f"✗ /whoami: {e}")
            results["whoami"] = {"status": "error", "error": str(e)}
        
        # Test /retrieve
        try:
            query = "AI machine learning"
            response = await client.get(f"{base_url}/retrieve", params={"query": query}, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                citation_count = len(data.get("results", []))
                print(f"✓ /retrieve: Found {citation_count} citations")
                results["retrieve"] = {"status": "success", "citation_count": citation_count}
            else:
                print(f"✗ /retrieve: Status {response.status_code}")
                results["retrieve"] = {"status": "failed", "status_code": response.status_code}
        except Exception as e:
            print(f"✗ /retrieve: {e}")
            results["retrieve"] = {"status": "error", "error": str(e)}
        
        # Test /query (full orchestrator)
        try:
            query = "machine learning neural networks"
            start_time = time.time()
            response = await client.get(f"{base_url}/query", params={"query": query}, timeout=120.0)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                agents = list(data.keys())
                print(f"✓ /query: Completed in {elapsed:.2f}s")
                print(f"  Agents: {', '.join(agents)}")
                
                # Check for CriticAgent
                if "critic" in data:
                    critic_data = data["critic"]
                    if "score" in critic_data:
                        print(f"  CriticAgent reward: {critic_data['score']:.3f}")
                
                results["query"] = {
                    "status": "success",
                    "elapsed_seconds": elapsed,
                    "agents": agents,
                    "has_critic": "critic" in data,
                }
            else:
                print(f"✗ /query: Status {response.status_code}")
                results["query"] = {"status": "failed", "status_code": response.status_code}
        except Exception as e:
            print(f"✗ /query: {e}")
            results["query"] = {"status": "error", "error": str(e)}
    
    return results


async def test_orchestrator_directly() -> Dict[str, Any]:
    """Test orchestrator directly (without FastAPI)."""
    print("\n=== Testing Orchestrator Directly ===")
    results = {}
    
    try:
        orch = Orchestrator()
        print(f"✓ Orchestrator created with {len(orch.agents)} agents")
        print(f"  Agents: {', '.join(orch.agents.keys())}")
        
        # Test query
        query = "AI machine learning patents"
        print(f"\nRunning query: '{query}'")
        start_time = time.time()
        
        agent_results = await orch.run_all(query)
        elapsed = time.time() - start_time
        
        print(f"✓ Query completed in {elapsed:.2f}s")
        
        # Check each agent
        for agent_name, result in agent_results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} {agent_name}: ", end="")
            if result.success:
                if agent_name == "critic":
                    score = result.data.get("score", 0)
                    print(f"Reward={score:.3f}")
                elif agent_name == "citation_mapper":
                    count = len(result.data.get("results", []))
                    print(f"{count} citations")
                elif agent_name == "litigation_scout":
                    cases = result.data.get("total_cases", 0)
                    print(f"{cases} cases")
                else:
                    print("Success")
            else:
                print(f"Failed: {result.error}")
        
        results = {
            "status": "success",
            "elapsed_seconds": elapsed,
            "agents": {name: {"success": r.success} for name, r in agent_results.items()},
        }
        
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        results = {"status": "error", "error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end testing for PatentSphere")
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker service checks",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip FastAPI endpoint tests",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI base URL",
    )
    args = parser.parse_args()
    
    settings = get_settings()
    
    print("=" * 60)
    print("PATENTSPHERE END-TO-END TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Check Docker services
    if not args.skip_docker:
        if not check_docker_services():
            all_passed = False
            print("\n⚠️  Docker services not running. Some tests may fail.")
    
    # Check database
    if not check_database_connectivity(settings):
        all_passed = False
    
    # Check Qdrant
    if not check_qdrant_connectivity(settings):
        all_passed = False
    
    # Test orchestrator directly
    orch_results = asyncio.run(test_orchestrator_directly())
    if orch_results.get("status") != "success":
        all_passed = False
    
    # Test FastAPI endpoints
    if not args.skip_api:
        api_results = asyncio.run(test_fastapi_endpoints(args.api_url))
        if any(r.get("status") != "success" for r in api_results.values()):
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Review output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()


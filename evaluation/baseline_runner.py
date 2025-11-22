#!/usr/bin/env python3
"""
Baseline evaluation runner for PatentSphere.

Runs queries through the orchestrator and collects metrics:
- Latency per agent
- Citation results (top-k patents)
- Success rates
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import asyncio

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import Orchestrator


async def run_query(orchestrator: Orchestrator, query: str, query_id: int) -> Dict[str, Any]:
    """Run a single query and collect metrics."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        results = await orchestrator.run_all(query)
        end_time = asyncio.get_event_loop().time()
        total_latency_ms = (end_time - start_time) * 1000
        
        # Extract metrics
        agent_latencies = {
            agent: result.data.get("latency_ms", 0)
            for agent, result in results.items()
        }
        
        # Get citation results
        citation_results = []
        if "citation_mapper" in results and results["citation_mapper"].success:
            citation_data = results["citation_mapper"].data.get("results", [])
            citation_results = [
                {
                    "patent_id": r.get("patent_id", ""),
                    "score": r.get("score", 0.0),
                    "chunk_type": r.get("chunk_type", ""),
                }
                for r in citation_data[:10]  # Top 10
            ]
        
        return {
            "query_id": query_id,
            "query": query,
            "success": all(r.success for r in results.values()),
            "total_latency_ms": total_latency_ms,
            "agent_latencies": agent_latencies,
            "citation_count": len(citation_results),
            "top_citations": citation_results,
            "all_agents": {k: {"success": v.success, "error": v.error} for k, v in results.items()},
        }
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        return {
            "query_id": query_id,
            "query": query,
            "success": False,
            "error": str(e),
            "total_latency_ms": (end_time - start_time) * 1000,
        }


async def run_baseline(queries: List[str], output_path: Path) -> Dict[str, Any]:
    """Run baseline evaluation on a list of queries."""
    orchestrator = Orchestrator()
    results = []
    
    print(f"Running baseline evaluation on {len(queries)} queries...")
    
    for idx, query in enumerate(queries, 1):
        print(f"[{idx}/{len(queries)}] Processing: {query[:60]}...")
        result = await run_query(orchestrator, query, idx)
        results.append(result)
        
        # Print summary
        if result["success"]:
            print(f"  ✓ Success ({result['total_latency_ms']:.1f}ms, {result['citation_count']} citations)")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Calculate aggregate metrics
    successful = [r for r in results if r["success"]]
    avg_latency = sum(r["total_latency_ms"] for r in successful) / len(successful) if successful else 0
    avg_citations = sum(r["citation_count"] for r in successful) / len(successful) if successful else 0
    
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "evaluation_type": "baseline",
        "total_queries": len(queries),
        "successful_queries": len(successful),
        "success_rate": len(successful) / len(queries) if queries else 0,
        "metrics": {
            "avg_latency_ms": avg_latency,
            "avg_citations_per_query": avg_citations,
        },
        "queries": results,
    }
    
    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Success rate: {output['success_rate']*100:.1f}%")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"  Avg citations: {avg_citations:.1f}")
    print(f"  Results saved to: {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument(
        "--queries",
        type=str,
        default="evaluation/data/baseline_queries.json",
        help="Path to JSON file with queries (list of strings)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/baseline_scores.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to run (for testing)",
    )
    args = parser.parse_args()
    
    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        sys.exit(1)
    
    with queries_path.open() as f:
        queries = json.load(f)
    
    if not isinstance(queries, list):
        print("Error: Queries file must contain a JSON array of strings")
        sys.exit(1)
    
    if args.limit:
        queries = queries[:args.limit]
    
    # Run evaluation
    output_path = Path(args.output)
    asyncio.run(run_baseline(queries, output_path))


if __name__ == "__main__":
    main()


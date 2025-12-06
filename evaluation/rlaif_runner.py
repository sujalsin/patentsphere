#!/usr/bin/env python3
"""
RLAIF Evaluation Runner.

Runs queries with trained AdaptiveRetrievalAgent policy and compares to baseline.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_query(
    orchestrator: Orchestrator, query: str, query_id: int
) -> Dict[str, Any]:
    """
    Run a single query through the orchestrator.
    
    Returns:
        Metrics dictionary with latency, reward, and other metrics
    """
    start_time = time.perf_counter()
    
    try:
        results = await orchestrator.run_all(query)
        elapsed = time.perf_counter() - start_time
        
        # Extract metrics
        citation_data = results.get("citation_mapper") or results.get("adaptive_retrieval")
        critic_data = results.get("critic")
        claims_data = results.get("claims_analyzer")
        
        metrics = {
            "query_id": query_id,
            "query": query,
            "latency_ms": elapsed * 1000,
            "success": all(r.success for r in results.values()),
        }
        
        # Citation metrics
        if citation_data and citation_data.success:
            results_list = citation_data.data.get("results", [])
            metrics["num_citations"] = len(results_list)
            metrics["citation_scores"] = [r.get("score", 0.0) for r in results_list]
        else:
            metrics["num_citations"] = 0
            metrics["citation_scores"] = []
        
        # Reward metrics
        if critic_data and critic_data.success:
            metrics["reward"] = critic_data.data.get("score", 0.0)
            metrics["reward_components"] = critic_data.data.get("components", {})
        else:
            metrics["reward"] = 0.0
            metrics["reward_components"] = {}
        
        # Query type
        if claims_data and claims_data.success:
            metrics["query_type"] = claims_data.data.get("query_type", "other")
        else:
            metrics["query_type"] = "other"
        
        # Adaptive retrieval metrics (if used)
        adaptive_data = results.get("adaptive_retrieval")
        if adaptive_data and adaptive_data.success:
            metrics["retrieval_depth"] = adaptive_data.data.get("retrieval_depth", 1)
            metrics["rl_metadata"] = adaptive_data.data.get("rl_metadata", {})
        else:
            metrics["retrieval_depth"] = 1
        
        return metrics
        
    except Exception as exc:
        logger.error("Query %d failed: %s", query_id, exc)
        return {
            "query_id": query_id,
            "query": query,
            "latency_ms": (time.perf_counter() - start_time) * 1000,
            "success": False,
            "error": str(exc),
        }


async def run_rlaif_evaluation(
    queries: List[str], output_path: Path
) -> Dict[str, Any]:
    """
    Run RLAIF evaluation on queries.
    
    Args:
        queries: List of queries to evaluate
        output_path: Path to save results JSON
    
    Returns:
        Summary statistics
    """
    logger.info("Starting RLAIF evaluation on %d queries", len(queries))
    
    # Create orchestrator (will use GPU if config.embeddings.device is set to "auto" or "cuda")
    orchestrator = Orchestrator()
    
    # Check if adaptive retrieval is enabled
    adaptive_enabled = orchestrator.agents.get("adaptive_retrieval") is not None
    logger.info("AdaptiveRetrievalAgent enabled: %s", adaptive_enabled)
    
    results = []
    start_time = time.time()
    
    for i, query in enumerate(queries, 1):
        logger.info("Running query %d/%d: %s", i, len(queries), query[:50])
        
        metrics = await run_query(orchestrator, query, i)
        results.append(metrics)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_latency = sum(r["latency_ms"] for r in results[-10:]) / 10
            logger.info(
                "Progress: %d/%d, avg_latency=%.0fms, elapsed=%.1fs",
                i,
                len(queries),
                avg_latency,
                elapsed,
            )
    
    # Calculate summary statistics
    successful = [r for r in results if r.get("success", False)]
    
    summary = {
        "total_queries": len(queries),
        "successful_queries": len(successful),
        "success_rate": len(successful) / len(queries) if queries else 0.0,
        "latency": {
            "mean": sum(r["latency_ms"] for r in successful) / len(successful) if successful else 0.0,
            "p50": sorted([r["latency_ms"] for r in successful])[len(successful) // 2] if successful else 0.0,
            "p95": sorted([r["latency_ms"] for r in successful])[int(len(successful) * 0.95)] if successful else 0.0,
        },
        "rewards": {
            "mean": sum(r.get("reward", 0.0) for r in successful) / len(successful) if successful else 0.0,
            "min": min([r.get("reward", 0.0) for r in successful]) if successful else 0.0,
            "max": max([r.get("reward", 0.0) for r in successful]) if successful else 0.0,
        },
        "citations": {
            "mean": sum(r.get("num_citations", 0) for r in successful) / len(successful) if successful else 0.0,
            "min": min([r.get("num_citations", 0) for r in successful]) if successful else 0,
            "max": max([r.get("num_citations", 0) for r in successful]) if successful else 0,
        },
        "retrieval_depth": {
            "mean": sum(r.get("retrieval_depth", 1) for r in successful) / len(successful) if successful else 1.0,
        },
    }
    
    # Save results
    output_data = {
        "summary": summary,
        "results": results,
        "timestamp": time.time(),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Evaluation complete. Results saved to %s", output_path)
    logger.info("Summary: %d/%d successful, avg_reward=%.3f, avg_latency=%.0fms",
                len(successful), len(queries), summary["rewards"]["mean"], summary["latency"]["mean"])
    
    return summary


def load_queries(query_file: Path) -> List[str]:
    """Load queries from JSON file."""
    with open(query_file, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "queries" in data:
            return data["queries"]
        else:
            raise ValueError("Invalid query file format")


async def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RLAIF evaluation")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/data/baseline_queries.json"),
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/rlaif_scores.json"),
        help="Output path for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of queries to evaluate",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage for embeddings (overrides config to 'cuda')",
    )
    
    args = parser.parse_args()
    
    # Override device setting if --gpu flag is provided
    if args.gpu:
        from config.settings import get_settings
        settings = get_settings()
        # Temporarily override device for this run
        original_device = settings.embeddings.device
        settings.embeddings.device = "cuda"
        logger.info("GPU mode enabled for embeddings (via --gpu flag, was: %s)", original_device)
    
    # Load queries
    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[:args.limit]
    
    # Run evaluation
    summary = await run_rlaif_evaluation(queries, args.output)
    
    print("\n" + "=" * 60)
    print("RLAIF Evaluation Summary")
    print("=" * 60)
    print(f"Total queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful_queries']} ({summary['success_rate']*100:.1f}%)")
    print(f"Average latency: {summary['latency']['mean']:.0f}ms")
    print(f"P50 latency: {summary['latency']['p50']:.0f}ms")
    print(f"P95 latency: {summary['latency']['p95']:.0f}ms")
    print(f"Average reward: {summary['rewards']['mean']:.3f}")
    print(f"Average citations: {summary['citations']['mean']:.1f}")
    print(f"Average retrieval depth: {summary['retrieval_depth']['mean']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


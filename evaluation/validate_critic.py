#!/usr/bin/env python3
"""
Validate CriticAgent reward scores against baseline evaluation results.

This script runs the CriticAgent on baseline queries and compares
reward scores to identify correlations and validate the reward system.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import Orchestrator


async def validate_critic_on_baseline(baseline_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Run CriticAgent on baseline queries and generate validation report.
    """
    # Load baseline results
    with baseline_path.open() as f:
        baseline_data = json.load(f)
    
    queries = baseline_data.get("queries", [])
    print(f"Validating CriticAgent on {len(queries)} baseline queries...")
    
    orchestrator = Orchestrator()
    results = []
    
    for idx, query_data in enumerate(queries, 1):
        query = query_data.get("query", "")
        if not query:
            continue
        
        print(f"[{idx}/{len(queries)}] Processing: {query[:60]}...")
        
        try:
            # Run orchestrator (includes CriticAgent if enabled)
            agent_results = await orchestrator.run_all(query)
            
            # Extract critic result
            critic_result = agent_results.get("critic")
            if critic_result and critic_result.success:
                reward_data = critic_result.data
                results.append({
                    "query_id": query_data.get("query_id"),
                    "query": query,
                    "total_reward": reward_data.get("score"),
                    "components": reward_data.get("components", {}),
                    "baseline_success": query_data.get("success", False),
                    "baseline_citation_count": query_data.get("citation_count", 0),
                })
                print(f"  ✓ Reward: {reward_data.get('score', 0):.3f}")
            else:
                print(f"  ✗ CriticAgent not available or failed")
                results.append({
                    "query_id": query_data.get("query_id"),
                    "query": query,
                    "error": "CriticAgent not available",
                })
        
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            results.append({
                "query_id": query_data.get("query_id"),
                "query": query,
                "error": str(exc),
            })
    
    # Compute statistics
    successful = [r for r in results if "total_reward" in r]
    if successful:
        rewards = [r["total_reward"] for r in successful]
        avg_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        # Component averages
        components_avg = {}
        for component in ["citation_overlap", "cpc_relevance", "temporal_diversity", "llm_fluency"]:
            component_scores = [r["components"].get(component, 0) for r in successful if "components" in r]
            if component_scores:
                components_avg[component] = sum(component_scores) / len(component_scores)
        
        stats = {
            "total_queries": len(queries),
            "successful_validations": len(successful),
            "reward_statistics": {
                "mean": avg_reward,
                "min": min_reward,
                "max": max_reward,
            },
            "component_averages": components_avg,
        }
    else:
        stats = {
            "total_queries": len(queries),
            "successful_validations": 0,
            "error": "No successful validations",
        }
    
    # Generate report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "baseline_file": str(baseline_path),
        "statistics": stats,
        "results": results,
    }
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Validation complete!")
    print(f"  Successful: {stats.get('successful_validations', 0)}/{stats.get('total_queries', 0)}")
    if successful:
        print(f"  Avg reward: {stats['reward_statistics']['mean']:.3f}")
        print(f"  Reward range: [{stats['reward_statistics']['min']:.3f}, {stats['reward_statistics']['max']:.3f}]")
    print(f"  Results saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate CriticAgent on baseline queries")
    parser.add_argument(
        "--baseline",
        type=str,
        default="evaluation/baseline_scores.json",
        help="Path to baseline evaluation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/critic_validation.json",
        help="Output path for validation results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to validate (for testing)",
    )
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline)
    output_path = Path(args.output)
    
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    # Run validation
    report = asyncio.run(validate_critic_on_baseline(baseline_path, output_path))
    
    # Print summary
    if report.get("statistics", {}).get("successful_validations", 0) > 0:
        print("\n=== Validation Summary ===")
        stats = report["statistics"]
        print(f"Success Rate: {stats['successful_validations']}/{stats['total_queries']}")
        if "reward_statistics" in stats:
            print(f"Average Reward: {stats['reward_statistics']['mean']:.3f}")
            if "component_averages" in stats:
                print("\nComponent Averages:")
                for component, avg in stats["component_averages"].items():
                    print(f"  {component}: {avg:.3f}")


if __name__ == "__main__":
    main()


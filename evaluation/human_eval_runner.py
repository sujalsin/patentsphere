#!/usr/bin/env python3
"""
Human Evaluation Runner.

Automates running queries for human evaluation and collecting scores.
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


async def run_query_for_evaluation(
    orchestrator: Orchestrator, query: str, mode: str = "baseline"
) -> Dict[str, Any]:
    """
    Run query and format for human evaluation.
    
    Args:
        orchestrator: Orchestrator instance
        query: Query string
        mode: "baseline" or "rlaif"
    
    Returns:
        Formatted result for human evaluation
    """
    start_time = time.perf_counter()
    
    try:
        results = await orchestrator.run_all(query)
        elapsed = time.perf_counter() - start_time
        
        # Extract synthesis output
        synthesis_data = results.get("synthesis")
        citation_data = results.get("citation_mapper") or results.get("adaptive_retrieval")
        
        if not synthesis_data or not synthesis_data.success:
            return {
                "query": query,
                "mode": mode,
                "success": False,
                "error": "Synthesis failed",
            }
        
        synthesis_output = synthesis_data.data
        
        evaluation_data = {
            "query": query,
            "mode": mode,
            "success": True,
            "latency_ms": elapsed * 1000,
            "summary": synthesis_output.get("summary", ""),
            "key_points": synthesis_output.get("key_points", []),
            "risk_assessment": synthesis_output.get("risk_assessment", ""),
            "citations": [],
        }
        
        # Add citations
        if citation_data and citation_data.success:
            results_list = citation_data.data.get("results", [])
            evaluation_data["citations"] = [
                {
                    "patent_id": r.get("patent_id", ""),
                    "score": r.get("score", 0.0),
                    "preview": r.get("chunk_text", "")[:200],
                }
                for r in results_list[:10]  # Top 10
            ]
        
        return evaluation_data
        
    except Exception as exc:
        logger.error("Query evaluation failed: %s", exc)
        return {
            "query": query,
            "mode": mode,
            "success": False,
            "error": str(exc),
        }


async def generate_evaluation_pairs(
    queries: List[str], output_dir: Path
) -> None:
    """
    Generate evaluation pairs (baseline and RLAIF) for human grading.
    
    Args:
        queries: List of queries to evaluate
        output_dir: Directory to save evaluation files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create orchestrators for baseline and RLAIF
    # For baseline: disable adaptive retrieval
    # For RLAIF: enable adaptive retrieval
    
    logger.info("Generating evaluation pairs for %d queries", len(queries))
    
    # Load config and modify for baseline
    from config.settings import get_settings
    settings = get_settings()
    
    # Baseline: disable adaptive retrieval
    settings.adaptive_retrieval.enabled = False
    baseline_orch = Orchestrator()
    
    # RLAIF: enable adaptive retrieval
    settings.adaptive_retrieval.enabled = True
    rlaif_orch = Orchestrator()
    
    evaluation_pairs = []
    
    for i, query in enumerate(queries, 1):
        logger.info("Processing query %d/%d: %s", i, len(queries), query[:50])
        
        # Run baseline
        baseline_result = await run_query_for_evaluation(
            baseline_orch, query, mode="baseline"
        )
        
        # Run RLAIF
        rlaif_result = await run_query_for_evaluation(
            rlaif_orch, query, mode="rlaif"
        )
        
        evaluation_pairs.append({
            "query_id": i,
            "query": query,
            "baseline": baseline_result,
            "rlaif": rlaif_result,
        })
    
    # Save evaluation pairs
    pairs_file = output_dir / "evaluation_pairs.json"
    with open(pairs_file, "w") as f:
        json.dump(evaluation_pairs, f, indent=2)
    
    logger.info("Saved %d evaluation pairs to %s", len(evaluation_pairs), pairs_file)
    
    # Generate human evaluation forms
    forms_file = output_dir / "human_eval_forms_generated.md"
    with open(forms_file, "w") as f:
        f.write("# Human Evaluation Forms\n\n")
        f.write("Please rate each response on a scale of 1-5:\n")
        f.write("- 1: Poor (irrelevant, incorrect, unhelpful)\n")
        f.write("- 2: Below Average (mostly irrelevant, some errors)\n")
        f.write("- 3: Average (partially relevant, minor errors)\n")
        f.write("- 4: Good (relevant, mostly correct, helpful)\n")
        f.write("- 5: Excellent (highly relevant, accurate, very helpful)\n\n")
        f.write("---\n\n")
        
        for pair in evaluation_pairs:
            f.write(f"## Query {pair['query_id']}: {pair['query']}\n\n")
            
            # Baseline
            f.write("### Baseline Response\n\n")
            if pair['baseline']['success']:
                f.write(f"**Summary:** {pair['baseline'].get('summary', 'N/A')}\n\n")
                f.write(f"**Citations:** {len(pair['baseline'].get('citations', []))}\n\n")
            else:
                f.write(f"**Error:** {pair['baseline'].get('error', 'Unknown')}\n\n")
            
            f.write("**Rating (1-5):** ____\n\n")
            f.write("**Comments:**\n\n")
            
            # RLAIF
            f.write("### RLAIF Response\n\n")
            if pair['rlaif']['success']:
                f.write(f"**Summary:** {pair['rlaif'].get('summary', 'N/A')}\n\n")
                f.write(f"**Citations:** {len(pair['rlaif'].get('citations', []))}\n\n")
            else:
                f.write(f"**Error:** {pair['rlaif'].get('error', 'Unknown')}\n\n")
            
            f.write("**Rating (1-5):** ____\n\n")
            f.write("**Comments:**\n\n")
            f.write("---\n\n")
    
    logger.info("Generated human evaluation forms: %s", forms_file)


def load_selected_queries(query_file: Path) -> List[str]:
    """Load selected queries for human evaluation."""
    with open(query_file, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            return [q if isinstance(q, str) else q.get("query", "") for q in data]
        elif isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
            return [q if isinstance(q, str) else q.get("query", "") for q in queries]
        else:
            raise ValueError("Invalid query file format")


async def aggregate_human_scores(scores_file: Path) -> Dict[str, Any]:
    """
    Aggregate human evaluation scores.
    
    Expected format: JSON with scores for each query
    """
    if not scores_file.exists():
        logger.warning("Scores file not found: %s", scores_file)
        return {}
    
    with open(scores_file, "r") as f:
        scores_data = json.load(f)
    
    baseline_scores = []
    rlaif_scores = []
    
    for item in scores_data:
        if "baseline_rating" in item:
            baseline_scores.append(float(item["baseline_rating"]))
        if "rlaif_rating" in item:
            rlaif_scores.append(float(item["rlaif_rating"]))
    
    if not baseline_scores or not rlaif_scores:
        return {"error": "Insufficient scores"}
    
    import numpy as np
    from scipy import stats
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(rlaif_scores, baseline_scores)
    
    summary = {
        "baseline": {
            "mean": float(np.mean(baseline_scores)),
            "std": float(np.std(baseline_scores)),
            "min": float(np.min(baseline_scores)),
            "max": float(np.max(baseline_scores)),
        },
        "rlaif": {
            "mean": float(np.mean(rlaif_scores)),
            "std": float(np.std(rlaif_scores)),
            "min": float(np.min(rlaif_scores)),
            "max": float(np.max(rlaif_scores)),
        },
        "improvement": float(np.mean(rlaif_scores) - np.mean(baseline_scores)),
        "improvement_pct": float(
            (np.mean(rlaif_scores) - np.mean(baseline_scores))
            / np.mean(baseline_scores)
            * 100
        ),
        "statistical_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        },
    }
    
    return summary


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Human evaluation runner")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/human_eval_selected.json"),
        help="Path to selected queries JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/human_eval"),
        help="Output directory for evaluation files",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        help="Path to human scores JSON (for aggregation)",
    )
    
    args = parser.parse_args()
    
    if args.scores:
        # Aggregate scores
        summary = await aggregate_human_scores(args.scores)
        print("\n" + "=" * 60)
        print("Human Evaluation Summary")
        print("=" * 60)
        print(f"Baseline mean: {summary.get('baseline', {}).get('mean', 0):.2f}")
        print(f"RLAIF mean:    {summary.get('rlaif', {}).get('mean', 0):.2f}")
        print(f"Improvement:   {summary.get('improvement', 0):+.2f} ({summary.get('improvement_pct', 0):+.1f}%)")
        if "statistical_test" in summary:
            test = summary["statistical_test"]
            print(f"p-value:       {test.get('p_value', 0):.4f} {'(significant)' if test.get('significant') else '(not significant)'}")
        print("=" * 60)
    else:
        # Generate evaluation pairs
        queries = load_selected_queries(args.queries)
        await generate_evaluation_pairs(queries, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())


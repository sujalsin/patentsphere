#!/usr/bin/env python3
"""
Statistical Comparison: Baseline vs RLAIF.

Performs paired t-test and generates comparison report.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_metrics(results: Dict[str, Any], metric_name: str) -> List[float]:
    """Extract metric values from results."""
    metrics = []
    for result in results.get("results", []):
        if result.get("success", False):
            value = result.get(metric_name)
            if value is not None:
                metrics.append(float(value))
    return metrics


def calculate_precision_at_k(results: Dict[str, Any], k: int = 10) -> List[float]:
    """
    Calculate precision@k for each query.
    
    Simplified: uses citation scores as relevance proxy.
    """
    precisions = []
    for result in results.get("results", []):
        if result.get("success", False):
            scores = result.get("citation_scores", [])
            if scores:
                # Top-k scores
                top_k_scores = sorted(scores, reverse=True)[:k]
                # Precision = average score (normalized)
                precision = sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0.0
                precisions.append(precision)
    return precisions


def calculate_recall_at_k(results: Dict[str, Any], k: int = 10) -> List[float]:
    """
    Calculate recall@k for each query.
    
    Simplified: uses number of citations as recall proxy.
    """
    recalls = []
    for result in results.get("results", []):
        if result.get("success", False):
            num_citations = result.get("num_citations", 0)
            # Normalize to [0, 1] range (assuming max ~50 citations)
            recall = min(1.0, num_citations / k)
            recalls.append(recall)
    return recalls


def paired_t_test(baseline: List[float], rlaif: List[float]) -> Dict[str, Any]:
    """
    Perform paired t-test.
    
    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    if len(baseline) != len(rlaif):
        # Pad shorter list with mean if lengths differ
        min_len = min(len(baseline), len(rlaif))
        baseline = baseline[:min_len]
        rlaif = rlaif[:min_len]
    
    if len(baseline) < 2:
        return {"error": "Insufficient data for t-test"}
    
    # Calculate differences
    differences = [r - b for b, r in zip(baseline, rlaif)]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(rlaif, baseline)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Confidence interval (95%)
    n = len(differences)
    se = std_diff / np.sqrt(n) if n > 0 else 0.0
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(mean_diff),
        "cohens_d": float(cohens_d),
        "confidence_interval_95": [float(ci_lower), float(ci_upper)],
        "n": n,
        "significant": p_value < 0.05,
    }


def generate_comparison_report(
    baseline_path: Path, rlaif_path: Path, output_path: Path
) -> None:
    """Generate comparison report between baseline and RLAIF."""
    baseline_results = load_results(baseline_path)
    rlaif_results = load_results(rlaif_path)
    
    # Extract metrics
    baseline_latency = extract_metrics(baseline_results, "latency_ms")
    rlaif_latency = extract_metrics(rlaif_results, "latency_ms")
    
    baseline_reward = extract_metrics(baseline_results, "reward")
    rlaif_reward = extract_metrics(rlaif_results, "reward")
    
    baseline_precision = calculate_precision_at_k(baseline_results, k=10)
    rlaif_precision = calculate_precision_at_k(rlaif_results, k=10)
    
    baseline_recall = calculate_recall_at_k(baseline_results, k=10)
    rlaif_recall = calculate_recall_at_k(rlaif_results, k=10)
    
    # Statistical tests
    latency_test = paired_t_test(baseline_latency, rlaif_latency)
    reward_test = paired_t_test(baseline_reward, rlaif_reward)
    precision_test = paired_t_test(baseline_precision, rlaif_precision)
    recall_test = paired_t_test(baseline_recall, rlaif_recall)
    
    # Generate report
    report = {
        "baseline_file": str(baseline_path),
        "rlaif_file": str(rlaif_path),
        "comparison": {
            "latency": {
                "baseline_mean": float(np.mean(baseline_latency)) if baseline_latency else 0.0,
                "rlaif_mean": float(np.mean(rlaif_latency)) if rlaif_latency else 0.0,
                "improvement_pct": float(
                    (np.mean(rlaif_latency) - np.mean(baseline_latency))
                    / np.mean(baseline_latency)
                    * 100
                ) if baseline_latency and np.mean(baseline_latency) > 0 else 0.0,
                "statistical_test": latency_test,
            },
            "reward": {
                "baseline_mean": float(np.mean(baseline_reward)) if baseline_reward else 0.0,
                "rlaif_mean": float(np.mean(rlaif_reward)) if rlaif_reward else 0.0,
                "improvement_pct": float(
                    (np.mean(rlaif_reward) - np.mean(baseline_reward))
                    / np.mean(baseline_reward)
                    * 100
                ) if baseline_reward and np.mean(baseline_reward) > 0 else 0.0,
                "statistical_test": reward_test,
            },
            "precision_at_10": {
                "baseline_mean": float(np.mean(baseline_precision)) if baseline_precision else 0.0,
                "rlaif_mean": float(np.mean(rlaif_precision)) if rlaif_precision else 0.0,
                "improvement_pct": float(
                    (np.mean(rlaif_precision) - np.mean(baseline_precision))
                    / np.mean(baseline_precision)
                    * 100
                ) if baseline_precision and np.mean(baseline_precision) > 0 else 0.0,
                "statistical_test": precision_test,
            },
            "recall_at_10": {
                "baseline_mean": float(np.mean(baseline_recall)) if baseline_recall else 0.0,
                "rlaif_mean": float(np.mean(rlaif_recall)) if rlaif_recall else 0.0,
                "improvement_pct": float(
                    (np.mean(rlaif_recall) - np.mean(baseline_recall))
                    / np.mean(baseline_recall)
                    * 100
                ) if baseline_recall and np.mean(baseline_recall) > 0 else 0.0,
                "statistical_test": recall_test,
            },
        },
    }
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Baseline vs RLAIF Comparison")
    print("=" * 60)
    
    print("\nLatency:")
    print(f"  Baseline: {report['comparison']['latency']['baseline_mean']:.0f}ms")
    print(f"  RLAIF:    {report['comparison']['latency']['rlaif_mean']:.0f}ms")
    print(f"  Change:   {report['comparison']['latency']['improvement_pct']:+.1f}%")
    if "statistical_test" in report['comparison']['latency']:
        test = report['comparison']['latency']['statistical_test']
        if "p_value" in test:
            print(f"  p-value:  {test['p_value']:.4f} {'(significant)' if test.get('significant') else '(not significant)'}")
    
    print("\nReward:")
    print(f"  Baseline: {report['comparison']['reward']['baseline_mean']:.3f}")
    print(f"  RLAIF:    {report['comparison']['reward']['rlaif_mean']:.3f}")
    print(f"  Change:   {report['comparison']['reward']['improvement_pct']:+.1f}%")
    if "statistical_test" in report['comparison']['reward']:
        test = report['comparison']['reward']['statistical_test']
        if "p_value" in test:
            print(f"  p-value:  {test['p_value']:.4f} {'(significant)' if test.get('significant') else '(not significant)'}")
    
    print("\nPrecision@10:")
    print(f"  Baseline: {report['comparison']['precision_at_10']['baseline_mean']:.3f}")
    print(f"  RLAIF:    {report['comparison']['precision_at_10']['rlaif_mean']:.3f}")
    print(f"  Change:   {report['comparison']['precision_at_10']['improvement_pct']:+.1f}%")
    if "statistical_test" in report['comparison']['precision_at_10']:
        test = report['comparison']['precision_at_10']['statistical_test']
        if "p_value" in test:
            print(f"  p-value:  {test['p_value']:.4f} {'(significant)' if test.get('significant') else '(not significant)'}")
    
    print("\nRecall@10:")
    print(f"  Baseline: {report['comparison']['recall_at_10']['baseline_mean']:.3f}")
    print(f"  RLAIF:    {report['comparison']['recall_at_10']['rlaif_mean']:.3f}")
    print(f"  Change:   {report['comparison']['recall_at_10']['improvement_pct']:+.1f}%")
    if "statistical_test" in report['comparison']['recall_at_10']:
        test = report['comparison']['recall_at_10']['statistical_test']
        if "p_value" in test:
            print(f"  p-value:  {test['p_value']:.4f} {'(significant)' if test.get('significant') else '(not significant)'}")
    
    print("\n" + "=" * 60)
    print(f"Report saved to: {output_path}")
    print("=" * 60)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline vs RLAIF results")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("evaluation/baseline_scores.json"),
        help="Path to baseline results JSON",
    )
    parser.add_argument(
        "--rlaif",
        type=Path,
        default=Path("evaluation/rlaif_scores.json"),
        help="Path to RLAIF results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/comparison_report.json"),
        help="Output path for comparison report",
    )
    
    args = parser.parse_args()
    
    generate_comparison_report(args.baseline, args.rlaif, args.output)


if __name__ == "__main__":
    main()


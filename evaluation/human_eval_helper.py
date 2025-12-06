#!/usr/bin/env python3
"""
Helper script for human evaluation of baseline results.

Selects 10 diverse queries for human grading and generates evaluation forms.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def select_queries_for_eval(baseline_results: Dict, num_queries: int = 10) -> List[Dict]:
    """Select diverse queries for human evaluation."""
    queries = baseline_results["queries"]
    
    # Sort by latency to get a mix of fast/slow
    sorted_queries = sorted(queries, key=lambda x: x["total_latency_ms"])
    
    # Select diverse set: first, last, and evenly spaced middle ones
    selected = []
    indices = [0]  # First (slowest)
    step = max(1, (len(sorted_queries) - 2) // (num_queries - 2))
    for i in range(1, num_queries - 1):
        idx = min(i * step, len(sorted_queries) - 1)
        indices.append(idx)
    indices.append(len(sorted_queries) - 1)  # Last (fastest)
    
    for idx in sorted(set(indices))[:num_queries]:
        selected.append(sorted_queries[idx])
    
    return selected


def generate_eval_forms(selected_queries: List[Dict], output_path: Path):
    """Generate markdown forms for human evaluation."""
    forms = []
    
    for query_data in selected_queries:
        query_id = query_data["query_id"]
        query = query_data["query"]
        citations = query_data.get("top_citations", [])[:5]  # Top 5 for review
        
        form = f"""## Query {query_id}: {query}

### Citations Retrieved:
"""
        for i, cit in enumerate(citations, 1):
            form += f"{i}. **{cit['patent_id']}** (score: {cit['score']:.3f}, type: {cit['chunk_type']})\n"
        
        form += f"""
### Evaluation:

- **Relevance (0-5)**: _How relevant are the retrieved patents to the query?_
  - 0: Not relevant
  - 1-2: Somewhat relevant
  - 3-4: Relevant
  - 5: Highly relevant

- **Actionability (0-5)**: _Can the results be used to answer the query?_
  - 0: Not actionable
  - 1-2: Somewhat actionable
  - 3-4: Actionable
  - 5: Highly actionable

- **Correctness (0-5)**: _Are the patent IDs and information correct?_
  - 0: Incorrect
  - 1-2: Partially correct
  - 3-4: Mostly correct
  - 5: Completely correct

**Your Scores:**
- Relevance: ___
- Actionability: ___
- Correctness: ___

**Notes:**
___

---
"""
        forms.append(form)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("# Human Evaluation Forms - Baseline Results\n\n")
        f.write(f"Total queries to evaluate: {len(selected_queries)}\n\n")
        f.write("---\n\n")
        f.write("\n".join(forms))
    
    print(f"✓ Generated {len(selected_queries)} evaluation forms at: {output_path}")


def main():
    baseline_path = Path("evaluation/baseline_scores.json")
    if not baseline_path.exists():
        print(f"Error: Baseline results not found at {baseline_path}")
        print("Run: python evaluation/baseline_runner.py")
        return
    
    with baseline_path.open() as f:
        baseline_results = json.load(f)
    
    # Select 10 queries for evaluation
    selected = select_queries_for_eval(baseline_results, num_queries=10)
    
    print(f"Selected {len(selected)} queries for human evaluation:")
    for q in selected:
        print(f"  [{q['query_id']}] {q['query'][:60]}... ({q['total_latency_ms']:.1f}ms)")
    
    # Generate evaluation forms
    output_path = Path("evaluation/human_eval_forms.md")
    generate_eval_forms(selected, output_path)
    
    # Also save selected queries as JSON for reference
    selected_path = Path("evaluation/human_eval_selected.json")
    with selected_path.open("w") as f:
        json.dump({"selected_queries": selected}, f, indent=2)
    print(f"✓ Saved selected queries to: {selected_path}")


if __name__ == "__main__":
    main()



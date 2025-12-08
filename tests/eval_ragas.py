"""E2E evaluation using Ragas for quality assessment."""
import pytest
import asyncio
from typing import List, Dict, Any
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: Ragas not installed. Install with: pip install ragas")


@pytest.mark.e2e
@pytest.mark.slow
class TestRagasEvaluation:
    """E2E evaluation tests using Ragas."""
    
    @pytest.fixture
    def golden_dataset(self):
        """Create golden dataset for evaluation."""
        return {
            'question': [
                'What are the key features of US-2016168309-A1?',
                'Has Apple been sued for touchscreen technology?',
                'Find patents related to car suspension systems',
            ],
            'answer': [
                'US-2016168309-A1 describes a composition obtained by blending a polythiol compound, an isocyanate group-containing compound and a radical generator. The invention provides a composition capable of bonding a rubber member strongly [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1).',
                'Based on the litigation data, there are cases involving touchscreen technology, though specific Apple cases would need further verification in the litigation database.',
                'Several patents relate to car suspension systems, including vehicle damping systems and active chassis control technologies. Key patents include [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1) which describes compositions for automotive applications.',
            ],
            'contexts': [
                [
                    'US-2016168309-A1: Composition, adhesive agent, adhesive sheet, and laminate. The present invention provides a composition obtained by blending a polythiol compound (A), an isocyanate group-containing compound (B) and a radical generator (C).',
                ],
                [
                    'Litigation case 1:00-cv-00037 involves patent US5655545 in E.D.Tex. court, filed 2000-01-13, status: active.',
                ],
                [
                    'US-2016168309-A1: Composition for automotive applications. US-2016168354-A1: Fluoroelastomer composition for vehicle components.',
                ],
            ],
            'ground_truth': [
                'US-2016168309-A1 features a composition with polythiol compounds, isocyanate groups, and radical generators for bonding rubber members.',
                'Yes, there have been litigation cases involving touchscreen and related technologies, though specific Apple cases require database verification.',
                'Patents related to car suspension include compositions and materials for automotive damping systems and chassis control.',
            ],
        }
    
    @pytest.mark.skipif(not RAGAS_AVAILABLE, reason="Ragas not installed")
    def test_ragas_faithfulness(self, golden_dataset):
        """Test faithfulness metric (no hallucinations)."""
        dataset = Dataset.from_dict(golden_dataset)
        
        results = evaluate(
            dataset,
            metrics=[faithfulness],
        )
        
        faithfulness_score = results['faithfulness']
        print(f"\nFaithfulness Score: {faithfulness_score}")
        
        assert faithfulness_score > 0.8, f"Faithfulness score {faithfulness_score} is below threshold 0.8"
    
    @pytest.mark.skipif(not RAGAS_AVAILABLE, reason="Ragas not installed")
    def test_ragas_answer_relevancy(self, golden_dataset):
        """Test answer relevancy metric."""
        dataset = Dataset.from_dict(golden_dataset)
        
        results = evaluate(
            dataset,
            metrics=[answer_relevancy],
        )
        
        relevancy_score = results['answer_relevancy']
        print(f"\nAnswer Relevancy Score: {relevancy_score}")
        
        assert relevancy_score > 0.8, f"Answer relevancy score {relevancy_score} is below threshold 0.8"
    
    @pytest.mark.skipif(not RAGAS_AVAILABLE, reason="Ragas not installed")
    def test_ragas_context_precision(self, golden_dataset):
        """Test context precision metric (retrieval quality)."""
        dataset = Dataset.from_dict(golden_dataset)
        
        results = evaluate(
            dataset,
            metrics=[context_precision],
        )
        
        precision_score = results['context_precision']
        print(f"\nContext Precision Score: {precision_score}")
        
        assert precision_score > 0.8, f"Context precision score {precision_score} is below threshold 0.8"
    
    @pytest.mark.skipif(not RAGAS_AVAILABLE, reason="Ragas not installed")
    def test_ragas_all_metrics(self, golden_dataset):
        """Test all Ragas metrics together."""
        dataset = Dataset.from_dict(golden_dataset)
        
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
            ],
        )
        
        print(f"\n{'='*60}")
        print("Ragas Evaluation Results")
        print(f"{'='*60}")
        for metric, score in results.items():
            print(f"{metric}: {score:.3f}")
        print(f"{'='*60}")
        
        # All metrics should be above threshold
        assert results['faithfulness'] > 0.8
        assert results['answer_relevancy'] > 0.8
        assert results['context_precision'] > 0.8


def run_ragas_evaluation():
    """Run Ragas evaluation manually (for CI/CD)."""
    if not RAGAS_AVAILABLE:
        print("Ragas not installed. Install with: pip install ragas")
        return
    
    from tests.eval_ragas import TestRagasEvaluation
    
    test_instance = TestRagasEvaluation()
    golden_dataset = test_instance.golden_dataset()
    
    dataset = Dataset.from_dict(golden_dataset)
    
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )
    
    print("\n" + "="*60)
    print("Ragas Evaluation Results")
    print("="*60)
    for metric, score in results.items():
        print(f"{metric}: {score:.3f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_ragas_evaluation()



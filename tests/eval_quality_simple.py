"""Simplified E2E Quality Evaluation - Tests with mock data first."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Ragas not installed. Install with: pip install ragas langchain-ollama")


# Test with mock/sample data to verify Ragas setup works
def test_ragas_setup():
    """Test Ragas evaluation with sample data to verify setup."""
    if not RAGAS_AVAILABLE:
        print("❌ Ragas not available")
        return False
    
    print("🧪 Testing Ragas Setup with Sample Data...")
    print()
    
    # Sample test data
    test_data = {
        "question": [
            "What is the primary CPC code for vehicle suspension?",
            "Find patents related to car suspension systems",
        ],
        "answer": [
            "The primary CPC classification for vehicle suspension arrangements is B60G. This code covers various suspension systems including active damping and chassis control technologies [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1).",
            "Patents related to car suspension systems include vehicle damping systems and active chassis control technologies. Key patents describe compositions for automotive applications [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1).",
        ],
        "contexts": [
            [
                "US-2016168309-A1: Composition, adhesive agent, adhesive sheet, and laminate. The present invention provides a composition obtained by blending a polythiol compound (A), an isocyanate group-containing compound (B) and a radical generator (C).",
                "B60G is the CPC code for vehicle suspension arrangements, covering various automotive suspension systems.",
            ],
            [
                "US-2016168309-A1: Composition for automotive applications with polythiol compounds and isocyanate groups for bonding rubber members.",
                "Vehicle suspension systems include damping mechanisms and active chassis control technologies classified under CPC B60G.",
            ],
        ],
        "ground_truth": [
            "The primary CPC classification for vehicle suspension arrangements is B60G.",
            "Patents related to car suspension systems include vehicle damping systems, active chassis control technologies, and automotive suspension components classified under CPC code B60G.",
        ],
    }
    
    dataset = Dataset.from_dict(test_data)
    
    try:
        print("Loading Judge Models (Ollama)...")
        judge_llm = ChatOllama(model="llama3.2:3b", temperature=0, base_url="http://localhost:11434")
        judge_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        print("✅ Judge models loaded")
        print()
        
        print("Running Evaluation...")
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        
        print("\n" + "=" * 60)
        print("          TEST RESULTS")
        print("=" * 60)
        
        # Handle results - may have parsing errors but still have metrics
        try:
            df = results.to_pandas()
            print("\n📊 Results DataFrame:")
            print(df)
            
            # Try to extract metrics
            metrics_available = []
            if 'faithfulness' in df.columns:
                avg_faithfulness = df['faithfulness'].mean()
                print(f"\nAverage Faithfulness:    {avg_faithfulness:.3f}")
                metrics_available.append(('faithfulness', avg_faithfulness))
            
            if 'answer_relevancy' in df.columns:
                avg_relevancy = df['answer_relevancy'].mean()
                print(f"Average Answer Relevancy: {avg_relevancy:.3f}")
                metrics_available.append(('answer_relevancy', avg_relevancy))
            
            if 'context_precision' in df.columns:
                avg_precision = df['context_precision'].mean()
                print(f"Average Context Precision: {avg_precision:.3f}")
                metrics_available.append(('context_precision', avg_precision))
            
            # Also try to get metrics from results dict
            if hasattr(results, '__dict__'):
                print("\n📈 Metrics from results object:")
                for key, value in results.__dict__.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3f}")
                        metrics_available.append((key, value))
            
            # Check if we got any metrics
            if metrics_available:
                print("\n✅ Ragas evaluation completed!")
                print("   Note: Some JSON parsing errors occurred, but metrics were calculated.")
                return True
            else:
                print("\n⚠️  Evaluation ran but no metrics extracted.")
                print("   This may be due to JSON parsing issues with the judge model.")
                return False
                
        except Exception as e:
            print(f"\n⚠️  Error extracting results: {e}")
            print("   Evaluation ran but results extraction failed.")
            print("   This is often due to JSON parsing issues with smaller LLM models.")
            print("\n   Recommendation: Use a larger model for the judge (e.g., llama3.2:1b or larger)")
            return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ragas_setup()
    sys.exit(0 if success else 1)


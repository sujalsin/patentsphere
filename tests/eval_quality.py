"""E2E Quality Evaluation using Ragas with local Ollama models."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import ChatOllama, OllamaEmbeddings

from graph.graph import workflow, AgentState


# ---------------------------------------------------------
# 1. SETUP: The "Golden Dataset"
# ---------------------------------------------------------
# These are the questions we expect the system to answer correctly.
test_data = [
    {
        "question": "What is the primary CPC code for vehicle suspension arrangements?",
        "ground_truth": "The primary CPC classification for vehicle suspension arrangements is B60G."
    },
    {
        "question": "Find patents related to car suspension systems",
        "ground_truth": "Patents related to car suspension systems include vehicle damping systems, active chassis control technologies, and automotive suspension components classified under CPC code B60G."
    },
    {
        "question": "What patents describe adhesive compositions for automotive applications?",
        "ground_truth": "Patents describing adhesive compositions for automotive applications include compositions with polythiol compounds, isocyanate groups, and radical generators for bonding rubber members in vehicles."
    },
    {
        "question": "Has Apple sued anyone over touchscreen technology?",
        "ground_truth": "Based on the litigation database, there have been various cases involving touchscreen technology, though specific Apple cases would need verification in the litigation records."
    },
    {
        "question": "What are the key features of fluoroelastomer compositions?",
        "ground_truth": "Fluoroelastomer compositions typically include at least one fluoroelastomer, meta-divinylbenzene in specific weight parts, and at least one peroxide for curing applications."
    }
]


# ---------------------------------------------------------
# 2. GENERATION: Run Your System
# ---------------------------------------------------------
async def generate_system_outputs():
    """Run the actual graph workflow and collect outputs."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("🤖 System is processing queries...")
    print()

    for i, item in enumerate(test_data, 1):
        print(f"[{i}/{len(test_data)}] Processing: {item['question'][:60]}...")

        # Initialize state for the workflow
        initial_state: AgentState = {
            "query": item["question"],
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "draft": None,
            "critique": None,
            "retry_count": 0,
            "final_response": None,
        }

        config = {"configurable": {"thread_id": f"eval_{i}"}}

        try:
            # Run the workflow
            final_state = None
            async for state in workflow.astream(initial_state, config=config, stream_mode="values"):
                final_state = state

            if not final_state:
                print(f"   ⚠️  No state returned for query {i}")
                continue

            # Capture data
            questions.append(item["question"])
            ground_truths.append(item["ground_truth"])

            # Get final response
            final_response = final_state.get("final_response") or final_state.get("draft", "No response generated.")
            answers.append(final_response)

            # Extract text from retrieved documents for the judge
            documents = final_state.get("documents", [])
            retrieved_texts = []
            for doc in documents:
                # Build context text from document
                text_parts = []
                if doc.get("title"):
                    text_parts.append(f"Title: {doc['title']}")
                if doc.get("abstract"):
                    text_parts.append(f"Abstract: {doc['abstract']}")
                if doc.get("patent_id"):
                    text_parts.append(f"Patent ID: {doc['patent_id']}")
                
                if text_parts:
                    retrieved_texts.append(" | ".join(text_parts))
                elif isinstance(doc, str):
                    retrieved_texts.append(doc)
                else:
                    retrieved_texts.append(str(doc))

            # If no documents retrieved, add a placeholder
            if not retrieved_texts:
                retrieved_texts = ["No documents retrieved"]

            contexts.append(retrieved_texts)

            print(f"   ✅ Completed: {len(retrieved_texts)} documents, {len(final_response)} chars response")
            print()

        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            # Add placeholder data to maintain structure
            questions.append(item["question"])
            ground_truths.append(item["ground_truth"])
            answers.append(f"Error: {str(e)}")
            contexts.append(["Error retrieving documents"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }


# ---------------------------------------------------------
# 3. EVALUATION: Run Ragas Judge
# ---------------------------------------------------------
def run_evaluation():
    """Run the complete evaluation pipeline."""
    print("=" * 60)
    print("🚀 Starting E2E Quality Evaluation")
    print("=" * 60)
    print()

    # 1. Generate Data
    print("Step 1: Generating system outputs...")
    data_dict = asyncio.run(generate_system_outputs())
    
    # Check if we have valid data
    if not data_dict["question"]:
        print("❌ No data generated. Cannot run evaluation.")
        return
    
    dataset = Dataset.from_dict(data_dict)
    print(f"✅ Generated {len(data_dict['question'])} test cases")
    print()

    # 2. Configure Local Judge (Llama 3.2 is fast and decent for grading)
    print("Step 2: Loading Judge Models (Ollama)...")
    try:
        judge_llm = ChatOllama(model="llama3.2:3b", temperature=0, base_url="http://localhost:11434")
        judge_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        print("✅ Judge models loaded")
    except Exception as e:
        print(f"❌ Error loading judge models: {e}")
        print("   Make sure Ollama is running and models are installed:")
        print("   - ollama pull llama3.2:3b")
        print("   - ollama pull nomic-embed-text")
        return
    print()

    # 3. Run Ragas
    print("Step 3: Running Evaluation Metrics...")
    print("   - Faithfulness (Hallucination Check)")
    print("   - Answer Relevancy (Topic Relevance)")
    print("   - Context Precision (Search Quality)")
    print()

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,      # Hallucination Checker
                answer_relevancy,  # Did we stay on topic?
                context_precision  # Search Quality
            ],
            llm=judge_llm,
            embeddings=judge_embeddings
        )

        # 4. Output Results
        print("\n" + "=" * 60)
        print("          EVALUATION REPORT")
        print("=" * 60)
        
        try:
            df = results.to_pandas()
            
            # Display all available columns for debugging
            print(f"\n📋 Available columns: {df.columns.tolist()}")
            print(f"📋 DataFrame shape: {df.shape}")
            
            # Display detailed results
            print("\n📊 Detailed Results:")
            print("-" * 60)
            
            available_cols = df.columns.tolist()
            metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision']
            
            # Find question column
            question_col = None
            for col in ['question', 'user_input', 'questions', 'user_question']:
                if col in available_cols:
                    question_col = col
                    break
            
            # Display results
            for idx in range(len(df)):
                row = df.iloc[idx]
                if question_col:
                    try:
                        question = str(row[question_col])[:60]
                    except:
                        question = f'Query {idx + 1}'
                else:
                    question = f'Query {idx + 1}'
                print(f"\nQuery {idx + 1}: {question}...")
                for metric in metric_cols:
                    if metric in available_cols:
                        try:
                            value = row[metric]
                            if PANDAS_AVAILABLE and pd.notna(value) or (not PANDAS_AVAILABLE and value is not None):
                                print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                            else:
                                print(f"  {metric.replace('_', ' ').title()}: N/A")
                        except:
                            print(f"  {metric.replace('_', ' ').title()}: Error")
            
            print("\n" + "-" * 60)
            print("\n📈 Summary Statistics:")
            print("-" * 60)
            
            # Calculate averages
            avg_faithfulness = None
            avg_relevancy = None
            avg_precision = None
            
            if 'faithfulness' in available_cols:
                faithfulness_values = [float(v) for v in df['faithfulness'] if (PANDAS_AVAILABLE and pd.notna(v)) or (not PANDAS_AVAILABLE and v is not None)]
                if faithfulness_values:
                    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values)
                    print(f"Average Faithfulness:    {avg_faithfulness:.3f} ({len(faithfulness_values)}/{len(df)} valid)")
                else:
                    print(f"Average Faithfulness:    N/A")
            
            if 'answer_relevancy' in available_cols:
                relevancy_values = [float(v) for v in df['answer_relevancy'] if (PANDAS_AVAILABLE and pd.notna(v)) or (not PANDAS_AVAILABLE and v is not None)]
                if relevancy_values:
                    avg_relevancy = sum(relevancy_values) / len(relevancy_values)
                    print(f"Average Answer Relevancy: {avg_relevancy:.3f} ({len(relevancy_values)}/{len(df)} valid)")
                else:
                    print(f"Average Answer Relevancy: N/A")
            
            if 'context_precision' in available_cols:
                precision_values = [float(v) for v in df['context_precision'] if (PANDAS_AVAILABLE and pd.notna(v)) or (not PANDAS_AVAILABLE and v is not None)]
                if precision_values:
                    avg_precision = sum(precision_values) / len(precision_values)
                    print(f"Average Context Precision: {avg_precision:.3f} ({len(precision_values)}/{len(df)} valid)")
                else:
                    print(f"Average Context Precision: N/A")
                    
        except Exception as e:
            print(f"\n⚠️  Error extracting results: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to get metrics from results object directly
            print("\n📈 Attempting alternative metric extraction...")
            avg_faithfulness = getattr(results, 'faithfulness', None)
            avg_relevancy = getattr(results, 'answer_relevancy', None)
            avg_precision = getattr(results, 'context_precision', None)
            
            if avg_faithfulness is not None:
                print(f"Average Faithfulness:    {avg_faithfulness:.3f}")
            if avg_relevancy is not None:
                print(f"Average Answer Relevancy: {avg_relevancy:.3f}")
            if avg_precision is not None:
                print(f"Average Context Precision: {avg_precision:.3f}")
        
        print("\n" + "=" * 60)
        print("          VERDICT")
        print("=" * 60)
        
        # Calculate Overall Pass/Fail (handle None values)
        if avg_faithfulness is None:
            print("\n⚠️  Could not calculate Faithfulness score due to parsing errors.")
            print("   This is common with smaller LLM models. Consider using a larger judge model.")
        elif avg_faithfulness < 0.7:
            print("\n❌ FAILED: System is hallucinating too much.")
            print(f"   Faithfulness score {avg_faithfulness:.3f} is below threshold 0.7")
            print("   Fix: Update Critic Agent in AGENTS.md to be stricter about 'Only use provided context.'")
        elif avg_faithfulness >= 0.9:
            print("\n✅ EXCELLENT: System shows excellent reliability.")
            print(f"   Faithfulness score {avg_faithfulness:.3f} indicates minimal hallucinations")
        else:
            print("\n⚠️  WARNING: System needs tuning.")
            print(f"   Faithfulness score {avg_faithfulness:.3f} is acceptable but could be improved")
        
        if avg_precision is None:
            print("\n⚠️  Could not calculate Context Precision score.")
        elif avg_precision < 0.6:
            print("\n❌ BAD SEARCH: Retrieved documents don't contain the answer.")
            print(f"   Context Precision {avg_precision:.3f} is below threshold 0.6")
            print("   Fix: Check if qdrant_client.py is correctly weighing Sparse Vectors (keywords).")
        elif avg_precision >= 0.7:
            print("✅ GOOD SEARCH: Retrieved documents are relevant.")
        else:
            print("⚠️  SEARCH NEEDS IMPROVEMENT: Some retrieved documents may not be relevant.")
        
        if avg_relevancy is None:
            print("\n⚠️  Could not calculate Answer Relevancy score.")
        elif avg_relevancy < 0.7:
            print("\n❌ OFF TOPIC: Answers are vague or don't address the question.")
            print(f"   Answer Relevancy {avg_relevancy:.3f} is below threshold 0.7")
            print("   Fix: Update Synthesizer prompt to be more direct and concise.")
        elif avg_relevancy >= 0.8:
            print("✅ ON TOPIC: Answers directly address the questions.")
        else:
            print("⚠️  RELEVANCY NEEDS IMPROVEMENT: Some answers may be slightly off-topic.")
        
        # Final verdict
        print("\n" + "=" * 60)
        if (avg_faithfulness is not None and avg_precision is not None and avg_relevancy is not None and
            avg_faithfulness >= 0.8 and avg_precision >= 0.7 and avg_relevancy >= 0.7):
            print("✅ PRODUCTION READY: All metrics meet quality thresholds!")
            print("   Your backend is ready for frontend integration.")
        elif (avg_faithfulness is not None and avg_precision is not None and
              avg_faithfulness >= 0.8 and avg_precision >= 0.7):
            print("✅ MOSTLY READY: Core metrics (Faithfulness, Precision) are good.")
            print("   Consider improving Answer Relevancy before production.")
        else:
            print("⚠️  NEEDS WORK: System requires improvements before production.")
            print("   Focus on the metrics below threshold.")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_evaluation()


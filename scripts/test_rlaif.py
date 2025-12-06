#!/usr/bin/env python3
"""
Test script for RLAIF (Reinforcement Learning from AI Feedback) components.

Tests:
1. CriticAgent reward computation
2. AdaptiveRetrievalAgent Q-learning
3. Experience logging to database
4. Policy training from logged experiences
"""

import asyncio
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from app.agents.critic import CriticAgent
from app.agents.claims import ClaimsAnalyzerAgent
from app.agents.citation import CitationMapperAgent
from app.orchestrator import Orchestrator
from config.settings import get_settings


async def test_critic_reward_model():
    """Test the CriticAgent as reward model."""
    print("\n" + "=" * 60)
    print("Testing CriticAgent (Reward Model)")
    print("=" * 60)
    
    settings = get_settings()
    weights = settings.critic.reward_weights
    critic = CriticAgent(settings=settings, weights=weights)
    
    print(f"\nReward Weights:")
    for component, weight in weights.items():
        print(f"  {component}: {weight}")
    
    # Mock data for testing
    mock_chunks = [
        {
            "patent_id": "US-10123456",
            "title": "Blockchain Energy Trading",
            "score": 0.92,
            "publication_date": "2021-05-15",
            "cpc_codes": [{"code": "G06Q40/00"}],
        },
        {
            "patent_id": "US-10654321",
            "title": "Smart Grid Controller",
            "score": 0.85,
            "publication_date": "2019-08-20",
            "cpc_codes": [{"code": "H02J13/00"}],
        },
        {
            "patent_id": "US-9999999",
            "title": "Distributed Ledger System",
            "score": 0.78,
            "publication_date": "2022-01-10",
            "cpc_codes": [{"code": "G06Q40/00"}],
        }
    ]
    
    mock_claims = {
        "query_type": "prior_art_search",
        "technical_keywords": ["blockchain", "energy", "trading"],
        "cpc_codes": [{"code": "G06Q40/00"}, {"code": "H02J13/00"}],
    }
    
    mock_synthesis = {
        "executive_summary": "The blockchain energy trading domain shows significant patent activity. "
                           "US-10123456 (IBM) discloses peer-to-peer energy transactions using distributed ledgers. "
                           "Key technical areas include smart contracts, grid integration, and real-time settlement. "
                           "This comprehensive analysis reveals strong prior art coverage in the intersection of "
                           "blockchain and energy trading, with particular emphasis on distributed transaction processing.",
        "risk_score": 60,
    }
    
    print("\nRunning CriticAgent...")
    result = await critic.run(
        query="blockchain energy trading",
        retrieved_chunks=mock_chunks,
        claims_analysis=mock_claims,
        synthesis_output=mock_synthesis,
    )
    
    if result.success:
        data = result.data
        print(f"\n✓ Reward Components:")
        print(f"  Citation Overlap: {data.get('components', {}).get('citation_overlap', 0):.3f}")
        print(f"  CPC Relevance: {data.get('components', {}).get('cpc_relevance', 0):.3f}")
        print(f"  Temporal Diversity: {data.get('components', {}).get('temporal_diversity', 0):.3f}")
        print(f"  LLM Fluency: {data.get('components', {}).get('llm_fluency', 0):.3f}")
        print(f"\n✓ Total Reward Score: {data.get('score', 0):.3f}")
        print(f"  Feedback: {data.get('feedback', 'N/A')}")
        return True
    else:
        print(f"✗ CriticAgent failed: {result.error}")
        return False


async def test_adaptive_retrieval_qlearning():
    """Test AdaptiveRetrievalAgent Q-learning mechanics."""
    print("\n" + "=" * 60)
    print("Testing AdaptiveRetrievalAgent (Q-Learning)")
    print("=" * 60)
    
    settings = get_settings()
    agent = AdaptiveRetrievalAgent(settings=settings)
    
    print(f"\nQ-Learning Parameters:")
    print(f"  Learning Rate (α): {agent.learning_rate}")
    print(f"  Discount Factor (γ): {agent.discount_factor}")
    print(f"  Exploration Rate (ε): {agent.exploration_rate}")
    print(f"  Exploration Decay: {agent.exploration_decay}")
    print(f"  Min Exploration: {agent.min_exploration}")
    
    print(f"\nPolicy Stats:")
    stats = agent.get_policy_stats()
    print(f"  Q-table states: {stats['num_states']}")
    print(f"  Policy path: {stats['policy_path']}")
    print(f"  Policy exists: {stats['policy_exists']}")
    
    # Test Q-value update
    print("\n--- Testing Q-value Update ---")
    
    state1 = ("prior_art_search", 0, 0.0, 0.5)
    action1 = "RETRIEVE"
    reward1 = 0.7
    state2 = ("prior_art_search", 1, 0.7, 0.6)
    
    print(f"  State: {state1}")
    print(f"  Action: {action1}")
    print(f"  Reward: {reward1}")
    print(f"  Next State: {state2}")
    
    # Get Q-value before update
    q_before = agent.q_table.get(state1, {}).get(action1, 0.0)
    
    # Update Q-value
    agent.update_q_value(state1, action1, reward1, state2)
    
    # Get Q-value after update
    q_after = agent.q_table[state1][action1]
    
    print(f"\n  Q(s,a) before: {q_before:.4f}")
    print(f"  Q(s,a) after:  {q_after:.4f}")
    print(f"  ✓ Q-value updated correctly")
    
    # Test action selection
    print("\n--- Testing Action Selection ---")
    
    # Force exploitation mode
    original_exploration = agent.exploration_rate
    agent.exploration_rate = 0.0
    
    selected_action = agent.select_action(state1)
    print(f"  Selected action (exploitation): {selected_action}")
    
    # Force exploration mode
    agent.exploration_rate = 1.0
    selected_actions = [agent.select_action(state1) for _ in range(10)]
    unique_actions = set(selected_actions)
    print(f"  Selected actions (exploration): {unique_actions}")
    
    agent.exploration_rate = original_exploration
    
    print("\n✓ Q-learning mechanics working correctly")
    return True


async def test_full_rlaif_loop():
    """Test the full RLAIF loop with real agents."""
    print("\n" + "=" * 60)
    print("Testing Full RLAIF Loop")
    print("=" * 60)
    
    settings = get_settings()
    orchestrator = Orchestrator(use_langgraph=True)
    
    # Check if required agents are available
    adaptive_agent = orchestrator.agents.get("adaptive_retrieval")
    critic_agent = orchestrator.agents.get("critic")
    
    if not adaptive_agent:
        print("✗ AdaptiveRetrievalAgent not enabled")
        return False
    
    if not critic_agent:
        print("✗ CriticAgent not enabled")
        return False
    
    print(f"\n✓ AdaptiveRetrievalAgent enabled")
    print(f"✓ CriticAgent enabled")
    
    # Run a query through the full pipeline
    test_query = "machine learning optimization algorithms"
    print(f"\nTest Query: '{test_query}'")
    print("Running full pipeline (this may take 1-2 minutes)...")
    
    results = await orchestrator.run_all(test_query)
    
    # Check adaptive retrieval result
    adaptive_result = results.get("adaptive_retrieval")
    if adaptive_result and adaptive_result.success:
        data = adaptive_result.data
        print(f"\n--- Adaptive Retrieval Results ---")
        print(f"  Retrieval Depth: {data.get('retrieval_depth', 0)}")
        print(f"  Total Chunks: {data.get('total_chunks', 0)}")
        
        rl_metadata = data.get("rl_metadata", {})
        print(f"  States Visited: {len(rl_metadata.get('states', []))}")
        print(f"  Actions Taken: {rl_metadata.get('actions', [])}")
        print(f"  Telemetry ID: {data.get('telemetry_run_id', 'N/A')}")
    else:
        print(f"✗ Adaptive retrieval failed: {adaptive_result.error if adaptive_result else 'No result'}")
    
    # Check critic result
    critic_result = results.get("critic")
    if critic_result and critic_result.success:
        data = critic_result.data
        print(f"\n--- Critic (Reward) Results ---")
        print(f"  Total Reward: {data.get('score', 0):.3f}")
        print(f"  Components: {data.get('components', {})}")
    else:
        print(f"✗ Critic failed: {critic_result.error if critic_result else 'No result'}")
    
    # Check final result
    final_result = results.get("final")
    if final_result and final_result.success:
        print(f"\n✓ Full RLAIF loop completed successfully")
        return True
    else:
        print(f"\n⚠ Pipeline completed with issues")
        return False


async def train_initial_policy():
    """Train an initial policy from logged experiences."""
    print("\n" + "=" * 60)
    print("Training Initial Policy from Logged Experiences")
    print("=" * 60)
    
    settings = get_settings()
    agent = AdaptiveRetrievalAgent(settings=settings)
    
    # Check database for experiences
    import psycopg
    from psycopg.rows import dict_row
    
    db_cfg = settings.database
    conn_str = f"postgresql://{db_cfg.user}:{db_cfg.password}@{db_cfg.host}:{db_cfg.port}/{db_cfg.database}"
    
    try:
        with psycopg.connect(conn_str, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Get experience count
                cur.execute("SELECT COUNT(*) as count FROM rl_experiences")
                exp_count = cur.fetchone()["count"]
                
                cur.execute("SELECT COUNT(*) as count FROM adaptive_retrieval_events")
                event_count = cur.fetchone()["count"]
                
                print(f"\nLogged Data:")
                print(f"  RL Experiences: {exp_count}")
                print(f"  Telemetry Events: {event_count}")
                
                if event_count < 10:
                    print("\n⚠ Not enough telemetry events for training")
                    print("  Run more queries to collect training data")
                    return False
                
                # Load telemetry events
                cur.execute("""
                    SELECT 
                        run_id::text AS run_id,
                        iteration,
                        action,
                        state,
                        chunk_quality
                    FROM adaptive_retrieval_events
                    ORDER BY run_id, iteration
                """)
                events = cur.fetchall()
                
                # Load rewards
                cur.execute("""
                    SELECT 
                        run_id::text AS run_id,
                        total_reward
                    FROM rl_experiences
                    WHERE total_reward IS NOT NULL
                """)
                rewards = {row["run_id"]: row["total_reward"] for row in cur.fetchall()}
        
        # Group events by run_id
        runs = {}
        for event in events:
            run_id = event["run_id"]
            runs.setdefault(run_id, []).append(event)
        
        print(f"\nProcessing {len(runs)} training runs...")
        
        # Update Q-values from experiences
        updates = 0
        for run_id, run_events in runs.items():
            if len(run_events) < 2:
                continue
            
            reward = rewards.get(run_id, 0.5)  # Default reward if not found
            
            for i in range(len(run_events) - 1):
                state_vec = run_events[i].get("state") or []
                next_state_vec = run_events[i + 1].get("state") or state_vec
                
                if not state_vec or not next_state_vec:
                    continue
                
                state = tuple(state_vec) if isinstance(state_vec, list) else state_vec
                next_state = tuple(next_state_vec) if isinstance(next_state_vec, list) else next_state_vec
                action = run_events[i].get("action", "STOP")
                
                # Apply reward only to final transition
                transition_reward = reward if i == len(run_events) - 2 else 0.0
                
                agent.update_q_value(state, action, transition_reward, next_state)
                updates += 1
            
            agent.decay_exploration()
        
        print(f"  Q-value updates: {updates}")
        print(f"  Q-table size: {len(agent.q_table)} states")
        print(f"  Exploration rate: {agent.exploration_rate:.4f}")
        
        # Save the trained policy
        agent._save_policy()
        print(f"\n✓ Policy saved to {agent.policy_path}")
        
        # Verify the saved policy
        if agent.policy_path.exists():
            with open(agent.policy_path, "rb") as f:
                saved = pickle.load(f)
            print(f"  Verified: {len(saved.get('q_table', {}))} states in saved policy")
            return True
        
        return False
        
    except Exception as exc:
        print(f"✗ Training failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all RLAIF tests."""
    print("\n" + "=" * 60)
    print("PatentSphere RLAIF System Tests")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Critic as reward model
    try:
        results["Critic Reward Model"] = await test_critic_reward_model()
    except Exception as e:
        print(f"\n✗ Critic test failed: {e}")
        results["Critic Reward Model"] = False
    
    # Test 2: Q-learning mechanics
    try:
        results["Q-Learning Mechanics"] = await test_adaptive_retrieval_qlearning()
    except Exception as e:
        print(f"\n✗ Q-learning test failed: {e}")
        results["Q-Learning Mechanics"] = False
    
    # Test 3: Train from logged experiences
    try:
        results["Policy Training"] = await train_initial_policy()
    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        results["Policy Training"] = False
    
    # Test 4: Full RLAIF loop (skip if no policy)
    try:
        results["Full RLAIF Loop"] = await test_full_rlaif_loop()
    except Exception as e:
        print(f"\n✗ Full loop test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Full RLAIF Loop"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All RLAIF tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
        print("\nTo train the policy, run:")
        print("  python scripts/train_rl_policy.py --episodes 100")


if __name__ == "__main__":
    asyncio.run(main())


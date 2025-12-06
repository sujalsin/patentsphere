#!/usr/bin/env python3
"""
RL Training Script for AdaptiveRetrievalAgent.

Trains Q-learning policy using CriticAgent rewards from rl_experiences table.
Optimized with GPU support and parallel processing.
"""

import asyncio
import json
import logging
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psycopg
from psycopg.rows import dict_row

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from app.orchestrator import Orchestrator
from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import PySpark for distributed processing
try:
    from pyspark.sql import SparkSession
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning("PySpark not available, using multiprocessing instead")

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch available, using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    logger.warning("PyTorch not available, using CPU")


def generate_synthetic_queries(count: int = 1000) -> List[str]:
    """
    Generate synthetic queries for training.
    
    Uses patterns from baseline queries plus variations.
    """
    base_queries = [
        "AI machine learning neural networks",
        "autonomous vehicles self-driving cars",
        "renewable energy solar panels batteries",
        "quantum computing algorithms",
        "CRISPR gene editing biotechnology",
        "blockchain cryptocurrency distributed ledger",
        "5G wireless communication networks",
        "robotic surgery medical devices",
        "semiconductor chips processors",
        "biometric authentication security",
    ]
    
    # Query templates
    templates = [
        "patents related to {topic}",
        "latest {topic} innovations",
        "{topic} technology developments",
        "recent {topic} research",
        "{topic} applications and methods",
        "emerging {topic} technologies",
        "{topic} systems and devices",
        "{topic} algorithms and processes",
    ]
    
    synthetic = []
    for _ in range(count):
        base = random.choice(base_queries)
        template = random.choice(templates)
        query = template.format(topic=base)
        synthetic.append(query)
    
    return synthetic


async def load_rewards_from_db(
    settings, limit: int = 1000
) -> List[Dict[str, Any]]:
    """Load rewards from rl_experiences table."""
    pg_cfg = settings.database
    conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
    
    try:
        with psycopg.connect(conn_str, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        id, query_text, query_type, retrieved_patent_ids,
                        retrieved_chunks,
                        total_reward,
                        reward_components,
                        agent_outputs,
                        run_id,
                        created_at
                    FROM rl_experiences
                    WHERE total_reward IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        logger.error("Failed to load rewards from database: %s", exc)
        return []


async def load_telemetry_events(
    settings, run_ids: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Load adaptive retrieval telemetry rows keyed by run_id."""
    if not run_ids:
        return {}

    pg_cfg = settings.database
    conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"

    try:
        with psycopg.connect(conn_str, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        run_id::text AS run_id,
                        iteration,
                        action,
                        state,
                        metadata
                    FROM adaptive_retrieval_events
                    WHERE run_id = ANY(%s)
                    ORDER BY run_id, iteration
                    """,
                    (run_ids,),
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.error("Failed to load telemetry events: %s", exc)
        return {}

    events: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        events.setdefault(row["run_id"], []).append(row)
    return events


async def replay_logged_experiences(
    settings,
    adaptive_agent: AdaptiveRetrievalAgent,
    limit: int = 500,
) -> None:
    """Warm up the policy using logged adaptive runs + critic rewards."""
    experiences = await load_rewards_from_db(settings, limit)
    run_ids = [str(exp["run_id"]) for exp in experiences if exp.get("run_id")]
    telemetry_map = await load_telemetry_events(settings, run_ids)

    if not telemetry_map:
        logger.info("No adaptive telemetry available to replay.")
        return

    updated = 0
    for exp in experiences:
        run_id = exp.get("run_id")
        if not run_id:
            continue
        run_key = str(run_id)
        events = telemetry_map.get(run_key, [])
        if len(events) < 2:
            continue

        reward = exp.get("total_reward") or 0.0

        for idx in range(len(events) - 1):
            state_vec = events[idx].get("state") or []
            next_state_vec = events[idx + 1].get("state") or state_vec
            if not state_vec or not next_state_vec:
                continue
            state = tuple(state_vec)
            next_state = tuple(next_state_vec)
            action = events[idx].get("action", "STOP")
            # Reward applied only to final transition to avoid double-counting
            transition_reward = reward if idx == len(events) - 2 else 0.0
            adaptive_agent.update_q_value(state, action, transition_reward, next_state)

        adaptive_agent.decay_exploration()
        updated += 1

    if updated:
        logger.info("Replayed %d logged experiences into the policy.", updated)
    else:
        logger.info("Logged experiences found but insufficient telemetry for replay.")


async def run_training_episode(
    query: str,
    episode_num: int,
    settings_dict: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run a single training episode.
    
    This function is designed to be called in parallel.
    It creates its own orchestrator instance to avoid conflicts.
    
    Returns:
        Episode data with state, action, reward, next_state
    """
    try:
        # Create fresh orchestrator for this episode (thread-safe)
        settings = get_settings() if settings_dict is None else None
        orchestrator = Orchestrator()
        adaptive_agent = orchestrator.agents.get("adaptive_retrieval")
        
        if not adaptive_agent:
            return {"error": "AdaptiveRetrievalAgent not found", "episode": episode_num}
        
        # Run orchestrator to get full context
        results = await orchestrator.run_all(query)
        
        # Extract data for RL update
        claims_data = results.get("claims_analyzer")
        adaptive_data = results.get("adaptive_retrieval")
        critic_data = results.get("critic")
        
        if not adaptive_data or not adaptive_data.success:
            return {"error": "AdaptiveRetrievalAgent failed", "episode": episode_num}
        
        if not critic_data or not critic_data.success:
            return {"error": "CriticAgent failed", "episode": episode_num}
        
        # Get reward from critic
        reward = critic_data.data.get("score", 0.0)
        
        # Get RL metadata from adaptive agent
        rl_metadata = adaptive_data.data.get("rl_metadata", {})
        states = rl_metadata.get("states", [])
        actions = rl_metadata.get("actions", [])
        
        # Update Q-values for each state-action pair
        query_type = claims_data.data.get("query_type", "other") if claims_data else "other"
        
        episode_data = {
            "episode": episode_num,
            "query": query,
            "query_type": query_type,
            "reward": reward,
            "states": states,
            "actions": actions,
            "retrieval_depth": adaptive_data.data.get("retrieval_depth", 0),
        }
        
        return episode_data
        
    except Exception as exc:
        logger.error("Training episode %d failed: %s", episode_num, exc)
        return {"error": str(exc), "episode": episode_num}


def run_episode_sync(args: Tuple[str, int]) -> Dict[str, Any]:
    """
    Synchronous wrapper for async episode execution.
    Used for multiprocessing.
    """
    query, episode_num = args
    return asyncio.run(run_training_episode(query, episode_num))


async def train_policy_parallel(
    episodes: int = 1000,
    batch_size: int = 32,
    save_frequency: int = 100,
    validation_split: float = 0.2,
    max_workers: int = 4,
    use_pyspark: bool = False,
) -> None:
    """
    Main training function with parallel processing support.
    
    Args:
        episodes: Number of training episodes
        batch_size: Batch size for Q-learning updates (not used in current implementation)
        save_frequency: Save checkpoint every N episodes
        validation_split: Fraction for validation set
        max_workers: Number of parallel workers (for multiprocessing)
        use_pyspark: Use PySpark for distributed processing
    """
    settings = get_settings()
    
    logger.info("Starting RL training (parallel mode)")
    logger.info("Episodes: %d, Batch size: %d, Workers: %d", episodes, batch_size, max_workers)
    logger.info("Device: %s, PySpark: %s", DEVICE, use_pyspark and PYSPARK_AVAILABLE)
    
    # Initialize agents
    orchestrator = Orchestrator()
    adaptive_agent = orchestrator.agents.get("adaptive_retrieval")
    
    if not adaptive_agent:
        logger.error("AdaptiveRetrievalAgent not found. Enable it in config.yaml")
        return

    await replay_logged_experiences(settings, adaptive_agent, limit=1000)
    
    # Generate synthetic queries
    logger.info("Generating %d synthetic queries...", episodes)
    queries = generate_synthetic_queries(episodes)
    
    # Split into train/validation
    split_idx = int(len(queries) * (1 - validation_split))
    train_queries = queries[:split_idx]
    val_queries = queries[split_idx:]
    
    logger.info("Training set: %d queries, Validation set: %d queries", len(train_queries), len(val_queries))
    
    # Training loop with parallel processing
    training_metrics = {
        "episodes": [],
        "rewards": [],
        "exploration_rates": [],
        "q_table_size": [],
    }
    
    start_time = time.time()
    
    if use_pyspark and PYSPARK_AVAILABLE:
        # Use PySpark for distributed processing
        logger.info("Using PySpark for distributed training...")
        spark = SparkSession.builder \
            .appName("RLTraining") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # Create RDD of (query, episode_num) pairs
        query_rdd = spark.sparkContext.parallelize(
            [(q, i) for i, q in enumerate(train_queries, 1)],
            numSlices=max_workers * 4
        )
        
        # Process episodes in parallel
        episode_results = query_rdd.map(run_episode_sync).collect()
        
        spark.stop()
    else:
        # Use multiprocessing for parallel execution
        logger.info("Using multiprocessing with %d workers...", max_workers)
        
        # Prepare episode arguments
        episode_args = [(query, i) for i, query in enumerate(train_queries, 1)]
        
        # Process episodes in batches to avoid memory issues
        episode_results = []
        batch_size_parallel = max_workers * 2  # Process 2x workers at a time
        
        for batch_start in range(0, len(episode_args), batch_size_parallel):
            batch = episode_args[batch_start:batch_start + batch_size_parallel]
            logger.info("Processing batch %d/%d (episodes %d-%d)...",
                       batch_start // batch_size_parallel + 1,
                       (len(episode_args) + batch_size_parallel - 1) // batch_size_parallel,
                       batch_start + 1,
                       min(batch_start + batch_size_parallel, len(episode_args)))
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(run_episode_sync, batch))
                episode_results.extend(batch_results)
    
    # Process results and update Q-values
    logger.info("Processing %d episode results and updating Q-values...", len(episode_results))
    
    for episode_data in episode_results:
        if "error" in episode_data:
            logger.warning("Episode %d failed: %s", episode_data.get("episode", "?"), episode_data.get("error"))
            continue
        
        episode_num = episode_data.get("episode", 0)
        reward = episode_data.get("reward", 0.0)
        states = episode_data.get("states", [])
        actions = episode_data.get("actions", [])
        
        # Update Q-values for each state-action transition
        for i in range(len(states) - 1):
            state = tuple(states[i])
            action = actions[i] if i < len(actions) else "STOP"
            next_state = tuple(states[i + 1]) if i + 1 < len(states) else tuple(states[i])
            
            # Use reward for final transition, 0 for intermediate
            transition_reward = reward if i == len(states) - 2 else 0.0
            
            adaptive_agent.update_q_value(state, action, transition_reward, next_state)
        
        # Decay exploration rate (do this once per episode)
        if episode_num % 10 == 0:
            adaptive_agent.decay_exploration()
        
        # Log metrics
        training_metrics["episodes"].append(episode_num)
        training_metrics["rewards"].append(reward)
        training_metrics["exploration_rates"].append(adaptive_agent.exploration_rate)
        training_metrics["q_table_size"].append(len(adaptive_agent.q_table))
        
        # Progress logging
        if episode_num % 10 == 0:
            recent_rewards = [r for e, r in zip(training_metrics["episodes"], training_metrics["rewards"]) if e >= episode_num - 9]
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            logger.info(
                "Episode %d/%d: avg_reward=%.3f, exploration=%.3f, q_table_size=%d",
                episode_num,
                len(train_queries),
                avg_reward,
                adaptive_agent.exploration_rate,
                len(adaptive_agent.q_table),
            )
        
        # Save checkpoint
        if episode_num % save_frequency == 0:
            adaptive_agent._save_policy()
            logger.info("Saved checkpoint at episode %d", episode_num)
    
    # Final save
    adaptive_agent._save_policy()
    logger.info("Training complete. Final policy saved.")
    
    # Validation
    logger.info("Running validation on %d queries...", len(val_queries))
    val_rewards = []
    val_limit = min(100, len(val_queries))  # Limit validation to 100 queries
    
    # Process validation in parallel too
    val_args = [(q, 0) for q in val_queries[:val_limit]]
    
    if use_pyspark and PYSPARK_AVAILABLE:
        spark = SparkSession.builder \
            .appName("RLValidation") \
            .getOrCreate()
        val_rdd = spark.sparkContext.parallelize(val_args, numSlices=max_workers)
        val_results = val_rdd.map(run_episode_sync).collect()
        spark.stop()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            val_results = list(executor.map(run_episode_sync, val_args))
    
    for result in val_results:
        if "reward" in result:
            val_rewards.append(result["reward"])
    
    if val_rewards:
        avg_val_reward = sum(val_rewards) / len(val_rewards)
        logger.info("Validation average reward: %.3f", avg_val_reward)
    
    # Save training metrics
    metrics_path = Path("models/training_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.1f seconds/episode)", elapsed, elapsed / len(train_queries) if train_queries else 0)
    logger.info("Final Q-table size: %d states", len(adaptive_agent.q_table))
    logger.info("Final exploration rate: %.3f", adaptive_agent.exploration_rate)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL policy for AdaptiveRetrievalAgent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (not used currently)")
    parser.add_argument("--save-frequency", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation set fraction")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--use-pyspark", action="store_true", help="Use PySpark for distributed processing")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available (for embeddings)")
    
    args = parser.parse_args()
    
    # Update device if GPU requested
    global DEVICE
    if args.gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("GPU enabled for embeddings")
    elif args.gpu:
        logger.warning("GPU requested but not available, using CPU")
    
    asyncio.run(
        train_policy_parallel(
            episodes=args.episodes,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency,
            validation_split=args.validation_split,
            max_workers=args.max_workers,
            use_pyspark=args.use_pyspark,
        )
    )


if __name__ == "__main__":
    main()

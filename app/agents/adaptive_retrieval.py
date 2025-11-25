from __future__ import annotations

import asyncio
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid

import psycopg
from psycopg.types.json import Jsonb

from app.agents.base import AgentResult, BaseAgent
from app.agents.citation import CitationMapperAgent

logger = logging.getLogger(__name__)


class AdaptiveRetrievalAgent(BaseAgent):
    """
    Adaptive retrieval agent using Q-learning to optimize retrieval depth.
    
    State: (query_type, retrieval_depth, cumulative_reward, chunk_quality)
    Actions: RETRIEVE, RETRIEVE_MORE, STOP
    """
    name = "adaptive_retrieval"
    
    def __init__(self, settings=None):
        super().__init__(settings)
        self.settings = settings
        
        # Initialize from config
        if settings:
            self.agent_cfg = settings.adaptive_retrieval
            self.actions = ["RETRIEVE", "RETRIEVE_MORE", "STOP"]
            self.exploration_rate = self.agent_cfg.exploration_rate if self.agent_cfg else 0.1
            self.policy_path = Path(self.agent_cfg.policy_path) if self.agent_cfg else Path("models/policy.pkl")
            self.max_depth = 3  # Maximum retrieval iterations
        else:
            self.agent_cfg = None
            self.actions = ["RETRIEVE", "RETRIEVE_MORE", "STOP"]
            self.exploration_rate = 0.1
            self.policy_path = Path("models/policy.pkl")
            self.max_depth = 3
        
        # Q-table: Dict[Tuple[str, int, float, float], Dict[str, float]]
        # State: (query_type, retrieval_depth, cumulative_reward, chunk_quality)
        self.q_table: Dict[Tuple[str, int, float, float], Dict[str, float]] = {}
        
        # Citation mapper for actual retrieval
        self.citation_agent = CitationMapperAgent(settings=settings) if settings else None
        
        # RL parameters from config
        # Settings merges q_learning and training into flat RLConfig
        if settings and hasattr(settings, 'rl'):
            rl_config = settings.rl
            self.learning_rate = getattr(rl_config, 'learning_rate', 0.1)
            self.discount_factor = getattr(rl_config, 'discount_factor', 0.95)
            self.exploration_decay = getattr(rl_config, 'exploration_decay', 0.995)
            self.min_exploration = getattr(rl_config, 'min_exploration', 0.01)
        else:
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.exploration_decay = 0.995
            self.min_exploration = 0.01
        
        # Load policy if exists
        self._load_policy()
    
    def _load_policy(self) -> None:
        """Load Q-table policy from disk."""
        if not self.policy_path.exists():
            logger.info("No policy file found at %s, starting with empty Q-table", self.policy_path)
            return
        
        try:
            with open(self.policy_path, "rb") as f:
                policy_data = pickle.load(f)
                if isinstance(policy_data, dict):
                    # Handle both old format (Tuple[str, int]) and new format
                    if "q_table" in policy_data:
                        self.q_table = policy_data["q_table"]
                        if "exploration_rate" in policy_data:
                            self.exploration_rate = policy_data["exploration_rate"]
                    else:
                        # Legacy format - convert if needed
                        self.q_table = policy_data
                else:
                    self.q_table = policy_data
            logger.info("Loaded policy from %s (%d states)", self.policy_path, len(self.q_table))
        except Exception as exc:
            logger.warning("Failed to load policy from %s: %s", self.policy_path, exc)
            self.q_table = {}
    
    def _save_policy(self) -> None:
        """Save Q-table policy to disk."""
        try:
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_data = {
                "q_table": self.q_table,
                "exploration_rate": self.exploration_rate,
                "version": "1.0",
            }
            with open(self.policy_path, "wb") as f:
                pickle.dump(policy_data, f)
            logger.info("Saved policy to %s (%d states)", self.policy_path, len(self.q_table))
        except Exception as exc:
            logger.error("Failed to save policy to %s: %s", self.policy_path, exc)
    
    def _get_state(
        self,
        query_type: str,
        retrieval_depth: int,
        cumulative_reward: float,
        chunk_quality: float,
    ) -> Tuple[str, int, float, float]:
        """
        Create state tuple from components.
        
        Args:
            query_type: Type of query (from ClaimsAnalyzerAgent)
            retrieval_depth: Current depth (0-indexed)
            cumulative_reward: Sum of rewards so far
            chunk_quality: Average quality score of retrieved chunks
        
        Returns:
            State tuple
        """
        # Normalize query_type
        query_type = query_type.lower() if query_type else "other"
        
        # Normalize cumulative_reward to [0, 1] range
        normalized_reward = max(0.0, min(1.0, cumulative_reward / 10.0))  # Assume max reward ~10
        
        # Normalize chunk_quality to [0, 1] range
        normalized_quality = max(0.0, min(1.0, chunk_quality))
        
        return (query_type, retrieval_depth, normalized_reward, normalized_quality)
    
    def _calculate_chunk_quality(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate average quality score of retrieved chunks.
        
        Uses similarity scores from Qdrant search.
        """
        if not results:
            return 0.0
        
        scores = [r.get("score", 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
    
    def select_action(self, state: Tuple[str, int, float, float]) -> str:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
        
        Returns:
            Selected action
        """
        if random.random() < self.exploration_rate:
            # Exploration: random action
            return random.choice(self.actions)
        
        # Exploitation: best action from Q-table
        state_values = self.q_table.get(state, {})
        if not state_values:
            # Unknown state: default to RETRIEVE
            return self.actions[0]
        
        return max(state_values, key=state_values.get)
    
    def update_q_value(
        self,
        state: Tuple[str, int, float, float],
        action: str,
        reward: float,
        next_state: Tuple[str, int, float, float],
    ) -> None:
        """
        Update Q-value using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
        """
        # Initialize Q-values for states if needed
        self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        self.q_table.setdefault(next_state, {a: 0.0 for a in self.actions})
        
        # Q-learning update: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        best_next = max(self.q_table[next_state].values())
        current = self.q_table[state][action]
        self.q_table[state][action] = current + self.learning_rate * (
            reward + self.discount_factor * best_next - current
        )
    
    def decay_exploration(self) -> None:
        """Decay exploration rate during training."""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
    
    async def run(
        self,
        query: str,
        query_type: str | None = None,
        initial_results: List[Dict[str, Any]] | None = None,
    ) -> AgentResult:
        """
        Run adaptive retrieval with Q-learning policy.
        
        Args:
            query: User query
            query_type: Query type from ClaimsAnalyzerAgent (optional)
            initial_results: Initial retrieval results (optional)
        
        Returns:
            AgentResult with retrieved chunks and RL metadata
        """
        start_time = time.perf_counter()
        
        if not self.citation_agent:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error="CitationMapperAgent not initialized",
            )
        
        try:
            # Initialize state + telemetry tracker
            telemetry_run_id = uuid.uuid4()
            telemetry_events: List[Dict[str, Any]] = []

            query_type = query_type or "other"
            retrieval_depth = 0
            cumulative_reward = 0.0
            all_results: List[Dict[str, Any]] = []
            rl_metadata = {
                "states": [],
                "actions": [],
                "rewards": [],
                "iterations": 0,
            }
            
            # Get initial results if provided, otherwise do first retrieval
            if initial_results:
                current_results = initial_results
            else:
                # First retrieval
                citation_result = await self.citation_agent.run(query)
                if not citation_result.success:
                    return AgentResult(
                        agent=self.name,
                        success=False,
                        data={},
                        error=f"Initial retrieval failed: {citation_result.error}",
                    )
                current_results = citation_result.data.get("results", [])
            
            all_results.extend(current_results)
            chunk_quality = self._calculate_chunk_quality(current_results)
            
            # Iterative retrieval loop
            while retrieval_depth < self.max_depth:
                # Create current state
                state = self._get_state(query_type, retrieval_depth, cumulative_reward, chunk_quality)
                
                # Select action
                action = self.select_action(state)
                rl_metadata["states"].append(state)
                rl_metadata["actions"].append(action)
                
                logger.debug(
                    "AdaptiveRetrievalAgent: depth=%d, state=%s, action=%s",
                    retrieval_depth,
                    state,
                    action,
                )
                
                # Execute action
                if action == "STOP":
                    telemetry_events.append(
                        self._build_telemetry_event(
                            iteration=len(telemetry_events),
                            state=state,
                            action=action,
                            chunk_quality=chunk_quality,
                            results=all_results,
                        )
                    )
                    break
                elif action == "RETRIEVE_MORE":
                    # Perform additional retrieval
                    citation_result = await self.citation_agent.run(query)
                    if citation_result.success:
                        new_results = citation_result.data.get("results", [])
                        # Filter out duplicates
                        existing_ids = {r.get("patent_id") for r in all_results}
                        new_results = [r for r in new_results if r.get("patent_id") not in existing_ids]
                        all_results.extend(new_results)
                        chunk_quality = self._calculate_chunk_quality(all_results)
                    telemetry_events.append(
                        self._build_telemetry_event(
                            iteration=len(telemetry_events),
                            state=state,
                            action=action,
                            chunk_quality=chunk_quality,
                            results=all_results,
                        )
                    )
                    retrieval_depth += 1
                else:  # RETRIEVE (already done on first iteration)
                    telemetry_events.append(
                        self._build_telemetry_event(
                            iteration=len(telemetry_events),
                            state=state,
                            action=action,
                            chunk_quality=chunk_quality,
                            results=all_results,
                        )
                    )
                    retrieval_depth += 1
                    if retrieval_depth >= self.max_depth:
                        break
                    # Continue to next iteration
                    continue
            
            rl_metadata["iterations"] = retrieval_depth + 1
            
            # Calculate final metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            data = {
                "results": all_results,
                "query_type": query_type,
                "retrieval_depth": retrieval_depth + 1,
                "total_chunks": len(all_results),
                "rl_metadata": rl_metadata,
                "latency_ms": latency_ms,
                "telemetry_run_id": str(telemetry_run_id),
            }

            if telemetry_events:
                await self._log_telemetry(
                    run_id=telemetry_run_id,
                    query=query,
                    query_type=query_type,
                    events=telemetry_events,
                )

            return AgentResult(agent=self.name, success=True, data=data)
            
        except Exception as exc:
            logger.error("AdaptiveRetrievalAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about the current policy."""
        return {
            "num_states": len(self.q_table),
            "exploration_rate": self.exploration_rate,
            "policy_path": str(self.policy_path),
            "policy_exists": self.policy_path.exists(),
        }

    def _build_telemetry_event(
        self,
        iteration: int,
        state: Tuple[str, int, float, float],
        action: str,
        chunk_quality: float,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        patent_ids: List[str] = []
        seen = set()
        for entry in results:
            pid = entry.get("patent_id")
            if pid and pid not in seen:
                seen.add(pid)
                patent_ids.append(pid)
            if len(patent_ids) >= 100:
                break

        metadata = {
            "unique_patent_count": len(seen),
            "total_results": len(results),
        }

        return {
            "iteration": iteration,
            "state": list(state),
            "action": action,
            "chunk_quality": chunk_quality,
            "total_chunks": len(results),
            "chunk_ids": patent_ids,
            "metadata": metadata,
        }

    async def _log_telemetry(
        self,
        run_id: uuid.UUID,
        query: str,
        query_type: str,
        events: List[Dict[str, Any]],
    ) -> None:
        """Persist adaptive retrieval telemetry events to Postgres."""
        if not events or not self.settings:
            return

        db_cfg = getattr(self.settings, "database", None)
        if not db_cfg:
            return

        required_str_attrs = ("user", "password", "host", "database")
        for attr in required_str_attrs:
            value = getattr(db_cfg, attr, None)
            if not isinstance(value, str) or not value:
                return
        port = getattr(db_cfg, "port", None)
        if not isinstance(port, int):
            return

        conn_str = (
            f"postgresql://{db_cfg.user}:{db_cfg.password}"
            f"@{db_cfg.host}:{db_cfg.port}/{db_cfg.database}"
        )

        rows = [
            (
                run_id,
                query,
                query_type,
                event.get("iteration"),
                event.get("action"),
                Jsonb(event.get("state")),
                event.get("chunk_ids"),
                event.get("total_chunks"),
                event.get("chunk_quality"),
                self.exploration_rate,
                Jsonb(event.get("metadata", {})),
            )
            for event in events
        ]

        def _write() -> None:
            try:
                with psycopg.connect(conn_str) as conn:
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO adaptive_retrieval_events (
                                run_id,
                                query_text,
                                query_type,
                                iteration,
                                action,
                                state,
                                chunk_ids,
                                total_chunks,
                                chunk_quality,
                                exploration_rate,
                                metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            rows,
                        )
                    conn.commit()
            except Exception as exc:
                logger.debug("Failed to log adaptive telemetry: %s", exc)

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:  # pragma: no cover
            logger.debug("Telemetry logging thread failed: %s", exc)

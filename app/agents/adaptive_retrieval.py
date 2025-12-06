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
from app.services.llm import LLMService, LLMRequest

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
        
        # LLM service for query expansion
        self.llm_service = LLMService(settings) if settings else None
        
        # Internal RLAIF settings
        if settings:
            self.internal_rlaif_threshold = getattr(
                settings.adaptive_retrieval, 'internal_rlaif_threshold', 0.7
            )
            self.max_internal_iterations = getattr(
                settings.adaptive_retrieval, 'max_internal_iterations', 3
            )
        else:
            self.internal_rlaif_threshold = 0.7
            self.max_internal_iterations = 3
        
        # Critic agent for scoring (reuse existing if available)
        self.critic_agent = None
        if settings and hasattr(settings, 'critic') and settings.critic.enabled:
            from app.agents.critic import CriticAgent
            self.critic_agent = CriticAgent(settings=settings, weights=settings.critic.reward_weights)
        
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
        
        if not state_values:
            return self.actions[0]
        return max(state_values.items(), key=lambda x: x[1])[0]
    
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
    
    async def _score_chunks_simple(
        self, 
        chunks: List[Dict[str, Any]], 
        query: str,
        claims_analysis: Dict[str, Any] | None = None
    ) -> float:
        """
        Simple scoring of chunks using CriticAgent (simplified).
        
        Returns a score between 0 and 1.
        """
        if not chunks:
            return 0.0
        
        # If we have a critic agent, use it for scoring
        if self.critic_agent is not None:
            try:
                # Create a mock synthesis output for scoring
                mock_synthesis = {
                    "technical_summary": f"Retrieved {len(chunks)} chunks for query: {query}",
                    "executive_summary": f"Found {len(chunks)} relevant patent chunks",
                }
                
                # Use critic to score (focus on citation overlap and CPC relevance)
                result = await self.critic_agent.run(
                    query=query,
                    retrieved_chunks=chunks,
                    claims_analysis=claims_analysis or {},
                    synthesis_output=mock_synthesis,
                )
                
                if result.success:
                    score = result.data.get("score", 0.0)
                    return float(score)
            except Exception as exc:
                logger.warning("Critic scoring failed in internal RLAIF: %s", exc)
        
        # Fallback: simple heuristic based on chunk quality
        avg_quality = self._calculate_chunk_quality(chunks)
        # Normalize to [0, 1] range (assuming quality is already normalized)
        return float(min(max(avg_quality, 0.0), 1.0))
    
    async def _expand_query(
        self, 
        query: str, 
        claims_analysis: Dict[str, Any] | None = None
    ) -> str:
        """
        Expand query using LLM to add synonyms and technical terms.
        
        Uses qwen2.5:1.5b-instruct for fast query expansion.
        """
        if not self.llm_service:
            # Fallback: simple expansion using claims analysis
            if claims_analysis:
                features = claims_analysis.get("features", [])
                if features:
                    feature_names = [f.get("name", "") if isinstance(f, dict) else str(f) for f in features]
                    expanded = f"{query} {' '.join(feature_names)}"
                    return expanded[:500]  # Limit length
            return query
        
        try:
            # Build expansion prompt
            cpc_codes = []
            if claims_analysis:
                cpc_list = claims_analysis.get("cpc_codes", [])
                if cpc_list:
                    cpc_codes = [
                        c.get("code", "") if isinstance(c, dict) else str(c)
                        for c in cpc_list
                    ]
            
            expansion_prompt = f"""Expand this patent query with synonyms and technical terms for better retrieval.

Original query: {query}

"""
            if cpc_codes:
                expansion_prompt += f"Relevant CPC codes: {', '.join(cpc_codes[:3])}\n\n"
            
            expansion_prompt += """Instructions:
1. Add technical synonyms and related terms
2. Include domain-specific terminology
3. Keep the core meaning intact
4. Return ONLY the expanded query (no explanations, no markdown)

Expanded query:"""
            
            llm_request = LLMRequest(
                agent="query_expansion",
                user_prompt=expansion_prompt,
                system_prompt="You are a patent search expert. Expand queries with technical synonyms.",
                temperature=0.3,
                max_tokens=200,
                response_format="text",
            )
            
            expanded = await self.llm_service.generate(llm_request, retries=1)
            # Clean up response (remove markdown, quotes, etc.)
            expanded = expanded.strip().strip('"').strip("'")
            # Limit length
            return expanded[:500] if expanded else query
            
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
            return query
    
    async def _internal_rlaif_loop(
        self, 
        query: str, 
        claims_analysis: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Internal RLAIF loop: search, score, expand if needed, retry.
        
        Returns top 20 high-relevance chunks.
        """
        max_iterations = self.max_internal_iterations
        current_query = query
        all_chunks = []
        seen_patent_ids = set()
        
        for i in range(max_iterations):
            # Step 1: Search with hybrid search
            try:
                if not self.citation_agent:
                    logger.warning("CitationMapperAgent not initialized in internal RLAIF loop")
                    break
                citation_result = await self.citation_agent.run(
                    current_query, 
                    claims_analysis=claims_analysis
                )
                
                if not citation_result.success:
                    logger.warning("CitationMapper failed in internal RLAIF loop: %s", citation_result.error)
                    break
                
                chunks = citation_result.data.get("results", [])
                
                # Filter duplicates
                new_chunks = []
                for chunk in chunks:
                    patent_id = chunk.get("patent_id")
                    if patent_id and patent_id not in seen_patent_ids:
                        seen_patent_ids.add(patent_id)
                        new_chunks.append(chunk)
                
                all_chunks.extend(new_chunks)
                
                if not new_chunks:
                    logger.debug("No new chunks found in iteration %d", i + 1)
                    break
                
                # Step 2: Score chunks
                score = await self._score_chunks_simple(new_chunks, query, claims_analysis)
                
                logger.debug(
                    "Internal RLAIF iteration %d: score=%.2f, chunks=%d",
                    i + 1, score, len(new_chunks)
                )
                
                # Step 3: Check if threshold met
                if score >= self.internal_rlaif_threshold:
                    logger.info(
                        "Internal RLAIF threshold met (%.2f >= %.2f), stopping",
                        score, self.internal_rlaif_threshold
                    )
                    break
                
                # Step 4: Expand query for next iteration
                if i < max_iterations - 1:  # Don't expand on last iteration
                    current_query = await self._expand_query(current_query, claims_analysis)
                    logger.debug("Query expanded for next iteration: %s", current_query[:100])
                
            except Exception as exc:
                logger.warning("Internal RLAIF loop error: %s", exc)
                break
        
        # Return top 20 chunks (sorted by score)
        all_chunks.sort(key=lambda x: x.get("score", x.get("hybrid_score", 0.0)), reverse=True)
        return all_chunks[:20]
    
    async def run(
        self,
        query: str,
        query_type: str | None = None,
        initial_results: List[Dict[str, Any]] | None = None,
        claims_analysis: Dict[str, Any] | None = None,
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
            
            # Use internal RLAIF loop for retrieval (with hybrid search)
            if initial_results:
                # If initial results provided, use them but still run RLAIF loop
                all_results = initial_results
            else:
                # Run internal RLAIF loop (search, score, expand, retry)
                all_results = await self._internal_rlaif_loop(query, claims_analysis)
            
            if not all_results:
                return AgentResult(
                    agent=self.name,
                    success=False,
                    data={},
                    error="No results retrieved from internal RLAIF loop",
                )
            
            chunk_quality = self._calculate_chunk_quality(all_results)
            
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

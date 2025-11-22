from __future__ import annotations

import random
from typing import Dict, Tuple


class AdaptiveRetrievalAgent:
    def __init__(self, actions=None, exploration=0.1) -> None:
        self.actions = actions or ["RETRIEVE", "RETRIEVE_MORE", "STOP"]
        self.q_table: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.exploration = exploration

    def select_action(self, state: Tuple[str, int]) -> str:
        if random.random() < self.exploration:
            return random.choice(self.actions)
        state_values = self.q_table.get(state, {})
        if not state_values:
            return self.actions[0]
        return max(state_values, key=state_values.get)

    def update(self, state: Tuple[str, int], action: str, reward: float, next_state: Tuple[str, int], alpha=0.1, gamma=0.95):
        self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        self.q_table.setdefault(next_state, {a: 0.0 for a in self.actions})

        best_next = max(self.q_table[next_state].values())
        current = self.q_table[state][action]
        self.q_table[state][action] = current + alpha * (reward + gamma * best_next - current)



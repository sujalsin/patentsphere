from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentResult:
    agent: str
    success: bool
    data: Dict[str, Any]
    error: str | None = None


class BaseAgent:
    name: str = "base"
    
    def __init__(self, settings=None):
        self.settings = settings

    async def run(self, query: str) -> AgentResult:
        raise NotImplementedError


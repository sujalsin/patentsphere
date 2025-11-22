from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from config.settings import Settings, LLMModelConfig

logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when the LLM backend fails after retries."""


@dataclass
class LLMRequest:
    """Represents a structured LLM generation request."""

    agent: str
    user_prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None  # Reserved for future use


class LLMService:
    """Thin async client around the Ollama HTTP API."""

    def __init__(self, settings: Optional[Settings]) -> None:
        self.settings = settings
        if settings and settings.llm:
            self.host = settings.llm.ollama_host.rstrip("/")
            self.timeout = settings.llm.timeout
            self.agent_models = settings.llm.agent_models or {}
            self.model_configs = settings.llm.models or {}
        else:
            self.host = "http://localhost:11434"
            self.timeout = 120
            self.agent_models = {}
            self.model_configs = {}

    def _resolve_model(self, agent_name: str) -> tuple[str, Optional[LLMModelConfig]]:
        """Return (model_key, model_config) for the requested agent."""
        model_key = self.agent_models.get(agent_name, agent_name)
        cfg = self.model_configs.get(model_key)
        if cfg:
            return cfg.name, cfg
        # Fall back to raw key name if config missing
        return model_key, None

    async def generate(self, request: LLMRequest, retries: int = 3) -> str:
        """Call Ollama's /api/generate endpoint with retry + timeout logic."""
        model_name, model_cfg = self._resolve_model(request.agent)
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": request.user_prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature or (model_cfg.temperature if model_cfg else 0.3),
                "top_p": request.top_p or (model_cfg.top_p if model_cfg else 0.9),
            },
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.max_tokens or (model_cfg and model_cfg.max_tokens):
            payload["options"]["num_predict"] = request.max_tokens or model_cfg.max_tokens

        last_exc: Exception | None = None
        url = f"{self.host}/api/generate"

        # Create timeout object with proper settings
        # Ollama can take time, especially on first request (model loading)
        # Add buffer for model loading: first call can take 2-3 minutes
        timeout_obj = httpx.Timeout(
            timeout=self.timeout,
            connect=30.0,  # Connection timeout (increased for slow connections)
            read=self.timeout,  # Read timeout (main timeout for generation)
            write=60.0,  # Write timeout (increased for large prompts)
            pool=30.0,  # Pool timeout (increased for connection pool)
        )

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_obj) as client:
                    response = await client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
                return body.get("response", "").strip()
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "LLM call timeout for agent=%s attempt=%s (timeout=%ss). "
                    "This may be normal on first model load. Retrying...",
                    request.agent,
                    attempt + 1,
                    self.timeout,
                )
                if attempt < retries:
                    # Exponential backoff with longer wait for timeouts
                    wait_time = 2.0 * (attempt + 1)
                    logger.info("Waiting %s seconds before retry...", wait_time)
                    await asyncio.sleep(wait_time)
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                logger.warning(
                    "LLM call failed for agent=%s attempt=%s error=%s",
                    request.agent,
                    attempt + 1,
                    exc,
                )
                if attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise LLMServiceError(f"Ollama call failed after {retries + 1} attempts") from last_exc



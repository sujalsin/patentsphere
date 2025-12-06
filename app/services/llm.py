from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config.settings import Settings, LLMModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Debug mode flag - set via environment or config
DEBUG_LLM = False


class LRUCache:
    """Simple LRU cache for LLM responses."""
    
    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, agent: str, prompt: str) -> str:
        """Create a cache key from agent and prompt."""
        content = f"{agent}:{prompt[:500]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, agent: str, prompt: str) -> Optional[str]:
        """Get cached response if available."""
        key = self._make_key(agent, prompt)
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, agent: str, prompt: str, response: str) -> None:
        """Cache a response."""
        key = self._make_key(agent, prompt)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = response
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
        }


class LLMServiceError(RuntimeError):
    """Base error for LLM service failures."""
    
    def __init__(self, message: str, agent: str = "", model: str = "", request_id: str = ""):
        super().__init__(message)
        self.agent = agent
        self.model = model
        self.request_id = request_id


class LLMTimeoutError(LLMServiceError):
    """Raised when LLM request times out."""
    pass


class LLMConnectionError(LLMServiceError):
    """Raised when cannot connect to LLM backend."""
    pass


class LLMModelNotFoundError(LLMServiceError):
    """Raised when requested model is not available."""
    pass


class LLMValidationError(LLMServiceError):
    """Raised when LLM output fails Pydantic validation."""
    
    def __init__(self, message: str, raw_response: str, validation_errors: list, 
                 agent: str = "", model: str = "", request_id: str = ""):
        super().__init__(message, agent, model, request_id)
        self.raw_response = raw_response
        self.validation_errors = validation_errors


@dataclass
class LLMRequest:
    """Represents a structured LLM generation request."""

    agent: str
    user_prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None  # "json" for JSON mode
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class LLMService:
    """Async client for Ollama HTTP API with Pydantic validation support and caching."""

    # Shared cache across all instances
    _cache: Optional[LRUCache] = None
    
    def __init__(self, settings: Optional[Settings]) -> None:
        self.settings = settings
        if settings and settings.llm:
            self.host = settings.llm.ollama_host.rstrip("/")
            self.timeout = settings.llm.timeout
            self.agent_models = settings.llm.agent_models or {}
            self.model_configs = settings.llm.models or {}
            self.num_ctx = getattr(settings.llm, 'num_ctx', 8192)
        else:
            self.host = "http://localhost:11434"
            self.timeout = 120
            self.agent_models = {}
            self.model_configs = {}
            self.num_ctx = 8192  # Default context window
        
        # Initialize shared cache
        if LLMService._cache is None:
            cache_size = 100
            if settings and hasattr(settings, 'claims_analyzer'):
                cache_size = getattr(settings.claims_analyzer, 'cache_size', 100)
            LLMService._cache = LRUCache(max_size=cache_size)
        
        self.cache = LLMService._cache
        self.use_cache = True  # Enable caching by default

    def _resolve_model(self, agent_name: str) -> tuple[str, Optional[LLMModelConfig]]:
        """Return (model_name, model_config) for the requested agent."""
        model_key = self.agent_models.get(agent_name)
        
        # If not found, check if it's a repair agent (e.g., synthesis_repair -> synthesis)
        if not model_key and "_repair" in agent_name:
            base_agent = agent_name.replace("_repair", "")
            model_key = self.agent_models.get(base_agent)
            # Repairs use qwen for fast JSON fixing
            if not model_key:
                model_key = "qwen"
        
        # Default fallback to qwen (fastest model)
        if not model_key:
            model_key = "qwen"
            logger.warning("No model configured for agent '%s', using fallback: %s", agent_name, model_key)
        
        cfg = self.model_configs.get(model_key)
        if cfg:
            return cfg.name, cfg
        
        # Final fallback if config is missing
        logger.warning("Model config '%s' not found, using qwen2.5:1.5b-instruct", model_key)
        return "qwen2.5:1.5b-instruct", None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text, handling markdown code blocks."""
        text = text.strip()
        
        # Try to find JSON in code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                extracted = text[start:end].strip()
                # Skip language identifier if present
                if extracted and not extracted.startswith("{"):
                    newline_pos = extracted.find("\n")
                    if newline_pos != -1:
                        extracted = extracted[newline_pos + 1:].strip()
                return extracted
        
        # Try to find JSON object directly
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return text[start:end + 1]
        
        return text

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling common LLM output issues with aggressive repair."""
        json_str = self._extract_json(text)
        
        # Try direct parse first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Aggressive JSON repair
        import re
        
        # Step 1: Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Step 2: Fix unquoted keys (but preserve already-quoted ones)
        fixed = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', fixed)
        
        # Step 3: Fix single quotes to double quotes
        fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        
        # Step 4: Fix missing quotes around string values (but not numbers/booleans)
        # This is tricky - only fix if it looks like a string value
        fixed = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', fixed)
        
        # Step 5: Fix missing commas between array/object elements
        fixed = re.sub(r'}\s*{', r'}, {', fixed)
        fixed = re.sub(r']\s*\[', r'], [', fixed)
        fixed = re.sub(r'"\s*"', r'", "', fixed)
        fixed = re.sub(r'}\s*"', r'}, "', fixed)
        fixed = re.sub(r']\s*"', r'], "', fixed)
        
        # Step 6: Fix unclosed brackets (try to balance)
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')
        if open_brackets > close_brackets:
            fixed += ']' * (open_brackets - close_brackets)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            # Last resort: try to extract just the first valid JSON object
            start = fixed.find('{')
            if start != -1:
                # Find matching closing brace
                depth = 0
                for i in range(start, len(fixed)):
                    if fixed[i] == '{':
                        depth += 1
                    elif fixed[i] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(fixed[start:i+1])
                            except json.JSONDecodeError:
                                pass
            raise e
    
    def _fill_missing_fields(self, data: Dict[str, Any], model_class: Type[T]) -> Dict[str, Any]:
        """Fill missing required fields with defaults and fix invalid enum values.
        
        Also handles schema mismatches by extracting fields from nested structures.
        """
        schema = model_class.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        filled = data.copy()
        
        # Handle schema mismatches - extract fields from nested structures
        # Common pattern: model returns {"classification": {...}, "notes": {...}}
        # but we need flat structure
        if isinstance(data, dict):
            # Look for nested structures that might contain our fields
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if nested dict has fields we need
                    for req_field in required:
                        if req_field not in filled and req_field in value:
                            filled[req_field] = value[req_field]
                    
                    # Special handling for common nested patterns
                    if "summary" in value or "description" in value:
                        if "summary" not in filled:
                            filled["summary"] = value.get("summary") or value.get("description", "")
                    
                    if "type" in value or "category" in value:
                        if "query_type" not in filled:
                            filled["query_type"] = value.get("type") or value.get("category", "")
                    
                    if "features" in value or "items" in value:
                        if "features" not in filled:
                            filled["features"] = value.get("features") or value.get("items", [])
                    
                    if "codes" in value or "cpc" in value or "classifications" in value:
                        if "cpc_codes" not in filled:
                            filled["cpc_codes"] = value.get("codes") or value.get("cpc") or value.get("classifications", [])
        
        for field_name in required:
            field_spec = properties.get(field_name, {})
            field_type = field_spec.get("type")
            default = field_spec.get("default")
            
            # Fix enum/literal fields with intelligent mapping
            if "enum" in field_spec:
                enum_values = field_spec["enum"]
                if field_name in filled:
                    value = str(filled[field_name]).lower().strip()
                    
                    # Special handling for query_type
                    if field_name == "query_type":
                        # Map common variations to valid values
                        mapping = {
                            "research": ["research", "general", "patent", "search", "analysis"],
                            "litigation": ["litigation", "lawsuit", "dispute", "infringement", "legal"],
                            "portfolio": ["portfolio", "acquisition", "m&a", "merger", "investment"],
                            "emergence": ["emergence", "emerging", "new", "novel", "innovation"],
                            "other": ["other", "unknown", "misc", "general"]
                        }
                        
                        matched = False
                        for valid_val, variations in mapping.items():
                            if value == valid_val or any(v in value for v in variations):
                                filled[field_name] = valid_val
                                matched = True
                                break
                        
                        if not matched:
                            # Default to research for most queries
                            filled[field_name] = "research"
                    else:
                        # Generic enum matching
                        for valid_val in enum_values:
                            if value == valid_val.lower() or value in valid_val.lower() or valid_val.lower() in value:
                                filled[field_name] = valid_val
                                break
                        else:
                            # Default to first enum value if no match
                            filled[field_name] = enum_values[0]
                else:
                    # Use smart default based on field name
                    if field_name == "query_type":
                        filled[field_name] = "research"  # Most common
                    else:
                        filled[field_name] = enum_values[0] if enum_values else default
            # Handle missing fields
            elif field_name not in filled or filled[field_name] is None:
                # Set defaults based on type
                if default is not None:
                    filled[field_name] = default
                elif field_type == "string":
                    if field_name == "summary":
                        filled[field_name] = "Analysis completed"
                    elif field_name == "query_type":
                        filled[field_name] = "research"
                    else:
                        filled[field_name] = ""
                elif field_type == "number":
                    filled[field_name] = 0.5
                elif field_type == "array":
                    filled[field_name] = []
                elif field_type == "object":
                    filled[field_name] = {}
        
        return filled

    async def generate(self, request: LLMRequest, retries: int = 3, use_cache: bool = True) -> str:
        """Call Ollama's /api/generate endpoint with retry + timeout logic and caching."""
        
        req_id = request.request_id
        start_time = time.perf_counter()
        
        # Check cache first (only for deterministic requests)
        if use_cache and self.use_cache and self.cache and request.temperature and request.temperature < 0.3:
            cached = self.cache.get(request.agent, request.user_prompt)
            if cached:
                logger.info("[%s] CACHE_HIT agent=%s", req_id, request.agent)
                return cached
        
        model_name, model_cfg = self._resolve_model(request.agent)
        
        # Log request details
        prompt_len = len(request.user_prompt)
        logger.info(
            "[%s] LLM_REQUEST agent=%s model=%s prompt_len=%d temp=%.2f",
            req_id, request.agent, model_name, prompt_len,
            request.temperature or (model_cfg.temperature if model_cfg else 0.3)
        )
        
        if DEBUG_LLM:
            logger.debug("[%s] PROMPT:\n%s", req_id, request.user_prompt[:1000])
        
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": request.user_prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature or (model_cfg.temperature if model_cfg else 0.3),
                "top_p": request.top_p or (model_cfg.top_p if model_cfg else 0.9),
                "num_ctx": model_cfg.context_window if model_cfg else self.num_ctx,
            },
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.response_format:
            payload["format"] = request.response_format
        if request.max_tokens or (model_cfg and model_cfg.max_tokens):
            payload["options"]["num_predict"] = request.max_tokens or model_cfg.max_tokens

        last_exc: Exception | None = None
        url = f"{self.host}/api/generate"

        timeout_obj = httpx.Timeout(
            timeout=self.timeout,
            connect=30.0,
            read=self.timeout,
            write=60.0,
            pool=30.0,
        )

        for attempt in range(retries + 1):
            attempt_start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=timeout_obj) as client:
                    response = await client.post(url, json=payload)
                
                # Check for model not found
                if response.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{model_name}' not found. Is it pulled?",
                        agent=request.agent, model=model_name, request_id=req_id
                    )
                
                response.raise_for_status()
                body = response.json()
                result = body.get("response", "").strip()
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                response_len = len(result)
                
                # Log success
                logger.info(
                    "[%s] LLM_SUCCESS agent=%s model=%s elapsed=%.0fms response_len=%d",
                    req_id, request.agent, model_name, elapsed_ms, response_len
                )
                
                if DEBUG_LLM:
                    logger.debug("[%s] RESPONSE:\n%s", req_id, result[:1000])
                
                # Cache the result for low-temperature requests
                if use_cache and self.use_cache and self.cache and request.temperature and request.temperature < 0.3:
                    self.cache.set(request.agent, request.user_prompt, result)
                
                return result
                
            except httpx.TimeoutException as exc:
                last_exc = exc
                attempt_ms = (time.perf_counter() - attempt_start) * 1000
                logger.warning(
                    "[%s] LLM_TIMEOUT agent=%s model=%s attempt=%d/%d elapsed=%.0fms",
                    req_id, request.agent, model_name, attempt + 1, retries + 1, attempt_ms
                )
                if attempt < retries:
                    wait_time = 2.0 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                    
            except httpx.ConnectError as exc:
                last_exc = exc
                logger.error(
                    "[%s] LLM_CONNECTION_ERROR agent=%s host=%s: %s",
                    req_id, request.agent, self.host, exc
                )
                raise LLMConnectionError(
                    f"Cannot connect to Ollama at {self.host}",
                    agent=request.agent, model=model_name, request_id=req_id
                ) from exc
                
            except LLMModelNotFoundError:
                raise
                
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] LLM_ERROR agent=%s model=%s attempt=%d/%d error=%s",
                    req_id, request.agent, model_name, attempt + 1, retries + 1, exc
                )
                if attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "[%s] LLM_FAILED agent=%s model=%s elapsed=%.0fms attempts=%d",
            req_id, request.agent, model_name, elapsed_ms, retries + 1
        )
        raise LLMTimeoutError(
            f"Ollama call failed after {retries + 1} attempts ({elapsed_ms:.0f}ms)",
            agent=request.agent, model=model_name, request_id=req_id
        ) from last_exc

    async def generate_structured(
        self,
        request: LLMRequest,
        output_model: Type[T],
        retries: int = 3,
        validation_retries: int = 2,
    ) -> T:
        """
        Generate LLM response and validate against a Pydantic model.
        
        Args:
            request: The LLM request configuration
            output_model: Pydantic model class to validate response against
            retries: Number of retries for LLM API calls
            validation_retries: Number of retries for validation failures
        
        Returns:
            Validated Pydantic model instance
        
        Raises:
            LLMValidationError: If validation fails after all retries
            LLMServiceError: If LLM API fails
        """
        req_id = request.request_id
        model_name, _ = self._resolve_model(request.agent)
        
        # Ensure JSON format is requested
        request.response_format = "json"
        
        logger.info(
            "[%s] STRUCTURED_REQUEST agent=%s model=%s output=%s",
            req_id, request.agent, model_name, output_model.__name__
        )
        
        last_validation_error: ValidationError | None = None
        raw_response: str = ""
        
        for validation_attempt in range(validation_retries + 1):
            try:
                raw_response = await self.generate(request, retries=retries)
                
                # Parse JSON from response
                json_data = self._parse_json(raw_response)
                
                # Fill missing required fields with defaults
                json_data = self._fill_missing_fields(json_data, output_model)
                
                # Validate against Pydantic model
                result = output_model.model_validate(json_data)
                
                logger.info(
                    "[%s] VALIDATION_SUCCESS agent=%s model=%s",
                    req_id, request.agent, output_model.__name__
                )
                return result
                
            except json.JSONDecodeError as exc:
                logger.warning(
                    "[%s] JSON_PARSE_ERROR agent=%s attempt=%d/%d: %s",
                    req_id, request.agent, validation_attempt + 1, validation_retries + 1, exc
                )
                if DEBUG_LLM:
                    logger.debug("[%s] RAW_RESPONSE:\n%s", req_id, raw_response[:500])
                
                # Create a simple validation error for JSON parse failures
                try:
                    last_validation_error = ValidationError.from_exception_data(
                        title=output_model.__name__,
                        line_errors=[{
                            "type": "json_invalid",
                            "loc": (),
                            "msg": str(exc),
                            "input": raw_response[:500],
                            "ctx": {"error": str(exc)},
                        }],
                    )
                except (TypeError, Exception):
                    # Fallback for different Pydantic versions
                    class FakeValidationError:
                        def errors(self):
                            return [{"loc": (), "msg": str(exc), "type": "json_invalid"}]
                    last_validation_error = FakeValidationError()
                
                if validation_attempt < validation_retries:
                    # Try repair prompt
                    repair_request = await self._create_repair_request(
                        request, raw_response, output_model
                    )
                    if repair_request:
                        logger.info("[%s] ATTEMPTING_REPAIR agent=%s", req_id, request.agent)
                        request = repair_request
                    await asyncio.sleep(0.5)
                    
            except ValidationError as exc:
                logger.warning(
                    "[%s] VALIDATION_ERROR agent=%s attempt=%d/%d errors=%d",
                    req_id, request.agent, validation_attempt + 1, validation_retries + 1, exc.error_count()
                )
                if DEBUG_LLM:
                    for err in exc.errors()[:3]:
                        logger.debug("[%s] - %s: %s", req_id, err["loc"], err["msg"])
                        
                last_validation_error = exc
                
                if validation_attempt < validation_retries:
                    # Try repair prompt
                    repair_request = await self._create_repair_request(
                        request, raw_response, output_model
                    )
                    if repair_request:
                        logger.info("[%s] ATTEMPTING_REPAIR agent=%s", req_id, request.agent)
                        request = repair_request
                    await asyncio.sleep(0.5)
        
        # All attempts failed
        validation_errors = []
        if last_validation_error:
            validation_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                for e in last_validation_error.errors()
            ]
        
        logger.error(
            "[%s] VALIDATION_FAILED agent=%s model=%s attempts=%d",
            req_id, request.agent, output_model.__name__, validation_retries + 1
        )
        
        raise LLMValidationError(
            f"Failed to generate valid {output_model.__name__} after {validation_retries + 1} attempts",
            raw_response=raw_response,
            validation_errors=validation_errors,
            agent=request.agent,
            model=model_name,
            request_id=req_id,
        )

    async def _create_repair_request(
        self,
        original_request: LLMRequest,
        malformed_response: str,
        output_model: Type[T],
    ) -> Optional[LLMRequest]:
        """Create a repair request to fix malformed JSON."""
        schema = output_model.model_json_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Build a simple example based on required fields
        example = {}
        for field in required:
            field_spec = properties.get(field, {})
            field_type = field_spec.get("type")
            if field_type == "string":
                example[field] = "example value"
            elif field_type == "number":
                example[field] = 0.5
            elif field_type == "array":
                example[field] = []
            elif field_type == "object":
                example[field] = {}
        
        repair_prompt = f"""Fix this JSON to match the required schema.

### Required Fields
{', '.join(required)}

### Example Structure
{json.dumps(example, indent=2)}

### Malformed Response
{malformed_response[:1500]}

### Your Task
Extract valid data from the malformed response and create correct JSON with ALL required fields: {', '.join(required)}

Return ONLY valid JSON. No other text."""

        return LLMRequest(
            agent=f"{original_request.agent}_repair",
            user_prompt=repair_prompt,
            system_prompt="You fix JSON. Return ONLY valid JSON matching the schema.",
            temperature=0.0,
            max_tokens=original_request.max_tokens or 1024,
            response_format="json",
        )

    async def generate_with_fallback(
        self,
        request: LLMRequest,
        output_model: Type[T],
        fallback_factory: callable,
        retries: int = 3,
    ) -> tuple[T, bool]:
        """
        Generate structured output with fallback on failure.
        
        Args:
            request: The LLM request
            output_model: Pydantic model for validation
            fallback_factory: Callable that returns a fallback instance
            retries: Number of retries
        
        Returns:
            Tuple of (model_instance, used_fallback)
        """
        try:
            result = await self.generate_structured(
                request, output_model, retries=retries
            )
            return result, False
        except (LLMServiceError, LLMValidationError) as exc:
            logger.warning(
                "Structured generation failed for agent=%s, using fallback: %s",
                request.agent,
                exc,
            )
            return fallback_factory(), True

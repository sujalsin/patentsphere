"""Configuration management using Pydantic Settings."""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # PostgreSQL Configuration
    # Note: Docker maps container port 5432 to host port 5432
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "patentuser"
    postgres_password: str = "patentpass"
    postgres_database: str = "patentsphere"
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "patents"
    
    # Ollama Configuration (privacy-first defaults)
    ollama_base_url: str = "http://localhost:11434"
    ollama_router_model: str = "phi4-mini:latest"
    ollama_extractor_model: str = "qwen2.5:1.5b-instruct"
    ollama_synthesizer_model: str = "gemma3:4b"
    ollama_critic_model: str = "llama3.2:3b"

    # Demo Mode Configuration
    # LLM priority order:
    #   1. GROQ_API_KEY set → Groq cloud model (fast public demo)
    #   2. No Groq key, demo_mode=True → Ollama cpu_fallback_model (free, local)
    #   3. Default (privacy-first) → per-agent Ollama models above
    groq_api_key: Optional[str] = None          # Set this to enable demo mode via Groq
    groq_model: str = "llama-3.1-8b-instant"    # Fast Groq model for public demos
    cpu_fallback_model: str = "llama3.2:3b"     # Upgraded CPU fallback (stronger than 1.5b)
    demo_mode: bool = False                      # Auto-set by get_pipeline_llm() when groq_api_key present

    # ENTERPRISE BRIDGE: tenant isolation support
    # Set default_tenant_id for single-tenant deployments.
    # Future: each request will carry its own tenant_id from auth layer.
    default_tenant_id: Optional[str] = None
    
    # Embedding Configuration
    # Note: Collection has 384-dim vectors, so we use a 384-dim model
    # If you want to use snowflake-arctic-embed-m (768-dim), you'll need to re-ingest
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions to match collection
    sparse_embedding_model: str = "Qdrant/sparse-splade-v1"  # SPLADE for sparse vectors
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_pipeline_llm(role: str = "synthesizer"):
    """
    Return the appropriate LangChain chat model for pipeline steps.

    Priority:
      1. GROQ_API_KEY is set → Groq (fast for public demos)
      2. No Groq, demo_mode=True → Ollama cpu_fallback_model (llama3.2:3b)
      3. Default (privacy-first) → Ollama synthesizer model

    Args:
        role: One of 'synthesizer', 'critic', 'extractor', 'router'.
              Used to select the appropriate default Ollama model.
    """
    if settings.groq_api_key:
        # Demo mode: use Groq cloud (fast, powerful, good for demos)
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0.3,
            )
        except ImportError:
            print("[config] langchain-groq not installed; falling back to Ollama cpu_fallback_model")

    if settings.demo_mode:
        # CPU fallback: stronger than qwen2.5:1.5b, still fully local and free
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.cpu_fallback_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            num_ctx=8192,
            num_predict=2048,
        )

    # Default: use the role-appropriate Ollama model (privacy-first)
    model_map = {
        "router": settings.ollama_router_model,
        "extractor": settings.ollama_extractor_model,
        "synthesizer": settings.ollama_synthesizer_model,
        "critic": settings.ollama_critic_model,
    }
    model = model_map.get(role, settings.ollama_synthesizer_model)
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        num_ctx=8192,
        num_predict=2048,
    )


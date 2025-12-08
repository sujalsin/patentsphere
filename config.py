"""Configuration management using Pydantic Settings."""
from pydantic_settings import BaseSettings
from typing import Optional


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
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_router_model: str = "phi4-mini:latest"
    ollama_extractor_model: str = "qwen2.5:1.5b-instruct"
    ollama_synthesizer_model: str = "gemma3:4b"
    ollama_critic_model: str = "llama3.2:3b"
    
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


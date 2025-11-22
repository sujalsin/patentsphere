"""
Settings management for PatentSphere.
Loads configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "patentsphere"
    user: str = "patentuser"
    password: str = "patentpass"
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False
    
    @property
    def url(self) -> str:
        """Generate SQLAlchemy database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    collection_name: str = "patents"
    vector_size: int = 384
    distance_metric: str = "Cosine"
    on_disk: bool = True


class GCPConfig(BaseModel):
    """Google Cloud Platform configuration."""
    project_id: Optional[str] = None
    credentials_path: Optional[str] = None
    bucket_name: Optional[str] = None
    
    class BigQueryConfig(BaseModel):
        dataset_id: str = "patents-public-data"
        table_id: str = "patents.publications"
        max_results: int = 1000000
    
    bigquery: BigQueryConfig = BigQueryConfig()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"
    normalize: bool = True


class LLMModelConfig(BaseModel):
    """Individual LLM model configuration."""
    name: str
    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 2048
    context_window: int = 8192


class LLMConfig(BaseModel):
    """LLM configuration."""
    ollama_host: str = "http://localhost:11434"
    timeout: int = 120
    models: Dict[str, LLMModelConfig] = {}
    agent_models: Dict[str, str] = {}


class AgentConfig(BaseModel):
    """Generic agent configuration."""
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3


class CitationMapperConfig(AgentConfig):
    """CitationMapperAgent specific configuration."""
    top_k: int = 50
    hybrid_search_alpha: float = 0.7
    cpc_filter_enabled: bool = True


class LitigationScoutConfig(AgentConfig):
    """LitigationScoutAgent specific configuration."""
    risk_threshold: int = 3  # High risk if 3+ cases per patent
    recent_years: int = 2     # Recent = last 2 years


class CriticConfig(AgentConfig):
    """CriticAgent specific configuration."""
    reward_weights: Dict[str, float] = {
        "citation_overlap": 0.4,
        "cpc_relevance": 0.3,
        "temporal_diversity": 0.2,
        "llm_fluency": 0.1
    }


class AdaptiveRetrievalConfig(AgentConfig):
    """AdaptiveRetrievalAgent specific configuration."""
    policy_path: str = "models/policy.pkl"
    exploration_rate: float = 0.1


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""
    execution_mode: str = "parallel"
    max_concurrent_agents: int = 4
    timeout: int = 120
    retry_failed_agents: bool = True
    aggregate_strategy: str = "weighted"
    agent_weights: Dict[str, float] = {}


class RLConfig(BaseModel):
    """Reinforcement Learning configuration."""
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.2
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    episodes: int = 1000
    batch_size: int = 32
    actions: List[str] = ["RETRIEVE", "RETRIEVE_MORE", "STOP"]


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: List[str] = []
    test_set_size: int = 100
    baseline_test_size: int = 20
    human_eval_size: int = 10
    significance_level: float = 0.05
    paired_test: bool = True
    targets: Dict[str, float] = {}


class Settings(BaseModel):
    """Main application settings."""
    
    # Application metadata
    app_name: str = "PatentSphere"
    app_version: str = "1.0.0"
    environment: str = "development"
    # High-level runtime profile, e.g. local_dev, gcp_free_tier, gcp_full
    app_profile: str = "local_dev"
    debug: bool = True
    log_level: str = "INFO"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Component configurations
    database: DatabaseConfig
    qdrant: QdrantConfig
    gcp: GCPConfig
    embeddings: EmbeddingConfig
    llm: LLMConfig
    orchestrator: OrchestratorConfig
    rl: RLConfig
    evaluation: EvaluationConfig
    
    # Agent configurations
    claims_analyzer: AgentConfig = AgentConfig()
    citation_mapper: CitationMapperConfig = CitationMapperConfig()
    litigation_scout: LitigationScoutConfig = LitigationScoutConfig()
    synthesis: AgentConfig = AgentConfig(timeout=60, max_retries=2)
    adaptive_retrieval: AdaptiveRetrievalConfig = AdaptiveRetrievalConfig(enabled=False)
    critic: CriticConfig = CriticConfig(enabled=False)
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    
    class Config:
        arbitrary_types_allowed = True


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Simple environment variable substitution: ${VAR_NAME:default_value}
    import re
    def replace_env_var(match):
        var_expr = match.group(1)
        if ':' in var_expr:
            var_name, default = var_expr.split(':', 1)
        else:
            var_name, default = var_expr, ''
        return os.getenv(var_name, default)
    
    content = re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    return yaml.safe_load(content)


def create_settings_from_yaml(yaml_config: Dict[str, Any]) -> Settings:
    """Create Settings object from parsed YAML configuration."""
    
    # Extract database configs
    postgres_config = yaml_config.get('database', {}).get('postgres', {})
    qdrant_config = yaml_config.get('database', {}).get('qdrant', {})
    
    # Extract GCP config
    gcp_config = yaml_config.get('gcp', {})
    
    # Extract embedding config
    embedding_config = yaml_config.get('embeddings', {})
    
    # Extract LLM config
    llm_config_data = yaml_config.get('llm', {})
    models_data = llm_config_data.get('models', {})
    llm_models = {
        name: LLMModelConfig(**config) 
        for name, config in models_data.items()
    }
    
    # Extract orchestrator config
    orchestrator_config = yaml_config.get('orchestrator', {})
    
    # Extract RL config
    rl_config_data = yaml_config.get('rl', {})
    q_learning = rl_config_data.get('q_learning', {})
    training = rl_config_data.get('training', {})
    rl_config = {**q_learning, **training}
    rl_config['actions'] = rl_config_data.get('actions', [])
    
    # Extract evaluation config
    eval_config = yaml_config.get('evaluation', {})
    
    # Extract agent configs
    agents_config = yaml_config.get('agents', {})
    
    # Build Settings object
    return Settings(
        app_name=yaml_config.get('app', {}).get('name', 'PatentSphere'),
        app_version=yaml_config.get('app', {}).get('version', '1.0.0'),
        environment=yaml_config.get('app', {}).get('environment', 'development'),
        app_profile=yaml_config.get('app', {}).get('profile', 'local_dev'),
        debug=yaml_config.get('app', {}).get('debug', True),
        log_level=yaml_config.get('app', {}).get('log_level', 'INFO'),
        api_host=yaml_config.get('api', {}).get('host', '0.0.0.0'),
        api_port=yaml_config.get('api', {}).get('port', 8000),
        api_workers=yaml_config.get('api', {}).get('workers', 4),
        database=DatabaseConfig(**postgres_config),
        qdrant=QdrantConfig(**qdrant_config),
        gcp=GCPConfig(
            project_id=gcp_config.get('project_id'),
            credentials_path=gcp_config.get('credentials_path'),
            bucket_name=gcp_config.get('storage', {}).get('bucket_name'),
            bigquery=GCPConfig.BigQueryConfig(**gcp_config.get('bigquery', {}))
        ),
        embeddings=EmbeddingConfig(**embedding_config),
        llm=LLMConfig(
            ollama_host=llm_config_data.get('ollama', {}).get('host', 'http://localhost:11434'),
            timeout=llm_config_data.get('ollama', {}).get('timeout', 120),
            models=llm_models,
            agent_models=llm_config_data.get('agent_models', {})
        ),
        orchestrator=OrchestratorConfig(**orchestrator_config),
        rl=RLConfig(**rl_config),
        evaluation=EvaluationConfig(**eval_config),
        claims_analyzer=AgentConfig(**agents_config.get('claims_analyzer', {})),
        citation_mapper=CitationMapperConfig(**agents_config.get('citation_mapper', {})),
        litigation_scout=LitigationScoutConfig(**agents_config.get('litigation_scout', {})),
        synthesis=AgentConfig(**agents_config.get('synthesis', {})),
        adaptive_retrieval=AdaptiveRetrievalConfig(**agents_config.get('adaptive_retrieval', {})),
        critic=CriticConfig(**agents_config.get('critic', {}))
    )


@lru_cache()
def get_settings(config_file: str = "config.yaml") -> Settings:
    """
    Load and cache application settings.
    
    Args:
        config_file: Name of the YAML config file (default: config.yaml)
        
    Returns:
        Settings object with all configuration
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Determine config file path
    config_path = Path(__file__).parent / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and parse YAML configuration
    yaml_config = load_yaml_config(config_path)
    
    # Create Settings object
    settings = create_settings_from_yaml(yaml_config)
    
    # Create necessary directories
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    
    return settings


if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    print(f"✓ Loaded configuration for {settings.app_name} v{settings.app_version}")
    print(f"✓ Environment: {settings.environment}")
    print(f"✓ Database URL: {settings.database.url}")
    print(f"✓ Qdrant: {settings.qdrant.host}:{settings.qdrant.port}")
    print(f"✓ Ollama: {settings.llm.ollama_host}")
    print(f"✓ Base directory: {settings.base_dir}")

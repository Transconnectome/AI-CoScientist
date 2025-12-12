"""Application configuration settings."""

from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "AI-CoScientist"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True

    # Security
    secret_key: str = Field(..., min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    # Database
    database_url: str
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Redis
    redis_url: str
    redis_cache_ttl: int = 3600
    redis_max_connections: int = 10

    # RabbitMQ
    rabbitmq_url: str
    rabbitmq_exchange: str = "ai_coscientist"
    rabbitmq_queue: str = "tasks"

    @property
    def openai_client(self) -> "openai.AsyncClient":
        """Return an async OpenAI client initialized with the API key.
        This property lazily creates the client when first accessed.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai package is required for OpenAI integration") from e
        # Ensure the key is set
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")
        # Configure the client
        client = openai.AsyncClient(api_key=self.openai_api_key)
        return client

    # OpenAI
    openai_api_key: str = ""

    # Anthropic
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-5"  # Claude Sonnet 4.5 (Released Sep 2025)
    anthropic_max_tokens: int = 8000  # Claude 4.5 supports up to 8K output

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"  # Gemini 2.5 Pro (Released Mar 2025)
    gemini_max_tokens: int = 8192
    gemini_temperature: float = 0.7

    # LLM Configuration
    llm_primary_provider: str = "openai"
    llm_fallback_provider: str = "anthropic"
    llm_cache_enabled: bool = True
    llm_cache_ttl: int = 3600
    llm_max_retries: int = 3
    llm_timeout: int = 60

    # Vector Database
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001
    chromadb_collection: str = "scientific_papers"
    embedding_model: str = "allenai/scibert_scivocab_uncased"
    embedding_dimension: int = 384

    # External APIs
    semantic_scholar_api_key: str = ""
    crossref_email: str = ""

    # Celery
    celery_broker_url: str
    celery_result_backend: str
    celery_task_always_eager: bool = False

    # Monitoring
    prometheus_port: int = 9090
    enable_metrics: bool = True

    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100

    # Storage
    upload_dir: str = "./uploads"
    max_upload_size: int = 10485760  # 10MB

    # RL System Configuration
    rl_enabled: bool = True
    rl_ab_testing_enabled: bool = True
    rl_initial_traffic_pct: int = 10
    rl_max_traffic_pct: int = 90
    rl_model_path: str = "./models/rl_agent_selection"
    rl_config_path: str = "./config/rl/rl_system_config.yaml"
    rl_training_enabled: bool = True
    rl_performance_threshold: float = 0.8

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key length."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    @property
    def chromadb_url(self) -> str:
        """Get ChromaDB URL."""
        return f"http://{self.chromadb_host}:{self.chromadb_port}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings

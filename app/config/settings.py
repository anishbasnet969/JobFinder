from typing import List, Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = Field(default="JobFinder", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")

    CORS_ORIGINS: str = Field(default="*", description="Allowed CORS origins")

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://username:password@db:5432/jobfinderdb",
        description="PostgreSQL database URL",
    )

    GOOGLE_API_KEY: Optional[str] = Field(
        default=None, description="Google API key for Gemini services"
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001", description="Gemini embedding model"
    )

    # Redis Configuration
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(
        default=None, description="Redis password (optional)"
    )

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(
        default="amqp://guest:guest@localhost:5672//",
        description="Celery broker URL (RabbitMQ)",
    )
    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None, description="Celery result backend (uses Redis if None)"
    )

    # Performance Settings
    CACHE_RESUME_TTL: int = Field(
        default=86400, description="Resume cache TTL in seconds (24 hours)"
    )
    CACHE_RECOMMENDATIONS_TTL: int = Field(
        default=3600, description="Recommendations cache TTL in seconds (1 hour)"
    )
    CACHE_EMBEDDINGS_TTL: int = Field(
        default=604800, description="Embeddings cache TTL in seconds (7 days)"
    )
    VECTOR_SEARCH_CANDIDATES: int = Field(
        default=100,
        description="Number of candidates to pre-filter in vector search",
    )

    # Worker Settings
    CELERY_TASK_ALWAYS_EAGER: bool = Field(
        default=False, description="Execute tasks synchronously (for testing)"
    )

    # Job ingestion
    JOB_INGEST_ON_STARTUP: bool = Field(
        default=False,
        description="Trigger CSV job ingestion automatically when the API starts",
    )
    JOB_INGEST_CSV_PATH: str = Field(
        default="datasets/Job/job_title_des.csv",
        description="Relative or absolute path to the job CSV",
    )
    JOB_INGEST_BATCH_SIZE: int = Field(
        default=50,
        description="Number of jobs processed concurrently per batch",
    )
    JOB_INGEST_LIMIT: Optional[int] = Field(
        default=500,
        description="Limit the number of jobs ingested from CSV (None for all)",
    )

    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS

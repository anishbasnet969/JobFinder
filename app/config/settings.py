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

    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS

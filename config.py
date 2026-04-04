"""
Configuration — Centralized settings via Pydantic BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── OpenAI ──────────────────────────────────
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    TEMPERATURE: float = Field(0.0, env="TEMPERATURE")

    # ── Agent ───────────────────────────────────
    MAX_AGENT_ITERATIONS: int = Field(8, env="MAX_AGENT_ITERATIONS")
    MAX_MEMORY_MESSAGES: int = Field(10, env="MAX_MEMORY_MESSAGES")

    # ── RAG ─────────────────────────────────────
    DOCS_DIR: str = Field("docs/", env="DOCS_DIR")
    FAISS_INDEX_PATH: str = Field("data/faiss_index", env="FAISS_INDEX_PATH")
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")

    # ── Memory backend ──────────────────────────
    USE_REDIS_MEMORY: bool = Field(False, env="USE_REDIS_MEMORY")
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    REDIS_TTL_SECONDS: int = Field(86400, env="REDIS_TTL_SECONDS")

    # ── External APIs ───────────────────────────
    OPENWEATHERMAP_API_KEY: str = Field("", env="OPENWEATHERMAP_API_KEY")

    # ── App ─────────────────────────────────────
    APP_TITLE: str = "JARVIS Agent"
    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

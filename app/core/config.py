from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Advanced RAG Platform"
    app_env: str = "production"
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    database_url: str

    qdrant_url: str
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    redis_url: str

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "advanced-rag"

    cors_origins: str = "http://localhost:5173"

    rate_limit_per_minute: int = 20
    max_request_size_mb: int = 5

    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    rag_max_iterations: int = 2

    jwt_secret_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application Configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & Services
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_TIMEOUT: int = 60
    QDRANT_BATCH_SIZE: int = 100
    S3_BUCKET: str = "medical-rag-docs-mohamed-2026"
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    # Supabase Database
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SECRET_KEY: str = ""
    SUPABASE_PUBLISHABLE_KEY: str = ""

    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    LLM_MODEL: str = "openai/gpt-oss-20b"

    # Chunking & Document Processing
    MAX_FILE_SIZE_MB: int = 50
    SEMANTIC_CHUNK_THRESHOLD: float = 0.50
    RECURSIVE_CHUNK_SIZE: int = 1500
    RECURSIVE_CHUNK_OVERLAP: int = 300

    # Retrieval, Reranking & Evaluation Thresholds
    RETRIEVAL_TOP_K: int = 10
    RERANKER_TOP_K: int = 5
    DEDUP_OVERLAP_THRESHOLD: float = 0.8
    EVIDENCE_THRESHOLD: float = 0.4
    COLLECTION_NAME: str = "medical_documents"


@lru_cache()
def get_config() -> AppConfig:
    """Returns a cached instance of AppConfig."""
    return AppConfig()

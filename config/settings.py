"""
config/settings.py
Configurações centrais via Pydantic Settings.
Todas as variáveis são lidas do .env ou do ambiente.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Aplicação ─────────────────────────────
    app_name: str = "Sueteres RAG Assistant"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = Field(default="sueteres-dev-key")

    # ── LLM ───────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_primary_model: str = "mistral:7b-instruct-v0.3-q4_K_M"
    llm_fallback_model: str = "qwen2.5:3b-instruct-q4_K_M"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 90
    llm_max_retries: int = 3

    # ── Embeddings ────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    models_cache_dir: Path = Path("./models")

    # ── ChromaDB ──────────────────────────────
    chroma_persist_dir: Path = Path("./chroma_db")
    chroma_collection_name: str = "sueteres_corpus"
    chroma_distance_metric: str = "cosine"

    # ── Document Store ────────────────────────
    sqlite_db_path: Path = Path("./document_store/sueteres.db")

    # ── Retrieval ─────────────────────────────
    retrieval_top_k_initial: int = 20
    retrieval_top_k_final: int = 5
    retrieval_score_threshold: float = 0.35
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    # ── Chunking ──────────────────────────────
    chunk_size_min: int = 512
    chunk_size_target: int = 768
    chunk_size_max: int = 1024
    chunk_overlap: int = 128

    # ── Corpus ────────────────────────────────
    corpus_dir: Path = Path("./corpus")

    # ── Logging ───────────────────────────────
    log_level: str = "INFO"
    log_format: str = "console"
    log_file: Path = Path("./logs/sueteres.log")

    @field_validator("embedding_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"embedding_device must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"

    @property
    def ollama_tags_url(self) -> str:
        return f"{self.ollama_base_url}/api/tags"

    def ensure_dirs(self) -> None:
        """Cria diretórios necessários se não existirem."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configurações — carregado uma única vez."""
    settings = Settings()
    settings.ensure_dirs()
    return settings

"""
rag/embedder.py
Serviço de embeddings usando intfloat/multilingual-e5-large.
Implementa a assimetria query/passage obrigatória do modelo e5.
"""

from __future__ import annotations

import structlog
import math
from functools import lru_cache
from typing import TYPE_CHECKING

from config.settings import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

logger = structlog.get_logger(__name__)

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class EmbedderService:
    """
    Gerador de embeddings com prefixação assimétrica (query: / passage:).

    A assimetria é fundamental para o modelo multilingual-e5-large:
    usar o mesmo prefixo para query e passagem degrada o recall em ~15-30%.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._device = settings.embedding_device
        self._batch_size = settings.embedding_batch_size
        self._model_cache_dir = str(settings.models_cache_dir)
        self._model: "_SentenceTransformer | None" = None

    def _load_model(self) -> "_SentenceTransformer":
        """Lazy loading — só carrega quando necessário."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            logger.info(
                "Loading embedding model",
                extra={"model": self._model_name, "device": self._device},
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._model_cache_dir,
            )
            logger.info(
                "Embedding model loaded",
                extra={"dimension": self.dimension},
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension() or 1024
        return 1024  # Dimensão padrão do multilingual-e5-large

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para passagens do corpus. Prefixo: 'passage: '."""
        if not texts:
            return []

        model = self._load_model()
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        vectors = model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Gera embedding para a query do usuário. Prefixo: 'query: '."""
        model = self._load_model()
        prefixed = QUERY_PREFIX + text
        vector = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Valida norma L2 ≈ 1.0 após normalização."""
        norm = math.sqrt(sum(x * x for x in embedding))
        return 0.9 <= norm <= 1.1


@lru_cache(maxsize=1)
def get_embedder() -> EmbedderService:
    """Singleton do embedder — carregado uma única vez."""
    return EmbedderService()

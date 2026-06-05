"""
rag/reranker.py
Reranker cross-encoder: reordena candidatos por relevância fina.
Modelo padrão: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import structlog
import math
from functools import lru_cache
from typing import TYPE_CHECKING

from config.settings import get_settings
from domain.entities import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as _CrossEncoder

logger = structlog.get_logger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    """
    Cross-encoder para reordenação fina de candidatos recuperados.

    Por que cross-encoder?
    O cross-encoder recebe (query + passagem) concatenados, capturando
    interação token-level entre os termos. Eleva MAP@5 em ~8-12% sobre
    o similaridade coseno do bi-encoder.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.reranker_model
        self._device = settings.reranker_device
        self._top_k_final = settings.retrieval_top_k_final
        self._score_threshold = settings.retrieval_score_threshold
        self._model: "_CrossEncoder | None" = None

    def _load_model(self) -> "_CrossEncoder":
        """Lazy loading do cross-encoder."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            logger.info(
                "Loading reranker model",
                extra={"model": self._model_name},
            )
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=512,
            )
            logger.info("Reranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Reordena candidatos por score do cross-encoder.
        Retorna os top-k com score normalizado [0, 1] via sigmoid.
        """
        if not candidates:
            return []

        k = top_k or self._top_k_final

        try:
            model = self._load_model()
            pairs = [(query, c.text_for_llm[:1024]) for c in candidates]
            raw_scores: list[float] = model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
        except Exception as exc:
            logger.warning(
                "Reranker failed, using original retrieval order",
                extra={"error": str(exc)},
            )
            return candidates[:k]

        min_s = min(raw_scores)
        max_s = max(raw_scores)
        span = max_s - min_s if max_s != min_s else 1.0

        scored = [
            (candidate, (score - min_s) / span)
            for candidate, score in zip(candidates, raw_scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        result: list[RetrievedChunk] = []
        for candidate, score in scored[:k]:
            result.append(RetrievedChunk(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                doc_title=candidate.doc_title,
                section_path=candidate.section_path,
                page_start=candidate.page_start,
                text_for_llm=candidate.text_for_llm,
                text_preview=candidate.text_preview,
                citation_short=candidate.citation_short,
                citation_abnt=candidate.citation_abnt,
                retrieval_score=round(score, 4),
                marker=candidate.marker,
            ))

        logger.debug(
            "Reranking complete",
            extra={
                "input_count": len(candidates),
                "output_count": len(result),
                "top_score": result[0].retrieval_score if result else 0.0,
            },
        )
        return result

    @property
    def score_threshold(self) -> float:
        return self._score_threshold


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Singleton do reranker."""
    return Reranker()

"""
rag/pipeline.py
Orquestrador principal do pipeline RAG — 8 passos sequenciais.
Retorna sempre RAGResponse, nunca lança exceção para o chamador.
"""

from __future__ import annotations

import hashlib
import structlog
import time
import uuid
from datetime import datetime
from typing import Any

from config.prompts import FALLBACK_RESPONSE
from config.settings import get_settings
from domain.entities import CoverageLevel, RAGResponse, RetrievedChunk
from domain.exceptions import (
    HallucinationDetectedError,
    LLMUnavailableError,
    OutOfCorpusError,
)
from rag.citation_formatter import CitationFormatter
from rag.context_builder import ContextBuilder
from rag.embedder import EmbedderService
from rag.guardrails import ConfidenceCalculator, HallucinationChecker, InCorpusChecker
from rag.llm_client import OllamaClient
from rag.query_processor import QueryProcessor
from rag.reranker import Reranker
from rag.retriever import Retriever

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """
    Pipeline RAG completo em 8 passos:

    1. QueryProcessor   — normalização e detecção de intent
    2. EmbedderService  — embedding da query (query: prefix)
    3. Retriever        — busca vetorial ChromaDB + cross-refs
    4. Reranker         — cross-encoder top-K
    5. InCorpusChecker  — guardrail pré-LLM (score threshold)
    6. ContextBuilder   — monta prompt com chunks numerados [T1]..[T5]
    7. OllamaClient     — geração LLM com retry + fallback
    8. HallucinationChecker + CitationFormatter — validação e formatação
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        embedder: EmbedderService,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        llm_client: OllamaClient,
        citation_formatter: CitationFormatter,
    ) -> None:
        self._qp = query_processor
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._ctx = context_builder
        self._llm = llm_client
        self._citation = citation_formatter

        settings = get_settings()
        self._score_threshold = settings.retrieval_score_threshold
        self._partial_threshold = self._score_threshold + 0.20

        self._in_corpus_checker = InCorpusChecker(threshold=self._score_threshold)
        self._hallucination_checker = HallucinationChecker()
        self._confidence_calculator = ConfidenceCalculator()

    def query(
        self,
        question: str,
        extra_filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Ponto de entrada público.
        Retorna sempre RAGResponse — NUNCA lança exceção para o chamador.
        """
        trace_id = self._generate_trace_id(question)
        start_time = time.monotonic()

        logger.info(
            "RAG query started",
            extra={"trace_id": trace_id, "question_preview": question[:80]},
        )

        try:
            return self._execute(question, extra_filters or {}, trace_id, start_time)

        except OutOfCorpusError as exc:
            logger.info(
                "Out-of-corpus query: returning fallback",
                extra={"trace_id": trace_id, "max_score": exc.max_score},
            )
            return self._fallback_response(
                trace_id=trace_id,
                question=question,
                max_score=exc.max_score,
                start_time=start_time,
            )

        except LLMUnavailableError as exc:
            logger.error(
                "LLM unavailable",
                extra={"trace_id": trace_id, "error": str(exc)},
            )
            return self._error_response(
                trace_id=trace_id,
                question=question,
                start_time=start_time,
                reason=str(exc),
            )

        except Exception as exc:
            logger.exception(
                "Unexpected pipeline error",
                extra={"trace_id": trace_id, "error_type": type(exc).__name__},
            )
            return self._error_response(
                trace_id=trace_id,
                question=question,
                start_time=start_time,
                reason=f"Unexpected error: {type(exc).__name__}: {exc}",
            )

    # ── Execução interna ──────────────────────────────────────

    def _execute(
        self,
        question: str,
        extra_filters: dict[str, Any],
        trace_id: str,
        start_time: float,
    ) -> RAGResponse:
        # ── Passo 1: Processar query ──────────────────
        processed = self._qp.process(question, extra_filters)

        # ── Passo 2: Embed query ──────────────────────
        query_vector = self._embedder.embed_query(processed.normalized or question)

        # ── Passo 3: Retrieve candidatos ─────────────
        candidates = self._retriever.retrieve(
            query_embedding=query_vector,
            chroma_filter=processed.chroma_filter,
        )

        # ── Passo 4: Rerank top-K ─────────────────────
        top_chunks = self._reranker.rerank(
            query=processed.normalized or question,
            candidates=candidates,
        )

        # ── Passo 5: Guardrail pré-LLM ───────────────
        # Lança OutOfCorpusError se nenhum chunk supera o threshold
        self._in_corpus_checker.check(top_chunks, question)

        best_score = top_chunks[0].retrieval_score if top_chunks else 0.0
        is_partial = best_score < self._partial_threshold

        # ── Passo 6: Build context ────────────────────
        context = self._ctx.build(
            query=processed.normalized or question,
            chunks=top_chunks,
            is_partial_context=is_partial,
        )

        # ── Passo 7: Gerar resposta LLM ───────────────
        llm_response = self._llm.generate(
            system_prompt=context.system_prompt,
            user_message=context.user_message,
        )

        # ── Passo 8: Validar + Formatar ───────────────
        hallucination_result = self._hallucination_checker.check(
            response_text=llm_response.text,
            chunks=context.chunks_with_markers,
        )

        if not hallucination_result.passed:
            logger.warning(
                "Response flagged by hallucination checker",
                extra={
                    "trace_id": trace_id,
                    "flags": hallucination_result.flags,
                    "numeric_discrepancies": hallucination_result.numeric_discrepancies,
                },
            )

        formatted = self._citation.format(
            raw_response=llm_response.text,
            chunks_with_markers=context.chunks_with_markers,
        )

        confidence = self._confidence_calculator.calculate(
            chunks=context.chunks_with_markers,
            response_text=llm_response.text,
            citation_coverage=formatted.citation_coverage,
            numeric_discrepancies=hallucination_result.numeric_discrepancies,
        )

        latency = max(1, int((time.monotonic() - start_time) * 1000))

        logger.info(
            "RAG query completed",
            extra={
                "trace_id": trace_id,
                "latency_ms": latency,
                "confidence": confidence,
                "model": llm_response.model_used,
                "chunks_cited": len(formatted.chunks_cited),
                "coverage": context.coverage_level.value,
            },
        )

        return RAGResponse(
            trace_id=trace_id,
            query=question,
            answer=formatted.answer_with_citations,
            chunks_used=formatted.chunks_cited,
            hallucination_flags=hallucination_result.flags,
            numeric_discrepancies=hallucination_result.numeric_discrepancies,
            response_confidence=confidence,
            coverage_level=context.coverage_level,
            model_used=llm_response.model_used,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=latency,
            generation_failed=not hallucination_result.passed,
            fallback_triggered=llm_response.fallback_used,
        )

    # ── Respostas de fallback/erro ────────────────────────────

    def _fallback_response(
        self,
        trace_id: str,
        question: str,
        max_score: float,
        start_time: float,
    ) -> RAGResponse:
        answer = FALLBACK_RESPONSE.format(
            max_score=max_score,
            threshold=self._score_threshold,
        )
        latency = max(1, int((time.monotonic() - start_time) * 1000))
        return RAGResponse(
            trace_id=trace_id,
            query=question,
            answer=answer,
            chunks_used=[],
            hallucination_flags=[],
            numeric_discrepancies=[],
            response_confidence=0.0,
            coverage_level=CoverageLevel.NONE,
            model_used="none",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency,
            generation_failed=False,
            fallback_triggered=True,
        )

    def _error_response(
        self,
        trace_id: str,
        question: str,
        start_time: float,
        reason: str,
    ) -> RAGResponse:
        latency = max(1, int((time.monotonic() - start_time) * 1000))
        return RAGResponse(
            trace_id=trace_id,
            query=question,
            answer=(
                "## Resposta técnica\n"
                "Ocorreu um erro interno ao processar sua consulta. "
                "Por favor, tente novamente em alguns instantes.\n\n"
                "## Documentos consultados\n—\n\n"
                "## Trechos utilizados\n—\n\n"
                "## Fontes\n—"
            ),
            chunks_used=[],
            hallucination_flags=[reason],
            numeric_discrepancies=[],
            response_confidence=0.0,
            coverage_level=CoverageLevel.NONE,
            model_used="error",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency,
            generation_failed=True,
            fallback_triggered=True,
        )

    @staticmethod
    def _generate_trace_id(question: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"aqe_{timestamp}_{suffix}"

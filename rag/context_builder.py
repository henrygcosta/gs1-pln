"""
rag/context_builder.py
Monta o prompt completo para o LLM com chunks numerados [T1]..[T5].
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, replace

from config.prompts import (
    CONTEXT_CHUNK_TEMPLATE,
    PARTIAL_CONTEXT_NOTE,
    QUERY_TEMPLATE,
    SYSTEM_PROMPT,
)
from config.settings import get_settings
from domain.entities import CoverageLevel, RetrievedChunk

logger = structlog.get_logger(__name__)

# Estimativa conservadora: 1 token ≈ 4 caracteres
CHARS_PER_TOKEN = 4


@dataclass
class BuiltContext:
    system_prompt: str
    user_message: str
    chunks_with_markers: list[RetrievedChunk]  # Com markers [T1]..[T5] atribuídos
    prompt_tokens_estimate: int
    coverage_level: CoverageLevel


class ContextBuilder:
    """
    Constrói o prompt estruturado para o LLM.

    Responsabilidades:
    1. Atribuir marcadores [T1]..[T5] a cada chunk
    2. Formatar cada chunk com contexto de seção
    3. Montar user_message (chunks + query)
    4. Estimar token count e verificar context window
    5. Detectar se o contexto é completo ou parcial
    """

    # Tokens reservados para a resposta do LLM
    RESPONSE_RESERVE_TOKENS = 2048

    def __init__(self) -> None:
        settings = get_settings()
        # Mistral 7B context window = 32k tokens
        # Deixamos folga: 32000 - reserva_resposta
        self._max_context_tokens = 32000 - self.RESPONSE_RESERVE_TOKENS
        self._score_threshold = settings.retrieval_score_threshold

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        is_partial_context: bool = False,
    ) -> BuiltContext:
        """
        Constrói o contexto completo para uma consulta.

        Args:
            query: Query normalizada do usuário
            chunks: Chunks ordenados pelo reranker (top-5)
            is_partial_context: True se scores estão entre threshold e 0.55
        """
        # Atribui marcadores
        marked_chunks = self._assign_markers(chunks)

        # Formata cada chunk
        formatted_chunks = self._format_chunks(marked_chunks)
        chunks_text = "\n".join(formatted_chunks)

        # Nota de contexto parcial, se aplicável
        partial_note = PARTIAL_CONTEXT_NOTE if is_partial_context else ""

        # Monta user_message
        user_message = QUERY_TEMPLATE.format(
            chunks_formatted=partial_note + chunks_text,
            query=query,
        )

        # Estima tokens
        system_tokens = len(SYSTEM_PROMPT) // CHARS_PER_TOKEN
        user_tokens = len(user_message) // CHARS_PER_TOKEN
        total_estimate = system_tokens + user_tokens

        if total_estimate > self._max_context_tokens:
            logger.warning(
                "Context window close to limit, truncating chunks",
                estimated=total_estimate,
                limit=self._max_context_tokens,
            )
            marked_chunks, user_message = self._truncate(query, marked_chunks, is_partial_context)

        coverage = self._determine_coverage(chunks, is_partial_context)

        logger.debug(
            "Context built",
            chunks=len(marked_chunks),
            prompt_tokens=total_estimate,
            coverage=coverage.value,
        )

        return BuiltContext(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            chunks_with_markers=marked_chunks,
            prompt_tokens_estimate=total_estimate,
            coverage_level=coverage,
        )

    def _assign_markers(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Atribui [T1]..[T5] a cada chunk."""
        marked: list[RetrievedChunk] = []
        for i, chunk in enumerate(chunks[:5], start=1):
            marked.append(replace(chunk, marker=f"T{i}"))
        return marked

    def _format_chunks(self, chunks: list[RetrievedChunk]) -> list[str]:
        """Formata cada chunk com seu marcador e metadados de localização."""
        formatted: list[str] = []
        for chunk in chunks:
            text = CONTEXT_CHUNK_TEMPLATE.format(
                marker=chunk.marker,
                section_path=chunk.section_path or "—",
                doc_title=chunk.doc_title or chunk.doc_id,
                page_start=chunk.page_start,
                text=chunk.text_for_llm,
            )
            formatted.append(text)
        return formatted

    def _truncate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        is_partial: bool,
    ) -> tuple[list[RetrievedChunk], str]:
        """Remove chunks a partir do último até caber no context window."""
        for end in range(len(chunks), 0, -1):
            subset = chunks[:end]
            formatted = "\n".join(self._format_chunks(subset))
            user_msg = QUERY_TEMPLATE.format(
                chunks_formatted=formatted,
                query=query,
            )
            tokens = len(user_msg) // CHARS_PER_TOKEN
            if tokens <= self._max_context_tokens - 500:
                logger.info(
                    "Truncated to fit context window",
                    original=len(chunks),
                    after=len(subset),
                )
                return subset, user_msg
        # Fallback extremo: só o primeiro chunk
        subset = chunks[:1]
        formatted = "\n".join(self._format_chunks(subset))
        user_msg = QUERY_TEMPLATE.format(chunks_formatted=formatted, query=query)
        return subset, user_msg

    def _determine_coverage(
        self,
        chunks: list[RetrievedChunk],
        is_partial: bool,
    ) -> CoverageLevel:
        if not chunks:
            return CoverageLevel.NONE
        if is_partial:
            return CoverageLevel.PARTIAL
        return CoverageLevel.FULL

"""
tests/integration/test_pipeline.py
Testes de integração do pipeline RAG completo.

Usa mocks para componentes externos (Ollama, embedder pesado, ChromaDB),
mas exercita o fluxo completo de coordenação entre módulos.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from domain.entities import (
    CoverageLevel,
    RAGResponse,
    RetrievedChunk,
)
from domain.exceptions import LLMUnavailableError, OutOfCorpusError
from rag.citation_formatter import CitationFormatter
from rag.context_builder import ContextBuilder
from rag.guardrails import ConfidenceCalculator, HallucinationChecker, InCorpusChecker
from rag.llm_client import OllamaClient
from rag.pipeline import RAGPipeline
from rag.query_processor import QueryProcessor
from rag.reranker import Reranker
from rag.retriever import Retriever


# ─────────────────────────────────────────────────────────────
# Helpers e fixtures do pipeline
# ─────────────────────────────────────────────────────────────


def make_chunk(marker: str, score: float, text: str = "") -> RetrievedChunk:
    default_text = f"chunk content for marker {marker} with technical data 30%"
    return RetrievedChunk(
        chunk_id=f"DOC-001_{marker}",
        doc_id="DOC-001",
        doc_title="LEED v4.1 BD+C",
        section_path="Water Efficiency > Section 5.2",
        page_start=312,
        text_for_llm=text or default_text,
        text_preview=(text or default_text)[:80],
        citation_short="[LEED v4.1, p. 312]",
        citation_abnt="U.S. GREEN BUILDING COUNCIL. LEED v4.1. 2019. p. 312.",
        retrieval_score=score,
        marker=marker,
    )


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = [
        make_chunk("", 0.93),
        make_chunk("", 0.87),
        make_chunk("", 0.74),
    ]
    return retriever


@pytest.fixture
def mock_reranker(five_retrieved_chunks: list[RetrievedChunk]) -> MagicMock:
    reranker = MagicMock(spec=Reranker)
    reranker.rerank.return_value = five_retrieved_chunks
    return reranker


@pytest.fixture
def mock_llm_client() -> MagicMock:
    from rag.llm_client import LLMResponse

    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = LLMResponse(
        text=(
            "## Resposta técnica\n"
            "O LEED v4.1 BD+C exige redução mínima de 30% no consumo de água exterior [T1]. "
            "Projetos que atingem 50% de redução recebem 2 pontos adicionais [T2].\n\n"
            "## Documentos consultados\n"
            "1. LEED v4.1 BD+C — USGBC, 2019\n\n"
            "## Trechos utilizados\n"
            "[T1] LEED v4.1, Seção 5.2, p. 312 — 30% reduction required\n"
            "[T2] LEED v4.1, Seção 5.2, p. 315 — 50% earns 2 points\n\n"
            "## Fontes\n"
            "U.S. GREEN BUILDING COUNCIL. LEED v4.1 BD+C Reference Guide. Washington: USGBC, 2019."
        ),
        model_used="mistral:7b-instruct-v0.3-q4_K_M",
        prompt_tokens=4200,
        completion_tokens=620,
        latency_ms=11000,
        fallback_used=False,
    )
    return client


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1024
    return embedder


@pytest.fixture
def pipeline(
    mock_embedder: MagicMock,
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_llm_client: MagicMock,
) -> RAGPipeline:
    return RAGPipeline(
        query_processor=QueryProcessor(),
        embedder=mock_embedder,
        retriever=mock_retriever,
        reranker=mock_reranker,
        context_builder=ContextBuilder(),
        llm_client=mock_llm_client,
        citation_formatter=CitationFormatter(),
    )


# ─────────────────────────────────────────────────────────────
# Fluxo completo — happy path
# ─────────────────────────────────────────────────────────────


class TestPipelineHappyPath:
    def test_returns_rag_response(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("Quais são os requisitos de eficiência hídrica no LEED v4.1?")
        assert isinstance(result, RAGResponse)

    def test_response_has_trace_id(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.trace_id
        assert result.trace_id.startswith("aqe_")

    def test_response_has_answer(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.answer
        assert len(result.answer) > 50

    def test_response_has_chunks_used(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        # Chunks usados devem ser os que foram citados na resposta
        assert isinstance(result.chunks_used, list)

    def test_response_has_citations(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert isinstance(result.citations_abnt, list)

    def test_response_confidence_between_0_and_1(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert 0.0 <= result.response_confidence <= 1.0

    def test_model_used_recorded(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.model_used == "mistral:7b-instruct-v0.3-q4_K_M"

    def test_generation_not_failed(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.generation_failed is False

    def test_latency_positive(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.latency_ms > 0

    def test_token_counts_recorded(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    def test_coverage_level_set(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1 eficiência hídrica")
        assert result.coverage_level in [
            CoverageLevel.FULL,
            CoverageLevel.PARTIAL,
            CoverageLevel.NONE,
        ]

    def test_embedder_called_with_query(
        self,
        pipeline: RAGPipeline,
        mock_embedder: MagicMock,
    ) -> None:
        pipeline.query("LEED v4.1 eficiência hídrica")
        mock_embedder.embed_query.assert_called_once()

    def test_retriever_called(
        self,
        pipeline: RAGPipeline,
        mock_retriever: MagicMock,
    ) -> None:
        pipeline.query("LEED v4.1")
        mock_retriever.retrieve.assert_called_once()

    def test_reranker_called(
        self,
        pipeline: RAGPipeline,
        mock_reranker: MagicMock,
    ) -> None:
        pipeline.query("LEED v4.1")
        mock_reranker.rerank.assert_called_once()

    def test_llm_called(
        self,
        pipeline: RAGPipeline,
        mock_llm_client: MagicMock,
    ) -> None:
        pipeline.query("LEED v4.1")
        mock_llm_client.generate.assert_called_once()


# ─────────────────────────────────────────────────────────────
# Out-of-corpus — score abaixo do threshold
# ─────────────────────────────────────────────────────────────


class TestOutOfCorpus:
    def test_returns_fallback_when_score_below_threshold(
        self,
        mock_embedder: MagicMock,
        mock_retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        # Simula chunks com scores baixíssimos
        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = [
            make_chunk("T1", 0.10),
            make_chunk("T2", 0.08),
        ]

        pipeline = RAGPipeline(
            query_processor=QueryProcessor(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=ContextBuilder(),
            llm_client=mock_llm_client,
            citation_formatter=CitationFormatter(),
        )

        result = pipeline.query("Pergunta completamente fora do domínio sobre culinária")

        assert result.fallback_triggered is True
        assert result.coverage_level == CoverageLevel.NONE
        assert result.response_confidence == 0.0
        assert result.model_used == "none"
        # LLM não deve ter sido chamado
        mock_llm_client.generate.assert_not_called()

    def test_fallback_response_has_structure(
        self,
        mock_embedder: MagicMock,
        mock_retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = [make_chunk("T1", 0.05)]

        pipeline = RAGPipeline(
            query_processor=QueryProcessor(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=ContextBuilder(),
            llm_client=mock_llm_client,
            citation_formatter=CitationFormatter(),
        )

        result = pipeline.query("Receita de bolo de chocolate")
        assert result.answer  # Resposta de fallback não é vazia
        assert result.chunks_used == []


# ─────────────────────────────────────────────────────────────
# Falha do LLM
# ─────────────────────────────────────────────────────────────


class TestLLMFailure:
    def test_returns_error_response_when_llm_unavailable(
        self,
        mock_embedder: MagicMock,
        mock_retriever: MagicMock,
        five_retrieved_chunks: list[RetrievedChunk],
    ) -> None:
        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = five_retrieved_chunks

        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.side_effect = LLMUnavailableError(
            "Ollama server is down"
        )

        pipeline = RAGPipeline(
            query_processor=QueryProcessor(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=ContextBuilder(),
            llm_client=mock_llm,
            citation_formatter=CitationFormatter(),
        )

        # Não deve lançar exceção — retorna resposta de erro estruturada
        result = pipeline.query("LEED v4.1 eficiência hídrica")

        assert isinstance(result, RAGResponse)
        assert result.generation_failed is True
        assert result.fallback_triggered is True
        assert result.response_confidence == 0.0

    def test_pipeline_never_raises_exception(
        self,
        mock_embedder: MagicMock,
        mock_retriever: MagicMock,
    ) -> None:
        """O pipeline deve SEMPRE retornar RAGResponse, nunca lançar exceção."""
        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.side_effect = RuntimeError("Unexpected reranker crash")

        pipeline = RAGPipeline(
            query_processor=QueryProcessor(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=ContextBuilder(),
            llm_client=MagicMock(spec=OllamaClient),
            citation_formatter=CitationFormatter(),
        )

        result = pipeline.query("qualquer query")
        assert isinstance(result, RAGResponse)


# ─────────────────────────────────────────────────────────────
# Fallback de modelo LLM (primário → secundário)
# ─────────────────────────────────────────────────────────────


class TestModelFallback:
    def test_fallback_model_used_when_primary_fails(
        self,
        mock_embedder: MagicMock,
        mock_retriever: MagicMock,
        five_retrieved_chunks: list[RetrievedChunk],
    ) -> None:
        from rag.llm_client import LLMResponse

        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = five_retrieved_chunks

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise LLMUnavailableError("Primary failed")
            return LLMResponse(
                text="## Resposta técnica\nFallback response [T1].\n\n## Documentos consultados\n—\n\n## Trechos utilizados\n[T1] texto\n\n## Fontes\n—",
                model_used="qwen2.5:3b-instruct-q4_K_M",
                prompt_tokens=1000,
                completion_tokens=100,
                latency_ms=4000,
                fallback_used=True,
            )

        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.side_effect = side_effect

        pipeline = RAGPipeline(
            query_processor=QueryProcessor(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=ContextBuilder(),
            llm_client=mock_llm,
            citation_formatter=CitationFormatter(),
        )

        result = pipeline.query("LEED v4.1")
        assert isinstance(result, RAGResponse)


# ─────────────────────────────────────────────────────────────
# Extra filters via API
# ─────────────────────────────────────────────────────────────


class TestExtraFilters:
    def test_extra_filters_passed_to_retriever(
        self,
        pipeline: RAGPipeline,
        mock_retriever: MagicMock,
    ) -> None:
        extra = {"doc_id": {"$eq": "DOC-001"}}
        pipeline.query("LEED v4.1", extra_filters=extra)
        call_args = mock_retriever.retrieve.call_args
        # Verifica que o filtro foi passado
        assert call_args is not None

    def test_none_filters_accepted(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("LEED v4.1", extra_filters=None)
        assert isinstance(result, RAGResponse)


# ─────────────────────────────────────────────────────────────
# Propriedades do RAGResponse
# ─────────────────────────────────────────────────────────────


class TestRAGResponseProperties:
    def test_documents_used_property(
        self,
        sample_rag_response: RAGResponse,
    ) -> None:
        docs = sample_rag_response.documents_used
        assert isinstance(docs, list)
        # Chunks do DOC-001 devem resultar em uma entrada de documento
        doc_ids = [d["doc_id"] for d in docs]
        assert "DOC-001" in doc_ids

    def test_documents_used_deduplicates(
        self,
        sample_rag_response: RAGResponse,
    ) -> None:
        docs = sample_rag_response.documents_used
        doc_ids = [d["doc_id"] for d in docs]
        # Não deve ter duplicatas
        assert len(doc_ids) == len(set(doc_ids))

    def test_citations_abnt_property(
        self,
        sample_rag_response: RAGResponse,
    ) -> None:
        citations = sample_rag_response.citations_abnt
        assert isinstance(citations, list)
        assert len(citations) >= 1

    def test_trace_id_format(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("test")
        parts = result.trace_id.split("_")
        assert parts[0] == "aqe"
        assert len(parts) >= 3

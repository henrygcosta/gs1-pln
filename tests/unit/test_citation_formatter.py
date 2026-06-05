"""
tests/unit/test_citation_formatter.py
Testes unitários do CitationFormatter — extração de marcadores e formatação ABNT.
"""

from __future__ import annotations

import pytest

from domain.entities import RetrievedChunk
from rag.citation_formatter import CitationFormatter, FormattedResponse


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def make_chunk(
    marker: str,
    doc_id: str = "DOC-001",
    doc_title: str = "LEED v4.1 BD+C",
    text: str = "sample text",
    score: float = 0.90,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}_{marker}",
        doc_id=doc_id,
        doc_title=doc_title,
        section_path=f"Section {marker}",
        page_start=100,
        text_for_llm=text,
        text_preview=text[:60],
        citation_short=f"[{doc_title[:20]}, p. 100]",
        citation_abnt=f"{doc_title}. USGBC, 2019. p. 100.",
        retrieval_score=score,
        marker=marker,
    )


@pytest.fixture
def formatter() -> CitationFormatter:
    return CitationFormatter()


@pytest.fixture
def three_chunks() -> list[RetrievedChunk]:
    return [
        make_chunk("T1", "DOC-001", "LEED v4.1 BD+C",
                   "minimum 30% reduction required for certification"),
        make_chunk("T2", "DOC-001", "LEED v4.1 BD+C",
                   "50% reduction earns 2 additional credit points"),
        make_chunk("T3", "DOC-002", "AQUA-HQE Referencial",
                   "requisito hídrico AQUA-HQE: medição individualizada"),
    ]


# ─────────────────────────────────────────────────────────────
# Extração de marcadores
# ─────────────────────────────────────────────────────────────


class TestMarkerExtraction:
    def test_extracts_single_marker(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Redução mínima de 30% exigida [T1]."
        result = formatter.format(raw, three_chunks)
        assert len(result.chunks_cited) == 1
        assert result.chunks_cited[0].marker == "T1"

    def test_extracts_multiple_markers(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Redução de 30% [T1]. Projetos com 50% obtêm 2 pontos [T2]. AQUA-HQE [T3]."
        result = formatter.format(raw, three_chunks)
        markers = [c.marker for c in result.chunks_cited]
        assert "T1" in markers
        assert "T2" in markers
        assert "T3" in markers

    def test_no_markers_returns_empty_cited(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Resposta sem nenhuma citação de trecho."
        result = formatter.format(raw, three_chunks)
        assert result.chunks_cited == []

    def test_duplicate_marker_deduplicated(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Redução de 30% [T1] é exigida. Ver também [T1] para detalhes."
        result = formatter.format(raw, three_chunks)
        t1_chunks = [c for c in result.chunks_cited if c.marker == "T1"]
        assert len(t1_chunks) == 1  # Não deve duplicar

    def test_invalid_marker_ignored(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        # [T6] e [T0] não são marcadores válidos ([T1]..[T5])
        raw = "Texto com [T6] e [T0] inválidos e [T1] válido."
        result = formatter.format(raw, three_chunks)
        markers = [c.marker for c in result.chunks_cited]
        assert "T6" not in markers
        assert "T0" not in markers
        assert "T1" in markers

    def test_all_five_markers_supported(
        self,
        formatter: CitationFormatter,
    ) -> None:
        chunks = [make_chunk(f"T{i}") for i in range(1, 6)]
        raw = "Texto [T1] [T2] [T3] [T4] [T5]."
        result = formatter.format(raw, chunks)
        assert len(result.chunks_cited) == 5


# ─────────────────────────────────────────────────────────────
# Formatação de saída
# ─────────────────────────────────────────────────────────────


class TestFormattedOutput:
    def test_returns_formatted_response_dataclass(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Texto [T1]."
        result = formatter.format(raw, three_chunks)
        assert isinstance(result, FormattedResponse)

    def test_citation_coverage_one_when_all_used(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Texto [T1] [T2] [T3]."
        result = formatter.format(raw, three_chunks)
        assert result.citation_coverage == pytest.approx(1.0)

    def test_citation_coverage_partial_when_some_used(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Texto [T1] apenas."
        result = formatter.format(raw, three_chunks)
        assert 0.0 < result.citation_coverage < 1.0

    def test_citation_coverage_zero_when_none_used(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Texto sem citações."
        result = formatter.format(raw, three_chunks)
        assert result.citation_coverage == 0.0

    def test_section_trechos_contains_marker_labels(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Texto [T1] [T2]."
        result = formatter.format(raw, three_chunks)
        assert "[T1]" in result.section_trechos or "T1" in result.section_trechos
        assert "[T2]" in result.section_trechos or "T2" in result.section_trechos

    def test_section_fontes_contains_abnt_citation(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "Texto [T1]."
        result = formatter.format(raw, three_chunks)
        # A seção de fontes deve conter a citação ABNT
        assert "USGBC" in result.section_fontes or "LEED" in result.section_fontes

    def test_section_documentos_deduplicates_same_doc(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        # T1 e T2 são ambos do DOC-001
        raw = "Texto [T1] e [T2]."
        result = formatter.format(raw, three_chunks)
        # Documento DOC-001 deve aparecer apenas uma vez
        count = result.section_documentos.count("LEED v4.1 BD+C")
        assert count == 1

    def test_answer_preserves_original_text(
        self,
        formatter: CitationFormatter,
        three_chunks: list[RetrievedChunk],
    ) -> None:
        raw = "O LEED exige 30% de redução [T1]. Aprovado."
        result = formatter.format(raw, three_chunks)
        # O texto original deve estar preservado (com ou sem as seções adicionais)
        assert "30% de redução" in result.answer_with_citations
        assert "Aprovado" in result.answer_with_citations

    def test_empty_chunks_with_no_markers(
        self,
        formatter: CitationFormatter,
    ) -> None:
        raw = "Não encontrei informação no corpus."
        result = formatter.format(raw, [])
        assert result.chunks_cited == []
        assert result.citation_coverage == 0.0
        assert isinstance(result, FormattedResponse)

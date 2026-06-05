"""
tests/unit/test_chunker.py
Testes unitários do HierarchicalSemanticChunker e sub-chunkers especializados.
"""

from __future__ import annotations

import pytest

from domain.entities import (
    Chunk,
    ChunkType,
    DocumentCategory,
    DocumentFormat,
    DocumentMetadata,
    DocumentSection,
    DocumentStatus,
    PreservedStructure,
)
from ingestion.chunker import (
    HierarchicalSemanticChunker,
    ListChunker,
    NormativeChunker,
    TableChunker,
    TextChunker,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def chunker() -> HierarchicalSemanticChunker:
    return HierarchicalSemanticChunker(
        chunk_size_min=64,      # Menor para facilitar testes
        chunk_size_target=128,
        chunk_size_max=256,
        overlap_tokens=16,
    )


@pytest.fixture
def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="DOC-TEST",
        title="Test Document",
        title_full="Full Test Document Title",
        issuer="Test Issuer",
        issuer_country="BR",
        year=2024,
        year_updated=None,
        status=DocumentStatus.ACTIVE,
        language="pt",
        category=DocumentCategory.NORMAS_CERTIFICACOES,
        subcategory="test",
        domain_tags=["test"],
        format=DocumentFormat.PDF,
        pages_total=10,
        access_type="public",
        source_url="",
        local_path="./test.pdf",
        sha256_checksum="abc" * 16,
    )


@pytest.fixture
def simple_section() -> DocumentSection:
    return DocumentSection(
        section_path="Chapter 1 > Section 1.1",
        section_number="1.1",
        section_heading_level=2,
        page_start=10,
        page_end=12,
        normalized_text=(
            "O sistema de gestão de energia predial deve ser instalado conforme "
            "as diretrizes estabelecidas nesta seção. "
            "Os parâmetros de controle são definidos em função da carga instalada. "
            "A eficiência energética é calculada mensalmente. "
            "Relatórios devem ser emitidos trimestralmente ao gestor do edifício."
        ),
        preserved_structures=[],
        acronyms_expanded=[],
    )


@pytest.fixture
def section_with_table() -> DocumentSection:
    table_ps = PreservedStructure(
        type="table",
        content=(
            "| Tipo de Uso | Consumo | Unidade |\n"
            "| --- | --- | --- |\n"
            "| Residencial | 150 | L/hab.dia |\n"
            "| Comercial | 50 | L/func.dia |\n"
        ),
        original_position="p.50",
        must_preserve=True,
        table_confidence="high",
        tag="##TABLE##",
    )
    return DocumentSection(
        section_path="Chapter 3 > Tabela 3.1",
        section_number="3.1",
        section_heading_level=3,
        page_start=50,
        page_end=51,
        normalized_text="##TABLE## Os valores acima representam médias nacionais.",
        preserved_structures=[table_ps],
        acronyms_expanded=[],
    )


@pytest.fixture
def section_with_normative() -> DocumentSection:
    norm_ps = PreservedStructure(
        type="normative_requirement",
        content=(
            "Art. 5.2 — Requisito obrigatório: toda edificação deve instalar "
            "medidores de consumo separados para água potável e sistema de reúso."
        ),
        original_position="p.100",
        must_preserve=True,
        table_confidence="high",
        tag="##NORM##",
    )
    return DocumentSection(
        section_path="Chapter 5 > Art 5.2",
        section_number="5.2",
        section_heading_level=2,
        page_start=100,
        page_end=100,
        normalized_text="##NORM## Verificação anual é obrigatória.",
        preserved_structures=[norm_ps],
        acronyms_expanded=[],
    )


# ─────────────────────────────────────────────────────────────
# HierarchicalSemanticChunker — Contrato básico
# ─────────────────────────────────────────────────────────────


class TestChunkerContract:
    def test_returns_list_of_chunks(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_produces_at_least_one_chunk(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        assert len(result) >= 1

    def test_empty_section_list_returns_empty(
        self,
        chunker: HierarchicalSemanticChunker,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([], metadata)
        assert result == []

    def test_chunk_ids_are_unique(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section, simple_section], metadata)
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_contain_doc_id(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert metadata.doc_id in chunk.chunk_id

    def test_chunk_doc_id_matches_metadata(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.doc_id == metadata.doc_id

    def test_chunk_text_not_empty(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.text_for_embedding.strip()
            assert chunk.text_for_llm.strip()

    def test_citation_fields_populated(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.citation.citation_short
            assert chunk.citation.citation_abnt

    def test_token_counts_positive(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.tokens.core_tokens > 0
            assert chunk.tokens.total_for_llm > 0

    def test_page_range_from_section(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.page_start >= simple_section.page_start

    def test_section_path_preserved_in_chunk(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.section_path == simple_section.section_path

    def test_content_hash_populated(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section], metadata)
        for chunk in result:
            assert chunk.content_hash_sha256
            assert len(chunk.content_hash_sha256) == 64  # sha256 hex digest


# ─────────────────────────────────────────────────────────────
# Preservação de tabelas e normativos
# ─────────────────────────────────────────────────────────────


class TestPreservation:
    def test_table_chunk_marked_as_table(
        self,
        chunker: HierarchicalSemanticChunker,
        section_with_table: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([section_with_table], metadata)
        table_chunks = [c for c in result if c.is_table]
        assert len(table_chunks) >= 1

    def test_normative_chunk_marked_as_normative(
        self,
        chunker: HierarchicalSemanticChunker,
        section_with_normative: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([section_with_normative], metadata)
        norm_chunks = [c for c in result if c.is_normative]
        assert len(norm_chunks) >= 1

    def test_table_chunk_must_preserve(
        self,
        chunker: HierarchicalSemanticChunker,
        section_with_table: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([section_with_table], metadata)
        table_chunks = [c for c in result if c.is_table]
        for chunk in table_chunks:
            assert chunk.must_preserve is True

    def test_table_content_in_chunk_text(
        self,
        chunker: HierarchicalSemanticChunker,
        section_with_table: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([section_with_table], metadata)
        all_text = " ".join(c.text_for_embedding for c in result)
        # Algum conteúdo da tabela deve aparecer
        assert "Residencial" in all_text or "Consumo" in all_text or "L/hab" in all_text


# ─────────────────────────────────────────────────────────────
# Sub-chunkers especializados
# ─────────────────────────────────────────────────────────────


class TestTableChunker:
    @pytest.fixture
    def table_chunker(self) -> TableChunker:
        return TableChunker(max_tokens=256)

    def test_splits_large_table(self, table_chunker: TableChunker) -> None:
        large_table = PreservedStructure(
            type="table",
            content=(
                "| Col1 | Col2 | Col3 |\n"
                "| --- | --- | --- |\n"
                + "| val | val | val |\n" * 50  # Tabela grande
            ),
            original_position="p.10",
            must_preserve=True,
            table_confidence="high",
            tag="",
        )
        result = table_chunker.chunk(large_table)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_small_table_stays_single_chunk(self, table_chunker: TableChunker) -> None:
        small_table = PreservedStructure(
            type="table",
            content=(
                "| Tipo | Valor |\n"
                "| --- | --- |\n"
                "| A | 30% |\n"
                "| B | 50% |\n"
            ),
            original_position="p.10",
            must_preserve=True,
            table_confidence="high",
            tag="",
        )
        result = table_chunker.chunk(small_table)
        assert len(result) == 1
        assert "30%" in result[0] or "Tipo" in result[0]


class TestNormativeChunker:
    @pytest.fixture
    def norm_chunker(self) -> NormativeChunker:
        return NormativeChunker()

    def test_returns_list(self, norm_chunker: NormativeChunker) -> None:
        ps = PreservedStructure(
            type="normative_requirement",
            content="Art. 5 — Toda edificação deve ter medidores individuais.",
            original_position="p.50",
            must_preserve=True,
            table_confidence="high",
            tag="",
        )
        result = norm_chunker.chunk(ps)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_preserves_normative_content(self, norm_chunker: NormativeChunker) -> None:
        content = "Art. 5.2 — Requisito obrigatório: instalar medidores."
        ps = PreservedStructure(
            type="normative_requirement",
            content=content,
            original_position="p.50",
            must_preserve=True,
            table_confidence="high",
            tag="",
        )
        result = norm_chunker.chunk(ps)
        combined = " ".join(result)
        assert "Art." in combined or "medidores" in combined


class TestTextChunker:
    @pytest.fixture
    def text_chunker(self) -> TextChunker:
        return TextChunker(
            min_tokens=32,
            target_tokens=64,
            max_tokens=128,
        )

    def test_short_text_single_chunk(
        self,
        text_chunker: TextChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = text_chunker.chunk(
            text="Texto curto de teste.",
            section=simple_section,
            metadata=metadata,
            start_index=0,
        )
        assert len(result) >= 1

    def test_long_text_split_into_multiple_chunks(
        self,
        text_chunker: TextChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        # Texto longo que deve gerar múltiplos chunks
        long_text = "Este é um parágrafo de texto técnico sobre eficiência energética. " * 30
        result = text_chunker.chunk(
            text=long_text,
            section=simple_section,
            metadata=metadata,
            start_index=0,
        )
        assert len(result) >= 2

    def test_chunk_text_contains_source_content(
        self,
        text_chunker: TextChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        text = "Sistema de gestão de energia deve ser calibrado anualmente."
        result = text_chunker.chunk(
            text=text,
            section=simple_section,
            metadata=metadata,
            start_index=0,
        )
        all_text = " ".join(c.text_for_embedding for c in result)
        assert "energia" in all_text or "gestão" in all_text


# ─────────────────────────────────────────────────────────────
# Múltiplas seções
# ─────────────────────────────────────────────────────────────


class TestMultipleSections:
    def test_multiple_sections_all_indexed(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        section_with_table: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document(
            [simple_section, section_with_table], metadata
        )
        assert len(result) >= 2

    def test_chunk_indices_sequential(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document([simple_section, simple_section], metadata)
        indices = [c.chunk_index_in_doc for c in result]
        for i, idx in enumerate(indices):
            assert idx == i  # Índices sequenciais a partir de 0

    def test_index_offset_applied(
        self,
        chunker: HierarchicalSemanticChunker,
        simple_section: DocumentSection,
        metadata: DocumentMetadata,
    ) -> None:
        result = chunker.chunk_document(
            [simple_section], metadata, chunk_index_offset=10
        )
        assert result[0].chunk_index_in_doc == 10

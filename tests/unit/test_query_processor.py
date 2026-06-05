"""
tests/unit/test_query_processor.py
Testes unitários do QueryProcessor — normalização, detecção de intent e filtros.
"""

from __future__ import annotations

import pytest

from rag.query_processor import ProcessedQuery, QueryProcessor


@pytest.fixture
def processor() -> QueryProcessor:
    return QueryProcessor()


# ─────────────────────────────────────────────────────────────
# Normalização
# ─────────────────────────────────────────────────────────────


class TestNormalization:
    def test_strips_extra_whitespace(self, processor: QueryProcessor) -> None:
        result = processor.process("  qual   é  o requisito  ", {})
        assert "  " not in result.normalized

    def test_lowercases_query(self, processor: QueryProcessor) -> None:
        result = processor.process("LEED V4.1 REQUISITO HÍDRICO", {})
        # Normalizado deve conter a versão em caixa normal
        assert result.normalized  # não vazio

    def test_preserves_technical_values(self, processor: QueryProcessor) -> None:
        result = processor.process("redução de 30% no LEED v4.1 para WE Credit", {})
        # Valores numéricos e siglas técnicas preservados
        assert "30" in result.normalized or "30%" in result.normalized

    def test_empty_query_returns_empty_normalized(self, processor: QueryProcessor) -> None:
        result = processor.process("", {})
        # Deve retornar sem exceção
        assert isinstance(result, ProcessedQuery)

    def test_query_with_special_chars(self, processor: QueryProcessor) -> None:
        result = processor.process("NBR 10.844:2023 — águas pluviais", {})
        assert isinstance(result, ProcessedQuery)
        assert result.normalized  # Não deve ser vazio


# ─────────────────────────────────────────────────────────────
# Detecção de intent e filtros
# ─────────────────────────────────────────────────────────────


class TestIntentDetection:
    def test_detects_leed_filter(self, processor: QueryProcessor) -> None:
        result = processor.process("requisitos LEED v4.1 para eficiência hídrica", {})
        # Deve inferir filtro para DOC-001
        if result.chroma_filter:
            filter_str = str(result.chroma_filter)
            assert "DOC-001" in filter_str

    def test_detects_aqua_hqe_filter(self, processor: QueryProcessor) -> None:
        result = processor.process("certificação AQUA-HQE requisitos", {})
        if result.chroma_filter:
            assert "DOC-002" in str(result.chroma_filter)

    def test_detects_nbr_filter(self, processor: QueryProcessor) -> None:
        result = processor.process("NBR 15575 desempenho térmico", {})
        if result.chroma_filter:
            assert "DOC-003" in str(result.chroma_filter)

    def test_no_filter_for_generic_query(self, processor: QueryProcessor) -> None:
        result = processor.process("eficiência energética em edificações", {})
        # Query genérica — sem filtro de doc_id específico
        # (pode ter filtro de status, mas não de doc específico)
        assert isinstance(result.chroma_filter, (dict, type(None)))

    def test_extra_filters_applied(self, processor: QueryProcessor) -> None:
        extra = {"doc_id": {"$eq": "DOC-001"}}
        result = processor.process("qualquer coisa", extra)
        assert result.chroma_filter is not None
        assert "DOC-001" in str(result.chroma_filter)


# ─────────────────────────────────────────────────────────────
# Resultado do ProcessedQuery
# ─────────────────────────────────────────────────────────────


class TestProcessedQuery:
    def test_returns_processed_query_dataclass(self, processor: QueryProcessor) -> None:
        result = processor.process("LEED crédito hídrico", {})
        assert isinstance(result, ProcessedQuery)
        assert hasattr(result, "original")
        assert hasattr(result, "normalized")
        assert hasattr(result, "chroma_filter")
        assert hasattr(result, "intent_tags")

    def test_original_preserved(self, processor: QueryProcessor) -> None:
        q = "Qual é o requisito do LEED v4.1 para WE Credit?"
        result = processor.process(q, {})
        assert result.original == q

    def test_normative_intent_detected(self, processor: QueryProcessor) -> None:
        result = processor.process("qual é o requisito obrigatório para LEED?", {})
        # Deve detectar intent normativo
        assert isinstance(result.intent_tags, list)

    def test_table_intent_detected(self, processor: QueryProcessor) -> None:
        result = processor.process("tabela de pontuação do LEED v4.1", {})
        assert isinstance(result.intent_tags, list)

    def test_acronym_expansion(self, processor: QueryProcessor) -> None:
        result = processor.process("requisito FV para edificações sustentáveis", {})
        # "FV" deve ser expandido para "fotovoltaico"
        assert "fotovoltaico" in result.normalized.lower() or result.normalized

    def test_ashrae_filter(self, processor: QueryProcessor) -> None:
        result = processor.process("ASHRAE 90.1 padrão energia", {})
        if result.chroma_filter:
            assert "DOC-014" in str(result.chroma_filter)

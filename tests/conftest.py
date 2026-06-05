"""
tests/conftest.py
Fixtures compartilhadas entre todos os testes.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# ── Garante que o diretório raiz está no PYTHONPATH ───────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configura variáveis de ambiente para testes ───────────────
# Feito ANTES de importar qualquer módulo que leia settings
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("API_KEY", "test-key-12345")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("LLM_PRIMARY_MODEL", "mistral:7b-instruct-v0.3-q4_K_M")
os.environ.setdefault("LLM_FALLBACK_MODEL", "qwen2.5:3b-instruct-q4_K_M")


from domain.entities import (
    Chunk,
    ChunkCitation,
    ChunkTokens,
    ChunkType,
    CoverageLevel,
    DocumentCategory,
    DocumentFormat,
    DocumentMetadata,
    DocumentSection,
    DocumentStatus,
    PreservedStructure,
    RAGResponse,
    RetrievedChunk,
)


# ─────────────────────────────────────────────────────────────
# Fixtures de entidades de domínio
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_document_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="DOC-001",
        title="LEED v4.1 BD+C",
        title_full="LEED v4.1 Building Design and Construction Reference Guide",
        issuer="USGBC",
        issuer_country="USA",
        year=2019,
        year_updated=2021,
        status=DocumentStatus.ACTIVE,
        language="en",
        category=DocumentCategory.NORMAS_CERTIFICACOES,
        subcategory="certificacao_internacional",
        domain_tags=["LEED", "certificação", "sustentabilidade"],
        format=DocumentFormat.PDF,
        pages_total=542,
        access_type="public",
        source_url="https://www.usgbc.org",
        local_path="./corpus/leed_v4.1.pdf",
        sha256_checksum="abc123def456" * 4,
    )


@pytest.fixture
def sample_document_section() -> DocumentSection:
    return DocumentSection(
        section_path="Water Efficiency > WE Credit > Outdoor Water Use Reduction",
        section_number="5.2",
        section_heading_level=2,
        page_start=312,
        page_end=318,
        normalized_text=(
            "Buildings pursuing this credit must demonstrate a minimum 30% reduction "
            "in outdoor water use compared to a calculated baseline. The baseline "
            "calculation must use standard landscaping for the climate zone as defined "
            "in the LEED v4.1 reference values table. Projects achieving 50% or greater "
            "reduction earn additional points up to 2 extra credits. Native and adapted "
            "plant species with low water demand combined with high-efficiency irrigation "
            "systems drip or micro-spray are the recommended approach. Metering of the "
            "irrigation system is required as a prerequisite under WE Prerequisite 1."
        ),
        preserved_structures=[],
        acronyms_expanded=["WE: Water Efficiency"],
    )


@pytest.fixture
def sample_chunk(sample_document_metadata: DocumentMetadata) -> Chunk:
    meta = sample_document_metadata
    return Chunk(
        chunk_id="DOC-001_s5.2_p312_c003",
        chunk_index_in_doc=3,
        doc_id="DOC-001",
        chunk_type=ChunkType.NORMATIVE,
        is_table=False,
        is_normative=True,
        is_prerequisite=False,
        is_oversized=False,
        section_path="Water Efficiency > WE Credit > Outdoor Water Use Reduction",
        section_number="5.2",
        page_start=312,
        page_end=314,
        tokens=ChunkTokens(core_tokens=180, overlap_tokens_prev=32, context_header_tokens=24),
        text_for_embedding=(
            "Buildings pursuing this credit must demonstrate a minimum 30% reduction "
            "in outdoor water use compared to a calculated baseline."
        ),
        text_for_llm=(
            "[DOC-001 · LEED v4.1 BD+C · Seção 5.2 · p. 312]\n"
            "Buildings pursuing this credit must demonstrate a minimum 30% reduction "
            "in outdoor water use compared to a calculated baseline. The baseline "
            "calculation must use standard landscaping for the climate zone."
        ),
        must_preserve=True,
        cross_references=["DOC-001_s5.1_p308_c001"],
        citation=ChunkCitation(
            citation_short="[LEED v4.1, Seção 5.2, p. 312]",
            citation_full="LEED v4.1 BD+C Reference Guide, Seção 5.2, p. 312-314",
            citation_abnt=(
                "U.S. GREEN BUILDING COUNCIL. LEED v4.1 BD+C Reference Guide. "
                "Washington: USGBC, 2019. Seção Water Efficiency, p. 312-318."
            ),
        ),
        doc_title=meta.title,
        doc_category=meta.category.value,
        doc_subcategory=meta.subcategory,
        doc_language=meta.language,
        doc_year=meta.year,
        doc_issuer=meta.issuer,
        doc_status=meta.status.value,
        content_hash_sha256="deadbeef" * 8,
    )


@pytest.fixture
def sample_retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="DOC-001_s5.2_p312_c003",
        doc_id="DOC-001",
        doc_title="LEED v4.1 BD+C",
        section_path="Water Efficiency > WE Credit > Outdoor Water Use Reduction",
        page_start=312,
        text_for_llm=(
            "[DOC-001 · LEED v4.1 BD+C · Seção 5.2 · p. 312]\n"
            "Buildings pursuing this credit must demonstrate a minimum 30% reduction "
            "in outdoor water use. Projects achieving 50% earn additional 2 points."
        ),
        text_preview="Buildings pursuing this credit must demonstrate a minimum 30% reduction...",
        citation_short="[LEED v4.1, Seção 5.2, p. 312]",
        citation_abnt=(
            "U.S. GREEN BUILDING COUNCIL. LEED v4.1 BD+C Reference Guide. "
            "Washington: USGBC, 2019. p. 312-318."
        ),
        retrieval_score=0.93,
        marker="T1",
    )


@pytest.fixture
def five_retrieved_chunks() -> list[RetrievedChunk]:
    """Cinco chunks com scores decrescentes para simular resultado do reranker."""
    base = [
        (0.93, "T1", "DOC-001_c001", "DOC-001", "LEED v4.1 BD+C",
         "30% reduction in outdoor water use compared to baseline. [T1]"),
        (0.87, "T2", "DOC-001_c002", "DOC-001", "LEED v4.1 BD+C",
         "50% reduction earns 2 additional points under WE Credit. [T2]"),
        (0.74, "T3", "DOC-002_c001", "DOC-002", "AQUA-HQE Referencial",
         "Requisito hídrico exige medição individualizada por setor. [T3]"),
        (0.65, "T4", "DOC-003_c001", "DOC-003", "ABNT NBR 15575",
         "Desempenho mínimo térmico conforme NBR 15575 partes 1 e 4. [T4]"),
        (0.58, "T5", "DOC-008_c001", "DOC-008", "CBCS — Água em Edificações",
         "Reúso de água cinza pode reduzir consumo em até 40% conforme CBCS. [T5]"),
    ]
    return [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=did,
            doc_title=title,
            section_path="Section Path",
            page_start=100,
            text_for_llm=text,
            text_preview=text[:80],
            citation_short=f"[{title[:20]}, p. 100]",
            citation_abnt=f"{title}. Issuer, 2022. p. 100.",
            retrieval_score=score,
            marker=marker,
        )
        for score, marker, cid, did, title, text in base
    ]


@pytest.fixture
def sample_rag_response(five_retrieved_chunks: list[RetrievedChunk]) -> RAGResponse:
    from datetime import datetime
    return RAGResponse(
        trace_id="aqe_20250915_143522_a7f3",
        query="Quais são os requisitos de eficiência hídrica no LEED v4.1?",
        answer=(
            "## Resposta técnica\n"
            "O LEED v4.1 exige redução mínima de 30% no consumo hídrico [T1]. "
            "Projetos com 50% de redução obtêm 2 pontos adicionais [T2].\n\n"
            "## Documentos consultados\n"
            "- LEED v4.1 BD+C\n\n"
            "## Trechos utilizados\n"
            "[T1] LEED v4.1, p. 312\n"
            "[T2] LEED v4.1, p. 315\n\n"
            "## Fontes\n"
            "U.S. GREEN BUILDING COUNCIL. LEED v4.1. 2019."
        ),
        chunks_used=five_retrieved_chunks[:2],
        hallucination_flags=[],
        numeric_discrepancies=[],
        response_confidence=0.91,
        coverage_level=CoverageLevel.FULL,
        model_used="mistral:7b-instruct-v0.3-q4_K_M",
        prompt_tokens=4218,
        completion_tokens=612,
        latency_ms=11340,
        generation_failed=False,
        fallback_triggered=False,
        created_at=datetime(2025, 9, 15, 14, 35, 22),
    )


# ─────────────────────────────────────────────────────────────
# Fixtures de armazenamento temporário
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_sqlite_db(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = temp_dir / "test.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    # Força re-leitura das settings
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    yield db_path
    settings_module.get_settings.cache_clear()


@pytest.fixture
def temp_chroma_dir(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    chroma_dir = temp_dir / "chroma"
    chroma_dir.mkdir()
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    yield chroma_dir
    settings_module.get_settings.cache_clear()


# ─────────────────────────────────────────────────────────────
# Fixtures de mocks
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Embedder que retorna vetor fixo de 1024 dimensões."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1024
    embedder.embed_passages.return_value = [[0.1] * 1024]
    return embedder


@pytest.fixture
def mock_ollama_available() -> Generator[None, None, None]:
    """Simula Ollama disponível sem chamada de rede real."""
    with patch("rag.llm_client.OllamaClient.check_availability", return_value=True):
        yield


@pytest.fixture
def mock_ollama_response() -> Generator[MagicMock, None, None]:
    """Simula resposta do Ollama com estrutura completa."""
    response_text = (
        "## Resposta técnica\n"
        "O LEED v4.1 exige redução mínima de 30% no consumo de água exterior [T1]. "
        "Para obtenção de pontuação adicional de 2 pontos, a redução deve atingir 50% [T2].\n\n"
        "## Documentos consultados\n"
        "1. LEED v4.1 BD+C — USGBC, 2019\n\n"
        "## Trechos utilizados\n"
        "[T1] LEED v4.1, Seção 5.2, p. 312\n"
        "[T2] LEED v4.1, Seção 5.2, p. 315\n\n"
        "## Fontes\n"
        "U.S. GREEN BUILDING COUNCIL. LEED v4.1 BD+C Reference Guide. Washington: USGBC, 2019."
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": response_text,
        "prompt_eval_count": 4200,
        "eval_count": 610,
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = mock_resp
        yield mock_resp


# ─────────────────────────────────────────────────────────────
# Fixtures de texto para chunker
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def normative_text() -> str:
    """Texto normativo representativo para testes de chunking."""
    return (
        "5.2 REQUISITOS DE EFICIÊNCIA HÍDRICA\n\n"
        "5.2.1 PRÉ-REQUISITO OBRIGATÓRIO\n"
        "Toda edificação deve instalar medidores de consumo de água separados para o sistema "
        "de irrigação. A medição individualizada é requisito para todas as categorias de "
        "certificação. O sistema deve registrar consumo diário com precisão de ±2%.\n\n"
        "5.2.2 CRÉDITO WE — OUTDOOR WATER USE REDUCTION\n"
        "Intenção: reduzir o consumo de água potável utilizada para irrigação.\n"
        "Pontuação: 1 a 3 pontos, conforme percentual de redução demonstrado.\n\n"
        "Requisito: demonstrar redução mínima de 30% no consumo de água em áreas externas "
        "em comparação com a linha de base calculada para a zona climática do projeto. "
        "A linha de base deve ser calculada usando o Landscape Irrigation Calculator "
        "disponibilizado pelo USGBC. Espécies nativas e adaptadas ao clima local são "
        "prioritariamente recomendadas.\n\n"
        "Tabela 5.2.1 — Pontuação por Nível de Redução\n"
        "| Redução | Pontos |\n"
        "| ------- | ------ |\n"
        "| 30%     | 1      |\n"
        "| 40%     | 2      |\n"
        "| 50%+    | 3      |\n\n"
        "Para edificações com área irrigada superior a 500 m², o dimensionamento deve "
        "incluir relatório de cálculo assinado por profissional habilitado. A verificação "
        "será realizada mediante apresentação de contas de água dos últimos 12 meses após "
        "a ocupação do edifício."
    )


@pytest.fixture
def table_text() -> str:
    """Texto com tabela Markdown para teste do TableChunker."""
    return (
        "Tabela 3.4 — Parâmetros de Consumo por Tipo de Uso\n\n"
        "| Uso | Consumo Médio | Unidade |\n"
        "| --- | ------------- | ------- |\n"
        "| Residencial unifamiliar | 150 | L/hab.dia |\n"
        "| Residencial multifamiliar | 120 | L/hab.dia |\n"
        "| Comercial escritório | 50 | L/func.dia |\n"
        "| Hospital | 700 | L/leito.dia |\n"
        "| Escola | 20 | L/aluno.dia |\n"
    )

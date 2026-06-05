"""
domain/entities.py
Entidades de domínio — objetos imutáveis que fluem pelo pipeline.
Não possuem dependências externas: apenas stdlib e dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────
# Enumerações
# ─────────────────────────────────────────────────────────────

class DocumentFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class DocumentCategory(str, Enum):
    NORMAS_CERTIFICACOES = "normas_certificacoes"
    RELATORIOS_TECNICOS = "relatorios_tecnicos"
    TECNOLOGIAS_HABILITADORAS = "tecnologias_habilitadoras"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    NORMATIVE = "normative"


class CoverageLevel(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class IngestionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


# ─────────────────────────────────────────────────────────────
# Objetos de página bruta
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundingBox:
    text: str
    y_top: float
    y_bottom: float


@dataclass
class RawPage:
    page_number: int
    raw_text: str
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    tables_detected: int = 0
    extraction_method: str = "unknown"
    ocr_applied: bool = False
    ocr_confidence: float | None = None


@dataclass
class RawDocument:
    doc_id: str
    source_path: str
    detected_format: DocumentFormat
    detected_encoding: str
    detected_language: str
    is_scanned: bool
    pages: list[RawPage]
    total_pages: int
    extraction_warnings: list[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Objetos após limpeza
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreservedStructure:
    type: str                  # "table" | "normative_requirement" | "formula" | "list"
    content: str               # Texto/markdown do elemento preservado
    original_position: str
    must_preserve: bool = True
    table_confidence: str = "high"  # "high" | "low"
    tag: str = ""


@dataclass
class CleanedPage:
    page_number: int
    cleaned_text: str
    header_removed: str = ""
    footer_removed: str = ""
    is_duplicate: bool = False
    cleaning_skipped: bool = False
    preserved_structures: list[PreservedStructure] = field(default_factory=list)
    cleaning_warnings: list[str] = field(default_factory=list)


@dataclass
class CleanedDocument:
    doc_id: str
    pages: list[CleanedPage]
    pages_removed_duplicate: int = 0
    pages_removed_blank: int = 0
    cleaning_warnings: list[str] = field(default_factory=list)
    cleaned_at: datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Objeto processado (entrada do chunker)
# ─────────────────────────────────────────────────────────────

@dataclass
class DocumentSection:
    section_path: str
    section_number: str
    section_heading_level: int
    page_start: int
    page_end: int
    normalized_text: str
    preserved_structures: list[PreservedStructure] = field(default_factory=list)
    acronyms_expanded: list[str] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    title_full: str
    issuer: str
    issuer_country: str
    year: int
    year_updated: int | None
    status: DocumentStatus
    language: str
    category: DocumentCategory
    subcategory: str
    domain_tags: list[str]
    format: DocumentFormat
    pages_total: int
    access_type: str
    source_url: str
    local_path: str
    sha256_checksum: str


@dataclass
class ProcessedDocument:
    doc_id: str
    processing_version: str
    quality_score: float
    quality_flags: list[str]
    total_chars: int
    total_tokens_estimated: int
    sections: list[DocumentSection]
    document_metadata: DocumentMetadata
    processing_warnings: list[str] = field(default_factory=list)
    ready_for_chunking: bool = True
    processed_at: datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Chunk — unidade atômica indexada
# ─────────────────────────────────────────────────────────────

@dataclass
class ChunkCitation:
    citation_short: str
    citation_full: str
    citation_abnt: str


@dataclass
class ChunkTokens:
    core_tokens: int
    overlap_tokens_prev: int
    context_header_tokens: int

    @property
    def total_for_llm(self) -> int:
        return self.core_tokens + self.overlap_tokens_prev + self.context_header_tokens


@dataclass
class Chunk:
    # Identity
    chunk_id: str
    chunk_index_in_doc: int
    doc_id: str

    # Type
    chunk_type: ChunkType
    is_table: bool
    is_normative: bool
    is_prerequisite: bool
    is_oversized: bool

    # Location
    section_path: str
    section_number: str
    page_start: int
    page_end: int

    # Tokens
    tokens: ChunkTokens

    # Content (two views of the same chunk)
    text_for_embedding: str   # Core content only — what gets vectorized
    text_for_llm: str         # Context header + overlap + core

    # Preservation
    must_preserve: bool
    cross_references: list[str]

    # Citation
    citation: ChunkCitation

    # Document context (denormalized for retrieval)
    doc_title: str
    doc_category: str
    doc_subcategory: str
    doc_language: str
    doc_year: int
    doc_issuer: str
    doc_status: str

    # Observability
    content_hash_sha256: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    retrieval_count: int = 0
    require_human_review: bool = False

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Serialize to ChromaDB-compatible flat dict (no lists, no nested objects)."""
        return {
            "doc_id": self.doc_id,
            "category": self.doc_category,
            "subcategory": self.doc_subcategory,
            "doc_language": self.doc_language,
            "doc_year": self.doc_year,
            "doc_issuer": self.doc_issuer,
            "doc_status": self.doc_status,
            "chunk_type": self.chunk_type.value,
            "is_table": int(self.is_table),
            "is_normative": int(self.is_normative),
            "is_prerequisite": int(self.is_prerequisite),
            "is_oversized": int(self.is_oversized),
            "must_preserve": int(self.must_preserve),
            "require_human_review": int(self.require_human_review),
            "page_start": self.page_start,
            "section_number": self.section_number,
            "core_tokens": self.tokens.core_tokens,
            "citation_short": self.citation.citation_short,
            "citation_abnt": self.citation.citation_abnt,
            "section_path": self.section_path,
            "doc_title": self.doc_title,
            "cross_references_json": ",".join(self.cross_references),
            "content_hash": self.content_hash_sha256,
        }


# ─────────────────────────────────────────────────────────────
# Objetos de resposta RAG
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: str
    page_start: int
    text_for_llm: str
    text_preview: str
    citation_short: str
    citation_abnt: str
    retrieval_score: float
    marker: str = ""           # [T1]..[T5] assigned by ContextBuilder


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    numeric_discrepancies: list[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    trace_id: str
    query: str
    answer: str
    chunks_used: list[RetrievedChunk]
    hallucination_flags: list[str]
    numeric_discrepancies: list[str]
    response_confidence: float
    coverage_level: CoverageLevel
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    generation_failed: bool
    fallback_triggered: bool
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def documents_used(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for c in self.chunks_used:
            if c.doc_id not in seen:
                seen[c.doc_id] = {
                    "doc_id": c.doc_id,
                    "title": c.doc_title,
                    "citation_abnt": c.citation_abnt,
                }
        return list(seen.values())

    @property
    def citations_abnt(self) -> list[str]:
        seen: dict[str, str] = {}
        for c in self.chunks_used:
            if c.doc_id not in seen:
                seen[c.doc_id] = c.citation_abnt
        return list(seen.values())


# ─────────────────────────────────────────────────────────────
# Relatório de ingestão
# ─────────────────────────────────────────────────────────────

@dataclass
class IngestionReport:
    doc_id: str
    status: IngestionStatus
    duration_seconds: float
    pages_extracted: int
    pages_after_dedup: int
    chunks_generated: int
    quality_score: float
    ocr_pages: int
    fallbacks_used: list[str]
    warnings: list[str]
    errors: list[str]
    sha256: str
    completed_at: datetime = field(default_factory=datetime.utcnow)

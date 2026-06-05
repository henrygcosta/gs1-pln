"""
ingestion/orchestrator.py
Orquestrador da ingestão: coordena loaders, cleaners, normalizers, chunker e indexação.
"""

from __future__ import annotations

import hashlib
import structlog
import time
from pathlib import Path
from typing import Any

from config.settings import get_settings
from domain.entities import (
    CleanedDocument,
    DocumentMetadata,
    DocumentSection,
    DocumentStatus,
    IngestionReport,
    IngestionStatus,
    ProcessedDocument,
)
from domain.exceptions import (
    DocumentLoadError,
    DuplicateDocumentError,
    LowQualityDocumentError,
)
from document_store.sqlite_store import SQLiteDocumentStore
from ingestion.chunker import HierarchicalSemanticChunker
from ingestion.cleaners.duplicate_page_filter import DuplicatePageFilter
from ingestion.cleaners.header_footer_remover import HeaderFooterRemover
from ingestion.cleaners.ocr_noise_cleaner import OcrNoiseCleaner
from ingestion.cleaners.structure_preserver import StructurePreserver
from ingestion.loaders.docx_loader import DocxLoader
from ingestion.loaders.html_loader import HtmlLoader
from ingestion.loaders.pdf_loader import PdfLoader
from ingestion.loaders.txt_loader import TxtLoader
from ingestion.normalizers.encoding_normalizer import EncodingNormalizer
from ingestion.normalizers.lexical_normalizer import LexicalNormalizer
from ingestion.normalizers.section_tagger import SectionTagger
from ingestion.normalizers.whitespace_normalizer import WhitespaceNormalizer
from rag.embedder import EmbedderService
from vector_store.base import VectorStoreBase

logger = structlog.get_logger(__name__)

QUALITY_THRESHOLD = 0.40


class IngestionOrchestrator:
    """
    Coordena o pipeline completo de ingestão de um documento.

    Etapas:
    1. Detecção de duplicata (checksum)
    2. Load (formato-específico)
    3. Limpeza (header/footer, dedup, OCR, estrutura)
    4. Normalização (encoding, whitespace, léxico, seções)
    5. Quality scoring
    6. Chunking (hierárquico semântico)
    7. Embedding (batch)
    8. Indexação (ChromaDB + SQLite)
    """

    PROCESSING_VERSION = "1.0.0"

    def __init__(
        self,
        vector_store: VectorStoreBase,
        document_store: SQLiteDocumentStore,
        embedder: EmbedderService,
    ) -> None:
        self._vs = vector_store
        self._ds = document_store
        self._embedder = embedder
        settings = get_settings()
        self._chunker = HierarchicalSemanticChunker(
            chunk_size_min=settings.chunk_size_min,
            chunk_size_target=settings.chunk_size_target,
            chunk_size_max=settings.chunk_size_max,
            overlap_tokens=settings.chunk_overlap,
        )

        # Loaders
        self._loaders = {
            ".pdf": PdfLoader(),
            ".docx": DocxLoader(),
            ".txt": TxtLoader(),
            ".html": HtmlLoader(),
            ".htm": HtmlLoader(),
        }

        # Cleaners
        self._hf_remover = HeaderFooterRemover()
        self._dup_filter = DuplicatePageFilter()
        self._ocr_cleaner = OcrNoiseCleaner()
        self._structure_preserver = StructurePreserver()

        # Normalizers
        self._encoding_norm = EncodingNormalizer()
        self._whitespace_norm = WhitespaceNormalizer()
        self._lexical_norm = LexicalNormalizer()
        self._section_tagger = SectionTagger()

    def ingest(
        self,
        path: Path,
        metadata: DocumentMetadata,
        force_reingest: bool = False,
    ) -> IngestionReport:
        """
        Ingere um documento completo.

        Args:
            path: Caminho para o arquivo
            metadata: Metadados do documento (definidos no catálogo do corpus)
            force_reingest: Se True, re-ingere mesmo que checksum já exista
        """
        start_time = time.monotonic()
        doc_id = metadata.doc_id
        warnings: list[str] = []
        errors: list[str] = []

        try:
            # ── Checksum e deduplicação ──────────
            sha256 = self._compute_sha256(path)
            if not force_reingest:
                existing_doc_id = self._ds.document_exists_by_sha256(sha256)
                if existing_doc_id:
                    logger.info("Document already ingested (SKIPPED)", extra={"doc_id": doc_id, "sha256": sha256[:12]})
                    return IngestionReport(
                        doc_id=doc_id,
                        status=IngestionStatus.SKIPPED,
                        duration_seconds=0.0,
                        pages_extracted=0,
                        pages_after_dedup=0,
                        chunks_generated=0,
                        quality_score=1.0,
                        ocr_pages=0,
                        fallbacks_used=[],
                        warnings=[f"Already ingested as {existing_doc_id}"],
                        errors=[],
                        sha256=sha256,
                    )

            # ── Passo 1: Load ─────────────────────
            suffix = path.suffix.lower()
            loader = self._loaders.get(suffix)
            if not loader:
                raise DocumentLoadError(doc_id, f"Unsupported format: {suffix}")

            logger.info("Loading document", extra={"doc_id": doc_id, "path": str(path)})
            raw_doc = loader.load(path, doc_id)

            # ── Passo 2: Limpeza ──────────────────
            cleaned_pages = self._hf_remover.clean(raw_doc)
            unique_pages, dup_count, blank_count = self._dup_filter.filter(cleaned_pages)

            if dup_count > 0:
                warnings.append(f"Removed {dup_count} duplicate pages")
            if blank_count > 0:
                warnings.append(f"Removed {blank_count} blank pages")

            # OCR noise + structure preservation
            for page in unique_pages:
                page.cleaned_text = self._ocr_cleaner.clean(page.cleaned_text)

            pages_with_structures = self._structure_preserver.preserve(unique_pages)

            # ── Passo 3: Normalização ─────────────
            sections: list[DocumentSection] = []
            for page in pages_with_structures:
                text = self._encoding_norm.normalize(page.cleaned_text)
                text = self._whitespace_norm.normalize(text)
                text = self._lexical_norm.normalize(text)

                # Detecta seções no texto
                detected = self._section_tagger.detect_sections(text)
                section_path = self._section_tagger.build_section_path(detected, 0)

                sections.append(DocumentSection(
                    section_path=section_path,
                    section_number=str(page.page_number),
                    section_heading_level=1,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    normalized_text=text,
                    preserved_structures=page.preserved_structures,
                ))

            # ── Passo 4: Quality score ─────────────
            quality_score = self._compute_quality_score(sections)
            if quality_score < QUALITY_THRESHOLD:
                raise LowQualityDocumentError(doc_id, quality_score, QUALITY_THRESHOLD)

            # ── Passo 5: Chunking ─────────────────
            chunks = self._chunker.chunk_document(sections, metadata)

            if not chunks:
                warnings.append("No chunks generated after chunking")
                logger.warning("No chunks generated", extra={"doc_id": doc_id})

            # ── Passo 6: Embedding ────────────────
            texts_for_embedding = [c.text_for_embedding for c in chunks]
            embeddings = self._embedder.embed_passages(texts_for_embedding)

            # ── Passo 7: Indexação ────────────────
            # Filtra chunks já existentes (dedup por content_hash)
            new_chunks = []
            new_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                if not self._vs.exists(chunk.chunk_id):
                    new_chunks.append(chunk)
                    new_embeddings.append(embedding)

            if new_chunks:
                self._vs.add(
                    ids=[c.chunk_id for c in new_chunks],
                    embeddings=new_embeddings,
                    documents=[c.text_for_llm for c in new_chunks],
                    metadatas=[c.to_chroma_metadata() for c in new_chunks],
                )
                self._ds.save_chunks(new_chunks)

            # Salva documento no SQLite
            self._ds.save_document(
                doc_id=doc_id,
                title=metadata.title,
                issuer=metadata.issuer,
                year=metadata.year,
                category=metadata.category.value,
                subcategory=metadata.subcategory,
                language=metadata.language,
                local_path=str(path),
                sha256=sha256,
                total_chunks=len(new_chunks),
            )

            duration = time.monotonic() - start_time
            report = IngestionReport(
                doc_id=doc_id,
                status=IngestionStatus.SUCCESS,
                duration_seconds=round(duration, 2),
                pages_extracted=raw_doc.total_pages,
                pages_after_dedup=len(unique_pages),
                chunks_generated=len(new_chunks),
                quality_score=quality_score,
                ocr_pages=sum(1 for p in raw_doc.pages if p.ocr_applied),
                fallbacks_used=[],
                warnings=warnings,
                errors=errors,
                sha256=sha256,
            )
            self._ds.log_ingestion(report)

            logger.info(
                "Ingestion completed",
                doc_id=doc_id,
                chunks=len(new_chunks),
                duration_s=round(duration, 1),
                quality=round(quality_score, 2),
            )
            return report

        except (DocumentLoadError, LowQualityDocumentError) as exc:
            duration = time.monotonic() - start_time
            logger.error("Ingestion failed", extra={"doc_id": doc_id, "error": str(exc)})
            report = IngestionReport(
                doc_id=doc_id,
                status=IngestionStatus.FAILED,
                duration_seconds=round(duration, 2),
                pages_extracted=0,
                pages_after_dedup=0,
                chunks_generated=0,
                quality_score=0.0,
                ocr_pages=0,
                fallbacks_used=[],
                warnings=warnings,
                errors=[str(exc)],
                sha256="",
            )
            self._ds.log_ingestion(report)
            return report

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _compute_quality_score(self, sections: list[DocumentSection]) -> float:
        if not sections:
            return 0.0

        all_text = " ".join(s.normalized_text for s in sections)
        if not all_text.strip():
            return 0.0

        total_chars = len(all_text)
        # Penaliza alta proporção de não-ASCII
        non_ascii = sum(1 for c in all_text if ord(c) > 127)
        ascii_ratio = 1.0 - (non_ascii / max(1, total_chars))

        # Penaliza falta de pontuação
        punct = sum(1 for c in all_text if c in ".!?,;:")
        punct_ratio = min(1.0, punct / max(1, total_chars / 100))

        # Score composto
        return round(0.6 * ascii_ratio + 0.4 * punct_ratio, 3)

"""
document_store/sqlite_store.py
Armazena chunks completos, metadados e log de ingestão em SQLite.
Complementa o ChromaDB (que guarda apenas o necessário para busca).
"""

from __future__ import annotations

import json
import structlog
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from config.settings import get_settings
from domain.entities import Chunk, IngestionReport, IngestionStatus

logger = structlog.get_logger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    chunk_type          TEXT NOT NULL,
    section_path        TEXT,
    section_number      TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    text_for_embedding  TEXT NOT NULL,
    text_for_llm        TEXT NOT NULL,
    citation_short      TEXT,
    citation_full       TEXT,
    citation_abnt       TEXT,
    cross_references    TEXT,       -- JSON array serialized as string
    core_tokens         INTEGER,
    content_hash        TEXT,
    is_normative        INTEGER DEFAULT 0,
    is_table            INTEGER DEFAULT 0,
    is_prerequisite     INTEGER DEFAULT 0,
    must_preserve       INTEGER DEFAULT 0,
    require_human_review INTEGER DEFAULT 0,
    retrieval_count     INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type   ON chunks(chunk_type);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    issuer          TEXT,
    year            INTEGER,
    category        TEXT,
    subcategory     TEXT,
    language        TEXT,
    status          TEXT DEFAULT 'active',
    local_path      TEXT,
    sha256          TEXT,
    total_chunks    INTEGER DEFAULT 0,
    ingested_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL,
    status          TEXT NOT NULL,
    duration_s      REAL,
    pages_extracted INTEGER,
    chunks_generated INTEGER,
    quality_score   REAL,
    warnings        TEXT,   -- JSON
    errors          TEXT,   -- JSON
    sha256          TEXT,
    completed_at    TEXT NOT NULL
);
"""


class SQLiteDocumentStore:
    """Document store relacional para chunks e metadados completos."""

    def __init__(self) -> None:
        settings = get_settings()
        self._db_path = settings.sqlite_db_path
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(DDL)
        logger.info("SQLite document store initialized", extra={"path": str(self._db_path)})

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Chunks ────────────────────────────────

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Salva lista de chunks em batch."""
        if not chunks:
            return

        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.chunk_type.value,
                c.section_path,
                c.section_number,
                c.page_start,
                c.page_end,
                c.text_for_embedding,
                c.text_for_llm,
                c.citation.citation_short,
                c.citation.citation_full,
                c.citation.citation_abnt,
                json.dumps(c.cross_references),
                c.tokens.core_tokens,
                c.content_hash_sha256,
                int(c.is_normative),
                int(c.is_table),
                int(c.is_prerequisite),
                int(c.must_preserve),
                int(c.require_human_review),
                0,
                c.created_at.isoformat(),
            )
            for c in chunks
        ]

        sql = """
        INSERT OR REPLACE INTO chunks
        (chunk_id, doc_id, chunk_type, section_path, section_number,
         page_start, page_end, text_for_embedding, text_for_llm,
         citation_short, citation_full, citation_abnt, cross_references,
         core_tokens, content_hash, is_normative, is_table, is_prerequisite,
         must_preserve, require_human_review, retrieval_count, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._connect() as conn:
            conn.executemany(sql, rows)

        logger.debug("Saved chunks to SQLite", extra={"count": len(chunks)})

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["cross_references"] = json.loads(data.get("cross_references") or "[]")
        return data

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["cross_references"] = json.loads(data.get("cross_references") or "[]")
            result.append(data)
        return result

    def chunk_exists(self, content_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
        return row is not None

    def increment_retrieval_count(self, chunk_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET retrieval_count = retrieval_count + 1 WHERE chunk_id = ?",
                (chunk_id,),
            )

    # ── Documents ─────────────────────────────

    def save_document(
        self,
        doc_id: str,
        title: str,
        issuer: str,
        year: int,
        category: str,
        subcategory: str,
        language: str,
        local_path: str,
        sha256: str,
        total_chunks: int,
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO documents
        (doc_id, title, issuer, year, category, subcategory,
         language, local_path, sha256, total_chunks, ingested_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                doc_id, title, issuer, year, category, subcategory,
                language, local_path, sha256, total_chunks,
                datetime.utcnow().isoformat(),
            ))

    def document_exists_by_sha256(self, sha256: str) -> str | None:
        """Retorna doc_id se já ingerido, None caso contrário."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return row["doc_id"] if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [dict(r) for r in rows]

    # ── Ingestion log ─────────────────────────

    def log_ingestion(self, report: IngestionReport) -> None:
        sql = """
        INSERT INTO ingestion_log
        (doc_id, status, duration_s, pages_extracted, chunks_generated,
         quality_score, warnings, errors, sha256, completed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                report.doc_id,
                report.status.value,
                report.duration_seconds,
                report.pages_extracted,
                report.chunks_generated,
                report.quality_score,
                json.dumps(report.warnings),
                json.dumps(report.errors),
                report.sha256,
                report.completed_at.isoformat(),
            ))

    def get_ingestion_history(self, doc_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_log WHERE doc_id = ? ORDER BY completed_at DESC",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]

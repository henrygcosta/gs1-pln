"""
ingestion/loaders/txt_loader.py
"""

from __future__ import annotations

import structlog
from pathlib import Path

from domain.entities import DocumentFormat, RawDocument, RawPage
from domain.exceptions import DocumentLoadError
from ingestion.loaders.base import LoaderBase

logger = structlog.get_logger(__name__)


class TxtLoader(LoaderBase):
    def load(self, path: Path, doc_id: str) -> RawDocument:
        if not path.exists():
            raise DocumentLoadError(doc_id, f"File not found: {path}")

        # Detecta encoding
        encoding = self._detect_encoding(path)

        try:
            text = path.read_text(encoding=encoding, errors="replace")
        except Exception as exc:
            raise DocumentLoadError(doc_id, f"Cannot read TXT: {exc}") from exc

        # Divide em páginas lógicas de 3000 chars
        page_size = 3000
        pages: list[RawPage] = []
        for i in range(0, max(1, len(text)), page_size):
            pages.append(RawPage(
                page_number=len(pages) + 1,
                raw_text=text[i : i + page_size],
                extraction_method="txt_read",
            ))

        if not pages:
            pages = [RawPage(page_number=1, raw_text=text, extraction_method="txt_read")]

        logger.info("TXT loaded", extra={"doc_id": doc_id, "pages": len(pages), "encoding": encoding})

        return RawDocument(
            doc_id=doc_id,
            source_path=str(path),
            detected_format=DocumentFormat.TXT,
            detected_encoding=encoding,
            detected_language="pt",
            is_scanned=False,
            pages=pages,
            total_pages=len(pages),
        )

    def _detect_encoding(self, path: Path) -> str:
        try:
            import chardet
            raw = path.read_bytes()[:10000]
            result = chardet.detect(raw)
            return result.get("encoding") or "utf-8"
        except ImportError:
            return "utf-8"

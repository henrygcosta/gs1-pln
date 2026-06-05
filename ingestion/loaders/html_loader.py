"""
ingestion/loaders/html_loader.py
"""

from __future__ import annotations

import structlog
from pathlib import Path

from domain.entities import DocumentFormat, RawDocument, RawPage
from domain.exceptions import DocumentLoadError
from ingestion.loaders.base import LoaderBase

logger = structlog.get_logger(__name__)

REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "iframe", "form", "button"}


class HtmlLoader(LoaderBase):
    def load(self, path: Path, doc_id: str) -> RawDocument:
        if not path.exists():
            raise DocumentLoadError(doc_id, f"File not found: {path}")

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise DocumentLoadError(doc_id, "beautifulsoup4 not installed") from exc

        try:
            raw_html = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise DocumentLoadError(doc_id, f"Cannot read HTML: {exc}") from exc

        soup = BeautifulSoup(raw_html, "lxml")

        # Remove tags indesejadas
        for tag in soup.find_all(REMOVE_TAGS):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Divide em páginas lógicas
        page_size = 3000
        pages: list[RawPage] = []
        for i in range(0, max(1, len(text)), page_size):
            pages.append(RawPage(
                page_number=len(pages) + 1,
                raw_text=text[i : i + page_size],
                extraction_method="beautifulsoup4",
            ))

        if not pages:
            pages = [RawPage(page_number=1, raw_text=text, extraction_method="beautifulsoup4")]

        logger.info("HTML loaded", extra={"doc_id": doc_id, "pages": len(pages)})

        return RawDocument(
            doc_id=doc_id,
            source_path=str(path),
            detected_format=DocumentFormat.HTML,
            detected_encoding="utf-8",
            detected_language="pt",
            is_scanned=False,
            pages=pages,
            total_pages=len(pages),
        )

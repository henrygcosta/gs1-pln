"""
ingestion/loaders/pdf_loader.py
Carrega PDFs com PyMuPDF (texto nativo) e pytesseract (fallback OCR).
"""

from __future__ import annotations

import structlog
from pathlib import Path

from domain.entities import BoundingBox, DocumentFormat, RawDocument, RawPage
from domain.exceptions import DocumentLoadError, ExtractionError
from ingestion.loaders.base import LoaderBase

logger = structlog.get_logger(__name__)


class PdfLoader(LoaderBase):
    """
    Carrega PDFs usando PyMuPDF.

    Estratégia:
    1. Tenta extração nativa de texto (PDFs digitais)
    2. Se página não tem texto (< 30 chars), aciona OCR com pytesseract
    3. Extrai bounding boxes para detecção de cabeçalhos/rodapés
    """

    def load(self, path: Path, doc_id: str) -> RawDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise DocumentLoadError(doc_id, "PyMuPDF not installed") from exc

        if not path.exists():
            raise DocumentLoadError(doc_id, f"File not found: {path}")

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            if "password" in str(exc).lower():
                from domain.exceptions import PasswordProtectedError
                raise PasswordProtectedError(doc_id, "PDF is password-protected") from exc
            raise DocumentLoadError(doc_id, f"Cannot open PDF: {exc}") from exc

        if doc.page_count == 0:
            raise DocumentLoadError(doc_id, "PDF has 0 pages")

        pages: list[RawPage] = []
        ocr_page_count = 0

        for page_num in range(doc.page_count):
            try:
                page = doc.load_page(page_num)
                text = page.get_text("text")  # type: ignore[attr-defined]
                bboxes: list[BoundingBox] = []

                # Extrai bounding boxes para heurística de cabeçalho/rodapé
                page_height = page.rect.height
                if page_height > 0:
                    blocks = page.get_text("blocks")  # type: ignore[attr-defined]
                    for block in blocks:
                        # block = (x0, y0, x1, y1, text, block_no, block_type)
                        if len(block) >= 5 and isinstance(block[4], str):
                            bboxes.append(BoundingBox(
                                text=block[4][:100],
                                y_top=block[1] / page_height,
                                y_bottom=block[3] / page_height,
                            ))

                ocr_applied = False
                method = "pymupdf_native"

                # Fallback para OCR se página sem texto
                if len(text.strip()) < 30:
                    text, ocr_applied = self._ocr_page(page, page_num + 1, doc_id)
                    method = "pytesseract_ocr" if ocr_applied else method
                    if ocr_applied:
                        ocr_page_count += 1

                pages.append(RawPage(
                    page_number=page_num + 1,
                    raw_text=text,
                    bounding_boxes=bboxes,
                    extraction_method=method,
                    ocr_applied=ocr_applied,
                ))

            except Exception as exc:
                logger.warning(
                    "Page extraction failed",
                    doc_id=doc_id,
                    page=page_num + 1,
                    error=str(exc),
                )

        doc.close()

        is_scanned = ocr_page_count > len(pages) * 0.5

        logger.info(
            "PDF loaded",
            doc_id=doc_id,
            pages=len(pages),
            ocr_pages=ocr_page_count,
            is_scanned=is_scanned,
        )

        return RawDocument(
            doc_id=doc_id,
            source_path=str(path),
            detected_format=DocumentFormat.PDF,
            detected_encoding="utf-8",
            detected_language="pt",  # Será refinado pelo normalizer
            is_scanned=is_scanned,
            pages=pages,
            total_pages=len(pages),
        )

    def _ocr_page(
        self, page: "fitz.Page", page_num: int, doc_id: str  # type: ignore[name-defined]
    ) -> tuple[str, bool]:
        """Tenta OCR com pytesseract. Retorna (texto, ocr_usado)."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Renderiza a página como imagem (300 DPI para melhor qualidade OCR)
            mat = page.get_pixmap(matrix=page.identity.prescale(300 / 72))  # type: ignore[attr-defined]
            img_bytes = mat.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(img, lang="por+eng")
            return text, True

        except ImportError:
            logger.debug("pytesseract not available, skipping OCR")
            return "", False
        except Exception as exc:
            logger.warning("OCR failed", extra={"doc_id": doc_id, "page": page_num, "error": str(exc)})
            return "", False

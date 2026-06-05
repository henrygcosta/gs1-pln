"""
ingestion/cleaners/duplicate_page_filter.py
Detecta e remove páginas duplicatas via SimHash.
"""

from __future__ import annotations

import structlog

from domain.entities import CleanedPage

logger = structlog.get_logger(__name__)

BLANK_THRESHOLD = 80      # Páginas com menos de N chars são "em branco"
SIMHASH_DISTANCE = 3      # Distância Hamming máxima para considerar duplicata


class DuplicatePageFilter:
    def filter(self, pages: list[CleanedPage]) -> tuple[list[CleanedPage], int, int]:
        """Retorna (páginas_únicas, removidas_duplicatas, removidas_brancas)."""
        try:
            from simhash import Simhash

            def simhash_value(text: str) -> int:
                return Simhash(text).value

        except ImportError:
            # Fallback: hash exato sem simhash
            def simhash_value(text: str) -> int:  # type: ignore[misc]
                return hash(text)

        seen_hashes: list[int] = []
        unique: list[CleanedPage] = []
        dup_count = 0
        blank_count = 0

        for page in pages:
            text = page.cleaned_text

            # Remove páginas em branco
            if len(text.strip()) < BLANK_THRESHOLD:
                blank_count += 1
                logger.debug("Blank page removed", extra={"page": page.page_number})
                continue

            h = simhash_value(text)

            # Verifica duplicata por distância Hamming
            is_dup = False
            for seen in seen_hashes:
                dist = bin(h ^ seen).count("1")
                if dist <= SIMHASH_DISTANCE:
                    is_dup = True
                    break

            if is_dup:
                dup_count += 1
                logger.debug("Duplicate page removed", extra={"page": page.page_number})
            else:
                seen_hashes.append(h)
                unique.append(page)

        return unique, dup_count, blank_count

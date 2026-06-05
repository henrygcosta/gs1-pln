"""
ingestion/cleaners/header_footer_remover.py
Remove cabeçalhos e rodapés por frequência e posição Y.
"""

from __future__ import annotations

import structlog
import re
from collections import Counter

from domain.entities import CleanedPage, RawDocument, RawPage

logger = structlog.get_logger(__name__)

# Threshold: texto presente em > 70% das páginas = cabeçalho/rodapé
FREQUENCY_THRESHOLD = 0.70
# Posição Y: primeiros/últimos 8% da altura da página
POSITION_THRESHOLD = 0.08


class HeaderFooterRemover:
    def clean(self, raw_doc: RawDocument) -> list[CleanedPage]:
        freq_map = self._build_frequency_map(raw_doc.pages)
        total = len(raw_doc.pages)
        cleaned: list[CleanedPage] = []

        for page in raw_doc.pages:
            header_lines: list[str] = []
            footer_lines: list[str] = []
            body_lines: list[str] = []

            for line in page.raw_text.split("\n"):
                normalized = line.strip()
                freq = freq_map.get(normalized, 0) / max(1, total)

                if freq >= FREQUENCY_THRESHOLD and normalized:
                    # Determina se é cabeçalho ou rodapé pela posição
                    if self._is_header_by_position(normalized, page):
                        header_lines.append(normalized)
                    elif self._is_footer_by_position(normalized, page):
                        footer_lines.append(normalized)
                    else:
                        header_lines.append(normalized)
                elif self._is_page_number(normalized):
                    footer_lines.append(normalized)
                else:
                    body_lines.append(line)

            cleaned.append(CleanedPage(
                page_number=page.page_number,
                cleaned_text="\n".join(body_lines).strip(),
                header_removed=" | ".join(header_lines),
                footer_removed=" | ".join(footer_lines),
            ))

        return cleaned

    def _build_frequency_map(self, pages: list[RawPage]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for page in pages:
            seen = set()
            for line in page.raw_text.split("\n"):
                normalized = line.strip()
                if normalized and normalized not in seen:
                    counter[normalized] += 1
                    seen.add(normalized)
        return counter

    def _is_header_by_position(self, text: str, page: RawPage) -> bool:
        for bbox in page.bounding_boxes:
            if bbox.text.strip()[:50] == text[:50] and bbox.y_top < POSITION_THRESHOLD:
                return True
        return False

    def _is_footer_by_position(self, text: str, page: RawPage) -> bool:
        for bbox in page.bounding_boxes:
            if bbox.text.strip()[:50] == text[:50] and bbox.y_bottom > (1 - POSITION_THRESHOLD):
                return True
        return False

    def _is_page_number(self, text: str) -> bool:
        return bool(re.match(r"^(página\s+)?\d+(\s+(de|of)\s+\d+)?$", text, re.IGNORECASE))

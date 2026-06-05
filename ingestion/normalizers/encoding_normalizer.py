"""
ingestion/normalizers/encoding_normalizer.py
Garante saída 100% UTF-8 NFC.
"""

from __future__ import annotations

import html
import re
import unicodedata


class EncodingNormalizer:
    # Caracteres de controle invisíveis (exceto tab, LF, CR)
    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def normalize(self, text: str) -> str:
        # Remove BOM
        text = text.lstrip("\ufeff")
        # Decodifica entidades HTML (&amp; → &, &#233; → é)
        text = html.unescape(text)
        # Forma NFC (canonicaliza codepoints compostos)
        text = unicodedata.normalize("NFC", text)
        # Remove caracteres de controle
        text = self.CONTROL_CHARS.sub(" ", text)
        return text

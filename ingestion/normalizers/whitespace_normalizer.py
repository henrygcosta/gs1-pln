"""
ingestion/normalizers/whitespace_normalizer.py
"""

from __future__ import annotations

import re


class WhitespaceNormalizer:
    # Múltiplos espaços → um
    MULTI_SPACE = re.compile(r"[ \t]+")
    # Tab → espaço
    TAB = re.compile(r"\t")
    # Mais de 2 newlines → 2
    MULTI_NEWLINE = re.compile(r"\n{3,}")
    # Espaço antes de pontuação
    SPACE_PUNCT = re.compile(r" +([.,;:!?])")
    # Palavra hifenizada por quebra de linha
    HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

    def normalize(self, text: str) -> str:
        text = self.TAB.sub(" ", text)
        text = self.HYPHEN_BREAK.sub(r"\1\2", text)  # Recompõe palavras
        text = self.MULTI_SPACE.sub(" ", text)
        text = self.SPACE_PUNCT.sub(r"\1", text)
        text = self.MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

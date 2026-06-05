"""
ingestion/cleaners/ocr_noise_cleaner.py
"""

from __future__ import annotations

import re

# Termos técnicos que NÃO devem ser "corrigidos" pelo limpador de OCR
PROTECTED_TERMS = frozenset({
    "LEED", "AQUA-HQE", "AQUA", "HQE", "GBC", "ABNT", "NBR", "PROCEL",
    "ABSOLAR", "ASHRAE", "BEMS", "HVAC", "FV", "kWh", "MWh", "W/m²",
    "fotovoltaico", "hídrico", "eficiência",
})

# Substituições comuns de erros de OCR
OCR_SUBSTITUTIONS = [
    (r"\b1uz\b", "luz"),
    (r"\brn\b", "m"),
    (r"(?<=[a-z])l(?=[0-9])", "1"),
    (r"(?<=[0-9])O(?=[0-9])", "0"),
]

# Recomposição de palavras hifenizadas por quebra de linha
HYPHEN_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")


class OcrNoiseCleaner:
    def clean(self, text: str) -> str:
        # Recompõe palavras hifenizadas
        text = HYPHEN_RE.sub(lambda m: m.group(1) + m.group(2), text)

        # Aplica ftfy para corrigir encoding
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            pass

        # Aplica substituições OCR apenas fora de termos protegidos
        for pattern, replacement in OCR_SUBSTITUTIONS:
            text = re.sub(pattern, replacement, text)

        return text

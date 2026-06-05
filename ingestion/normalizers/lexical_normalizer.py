"""
ingestion/normalizers/lexical_normalizer.py
Expande siglas do domínio e normaliza variações ortográficas.
"""

from __future__ import annotations

import re

ACRONYM_EXPANSIONS: dict[str, str] = {
    r"\bFV\b": "fotovoltaico (FV)",
    r"\bGEE\b": "gases de efeito estufa (GEE)",
    r"\bHVAC\b": "aquecimento, ventilação e ar condicionado (HVAC)",
    r"\bBEMS\b": "Building Energy Management System (BEMS)",
    r"\bHQE\b": "Haute Qualité Environnementale (HQE)",
}

ORTHO_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\breuso\b", re.I), "reúso"),
    (re.compile(r"\bfoto-voltaico\b", re.I), "fotovoltaico"),
    (re.compile(r"\baqua-hqe\b", re.I), "AQUA-HQE"),
]


class LexicalNormalizer:
    def normalize(self, text: str) -> str:
        # Expansão de siglas
        for pattern, replacement in ACRONYM_EXPANSIONS.items():
            text = re.sub(pattern, replacement, text)
        # Correções ortográficas
        for pat, rep in ORTHO_FIXES:
            text = pat.sub(rep, text)
        return text

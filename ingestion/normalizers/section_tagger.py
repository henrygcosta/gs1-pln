"""
ingestion/normalizers/section_tagger.py
Detecta hierarquia de seções e constrói section_path para cada bloco.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


HEADING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(\d+)\s+([A-ZÁÉÍÓÚ][^.\n]{5,60})$", re.MULTILINE),            # "1 Introdução"
    re.compile(r"^(\d+\.\d+)\s+([A-ZÁÉÍÓÚ][^.\n]{5,60})$", re.MULTILINE),      # "1.2 Eficiência"
    re.compile(r"^(\d+\.\d+\.\d+)\s+([A-ZÁÉÍÓÚ][^.\n]{5,60})$", re.MULTILINE),  # "1.2.3 Requisitos"
    re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE),                                 # Markdown headings
]


@dataclass
class DetectedSection:
    number: str
    title: str
    position: int
    level: int


class SectionTagger:
    def detect_sections(self, text: str) -> list[DetectedSection]:
        sections: list[DetectedSection] = []

        for level, pattern in enumerate(HEADING_PATTERNS, start=1):
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 2:
                    number, title = groups
                elif len(groups) == 1:
                    number, title = "", groups[0]
                else:
                    continue

                sections.append(DetectedSection(
                    number=number.strip(),
                    title=title.strip(),
                    position=match.start(),
                    level=level,
                ))

        # Ordena por posição no texto
        sections.sort(key=lambda s: s.position)
        return sections

    def build_section_path(self, sections: list[DetectedSection], position: int) -> str:
        """Retorna o caminho hierárquico de seção para um ponto do texto."""
        path_parts: list[str] = []

        for section in sections:
            if section.position <= position:
                # Mantém hierarquia (substitui partes do mesmo nível)
                level = section.level
                while len(path_parts) >= level:
                    path_parts.pop()
                path_parts.append(section.title)

        return " > ".join(path_parts) if path_parts else "Introdução"

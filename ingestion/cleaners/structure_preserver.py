"""
ingestion/cleaners/structure_preserver.py
Detecta e preserva tabelas, requisitos normativos e fórmulas.
"""

from __future__ import annotations

import re
from typing import ClassVar

from domain.entities import CleanedPage, PreservedStructure

# Padrões para detecção de elementos preservados
NORMATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:Art(?:igo)?\.?\s*\d+|§\s*\d+|Inciso\s+[IVXLC]+|\d+\.\d+(?:\.\d+)*\s+(?:Requisito|Deve|Deverá))", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\d+\.\d+\.\d+\s+[A-Z]", re.MULTILINE),
]

FORMULA_PATTERN = re.compile(r"[A-Z]\s*=\s*[\w\s\+\-\*/\^\.]+(?:kWh|W|m²|L|%)", re.MULTILINE)

TABLE_SIGNAL_PATTERN = re.compile(r"^\s*[\|│].*[\|│]\s*$", re.MULTILINE)


class StructurePreserver:
    """
    Identifica estruturas que NÃO devem ser fragmentadas pelo chunker.
    Marca blocos com PreservedStructure antes da limpeza de espaços.
    """

    def preserve(self, pages: list[CleanedPage]) -> list[CleanedPage]:
        result: list[CleanedPage] = []
        for page in pages:
            structures: list[PreservedStructure] = []
            text = page.cleaned_text

            # Detecta tabelas Markdown (linhas com |)
            if TABLE_SIGNAL_PATTERN.search(text):
                tables = self._extract_tables(text)
                for i, tbl in enumerate(tables):
                    structures.append(PreservedStructure(
                        type="table",
                        content=tbl,
                        original_position=f"page_{page.page_number}_table_{i+1}",
                        must_preserve=True,
                    ))

            # Detecta requisitos normativos
            for pat in NORMATIVE_PATTERNS:
                for m in pat.finditer(text):
                    block = self._extract_paragraph(text, m.start())
                    if block and len(block) > 20:
                        structures.append(PreservedStructure(
                            type="normative_requirement",
                            content=block,
                            original_position=f"page_{page.page_number}_norm_{m.start()}",
                            must_preserve=True,
                        ))

            # Detecta fórmulas
            for m in FORMULA_PATTERN.finditer(text):
                structures.append(PreservedStructure(
                    type="formula",
                    content=m.group(0).strip(),
                    original_position=f"page_{page.page_number}_formula_{m.start()}",
                    must_preserve=True,
                ))

            # Remove duplicatas de structures
            seen: set[str] = set()
            unique_structures: list[PreservedStructure] = []
            for s in structures:
                key = s.content[:50]
                if key not in seen:
                    seen.add(key)
                    unique_structures.append(s)

            result.append(CleanedPage(
                page_number=page.page_number,
                cleaned_text=page.cleaned_text,
                header_removed=page.header_removed,
                footer_removed=page.footer_removed,
                is_duplicate=page.is_duplicate,
                cleaning_skipped=page.cleaning_skipped,
                preserved_structures=unique_structures,
                cleaning_warnings=page.cleaning_warnings,
            ))

        return result

    def _extract_tables(self, text: str) -> list[str]:
        """Extrai blocos de tabela Markdown do texto."""
        lines = text.split("\n")
        tables: list[str] = []
        in_table = False
        current: list[str] = []

        for line in lines:
            if "|" in line:
                in_table = True
                current.append(line)
            else:
                if in_table and current:
                    tables.append("\n".join(current))
                    current = []
                    in_table = False

        if current:
            tables.append("\n".join(current))

        return tables

    def _extract_paragraph(self, text: str, pos: int) -> str:
        """Extrai o parágrafo completo a partir de uma posição."""
        # Busca fim do parágrafo (linha em branco ou outro parágrafo)
        end = text.find("\n\n", pos)
        if end == -1:
            end = min(pos + 800, len(text))

        # Busca início do parágrafo
        start = text.rfind("\n\n", 0, pos)
        start = start + 2 if start != -1 else 0

        return text[start:end].strip()

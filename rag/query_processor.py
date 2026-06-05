"""
rag/query_processor.py
Normaliza a query, expande termos técnicos e infere filtros de metadados ChromaDB.
"""

from __future__ import annotations

import structlog
import re
from dataclasses import dataclass, field
from typing import Any

logger = structlog.get_logger(__name__)

# ── Mapeamento de termos → doc_id ─────────────────────────────
TERM_TO_DOC: dict[str, str] = {
    "leed": "DOC-001",
    "aqua-hqe": "DOC-002",
    "aqua hqe": "DOC-002",
    "nbr 15575": "DOC-003",
    "abnt 15575": "DOC-003",
    "nbr 10844": "DOC-004",
    "10844": "DOC-004",
    "casa azul": "DOC-005",
    "iea": "DOC-006",
    "procel edifica": "DOC-007",
    "cbcs": "DOC-008",
    "absolar": "DOC-011",
    "sinduscon": "DOC-013",
    "ashrae 90.1": "DOC-014",
    "ashrae": "DOC-014",
    "abesco": "DOC-015",
    "bems": "DOC-015",
}

# ── Tags de intent por categoria de conteúdo ──────────────────
TABLE_TRIGGERS = {
    "tabela", "valores", "dimensão", "dimensionamento", "coeficiente",
    "pontuação", "pontos", "créditos", "tabela de", "tabelas de",
}
NORMATIVE_TRIGGERS = {
    "requisito", "obrigatório", "deve", "deverá", "artigo", "art.", "§",
    "pré-requisito", "prereq", "prerequisite", "must", "shall",
}

# ── Expansão de siglas do domínio ─────────────────────────────
ACRONYM_MAP: dict[str, str] = {
    r"\bFV\b": "fotovoltaico (FV)",
    r"\bGEE\b": "gases de efeito estufa (GEE)",
    r"\bHVAC\b": "sistemas de ventilação, aquecimento e ar condicionado (HVAC)",
    r"\bWE\b": "Water Efficiency (WE)",
    r"\bEA\b": "Energy and Atmosphere (EA)",
    r"\bSS\b": "Sustainable Sites (SS)",
    r"\bBEMS\b": "Building Energy Management System (BEMS)",
}


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    chroma_filter: dict[str, Any]
    is_table_query: bool
    is_normative_query: bool
    detected_doc_ids: list[str]
    intent_tags: list[str] = field(default_factory=list)


class QueryProcessor:
    """
    Pré-processamento da query antes do embedding.

    Responsabilidades (SRP):
    1. Normalizar texto
    2. Expandir siglas do domínio
    3. Detectar intent e doc_ids
    4. Construir where clause ChromaDB
    """

    def process(self, query: str, extra_filters: dict[str, Any] | None = None) -> ProcessedQuery:
        """
        Processa a query e retorna ProcessedQuery.
        Query vazia retorna ProcessedQuery com campos vazios — não levanta exceção.
        """
        query = query.strip() if query else ""
        if not query:
            return ProcessedQuery(
                original="",
                normalized="",
                chroma_filter=self._build_filter([], False, False, extra_filters),
                is_table_query=False,
                is_normative_query=False,
                detected_doc_ids=[],
                intent_tags=[],
            )

        normalized = self._normalize(query)
        expanded = self._expand_acronyms(normalized)
        lower = query.lower()
        doc_ids = self._detect_doc_ids(lower)
        is_table = self._has_table_intent(lower)
        is_norm = self._has_normative_intent(lower)
        intent_tags = self._build_intent_tags(is_table, is_norm, doc_ids)
        chroma_filter = self._build_filter(doc_ids, is_table, is_norm, extra_filters)

        logger.debug(
            "Query processed",
            extra={
                "original": query[:60],
                "doc_ids": doc_ids,
                "is_table": is_table,
                "is_normative": is_norm,
                "intent_tags": intent_tags,
            },
        )

        return ProcessedQuery(
            original=query,
            normalized=expanded,
            chroma_filter=chroma_filter,
            is_table_query=is_table,
            is_normative_query=is_norm,
            detected_doc_ids=doc_ids,
            intent_tags=intent_tags,
        )

    # ── Métodos privados ──────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = " ".join(text.split())
        text = re.sub(r"[^\w\s\-\.,?!áéíóúãõâêîôûçÁÉÍÓÚÃÕÂÊÎÔÛÇ]", " ", text)
        return text.strip()

    def _expand_acronyms(self, text: str) -> str:
        for pattern, replacement in ACRONYM_MAP.items():
            text = re.sub(pattern, replacement, text)
        return text

    def _detect_doc_ids(self, lower_query: str) -> list[str]:
        found: list[str] = []
        for term, doc_id in TERM_TO_DOC.items():
            if term in lower_query and doc_id not in found:
                found.append(doc_id)
        return found

    def _has_table_intent(self, lower_query: str) -> bool:
        return any(trigger in lower_query for trigger in TABLE_TRIGGERS)

    def _has_normative_intent(self, lower_query: str) -> bool:
        return any(trigger in lower_query for trigger in NORMATIVE_TRIGGERS)

    def _build_intent_tags(
        self, is_table: bool, is_norm: bool, doc_ids: list[str]
    ) -> list[str]:
        tags: list[str] = []
        if is_table:
            tags.append("table_query")
        if is_norm:
            tags.append("normative_query")
        if doc_ids:
            tags.append(f"doc_filter:{','.join(doc_ids)}")
        return tags

    def _build_filter(
        self,
        doc_ids: list[str],
        is_table: bool,
        is_normative: bool,
        extra_filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Monta where clause ChromaDB com $and de condições."""
        conditions: list[dict[str, Any]] = [
            {"doc_status": {"$eq": "active"}},
            {"require_human_review": {"$eq": 0}},
        ]

        if len(doc_ids) == 1:
            conditions.append({"doc_id": {"$eq": doc_ids[0]}})
        elif len(doc_ids) > 1:
            conditions.append({"doc_id": {"$in": doc_ids}})

        if extra_filters:
            for k, v in extra_filters.items():
                conditions.append({k: {"$eq": v}})

        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

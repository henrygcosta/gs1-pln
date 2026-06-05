"""
rag/citation_formatter.py
Extrai marcadores [Tx] da resposta do LLM e formata as citações.
"""

from __future__ import annotations

import structlog
import re
from dataclasses import dataclass

from domain.entities import RetrievedChunk

logger = structlog.get_logger(__name__)

MARKER_RE = re.compile(r"\[T([1-5])\]")


@dataclass
class FormattedResponse:
    answer_with_citations: str        # Texto com marcações inline
    chunks_cited: list[RetrievedChunk]  # Apenas os chunks efetivamente citados
    section_trechos: str               # Seção "Trechos utilizados"
    section_documentos: str            # Seção "Documentos consultados"
    section_fontes: str                # Seção "Fontes" (ABNT)
    citation_coverage: float           # Fração de marcadores esperados que foram usados


class CitationFormatter:
    """
    Formata as citações da resposta do LLM em três etapas:

    1. Extrai quais marcadores [T1]..[T5] foram realmente usados
    2. Busca os metadados dos chunks correspondentes
    3. Formata as três seções obrigatórias: Trechos, Documentos, Fontes
    """

    def format(
        self,
        raw_response: str,
        chunks_with_markers: list[RetrievedChunk],
    ) -> FormattedResponse:
        """
        Processa a resposta bruta do LLM e retorna a resposta formatada.

        Args:
            raw_response: Texto gerado pelo LLM (pode já conter as seções)
            chunks_with_markers: Chunks com markers [T1]..[T5] atribuídos
        """
        # Mapeia marker → chunk
        marker_map: dict[str, RetrievedChunk] = {
            c.marker: c for c in chunks_with_markers if c.marker
        }

        # Encontra quais markers foram usados na resposta
        used_markers = self._extract_used_markers(raw_response)

        # Chunks efetivamente citados (deduplica por doc_id para as seções)
        cited_chunks: list[RetrievedChunk] = []
        for marker in sorted(used_markers):
            chunk = marker_map.get(marker)
            if chunk:
                cited_chunks.append(chunk)

        # Extrai apenas a seção "Resposta técnica" se o LLM já gerou as outras
        answer_part = self._extract_answer_section(raw_response)

        # Formata as três seções obrigatórias
        section_trechos = self._build_trechos_section(cited_chunks)
        section_documentos = self._build_documentos_section(cited_chunks)
        section_fontes = self._build_fontes_section(cited_chunks)

        # Monta resposta final completa
        final = self._assemble_response(
            answer_part,
            section_trechos,
            section_documentos,
            section_fontes,
        )

        # Calcula coverage
        expected = len(chunks_with_markers)
        coverage = len(cited_chunks) / expected if expected > 0 else 0.0

        logger.debug(
            "Citations formatted",
            used_markers=list(used_markers),
            cited_chunks=len(cited_chunks),
            coverage=f"{coverage:.0%}",
        )

        return FormattedResponse(
            answer_with_citations=final,
            chunks_cited=cited_chunks,
            section_trechos=section_trechos,
            section_documentos=section_documentos,
            section_fontes=section_fontes,
            citation_coverage=round(coverage, 3),
        )

    def _extract_used_markers(self, text: str) -> set[str]:
        """Extrai todos os marcadores [Tx] usados no texto."""
        matches = MARKER_RE.findall(text)
        return {f"T{m}" for m in matches}

    def _extract_answer_section(self, text: str) -> str:
        """
        Se o LLM seguiu o formato, extrai apenas a seção '## Resposta técnica'.
        Caso contrário, usa o texto completo.
        """
        # Procura pelo padrão de seção gerado pelo LLM
        match = re.search(
            r"##\s*Resposta técnica\s*\n(.*?)(?=##\s*Documentos consultados|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return f"## Resposta técnica\n{match.group(1).strip()}"
        return f"## Resposta técnica\n{text.strip()}"

    def _build_trechos_section(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "## Trechos utilizados\nNenhum trecho foi utilizado diretamente nesta resposta."

        lines = ["## Trechos utilizados", ""]
        for chunk in chunks:
            lines.append(f"**[{chunk.marker}]** — {chunk.citation_short}")
            lines.append(f"*Seção:* {chunk.section_path or '—'}")
            preview = chunk.text_preview[:250] + "..." if len(chunk.text_preview) > 250 else chunk.text_preview
            lines.append(f'> "{preview}"')
            lines.append(f"*Relevância:* {chunk.retrieval_score:.2f}")
            lines.append("")
        return "\n".join(lines)

    def _build_documentos_section(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "## Documentos consultados\n"
                "Nenhum documento do corpus continha a informação solicitada "
                "com confiança suficiente."
            )

        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = chunk

        lines = ["## Documentos consultados", ""]
        for i, (doc_id, chunk) in enumerate(seen.items(), 1):
            # Usa apenas o título em negrito — sem citation_short para evitar repetição
            title = chunk.doc_title or doc_id
            lines.append(f"{i}. **{title}** ({doc_id})")
            lines.append("")
        return "\n".join(lines)

    def _build_fontes_section(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "## Fontes\n—"

        seen: dict[str, str] = {}
        for chunk in chunks:
            if chunk.doc_id not in seen and chunk.citation_abnt:
                seen[chunk.doc_id] = chunk.citation_abnt

        if not seen:
            return "## Fontes\n—"

        lines = ["## Fontes", ""]
        for abnt in seen.values():
            lines.append(abnt)
            lines.append("")
        return "\n".join(lines)

    def _assemble_response(
        self,
        answer: str,
        trechos: str,
        documentos: str,
        fontes: str,
    ) -> str:
        return "\n\n".join([answer, documentos, trechos, fontes])

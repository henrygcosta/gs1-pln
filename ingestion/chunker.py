"""
ingestion/chunker.py
Chunker hierárquico semântico — Clean Architecture, SOLID, tipagem completa.

Estratégia de chunking por tipo:
- Tabela          → TableChunker
- Normativo       → NormativeChunker
- Lista           → ListChunker
- Texto corrido   → TextChunker (R4→R5→R6)

O HierarchicalSemanticChunker orquestra e constrói os objetos Chunk finais.
"""

from __future__ import annotations

import hashlib
import structlog
import re
from typing import Optional

from domain.entities import (
    Chunk,
    ChunkCitation,
    ChunkTokens,
    ChunkType,
    DocumentMetadata,
    DocumentSection,
    PreservedStructure,
)

logger = structlog.get_logger(__name__)

CHARS_PER_TOKEN = 4  # Estimativa: 1 token ≈ 4 chars em português técnico


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _build_chunk(
    text: str,
    section: DocumentSection,
    metadata: DocumentMetadata,
    index: int,
    chunk_type: ChunkType = ChunkType.TEXT,
    must_preserve: bool = False,
    is_table: bool = False,
    is_normative: bool = False,
    max_tokens: int = 1024,
) -> Chunk:
    """Constrói um objeto Chunk a partir de texto e metadados. Função pura."""
    tokens_count = _estimate_tokens(text)
    context_header = (
        f"[DOC: {metadata.title} | SEÇÃO: {section.section_path} | p. {section.page_start}]"
    )
    text_for_llm = f"{context_header}\n\n{text}"
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    chunk_id = f"{metadata.doc_id}_s{section.section_number}_c{index:04d}"

    citation = ChunkCitation(
        citation_short=(
            f"[{metadata.title}, Seção {section.section_number}, p. {section.page_start}]"
        ),
        citation_full=(
            f"{metadata.issuer}. {metadata.title_full}. {metadata.year}. "
            f"Seção {section.section_path}, p. {section.page_start}-{section.page_end}."
        ),
        citation_abnt=(
            f"{metadata.issuer.upper()}. {metadata.title_full}. "
            f"{metadata.issuer_country}: {metadata.issuer}, {metadata.year}. "
            f"p. {section.page_start}-{section.page_end}."
        ),
    )

    section_lower = section.section_path.lower()
    is_prereq = any(k in section_lower for k in ("prereq", "pré-requisito", "prerequisite"))

    return Chunk(
        chunk_id=chunk_id,
        chunk_index_in_doc=index,
        doc_id=metadata.doc_id,
        chunk_type=chunk_type,
        is_table=is_table,
        is_normative=is_normative,
        is_prerequisite=is_prereq,
        is_oversized=tokens_count > max_tokens and must_preserve,
        section_path=section.section_path,
        section_number=section.section_number,
        page_start=section.page_start,
        page_end=section.page_end,
        tokens=ChunkTokens(
            core_tokens=tokens_count,
            overlap_tokens_prev=0,
            context_header_tokens=_estimate_tokens(context_header),
        ),
        text_for_embedding=text,
        text_for_llm=text_for_llm,
        must_preserve=must_preserve,
        cross_references=[],
        citation=citation,
        doc_title=metadata.title,
        doc_category=metadata.category.value,
        doc_subcategory=metadata.subcategory,
        doc_language=metadata.language,
        doc_year=metadata.year,
        doc_issuer=metadata.issuer,
        doc_status=metadata.status.value,
        content_hash_sha256=content_hash,
    )


class HierarchicalSemanticChunker:
    """
    Orquestrador de chunking — responsabilidade única: coordenar sub-chunkers
    e produzir a lista final de objetos Chunk com overlaps injetados.
    """

    def __init__(
        self,
        chunk_size_min: int = 512,
        chunk_size_target: int = 768,
        chunk_size_max: int = 1024,
        overlap_tokens: int = 128,
    ) -> None:
        self._min = chunk_size_min
        self._target = chunk_size_target
        self._max = chunk_size_max
        self._overlap = overlap_tokens
        self._table_chunker = TableChunker(max_tokens=chunk_size_max)
        self._norm_chunker = NormativeChunker()
        self._list_chunker = ListChunker(max_tokens=chunk_size_max, overlap=overlap_tokens)
        self._text_chunker = TextChunker(
            min_tokens=chunk_size_min,
            target_tokens=chunk_size_target,
            max_tokens=chunk_size_max,
            overlap_tokens=overlap_tokens,
        )

    def chunk_document(
        self,
        sections: list[DocumentSection],
        metadata: DocumentMetadata,
        chunk_index_offset: int = 0,
    ) -> list[Chunk]:
        """Ponto de entrada público — processa todas as seções."""
        all_chunks: list[Chunk] = []
        global_index = chunk_index_offset

        for section in sections:
            section_chunks = self._chunk_section(section, metadata, global_index)
            self._inject_overlaps(section_chunks)
            all_chunks.extend(section_chunks)
            global_index += len(section_chunks)

        logger.debug(
            "Document chunked",
            extra={"doc_id": metadata.doc_id, "sections": len(sections), "total": len(all_chunks)},
        )
        return all_chunks

    def _chunk_section(
        self,
        section: DocumentSection,
        metadata: DocumentMetadata,
        global_index: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        # 1. Estruturas preservadas (tabelas, normativos, listas)
        for ps in section.preserved_structures:
            ps_chunks = self._chunk_preserved(ps, section, metadata, global_index + len(chunks))
            chunks.extend(ps_chunks)

        # 2. Texto corrido — delega ao TextChunker que retorna Chunk objetos
        text = self._remove_preserved_markers(section.normalized_text)
        if text.strip():
            text_chunks = self._text_chunker.chunk(
                text=text,
                section=section,
                metadata=metadata,
                start_index=global_index + len(chunks),
            )
            chunks.extend(text_chunks)

        return chunks

    def _chunk_preserved(
        self,
        ps: PreservedStructure,
        section: DocumentSection,
        metadata: DocumentMetadata,
        index: int,
    ) -> list[Chunk]:
        if ps.type == "table":
            raw_texts = self._table_chunker.chunk(ps)
            chunk_type, is_table, is_norm = ChunkType.TABLE, True, False
        elif ps.type in ("normative_requirement", "formula"):
            raw_texts = self._norm_chunker.chunk(ps)
            chunk_type, is_table, is_norm = ChunkType.NORMATIVE, False, True
        elif ps.type == "list":
            raw_texts = self._list_chunker.chunk(ps)
            chunk_type, is_table, is_norm = ChunkType.LIST, False, False
        else:
            raw_texts = [ps.content]
            chunk_type, is_table, is_norm = ChunkType.TEXT, False, False

        return [
            _build_chunk(
                text=text,
                section=section,
                metadata=metadata,
                index=index + i,
                chunk_type=chunk_type,
                must_preserve=ps.must_preserve,
                is_table=is_table,
                is_normative=is_norm,
                max_tokens=self._max,
            )
            for i, text in enumerate(raw_texts)
        ]

    def _inject_overlaps(self, chunks: list[Chunk]) -> None:
        """Injeta tail do chunk anterior no início do próximo (sem reembedding)."""
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text_for_embedding
            overlap_chars = self._overlap * CHARS_PER_TOKEN
            overlap_text = (
                prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text
            )
            current = chunks[i]
            new_llm_text = f"...[continuação de trecho anterior]\n{overlap_text}\n\n{current.text_for_embedding}"
            chunks[i] = Chunk(
                chunk_id=current.chunk_id,
                chunk_index_in_doc=current.chunk_index_in_doc,
                doc_id=current.doc_id,
                chunk_type=current.chunk_type,
                is_table=current.is_table,
                is_normative=current.is_normative,
                is_prerequisite=current.is_prerequisite,
                is_oversized=current.is_oversized,
                section_path=current.section_path,
                section_number=current.section_number,
                page_start=current.page_start,
                page_end=current.page_end,
                tokens=ChunkTokens(
                    core_tokens=current.tokens.core_tokens,
                    overlap_tokens_prev=_estimate_tokens(overlap_text),
                    context_header_tokens=current.tokens.context_header_tokens,
                ),
                text_for_embedding=current.text_for_embedding,
                text_for_llm=new_llm_text,
                must_preserve=current.must_preserve,
                cross_references=current.cross_references,
                citation=current.citation,
                doc_title=current.doc_title,
                doc_category=current.doc_category,
                doc_subcategory=current.doc_subcategory,
                doc_language=current.doc_language,
                doc_year=current.doc_year,
                doc_issuer=current.doc_issuer,
                doc_status=current.doc_status,
                content_hash_sha256=current.content_hash_sha256,
            )

    @staticmethod
    def _remove_preserved_markers(text: str) -> str:
        return re.sub(r"\[PRESERVED:[^\]]+\]", "", text).strip()


class TableChunker:
    """Divide tabelas Markdown grandes. Preserva cabeçalho em cada parte."""

    def __init__(self, max_tokens: int = 1024) -> None:
        self._max = max_tokens

    def chunk(self, ps: PreservedStructure) -> list[str]:
        if _estimate_tokens(ps.content) <= self._max:
            return [ps.content]

        lines = ps.content.split("\n")
        header_lines: list[str] = []
        data_lines: list[str] = []
        separator_found = False

        for line in lines:
            if not separator_found and re.match(r"^\|[-| :]+\|$", line.strip()):
                separator_found = True
                header_lines.append(line)
            elif not separator_found:
                header_lines.append(line)
            else:
                data_lines.append(line)

        if not data_lines:
            return [ps.content]

        header = "\n".join(header_lines)
        rows_per_chunk = max(2, (self._max * CHARS_PER_TOKEN) // max(1, len(data_lines)))
        result = []
        for i in range(0, len(data_lines), rows_per_chunk):
            batch = data_lines[i : i + rows_per_chunk]
            result.append(f"{header}\n" + "\n".join(batch))

        return result or [ps.content]


class NormativeChunker:
    """Preserva requisitos normativos como unidade atômica (must_preserve=True)."""

    def chunk(self, ps: PreservedStructure) -> list[str]:
        return [ps.content]


class ListChunker:
    """Divide listas grandes mantendo itens coesos."""

    def __init__(self, max_tokens: int = 1024, overlap: int = 64) -> None:
        self._max = max_tokens
        self._overlap = overlap

    def chunk(self, ps: PreservedStructure) -> list[str]:
        if _estimate_tokens(ps.content) <= self._max:
            return [ps.content]

        items = ps.content.split("\n")
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for item in items:
            item_tokens = _estimate_tokens(item)
            if current_tokens + item_tokens > self._max and current:
                result.append("\n".join(current))
                overlap_count = max(1, self._overlap // max(1, item_tokens))
                current = current[-overlap_count:]
                current_tokens = _estimate_tokens("\n".join(current))

            current.append(item)
            current_tokens += item_tokens

        if current:
            result.append("\n".join(current))

        return result or [ps.content]


class TextChunker:
    """
    Chunker para texto corrido.

    Hierarquia de divisão:
    R4 — parágrafo duplo (\n\n)
    R5 — heading (implícito ao agrupar parágrafos)
    R6 — sentença (fallback para parágrafos > max_tokens)

    Interface pública dupla:
    - chunk(text, section, metadata, start_index) → list[Chunk]  (uso pelo orchestrador)
    - chunk(text) → list[str]                                     (uso interno)
    """

    DOUBLE_NEWLINE = re.compile(r"\n{2,}")
    SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        min_tokens: int = 512,
        target_tokens: int = 768,
        max_tokens: int = 1024,
        overlap_tokens: int = 128,
    ) -> None:
        self._min = min_tokens
        self._target = target_tokens
        self._max = max_tokens
        self._overlap = overlap_tokens

    def chunk(
        self,
        text: str,
        section: Optional[DocumentSection] = None,
        metadata: Optional[DocumentMetadata] = None,
        start_index: int = 0,
    ) -> list[Chunk] | list[str]:
        """
        Divide texto em chunks semânticos.

        Se section e metadata forem fornecidos, retorna list[Chunk].
        Caso contrário, retorna list[str] (uso interno pelo orchestrador).
        """
        raw_texts = self._split_to_raw(text)
        if section is None or metadata is None:
            return raw_texts

        return [
            _build_chunk(
                text=raw,
                section=section,
                metadata=metadata,
                index=start_index + i,
                chunk_type=ChunkType.TEXT,
            )
            for i, raw in enumerate(raw_texts)
            if raw.strip()
        ]

    def _split_to_raw(self, text: str) -> list[str]:
        """Retorna list[str] de chunks semânticos."""
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in self.DOUBLE_NEWLINE.split(text) if p.strip()]
        return self._merge_to_target(paragraphs)

    def _merge_to_target(self, paragraphs: list[str]) -> list[str]:
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _estimate_tokens(para)

            if para_tokens > self._max:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts, current_tokens = [], 0
                chunks.extend(self._split_by_sentence(para))
                continue

            if current_tokens + para_tokens > self._max and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_tokens = [para], para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens
                if current_tokens >= self._target:
                    chunks.append("\n\n".join(current_parts))
                    current_parts, current_tokens = [], 0

        if current_parts:
            remaining = "\n\n".join(current_parts)
            if chunks and _estimate_tokens(remaining) < self._min // 2:
                chunks[-1] = chunks[-1] + "\n\n" + remaining
            else:
                chunks.append(remaining)

        return [c for c in chunks if c.strip()]

    def _split_by_sentence(self, text: str) -> list[str]:
        """R6 — fallback de divisão por sentença."""
        sentences = self.SENTENCE_END.split(text)
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _estimate_tokens(sent)
            if current_tokens + sent_tokens > self._max and current:
                result.append(" ".join(current))
                current, current_tokens = [sent], sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            result.append(" ".join(current))

        return [r for r in result if r.strip()] or [text]

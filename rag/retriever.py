"""
rag/retriever.py
Recuperação vetorial em dois estágios:
  1. ChromaDB top-K com pré-filtro de metadados
  2. Expansão por referências cruzadas (cross_references)
"""

from __future__ import annotations

import structlog
from typing import Any

from config.settings import get_settings
from domain.entities import RetrievedChunk
from document_store.sqlite_store import SQLiteDocumentStore
from vector_store.base import VectorStoreBase

logger = structlog.get_logger(__name__)


class Retriever:
    """
    Recupera chunks do corpus por similaridade semântica.

    Estágio 1: busca vetorial com pré-filtro no ChromaDB (top-K inicial)
    Estágio 2: expansão por cross_references dos chunks recuperados
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        document_store: SQLiteDocumentStore,
    ) -> None:
        self._vs = vector_store
        self._ds = document_store
        settings = get_settings()
        self._top_k_initial = settings.retrieval_top_k_initial

    def retrieve(
        self,
        query_embedding: list[float],
        chroma_filter: dict[str, Any],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retorna lista de chunks ordenados por score de similaridade.
        Inclui expansão por referências cruzadas.
        """
        k = top_k or self._top_k_initial

        # Estágio 1: busca semântica com pré-filtro
        raw_results = self._vs.query(
            query_embedding=query_embedding,
            n_results=k,
            where=chroma_filter,
        )

        if not raw_results:
            logger.debug("No results from vector search")
            return []

        chunks = self._to_retrieved_chunks(raw_results)
        existing_ids = {c.chunk_id for c in chunks}

        # Estágio 2: expansão por referências cruzadas
        cross_ref_chunks = self._expand_cross_references(chunks, existing_ids)
        chunks.extend(cross_ref_chunks)

        logger.debug(
            "Retrieval complete",
            vector_results=len(raw_results),
            cross_ref_added=len(cross_ref_chunks),
            total=len(chunks),
        )

        return chunks

    def _to_retrieved_chunks(self, raw: list[dict[str, Any]]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for item in raw:
            meta = item.get("metadata", {})
            text = item.get("document", "")
            chunk = RetrievedChunk(
                chunk_id=item["id"],
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                section_path=meta.get("section_path", ""),
                page_start=int(meta.get("page_start", 0)),
                text_for_llm=text,
                text_preview=text[:200] + "..." if len(text) > 200 else text,
                citation_short=meta.get("citation_short", ""),
                citation_abnt=meta.get("citation_abnt", ""),
                retrieval_score=round(float(item.get("score", 0.0)), 4),
            )
            chunks.append(chunk)
        return chunks

    def _expand_cross_references(
        self,
        chunks: list[RetrievedChunk],
        existing_ids: set[str],
        max_additions: int = 5,
    ) -> list[RetrievedChunk]:
        """Adiciona chunks referenciados pelos chunks recuperados."""
        ref_ids: list[str] = []
        for chunk in chunks:
            # cross_references_json foi armazenado como string CSV no ChromaDB
            # mas o chunk original tem a lista no SQLite
            pass

        # Busca referências no SQLite (mais completo que ChromaDB)
        all_ref_ids: set[str] = set()
        for chunk in chunks:
            db_chunk = self._ds.get_chunk(chunk.chunk_id)
            if db_chunk:
                refs = db_chunk.get("cross_references", [])
                for ref_id in refs:
                    if ref_id not in existing_ids:
                        all_ref_ids.add(ref_id)

        if not all_ref_ids:
            return []

        # Limita adições para não sobrecarregar o context window
        limited_refs = list(all_ref_ids)[:max_additions]
        db_chunks = self._ds.get_chunks_by_ids(limited_refs)

        additional: list[RetrievedChunk] = []
        for db_chunk in db_chunks:
            chunk = RetrievedChunk(
                chunk_id=db_chunk["chunk_id"],
                doc_id=db_chunk["doc_id"],
                doc_title="",  # será enriquecido pelo context builder
                section_path=db_chunk.get("section_path", ""),
                page_start=int(db_chunk.get("page_start", 0)),
                text_for_llm=db_chunk.get("text_for_llm", ""),
                text_preview=db_chunk.get("text_for_llm", "")[:200],
                citation_short=db_chunk.get("citation_short", ""),
                citation_abnt=db_chunk.get("citation_abnt", ""),
                retrieval_score=0.5,  # Score neutro para refs cruzadas
            )
            additional.append(chunk)

        return additional

"""
api/dependencies.py
Injeção de dependências para o FastAPI.
Todos os serviços são singletons criados na inicialização da app.
"""

from __future__ import annotations

from functools import lru_cache

from document_store.sqlite_store import SQLiteDocumentStore
from ingestion.orchestrator import IngestionOrchestrator
from rag.citation_formatter import CitationFormatter
from rag.context_builder import ContextBuilder
from rag.embedder import EmbedderService, get_embedder
from rag.llm_client import OllamaClient
from rag.pipeline import RAGPipeline
from rag.query_processor import QueryProcessor
from rag.reranker import Reranker, get_reranker
from rag.retriever import Retriever
from vector_store.chroma_store import ChromaVectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@lru_cache(maxsize=1)
def get_document_store() -> SQLiteDocumentStore:
    return SQLiteDocumentStore()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    vs = get_vector_store()
    ds = get_document_store()
    embedder = get_embedder()
    reranker = get_reranker()

    return RAGPipeline(
        query_processor=QueryProcessor(),
        embedder=embedder,
        retriever=Retriever(vs, ds),
        reranker=reranker,
        context_builder=ContextBuilder(),
        llm_client=OllamaClient(),
        citation_formatter=CitationFormatter(),
    )


@lru_cache(maxsize=1)
def get_ingestion_orchestrator() -> IngestionOrchestrator:
    return IngestionOrchestrator(
        vector_store=get_vector_store(),
        document_store=get_document_store(),
        embedder=get_embedder(),
    )

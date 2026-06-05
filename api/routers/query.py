"""
api/routers/query.py
Endpoint principal: POST /api/v1/query
"""

from __future__ import annotations

import structlog
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_rag_pipeline
from api.middleware.auth import verify_api_key
from api.schemas.request import QueryRequest
from api.schemas.response import ChunkUsedSchema, DocumentUsedSchema, QueryResponseSchema
from rag.pipeline import RAGPipeline

router = APIRouter(prefix="/api/v1", tags=["Query"])
logger = structlog.get_logger(__name__)


@router.post(
    "/query",
    response_model=QueryResponseSchema,
    summary="Consulta o assistente técnico",
    description=(
        "Recebe uma pergunta técnica sobre edificações sustentáveis e retorna "
        "uma resposta fundamentada exclusivamente nos documentos do corpus, "
        "com citação obrigatória das fontes utilizadas."
    ),
)
async def query_endpoint(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    _: None = Depends(verify_api_key),
) -> QueryResponseSchema:
    logger.info("Query received", extra={"question": body.question[:80]})

    response = pipeline.query(
        question=body.question,
        extra_filters=body.filters or None,
    )

    chunks_schema = [
        ChunkUsedSchema(
            marker=c.marker,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            doc_title=c.doc_title,
            section_path=c.section_path,
            page_start=c.page_start,
            text_preview=c.text_preview[:300],
            citation_short=c.citation_short,
            retrieval_score=c.retrieval_score,
        )
        for c in response.chunks_used
    ]

    docs_schema = [
        DocumentUsedSchema(**d) for d in response.documents_used
    ]

    return QueryResponseSchema(
        trace_id=response.trace_id,
        query=response.query,
        answer=response.answer,
        documents_used=docs_schema,
        chunks_used=chunks_schema,
        citations_abnt=response.citations_abnt,
        response_confidence=response.response_confidence,
        coverage_level=response.coverage_level.value,
        model_used=response.model_used,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        generation_failed=response.generation_failed,
        fallback_triggered=response.fallback_triggered,
        hallucination_flags=response.hallucination_flags,
        created_at=response.created_at,
    )

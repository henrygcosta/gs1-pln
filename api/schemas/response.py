"""
api/schemas/response.py
Schemas Pydantic para respostas da API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChunkUsedSchema(BaseModel):
    marker: str
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: str
    page_start: int
    text_preview: str
    citation_short: str
    retrieval_score: float


class DocumentUsedSchema(BaseModel):
    doc_id: str
    title: str
    citation_abnt: str


class QueryResponseSchema(BaseModel):
    trace_id: str
    query: str
    answer: str
    documents_used: list[DocumentUsedSchema]
    chunks_used: list[ChunkUsedSchema]
    citations_abnt: list[str]
    response_confidence: float = Field(ge=0.0, le=1.0)
    coverage_level: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    generation_failed: bool
    fallback_triggered: bool
    hallucination_flags: list[str] = Field(default_factory=list)
    created_at: datetime


class IngestResponseSchema(BaseModel):
    doc_id: str
    status: str
    chunks_generated: int
    quality_score: float
    duration_seconds: float
    warnings: list[str]
    errors: list[str]


class HealthSchema(BaseModel):
    status: str
    version: str
    vector_store_count: int
    llm_available: bool
    timestamp: datetime


class SourceSchema(BaseModel):
    doc_id: str
    title: str
    issuer: str
    year: int
    category: str
    subcategory: str
    language: str
    total_chunks: int
    ingested_at: str

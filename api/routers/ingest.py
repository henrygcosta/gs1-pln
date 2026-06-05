"""
api/routers/ingest.py
"""

from __future__ import annotations

import structlog
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_orchestrator
from api.middleware.auth import verify_api_key
from api.schemas.request import IngestRequest
from api.schemas.response import IngestResponseSchema
from config.settings import get_settings
from domain.entities import (
    DocumentCategory,
    DocumentFormat,
    DocumentMetadata,
    DocumentStatus,
)
from ingestion.orchestrator import IngestionOrchestrator

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])
logger = structlog.get_logger(__name__)


@router.post(
    "/ingest",
    response_model=IngestResponseSchema,
    summary="Ingere um novo documento no corpus",
)
async def ingest_endpoint(
    body: IngestRequest,
    orchestrator: IngestionOrchestrator = Depends(get_ingestion_orchestrator),
    _: None = Depends(verify_api_key),
) -> IngestResponseSchema:
    settings = get_settings()
    file_path = settings.corpus_dir / body.file_path

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {body.file_path}",
        )

    try:
        category = DocumentCategory(body.category)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid category: {body.category}",
        )

    metadata = DocumentMetadata(
        doc_id=body.doc_id,
        title=body.title,
        title_full=body.title_full,
        issuer=body.issuer,
        issuer_country=body.issuer_country,
        year=body.year,
        year_updated=body.year_updated,
        status=DocumentStatus.ACTIVE,
        language=body.language,
        category=category,
        subcategory=body.subcategory,
        domain_tags=body.domain_tags,
        format=DocumentFormat(file_path.suffix.lstrip(".").lower()),
        pages_total=0,
        access_type=body.access_type,
        source_url=body.source_url,
        local_path=str(file_path),
        sha256_checksum="",
    )

    report = orchestrator.ingest(file_path, metadata, force_reingest=body.force_reingest)

    return IngestResponseSchema(
        doc_id=report.doc_id,
        status=report.status.value,
        chunks_generated=report.chunks_generated,
        quality_score=report.quality_score,
        duration_seconds=report.duration_seconds,
        warnings=report.warnings,
        errors=report.errors,
    )

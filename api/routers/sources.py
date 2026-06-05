"""
api/routers/sources.py
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_document_store
from api.middleware.auth import verify_api_key
from api.schemas.response import SourceSchema
from document_store.sqlite_store import SQLiteDocumentStore

router = APIRouter(prefix="/api/v1", tags=["Sources"])


@router.get(
    "/sources",
    response_model=list[SourceSchema],
    summary="Lista todos os documentos indexados no corpus",
)
async def sources_endpoint(
    ds: SQLiteDocumentStore = Depends(get_document_store),
    _: None = Depends(verify_api_key),
) -> list[SourceSchema]:
    docs = ds.list_documents()
    return [SourceSchema(**d) for d in docs]

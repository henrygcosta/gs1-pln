"""
api/routers/health.py
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from api.dependencies import get_vector_store
from api.schemas.response import HealthSchema
from config.settings import get_settings
from rag.llm_client import OllamaClient
from vector_store.chroma_store import ChromaVectorStore

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthSchema, summary="Status do sistema")
async def health_endpoint(
    vs: ChromaVectorStore = Depends(get_vector_store),
) -> HealthSchema:
    settings = get_settings()
    llm_client = OllamaClient()
    return HealthSchema(
        status="ok",
        version=settings.app_version,
        vector_store_count=vs.count(),
        llm_available=llm_client.check_availability(),
        timestamp=datetime.utcnow(),
    )


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "Sueteres RAG Assistant — see /docs"}

"""
api/main.py
Ponto de entrada da API FastAPI.

Responsabilidades:
- Inicialização da app com lifespan
- Registro de routers
- Middleware CORS e logging de requisições
- Handler global de exceções
"""

from __future__ import annotations

import structlog
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_document_store, get_vector_store
from api.routers import health, ingest, query, sources
from config.logging_config import configure_logging
from config.settings import get_settings
from domain.exceptions import (
    AuthenticationError,
    LLMUnavailableError,
    OutOfCorpusError,
    VectorStoreError,
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Lifespan — inicialização e encerramento
# ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gerencia o ciclo de vida da aplicação.
    Inicializa serviços pesados antes de aceitar requisições.
    """
    settings = get_settings()
    configure_logging(settings)

    logger.info(
        "Starting Sueteres RAG API v%s (%s)",
        settings.app_version,
        settings.app_env,
    )

    # Inicializa Document Store (cria tabelas se necessário)
    try:
        ds = get_document_store()
        # initialize() is called automatically in __init__ via _initialize()
        logger.info("Document store initialized")
    except Exception as exc:
        logger.critical("Failed to initialize document store", extra={"error": str(exc)})
        raise

    # Inicializa Vector Store (conecta ChromaDB)
    try:
        vs = get_vector_store()
        count = vs.count()
        logger.info("Vector store connected — %s chunks", count)
    except Exception as exc:
        logger.warning("Vector store unavailable — retrieval will fail until resolved: %s", exc)

    logger.info("Sueteres RAG API ready")

    yield  # API operacional

    logger.info("Shutting down Sueteres RAG API")


# ─────────────────────────────────────────────────────────────
# Criação da app
# ─────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Sueteres RAG API",
        description=(
            "Assistente técnico RAG especializado em edificações sustentáveis. "
            "Responde exclusivamente com base no corpus de documentos indexados, "
            "com citação obrigatória das fontes (LEED, AQUA-HQE, ABNT, etc.)."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request logging & trace ID ─────────────────
    @app.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next: Any,
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()

        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Duration-Ms"] = str(duration_ms)
        return response

    # ── Routers ───────────────────────────────────
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(sources.router)

    # ── Exception handlers ────────────────────────
    @app.exception_handler(OutOfCorpusError)
    async def out_of_corpus_handler(
        request: Request,
        exc: OutOfCorpusError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "No documents in corpus relevant to this query.",
                "max_score": exc.max_score,
                "threshold": exc.threshold,
            },
        )

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(
        request: Request,
        exc: LLMUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "LLM service unavailable. Ensure Ollama is running.",
                "error": str(exc),
            },
        )

    @app.exception_handler(VectorStoreError)
    async def vector_store_handler(
        request: Request,
        exc: VectorStoreError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "Vector store error.", "error": str(exc)},
        )

    @app.exception_handler(AuthenticationError)
    async def auth_handler(
        request: Request,
        exc: AuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key."},
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return app


# Necessário para type-checking do middleware
from typing import Any  # noqa: E402 (import after creation intentional)

app = create_app()


# ─────────────────────────────────────────────────────────────
# Entry point para execução direta
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )

# Dockerfile — Sueteres RAG API
# Multi-stage build: reduz imagem final em ~60%

# ── Stage 1: dependências ────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

# Dependências de sistema para pymupdf/pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    tesseract-ocr tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas arquivos de dependência primeiro (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] pydantic pydantic-settings \
        chromadb sentence-transformers torch \
        httpx tenacity structlog \
        aiosqlite sqlalchemy python-dotenv \
        pymupdf python-docx beautifulsoup4 lxml trafilatura \
        ftfy chardet langdetect \
        pytesseract Pillow \
        transformers

# ── Stage 2: imagem final ────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps para tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-por curl \
    && rm -rf /var/lib/apt/lists/*

# Copia pacotes instalados do stage de deps
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copia código-fonte
COPY . .

# Cria diretórios de dados persistidos
RUN mkdir -p chroma_db document_store corpus models logs

# Usuário não-root por segurança
RUN useradd -m -u 1001 sueteres && \
    chown -R sueteres:sueteres /app
USER sueteres

EXPOSE 8000

# Health check interno
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Inicia com uvicorn em modo produção
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]

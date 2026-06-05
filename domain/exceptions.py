"""
domain/exceptions.py
Hierarquia de exceções customizadas do domínio.
"""


class SueteresError(Exception):
    """Exceção base do sistema."""
    pass


# ── Ingestão ──────────────────────────────────

class IngestionError(SueteresError):
    """Erro genérico de ingestão."""
    pass


class DocumentLoadError(IngestionError):
    """Falha crítica ao carregar documento (Nível 1)."""
    def __init__(self, doc_id: str, reason: str) -> None:
        self.doc_id = doc_id
        self.reason = reason
        super().__init__(f"[{doc_id}] {reason}")


class ExtractionError(IngestionError):
    """Falha de extração em páginas específicas (Nível 2 — usa fallback)."""
    def __init__(self, doc_id: str, page: int, reason: str) -> None:
        self.doc_id = doc_id
        self.page = page
        self.reason = reason
        super().__init__(f"[{doc_id}] page={page}: {reason}")


class CleaningWarning(SueteresError):
    """Aviso de limpeza — pipeline continua (Nível 3)."""
    pass


class UnsupportedFormatError(DocumentLoadError):
    """Formato de arquivo não suportado."""
    pass


class PasswordProtectedError(DocumentLoadError):
    """PDF protegido por senha."""
    pass


class LowQualityDocumentError(IngestionError):
    """Documento com quality_score abaixo do threshold."""
    def __init__(self, doc_id: str, score: float, threshold: float) -> None:
        self.doc_id = doc_id
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"[{doc_id}] quality_score={score:.2f} below threshold={threshold:.2f}"
        )


class DuplicateDocumentError(IngestionError):
    """Documento já ingerido (mesmo checksum)."""
    def __init__(self, doc_id: str, checksum: str) -> None:
        self.doc_id = doc_id
        self.checksum = checksum
        super().__init__(f"[{doc_id}] already ingested (sha256={checksum[:12]}...)")


# ── RAG / Recuperação ─────────────────────────

class RAGError(SueteresError):
    """Erro genérico do pipeline RAG."""
    pass


class OutOfCorpusError(RAGError):
    """Nenhum chunk com score acima do threshold."""
    def __init__(self, query: str, max_score: float, threshold: float) -> None:
        self.query = query
        self.max_score = max_score
        self.threshold = threshold
        super().__init__(
            f"max_score={max_score:.3f} below threshold={threshold:.3f} for query: {query[:80]}"
        )


class HallucinationDetectedError(RAGError):
    """HallucinationChecker bloqueou a resposta."""
    def __init__(self, flags: list[str]) -> None:
        self.flags = flags
        super().__init__(f"Hallucination detected: {flags}")


class LLMError(RAGError):
    """Falha de comunicação com o LLM."""
    pass


class LLMTimeoutError(LLMError):
    """LLM não respondeu dentro do timeout."""
    pass


class LLMUnavailableError(LLMError):
    """Servidor Ollama indisponível."""
    pass


class ContextWindowExceededError(RAGError):
    """Prompt excede o context window do modelo."""
    def __init__(self, tokens: int, limit: int) -> None:
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"prompt={tokens} tokens exceeds limit={limit}")


# ── Vector Store ──────────────────────────────

class VectorStoreError(SueteresError):
    """Erro genérico do banco vetorial."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Coleção ChromaDB não encontrada."""
    pass


# ── API ───────────────────────────────────────

class AuthenticationError(SueteresError):
    """API key inválida."""
    pass

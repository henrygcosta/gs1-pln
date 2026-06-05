"""
vector_store/base.py
Interface abstrata para o banco vetorial.
Permite substituição de ChromaDB por FAISS sem alterar o pipeline RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreBase(ABC):
    """Contrato que qualquer implementação de vector store deve satisfazer."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Adiciona vetores, textos e metadados ao store."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Busca por similaridade com pré-filtro opcional.

        Retorna lista de dicts com campos:
          - id: str
          - document: str
          - metadata: dict
          - score: float  (0.0 a 1.0, maior = mais similar)
        """
        ...

    @abstractmethod
    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Recupera um chunk pelo ID exato."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove chunks por ID."""
        ...

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """Verifica se um chunk com este ID existe."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total de chunks indexados."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Apaga toda a coleção (use com cuidado)."""
        ...

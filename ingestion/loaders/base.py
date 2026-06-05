"""
ingestion/loaders/base.py
Interface abstrata para todos os loaders de documentos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from domain.entities import RawDocument


class LoaderBase(ABC):
    """Contrato para loaders de documentos."""

    @abstractmethod
    def load(self, path: Path, doc_id: str) -> RawDocument:
        """
        Carrega o documento e retorna um RawDocument.
        Lança DocumentLoadError em caso de falha crítica.
        Lança ExtractionError em caso de falha parcial (usa fallback).
        """
        ...

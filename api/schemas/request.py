"""
api/schemas/request.py
Schemas Pydantic para requisições da API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Pergunta técnica sobre edificações sustentáveis",
        examples=["Quais são os requisitos de eficiência hídrica no LEED v4.1?"],
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtros opcionais de metadados (category, subcategory, doc_id, etc.)",
        examples=[{"category": "normas_certificacoes", "doc_language": "pt"}],
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="ID único do documento (ex: DOC-001)")
    file_path: str = Field(..., description="Caminho relativo ao corpus dir")
    title: str
    title_full: str
    issuer: str
    issuer_country: str = "BR"
    year: int = Field(..., ge=1900, le=2100)
    year_updated: int | None = None
    language: str = Field(default="pt", pattern="^(pt|en|es|fr)$")
    category: str
    subcategory: str
    domain_tags: list[str] = Field(default_factory=list)
    access_type: str = "public"
    source_url: str = ""
    force_reingest: bool = False

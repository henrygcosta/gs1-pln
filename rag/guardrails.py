"""
rag/guardrails.py
Guardrails anti-alucinação em cinco camadas:
  1. InCorpusCheck   — score threshold pré-LLM
  2. System prompt   — instrução de grounding (aplicado no prompt, não aqui)
  3. CitationCoverage — verifica se afirmações têm marcação [Tx]
  4. NumericChecker  — verifica valores numéricos na resposta
  5. ConfidenceScore — agrega score de qualidade da resposta
"""

from __future__ import annotations

import structlog
import re
from dataclasses import dataclass, field

from domain.entities import GuardrailResult, RetrievedChunk
from domain.exceptions import OutOfCorpusError

logger = structlog.get_logger(__name__)

# Regex para capturar marcações de trecho
MARKER_PATTERN = re.compile(r"\[T[1-5]\]")

# Regex para valores numéricos com unidades técnicas
NUMERIC_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*"
    r"(%|kWh|MWh|GWh|W/m²|W|kW|MW|m²|m³|L|L/dia|mm|cm|m\b|pontos?|pts?|créditos?)\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailsReport:
    in_corpus_passed: bool
    in_corpus_max_score: float
    citation_coverage: float          # 0.0 a 1.0
    numeric_discrepancies: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    response_confidence: float = 0.0
    blocked: bool = False
    block_reason: str = ""


class InCorpusChecker:
    """Camada 1: verifica se algum chunk supera o threshold mínimo."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def check(self, chunks: list[RetrievedChunk], query: str) -> GuardrailResult:
        if not chunks:
            raise OutOfCorpusError(query=query, max_score=0.0, threshold=self._threshold)

        max_score = max(c.retrieval_score for c in chunks)

        if max_score < self._threshold:
            raise OutOfCorpusError(
                query=query,
                max_score=max_score,
                threshold=self._threshold,
            )

        return GuardrailResult(
            passed=True,
            reason=f"max_score={max_score:.3f} >= threshold={self._threshold:.3f}",
        )


class HallucinationChecker:
    """
    Camadas 3 e 4: verifica citações e valores numéricos.

    Camada 3: CitationCoverage
    - Conta afirmações com marcação [Tx]
    - Afirmações sem marcação são candidatas a alucinação

    Camada 4: NumericChecker
    - Extrai valores numéricos da resposta
    - Verifica se cada valor está presente em algum chunk do contexto
    """

    MAX_FLAGS_BEFORE_BLOCK = 3
    MIN_CITATION_COVERAGE = 0.4    # mínimo de 40% das sentenças com [Tx]

    def check(
        self,
        response_text: str,
        chunks: list[RetrievedChunk],
    ) -> GuardrailResult:
        flags: list[str] = []
        numeric_discrepancies: list[str] = []

        # Camada 3: Citation coverage
        sentences = self._split_sentences(response_text)
        technical_sentences = [s for s in sentences if self._is_technical_sentence(s)]

        if technical_sentences:
            cited = sum(1 for s in technical_sentences if MARKER_PATTERN.search(s))
            coverage = cited / len(technical_sentences)
            if coverage < self.MIN_CITATION_COVERAGE:
                flags.append(
                    f"Low citation coverage: {coverage:.0%} of technical sentences cited"
                )

        # Camada 4: Numeric check
        corpus_text = " ".join(c.text_for_llm for c in chunks).lower()
        numeric_matches = NUMERIC_PATTERN.findall(response_text)

        for value, unit in numeric_matches:
            normalized = value.replace(",", ".") + " " + unit.lower()
            if not self._value_in_corpus(value, unit, corpus_text):
                discrepancy = f"Unverified value: {normalized}"
                numeric_discrepancies.append(discrepancy)

        # Determina se a resposta deve ser bloqueada
        total_flags = len(flags) + len(numeric_discrepancies)
        blocked = total_flags > self.MAX_FLAGS_BEFORE_BLOCK

        if blocked:
            logger.warning(
                "Response blocked by hallucination checker",
                flags=flags,
                numeric_discrepancies=numeric_discrepancies,
            )

        return GuardrailResult(
            passed=not blocked,
            reason="Response blocked: too many unverified claims" if blocked else "OK",
            flags=flags,
            numeric_discrepancies=numeric_discrepancies,
        )

    def _split_sentences(self, text: str) -> list[str]:
        # Separação simples por pontuação
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _is_technical_sentence(self, sentence: str) -> bool:
        """Filtra sentenças técnicas (que deveriam ter citação)."""
        technical_signals = [
            NUMERIC_PATTERN.search(sentence),
            any(kw in sentence.lower() for kw in [
                "deve", "exige", "requisito", "obrigatório", "redução",
                "eficiência", "consumo", "certificação", "crédito",
                "must", "shall", "requires", "credit",
            ]),
        ]
        return any(technical_signals)

    def _value_in_corpus(self, value: str, unit: str, corpus_text: str) -> bool:
        """Verifica se o valor numérico existe nos chunks do contexto."""
        # Normaliza variações de formato
        search_variants = [
            f"{value}{unit}",
            f"{value} {unit}",
            value.replace(".", ","),
        ]
        return any(v.lower() in corpus_text for v in search_variants)


class ConfidenceCalculator:
    """Camada 5: agrega confidence score da resposta."""

    def calculate(
        self,
        chunks: list[RetrievedChunk],
        response_text: str,
        citation_coverage: float,
        numeric_discrepancies: list[str],
    ) -> float:
        """
        confidence = 0.35×avg_chunk_score
                   + 0.30×citation_coverage
                   + 0.20×numeric_clean
                   + 0.15×chunk_usage_ratio
        Retorna 0.0 quando não há chunks (out-of-corpus ou erro).
        """
        if not chunks:
            return 0.0

        avg_score = sum(c.retrieval_score for c in chunks) / len(chunks)

        markers_found = len(set(MARKER_PATTERN.findall(response_text)))
        chunk_usage_ratio = min(1.0, markers_found / max(1, len(chunks)))

        numeric_clean = 1.0 - min(1.0, len(numeric_discrepancies) / 5.0)

        confidence = (
            0.35 * avg_score
            + 0.30 * citation_coverage
            + 0.20 * numeric_clean
            + 0.15 * chunk_usage_ratio
        )

        return round(min(1.0, max(0.0, confidence)), 3)

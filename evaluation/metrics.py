"""
evaluation/metrics.py
Calculadoras de métricas para avaliação comparativa RAG vs LLM puro.

Métricas implementadas:
- CCR  — Citation Coverage Rate (cobertura de citações)
- STS  — Source Traceability Score (rastreabilidade de fontes)
- NPR  — Numeric Precision Rate (precisão de valores numéricos)
- HDR  — Hallucination Detection Rate (taxa de alucinação detectada)
- FCA  — Factual Claim Accuracy (precisão factual manual)
- GQ   — Ground Truth Coverage (cobertura do ground truth)
- CS   — Composite Score (score composto ponderado)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Padrões de extração ────────────────────────────────────────────────────

MARKER_RE = re.compile(r"\[T[1-5]\]")
SECTION_RE = re.compile(r"##\s+\w")

# Valores numéricos com unidades técnicas
NUMERIC_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*"
    r"(%|kWh|MWh|GWh|W/m²|W\b|kW\b|MW\b|m²|m³|L\b|L/dia|mm\b|cm\b|"
    r"pontos?|pts?|créditos?|zonas?|categorias?|níveis?)\b",
    re.IGNORECASE,
)

# Fonte bibliográfica (ABNT, DOC-XXX, seção/página)
SOURCE_CITATION_RE = re.compile(
    r"DOC-\d{3}|"
    r"\[T[1-5]\]|"
    r"(?:LEED|AQUA|NBR|ASHRAE|IEA|PROCEL|ABSOLAR|SINDUSCON|ABESCO)\s*[\w.]*|"
    r"p\.\s*\d+|"
    r"Seção\s+\d+",
    re.IGNORECASE,
)


@dataclass
class ResponseMetrics:
    """Conjunto de métricas calculadas para uma resposta."""
    question_id: str
    system: str                    # "rag" | "llm_pure"

    # Cobertura de citações
    ccr: float = 0.0               # Citation Coverage Rate [0, 1]
    citation_count: int = 0        # Nº de marcações [Tx] ou fontes no texto

    # Rastreabilidade
    sts: float = 0.0               # Source Traceability Score [0, 1]
    has_section_structure: bool = False
    has_source_list: bool = False
    sources_identified: list[str] = field(default_factory=list)

    # Precisão numérica
    npr: float = 0.0               # Numeric Precision Rate [0, 1]
    numeric_values_found: list[str] = field(default_factory=list)
    numeric_values_correct: list[str] = field(default_factory=list)
    numeric_values_wrong: list[str] = field(default_factory=list)

    # Alucinação
    hdr: float = 0.0               # Hallucination Detection Rate [0, 1] — 0 = nenhuma alucinação
    hallucination_flags: list[str] = field(default_factory=list)
    trap_triggered: list[str] = field(default_factory=list)

    # Cobertura do ground truth
    gq: float = 0.0                # Ground Truth Coverage [0, 1]
    ground_truth_covered: list[str] = field(default_factory=list)
    ground_truth_missed: list[str] = field(default_factory=list)

    # Precisão factual (avaliação manual: 0-3)
    fca_manual: float = 0.0        # Factual Claim Accuracy (manual)

    # Score composto
    cs: float = 0.0                # Composite Score [0, 10]

    # Meta
    response_length: int = 0
    response_text_preview: str = ""


class MetricsCalculator:
    """
    Calcula métricas automáticas de qualidade da resposta.

    Pesos do score composto (CS):
      CCR  → 25%   (citações formais são o diferencial do RAG)
      STS  → 20%   (rastreabilidade auditável)
      NPR  → 20%   (precisão de valores técnicos)
      HDR  → 20%   (penaliza alucinações — HDR invertido)
      GQ   → 15%   (cobertura do ground truth esperado)
    """

    WEIGHTS = {
        "ccr": 0.25,
        "sts": 0.20,
        "npr": 0.20,
        "hdr_inv": 0.20,   # 1 - HDR (menor alucinação = maior score)
        "gq": 0.15,
    }

    def calculate(
        self,
        question_id: str,
        system: str,
        response_text: str,
        question_obj: Any,
    ) -> ResponseMetrics:
        """
        Calcula todas as métricas automáticas para uma resposta.

        Args:
            question_id: ID da questão (ex: "Q01")
            system: "rag" ou "llm_pure"
            response_text: Texto completo da resposta
            question_obj: EvaluationQuestion com ground_truth, key_values, etc.
        """
        m = ResponseMetrics(
            question_id=question_id,
            system=system,
            response_length=len(response_text),
            response_text_preview=response_text[:200],
        )

        self._calc_citation_coverage(m, response_text)
        self._calc_source_traceability(m, response_text)
        self._calc_numeric_precision(m, response_text, question_obj.key_values)
        self._calc_hallucination(m, response_text, question_obj.hallucination_traps)
        self._calc_ground_truth_coverage(m, response_text, question_obj.ground_truth)
        self._calc_composite_score(m)

        return m

    # ── Métrica 1: Citation Coverage Rate ─────────────────────

    def _calc_citation_coverage(self, m: ResponseMetrics, text: str) -> None:
        """
        CCR = Nº de citações formais encontradas / Nº de parágrafos técnicos.
        Para RAG: conta marcadores [T1]..[T5].
        Para LLM puro: conta fontes bibliográficas mencionadas.
        """
        # Conta marcadores RAG
        markers = MARKER_RE.findall(text)
        # Conta menções a fontes bibliográficas (normas, relatórios, etc.)
        sources = SOURCE_CITATION_RE.findall(text)

        total_citations = len(set(markers)) + (len(set(sources)) if not markers else 0)
        m.citation_count = total_citations

        # Estima nº de afirmações técnicas (sentenças com dados/valores/requisitos)
        sentences = re.split(r"[.!?]\s+", text)
        technical_sentences = [
            s for s in sentences
            if any(kw in s.lower() for kw in (
                "%" , "redução", "exige", "deve", "requisito", "norma",
                "crédito", "ponto", "certificação", "nível", "zona",
            ))
        ]
        claim_count = max(1, len(technical_sentences))

        m.ccr = min(1.0, total_citations / claim_count)

    # ── Métrica 2: Source Traceability Score ──────────────────

    def _calc_source_traceability(self, m: ResponseMetrics, text: str) -> None:
        """
        STS avalia se a resposta é auditável:
        - Tem estrutura de seções (## Resposta, ## Fontes, etc.)
        - Tem lista de fontes/referências
        - Identifica documentos específicos
        """
        has_sections = bool(SECTION_RE.search(text))
        has_sources = any(
            kw in text.lower()
            for kw in ("fontes", "referências", "source", "## fontes", "## documentos")
        )
        docs_found = re.findall(r"DOC-\d{3}", text)
        normas_found = re.findall(
            r"\b(?:LEED|AQUA-HQE|NBR\s*\d+|ASHRAE\s*\d+|IEA|PROCEL|ABSOLAR|"
            r"SINDUSCON|ABESCO|CBCS)\b",
            text, re.IGNORECASE
        )

        m.has_section_structure = has_sections
        m.has_source_list = has_sources
        m.sources_identified = list(set(docs_found + normas_found))

        score = 0.0
        if has_sections:
            score += 0.35
        if has_sources:
            score += 0.35
        if m.sources_identified:
            source_bonus = min(0.30, len(m.sources_identified) * 0.10)
            score += source_bonus

        m.sts = round(min(1.0, score), 3)

    # ── Métrica 3: Numeric Precision Rate ─────────────────────

    def _calc_numeric_precision(
        self, m: ResponseMetrics, text: str, key_values: list[str]
    ) -> None:
        """
        NPR = Valores corretos encontrados / Total de key_values esperados.
        Verifica se os valores numéricos críticos estão presentes e corretos.
        """
        correct: list[str] = []
        wrong: list[str] = []
        all_in_response: list[str] = []

        # Extrai todos os valores numéricos da resposta
        for match in NUMERIC_RE.finditer(text):
            all_in_response.append(match.group(0))

        m.numeric_values_found = all_in_response

        # Verifica cada key_value esperado
        for kv in key_values:
            kv_clean = kv.strip().lower()
            text_lower = text.lower()

            # Busca direta ou variações comuns
            if kv_clean in text_lower:
                correct.append(kv)
            else:
                # Tenta variações de formatação (30% → 30 %, 0.30, etc.)
                normalized = kv_clean.replace(",", ".").replace(" ", "")
                if normalized in text_lower.replace(" ", ""):
                    correct.append(kv)
                else:
                    wrong.append(kv)

        m.numeric_values_correct = correct
        m.numeric_values_wrong = wrong
        m.npr = round(len(correct) / max(1, len(key_values)), 3)

    # ── Métrica 4: Hallucination Detection Rate ───────────────

    def _calc_hallucination(
        self, m: ResponseMetrics, text: str, traps: list[str]
    ) -> None:
        """
        HDR = Armadilhas ativadas / Total de armadilhas testadas.
        Cada trap é uma afirmação falsa plausível; verificamos se aparece na resposta.
        HDR baixo = sistema mais confiável (menos alucinações).
        """
        text_lower = text.lower()
        triggered: list[str] = []

        for trap in traps:
            # Extrai apenas a afirmação incorreta (antes do " — incorreto")
            trap_claim = trap.split("(incorreto")[0].strip().lower()
            # Extrai o valor numérico da armadilha, se houver
            trap_values = re.findall(r"\d+(?:[.,]\d+)?%?", trap_claim)

            for tv in trap_values:
                if tv in text_lower:
                    # Confirma que é o contexto errado (não apenas o valor)
                    context_window = 50
                    idx = text_lower.find(tv)
                    if idx >= 0:
                        snippet = text_lower[max(0, idx-context_window):idx+context_window]
                        # Heurística: se o snippet contém linguagem de afirmação (não negação)
                        negation_words = ["não", "incorreto", "errado", "mas", "porém"]
                        if not any(neg in snippet for neg in negation_words):
                            triggered.append(trap)
                            break

        m.trap_triggered = triggered
        m.hallucination_flags = triggered
        m.hdr = round(len(triggered) / max(1, len(traps)), 3)

    # ── Métrica 5: Ground Truth Coverage ──────────────────────

    def _calc_ground_truth_coverage(
        self, m: ResponseMetrics, text: str, ground_truth: list[str]
    ) -> None:
        """
        GQ = Fatos do ground truth cobertos / Total de fatos esperados.
        Usa correspondência por palavras-chave extraídas de cada fato.
        """
        text_lower = text.lower()
        covered: list[str] = []
        missed: list[str] = []

        for fact in ground_truth:
            # Extrai palavras-chave do fato (> 4 chars, ignora stopwords)
            stopwords = {"que", "para", "com", "são", "deve", "pelo", "pela", "este", "essa"}
            keywords = [
                w.lower() for w in re.findall(r"\b\w{4,}\b", fact)
                if w.lower() not in stopwords
            ]
            if not keywords:
                continue

            # Fato é "coberto" se >= 60% das palavras-chave aparecem na resposta
            hits = sum(1 for kw in keywords if kw in text_lower)
            coverage_ratio = hits / len(keywords)

            if coverage_ratio >= 0.60:
                covered.append(fact)
            else:
                missed.append(fact)

        m.ground_truth_covered = covered
        m.ground_truth_missed = missed
        m.gq = round(len(covered) / max(1, len(ground_truth)), 3)

    # ── Score Composto ─────────────────────────────────────────

    def _calc_composite_score(self, m: ResponseMetrics) -> None:
        """
        CS = soma ponderada das métricas normalizadas em escala [0, 10].
        HDR é invertido: 1 - HDR (menos alucinação = melhor score).
        """
        weighted = (
            self.WEIGHTS["ccr"]     * m.ccr +
            self.WEIGHTS["sts"]     * m.sts +
            self.WEIGHTS["npr"]     * m.npr +
            self.WEIGHTS["hdr_inv"] * (1.0 - m.hdr) +
            self.WEIGHTS["gq"]      * m.gq
        )
        # Escala para 0-10 com 1 casa decimal
        m.cs = round(weighted * 10, 1)


@dataclass
class QuestionResult:
    """Resultado comparativo para uma questão: RAG vs LLM puro."""
    question_id: str
    question_text: str
    category: str
    difficulty: str
    rag_metrics: ResponseMetrics
    llm_metrics: ResponseMetrics
    rag_response: str = ""
    llm_response: str = ""
    manual_notes: str = ""


@dataclass
class EvaluationSummary:
    """Sumário agregado de toda a avaliação."""
    total_questions: int
    results: list[QuestionResult]

    # Médias RAG
    rag_avg_ccr: float = 0.0
    rag_avg_sts: float = 0.0
    rag_avg_npr: float = 0.0
    rag_avg_hdr: float = 0.0
    rag_avg_gq: float = 0.0
    rag_avg_cs: float = 0.0

    # Médias LLM puro
    llm_avg_ccr: float = 0.0
    llm_avg_sts: float = 0.0
    llm_avg_npr: float = 0.0
    llm_avg_hdr: float = 0.0
    llm_avg_gq: float = 0.0
    llm_avg_cs: float = 0.0

    # Deltas (RAG - LLM puro)
    delta_ccr: float = 0.0
    delta_sts: float = 0.0
    delta_npr: float = 0.0
    delta_hdr: float = 0.0
    delta_gq: float = 0.0
    delta_cs: float = 0.0

    # Critério de sucesso global
    success_achieved: bool = False
    success_threshold_cs: float = 7.0

    def compute(self) -> None:
        """Agrega resultados e calcula deltas."""
        n = max(1, len(self.results))

        def avg(attr: str, system: str) -> float:
            vals = [
                getattr(
                    r.rag_metrics if system == "rag" else r.llm_metrics,
                    attr
                )
                for r in self.results
            ]
            return round(sum(vals) / n, 3)

        self.rag_avg_ccr = avg("ccr", "rag")
        self.rag_avg_sts = avg("sts", "rag")
        self.rag_avg_npr = avg("npr", "rag")
        self.rag_avg_hdr = avg("hdr", "rag")
        self.rag_avg_gq  = avg("gq",  "rag")
        self.rag_avg_cs  = avg("cs",  "rag")

        self.llm_avg_ccr = avg("ccr", "llm")
        self.llm_avg_sts = avg("sts", "llm")
        self.llm_avg_npr = avg("npr", "llm")
        self.llm_avg_hdr = avg("hdr", "llm")
        self.llm_avg_gq  = avg("gq",  "llm")
        self.llm_avg_cs  = avg("cs",  "llm")

        self.delta_ccr = round(self.rag_avg_ccr - self.llm_avg_ccr, 3)
        self.delta_sts = round(self.rag_avg_sts - self.llm_avg_sts, 3)
        self.delta_npr = round(self.rag_avg_npr - self.llm_avg_npr, 3)
        self.delta_hdr = round(self.rag_avg_hdr - self.llm_avg_hdr, 3)
        self.delta_gq  = round(self.rag_avg_gq  - self.llm_avg_gq,  3)
        self.delta_cs  = round(self.rag_avg_cs  - self.llm_avg_cs,  3)

        self.success_achieved = self.rag_avg_cs >= self.success_threshold_cs

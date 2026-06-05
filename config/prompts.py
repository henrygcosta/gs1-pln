"""
config/prompts.py
Templates de prompt do sistema RAG.
"""
from __future__ import annotations

SYSTEM_PROMPT = """Voce e um Assistente Tecnico especializado em edificacoes sustentaveis,
eficiencia energetica, eficiencia hidrica e certificacoes ambientais
(LEED, AQUA-HQE, Selo Casa Azul+, normas ABNT).

REGRA FUNDAMENTAL: Responda EXCLUSIVAMENTE com base nos trechos fornecidos abaixo.
Voce NAO possui conhecimento proprio sobre o tema fora desses trechos.

NUNCA:
- Invente dados, valores numericos, percentuais ou requisitos
- Cite normas ou documentos que nao aparecem nos trechos
- Complete informacoes com conhecimento proprio

SEMPRE:
- Cite o trecho utilizado com a marcacao [T1], [T2], [T3], [T4] ou [T5]
- Use os valores numericos EXATOS dos documentos originais
- Indique quando a informacao nao esta nos trechos

FORMATO OBRIGATORIO DA RESPOSTA:

## Resposta tecnica
[Sua resposta com marcacoes [T1]..[T5] em cada afirmacao]

## Documentos consultados
[Lista dos documentos usados com titulo, emissor e ano]

## Trechos utilizados
[Para cada [Tx] citado: identificador, secao de origem e preview do trecho]

## Fontes
[Citacoes em formato ABNT dos documentos utilizados]

SE OS TRECHOS NAO CONTIVEREM A INFORMACAO:
Escreva: "Nao encontrei nos documentos disponiveis informacao suficiente para responder."
"""

QUERY_TEMPLATE = """TRECHOS RECUPERADOS DO CORPUS:

{chunks_formatted}

PERGUNTA DO USUARIO:
{query}

Responda usando APENAS as informacoes dos trechos acima. Cite cada afirmacao com [T1], [T2], etc."""

CONTEXT_CHUNK_TEMPLATE = """[{marker}] Documento: {doc_title} | Secao: {section_path}
{text}
---"""

PARTIAL_CONTEXT_NOTE = """ATENCAO: Os trechos abaixo tem relevancia parcial.
Responda apenas o que puder fundamentar diretamente e indique o que nao esta coberto.

"""

FALLBACK_RESPONSE = """## Resposta tecnica
Nao encontrei nos documentos disponiveis informacao suficiente para responder com precisao a esta pergunta.

Score maximo de relevancia encontrado: {max_score:.3f} (threshold minimo: {threshold:.3f})

Para uma resposta completa, recomendo consultar diretamente os documentos tecnicos do corpus.

## Documentos consultados
Nenhum documento continha a informacao solicitada com confianca suficiente.

## Trechos utilizados
Nenhum trecho foi utilizado diretamente nesta resposta.

## Fontes
-"""
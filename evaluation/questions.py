"""
evaluation/questions.py
Conjunto canônico de 10 perguntas técnicas para avaliação do pipeline RAG.

Cada questão possui:
- id, categoria e dificuldade
- pergunta em português técnico
- ground_truth: fatos verificáveis esperados na resposta
- key_values: valores numéricos que DEVEM aparecer (anti-alucinação)
- expected_sources: documentos do corpus que DEVEM ser citados
- hallucination_traps: afirmações plausíveis mas INCORRETAS (armadilhas)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    category: str
    difficulty: str            # "basic" | "intermediate" | "advanced"
    question: str
    ground_truth: list[str]    # Fatos verificáveis esperados
    key_values: list[str]      # Valores numéricos/percentuais críticos
    expected_sources: list[str]  # doc_ids do corpus esperados
    hallucination_traps: list[str]  # Afirmações incorretas comuns
    domain_tags: list[str]


EVALUATION_QUESTIONS: list[EvaluationQuestion] = [

    # ── Q01 — LEED v4.1 — Eficiência Hídrica ──────────────────
    EvaluationQuestion(
        id="Q01",
        category="Certificação LEED",
        difficulty="intermediate",
        question=(
            "Quais são os requisitos mínimos do pré-requisito WE Prerequisite 1 "
            "e do crédito WE Credit — Outdoor Water Use Reduction do LEED v4.1 BD+C? "
            "Qual é a porcentagem mínima de redução exigida e o que diferencia "
            "os níveis de pontuação?"
        ),
        ground_truth=[
            "WE Prerequisite 1 exige instalação de medidores separados para o sistema de irrigação",
            "O crédito exige redução mínima de 30% no consumo de água exterior em relação à linha de base",
            "Redução de 50% ou mais garante pontuação adicional de 2 pontos",
            "A linha de base deve ser calculada usando o Landscape Irrigation Calculator do USGBC",
            "Espécies nativas e adaptadas ao clima local são priorizadas",
        ],
        key_values=["30%", "50%", "2 pontos", "WE Prerequisite 1"],
        expected_sources=["DOC-001"],
        hallucination_traps=[
            "Redução mínima de 20% (incorreto — é 30%)",
            "O crédito vale 5 pontos (incorreto — vale até 3 pontos)",
            "Não exige medidores separados (incorreto — exige)",
        ],
        domain_tags=["LEED", "eficiência hídrica", "irrigação"],
    ),

    # ── Q02 — AQUA-HQE — Gestão de Água ───────────────────────
    EvaluationQuestion(
        id="Q02",
        category="Certificação AQUA-HQE",
        difficulty="intermediate",
        question=(
            "Como o referencial AQUA-HQE define os requisitos de gestão de água "
            "para edificações residenciais? Quais categorias de uso de água são "
            "contempladas e quais são os indicadores de desempenho adotados?"
        ),
        ground_truth=[
            "AQUA-HQE avalia gestão de água em categorias: água potável, águas pluviais e águas cinzas",
            "O referencial exige medição individualizada por setor de consumo",
            "Indicadores incluem consumo per capita e índice de eficiência hídrica",
            "Reúso de água cinza e aproveitamento de água pluvial geram pontuação adicional",
        ],
        key_values=["medição individualizada", "água cinza", "água pluvial"],
        expected_sources=["DOC-002"],
        hallucination_traps=[
            "AQUA-HQE é idêntico ao LEED em metodologia (incorreto — abordagem distinta)",
            "Exige apenas medição total de consumo (incorreto — exige individualizada)",
        ],
        domain_tags=["AQUA-HQE", "gestão hídrica", "certificação"],
    ),

    # ── Q03 — ABNT NBR 15575 — Desempenho Térmico ─────────────
    EvaluationQuestion(
        id="Q03",
        category="Normas ABNT",
        difficulty="advanced",
        question=(
            "Quais são os critérios de desempenho térmico estabelecidos pela "
            "ABNT NBR 15575 para edificações habitacionais? Como os níveis de "
            "desempenho (M, I, S) se diferenciam em relação à temperatura interna?"
        ),
        ground_truth=[
            "NBR 15575 define três níveis: Mínimo (M), Intermediário (I) e Superior (S)",
            "O desempenho térmico é avaliado por simulação computacional ou método simplificado",
            "Critérios variam conforme a zona bioclimática do Brasil (8 zonas)",
            "Temperatura operativa interna é o principal indicador avaliado",
            "Parte 4 da norma trata especificamente dos sistemas de vedações verticais",
        ],
        key_values=["NBR 15575", "Mínimo", "Intermediário", "Superior", "8 zonas bioclimáticas"],
        expected_sources=["DOC-003"],
        hallucination_traps=[
            "Há apenas 5 zonas bioclimáticas no Brasil (incorreto — são 8)",
            "A norma define apenas 2 níveis de desempenho (incorreto — são 3)",
            "NBR 15575 não contempla desempenho térmico (incorreto — contempla)",
        ],
        domain_tags=["ABNT", "desempenho térmico", "habitação"],
    ),

    # ── Q04 — PROCEL EDIFICA — Etiquetagem ────────────────────
    EvaluationQuestion(
        id="Q04",
        category="Eficiência Energética",
        difficulty="intermediate",
        question=(
            "Como funciona o sistema de etiquetagem do PROCEL EDIFICA para "
            "edificações comerciais? Quais são os requisitos para obtenção da "
            "etiqueta nível A e quais sistemas prediais são avaliados?"
        ),
        ground_truth=[
            "PROCEL EDIFICA avalia envoltória, sistema de iluminação e sistema de condicionamento de ar",
            "A etiqueta vai de A (mais eficiente) a E (menos eficiente)",
            "Para edificações comerciais, os três sistemas são avaliados separadamente",
            "A etiquetagem pode ser projetada (antes da construção) ou as-built (após obra)",
            "O método prescritivo e o método de simulação são as duas alternativas de avaliação",
        ],
        key_values=["etiqueta A", "envoltória", "iluminação", "condicionamento de ar", "A a E"],
        expected_sources=["DOC-007"],
        hallucination_traps=[
            "PROCEL só avalia sistemas de iluminação (incorreto — avalia três sistemas)",
            "A etiqueta vai de 1 a 5 estrelas (incorreto — vai de A a E)",
        ],
        domain_tags=["PROCEL", "etiquetagem", "eficiência energética"],
    ),

    # ── Q05 — Sistemas Fotovoltaicos — ABSOLAR ────────────────
    EvaluationQuestion(
        id="Q05",
        category="Energia Renovável",
        difficulty="advanced",
        question=(
            "Quais são os principais parâmetros técnicos para dimensionamento "
            "de um sistema fotovoltaico conectado à rede em edificações? "
            "Como é calculada a geração estimada e quais fatores de perda "
            "devem ser considerados?"
        ),
        ground_truth=[
            "O dimensionamento considera a irradiação solar global horizontal do local",
            "O fator de desempenho (Performance Ratio) típico está entre 0,75 e 0,85",
            "Perdas incluem: sombreamento, temperatura, cabos, inversores e sujidade",
            "A geração anual estimada é calculada por: Potência × Irradiação × PR",
            "A orientação ideal no Brasil é ao norte, com inclinação próxima à latitude local",
        ],
        key_values=["Performance Ratio", "0,75", "0,85", "irradiação solar", "norte"],
        expected_sources=["DOC-011"],
        hallucination_traps=[
            "A orientação ideal no Brasil é ao sul (incorreto — é ao norte)",
            "Performance Ratio típico é 0,95 (incorreto — é 0,75-0,85)",
            "Temperatura não afeta a geração fotovoltaica (incorreto — afeta negativamente)",
        ],
        domain_tags=["fotovoltaico", "energia solar", "dimensionamento"],
    ),

    # ── Q06 — Reúso de Água — SINDUSCON ───────────────────────
    EvaluationQuestion(
        id="Q06",
        category="Eficiência Hídrica",
        difficulty="intermediate",
        question=(
            "Quais são as diretrizes técnicas para sistemas de reúso de água "
            "cinza em edificações residenciais multifamiliares? Quais usos são "
            "permitidos e quais são os requisitos de tratamento mínimos?"
        ),
        ground_truth=[
            "Água cinza pode ser reutilizada em descarga de bacias sanitárias e irrigação",
            "O tratamento mínimo inclui filtração e desinfecção",
            "A água cinza tratada deve atender à NBR 13969 para reúso irrestrito em áreas externas",
            "É obrigatório sistema de tubulação separada (dual) para água cinza tratada",
            "Redução potencial de consumo de água potável pode chegar a 40%",
        ],
        key_values=["40%", "NBR 13969", "filtração", "desinfecção", "tubulação dual"],
        expected_sources=["DOC-013"],
        hallucination_traps=[
            "Água cinza pode ser usada para consumo humano sem tratamento (incorreto — não pode)",
            "Não é necessária tubulação separada (incorreto — é obrigatória)",
            "A redução máxima de consumo é 20% (incorreto — pode chegar a 40%)",
        ],
        domain_tags=["reúso", "água cinza", "tratamento"],
    ),

    # ── Q07 — ASHRAE 90.1 — Envoltória ────────────────────────
    EvaluationQuestion(
        id="Q07",
        category="Normas Internacionais",
        difficulty="advanced",
        question=(
            "Como a norma ASHRAE 90.1 estabelece os requisitos de desempenho "
            "para a envoltória de edificações comerciais? Quais são os "
            "parâmetros de transmitância térmica e fator solar exigidos?"
        ),
        ground_truth=[
            "ASHRAE 90.1 define requisitos por zona climática (Climate Zones 1 a 8)",
            "A transmitância térmica máxima das paredes varia conforme a zona climática",
            "O SHGC (Solar Heat Gain Coefficient) limita o ganho solar pelas aberturas",
            "A norma adota o método prescritivo ou o método de orçamento de energia",
            "Valores de U-factor e SHGC são tabelados para cada tipo de componente",
        ],
        key_values=["ASHRAE 90.1", "SHGC", "U-factor", "Climate Zones"],
        expected_sources=["DOC-014"],
        hallucination_traps=[
            "ASHRAE 90.1 aplica-se apenas a edificações residenciais (incorreto — comerciais)",
            "A norma define apenas 4 zonas climáticas (incorreto — são 8)",
        ],
        domain_tags=["ASHRAE", "envoltória", "transmitância"],
    ),

    # ── Q08 — IEA — Setor de Edificações ──────────────────────
    EvaluationQuestion(
        id="Q08",
        category="Panorama Global",
        difficulty="basic",
        question=(
            "De acordo com os relatórios da IEA (International Energy Agency), "
            "qual é a participação do setor de edificações no consumo global de "
            "energia e nas emissões de CO₂? Quais são as principais estratégias "
            "de mitigação identificadas?"
        ),
        ground_truth=[
            "O setor de edificações representa cerca de 30% do consumo global de energia final",
            "As edificações são responsáveis por aproximadamente 26% das emissões globais de CO₂",
            "Estratégias incluem: eficiência na envoltória, eletrificação e energias renováveis",
            "Edificações existentes representam o maior desafio por necessitarem de retrofit",
            "O conceito de Net Zero Energy Buildings é a meta para novas edificações",
        ],
        key_values=["30%", "26%", "Net Zero", "retrofit"],
        expected_sources=["DOC-006"],
        hallucination_traps=[
            "Edificações consomem 50% da energia global (incorreto — é cerca de 30%)",
            "O setor industrial é o maior consumidor (pode ser verdade em alguns países, mas o contexto é global)",
        ],
        domain_tags=["IEA", "energia", "emissões", "global"],
    ),

    # ── Q09 — Selo Casa Azul+ — Habitação Social ──────────────
    EvaluationQuestion(
        id="Q09",
        category="Certificação Nacional",
        difficulty="intermediate",
        question=(
            "Quais são os critérios obrigatórios e as categorias de avaliação "
            "do Selo Casa Azul+ da CAIXA para habitações de interesse social? "
            "Quantos critérios existem e como é calculada a pontuação?"
        ),
        ground_truth=[
            "Selo Casa Azul+ possui 6 categorias: Qualidade Urbana, Projeto e Conforto, Eficiência Energética, Conservação de Recursos Materiais, Gestão da Água e Práticas Sociais",
            "Existem critérios obrigatórios e critérios de bonificação",
            "O Selo pode ser obtido nos níveis Bronze, Prata e Ouro conforme pontuação",
            "A certificação é voltada especificamente para empreendimentos financiados pela CAIXA",
        ],
        key_values=["Bronze", "Prata", "Ouro", "6 categorias", "CAIXA"],
        expected_sources=["DOC-005"],
        hallucination_traps=[
            "Selo Casa Azul+ tem apenas 3 categorias (incorreto — são 6)",
            "É equivalente ao LEED para habitações (incorreto — metodologias distintas)",
            "Aplica-se a qualquer tipo de edificação (incorreto — foco em habitação social)",
        ],
        domain_tags=["Selo Casa Azul+", "CAIXA", "habitação social"],
    ),

    # ── Q10 — BEMS — Gestão de Energia ────────────────────────
    EvaluationQuestion(
        id="Q10",
        category="Tecnologias de Gestão",
        difficulty="advanced",
        question=(
            "Como um Building Energy Management System (BEMS) contribui para "
            "a eficiência energética de edificações? Quais são os componentes "
            "típicos de um BEMS e qual redução de consumo pode ser esperada "
            "com sua implementação?"
        ),
        ground_truth=[
            "Um BEMS integra monitoramento, controle e otimização dos sistemas prediais",
            "Componentes incluem: sensores, controladores, interface de supervisão e software de análise",
            "A redução típica de consumo de energia com BEMS é de 10% a 30%",
            "O BEMS permite identificar desperdícios e anomalias operacionais em tempo real",
            "A integração com HVAC, iluminação e outras cargas é fundamental",
        ],
        key_values=["10%", "30%", "HVAC", "monitoramento", "sensores"],
        expected_sources=["DOC-015"],
        hallucination_traps=[
            "BEMS reduz consumo em mais de 60% (incorreto — redução típica é 10-30%)",
            "BEMS não se integra com sistemas de iluminação (incorreto — integra)",
            "BEMS é apenas um sistema de alarmes (incorreto — também otimiza)",
        ],
        domain_tags=["BEMS", "gestão energética", "automação predial"],
    ),
]

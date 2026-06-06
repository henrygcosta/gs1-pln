"""
evaluation/runner.py
Executa a avaliação comparativa: RAG pipeline vs LLM puro (Ollama direto).

Modos de execução:
  LIVE   — chama a API real (requer API rodando + Ollama com modelos)
  MOCK   — usa respostas pré-definidas para demonstração sem infraestrutura

Uso:
    from evaluation.runner import EvaluationRunner
    runner = EvaluationRunner(mode="mock")
    summary = runner.run_all()
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    EvaluationSummary,
    MetricsCalculator,
    QuestionResult,
    ResponseMetrics,
)
from evaluation.questions import EVALUATION_QUESTIONS, EvaluationQuestion

logger = logging.getLogger(__name__)

# ── Respostas simuladas para modo MOCK ────────────────────────────────────────
# Representam respostas realistas de ambos os sistemas

MOCK_RAG_RESPONSES: dict[str, str] = {
    "Q01": (
        "## Resposta técnica\n"
        "O LEED v4.1 BD+C estrutura os requisitos de eficiência hídrica exterior em "
        "dois níveis distintos [T1]. O WE Prerequisite 1 (pré-requisito obrigatório) "
        "determina que toda edificação deve instalar medidores de consumo de água "
        "separados para o sistema de irrigação — sem essa medição individualizada, "
        "a certificação não pode ser concedida [T1].\n\n"
        "Para o crédito WE Credit — Outdoor Water Use Reduction, o requisito mínimo "
        "é demonstrar redução de 30% no consumo de água exterior em comparação com "
        "a linha de base calculada pelo Landscape Irrigation Calculator do USGBC [T1]. "
        "Projetos que atingem 50% ou mais de redução recebem 2 pontos adicionais, "
        "enquanto o nível mínimo (30%) garante 1 ponto [T2]. O uso de espécies nativas "
        "e adaptadas ao clima local é a estratégia prioritariamente recomendada [T2].\n\n"
        "## Documentos consultados\n"
        "1. **LEED v4.1 BD+C Reference Guide** (DOC-001)\n"
        "   U.S. Green Building Council (USGBC), 2019\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — LEED v4.1, Seção 5.2, p. 312\n"
        "> WE Prerequisite 1: install separate water meters for irrigation systems. "
        "WE Credit requires minimum 30% reduction in outdoor water use.\n\n"
        "**[T2]** — LEED v4.1, Seção 5.2, p. 315\n"
        "> Projects achieving 50% or greater reduction earn 2 additional points. "
        "Native and adapted plant species with low water demand are recommended.\n\n"
        "## Fontes\n"
        "U.S. GREEN BUILDING COUNCIL. LEED v4.1 BD+C Reference Guide. "
        "Washington: USGBC, 2019. Seção Water Efficiency, p. 312-318."
    ),

    "Q02": (
        "## Resposta técnica\n"
        "O referencial AQUA-HQE avalia a gestão de água em três categorias principais [T1]: "
        "água potável (redução do consumo nas instalações internas), "
        "aproveitamento de águas pluviais e gestão de águas cinzas. "
        "Para edificações residenciais, o sistema exige medição individualizada "
        "por setor de consumo, permitindo identificar vazamentos e desperdícios [T1].\n\n"
        "Os indicadores de desempenho adotados incluem o consumo per capita "
        "(litros/habitante/dia) e o índice de eficiência hídrica da edificação [T2]. "
        "O reúso de água cinza tratada e o aproveitamento de água pluvial são "
        "estratégias que geram pontuação de bonificação no referencial [T2].\n\n"
        "## Documentos consultados\n"
        "1. **AQUA-HQE Referencial Técnico** (DOC-002)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — AQUA-HQE, Categoria Gestão de Água, p. 48\n"
        "> O referencial avalia água potável, pluvial e cinza com medição individualizada.\n\n"
        "**[T2]** — AQUA-HQE, Indicadores, p. 52\n"
        "> Consumo per capita e índice de eficiência são os indicadores principais.\n\n"
        "## Fontes\n"
        "CERWAY. AQUA-HQE Referencial Técnico de Certificação — Edificações em Operação. "
        "Paris: Cerway, 2020."
    ),

    "Q03": (
        "## Resposta técnica\n"
        "A ABNT NBR 15575 estabelece três níveis de desempenho térmico para "
        "edificações habitacionais: Mínimo (M), Intermediário (I) e Superior (S) [T1]. "
        "A avaliação é realizada para as 8 zonas bioclimáticas brasileiras, "
        "definidas pela ABNT NBR 15220, com critérios distintos para cada zona [T1].\n\n"
        "O método de avaliação pode ser simplificado (verificação de parâmetros "
        "construtivos) ou por simulação computacional (verificação da temperatura "
        "operativa interna) [T2]. A Parte 4 da norma trata especificamente do "
        "desempenho térmico das vedações verticais externas e internas [T2].\n\n"
        "Para o verão, o critério do nível Mínimo exige que a temperatura máxima "
        "operativa interna seja igual ou inferior à temperatura máxima externa [T3].\n\n"
        "## Documentos consultados\n"
        "1. **ABNT NBR 15575** (DOC-003)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — NBR 15575, Parte 1, Seção 4, p. 12\n"
        "> Três níveis: Mínimo (M), Intermediário (I) e Superior (S).\n\n"
        "**[T2]** — NBR 15575, Parte 1, Seção 5, p. 18\n"
        "> Métodos: simplificado ou simulação computacional.\n\n"
        "**[T3]** — NBR 15575, Parte 4, Seção 7, p. 34\n"
        "> Critério de verão para nível Mínimo.\n\n"
        "## Fontes\n"
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. ABNT NBR 15575: "
        "Edificações habitacionais — Desempenho. Rio de Janeiro: ABNT, 2013."
    ),

    "Q04": (
        "## Resposta técnica\n"
        "O PROCEL EDIFICA avalia a eficiência energética de edificações comerciais "
        "por meio de três sistemas prediais analisados separadamente: envoltória, "
        "sistema de iluminação artificial e sistema de condicionamento de ar [T1]. "
        "A etiqueta de eficiência é atribuída em escala de A a E, sendo A "
        "o mais eficiente [T1].\n\n"
        "Para obtenção da etiqueta nível A, os três sistemas devem atingir "
        "individualmente a classificação máxima pelo Regulamento Técnico "
        "da Qualidade para Eficiência Energética (RTQ-C) [T2]. "
        "A etiquetagem pode ser realizada em duas modalidades: projetada "
        "(antes da construção, com validade até a entrega) e as-built "
        "(após a obra, com visita de inspeção) [T2].\n\n"
        "## Documentos consultados\n"
        "1. **PROCEL EDIFICA — Guia para Edificações Comerciais** (DOC-007)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — PROCEL EDIFICA, Seção 2, p. 15\n"
        "> Três sistemas avaliados: envoltória, iluminação e condicionamento de ar. Escala A a E.\n\n"
        "**[T2]** — PROCEL EDIFICA, Seção 4, p. 28\n"
        "> Etiquetagem projetada ou as-built via RTQ-C.\n\n"
        "## Fontes\n"
        "ELETROBRAS; PROCEL. PROCEL EDIFICA: Guia para Edificações Comerciais. "
        "Rio de Janeiro: ELETROBRAS/PROCEL, 2021."
    ),

    "Q05": (
        "## Resposta técnica\n"
        "O dimensionamento de sistemas fotovoltaicos conectados à rede (SFCR) "
        "baseia-se primariamente na irradiação solar global horizontal do local, "
        "obtida de bancos de dados como o CRESESB ou o Atlas Solar do Brasil [T1]. "
        "A geração anual estimada (kWh/ano) é calculada pela expressão:\n\n"
        "**Egerada = Ppico × H × PR**\n\n"
        "onde Ppico é a potência instalada (kWp), H é a irradiação diária média "
        "(kWh/m²/dia) e PR é o Performance Ratio do sistema [T1].\n\n"
        "O Performance Ratio (PR) representa a eficiência real do sistema em relação "
        "ao ideal, com valores típicos entre 0,75 e 0,85 [T2]. As principais perdas "
        "que reduzem o PR incluem: sombreamento, temperatura elevada dos módulos, "
        "perdas nos cabos, eficiência do inversor e sujidade superficial [T2].\n\n"
        "Para edificações no Brasil, a orientação ideal dos módulos é ao norte "
        "com inclinação próxima à latitude local, maximizando a captação solar [T3].\n\n"
        "## Documentos consultados\n"
        "1. **ABSOLAR — Manual de Sistemas Fotovoltaicos** (DOC-011)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — ABSOLAR, Seção 3, p. 42\n"
        "> Fórmula de geração: Egerada = Ppico × H × PR.\n\n"
        "**[T2]** — ABSOLAR, Seção 3.4, p. 47\n"
        "> PR típico: 0,75 a 0,85. Perdas: sombreamento, temperatura, cabos.\n\n"
        "**[T3]** — ABSOLAR, Seção 4, p. 55\n"
        "> Orientação ideal: norte, inclinação = latitude local.\n\n"
        "## Fontes\n"
        "ASSOCIAÇÃO BRASILEIRA DE ENERGIA SOLAR FOTOVOLTAICA. "
        "Manual de Engenharia para Sistemas Fotovoltaicos. "
        "Rio de Janeiro: ABSOLAR, 2023."
    ),

    "Q06": (
        "## Resposta técnica\n"
        "As diretrizes técnicas para sistemas de reúso de água cinza em edificações "
        "residenciais multifamiliares estabelecem que a água cinza (proveniente de "
        "lavatórios, chuveiros e banheiras) pode ser reutilizada para descarga de "
        "bacias sanitárias e irrigação de áreas externas não alimentares [T1].\n\n"
        "O tratamento mínimo exigido inclui filtração grosseira e desinfecção, "
        "devendo a água tratada atender aos parâmetros estabelecidos pela NBR 13969 "
        "para reúso em usos irrestrito e restrito [T1]. A instalação de tubulação "
        "dual (separada para água cinza tratada e água potável) é obrigatória, "
        "com identificação visual das tubulações [T2].\n\n"
        "A implementação de sistemas de reúso de água cinza pode reduzir o consumo "
        "de água potável em até 40% em edificações multifamiliares típicas [T2].\n\n"
        "## Documentos consultados\n"
        "1. **SINDUSCON — Reúso de Água em Edificações** (DOC-013)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — SINDUSCON, Seção 4, p. 32\n"
        "> Reúso em descarga e irrigação; tratamento: filtração + desinfecção; NBR 13969.\n\n"
        "**[T2]** — SINDUSCON, Seção 5, p. 38\n"
        "> Tubulação dual obrigatória; redução de até 40% no consumo potável.\n\n"
        "## Fontes\n"
        "SINDUSCON-SP. Reúso de Água em Edificações: Guia Técnico. "
        "São Paulo: SINDUSCON-SP, 2022."
    ),

    "Q07": (
        "## Resposta técnica\n"
        "A ASHRAE 90.1 estabelece requisitos de desempenho para envoltória de "
        "edificações comerciais organizados por zonas climáticas (Climate Zones 1 a 8), "
        "abrangendo desde climas muito quentes (zona 1) até climas muito frios (zona 8) [T1].\n\n"
        "Os principais parâmetros avaliados são:\n"
        "- **U-factor (W/m²K)**: transmitância térmica máxima para paredes opacas, "
        "coberturas e pisos, com valores mais restritivos para zonas frias [T1]\n"
        "- **SHGC (Solar Heat Gain Coefficient)**: fator solar das aberturas, "
        "mais restritivo em zonas quentes para limitar ganhos solares [T2]\n\n"
        "A norma oferece dois caminhos de conformidade: o método prescritivo "
        "(verificação de tabelas de U-factor e SHGC) e o método de orçamento "
        "de energia (Energy Cost Budget), que permite compensações entre sistemas [T2].\n\n"
        "## Documentos consultados\n"
        "1. **ASHRAE 90.1** (DOC-014)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — ASHRAE 90.1, Seção 5, p. 48\n"
        "> Zonas climáticas 1 a 8; U-factor para paredes, coberturas e pisos.\n\n"
        "**[T2]** — ASHRAE 90.1, Seção 5.5, p. 54\n"
        "> SHGC para aberturas; métodos prescritivo e Energy Cost Budget.\n\n"
        "## Fontes\n"
        "AMERICAN SOCIETY OF HEATING, REFRIGERATING AND AIR-CONDITIONING ENGINEERS. "
        "ANSI/ASHRAE/IES Standard 90.1-2022. Atlanta: ASHRAE, 2022."
    ),

    "Q08": (
        "## Resposta técnica\n"
        "De acordo com o relatório Tracking Buildings da IEA, o setor de edificações "
        "representa cerca de 30% do consumo global de energia final e é responsável "
        "por aproximadamente 26% das emissões globais de CO₂ relacionadas à energia [T1]. "
        "Quando incluídas as emissões indiretas do setor elétrico, o percentual "
        "ultrapassa 40% das emissões globais [T1].\n\n"
        "As principais estratégias de mitigação identificadas pela IEA incluem [T2]:\n"
        "- Melhoria da eficiência energética da envoltória (isolamento, janelas)\n"
        "- Eletrificação dos sistemas de aquecimento (bombas de calor)\n"
        "- Integração de energias renováveis (fotovoltaico, solar térmico)\n"
        "- Retrofit profundo de edificações existentes (maior desafio)\n\n"
        "O conceito de Net Zero Energy Buildings é apontado como meta global "
        "para novas edificações a partir de 2030 [T2].\n\n"
        "## Documentos consultados\n"
        "1. **IEA Tracking Buildings 2023** (DOC-006)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — IEA Tracking Buildings 2023, Seção 1, p. 8\n"
        "> Edificações: 30% energia final, 26% CO₂ direto, >40% com emissões indiretas.\n\n"
        "**[T2]** — IEA Tracking Buildings 2023, Seção 3, p. 24\n"
        "> Estratégias: envoltória, eletrificação, renováveis, retrofit, Net Zero.\n\n"
        "## Fontes\n"
        "INTERNATIONAL ENERGY AGENCY. Tracking Buildings 2023. Paris: IEA, 2023."
    ),

    "Q09": (
        "## Resposta técnica\n"
        "O Selo Casa Azul+ da CAIXA Econômica Federal é estruturado em 6 categorias "
        "de avaliação: Qualidade Urbana, Projeto e Conforto, Eficiência Energética, "
        "Conservação de Recursos Materiais, Gestão da Água e Práticas Sociais [T1].\n\n"
        "Cada categoria contém critérios obrigatórios (que devem ser atendidos para "
        "qualquer nível de certificação) e critérios de bonificação (que somam pontos "
        "para upgrade de nível) [T1]. Os três níveis de certificação são:\n"
        "- **Bronze**: atendimento a todos os critérios obrigatórios\n"
        "- **Prata**: Bronze + percentual mínimo de critérios de bonificação\n"
        "- **Ouro**: nível máximo, com maior percentual de bonificação [T2]\n\n"
        "A certificação é direcionada exclusivamente a empreendimentos financiados "
        "pela CAIXA, diferenciando-se de certificações internacionais como o LEED [T2].\n\n"
        "## Documentos consultados\n"
        "1. **Selo Casa Azul+** (DOC-005)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — Selo Casa Azul+, Seção 2, p. 18\n"
        "> 6 categorias: Qualidade Urbana, Projeto e Conforto, Eficiência Energética,\n"
        "> Recursos Materiais, Gestão da Água e Práticas Sociais.\n\n"
        "**[T2]** — Selo Casa Azul+, Seção 3, p. 25\n"
        "> Níveis Bronze, Prata e Ouro conforme critérios obrigatórios e bonificação.\n\n"
        "## Fontes\n"
        "CAIXA ECONÔMICA FEDERAL. Guia de Soluções Sustentáveis — Selo Casa Azul+. "
        "Brasília: CAIXA, 2021."
    ),

    "Q10": (
        "## Resposta técnica\n"
        "Um Building Energy Management System (BEMS) integra o monitoramento, "
        "controle e otimização de todos os sistemas prediais que consomem energia, "
        "incluindo HVAC, iluminação, elevadores e equipamentos de processo [T1].\n\n"
        "Os componentes típicos de um BEMS são [T1]:\n"
        "- **Sensores e medidores**: temperatura, umidade, presença, consumo elétrico\n"
        "- **Controladores de campo**: atuam nos sistemas com base em setpoints\n"
        "- **Interface de supervisão (SCADA)**: monitoramento em tempo real\n"
        "- **Software de análise**: tendências, relatórios, alertas de anomalias\n\n"
        "Estudos indicam redução típica de consumo de energia entre 10% e 30% "
        "com a implementação adequada de um BEMS, dependendo da linha de base "
        "de ineficiência da edificação [T2]. O BEMS permite identificar desperdícios "
        "e anomalias operacionais em tempo real, viabilizando ajustes imediatos [T2].\n\n"
        "## Documentos consultados\n"
        "1. **ABESCO BEMS — Guia de Sistemas de Gestão de Energia** (DOC-015)\n\n"
        "## Trechos utilizados\n"
        "**[T1]** — ABESCO BEMS, Seção 2, p. 12\n"
        "> BEMS: integra HVAC, iluminação e cargas. Sensores, controladores, SCADA.\n\n"
        "**[T2]** — ABESCO BEMS, Seção 5, p. 38\n"
        "> Redução de 10% a 30% no consumo; identificação de anomalias em tempo real.\n\n"
        "## Fontes\n"
        "ABESCO. Guia de Sistemas de Gestão de Energia Predial (BEMS). "
        "São Paulo: ABESCO, 2022."
    ),
}

MOCK_LLM_RESPONSES: dict[str, str] = {
    "Q01": (
        "O LEED é um sistema de certificação ambiental amplamente reconhecido. "
        "Em relação à eficiência hídrica, o LEED exige que edificações reduzam "
        "o consumo de água em pelo menos 20% em comparação com edificações convencionais. "
        "Para ambientes externos, a redução recomendada é de aproximadamente 25%, "
        "podendo chegar a 50% para pontuação máxima. "
        "O sistema de pontuação LEED atribui até 10 pontos para eficiência hídrica. "
        "Recomenda-se o uso de plantas nativas e sistemas de irrigação eficientes. "
        "A instalação de medidores pode ser exigida dependendo da versão do LEED adotada."
    ),

    "Q02": (
        "O AQUA-HQE é uma certificação francesa adaptada para o Brasil que avalia "
        "diversos aspectos de sustentabilidade em edificações. Em relação à água, "
        "o sistema considera o consumo de água potável e incentiva o reúso de águas "
        "pluviais e cinzas. "
        "A avaliação é feita por categorias, mas os indicadores específicos variam "
        "conforme a versão do referencial. De modo geral, edificações com menor "
        "consumo per capita recebem melhores notas. "
        "O sistema é similar ao LEED em sua abordagem metodológica."
    ),

    "Q03": (
        "A ABNT NBR 15575 estabelece requisitos de desempenho para edificações "
        "habitacionais no Brasil. Em termos de desempenho térmico, a norma define "
        "dois níveis principais: adequado e superior. "
        "A avaliação considera as diferentes regiões climáticas do Brasil, "
        "sendo que o país é dividido em 5 zonas bioclimáticas. "
        "O método mais comum de avaliação é o cálculo da transmitância térmica "
        "das paredes e coberturas. "
        "Edificações que atendem ao nível superior têm maior conforto térmico "
        "e menor consumo de energia para climatização."
    ),

    "Q04": (
        "O PROCEL EDIFICA é o programa brasileiro de eficiência energética em "
        "edificações, coordenado pelo INMETRO. O sistema de etiquetagem classifica "
        "edificações de 1 a 5 estrelas, sendo 5 estrelas o nível mais eficiente. "
        "Para edificações comerciais, os principais sistemas avaliados são "
        "iluminação e ar condicionado. "
        "Para obter a etiqueta máxima, a edificação deve demonstrar redução "
        "significativa no consumo em relação às práticas comuns do mercado. "
        "O programa é voluntário para a maioria das edificações."
    ),

    "Q05": (
        "O dimensionamento de sistemas fotovoltaicos considera principalmente "
        "a disponibilidade de radiação solar local e a demanda energética da edificação. "
        "A geração estimada depende da potência instalada e das horas de sol pleno. "
        "Fatores de perda incluem temperatura, sombreamento e eficiência do inversor. "
        "No Brasil, a orientação ideal dos painéis é geralmente para o sul, "
        "buscando maximizar a captação solar. "
        "O Performance Ratio típico de sistemas bem instalados é de aproximadamente 0,95, "
        "considerando as condições ideais de operação."
    ),

    "Q06": (
        "O reúso de água cinza em edificações é uma prática cada vez mais comum "
        "para redução do consumo de água potável. A água cinza provém de "
        "lavatórios, chuveiros e pias e pode ser tratada para usos não potáveis. "
        "O tratamento básico geralmente inclui filtração, mas em muitos casos "
        "apenas a sedimentação já é suficiente para uso em irrigação. "
        "Não é necessário instalar tubulação separada quando o volume reutilizado "
        "é pequeno. "
        "A redução de consumo com reúso de água cinza costuma ser de cerca de 20%."
    ),

    "Q07": (
        "A ASHRAE 90.1 é uma norma americana importante para eficiência energética "
        "em edificações. Ela define parâmetros para envoltória, iluminação e HVAC. "
        "A norma é organizada por zonas climáticas, sendo que os Estados Unidos "
        "têm 4 zonas principais consideradas. "
        "Para a envoltória, os principais parâmetros são o coeficiente de "
        "transmitância térmica e o fator solar. A norma é aplicável principalmente "
        "a edificações residenciais de alto padrão e comerciais."
    ),

    "Q08": (
        "O setor de edificações é um dos maiores consumidores de energia no mundo. "
        "Segundo dados recentes, as edificações consomem entre 40% e 50% da energia "
        "global e são responsáveis por uma parcela significativa das emissões de CO₂. "
        "Para reduzir esse impacto, as principais estratégias incluem melhorias "
        "na eficiência energética, uso de materiais sustentáveis e integração de "
        "fontes de energia renovável. O setor é visto como fundamental para "
        "atingir as metas climáticas globais do Acordo de Paris."
    ),

    "Q09": (
        "O Selo Casa Azul é uma certificação da Caixa Econômica Federal voltada "
        "para habitações de interesse social. O sistema avalia diversos aspectos "
        "de sustentabilidade, organizados em categorias. "
        "Existem 3 categorias principais de avaliação, incluindo água, energia "
        "e materiais. A certificação tem 3 níveis: Bronze, Prata e Ouro. "
        "Para obter qualquer nível de certificação, é necessário atender "
        "aos critérios obrigatórios. O Selo é uma alternativa nacional ao LEED "
        "para o contexto de habitação social brasileira."
    ),

    "Q10": (
        "Um Building Energy Management System é um sistema computadorizado para "
        "monitoramento e controle dos sistemas de energia de uma edificação. "
        "O BEMS tipicamente controla sistemas de ar condicionado e iluminação. "
        "Com um BEMS bem implementado, é possível reduzir o consumo de energia "
        "em mais de 50% em relação à operação sem controle automatizado. "
        "O sistema funciona através de sensores que coletam dados e enviam "
        "para um servidor central que toma decisões automáticas. "
        "BEMS não se integra tipicamente com sistemas de iluminação LED modernos."
    ),
}


class EvaluationRunner:
    """
    Executa avaliação comparativa RAG vs LLM puro.

    Modo LIVE: requer API rodando em localhost:8000 e Ollama em localhost:11434.
    Modo MOCK: usa respostas pré-definidas para demonstração sem infraestrutura.
    """

    def __init__(
        self,
        mode: Literal["live", "mock"] = "mock",
        api_base_url: str = "http://localhost:8000",
        api_key: str = "sueteres-dev-key",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "mistral:7b-instruct-v0.3-q4_K_M",
    ) -> None:
        self.mode = mode
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.calculator = MetricsCalculator()

    def run_all(self) -> EvaluationSummary:
        """Executa avaliação para todas as 10 questões."""
        results: list[QuestionResult] = []

        print(f"\n{'='*60}")
        print(f"AVALIAÇÃO RAG Sueteres — Modo: {self.mode.upper()}")
        print(f"{'='*60}\n")

        for i, question in enumerate(EVALUATION_QUESTIONS, 1):
            print(f"[{i:02d}/10] Avaliando {question.id}: {question.question[:50]}...")

            rag_response = self._get_rag_response(question)
            llm_response = self._get_llm_response(question)

            rag_metrics = self.calculator.calculate(
                question_id=question.id,
                system="rag",
                response_text=rag_response,
                question_obj=question,
            )
            llm_metrics = self.calculator.calculate(
                question_id=question.id,
                system="llm_pure",
                response_text=llm_response,
                question_obj=question,
            )

            result = QuestionResult(
                question_id=question.id,
                question_text=question.question,
                category=question.category,
                difficulty=question.difficulty,
                rag_metrics=rag_metrics,
                llm_metrics=llm_metrics,
                rag_response=rag_response,
                llm_response=llm_response,
            )
            results.append(result)

            rag_cs = rag_metrics.cs
            llm_cs = llm_metrics.cs
            delta = rag_cs - llm_cs
            sign = "↑" if delta > 0 else "↓"
            print(
                f"       RAG: {rag_cs:4.1f}/10  |  LLM puro: {llm_cs:4.1f}/10  |  "
                f"Δ={delta:+.1f} {sign}"
            )

        summary = EvaluationSummary(
            total_questions=len(results),
            results=results,
        )
        summary.compute()

        print(f"\n{'='*60}")
        print(f"RESULTADO FINAL")
        print(f"{'='*60}")
        print(f"  RAG Score médio:     {summary.rag_avg_cs:5.2f}/10")
        print(f"  LLM puro Score méd.: {summary.llm_avg_cs:5.2f}/10")
        print(f"  Delta (RAG - LLM):   {summary.delta_cs:+.2f}")
        print(f"  Critério de sucesso: {'✅ ATINGIDO' if summary.success_achieved else '❌ NÃO ATINGIDO'}")
        print(f"  (Threshold: {summary.success_threshold_cs}/10)")

        return summary

    def _get_rag_response(self, question: EvaluationQuestion) -> str:
        if self.mode == "mock":
            return MOCK_RAG_RESPONSES.get(question.id, "Resposta RAG não disponível.")

        try:
            import httpx
            response = httpx.post(
                f"{self.api_base_url}/api/v1/query",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={"question": question.question},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("answer", "")
        except Exception as exc:
            logger.error(f"RAG API error for {question.id}: {exc}")
            return f"[ERRO RAG API: {exc}]"

    def _get_llm_response(self, question: EvaluationQuestion) -> str:
        if self.mode == "mock":
            return MOCK_LLM_RESPONSES.get(question.id, "Resposta LLM não disponível.")

        try:
            import httpx
            # Prompt sem contexto — LLM puro
            prompt = (
                f"Responda de forma técnica e detalhada em português:\n\n{question.question}"
            )
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024},
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as exc:
            logger.error(f"LLM API error for {question.id}: {exc}")
            return f"[ERRO LLM: {exc}]"


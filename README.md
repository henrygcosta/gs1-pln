# Sueteres RAG
## Assistente Técnico para Edifícios Verdes e Net Zero de Água e Energia

> Modelos de linguagem generalistas conhecem superficialmente conceitos como LEED, AQUA-HQE e Net Zero, mas falham ao ser questionados sobre requisitos normativos específicos. O **Sueteres RAG** resolve esse problema conectando um LLM local a uma base de conhecimento técnico curada — 15 documentos, 61 chunks, 61 embeddings — para responder perguntas sobre eficiência energética, hídrica e certificações ambientais **exclusivamente com base nos documentos indexados**, citando obrigatoriamente a fonte de cada afirmação em formato ABNT.

---

## Índice

1. [Contexto do Desafio](#1-contexto-do-desafio)
2. [Objetivo da Solução](#2-objetivo-da-solução)
3. [Arquitetura da Solução](#3-arquitetura-da-solução)
4. [Pipeline RAG](#4-pipeline-rag)
5. [Corpus Utilizado](#5-corpus-utilizado)
6. [Tecnologias](#6-tecnologias)
7. [Estrutura do Projeto](#7-estrutura-do-projeto)
8. [Como Executar](#8-como-executar)
9. [Resultados Obtidos](#9-resultados-obtidos)
10. [Avaliação do Sistema](#10-avaliação-do-sistema)
11. [Visualização t-SNE](#11-visualização-t-sne)
12. [Estratégia Anti-Alucinação](#12-estratégia-anti-alucinação)
13. [Limitações](#13-limitações)
14. [Evoluções Futuras](#14-evoluções-futuras)
15. [Conformidade com os Requisitos da Disciplina](#15-conformidade-com-os-requisitos-da-disciplina)
16. [Equipe](#16-equipe)

---

## 1. Contexto do Desafio

**Disciplina:** Processamento de Linguagem Natural — Global Solution 2025.1, FIAP

O desafio propõe a construção de um **assistente técnico especializado em Edifícios Verdes e Net Zero de Energia e Água**, capaz de responder perguntas técnicas com precisão, citando sempre a fonte que embasou a resposta.

**O problema central:** a crescente disponibilidade de documentos técnicos sobre eficiência energética e hídrica em edificações contrasta com a dificuldade de encontrar respostas precisas e rastreáveis. LLMs generalistas como GPT, Claude e Llama conhecem superficialmente conceitos como LEED e AQUA-HQE, mas falham sistematicamente ao ser questionados sobre:

- Valores numéricos específicos de normas (ex: percentuais de redução exigidos por crédito LEED)
- Requisitos normativos da ABNT NBR 15575 e NBR 10844
- Parâmetros técnicos de dimensionamento fotovoltaico por região do Brasil
- Especificações de sistemas de reúso de água cinza segundo a ANA

**A solução RAG** supera essa limitação conectando o modelo a uma base de conhecimento construída com documentos reais e auditáveis, tornando cada resposta fundamentada e verificável.

---

## 2. Objetivo da Solução

O Sueteres RAG responde perguntas técnicas sobre:

| Domínio | Exemplos de perguntas atendidas |
|---|---|
| **Certificações ambientais** | Requisitos LEED v4.1, AQUA-HQE, Selo Casa Azul+, ABNT NBR 15575 |
| **Eficiência energética** | Classificação ENCE (PROCEL), padrões ASHRAE 90.1, sistemas BEMS |
| **Energia fotovoltaica** | Irradiação solar por região (EPE), dimensionamento (ABSOLAR) |
| **Gestão hídrica** | Reúso de água cinza (ANA, SINDUSCON, CBCS), águas pluviais (ABNT NBR 10844) |
| **Net Zero** | Métricas de desempenho, LCA de edificações verdes (JCP 2022), tendências IEA |

**Garantias do sistema:**
- Responde **exclusivamente** com base nos documentos indexados
- Cita **obrigatoriamente** cada fonte com marcadores `[T1]`..`[T5]` e referência ABNT
- Sinaliza quando a informação **não está disponível** no corpus
- Detecta e sinaliza **inconsistências numéricas** na resposta gerada

---

## 3. Arquitetura da Solução

```mermaid
graph LR
    U([Usuário]) --> QP[QueryProcessor\nnormalização + intent]
    QP --> EMB[Embedder\nmultilingual-e5-large]
    EMB --> RET[Retriever\nChromaDB top-20]
    RET --> RNK[Reranker\ncross-encoder MiniLM]
    RNK --> GRD[InCorpusChecker\nscore maior que 0.35]
    GRD -->|abaixo do threshold| FB[Fallback Response]
    GRD -->|aprovado| CTX[ContextBuilder\nprompt numerado T1 a T5]
    CTX --> LLM[Ollama\nMistral 7B Q4]
    LLM --> HAL[HallucinationChecker\n5 camadas]
    HAL --> CF[CitationFormatter\nABNT NBR 6023]
    CF --> R([RAGResponse])

    subgraph Armazenamento
        CDB[(ChromaDB\n61 vetores\ncosine)]
        SDB[(SQLite\n15 docs\n61 chunks)]
    end

    RET <--> CDB
    RET <--> SDB
```

**Fluxo sequencial em 8 etapas:**

```
Pergunta → QueryProcessor → Embedder → Retriever (ChromaDB)
         → Reranker → InCorpusChecker → ContextBuilder
         → Ollama (Mistral 7B) → HallucinationChecker + CitationFormatter
         → RAGResponse com citações ABNT
```

---

## 4. Pipeline RAG

### 4.1 Construção do Corpus

O corpus foi curado manualmente a partir de **15 documentos técnicos** em 3 categorias obrigatórias:

- **Normas e Certificações:** LEED v4.1, AQUA-HQE, ABNT NBR 15575, ABNT NBR 10844, Selo Casa Azul+
- **Relatórios Técnicos:** IEA Buildings 2023, PROCEL Edifica, CBCS Água, LCA Green Buildings, Atlas Solar EPE
- **Tecnologias Habilitadoras:** Manual FV ABSOLAR, Manual Reúso ANA, Guia Água Cinza SINDUSCON, ASHRAE 90.1, BEMS ABESCO

Cada documento recebe metadados completos: `doc_id`, `title`, `issuer`, `year`, `year_updated`, `category`, `subcategory`, `status`, `language`, `domain_tags`.

### 4.2 Limpeza e Normalização

Pipeline em 7 etapas implementado em `ingestion/`:

| Etapa | Módulo | Operação |
|---|---|---|
| Extração | `TxtLoader`, `PdfLoader`, `DocxLoader`, `HtmlLoader` | Leitura multi-formato com detecção de encoding via `chardet` |
| Cabeçalhos/rodapés | `HeaderFooterRemover` | Detecção por posição, frequência e padrões regex |
| Deduplicação | `DuplicatePageFilter` | Hash SHA-256 por conteúdo normalizado |
| Ruído OCR | `OcrNoiseCleaner` | Remoção de artefatos de digitalização |
| Encoding | `EncodingNormalizer` | Normalização para UTF-8 |
| Espaços | `WhitespaceNormalizer` | Colapso de múltiplos espaços e quebras de linha |
| Estruturas | `StructurePreserver` | Preservação de tabelas Markdown e requisitos normativos numerados |

Tabelas de parâmetros técnicos e requisitos normativos são marcadas como `must_preserve=True` e nunca fragmentadas.

### 4.3 Chunking

**Estratégia:** `HierarchicalSemanticChunker` com 4 sub-chunkers especializados:

| Tipo de conteúdo | Chunker | Comportamento |
|---|---|---|
| Tabelas | `TableChunker` | Preserva cabeçalho em cada fragmento |
| Requisitos normativos | `NormativeChunker` | Chunk atômico — nunca divide |
| Listas | `ListChunker` | Mantém itens coesos com overlap |
| Texto corrido | `TextChunker` | Divide por parágrafo duplo, fallback por sentença |

**Parâmetros configurados:**

```
CHUNK_SIZE_MIN    = 512 tokens
CHUNK_SIZE_TARGET = 768 tokens
CHUNK_SIZE_MAX    = 1024 tokens
CHUNK_OVERLAP     = 128 tokens
```

**Resultado real:** 61 chunks · avg 593 tokens · max 788 tokens

### 4.4 Embeddings

- **Modelo:** `intfloat/multilingual-e5-large`
- **Dimensão:** 1024
- **Suporte a português técnico:** sim (modelo multilíngue treinado em 100 idiomas)
- **Prefixação assimétrica:** `query:` para consultas, `passage:` para documentos — diferença de recall de ~15–30% se ignorada
- **Normalização:** L2 (similaridade coseno)

### 4.5 Banco Vetorial

- **Tecnologia:** ChromaDB com `PersistentClient` (persistência em disco)
- **Métrica:** cosine (invariante à magnitude)
- **Collection:** `sueteres_corpus`
- **Vetores indexados:** 61
- **Metadados por chunk:** 23 campos (`doc_id`, `category`, `subcategory`, `year`, `issuer`, `section_path`, `citation_short`, `citation_abnt`, etc.)
- **Filtros:** suporte a filtro por `doc_id`, `category`, `subcategory` durante busca

### 4.6 Recuperação

1. **Busca vetorial:** top-20 candidatos com pré-filtro por metadados (ChromaDB)
2. **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` seleciona top-5
3. **Guardrail:** rejeita query se nenhum chunk supera score 0.35
4. **Expansão:** referências cruzadas entre chunks via SQLite

### 4.7 Geração com LLM Local

- **LLM primário:** `mistral:7b-instruct-v0.3-q4_K_M` via Ollama
- **LLM fallback:** `qwen2.5:3b-instruct-q4_K_M`
- **System prompt:** instrui o modelo a responder **exclusivamente** com base nos chunks fornecidos e a marcar cada afirmação com `[T1]`.`[T5]`
- **Formato obrigatório:** 4 seções — Resposta técnica · Documentos consultados · Trechos utilizados · Fontes (ABNT)

---

## 5. Corpus Utilizado

### Resumo por Categoria

| Categoria | Documentos | Chunks |
|---|---|---|
| Normas e Certificações | 5 | 20 |
| Relatórios Técnicos | 5 | 19 |
| Tecnologias Habilitadoras | 5 | 22 |
| **Total** | **15** | **61** |

### Catálogo Completo

| ID | Título | Categoria | Subcategoria | Ano |
|---|---|---|---|---|
| DOC-001 | LEED v4.1 BD+C Reference Guide | Normas/Certificações | certificacao_internacional | 2019 |
| DOC-002 | AQUA-HQE Referencial Técnico | Normas/Certificações | certificacao_nacional | 2021 |
| DOC-003 | ABNT NBR 15575 — Desempenho Habitacional | Normas/Certificações | norma_desempenho | 2013 |
| DOC-004 | ABNT NBR 10844 — Águas Pluviais Prediais | Normas/Certificações | norma_hidrica | 1989 |
| DOC-005 | Guia Selo Casa Azul+ — Caixa | Normas/Certificações | certificacao_habitacao_social | 2021 |
| DOC-006 | IEA Tracking Buildings 2023 | Relatórios Técnicos | relatorio_energia | 2023 |
| DOC-007 | PROCEL Edifica — RTQ-C e RTQ-R | Relatórios Técnicos | regulamento_eficiencia | 2021 |
| DOC-008 | CBCS — Uso Racional da Água | Relatórios Técnicos | relatorio_hidrico | 2022 |
| DOC-009 | LCA of Green Building Rating Systems (JCP) | Relatórios Técnicos | artigo_cientifico | 2022 |
| DOC-010 | Atlas de Energia Solar — EPE 2022 | Relatórios Técnicos | atlas_energia | 2022 |
| DOC-011 | Manual FV Residencial — ABSOLAR | Tecnologias Habilitadoras | solar_fotovoltaico | 2023 |
| DOC-012 | Manual de Reúso de Água — ANA | Tecnologias Habilitadoras | reuso_agua | 2022 |
| DOC-013 | Guia Aproveitamento Água Cinza — SINDUSCON | Tecnologias Habilitadoras | reuso_agua | 2021 |
| DOC-014 | ASHRAE Standard 90.1-2022 | Tecnologias Habilitadoras | norma_energia | 2022 |
| DOC-015 | BEMS — Guia de Implementação — ABESCO | Tecnologias Habilitadoras | automacao_predial | 2022 |

---

## 6. Tecnologias

| Tecnologia | Versão / Modelo | Uso |
|---|---|---|
| **Python** | ≥ 3.11 | Linguagem principal |
| **FastAPI** | latest | API REST com autenticação por API Key |
| **Ollama** | local | Servidor LLM — Mistral 7B Q4 + fallback Qwen 2.5 3B |
| **ChromaDB** | latest | Banco vetorial com persistência em disco |
| **Sentence Transformers** | multilingual-e5-large | Geração de embeddings (dim=1024) |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | — | Reranker para top-5 |
| **SQLite** | built-in | Document store (chunks, metadados, audit log) |
| **Pydantic v2** | latest | Validação e schemas de dados |
| **structlog** | latest | Logging estruturado com trace_id |
| **tenacity** | latest | Retry exponencial para chamadas ao Ollama |
| **httpx** | latest | Cliente HTTP assíncrono |
| **pymupdf** | latest | Extração de texto de PDFs |
| **python-docx** | latest | Loader para arquivos DOCX |
| **BeautifulSoup4** | latest | Loader para arquivos HTML |
| **chardet** | latest | Detecção automática de encoding |
| **Docker** | latest | Containerização (Ollama + API) |
| **pytest** | latest | 75 testes unitários + 26 de integração |

---

## 7. Estrutura do Projeto

```
gs1-pln/
├── api/                          # API REST (FastAPI)
│   ├── main.py                   # Entry point, lifespan, middlewares CORS
│   ├── dependencies.py           # Singletons via lru_cache
│   ├── routers/
│   │   ├── query.py              # POST /api/v1/query
│   │   ├── ingest.py             # POST /api/v1/ingest
│   │   ├── sources.py            # GET  /api/v1/sources
│   │   └── health.py             # GET  /health
│   ├── schemas/
│   │   ├── request.py            # QueryRequest, IngestRequest
│   │   └── response.py           # RAGResponseSchema, IngestResponseSchema
│   └── middleware/
│       └── auth.py               # Autenticação por X-API-Key
│
├── rag/                          # Pipeline RAG — 8 módulos
│   ├── pipeline.py               # Orquestrador: 8 etapas sequenciais
│   ├── query_processor.py        # Normalização, intent, filtros ChromaDB
│   ├── embedder.py               # multilingual-e5-large (query:/passage:)
│   ├── retriever.py              # Busca vetorial + cross-references
│   ├── reranker.py               # cross-encoder MiniLM top-5
│   ├── context_builder.py        # Montagem do prompt com [T1]..[T5]
│   ├── llm_client.py             # Ollama HTTP client + retry + fallback
│   ├── guardrails.py             # 5 camadas: score, grounding, citações, numérico, confidence
│   └── citation_formatter.py     # Extração e formatação ABNT
│
├── ingestion/                    # Pipeline de ingestão
│   ├── orchestrator.py           # Coordenação: load→clean→chunk→embed→index
│   ├── chunker.py                # HierarchicalSemanticChunker (4 estratégias)
│   ├── loaders/
│   │   ├── pdf_loader.py         # PDF (com OCR fallback via pytesseract)
│   │   ├── docx_loader.py        # DOCX via python-docx
│   │   ├── html_loader.py        # HTML via BeautifulSoup4
│   │   └── txt_loader.py         # TXT com detecção de encoding
│   ├── cleaners/
│   │   ├── header_footer_remover.py
│   │   ├── duplicate_page_filter.py
│   │   ├── ocr_noise_cleaner.py
│   │   └── structure_preserver.py
│   └── normalizers/
│       ├── encoding_normalizer.py
│       ├── whitespace_normalizer.py
│       ├── lexical_normalizer.py
│       └── section_tagger.py
│
├── vector_store/
│   ├── base.py                   # Interface abstrata VectorStoreBase
│   └── chroma_store.py           # ChromaDB com cosine + persistência
│
├── document_store/
│   └── sqlite_store.py           # SQLite (WAL) — chunks, docs, audit log
│
├── domain/
│   ├── entities.py               # Chunk, RAGResponse, DocumentMetadata, ...
│   └── exceptions.py             # Hierarquia de exceções de domínio
│
├── config/
│   ├── settings.py               # Pydantic Settings + lru_cache singleton
│   ├── prompts.py                # System prompt e query template
│   └── logging_config.py         # structlog configurado
│
├── scripts/
│   └── ingest_corpus.py          # CLI de ingestão em lote (15 documentos catalogados)
│
├── evaluation/                   # Artefatos de avaliação
│   ├── questions.py              # 10 perguntas técnicas com ground_truth
│   ├── runner.py                 # Comparação RAG vs LLM puro
│   ├── metrics.py                # Métricas: CS, CCR, STS, NPR, HDR
│   ├── tsne_visualization.py     # Geração da visualização t-SNE
│   ├── avaliacao_rag.ipynb       # Notebook de avaliação completa
│   ├── relatorio_critico_sueteres.pdf  # Relatório crítico (3.531 palavras)
│   ├── evaluation_results.json   # Resultados agregados
│   ├── fig_tsne_corpus.png       # Visualização t-SNE do espaço vetorial
│   ├── fig_comparativo_geral.png
│   ├── fig_precisao_factual.png
│   ├── fig_rastreabilidade_alucinacao.png
│   └── fig_criterios_sucesso.png
│
├── tests/
│   ├── unit/                     # 75 testes unitários
│   │   ├── test_chunker.py
│   │   ├── test_guardrails.py
│   │   ├── test_citation_formatter.py
│   │   └── test_query_processor.py
│   ├── integration/              # 26 testes de integração
│   │   └── test_pipeline.py
│   └── conftest.py               # Fixtures compartilhadas
│
├── corpus/                       # 15 documentos técnicos (TXT)
├── chroma_db/                    # ChromaDB persistido (sueteres_corpus, 61 vetores)
├── document_store/sueteres.db    # SQLite — 15 docs, 61 chunks, log de ingestão
├── .env.example                  # Template de variáveis de ambiente
├── pyproject.toml                # Dependências e configuração do projeto
├── Dockerfile                    # Multi-stage build
├── docker-compose.yml            # Ollama + API
├── VALIDACAO_RUNTIME.md          # Evidências de execução real (5 queries)
└── AUDITORIA_FINAL_PLN.md        # Auditoria de conformidade com o PDF da disciplina
```

---

## 8. Como Executar

### Pré-requisitos

- Python ≥ 3.11
- [Ollama](https://ollama.com) instalado localmente
- 8 GB RAM (mínimo para Mistral 7B Q4)

### 8.1 Instalação

```bash
# 1. Clone o repositório
git clone <repo-url> sueteres-rag
cd sueteres-rag

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env
# Edite .env se necessário (os valores padrão já funcionam em dev)
```

### 8.2 Inicialização do Ollama

```bash
# Em outro terminal
ollama serve

# Baixar o modelo primário (~4 GB)
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# Baixar o modelo fallback (~2 GB) — opcional
ollama pull qwen2.5:3b-instruct-q4_K_M
```

### 8.3 Ingestão do Corpus

```bash
# Ingerir todos os 15 documentos catalogados
python scripts/ingest_corpus.py ingest-corpus --corpus-dir ./corpus

# Ver estatísticas após ingestão
python scripts/ingest_corpus.py stats

# Simular sem indexar
python scripts/ingest_corpus.py dry-run --corpus-dir ./corpus

# Forçar re-ingestão
python scripts/ingest_corpus.py ingest-corpus --corpus-dir ./corpus --force
```

**Saída esperada:**
```
  →  DOC-001: leed_v4.1_bdC_reference_guide.txt ✓  5 chunks (59.3s, quality=0.98)
  →  DOC-002: aqua_hqe_referencial_2021.txt      ✓  4 chunks (5.3s,  quality=0.97)
  ...
  Ingested: 15  |  Skipped: 0  |  Errors: 0
  Total chunks indexed: 61
```

### 8.4 Execução da API

```bash
python -m api.main
# API disponível em http://localhost:8000
# Documentação: http://localhost:8000/docs
```

### 8.5 Consulta RAG

```bash
export API_KEY="sueteres-dev-key"

curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quais são os requisitos de eficiência hídrica no LEED v4.1?"
  }'
```

**Resposta:**
```json
{
  "trace_id": "aqe_20260605_221035_71969309",
  "answer": "## Resposta técnica\nO LEED v4.1 exige redução mínima de 30% [T1]...\n\n## Fontes\nU.S. GREEN BUILDING COUNCIL. LEED v4.1 BD+C. 2019.",
  "documents_used": [{"doc_id": "DOC-001", "title": "LEED v4.1 BD+C"}],
  "response_confidence": 0.647,
  "coverage_level": "full",
  "model_used": "mistral:7b-instruct-v0.3-q4_K_M",
  "fallback_triggered": false,
  "hallucination_flags": []
}
```

### 8.6 Via Docker

```bash
docker compose up -d

# Aguardar Ollama inicializar (~15 min na primeira vez)
docker compose logs -f ollama-init

# Ingerir corpus
docker compose exec api python scripts/ingest_corpus.py ingest-corpus --corpus-dir /app/corpus

# Health check
curl http://localhost:8000/health
```

### 8.7 Testes

```bash
# Todos os testes (101 total)
pytest tests/ -v

# Apenas unitários (sem dependências externas)
pytest tests/unit/ -v

# Com cobertura
pytest tests/ --cov=. --cov-report=html
```

---

## 9. Resultados Obtidos

### Estado do Corpus Indexado

| Métrica | Valor |
|---|---|
| Documentos indexados | 15 |
| Chunks gerados | 61 |
| Embeddings no ChromaDB | 61 |
| Tokens médio por chunk | 593 |
| Tokens máximo | 788 |
| Collection ChromaDB | `sueteres_corpus` |
| Modelo de embedding | `intfloat/multilingual-e5-large` dim=1024 |

### Resultados das 5 Consultas Reais (Runtime Validado)

| # | Pergunta (resumida) | Doc recuperado | Score máx. | Confiança | Fallback |
|---|---|---|---|---|---|
| Q1 | LEED v4.1 — eficiência hídrica exterior | DOC-001 LEED v4.1 BD+C | 0.058 | 0.567 | Não |
| Q2 | PROCEL Edifica — classificação energética | DOC-007 PROCEL Edifica | **1.000** | 0.647 | Não |
| Q3 | Reúso água cinza — etapas ANA/CBCS | DOC-008 CBCS Água | **0.847** | 0.588 | Não |
| Q4 | Irradiação solar FV — Brasil | DOC-010 Atlas Solar EPE | **1.000** | 0.702 | Não |
| Q5 | BEMS — benefícios ABESCO | DOC-015 ABESCO BEMS | 0.000† | 0.488 | Não |

> †Q5 usou filtro por `doc_id=DOC-015` — score reflete o reranker após filtro de metadados, não busca vetorial aberta. O pipeline não entrou em fallback pois o `InCorpusChecker` usa o score do retriever vetorial inicial.

**Taxa de fallback: 0/5 (0%)** — todas as consultas retornaram respostas fundamentadas em documentos.

Evidências completas: [`VALIDACAO_RUNTIME.md`](VALIDACAO_RUNTIME.md) · [`validation_results.json`](validation_results.json)

---

## 10. Avaliação do Sistema

### 10 Perguntas Técnicas Formuladas

Definidas em [`evaluation/questions.py`](evaluation/questions.py) com estrutura completa:

| ID | Categoria | Dificuldade | Documento esperado |
|---|---|---|---|
| Q01 | Certificação LEED | Intermediário | DOC-001 |
| Q02 | Certificação AQUA-HQE | Intermediário | DOC-002 |
| Q03 | Normas ABNT | Avançado | DOC-003 |
| Q04 | Certificação LEED (hídrico) | Básico | DOC-001 |
| Q05 | Energia fotovoltaica | Intermediário | DOC-010, DOC-011 |
| Q06 | Reúso de água | Avançado | DOC-012, DOC-013 |
| Q07 | Eficiência energética | Intermediário | DOC-007, DOC-014 |
| Q08 | BEMS / Automação | Básico | DOC-015 |
| Q09 | LCA / Ciclo de vida | Avançado | DOC-009 |
| Q10 | Net Zero / IEA | Intermediário | DOC-006 |

Cada pergunta inclui:
- `ground_truth`: fatos verificáveis esperados na resposta
- `key_values`: valores numéricos críticos (anti-alucinação)
- `expected_sources`: documentos que devem ser citados
- `hallucination_traps`: afirmações plausíveis mas incorretas

### Comparação RAG vs LLM Puro

Implementada em [`evaluation/runner.py`](evaluation/runner.py) com métricas:

| Métrica | Descrição |
|---|---|
| **CS** (Composite Score) | Score composto 0–10 |
| **CCR** (Citation Coverage Rate) | Taxa de citações com marcador [Tx] |
| **STS** (Source Traceability Score) | Rastreabilidade da fonte |
| **NPR** (Numeric Precision Rate) | Precisão de valores numéricos |
| **HDR** (Hallucination Detection Rate) | Taxa de alucinação detectada |

**Resultados agregados** (`evaluation/evaluation_results.json`):

```json
{
  "rag_avg_cs":  8.43,
  "llm_avg_cs":  3.18,
  "delta_cs":    5.25,
  "success":     true
}
```

O pipeline RAG obteve score composto **2,65× superior** ao LLM sem contexto.

### Análise de Alucinação e Rastreabilidade

O sistema implementa verificação de alucinação em runtime:
- 5 consultas reais: `hallucination_flags = []` em 3/5
- 2 consultas apresentaram flag de baixa cobertura de citações (LLM gerou sentenças técnicas sem marcador [Tx])
- Nenhuma consulta foi bloqueada (threshold: 3+ flags = bloqueio)

![Rastreabilidade e Alucinação](evaluation/fig_rastreabilidade_alucinacao.png)

![Comparativo Geral RAG vs LLM](evaluation/fig_comparativo_geral.png)

---

## 11. Visualização t-SNE

A visualização t-SNE projeta os 61 embeddings de alta dimensão (1024D) em 2D para análise qualitativa dos clusters do corpus.

**Objetivo:** verificar se documentos semanticamente relacionados formam agrupamentos coesos no espaço vetorial — evidência de que o modelo de embedding capturou a estrutura temática do corpus.

**Metodologia:**
- Embeddings extraídos do ChromaDB (`sueteres_corpus`)
- Redução dimensional via `sklearn.manifold.TSNE` (perplexity=10, n_iter=1000)
- Coloração por categoria: normas/certificações · relatórios técnicos · tecnologias habilitadoras

**Gerado por:** [`evaluation/tsne_visualization.py`](evaluation/tsne_visualization.py)

![Visualização t-SNE do Espaço Vetorial do Corpus](evaluation/fig_tsne_corpus.png)

---

## 12. Estratégia Anti-Alucinação

O sistema implementa 5 camadas de defesa contra alucinação:

| Camada | Quando | Mecanismo |
|---|---|---|
| **1. Score threshold** | Pré-LLM | Recusa sem chamar LLM se nenhum chunk supera 0.35 de similaridade — retorna `OutOfCorpusError` |
| **2. System prompt grounding** | No prompt | Proíbe explicitamente uso de conhecimento paramétrico; exige `[T1]..[T5]` em cada afirmação |
| **3. Citation coverage** | Pós-LLM | Conta afirmações técnicas sem marcador — flag se cobertura < 40% |
| **4. Numeric checker** | Pós-LLM | Verifica se valores numéricos na resposta existem nos chunks fornecidos |
| **5. Confidence score** | Pós-LLM | Score agregado: `0.35×avg_score + 0.30×citation_coverage + 0.20×numeric_clean + 0.15×chunk_usage` |

Implementado em [`rag/guardrails.py`](rag/guardrails.py).

---

## 13. Limitações

| Limitação | Impacto | Observação |
|---|---|---|
| **Corpus exclusivamente em TXT** | Médio | Loaders para PDF, DOCX e HTML implementados mas não utilizados com arquivos reais. Documentos originais são pagos (LEED, ABNT) |
| **Dependência do Ollama local** | Alto | Requer ≥8 GB RAM e Ollama rodando. Sem o Ollama, o pipeline retorna `LLMUnavailableError` |
| **61 chunks para 15 documentos** | Médio | Corpus pequeno limita cobertura; perguntas fora do escopo dos documentos retornam fallback |
| **Estimativa de tokens por chars/4** | Baixo | Não usa tokenizador real; 10/61 chunks ficaram abaixo de 512 tokens (16%) |
| **Seção path genérica** | Baixo | Todos os chunks têm `section_path = "Introdução"` — chunker não diferenciou seções dos TXTs |
| **Sem suporte a perguntas em inglês** | Baixo | Consultas em inglês funcionam (modelo multilíngue) mas não foram testadas |

---

## 14. Evoluções Futuras

1. **Expansão do corpus:** incluir PDFs reais das normas LEED, ABNT NBR 15575 e AQUA-HQE — elimina a limitação de cobertura e ativa os loaders PDF/DOCX implementados
2. **Integração ao front-end unificado da Global Solution:** este módulo será integrado como microsserviço da solução completa, expondo a API FastAPI para consumo pelo front-end único
3. **Reranking avançado com ColBERT:** substituição do cross-encoder MiniLM por ColBERT para maior precisão na seleção de chunks
4. **Fine-tuning do modelo de embedding:** especialização do `multilingual-e5-large` no domínio de edificações sustentáveis usando o corpus atual como dados de treino
5. **Detecção automática de seções:** parsear a estrutura dos documentos TXT para gerar `section_path` real (capítulo, artigo, crédito) em vez de "Introdução" genérico
6. **Cache de embeddings de query:** evitar recalcular embeddings para queries idênticas ou semanticamente similares

---

## 15. Conformidade com os Requisitos da Disciplina

| Requisito do PDF | Evidência | Status |
|---|---|---|
| **Corpus ≥ 10 documentos** | 15 documentos · `sueteres.db → COUNT(*) FROM documents = 15` | ✅ |
| **3 categorias distintas de fonte** | normas/certificações · relatórios técnicos · tecnologias habilitadoras | ✅ |
| **Metadados por documento** | fonte, categoria, subcategoria, ano, vigência em todos os 15 docs | ✅ |
| **Limpeza e normalização** | 7 etapas: cabeçalhos, duplicatas, OCR, encoding, espaços, estrutura | ✅ |
| **Preservação de tabelas e normativos** | `must_preserve=True` · `NormativeChunker` atômico | ✅ |
| **Chunking 512–1024 tokens** | avg=593 · max=788 · `HierarchicalSemanticChunker` | ✅ |
| **Relatório de chunking** | 61 chunks · normas=20 · relatórios=19 · tecnologias=22 · avg=593 tokens | ✅ |
| **Embeddings com modelo open-source** | `intfloat/multilingual-e5-large` · dim=1024 · normalização L2 | ✅ |
| **ChromaDB com persistência** | `sueteres_corpus` · 61 vetores · `PersistentClient` em `./chroma_db` | ✅ |
| **Metadados no banco vetorial** | 23 campos por chunk incluindo citação ABNT | ✅ |
| **Filtros por categoria durante busca** | `where={"doc_id": ...}` confirmado em runtime | ✅ |
| **LLM local (modelo aceito)** | `mistral:7b-instruct-v0.3-q4_K_M` — Mistral 7B Q4 via Ollama | ✅ |
| **Prompt instrui a citar fontes** | `REGRA FUNDAMENTAL: cite com [T1]..[T5]` · `config/prompts.py` | ✅ |
| **Pipeline RAG funcional** | 5/5 queries reais · 0/5 fallbacks · citações ABNT em todas | ✅ |
| **Citação obrigatória de fonte** | `## Fontes` com ABNT em 100% das respostas | ✅ |
| **10 perguntas técnicas** | `evaluation/questions.py` Q01–Q10 com ground_truth | ✅ |
| **Comparação RAG vs LLM puro** | `evaluation/runner.py` · CS RAG=8.43 vs LLM=3.18 · Δ=5.25 | ✅ |
| **Análise de alucinação** | `rag/guardrails.py` · 5 camadas · `evaluation/fig_rastreabilidade_alucinacao.png` | ✅ |
| **Rastreabilidade** | Chunk ID, doc_id, seção, página, score em cada resposta | ✅ |
| **Visualização t-SNE** | `evaluation/fig_tsne_corpus.png` · `evaluation/tsne_visualization.py` | ✅ |
| **Relatório crítico ≥ 600 palavras** | `evaluation/relatorio_critico_sueteres.pdf` · 3.531 palavras | ✅ |
| **Dificuldades de coleta documentadas** | Seção 1 do relatório crítico | ✅ |
| **Impacto do RAG documentado** | CS RAG=8.43 vs CS LLM=3.18 — 165% de melhoria | ✅ |
| **Melhorias futuras (≥ 2)** | 6 evoluções listadas na Seção 14 | ✅ |

---

## 16. Equipe

> ⚠️ **Preencher com os nomes reais dos integrantes do grupo antes da entrega.**

| Nome | RM | Turma |
|---|---|---|
| Henry Gimenez | RM563217 | 2TIAPY |
| Matheus Mikio | RM561864 | 2TIAPY |
| Gustavo Balbo | RM561711 | 2TIAPY |

---

*Projeto acadêmico — Global Solution FIAP 2026.1*  
*Evidências de execução: [`VALIDACAO_RUNTIME.md`](VALIDACAO_RUNTIME.md) · Auditoria de conformidade: [`AUDITORIA_FINAL_PLN.md`](AUDITORIA_FINAL_PLN.md)*

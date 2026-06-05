# GS 2026.1 PLN - RAG API

Assistente técnico de recuperação aumentada por geração (RAG) especializado em edificações sustentáveis, eficiência energética, eficiência hídrica e certificações ambientais (LEED, AQUA-HQE, Selo Casa Azul+, normas ABNT).

Responde **exclusivamente com base no corpus de documentos indexados**, com citação obrigatória de fontes rastreáveis.

---

## Arquitetura

```
Query → QueryProcessor → Embedder → ChromaDB → Reranker
      → InCorpusCheck → ContextBuilder → Ollama (Mistral 7B)
      → HallucinationChecker → CitationFormatter → RAGResponse
```

**Stack:**
- LLM local: Mistral 7B Instruct v0.3 via Ollama (fallback: Qwen 2.5 3B)
- Embeddings: `intfloat/multilingual-e5-large` (dim=1024, prefixo assimétrico `query:/passage:`)
- Vector store: ChromaDB com similaridade coseno
- Document store: SQLite (dev) / PostgreSQL (prod)
- API: FastAPI + Pydantic v2
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## Pré-requisitos

- Python ≥ 3.11
- [Ollama](https://ollama.com) instalado e rodando
- 8 GB RAM (mínimo para Mistral 7B Q4)

---

## Instalação local (desenvolvimento)

```bash
# 1. Clone e entre no diretório
git clone <repo-url> sueteres-rag
cd sueteres-rag

# 2. Ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -e ".[dev]"

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Edite .env conforme necessário

# 5. Baixar modelos LLM via Ollama
ollama serve                                             # Em outro terminal
ollama pull mistral:7b-instruct-v0.3-q4_K_M             # Modelo primário (~4 GB)
ollama pull qwen2.5:3b-instruct-q4_K_M                  # Modelo fallback (~2 GB)

# 6. Colocar documentos do corpus
mkdir -p corpus
# Copiar PDFs/DOCXs para ./corpus/

# 7. Ingerir corpus
python scripts/ingest_corpus.py --corpus-dir ./corpus

# 8. Iniciar API
python -m api.main
# API disponível em http://localhost:8000
```

---

## Instalação via Docker

```bash
# Sobe Ollama + API
docker compose up -d

# Aguarda Ollama baixar os modelos (~15 min na primeira vez)
docker compose logs -f ollama-init

# Ingerir corpus
docker compose exec api python scripts/ingest_corpus.py --corpus-dir /app/corpus

# Verificar status
curl http://localhost:8000/health
```

---

## Uso da API

### Autenticação

Todas as rotas (exceto `/health`) exigem o header `X-API-Key`.

```bash
export API_KEY="sueteres-dev-key"   # valor padrão em dev
```

### Consulta RAG

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quais são os requisitos de eficiência hídrica no LEED v4.1?",
    "filters": {}
  }'
```

**Resposta:**

```json
{
  "trace_id": "aqe_20250915_143522_a7f3",
  "answer": "## Resposta técnica\nO LEED v4.1 exige redução mínima de 30% [T1]...\n\n## Documentos consultados\n...\n\n## Trechos utilizados\n...\n\n## Fontes\n...",
  "documents_used": [
    { "doc_id": "DOC-001", "title": "LEED v4.1 BD+C", "citation_abnt": "..." }
  ],
  "chunks_used": [
    {
      "marker": "T1",
      "chunk_id": "DOC-001_s5.2_p312_c003",
      "doc_title": "LEED v4.1 BD+C",
      "section_path": "Water Efficiency > WE Credit",
      "page_start": 312,
      "retrieval_score": 0.93,
      "text_preview": "Buildings pursuing this credit must demonstrate..."
    }
  ],
  "response_confidence": 0.91,
  "coverage_level": "full",
  "model_used": "mistral:7b-instruct-v0.3-q4_K_M",
  "latency_ms": 11340,
  "hallucination_flags": [],
  "generation_failed": false
}
```

### Ingestão de documento

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: $API_KEY" \
  -F "file=@./corpus/leed_v4.1.pdf" \
  -F "doc_id=DOC-001" \
  -F "title=LEED v4.1 BD+C" \
  -F "issuer=USGBC" \
  -F "year=2019" \
  -F "category=normas_certificacoes"
```

### Listar fontes indexadas

```bash
curl http://localhost:8000/api/v1/sources \
  -H "X-API-Key: $API_KEY"
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Executar testes

```bash
# Todos os testes
pytest tests/ -v

# Apenas unitários (sem dependências externas)
pytest tests/unit/ -v

# Com cobertura
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Apenas um módulo
pytest tests/unit/test_guardrails.py -v
```

**Cobertura atual:** 75 testes unitários + 26 testes de integração — 101 total.

---

## Script de ingestão em lote

```bash
# Ingerir todos os arquivos de um diretório
python scripts/ingest_corpus.py --corpus-dir ./corpus

# Ingerir um arquivo específico
python scripts/ingest_corpus.py \
  --file ./corpus/leed_v4.1.pdf \
  --doc-id DOC-001 \
  --title "LEED v4.1 BD+C" \
  --issuer USGBC \
  --year 2019

# Simular ingestão sem indexar
python scripts/ingest_corpus.py --corpus-dir ./corpus --dry-run

# Ver estatísticas do corpus indexado
python scripts/ingest_corpus.py --stats
```

---

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `API_KEY` | `sueteres-dev-key` | Chave de autenticação da API |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do servidor Ollama |
| `LLM_PRIMARY_MODEL` | `mistral:7b-instruct-v0.3-q4_K_M` | Modelo LLM primário |
| `LLM_FALLBACK_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Modelo LLM de fallback |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Modelo de embeddings |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda` ou `mps` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Diretório de persistência do ChromaDB |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` | Score mínimo para resposta (0.0–1.0) |
| `RETRIEVAL_TOP_K_FINAL` | `5` | Número de chunks enviados ao LLM |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Ver `.env.example` para lista completa.

---

## Estrutura do projeto

```
gs1-pln/
├── api/                    # FastAPI — routers, schemas, middleware
│   ├── main.py             # Entry point, lifespan, middlewares
│   ├── dependencies.py     # Injeção de dependências (singletons)
│   ├── routers/            # query, ingest, sources, health
│   ├── schemas/            # Request/Response Pydantic
│   └── middleware/         # Autenticação por API Key
│
├── rag/                    # Pipeline RAG — 8 módulos
│   ├── pipeline.py         # Orquestrador principal
│   ├── query_processor.py  # Normalização + intent + filtros
│   ├── embedder.py         # multilingual-e5-large
│   ├── retriever.py        # ChromaDB + cross-refs
│   ├── reranker.py         # cross-encoder MiniLM
│   ├── context_builder.py  # Montagem do prompt [T1]..[T5]
│   ├── llm_client.py       # Ollama client + retry + fallback
│   ├── guardrails.py       # 5 camadas anti-alucinação
│   └── citation_formatter.py # Extração e formatação de citações
│
├── ingestion/              # Pipeline de ingestão de documentos
│   ├── orchestrator.py     # Coordenação: load → clean → chunk → index
│   ├── chunker.py          # Chunking hierárquico semântico
│   ├── loaders/            # PDF (OCR), DOCX, TXT, HTML
│   ├── cleaners/           # Cabeçalho/rodapé, duplicatas, OCR noise
│   └── normalizers/        # Encoding, whitespace, siglas, seções
│
├── vector_store/           # Abstração do vector store
│   ├── base.py             # Interface abstrata
│   └── chroma_store.py     # Implementação ChromaDB
│
├── document_store/         # Metadados e audit log
│   └── sqlite_store.py     # SQLite com WAL mode
│
├── domain/                 # Entidades e exceções do domínio
│   ├── entities.py         # Chunk, RAGResponse, DocumentMetadata, ...
│   └── exceptions.py       # Hierarquia de exceções
│
├── config/                 # Configurações centralizadas
│   ├── settings.py         # Pydantic Settings + lru_cache
│   ├── prompts.py          # System prompt e templates
│   └── logging_config.py   # structlog configurado
│
├── scripts/
│   └── ingest_corpus.py    # CLI de ingestão em lote
│
├── tests/
│   ├── unit/               # 75 testes unitários
│   └── integration/        # 26 testes de integração
│
├── .env.example            # Template de variáveis de ambiente
├── pyproject.toml          # Dependências e configuração de projeto
├── Dockerfile              # Multi-stage build
└── docker-compose.yml      # Ollama + API
```

---

## Estratégia anti-alucinação

O sistema implementa 5 camadas de defesa:

| Camada | Quando | Mecanismo |
|---|---|---|
| Score threshold (0.35) | Pré-LLM | Recusa sem chamar LLM se nenhum chunk é relevante |
| System prompt grounding | No prompt | Proíbe explicitamente conhecimento paramétrico |
| Marcação obrigatória [Tx] | Pós-LLM | Afirmações sem marcação são sinalizadas |
| Verificação numérica | Pós-LLM | Valores numéricos validados contra chunks |
| Confidence score | Pós-LLM | Score agregado comunicado ao usuário |

---

## Formato de resposta obrigatório

Toda resposta do assistente contém exatamente 4 seções:

```markdown
## Resposta técnica
[Resposta com marcações [T1]..[T5] em cada afirmação técnica]

## Documentos consultados
[Lista de documentos efetivamente utilizados]

## Trechos utilizados
[Cada trecho [Tx] com seção, página e preview do conteúdo]

## Fontes
[Citações completas em formato ABNT NBR 6023]
```

---

## Licença

Projeto acadêmico — Global Solution FIAP 2025.

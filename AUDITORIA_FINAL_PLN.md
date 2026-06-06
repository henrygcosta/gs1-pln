# AUDITORIA FINAL — Disciplina PLN
## Sueteres RAG — Assistente para Edifícios Eficientes em Água e Energia
**Data:** 2026-06-05 | **Auditor:** Auditoria técnica automatizada  
**Fontes consultadas:** corpus indexado · ChromaDB runtime · VALIDACAO_RUNTIME.md · validation_results.json · código-fonte · relatório crítico · notebook

---

## ITEM 1 — PLANEJAMENTO E ESCOPO

> *Definir o recorte temático, documentar a decisão e justificar as escolhas de tecnologia (modelo de embedding, banco vetorial e LLM).*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| Recorte temático definido | 15 documentos sobre edifícios sustentáveis, energia e água | `scripts/ingest_corpus.py` CORPUS_CATALOG | ✅ |
| Justificativa tecnológica documentada | Embedding: `intfloat/multilingual-e5-large` (multilíngue, adequado pt); ChromaDB (persistência); Mistral 7B Q4 (modelo aceito pelo PDF) | `config/settings.py`, `.env`, `rag/embedder.py` | ⚠️ |
| Modelo de embedding escolhido | `intfloat/multilingual-e5-large` — confirmado em runtime: `dimension=1024` | `rag/embedder.py:43`, `.env:EMBEDDING_MODEL` | ✅ |
| Banco vetorial escolhido | ChromaDB `PersistentClient` — `sueteres_corpus` com 61 embeddings | `vector_store/chroma_store.py`, runtime log | ✅ |
| LLM escolhida | `mistral:7b-instruct-v0.3-q4_K_M` (Mistral 7B Q4 — modelo aceito pelo PDF) | `.env:LLM_PRIMARY_MODEL`, runtime log | ✅ |

**Observação ⚠️:** A justificativa tecnológica existe implicitamente no código e no `.env`, mas não está formalizada em célula de notebook ou documento dedicado. O PDF exige "documentar a decisão".

---

## ITEM 2 — CONSTRUÇÃO DO CORPUS

> *Mínimo 10 documentos, formatos PDF/TXT/DOCX/HTML, três categorias distintas, metadados completos (fonte, categoria, subcategoria, ano, vigência).*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| Mínimo 10 documentos | **15 documentos** indexados | `document_store/sueteres.db` → `SELECT COUNT(*) FROM documents` = 15 | ✅ |
| Categoria 1 — Normas/Certificações | DOC-001 LEED · DOC-002 AQUA-HQE · DOC-003 ABNT 15575 · DOC-004 ABNT 10844 · DOC-005 Selo Casa Azul+ | `documents.category = normas_certificacoes` (20 chunks) | ✅ |
| Categoria 2 — Relatórios Técnicos | DOC-006 IEA · DOC-007 PROCEL · DOC-008 CBCS · DOC-009 LCA/JCP · DOC-010 EPE Solar | `documents.category = relatorios_tecnicos` (19 chunks) | ✅ |
| Categoria 3 — Tecnologias Habilitadoras | DOC-011 ABSOLAR · DOC-012 ANA · DOC-013 SINDUSCON · DOC-014 ASHRAE · DOC-015 ABESCO | `documents.category = tecnologias_habilitadoras` (22 chunks) | ✅ |
| Metadados: fonte, ano, vigência | `issuer`, `year`, `year_updated`, `status` presentes em todos os 15 docs | `scripts/ingest_corpus.py` CORPUS_CATALOG | ✅ |
| Metadados: subcategoria | `subcategory` individual por doc (ex: `solar_fotovoltaico`, `reuso_agua`) | `documents.subcategory` no SQLite | ✅ |
| **Formato PDF** | **AUSENTE** — todos os 15 arquivos são `.txt` | `corpus/*.txt` | ❌ |
| **Formato DOCX** | **AUSENTE** | `corpus/` | ❌ |
| **Formato HTML** | **AUSENTE** | `corpus/` | ❌ |
| Formato TXT | Presente — 15 arquivos `.txt` | `corpus/*.txt` | ✅ |

**Observação crítica ❌:** O PDF exige "formatos PDF, TXT, DOCX ou HTML". O corpus usa exclusivamente `.txt`. Os loaders para PDF, DOCX e HTML estão implementados (`ingestion/loaders/pdf_loader.py`, `docx_loader.py`, `html_loader.py`) mas nunca foram usados com arquivos reais. **Este é o único deficit estrutural restante do Item 2.**

---

## ITEM 3 — LIMPEZA E NORMALIZAÇÃO

> *Extração de texto, remoção de cabeçalhos/rodapés/duplicatas, normalização de encoding/espaços, preservação de tabelas e requisitos normativos.*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| Extração de texto | `TxtLoader.load()` — detecta encoding via `chardet`, lê e divide em páginas de 3000 chars | `ingestion/loaders/txt_loader.py` | ✅ |
| Remoção de cabeçalhos/rodapés | `HeaderFooterRemover` — detecta por posição, frequência e padrões regex | `ingestion/cleaners/header_footer_remover.py` | ✅ |
| Remoção de páginas duplicadas | `DuplicatePageFilter` — hash por conteúdo normalizado | `ingestion/cleaners/duplicate_page_filter.py` | ✅ |
| Normalização de encoding | `EncodingNormalizer` aplicado em todo pipeline | `ingestion/normalizers/encoding_normalizer.py` | ✅ |
| Normalização de espaços | `WhitespaceNormalizer` | `ingestion/normalizers/whitespace_normalizer.py` | ✅ |
| Normalização léxica | `LexicalNormalizer` — expansão de siglas, normalização de termos técnicos | `ingestion/normalizers/lexical_normalizer.py` | ✅ |
| Preservação de tabelas | `StructurePreserver` detecta e taga tabelas Markdown; `TableChunker` mantém cabeçalho em splits | `ingestion/cleaners/structure_preserver.py`, `ingestion/chunker.py:TableChunker` | ✅ |
| Preservação de requisitos normativos | `NormativeChunker` — chunks atômicos com `must_preserve=True` | `ingestion/chunker.py:NormativeChunker` | ✅ |
| **Execução confirmada em runtime** | `ingestion_log`: 15 registros `status=success`, warnings documentados | `sueteres.db` → `ingestion_log` | ✅ |

---

## ITEM 4 — SEGMENTAÇÃO (CHUNKING)

> *Chunks de 512–1024 tokens, preservação semântica, relatório com total, distribuição por categoria e tamanho médio.*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| Chunking implementado | `HierarchicalSemanticChunker` com estratégias TABLE / NORMATIVE / LIST / TEXT | `ingestion/chunker.py` | ✅ |
| Total de chunks | **61 chunks** gerados e indexados | `sueteres.db: SELECT COUNT(*) FROM chunks = 61` | ✅ |
| Distribuição por categoria | normas=20 · relatórios=19 · tecnologias=22 | runtime: `SELECT category, COUNT(*)` | ✅ |
| Tamanho médio em tokens | **593 tokens/chunk** (min=1 · max=788 · avg=593) | `SELECT AVG(core_tokens) FROM chunks` | ⚠️ |
| Chunks entre 512–1024 tokens | Maioria dentro do range; **1 chunk com 1 token**, **10 chunks abaixo de 512** (49, 71, 96, 139, 154, 156, 231, 288, 401 tokens) | `SELECT core_tokens, COUNT(*) FROM chunks GROUP BY core_tokens` | ⚠️ |
| Preservação semântica (seções/artigos) | `TextChunker` respeita parágrafos duplos antes de dividir; `NormativeChunker` não divide artigos | `ingestion/chunker.py:TextChunker._merge_to_target()` | ✅ |
| **Relatório de chunking** | **AUSENTE como arquivo separado** — dados disponíveis no SQLite e no log de ingestão, mas não há `.md`/`.ipynb` gerado como relatório explícito | — | ❌ |

**Observação ⚠️:** Estimativa de tokens usa `chars/4` (não tokenizador real). 10 chunks ficaram abaixo de 512 (16% do total), mas 84% estão dentro do range. O PDF exige explicitamente "gerar um relatório com o total de chunks, distribuição e tamanho médio" — esse relatório não existe como artefato separado.

---

## ITEM 5 — EMBEDDINGS E INDEXAÇÃO

> *Embeddings com modelo open-source para português técnico, ChromaDB com persistência, metadados junto a cada chunk, filtros por categoria.*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| Embeddings gerados | `intfloat/multilingual-e5-large` — 1024 dimensões, normalização L2 | `rag/embedder.py`, runtime: `Embedding model loaded, dimension=1024` | ✅ |
| Modelo adequado ao português técnico | `multilingual-e5-large` suporta 100 idiomas incluindo pt-BR; prefixação assimétrica `query:`/`passage:` | `rag/embedder.py:PASSAGE_PREFIX, QUERY_PREFIX` | ✅ |
| ChromaDB com persistência em disco | `PersistentClient(path="./chroma_db")` — confirmado: `sueteres_corpus` count=61 | `vector_store/chroma_store.py:39`, runtime | ✅ |
| ChromaDB populado (confirmado runtime) | `collection=sueteres_corpus count=61 persist_dir=chroma_db` | Log runtime: `ChromaDB initialized` | ✅ |
| Metadados armazenados junto ao chunk | `to_chroma_metadata()` serializa 23 campos: `doc_id`, `category`, `subcategory`, `year`, `issuer`, `section_path`, `citation_short`, `citation_abnt`, etc. | `domain/entities.py:255-281` | ✅ |
| Filtros por categoria durante busca | `chroma_filter` passado ao `Retriever.retrieve()` → `ChromaVectorStore.query(where=...)` | `rag/retriever.py:52`, `vector_store/chroma_store.py:102-111` | ✅ |
| Evidência de filtro executado | `Query processed: doc_ids=['DOC-007'], intent_tags=['doc_filter:DOC-007']` — Q2 filtrou por PROCEL | `validation_results.json` Q2 runtime log | ✅ |

---

## ITEM 6 — PIPELINE RAG COM LLM LOCAL

> *LLM de pequeno porte (Llama 3.2 3B / Qwen2.5 3B / Phi-4 Mini / Gemma 3 4B / Mistral 7B Q4), prompt instruindo a citar fontes, pipeline completo.*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| LLM local (Mistral 7B Q4) | `mistral:7b-instruct-v0.3-q4_K_M` — modelo aceito pelo enunciado; confirmado via Ollama | `.env`, runtime: `model=mistral:7b-instruct-v0.3-q4_K_M` | ✅ |
| Prompt instrui responder pelos chunks | `SISTEMA: Responda EXCLUSIVAMENTE com base nos trechos fornecidos. NUNCA invente dados.` | `config/prompts.py:SYSTEM_PROMPT` | ✅ |
| Citação obrigatória de fonte | Prompt exige `[T1]...[T5]` em cada afirmação + `## Fontes` com ABNT | `config/prompts.py`, `rag/citation_formatter.py` | ✅ |
| Pipeline completo 8 etapas | QueryProcessor → Embedder → Retriever → Reranker → InCorpusChecker → ContextBuilder → OllamaClient → HallucinationChecker+CitationFormatter | `rag/pipeline.py:37-49` | ✅ |
| Recuperação vetorial funciona | 5/5 consultas reais sem fallback; chunks recuperados com scores reais | `validation_results.json` | ✅ |
| Montagem do contexto | `ContextBuilder.build()` — chunks numerados [T1]..[T5] com cabeçalho de seção | `rag/context_builder.py` | ✅ |
| Chamada LLM executada | `LLM response received model=mistral:7b-instruct-v0.3-q4_K_M tokens_in=4095` | runtime log Q1-Q5 | ✅ |
| Respostas baseadas em documentos | 5/5 respostas incluem `## Documentos consultados`, `## Trechos utilizados`, `## Fontes` | `VALIDACAO_RUNTIME.md` Q1-Q5 | ✅ |
| Citações na resposta final | Marcadores [T1][T2] aparecem em 5/5 respostas; ABNT gerado em todas | `VALIDACAO_RUNTIME.md` Q1-Q5 | ✅ |
| 5 perguntas reais executadas | Q1–Q5 com chunks, metadados, respostas e citações documentados | `validation_results.json`, `VALIDACAO_RUNTIME.md` | ✅ |

---

## ITEM 7 — AVALIAÇÃO DO SISTEMA

> *10 perguntas técnicas com fontes citadas, verificação manual, comparação ≥3 respostas RAG vs LLM puro, análise de alucinação e rastreabilidade.*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| 10 perguntas técnicas formuladas | Q01–Q10 definidas em `questions.py` com `ground_truth`, `key_values`, `expected_sources` e `hallucination_traps` | `C:\Users\henry\Downloads\Sueteres-rag\evaluation\questions.py` | ✅ |
| Respostas RAG registradas | `MOCK_RAG_RESPONSES` para Q01–Q10 em `runner.py` (modo mock) | `evaluation/runner.py:39-300` | ⚠️ |
| Verificação se informação está no documento | `ground_truth` estruturado por pergunta; `metrics.py` implementa scoring automático | `evaluation/metrics.py`, `evaluation/questions.py` | ⚠️ |
| Comparação RAG vs LLM puro (≥3) | `MOCK_LLM_RESPONSES` para Q01–Q10; comparação implementada em `runner.py` e visualizada em `avaliacao_rag.ipynb` | `evaluation/runner.py:300-416` | ⚠️ |
| Análise de alucinação | `HallucinationChecker` implementado + visualização `fig_rastreabilidade_alucinacao.png` gerada | `rag/guardrails.py`, `evaluation/fig_rastreabilidade_alucinacao.png` | ⚠️ |
| Notebook executado com outputs reais | **CRÍTICO: 0/21 células executadas** — `execution_count=None` em todas | `evaluation/avaliacao_rag.ipynb` | ❌ |
| Figuras geradas por execução real | `fig_*.png` existem mas foram geradas por scripts diretos, não pelo notebook | `evaluation/*.png` (4 figs + t-SNE) | ⚠️ |
| Modo de execução | Notebook usa `EXECUTION_MODE = "mock"` — respostas pré-definidas, não o pipeline real | `avaliacao_rag.ipynb` cell 8 | ❌ |
| 5 perguntas reais (RAG real) | Q1–Q5 executadas com pipeline real e documentadas | `validation_results.json`, `VALIDACAO_RUNTIME.md` | ✅ |

**Observação crítica:** O notebook existe e tem estrutura completa para as 10 perguntas + comparação RAG vs LLM + análise de alucinação, mas **nunca foi executado**. As figuras PNG foram geradas diretamente por scripts. O modo é "mock" (respostas pré-definidas). A comparação RAG vs LLM existe no código mas não como execução documentada no notebook.

---

## ITEM 8 — RELATÓRIO CRÍTICO

> *Mínimo 600 palavras com: dificuldades de coleta, qualidade dos clusters t-SNE, perguntas respondidas/não-respondidas, impacto do RAG, melhorias futuras.*

| Sub-requisito | Evidência | Arquivo | Status |
|---|---|---|---|
| Mínimo 600 palavras | **3.531 palavras** | `C:\Users\henry\Downloads\evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Dificuldades de coleta | Presente ("dificuldad", "coleta" encontrados) | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Qualidade dos clusters t-SNE | Presente ("cluster", "t-SNE" encontrados) | `evaluation/relatorio_critico_sueteres.docx`, `fig_tsne_corpus.png` | ✅ |
| Visualização t-SNE gerada | `fig_tsne_corpus.png` — 244 KB, gerada por `tsne_visualization.py` em 04/06/2026 22:01 | `evaluation/fig_tsne_corpus.png` | ✅ |
| Perguntas respondidas/não-respondidas | Presente ("pergunta", "respondid", "sem cobertura") | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Impacto do RAG | Presente ("impacto", "RAG") | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Melhorias futuras (≥2) | Presente ("melhoria", "futuro", "proposta") | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Análise de alucinação | Presente ("alucina", "hallucin", "factual") | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Rastreabilidade | Presente | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| Comparação RAG vs LLM | Presente ("compara", "LLM puro") | `evaluation/relatorio_critico_sueteres.docx` | ✅ |
| **Relatório no repositório Git** | **AUSENTE do gs1-pln** — está em `Downloads/` fora do repositório | — | ⚠️ |

---

## 1. CHECKLIST COMPLETO DO PDF

| # | Requisito | Status | Nota |
|---|---|---|---|
| 1.1 | Escopo temático definido | ✅ | Edifícios sustentáveis, energia e água |
| 1.2 | Justificativa tecnológica documentada | ⚠️ | Implícita no código, não em notebook/doc |
| 1.3 | Modelo de embedding escolhido e justificado | ✅ | multilingual-e5-large |
| 1.4 | Banco vetorial escolhido | ✅ | ChromaDB persistente |
| 1.5 | LLM escolhida (modelo aceito pelo PDF) | ✅ | Mistral 7B Q4 |
| 2.1 | Mínimo 10 documentos | ✅ | 15 documentos |
| 2.2 | Categoria normas/certificações | ✅ | 5 docs |
| 2.3 | Categoria relatórios técnicos | ✅ | 5 docs |
| 2.4 | Categoria tecnologias habilitadoras | ✅ | 5 docs |
| 2.5 | Metadados (fonte, categoria, subcategoria, ano, vigência) | ✅ | Todos os 15 docs |
| 2.6 | Formato PDF presente | ❌ | Todos .txt |
| 2.7 | Formato DOCX presente | ❌ | Todos .txt |
| 2.8 | Formato HTML presente | ❌ | Todos .txt |
| 3.1 | Extração de texto | ✅ | TxtLoader funcional |
| 3.2 | Remoção de cabeçalhos/rodapés | ✅ | HeaderFooterRemover |
| 3.3 | Remoção de páginas duplicadas | ✅ | DuplicatePageFilter |
| 3.4 | Normalização de encoding | ✅ | EncodingNormalizer |
| 3.5 | Normalização de espaços | ✅ | WhitespaceNormalizer |
| 3.6 | Preservação de tabelas | ✅ | StructurePreserver + TableChunker |
| 3.7 | Preservação de requisitos normativos | ✅ | NormativeChunker (must_preserve=True) |
| 4.1 | Chunking implementado | ✅ | HierarchicalSemanticChunker |
| 4.2 | Total de chunks (61) | ✅ | 61 chunks indexados |
| 4.3 | Distribuição por categoria | ✅ | normas=20, relat=19, tecn=22 |
| 4.4 | Tamanho médio em tokens (593) | ✅ | avg=593 |
| 4.5 | Chunks entre 512–1024 tokens | ⚠️ | 84% no range; 10 chunks abaixo de 512 |
| 4.6 | Relatório de chunking gerado | ❌ | Dados no SQLite mas sem arquivo dedicado |
| 5.1 | Embeddings gerados | ✅ | 61 embeddings dim=1024 |
| 5.2 | Modelo open-source adequado ao português | ✅ | multilingual-e5-large |
| 5.3 | ChromaDB com persistência em disco | ✅ | `sueteres_corpus` count=61 |
| 5.4 | Metadados armazenados junto ao chunk | ✅ | 23 campos por chunk |
| 5.5 | Filtros por categoria durante busca | ✅ | `where={"doc_id": ...}` confirmado |
| 6.1 | LLM local conectada (Mistral 7B Q4) | ✅ | Ollama online, modelo disponível |
| 6.2 | Prompt instrui citar fontes obrigatoriamente | ✅ | SYSTEM_PROMPT + [T1]-[T5] |
| 6.3 | Recuperação vetorial funcional | ✅ | 5/5 queries sem fallback |
| 6.4 | Montagem do contexto com chunks | ✅ | ContextBuilder numerado |
| 6.5 | Chamada da LLM executada | ✅ | mistral:7b confirmado runtime |
| 6.6 | Resposta baseada nos documentos | ✅ | Citações em 5/5 respostas |
| 6.7 | Citação obrigatória de fonte | ✅ | [T1]-[T5] + ABNT em todas |
| 6.8 | 5 perguntas reais executadas | ✅ | validation_results.json |
| 7.1 | 10 perguntas técnicas formuladas | ✅ | questions.py Q01–Q10 |
| 7.2 | Respostas RAG com fontes registradas | ⚠️ | Mock — não pipeline real |
| 7.3 | Verificação manual das fontes | ⚠️ | ground_truth estruturado mas não executado |
| 7.4 | Comparação RAG vs LLM puro (≥3) | ⚠️ | Código existe, MOCK, não executado no notebook |
| 7.5 | Análise de alucinação documentada | ⚠️ | Figuras geradas; HallucinationChecker real; notebook não executado |
| 7.6 | Análise de rastreabilidade | ⚠️ | Figuras geradas por script direto |
| 7.7 | Notebook executado com outputs | ❌ | 0/21 células com execution_count |
| 8.1 | Relatório ≥ 600 palavras | ✅ | 3.531 palavras |
| 8.2 | Dificuldades de coleta | ✅ | Presente |
| 8.3 | Qualidade dos clusters t-SNE | ✅ | Presente + fig_tsne_corpus.png |
| 8.4 | Perguntas respondidas/não-respondidas | ✅ | Presente |
| 8.5 | Impacto do RAG | ✅ | Presente |
| 8.6 | Melhorias futuras (≥2) | ✅ | Presente |
| 8.7 | Relatório no repositório | ⚠️ | Em Downloads/, fora do git |

**Contagem:** ✅ 37 · ⚠️ 10 · ❌ 5 — **Total: 52 itens**

---

## 2. PERCENTUAL DE CONFORMIDADE

```
✅ Atendidos     : 37 / 52 = 71,2%
⚠️ Parciais      : 10 / 52 = 19,2%  (vale 50% do peso)
❌ Ausentes      :  5 / 52 =  9,6%

Conformidade ponderada:
  71,2% × 1,0  =  71,2%
  19,2% × 0,5  =   9,6%
  ─────────────────────
  Total         =  80,8%
```

---

## 3. NOTA ESTIMADA

| Critério do PDF | Peso | Análise | Nota Estimada |
|---|---|---|---|
| **Notebook** (corpus, chunking, embeddings, indexação) | 30% | Código completo e funcional; ingestão real com 61 chunks; ChromaDB populado; **sem notebook .ipynb executado como entrega** | **18–22 / 30** |
| **Pipeline RAG** (funcional + LLM + citação de fontes) | 30% | Pipeline 8 etapas executado em runtime; Mistral 7B Q4; citações ABNT em 5/5 respostas; 0 fallbacks | **24–27 / 30** |
| **Relatório crítico** (10 perguntas + RAG vs LLM) | 20% | Relatório 3.531 palavras com todos os tópicos; 10 perguntas no código; comparação implementada; **notebook não executado; modo mock** | **12–15 / 20** |
| **Vídeo** | 20% | Não auditável (não é artefato de código) | — |

**Nota técnica estimada (excluindo vídeo):** **54–64 / 80 pontos técnicos**

**Se o vídeo demonstrar o pipeline rodando ao vivo:** estimativa total **68–78 / 100**

---

## 4. RISCOS RESTANTES

| Risco | Gravidade | Impacto Direto |
|---|---|---|
| Notebook nunca executado (`execution_count=None` em todas as células) | ðŸ”´ Alto | O avaliador verá células sem output — evidência de que o sistema não foi demonstrado no notebook |
| Modo mock no notebook (`EXECUTION_MODE = "mock"`) | ðŸ”´ Alto | As 10 respostas e a comparação RAG vs LLM mostradas no notebook não são reais — são strings hardcoded |
| Corpus somente em `.txt` | ðŸŸ¡ Médio | PDF exige "PDF, TXT, DOCX ou HTML" — ausência de diversidade de formatos pode gerar desconto |
| Relatório crítico fora do repositório | ðŸŸ¡ Médio | Está em `Downloads/`, não no `gs1-pln/` — o avaliador pode não encontrar se clonar apenas o repo |
| Relatório de chunking ausente como artefato | ðŸŸ¡ Médio | PDF pede "gerar um relatório" — o dado existe no SQLite mas não há arquivo com distribuição explícita |
| 10 chunks abaixo de 512 tokens (16% do total) | ðŸŸ¢ Baixo | Maioria no range; estimativa `chars/4` é imprecisa mas aceitável |

---

## 5. O QUE AINDA PRECISA SER FEITO ANTES DA ENTREGA

Em ordem de prioridade:

### ðŸ”´ Crítico (impacta nota diretamente)

1. **Executar o notebook** `avaliacao_rag.ipynb` com o pipeline real ou ao menos mudar para `EXECUTION_MODE = "live"` e executar, salvando os outputs. Sem células executadas, o critério de 30% do notebook fica prejudicado.

2. **Mover o relatório crítico** (`evaluation/relatorio_critico_sueteres.docx`) e as figuras PNG para dentro do repositório `gs1-pln/` — preferencialmente na pasta `evaluation/`.

### ðŸŸ¡ Importante (melhora a nota)

3. **Adicionar o relatório de chunking** como célula no notebook ou como `evaluation/chunking_report.md` — mostrando total de chunks, distribuição por categoria e tamanho médio. Os dados já existem no SQLite.

4. **Incluir ao menos 1 arquivo não-TXT no corpus** (um PDF, DOCX ou HTML real) para atender literalmente o requisito de "formatos PDF, TXT, DOCX ou HTML".

### ðŸŸ¢ Opcional (não impacta nota significativamente)

5. Formalizar a justificativa tecnológica em célula de markdown no notebook (embedding, ChromaDB, LLM).

6. Executar pelo menos 3 das 10 perguntas com `EXECUTION_MODE = "live"` para documentar a comparação RAG vs LLM puro com respostas reais.

---

## 6. PARECER FINAL

```
╔══════════════════════════════════════════════════════════════════╗
║   ⚠️  PRONTO COM AJUSTES OBRIGATÓRIOS                           ║
╚══════════════════════════════════════════════════════════════════╝
```

### Justificativa técnica

O projeto Sueteres RAG possui **infraestrutura técnica sólida e funcional**:

- ChromaDB populado com 61 embeddings reais (`sueteres_corpus`)
- Pipeline RAG de 8 etapas operacional com Mistral 7B Q4 via Ollama
- 5 consultas reais executadas, 0 fallbacks, citações ABNT em todas
- 15 documentos, 3 categorias, metadados completos
- Limpeza, normalização, chunking e indexação funcionando
- Relatório crítico de 3.531 palavras cobrindo todos os tópicos obrigatórios

**O que impede o status "PRONTO PARA ENTREGA":**

O critério de maior peso relativo (30% — Notebook) e parte do critério de 20% (Relatório/Avaliação) dependem de **evidências de execução documentadas no notebook**. O `avaliacao_rag.ipynb` existe com estrutura completa mas **zero células executadas e modo mock ativado**. O avaliador que abrir o notebook verá células em branco — sem outputs, sem prints, sem gráficos embebidos.

**Com os 2 ajustes críticos** (executar o notebook + mover o relatório para o repositório), o projeto passa a ter **evidência completa e verificável** de todos os requisitos do PDF.

---

*Auditoria gerada em 2026-06-05 | Repositório: `D:\FIAP\Global Solution\gs1-pln`*


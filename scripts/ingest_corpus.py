#!/usr/bin/env python3
"""
scripts/ingest_corpus.py
CLI para ingestão em lote do corpus Sueteres.

Uso:
    cd sueteres-rag
    python scripts/ingest_corpus.py --corpus-dir ./corpus
    python scripts/ingest_corpus.py --file ./corpus/leed_v4.1.pdf --doc-id DOC-001
    python scripts/ingest_corpus.py --dry-run          # Apenas lista arquivos, não ingere
    python scripts/ingest_corpus.py --stats            # Mostra estatísticas do corpus indexado
"""

from __future__ import annotations

import argparse
import json
import structlog
import sys
import time
from pathlib import Path

# Garante que o diretório raiz do projeto está no PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import configure_logging
from config.settings import get_settings
from document_store.sqlite_store import SQLiteDocumentStore
from domain.entities import (
    DocumentCategory,
    DocumentMetadata,
    DocumentStatus,
)
from ingestion.orchestrator import IngestionOrchestrator
from rag.embedder import get_embedder
from vector_store.chroma_store import ChromaVectorStore

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Catálogo do corpus (15 documentos definidos na arquitetura)
# ─────────────────────────────────────────────────────────────

CORPUS_CATALOG: list[dict] = [
    # CATEGORIA 1 — Normas e Certificações
    {
        "doc_id": "DOC-001",
        "title": "LEED v4.1 BD+C",
        "title_full": "LEED v4.1 Building Design and Construction Reference Guide",
        "issuer": "U.S. Green Building Council",
        "issuer_country": "USA",
        "year": 2019,
        "year_updated": 2021,
        "status": DocumentStatus.ACTIVE,
        "language": "en",
        "category": DocumentCategory.NORMAS_CERTIFICACOES,
        "subcategory": "certificacao_internacional",
        "domain_tags": ["LEED", "certificação", "sustentabilidade", "energia", "água"],
        "access_type": "public",
        "source_url": "https://www.usgbc.org/leed/v41",
        "filename": "leed_v4.1_bdC_reference_guide.pdf",
    },
    {
        "doc_id": "DOC-002",
        "title": "AQUA-HQE Referencial",
        "title_full": "Referencial Técnico de Certificação AQUA-HQE — Edifícios em Construção",
        "issuer": "Fundação Vanzolini",
        "issuer_country": "BR",
        "year": 2021,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.NORMAS_CERTIFICACOES,
        "subcategory": "certificacao_nacional",
        "domain_tags": ["AQUA-HQE", "certificação", "Brasil", "sustentabilidade"],
        "access_type": "paid",
        "source_url": "https://vanzolini.org.br/aqua",
        "filename": "aqua_hqe_referencial_2021.pdf",
    },
    {
        "doc_id": "DOC-003",
        "title": "ABNT NBR 15575",
        "title_full": "ABNT NBR 15575 — Edificações Habitacionais — Desempenho",
        "issuer": "ABNT",
        "issuer_country": "BR",
        "year": 2013,
        "year_updated": 2021,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.NORMAS_CERTIFICACOES,
        "subcategory": "norma_desempenho",
        "domain_tags": ["ABNT", "NBR 15575", "desempenho", "habitação"],
        "access_type": "paid",
        "source_url": "https://www.abnt.org.br",
        "filename": "abnt_nbr_15575_2021.pdf",
    },
    {
        "doc_id": "DOC-004",
        "title": "ABNT NBR 10844",
        "title_full": "ABNT NBR 10844 — Instalações Prediais de Águas Pluviais",
        "issuer": "ABNT",
        "issuer_country": "BR",
        "year": 1989,
        "year_updated": 2023,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.NORMAS_CERTIFICACOES,
        "subcategory": "norma_hidrica",
        "domain_tags": ["ABNT", "NBR 10844", "águas pluviais", "drenagem"],
        "access_type": "paid",
        "source_url": "https://www.abnt.org.br",
        "filename": "abnt_nbr_10844_2023.pdf",
    },
    {
        "doc_id": "DOC-005",
        "title": "Selo Casa Azul+",
        "title_full": "Guia de Boas Práticas — Selo Casa Azul+ da Caixa Econômica Federal",
        "issuer": "Caixa Econômica Federal",
        "issuer_country": "BR",
        "year": 2021,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.NORMAS_CERTIFICACOES,
        "subcategory": "certificacao_habitacao_social",
        "domain_tags": ["Selo Casa Azul+", "Caixa", "habitação social", "eficiência"],
        "access_type": "public",
        "source_url": "https://www.caixa.gov.br/sustentabilidade",
        "filename": "selo_casa_azul_plus_guia_2021.pdf",
    },
    # CATEGORIA 2 — Relatórios Técnicos e Pesquisa
    {
        "doc_id": "DOC-006",
        "title": "IEA Tracking Buildings 2023",
        "title_full": "Tracking Clean Energy Progress: Buildings — IEA 2023",
        "issuer": "International Energy Agency",
        "issuer_country": "INT",
        "year": 2023,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "en",
        "category": DocumentCategory.RELATORIOS_TECNICOS,
        "subcategory": "relatorio_energia",
        "domain_tags": ["IEA", "energia", "edificações", "global", "tendências"],
        "access_type": "public",
        "source_url": "https://www.iea.org/reports/buildings",
        "filename": "iea_tracking_buildings_2023.pdf",
    },
    {
        "doc_id": "DOC-007",
        "title": "PROCEL Edifica",
        "title_full": "Regulamento Técnico da Qualidade para o Nível de Eficiência Energética — RTQ-C e RTQ-R",
        "issuer": "INMETRO / ELETROBRAS PROCEL",
        "issuer_country": "BR",
        "year": 2021,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.RELATORIOS_TECNICOS,
        "subcategory": "regulamento_eficiencia",
        "domain_tags": ["PROCEL", "INMETRO", "eficiência energética", "etiquetagem"],
        "access_type": "public",
        "source_url": "http://www.procelinfo.com.br",
        "filename": "procel_edifica_rtq_2021.pdf",
    },
    {
        "doc_id": "DOC-008",
        "title": "CBCS — Água em Edificações",
        "title_full": "Uso Racional da Água em Edificações — Conselho Brasileiro de Construção Sustentável",
        "issuer": "CBCS",
        "issuer_country": "BR",
        "year": 2022,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.RELATORIOS_TECNICOS,
        "subcategory": "relatorio_hidrico",
        "domain_tags": ["CBCS", "água", "uso racional", "Brasil"],
        "access_type": "public",
        "source_url": "https://www.cbcs.org.br",
        "filename": "cbcs_agua_edificacoes_2022.pdf",
    },
    {
        "doc_id": "DOC-009",
        "title": "Life Cycle Assessment — Green Buildings (JCP 2022)",
        "title_full": "Life Cycle Assessment of Green Building Rating Systems: A Systematic Review",
        "issuer": "Journal of Cleaner Production",
        "issuer_country": "INT",
        "year": 2022,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "en",
        "category": DocumentCategory.RELATORIOS_TECNICOS,
        "subcategory": "artigo_cientifico",
        "domain_tags": ["LCA", "ACV", "revisão sistemática", "certificações"],
        "access_type": "paid",
        "source_url": "https://doi.org/10.1016/j.jclepro.2022.xxx",
        "filename": "jcp_2022_lca_green_buildings.pdf",
    },
    {
        "doc_id": "DOC-010",
        "title": "Atlas Solar EPE 2022",
        "title_full": "Atlas de Energia Solar do Brasil — Empresa de Pesquisa Energética 2022",
        "issuer": "Empresa de Pesquisa Energética",
        "issuer_country": "BR",
        "year": 2022,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.RELATORIOS_TECNICOS,
        "subcategory": "atlas_energia",
        "domain_tags": ["solar", "irradiação", "fotovoltaico", "Atlas", "EPE"],
        "access_type": "public",
        "source_url": "https://www.epe.gov.br/atlas-solar",
        "filename": "epe_atlas_solar_2022.pdf",
    },
    # CATEGORIA 3 — Tecnologias Habilitadoras
    {
        "doc_id": "DOC-011",
        "title": "ABSOLAR — Manual FV Residencial",
        "title_full": "Manual de Energia Solar Fotovoltaica para Residências",
        "issuer": "ABSOLAR",
        "issuer_country": "BR",
        "year": 2023,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.TECNOLOGIAS_HABILITADORAS,
        "subcategory": "solar_fotovoltaico",
        "domain_tags": ["fotovoltaico", "ABSOLAR", "residencial", "geração"],
        "access_type": "public",
        "source_url": "https://www.absolar.org.br",
        "filename": "absolar_manual_fv_2023.pdf",
    },
    {
        "doc_id": "DOC-012",
        "title": "ANA — Manual de Reúso de Água",
        "title_full": "Manual de Reúso de Água em Edificações — Agência Nacional de Águas",
        "issuer": "ANA",
        "issuer_country": "BR",
        "year": 2022,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.TECNOLOGIAS_HABILITADORAS,
        "subcategory": "reuso_agua",
        "domain_tags": ["reúso", "água cinza", "ANA", "tratamento"],
        "access_type": "public",
        "source_url": "https://www.gov.br/ana",
        "filename": "ana_manual_reuso_agua_2022.pdf",
    },
    {
        "doc_id": "DOC-013",
        "title": "SINDUSCON — Guia Reúso Água Cinza",
        "title_full": "Guia Técnico para Aproveitamento de Água Cinza em Edificações",
        "issuer": "SINDUSCON-SP",
        "issuer_country": "BR",
        "year": 2021,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.TECNOLOGIAS_HABILITADORAS,
        "subcategory": "reuso_agua",
        "domain_tags": ["SINDUSCON", "água cinza", "aproveitamento", "tratamento"],
        "access_type": "public",
        "source_url": "https://www.sindusconsp.com.br",
        "filename": "sinduscon_guia_agua_cinza_2021.pdf",
    },
    {
        "doc_id": "DOC-014",
        "title": "ASHRAE 90.1-2022",
        "title_full": "ASHRAE Standard 90.1-2022 — Energy Standard for Sites and Buildings",
        "issuer": "ASHRAE",
        "issuer_country": "USA",
        "year": 2022,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "en",
        "category": DocumentCategory.TECNOLOGIAS_HABILITADORAS,
        "subcategory": "norma_energia",
        "domain_tags": ["ASHRAE", "90.1", "energia", "padrão", "sistemas HVAC"],
        "access_type": "paid",
        "source_url": "https://www.ashrae.org/90.1",
        "filename": "ashrae_90.1_2022.pdf",
    },
    {
        "doc_id": "DOC-015",
        "title": "ABESCO — BEMS e Automação Predial",
        "title_full": "Sistemas de Gestão de Energia Predial (BEMS) — Guia de Implementação",
        "issuer": "ABESCO",
        "issuer_country": "BR",
        "year": 2022,
        "year_updated": None,
        "status": DocumentStatus.ACTIVE,
        "language": "pt",
        "category": DocumentCategory.TECNOLOGIAS_HABILITADORAS,
        "subcategory": "automacao_predial",
        "domain_tags": ["BEMS", "automação", "gestão energia", "ABESCO"],
        "access_type": "public",
        "source_url": "https://www.abesco.com.br",
        "filename": "abesco_bems_guia_2022.pdf",
    },
]


def find_file_for_doc(corpus_dir: Path, filename: str) -> Path | None:
    """Procura o arquivo do documento no diretório do corpus."""
    candidate = corpus_dir / filename
    if candidate.exists():
        return candidate

    # Busca fuzzy por stem (sem extensão)
    stem = Path(filename).stem
    for ext in (".pdf", ".docx", ".txt", ".html"):
        for f in corpus_dir.glob(f"*{stem}*{ext}"):
            return f

    return None


def build_metadata(catalog_entry: dict, file_path: Path, sha256: str) -> DocumentMetadata:
    """Constrói DocumentMetadata a partir do catálogo."""
    entry = catalog_entry.copy()
    return DocumentMetadata(
        doc_id=entry["doc_id"],
        title=entry["title"],
        title_full=entry["title_full"],
        issuer=entry["issuer"],
        issuer_country=entry["issuer_country"],
        year=entry["year"],
        year_updated=entry.get("year_updated"),
        status=entry["status"],
        language=entry["language"],
        category=entry["category"],
        subcategory=entry["subcategory"],
        domain_tags=entry["domain_tags"],
        format=_infer_format(file_path),
        pages_total=0,  # Preenchido pelo loader
        access_type=entry["access_type"],
        source_url=entry["source_url"],
        local_path=str(file_path),
        sha256_checksum=sha256,
    )


def _infer_format(path: Path):  # type: ignore[return]
    from domain.entities import DocumentFormat
    ext = path.suffix.lower()
    return {
        ".pdf": DocumentFormat.PDF,
        ".docx": DocumentFormat.DOCX,
        ".txt": DocumentFormat.TXT,
        ".html": DocumentFormat.HTML,
    }.get(ext, DocumentFormat.UNKNOWN)


# ─────────────────────────────────────────────────────────────
# Comandos CLI
# ─────────────────────────────────────────────────────────────


def cmd_ingest_corpus(args: argparse.Namespace) -> int:
    """Ingere todos os documentos do catálogo encontrados no corpus_dir."""
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus dir not found: {corpus_dir}")
        return 1

    settings = get_settings()
    orchestrator = IngestionOrchestrator(
        vector_store=ChromaVectorStore(),
        document_store=SQLiteDocumentStore(),
        embedder=get_embedder(),
    )

    success_count = 0
    skip_count = 0
    error_count = 0
    total_chunks = 0
    start = time.monotonic()

    for entry in CORPUS_CATALOG:
        doc_id = entry["doc_id"]
        file_path = find_file_for_doc(corpus_dir, entry["filename"])

        if file_path is None:
            print(f"  ⚠  {doc_id}: file not found ({entry['filename']}) — skipping")
            skip_count += 1
            continue

        print(f"  →  {doc_id}: {file_path.name}", end="", flush=True)

        import hashlib
        sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()

        meta = build_metadata(entry, file_path, sha256)

        try:
            report = orchestrator.ingest(
                path=file_path,
                metadata=meta,
                force_reingest=args.force,
            )

            if report.status.value == "skipped":
                print(f" ↩  SKIPPED (already indexed)")
                skip_count += 1
            elif report.status.value == "success":
                total_chunks += report.chunks_generated
                print(
                    f" ✓  {report.chunks_generated} chunks "
                    f"({report.duration_seconds:.1f}s, quality={report.quality_score:.2f})"
                )
                if report.warnings:
                    for w in report.warnings[:3]:
                        print(f"     ⚠ {w}")
                success_count += 1
            else:
                print(f" ✗  FAILED: {report.errors}")
                error_count += 1

        except Exception as exc:
            print(f" ✗  ERROR: {exc}")
            error_count += 1

    duration = time.monotonic() - start
    print("\n" + "=" * 60)
    print(f"  Ingested: {success_count}  |  Skipped: {skip_count}  |  Errors: {error_count}")
    print(f"  Total chunks indexed: {total_chunks}")
    print(f"  Duration: {duration:.1f}s")
    print("=" * 60)

    return 0 if error_count == 0 else 1


def cmd_ingest_file(args: argparse.Namespace) -> int:
    """Ingere um único arquivo."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return 1

    import hashlib
    sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()

    # Monta metadata mínimo
    meta = DocumentMetadata(
        doc_id=args.doc_id or f"DOC-{file_path.stem[:20].upper()}",
        title=args.title or file_path.stem,
        title_full=args.title or file_path.stem,
        issuer=args.issuer or "Unknown",
        issuer_country="BR",
        year=args.year or 2024,
        year_updated=None,
        status=DocumentStatus.ACTIVE,
        language=args.language or "pt",
        category=DocumentCategory.RELATORIOS_TECNICOS,
        subcategory="custom",
        domain_tags=[],
        format=_infer_format(file_path),
        pages_total=0,
        access_type="public",
        source_url="",
        local_path=str(file_path),
        sha256_checksum=sha256,
    )

    orchestrator = IngestionOrchestrator(
        vector_store=ChromaVectorStore(),
        document_store=SQLiteDocumentStore(),
        embedder=get_embedder(),
    )

    print(f"Ingesting: {file_path}")
    report = orchestrator.ingest(
        path=file_path,
        metadata=meta,
        force_reingest=args.force,
    )

    print(json.dumps(
        {
            "doc_id": report.doc_id,
            "status": report.status.value,
            "chunks_generated": report.chunks_generated,
            "quality_score": report.quality_score,
            "duration_seconds": report.duration_seconds,
            "warnings": report.warnings[:5],
            "errors": report.errors[:5],
        },
        indent=2,
        ensure_ascii=False,
    ))

    return 0 if report.status.value != "failed" else 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Exibe estatísticas do corpus indexado."""
    ds = SQLiteDocumentStore()
    vs = ChromaVectorStore()

    docs = ds.list_documents()
    total_chunks = vs.count()

    print(f"\nSueteres RAG — Corpus Statistics")
    print("=" * 60)
    print(f"  Documents indexed : {len(docs)}")
    print(f"  Chunks in vector  : {total_chunks}")
    print()

    for doc in docs:
        print(
            f"  [{doc['doc_id']}] {doc['title'][:45]:<45} "
            f"{doc['total_chunks']:>5} chunks  ({doc['ingested_at'][:10]})"
        )

    print("=" * 60)
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Lista documentos disponíveis sem ingerir."""
    corpus_dir = Path(args.corpus_dir)

    print(f"\nDRY RUN — Corpus directory: {corpus_dir}")
    print("=" * 60)

    found = 0
    missing = 0
    for entry in CORPUS_CATALOG:
        file_path = find_file_for_doc(corpus_dir, entry["filename"])
        status = "✓ FOUND   " if file_path else "✗ MISSING "
        path_str = str(file_path) if file_path else entry["filename"]
        print(f"  {status} [{entry['doc_id']}] {path_str}")
        if file_path:
            found += 1
        else:
            missing += 1

    print("=" * 60)
    print(f"  Found: {found}  |  Missing: {missing}")
    return 0


# ─────────────────────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sueteres RAG — Corpus Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", help="Log level")

    sub = parser.add_subparsers(dest="command")

    # ingest-corpus
    p_corpus = sub.add_parser("ingest-corpus", help="Ingest all corpus documents")
    p_corpus.add_argument(
        "--corpus-dir",
        default="./corpus",
        help="Directory containing corpus files",
    )
    p_corpus.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if already indexed",
    )

    # ingest-file
    p_file = sub.add_parser("ingest-file", help="Ingest a single file")
    p_file.add_argument("--file", required=True, help="Path to file")
    p_file.add_argument("--doc-id", help="Document ID (e.g. DOC-001)")
    p_file.add_argument("--title", help="Document title")
    p_file.add_argument("--issuer", help="Document issuer")
    p_file.add_argument("--year", type=int, help="Publication year")
    p_file.add_argument("--language", default="pt", help="Language (pt or en)")
    p_file.add_argument("--force", action="store_true")

    # stats
    sub.add_parser("stats", help="Show corpus statistics")

    # dry-run
    p_dry = sub.add_parser("dry-run", help="List corpus files without ingesting")
    p_dry.add_argument("--corpus-dir", default="./corpus")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(settings=get_settings())

    if args.command == "ingest-corpus":
        return cmd_ingest_corpus(args)
    elif args.command == "ingest-file":
        return cmd_ingest_file(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "dry-run":
        return cmd_dry_run(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

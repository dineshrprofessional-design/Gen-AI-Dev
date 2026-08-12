"""Ingest CLI.

    python -m app.rag.ingest
    python -m app.rag.ingest --docs sample-corpus
    python -m app.rag.ingest --strict --json

Phase scope is deliberately narrow: read the corpus, resolve metadata, report
what loaded and what was quarantined. Chunking, embedding and indexing are
later phases and are not wired in here.
"""

import argparse
import sys
from pathlib import Path

from app.rag.loader import IngestError, load_documents
from app.rag.models import IngestReport

DEFAULT_DOCS = Path("docs")


def print_report(report: IngestReport) -> None:
    counts = report.counts()
    print(
        f"\n{counts['loaded']} loaded · {counts['failed']} failed · "
        f"{counts['skipped']} skipped   ({report.docs_root})\n"
    )

    if report.documents:
        header = (
            f"{'sdk':8} {'page_id':18} {'type':10} {'format':9} {'chars':>7}  source_file"
        )
        print(header)
        print("-" * len(header))
        for doc in sorted(report.documents, key=lambda d: (d.sdk_version, d.page_id)):
            print(
                f"{doc.sdk_version:8} {doc.page_id:18} {doc.page_type.value:10} "
                f"{doc.source_format:9} {len(doc.text):>7}  {doc.source_file}"
            )

    if report.failures:
        print("\nFAILED (quarantined — the run continued)")
        print("-" * 60)
        for failure in report.failures:
            print(f"  [{failure.stage.value:8}] {failure.source_file}")
            print(f"             {failure.reason}")

    if report.skipped:
        print("\nSKIPPED (not read at all)")
        print("-" * 60)
        for skipped in report.skipped:
            print(f"  {skipped.source_file} — {skipped.reason}")

    if report.documents:
        versions = sorted({d.sdk_version for d in report.documents})
        print(f"\nversions: {', '.join(versions)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a documentation corpus.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--version", default=None, help="ingest this sdk_version only")
    parser.add_argument("--show", action="store_true", help="print each document body")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if anything was quarantined"
    )
    args = parser.parse_args(argv)

    try:
        report = load_documents(args.docs, args.version)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        if args.show:
            for doc in report.documents:
                sources = {k: v.value for k, v in doc.metadata_sources.items()}
                print("\n" + "=" * 72)
                print(f"{doc.source_file}  [{doc.source_format}]")
                print(f"metadata : {doc.metadata()}")
                print(f"resolved : {sources}")
                print("=" * 72)
                print(doc.text.strip())
        print_report(report)

    if not report.documents:
        print("\nno documents loaded", file=sys.stderr)
        return 1
    if args.strict and report.failures:
        print(f"\nstrict: {report.failed_count} file(s) quarantined", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

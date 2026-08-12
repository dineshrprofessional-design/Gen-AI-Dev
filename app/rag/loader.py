"""Walk a corpus and turn it into an IngestReport.

The contract that matters: **per-file problems are recorded, not raised.** One
malformed page in a corpus of fifty thousand must not stop the other 49,999.
Only a genuinely fatal condition — the docs root not existing — raises.
"""

from pathlib import Path

from app.rag.extractors import ExtractError, get_extractor
from app.rag.metadata import MetadataError, resolve
from app.rag.models import Document, FailedFile, IngestReport, SkippedFile, Stage


class IngestError(ValueError):
    """Fatal: the run cannot start at all."""


def load_document(path: Path, root: Path) -> Document:
    """Read one file into a Document, or raise for this file alone.

    `source_file` is stored relative to the docs root so the metadata stays
    stable across machines instead of baking in an absolute path.
    """
    relative = path.relative_to(root)
    source_file = relative.as_posix()

    extractor = get_extractor(path)
    if extractor is None:
        raise ExtractError(f"unsupported file type {path.suffix or '(none)'}")

    try:
        extracted = extractor.extract(path)
    except ExtractError:
        raise
    except OSError as exc:
        raise ExtractError(f"cannot read file: {exc}") from exc

    if not extracted.text.strip():
        raise ExtractError("no text content")

    values, sources = resolve(relative, extracted)

    return Document(
        page_id=str(values["page_id"]),
        sdk_version=str(values["sdk_version"]),
        page_type=values["page_type"],  # type: ignore[arg-type]
        title=str(values["title"]),
        source_file=source_file,
        source_format=extractor.name,
        text=extracted.text,
        metadata_sources=sources,
    )


def load_documents(root: Path, sdk_version: str | None = None) -> IngestReport:
    """Ingest every file under `root`, quarantining whatever fails.

    The version filter keeps a run scoped to the new pages instead of
    re-reading the whole docs tree.
    """
    if not root.is_dir():
        raise IngestError(f"docs root not found: {root}")

    report = IngestReport(docs_root=str(root), sdk_version_filter=sdk_version)
    seen: dict[tuple[str, str], str] = {}

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        source_file = path.relative_to(root).as_posix()

        if get_extractor(path) is None:
            report.skipped.append(
                SkippedFile(
                    source_file=source_file,
                    reason=f"unsupported file type {path.suffix or '(none)'}",
                )
            )
            continue

        try:
            document = load_document(path, root)
        except ExtractError as exc:
            report.failures.append(
                FailedFile(source_file=source_file, stage=Stage.extract, reason=str(exc))
            )
            continue
        except MetadataError as exc:
            report.failures.append(
                FailedFile(
                    source_file=source_file, stage=Stage.metadata, reason=str(exc)
                )
            )
            continue
        except Exception as exc:  # a bad file must never crash the run
            report.failures.append(
                FailedFile(
                    source_file=source_file,
                    stage=Stage.validate,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        # Identity must be unique. Keep the first, quarantine the second, and
        # name both files — otherwise one would silently overwrite the other in
        # the index later.
        key = (document.sdk_version, document.page_id)
        if key in seen:
            report.failures.append(
                FailedFile(
                    source_file=source_file,
                    stage=Stage.validate,
                    reason=(
                        f"duplicate page {key[0]}/{key[1]} — "
                        f"already ingested from {seen[key]}"
                    ),
                )
            )
            continue
        seen[key] = source_file

        if sdk_version is not None and document.sdk_version != sdk_version:
            continue

        report.documents.append(document)

    return report

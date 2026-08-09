"""Resolve document metadata.

Front matter is the best evidence but it is usually absent — real corpora are
not authored for your pipeline. So each field falls back down a chain, and we
record which link in the chain answered.
"""

import re
from pathlib import Path

from app.rag.extractors import Extracted
from app.rag.models import MetadataSource, PageType

_VERSION = re.compile(r"^v\d+(?:[._]\d+)*$", re.IGNORECASE)
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

UNKNOWN_VERSION = "unknown"
DEFAULT_PAGE_TYPE = PageType.reference


class MetadataError(ValueError):
    """Metadata could not be resolved. The file is quarantined."""


def slugify(value: str) -> str:
    return _SLUG_STRIP.sub("-", value.strip().lower()).strip("-")


def humanise(stem: str) -> str:
    return " ".join(word for word in re.split(r"[-_\s]+", stem) if word).title()


def _version_from_path(parts: tuple[str, ...]) -> str | None:
    return next((p.lower() for p in parts if _VERSION.match(p)), None)


def _page_type_from_path(parts: tuple[str, ...]) -> PageType | None:
    allowed = {p.value: p for p in PageType}
    return next((allowed[p.lower()] for p in parts if p.lower() in allowed), None)


def resolve(
    relative_path: Path, extracted: Extracted
) -> tuple[dict[str, object], dict[str, MetadataSource]]:
    """Work out the four required fields plus the title.

    Returns (values, sources). Raises MetadataError only when a *declared*
    value is invalid — a missing value is derived, never fatal.
    """
    front = extracted.front_matter
    # Only the directories are evidence about the document; the filename is
    # handled separately as page_id.
    parts = tuple(relative_path.parts[:-1])
    sources: dict[str, MetadataSource] = {}

    # --- sdk_version ---
    if front.get("sdk_version"):
        sdk_version = front["sdk_version"]
        sources["sdk_version"] = MetadataSource.front_matter
    elif (derived := _version_from_path(parts)) is not None:
        sdk_version = derived
        sources["sdk_version"] = MetadataSource.path
    else:
        sdk_version = UNKNOWN_VERSION
        sources["sdk_version"] = MetadataSource.default

    # --- page_type ---
    if front.get("page_type"):
        try:
            page_type = PageType(front["page_type"])
        except ValueError as exc:
            allowed = ", ".join(p.value for p in PageType)
            raise MetadataError(
                f"page_type {front['page_type']!r} must be one of {allowed}"
            ) from exc
        sources["page_type"] = MetadataSource.front_matter
    elif (from_path := _page_type_from_path(parts)) is not None:
        page_type = from_path
        sources["page_type"] = MetadataSource.path
    else:
        page_type = DEFAULT_PAGE_TYPE
        sources["page_type"] = MetadataSource.default

    # --- page_id ---
    if front.get("page_id"):
        page_id = front["page_id"]
        sources["page_id"] = MetadataSource.front_matter
    else:
        page_id = slugify(relative_path.stem)
        sources["page_id"] = MetadataSource.filename
    if not page_id:
        raise MetadataError(f"cannot derive page_id from {relative_path.name!r}")

    # --- title ---
    if front.get("title"):
        title = front["title"]
        sources["title"] = MetadataSource.front_matter
    elif extracted.title_hint:
        title = extracted.title_hint.strip()
        sources["title"] = MetadataSource.content
    else:
        title = humanise(relative_path.stem)
        sources["title"] = MetadataSource.filename

    values: dict[str, object] = {
        "sdk_version": sdk_version,
        "page_type": page_type,
        "page_id": page_id,
        "title": title or humanise(relative_path.stem),
    }
    return values, sources

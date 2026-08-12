from enum import Enum

from pydantic import BaseModel, Field


class PageType(str, Enum):
    """What kind of page this is. Recorded now so retrieval can filter on it."""

    reference = "reference"
    guide = "guide"
    changelog = "changelog"


class MetadataSource(str, Enum):
    """Where a metadata value came from.

    "The author declared it" and "I guessed it from a folder name" are very
    different levels of trust, and only one of them is worth chasing down when
    a value looks wrong.
    """

    front_matter = "front_matter"
    path = "path"
    content = "content"
    filename = "filename"
    default = "default"


class Stage(str, Enum):
    """Which step of the pipeline rejected a file."""

    extract = "extract"
    metadata = "metadata"
    validate = "validate"


class Document(BaseModel):
    """One successfully ingested file: metadata contract + body."""

    page_id: str = Field(min_length=1)
    sdk_version: str = Field(min_length=1)
    page_type: PageType
    title: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_format: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata_sources: dict[str, MetadataSource] = Field(default_factory=dict)

    def metadata(self) -> dict[str, str]:
        """The required fields, flat. This is what travels downstream."""
        return {
            "source_file": self.source_file,
            "page_id": self.page_id,
            "sdk_version": self.sdk_version,
            "page_type": self.page_type.value,
        }


class Chunk(BaseModel):
    """One retrievable piece of a Document.

    Everything a Document carries is copied onto every chunk. A chunk that
    can't say which page it came from is useless no matter how well it matches
    a query.
    """

    chunk_id: str
    text: str

    # Inherited from the parent Document — the metadata contract, unchanged.
    source_file: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    sdk_version: str = Field(min_length=1)
    page_type: PageType

    # Where this chunk sits inside the page.
    heading_path: str = ""
    anchor: str = ""
    char_start: int = 0
    char_end: int = 0

    # Which chunker produced it, and what structure it happens to contain.
    strategy: str = ""
    chunk_ord: int = 0
    has_table_row: bool = False
    has_code_fence: bool = False

    def to_metadata(self) -> dict[str, str | int | bool]:
        """Flatten for the vector store, which only accepts scalar values.

        Every field the ingestion contract requires travels with the chunk —
        a chunk that cannot say which page it came from is useless no matter
        how well it matches a query.
        """
        return {
            "source_file": self.source_file,
            "page_id": self.page_id,
            "sdk_version": self.sdk_version,
            "page_type": self.page_type.value,
            "heading_path": self.heading_path,
            "anchor": self.anchor,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "strategy": self.strategy,
            "chunk_ord": self.chunk_ord,
            "has_table_row": self.has_table_row,
            "has_code_fence": self.has_code_fence,
        }


class FailedFile(BaseModel):
    """A file that should have ingested but didn't. Quarantined, not fatal."""

    source_file: str
    stage: Stage
    reason: str


class SkippedFile(BaseModel):
    """A file we never tried to read. Recorded so it is never silent."""

    source_file: str
    reason: str


class IngestReport(BaseModel):
    """The outcome of one ingest run — successes and failures together.

    Returning this instead of raising is the point: one bad file in a corpus of
    fifty thousand must not stop the other 49,999.
    """

    docs_root: str
    sdk_version_filter: str | None = None
    documents: list[Document] = Field(default_factory=list)
    failures: list[FailedFile] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return len(self.documents)

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def ok(self) -> bool:
        """Nothing was quarantined and at least one document came through."""
        return not self.failures and bool(self.documents)

    def counts(self) -> dict[str, int]:
        return {
            "loaded": self.loaded_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
        }

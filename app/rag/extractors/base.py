from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ExtractError(Exception):
    """This file could not be read into text. It gets quarantined."""


@dataclass
class Extracted:
    """Raw material handed to metadata resolution.

    `front_matter` is whatever the format could declare about itself — markdown
    has it, a PDF does not. `title_hint` is a title found in the content, which
    is weaker evidence than a declaration but stronger than a filename.
    """

    text: str
    front_matter: dict[str, str] = field(default_factory=dict)
    title_hint: str | None = None


class Extractor(Protocol):
    name: str
    extensions: tuple[str, ...]

    def extract(self, path: Path) -> Extracted: ...

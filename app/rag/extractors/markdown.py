import re
from pathlib import Path

from app.rag.extractors.base import Extracted, ExtractError

# Front matter is the leading `---` fenced block at the very top of the file.
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split front matter off the body. Absent front matter is not an error.

    Deliberately a tiny key: value parser rather than a YAML dependency — front
    matter here is flat, and a real YAML parser would silently accept nested
    shapes that nothing downstream can carry.
    """
    match = _FRONT_MATTER.match(raw)
    if match is None:
        return {}, raw

    fields: dict[str, str] = {}
    for number, line in enumerate(match.group(1).splitlines(), start=2):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ExtractError(f"malformed front matter at line {number}: {line!r}")
        fields[key.strip()] = value.strip().strip("\"'")

    return fields, raw[match.end() :]


class MarkdownExtractor:
    name = "markdown"
    extensions = (".md", ".markdown")

    def extract(self, path: Path) -> Extracted:
        raw = path.read_text(encoding="utf-8")
        front_matter, body = parse_front_matter(raw)
        heading = _H1.search(body)
        return Extracted(
            text=body,
            front_matter=front_matter,
            title_hint=heading.group(1).strip() if heading else None,
        )

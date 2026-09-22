"""Render report/w8/week8_report.md to a PDF.

    python report/w8/build_week8_pdf.py

The Markdown is the single source: this script parses it rather than holding a
second copy of the prose, so the .md and the .pdf cannot drift. Palette and
page furniture follow report/build_report.py, the repo's existing convention.

Deliberately a small hand-rolled Markdown subset (headings, paragraphs, tables,
fenced code, lists, blockquotes, rules, inline bold/italic/code) rather than a
dependency: the document is ours, we know exactly which constructs it uses, and
--verify asserts every line was consumed by some rule.
"""

import argparse
import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "report/w8/week8_report.md"
TARGET = ROOT / "report/w8/week8_report.pdf"

# palette — report/build_report.py
INK = colors.HexColor("#1c1b19")
MUTED = colors.HexColor("#6e6a63")
ACCENT = colors.HexColor("#2f6f4f")
RULE = colors.HexColor("#d9d6cf")
BAND = colors.HexColor("#f2f1ec")
CODEBG = colors.HexColor("#f5f4f0")

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
_CODE = re.compile(r"`([^`]+?)`")


def inline(text: str) -> str:
    """Markdown inline -> reportlab mini-HTML, escaped first."""
    out = html.escape(text, quote=False)
    out = _CODE.sub(lambda m: f'<font face="Courier" size="8.5" '
                              f'color="#2f6f4f">{m.group(1)}</font>', out)
    out = _BOLD.sub(r"<b>\1</b>", out)
    out = _ITAL.sub(r"<i>\1</i>", out)
    return out


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    mk = lambda **kw: ParagraphStyle(parent=base, **kw)
    return {
        "h1": mk(name="h1", fontName="Helvetica-Bold", fontSize=19, leading=23,
                 textColor=INK, spaceBefore=2, spaceAfter=10),
        "h2": mk(name="h2", fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                 textColor=ACCENT, spaceBefore=16, spaceAfter=6),
        "h3": mk(name="h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                 textColor=INK, spaceBefore=10, spaceAfter=4),
        "body": mk(name="body", fontName="Helvetica", fontSize=9.2, leading=13.4,
                   textColor=INK, spaceAfter=6),
        "meta": mk(name="meta", fontName="Helvetica-Oblique", fontSize=8.6,
                   leading=12, textColor=MUTED, spaceAfter=10),
        "li": mk(name="li", fontName="Helvetica", fontSize=9.2, leading=13.2,
                 textColor=INK, leftIndent=11, bulletIndent=2, spaceAfter=3),
        "quote": mk(name="quote", fontName="Helvetica-Oblique", fontSize=9,
                    leading=13, textColor=MUTED, leftIndent=10, spaceAfter=6),
        "cell": mk(name="cell", fontName="Helvetica", fontSize=8.1, leading=11,
                   textColor=INK),
        "cellh": mk(name="cellh", fontName="Helvetica-Bold", fontSize=8.1,
                    leading=11, textColor=INK),
    }


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_table(rows: list[list[str]], st: dict, width: float) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    data = [[Paragraph(inline(c), st["cellh"]) for c in header]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
    # first column wider; the rest share what is left
    first = min(width * 0.34, max(width / ncols, width * 0.18))
    rest = (width - first) / max(1, ncols - 1)
    table = Table(data, colWidths=[first] + [rest] * (ncols - 1), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def parse(md: str, st: dict, width: float) -> tuple[list, int]:
    flow, consumed = [], 0
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1; consumed += 1; continue

        if stripped.startswith("```"):
            i += 1; consumed += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1; consumed += 1
            i += 1; consumed += 1
            block = Preformatted("\n".join(buf), ParagraphStyle(
                name="code", fontName="Courier", fontSize=7.6, leading=10,
                textColor=INK, backColor=CODEBG, borderPadding=6,
                leftIndent=2, spaceAfter=8))
            flow.append(block)
            continue

        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = split_row(lines[i])
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1; consumed += 1
            if rows:
                flow.append(Spacer(1, 3))
                flow.append(build_table(rows, st, width))
                flow.append(Spacer(1, 8))
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            key = {1: "h1", 2: "h2", 3: "h3"}.get(level, "h3")
            flow.append(Paragraph(inline(text), st[key]))
            i += 1; consumed += 1
            continue

        if stripped.startswith("> "):
            flow.append(Paragraph(inline(stripped[2:]), st["quote"]))
            i += 1; consumed += 1
            continue

        if set(stripped) == {"-"} and len(stripped) >= 3:
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
            flow.append(Spacer(1, 6))
            i += 1; consumed += 1
            continue

        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            marker = re.match(r"^([-*]|\d+\.)\s+", stripped).group(1)
            text = re.sub(r"^([-*]|\d+\.)\s+", "", stripped)
            bullet = "•" if marker in "-*" else marker
            flow.append(Paragraph(inline(text), st["li"], bulletText=bullet))
            i += 1; consumed += 1
            continue

        if stripped.startswith("*") and stripped.endswith("*") and i < 8:
            flow.append(Paragraph(inline(stripped.strip("*")), st["meta"]))
            i += 1; consumed += 1
            continue

        flow.append(Paragraph(inline(stripped), st["body"]))
        i += 1; consumed += 1
    return flow, consumed


def furniture(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 11 * mm,
                      "Week 8 — Agent Failure Modes & Trajectory Evals")
    canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the Week 8 report to PDF.")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=TARGET)
    ap.add_argument("--verify", action="store_true",
                    help="assert every source line was consumed")
    args = ap.parse_args(argv)

    md = args.source.read_text(encoding="utf-8")
    total = len(md.splitlines())
    st = styles()
    margin = 18 * mm
    width = A4[0] - 2 * margin

    flow, consumed = parse(md, st, width)
    if args.verify and consumed != total:
        print(f"FAIL  consumed {consumed} of {total} source lines", file=sys.stderr)
        return 1

    doc = BaseDocTemplate(str(args.out), pagesize=A4,
                          leftMargin=margin, rightMargin=margin,
                          topMargin=18 * mm, bottomMargin=22 * mm,
                          title="Week 8 — Agent Failure Modes & Trajectory Evals",
                          author="dineshrprofessional-design")
    frame = Frame(margin, 22 * mm, width, A4[1] - 40 * mm, id="body")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=furniture)])
    # doc.build() drains the list it is given, so count first.
    flowables = len(flow)
    doc.build(flow)

    print(f"wrote {args.out}")
    print(f"  {total} source lines -> {consumed} consumed, "
          f"{flowables} flowables, {doc.page} pages, "
          f"{args.out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Document extraction across the formats the README advertises.

Formats whose library is not installed are skipped rather than failed -- the
suite has to stay green on a bare clone, and an optional extractor is exactly
that. What is *not* optional: an unreadable file raises a typed error naming the
format, and a file that is really a different format than its extension claims
does not silently produce garbage.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from ciws.core.errors import ValidationFailed
from ciws.ingest import extractors


def _skip_without(package: str) -> None:
    pytest.importorskip(package, reason=f"{package} is an optional extractor dependency")


# ---------------------------------------------------------------------------
# Plain text family
# ---------------------------------------------------------------------------


async def test_markdown(tmp_path: Path):
    source = tmp_path / "notes.md"
    source.write_text("# Title\n\nSome body text about turbines.\n", "utf-8")
    result = await extractors.extract(source)
    assert "turbines" in result.text


async def test_plain_text(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("just words", "utf-8")
    assert "just words" in (await extractors.extract(source)).text


async def test_source_code_is_extracted_verbatim(tmp_path: Path):
    source = tmp_path / "module.py"
    source.write_text("def compute():\n    return 42\n", "utf-8")
    result = await extractors.extract(source)
    assert "def compute" in result.text


async def test_a_file_with_a_bom_does_not_leak_it(tmp_path: Path):
    source = tmp_path / "bom.txt"
    source.write_bytes("﻿clean text".encode())
    result = await extractors.extract(source)
    assert not result.text.startswith("﻿")


async def test_latin1_content_does_not_explode(tmp_path: Path):
    source = tmp_path / "legacy.txt"
    source.write_bytes("caf\xe9 r\xe9sum\xe9".encode("latin-1"))
    result = await extractors.extract(source)
    assert result.text.strip(), "a non-UTF-8 file produced nothing"


# ---------------------------------------------------------------------------
# Structured text
# ---------------------------------------------------------------------------


async def test_csv_becomes_readable_rows(tmp_path: Path):
    source = tmp_path / "data.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "latency_ms"])
        writer.writerow(["eu-west-2", "38"])
        writer.writerow(["us-east-1", "51"])
    result = await extractors.extract(source)
    assert "eu-west-2" in result.text and "38" in result.text


async def test_json_is_flattened_not_dumped_raw(tmp_path: Path):
    source = tmp_path / "config.json"
    source.write_text(json.dumps({"deploy": {"day": "Friday", "hour": 16}}), "utf-8")
    result = await extractors.extract(source)
    assert "Friday" in result.text


async def test_html_is_stripped_to_text_and_keeps_its_title(tmp_path: Path):
    source = tmp_path / "page.html"
    source.write_text(
        "<html><head><title>Ops Handbook</title></head>"
        "<body><script>alert(1)</script><p>Retention is 90 days.</p></body></html>",
        "utf-8",
    )
    result = await extractors.extract(source)
    assert "90 days" in result.text
    assert "alert(1)" not in result.text, "script contents leaked into the extracted text"


def test_strip_html_drops_tags_and_scripts():
    text = extractors.strip_html("<p>hello <b>there</b></p><style>p{}</style>")
    assert "hello" in text and "there" in text
    assert "<" not in text and "p{}" not in text


def test_html_title_is_found_or_empty():
    assert extractors.html_title("<title>Set</title>") == "Set"
    assert extractors.html_title("<html><body>no title</body></html>") == ""


# ---------------------------------------------------------------------------
# Office formats (optional dependencies)
# ---------------------------------------------------------------------------


async def test_docx(tmp_path: Path):
    _skip_without("docx")
    import docx

    source = tmp_path / "memo.docx"
    document = docx.Document()
    document.add_heading("Quarterly Memo", level=1)
    document.add_paragraph("Redpanda cut p99 latency to 38ms.")
    document.save(source)

    result = await extractors.extract(source)
    assert "38ms" in result.text


async def test_xlsx(tmp_path: Path):
    _skip_without("openpyxl")
    import openpyxl

    source = tmp_path / "sheet.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet["A1"] = "region"
    sheet["B1"] = "latency"
    sheet["A2"] = "eu-west-2"
    sheet["B2"] = 38
    book.save(source)

    result = await extractors.extract(source)
    assert "eu-west-2" in result.text


async def test_pptx(tmp_path: Path):
    _skip_without("pptx")
    from pptx import Presentation

    source = tmp_path / "deck.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Migration Plan"
    deck.save(source)

    result = await extractors.extract(source)
    assert "Migration Plan" in result.text


async def test_pdf(tmp_path: Path):
    _skip_without("pypdf")
    reportlab = pytest.importorskip("reportlab", reason="need a PDF writer to make a fixture")
    from reportlab.pdfgen import canvas

    source = tmp_path / "doc.pdf"
    page = canvas.Canvas(str(source))
    page.drawString(72, 720, "Raw logs are retained for 90 days.")
    page.save()

    result = await extractors.extract(source)
    assert "90 days" in result.text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def test_a_binary_blob_raises_a_typed_error(tmp_path: Path):
    source = tmp_path / "thing.bin"
    source.write_bytes(bytes(range(256)) * 40)
    with pytest.raises(ValidationFailed):
        await extractors.extract(source)


async def test_a_missing_file_raises_rather_than_returning_empty(tmp_path: Path):
    with pytest.raises(Exception):
        await extractors.extract(tmp_path / "not-here.md")


async def test_a_zip_pretending_to_be_a_docx_fails_cleanly(tmp_path: Path):
    """A wrong-format file must not produce plausible-looking garbage."""
    source = tmp_path / "fake.docx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("hello.txt", "not a word document")

    with pytest.raises(Exception):
        await extractors.extract(source)


def test_supported_extensions_covers_what_the_readme_promises():
    supported = {e.lower().lstrip(".") for e in extractors.supported_extensions()}
    for extension in ("md", "txt", "csv", "json", "html", "pdf", "docx", "xlsx", "pptx", "py"):
        assert extension in supported, f"the README advertises .{extension}"

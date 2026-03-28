"""Tests for PDF text extraction."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from utils.pdf_parser import extract_text_from_pdf


def test_extract_text_from_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello peer review")
    doc.save(pdf_path)
    doc.close()

    text = extract_text_from_pdf(pdf_path)
    assert "Hello peer review" in text


def test_extract_text_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(tmp_path / "nope.pdf")

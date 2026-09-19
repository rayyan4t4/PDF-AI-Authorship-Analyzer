"""
tests/test_pdf_extraction.py

Unit tests for core/pdf_extractor.py
Uses synthetic minimal PDFs generated in-memory with fitz.
"""

import io
import pytest
import fitz

from core.pdf_extractor import extract_pdf, PDFExtractionResult


# ---------------------------------------------------------------------------
# Helpers: create minimal PDFs in memory
# ---------------------------------------------------------------------------

def _make_text_pdf(text: str) -> bytes:
    """Create a single-page PDF with the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_empty_pdf() -> bytes:
    """Create a valid PDF with no text content."""
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_multipage_pdf(texts: list[str]) -> bytes:
    """Create a multi-page PDF with one text block per page."""
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPDFExtraction:

    def test_basic_text_extraction(self):
        """A text PDF should return status=success and non-empty raw_text."""
        # Use text long enough to exceed MIN_EXTRACTABLE_CHARS (50 chars)
        sample_text = (
            "This is a test PDF document. It contains several sentences. "
            "The purpose of this document is to verify that text extraction works correctly. "
            "Additional text is included here to ensure sufficient character count."
        )
        pdf_bytes = _make_text_pdf(sample_text)
        result = extract_pdf(pdf_bytes, "test.pdf")

        assert result.status == "success"
        assert result.filename == "test.pdf"
        assert len(result.raw_text) > 0
        assert result.word_count > 0
        assert result.page_count == 1

    def test_empty_pdf_has_no_text_status(self):
        """A PDF with no text should return status=no_text."""
        pdf_bytes = _make_empty_pdf()
        result = extract_pdf(pdf_bytes, "empty.pdf")
        assert result.status == "no_text"
        assert result.char_count < 50

    def test_multipage_pdf(self):
        """Multi-page PDF should accumulate text from all pages."""
        texts = ["Page one content.", "Page two content.", "Page three content."]
        pdf_bytes = _make_multipage_pdf(texts)
        result = extract_pdf(pdf_bytes, "multipage.pdf")

        assert result.status == "success"
        assert result.page_count == 3
        assert result.word_count >= 9  # at least 3 words per page

    def test_corrupt_bytes_returns_error(self):
        """Random bytes should not crash — should return error status."""
        result = extract_pdf(b"NOT A PDF AT ALL!!!", "corrupt.pdf")
        assert result.status == "error"
        assert result.error is not None

    def test_empty_bytes_returns_error(self):
        """Empty byte string should return error gracefully."""
        result = extract_pdf(b"", "empty_bytes.pdf")
        assert result.status in ("error", "no_text")

    def test_filename_preserved(self):
        """Filename must be included in the result."""
        pdf_bytes = _make_text_pdf("Some text")
        result = extract_pdf(pdf_bytes, "my_assignment.pdf")
        assert result.filename == "my_assignment.pdf"

    def test_word_count_accuracy(self):
        """Word count should be approximately correct for known text."""
        # fitz insert_text wraps lines; we get fewer words due to overflow truncation.
        # Use a realistic sentence rather than repeated words to avoid truncation.
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains multiple unique words. "
            "Word count verification confirms the extractor is working."
        )
        pdf_bytes = _make_text_pdf(text)
        result = extract_pdf(pdf_bytes, "words.pdf")
        # Should extract at least 10 words from this 25-word text
        assert result.word_count >= 10

    def test_page_count_matches(self):
        """Page count should match the actual number of pages."""
        pdf_bytes = _make_multipage_pdf(["A"] * 5)
        result = extract_pdf(pdf_bytes, "fivepages.pdf")
        assert result.page_count == 5

    def test_char_count_populated(self):
        """Character count should be non-zero for text PDF."""
        pdf_bytes = _make_text_pdf("Characters here.")
        result = extract_pdf(pdf_bytes, "chars.pdf")
        assert result.char_count > 0

    def test_no_ocr_invoked(self):
        """Verify no OCR imports are used anywhere in pdf_extractor."""
        import ast
        import pathlib
        source = pathlib.Path("core/pdf_extractor.py").read_text()
        forbidden = ["tesseract", "pytesseract", "cv2", "PIL", "easyocr", "paddleocr"]
        for lib in forbidden:
            assert lib.lower() not in source.lower(), (
                f"OCR library '{lib}' found in pdf_extractor.py — OCR is forbidden in V1."
            )

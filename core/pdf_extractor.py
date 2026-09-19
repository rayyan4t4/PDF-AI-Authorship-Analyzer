"""
core/pdf_extractor.py

Extracts text from PDF files using PyMuPDF (fitz).
No OCR. Text-only extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config.settings import MIN_EXTRACTABLE_CHARS

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """Result of extracting text from a single PDF."""

    filename: str
    page_count: int = 0
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    raw_text: str = ""
    status: str = "pending"   # pending | success | no_text | encrypted | error
    error: Optional[str] = None
    page_texts: list[str] = field(default_factory=list)


def _count_sentences(text: str) -> int:
    """Approximate sentence count by splitting on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return sum(1 for p in parts if p.strip())


def _count_paragraphs(text: str) -> int:
    """Count non-empty paragraphs (separated by blank lines)."""
    paras = re.split(r"\n\s*\n", text.strip())
    return sum(1 for p in paras if p.strip())


def extract_pdf(pdf_bytes: bytes, filename: str) -> PDFExtractionResult:
    """
    Extract text from a PDF given its raw bytes and filename.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        filename:  Original filename (used for display and logging only).

    Returns:
        PDFExtractionResult with all text and metadata.
    """
    result = PDFExtractionResult(filename=filename)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.EmptyFileError:
        result.status = "error"
        result.error = "Empty or invalid PDF file."
        logger.warning(f"[{filename}] Empty/invalid PDF.")
        return result
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        logger.warning(f"[{filename}] Failed to open PDF: {exc}")
        return result

    # Handle encrypted PDFs
    if doc.is_encrypted:
        # Attempt to open with empty password (some PDFs only have permissions lock)
        try:
            doc.authenticate("")
        except Exception:
            pass
        if doc.is_encrypted:
            result.status = "encrypted"
            result.error = "PDF is password-protected."
            result.page_count = doc.page_count
            logger.warning(f"[{filename}] Encrypted PDF, skipping.")
            doc.close()
            return result

    result.page_count = doc.page_count
    page_texts: list[str] = []

    for page_num in range(doc.page_count):
        try:
            page = doc[page_num]
            text = page.get_text()  # plain text extraction only
            page_texts.append(text)
        except Exception as exc:
            logger.debug(f"[{filename}] Page {page_num + 1} extraction failed: {exc}")
            page_texts.append("")

    doc.close()

    raw_text = "\n".join(page_texts)
    result.page_texts = page_texts
    result.raw_text = raw_text
    result.char_count = len(raw_text)
    result.word_count = len(raw_text.split()) if raw_text.strip() else 0
    result.sentence_count = _count_sentences(raw_text)
    result.paragraph_count = _count_paragraphs(raw_text)

    if result.char_count < MIN_EXTRACTABLE_CHARS:
        result.status = "no_text"
        logger.info(f"[{filename}] No extractable text ({result.char_count} chars).")
    else:
        result.status = "success"
        logger.info(
            f"[{filename}] Extracted {result.word_count} words "
            f"from {result.page_count} pages."
        )

    return result

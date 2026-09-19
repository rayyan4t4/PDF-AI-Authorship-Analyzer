"""
core/text_preprocessor.py

Preprocesses extracted PDF text for analysis.
Preserves linguistic characteristics; does not aggressively modify text.
Always maintains both raw_text and processed_text.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedText:
    """Container for preprocessed text and its components."""

    raw_text: str                        # Never modified
    processed_text: str = ""            # Cleaned/normalized
    sentences: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

# Common PDF extraction artifacts
_ARTIFACT_PATTERNS = [
    (re.compile(r"\x00"), ""),                                 # null bytes
    (re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]"), " "), # control chars
    (re.compile(r"[ \t]{3,}"), "  "),                          # excessive spaces
    (re.compile(r"-\n(\w)"), r"\1"),                           # hyphenated line breaks
    (re.compile(r"(?<=[a-z])\n(?=[a-z])"), " "),              # mid-word line breaks
]

# Unicode punctuation normalization
_UNICODE_PUNCT = {
    "\u2018": "'",  "\u2019": "'",   # curly single quotes
    "\u201c": '"',  "\u201d": '"',   # curly double quotes
    "\u2013": "-",  "\u2014": "--",  # en/em dash
    "\u2026": "...",                  # ellipsis
    "\u00b7": "·",                   # middle dot (keep)
    "\u2022": "-",                   # bullet point
    "\u00ad": "",                    # soft hyphen
}

# Sentence boundary (simplified — avoids NLTK dependency)
_SENTENCE_SPLIT = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+"
)


def _normalize_unicode_punct(text: str) -> str:
    """Replace typographic punctuation with ASCII equivalents."""
    for src, dst in _UNICODE_PUNCT.items():
        text = text.replace(src, dst)
    return text


def _remove_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts."""
    for pattern, replacement in _ARTIFACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Preserve double newlines as paragraph markers
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces within lines
    lines = [re.sub(r" {2,}", " ", line.rstrip()) for line in text.splitlines()]
    return "\n".join(lines)


def _reconstruct_paragraphs(text: str) -> list[str]:
    """Split into paragraphs on blank lines."""
    raw_paras = re.split(r"\n\s*\n", text)
    paragraphs = []
    for para in raw_paras:
        cleaned = para.strip().replace("\n", " ")
        cleaned = re.sub(r" {2,}", " ", cleaned)
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _segment_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex-based heuristic.
    Handles common abbreviations imperfectly but avoids heavy NLP dependencies.
    """
    # Split on sentence-ending punctuation followed by whitespace + capital
    raw = _SENTENCE_SPLIT.split(text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 5:  # filter noise
            sentences.append(s)
    return sentences


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer — splits on non-alphanumeric characters."""
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(raw_text: str) -> PreprocessedText:
    """
    Preprocess extracted PDF text.

    The raw_text is never modified. All changes are applied to a working copy
    that becomes processed_text.

    Args:
        raw_text: Text as returned by PyMuPDF.

    Returns:
        PreprocessedText with all components populated.
    """
    result = PreprocessedText(raw_text=raw_text)

    working = raw_text

    # Step 1: Remove artifacts
    working = _remove_artifacts(working)

    # Step 2: Normalize Unicode punctuation
    working = _normalize_unicode_punct(working)

    # Step 3: Normalize whitespace
    working = _normalize_whitespace(working)

    result.processed_text = working.strip()

    # Step 4: Reconstruct paragraphs
    result.paragraphs = _reconstruct_paragraphs(result.processed_text)
    result.paragraph_count = len(result.paragraphs)

    # Step 5: Segment sentences (from paragraph text)
    all_sentences: list[str] = []
    for para in result.paragraphs:
        all_sentences.extend(_segment_sentences(para))
    result.sentences = all_sentences
    result.sentence_count = len(all_sentences)

    # Step 6: Tokenize
    result.tokens = _tokenize(result.processed_text)
    result.word_count = len(result.tokens)

    logger.debug(
        f"Preprocessed: {result.word_count} words, "
        f"{result.sentence_count} sentences, "
        f"{result.paragraph_count} paragraphs."
    )

    return result

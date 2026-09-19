"""
utils/helpers.py

Miscellaneous utility functions.
"""

import hashlib
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file bytes (for cache keys)."""
    return hashlib.sha256(data).hexdigest()


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def normalize_to_100(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Linearly scale value from [min_val, max_val] to [0, 100].
    Returns 0 if min_val == max_val.
    """
    if max_val == min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val) * 100.0
    return clamp(normalized)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_score(score: Any) -> str:
    """Format a score for display. Returns 'N/A' if None."""
    if score is None:
        return "N/A"
    try:
        return f"{float(score):.0f}"
    except (TypeError, ValueError):
        return str(score)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten a nested dict into a single-level dict with joined keys."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def count_words(text: str) -> int:
    """Count words using whitespace splitting."""
    return len(text.split()) if text.strip() else 0


def count_sentences(text: str) -> int:
    """Rough sentence count using terminal punctuation."""
    sentences = re.split(r"[.!?]+", text)
    return sum(1 for s in sentences if s.strip())


def seconds_to_human(seconds: float) -> str:
    """Convert seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    return f"{minutes:.1f}min"

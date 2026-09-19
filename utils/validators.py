"""
utils/validators.py

Validation utilities for uploaded files and configuration.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 100


def is_valid_pdf_extension(filename: str) -> bool:
    """Return True if the filename has a .pdf extension (case-insensitive)."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_valid_file_size(size_bytes: int, max_mb: float = MAX_FILE_SIZE_MB) -> bool:
    """Return True if file size is within the allowed limit."""
    return size_bytes <= max_mb * 1024 * 1024


def validate_uploaded_file(filename: str, size_bytes: int) -> tuple[bool, str]:
    """
    Validate an uploaded file for basic sanity.

    Returns:
        (is_valid, error_message)
    """
    if not is_valid_pdf_extension(filename):
        return False, f"File '{filename}' is not a PDF."

    if not is_valid_file_size(size_bytes):
        return False, (
            f"File '{filename}' exceeds the {MAX_FILE_SIZE_MB} MB limit "
            f"({size_bytes / 1024 / 1024:.1f} MB)."
        )

    return True, ""


def validate_weights(weights: dict[str, float]) -> tuple[bool, str]:
    """
    Check that scoring weights are all non-negative and approximately sum to 1.0.

    Returns:
        (is_valid, error_message)
    """
    for key, val in weights.items():
        if val < 0:
            return False, f"Weight '{key}' must be non-negative, got {val}."

    total = sum(weights.values())
    if total == 0:
        return False, "All weights are zero."
    if abs(total - 1.0) > 0.05:
        return (
            False,
            f"Weights sum to {total:.3f}. They should sum to 1.0. "
            "They will be normalized automatically.",
        )
    return True, ""

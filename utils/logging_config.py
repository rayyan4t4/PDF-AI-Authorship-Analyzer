"""
utils/logging_config.py

Configures application-wide logging.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """
    Configure root logger with console and optional file handler.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path to write log file.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    handlers: list[logging.Handler] = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Suppress noisy third-party loggers
    for lib in ("transformers", "torch", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() once at startup."""
    return logging.getLogger(name)

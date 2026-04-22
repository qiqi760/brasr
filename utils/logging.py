"""
logging.py
──────────
Logging configuration for GLCLAP experiments.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str | Path] = None,
    log_level: int = logging.INFO,
    log_filename: str = "train.log",
) -> logging.Logger:
    """
    Configure root logger to write to stdout and optionally a file.

    Args:
        log_dir:      Directory to write the log file to. If None, no file handler.
        log_level:    Logging level (default: INFO).
        log_filename: Name of the log file within log_dir.

    Returns:
        Root logger instance.
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)

    # Optional file handler
    if log_dir is not None:
        log_path = Path(log_dir) / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)
        root_logger.info(f"Logging to {log_path}")

    return root_logger

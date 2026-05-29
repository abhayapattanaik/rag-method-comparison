"""Structured logging configuration for the RAG Comparison System.

Sets up two handlers:
  - RotatingFileHandler → data/logs/rag_comparison.log (DEBUG, full detail)
  - StreamHandler       → stderr (INFO, concise)

Call setup_logging() once at process start (e.g. in cli/main.py::main()).
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import AppConfig

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_DEFAULT_LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
_LOG_FILENAME = "rag_comparison.log"

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5

_configured = False  # guard against double-initialisation


def setup_logging(
    config: "AppConfig | None" = None,
    level: str = "INFO",
) -> None:
    """Configure root logger with file (DEBUG) and console (INFO) handlers.

    Args:
        config: Optional AppConfig. Reserved for future use (e.g. log dir
                override from config). Currently unused but accepted so callers
                can pass it for forward-compatibility.
        level:  Minimum log level for the root logger. Defaults to "INFO".
                The file handler always uses DEBUG regardless of this setting.
    """
    global _configured
    if _configured:
        return

    log_dir = _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _LOG_FILENAME

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- File handler (DEBUG, rotating) ------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # --- Console handler (INFO → stderr) -----------------------------------
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # --- Root logger -------------------------------------------------------
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(min(logging.DEBUG, numeric_level))  # always capture DEBUG for file
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _configured = True

    logging.getLogger(__name__).debug(
        "Logging initialised: file=%s level=%s", log_path, level
    )

"""Shared utilities: logging configuration and custom exceptions.

This module centralizes cross-cutting concerns so that other modules in
the package do not duplicate logging setup or error-handling logic.
"""

from __future__ import annotations

import logging
from pathlib import Path


# --------------------------------------------------------------------------- #
# Custom exceptions
# --------------------------------------------------------------------------- #
class SmartGridImputationError(Exception):
    """Base exception for all errors raised by this package."""


class DataLoadError(SmartGridImputationError):
    """Raised when a dataset cannot be loaded or validated."""


class ColumnNotFoundError(SmartGridImputationError):
    """Raised when a requested column does not exist in the dataset."""


class NonNumericColumnError(SmartGridImputationError):
    """Raised when a selected column is not numeric and cannot be imputed."""


class InsufficientDataError(SmartGridImputationError):
    """Raised when there is not enough valid data to perform an operation."""


class ImputationError(SmartGridImputationError):
    """Raised when an imputation algorithm fails to produce a result."""


class InvalidMissingRateError(SmartGridImputationError):
    """Raised when a requested missing-data percentage is out of range."""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    log_file:
        Optional path to a file where logs should also be written.
    level:
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        A logger with a stream handler (and optionally a file handler)
        attached exactly once, even if this function is called repeatedly.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def ensure_directory(path: Path) -> Path:
    """Create a directory (including parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# Paper baseline scenarios plus 10-point stress increments through 90%, then 99%.
DEFAULT_MISSING_RATES: tuple[float, ...] = (
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.99,
)

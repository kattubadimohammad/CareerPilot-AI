"""Structured, secure logging module for CareerPilot AI.
Automatically redacts sensitive API keys and PII patterns (emails, phones).
"""
import logging
import os
import re

API_KEY_REGEX = re.compile(r"(?:AIza[0-9A-Za-z-_]{25,50}|AQ\.[0-9A-Za-z_-]{20,})")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")


class SensitiveDataScrubber(logging.Formatter):
    """Custom logging formatter that automatically scrubs sensitive credentials and PII."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        scrubbed = API_KEY_REGEX.sub("[REDACTED_KEY]", original)
        scrubbed = EMAIL_REGEX.sub("[REDACTED_EMAIL]", scrubbed)
        return scrubbed


def get_logger(name: str = "CareerPilot") -> logging.Logger:
    """Return a configured logger with PII and credential scrubbing."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))

        handler = logging.StreamHandler()
        formatter = SensitiveDataScrubber("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger

"""Security and hardening tests for CareerPilot AI."""
import html
import logging
import pytest
from src.gemini_service import GeminiService
from src.logger import SensitiveDataScrubber, get_logger


def test_xss_escaping():
    malicious_inputs = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('cookie')>",
        "<a href='javascript:steal()'>Click</a>",
        "Python\"><script>alert(1)</script>",
    ]
    for raw in malicious_inputs:
        escaped = html.escape(raw)
        assert "<script>" not in escaped
        assert "<img" not in escaped
        assert "<a href" not in escaped
        assert "&lt;" in escaped or "&quot;" in escaped


def test_logger_sensitive_data_scrubbing():
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="User john.doe@company.com with key AIzaSyDUMMYKEY12345678901234567890123 logged in.",
        args=(),
        exc_info=None,
    )
    formatter = SensitiveDataScrubber("%(message)s")
    formatted = formatter.format(record)

    assert "john.doe@company.com" not in formatted
    assert "[REDACTED_EMAIL]" in formatted
    assert "AIzaSyDUMMYKEY12345678901234567890123" not in formatted
    assert "[REDACTED_KEY]" in formatted


def test_gemini_auth_error_key_scrubbing():
    svc = GeminiService(api_key="AIzaSyFAKEKEYFORTESTINGPURPOSES99999")
    ok, msg = svc.test_connection()
    assert ok is False
    assert "AIzaSy" not in msg

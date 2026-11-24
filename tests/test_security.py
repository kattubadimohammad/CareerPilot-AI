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


def test_deployment_config_security():
    """Verify production and cloud configuration security parameters."""
    import tomllib
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    assert config_path.exists(), "config.toml must exist"

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    # Server security assertions
    server_cfg = cfg.get("server", {})
    assert server_cfg.get("enableCORS") is True, "CORS protection must be enabled"
    assert server_cfg.get("enableXsrfProtection") is True, "XSRF protection must be enabled"
    assert server_cfg.get("headless") is True, "Headless must be enabled for cloud/Docker"
    assert server_cfg.get("maxUploadSize") == 10, "Upload size must be capped at 10MB"

    # Cloud compatibility assertions: no hardcoded local-only address or restrictions
    assert "address" not in server_cfg or server_cfg.get("address") != "localhost", (
        "server.address must not be hardcoded to localhost"
    )
    assert "allowedHosts" not in server_cfg or len(server_cfg.get("allowedHosts", [])) == 0, (
        "server.allowedHosts must not hardcode local-only hosts for cloud compatibility"
    )


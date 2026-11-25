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
        msg="User john.doe@company.com with keys AIzaSyDUMMYKEY12345678901234567890123 and AQ.AQTESTKEY012345678901234567890123456789 logged in.",
        args=(),
        exc_info=None,
    )
    formatter = SensitiveDataScrubber("%(message)s")
    formatted = formatter.format(record)

    assert "john.doe@company.com" not in formatted
    assert "[REDACTED_EMAIL]" in formatted
    assert "AIzaSyDUMMYKEY12345678901234567890123" not in formatted
    assert "[REDACTED_KEY]" in formatted
    assert "AQ.AQTESTKEY012345678901234567890123456789" not in formatted


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


def test_secure_key_loading_from_streamlit_secrets(monkeypatch):
    """Confirm GEMINI_API_KEY is loaded directly from st.secrets on Streamlit Cloud."""
    import streamlit as st
    from app import get_gemini_api_key

    # 1. Test top-level st.secrets["GEMINI_API_KEY"]
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(st, "secrets", {"GEMINI_API_KEY": "AIzaSyCloudSecretsKeyTest12345"})

    key, source = get_gemini_api_key()
    assert key == "AIzaSyCloudSecretsKeyTest12345"
    assert source == "st.secrets['GEMINI_API_KEY']"

    # Also confirm GeminiService picks up st.secrets when no key passed
    svc = GeminiService()
    assert svc.api_key == "AIzaSyCloudSecretsKeyTest12345"
    assert svc.is_configured() is True


def test_secure_key_loading_from_nested_streamlit_secrets(monkeypatch):
    """Confirm GEMINI_API_KEY is loaded from [general] section in st.secrets."""
    import streamlit as st
    from app import get_gemini_api_key

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(st, "secrets", {"general": {"GEMINI_API_KEY": "AIzaSyNestedSecretsKey67890"}})

    key, source = get_gemini_api_key()
    assert key == "AIzaSyNestedSecretsKey67890"
    assert source == "st.secrets['general']['GEMINI_API_KEY']"


def test_secure_key_loading_environment_fallback(monkeypatch):
    """Confirm environment-variable fallback when st.secrets is empty or absent."""
    import streamlit as st
    from app import get_gemini_api_key

    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyEnvFallbackKey98765")

    key, source = get_gemini_api_key()
    assert key == "AIzaSyEnvFallbackKey98765"
    assert source == "environment:GEMINI_API_KEY"

    svc = GeminiService()
    assert svc.api_key == "AIzaSyEnvFallbackKey98765"
    assert svc.is_configured() is True


def test_secure_key_loading_unconfigured(monkeypatch):
    """Confirm unconfigured behavior when neither secrets nor env are present."""
    import streamlit as st
    from app import get_gemini_api_key

    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    key, source = get_gemini_api_key()
    assert key == ""
    assert source == "none"

    svc = GeminiService()
    assert svc.api_key == ""
    assert svc.is_configured() is False


def test_no_api_key_text_input_in_ui():
    """Verify that no visible text_input widget for Gemini API key exists in the app UI."""
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=10)

    # Check all text_input widgets in the entire application
    text_inputs = at.text_input
    labels = [ti.label.lower() for ti in text_inputs]
    for lbl in labels:
        assert "api key" not in lbl, f"Found exposed API key text_input: {lbl}"
        assert "gemini" not in lbl, f"Found exposed Gemini text_input: {lbl}"


def test_api_key_not_exposed_in_diagnostics_or_markup(monkeypatch):
    """Verify that configured API keys are never leaked into rendered UI markup or diagnostics."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    secret_key = "AIzaSySuperSecretKeyDoNotLeak99999"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=10)

    # Inspect all rendered text and markdown
    for md in at.markdown:
        assert secret_key not in md.value, "API key leaked in markdown!"

    for c in at.caption:
        assert secret_key not in c.value, "API key leaked in caption!"

    # Verify safe status badge displays active status without leaking key
    md_texts = " ".join([m.value for m in at.markdown])
    assert "Key format accepted (Environment)" in md_texts
    assert "Connection not verified" in md_texts
    assert secret_key not in md_texts



def test_gemini_configuration_diagnostics_do_not_expose_key(monkeypatch):
    """Show safe source/format diagnostics for an invalid key without revealing it."""
    import streamlit as st
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    secret_value = "not-a-real-gemini-key"
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("GEMINI_API_KEY", secret_value)

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=10)

    rendered = " ".join(
        [item.value for item in at.markdown]
        + [item.value for item in at.caption]
        + [item.value for item in at.text]
    )
    assert "Secret detected:" in rendered
    assert "environment:GEMINI_API_KEY" in rendered
    assert "Missing or invalid format" in rendered
    assert secret_value not in rendered


def test_aq_format_gemini_key_is_accepted_and_redacted(monkeypatch):
    """Accept the newer Google AI Studio AQ. key format without exposing it."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    aq_key = "AQ.AQTESTKEY012345678901234567890123456789"
    monkeypatch.setattr(st, "secrets", {"GEMINI_API_KEY": aq_key})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    svc = GeminiService()
    assert svc.api_key == aq_key
    assert svc.is_configured() is True
    diagnostics = svc.get_diagnostic_info()
    assert diagnostics["key_prefix"] == "AQ."
    assert diagnostics["starts_with_aq"] is True
    assert aq_key not in str(diagnostics)

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=10)
    rendered = " ".join(
        [item.value for item in at.markdown]
        + [item.value for item in at.caption]
        + [item.value for item in at.text]
    )
    assert "Recognized key prefix: AQ." in rendered
    assert aq_key not in rendered

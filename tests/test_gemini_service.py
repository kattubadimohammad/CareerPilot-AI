"""Tests for Gemini Service, Secrets Loading, Model Failover, and Error Handling."""
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from src.gemini_service import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_UNVERIFIED_MODELS,
    PREFERRED_MODELS,
    GeminiService,
    load_gemini_api_key,
)


def test_gemini_service_initialization_no_key():
    service = GeminiService(api_key="")
    assert service.is_configured() is False

    success, msg = service.test_connection()
    assert success is False
    assert "No valid Gemini API key" in msg


def test_gemini_service_json_extraction():
    raw_markdown = """```json
    {
        "score": 92,
        "skills": ["Python", "Pandas"]
    }
    ```"""
    extracted = GeminiService._extract_json_text(raw_markdown)
    assert '"score": 92' in extracted
    assert not extracted.startswith("```")


def test_gemini_service_fallback_when_unconfigured():
    service = GeminiService(api_key="")
    fallback = {"status": "fallback_ok", "skills": ["Python"]}
    result, is_ai, note = service.generate_json_response(
        prompt="Analyze this text",
        fallback_data=fallback,
    )
    assert is_ai is False
    assert result["status"] == "fallback_ok"
    assert "Deterministic Fallback" in note


def test_gemini_set_api_key():
    service = GeminiService(api_key="")
    assert service.is_configured() is False
    service.set_api_key("AIzaSyFakeKeyForTestingPurposes1234567")
    assert service.is_configured() is True


# ==============================================================================
# KEY SANITIZATION & STRICT FORMAT VALIDATION TESTS
# ==============================================================================


def test_gemini_sanitize_key():
    assert GeminiService.sanitize_key(None) == ""
    assert GeminiService.sanitize_key("") == ""
    assert GeminiService.sanitize_key("   ") == ""
    assert GeminiService.sanitize_key("  AIzaSyValidKey1234567890123456789012  ") == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key('"AIzaSyValidKey1234567890123456789012"') == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key("'AIzaSyValidKey1234567890123456789012'") == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key('""AIzaSyValidKey1234567890123456789012""') == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key('  "AIzaSyValidKey1234567890123456789012"  ') == "AIzaSyValidKey1234567890123456789012"
    # Smart quotes & bearer prefixes
    assert GeminiService.sanitize_key('“AIzaSyValidKey1234567890123456789012”') == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key('‘AIzaSyValidKey1234567890123456789012’') == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key('`AIzaSyValidKey1234567890123456789012`') == "AIzaSyValidKey1234567890123456789012"
    assert GeminiService.sanitize_key('Bearer AIzaSyValidKey1234567890123456789012') == "AIzaSyValidKey1234567890123456789012"


def test_gemini_is_configured_strict_format_validation():
    # Valid Google AI Studio key format: starts with AIza, length 39
    valid_key = "AIzaSyFakeKeyForTestingPurposes1234567"
    assert GeminiService(api_key=valid_key).is_configured() is True
    assert GeminiService(api_key=f'"{valid_key}"').is_configured() is True
    assert GeminiService(api_key=f'“{valid_key}”').is_configured() is True

    # Valid base64 / Google Cloud key with + / = characters
    valid_cloud_key = "AIzaSyTestKeyWithSpecial+Chars/123456=="
    assert GeminiService(api_key=valid_cloud_key).is_configured() is True

    # Must NOT be considered configured merely because it has characters or is a placeholder
    bad_strings = [
        "",
        "1234567890",
        "AIzaShort",                     # Too short (<25 chars)
        "AIzaSy_placeholder_key_example_12345",  # Placeholder substring
        "AIzaSy_dummy_key_not_real_1234567890",  # Dummy substring
        "your_gemini_api_key_here",
        "key with spaces inside 12345678901234567890",
    ]
    for bad_key in bad_strings:
        svc = GeminiService(api_key=bad_key)
        assert svc.is_configured() is False, f"Key '{bad_key}' should not pass is_configured()"


# ==============================================================================
# MODEL CONFIGURATION & DISCOVERY TESTS
# ==============================================================================


def test_gemini_primary_model_preference():
    assert DEFAULT_MODEL == "gemini-3.5-flash"
    assert PREFERRED_MODELS[0] == "gemini-3.5-flash"
    assert PREFERRED_MODELS[1] == "gemini-3.5-flash-lite"

    service = GeminiService()
    assert service.model_name == "gemini-3.5-flash"


def test_gemini_available_models_export():
    assert isinstance(AVAILABLE_MODELS, list)
    assert len(AVAILABLE_MODELS) > 0
    assert DEFAULT_MODEL in AVAILABLE_MODELS
    assert AVAILABLE_MODELS == DEFAULT_UNVERIFIED_MODELS


def test_gemini_model_discovery_success():
    service = GeminiService(api_key="AIzaSyFakeKeyForTestingPurposes1234567")
    mock_client = MagicMock()
    mock_m1 = MagicMock()
    mock_m1.name = "models/gemini-3.5-flash"
    mock_m2 = MagicMock()
    mock_m2.name = "models/gemini-3.5-flash-lite"
    mock_m3 = MagicMock()
    mock_m3.name = "models/gemini-embedding-001"
    mock_client.models.list.return_value = [mock_m1, mock_m2, mock_m3]
    service._client = mock_client

    models, is_live = service.get_available_models()
    assert is_live is True
    assert "gemini-3.5-flash" in models
    assert "gemini-3.5-flash-lite" in models
    assert "gemini-embedding-001" not in models  # Embeddings excluded from conversational selector


def test_gemini_model_discovery_failure_graceful_fallback():
    service = GeminiService(api_key="AIzaSyFakeKeyForTestingPurposes1234567")
    mock_client = MagicMock()
    mock_client.models.list.side_effect = Exception("Network offline / API down")
    service._client = mock_client

    models, is_live = service.get_available_models()
    assert is_live is False
    assert models == DEFAULT_UNVERIFIED_MODELS
    assert "gemini-3.5-flash" in models


def test_gemini_model_discovery_empty_list_fallback():
    service = GeminiService(api_key="AIzaSyFakeKeyForTestingPurposes1234567")
    mock_client = MagicMock()
    mock_client.models.list.return_value = []
    service._client = mock_client

    models, is_live = service.get_available_models()
    assert is_live is False
    assert models == DEFAULT_UNVERIFIED_MODELS


# ==============================================================================
# STREAMLIT SECRETS & ENVIRONMENT LOADING TESTS
# ==============================================================================


def test_load_gemini_api_key_from_streamlit_secrets_standard():
    mock_secrets = {"GEMINI_API_KEY": "AIzaSySecretFromCloudSecrets12345"}
    with patch("streamlit.secrets", mock_secrets, create=True):
        loaded = load_gemini_api_key()
        assert loaded == "AIzaSySecretFromCloudSecrets12345"


def test_load_gemini_api_key_from_streamlit_secrets_with_quotes():
    mock_secrets = {"GEMINI_API_KEY": '"AIzaSySecretWithQuotes1234567"'}
    with patch("streamlit.secrets", mock_secrets, create=True):
        loaded = load_gemini_api_key()
        assert loaded == "AIzaSySecretWithQuotes1234567"


def test_load_gemini_api_key_from_streamlit_secrets_case_variants():
    # Lowercase gemini_api_key
    mock_secrets = {"gemini_api_key": "AIzaSyLowercaseSecret12345"}
    with patch("streamlit.secrets", mock_secrets, create=True):
        assert load_gemini_api_key() == "AIzaSyLowercaseSecret12345"

    # GOOGLE_API_KEY
    mock_secrets2 = {"GOOGLE_API_KEY": "AIzaSyGoogleKeySecret12345"}
    with patch("streamlit.secrets", mock_secrets2, create=True):
        assert load_gemini_api_key() == "AIzaSyGoogleKeySecret12345"


def test_load_gemini_api_key_from_nested_secrets():
    # [gemini] table
    mock_secrets_table = {"gemini": {"api_key": "AIzaSyNestedGeminiTable123"}}
    with patch("streamlit.secrets", mock_secrets_table, create=True):
        assert load_gemini_api_key() == "AIzaSyNestedGeminiTable123"

    # [general] table
    mock_secrets_general = {"general": {"GEMINI_API_KEY": "AIzaSyGeneralTable123"}}
    with patch("streamlit.secrets", mock_secrets_general, create=True):
        assert load_gemini_api_key() == "AIzaSyGeneralTable123"


def test_load_gemini_api_key_env_fallback():
    # Empty secrets falls back to os.environ
    with patch("streamlit.secrets", {}, create=True):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFromEnvVariable123"}, clear=True):
            assert load_gemini_api_key() == "AIzaSyFromEnvVariable123"


def test_load_gemini_api_key_missing_everywhere():
    with patch("streamlit.secrets", {}, create=True):
        with patch.dict(os.environ, {}, clear=True):
            assert load_gemini_api_key() == ""


def test_load_gemini_api_key_documented_nested_table():
    # Documented nested tables [gemini] and [google]
    mock_secrets_google = {"google": {"api_key": "AIzaSyDocumentedGoogleKey12345"}}
    with patch("streamlit.secrets", mock_secrets_google, create=True):
        assert load_gemini_api_key() == "AIzaSyDocumentedGoogleKey12345"

    mock_secrets_gemini = {"gemini": {"api_key": "AIzaSyDocumentedGeminiKey12345"}}
    with patch("streamlit.secrets", mock_secrets_gemini, create=True):
        assert load_gemini_api_key() == "AIzaSyDocumentedGeminiKey12345"


def test_load_gemini_api_key_never_selects_unrelated_credentials():
    # Secrets containing unrelated credentials in other tables must NOT be selected
    unrelated_secrets = {
        "database": {"password": "AIzaSyDatabasePasswordDoNotSelect123"},
        "stripe": {"secret_key": "AIzaSyStripeSecretKeyDoNotSelect123"},
        "analytics": {"token": "AIzaSyAnalyticsTokenDoNotSelect123"},
    }
    with patch("streamlit.secrets", unrelated_secrets, create=True):
        with patch.dict(os.environ, {}, clear=True):
            assert load_gemini_api_key() == ""


def test_gemini_diagnostic_info_structure():
    mock_secrets = {"GEMINI_API_KEY": "AIzaSySecretDiagnosticKey12345"}
    with patch("streamlit.secrets", mock_secrets, create=True):
        service = GeminiService(api_key="AIzaSySecretDiagnosticKey12345")
        diag = service.get_diagnostic_info()

        assert "git_commit" in diag and diag["git_commit"]
        assert diag["st_secrets_available"] is True
        assert "GEMINI_API_KEY" in diag["st_secrets_keys"]
        assert "st.secrets['GEMINI_API_KEY']" in diag["source_detected"]
        assert diag["key_detected"] is True
        assert diag["key_length"] == len("AIzaSySecretDiagnosticKey12345")
        assert diag["starts_with_aiza"] is True
        assert diag["is_configured"] is True
        assert "Valid" in diag["format_status"]

        # Ensure raw key string is never leaked in diagnostic output
        diag_str = str(diag)
        assert "AIzaSySecretDiagnosticKey12345" not in diag_str


# ==============================================================================
# API CONNECTION & DIAGNOSTIC FAILURE HANDLING TESTS
# ==============================================================================


def test_test_connection_404_model_not_found():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234", model_name="gemini-3.5-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "404 Not Found: models/gemini-3.5-flash is not found for API version v1beta"
    )
    service._client = mock_client

    ok, msg = service.test_connection()
    assert ok is False
    assert "404" in msg
    assert "not found" in msg.lower() or "not enabled" in msg.lower()
    assert "gemini-3.5-flash-lite" in msg


def test_test_connection_503_high_demand():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234", model_name="gemini-3.5-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "503 The model is overloaded. Please try again later."
    )
    service._client = mock_client

    ok, msg = service.test_connection()
    assert ok is False
    assert "503" in msg
    assert "high demand" in msg.lower()
    assert "gemini-3.5-flash-lite" in msg


def test_test_connection_400_invalid_key():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "400 API_KEY_INVALID: API key not valid. Please pass a valid API key."
    )
    service._client = mock_client

    ok, msg = service.test_connection()
    assert ok is False
    assert "Invalid Gemini API Key" in msg


def test_test_connection_429_quota():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric"
    )
    service._client = mock_client

    ok, msg = service.test_connection()
    assert ok is False
    assert "rate limit" in msg.lower() or "quota" in msg.lower()


def test_test_connection_403_permission_denied():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "403 PERMISSION_DENIED: Generative Language API has not been used in project"
    )
    service._client = mock_client

    ok, msg = service.test_connection()
    assert ok is False
    assert "403" in msg
    assert "Generative Language API" in msg


# ==============================================================================
# FAILOVER & FALLBACK TESTS
# ==============================================================================


def test_generate_json_response_model_failover_success():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234", model_name="gemini-3.5-flash")
    mock_client = MagicMock()

    # First model (gemini-3.5-flash) throws 503 high demand
    # Second model (gemini-3.5-flash-lite) succeeds
    successful_resp = MagicMock()
    successful_resp.text = '{"analysis": "successful failover", "score": 95}'

    mock_client.models.generate_content.side_effect = [
        Exception("503 The model is overloaded."),
        successful_resp,
    ]
    service._client = mock_client

    result, is_ai, note = service.generate_json_response(
        prompt="Analyze candidate",
        fallback_data={"analysis": "fallback"},
    )
    assert is_ai is True
    assert result["analysis"] == "successful failover"
    assert "gemini-3.5-flash-lite" in note
    assert "failover from gemini-3.5-flash" in note


def test_generate_json_response_all_fail_preserves_deterministic_fallback():
    service = GeminiService(api_key="AIzaSyTestKey123456789012345678901234")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable")
    service._client = mock_client

    fallback = {"status": "deterministic_guaranteed", "score": 80}
    result, is_ai, note = service.generate_json_response(
        prompt="Analyze candidate",
        fallback_data=fallback,
    )
    assert is_ai is False
    assert result["status"] == "deterministic_guaranteed"
    assert "Deterministic Fallback" in note


def test_error_scrubbing_redacts_actual_key():
    custom_key = "AIzaSySuperSecretKeyNotToBeLeaked1234567"
    service = GeminiService(api_key=custom_key)
    error_with_key = f"Error occurred with credential {custom_key} at endpoint."
    scrubbed = service._scrub_sensitive_string(error_with_key)

    assert custom_key not in scrubbed
    assert "[REDACTED_KEY]" in scrubbed

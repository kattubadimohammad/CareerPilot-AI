"""Tests for Gemini Service."""
import pytest
from src.gemini_service import GeminiService


def test_gemini_service_initialization_no_key():
    service = GeminiService(api_key="")
    assert service.is_configured() is False

    success, msg = service.test_connection()
    assert success is False
    assert "No Gemini API key" in msg


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

"""Automated tests for CareerPilot AI Resume Intelligence upload-to-analysis workflow.
Uses Streamlit AppTest to verify real interactive UI behaviors, file uploads,
session state preservation, and error handling.
"""
from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = str(BASE_DIR / "app.py")
SAMPLE_RESUME_TEXT = """
ALEX MORGAN
Senior Data Analyst
Email: alex.morgan@example.com
Phone: +1 555 010 2020

PROFESSIONAL SUMMARY
Data analyst with experience building dashboards, analyzing business performance,
and automating reporting workflows.

SKILLS
Python, SQL, Excel, Power BI, Tableau, Data Analysis, Data Visualization

EXPERIENCE
Senior Data Analyst | Example Company | 2021 - 2025
Built SQL reporting pipelines and Power BI dashboards for business stakeholders.

EDUCATION
Bachelor of Science in Information Systems
""".strip()


def make_sample_pdf():
    """Build a deterministic PDF fixture at test time."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), SAMPLE_RESUME_TEXT, fontsize=10)
    content = document.tobytes()
    document.close()
    return content


def make_sample_docx():
    """Build a deterministic DOCX fixture at test time."""
    document = Document()
    for line in SAMPLE_RESUME_TEXT.splitlines():
        if line.strip():
            document.add_paragraph(line)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_resume_intelligence_navigation_and_ready_state():
    """Verify that navigating to Resume Intelligence renders the file uploader and ready state."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    # Navigate to Resume Intelligence
    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()

    # Verify file uploader is present
    assert len(at.file_uploader) >= 1
    assert "Upload Resume" in at.file_uploader[0].label
    assert at.session_state.active_resume_data is None

    # Verify no API key text input is displayed in sidebar or page
    for ti in at.text_input:
        assert "api key" not in ti.label.lower()

    # Verify guidance message is displayed when no resume is loaded
    info_texts = [i.value for i in at.info]
    assert any("Ready to analyze" in t for t in info_texts)


def test_pdf_resume_upload_and_analysis_workflow():
    """Verify end-to-end upload of a PDF resume produces active analysis results."""
    pdf_bytes = make_sample_pdf()
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    # Navigate to Resume Intelligence
    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()

    # Upload PDF
    at.file_uploader[0].upload("sample_resume.pdf", pdf_bytes).run()

    # Assert active resume data is populated
    active_data = at.session_state.active_resume_data
    assert active_data is not None
    assert "ALEX MORGAN" in active_data.get("candidate_name", "").upper()
    assert len(active_data.get("skills", [])) > 0
    assert active_data.get("score", 0) > 0
    assert at.session_state.active_resume_id is not None

    # Assert UI rendered candidate header and metrics
    subheaders = [sh.value for sh in at.subheader]
    assert any("ALEX MORGAN" in sh.upper() for sh in subheaders)
    metrics = [(m.label, m.value) for m in at.metric]
    assert any(m[0] == "Resume Score" for m in metrics)


def test_docx_resume_upload_and_analysis_workflow():
    """Verify end-to-end upload of a DOCX resume produces active analysis results."""
    docx_bytes = make_sample_docx()
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    # Navigate to Resume Intelligence
    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()

    # Upload DOCX
    at.file_uploader[0].upload("sample_resume.docx", docx_bytes).run()

    # Assert active resume data is populated
    active_data = at.session_state.active_resume_data
    assert active_data is not None
    assert "ALEX MORGAN" in active_data.get("candidate_name", "").upper()
    assert len(active_data.get("skills", [])) > 0
    assert active_data.get("score", 0) > 0


def test_mobile_bom_pdf_upload_and_analysis_workflow():
    """Verify that a mobile-exported PDF with UTF-8 BOM uploads and analyzes seamlessly."""
    pdf_bytes = b"\xef\xbb\xbf" + make_sample_pdf()
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()
    at.file_uploader[0].upload("mobile_exported_resume.pdf", pdf_bytes).run()

    active_data = at.session_state.active_resume_data
    assert active_data is not None
    assert "ALEX MORGAN" in active_data.get("candidate_name", "").upper()
    assert active_data.get("score", 0) > 0
    subheaders = [sh.value for sh in at.subheader]
    assert any("ALEX MORGAN" in sh.upper() for sh in subheaders)



def test_empty_file_upload_error_handling():
    """Verify uploading an empty (0 bytes) file shows an explicit error instead of silent failure."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()
    at.file_uploader[0].upload("empty_resume.pdf", b"").run()

    # Assert clear error message is displayed
    errors = [e.value for e in at.error]
    assert any("empty (0 bytes)" in err.lower() for err in errors)
    assert at.session_state.active_resume_data is None


def test_invalid_magic_bytes_upload_error_handling():
    """Verify uploading an invalid/corrupted document displays DocumentValidationError."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()
    # Upload text file disguised as PDF
    at.file_uploader[0].upload("corrupt.pdf", b"Not a real PDF content").run()

    errors = [e.value for e in at.error]
    assert any("validation error" in err.lower() for err in errors)
    assert at.session_state.active_resume_data is None


def test_rerun_preserves_active_session_state():
    """Verify that rerunning the script (simulating tab switches or UI clicks) preserves analysis results."""
    pdf_bytes = make_sample_pdf()
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    at.sidebar.radio[0].set_value("📄 Resume Intelligence").run()
    at.file_uploader[0].upload("sample_resume.pdf", pdf_bytes).run()

    assert at.session_state.active_resume_data is not None
    orig_id = at.session_state.active_resume_id

    # Simulate rerun
    at.run()

    # Ensure data was preserved and not wiped
    assert at.session_state.active_resume_data is not None
    assert at.session_state.active_resume_id == orig_id
    subheaders = [sh.value for sh in at.subheader]
    assert any("ALEX MORGAN" in sh.upper() for sh in subheaders)


def test_gemini_model_fallback_hierarchy():
    """Verify Gemini fallback order: gemini-3.5-flash -> gemini-3.5-flash-lite -> Deterministic."""
    from unittest.mock import MagicMock
    from src.gemini_service import GeminiService, PREFERRED_MODELS

    # 1. Verify strict model configuration
    assert PREFERRED_MODELS == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
    for deprecated in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        assert deprecated not in PREFERRED_MODELS

    svc = GeminiService(api_key="AIzaSyTestKeyForModelHierarchy12345")
    assert svc.model_name == "gemini-3.5-flash"

    # 2. Mock client to simulate primary failure and fallback-lite success
    call_log = []

    def mock_generate_content(model, contents):
        call_log.append(model)
        if model == "gemini-3.5-flash":
            raise Exception("503 Service Unavailable on primary 3.5-flash")
        elif model == "gemini-3.5-flash-lite":
            mock_resp = MagicMock()
            mock_resp.text = '{"candidate_name": "Fallback Lite Candidate", "score": 90}'
            return mock_resp
        raise Exception("Unknown model called")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_generate_content
    svc._client = mock_client

    data, is_ai, note = svc.generate_json_response("Test prompt", fallback_data={"status": "deterministic"})
    assert is_ai is True
    assert "gemini-3.5-flash-lite" in note
    assert call_log == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
    assert data.get("candidate_name") == "Fallback Lite Candidate"

    # 3. Simulate both Gemini models failing -> Deterministic fallback
    call_log.clear()

    def mock_all_fail(model, contents):
        call_log.append(model)
        raise Exception(f"500 Internal error on {model}")

    mock_client.models.generate_content.side_effect = mock_all_fail
    data2, is_ai2, note2 = svc.generate_json_response("Test prompt", fallback_data={"status": "deterministic_fallback"})
    assert is_ai2 is False
    assert "Deterministic Fallback" in note2
    assert data2["status"] == "deterministic_fallback"
    assert call_log == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


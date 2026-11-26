"""Targeted regression tests for VibeForge 2026 improvements:
1. Career Roadmap personalization & proven skill protection.
2. Interview Coach custom role question generation (Data Analyst / Supply Chain Analytics).
3. Gemini failover safe diagnostics and fallback integrity.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from src.career_roadmap import CareerRoadmapGenerator, _has_skill
from src.database import Database
from src.gemini_service import GeminiService
from src.interview_coach import InterviewCoach

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = str(BASE_DIR / "app.py")


# ==============================================================================
# PRIORITY 1: CAREER ROADMAP PERSONALIZATION & SKILL PROTECTION
# ==============================================================================


def test_has_skill_matching_and_synonyms():
    """Verify that _has_skill accurately recognizes exact, synonym, and token variants."""
    skills = ["Python 3.11", "AWS (EC2, S3, RDS)", "PostgreSQL", "Power BI", "Docker & Kubernetes"]

    assert _has_skill("Python", skills) is True
    assert _has_skill("AWS", skills) is True
    assert _has_skill("SQL", skills) is True  # PostgreSQL matches SQL via synonym/token
    assert _has_skill("Docker", skills) is True
    assert _has_skill("Kubernetes", skills) is True
    assert _has_skill("Power BI", skills) is True
    assert _has_skill("Kafka", skills) is False
    assert _has_skill("Terraform", skills) is False


def test_proven_skills_never_recommended_as_critical_gaps():
    """Verify that already-proven skills are never categorized into missing_skills or P0_Critical."""
    candidate_skills = ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "CI/CD", "Redis"]
    gaps = CareerRoadmapGenerator.identify_skill_gaps(candidate_skills, "Senior Backend Engineer")

    # Proven skills must be in existing_skills
    for skill in ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "CI/CD", "Redis"]:
        assert skill in gaps["existing_skills"], f"{skill} was not recognized in existing_skills!"
        assert skill not in gaps["missing_skills"], f"{skill} was falsely flagged as missing!"
        assert skill not in gaps["learning_priorities"]["P0_Critical"], f"{skill} was falsely flagged as P0_Critical!"

    # Unproven benchmark skills should be in missing_skills
    assert "Kubernetes" in gaps["missing_skills"]
    assert "System Design" in gaps["missing_skills"]


def test_data_analyst_supply_chain_custom_role_benchmark():
    """Verify that custom roles like Data Analyst – Supply Chain Analytics receive domain-relevant benchmarks."""
    candidate_skills = ["SQL", "Excel", "Data Analysis", "Tableau"]
    gaps = CareerRoadmapGenerator.identify_skill_gaps(
        candidate_skills, "Data Analyst – Supply Chain Analytics"
    )

    # Benchmark skills must be supply chain and analytics oriented, NOT software architecture
    benchmarks = gaps["benchmark_skills"]
    assert any("supply chain" in b.lower() or "inventory" in b.lower() or "forecast" in b.lower() for b in benchmarks)
    assert not any("microservices" in b.lower() or "system design" in b.lower() for b in benchmarks)

    # Proven skills (SQL, Excel, Data Analysis) must not be in P0
    assert "SQL" in gaps["existing_skills"]
    assert "Excel" in gaps["existing_skills"]
    assert "SQL" not in gaps["learning_priorities"]["P0_Critical"]
    assert "Excel" not in gaps["learning_priorities"]["P0_Critical"]


def test_career_roadmap_apptest_resume_selector():
    """Verify that Career Roadmap UI renders the candidate profile selector and loads verified skills."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)
    at.sidebar.radio[0].set_value("🗺️ Career Roadmap").run(timeout=10)

    assert not at.exception
    # Profile selector should be present
    selector_labels = [sb.label for sb in at.selectbox]
    assert any("Candidate Profile" in l for l in selector_labels)

    # Verified candidate profile should be displayed
    success_texts = [s.value for s in at.success]
    assert any("Loaded Profile" in t for t in success_texts)


# ==============================================================================
# PRIORITY 2: INTERVIEW COACH CUSTOM ROLE DOMAIN QUESTIONS
# ==============================================================================


def test_interview_coach_supply_chain_custom_role():
    """Verify that Data Analyst – Supply Chain Analytics generates supply chain questions, not software architecture."""
    coach = InterviewCoach()
    questions = coach.get_questions_for_role("Data Analyst – Supply Chain Analytics", count=3)

    assert len(questions) == 3
    q_texts = " ".join([q["question"].lower() for q in questions])

    # Must contain domain concepts: inventory, lead-time, forecast, supply chain, or stockout
    assert any(term in q_texts for term in ["inventory", "lead-time", "supply chain", "stockout", "forecast"])

    # Must NOT ask about software systems architecture or legacy code
    assert "architecture of the most technically challenging software system" not in q_texts
    assert "legacy code" not in q_texts


def test_interview_coach_data_analyst_role():
    """Verify that general Data Analyst receives quantitative SQL and data integrity questions."""
    coach = InterviewCoach()
    questions = coach.get_questions_for_role("Data Analyst", count=3)

    assert len(questions) == 3
    q_texts = " ".join([q["question"].lower() for q in questions])

    assert "sql" in q_texts or "queries" in q_texts
    assert "missing" in q_texts or "anomalies" in q_texts or "integrity" in q_texts
    assert "architecture of the most technically challenging software system" not in q_texts


def test_interview_coach_evaluation_and_session_persistence():
    """Verify that answer evaluation and database persistence remain completely functional."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        temp_db_path = str(Path(tmpdir) / "test_interview.db")
        test_db = Database(temp_db_path)
        coach = InterviewCoach()

        question = "How do you optimize safety stock and reduce lead-time variability in supply chain analytics?"
        user_answer = (
            "In my previous role as supply chain analyst, our inventory stockouts spiked by 18% due to supplier shipping delays. "
            "My task was to build a dynamic safety stock model factoring historical lead-time volatility. "
            "I analyzed 50,000 purchase orders in SQL and built a dynamic reorder point formula in Python. "
            "Consequently, we reduced stockout occurrences by 35% while lowering overall warehouse holding costs by $120,000."
        )

        evaluation = coach.evaluate_answer_deterministic(question, user_answer, role="Supply Chain Analyst")
        assert evaluation["overall_score"] >= 70
        assert evaluation["star_analysis"]["action"] is True
        assert evaluation["star_analysis"]["result"] is True

        sess_id = test_db.save_interview_session(
            resume_id=None,
            target_role="Supply Chain Analyst",
            session_data={"question": question, "answer": user_answer, "evaluation": evaluation},
            overall_score=evaluation["overall_score"],
        )
        assert sess_id > 0
        saved = test_db.get_interview_session(sess_id)
        assert saved is not None
        assert saved["target_role"] == "Supply Chain Analyst"
        assert saved["overall_score"] == evaluation["overall_score"]


# ==============================================================================
# PRIORITY 3: GEMINI RELIABILITY & SAFE FAILOVER DIAGNOSTICS
# ==============================================================================


def test_gemini_safe_failover_diagnostics_on_404():
    """Verify that 404 failover from gemini-3.5-flash to gemini-3.5-flash-lite records safe diagnostics."""
    service = GeminiService(api_key="AIzaSyValidFormatKeyForTestingOnly4567", model_name="gemini-3.5-flash")
    mock_client = MagicMock()

    def mock_generate(model, contents):
        if model == "gemini-3.5-flash":
            raise Exception("404 Not Found: models/gemini-3.5-flash is not found for API version v1beta")
        # Second model succeeds
        resp = MagicMock()
        resp.text = '{"status": "ok", "provider": "gemini-3.5-flash-lite"}'
        return resp

    mock_client.models.generate_content.side_effect = mock_generate
    service._client = mock_client

    result, is_ai, note = service.generate_json_response("Test prompt")
    assert is_ai is True
    assert result.get("status") == "ok"
    assert "failover" in note
    assert service.last_used_model == "gemini-3.5-flash-lite"

    # Check safe diagnostics
    diag = service.get_diagnostic_info()
    assert diag["last_used_model"] == "gemini-3.5-flash-lite"
    assert len(diag["last_failover_diagnostics"]) == 2
    assert diag["last_failover_diagnostics"][0]["model"] == "gemini-3.5-flash"
    assert diag["last_failover_diagnostics"][0]["status"] == "failed"
    assert "404" in diag["last_failover_diagnostics"][0]["error"]
    assert diag["last_failover_diagnostics"][1]["model"] == "gemini-3.5-flash-lite"
    assert diag["last_failover_diagnostics"][1]["status"] == "success"

    # Verify no credential leaks in diagnostics string representation
    diag_repr = str(diag)
    assert "AIzaSyValidFormatKeyForTestingOnly4567" not in diag_repr


def test_gemini_failover_exhaustion_preserves_deterministic_fallback():
    """Verify that when both models fail (e.g. 404s), fallback is cleanly returned and diagnostics captured."""
    service = GeminiService(api_key="AIzaSyValidFormatKeyForTestingOnly4567", model_name="gemini-3.5-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("404 Not Found: model is not found")
    service._client = mock_client

    fallback = {"status": "deterministic_fallback_ok"}
    result, is_ai, note = service.generate_json_response("Test prompt", fallback_data=fallback)

    assert is_ai is False
    assert result["status"] == "deterministic_fallback_ok"
    assert "Deterministic Fallback" in note

    diag = service.get_diagnostic_info()
    assert len(diag["last_failover_diagnostics"]) == 2
    assert all(d["status"] == "failed" for d in diag["last_failover_diagnostics"])
    assert diag["last_error_safe"] is not None
    assert "Deterministic fallback engaged" in diag["last_error_safe"]

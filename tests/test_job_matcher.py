"""Tests for Job Matcher and ATS Optimizer."""
import pytest
from src.job_matcher import JobMatcher

SAMPLE_JOB_DESC = """
Position: Senior Backend Engineer
Company: Apex Cloud
Requirements:
- 5+ years of software engineering experience.
- Deep expertise in Python, FastAPI, and PostgreSQL.
- Strong hands-on experience with Docker, Kubernetes, and AWS (EC2, S3).
- Experience building event-driven systems using Kafka or Redis.
- Familiarity with CI/CD pipelines, Terraform, and Git.
Responsibilities:
- Build resilient microservices handling high transaction volume.
- Optimize database performance and implement caching strategies.
- Collaborate with frontend team and lead code reviews.
"""

SAMPLE_RESUME_TEXT = """
Alex Morgan
Senior Software Engineer
Skills: Python, FastAPI, Docker, PostgreSQL, Redis, Git, CI/CD
Experience: 6 years developing scalable web applications.
"""


def test_extract_job_skills():
    skills = JobMatcher.extract_job_skills(SAMPLE_JOB_DESC)
    assert "Python" in skills
    assert "FastAPI" in skills
    assert "Kubernetes" in skills
    assert "AWS" in skills
    assert "Docker" in skills
    assert "Kafka" in skills


def test_extract_ats_keywords():
    keywords = JobMatcher.extract_ats_keywords(SAMPLE_JOB_DESC, top_n=10)
    kw_names = [k for k, _ in keywords]
    assert any("microservices" in k or "experience" in k or "database" in k or "backend" in k for k in kw_names)


def test_calculate_explainable_score():
    resume_skills = ["Python", "FastAPI", "Docker", "PostgreSQL", "Redis", "Git", "CI/CD"]
    job_skills = JobMatcher.extract_job_skills(SAMPLE_JOB_DESC)

    scoring = JobMatcher.calculate_explainable_score(
        resume_skills=resume_skills,
        job_skills=job_skills,
        resume_text=SAMPLE_RESUME_TEXT,
        job_text=SAMPLE_JOB_DESC,
    )

    assert "overall_score" in scoring
    assert 40.0 <= scoring["overall_score"] <= 95.0
    assert "sub_scores" in scoring
    assert "skills_alignment" in scoring["sub_scores"]
    assert "ats_keyword_density" in scoring["sub_scores"]

    assert "Python" in scoring["matched_skills"]
    assert "FastAPI" in scoring["matched_skills"]
    assert "Kubernetes" in scoring["missing_skills"]
    assert "AWS" in scoring["missing_skills"]


def test_match_resume_to_job_pipeline():
    matcher = JobMatcher()
    result = matcher.match_resume_to_job(
        resume_text=SAMPLE_RESUME_TEXT,
        resume_skills=["Python", "FastAPI", "Docker"],
        job_text=SAMPLE_JOB_DESC,
        job_title="Senior Backend Engineer",
    )

    assert result["job_title"] == "Senior Backend Engineer"
    assert result["overall_score"] > 0
    assert "ethical_notice" in result
    assert "ETHICAL ATS OPTIMIZATION GUARDRAIL" in result["ethical_notice"]
    assert len(result["recommendations"]) >= 2
    assert result["is_ai_generated"] is False


def test_skill_synonyms_match_resume_evidence():
    score = JobMatcher.calculate_explainable_score(
        resume_skills=["Excel", "Power BI", "Inventory Management"],
        job_skills=["Advanced Excel", "Business Intelligence", "Inventory Management"],
        resume_text="Excel PivotTables, Power BI dashboards, inventory management.",
        job_text="Advanced Excel, Business Intelligence, Inventory Management",
    )
    assert score["matched_skills"] == ["Advanced Excel", "Business Intelligence", "Inventory Management"]
    assert score["missing_skills"] == []


def test_behavioral_evidence_is_partial_not_full_match():
    score = JobMatcher.calculate_explainable_score(
        resume_skills=[],
        job_skills=["Communication", "Leadership"],
        resume_text="Coordinated with stakeholders and collaborated across teams.",
        job_text="Communication and Leadership required.",
        job_title="Analyst",
    )
    assert "Communication" in score["partially_matched_skills"]
    assert "Leadership" in score["missing_skills"]
    assert score["partial_skills_count"] == 1


def test_short_ats_terms_are_retained_and_stopwords_excluded():
    keywords = dict(JobMatcher.extract_ats_keywords("PO lifecycle, BI tools, AI and ML. PO reporting.", top_n=20))
    assert keywords["po"] == 2
    assert "bi" in keywords
    assert "ai" in keywords
    assert "ml" in keywords
    assert "and" not in keywords


def test_seniority_uses_explicit_job_title_not_body_mentions():
    junior = JobMatcher.calculate_explainable_score(
        resume_skills=[], job_skills=[], resume_text="Analyst with 2 years experience.",
        job_text="The analyst reports to a senior manager.", job_title="Data Analyst"
    )
    senior = JobMatcher.calculate_explainable_score(
        resume_skills=[], job_skills=[], resume_text="Analyst with 2 years experience.",
        job_text="The role works with multiple teams.", job_title="Senior Data Analyst"
    )
    assert junior["sub_scores"]["seniority_experience_alignment"] == 90.0
    assert senior["sub_scores"]["seniority_experience_alignment"] == 75.0


def test_seniority_years_gap_is_reflected_when_available():
    score = JobMatcher.calculate_explainable_score(
        resume_skills=[], job_skills=[], resume_text="2 years of experience.",
        job_text="Requires 5+ years of experience.", job_title="Senior Analyst"
    )
    assert score["required_years_experience"] == 5
    assert score["candidate_years_experience"] == 2
    assert score["sub_scores"]["seniority_experience_alignment"] == 75.0


def test_ai_rewrite_guard_rejects_unsupported_metrics_and_tools():
    rewrites = [
        {"original_theme": "Impact", "suggested_rewrite": "Improved output by 40%.", "justification": "Metric"},
        {"original_theme": "Tools", "suggested_rewrite": "Built pipelines using Kubernetes.", "justification": "Tool"},
        {"original_theme": "Supported", "suggested_rewrite": "Analyzed data using SQL.", "justification": "Supported tool"},
    ]
    safe = JobMatcher._safe_ai_rewrites(rewrites, "Analyzed data using SQL.")
    assert len(safe) == 1
    assert "SQL" in safe[0]["suggested_rewrite"]


def test_no_job_skills_uses_neutral_skill_alignment_and_score_is_bounded():
    score = JobMatcher.calculate_explainable_score(
        resume_skills=[], job_skills=[], resume_text="", job_text="",
        job_title="Analyst"
    )
    assert score["sub_scores"]["skills_alignment"] == 70.0
    assert 0.0 <= score["overall_score"] <= 100.0


def test_job_matcher_clear_jd_button_apptest():
    """Verify Clear JD sits beside Load Sample JD and clears the job description safely."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception

    # Seed active session state from another workflow to verify Clear JD doesn't disturb it
    at.session_state["active_resume_data"] = {"candidate_name": "Alex Morgan", "skills": ["Python"]}
    at.sidebar.radio[0].set_value("🎯 Job Matcher & ATS").run()
    assert not at.exception

    btn_labels = [b.label for b in at.button]
    assert "📋 Load Sample JD" in btn_labels
    assert "🧹 Clear JD" in btn_labels

    # 1. Enter custom JD manually and clear it
    at.text_area[0].set_value("Custom Supply Chain Analyst Job Description").run()
    assert not at.exception
    assert at.text_area[0].value == "Custom Supply Chain Analyst Job Description"
    assert at.session_state["cached_jd_text"] == "Custom Supply Chain Analyst Job Description"

    clear_btn = next(b for b in at.button if b.label == "🧹 Clear JD")
    clear_btn.click().run()
    assert not at.exception
    assert at.text_area[0].value == ""
    assert at.session_state["cached_jd_text"] == ""

    # 2. Verify cached_jd_text populated in session state also clears cleanly
    at.session_state["cached_jd_text"] = "Cached Sample JD Content"
    at.session_state["job_description_input"] = "Cached Sample JD Content"
    at.run()
    assert at.text_area[0].value == "Cached Sample JD Content"

    clear_btn = next(b for b in at.button if b.label == "🧹 Clear JD")
    clear_btn.click().run()
    assert not at.exception
    assert at.text_area[0].value == ""
    assert at.session_state["cached_jd_text"] == ""

    # 3. Navigate away and back to verify cleared state persists and other workflow state stays intact
    at.sidebar.radio[0].set_value("📊 Executive Dashboard").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("🎯 Job Matcher & ATS").run()
    assert not at.exception
    assert at.text_area[0].value == ""
    assert at.session_state["active_resume_data"]["candidate_name"] == "Alex Morgan"

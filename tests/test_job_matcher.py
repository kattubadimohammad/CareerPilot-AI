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

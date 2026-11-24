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

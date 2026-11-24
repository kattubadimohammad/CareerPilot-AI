"""Tests for Resume Analyzer."""
import pytest
from src.resume_analyzer import ResumeAnalyzer

SAMPLE_RESUME_TEXT = """
Alex Morgan
alex.morgan@email.com | (555) 987-6543 | San Francisco, CA

Professional Summary:
Versatile Senior Software Engineer with 6+ years of experience designing scalable distributed systems.

Technical Skills:
Python, FastAPI, Docker, Kubernetes, PostgreSQL, AWS, Redis, Git, Unit Testing, CI/CD

Work Experience:
Senior Software Engineer - CloudScale Inc (2021 - Present)
- Architected microservices platform handling 15M daily requests, reducing response latency by 35%.
- Implemented automated CI/CD pipeline using GitHub Actions, cutting deployment time from 45 minutes to 8 minutes.
- Spearheaded database migration to PostgreSQL, cutting infrastructure costs by $120,000 annually.

Software Engineer - DataCorp (2018 - 2021)
- Developed RESTful APIs using Python and Flask serving 500k active users.
- Automated ETL pipelines with Pandas and Redis, improving data processing throughput by 40%.

Education:
Bachelor of Science in Computer Science - University of California, Berkeley (2018)
"""


def test_extract_skills_deterministic():
    skills = ResumeAnalyzer.extract_skills_deterministic(SAMPLE_RESUME_TEXT)
    assert "Python" in skills
    assert "Docker" in skills
    assert "Kubernetes" in skills
    assert "PostgreSQL" in skills
    assert "AWS" in skills
    assert "FastAPI" in skills


def test_extract_education_deterministic():
    edu = ResumeAnalyzer.extract_education_deterministic(SAMPLE_RESUME_TEXT)
    assert len(edu) >= 1
    assert any("Computer Science" in e["degree"] for e in edu)


def test_extract_experience_deterministic():
    exp = ResumeAnalyzer.extract_experience_deterministic(SAMPLE_RESUME_TEXT)
    assert len(exp) >= 1
    roles = [e["role"] for e in exp]
    assert any("Senior Software Engineer" in r or "Software Engineer" in r for r in roles)


def test_evaluate_strengths_and_weaknesses():
    skills = ResumeAnalyzer.extract_skills_deterministic(SAMPLE_RESUME_TEXT)
    exp = ResumeAnalyzer.extract_experience_deterministic(SAMPLE_RESUME_TEXT)
    strengths, weaknesses, feedback, score = ResumeAnalyzer.evaluate_strengths_and_weaknesses(
        SAMPLE_RESUME_TEXT, skills, exp
    )
    assert score >= 75
    assert len(strengths) >= 2
    assert any("quantification" in s.lower() or "metrics" in s.lower() for s in strengths)


def test_full_resume_analysis_fallback():
    analyzer = ResumeAnalyzer()
    result = analyzer.analyze_resume(SAMPLE_RESUME_TEXT, "alex_resume.pdf")

    assert result["candidate_name"] == "Alex Morgan"
    assert result["email"] == "alex.morgan@email.com"
    assert "Python" in result["skills"]
    assert result["score"] >= 70
    assert result["is_ai_generated"] is False
    assert "Deterministic" in result["provider_note"]

"""Tests for Database persistence layer."""
import gc
import tempfile
import pytest
from pathlib import Path
from src.database import Database


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    db = Database(db_path=db_path)
    yield db
    gc.collect()
    try:
        if db_path.exists():
            db_path.unlink()
        Path(temp_dir).rmdir()
    except Exception:
        pass



def test_save_and_get_resume(temp_db):
    parsed = {
        "candidate_name": "Jane Doe",
        "email": "jane@example.com",
        "skills": ["Python", "Docker", "SQL"],
        "education": [{"degree": "B.S. Computer Science", "school": "MIT", "year": "2020"}],
        "experience": [{"role": "Software Engineer", "company": "Tech Corp", "years": "3"}],
        "strengths": ["Strong backend fundamentals"],
        "weaknesses": ["Limited cloud experience"],
        "feedback": ["Highlight system design achievements"],
        "score": 85,
    }
    resume_id = temp_db.save_resume(
        filename="jane_resume.pdf",
        file_type="pdf",
        raw_text="Jane Doe\nSoftware Engineer\nPython Docker SQL",
        candidate_name="Jane Doe",
        parsed_data=parsed,
    )
    assert resume_id > 0

    fetched = temp_db.get_resume(resume_id)
    assert fetched is not None
    assert fetched["candidate_name"] == "Jane Doe"
    assert fetched["parsed_data"]["skills"] == ["Python", "Docker", "SQL"]
    assert fetched["parsed_data"]["score"] == 85


def test_list_and_delete_resume(temp_db):
    temp_db.save_resume(
        filename="test1.pdf",
        file_type="pdf",
        raw_text="Test 1",
        candidate_name="User 1",
        parsed_data={"skills": ["Python"]},
    )
    temp_db.save_resume(
        filename="test2.docx",
        file_type="docx",
        raw_text="Test 2",
        candidate_name="User 2",
        parsed_data={"skills": ["Go"]},
    )

    resumes = temp_db.list_resumes()
    assert len(resumes) == 2

    # Delete first
    first_id = resumes[0]["id"]
    deleted = temp_db.delete_resume(first_id)
    assert deleted is True
    assert len(temp_db.list_resumes()) == 1


def test_job_match_crud(temp_db):
    res_id = temp_db.save_resume(
        filename="job_cand.pdf",
        file_type="pdf",
        raw_text="Python Engineer",
        candidate_name="Python Dev",
        parsed_data={"skills": ["Python"]},
    )
    match_id = temp_db.save_job_match(
        resume_id=res_id,
        job_title="Senior Python Engineer",
        job_description="Seeking Python and AWS expert",
        match_score=78.5,
        match_data={
            "matched_skills": ["Python"],
            "missing_skills": ["AWS"],
            "recommendations": ["Learn AWS EC2 and S3"],
        },
    )
    assert match_id > 0
    fetched = temp_db.get_job_match(match_id)
    assert fetched is not None
    assert fetched["job_title"] == "Senior Python Engineer"
    assert fetched["match_score"] == 78.5
    assert fetched["match_data"]["missing_skills"] == ["AWS"]


def test_interview_session_and_roadmap_crud(temp_db):
    res_id = temp_db.save_resume(
        filename="interview_cand.pdf",
        file_type="pdf",
        raw_text="Data Scientist",
        candidate_name="Data Sci",
        parsed_data={"skills": ["Python", "SQL"]},
    )
    session_id = temp_db.save_interview_session(
        resume_id=res_id,
        target_role="Data Scientist",
        session_data={
            "questions": ["What is regularization?"],
            "answers": [{"relevance": 90, "clarity": 85, "completeness": 80}],
        },
        overall_score=85.0,
    )
    assert session_id > 0
    interview = temp_db.get_interview_session(session_id)
    assert interview["target_role"] == "Data Scientist"

    roadmap_id = temp_db.save_career_roadmap(
        resume_id=res_id,
        target_role="MLOps Engineer",
        timeline_months=6,
        roadmap_data={"milestones": ["Month 1: CI/CD"]},
    )
    assert roadmap_id > 0
    roadmap = temp_db.get_career_roadmap(roadmap_id)
    assert roadmap["target_role"] == "MLOps Engineer"

    stats = temp_db.get_dashboard_stats()
    assert stats["matches_count"] == 0
    assert stats["interviews_count"] == 1
    assert stats["avg_interview_score"] == 85.0
    assert stats["roadmaps_count"] == 1


def test_delete_resume_and_associated_cascade(temp_db):
    res_id = temp_db.save_resume(
        filename="cascade_test.pdf",
        file_type="pdf",
        raw_text="Test",
        candidate_name="Cascade User",
        parsed_data={"skills": ["Python"]},
    )
    temp_db.save_job_match(
        resume_id=res_id,
        job_title="Engineer",
        job_description="Python",
        match_score=90.0,
        match_data={},
    )
    temp_db.save_interview_session(
        resume_id=res_id,
        target_role="Engineer",
        session_data={},
        overall_score=80.0,
    )
    temp_db.save_career_roadmap(
        resume_id=res_id,
        target_role="Engineer",
        timeline_months=6,
        roadmap_data={},
    )

    result = temp_db.delete_resume_and_associated(res_id)
    assert result["resumes_deleted"] == 1
    assert result["matches_deleted"] == 1
    assert result["interviews_deleted"] == 1
    assert result["roadmaps_deleted"] == 1

    assert temp_db.get_resume(res_id) is None


def test_sqlite_pragmas(temp_db):
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal = cursor.fetchone()[0]
        assert journal.lower() == "wal"

        cursor.execute("PRAGMA foreign_keys;")
        fk = cursor.fetchone()[0]
        assert fk == 1


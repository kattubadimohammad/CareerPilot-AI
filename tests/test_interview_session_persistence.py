"""Regression tests for complete interview-session persistence."""
import tempfile
from pathlib import Path

from src.database import Database


def test_completed_interview_persists_questions_answers_and_evaluations():
    with tempfile.TemporaryDirectory() as temp_dir:
        db = Database(Path(temp_dir) / "careerpilot.db")
        resume_id = db.save_resume(
            filename="candidate.pdf",
            file_type="pdf",
            raw_text="Python SQL analyst",
            candidate_name="Test Candidate",
            parsed_data={"skills": ["Python", "SQL"]},
        )
        questions = [
            {"id": 1, "question": "Explain data leakage."},
            {"id": 2, "question": "How do you monitor drift?"},
        ]
        answers = {
            "0": "I use chronological validation to prevent leakage.",
            "1": "I monitor feature distributions and retrain after validation.",
        }
        evaluations = {
            "0": {"overall_score": 80.0, "relevance_score": 80},
            "1": {"overall_score": 90.0, "relevance_score": 90},
        }

        session_id = db.save_interview_session(
            resume_id=resume_id,
            target_role="Data Scientist / ML Engineer",
            session_data={
                "questions": questions,
                "answers": answers,
                "evaluations": evaluations,
            },
            overall_score=85.0,
        )

        saved = db.get_interview_session(session_id)
        assert saved is not None
        assert saved["session_data"]["questions"] == questions
        assert saved["session_data"]["answers"] == answers
        assert saved["session_data"]["evaluations"] == evaluations
        assert saved["overall_score"] == 85.0

        stats = db.get_dashboard_stats()
        assert stats["interviews_count"] == 1
        assert stats["avg_interview_score"] == 85.0

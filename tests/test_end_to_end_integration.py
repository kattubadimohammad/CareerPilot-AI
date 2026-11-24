"""End-to-end integration test exercising all CareerPilot AI core capabilities."""
import tempfile
import pytest
from pathlib import Path

from src.database import Database
from src.document_parser import DocumentParser
from src.resume_analyzer import ResumeAnalyzer
from src.job_matcher import JobMatcher
from src.interview_coach import InterviewCoach
from src.career_roadmap import CareerRoadmapGenerator

SAMPLE_RESUME = """
Jordan Lee
jordan.lee@example.com | (555) 123-4567 | New York, NY
Summary: Experienced Data and Backend Engineer skilled in distributed data pipelines.
Skills: Python, SQL, PostgreSQL, Docker, Redis, Pandas, Git, FastAPI, AWS
Experience:
Senior Data Engineer | MegaCorp (2021 - Present)
- Architected ETL pipeline processing 50M records daily using Python and PostgreSQL.
- Reduced query latency by 40% using Redis caching and index optimization.
- Led migration of 12 microservices to AWS with Docker containers.
Education:
Bachelor of Science in Computer Science | Stanford University (2020)
"""

SAMPLE_JOB = """
Role: Senior Cloud Data Engineer
Requirements:
- 4+ years software engineering experience with Python and SQL.
- Strong hands-on experience with AWS, Docker, Kubernetes, and PostgreSQL.
- Experience with Redis, Kafka, and data pipelines.
- Excellent communication and problem-solving skills.
"""


def test_full_careerpilot_flow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_careerpilot.db"
        db = Database(db_path=db_path)

        # 1. Document Extraction & Contact Parsing
        contact = DocumentParser.quick_contact_extraction(SAMPLE_RESUME)
        assert contact["name"] == "Jordan Lee"
        assert contact["email"] == "jordan.lee@example.com"

        # 2. Resume Intelligence Analysis
        analyzer = ResumeAnalyzer()
        resume_analysis = analyzer.analyze_resume(SAMPLE_RESUME, "jordan_resume.pdf")
        assert "Python" in resume_analysis["skills"]
        assert resume_analysis["score"] >= 70

        # Save to DB
        res_id = db.save_resume(
            filename="jordan_resume.pdf",
            file_type="pdf",
            raw_text=SAMPLE_RESUME,
            candidate_name=resume_analysis["candidate_name"],
            parsed_data=resume_analysis,
        )
        assert res_id > 0

        # 3. Job Matcher & ATS Optimization
        matcher = JobMatcher()
        match_result = matcher.match_resume_to_job(
            resume_text=SAMPLE_RESUME,
            resume_skills=resume_analysis["skills"],
            job_text=SAMPLE_JOB,
            job_title="Senior Cloud Data Engineer",
        )
        assert match_result["overall_score"] > 50
        assert "Python" in match_result["matched_skills"]
        assert "Kubernetes" in match_result["missing_skills"]
        assert "ETHICAL ATS OPTIMIZATION GUARDRAIL" in match_result["ethical_notice"]

        # Save Job Match to DB
        match_id = db.save_job_match(
            resume_id=res_id,
            job_title="Senior Cloud Data Engineer",
            job_description=SAMPLE_JOB,
            match_score=match_result["overall_score"],
            match_data=match_result,
        )
        assert match_id > 0

        # 4. Interview Coach
        coach = InterviewCoach()
        questions = coach.get_questions_for_role("Senior Cloud Data Engineer", skills=resume_analysis["skills"], count=2)
        assert len(questions) == 2

        answer = (
            "In my previous role at MegaCorp, our ETL pipeline began timing out due to 3x data growth. "
            "I was tasked with restoring our SLA of sub-hour completions. "
            "I implemented parallel partition ingestion with Redis caching and rewritten PostgreSQL queries. "
            "Consequently, pipeline execution time dropped by 40% and we met all customer SLAs."
        )
        eval_result = coach.evaluate_answer(questions[0]["question"], answer, role="Senior Cloud Data Engineer")
        assert eval_result["overall_score"] >= 60
        assert eval_result["star_analysis"]["action"] is True

        sess_id = db.save_interview_session(
            resume_id=res_id,
            target_role="Senior Cloud Data Engineer",
            session_data={"questions": questions, "evaluations": {0: eval_result}},
            overall_score=eval_result["overall_score"],
        )
        assert sess_id > 0

        # 5. Career Roadmap
        roadmap_gen = CareerRoadmapGenerator()
        roadmap = roadmap_gen.generate_roadmap(
            current_skills=resume_analysis["skills"],
            target_role="Cloud & DevOps Architect",
            timeline_months=6,
        )
        assert "P0_Critical" in roadmap["learning_priorities"]
        assert len(roadmap["milestones"]) >= 3

        rm_id = db.save_career_roadmap(
            resume_id=res_id,
            target_role="Cloud & DevOps Architect",
            timeline_months=6,
            roadmap_data=roadmap,
        )
        assert rm_id > 0

        # 6. Verify Dashboard Metrics Aggregation
        stats = db.get_dashboard_stats()
        assert stats["resumes_count"] == 1
        assert stats["matches_count"] == 1
        assert stats["interviews_count"] == 1
        assert stats["roadmaps_count"] == 1
        assert stats["avg_match_score"] == match_result["overall_score"]

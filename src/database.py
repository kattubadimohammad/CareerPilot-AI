"""SQLite Database persistence layer for CareerPilot AI."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "careerpilot.db"


class Database:
    """Manages SQLite storage for CareerPilot AI."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def init_db(self) -> None:
        """Create necessary tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Resumes Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    candidate_name TEXT,
                    parsed_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Job Matches Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS job_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    job_title TEXT NOT NULL,
                    job_description TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    match_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE SET NULL
                )
                """
            )

            # Interview Sessions Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    target_role TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    overall_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE SET NULL
                )
                """
            )

            # Career Roadmaps Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS career_roadmaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    target_role TEXT NOT NULL,
                    timeline_months INTEGER DEFAULT 6,
                    roadmap_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE SET NULL
                )
                """
            )
            conn.commit()

    # Resume Operations
    def save_resume(
        self,
        filename: str,
        file_type: str,
        raw_text: str,
        candidate_name: str,
        parsed_data: Dict[str, Any],
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO resumes (filename, file_type, raw_text, candidate_name, parsed_data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    file_type,
                    raw_text,
                    candidate_name,
                    json.dumps(parsed_data),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_resume(self, resume_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["parsed_data"] = json.loads(data["parsed_data"])
            return data

    def list_resumes(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename, file_type, candidate_name, created_at FROM resumes ORDER BY id DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_resume_and_associated(self, resume_id: int) -> Dict[str, int]:
        """Delete a resume and all child records (job matches, interview sessions, roadmaps)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM job_matches WHERE resume_id = ?", (resume_id,))
            matches_deleted = cursor.rowcount
            cursor.execute("DELETE FROM interview_sessions WHERE resume_id = ?", (resume_id,))
            interviews_deleted = cursor.rowcount
            cursor.execute("DELETE FROM career_roadmaps WHERE resume_id = ?", (resume_id,))
            roadmaps_deleted = cursor.rowcount
            cursor.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
            resumes_deleted = cursor.rowcount
            conn.commit()
            return {
                "resumes_deleted": resumes_deleted,
                "matches_deleted": matches_deleted,
                "interviews_deleted": interviews_deleted,
                "roadmaps_deleted": roadmaps_deleted,
            }

    def delete_resume(self, resume_id: int) -> bool:
        """Backwards-compatible resume deletion."""
        res = self.delete_resume_and_associated(resume_id)
        return res["resumes_deleted"] > 0

    # Job Match Operations
    def save_job_match(
        self,
        resume_id: Optional[int],
        job_title: str,
        job_description: str,
        match_score: float,
        match_data: Dict[str, Any],
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO job_matches (resume_id, job_title, job_description, match_score, match_data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resume_id,
                    job_title,
                    job_description,
                    match_score,
                    json.dumps(match_data),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_job_match(self, match_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM job_matches WHERE id = ?", (match_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["match_data"] = json.loads(data["match_data"])
            return data

    def list_job_matches(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, resume_id, job_title, match_score, created_at 
                FROM job_matches 
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # Interview Operations
    def save_interview_session(
        self,
        resume_id: Optional[int],
        target_role: str,
        session_data: Dict[str, Any],
        overall_score: float = 0.0,
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interview_sessions (resume_id, target_role, session_data, overall_score)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resume_id,
                    target_role,
                    json.dumps(session_data),
                    overall_score,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_interview_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interview_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["session_data"] = json.loads(data["session_data"])
            return data

    def list_interview_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, resume_id, target_role, overall_score, created_at 
                FROM interview_sessions 
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # Career Roadmap Operations
    def save_career_roadmap(
        self,
        resume_id: Optional[int],
        target_role: str,
        timeline_months: int,
        roadmap_data: Dict[str, Any],
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO career_roadmaps (resume_id, target_role, timeline_months, roadmap_data)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resume_id,
                    target_role,
                    timeline_months,
                    json.dumps(roadmap_data),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_career_roadmap(self, roadmap_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM career_roadmaps WHERE id = ?", (roadmap_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["roadmap_data"] = json.loads(data["roadmap_data"])
            return data

    def list_career_roadmaps(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, resume_id, target_role, timeline_months, created_at 
                FROM career_roadmaps 
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Aggregate counts and averages for dashboard reporting."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM resumes")
            resumes_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*), AVG(match_score) FROM job_matches")
            row = cursor.fetchone()
            matches_count = row[0]
            avg_match_score = round(row[1], 1) if row[1] is not None else 0.0

            cursor.execute("SELECT COUNT(*), AVG(overall_score) FROM interview_sessions")
            row = cursor.fetchone()
            interviews_count = row[0]
            avg_interview_score = round(row[1], 1) if row[1] is not None else 0.0

            cursor.execute("SELECT COUNT(*) FROM career_roadmaps")
            roadmaps_count = cursor.fetchone()[0]

            return {
                "resumes_count": resumes_count,
                "matches_count": matches_count,
                "avg_match_score": avg_match_score,
                "interviews_count": interviews_count,
                "avg_interview_score": avg_interview_score,
                "roadmaps_count": roadmaps_count,
            }

"""Tests for Career Roadmap Generator."""
import pytest
from src.career_roadmap import CareerRoadmapGenerator


def test_identify_skill_gaps():
    current_skills = ["Python", "FastAPI", "Docker"]
    gaps = CareerRoadmapGenerator.identify_skill_gaps(current_skills, "Senior Backend Engineer")

    assert "PostgreSQL" in gaps["missing_skills"] or "Kubernetes" in gaps["missing_skills"]
    assert "Python" in gaps["existing_skills"]
    assert "P0_Critical" in gaps["learning_priorities"]
    assert len(gaps["milestones"]) >= 3


def test_generate_roadmap_pipeline():
    generator = CareerRoadmapGenerator()
    roadmap = generator.generate_roadmap(
        current_skills=["Python", "SQL", "Pandas"],
        target_role="AI / Machine Learning Engineer",
        timeline_months=6,
    )

    assert roadmap["target_role"] == "AI / Machine Learning Engineer"
    assert "P0_Critical" in roadmap["learning_priorities"]
    assert len(roadmap["milestones"]) >= 3
    assert roadmap["is_ai_generated"] is False
    assert "Deterministic" in roadmap["provider_note"]

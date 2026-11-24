"""Tests for Interview Coach."""
import pytest
from src.interview_coach import InterviewCoach


def test_get_questions_for_role():
    coach = InterviewCoach()
    questions = coach.get_questions_for_role("Backend Engineer", count=3)
    assert len(questions) == 3
    assert any("API" in q["question"] or "database" in q["question"].lower() for q in questions)


def test_evaluate_answer_deterministic_brief():
    coach = InterviewCoach()
    evaluation = coach.evaluate_answer_deterministic(
        question="How do you handle API scaling?",
        user_answer="I use Redis cache.",
        role="Backend Engineer",
    )
    assert evaluation["overall_score"] < 40
    assert any("too brief" in f.lower() for f in evaluation["feedback"])


def test_evaluate_answer_deterministic_star():
    coach = InterviewCoach()
    user_answer = (
        "At my previous company, our payment microservice experienced sudden 5x traffic surges during holiday sales. "
        "My task was to eliminate the database timeouts causing transaction failures. "
        "I implemented Redis caching for user sessions and added an asynchronous Kafka queue for billing workers. "
        "As a result, API response latency decreased by 45% and system uptime remained at 99.99% throughout peak week."
    )
    evaluation = coach.evaluate_answer_deterministic(
        question="Describe a scenario where you optimized API performance under heavy load.",
        user_answer=user_answer,
        role="Backend Engineer",
    )
    assert evaluation["overall_score"] >= 75
    assert evaluation["star_analysis"]["situation"] is True
    assert evaluation["star_analysis"]["action"] is True
    assert evaluation["star_analysis"]["result"] is True
    assert evaluation["is_ai_generated"] is False

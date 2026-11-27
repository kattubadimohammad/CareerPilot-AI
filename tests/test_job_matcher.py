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


def test_job_title_detection_explicit_inferred_and_missing():
    """Verify explicit, JD-labeled, inferred, and missing job titles do not get blocked by default placeholder."""
    matcher = JobMatcher()

    # 1. Explicit passed title always takes precedence over JD body or label
    explicit = JobMatcher._resolve_job_title("Title: Senior Director\nRequirements: SQL", "Data Analyst")
    assert explicit == "Data Analyst"

    # 2. Default 'Target Position' must NOT block extraction of explicit JD title labels (e.g. Title:, Position:)
    jd_labeled = "Title: Senior Supply Chain Analyst\nCompany: Global Retail\nRequires Python, SQL."
    res_labeled = matcher.match_resume_to_job("Analyst with 2 years of experience. Python SQL.", ["Python", "SQL"], jd_labeled)
    assert res_labeled["job_title"] == "Senior Supply Chain Analyst"
    assert res_labeled["job_title_used_for_seniority"] == "Senior Supply Chain Analyst"
    assert res_labeled["sub_scores"]["seniority_experience_alignment"] == 75.0

    # 3. Inferred from standalone heading or hiring phrase when placeholder is used
    jd_heading = "# Lead Data Engineer\nCompany: Cloud Data Inc.\nRequires Python and SQL."
    assert JobMatcher._resolve_job_title(jd_heading, "Target Position") == "Lead Data Engineer"

    jd_hiring = "We are seeking a Principal Analytics Engineer to build scalable ETL workflows."
    assert JobMatcher._resolve_job_title(jd_hiring, "Target Position") == "Principal Analytics Engineer"

    # 4. Missing title: body mentions ('reports to senior manager') must NOT trigger false seniority
    jd_missing = "The analyst reports to a senior manager.\nRequirements: Python, SQL."
    res_missing = matcher.match_resume_to_job("Analyst with Python and SQL.", ["Python", "SQL"], jd_missing)
    assert res_missing["job_title"] == "Target Position"
    assert res_missing["job_title_used_for_seniority"] == ""
    assert res_missing["sub_scores"]["seniority_experience_alignment"] == 90.0


def test_safe_ai_rewrites_rejects_all_unsupported_claim_categories():
    """Ensure fail-closed validation rejects unsupported tools, employers, qualifications, dates, metrics, and achievements."""
    resume = (
        "Data Operations Analyst at Flipkart (Sep 2024 - Sep 2026).\n"
        "Built MIS reports and KPI dashboards using Advanced Excel and SQL across 16+ product verticals, "
        "improving Instock availability to 98% and reducing stockouts by 25%. "
        "Managed inventory for 25 units in pilot testing."
    )

    unsupported_rewrites = [
        # Unsupported tools (in-taxonomy and out-of-taxonomy)
        {"original_theme": "Tool1", "suggested_rewrite": "Built Snowflake and Looker dashboards using SQL.", "justification": "t"},
        {"original_theme": "Tool2", "suggested_rewrite": "Orchestrated SAP HANA procurement workflows using Excel.", "justification": "t"},
        # Unsupported employers
        {"original_theme": "Emp1", "suggested_rewrite": "Built SQL dashboards at Amazon Logistics.", "justification": "e"},
        {"original_theme": "Emp2", "suggested_rewrite": "Supported KPI reporting for Horizon Systems Inc.", "justification": "e"},
        # Unsupported qualifications / credentials
        {"original_theme": "Qual1", "suggested_rewrite": "Applied PMP certification to build KPI dashboards using SQL.", "justification": "q"},
        {"original_theme": "Qual2", "suggested_rewrite": "MBA graduate who built SQL KPI dashboards.", "justification": "q"},
        {"original_theme": "Qual3", "suggested_rewrite": "AWS Certified analyst maintaining Excel reports.", "justification": "q"},
        # Unsupported dates / durations
        {"original_theme": "Date1", "suggested_rewrite": "Built SQL KPI dashboards from January 2019 to December 2020.", "justification": "d"},
        {"original_theme": "Date2", "suggested_rewrite": "Delivered Q4 Excel dashboards for Flipkart.", "justification": "d"},
        {"original_theme": "Date3", "suggested_rewrite": "Built SQL dashboards over 5 years.", "justification": "d"},
        # Unsupported metrics / scale mismatches
        {"original_theme": "Met1", "suggested_rewrite": "Reduced stockouts by 45% using SQL.", "justification": "m"},
        {"original_theme": "Met2", "suggested_rewrite": "Managed 25% growth across pilot testing.", "justification": "m"},  # 25 units != 25%
        {"original_theme": "Met3", "suggested_rewrite": "Managed 16+ engineers using Advanced Excel.", "justification": "m"},  # 16+ verticals != engineers
        {"original_theme": "Met4", "suggested_rewrite": "Doubled Instock availability using SQL.", "justification": "m"},
        # Unsupported achievements / lowercase fabrication
        {"original_theme": "Ach1", "suggested_rewrite": "Negotiated supplier contracts achieving revenue savings using SQL.", "justification": "a"},
        {"original_theme": "Ach2", "suggested_rewrite": "Led global enterprise cost transformation with Advanced Excel.", "justification": "a"},
    ]

    filtered = JobMatcher._safe_ai_rewrites(unsupported_rewrites, resume)
    assert filtered == []


def test_safe_ai_rewrites_preserves_fully_grounded_rewrites():
    """Verify faithful rewrites using only resume-verified facts, tools, employers, and metrics pass."""
    resume = (
        "Data Operations Analyst at Flipkart (Sep 2024 - Sep 2026).\n"
        "Built MIS reports and KPI dashboards using Advanced Excel and SQL across 16+ product verticals, "
        "improving Instock availability to 98% and reducing stockouts by 25%."
    )
    faithful_rewrites = [
        {
            "original_theme": "Inventory KPI Reporting",
            "suggested_rewrite": "Built MIS reports and KPI dashboards at Flipkart using Advanced Excel and SQL across 16+ product verticals, improving Instock availability to 98% and reducing stockouts by 25%.",
            "justification": "Highlights verified KPI impact and tools.",
        }
    ]
    filtered = JobMatcher._safe_ai_rewrites(faithful_rewrites, resume)
    assert len(filtered) == 1
    assert "98%" in filtered[0]["suggested_rewrite"]
    assert "Flipkart" in filtered[0]["suggested_rewrite"]


def test_safe_ai_rewrites_preserves_legitimate_variants_and_abbreviations():
    """Verify legitimate bullets with bullets, plural acronyms, morphological lemmas, slashes, and possessives pass."""
    resume = (
        "Data Operations Analyst at Flipkart (Sep 2024 - Sep 2026).\n"
        "Analyzed operational data with Python, SQL, CI/CD, and REST APIs. "
        "Built KPI dashboards and optimized workflows saving $120,000."
    )
    legitimate = [
        {
            "original_theme": "Morphological & Framing",
            "suggested_rewrite": "• Applied Python/SQL to analyze operational data with REST API.",
            "justification": "Morphological lemma + slash + singular API.",
        },
        {
            "original_theme": "Possessive & Acronym Plural",
            "suggested_rewrite": "Built KPIs dashboard at Flipkart's workflows using optimization for $120000.",
            "justification": "Plural acronym + possessive + comma normalization.",
        },
    ]
    filtered = JobMatcher._safe_ai_rewrites(legitimate, resume)
    assert len(filtered) == 2


def test_safe_ai_rewrites_blocks_strict_bypass_attempts_and_75pct_loopholes():
    """Ensure zero-tolerance grounding closes 75% padding, index-0 employers, alphanumeric tools, and unit swaps."""
    resume = (
        "Analyzed operational datasets using SQL and Excel across 16+ product verticals to monitor inventory and weekly reporting. "
        "Completed pilot in 30 days. Graduated 2020; certified project 2023."
    )
    bypasses = [
        # Long grounded bullet + 2 ungrounded words (previously passed >= 75% threshold)
        {"original_theme": "Padding", "suggested_rewrite": "Analyzed operational datasets using SQL and Excel across 16+ product verticals to monitor inventory, weekly reporting, and automated invoicing.", "justification": "75% pad"},
        # Ungrounded evaluative adjective
        {"original_theme": "Adj", "suggested_rewrite": "Analyzed datasets using accurate scalable SQL reporting.", "justification": "Adjective claim"},
        # Index-0 unsupported employer
        {"original_theme": "Idx0Emp", "suggested_rewrite": "Deloitte analyst analyzing datasets using SQL and Excel.", "justification": "Index 0 employer"},
        # Lowercase employer / lowercase unknown tool / alphanumeric tool code
        {"original_theme": "LowerEmp", "suggested_rewrite": "Analyzed operational datasets using SQL and Excel at amazon.", "justification": "Lowercase employer"},
        {"original_theme": "LowerTool", "suggested_rewrite": "Analyzed operational datasets using SQL, Excel, and duckdb.", "justification": "Lowercase tool"},
        {"original_theme": "Alnum", "suggested_rewrite": "Analyzed operational datasets using SQL, Excel, and s3.", "justification": "Alphanumeric code"},
        # Spelled numbers & swapped unit contexts
        {"original_theme": "Spelled", "suggested_rewrite": "Analyzed ten operational datasets using SQL.", "justification": "Spelled number"},
        {"original_theme": "UnitSwap", "suggested_rewrite": "Delivered 30 dashboards using SQL and Excel.", "justification": "30 days != 30 dashboards"},
        # Ambiguous month May & spliced year range
        {"original_theme": "MayDate", "suggested_rewrite": "Analyzed operational datasets in May using SQL.", "justification": "Month May"},
        {"original_theme": "SplicedDate", "suggested_rewrite": "Analyzed operational datasets 2020-2023 using SQL.", "justification": "Spliced range"},
    ]
    filtered = JobMatcher._safe_ai_rewrites(bypasses, resume)
    assert filtered == []

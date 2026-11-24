"""CareerPilot AI - Main Streamlit Application.
Your Career. Your Potential. Powered by AI.
"""
import html
import io
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.database import Database
from src.document_parser import DocumentParser, DocumentValidationError
from src.gemini_service import (
    GeminiService,
    load_gemini_api_key,
    DEFAULT_MODEL,
)
from src.resume_analyzer import ResumeAnalyzer
from src.job_matcher import JobMatcher
from src.interview_coach import InterviewCoach
from src.career_roadmap import CareerRoadmapGenerator

# Page Configuration
st.set_page_config(
    page_title="CareerPilot AI | Your Career Powered by AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STYLE_PATH = BASE_DIR / "assets" / "style.css"

# Load Custom CSS
if STYLE_PATH.exists():
    with open(STYLE_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Shared Services in Session State
@st.cache_resource
def get_database():
    return Database(DATA_DIR / "careerpilot.db")


db = get_database()

# API Key and Gemini Service Initialization with Secrets Auto-Sync
current_gemini_key = load_gemini_api_key()

if "gemini_service" not in st.session_state:
    st.session_state.gemini_api_key = current_gemini_key
    st.session_state.gemini_service = GeminiService(
        api_key=current_gemini_key,
        model_name=DEFAULT_MODEL,
    )
else:
    # If session state was initialized before secrets were configured or key changed,
    # automatically synchronize without requiring session reset
    if current_gemini_key and (
        not st.session_state.gemini_service.is_configured()
        or st.session_state.gemini_service.api_key != current_gemini_key
    ):
        st.session_state.gemini_api_key = current_gemini_key
        st.session_state.gemini_service.set_api_key(current_gemini_key)

if "active_resume_data" not in st.session_state:
    st.session_state.active_resume_data = None

if "active_resume_text" not in st.session_state:
    st.session_state.active_resume_text = ""

if "active_resume_id" not in st.session_state:
    st.session_state.active_resume_id = None


# Helper to load sample files
def load_sample_file(filename: str) -> bytes:
    filepath = DATA_DIR / filename
    if filepath.exists():
        return filepath.read_bytes()
    return b""


def load_sample_text(filename: str) -> str:
    filepath = DATA_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


# Sidebar Configuration & Navigation
with st.sidebar:
    st.markdown("## 🚀 **CareerPilot AI**")
    st.caption("Your Career. Your Potential. Powered by AI.")
    st.divider()

    # Navigation Menu
    nav_selection = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "📄 Resume Intelligence",
            "🎯 Job Matcher & ATS",
            "🎤 Interview Coach",
            "🗺️ Career Roadmap",
        ],
        index=0,
    )

    st.divider()

    # Gemini API Configuration
    st.markdown("### 🔑 AI Service Settings")
    # Never render the server-side API key in the public app UI.
    # Configure GEMINI_API_KEY through Streamlit Community Cloud Secrets
    # or a local environment variable instead.
    if st.session_state.gemini_service.is_configured():
        st.caption("Gemini API key is securely loaded from server configuration.")
    else:
        st.caption("Gemini API key is not configured. Using deterministic fallback mode.")

    # Model Selection (verified via Google GenAI SDK where possible)
    models_list, is_live_verified = st.session_state.gemini_service.get_available_models()
    current_model = st.session_state.gemini_service.model_name
    if current_model not in models_list:
        current_model = DEFAULT_MODEL
        st.session_state.gemini_service.model_name = current_model

    model_idx = models_list.index(current_model) if current_model in models_list else 0
    selected_model = st.selectbox(
        "Gemini Model",
        models_list,
        index=model_idx,
        help="Select Gemini model. 'gemini-3.8-flash' is the primary model with automatic failover to 'gemini-3.5-flash-lite'.",
    )
    if selected_model != st.session_state.gemini_service.model_name:
        st.session_state.gemini_service.model_name = selected_model

    if is_live_verified:
        st.caption("✅ Live model access confirmed for project.")
    elif st.session_state.gemini_service.is_configured():
        st.caption("ℹ️ Unverified default models shown (click 'Test API' to verify).")

    col_btn, col_badge = st.columns([1, 1])
    with col_btn:
        if st.button("Test API", width="stretch"):
            with st.spinner("Testing..."):
                ok, msg = st.session_state.gemini_service.test_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with col_badge:
        if st.session_state.gemini_service.is_configured():
            st.markdown("<span style='color: #16A34A; font-weight:600;'>● AI Key Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #64748B; font-weight:600;'>○ Fallback Mode</span>", unsafe_allow_html=True)

    if not st.session_state.gemini_service.is_configured():
        with st.expander("ℹ️ Secrets Setup Guide", expanded=False):
            st.markdown(
                """
                **Streamlit Community Cloud:**
                1. Go to **App Settings** > **Secrets**.
                2. Add your key:
                ```toml
                GEMINI_API_KEY = "AIzaSy..."
                ```
                3. Click **Save** then **🔄 Sync Secrets** below.
                """
            )

    with st.expander("🔍 Safe Configuration Diagnostics", expanded=False):
        diag = st.session_state.gemini_service.get_diagnostic_info()
        st.markdown(f"**App Version:** `{diag['git_commit']}`")
        st.markdown(f"**Secrets Accessible:** {'✅ Yes' if diag['st_secrets_available'] else '❌ No'}")
        if diag.get("st_secrets_keys"):
            st.markdown(f"**Secrets Keys Detected:** `{', '.join(diag['st_secrets_keys'])}`")
        if diag.get("st_secrets_error"):
            st.caption(f"⚠️ Secrets Notice: {diag['st_secrets_error']}")
        st.markdown(f"**Config Source Detected:** `{diag['source_detected']}`")
        st.markdown(f"**Key Present:** {'✅ Yes' if diag['key_detected'] else '❌ No'}")
        if diag['key_detected']:
            st.markdown(f"**Key Length:** {diag['key_length']} characters")
            st.markdown(f"**Starts with 'AIza':** {'✅ Yes' if diag['starts_with_aiza'] else 'ℹ️ No (Non-AIza prefix)'}")
        st.markdown(f"**Format Status:** `{diag['format_status']}`")
        st.markdown(f"**Format Validation (`is_configured`):** {'✅ Passed' if diag['is_configured'] else '❌ Failed'}")
        st.markdown(f"**Gemini Client Initialized:** {'✅ Yes' if diag['client_initialized'] else '❌ No'}")
        if diag.get("client_error"):
            st.caption(f"⚠️ Client Notice: {diag['client_error']}")

    st.divider()

    # Database Quick Stats
    stats = db.get_dashboard_stats()
    st.markdown("### 📈 Stored Records")
    st.write(f"- **Resumes Saved:** {stats['resumes_count']}")
    st.write(f"- **Job Matches:** {stats['matches_count']}")
    st.write(f"- **Interview Runs:** {stats['interviews_count']}")
    st.write(f"- **Roadmaps:** {stats['roadmaps_count']}")

    col_sync, col_clear = st.columns([1, 1])
    with col_sync:
        if st.button("🔄 Sync Secrets", width="stretch", help="Reload API key from secrets or environment"):
            reloaded_key = load_gemini_api_key(force_refresh=True)
            st.session_state.gemini_api_key = reloaded_key
            st.session_state.gemini_service.set_api_key(reloaded_key)
            if st.session_state.gemini_service.is_configured():
                st.success("Secrets successfully synchronized and verified!")
            else:
                st.warning("No valid API key detected. If you just added secrets, please wait a moment or reboot the app in Streamlit Cloud Settings.")
            st.rerun()

    with col_clear:
        if st.button("🗑️ Clear Session", width="stretch", help="Reset active resume data and reload secrets"):
            st.session_state.active_resume_data = None
            st.session_state.active_resume_text = ""
            st.session_state.active_resume_id = None
            reloaded_key = load_gemini_api_key(force_refresh=True)
            st.session_state.gemini_api_key = reloaded_key
            st.session_state.gemini_service.set_api_key(reloaded_key)
            st.rerun()


# ==============================================================================
# 1. EXECUTIVE DASHBOARD
# ==============================================================================
if nav_selection == "📊 Executive Dashboard":
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">Welcome to CareerPilot AI</h1>
            <p class="hero-tagline">Comprehensive, explainable AI intelligence for your resume, job fit, interview prep, and career velocity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = db.get_dashboard_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Parsed Resumes</div>
                <div class="metric-value">{stats['resumes_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Job Match</div>
                <div class="metric-value">{stats['avg_match_score']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Interview Score</div>
                <div class="metric-value">{stats['avg_interview_score']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Roadmaps Built</div>
                <div class="metric-value">{stats['roadmaps_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Start Guide & Active Context
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🎯 Quick Start Workflow")
        st.markdown(
            """
            1. **Resume Intelligence**: Upload your resume in PDF or DOCX format (or load our demonstration profile). Get instant skill categorization, strengths, weaknesses, and score.
            2. **Job Matcher & ATS**: Compare your resume against any job description. Uncover matched vs missing skills, keyword frequencies, and ethical ATS optimization advice.
            3. **Interview Coach**: Generate role-specific behavioral (STAR) and technical questions. Receive instant scoring on relevance, clarity, and completeness.
            4. **Career Roadmap**: Map your skill gaps against target roles and receive prioritized P0/P1/P2 milestones with capstone projects.
            """
        )

        st.info("💡 **Tip**: CareerPilot AI operates fully standalone with a deterministic engine and upgrades dynamically to Google Gemini when an API key is provided.")

    with col_right:
        st.subheader("📄 Active Candidate Profile")
        if st.session_state.active_resume_data:
            res_data = st.session_state.active_resume_data
            st.success(f"**Loaded:** {res_data.get('candidate_name', 'Candidate')}")
            st.write(f"- **Email:** {res_data.get('email', 'N/A')}")
            st.write(f"- **Verified Skills:** {len(res_data.get('skills', []))} skills")
            st.write(f"- **Resume Score:** {res_data.get('score', 0)} / 100")
            st.write(f"- **Engine:** {'Gemini AI' if res_data.get('is_ai_generated') else 'Deterministic'}")
        else:
            st.warning("No active resume loaded in this session.")
            if st.button("🚀 Load Sample Resume Now", type="primary", width="stretch"):
                sample_pdf = load_sample_file("sample_resume.pdf")
                if sample_pdf:
                    text, ftype = DocumentParser.extract_text(sample_pdf, "sample_resume.pdf")
                    analyzer = ResumeAnalyzer(st.session_state.gemini_service)
                    parsed = analyzer.analyze_resume(text, "sample_resume.pdf")
                    res_id = db.save_resume(
                        filename="sample_resume.pdf",
                        file_type="pdf",
                        raw_text=text,
                        candidate_name=parsed.get("candidate_name", "Alex Morgan"),
                        parsed_data=parsed,
                    )
                    st.session_state.active_resume_data = parsed
                    st.session_state.active_resume_text = text
                    st.session_state.active_resume_id = res_id
                    st.success("Loaded demonstration resume into active session!")
                    st.rerun()

    # Recent Records Table & Privacy Management
    st.subheader("🕒 Recent Analysis History")
    recent_resumes = db.list_resumes()
    if recent_resumes:
        df_resumes = pd.DataFrame(recent_resumes)
        st.dataframe(df_resumes, width="stretch")

        with st.expander("🛡️ Privacy & Data Management (Delete Resume & Associated Records)"):
            st.caption("Under GDPR / CCPA right-to-be-forgotten, you can permanently purge any stored candidate resume and all linked job matches, interviews, and roadmaps.")
            del_options = {r["id"]: f"#{r['id']} - {r['candidate_name']} ({r['filename']}) - {r['created_at']}" for r in recent_resumes}
            selected_del_id = st.selectbox("Select Resume to Permanently Delete", options=list(del_options.keys()), format_func=lambda x: del_options[x], key="del_resume_select")
            
            confirm_del = st.checkbox("I confirm I want to permanently delete this resume and all associated evaluations.", key="confirm_del_check")
            if st.button("🗑️ Permanently Delete Resume & Associated Records", type="secondary", disabled=not confirm_del):
                deleted_counts = db.delete_resume_and_associated(selected_del_id)
                if st.session_state.active_resume_id == selected_del_id:
                    st.session_state.active_resume_data = None
                    st.session_state.active_resume_text = ""
                    st.session_state.active_resume_id = None
                st.success(f"Permanently purged Resume #{selected_del_id} and all associated records: {deleted_counts['matches_deleted']} job matches, {deleted_counts['interviews_deleted']} interviews, {deleted_counts['roadmaps_deleted']} roadmaps.")
                st.rerun()
    else:
        st.caption("No historical records saved yet. Start by parsing a resume.")


# ==============================================================================
# 2. RESUME INTELLIGENCE
# ==============================================================================
elif nav_selection == "📄 Resume Intelligence":
    st.header("📄 Resume Intelligence")
    st.caption("Upload a resume in PDF or DOCX format for deep skill extraction, impact scoring, and actionable feedback.")

    col_upload, col_demo = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF or DOCX)",
            type=["pdf", "docx"],
            help="Supported formats: .pdf, .docx. Maximum file size: 10MB.",
        )

    with col_demo:
        st.write("Or test instantly:")
        load_demo_pdf = st.button("📋 Load Sample PDF", width="stretch")
        load_demo_docx = st.button("📋 Load Sample DOCX", width="stretch")

    file_bytes = None
    filename = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
    elif load_demo_pdf:
        file_bytes = load_sample_file("sample_resume.pdf")
        filename = "sample_resume.pdf"
    elif load_demo_docx:
        file_bytes = load_sample_file("sample_resume.docx")
        filename = "sample_resume.docx"

    if file_bytes and filename:
        try:
            with st.spinner("Extracting and parsing document text..."):
                raw_text, file_type = DocumentParser.extract_text(file_bytes, filename)
                analyzer = ResumeAnalyzer(st.session_state.gemini_service)
                analysis_result = analyzer.analyze_resume(raw_text, filename)

                # Persist to SQLite
                res_id = db.save_resume(
                    filename=filename,
                    file_type=file_type,
                    raw_text=raw_text,
                    candidate_name=analysis_result.get("candidate_name", "Candidate"),
                    parsed_data=analysis_result,
                )

                st.session_state.active_resume_data = analysis_result
                st.session_state.active_resume_text = raw_text
                st.session_state.active_resume_id = res_id
                st.success(f"Successfully parsed and saved: **{filename}** (ID #{res_id})")

        except DocumentValidationError as dve:
            st.error(f"Validation Error: {str(dve)}")
        except Exception as e:
            st.error(f"Unexpected parsing error: {str(e)}")

    # Display Active Resume Analysis
    if st.session_state.active_resume_data:
        data = st.session_state.active_resume_data
        st.divider()

        # Engine Badge
        if data.get("is_ai_generated"):
            st.markdown(
                f"<div class='provider-badge-ai'>✨ {html.escape(str(data.get('provider_note', 'Generated with Gemini')))}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='provider-badge-fallback'>⚙️ {html.escape(str(data.get('provider_note', 'Deterministic Fallback')))}</div>",
                unsafe_allow_html=True,
            )

        # Candidate Header & Score
        col_cand, col_score = st.columns([3, 1])
        with col_cand:
            st.subheader(f"👤 {data.get('candidate_name', 'Candidate')}")
            contact_line = []
            if data.get("email"):
                contact_line.append(f"📧 {data['email']}")
            if data.get("phone"):
                contact_line.append(f"📞 {data['phone']}")
            if contact_line:
                st.markdown(" • ".join(contact_line))
            if data.get("summary"):
                st.markdown(f"*{data['summary']}*")

        with col_score:
            score = data.get("score", 75)
            st.metric("Resume Score", f"{score}/100")
            st.progress(score / 100)

        # Tabbed Details
        tab_skills, tab_exp_edu, tab_critique, tab_raw = st.tabs(
            ["🛠️ Verified Skills", "💼 Experience & Education", "💡 Strengths & Feedback", "📝 Raw Extracted Text"]
        )

        with tab_skills:
            st.write(f"**Identified Skills ({len(data.get('skills', []))}):**")
            pills_html = " ".join(
                [f"<span class='skill-pill pill-neutral'>{html.escape(str(s))}</span>" for s in data.get("skills", [])]
            )
            st.markdown(pills_html if pills_html else "*No explicit skills detected.*", unsafe_allow_html=True)

        with tab_exp_edu:
            col_exp, col_edu = st.columns(2)
            with col_exp:
                st.markdown("#### 💼 Experience Entries")
                exp_list = data.get("experience", [])
                if exp_list:
                    for exp in exp_list:
                        with st.expander(f"{exp.get('role', 'Role')} - {exp.get('company', 'Company')}", expanded=True):
                            st.write(exp.get("highlights", ""))
                else:
                    st.caption("No structured experience entries identified.")

            with col_edu:
                st.markdown("#### 🎓 Education Entries")
                edu_list = data.get("education", [])
                if edu_list:
                    for edu in edu_list:
                        st.markdown(f"- **{edu.get('degree', 'Degree')}**")
                        st.caption(f"{edu.get('institution', 'Institution')} • {edu.get('year', 'Year')}")
                else:
                    st.caption("No structured education entries identified.")

        with tab_critique:
            col_str, col_weak = st.columns(2)
            with col_str:
                st.markdown("#### ✅ Strengths")
                for s in data.get("strengths", []):
                    st.markdown(f"- {s}")

            with col_weak:
                st.markdown("#### ⚠️ Areas for Improvement")
                for w in data.get("weaknesses", []):
                    st.markdown(f"- {w}")

            st.markdown("#### 🎯 Actionable Recommendations")
            for f in data.get("feedback", []):
                st.info(f)

        with tab_raw:
            st.text_area("Full Document Text", st.session_state.active_resume_text, height=300)


# ==============================================================================
# 3. JOB MATCHER & ATS OPTIMIZER
# ==============================================================================
elif nav_selection == "🎯 Job Matcher & ATS":
    st.header("🎯 Job Matcher & ATS Optimizer")
    st.caption("Compare your resume against any target job description. Identify keyword gaps and receive honest ATS optimization advice.")

    # Guardrail Callout
    st.markdown(
        """
        <div class="guardrail-box">
            <strong>🛡️ Ethical ATS Optimization Guardrail:</strong>
            CareerPilot AI strictly advises against fabricating or exaggerating qualifications.
            Only incorporate keywords and competencies that truthfully represent your experience.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Candidate Resume Selection
    resumes_list = db.list_resumes()
    selected_resume_id = None

    if resumes_list:
        options = {r["id"]: f"#{r['id']} - {r['candidate_name']} ({r['filename']})" for r in resumes_list}
        default_index = 0
        if st.session_state.active_resume_id in options:
            default_index = list(options.keys()).index(st.session_state.active_resume_id)

        selected_resume_id = st.selectbox(
            "Select Resume for Job Comparison",
            options=list(options.keys()),
            format_func=lambda x: options[x],
            index=default_index,
        )
    else:
        st.warning("Please upload or load a resume first in the Resume Intelligence tab.")

    col_jd_head, col_demo_job = st.columns([3, 1])
    with col_jd_head:
        target_title = st.text_input("Target Job Title", value="Senior Backend Cloud Engineer")
    with col_demo_job:
        st.write(" ")
        load_demo_jd = st.button("📋 Load Sample JD", width="stretch")

    if "cached_jd_text" not in st.session_state:
        st.session_state.cached_jd_text = ""

    if load_demo_jd:
        st.session_state.cached_jd_text = load_sample_text("sample_job.txt")
        st.rerun()

    job_description_text = st.text_area(
        "Paste Job Description",
        value=st.session_state.cached_jd_text,
        height=220,
        placeholder="Paste full job description including requirements and responsibilities here...",
    )

    if st.button("⚡ Run Job & ATS Match", type="primary", width="stretch"):
        if not selected_resume_id:
            st.error("Please select a resume before matching.")
        elif not job_description_text.strip():
            st.error("Please paste or load a job description.")
        else:
            resume_record = db.get_resume(selected_resume_id)
            if not resume_record:
                st.error("Could not fetch selected resume.")
            else:
                with st.spinner("Analyzing skill compatibility and ATS keyword alignment..."):
                    resume_skills = resume_record["parsed_data"].get("skills", [])
                    raw_resume_text = resume_record["raw_text"]

                    matcher = JobMatcher(st.session_state.gemini_service)
                    match_result = matcher.match_resume_to_job(
                        resume_text=raw_resume_text,
                        resume_skills=resume_skills,
                        job_text=job_description_text,
                        job_title=target_title,
                    )

                    # Persist to database
                    match_id = db.save_job_match(
                        resume_id=selected_resume_id,
                        job_title=target_title,
                        job_description=job_description_text,
                        match_score=match_result["overall_score"],
                        match_data=match_result,
                    )

                    st.session_state.latest_match_result = match_result
                    st.success(f"Match analysis completed and saved (Record #{match_id})!")

    # Display Results
    if "latest_match_result" in st.session_state:
        m = st.session_state.latest_match_result
        st.divider()

        # Engine note
        if m.get("is_ai_generated"):
            st.markdown(f"<div class='provider-badge-ai'>✨ {html.escape(str(m.get('provider_note', '')))}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='provider-badge-fallback'>⚙️ {html.escape(str(m.get('provider_note', '')))}</div>", unsafe_allow_html=True)

        # Score and Breakdown Cards
        col_main_score, col_sub1, col_sub2, col_sub3 = st.columns(4)
        with col_main_score:
            st.metric("Compatibility Score", f"{m['overall_score']}%")
            st.progress(m["overall_score"] / 100)

        subs = m.get("sub_scores", {})
        with col_sub1:
            st.metric("Skills Alignment (55%)", f"{subs.get('skills_alignment', 0)}%")
        with col_sub2:
            st.metric("ATS Keyword Overlap (25%)", f"{subs.get('ats_keyword_density', 0)}%")
        with col_sub3:
            st.metric("Seniority Alignment (20%)", f"{subs.get('seniority_experience_alignment', 0)}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Matched vs Missing Skills
        col_matched, col_missing = st.columns(2)
        with col_matched:
            st.markdown(f"#### ✅ Matched Requirements ({len(m.get('matched_skills', []))})")
            if m.get("matched_skills"):
                pills = " ".join([f"<span class='skill-pill pill-matched'>✓ {html.escape(str(s))}</span>" for s in m["matched_skills"]])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("No direct skill matches detected.")

        with col_missing:
            st.markdown(f"#### ⚠️ Missing Requirements ({len(m.get('missing_skills', []))})")
            if m.get("missing_skills"):
                pills = " ".join([f"<span class='skill-pill pill-missing'>✗ {html.escape(str(s))}</span>" for s in m["missing_skills"]])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.success("All target technical skills matched!")

        # High-Frequency ATS Keywords
        st.markdown("#### 🔍 High-Frequency Job Posting Keywords")
        kw_pills = " ".join([f"<span class='skill-pill pill-neutral'>{html.escape(str(kw))}</span>" for kw in m.get("top_keywords", [])])
        st.markdown(kw_pills, unsafe_allow_html=True)

        # Tactical Bullet Rewrites if available
        if m.get("bullet_rewrites"):
            st.markdown("#### ✏️ AI Bullet Point Rewrite Recommendations")
            for rw in m["bullet_rewrites"]:
                with st.expander(f"Improvement Area: {rw.get('original_theme', 'Bullet')}", expanded=True):
                    st.markdown(f"**Recommended Draft:** `{rw.get('suggested_rewrite', '')}`")
                    st.caption(f"**Why this works:** {rw.get('justification', '')}")

        # Actionable Recommendations
        st.markdown("#### 📋 ATS Recommendations")
        for rec in m.get("recommendations", []):
            st.info(rec)


# ==============================================================================
# 4. INTERVIEW COACH
# ==============================================================================
elif nav_selection == "🎤 Interview Coach":
    st.header("🎤 Interview Coach")
    st.caption("Practice realistic behavioral (STAR) and technical questions. Receive instant scoring on relevance, clarity, and completeness.")

    col_role, col_count = st.columns([3, 1])
    with col_role:
        target_role = st.selectbox(
            "Select Target Role",
            [
                "Backend Engineer",
                "Full Stack Developer",
                "Data Scientist / ML Engineer",
                "DevOps / Cloud Engineer",
                "Custom Position",
            ],
            index=0,
        )
        if target_role == "Custom Position":
            target_role = st.text_input("Enter Custom Position Title", "Senior Software Engineer")

    with col_count:
        q_count = st.number_input("Question Count", min_value=1, max_value=5, value=3)

    coach = InterviewCoach(st.session_state.gemini_service)

    if st.button("🎲 Generate Interview Questions", width="stretch"):
        active_skills = []
        if st.session_state.active_resume_data:
            active_skills = st.session_state.active_resume_data.get("skills", [])

        with st.spinner("Generating role-specific interview simulation..."):
            questions = coach.get_questions_for_role(target_role, skills=active_skills, count=q_count)
            st.session_state.current_interview_questions = questions
            st.session_state.current_interview_answers = {}
            st.session_state.current_interview_evaluations = {}

    if "current_interview_questions" in st.session_state:
        questions = st.session_state.current_interview_questions
        st.divider()

        st.subheader(f"Session Questions: {target_role}")

        for idx, q in enumerate(questions):
            st.markdown(f"### Question {idx + 1}: *{q.get('type', 'General')}*")
            st.markdown(f"**{q.get('question')}**")
            if q.get("context"):
                st.caption(f"💡 *Interviewer Context:* {q.get('context')}")

            # Answer input
            ans_key = f"ans_{idx}"
            user_ans = st.text_area(
                f"Your Answer to Question {idx + 1}",
                key=ans_key,
                height=150,
                placeholder="Structure your answer using Context/Situation, Action taken, and Measurable Results...",
            )

            col_eval, _ = st.columns([1, 3])
            with col_eval:
                eval_btn = st.button(f"Submit & Evaluate Q{idx + 1}", key=f"btn_eval_{idx}")

            if eval_btn and user_ans.strip():
                with st.spinner("Evaluating your answer against professional interview rubric..."):
                    evaluation = coach.evaluate_answer(q.get("question"), user_ans, role=target_role)
                    st.session_state.current_interview_evaluations[idx] = evaluation

            # Display evaluation for this question if available
            if idx in st.session_state.get("current_interview_evaluations", {}):
                ev = st.session_state.current_interview_evaluations[idx]
                st.markdown(
                    f"<div class='metric-card' style='margin-top:10px;'>"
                    f"<h4>Assessment Score: {html.escape(str(ev.get('overall_score', 0)))}/100</h4>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Relevance", f"{ev.get('relevance_score', 0)}%")
                c2.metric("Clarity", f"{ev.get('clarity_score', 0)}%")
                c3.metric("Completeness", f"{ev.get('completeness_score', 0)}%")

                # STAR analysis
                star = ev.get("star_analysis", {})
                st.markdown(
                    f"**STAR Elements Detected:** "
                    f"{'✅' if star.get('situation') else '❌'} Situation | "
                    f"{'✅' if star.get('task') else '❌'} Task | "
                    f"{'✅' if star.get('action') else '❌'} Action | "
                    f"{'✅' if star.get('result') else '❌'} Result"
                )

                if ev.get("feedback"):
                    st.markdown("**Coach Feedback:**")
                    for fb in ev["feedback"]:
                        st.info(fb)

                if ev.get("model_answer_summary"):
                    st.success(f"**Ideal Response Approach:** {ev['model_answer_summary']}")

            st.divider()

        # Save session button
        if st.session_state.get("current_interview_evaluations"):
            if st.button("💾 Save Completed Interview Session to Database", type="primary"):
                evals = st.session_state.current_interview_evaluations
                scores = [e.get("overall_score", 0) for e in evals.values()]
                avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

                sess_id = db.save_interview_session(
                    resume_id=st.session_state.active_resume_id,
                    target_role=target_role,
                    session_data={
                        "questions": questions,
                        "evaluations": evals,
                    },
                    overall_score=avg_score,
                )
                st.success(f"Interview session saved successfully (ID #{sess_id}) with overall score of {avg_score}%!")


# ==============================================================================
# 5. CAREER ROADMAP
# ==============================================================================
elif nav_selection == "🗺️ Career Roadmap":
    st.header("🗺️ Career Roadmap & Velocity Planner")
    st.caption("Identify skill gaps, prioritize learning objectives (P0/P1/P2), and chart actionable milestones toward your target role.")

    col_target, col_months = st.columns([3, 1])
    with col_target:
        target_role = st.selectbox(
            "Select Target Role to Benchmark",
            [
                "Senior Backend Engineer",
                "AI / Machine Learning Engineer",
                "Cloud & DevOps Architect",
                "Custom Role",
            ],
            index=0,
        )
        if target_role == "Custom Role":
            target_role = st.text_input("Enter Target Role", "Principal Architect")

    with col_months:
        timeline_months = st.selectbox("Target Horizon", [3, 6, 12], index=1)

    # Active Skills Context
    current_skills = []
    if st.session_state.active_resume_data:
        current_skills = st.session_state.active_resume_data.get("skills", [])
        st.write(f"Using **{len(current_skills)} verified skills** from active resume.")
    else:
        st.caption("No active resume loaded. Evaluating against standard junior-to-mid foundation.")

    roadmap_gen = CareerRoadmapGenerator(st.session_state.gemini_service)

    if st.button("🚀 Generate Career Roadmap", type="primary", width="stretch"):
        with st.spinner("Analyzing skill gaps and building milestone phases..."):
            roadmap = roadmap_gen.generate_roadmap(
                current_skills=current_skills,
                target_role=target_role,
                timeline_months=timeline_months,
            )

            # Save to SQLite
            rm_id = db.save_career_roadmap(
                resume_id=st.session_state.active_resume_id,
                target_role=target_role,
                timeline_months=timeline_months,
                roadmap_data=roadmap,
            )

            st.session_state.current_roadmap = roadmap
            st.success(f"Career roadmap generated and persisted (ID #{rm_id})!")

    # Display Roadmap
    if "current_roadmap" in st.session_state:
        rm = st.session_state.current_roadmap
        st.divider()

        # Engine note
        if rm.get("is_ai_generated"):
            st.markdown(f"<div class='provider-badge-ai'>✨ {html.escape(str(rm.get('provider_note', '')))}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='provider-badge-fallback'>⚙️ {html.escape(str(rm.get('provider_note', '')))}</div>", unsafe_allow_html=True)

        if rm.get("strategic_overview"):
            st.markdown(f"*{rm['strategic_overview']}*")

        # Prioritized Learning Objectives
        st.subheader("🎯 Prioritized Learning Objectives")
        priors = rm.get("learning_priorities", {})

        col_p0, col_p1, col_p2 = st.columns(3)
        with col_p0:
            st.markdown("#### 🔴 P0: Critical Prerequisites")
            p0_items = priors.get("P0_Critical", [])
            if p0_items:
                pills = " ".join([f"<span class='skill-pill pill-priority-p0'>{html.escape(str(s))}</span>" for s in p0_items])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("All core prerequisites fulfilled!")

        with col_p1:
            st.markdown("#### 🟡 P1: High-Value Differentiators")
            p1_items = priors.get("P1_Important", [])
            if p1_items:
                pills = " ".join([f"<span class='skill-pill pill-priority-p1'>{html.escape(str(s))}</span>" for s in p1_items])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("No secondary priorities pending.")

        with col_p2:
            st.markdown("#### 🟢 P2: Specialized Niches")
            p2_items = priors.get("P2_Nice_To_Have", [])
            if p2_items:
                pills = " ".join([f"<span class='skill-pill pill-priority-p2'>{html.escape(str(s))}</span>" for s in p2_items])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("All niche competencies mastered.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Milestone Roadmap
        st.subheader(f"📅 Actionable Development Milestones ({timeline_months} Months)")
        for m_idx, ms in enumerate(rm.get("milestones", [])):
            with st.expander(f"{ms.get('phase', f'Phase {m_idx + 1}')}: {ms.get('focus', '')}", expanded=True):
                st.markdown("**Action Items:**")
                for action in ms.get("action_items", []):
                    st.markdown(f"- [ ] {action}")

                if ms.get("capstone_project"):
                    st.info(f"🏆 **Capstone Project:** {ms['capstone_project']}")

        # Recommended Resources
        if rm.get("recommended_certifications_and_resources"):
            st.subheader("📚 Recommended Certifications & Resources")
            for res in rm["recommended_certifications_and_resources"]:
                st.markdown(f"- 📖 {res}")

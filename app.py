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
from src.gemini_service import GeminiService, load_gemini_api_key_with_source
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

def get_gemini_api_key() -> tuple[str, str]:
    """Load the key and preserve the exact source reported by the service helper."""
    return load_gemini_api_key_with_source()

# Load key securely on every run/rerun to reflect secrets or env changes
current_api_key, current_key_source = get_gemini_api_key()
st.session_state.gemini_api_key = current_api_key
st.session_state.gemini_key_source = current_key_source

if "gemini_service" not in st.session_state:
    st.session_state.gemini_service = GeminiService(api_key=st.session_state.gemini_api_key)
elif st.session_state.gemini_service.api_key != st.session_state.gemini_api_key:
    st.session_state.gemini_service.set_api_key(st.session_state.gemini_api_key)

if "active_resume_data" not in st.session_state:
    st.session_state.active_resume_data = None

if "active_resume_text" not in st.session_state:
    st.session_state.active_resume_text = ""

if "active_resume_id" not in st.session_state:
    st.session_state.active_resume_id = None

if "last_analyzed_file_sig" not in st.session_state:
    st.session_state.last_analyzed_file_sig = None


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

    # Gemini AI Service Settings & Diagnostics (Key loaded securely via Secrets/Env)
    st.markdown("### 🔑 AI Service Settings")

    is_configured = st.session_state.gemini_service.is_configured()
    key_source = st.session_state.get("gemini_key_source", "none")

    # Format validation is not proof of a successful API connection.
    # Safe status badge without exposing secret values.
    if is_configured:
        source_label = "Streamlit Secrets" if key_source.startswith("st.secrets[") else "Environment"
        st.markdown(
            f"<div style='background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 6px; padding: 6px 10px; margin-bottom: 8px;'>"
            f"<span style='color: #065F46; font-weight: 600; font-size: 0.85rem;'>● Key format accepted ({source_label})</span>"
            "<div style='color: #065F46; font-size: 0.75rem;'>Connection not verified — use Test API Connection.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 6px 10px; margin-bottom: 8px;'>"
            "<span style='color: #475569; font-weight: 600; font-size: 0.85rem;'>○ Fallback Mode (Deterministic)</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Safe diagnostics explain why Gemini is inactive without exposing credentials.
    with st.expander("Gemini configuration diagnostics"):
        diagnostics = st.session_state.gemini_service.get_diagnostic_info()
        st.write(f"**Secret detected:** {'Yes' if diagnostics['key_detected'] else 'No'}")
        st.write(f"**Detected source:** {diagnostics['source_detected']}")
        st.write(f"**Key length:** {diagnostics['key_length']} characters")
        st.write(f"**Recognized key prefix:** {diagnostics['key_prefix']}")
        st.write(f"**Format check:** {diagnostics['format_status']}")
        st.write(f"**Streamlit Secrets available:** {'Yes' if diagnostics['st_secrets_available'] else 'No'}")
        st.write(f"**Secret names visible to app:** {', '.join(diagnostics['st_secrets_keys']) or 'None'}")
        st.write(f"**Primary model:** {diagnostics.get('primary_model', 'gemini-3.5-flash')}")
        st.write(f"**Fallback model:** {diagnostics.get('fallback_model', 'gemini-3.5-flash-lite')}")
        if diagnostics.get("last_used_model"):
            st.write(f"**Active model in session:** `{diagnostics['last_used_model']}`")
        if diagnostics.get("last_failover_diagnostics"):
            st.markdown("**Recent Attempt / Failover Log:**")
            for item in diagnostics["last_failover_diagnostics"]:
                err_detail = f" — {item.get('error', '')}" if item.get('error') else ""
                st.write(f"- `{item['model']}`: {item['status']}{err_detail}")
        if diagnostics.get("last_error_safe"):
            st.caption(f"ℹ️ {diagnostics['last_error_safe']}")
        st.caption("For security, the API key value is never displayed.")

    # Model Selection: Primary gemini-3.5-flash, Fallback gemini-3.5-flash-lite
    selected_model = st.selectbox(
        "Gemini Model",
        ["gemini-3.5-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="Primary model is gemini-3.5-flash with automatic failover to gemini-3.5-flash-lite.",
    )
    st.session_state.gemini_service.model_name = selected_model

    if st.button("Test API Connection", width="stretch"):
        with st.spinner("Testing connection..."):
            ok, msg = st.session_state.gemini_service.test_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # Database Quick Stats
    stats = db.get_dashboard_stats()
    st.markdown("### 📈 Stored Records")
    st.write(f"- **Resumes Saved:** {stats['resumes_count']}")
    st.write(f"- **Job Matches:** {stats['matches_count']}")
    st.write(f"- **Interview Runs:** {stats['interviews_count']}")
    st.write(f"- **Roadmaps:** {stats['roadmaps_count']}")

    if st.button("🗑️ Clear Active Session", width="stretch"):
        st.session_state.active_resume_data = None
        st.session_state.active_resume_text = ""
        st.session_state.active_resume_id = None
        st.session_state.last_analyzed_file_sig = None
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
                    st.session_state.last_analyzed_file_sig = "sample_resume.pdf"
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
                    st.session_state.last_analyzed_file_sig = None
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
            key="resume_file_uploader",
        )

    with col_demo:
        st.write("Or test instantly:")
        load_demo_pdf = st.button("📋 Load Sample PDF", width="stretch")
        load_demo_docx = st.button("📋 Load Sample DOCX", width="stretch")

    file_bytes = None
    filename = ""

    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
    elif load_demo_pdf:
        file_bytes = load_sample_file("sample_resume.pdf")
        filename = "sample_resume.pdf"
    elif load_demo_docx:
        file_bytes = load_sample_file("sample_resume.docx")
        filename = "sample_resume.docx"

    # Handle Upload & Analysis Workflow
    if uploaded_file is not None and len(file_bytes) == 0:
        st.error("⚠️ The uploaded file is empty (0 bytes). Please select a valid PDF or DOCX resume.")
    elif file_bytes and filename:
        import hashlib
        file_sig = hashlib.sha256(file_bytes).hexdigest()[:16]
        is_new_file = st.session_state.get("last_analyzed_file_sig") != file_sig

        # File status bar and manual Re-analyze trigger
        col_status, col_action = st.columns([3, 1])
        with col_status:
            file_kb = len(file_bytes) / 1024
            st.caption(f"📁 Document: **{html.escape(filename)}** ({file_kb:.1f} KB)")
        with col_action:
            if not is_new_file and st.session_state.active_resume_data:
                if st.button("🔄 Re-analyze Resume", width="stretch"):
                    is_new_file = True

        if is_new_file:
            try:
                with st.status(f"Processing {filename}...", expanded=True) as status_box:
                    status_box.write("📄 Step 1/4: Validating file integrity and magic bytes...")
                    raw_text, file_type = DocumentParser.extract_text(file_bytes, filename)

                    status_box.write("🔍 Step 2/4: Extracting structured skills, education, and career experience...")
                    analyzer = ResumeAnalyzer(st.session_state.gemini_service)
                    analysis_result = analyzer.analyze_resume(raw_text, filename)

                    status_box.write("💾 Step 3/4: Storing candidate record in database...")
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
                    st.session_state.last_analyzed_file_sig = file_sig

                    status_box.write("✨ Step 4/4: Finalizing resume score and intelligence...")
                    status_box.update(label=f"✅ Analysis Complete: {filename} (ID #{res_id})", state="complete", expanded=False)

                st.success(f"Successfully parsed and analyzed: **{filename}** (ID #{res_id})")
                st.rerun()

            except DocumentValidationError as dve:
                st.error(f"❌ Document Validation Error: {str(dve)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error During Analysis: {str(e)}")

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
    else:
        st.divider()
        st.info("💡 **Ready to analyze**: Upload your resume in PDF or DOCX format above (or load a sample profile) to view instant AI & deterministic skill extraction, scoring, and actionable feedback.")


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

        partial_skills = m.get("partially_matched_skills", [])
        if partial_skills:
            st.markdown(f"#### ◐ Related Evidence — Review ({len(partial_skills)})")
            st.caption("Related wording was found, but it is not treated as a confirmed skill match.")
            partial_pills = " ".join(
                [f"<span class='skill-pill pill-neutral'>◐ {html.escape(str(skill))}</span>" for skill in partial_skills]
            )
            st.markdown(partial_pills, unsafe_allow_html=True)
            with st.expander("Why these requirements were partially matched"):
                for skill in partial_skills:
                    reason = m.get("skill_evidence", {}).get(skill, {}).get("reason", "Related evidence found")
                    st.write(f"- **{skill}:** {reason}")

        with st.expander("How the compatibility score is calculated"):
            st.write("Skills alignment: 55% · ATS keyword overlap: 25% · Seniority/experience alignment: 20%.")
            st.write(f"Direct skill matches: {m.get('matched_skills_count', 0)} / {m.get('required_skills_count', 0)}")
            st.write(f"Partial evidence: {m.get('partial_skills_count', 0)}")
            st.write(f"Seniority evaluated against job title: {m.get('job_title_used_for_seniority', target_title)}")
            if m.get("required_years_experience") is not None:
                st.write(f"Experience requirement: {m.get('required_years_experience')} years")
            if m.get("candidate_years_experience") is not None:
                st.write(f"Resume-stated experience: {m.get('candidate_years_experience')} years")

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

    # Persisted match history for the selected candidate.
    st.divider()
    st.subheader("🕒 Recent Job Match History")
    recent_matches = db.list_job_matches(limit=10)
    candidate_matches = [item for item in recent_matches if item.get("resume_id") == selected_resume_id]
    if candidate_matches:
        history_df = pd.DataFrame(candidate_matches)
        st.dataframe(history_df, width="stretch", hide_index=True)
    else:
        st.caption("No saved job matches are available for this resume yet.")


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
                "Data Analyst",
                "Supply Chain Analyst",
                "Business Analyst",
                "Custom Position",
            ],
            index=0,
        )
        if target_role == "Custom Position":
            target_role = st.text_input("Enter Custom Position Title", "Data Analyst – Supply Chain Analytics")

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
                "Data Analyst / Business Intelligence",
                "Supply Chain Analyst / Operations Analytics",
                "Data Engineer",
                "Full Stack Developer",
                "Custom Role",
            ],
            index=0,
        )
        if target_role == "Custom Role":
            target_role = st.text_input("Enter Target Role", "Data Analyst – Supply Chain Analytics")

    with col_months:
        timeline_months = st.selectbox("Target Horizon", [3, 6, 12], index=1)

    # Candidate Resume Profile Selection & Personalization
    st.markdown("### 👤 Candidate Profile & Skills Baseline")
    saved_resumes = db.list_resumes()

    # Build options dictionary: saved database resumes + demonstration profile
    resume_options = {}
    for r in saved_resumes:
        resume_options[r["id"]] = f"#{r['id']} - {r['candidate_name']} ({r['filename']}) — {str(r.get('created_at', ''))[:16]}"

    DEMO_KEY = "demo_profile"
    resume_options[DEMO_KEY] = "🚀 Demonstration Profile (Alex Morgan — Cloud & Data Engineer)"

    option_keys = list(resume_options.keys())

    # Determine default selection index
    default_idx = 0
    if st.session_state.active_resume_id in option_keys:
        default_idx = option_keys.index(st.session_state.active_resume_id)
    elif saved_resumes:
        default_idx = 0
    elif DEMO_KEY in option_keys:
        default_idx = option_keys.index(DEMO_KEY)

    selected_key = st.selectbox(
        "Select Candidate Profile to Personalize Roadmap",
        options=option_keys,
        index=default_idx,
        format_func=lambda x: resume_options[x],
        key="roadmap_profile_selector",
    )

    # Synchronize candidate profile state based on selection
    if selected_key == DEMO_KEY:
        if st.session_state.active_resume_id != DEMO_KEY or not st.session_state.active_resume_data:
            st.session_state.active_resume_id = DEMO_KEY
            st.session_state.active_resume_data = {
                "candidate_name": "Alex Morgan",
                "email": "alex.morgan@example.com",
                "skills": ["Python", "SQL", "Docker", "Kubernetes", "AWS", "PostgreSQL", "Redis", "CI/CD", "Terraform", "Go"],
                "score": 85,
            }
            st.session_state.active_resume_text = "Alex Morgan Senior Cloud Engineer Python SQL Docker Kubernetes AWS PostgreSQL Redis CI/CD Terraform"
    else:
        if st.session_state.active_resume_id != selected_key or not st.session_state.active_resume_data:
            chosen = db.get_resume(selected_key)
            if chosen:
                st.session_state.active_resume_id = selected_key
                st.session_state.active_resume_data = chosen.get("parsed_data", {})
                st.session_state.active_resume_text = chosen.get("raw_text", "")

    # Display active candidate verified skills
    current_skills = []
    if st.session_state.active_resume_data:
        res_data = st.session_state.active_resume_data
        cand_name = res_data.get("candidate_name") or "Candidate"
        current_skills = res_data.get("skills", [])
        st.success(f"**Loaded Profile:** {cand_name} (ID #{st.session_state.active_resume_id})")
        if current_skills:
            st.write(f"**Verified Proven Skills ({len(current_skills)}):**")
            pills = " ".join([f"<span class='skill-pill pill-match'>{html.escape(str(s))}</span>" for s in current_skills])
            st.markdown(pills, unsafe_allow_html=True)
            st.caption("✅ Proven skills will be respected and excluded from critical prerequisites.")
        else:
            st.caption("No specific skills extracted from this profile. Evaluating standard foundations.")
    else:
        st.caption("Evaluating against standard junior-to-mid foundation.")

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

# VibeForge submission attribution
st.divider()
st.caption("Built for VibeForge 2026 by Mohammad Kattubadi")

# CareerPilot AI 🚀
> **Your Career. Your Potential. Powered by AI.**

[![CI Test Suite](https://img.shields.io/badge/pytest-60%20passed-brightgreen.svg)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Google GenAI](https://img.shields.io/badge/Gemini%20API-Free--Tier%20Ready-orange.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

CareerPilot AI is an autonomous, production-grade career acceleration platform built with Python and Streamlit. It empowers candidates to analyze resumes, benchmark against competitive job descriptions, pass Applicant Tracking Systems (ATS) ethically, simulate rigorous technical interviews, and chart clear skill development roadmaps.

---

## 🌟 Key Features

### 1. 📄 Resume Intelligence
* **PDF & DOCX Extraction:** Native text parsing with validation and contact extraction powered by PyMuPDF and `python-docx`.
* **Deep Skill Extraction:** Automatic classification across languages, frameworks, cloud platforms, databases, and methodologies.
* **Impact & Strength Scoring:** Evaluates quantifiable business metrics (%, $, scale), action verb density, and structural clarity.
* **Resume Quality Score:** 0–100 benchmark score with actionable suggestions.

### 2. 🎯 Job Matcher & Explainable Fit
* **Requirements Extraction:** Pulls required skills, experience thresholds, and responsibilities from any job posting.
* **Explainable Scoring:** Transparent 3-component formula:
  * **Skills Alignment (55% weight)**
  * **ATS Keyword Density (25% weight)**
  * **Seniority Alignment (20% weight)**
* **Visual Skill Breakdown:** Interactive green (matched) and red (missing) skill pills.

### 3. 🛡️ Ethical ATS Optimizer
* **Keyword Gap Analysis:** High-frequency job keyword density comparison.
* **Tactical Bullet Point Rewrites:** Suggests impactful rewording based on the Google XYZ formula (`Accomplished [X] measured by [Y] by doing [Z]`).
* **Anti-Fabrication Guardrail:** Strict ethical policy that prevents qualification invention and encourages honest positioning.

### 4. 🎤 Interactive Interview Coach
* **Role-Specific Simulation:** Generates technical, system design, and behavioral questions across Backend, Full Stack, Data Science, DevOps, and custom roles.
* **3-Dimensional Rubric:** Scores answers on **Relevance (0–100)**, **Clarity (0–100)**, and **Completeness (0–100)**.
* **STAR Framework Detection:** Validates the presence of **Situation**, **Task**, **Action**, and **Result** components.

### 5. 🗺️ Career Roadmap & Velocity Planner
* **Skill Gap Benchmarking:** Identifies exact missing competencies against industry standards.
* **Priority Categorization:** Categorizes goals into **P0 (Critical)**, **P1 (Important)**, and **P2 (Nice-to-Have)**.
* **Actionable Milestones:** Phased 3, 6, or 12-month development plans with concrete capstone projects and certifications.

### 6. 📊 Executive Dashboard & SQLite Persistence
* Unified SQLite database persisting resumes, job matches, interview evaluations, and roadmaps.
* Aggregated metrics (total resumes, average match score, interview run statistics).

---

## 🏗️ Architecture & Approved Stack

```
careerpilot-ai/
├── .github/workflows/ci.yml   # GitHub Actions CI matrix
├── .streamlit/config.toml      # Theme & server configuration
├── assets/style.css            # Executive styling and metric card CSS
├── data/                       # Sample files & SQLite storage
│   ├── sample_resume.pdf       # Sample demonstration PDF
│   ├── sample_resume.docx      # Sample demonstration DOCX
│   ├── sample_job.txt          # Sample job posting
│   └── careerpilot.db          # SQLite database
├── src/                        # Modular engine components
│   ├── database.py             # SQLite persistence layer
│   ├── document_parser.py      # PyMuPDF & docx parser with validation
│   ├── gemini_service.py       # Google GenAI integration + deterministic fallback
│   ├── resume_analyzer.py      # Resume intelligence & scoring
│   ├── job_matcher.py          # Explainable job matcher & ATS optimizer
│   ├── interview_coach.py      # Role-specific question bank & STAR evaluator
│   └── career_roadmap.py       # Skill gap & prioritized milestones engine
├── tests/                      # Automated test suite (27 tests)
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Dependency manifest
├── LIMITATIONS.md              # Known limitations report
└── README.md                   # Documentation
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
* Python 3.11, 3.12, or 3.13
* Git

### 2. Clone the Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/careerpilot-ai.git
cd careerpilot-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemini API (Optional)
CareerPilot AI runs out-of-the-box in **Deterministic Fallback Mode** without any API key. To unlock generative AI reasoning:
```bash
# Copy example configuration
cp .env.example .env

# Set your API key in .env or provide it directly in the Streamlit UI sidebar:
GEMINI_API_KEY="AIzaSy..."
```
*(Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/))*

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Automated Tests

The application features a complete test suite with **60 automated tests** covering database CRUD, document parsing, validation errors, Gemini service fallbacks, secret loading and quote sanitization, model migration and failover, ATS scoring, STAR rubric evaluation, and an end-to-end integration test:

```bash
# Run all tests with pytest
python -m pytest tests/ -v
```

Output:
```
============================= 60 passed in 96.2s ==============================
```

---

## 🐳 Docker Container Deployment

You can build and run CareerPilot AI anywhere using Docker:

```bash
# Build Docker image
docker build -t careerpilot-ai .

# Run container
docker run -p 8501:8501 -e GEMINI_API_KEY="your_key" careerpilot-ai
```
Access the application at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. (Optional) In **Advanced settings**, add your `GEMINI_API_KEY` under Secrets.
5. Click **Deploy!**

---

## 🛡️ Engineering Reliability & Guardrails

* **Zero Hardcoded Fake AI:** When Gemini is unavailable, all outputs are generated by deterministic algorithms and clearly badged as `⚙️ Deterministic Fallback`.
* **No Qualification Invention:** ATS recommendations emphasize authentic positioning and reject credential fabrication.
* **Secure Credential Handling:** API keys entered via the UI are masked and held in transient memory only; keys are never logged or stored in the database.

---

## 📄 License
This project is licensed under the MIT License.

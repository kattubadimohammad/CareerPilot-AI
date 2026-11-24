# CareerPilot AI — Cloud & Production Deployment Guide

## 1. Overview
CareerPilot AI is packaged for zero-dependency local execution, Docker containers, and Streamlit Community Cloud.

---

## 2. Streamlit Community Cloud Deployment

### Prerequisites
* A GitHub repository containing the CareerPilot AI source code.
* A free account on [share.streamlit.io](https://share.streamlit.io).

### Steps
1. Push your repository to GitHub.
2. Sign in to **Streamlit Community Cloud** and click **Create app**.
3. Select your repository, set the branch to `main`, and set **Main file path** to `app.py`.
4. Open **Advanced settings**:
   * **Python version**: Select `3.11` (enforced via repository `.python-version`).
   * **Secrets**: Add your Gemini API Key in TOML format:
     ```toml
     GEMINI_API_KEY = "AIzaSy..."
     ```
5. Click **Deploy!**

### Storage Architecture Note on Community Cloud
* Streamlit Community Cloud utilizes **ephemeral container storage**. Any new records written to `data/careerpilot.db` will persist throughout the running container lifecycle but will reset when the app is rebooted or redeployed.
* For **permanent multi-user persistence** in cloud environments, configure an external cloud database URI (e.g. Supabase PostgreSQL, Neon, or Turso SQLite) via environment variable or Streamlit Secrets.

---

## 3. Docker Container Deployment

CareerPilot AI provides a production-grade multi-stage Docker container:

```bash
# 1. Build Docker image
docker build -t careerpilot-ai .

# 2. Run container with persistent local volume for database
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e GEMINI_API_KEY="AIzaSy..." \
  --name careerpilot \
  careerpilot-ai
```

---

## 4. Production Security & Reliability Checklist

* [x] **XSS Sanitization:** Dynamic HTML pill and badge rendering escaped via `html.escape()`.
* [x] **File Validation:** Magic byte signatures validated for PDF (`%PDF-`) and DOCX (`PK\x03\x04`).
* [x] **ZIP Bomb Defense:** DOCX extraction guarded against uncompressed size bombs (>25MB) and extreme compression ratios (>50:1).
* [x] **Database Concurrency:** SQLite WAL mode, foreign keys (`PRAGMA foreign_keys = ON;`), and 30-second busy timeout enabled.
* [x] **Right to be Forgotten:** Full cascade resume deletion exposed in the UI.
* [x] **Credential Protection:** Automatic scrubbing of API keys and PII in application loggers.
* [x] **Prompt Isolation:** XML containment tags (`<untrusted_resume_content>`) used to prevent prompt injection.

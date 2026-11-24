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
2. Sign in to **Streamlit Community Cloud** and select your app settings.
3. In **App Settings** > **Secrets**, enter your Gemini API Key in valid TOML format:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
   > **Note on quotes:** The app automatically strips accidental leading/trailing quotes, whitespace, and handles case-insensitive naming (`gemini_api_key`, `GOOGLE_API_KEY`, or `[gemini] api_key = "..."`).
4. Click **Save**.
5. In the app sidebar, click **🔄 Sync Secrets** or **Test API** to verify immediate connectivity.

### Supported Models for Gemini Free Tier
* **`gemini-3.8-flash`** (Primary Default): Google's workhorse multimodal Flash model for complex agentic and reasoning tasks.
* **`gemini-3.5-flash-lite`** (Direct Fallback): Ultra-low latency, high-throughput model optimized for high availability during peak traffic spikes.
> **Note on Model Availability:** Model availability is determined by Google AI Studio and project enablement. Do not assume older models (such as `gemini-2.0-flash` or `gemini-1.5-flash`) are supported or available for new Google Cloud projects. The application dynamically queries available models via Google GenAI SDK discovery when connected.

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

# CareerPilot AI - Known Limitations & Engineering Analysis

**Version:** 1.0.0 (MVP)  
**Date:** November 2026  
**Status:** MVP Production-Ready (Local & Containerized)

---

## 1. Document Parsing & OCR Scope
* **Text-Embedded Files Only:** The document parser uses PyMuPDF (`fitz`) and `python-docx` for high-fidelity native text extraction. Scanned image-only PDFs (photographs or rasterized documents lacking an embedded text layer) are detected and gracefully rejected with a helpful warning rather than misparsed.
* **Complex Multi-Column Tables:** Complex tabular multi-column resumes with non-standard reading orders may occasionally experience reordered line breaks during plain text extraction. A two-pass visual layout engine (e.g. `pymupdf4llm` or Vision LLM) is planned for v2.0.

---

## 2. Gemini API Free-Tier Quotas & Fallback Engine
* **Rate Limits:** Google Gemini Free-Tier provides 15 Requests Per Minute (RPM) and a daily limit. To protect user experience:
  * CareerPilot AI implements a **clearly labelled deterministic fallback** for every module.
  * If the API key is absent, rate-limited (HTTP 429), or unavailable, the application immediately switches to its rule-based heuristic algorithms without throwing unhandled exceptions or presenting fake data.
  * Every card and report transparently surfaces an **Engine Provider Badge** indicating whether results were generated via Gemini AI (`✨ Generated with Gemini`) or the Heuristic Engine (`⚙️ Deterministic Fallback`).

---

## 3. Ethical Anti-Fabrication Guardrail
* **No Qualification Invention:** Unlike generative systems that hallucinate credentials to artificially inflate ATS scores, CareerPilot AI strictly adheres to an ethical policy:
  * It highlights genuine skill overlaps and identifies true missing requirements.
  * It suggests rewording and better framing of authentic achievements using the Google XYZ formula (`Accomplished [X] measured by [Y] by doing [Z]`).
  * It warns candidates against injecting unverified keywords.

---

## 4. SQLite Persistence & Scalability
* **Concurrency Model:** SQLite was chosen for minimal operational overhead, zero-configuration local deployment, and portable data files.
* **Multi-User Production:** For scaled multi-user enterprise deployments with high concurrent write volume, SQLite should be swapped for PostgreSQL (supported via SQLAlchemy / Psycopg2).

---

## 5. Deployment Status
* **Local & Docker Verification:** The application has been verified locally, booting successfully with HTTP 200 health responses.
* **Public URL Testing:** Per project execution rules, no public URL is claimed as active until verified via live network probe. A tested `Dockerfile`, `.streamlit/config.toml`, and GitHub Actions workflow are supplied for instant deployment to Streamlit Community Cloud, Google Cloud Run, or AWS App Runner.

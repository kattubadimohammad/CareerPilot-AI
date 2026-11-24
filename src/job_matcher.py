"""Job Matcher and ATS Optimizer Module.
Calculates explainable compatibility scores, identifies skill overlaps and gaps,
analyzes ATS keyword frequency, and generates ethical resume optimization advice.
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.gemini_service import GeminiService
from src.resume_analyzer import COMPILED_SKILL_PATTERNS, TAXONOMY_SKILLS


class JobMatcher:
    """Matches candidate resume against job descriptions with explainable scoring and ATS optimization."""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    @classmethod
    def extract_job_skills(cls, job_text: str) -> List[str]:
        """Extract skills mentioned in job description using precompiled patterns."""
        text_lower = f" {job_text.lower()} "
        found_skills = set()
        for skill, pattern in COMPILED_SKILL_PATTERNS.items():
            if pattern.search(text_lower):
                found_skills.add(skill)
        return sorted(list(found_skills))

    @classmethod
    def extract_ats_keywords(cls, job_text: str, top_n: int = 25) -> List[Tuple[str, int]]:
        """Extract top frequent significant keywords from job description (excluding stop words)."""
        stop_words = {
            "the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "by", "this",
            "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your",
            "all", "have", "new", "more", "an", "was", "we", "will", "home", "can", "us",
            "about", "if", "my", "has", "into", "their", "our", "work", "team", "role",
            "experience", "years", "candidate", "responsibilities", "requirements", "skills",
            "ability", "must", "strong", "preferred", "including", "working", "knowledge",
            "opportunity", "join", "company", "looking", "qualified", "applicant", "employment",
            "equal", "status", "position", "environment", "support", "help", "across", "other"
        }
        words = re.findall(r"\b[a-zA-Z]{3,25}\b", job_text.lower())
        frequency: Dict[str, int] = {}
        for w in words:
            if w not in stop_words:
                frequency[w] = frequency.get(w, 0) + 1

        sorted_words = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]

    @classmethod
    def calculate_explainable_score(
        cls, resume_skills: List[str], job_skills: List[str], resume_text: str, job_text: str
    ) -> Dict[str, Any]:
        """
        Calculate an explainable compatibility score with granular sub-scores.
        Weights:
        - Skills Alignment: 55%
        - ATS Keyword Overlap: 25%
        - Experience & Seniority Alignment: 20%
        """
        resume_skill_set = {s.lower() for s in resume_skills}
        job_skill_set = {s.lower() for s in job_skills}

        # 1. Skills Alignment Score
        if job_skill_set:
            matched_set = resume_skill_set.intersection(job_skill_set)
            skill_score = (len(matched_set) / len(job_skill_set)) * 100
        else:
            matched_set = set()
            skill_score = 70.0  # Default neutral score if no specific technical skills detected in JD

        matched_skills = [s for s in job_skills if s.lower() in matched_set]
        missing_skills = [s for s in job_skills if s.lower() not in matched_set]

        # 2. ATS Keyword Overlap Score
        top_job_keywords = cls.extract_ats_keywords(job_text, top_n=20)
        resume_text_lower = resume_text.lower()
        matched_kw_count = sum(1 for kw, _ in top_job_keywords if kw in resume_text_lower)
        keyword_score = (matched_kw_count / max(1, len(top_job_keywords))) * 100

        # 3. Experience & Seniority Alignment Score
        senior_job = bool(re.search(r"\b(senior|lead|principal|staff|manager|architect|director)\b", job_text, re.I))
        senior_res = bool(re.search(r"\b(senior|lead|principal|staff|architect|head)\b", resume_text, re.I))

        # Check years requirement vs resume
        req_years_match = re.search(r"(\d+)\+?\s*years(?:\s+of)?\s+experience", job_text, re.I)
        req_years = int(req_years_match.group(1)) if req_years_match else None

        if senior_job and senior_res:
            exp_score = 95.0
        elif not senior_job:
            exp_score = 90.0
        else:
            exp_score = 65.0  # Senior job but resume does not feature senior keywords prominently

        # Overall composite score
        overall_score = round(
            (0.55 * skill_score) + (0.25 * keyword_score) + (0.20 * exp_score), 1
        )
        overall_score = max(10.0, min(99.0, overall_score))

        return {
            "overall_score": overall_score,
            "sub_scores": {
                "skills_alignment": round(skill_score, 1),
                "ats_keyword_density": round(keyword_score, 1),
                "seniority_experience_alignment": round(exp_score, 1),
            },
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "required_skills_count": len(job_skills),
            "matched_skills_count": len(matched_skills),
            "top_job_keywords": top_job_keywords,
        }

    @classmethod
    def generate_ats_recommendations(
        cls, missing_skills: List[str], top_keywords: List[Tuple[str, int]], resume_text: str
    ) -> List[str]:
        """Generate actionable ATS optimization suggestions with strict anti-fabrication guidelines."""
        recs = []
        resume_lower = resume_text.lower()

        if missing_skills:
            top_missing = ", ".join(missing_skills[:5])
            recs.append(
                f"Integrate Verified Missing Skills: Your resume is missing key technical requirements: {top_missing}. "
                "If you have relevant project or academic exposure to any of these, incorporate them into your Skills and Experience bullets."
            )

        # Check missing high-frequency keywords
        missing_kws = [kw for kw, _ in top_keywords if kw not in resume_lower][:6]
        if missing_kws:
            recs.append(
                f"Incorporate Job-Specific Terminology: The job posting frequently uses terms like '{', '.join(missing_kws)}'. "
                "Align your resume vocabulary to match these terms where truthful and applicable."
            )

        recs.append(
            "Adopt ATS-Friendly Formatting: Ensure clean single-column flow, standard headers (e.g., 'Summary', 'Skills', 'Experience', 'Education'), "
            "and eliminate complex graphic tables or text boxes which cause parsing failures."
        )

        recs.append(
            "Contextualize Tools in Achievement Bullets: Rather than simply listing tools in a skills bank, demonstrate how you applied them "
            "using the format: 'Developed [feature] using [Technology] to achieve [Measurable Outcome]'."
        )

        return recs

    def match_resume_to_job(
        self, resume_text: str, resume_skills: List[str], job_text: str, job_title: str = "Target Position"
    ) -> Dict[str, Any]:
        """
        Complete matching and ATS optimization pipeline.
        Combines deterministic scoring with Gemini semantic recommendations if configured.
        """
        job_skills = self.extract_job_skills(job_text)
        scoring = self.calculate_explainable_score(resume_skills, job_skills, resume_text, job_text)
        deterministic_recs = self.generate_ats_recommendations(
            scoring["missing_skills"], scoring["top_job_keywords"], resume_text
        )

        result_data = {
            "job_title": job_title,
            "overall_score": scoring["overall_score"],
            "sub_scores": scoring["sub_scores"],
            "matched_skills": scoring["matched_skills"],
            "missing_skills": scoring["missing_skills"],
            "required_skills_count": scoring["required_skills_count"],
            "matched_skills_count": scoring["matched_skills_count"],
            "top_keywords": [kw for kw, _ in scoring["top_job_keywords"][:15]],
            "recommendations": deterministic_recs,
            "ethical_notice": (
                "ETHICAL ATS OPTIMIZATION GUARDRAIL: CareerPilot AI strictly advises against fabricating or exaggerating qualifications. "
                "Only incorporate keywords and skills that accurately reflect your verified competencies and project experience."
            ),
        }

        # If Gemini is configured, enrich with deep semantic job-fit reasoning
        if self.gemini_service and self.gemini_service.is_configured():
            prompt = f"""
You are an expert ATS optimization consultant and hiring manager.
Compare the candidate's resume with the target job description.

<untrusted_resume_content>
{resume_text[:3000]}
</untrusted_resume_content>

<untrusted_job_description title="{job_title}">
{job_text[:3000]}
</untrusted_job_description>

Provide an objective ATS optimization and job-fit analysis.
Return ONLY valid JSON matching this structure:
{{
    "fit_summary": "2-3 sentence executive summary of job compatibility",
    "recommended_bullet_rewrites": [
        {{
            "original_theme": "Area of resume to improve",
            "suggested_rewrite": "High-impact bullet draft incorporating job keywords ethically",
            "justification": "Why this improves ATS ranking"
        }}
    ],
    "ats_tactical_tips": [
        "Specific actionable ATS tip 1",
        "Specific actionable ATS tip 2",
        "Specific actionable ATS tip 3"
    ]
}}
"""
            ai_data, is_ai, note = self.gemini_service.generate_json_response(
                prompt=prompt,
                system_instruction="You are an expert ATS optimizer. Never advocate falsifying qualifications. Respond ONLY in raw JSON.",
                fallback_data={"fit_summary": "", "recommended_bullet_rewrites": [], "ats_tactical_tips": []},
            )

            if is_ai and isinstance(ai_data, dict):
                result_data["fit_summary"] = ai_data.get("fit_summary", "")
                result_data["bullet_rewrites"] = ai_data.get("recommended_bullet_rewrites", [])
                if ai_data.get("ats_tactical_tips"):
                    result_data["recommendations"] = ai_data.get("ats_tactical_tips") + deterministic_recs
                result_data["is_ai_generated"] = True
                result_data["provider_note"] = note
                return result_data

        result_data["is_ai_generated"] = False
        result_data["provider_note"] = (
            "Deterministic Rule-Based Matching (Gemini API key not configured. Enable in sidebar for semantic bullet rewrites.)"
        )
        return result_data

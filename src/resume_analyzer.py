"""Resume Intelligence Module: Analyzes skills, education, experience, strengths,
weaknesses, actionable feedback, and calculates resume scores.
"""
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple
from src.document_parser import DocumentParser
from src.gemini_service import GeminiService

# Canonical skill taxonomy for deterministic extraction
TAXONOMY_SKILLS = {
    # Programming Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Golang",
    "Rust", "Ruby", "PHP", "SQL", "HTML", "CSS", "Bash", "Shell", "R", "Kotlin", "Swift",
    # Frameworks & Libraries
    "React", "Next.js", "Angular", "Vue", "Node.js", "Express", "Django", "FastAPI",
    "Flask", "Spring Boot", "ASP.NET", ".NET", "Pandas", "NumPy", "Scikit-Learn",
    "PyTorch", "TensorFlow", "Streamlit", "Pytest", "Jest", "GraphQL", "REST APIs",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "Terraform",
    "CI/CD", "GitHub Actions", "GitLab CI", "Jenkins", "Linux", "Ansible",
    "Prometheus", "Grafana", "Helm", "Serverless", "Lambda",
    # Databases & Storage
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Elasticsearch",
    "Kafka", "RabbitMQ", "DynamoDB", "Snowflake", "BigQuery", "Cassandra",
    # Concepts & Methodologies
    "System Design", "Microservices", "Agile", "Scrum", "Git", "Unit Testing",
    "Machine Learning", "Deep Learning", "Data Engineering", "Object-Oriented Programming",
    "TDD", "CI/CD Pipelines", "Code Review", "Data Structures", "Algorithms",
    # Soft & Management Skills
    "Leadership", "Mentoring", "Project Management", "Problem Solving",
    "Communication", "Cross-functional Collaboration", "Analytical Skills",
}

# Precompiled regex patterns for high-throughput skill extraction
COMPILED_SKILL_PATTERNS = {
    skill: re.compile(rf"(?<![a-zA-Z0-9_]){re.escape(skill.lower())}(?![a-zA-Z0-9_])")
    for skill in TAXONOMY_SKILLS
}

ACTION_VERBS = [
    "architected", "developed", "built", "implemented", "designed", "engineered",
    "spearheaded", "optimized", "reduced", "increased", "launched", "deployed",
    "automated", "created", "led", "managed", "improved", "delivered", "scaled"
]


class ResumeAnalyzer:
    """Performs deep analysis of resume content using Gemini or deterministic rules."""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    @classmethod
    def extract_skills_deterministic(cls, text: str) -> List[str]:
        """Extract skills from text by matching against precompiled taxonomy with boundary checks."""
        found_skills = set()
        text_lower = f" {text.lower()} "

        for skill, pattern in COMPILED_SKILL_PATTERNS.items():
            if pattern.search(text_lower):
                found_skills.add(skill)

        return sorted(list(found_skills))

    @classmethod
    def extract_education_deterministic(cls, text: str) -> List[Dict[str, str]]:
        """Identify education entries (degree, institution, year)."""
        degrees = []
        degree_patterns = [
            r"(Bachelor|B\.?S\.?|B\.?A\.?|B\.?Tech|B\.?E\.?)\s*(?:of|in)?\s*([A-Za-z\s]{3,35})",
            r"(Master|M\.?S\.?|M\.?A\.?|M\.?Tech|MBA)\s*(?:of|in)?\s*([A-Za-z\s]{3,35})",
            r"(Ph\.?D\.?|Doctorate)\s*(?:in)?\s*([A-Za-z\s]{3,35})",
            r"(Associate\s+Degree)\s*(?:in)?\s*([A-Za-z\s]{3,35})",
        ]

        lines = text.split("\n")
        for line in lines:
            for pat in degree_patterns:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    deg_type = match.group(1).strip()
                    major = match.group(2).strip()
                    # Find year in same or nearby line
                    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", line)
                    year = year_match.group(1) if year_match else "Not specified"
                    degrees.append({
                        "degree": f"{deg_type} in {major}".strip(),
                        "institution": "Recognized Institution",
                        "year": year,
                    })
                    break

        if not degrees:
            # Check for general education keywords
            if re.search(r"\b(university|college|institute|bachelor|master|b\.s\.|m\.s\.)\b", text, re.IGNORECASE):
                degrees.append({
                    "degree": "Higher Education Degree",
                    "institution": "University / College",
                    "year": "Listed in resume",
                })

        return degrees

    @classmethod
    def extract_experience_deterministic(cls, text: str) -> List[Dict[str, str]]:
        """Identify work experience entries and estimate total experience."""
        experiences = []
        role_patterns = [
            r"((?:Senior|Lead|Staff|Principal|Junior|Associate)?\s*(?:Software Engineer|Developer|Data Scientist|DevOps Engineer|Cloud Architect|Product Manager|Data Analyst|Backend Engineer|Frontend Engineer|Full Stack Developer|QA Engineer|Machine Learning Engineer))",
        ]

        lines = text.split("\n")
        for line in lines:
            for pat in role_patterns:
                match = re.search(pat, line, re.IGNORECASE)
                if match and len(match.group(1).strip()) > 5:
                    role_title = match.group(1).strip().title()
                    # Check if already added
                    if not any(e["role"].lower() == role_title.lower() for e in experiences):
                        experiences.append({
                            "role": role_title,
                            "company": "Organization",
                            "highlights": "Key responsibilities documented in resume",
                        })

        # Calculate rough years of experience from year ranges like 2018 - 2023 or 2019-Present
        year_ranges = re.findall(r"\b(20\d{2})\s*(?:-|to|–)\s*(20\d{2}|present|current)\b", text, re.IGNORECASE)
        total_years = 0
        current_year = datetime.now().year
        for start_str, end_str in year_ranges:
            try:
                start = int(start_str)
                end = current_year if end_str.lower() in ["present", "current"] else int(end_str)
                diff = max(0, end - start)
                total_years += diff
            except ValueError:
                pass

        if not experiences and total_years > 0:
            experiences.append({
                "role": "Professional Experience",
                "company": "Industry Experience",
                "highlights": f"Approximately {total_years} years of accumulated industry experience.",
            })

        return experiences

    @classmethod
    def evaluate_strengths_and_weaknesses(
        cls, text: str, skills: List[str], experiences: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str], List[str], int]:
        """Compute strengths, weaknesses, actionable recommendations, and resume score."""
        strengths = []
        weaknesses = []
        feedback = []
        score = 50  # Base score

        # 1. Skill breadth check
        if len(skills) >= 10:
            strengths.append(f"Broad technical skill set identified ({len(skills)} verified skills across multiple domains).")
            score += 15
        elif len(skills) >= 5:
            strengths.append(f"Good foundational skill portfolio ({len(skills)} skills detected).")
            score += 10
        else:
            weaknesses.append("Limited technical skill keywords detected. ATS scanners may rank the resume lower.")
            feedback.append("Enrich the Skills section with explicit programming languages, frameworks, and tools.")

        # 2. Metric quantification check (numbers, percentages, revenue impact)
        metrics = re.findall(r"(\b\d+%\b|\$\d+[\d,]*|\b\d+x\b|\b\d+\s*(?:users|clients|ms|seconds|minutes|hours|team members|engineers|requests)\b)", text, re.IGNORECASE)
        if len(metrics) >= 4:
            strengths.append(f"Strong quantification of impact with {len(metrics)} measurable metrics (%, $, scale).")
            score += 15
        elif len(metrics) >= 1:
            strengths.append(f"Quantifiable metrics present ({len(metrics)} measured outcomes), showing evidence of business impact.")
            score += 8
            feedback.append("Further quantify achievements by stating how much latency was reduced, revenue generated, or time saved.")
        else:
            weaknesses.append("Lack of quantifiable business metrics. Bullets appear task-oriented rather than achievement-oriented.")
            feedback.append("Use the Google XYZ formula: 'Accomplished [X] as measured by [Y], by doing [Z]'.")

        # 3. Action verb density
        text_lower = text.lower()
        verb_count = sum(1 for verb in ACTION_VERBS if f" {verb} " in text_lower or f"\n{verb} " in text_lower)
        if verb_count >= 5:
            strengths.append("Effective use of strong, executive action verbs throughout experience entries.")
            score += 10
        else:
            weaknesses.append("Low density of dynamic action verbs. Responsibilities may appear passive.")
            feedback.append("Start bullet points with high-impact verbs like 'Architected', 'Spearheaded', 'Optimized', or 'Automated'.")

        # 4. Contact info check
        contacts = DocumentParser.quick_contact_extraction(text)
        if contacts.get("email") and contacts.get("phone"):
            score += 5
        else:
            weaknesses.append("Incomplete contact information found (missing direct phone or email).")
            feedback.append("Ensure your email, phone, and LinkedIn URL are clearly visible in the header.")

        # 5. Experience depth
        if len(experiences) >= 2:
            strengths.append("Clear career progression with multiple structured roles.")
            score += 5
        elif not experiences:
            weaknesses.append("Experience section could not be clearly parsed into structured roles.")
            feedback.append("Use standard headings like 'Work Experience' or 'Professional Experience' with clear role titles and dates.")

        # Clamp score between 20 and 98
        final_score = max(20, min(98, score))

        if not strengths:
            strengths.append("Resume provides a concise summary of candidate background.")
        if not weaknesses:
            weaknesses.append("No major structural flaws detected.")
        if not feedback:
            feedback.append("Keep the resume updated with recent project wins and certifications.")

        return strengths, weaknesses, feedback, final_score

    def analyze_resume(self, text: str, filename: str = "resume.pdf") -> Dict[str, Any]:
        """
        Full analysis pipeline combining deterministic extraction with Gemini AI semantic enrichment.
        """
        # 1. Deterministic baseline
        contact = DocumentParser.quick_contact_extraction(text)
        skills = self.extract_skills_deterministic(text)
        education = self.extract_education_deterministic(text)
        experience = self.extract_experience_deterministic(text)
        strengths, weaknesses, feedback, score = self.evaluate_strengths_and_weaknesses(text, skills, experience)

        deterministic_data = {
            "candidate_name": contact.get("name") or "Candidate",
            "email": contact.get("email"),
            "phone": contact.get("phone"),
            "skills": skills,
            "education": education,
            "experience": experience,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "feedback": feedback,
            "score": score,
            "summary": f"Professional profile with expertise in {', '.join(skills[:5]) if skills else 'technical development'}.",
        }

        # 2. If Gemini is available, request deep semantic enhancement
        if self.gemini_service and self.gemini_service.is_configured():
            prompt = f"""
You are an expert executive resume reviewer and technical recruiter.
Analyze the following resume text in detail and provide a comprehensive, objective assessment.

<untrusted_resume_content>
{text[:4000]}
</untrusted_resume_content>

Return ONLY a valid JSON object with EXACTLY this structure:
{{
    "candidate_name": "Full name of candidate",
    "email": "email address or null",
    "phone": "phone number or null",
    "summary": "2-3 sentence executive professional summary",
    "skills": ["list", "of", "all", "technical", "and", "soft", "skills"],
    "education": [
        {{"degree": "Degree name", "institution": "Institution name", "year": "Graduation year or dates"}}
    ],
    "experience": [
        {{"role": "Title", "company": "Company", "highlights": "Key achievements and metrics"}}
    ],
    "strengths": ["3-5 concrete strengths with specific evidence"],
    "weaknesses": ["2-4 critical areas needing improvement or missing information"],
    "feedback": ["3-5 actionable, high-priority resume improvement suggestions"],
    "score": 85
}}
"""
            ai_data, is_ai, note = self.gemini_service.generate_json_response(
                prompt=prompt,
                system_instruction="You are an expert AI resume reviewer. Always return valid raw JSON.",
                fallback_data=deterministic_data,
            )

            if is_ai and isinstance(ai_data, dict) and "skills" in ai_data:
                # Merge and sanitize
                result = {
                    "candidate_name": ai_data.get("candidate_name") or deterministic_data["candidate_name"],
                    "email": ai_data.get("email") or deterministic_data["email"],
                    "phone": ai_data.get("phone") or deterministic_data["phone"],
                    "summary": ai_data.get("summary") or deterministic_data["summary"],
                    "skills": sorted(list(set(ai_data.get("skills", []) + skills))),
                    "education": ai_data.get("education") or education,
                    "experience": ai_data.get("experience") or experience,
                    "strengths": ai_data.get("strengths") or strengths,
                    "weaknesses": ai_data.get("weaknesses") or weaknesses,
                    "feedback": ai_data.get("feedback") or feedback,
                    "score": int(ai_data.get("score", score)),
                    "is_ai_generated": True,
                    "provider_note": note,
                }
                return result

        # Deterministic return
        deterministic_data["is_ai_generated"] = False
        deterministic_data["provider_note"] = (
            "Deterministic Rule-Based Intelligence (Gemini API key not configured. Enable in sidebar for AI semantic review.)"
        )
        return deterministic_data

"""Interview Coach Module.
Generates role-specific behavioral and technical interview questions,
evaluates candidate responses across relevance, clarity, and completeness,
and provides structured coaching using the STAR method.
"""
import re
from typing import Any, Dict, List, Optional
from src.gemini_service import GeminiService

ROLE_QUESTION_BANKS = {
    "Backend Engineer": [
        {
            "id": 1,
            "type": "Technical / Architecture",
            "question": "How do you design a high-throughput, low-latency REST API that handles sudden 10x traffic spikes?",
            "context": "Focus on horizontal scaling, caching strategies (Redis/CDN), connection pooling, and circuit breakers.",
        },
        {
            "id": 2,
            "type": "Database & Storage",
            "question": "Can you walk through a scenario where a database query became a performance bottleneck in production and how you optimized it?",
            "context": "Discuss indexing strategies, query plan analysis (EXPLAIN ANALYZE), schema normalization/denormalization, and read replicas.",
        },
        {
            "id": 3,
            "type": "Behavioral / STAR",
            "question": "Describe a major production outage or bug you introduced or investigated. How did you diagnose the issue, restore service, and prevent recurrence?",
            "context": "Use STAR format: Situation, Task, Action, Result. Highlight blameless post-mortem and observability improvements.",
        },
    ],
    "Full Stack Developer": [
        {
            "id": 1,
            "type": "Architecture & State",
            "question": "How do you manage complex application state between a modern client-side frontend (e.g. React) and asynchronous backend microservices?",
            "context": "Address optimistic updates, cache invalidation, error handling, and WebSocket/polling trade-offs.",
        },
        {
            "id": 2,
            "type": "Security & Auth",
            "question": "Explain how you implement end-to-end authentication and authorization (e.g., OAuth2, JWTs) securely across web clients and APIs.",
            "context": "Mention token refresh mechanics, XSS/CSRF mitigation, HttpOnly cookies, and role-based access control.",
        },
        {
            "id": 3,
            "type": "Behavioral / Collaboration",
            "question": "Tell me about a time you disagreed with a product designer or manager regarding technical feasibility. How did you arrive at an alignment?",
            "context": "Emphasize collaborative trade-off evaluation, user experience focus, and incremental delivery.",
        },
    ],
    "Data Scientist / ML Engineer": [
        {
            "id": 1,
            "type": "ML Modeling",
            "question": "How do you prevent data leakage and overfitting when training predictive machine learning models on high-dimensional temporal datasets?",
            "context": "Discuss time-series cross-validation, feature store isolation, regularization, and out-of-fold validation.",
        },
        {
            "id": 2,
            "type": "Production MLOps",
            "question": "Describe your strategy for deploying a machine learning model to production and monitoring data and concept drift over time.",
            "context": "Cover shadow/canary deployments, latency SLAs, feature distribution monitoring, and automated retraining pipelines.",
        },
        {
            "id": 3,
            "type": "Behavioral / Communication",
            "question": "Describe a project where you translated complex model outputs or mathematical insights into business decisions for non-technical stakeholders.",
            "context": "Highlight visualizations, ROI metrics, stakeholder engagement, and business impact.",
        },
    ],
    "DevOps / Cloud Engineer": [
        {
            "id": 1,
            "type": "Infrastructure as Code",
            "question": "How do you structure modular Terraform or Infrastructure-as-Code repositories to support multiple environments (dev, staging, prod) safely?",
            "context": "Discuss remote state locking, workspace vs directory structuring, parameterization, and drift detection.",
        },
        {
            "id": 2,
            "type": "Resilience & Observability",
            "question": "What is your approach to designing Kubernetes cluster autoscaling, disaster recovery, and multi-region failover?",
            "context": "Address HPA/Cluster Autoscaler, backup strategies, traffic routing, and health probes.",
        },
        {
            "id": 3,
            "type": "Behavioral / Incident Response",
            "question": "Walk through an on-call incident where telemetry alarms fired in the middle of the night. How did you triage and resolve it under pressure?",
            "context": "Demonstrate systematic troubleshooting, communication to leadership, and long-term remediation.",
        },
    ],
    "Data Analyst": [
        {
            "id": 1,
            "type": "SQL & Quantitative Analysis",
            "question": "How do you design, optimize, and validate complex SQL queries (e.g., window functions, CTEs, self-joins) to calculate critical business metrics from large datasets?",
            "context": "Address performance optimization, aggregation accuracy, handling nulls/duplicates, and dimensional modeling.",
        },
        {
            "id": 2,
            "type": "Data Integrity & Problem Solving",
            "question": "When analyzing business performance, how do you handle missing values, anomalies, or discrepancies across disparate data sources before presenting findings to leadership?",
            "context": "Focus on data validation pipelines, outlier detection, stakeholder reconciliation, and reporting confidence intervals.",
        },
        {
            "id": 3,
            "type": "Behavioral / Stakeholder Storytelling",
            "question": "Describe a project where your data analysis directly uncovered an operational bottleneck or influenced a major business decision. How did you present your findings to non-technical stakeholders?",
            "context": "Use STAR format. Emphasize data visualization, translation of statistical findings into business ROI, and executive alignment.",
        },
    ],
    "Supply Chain Analyst": [
        {
            "id": 1,
            "type": "Supply Chain & Inventory Modeling",
            "question": "How do you calculate and optimize inventory reorder points, safety stock, and economic order quantity (EOQ) while balancing customer service levels against working capital holding costs?",
            "context": "Discuss demand volatility, supplier lead-time distribution, stockout cost modeling, and multi-echelon inventory strategies.",
        },
        {
            "id": 2,
            "type": "Logistics Problem Solving",
            "question": "Walk me through how you would diagnose a sudden spike in supplier lead-time variance or inventory stockouts using logistics and ERP purchase-order data.",
            "context": "Cover root cause analysis, supplier scorecarding, freight route optimization, and buffer stock adjustments.",
        },
        {
            "id": 3,
            "type": "Behavioral / Cross-Functional Alignment",
            "question": "Tell me about a situation where your supply chain forecast or capacity model contradicted the intuition of procurement or operations leadership. How did you build consensus and implement changes?",
            "context": "Use STAR format. Highlight empirical model validation, cross-functional empathy, and operational impact.",
        },
    ],
    "Business Analyst": [
        {
            "id": 1,
            "type": "Business Metrics & KPI Frameworks",
            "question": "What analytical frameworks and KPIs do you use to evaluate user funnel drop-offs, customer unit economics, and operational efficiency?",
            "context": "Address cohort analysis, conversion metrics, leading vs lagging indicators, and executive dashboards.",
        },
        {
            "id": 2,
            "type": "Requirements & Process Optimization",
            "question": "How do you prioritize competing business requirements or resolve ambiguous analytical requests from different executive stakeholders?",
            "context": "Focus on impact vs effort prioritization, requirement traceability, and trade-off communication.",
        },
        {
            "id": 3,
            "type": "Behavioral / Change Management",
            "question": "Describe a time when an analytical report you produced was challenged by leadership. How did you validate your assumptions and maintain credibility?",
            "context": "Use STAR format. Highlight transparent methodology, active listening, and objective verification.",
        },
    ],
}

DEFAULT_QUESTIONS = [
    {
        "id": 1,
        "type": "System Design / Technical",
        "question": "Describe the architecture of the most technically challenging software system you have built. What trade-offs did you make?",
        "context": "Discuss scalability, maintainability, technology choices, and bottlenecks.",
    },
    {
        "id": 2,
        "type": "Problem Solving",
        "question": "How do you approach a critical problem when you have incomplete requirements or are working with unfamiliar legacy code?",
        "context": "Focus on root cause analysis, telemetry/logging, test coverage, and incremental refactoring.",
    },
    {
        "id": 3,
        "type": "Behavioral / STAR",
        "question": "Tell me about a time you led or influenced a technical decision that initially faced pushback from teammates. What was the outcome?",
        "context": "Use Situation, Task, Action, Result framework. Highlight empathy, data-driven prototyping, and shared ownership.",
    },
]


class InterviewCoach:
    """Manages interview simulation and structured rubric evaluation."""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    def get_questions_for_role(
        self, role: str, skills: Optional[List[str]] = None, count: int = 3
    ) -> List[Dict[str, Any]]:
        """Return tailored interview questions using Gemini or predefined role banks."""
        role_lower = role.lower().strip()
        selected_bank = None

        # 1. Deterministic bank lookup with domain-awareness
        if any(k in role_lower for k in ["supply chain", "logistics", "procurement", "inventory"]):
            selected_bank = ROLE_QUESTION_BANKS["Supply Chain Analyst"]
        elif any(k in role_lower for k in ["data analyst", "business intelligence", "bi analyst", "analytics"]):
            selected_bank = ROLE_QUESTION_BANKS["Data Analyst"]
        elif any(k in role_lower for k in ["business analyst", "product analyst", "operations analyst"]):
            selected_bank = ROLE_QUESTION_BANKS["Business Analyst"]
        elif any(k in role_lower for k in ["data sci", "machine learning", "ml "]):
            selected_bank = ROLE_QUESTION_BANKS["Data Scientist / ML Engineer"]
        elif any(k in role_lower for k in ["devops", "cloud", "sre", "infrastructure"]):
            selected_bank = ROLE_QUESTION_BANKS["DevOps / Cloud Engineer"]
        elif any(k in role_lower for k in ["full stack", "frontend", "web developer"]):
            selected_bank = ROLE_QUESTION_BANKS["Full Stack Developer"]
        elif any(k in role_lower for k in ["backend", "software engineer", "systems engineer", "api"]):
            selected_bank = ROLE_QUESTION_BANKS["Backend Engineer"]

        if not selected_bank:
            for key in ROLE_QUESTION_BANKS:
                if key.lower() in role_lower or role_lower in key.lower():
                    selected_bank = ROLE_QUESTION_BANKS[key]
                    break

        if not selected_bank:
            if any(k in role_lower for k in ["analyst", "analytics"]):
                selected_bank = ROLE_QUESTION_BANKS["Data Analyst"]
            else:
                selected_bank = [
                    {
                        "id": 1,
                        "type": "Technical & Domain Expertise",
                        "question": f"In your work as a {role}, what core analytical methodologies, domain tools, or technical frameworks do you rely on to deliver rigorous results?",
                        "context": "Demonstrate practical tool mastery, workflow design, and domain depth.",
                    },
                    {
                        "id": 2,
                        "type": "Problem Solving",
                        "question": f"Describe a scenario in your work as a {role} where you faced ambiguous requirements, conflicting data, or tight operational constraints. How did you structure your investigation and deliver a resolution?",
                        "context": "Focus on root cause analysis, methodical hypothesis testing, and stakeholder alignment.",
                    },
                    {
                        "id": 3,
                        "type": "Behavioral / STAR",
                        "question": f"Tell me about a high-impact initiative or project you delivered as a {role}. Walk through the Situation, your specific Task, the Actions you personally executed, and the quantifiable Result achieved.",
                        "context": "Use STAR format: Situation, Task, Action, Result with quantifiable business outcomes.",
                    },
                ]

        # If Gemini is configured, generate personalized questions tailored to candidate's skills
        if self.gemini_service and self.gemini_service.is_configured():
            skills_str = ", ".join(skills[:8]) if skills else "Standard technical competencies"
            prompt = f"""
You are an expert interviewer at a leading organization.
Generate {count} rigorous, realistic interview questions for the role: "{role}".
The candidate has background in: {skills_str}.

CRITICAL GUIDELINE: Tailor all questions specifically to the domain of "{role}". Do NOT ask software architecture or distributed systems questions if the role is a Data Analyst, Supply Chain Analyst, or business analytics position.
Include 1 technical/domain question, 1 problem-solving question, and 1 behavioral (STAR method) question.
Return ONLY valid JSON matching this schema:
[
    {{
        "id": 1,
        "type": "Technical Domain",
        "question": "Full question text",
        "context": "What a top interviewer looks for in a strong response"
    }}
]
"""
            ai_data, is_ai, _ = self.gemini_service.generate_json_response(
                prompt=prompt,
                system_instruction="You are a senior interview coach. Always respond in valid raw JSON array.",
                fallback_data=selected_bank,
            )
            if is_ai and isinstance(ai_data, list) and len(ai_data) > 0:
                return ai_data[:count]

        return selected_bank[:count]

    @classmethod
    def evaluate_answer_deterministic(
        cls, question: str, user_answer: str, role: str = "Software Engineer"
    ) -> Dict[str, Any]:
        """
        Evaluate candidate response using structured heuristics:
        - Relevance (keyword alignment with question)
        - Clarity (sentence length, readability, paragraph organization)
        - Completeness (presence of STAR components and quantified results)
        """
        user_answer = user_answer.strip()
        word_count = len(user_answer.split())

        if word_count < 10:
            return {
                "relevance_score": 25,
                "clarity_score": 30,
                "completeness_score": 15,
                "overall_score": 23.3,
                "feedback": [
                    "Response is too brief for an interview setting.",
                    "Provide a detailed narrative establishing context, specific actions, and measurable outcomes.",
                ],
                "star_analysis": {
                    "situation": False,
                    "task": False,
                    "action": False,
                    "result": False,
                },
                "is_ai_generated": False,
                "provider_note": "Deterministic Heuristic Rubric (Answer too brief for deep analysis)",
            }

        # 1. Relevance: check keyword overlap between question and answer
        q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", question.lower()))
        ans_lower = user_answer.lower()
        matched_q_words = sum(1 for w in q_words if w in ans_lower)
        relevance_score = min(95, max(40, 50 + (matched_q_words * 8)))

        # 2. Clarity: length check, punctuation, capitalization
        clarity_score = 70
        if 50 <= word_count <= 250:
            clarity_score += 20
        elif word_count > 250:
            clarity_score += 10  # A bit verbose
        else:
            clarity_score -= 10  # A bit short

        if re.search(r"[.!?]\s+[A-Z]", user_answer):
            clarity_score += 5  # Proper sentence transitions

        # 3. Completeness & STAR Detection
        star_situation = bool(re.search(r"\b(when|situation|context|project|company|at my|working on)\b", ans_lower))
        star_task = bool(re.search(r"\b(task|goal|objective|needed to|required|responsible for)\b", ans_lower))
        star_action = bool(re.search(r"\b(i built|i designed|i developed|i implemented|i led|i analyzed|i created|i decided)\b", ans_lower))
        star_result = bool(re.search(r"\b(result|outcome|impact|\d+%|\$\d+|improved|reduced|increased|successfully|delivered)\b", ans_lower))

        star_detected_count = sum([star_situation, star_task, star_action, star_result])
        completeness_score = min(98, 40 + (star_detected_count * 14))

        overall_score = round((relevance_score + clarity_score + completeness_score) / 3, 1)

        # Actionable feedback
        feedback = []
        if not star_situation:
            feedback.append("Frame the Context: Briefly specify the project, timeline, and company context before diving into details.")
        if not star_action:
            feedback.append("Own Your Actions: Use first-person active verbs ('I architected', 'I investigated') rather than generic team descriptions ('We did').")
        if not star_result:
            feedback.append("Quantify the Result: Close with measurable business or technical outcomes (e.g., latency dropped by 30%, zero downtime).")
        if word_count < 50:
            feedback.append("Elaborate Further: Aim for 100-200 words to demonstrate depth of technical judgment.")
        elif word_count > 300:
            feedback.append("Practice Conciseness: Avoid over-explaining background details; prioritize the decision-making trade-offs.")

        if not feedback:
            feedback.append("Excellent structured answer! Highlights clear ownership and strong delivery.")

        return {
            "relevance_score": min(100, relevance_score),
            "clarity_score": min(100, clarity_score),
            "completeness_score": min(100, completeness_score),
            "overall_score": overall_score,
            "feedback": feedback,
            "star_analysis": {
                "situation": star_situation,
                "task": star_task,
                "action": star_action,
                "result": star_result,
            },
            "is_ai_generated": False,
            "provider_note": "Deterministic STAR Heuristic Rubric",
        }

    def evaluate_answer(
        self, question: str, user_answer: str, role: str = "Software Engineer"
    ) -> Dict[str, Any]:
        """Evaluate candidate response using Gemini AI or structured rubric fallback."""
        deterministic_eval = self.evaluate_answer_deterministic(question, user_answer, role)

        if self.gemini_service and self.gemini_service.is_configured():
            prompt = f"""
You are a principal engineer and hiring committee chair interviewing a candidate for: "{role}".

INTERVIEW QUESTION:
"{question}"

CANDIDATE ANSWER:
\"\"\"
{user_answer}
\"\"\"

Critique the answer rigorously across three dimensions:
1. Relevance (Does it answer the exact technical question asked?)
2. Clarity (Is it structured, concise, and professional?)
3. Completeness (Does it demonstrate situation, technical depth, actions taken, and measurable impact?)

Return ONLY a valid JSON object matching this schema:
{{
    "relevance_score": 85,
    "clarity_score": 90,
    "completeness_score": 80,
    "overall_score": 85.0,
    "feedback": [
        "Constructive critique point 1",
        "Constructive critique point 2",
        "Constructive critique point 3"
    ],
    "model_answer_summary": "A 2-3 sentence example of how an ideal candidate might articulate the core trade-off or outcome",
    "star_analysis": {{
        "situation": true,
        "task": true,
        "action": true,
        "result": true
    }}
}}
"""
            ai_data, is_ai, note = self.gemini_service.generate_json_response(
                prompt=prompt,
                system_instruction="You are a hiring manager evaluating interview answers. Respond ONLY in raw JSON.",
                fallback_data=deterministic_eval,
            )

            if is_ai and isinstance(ai_data, dict) and "overall_score" in ai_data:
                ai_data["is_ai_generated"] = True
                ai_data["provider_note"] = note
                return ai_data

        return deterministic_eval

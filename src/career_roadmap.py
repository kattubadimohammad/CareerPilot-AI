"""Career Roadmap Module.
Identifies skill gaps, prioritizes learning objectives (P0/P1/P2),
and generates structured development milestones over customizable time horizons.
"""
from typing import Any, Dict, List, Optional
from src.gemini_service import GeminiService

ROLE_TARGET_REQUIREMENTS = {
    "Senior Backend Engineer": {
        "core_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "Redis", "Kafka", "System Design", "AWS", "CI/CD"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Advanced Architecture & Concurrency",
                "focus": "Deep dive into distributed systems, async programming, and database optimization.",
                "action_items": [
                    "Master async/await patterns and background worker pools (Celery / FastAPI background tasks).",
                    "Optimize PostgreSQL query plans (indexing, partitioning, vacuuming) for millions of records.",
                    "Design and implement distributed caching with Redis (cache-aside, write-through, TTL invalidation).",
                ],
                "capstone_project": "Build an event-driven rate limiter and billing service handling 5,000 req/sec.",
            },
            {
                "phase": "Months 3-4: Cloud Infrastructure & Resilience",
                "focus": "Container orchestration, automated deployments, and observability.",
                "action_items": [
                    "Containerize multi-service applications with Docker and helm charts for Kubernetes.",
                    "Set up Prometheus metrics and Grafana dashboards for latency and error monitoring.",
                    "Implement circuit breaker and retry mechanisms using Resilience patterns.",
                ],
                "capstone_project": "Deploy an auto-scaling Kubernetes cluster on AWS EKS with GitOps via GitHub Actions.",
            },
            {
                "phase": "Months 5-6: System Design Mastery & Leadership",
                "focus": "Enterprise scalability, technical RFC authoring, and interview readiness.",
                "action_items": [
                    "Practice 15 high-level system design architectures (e.g., URL shortener, Chat system, Video streaming).",
                    "Author architectural RFCs documenting trade-offs, CAP theorem constraints, and cost estimations.",
                    "Conduct peer code reviews and lead technical discussions on reliability engineering.",
                ],
                "capstone_project": "Complete end-to-end system design portfolio and mock interview simulations.",
            },
        ],
    },
    "AI / Machine Learning Engineer": {
        "core_skills": ["Python", "PyTorch", "Pandas", "NumPy", "Scikit-Learn", "Docker", "FastAPI", "MLOps", "Git", "REST APIs"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Foundations & Model Engineering",
                "focus": "Deep mathematical foundations, neural architectures, and data engineering.",
                "action_items": [
                    "Train and fine-tune Transformer and LLM models using PyTorch and Hugging Face.",
                    "Implement feature engineering pipelines with Pandas and Scikit-Learn.",
                    "Prevent data leakage and validate model generalization across temporal splits.",
                ],
                "capstone_project": "Fine-tune a quantized open-source LLM for domain-specific question answering.",
            },
            {
                "phase": "Months 3-4: Production MLOps & Serving",
                "focus": "Low-latency inference, model registry, and monitoring.",
                "action_items": [
                    "Package PyTorch models with ONNX Runtime or vLLM for high-throughput inference.",
                    "Build model serving endpoints with FastAPI and Docker.",
                    "Implement model drift detection and automated evaluation pipelines.",
                ],
                "capstone_project": "Deploy an end-to-end MLOps pipeline with model tracking and automated retraining triggers.",
            },
            {
                "phase": "Months 5-6: Scalable AI Architecture & Production Systems",
                "focus": "Distributed training, vector databases, and real-time retrieval.",
                "action_items": [
                    "Implement RAG (Retrieval-Augmented Generation) architectures with Vector DBs.",
                    "Evaluate LLM outputs using automated metrics and safety guardrails.",
                    "Scale batch embedding pipelines across cloud compute clusters.",
                ],
                "capstone_project": "Enterprise Agentic RAG system with document citation and evaluation suite.",
            },
        ],
    },
    "Cloud & DevOps Architect": {
        "core_skills": ["AWS", "Terraform", "Kubernetes", "Docker", "Linux", "CI/CD", "GitHub Actions", "Prometheus", "Grafana", "Python"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Infrastructure as Code & Automation",
                "focus": "Production Terraform architecture and immutable infrastructure.",
                "action_items": [
                    "Structure reusable, multi-environment Terraform modules with remote state locking.",
                    "Automate linting, security scanning (Tfsec), and CI/CD for cloud infrastructure.",
                    "Master Linux networking, systemd, and bash automation.",
                ],
                "capstone_project": "Automated multi-tier VPC network with private subnets and NAT gateways via Terraform.",
            },
            {
                "phase": "Months 3-4: Kubernetes Mastery & GitOps",
                "focus": "Production cluster administration and progressive delivery.",
                "action_items": [
                    "Configure Kubernetes RBAC, NetworkPolicies, and ingress controllers.",
                    "Implement GitOps deployments with ArgoCD or Flux.",
                    "Configure automated cluster autoscaling and spot instance optimization.",
                ],
                "capstone_project": "Multi-tenant Kubernetes platform with zero-downtime rolling deployments.",
            },
            {
                "phase": "Months 5-6: Observability, Security & Cost Governance",
                "focus": "Site Reliability Engineering (SRE) practices and enterprise governance.",
                "action_items": [
                    "Define SLOs, SLIs, and error budgets for mission-critical microservices.",
                    "Implement distributed tracing with OpenTelemetry and Jaeger.",
                    "Conduct cloud spend audits and optimize resource sizing.",
                ],
                "capstone_project": "Enterprise SRE dashboard with automated alerting, runbooks, and FinOps policies.",
            },
        ],
    },
}

GENERIC_ROADMAP_TEMPLATE = [
    {
        "phase": "Phase 1 (Months 1-2): Core Prerequisite Competencies",
        "focus": "Closing highest-priority technical skill gaps and modernizing toolchain.",
        "action_items": [
            "Complete structured deep dive on core missing skills identified in assessment.",
            "Build practical, hands-on portfolio mini-projects demonstrating competency.",
            "Establish automated unit testing and version control best practices.",
        ],
        "capstone_project": "Production-grade microservice or tool incorporating primary priority skills.",
    },
    {
        "phase": "Phase 2 (Months 3-4): System Architecture & Integration",
        "focus": "End-to-end integration, performance optimization, and best practices.",
        "action_items": [
            "Implement CI/CD deployment pipelines and containerization.",
            "Optimize database queries and implement caching layers.",
            "Document architecture decisions and trade-offs.",
        ],
        "capstone_project": "Full-stack scalable application deployed to cloud with automated tests.",
    },
    {
        "phase": "Phase 3 (Months 5-6): Industry Positioning & Interview Mastery",
        "focus": "Portfolio presentation, leadership presence, and interview execution.",
        "action_items": [
            "Publish open-source code and technical articles detailing design challenges.",
            "Practice behavioral and technical architecture mock interviews.",
            "Target applications to high-fit roles aligned with updated resume.",
        ],
        "capstone_project": "Comprehensive technical portfolio and active role applications.",
    },
]


class CareerRoadmapGenerator:
    """Generates structured career roadmaps, skill gap analyses, and prioritized milestones."""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    @classmethod
    def identify_skill_gaps(
        cls, current_skills: List[str], target_role: str
    ) -> Dict[str, Any]:
        """Compare candidate's verified skills against target role benchmarks."""
        target_norm = target_role.strip()
        matched_role_key = None
        for key in ROLE_TARGET_REQUIREMENTS:
            if key.lower() in target_norm.lower() or target_norm.lower() in key.lower():
                matched_role_key = key
                break

        current_lower = {s.lower() for s in current_skills}

        if matched_role_key:
            benchmark_skills = ROLE_TARGET_REQUIREMENTS[matched_role_key]["core_skills"]
            milestones = ROLE_TARGET_REQUIREMENTS[matched_role_key]["milestones_template"]
        else:
            # Create standard benchmark based on target role
            benchmark_skills = ["System Design", "Cloud Infrastructure", "CI/CD", "Database Optimization", "Testing", "Security"]
            milestones = GENERIC_ROADMAP_TEMPLATE

        missing_skills = [s for s in benchmark_skills if s.lower() not in current_lower]
        existing_skills = [s for s in benchmark_skills if s.lower() in current_lower]

        # Categorize into P0, P1, P2
        p0_critical = missing_skills[:3]
        p1_important = missing_skills[3:6]
        p2_nice_to_have = missing_skills[6:]

        return {
            "target_role": target_role,
            "benchmark_skills": benchmark_skills,
            "existing_skills": existing_skills,
            "missing_skills": missing_skills,
            "learning_priorities": {
                "P0_Critical": p0_critical,
                "P1_Important": p1_important,
                "P2_Nice_To_Have": p2_nice_to_have,
            },
            "milestones": milestones,
        }

    def generate_roadmap(
        self, current_skills: List[str], target_role: str, timeline_months: int = 6
    ) -> Dict[str, Any]:
        """
        Generate a personalized career roadmap.
        Combines deterministic baseline with Gemini AI enrichment when configured.
        """
        deterministic_gap = self.identify_skill_gaps(current_skills, target_role)

        result_data = {
            "target_role": target_role,
            "timeline_months": timeline_months,
            "current_skills_count": len(current_skills),
            "skill_gaps": deterministic_gap["missing_skills"],
            "existing_strengths": deterministic_gap["existing_skills"],
            "learning_priorities": deterministic_gap["learning_priorities"],
            "milestones": deterministic_gap["milestones"],
            "is_ai_generated": False,
            "provider_note": "Deterministic Career Roadmap Engine",
        }

        # If Gemini is configured, generate a personalized AI roadmap
        if self.gemini_service and self.gemini_service.is_configured():
            skills_str = ", ".join(current_skills) if current_skills else "General technical skills"
            prompt = f"""
You are an executive career coach and engineering director.
Create a tailored {timeline_months}-month career development roadmap for a professional aiming to become a:
TARGET ROLE: "{target_role}"

CURRENT VERIFIED SKILLS:
{skills_str}

Return ONLY valid JSON matching this structure:
{{
    "target_role": "{target_role}",
    "strategic_overview": "2-3 sentence strategic roadmap perspective",
    "learning_priorities": {{
        "P0_Critical": ["Skill 1", "Skill 2"],
        "P1_Important": ["Skill 3", "Skill 4"],
        "P2_Nice_To_Have": ["Skill 5"]
    }},
    "milestones": [
        {{
            "phase": "Phase 1: Months 1-2",
            "focus": "Theme of this phase",
            "action_items": [
                "Actionable step 1",
                "Actionable step 2",
                "Actionable step 3"
            ],
            "capstone_project": "Concrete portfolio project demonstrating mastery"
        }}
    ],
    "recommended_certifications_and_resources": [
        "Resource 1",
        "Resource 2"
    ]
}}
"""
            ai_data, is_ai, note = self.gemini_service.generate_json_response(
                prompt=prompt,
                system_instruction="You are an expert career roadmap strategist. Respond ONLY with valid raw JSON.",
                fallback_data=result_data,
            )

            if is_ai and isinstance(ai_data, dict) and "milestones" in ai_data:
                ai_data["is_ai_generated"] = True
                ai_data["provider_note"] = note
                ai_data["current_skills_count"] = len(current_skills)
                return ai_data

        return result_data

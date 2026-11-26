"""Career Roadmap Module.
Identifies skill gaps, prioritizes learning objectives (P0/P1/P2),
and generates structured development milestones over customizable time horizons.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from src.gemini_service import GeminiService

SKILL_SYNONYMS = {
    "python": ["python", "python3", "python 3", "py"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "pl/sql", "tsql", "t-sql", "sqlite", "oracle sql"],
    "aws": ["aws", "amazon web services", "aws cloud", "ec2", "s3", "rds"],
    "docker": ["docker", "dockerfile", "container", "containers", "containerization"],
    "kubernetes": ["kubernetes", "k8s", "helm"],
    "ci/cd": ["ci/cd", "ci/cd pipelines", "continuous integration", "github actions", "gitlab ci", "jenkins"],
    "fastapi": ["fastapi", "fast api"],
    "tableau": ["tableau", "tableau desktop", "tableau server"],
    "power bi": ["power bi", "powerbi", "dax", "power query"],
    "excel": ["excel", "ms excel", "microsoft excel", "spreadsheets", "advanced excel"],
    "supply chain analytics": ["supply chain", "supply chain analytics", "supply chain management", "logistics", "inventory"],
    "data visualization": ["data visualization", "visualization", "tableau", "power bi", "dashboarding", "matplotlib", "seaborn"],
    "etl": ["etl", "elt", "data pipeline", "data pipelines", "data ingestion"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn", "predictive modeling"],
    "git": ["git", "github", "gitlab"],
    "rest apis": ["rest apis", "rest", "restful", "rest api", "fastapi", "flask", "django"],
}

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
    "Data Analyst / Business Intelligence": {
        "core_skills": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Data Visualization", "ETL", "Statistical Analysis", "Data Modeling", "Business Metrics"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Advanced SQL & Business Metric Modeling",
                "focus": "Complex analytical queries, dimensional modeling, and metric definitions.",
                "action_items": [
                    "Master window functions, CTEs, self-joins, and query execution plans for analytics datasets.",
                    "Build star and snowflake schemas supporting automated BI reporting tables.",
                    "Define and document core business KPIs (CAC, LTV, churn, conversion rates).",
                ],
                "capstone_project": "Automated analytical data mart and financial KPI reporting pipeline in SQL.",
            },
            {
                "phase": "Months 3-4: Interactive BI Storytelling & Automation",
                "focus": "Enterprise dashboards, automated refresh schedules, and stakeholder communication.",
                "action_items": [
                    "Design interactive executive dashboards in Power BI / Tableau with drill-through analytics.",
                    "Automate data extraction and cleaning scripts with Python (Pandas) and scheduled jobs.",
                    "Conduct stakeholder interview simulations translating ambiguous business goals into analytics specs.",
                ],
                "capstone_project": "End-to-end executive performance dashboard adopted by business decision-makers.",
            },
            {
                "phase": "Months 5-6: Statistical Analysis & Strategic Insights",
                "focus": "Hypothesis testing, exploratory forecasting, and decision advocacy.",
                "action_items": [
                    "Implement A/B test analysis, cohort segmentation, and variance decomposition.",
                    "Author strategic insight memos detailing actionable recommendations based on data findings.",
                    "Lead data review sessions with cross-functional leadership.",
                ],
                "capstone_project": "Comprehensive strategic analytics case study with executive presentation deck.",
            },
        ],
    },
    "Supply Chain Analyst / Operations Analytics": {
        "core_skills": ["Supply Chain Analytics", "SQL", "Inventory Optimization", "Demand Forecasting", "Excel", "Power BI", "Python", "Logistics Modeling", "ERP Systems", "Data Visualization"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Inventory Modeling & Operational Data Foundations",
                "focus": "Safety stock calculation, reorder point modeling, and ERP transaction analysis.",
                "action_items": [
                    "Model Economic Order Quantity (EOQ), reorder triggers, and safety stock factoring lead-time variance.",
                    "Extract and transform purchase orders, shipping manifests, and inventory balances with SQL.",
                    "Analyze SKU-level turnover rates and identify slow-moving or obsolete inventory.",
                ],
                "capstone_project": "Multi-echelon inventory optimization model reducing working capital requirements.",
            },
            {
                "phase": "Months 3-4: Demand Forecasting & Supply Network Visibility",
                "focus": "Time-series forecasting, logistics tracking, and supplier scorecarding.",
                "action_items": [
                    "Implement moving average, exponential smoothing, and seasonal demand forecasts in Python/Excel.",
                    "Build Power BI dashboards tracking on-time in-full (OTIF) delivery and supplier lead times.",
                    "Simulate supply chain disruption scenarios and bottleneck mitigation paths.",
                ],
                "capstone_project": "Interactive supply chain control tower dashboard with automated stockout alerts.",
            },
            {
                "phase": "Months 5-6: Strategic Logistics Planning & S&OP Execution",
                "focus": "Sales and Operations Planning (S&OP), freight optimization, and executive reporting.",
                "action_items": [
                    "Design S&OP monthly consensus models aligning sales forecasts with procurement capacity.",
                    "Optimize freight route allocation and carrier cost models.",
                    "Present quarterly supply chain efficiency recommendations to operations leadership.",
                ],
                "capstone_project": "Complete S&OP analytical framework with procurement cost-reduction roadmap.",
            },
        ],
    },
    "Data Engineer": {
        "core_skills": ["Python", "SQL", "Apache Spark", "Airflow", "Data Warehousing", "PostgreSQL", "Kafka", "Docker", "AWS", "Data Modeling"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Batch & Stream Ingestion Pipelines",
                "focus": "Distributed data processing and workflow orchestration.",
                "action_items": [
                    "Build robust DAG workflows with Apache Airflow incorporating retries and SLAs.",
                    "Process large-scale datasets with PySpark using partitioning and memory caching.",
                    "Design schema evolution strategies for raw and curated data lakes.",
                ],
                "capstone_project": "Production batch ingestion pipeline processing 10M records daily with Airflow.",
            },
            {
                "phase": "Months 3-4: Cloud Data Warehousing & Real-Time Streams",
                "focus": "High-throughput messaging and modern cloud warehouses.",
                "action_items": [
                    "Stream events with Apache Kafka or AWS Kinesis to real-time subscribers.",
                    "Design dimensional models in Snowflake or BigQuery with clustering keys.",
                    "Implement automated data quality testing with Great Expectations or dbt.",
                ],
                "capstone_project": "Real-time analytics pipeline feeding a cloud data warehouse with automated quality gates.",
            },
            {
                "phase": "Months 5-6: Performance Tuning, Cost Governance & Architecture",
                "focus": "Query acceleration, data governance, and scalable storage tiering.",
                "action_items": [
                    "Tune Spark cluster configurations, shuffle partitions, and broadcast joins.",
                    "Establish data lineage and access governance frameworks.",
                    "Implement cold-storage lifecycle policies cutting warehouse cost by 30%.",
                ],
                "capstone_project": "Enterprise data platform blueprint with cost optimization audit.",
            },
        ],
    },
    "Full Stack Developer": {
        "core_skills": ["JavaScript", "TypeScript", "React", "Python", "Node.js", "REST APIs", "SQL", "Docker", "Git", "HTML/CSS"],
        "milestones_template": [
            {
                "phase": "Months 1-2: Frontend State Architecture & Modern UI",
                "focus": "Component lifecycle, responsive design, and state management.",
                "action_items": [
                    "Build accessible, responsive UI components with React, TypeScript, and modern CSS.",
                    "Implement client-side caching and optimistic UI updates.",
                    "Write automated end-to-end frontend tests with Playwright or Cypress.",
                ],
                "capstone_project": "Responsive enterprise web dashboard with real-time state synchronization.",
            },
            {
                "phase": "Months 3-4: Scalable API Services & Database Integration",
                "focus": "Secure RESTful/GraphQL APIs and relational schema design.",
                "action_items": [
                    "Develop secure REST APIs with Python (FastAPI) or Node.js with JWT authentication.",
                    "Model relational entities in PostgreSQL with automated database migrations.",
                    "Containerize the full-stack application using Docker Compose.",
                ],
                "capstone_project": "Full-stack SaaS application with authentication, billing, and automated test suite.",
            },
            {
                "phase": "Months 5-6: Cloud Deployment, Performance & Production Delivery",
                "focus": "CI/CD automation, asset optimization, and observability.",
                "action_items": [
                    "Deploy frontend and backend services to cloud containers with automated CI/CD.",
                    "Optimize Core Web Vitals, bundle size, and server response times.",
                    "Implement error tracking with Sentry and performance monitoring.",
                ],
                "capstone_project": "Production-deployed application with 99.9% uptime SLA and zero-downtime releases.",
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
        "capstone_project": "Production-grade project or tool incorporating primary priority skills.",
    },
    {
        "phase": "Phase 2 (Months 3-4): System Architecture & Integration",
        "focus": "End-to-end integration, performance optimization, and best practices.",
        "action_items": [
            "Implement automated deployment workflows and containerization.",
            "Optimize data flow, query performance, and reporting speed.",
            "Document architecture decisions, trade-offs, and governance guidelines.",
        ],
        "capstone_project": "Full-scale application or analytical dashboard deployed with automated tests.",
    },
    {
        "phase": "Phase 3 (Months 5-6): Industry Positioning & Interview Mastery",
        "focus": "Portfolio presentation, leadership presence, and interview execution.",
        "action_items": [
            "Publish open-source code and technical write-ups detailing operational challenges.",
            "Practice behavioral and technical architecture mock interviews.",
            "Target applications to high-fit roles aligned with verified competency roadmap.",
        ],
        "capstone_project": "Comprehensive professional portfolio and active role applications.",
    },
]


def _has_skill(benchmark_skill: str, current_skills: List[str]) -> bool:
    """Return True if benchmark_skill is matched or proven in current_skills."""
    b_norm = benchmark_skill.strip().lower()
    if not b_norm or not current_skills:
        return False

    # 1. Exact case-insensitive match
    current_norms = [s.strip().lower() for s in current_skills if s and s.strip()]
    if b_norm in current_norms:
        return True

    # 2. Known synonym dictionary check
    for canonical, syns in SKILL_SYNONYMS.items():
        canonical_matches_b = (b_norm == canonical or b_norm in syns or canonical in b_norm)
        if canonical_matches_b:
            for c_skill in current_norms:
                if c_skill == canonical or c_skill in syns or canonical in c_skill:
                    return True

    # 3. Substring / token boundary match (e.g. "AWS (EC2, S3)" matches "AWS")
    b_tokens = set(re.findall(r"[a-z0-9+#]+", b_norm))
    for c_skill in current_norms:
        c_tokens = set(re.findall(r"[a-z0-9+#]+", c_skill))
        # Direct word match (e.g. 'aws' in {'aws', 'ec2', 's3'})
        if b_tokens and b_tokens.issubset(c_tokens):
            return True
        # If benchmark is short single word (>= 2 chars), check if token matches
        if len(b_norm) >= 2 and b_norm in c_tokens:
            return True
        # Check if candidate skill is a token in benchmark (e.g. 'postgres' in 'postgresql')
        for bt in b_tokens:
            if bt in c_tokens:
                if len(bt) >= 3:
                    return True

    return False


def _get_custom_role_benchmark(target_role: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Dynamically determine domain-appropriate benchmark skills and milestones for custom roles."""
    role_lower = target_role.lower()

    if any(k in role_lower for k in ["supply chain", "logistics", "procurement", "inventory"]):
        return (
            ROLE_TARGET_REQUIREMENTS["Supply Chain Analyst / Operations Analytics"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["Supply Chain Analyst / Operations Analytics"]["milestones_template"],
        )
    if any(k in role_lower for k in ["data analyst", "business intelligence", "bi analyst", "analytics"]):
        return (
            ROLE_TARGET_REQUIREMENTS["Data Analyst / Business Intelligence"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["Data Analyst / Business Intelligence"]["milestones_template"],
        )
    if any(k in role_lower for k in ["data engineer", "etl", "data warehouse", "pipeline"]):
        return (
            ROLE_TARGET_REQUIREMENTS["Data Engineer"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["Data Engineer"]["milestones_template"],
        )
    if any(k in role_lower for k in ["machine learning", "data scientist", "deep learning", "ai engineer", "nlp"]):
        return (
            ROLE_TARGET_REQUIREMENTS["AI / Machine Learning Engineer"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["AI / Machine Learning Engineer"]["milestones_template"],
        )
    if any(k in role_lower for k in ["devops", "cloud", "sre", "platform", "infrastructure"]):
        return (
            ROLE_TARGET_REQUIREMENTS["Cloud & DevOps Architect"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["Cloud & DevOps Architect"]["milestones_template"],
        )
    if any(k in role_lower for k in ["full stack", "frontend", "web developer", "react"]):
        return (
            ROLE_TARGET_REQUIREMENTS["Full Stack Developer"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["Full Stack Developer"]["milestones_template"],
        )
    if any(k in role_lower for k in ["backend", "api", "microservice", "python developer"]):
        return (
            ROLE_TARGET_REQUIREMENTS["Senior Backend Engineer"]["core_skills"],
            ROLE_TARGET_REQUIREMENTS["Senior Backend Engineer"]["milestones_template"],
        )

    # General fallback tailored to professional technical competencies (avoiding strict software architecture)
    default_skills = ["Data Analysis", "Project Management", "SQL", "Technical Communication", "Automation", "Process Optimization", "Reporting"]
    return default_skills, GENERIC_ROADMAP_TEMPLATE


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

        if matched_role_key:
            benchmark_skills = ROLE_TARGET_REQUIREMENTS[matched_role_key]["core_skills"]
            milestones = ROLE_TARGET_REQUIREMENTS[matched_role_key]["milestones_template"]
        else:
            benchmark_skills, milestones = _get_custom_role_benchmark(target_role)

        existing_skills = [s for s in benchmark_skills if _has_skill(s, current_skills)]
        missing_skills = [s for s in benchmark_skills if not _has_skill(s, current_skills)]

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
            missing_str = ", ".join(deterministic_gap["missing_skills"]) if deterministic_gap["missing_skills"] else "None identified"
            prompt = f"""
You are an executive career coach and engineering director.
Create a tailored {timeline_months}-month career development roadmap for a professional aiming to become a:
TARGET ROLE: "{target_role}"

CURRENT VERIFIED SKILLS (DO NOT recommend these as gaps or priorities):
{skills_str}

IDENTIFIED GAPS:
{missing_str}

CRITICAL GUARDRAIL: Do NOT recommend any of the candidate's existing verified skills as critical gaps.
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

#!/usr/bin/env python3
"""Generate AI·데이터 기반 심리학과 Inno-Edu MVP using AI-CoScientist multi-agent stack."""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.base import ResearchAgent
from src.agents.domain_experts import NeuroscienceExpertAgent
from src.agents.pool import AgentPool
from src.agents.types import AgentTask, AgentResult, TaskType
from src.context.manager import ResearchContextManager
from src.context.types import Insight
from src.router.meta_router import MetaRouter
from src.router.types import ResearchTask
from src.services.llm.types import LLMResponse, ModelProvider


class StubLLMService:
    """LLM stub that returns deterministic task profiles based on description keywords."""

    async def complete(self, request):  # type: ignore[override]
        prompt = request.prompt
        text = prompt.lower()
        domains: List[str] = []
        keywords: List[str] = []

        def add_domain(name: str, tokens: List[str]):
            if name not in domains:
                domains.append(name)
            keywords.extend(tokens)

        if any(word in text for word in ["curriculum", "커리큘럼", "모듈", "roadmap", "학사"]):
            add_domain("curriculum_design", ["curriculum", "모듈", "pathway"])
        if any(word in text for word in ["project", "pbl", "project-based", "프로젝트", "탐험"]):
            add_domain("project_based_learning", ["pbl", "studio", "탐험"])
        if any(word in text for word in ["ai", "인공지능", "데이터", "ml", "머신"]):
            add_domain("ai_integration", ["ai", "data", "ml"])
        if any(word in text for word in ["pipeline", "데이터", "infra", "인프라"]):
            add_domain("data_infrastructure", ["pipeline", "infra"])
        if any(word in text for word in ["assessment", "평가", "kpi", "성과"]):
            add_domain("assessment", ["kpi", "evaluation"])
        if any(word in text for word in ["governance", "거버넌스", "tf", "운영", "로드맵", "mvp"]):
            add_domain("governance", ["governance", "roadmap"])
        if any(word in text for word in ["neuro", "brain", "뇌"]):
            add_domain("neuroscience", ["neuro", "brain", "뇌"])
        if "심리" in text or "psychology" in text:
            add_domain("psychology", ["psychology", "심리"])

        if not domains:
            domains = ["general"]

        complexity = "high" if len(domains) >= 3 else ("medium" if len(domains) == 2 else "simple")

        profile = {
            "domains": domains,
            "complexity": complexity,
            "task_type": request.task_type.value if hasattr(request.task_type, "value") else "proposal_development",
            "sub_tasks": [{"id": "main", "type": "synthesis"}],
            "required_expertise": domains,
            "quality_gates": ["alignment", "feasibility"],
            "context_dependencies": ["educational_philosophy", "stakeholder_feedback"],
            "keywords": keywords,
        }

        return LLMResponse(
            content=json.dumps(profile, ensure_ascii=False),
            model="stub-llm",
            provider=ModelProvider.LOCAL,
            tokens_used=0,
            cost=0.0,
            latency_ms=5.0,
            finish_reason="stop",
            metadata={"stub": True},
        )


class CustomNeuroscienceAgent(NeuroscienceExpertAgent):
    """Override to return meaningful insight without external LLM."""

    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities.extend(["neuroscience", "curriculum_design", "project_based_learning"])
        self.domains.extend(["curriculum_design", "project_based_learning"])
        self.specializations.extend(["brain_based_learning", "complex_systems"])

    async def process(self, task: AgentTask, relevant_context: Dict[str, Any]) -> AgentResult:
        insights = relevant_context.get("insights", [])
        summary_points = [ins.content for ins in insights[:3]]
        output = {
            "핵심철학": "행동이 의미를 만들고 복잡계로서의 뇌가 탐험에서 학습을 창발시킨다는 관점을 커리큘럼의 출발점으로 삼습니다.",
            "요약": summary_points,
            "신경과학_시사점": [
                "행동→감각→의미 순환을 수업 구조에 반영하여 프로젝트 탐색을 선행",
                "복잡계 관점으로 다중 모달 데이터를 연결해 자기조직화 학습 경험 설계",
                "Corollary discharge 개념을 적용해 자기 예측·리플렉션 루프를 강조",
            ],
        }

        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.9,
        )


class CurriculumStrategistAgent(ResearchAgent):
    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = ["curriculum_design", "project_based_learning", "psychology"]
        self.domains = ["curriculum_design", "project_based_learning", "psychology"]
        self.specializations = ["modular_pathways", "brain_based_learning"]

    async def process(self, task: AgentTask, relevant_context: Dict[str, Any]) -> AgentResult:
        output = {
            "비전": "행동·탐험 중심 Brain-Data Studio로 심리학과를 AI 시대 두뇌과학 허브로 전환",
            "모듈": [
                "Brain & Behavior Data Lab (신경·행동 데이터 탐구)",
                "AI Psychology Studio (생성형 AI를 활용한 심리 서비스 디자인)",
                "Human-Centered Interventions Clinic (정책·서비스 실험)",
                "Ethics & Societal Impact Lab (책임 AI 및 법·윤리)"
            ],
            "학사구조": [
                "1학년: '행동하는 뇌' 기반 탐색 세미나 + 데이터 리터러시 부트캠프",
                "2학년: 역량 모듈별 프로젝트 스튜디오(15학점 마이크로 크레덴셜)",
                "3학년: 산학·지역 연계 PBL, AI 모듈과 교차 수강, 문제정의 Practicum",
                "4학년: 통합 캡스톤 + 연구·창업 옵션(뇌-데이터 논문, 제품, 정책 제안 중 선택)"
            ],
        }
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.88,
        )


class AIInfrastructureAgent(ResearchAgent):
    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = ["ai_integration", "data_infrastructure", "psychology"]
        self.domains = ["ai_integration", "data_infrastructure", "psychology"]
        self.specializations = ["learning_stack", "data_governance"]

    async def process(self, task: AgentTask, relevant_context: Dict[str, Any]) -> AgentResult:
        output = {
            "플랫폼": {
                "데이터": "Secure Behavioral Data Lake (IRB 가이드 연동) + Synthetic Data Sandbox",
                "AI도구": ["오픈소스 LLM 파운데이션(Colab/Vertex)", "심리 전공 맞춤 프롬프트 허브", "시뮬레이션 도구(EEG/fMRI 가상 데이터)"]
            },
            "지원체계": [
                "데이터 큐레이터 + 윤리 담당이 포함된 AI 학습 지원센터",
                "Cloud Credits 확보(삼성/SKT) 및 GPU Lab 타임쉐어링",
                "학생용 AI Toolchain 인증(버전 관리, 재현성 스택)"
            ],
            "정책": [
                "생성형 AI 활용 가이드라인(투명성·인용·리스크 관리)",
                "데이터 활용 3단계 승인(연구용, 수업용, 공개용)",
                "다전공·비전공 학생의 접근을 위한 계층형 권한 체계"
            ],
        }
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.87,
        )


class PBLEcosystemAgent(ResearchAgent):
    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = ["project_based_learning", "curriculum_design", "psychology"]
        self.domains = ["project_based_learning", "psychology"]
        self.specializations = ["studio_model", "community_partnerships"]

    async def process(self, task: AgentTask, relevant_context: Dict[str, Any]) -> AgentResult:
        output = {
            "프로젝트_팩": [
                "NeuroAI Discovery Sprint: 신경데이터 기반 행동 예측 모델 개발",
                "Wellbeing Policy Lab: 공공 데이터로 정신건강 정책 시나리오 시뮬레이션",
                "UX for Cognitive Tech: HCI 실험과 사용자 여정 데이터 분석",
            ],
            "운영": [
                "4주 모듈형 스프린트 + 8주 확장형 프로젝트",
                "산학 멘토풀(디지털 헬스케어, 에듀테크, 정책 연구소) 매칭",
                "Action→Reflection→Generalization 루프를 기록하는 Project Logbook"],
            "학생지원": [
                "역량 뱃지 시스템(모델링, 인터벤션 디자인, 윤리 심사)",
                "동료 코칭과 AI 코파일럿 튜터 결합",
                "다전공 학생이 참여할 수 있는 교차 등록 좌석 30% 확보"
            ],
        }
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.86,
        )


class AssessmentGovernanceAgent(ResearchAgent):
    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = ["assessment", "governance", "psychology"]
        self.domains = ["assessment", "governance", "psychology"]
        self.specializations = ["kpi_design", "continuous_improvement"]

    async def process(self, task: AgentTask, relevant_context: Dict[str, Any]) -> AgentResult:
        output = {
            "공통지표": {
                "외부리뷰": "2026.06까지 다중 전문가 리뷰 완료, 권고사항 이행률 80%",
                "교과 개편": "30% 이상 과목 구조 개편, 모듈별 학습성과 루브릭 구축",
                "수업혁신": "PBL·스튜디오 수업 비중 50%, 행동기반 평가 도입"
            },
            "자율지표": [
                "AI 심리데이터 캡스톤 참여 학생 150명/년",
                "비전공 학생 교차 등록 120명/년",
                "산학공동 프로젝트 후속 연구·창업 10건/년",
                "학생 만족도(프로젝트 몰입감) 75% 이상"
            ],
            "거버넌스": [
                "Psychology Innovation Council(TF) + Data/Ethics Subcommittee",
                "분기별 KPI 대시보드 공개, 연 2회 타 단대 협력 점검",
                "학생·동문 자문 패널과 산업 자문위원회 이중 구조"
            ],
        }
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.85,
        )


class ImplementationAgent(ResearchAgent):
    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = ["governance", "implementation", "psychology"]
        self.domains = ["governance", "implementation", "psychology"]
        self.specializations = ["mvp_delivery", "milestone_planning"]

    async def process(self, task: AgentTask, relevant_context: Dict[str, Any]) -> AgentResult:
        output = {
            "타임라인": [
                "2025.11~12: TF 구성, 교육철학 합의, 데이터 자산 현황 조사",
                "2026 Q1: 학생·동문·산업 설문 및 심층 인터뷰, 외부 리뷰 위촉",
                "2026 Q2: 모듈 설계 워크숍, AI 인프라 PoC, 파일럿 프로젝트 착수",
                "2026 Q3: 교과심의 및 학사제도 정비, 평가 루브릭 베타 테스트",
                "2026 Q4: 1차년도 결과보고(연차점검) + 2차년도 실행 계획 확정"
            ],
            "MVP_산출물": [
                "Brain-Data Studio 커리큘럼 맵 & 학습자 여정",
                "AI 학습 지원 플랫폼 베타 + 안전 가이드라인",
                "3종 프로젝트 팩 & 평가 루브릭",
                "KPI 대시보드(실시간 데이터 파이프라인)",
                "산학·지역 파트너십 MOA 5건"
            ],
            "리스크관리": [
                "데이터 윤리: IRB·정보보호팀과 공동 프로토콜",
                "교원 역량: Teaching Innovation Pods 및 마이크로 자격",
                "자원: 학내 공용 GPU, 산학 협약을 통한 추가 예산 레버리지"
            ],
        }
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.86,
        )


async def main():
    """Run multi-agent pipeline and save aggregated outputs."""

    output_dir = Path("/Users/jiookcha/Library/CloudStorage/OneDrive-Personal/_Documents/_그랜트/00INOEDU/SNU-Psychology-InnoEdu2031/docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    stub_llm = StubLLMService()
    context_manager = ResearchContextManager(vector_store=None, graph_db=None)
    agent_pool = AgentPool(stub_llm, context_manager)

    # Override neuroscience agent with richer output
    agent_pool.agents["neuroscience_expert"] = CustomNeuroscienceAgent(
        "neuroscience_expert",
        stub_llm,
        context_manager,
    )

    # Register additional agents
    agent_pool.agents["curriculum_strategist"] = CurriculumStrategistAgent("curriculum_strategist", stub_llm, context_manager)
    agent_pool.agents["ai_infrastructure"] = AIInfrastructureAgent("ai_infrastructure", stub_llm, context_manager)
    agent_pool.agents["pbl_ecosystem"] = PBLEcosystemAgent("pbl_ecosystem", stub_llm, context_manager)
    agent_pool.agents["assessment_governance"] = AssessmentGovernanceAgent("assessment_governance", stub_llm, context_manager)
    agent_pool.agents["implementation_lead"] = ImplementationAgent("implementation_lead", stub_llm, context_manager)

    # Seed context with key insights from 보육학회 자료와 학과 요구
    seed_insights = [
        Insight(
            content="행동은 인지와 의미의 선행조건이며 탐험-예측-피드백 구조가 학습의 본질",
            type="philosophy",
            domains=["curriculum_design", "psychology"],
            score=0.92,
            concepts=["behavior-first", "corollary discharge"],
        ),
        Insight(
            content="뇌는 복잡계로서 상호작용과 창발을 통해 발달하므로 프로젝트 기반 상호작용 설계가 필수",
            type="philosophy",
            domains=["project_based_learning", "psychology"],
            score=0.9,
            concepts=["complex systems", "emergence"],
        ),
        Insight(
            content="AI 시대 심리학은 데이터 해석·윤리·정책·서비스를 통합해야 하며 다전공 학습자의 교차 진입을 보장해야 함",
            type="needs",
            domains=["ai_integration", "governance"],
            score=0.88,
            concepts=["ai-readiness", "cross-disciplinary"],
        ),
    ]
    for insight in seed_insights:
        await context_manager.store_insight(insight, "seed", "init", {})

    meta_router = MetaRouter(stub_llm, agent_pool, context_manager)
    task_analyzer = meta_router.task_analyzer
    agent_matcher = meta_router.agent_matcher

    tasks = [
        ResearchTask(
            description="행동하는 뇌 교육철학과 신경과학 근거를 제시",
            task_type="proposal_development",
            prior_work="보육학회 발표 슬라이드",
        ),
        ResearchTask(
            description="심리학과 학부 AI·데이터 기반 커리큘럼 아키텍처와 모듈 설계를 정의",
            task_type="proposal_development",
            prior_work="보육학회 발표 철학 요약",
        ),
        ResearchTask(
            description="생성형 AI와 행동데이터를 활용하는 학습 인프라·정책을 설계",
            task_type="proposal_development",
            prior_work="심리학과 데이터 자산 현황 조사",
        ),
        ResearchTask(
            description="프로젝트 기반 학습 운영 모델과 산학·지역 협력 생태계를 구체화",
            task_type="proposal_development",
        ),
        ResearchTask(
            description="성과지표와 거버넌스·평가 체계를 수립",
            task_type="proposal_development",
        ),
        ResearchTask(
            description="1차년도 MVP 로드맵과 산출물을 설계",
            task_type="proposal_development",
        ),
    ]

    aggregated = []
    for idx, task in enumerate(tasks, start=1):
        task_profile = await task_analyzer.analyze_task(task)
        agent_configs = agent_matcher.select_agents(task_profile, {})

        agent_results = []
        for config in agent_configs:
            context = await context_manager.get_relevant(
                agent_id=config.agent_id,
                task_type=task.task_type,
                max_tokens=4000,
            )

            agent_task = AgentTask(
                task_id=f"task_{idx}_{config.agent_id}",
                task_type=TaskType.DOMAIN_VALIDATION if hasattr(TaskType, "DOMAIN_VALIDATION") else TaskType.HYPOTHESIS_GENERATION,
                description=task.description,
            )

            result = await config.agent.process(agent_task, context)
            agent_results.append(result)

            if result.confidence > 0.7:
                await context_manager.store_insight(
                    Insight(
                        content=json.dumps(result.output, ensure_ascii=False) if isinstance(result.output, dict) else str(result.output),
                        type="agent_result",
                        domains=config.agent.domains,
                        score=result.confidence,
                    ),
                    config.agent_id,
                    agent_task.task_id,
                    {},
                )

        aggregated.append(
            {
                "task": task.description,
                "status": "success" if agent_results else "no_agent",
                "quality_score": sum(r.confidence for r in agent_results) / len(agent_results) if agent_results else 0.0,
                "agents": [
                    {
                        "agent_id": r.agent_id,
                        "confidence": r.confidence,
                        "output": r.output,
                    }
                    for r in agent_results
                ],
                "profile": {
                    "domains": task_profile.domains,
                    "required_expertise": task_profile.required_expertise,
                },
            }
        )

    output_json = output_dir / "multiagent_outputs.json"
    output_json.write_text(json.dumps({"runs": aggregated}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# Multi-Agent Summary",
        f"- 총 작업 수: {len(aggregated)}",
    ]
    for entry in aggregated:
        summary_lines.append(f"\n## 작업: {entry['task']}")
        summary_lines.append(f"- 상태: {entry['status']} (quality={entry['quality_score']:.2f})")
        for agent in entry["agents"]:
            summary_lines.append(f"  - {agent['agent_id']}: 신뢰도 {agent['confidence']:.2f}")
    (output_dir / "multiagent_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved multi-agent outputs to {output_json}")


if __name__ == "__main__":
    asyncio.run(main())

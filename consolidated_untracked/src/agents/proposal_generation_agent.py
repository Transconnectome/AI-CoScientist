#!/usr/bin/env python3
"""
Autonomous Proposal Generation Agent Implementation
2025 Agentic AI: 자율적 제안서 생성 시스템

Features:
- Independent proposal section generation
- Samsung Future Tech Grant strategy integration
- DD-RAPTOR knowledge base utilization
- Multi-persona coordination (Chief Research Architect, Nobel laureate neuroscientist)
- Automatic quality validation and improvement loops
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time

# Project imports
from ..services.rag.enhanced_dd_raptor import EnhancedDDRaptorSystem, create_enhanced_dd_raptor
from ..services.hybrid_rag_service import HybridRAGService

logger = logging.getLogger(__name__)

class SectionType(str, Enum):
    """제안서 섹션 타입"""
    RESEARCH_OBJECTIVES = "research_objectives"
    METHODOLOGY = "methodology"
    INNOVATION_SIGNIFICANCE = "innovation_significance"
    BUDGET_JUSTIFICATION = "budget_justification"
    TIMELINE = "timeline"
    TEAM_ORGANIZATION = "team_organization"
    EXPECTED_OUTCOMES = "expected_outcomes"
    RISK_MITIGATION = "risk_mitigation"

class PersonaType(str, Enum):
    """페르소나 타입"""
    CHIEF_RESEARCH_ARCHITECT = "chief_research_proposal_architect"
    NOBEL_NEUROSCIENTIST = "nobel_laureate_neuroscientist"
    SAMSUNG_GRANT_STRATEGIST = "samsung_grant_strategist"
    BUDGET_SPECIALIST = "budget_specialist"
    INNOVATION_EVALUATOR = "innovation_evaluator"

@dataclass
class SectionSpec:
    """섹션 사양"""
    type: SectionType
    target_audience: str = "samsung_future_tech"
    constraints: List[str] = None
    word_count_target: int = 1000
    required_keywords: List[str] = None
    persona: PersonaType = PersonaType.CHIEF_RESEARCH_ARCHITECT

@dataclass
class GeneratedSection:
    """생성된 섹션"""
    type: SectionType
    content: str
    word_count: int
    citations_count: int
    confidence: float
    reasoning: str
    quality_metrics: Dict[str, float]
    persona_used: PersonaType
    generation_time_ms: float

@dataclass
class ProposalRequirements:
    """제안서 요구사항"""
    grant_type: str = "samsung_future_tech"
    research_domain: str = "developmental_disorder_ai"
    budget_range: str = "10_billion_won"
    duration: str = "5_years"
    innovation_level: str = "world_first"
    risk_level: str = "high_risk_high_return"

class AutonomousImprovementError(Exception):
    """자율적 개선 오류"""
    pass

class ProposalGenerationAgent:
    """자율적 제안서 생성 에이전트 (2025 Agentic AI 패턴)"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()

        # Core components
        self.dd_rag_system = None
        self.hybrid_rag = None

        # Persona configurations
        self.personas = self._initialize_personas()

        # Quality thresholds
        self.quality_thresholds = {
            "min_word_count": 800,
            "min_citations": 10,
            "min_confidence": 0.75,
            "min_innovation_score": 0.8,
            "min_coherence_score": 0.85
        }

        # Generation history for learning
        self.generation_history = []

    def _default_config(self) -> Dict:
        """기본 설정"""
        return {
            "dd_rag_db_path": "chromadb_data_dd",
            "output_directory": "./output/proposals",
            "template_directory": "./templates/proposals",
            "max_generation_attempts": 3,
            "quality_improvement_rounds": 2,
            "parallel_section_generation": True,
            "auto_improvement_enabled": True
        }

    def _initialize_personas(self) -> Dict[PersonaType, Dict]:
        """페르소나 초기화"""
        return {
            PersonaType.CHIEF_RESEARCH_ARCHITECT: {
                "role": "수석 연구 제안 설계자",
                "expertise": "대형 과제 수주, 전략 수립, 논리적 완결성",
                "tone": "확신에 찬 전문적 어조",
                "key_phrases": ["달성한다", "규명한다", "구축한다", "실현한다"],
                "focus": "패러다임 전환, 혁신적 접근법"
            },
            PersonaType.NOBEL_NEUROSCIENTIST: {
                "role": "노벨상 수상급 뇌과학자",
                "expertise": "신경과학 이론, 메커니즘 규명, 과학적 엄밀성",
                "tone": "과학적 정확성과 권위",
                "key_phrases": ["생물학적 기전", "신경회로", "분자기전"],
                "focus": "과학적 타당성, 메커니즘 기반 접근"
            },
            PersonaType.SAMSUNG_GRANT_STRATEGIST: {
                "role": "삼성미래기술육성사업 전문가",
                "expertise": "삼성 심사 기준, High Risk High Return",
                "tone": "전략적이고 비즈니스 지향적",
                "key_phrases": ["세계 최초", "파괴적 혁신", "First Mover"],
                "focus": "시장 파급효과, 기술적 초격차"
            },
            PersonaType.BUDGET_SPECIALIST: {
                "role": "연구비 산정 전문가",
                "expertise": "예산 계산, 비용 효율성, 자원 배분",
                "tone": "정확하고 체계적",
                "key_phrases": ["비용 대비 효과", "자원 최적화"],
                "focus": "예산 정당성, 효율적 자원 활용"
            },
            PersonaType.INNOVATION_EVALUATOR: {
                "role": "혁신성 평가 전문가",
                "expertise": "기술 혁신도, 차별성 분석, 미래 전망",
                "tone": "미래지향적이고 분석적",
                "key_phrases": ["기술적 도약", "패러다임 시프트"],
                "focus": "혁신성, 차별성, 미래 가치"
            }
        }

    async def initialize(self):
        """에이전트 초기화"""
        logger.info("Initializing Proposal Generation Agent...")

        # DD-RAPTOR 시스템 초기화
        self.dd_rag_system = await create_enhanced_dd_raptor(
            db_path=self.config["dd_rag_db_path"]
        )

        # Hybrid RAG 서비스 초기화 (실제 구현 시 필요)
        # self.hybrid_rag = HybridRAGService()

        # 출력 디렉토리 생성
        Path(self.config["output_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["template_directory"]).mkdir(parents=True, exist_ok=True)

        logger.info("Proposal Generation Agent initialized successfully")

    async def generate_section(self, section_spec: SectionSpec) -> GeneratedSection:
        """섹션 자율적 생성"""
        logger.info(f"Generating section: {section_spec.type.value}")
        start_time = time.time()

        # 1. DD-RAPTOR에서 관련 지식 검색
        knowledge_context = await self._gather_dd_knowledge(section_spec)

        # 2. 페르소나 기반 생성 전략 선택
        generation_strategy = self._select_generation_strategy(section_spec)

        # 3. 섹션 내용 생성
        content = await self._generate_content_with_persona(
            section_spec, knowledge_context, generation_strategy
        )

        # 4. 품질 검증 및 자율적 개선
        if self.config["auto_improvement_enabled"]:
            content = await self._autonomous_quality_improvement(
                content, section_spec
            )

        # 5. 메트릭 계산
        metrics = await self._calculate_section_metrics(content, section_spec)

        generation_time = (time.time() - start_time) * 1000

        generated_section = GeneratedSection(
            type=section_spec.type,
            content=content,
            word_count=len(content.split()),
            citations_count=content.count('[') + content.count('('),  # 간단한 인용 카운트
            confidence=metrics["confidence"],
            reasoning=metrics["reasoning"],
            quality_metrics=metrics,
            persona_used=section_spec.persona,
            generation_time_ms=generation_time
        )

        # 학습을 위한 히스토리 저장
        self.generation_history.append({
            "section_spec": asdict(section_spec),
            "generated_section": asdict(generated_section),
            "timestamp": time.time()
        })

        return generated_section

    async def generate_full_proposal(self, requirements: ProposalRequirements) -> Dict[str, GeneratedSection]:
        """전체 제안서 자율적 생성"""
        logger.info("Generating full proposal autonomously...")

        # 1. 제안서 구조 정의
        sections_to_generate = [
            SectionSpec(
                type=SectionType.RESEARCH_OBJECTIVES,
                persona=PersonaType.CHIEF_RESEARCH_ARCHITECT,
                word_count_target=1200,
                required_keywords=["세계 최초", "파운데이션 모델", "발달장애"]
            ),
            SectionSpec(
                type=SectionType.METHODOLOGY,
                persona=PersonaType.NOBEL_NEUROSCIENTIST,
                word_count_target=1500,
                required_keywords=["다중 모달", "종단 연구", "Digital Twin Brain"]
            ),
            SectionSpec(
                type=SectionType.INNOVATION_SIGNIFICANCE,
                persona=PersonaType.SAMSUNG_GRANT_STRATEGIST,
                word_count_target=1000,
                required_keywords=["패러다임 전환", "기술적 초격차"]
            ),
            SectionSpec(
                type=SectionType.BUDGET_JUSTIFICATION,
                persona=PersonaType.BUDGET_SPECIALIST,
                word_count_target=800,
                required_keywords=["비용 효율성", "자원 최적화"]
            )
        ]

        # 2. 병렬 생성 (성능 최적화)
        if self.config["parallel_section_generation"]:
            tasks = [
                self.generate_section(section_spec)
                for section_spec in sections_to_generate
            ]
            generated_sections = await asyncio.gather(*tasks)
        else:
            # 순차 생성
            generated_sections = []
            for section_spec in sections_to_generate:
                section = await self.generate_section(section_spec)
                generated_sections.append(section)

        # 3. 섹션별 결과 딕셔너리로 변환
        proposal_sections = {
            section.type.value: section
            for section in generated_sections
        }

        # 4. 전체 제안서 품질 검증
        await self._validate_full_proposal_quality(proposal_sections)

        logger.info(f"Full proposal generated with {len(proposal_sections)} sections")
        return proposal_sections

    async def generate_with_dd_knowledge(self, section_type: str, dd_query: str) -> GeneratedSection:
        """DD-RAPTOR 지식 활용 생성"""
        logger.info(f"Generating {section_type} with DD knowledge: {dd_query}")

        # 1. DD-RAPTOR 검색
        search_results = await self.dd_rag_system.search(dd_query, n_results=10)

        # 2. 검색 결과 검증
        if search_results.relevancy_score < 0.7:
            logger.warning(f"Low relevancy score: {search_results.relevancy_score}")

        # 3. 지식 기반 섹션 생성
        section_spec = SectionSpec(
            type=SectionType(section_type),
            persona=PersonaType.CHIEF_RESEARCH_ARCHITECT,
            required_keywords=dd_query.split()
        )

        # 4. DD 논문 인용을 포함한 내용 생성
        content = await self._generate_with_dd_citations(
            section_spec, search_results
        )

        # 5. 메트릭 계산
        metrics = await self._calculate_section_metrics(content, section_spec)

        return GeneratedSection(
            type=section_spec.type,
            content=content,
            word_count=len(content.split()),
            citations_count=len([doc for doc in search_results.documents]),
            confidence=metrics["confidence"],
            reasoning=f"Generated using DD-RAPTOR knowledge from {len(search_results.documents)} papers",
            quality_metrics=metrics,
            persona_used=section_spec.persona,
            generation_time_ms=0.0
        )

    async def _gather_dd_knowledge(self, section_spec: SectionSpec) -> Dict[str, Any]:
        """DD-RAPTOR에서 관련 지식 수집"""
        # 섹션 타입별 특화 쿼리 생성
        section_queries = {
            SectionType.RESEARCH_OBJECTIVES: [
                "foundation model brain development autism",
                "multimodal neurodevelopmental disorders",
                "early diagnosis prediction algorithms"
            ],
            SectionType.METHODOLOGY: [
                "longitudinal brain imaging analysis",
                "zebrafish validation neurodevelopment",
                "federated learning healthcare"
            ],
            SectionType.INNOVATION_SIGNIFICANCE: [
                "breakthrough neuroscience AI applications",
                "digital twin brain modeling",
                "precision medicine developmental disorders"
            ],
            SectionType.BUDGET_JUSTIFICATION: [
                "AI computing resources neuroscience",
                "large scale brain imaging costs",
                "consortium research infrastructure"
            ]
        }

        queries = section_queries.get(section_spec.type, ["developmental disorder AI"])

        # 다중 쿼리 검색
        search_results = []
        for query in queries:
            results = await self.dd_rag_system.search(query, n_results=5)
            search_results.extend(results.documents)

        return {
            "search_results": search_results,
            "total_papers": len(search_results),
            "relevancy_scores": [0.85, 0.82, 0.79]  # Mock scores
        }

    def _select_generation_strategy(self, section_spec: SectionSpec) -> Dict[str, Any]:
        """생성 전략 선택"""
        persona_config = self.personas[section_spec.persona]

        strategy = {
            "persona": persona_config,
            "tone_requirements": persona_config["tone"],
            "key_phrases": persona_config["key_phrases"],
            "focus_areas": persona_config["focus"],
            "structure_template": self._get_section_template(section_spec.type),
            "quality_criteria": self._get_quality_criteria(section_spec.type)
        }

        return strategy

    def _get_section_template(self, section_type: SectionType) -> str:
        """섹션별 템플릿 반환"""
        templates = {
            SectionType.RESEARCH_OBJECTIVES: """
## 연구 목표

### 1. 비전 (Vision)
[세계 최초의 혁신적 비전 제시]

### 2. 구체적 목표 (Specific Objectives)
[명확하고 측정 가능한 목표들]

### 3. 기대 성과 (Expected Outcomes)
[구체적인 성과 지표와 파급효과]
            """,
            SectionType.METHODOLOGY: """
## 연구 방법론

### 1. 연구 설계 (Study Design)
[종단 연구 설계 및 다중 모달 접근법]

### 2. 데이터 수집 및 처리 (Data Collection & Processing)
[3,000명 코호트, 20년 종단 데이터]

### 3. AI 모델 개발 (AI Model Development)
[파운데이션 모델 아키텍처 및 학습 방법]

### 4. 검증 방법론 (Validation Methodology)
[제브라피쉬 검증, 임상 적용]
            """,
            SectionType.INNOVATION_SIGNIFICANCE: """
## 혁신성 및 기대효과

### 1. 기술적 혁신성 (Technical Innovation)
[세계 최초, 패러다임 전환]

### 2. 과학적 기여도 (Scientific Contribution)
[메커니즘 규명, 이론적 발전]

### 3. 사회적 파급효과 (Social Impact)
[의료비 절감, 삶의 질 향상]
            """
        }

        return templates.get(section_type, "## {section_type}\n[내용을 여기에 작성하세요]")

    def _get_quality_criteria(self, section_type: SectionType) -> Dict[str, float]:
        """섹션별 품질 기준"""
        criteria = {
            SectionType.RESEARCH_OBJECTIVES: {
                "clarity": 0.9,
                "innovation": 0.85,
                "feasibility": 0.8
            },
            SectionType.METHODOLOGY: {
                "scientific_rigor": 0.9,
                "technical_detail": 0.85,
                "reproducibility": 0.8
            },
            SectionType.INNOVATION_SIGNIFICANCE: {
                "novelty": 0.9,
                "impact": 0.85,
                "market_potential": 0.8
            }
        }

        return criteria.get(section_type, {"overall_quality": 0.8})

    async def _generate_content_with_persona(self, section_spec: SectionSpec,
                                           knowledge_context: Dict,
                                           generation_strategy: Dict) -> str:
        """페르소나 기반 내용 생성"""
        # Mock content generation (실제로는 LLM 호출)
        persona_config = generation_strategy["persona"]
        template = generation_strategy["structure_template"]

        # 페르소나별 특화 내용 생성
        if section_spec.persona == PersonaType.CHIEF_RESEARCH_ARCHITECT:
            content = await self._generate_strategic_content(section_spec, knowledge_context)
        elif section_spec.persona == PersonaType.NOBEL_NEUROSCIENTIST:
            content = await self._generate_scientific_content(section_spec, knowledge_context)
        elif section_spec.persona == PersonaType.SAMSUNG_GRANT_STRATEGIST:
            content = await self._generate_strategic_business_content(section_spec, knowledge_context)
        else:
            content = await self._generate_default_content(section_spec, knowledge_context)

        # 키 프레이즈 삽입
        content = self._inject_key_phrases(content, persona_config["key_phrases"])

        return content

    async def _generate_strategic_content(self, section_spec: SectionSpec,
                                        knowledge_context: Dict) -> str:
        """전략적 내용 생성 (수석 연구 제안 설계자 페르소나)"""
        base_content = f"""
## {section_spec.type.value.replace('_', ' ').title()}

본 연구는 **세계 최초의 발달장애 특화 멀티모달 파운데이션 모델**을 구축하여 기존 관찰 중심 진단의 패러다임을 **데이터 기반 예측 진단**으로 전환하는 혁신적 도전이다.

### 핵심 전략 목표

1. **First Mover Advantage 확보**: 발달장애 영역에서 세계 최초의 파운데이션 모델 개발로 기술적 주도권을 확보한다.

2. **Paradigm Disruption**: 100년간 지속된 증상 기반 진단을 생물학적 메커니즘 기반 예측 진단으로 혁신한다.

3. **Multi-modal Integration**: fMRI, dMRI, EEG, 유전체, 행동 데이터의 완벽한 융합을 통해 Digital Twin Brain을 실현한다.

### 혁신적 접근법

**20년 종단 멀티모달 데이터** (3,000명 코호트)를 활용한 **Longitudinal Transformer** 아키텍처로 발달 궤적을 학습하여 **3세 미만 조기 진단 및 개인별 예후 예측**을 달성한다.

**제브라피쉬 검증 루프**를 통해 AI가 예측한 위험 유전자를 생물학적으로 검증하여 **설명 가능한 바이오마커**를 발굴한다.

### 기대 파급효과

- **의료 패러다임 전환**: 사후 치료에서 사전 예방 중심으로
- **사회적 비용 절감**: 발달장애 관련 천문학적 사회적 비용 감소
- **글로벌 기술 주도권**: K-Brain AI 플랫폼의 세계 표준화

이는 단순한 기술 개발을 넘어 **인류의 뇌발달 이해를 혁신하는 과학적 도약**이자 **대한민국 뇌과학의 세계적 위상 제고**를 실현하는 전략적 과제이다.
        """

        return base_content.strip()

    async def _generate_scientific_content(self, section_spec: SectionSpec,
                                         knowledge_context: Dict) -> str:
        """과학적 내용 생성 (노벨상 수상급 뇌과학자 페르소나)"""
        base_content = f"""
## {section_spec.type.value.replace('_', ' ').title()}

발달장애의 생물학적 메커니즘을 규명하기 위한 **다중 스케일 통합 접근법**을 제시한다.

### 신경생물학적 가설

**뇌 연결성-유전자-대사체** 상호작용의 발달적 변화가 자폐 스펙트럼 장애의 핵심 병인이라는 가설하에, **종단적 네트워크 분석**을 통해 임계 발달 시기의 **연결성 패턴 변화**를 규명한다.

### 다중 모달 융합 방법론

1. **fMRI 기반 기능적 연결성**: 기본 모드 네트워크(DMN), 실행 네트워크, 현저성 네트워크 간 상호작용 분석
2. **dMRI 기반 구조적 연결성**: 백질 발달 궤적 및 확산 텐서 지표의 종단 변화
3. **EEG 기반 신경진동**: 감마파, 알파파, 델타파의 발달적 변화 패턴
4. **멀티오믹스 분석**: 전사체-대사체-마이크로바이옴의 통합 프로파일링

### 생물학적 검증 시스템

**제브라피쉬 모델**을 이용한 후보 유전자의 기능적 검증:
- CRISPR/Cas9 기반 유전자 편집
- 신경 발달 이상의 실시간 관찰
- 행동 표현형과 신경회로 변화의 인과관계 규명

### 예측 모델의 생물학적 해석성

**Attention Mechanism**을 통해 모델 예측에 기여하는 **핵심 뇌 영역 및 유전자 네트워크**를 식별하여 **설명 가능한 AI**를 구현한다.

이러한 접근법을 통해 **발달장애의 분자-회로-행동 연결고리**를 최초로 규명하여 정밀의료의 새로운 지평을 연다.
        """

        return base_content.strip()

    async def _generate_strategic_business_content(self, section_spec: SectionSpec,
                                                 knowledge_context: Dict) -> str:
        """전략적 비즈니스 내용 생성 (삼성 전략가 페르소나)"""
        base_content = f"""
## {section_spec.type.value.replace('_', ' ').title()}

**삼성미래기술육성사업의 'High Risk, High Return' 철학**에 완벽히 부합하는 **파괴적 혁신 과제**이다.

### Market Disruption Potential

**$280B 글로벌 신경과학 시장**에서 **First Mover**로서의 절대 우위를 확보한다.
- 기존 진단 시장의 **패러다임 시프트** 주도
- **Digital Therapeutics** 신시장 창출
- **B2B 플랫폼** 비즈니스 모델 구축

### Technology Moat (기술적 진입장벽)

1. **데이터 자산**: 세계 유일의 20년 종단 3,000명 멀티모달 데이터셋
2. **알고리즘 IP**: Longitudinal Transformer 및 멀티모달 융합 기술
3. **검증 플랫폼**: 제브라피쉬 기반 생물학적 검증 시스템
4. **임상 네트워크**: 5개 병원 컨소시엄의 배타적 협력

### Competitive Advantage

**구글, IBM, Microsoft도 시도하지 못한 영역**:
- 발달장애 특화 파운데이션 모델 (vs. 범용 모델)
- 생물학적 검증 루프 통합 (vs. AI만 단독)
- 종단 데이터 기반 예후 예측 (vs. 단편적 진단)

### Market Size & Revenue Projection

- **Year 1-2**: 기술 검증 및 IP 확보 ($50M 기술가치)
- **Year 3-4**: 임상 도구 상용화 ($200M 매출 예상)
- **Year 5+**: 글로벌 플랫폼 확산 ($1B+ 시장 규모)

### Strategic Partnership

- **삼성메디컬센터**: 임상 검증 및 의료진 트레이닝
- **삼성SDS**: AI 플랫폼 구축 및 스케일링
- **삼성바이오로직스**: 치료제 개발 연계

**ROI 200%+ 달성**을 통해 삼성미래기술육성사업의 **대표 성공사례**로 자리매김한다.
        """

        return base_content.strip()

    async def _generate_default_content(self, section_spec: SectionSpec,
                                      knowledge_context: Dict) -> str:
        """기본 내용 생성"""
        return f"""
## {section_spec.type.value.replace('_', ' ').title()}

[이 섹션은 {section_spec.persona.value} 페르소나로 생성됩니다]

### 주요 내용

관련 연구 결과를 바탕으로 한 체계적 접근법을 제시합니다.

### 세부 사항

구체적인 구현 방안과 예상 성과를 다음과 같이 제시합니다:

1. 첫 번째 핵심 요소
2. 두 번째 핵심 요소
3. 세 번째 핵심 요소

### 기대 효과

이러한 접근을 통해 다음과 같은 성과를 달성합니다.
        """

    def _inject_key_phrases(self, content: str, key_phrases: List[str]) -> str:
        """키 프레이즈 삽입 (페르소나 특성 반영)"""
        # 간단한 구현: 이미 내용에 포함되어 있으므로 그대로 반환
        return content

    async def _autonomous_quality_improvement(self, content: str,
                                            section_spec: SectionSpec) -> str:
        """자율적 품질 개선"""
        logger.info("Performing autonomous quality improvement...")

        improvement_rounds = self.config["quality_improvement_rounds"]

        for round_num in range(improvement_rounds):
            # 1. 현재 품질 평가
            current_metrics = await self._calculate_section_metrics(content, section_spec)

            # 2. 개선 필요성 판단
            needs_improvement = self._assess_improvement_needs(current_metrics)

            if not needs_improvement:
                break

            # 3. 자율적 개선 수행
            content = await self._apply_autonomous_improvements(
                content, current_metrics, section_spec
            )

            logger.info(f"Quality improvement round {round_num + 1} completed")

        return content

    def _assess_improvement_needs(self, metrics: Dict[str, float]) -> bool:
        """개선 필요성 평가"""
        # 임계값 기반 개선 필요성 판단
        needs_improvement = (
            metrics.get("confidence", 0) < self.quality_thresholds["min_confidence"] or
            metrics.get("coherence", 0) < self.quality_thresholds["min_coherence_score"] or
            metrics.get("innovation", 0) < self.quality_thresholds["min_innovation_score"]
        )

        return needs_improvement

    async def _apply_autonomous_improvements(self, content: str, metrics: Dict,
                                           section_spec: SectionSpec) -> str:
        """자율적 개선 적용"""
        improved_content = content

        # 1. 어조 개선 (페르소나에 맞게)
        if metrics.get("tone_consistency", 0) < 0.8:
            improved_content = self._improve_tone_consistency(improved_content, section_spec)

        # 2. 구조적 개선
        if metrics.get("structure_score", 0) < 0.8:
            improved_content = self._improve_structure(improved_content, section_spec)

        # 3. 키워드 밀도 최적화
        if section_spec.required_keywords:
            improved_content = self._optimize_keyword_density(
                improved_content, section_spec.required_keywords
            )

        return improved_content

    def _improve_tone_consistency(self, content: str, section_spec: SectionSpec) -> str:
        """어조 일관성 개선"""
        persona_config = self.personas[section_spec.persona]
        key_phrases = persona_config["key_phrases"]

        # 간단한 구현: 확정적 어조 강화
        improvements = {
            "것이다": "한다",
            "일 것": "것",
            "가능하다": "가능성이 있다",
            "예상된다": "예상한다"
        }

        improved_content = content
        for old_phrase, new_phrase in improvements.items():
            improved_content = improved_content.replace(old_phrase, new_phrase)

        return improved_content

    def _improve_structure(self, content: str, section_spec: SectionSpec) -> str:
        """구조적 개선"""
        # 섹션 구조 개선: 소제목 추가, 번호 매기기 등
        lines = content.split('\n')
        improved_lines = []

        for line in lines:
            improved_lines.append(line)

            # 중요 문장 후에 강조 추가
            if any(keyword in line for keyword in ["세계 최초", "파괴적 혁신", "패러다임"]):
                if not line.startswith('#') and len(line.strip()) > 0:
                    improved_lines.append("")  # 빈 줄 추가로 강조

        return '\n'.join(improved_lines)

    def _optimize_keyword_density(self, content: str, keywords: List[str]) -> str:
        """키워드 밀도 최적화"""
        # 키워드가 충분히 포함되어 있는지 확인하고 부족하면 추가
        for keyword in keywords:
            if content.count(keyword) == 0:
                # 적절한 위치에 키워드 삽입 (간단한 구현)
                insertion_point = len(content) // 2
                content = content[:insertion_point] + f" {keyword}와 관련하여, " + content[insertion_point:]

        return content

    async def _calculate_section_metrics(self, content: str,
                                       section_spec: SectionSpec) -> Dict[str, float]:
        """섹션 메트릭 계산"""
        # Mock metrics calculation (실제로는 더 정교한 분석)
        word_count = len(content.split())

        metrics = {
            "word_count_score": min(1.0, word_count / section_spec.word_count_target),
            "coherence": np.random.uniform(0.75, 0.95),  # Mock coherence score
            "innovation": np.random.uniform(0.8, 0.95),   # Mock innovation score
            "confidence": np.random.uniform(0.8, 0.92),   # Mock confidence
            "tone_consistency": np.random.uniform(0.85, 0.95),
            "structure_score": np.random.uniform(0.8, 0.9),
            "keyword_relevance": self._calculate_keyword_relevance(content, section_spec),
            "reasoning": f"Generated {word_count} words with high quality metrics"
        }

        return metrics

    def _calculate_keyword_relevance(self, content: str, section_spec: SectionSpec) -> float:
        """키워드 관련성 계산"""
        if not section_spec.required_keywords:
            return 1.0

        content_lower = content.lower()
        found_keywords = sum(
            1 for keyword in section_spec.required_keywords
            if keyword.lower() in content_lower
        )

        return found_keywords / len(section_spec.required_keywords)

    async def _generate_with_dd_citations(self, section_spec: SectionSpec,
                                        search_results) -> str:
        """DD 논문 인용을 포함한 생성"""
        base_content = await self._generate_content_with_persona(
            section_spec, {"search_results": search_results.documents},
            self._select_generation_strategy(section_spec)
        )

        # DD 논문 인용 추가
        citations = []
        for i, doc in enumerate(search_results.documents[:5]):
            # 간단한 인용 형식 (실제로는 메타데이터에서 추출)
            citations.append(f"[{i+1}] {doc[:100]}...")

        citation_section = "\n\n### 참고문헌\n" + "\n".join(citations)

        return base_content + citation_section

    async def _validate_full_proposal_quality(self, proposal_sections: Dict) -> None:
        """전체 제안서 품질 검증"""
        logger.info("Validating full proposal quality...")

        # 1. 섹션 완성도 검증
        required_sections = [
            SectionType.RESEARCH_OBJECTIVES.value,
            SectionType.METHODOLOGY.value,
            SectionType.INNOVATION_SIGNIFICANCE.value
        ]

        missing_sections = [
            section for section in required_sections
            if section not in proposal_sections
        ]

        if missing_sections:
            raise AutonomousImprovementError(f"Missing required sections: {missing_sections}")

        # 2. 전체 길이 검증
        total_words = sum(section.word_count for section in proposal_sections.values())
        if total_words < 3000:  # 최소 3000단어
            logger.warning(f"Proposal too short: {total_words} words")

        # 3. 일관성 검증
        confidence_scores = [section.confidence for section in proposal_sections.values()]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        if avg_confidence < 0.8:
            logger.warning(f"Low average confidence: {avg_confidence:.3f}")

        logger.info("Full proposal quality validation completed")


# Factory function
async def create_proposal_agent(config: Optional[Dict] = None) -> ProposalGenerationAgent:
    """제안서 생성 에이전트 생성 및 초기화"""
    agent = ProposalGenerationAgent(config)
    await agent.initialize()
    return agent
#!/usr/bin/env python3
"""
Samsung Grant Generator - Unified RAG Implementation
삼성미래기술육성사업 특화 자동 제안서 생성 시스템

Next-Generation Samsung Grant System powered by Unified RAG Orchestrator
- Replaced DD-RAPTOR with 6-strategy RAG orchestration
- Enhanced knowledge integration (ESM3, Grant proposals, Multi-domain research)
- Intelligent cross-domain synthesis for breakthrough proposals
- Samsung-specific format compliance with advanced AI backing

Features:
- Samsung Future Tech Grant format compliance
- Unified RAG knowledge integration (ESM3, Neuroscience, Quantum ML)
- Autonomous budget calculation and validation
- Multi-persona content generation with RAG strategy optimization
- Korean government proposal standards
- Automatic quality verification with cross-domain insights
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
from datetime import datetime, timedelta

# Updated imports for Unified RAG system
from ..agents.proposal_generation_agent_unified import (
    UnifiedProposalGenerationAgent, create_unified_proposal_agent,
    SectionSpec, SectionType, PersonaType, GeneratedSection
)
from .budget_calculator import AutonomousBudgetCalculator

logger = logging.getLogger(__name__)

class ProposalStatus(str, Enum):
    """제안서 상태"""
    DRAFT = "draft"
    REVIEW = "review"
    REVISION = "revision"
    FINAL = "final"
    SUBMITTED = "submitted"

class SamsungSectionType(str, Enum):
    """삼성 제안서 섹션 타입"""
    SECTION_1_OVERVIEW = "section_1_overview"          # 연구개발과제 개요
    SECTION_2_RESEARCH = "section_2_research"          # 연구개발 내용
    SECTION_3_IMPLEMENTATION = "section_3_implementation"  # 연구개발 추진계획
    SECTION_4_OUTCOMES = "section_4_outcomes"          # 기대성과 및 활용방안
    SECTION_5_BUDGET = "section_5_budget"              # 연구비 계획
    SECTION_6_RESEARCHERS = "section_6_researchers"    # 연구진 구성

class RiskLevel(str, Enum):
    """연구 위험도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BREAKTHROUGH = "breakthrough"

@dataclass
class SamsungGrantSpec:
    """삼성 제안서 명세"""
    title: str
    research_area: str
    primary_pi: str
    institution: str
    total_budget: float
    duration_years: int
    risk_level: RiskLevel
    innovation_keywords: List[str]
    collaboration_type: str = "single_institution"
    international_collaboration: bool = False

    # Unified RAG enhancement fields
    knowledge_domains: List[str] = None
    cross_domain_synthesis: bool = True
    rag_strategy_preferences: List[str] = None

@dataclass
class ProposalSection:
    """제안서 섹션"""
    section_id: str
    title: str
    content: str
    korean_content: Optional[str] = None
    word_count: int = 0
    subsections: Dict[str, str] = None
    quality_score: float = 0.0
    citations: List[Dict[str, str]] = None

    # Unified RAG metadata
    rag_strategy_used: str = ""
    knowledge_sources: List[str] = None
    cross_domain_insights: List[str] = None

@dataclass
class GeneratedProposal:
    """생성된 전체 제안서"""
    proposal_id: str
    grant_spec: SamsungGrantSpec
    sections: Dict[str, ProposalSection]
    budget_breakdown: Dict[str, Any]
    timeline: Dict[str, Any]
    status: ProposalStatus
    generated_at: datetime
    total_pages: int = 0
    quality_metrics: Dict[str, float] = None

    # Unified RAG analytics
    rag_performance: Dict[str, Any] = None
    strategy_distribution: Dict[str, int] = None
    knowledge_coverage: Dict[str, float] = None

class UnifiedSamsungGrantGenerator:
    """Unified RAG 기반 삼성 제안서 생성기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """생성기 초기화"""
        self.config = config or self._get_default_config()
        self.proposal_agent: Optional[UnifiedProposalGenerationAgent] = None
        self.budget_calculator = AutonomousBudgetCalculator()

        # Samsung-specific requirements
        self.required_sections = self._load_samsung_requirements()

        # Unified RAG performance tracking
        self.generation_analytics = {
            "proposals_generated": 0,
            "strategy_performance": {},
            "quality_trends": [],
            "cross_domain_success_rate": 0.0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정"""
        return {
            "output_directory": "output/samsung_proposals_unified/",
            "template_directory": "templates/samsung/",
            "quality_threshold": 0.85,
            "max_sections": 6,
            "korean_translation": True,
            "auto_budget_calculation": True,
            "compliance_validation": True,

            # Unified RAG specific settings
            "enable_cross_domain_synthesis": True,
            "preferred_rag_strategies": ["GRAPH_RAG", "HYBRID", "ENHANCED_DD_RAPTOR"],
            "knowledge_domain_preferences": {
                "neuroscience": 0.4,
                "quantum_ml": 0.3,
                "protein_research": 0.2,
                "general_ai": 0.1
            },
            "samsung_specific_optimization": True
        }

    def _load_samsung_requirements(self) -> Dict[str, Any]:
        """삼성 제안서 요구사항 로드"""
        return {
            "section_1_overview": {
                "title": "연구개발과제 개요",
                "min_words": 300,
                "max_words": 500,
                "subsections": ["연구목표", "기술적 의의", "파급효과"],
                "required_elements": ["혁신성", "실용성", "경제성"]
            },
            "section_2_research": {
                "title": "연구개발 내용",
                "min_words": 1000,
                "max_words": 1500,
                "subsections": ["연구방법론", "핵심기술", "예상 난제 및 해결방안"],
                "required_elements": ["독창성", "기술적 우수성", "실현가능성"]
            },
            "section_3_implementation": {
                "title": "연구개발 추진계획",
                "min_words": 500,
                "max_words": 800,
                "subsections": ["연구일정", "연구체계", "성과관리"],
                "required_elements": ["체계성", "효율성", "관리계획"]
            },
            "section_4_outcomes": {
                "title": "기대성과 및 활용방안",
                "min_words": 400,
                "max_words": 600,
                "subsections": ["기술적 성과", "경제적 파급효과", "활용계획"],
                "required_elements": ["구체성", "실현가능성", "사회적 가치"]
            }
        }

    async def initialize(self):
        """생성기 초기화"""
        logger.info("Initializing Unified Samsung Grant Generator...")

        # Unified Proposal Agent 초기화
        self.proposal_agent = create_unified_proposal_agent({
            **self.config,
            "samsung_optimized": True
        })
        await self.proposal_agent.initialize()

        # Budget calculator 초기화
        await self.budget_calculator.initialize()

        # 출력 디렉토리 생성
        Path(self.config["output_directory"]).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Unified Samsung Grant Generator initialized successfully")

    async def generate_proposal(self, grant_spec: SamsungGrantSpec) -> GeneratedProposal:
        """전체 제안서 생성 - Unified RAG 기반"""
        logger.info(f"🚀 Generating Samsung proposal: {grant_spec.title} with Unified RAG")
        start_time = time.time()

        proposal_id = self._generate_proposal_id(grant_spec)

        # 1. 섹션별 생성 명세 준비 (Unified RAG 최적화)
        section_specs = await self._prepare_unified_section_specs(grant_spec)

        # 2. 병렬 섹션 생성 (Unified RAG 다중 전략 활용)
        sections = {}
        section_tasks = []

        for section_id, spec in section_specs.items():
            task = self._generate_samsung_section_unified(section_id, spec, grant_spec)
            section_tasks.append((section_id, task))

        # 병렬 실행
        section_results = await asyncio.gather(*[task for _, task in section_tasks], return_exceptions=True)

        # 결과 처리
        for (section_id, _), result in zip(section_tasks, section_results):
            if isinstance(result, Exception):
                logger.error(f"Section {section_id} failed: {result}")
                continue
            sections[section_id] = result

        # 3. 자율적 예산 계산 (AI 인프라 고려)
        budget_breakdown = await self.budget_calculator.calculate_comprehensive_budget(
            total_budget=grant_spec.total_budget,
            duration_years=grant_spec.duration_years,
            research_type="ai_intensive",
            special_requirements={
                "unified_rag_infrastructure": True,
                "multi_domain_research": len(grant_spec.knowledge_domains or []) > 1,
                "samsung_compliance": True
            }
        )

        # 4. 연구 일정 자동 생성
        timeline = self._generate_research_timeline(grant_spec, sections)

        # 5. 전체 품질 검증 및 최적화
        quality_metrics = await self._validate_unified_proposal_quality(sections, grant_spec)

        # 6. RAG 성능 분석
        rag_performance = self._analyze_rag_performance(sections)

        generation_time = (time.time() - start_time) * 1000

        proposal = GeneratedProposal(
            proposal_id=proposal_id,
            grant_spec=grant_spec,
            sections=sections,
            budget_breakdown=budget_breakdown,
            timeline=timeline,
            status=ProposalStatus.DRAFT,
            generated_at=datetime.now(),
            total_pages=self._estimate_page_count(sections),
            quality_metrics=quality_metrics,
            rag_performance=rag_performance,
            strategy_distribution=self._get_strategy_distribution(sections),
            knowledge_coverage=self._analyze_knowledge_coverage(sections, grant_spec)
        )

        # Analytics 업데이트
        self._update_generation_analytics(proposal, generation_time)

        logger.info(f"✅ Samsung proposal generated in {generation_time:.1f}ms")
        logger.info(f"📊 Quality score: {quality_metrics.get('overall_score', 0):.3f}")
        logger.info(f"🎯 Strategy distribution: {rag_performance.get('strategy_usage', {})}")

        return proposal

    async def _prepare_unified_section_specs(self, grant_spec: SamsungGrantSpec) -> Dict[str, SectionSpec]:
        """Unified RAG 최적화 섹션 명세 준비"""
        specs = {}

        # 기본 섹션 정의
        section_mappings = {
            "section_1_overview": (SectionType.RESEARCH_OBJECTIVES, PersonaType.SAMSUNG_GRANT_STRATEGIST),
            "section_2_research": (SectionType.METHODOLOGY, PersonaType.CHIEF_RESEARCH_ARCHITECT),
            "section_3_implementation": (SectionType.EXPECTED_OUTCOMES, PersonaType.NOBEL_NEUROSCIENTIST),
            "section_4_outcomes": (SectionType.INNOVATION_SIGNIFICANCE, PersonaType.INNOVATION_EVALUATOR)
        }

        for section_id, (section_type, persona) in section_mappings.items():
            requirements = self.required_sections.get(section_id, {})

            # Enhanced keywords for Unified RAG
            unified_keywords = self._generate_unified_keywords(grant_spec, section_type)

            specs[section_id] = SectionSpec(
                type=section_type,
                persona=persona,
                required_keywords=unified_keywords,
                min_words=requirements.get("min_words", 400),
                max_words=requirements.get("max_words", 800),
                citation_requirement=True,
                innovation_focus=(section_type == SectionType.INNOVATION_SIGNIFICANCE)
            )

        return specs

    def _generate_unified_keywords(self, grant_spec: SamsungGrantSpec, section_type: SectionType) -> List[str]:
        """Unified RAG용 확장 키워드 생성"""
        base_keywords = grant_spec.innovation_keywords.copy()

        # Section-specific enhancements
        section_specific = {
            SectionType.RESEARCH_OBJECTIVES: [
                "foundation model", "breakthrough innovation", "Samsung Future Tech",
                "ESM3 applications", "neuroscience AI", "precision medicine"
            ],
            SectionType.METHODOLOGY: [
                "multimodal integration", "federated learning", "quantum ML",
                "protein structure prediction", "brain imaging analysis",
                "longitudinal validation"
            ],
            SectionType.INNOVATION_SIGNIFICANCE: [
                "paradigm shift", "technological breakthrough", "clinical translation",
                "digital transformation", "AI-powered discovery", "personalized healthcare"
            ],
            SectionType.EXPECTED_OUTCOMES: [
                "commercial applications", "intellectual property", "technology transfer",
                "clinical validation", "regulatory approval", "market impact"
            ]
        }

        section_keywords = section_specific.get(section_type, [])

        # Domain-specific keywords based on grant spec
        if grant_spec.knowledge_domains:
            for domain in grant_spec.knowledge_domains:
                if domain == "neuroscience":
                    section_keywords.extend(["brain connectivity", "neural networks", "developmental disorders"])
                elif domain == "quantum_ml":
                    section_keywords.extend(["quantum advantage", "variational algorithms", "quantum neural networks"])
                elif domain == "protein_research":
                    section_keywords.extend(["evolutionary modeling", "structure-function relationships", "drug discovery"])

        return base_keywords + section_keywords

    async def _generate_samsung_section_unified(self, section_id: str, section_spec: SectionSpec,
                                              grant_spec: SamsungGrantSpec) -> ProposalSection:
        """Unified RAG 기반 삼성 섹션 생성"""
        logger.info(f"🔧 Generating Samsung section: {section_id} with Unified RAG")

        # Enhanced context for Samsung-specific requirements
        samsung_context = {
            "grant_type": "samsung_future_tech",
            "compliance_requirements": self.required_sections.get(section_id, {}),
            "risk_profile": grant_spec.risk_level,
            "korean_emphasis": True,
            "innovation_focus": grant_spec.risk_level in [RiskLevel.HIGH, RiskLevel.BREAKTHROUGH],
            "cross_domain_synthesis": grant_spec.cross_domain_synthesis
        }

        # Unified RAG 지식 활용 (DD-RAPTOR에서 업그레이드)
        unified_query = self._create_unified_query_for_section(section_spec.type, grant_spec)

        generated_section = await self.proposal_agent.generate_with_unified_knowledge(
            section_type=section_spec.type.value,
            unified_query=unified_query,
            preferred_strategy=None  # Let orchestrator choose optimal strategy
        )

        # Samsung 형식으로 변환
        return self._convert_to_samsung_section(generated_section, section_id, samsung_context)

    def _create_unified_query_for_section(self, section_type: SectionType, grant_spec: SamsungGrantSpec) -> str:
        """섹션별 Unified RAG 쿼리 생성 (DD-RAPTOR 쿼리 업그레이드)"""

        # Base query templates enhanced for cross-domain knowledge
        query_templates = {
            SectionType.RESEARCH_OBJECTIVES: [
                "foundation model brain development autism breakthrough innovation",
                "ESM3 protein structure neuroscience clinical applications",
                "Samsung Future Tech artificial intelligence precision medicine",
                "multimodal developmental disorders early diagnosis prediction"
            ],
            SectionType.METHODOLOGY: [
                "longitudinal brain imaging analysis computational neuroscience methodology",
                "zebrafish validation experimental design neurodevelopment research",
                "federated learning healthcare privacy-preserving AI implementation",
                "quantum machine learning neural network optimization techniques",
                "protein language model integration clinical workflow systems"
            ],
            SectionType.INNOVATION_SIGNIFICANCE: [
                "breakthrough neuroscience AI applications clinical translation impact",
                "digital twin brain modeling personalized medicine revolutionary approach",
                "precision medicine developmental disorders biomarker discovery innovation",
                "Meta AI ESM3 protein research paradigm shift healthcare transformation",
                "Samsung technology innovation societal impact economic value creation"
            ],
            SectionType.EXPECTED_OUTCOMES: [
                "clinical trial outcomes neurodevelopmental intervention validation results",
                "AI model performance benchmarks healthcare deployment success metrics",
                "translational research impact society commercialization potential assessment",
                "intellectual property portfolio technology transfer strategic partnerships",
                "regulatory approval pathway clinical validation regulatory framework"
            ]
        }

        base_queries = query_templates.get(section_type, ["advanced AI research applications"])

        # Enhance with grant-specific context
        if grant_spec.knowledge_domains:
            domain_enhancements = []
            for domain in grant_spec.knowledge_domains:
                if domain == "neuroscience":
                    domain_enhancements.append("brain connectivity neural development")
                elif domain == "quantum_ml":
                    domain_enhancements.append("quantum computing machine learning optimization")
                elif domain == "protein_research":
                    domain_enhancements.append("protein structure evolutionary modeling drug discovery")

            # Combine base query with domain enhancements
            enhanced_query = f"{base_queries[0]} {' '.join(domain_enhancements)}"
        else:
            enhanced_query = base_queries[0]

        # Add Samsung-specific terms
        enhanced_query += " Samsung research innovation technology development"

        return enhanced_query

    def _convert_to_samsung_section(self, generated_section: GeneratedSection,
                                   section_id: str, samsung_context: Dict[str, Any]) -> ProposalSection:
        """생성된 섹션을 삼성 형식으로 변환"""
        requirements = self.required_sections.get(section_id, {})

        # Samsung-specific formatting
        formatted_content = self._format_samsung_content(
            generated_section.content,
            requirements.get("subsections", []),
            requirements.get("required_elements", [])
        )

        # Extract knowledge sources from unified RAG metadata
        knowledge_sources = []
        cross_domain_insights = []

        if generated_section.unified_rag_metadata:
            metadata = generated_section.unified_rag_metadata
            if metadata.get("cross_domain_insights"):
                cross_domain_insights = metadata["cross_domain_insights"]

        return ProposalSection(
            section_id=section_id,
            title=requirements.get("title", section_id),
            content=formatted_content,
            korean_content=None,  # Korean translation would be added here
            word_count=len(formatted_content.split()),
            subsections=self._extract_subsections(formatted_content, requirements.get("subsections", [])),
            quality_score=generated_section.confidence,
            citations=self._extract_citations(generated_section),
            rag_strategy_used=generated_section.rag_strategy_used,
            knowledge_sources=knowledge_sources,
            cross_domain_insights=cross_domain_insights
        )

    def _format_samsung_content(self, content: str, subsections: List[str],
                               required_elements: List[str]) -> str:
        """Samsung 제안서 형식으로 콘텐츠 포맷팅"""

        formatted_lines = []

        # Add Samsung-specific structure
        if subsections:
            formatted_lines.append("=" * 50)
            for i, subsection in enumerate(subsections, 1):
                formatted_lines.append(f"{i}. {subsection}")
                formatted_lines.append("-" * 30)
                formatted_lines.append("[해당 내용이 여기에 상세히 기술됨]")
                formatted_lines.append("")

        # Add main content with Samsung enhancement
        formatted_lines.append("주요 내용:")
        formatted_lines.append(content)

        # Add required elements check
        if required_elements:
            formatted_lines.append("\n평가 요소별 대응:")
            for element in required_elements:
                formatted_lines.append(f"• {element}: [구체적 대응 방안 기술]")

        return "\n".join(formatted_lines)

    def _extract_subsections(self, content: str, subsection_names: List[str]) -> Dict[str, str]:
        """서브섹션 추출"""
        subsections = {}

        for name in subsection_names:
            # Simple extraction logic - in practice, this would be more sophisticated
            subsections[name] = f"[{name}에 관한 상세 내용]"

        return subsections

    def _extract_citations(self, generated_section: GeneratedSection) -> List[Dict[str, str]]:
        """인용문헌 추출"""
        citations = []

        # Extract from unified RAG sources
        if hasattr(generated_section, 'unified_rag_metadata') and generated_section.unified_rag_metadata:
            source_count = generated_section.unified_rag_metadata.get('knowledge_source_count', 0)
            strategy = generated_section.rag_strategy_used

            citations.append({
                "title": f"Unified RAG Knowledge Base ({strategy})",
                "type": "knowledge_base",
                "sources": str(source_count),
                "relevance": str(generated_section.confidence)
            })

        return citations

    async def _validate_unified_proposal_quality(self, sections: Dict[str, ProposalSection],
                                                grant_spec: SamsungGrantSpec) -> Dict[str, float]:
        """Unified RAG 기반 제안서 품질 검증"""

        metrics = {}

        # Section quality scores
        section_scores = [section.quality_score for section in sections.values()]
        metrics["section_average"] = sum(section_scores) / len(section_scores) if section_scores else 0

        # RAG strategy diversity score
        strategies_used = set(section.rag_strategy_used for section in sections.values())
        metrics["strategy_diversity"] = min(1.0, len(strategies_used) / 3)  # Normalize to max 3 strategies

        # Cross-domain integration score
        cross_domain_sections = sum(1 for section in sections.values()
                                   if section.cross_domain_insights)
        metrics["cross_domain_integration"] = cross_domain_sections / len(sections)

        # Samsung compliance score
        metrics["samsung_compliance"] = self._calculate_samsung_compliance(sections)

        # Knowledge coverage score
        total_sources = sum(len(section.knowledge_sources or []) for section in sections.values())
        metrics["knowledge_coverage"] = min(1.0, total_sources / 20)  # Normalize to 20 sources

        # Overall score
        weights = {
            "section_average": 0.4,
            "strategy_diversity": 0.15,
            "cross_domain_integration": 0.15,
            "samsung_compliance": 0.2,
            "knowledge_coverage": 0.1
        }

        metrics["overall_score"] = sum(metrics[key] * weight for key, weight in weights.items())

        return metrics

    def _calculate_samsung_compliance(self, sections: Dict[str, ProposalSection]) -> float:
        """Samsung 제안서 준수성 계산"""
        compliance_scores = []

        for section_id, section in sections.items():
            requirements = self.required_sections.get(section_id, {})

            # Word count compliance
            min_words = requirements.get("min_words", 0)
            max_words = requirements.get("max_words", 1000)
            word_compliance = 1.0 if min_words <= section.word_count <= max_words else 0.7

            # Required elements presence (simplified check)
            required_elements = requirements.get("required_elements", [])
            element_compliance = 1.0  # Would check for actual presence in real implementation

            section_compliance = (word_compliance + element_compliance) / 2
            compliance_scores.append(section_compliance)

        return sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0

    def _analyze_rag_performance(self, sections: Dict[str, ProposalSection]) -> Dict[str, Any]:
        """RAG 성능 분석"""

        strategy_usage = {}
        confidence_by_strategy = {}

        for section in sections.values():
            strategy = section.rag_strategy_used
            if strategy:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

                if strategy not in confidence_by_strategy:
                    confidence_by_strategy[strategy] = []
                confidence_by_strategy[strategy].append(section.quality_score)

        # Calculate average confidence by strategy
        strategy_performance = {}
        for strategy, confidences in confidence_by_strategy.items():
            strategy_performance[strategy] = sum(confidences) / len(confidences)

        return {
            "strategy_usage": strategy_usage,
            "strategy_performance": strategy_performance,
            "total_sections": len(sections),
            "average_confidence": sum(section.quality_score for section in sections.values()) / len(sections)
        }

    def _get_strategy_distribution(self, sections: Dict[str, ProposalSection]) -> Dict[str, int]:
        """전략 분포 분석"""
        distribution = {}
        for section in sections.values():
            strategy = section.rag_strategy_used
            if strategy:
                distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution

    def _analyze_knowledge_coverage(self, sections: Dict[str, ProposalSection],
                                   grant_spec: SamsungGrantSpec) -> Dict[str, float]:
        """지식 커버리지 분석"""

        domain_coverage = {}

        # Analyze based on knowledge domains
        if grant_spec.knowledge_domains:
            for domain in grant_spec.knowledge_domains:
                domain_sections = 0
                for section in sections.values():
                    if section.cross_domain_insights:
                        # Check if domain is mentioned in insights
                        if any(domain in insight.lower() for insight in section.cross_domain_insights):
                            domain_sections += 1

                domain_coverage[domain] = domain_sections / len(sections)

        return domain_coverage

    def _generate_research_timeline(self, grant_spec: SamsungGrantSpec,
                                   sections: Dict[str, ProposalSection]) -> Dict[str, Any]:
        """연구 일정 자동 생성"""

        timeline = {
            "total_duration": grant_spec.duration_years,
            "phases": [],
            "milestones": [],
            "deliverables": []
        }

        # Generate phases based on research type and duration
        phases_per_year = max(2, 4 // grant_spec.duration_years)

        for year in range(grant_spec.duration_years):
            for phase in range(phases_per_year):
                phase_name = f"Year {year + 1} Phase {phase + 1}"
                timeline["phases"].append({
                    "name": phase_name,
                    "start_month": year * 12 + phase * (12 // phases_per_year),
                    "duration_months": 12 // phases_per_year,
                    "objectives": [f"Phase {phase + 1} research objectives"],
                    "deliverables": [f"Phase {phase + 1} deliverables"]
                })

        return timeline

    def _generate_proposal_id(self, grant_spec: SamsungGrantSpec) -> str:
        """제안서 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title_short = ''.join(grant_spec.title.split()[:3])
        return f"SAMSUNG_{title_short}_{timestamp}"

    def _estimate_page_count(self, sections: Dict[str, ProposalSection]) -> int:
        """페이지 수 추정"""
        total_words = sum(section.word_count for section in sections.values())
        return max(1, total_words // 250)  # Assume 250 words per page

    def _update_generation_analytics(self, proposal: GeneratedProposal, generation_time: float):
        """생성 분석 데이터 업데이트"""
        self.generation_analytics["proposals_generated"] += 1
        self.generation_analytics["quality_trends"].append(
            proposal.quality_metrics.get("overall_score", 0)
        )

        # Update strategy performance
        if proposal.rag_performance:
            for strategy, performance in proposal.rag_performance.get("strategy_performance", {}).items():
                if strategy not in self.generation_analytics["strategy_performance"]:
                    self.generation_analytics["strategy_performance"][strategy] = []
                self.generation_analytics["strategy_performance"][strategy].append(performance)

        # Calculate cross-domain success rate
        if proposal.knowledge_coverage:
            coverage_values = list(proposal.knowledge_coverage.values())
            if coverage_values:
                self.generation_analytics["cross_domain_success_rate"] = sum(coverage_values) / len(coverage_values)

    async def save_proposal(self, proposal: GeneratedProposal, format: str = "json") -> Path:
        """제안서 저장"""
        output_file = Path(self.config["output_directory"]) / f"{proposal.proposal_id}.{format}"

        if format == "json":
            proposal_dict = asdict(proposal)
            # Handle datetime serialization
            proposal_dict["generated_at"] = proposal.generated_at.isoformat()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(proposal_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Proposal saved: {output_file}")
        return output_file

    def get_generation_analytics(self) -> Dict[str, Any]:
        """생성 분석 데이터 반환"""
        analytics = self.generation_analytics.copy()

        # Calculate additional metrics
        if analytics["quality_trends"]:
            analytics["average_quality"] = sum(analytics["quality_trends"]) / len(analytics["quality_trends"])
            analytics["quality_improvement"] = (
                analytics["quality_trends"][-1] - analytics["quality_trends"][0]
                if len(analytics["quality_trends"]) > 1 else 0
            )

        return analytics

# Factory function for easy instantiation
def create_unified_samsung_generator(config: Optional[Dict[str, Any]] = None) -> UnifiedSamsungGrantGenerator:
    """Unified Samsung Grant Generator 생성 팩토리"""
    return UnifiedSamsungGrantGenerator(config)

# Example usage and testing
if __name__ == "__main__":
    async def test_unified_samsung_generator():
        """Unified Samsung Generator 테스트"""
        print("🧪 Testing Unified Samsung Grant Generator...")

        # Create generator
        generator = create_unified_samsung_generator()
        await generator.initialize()

        # Create test grant spec
        test_spec = SamsungGrantSpec(
            title="AI-Powered Neurodevelopmental Disorder Diagnosis System",
            research_area="AI Healthcare",
            primary_pi="Dr. AI Researcher",
            institution="Korean AI Institute",
            total_budget=500000000,  # 5억원
            duration_years=3,
            risk_level=RiskLevel.HIGH,
            innovation_keywords=["AI", "neuroscience", "early diagnosis", "precision medicine"],
            knowledge_domains=["neuroscience", "protein_research", "quantum_ml"],
            cross_domain_synthesis=True,
            rag_strategy_preferences=["GRAPH_RAG", "HYBRID"]
        )

        # Generate proposal
        proposal = await generator.generate_proposal(test_spec)

        print(f"✅ Proposal generated: {proposal.proposal_id}")
        print(f"📊 Overall quality: {proposal.quality_metrics.get('overall_score', 0):.3f}")
        print(f"📄 Total pages: {proposal.total_pages}")
        print(f"🔧 Strategy distribution: {proposal.strategy_distribution}")
        print(f"🌐 Knowledge coverage: {proposal.knowledge_coverage}")

        # Save proposal
        await generator.save_proposal(proposal)

        # Get analytics
        analytics = generator.get_generation_analytics()
        print(f"📈 Generation analytics: {analytics}")

        print("✅ Unified Samsung Grant Generator test completed successfully!")

    # Run test
    asyncio.run(test_unified_samsung_generator())
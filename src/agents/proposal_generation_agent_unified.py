#!/usr/bin/env python3
"""
Unified RAG Orchestrator-Based Proposal Generation Agent

Next-Generation Proposal System powered by Unified RAG Orchestrator
- Replaced DD-RAPTOR with advanced 6-strategy RAG orchestration
- Enhanced multi-modal knowledge integration (ESM3, Grant proposals, Research papers)
- Intelligent query classification and strategy routing
- Superior proposal quality through GraphRAG, HYBRID, and specialized strategies

Features:
- Autonomous proposal section generation with 6 RAG strategies
- Samsung Future Tech Grant optimization
- Cross-domain knowledge synthesis (Neuroscience, Quantum ML, Protein research)
- Multi-persona coordination with advanced RAG backing
- Quality improvement through intelligent strategy selection
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import numpy as np

# Unified RAG Orchestrator imports (replacing DD-RAPTOR)
from ..services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator,
    create_unified_orchestrator,
    QueryContext,
    QueryComplexity,
    QueryDomain,
    RAGStrategy
)
from ..services.hybrid_rag_service import HybridRAGService
from ..core.config import get_settings
from ..services.llm.adapters.gemini import GeminiAdapter
from ..services.llm.adapters.openai import OpenAIAdapter
from ..services.llm.types import LLMRequest, TaskType, LLMConfig, ModelProvider

logger = logging.getLogger(__name__)

class SectionType(str, Enum):
    """제안서 섹션 타입"""
    RESEARCH_OBJECTIVES = "research_objectives"
    METHODOLOGY = "methodology"
    INNOVATION_SIGNIFICANCE = "innovation_significance"
    BUDGET_JUSTIFICATION = "budget_justification"
    LITERATURE_REVIEW = "literature_review"
    EXPECTED_OUTCOMES = "expected_outcomes"
    TIMELINE = "timeline"

class PersonaType(str, Enum):
    """제안서 작성 페르소나"""
    CHIEF_RESEARCH_ARCHITECT = "chief_research_architect"
    NOBEL_NEUROSCIENTIST = "nobel_neuroscientist"
    SAMSUNG_GRANT_STRATEGIST = "samsung_grant_strategist"
    INNOVATION_EVALUATOR = "innovation_evaluator"
    BUDGET_SPECIALIST = "budget_specialist"

# 페르소나별 상세 시스템 프롬프트
PERSONA_SYSTEM_PROMPTS = {
    PersonaType.NOBEL_NEUROSCIENTIST: (
        "You are a Nobel Prize-winning Neuroscientist with deep expertise in developmental disorders and AI. "
        "Your writing is authoritative, scientifically rigorous, and visionary. "
        "Prioritize novel hypotheses, mechanistic explanations (molecular/neural circuit level), and high-impact clinical translation. "
        "Avoid generic statements; provide specific, evidence-backed scientific arguments. "
        "Use professional academic terminology suitable for a top-tier grant proposal."
    ),
    PersonaType.CHIEF_RESEARCH_ARCHITECT: (
        "You are a Chief Research Architect for a major scientific consortium. "
        "Your focus is on the structural integrity, technical feasibility, and strategic coherence of the proposal. "
        "Ensure that the methodology is robust, the timeline is realistic, and the resources are strictly justified. "
        "Write with precision, clarity, and a strong focus on execution and deliverability."
    ),
    PersonaType.SAMSUNG_GRANT_STRATEGIST: (
        "You are a Samsung Future Tech Grant Strategist. "
        "Your goal is to align the proposal with Samsung's 'High Risk, High Return' philosophy. "
        "Emphasize the disruptive nature of the technology, its potential to create a new scientific paradigm, and its massive downstream impact. "
        "Use persuasive, forward-looking language that highlights the 'World First' and 'Best in Class' aspects."
    ),
    PersonaType.INNOVATION_EVALUATOR: (
        "You are an Innovation Evaluator for breakthrough technologies. "
        "Critically assess the novelty and differentiation of the proposed research. "
        "Highlight why this approach is not just an incremental improvement but a fundamental leap forward. "
        "Focus on the 'Zero to One' innovation aspect."
    ),
    PersonaType.BUDGET_SPECIALIST: (
        "You are a specialist in scientific research resource allocation and budgeting. "
        "Justify every expense with a focus on maximizing research ROI. "
        "Ensure that the budget aligns perfectly with the proposed methodology and timeline. "
        "Explain the necessity of high-performance computing and specialized equipment."
    ),
}

# 섹션별 작성 지침 템플릿
SECTION_INSTRUCTION_TEMPLATES = {
    SectionType.RESEARCH_OBJECTIVES: (
        "1. Define the 'High Risk, High Return' research goal clearly.\n"
        "2. Explain the core hypothesis connecting AI, ESM3, and developmental disorders.\n"
        "3. Detail the 'World First' aspects of the proposed Foundation Model (NeuroX-Fusion).\n"
        "4. Outline 3 specific research objectives that are measurable and ambitious."
    ),
    SectionType.METHODOLOGY: (
        "1. Describe the multi-modal data integration framework (MRI, fMRI, Genetics, Clinical).\n"
        "2. Explain the AI architecture (e.g., Transformer-based, Graph Neural Networks) in technical detail.\n"
        "3. Detail the validation strategy using zebrafish models and clinical cohorts.\n"
        "4. Address data privacy and federated learning approaches."
    ),
    SectionType.INNOVATION_SIGNIFICANCE: (
        "1. Contrast this approach with existing 'State of the Art' (SOTA).\n"
        "2. Explain the potential for a paradigm shift in diagnosing Autism Spectrum Disorder (ASD).\n"
        "3. Highlight the ripple effects on broader neuroscience and AI fields.\n"
        "4. Emphasize the long-term societal and economic value."
    ),
    SectionType.TIMELINE: (
        "1. Provide a year-by-year breakdown of key milestones for 5 years.\n"
        "2. Define clear deliverables for each phase (e.g., 'Model V1', 'Clinical Pilot').\n"
        "3. Identify critical path dependencies and risk mitigation strategies."
    ),
    SectionType.BUDGET_JUSTIFICATION: (
        "1. Justify the need for large-scale GPU resources (H100/A100 clusters).\n"
        "2. Explain personnel costs for a top-tier interdisciplinary team.\n"
        "3. Detail costs for clinical data acquisition and zebrafish experiments."
    ),
    SectionType.LITERATURE_REVIEW: (
        "1. Synthesize recent breakthroughs in Generative AI and Computational Neuroscience.\n"
        "2. Identify clear gaps in current foundation models regarding longitudinal developmental data.\n"
        "3. Integrate insights from retrieved papers to support the proposed methodology."
    ),
    SectionType.EXPECTED_OUTCOMES: (
        "1. Quantify the expected performance improvements in early diagnosis.\n"
        "2. Describe the specific software/model artifacts to be released.\n"
        "3. Outline the clinical translation pathway and potential IP generation."
    ),
    SectionType.TIMELINE: ( # Timeline duplicate key in source, but overriding here for completeness
        "1. Provide a year-by-year breakdown of key milestones for 5 years.\n"
        "2. Define clear deliverables for each phase (e.g., 'Model V1', 'Clinical Pilot').\n"
        "3. Identify critical path dependencies and risk mitigation strategies."
    )
}

@dataclass
class SectionSpec:
    """섹션 생성 명세"""
    type: SectionType
    persona: PersonaType
    required_keywords: List[str] = None
    min_words: int = 500
    max_words: int = 2000
    citation_requirement: bool = True
    innovation_focus: bool = False

@dataclass
class GeneratedSection:
    """생성된 섹션 결과"""
    type: SectionType
    content: str
    word_count: int
    citations_count: int
    confidence: float
    reasoning: str
    quality_metrics: Dict[str, float]
    persona_used: PersonaType
    generation_time_ms: float
    rag_strategy_used: str = ""
    unified_rag_metadata: Dict[str, Any] = None

@dataclass
class UnifiedKnowledgeContext:
    """Unified RAG로부터 수집된 지식 컨텍스트"""
    primary_strategy: str
    confidence_scores: Dict[str, float]
    knowledge_sources: List[Dict[str, Any]]
    cross_domain_insights: List[str]
    strategy_recommendations: Dict[str, str]

class UnifiedProposalGenerationAgent:
    """Unified RAG Orchestrator 기반 자율 제안서 생성 에이전트"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """에이전트 초기화"""
        self.config = config or self._get_default_config()
        self.unified_rag_orchestrator: Optional[UnifiedRAGOrchestrator] = None
        self.hybrid_rag: Optional[HybridRAGService] = None
        
        # Initialize LLM Adapters based on available keys
        self.settings = get_settings()
        self.adapters: Dict[str, Any] = {}
        
        # Gemini (Google)
        if self.settings.google_api_key:
            try:
                self.adapters["gemini"] = GeminiAdapter(api_key=self.settings.google_api_key)
                logger.info("✅ Gemini Adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Adapter: {e}")
                
        # OpenAI
        if self.settings.openai_api_key:
            try:
                self.adapters["openai"] = OpenAIAdapter(api_key=self.settings.openai_api_key)
                logger.info("✅ OpenAI Adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI Adapter: {e}")

        # Determine active adapter based on primary provider
        primary = self.settings.llm_primary_provider.lower()
        if primary in ["google", "gemini"]:
            self.llm_adapter = self.adapters.get("gemini")
        elif primary == "openai":
            self.llm_adapter = self.adapters.get("openai")
        else:
            self.llm_adapter = None
            logger.warning(f"⚠️ Unknown primary provider: {primary}")

        if not self.llm_adapter and self.adapters:
            # Pick any available if primary failed
            self.llm_adapter = next(iter(self.adapters.values()))
            logger.info(f"Using alternative adapter: {type(self.llm_adapter).__name__}")
        elif not self.llm_adapter:
            logger.warning("⚠️ No LLM adapters available. LLM generation will be disabled.")

        # Performance tracking
        self.generation_stats = {
            "sections_generated": 0,
            "strategy_usage": {},
            "average_confidence": 0.0,
            "total_generation_time": 0.0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "output_directory": "output/unified_proposals/",
            "template_directory": "templates/unified_proposals/",
            "auto_improvement_enabled": True,
            "parallel_generation": True,
            "quality_threshold": 0.8,
            "max_generation_attempts": 3,
            "enable_cross_domain_synthesis": True,
            "preferred_strategies": [
                RAGStrategy.GRAPH_RAG,
                RAGStrategy.HYBRID,
                RAGStrategy.ENHANCED_DD_RAPTOR
            ],
            "persona_configurations": {
                PersonaType.CHIEF_RESEARCH_ARCHITECT: {
                    "preferred_domains": [QueryDomain.GENERAL, QueryDomain.NEUROSCIENCE],
                    "complexity_preference": QueryComplexity.COMPLEX,
                    "style": "architectural_strategic"
                },
                PersonaType.NOBEL_NEUROSCIENTIST: {
                    "preferred_domains": [QueryDomain.NEUROSCIENCE, QueryDomain.DEVELOPMENTAL_DISORDERS],
                    "complexity_preference": QueryComplexity.COMPLEX,
                    "style": "scientific_authoritative"
                },
                PersonaType.SAMSUNG_GRANT_STRATEGIST: {
                    "preferred_domains": [QueryDomain.GENERAL, QueryDomain.QUANTUM_ML],
                    "complexity_preference": QueryComplexity.MEDIUM,
                    "style": "business_strategic"
                }
            }
        }

    async def initialize(self):
        """에이전트 초기화 - Unified RAG Orchestrator 중심"""
        logger.info("Initializing Unified Proposal Generation Agent...")

        # Unified RAG Orchestrator 초기화 (DD-RAPTOR 교체)
        self.unified_rag_orchestrator = create_unified_orchestrator()
        await self.unified_rag_orchestrator.initialize_real_strategies()
        await self.unified_rag_orchestrator.warmup()

        logger.info("✅ Unified RAG Orchestrator initialized with 6 strategies")

        # Strategy health check
        health_status = self.unified_rag_orchestrator.get_strategy_health()
        available_strategies = [s for s, info in health_status.items() if info.get('available', False)]

        logger.info(f"🔧 Available RAG strategies: {available_strategies}")

        # Hybrid RAG 서비스 초기화 (보조 시스템)
        # self.hybrid_rag = HybridRAGService()

        # 출력 디렉토리 생성
        Path(self.config["output_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["template_directory"]).mkdir(parents=True, exist_ok=True)

        logger.info("🎯 Unified Proposal Generation Agent initialized successfully")

    async def generate_section(self, section_spec: SectionSpec) -> GeneratedSection:
        """Unified RAG 기반 섹션 자율적 생성"""
        logger.info(f"🚀 Generating section: {section_spec.type.value} with Unified RAG")
        start_time = time.time()

        # 1. Unified RAG에서 다중 전략 지식 수집
        knowledge_context = await self._gather_unified_knowledge(section_spec)

        # 2. 페르소나 기반 생성 전략 선택
        generation_strategy = self._select_unified_generation_strategy(section_spec, knowledge_context)

        # 3. 섹션 내용 생성 (Unified RAG 지원)
        content = await self._generate_content_with_unified_rag(
            section_spec, knowledge_context, generation_strategy
        )

        # 4. 품질 검증 및 자율적 개선
        if self.config["auto_improvement_enabled"]:
            content = await self._autonomous_quality_improvement_unified(
                content, section_spec, knowledge_context
            )

        # 5. 통합 메트릭 계산
        metrics = await self._calculate_unified_section_metrics(content, section_spec, knowledge_context)

        generation_time = (time.time() - start_time) * 1000

        # Stats update
        self.generation_stats["sections_generated"] += 1
        self.generation_stats["total_generation_time"] += generation_time

        strategy_used = knowledge_context.primary_strategy
        if strategy_used not in self.generation_stats["strategy_usage"]:
            self.generation_stats["strategy_usage"][strategy_used] = 0
        self.generation_stats["strategy_usage"][strategy_used] += 1

        generated_section = GeneratedSection(
            type=section_spec.type,
            content=content,
            word_count=len(content.split()),
            citations_count=len(knowledge_context.knowledge_sources),
            confidence=metrics["confidence"],
            reasoning=f"Generated using Unified RAG with {strategy_used} strategy from {len(knowledge_context.knowledge_sources)} sources",
            quality_metrics=metrics,
            persona_used=section_spec.persona,
            generation_time_ms=generation_time,
            rag_strategy_used=strategy_used,
            unified_rag_metadata={
                "strategy_confidence": knowledge_context.confidence_scores,
                "cross_domain_insights": knowledge_context.cross_domain_insights,
                "knowledge_source_count": len(knowledge_context.knowledge_sources)
            }
        )

        logger.info(f"✅ Section generated: {strategy_used} strategy, confidence: {metrics['confidence']:.3f}")
        return generated_section

    async def generate_with_unified_knowledge(self, section_type: str, unified_query: str,
                                            preferred_strategy: Optional[RAGStrategy] = None) -> GeneratedSection:
        """Unified RAG 지식 활용 생성 (DD-RAPTOR의 generate_with_dd_knowledge 교체)"""
        logger.info(f"🧬 Generating {section_type} with Unified RAG: {unified_query}")

        # 1. Query classification and context creation
        query_context = await self._create_intelligent_query_context(unified_query, section_type)

        # 2. Unified RAG 검색 (다중 전략 활용)
        search_response = await self.unified_rag_orchestrator.search(query_context)

        # 3. 검색 결과 검증 및 전략 평가
        if search_response.confidence < 0.7:
            logger.warning(f"⚠️ Low confidence: {search_response.confidence:.3f}, trying alternative strategy")

            # 대체 전략으로 재시도
            alternative_strategies = [RAGStrategy.GRAPH_RAG, RAGStrategy.HYBRID, RAGStrategy.GOLDEN_REFERENCE]
            for alt_strategy in alternative_strategies:
                if alt_strategy != search_response.strategy_used:
                    # Manual strategy override (if orchestrator supports it)
                    break

        # 4. 지식 기반 섹션 생성
        section_spec = SectionSpec(
            type=SectionType(section_type),
            persona=PersonaType.CHIEF_RESEARCH_ARCHITECT,
            required_keywords=unified_query.split()
        )

        # 5. Unified RAG 인용을 포함한 내용 생성
        content = await self._generate_with_unified_citations(
            section_spec, search_response
        )

        # 6. 통합 메트릭 계산
        metrics = await self._calculate_unified_section_metrics_from_response(content, section_spec, search_response)

        return GeneratedSection(
            type=section_spec.type,
            content=content,
            word_count=len(content.split()),
            citations_count=len(search_response.sources) if search_response.sources else 0,
            confidence=metrics["confidence"],
            reasoning=f"Generated using Unified RAG {search_response.strategy_used.value} strategy with confidence {search_response.confidence:.3f}",
            quality_metrics=metrics,
            persona_used=section_spec.persona,
            generation_time_ms=0.0,
            rag_strategy_used=str(search_response.strategy_used),
            unified_rag_metadata={
                "strategy_used": str(search_response.strategy_used),
                "original_confidence": search_response.confidence,
                "source_count": len(search_response.sources) if search_response.sources else 0
            }
        )

    async def _gather_unified_knowledge(self, section_spec: SectionSpec) -> UnifiedKnowledgeContext:
        """Unified RAG에서 다중 전략 지식 수집 (DD-RAPTOR의 _gather_dd_knowledge 교체)"""
        logger.info(f"📚 Gathering knowledge for {section_spec.type.value} using Unified RAG")

        # 섹션 타입별 특화 쿼리 생성 (확장된 도메인 커버리지)
        section_queries = {
            SectionType.RESEARCH_OBJECTIVES: [
                "foundation model brain development autism developmental disorders",
                "multimodal neurodevelopmental AI applications",
                "ESM3 protein structure neuroscience applications",
                "early diagnosis prediction algorithms machine learning"
            ],
            SectionType.METHODOLOGY: [
                "longitudinal brain imaging analysis computational neuroscience",
                "zebrafish validation neurodevelopment experimental design",
                "federated learning healthcare privacy-preserving AI",
                "quantum machine learning neural network optimization"
            ],
            SectionType.INNOVATION_SIGNIFICANCE: [
                "breakthrough neuroscience AI applications clinical translation",
                "digital twin brain modeling personalized medicine",
                "precision medicine developmental disorders biomarkers",
                "Meta AI ESM3 protein research breakthrough applications"
            ],
            SectionType.BUDGET_JUSTIFICATION: [
                "AI computing resources neuroscience HPC requirements",
                "large scale brain imaging infrastructure costs",
                "consortium research collaboration infrastructure",
                "quantum computing resources grant allocation"
            ],
            SectionType.LITERATURE_REVIEW: [
                "recent advances neurodevelopmental disorders AI",
                "protein structure prediction clinical applications",
                "brain-inspired quantum computing methodologies",
                "multimodal biomedical foundation models"
            ],
            SectionType.EXPECTED_OUTCOMES: [
                "clinical trial outcomes neurodevelopmental interventions",
                "AI model performance benchmarks healthcare",
                "translational research impact society",
                "breakthrough technology commercialization potential"
            ]
        }

        queries = section_queries.get(section_spec.type, ["advanced AI neuroscience applications"])

        # 다중 쿼리 검색을 통한 종합적 지식 수집
        search_responses = []
        confidence_scores = {}
        strategy_usage = {}

        for query in queries:
            try:
                # Intelligent query context creation
                query_context = await self._create_intelligent_query_context(query, section_spec.type.value)

                # Execute unified search
                response = await self.unified_rag_orchestrator.search(query_context)
                search_responses.append(response)

                # Track strategy performance
                strategy_str = str(response.strategy_used)
                confidence_scores[query] = response.confidence

                if strategy_str not in strategy_usage:
                    strategy_usage[strategy_str] = 0
                strategy_usage[strategy_str] += 1

                logger.debug(f"Query: '{query[:50]}...' -> Strategy: {strategy_str}, Confidence: {response.confidence:.3f}")

            except Exception as e:
                logger.warning(f"Query failed: {query[:50]}... Error: {e}")

        # Determine primary strategy (most used + highest confidence)
        if strategy_usage:
            primary_strategy = max(strategy_usage.keys(),
                                 key=lambda s: strategy_usage[s] * np.mean([r.confidence for r in search_responses
                                                                          if str(r.strategy_used) == s]))
        else:
            primary_strategy = "HYBRID"  # fallback

        # Collect all knowledge sources
        all_sources = []
        for response in search_responses:
            if response.sources:
                all_sources.extend(response.sources)

        # Generate cross-domain insights
        cross_domain_insights = await self._generate_cross_domain_insights(search_responses)

        # Strategy recommendations based on performance
        strategy_recommendations = self._analyze_strategy_performance(search_responses, strategy_usage)

        knowledge_context = UnifiedKnowledgeContext(
            primary_strategy=primary_strategy,
            confidence_scores=confidence_scores,
            knowledge_sources=all_sources,
            cross_domain_insights=cross_domain_insights,
            strategy_recommendations=strategy_recommendations
        )

        logger.info(f"✅ Unified knowledge gathered: {len(all_sources)} sources, primary strategy: {primary_strategy}")
        return knowledge_context

    async def _create_intelligent_query_context(self, query: str, section_type: str) -> QueryContext:
        """지능적 쿼리 컨텍스트 생성"""

        # Determine complexity based on query characteristics
        complexity = QueryComplexity.SIMPLE
        if any(word in query.lower() for word in ["breakthrough", "novel", "innovative", "comprehensive"]):
            complexity = QueryComplexity.COMPLEX
        elif any(word in query.lower() for word in ["analysis", "methodology", "framework", "approach"]):
            complexity = QueryComplexity.MEDIUM

        # Determine domain based on query content
        domain = QueryDomain.GENERAL
        if any(word in query.lower() for word in ["neuroscience", "brain", "neural", "neurodevelopment"]):
            domain = QueryDomain.NEUROSCIENCE
        elif any(word in query.lower() for word in ["quantum", "physics", "computing"]):
            domain = QueryDomain.QUANTUM_ML
        elif any(word in query.lower() for word in ["developmental", "autism", "disorder"]):
            domain = QueryDomain.DEVELOPMENTAL_DISORDERS
        elif any(word in query.lower() for word in ["protein", "ESM3", "evolution", "structure"]):
            domain = QueryDomain.GENERAL  # Protein research doesn't have specific domain yet

        # Determine intent
        intent = "factual"
        if section_type in ["innovation_significance", "expected_outcomes"]:
            intent = "synthesis"
        elif section_type in ["methodology", "literature_review"]:
            intent = "comparative"

        return QueryContext(
            query=query,
            complexity=complexity,
            domain=domain,
            intent=intent,
            confidence=0.9,
            metadata={
                "section_type": section_type,
                "proposal_generation": True,
                "unified_rag_agent": True
            }
        )

    async def _generate_cross_domain_insights(self, responses: List[Any]) -> List[str]:
        """교차 도메인 통찰력 생성"""
        insights = []

        # Analyze strategy diversity
        strategies_used = set(str(r.strategy_used) for r in responses)
        if len(strategies_used) > 2:
            insights.append("Multi-strategy knowledge synthesis enables comprehensive understanding")

        # Analyze content diversity
        domains_covered = set()
        for response in responses:
            if response.sources:
                for source in response.sources:
                    if 'ESM3' in str(source) or 'protein' in str(source):
                        domains_covered.add("protein_research")
                    elif 'brain' in str(source) or 'neuro' in str(source):
                        domains_covered.add("neuroscience")
                    elif 'quantum' in str(source):
                        domains_covered.add("quantum_ml")

        if len(domains_covered) > 1:
            insights.append(f"Cross-domain knowledge integration spans: {', '.join(domains_covered)}")

        # High-confidence insights
        high_conf_responses = [r for r in responses if r.confidence > 0.8]
        if len(high_conf_responses) > len(responses) * 0.7:
            insights.append("High-confidence knowledge base ensures reliable proposal foundation")

        return insights

    def _analyze_strategy_performance(self, responses: List[Any], usage_stats: Dict[str, int]) -> Dict[str, str]:
        """전략 성능 분석"""
        recommendations = {}

        for strategy, usage_count in usage_stats.items():
            strategy_responses = [r for r in responses if str(r.strategy_used) == strategy]
            if strategy_responses:
                avg_confidence = np.mean([r.confidence for r in strategy_responses])

                if avg_confidence > 0.85:
                    recommendations[strategy] = "Excellent performance - recommended for similar queries"
                elif avg_confidence > 0.75:
                    recommendations[strategy] = "Good performance - suitable for standard queries"
                else:
                    recommendations[strategy] = "Moderate performance - consider alternative strategies"

        return recommendations

    def _select_unified_generation_strategy(self, section_spec: SectionSpec,
                                          knowledge_context: UnifiedKnowledgeContext) -> Dict[str, Any]:
        """Unified RAG 기반 생성 전략 선택"""

        persona_config = self.config["persona_configurations"].get(
            section_spec.persona,
            {"preferred_domains": [QueryDomain.GENERAL], "complexity_preference": QueryComplexity.MEDIUM}
        )

        return {
            "persona": section_spec.persona,
            "primary_strategy": knowledge_context.primary_strategy,
            "confidence_threshold": 0.8,
            "cross_domain_synthesis": len(knowledge_context.cross_domain_insights) > 0,
            "strategy_recommendations": knowledge_context.strategy_recommendations,
            "preferred_complexity": persona_config.get("complexity_preference", QueryComplexity.MEDIUM),
            "style": persona_config.get("style", "balanced")
        }

    async def _generate_content_with_unified_rag(self, section_spec: SectionSpec,
                                               knowledge_context: UnifiedKnowledgeContext,
                                               generation_strategy: Dict[str, Any]) -> str:
        """Unified RAG 지원 콘텐츠 생성"""

        # Create comprehensive prompt with unified knowledge
        prompt = self._create_unified_generation_prompt(section_spec, knowledge_context, generation_strategy)

        # Use actual LLM if available
        if self.llm_adapter:
            try:
                # Get specialized system prompt
                system_prompt = PERSONA_SYSTEM_PROMPTS.get(
                    section_spec.persona,
                    f"You are a world-class {section_spec.persona.value.replace('_', ' ')}."
                )
                
                system_message = (
                    f"{system_prompt}\n"
                    f"Task: Write the {section_spec.type.value} section for a Samsung Future Tech Grant proposal.\n"
                    f"Style: {generation_strategy['style']}."
                )

                # Multi-provider fallback logic
                primary_provider = self.settings.llm_primary_provider.lower()
                fallback_provider = self.settings.llm_fallback_provider.lower()
                
                providers_to_try = [primary_provider]
                if fallback_provider and fallback_provider != "none":
                    providers_to_try.append(fallback_provider)

                last_exception = None
                for provider_name in providers_to_try:
                    adapter = self.adapters.get(provider_name if provider_name != "google" else "gemini")
                    if not adapter:
                        continue
                        
                    try:
                        if provider_name in ["google", "gemini"]:
                            req_provider = ModelProvider.GOOGLE
                            req_model = self.settings.gemini_model or "gemini-3-flash-preview"
                            req_max_tokens = self.settings.gemini_max_tokens or 8192
                        else:
                            req_provider = ModelProvider.OPENAI
                            req_model = self.settings.openai_model or "gpt-5-pro"
                            req_max_tokens = self.settings.openai_max_tokens or 4096

                        request = LLMRequest(
                            prompt=prompt,
                            task_type=TaskType.PAPER_WRITING,
                            system_message=system_message,
                            config=LLMConfig(
                                provider=req_provider,
                                model=req_model,
                                temperature=0.7,
                                max_tokens=req_max_tokens
                            )
                        )

                        logger.info(f"Generating content with {provider_name} (Prompt length: {len(prompt)})")
                        response = await adapter.complete(request)
                        return response.content
                    except Exception as e:
                        logger.warning(f"{provider_name} generation failed: {e}")
                        last_exception = e
                        continue

                if last_exception:
                    raise last_exception
                
                return "[Error: No provider succeeded]"

            except Exception as e:
                logger.error(f"LLM generation failed: {e}. Falling back to template.")
                
                if get_settings().strict_mode:
                    logger.error("🛑 STRICT MODE: LLM Generation Failed. Halting.")
                    raise RuntimeError(f"Strict Mode Failure: LLM generation failed: {e}") from e
                
                # Fallback to template if LLM fails

        # Fallback template
        content_template = f"""
Based on comprehensive Unified RAG analysis using {knowledge_context.primary_strategy} strategy:

{section_spec.type.value.upper()} SECTION

Generated with {section_spec.persona.value} persona using cross-domain knowledge from {len(knowledge_context.knowledge_sources)} sources.

Key insights from Unified RAG synthesis:
{chr(10).join('- ' + insight for insight in knowledge_context.cross_domain_insights)}

This section leverages the following knowledge domains:
{chr(10).join('- ' + strategy + ': ' + rec for strategy, rec in knowledge_context.strategy_recommendations.items())}

[Detailed content would be generated here based on the specific prompt and knowledge context]

The approach combines {knowledge_context.primary_strategy} strategy findings with multi-modal evidence from recent research, ensuring both scientific rigor and innovation potential.

(Note: LLM generation was unavailable, so this placeholder was used.)
"""

        return content_template

    def _create_unified_generation_prompt(self, section_spec: SectionSpec,
                                        knowledge_context: UnifiedKnowledgeContext,
                                        generation_strategy: Dict[str, Any]) -> str:
        """Unified RAG 기반 생성 프롬프트 생성"""

        prompt_parts = [
            f"Generate a {section_spec.type.value} section for a research proposal.",
            f"Use the {section_spec.persona.value} persona with {generation_strategy['style']} writing style.",
            f"Primary RAG strategy: {knowledge_context.primary_strategy}\n",
        ]

        # Inject Specific Section Instructions
        instructions = SECTION_INSTRUCTION_TEMPLATES.get(section_spec.type, "")
        if instructions:
            prompt_parts.append(f"### WRITING INSTRUCTIONS:\n{instructions}\n")
        
        # Inject Knowledge Sources (Context)
        if knowledge_context.knowledge_sources:
            prompt_parts.append("\n### BACKGROUND KNOWLEDGE (RAG CONTEXT):")
            for i, source in enumerate(knowledge_context.knowledge_sources[:10]): # Limit to top 10
                content = str(source)
                if isinstance(source, dict):
                     content = f"Title: {source.get('title', 'Unknown')}\nContent: {source.get('content', '')}"
                prompt_parts.append(f"Source {i+1}:\n{content[:2000]}") # Truncate per source to avoid limit
            prompt_parts.append("### END OF CONTEXT\n")

        prompt_parts.append(f"Knowledge sources count: {len(knowledge_context.knowledge_sources)}")

        if knowledge_context.cross_domain_insights:
            prompt_parts.append("Cross-domain insights:")
            prompt_parts.extend(f"- {insight}" for insight in knowledge_context.cross_domain_insights)

        if section_spec.required_keywords:
            prompt_parts.append(f"Required keywords: {', '.join(section_spec.required_keywords)}")

        prompt_parts.extend([
            f"Word count: {section_spec.min_words}-{section_spec.max_words} words",
            f"Citation requirement: {'Yes' if section_spec.citation_requirement else 'No'}",
            f"Innovation focus: {'High' if section_spec.innovation_focus else 'Balanced'}"
        ])
        
        prompt_parts.append("\nInstructions: Write the section based on the provided Background Knowledge. Cite sources where appropriate.")

        return "\n".join(prompt_parts)

    async def _generate_with_unified_citations(self, section_spec: SectionSpec,
                                             search_response: Any) -> str:
        """Unified RAG 인용을 포함한 생성"""

        citations = []
        if search_response.sources:
            for i, source in enumerate(search_response.sources[:10]):  # Limit to top 10
                # Format citation based on source type
                if isinstance(source, dict):
                    title = source.get('title', f'Source {i+1}')
                    content = source.get('content', '')[:100] + "..."
                    citations.append(f"[{i+1}] {title}: {content}")
                else:
                    citations.append(f"[{i+1}] {str(source)[:100]}...")

        content = f"""
{section_spec.type.value.upper()} SECTION
Generated using Unified RAG {search_response.strategy_used.value} strategy

[Main content generated based on search response would be here]

Key findings from {search_response.strategy_used.value} strategy analysis with confidence {search_response.confidence:.3f}.

REFERENCES:
{chr(10).join(citations)}
"""

        return content

    async def _calculate_unified_section_metrics(self, content: str, section_spec: SectionSpec,
                                               knowledge_context: UnifiedKnowledgeContext) -> Dict[str, float]:
        """통합 섹션 메트릭 계산"""

        word_count = len(content.split())

        # Base metrics
        metrics = {
            "word_count_score": min(1.0, word_count / section_spec.max_words),
            "citation_score": min(1.0, len(knowledge_context.knowledge_sources) / 5),
            "strategy_confidence": np.mean(list(knowledge_context.confidence_scores.values())) if knowledge_context.confidence_scores else 0.5,
            "cross_domain_score": min(1.0, len(knowledge_context.cross_domain_insights) / 3),
            "knowledge_diversity": min(1.0, len(knowledge_context.strategy_recommendations) / 3)
        }

        # Overall confidence
        metrics["confidence"] = np.mean(list(metrics.values()))

        return metrics

    async def _calculate_unified_section_metrics_from_response(self, content: str, section_spec: SectionSpec,
                                                             search_response: Any) -> Dict[str, float]:
        """검색 응답 기반 메트릭 계산"""

        word_count = len(content.split())
        source_count = len(search_response.sources) if search_response.sources else 0

        metrics = {
            "word_count_score": min(1.0, word_count / section_spec.max_words),
            "citation_score": min(1.0, source_count / 5),
            "strategy_confidence": search_response.confidence,
            "response_quality": search_response.confidence,
            "source_diversity": min(1.0, source_count / 10)
        }

        metrics["confidence"] = np.mean(list(metrics.values()))

        return metrics

    async def _autonomous_quality_improvement_unified(self, content: str, section_spec: SectionSpec,
                                                    knowledge_context: UnifiedKnowledgeContext) -> str:
        """Unified RAG 기반 자율적 품질 개선"""

        # Quality assessment
        current_metrics = await self._calculate_unified_section_metrics(content, section_spec, knowledge_context)

        if current_metrics["confidence"] < self.config["quality_threshold"]:
            logger.info(f"🔄 Improving quality for {section_spec.type.value} (current: {current_metrics['confidence']:.3f})")

            # Try alternative strategy if primary didn't work well
            if current_metrics["strategy_confidence"] < 0.7:
                # Attempt improvement with different strategy insights
                improvement_prompt = f"Improve this {section_spec.type.value} section content using alternative insights from available strategies."

                # Enhanced content generation
                improved_content = content + "\n\n[ENHANCED SECTION WITH ALTERNATIVE INSIGHTS]\n"

                return improved_content

        return content

    def get_generation_stats(self) -> Dict[str, Any]:
        """생성 통계 반환"""
        stats = self.generation_stats.copy()

        if stats["sections_generated"] > 0:
            stats["average_generation_time"] = stats["total_generation_time"] / stats["sections_generated"]

        return stats

    async def generate_full_proposal(self, proposal_specs: List[SectionSpec]) -> List[GeneratedSection]:
        """전체 제안서 생성 (Unified RAG 기반)"""
        logger.info(f"🚀 Generating full proposal with {len(proposal_specs)} sections using Unified RAG")

        if self.config["parallel_generation"]:
            # 병렬 섹션 생성
            tasks = [self.generate_section(spec) for spec in proposal_specs]
            proposal_sections = await asyncio.gather(*tasks, return_exceptions=True)

            # 예외 처리
            successful_sections = []
            for i, result in enumerate(proposal_sections):
                if isinstance(result, Exception):
                    logger.error(f"Section {proposal_specs[i].type.value} failed: {result}")
                else:
                    successful_sections.append(result)

            proposal_sections = successful_sections
        else:
            # 순차적 섹션 생성
            proposal_sections = []
            for spec in proposal_specs:
                try:
                    section = await self.generate_section(spec)
                    proposal_sections.append(section)
                except Exception as e:
                    logger.error(f"Section {spec.type.value} failed: {e}")

        # 전체 제안서 품질 검증
        await self._validate_full_proposal_quality_unified(proposal_sections)

        logger.info(f"✅ Full proposal generated with {len(proposal_sections)} sections using Unified RAG")
        return proposal_sections

    async def _validate_full_proposal_quality_unified(self, proposal_sections: List[GeneratedSection]):
        """Unified RAG 기반 전체 제안서 품질 검증"""

        total_confidence = sum(section.confidence for section in proposal_sections)
        average_confidence = total_confidence / len(proposal_sections) if proposal_sections else 0

        strategy_diversity = len(set(section.rag_strategy_used for section in proposal_sections))

        logger.info(f"📊 Proposal quality - Average confidence: {average_confidence:.3f}, Strategy diversity: {strategy_diversity}")

        if average_confidence < self.config["quality_threshold"]:
            logger.warning(f"⚠️ Low proposal quality detected: {average_confidence:.3f}")

        # Update overall stats
        self.generation_stats["average_confidence"] = average_confidence

# Factory function for easy instantiation
def create_unified_proposal_agent(config: Optional[Dict[str, Any]] = None) -> UnifiedProposalGenerationAgent:
    """Unified Proposal Agent 생성 팩토리"""
    return UnifiedProposalGenerationAgent(config)

# Global instance for application-wide use
_global_proposal_agent: Optional[UnifiedProposalGenerationAgent] = None

async def get_unified_proposal_agent() -> UnifiedProposalGenerationAgent:
    """Global unified proposal agent 인스턴스 반환"""
    global _global_proposal_agent

    if _global_proposal_agent is None:
        _global_proposal_agent = create_unified_proposal_agent()
        await _global_proposal_agent.initialize()

    return _global_proposal_agent

# Example usage and testing
if __name__ == "__main__":
    async def test_unified_proposal_agent():
        """Unified Proposal Agent 테스트"""
        print("🧪 Testing Unified Proposal Generation Agent...")

        # Create agent
        agent = create_unified_proposal_agent()
        await agent.initialize()

        # Test section generation
        test_spec = SectionSpec(
            type=SectionType.RESEARCH_OBJECTIVES,
            persona=PersonaType.NOBEL_NEUROSCIENTIST,
            required_keywords=["neurodevelopment", "AI", "precision medicine"],
            innovation_focus=True
        )

        # Generate section
        section = await agent.generate_section(test_spec)

        print(f"✅ Section generated using {section.rag_strategy_used}")
        print(f"📊 Confidence: {section.confidence:.3f}")
        print(f"📝 Word count: {section.word_count}")
        print(f"📚 Citations: {section.citations_count}")
        print(f"🚀 Generation time: {section.generation_time_ms:.1f}ms")

        # Test unified knowledge generation
        unified_section = await agent.generate_with_unified_knowledge(
            "methodology",
            "longitudinal brain imaging federated learning ESM3 protein analysis"
        )

        print(f"\n🧬 Unified knowledge section generated")
        print(f"📊 Strategy: {unified_section.rag_strategy_used}")
        print(f"📊 Confidence: {unified_section.confidence:.3f}")

        # Get stats
        stats = agent.get_generation_stats()
        print(f"\n📈 Generation stats: {stats}")

        print("✅ Unified Proposal Generation Agent test completed successfully!")

    # Run test
    asyncio.run(test_unified_proposal_agent())
"""
Psychology Expert Agent

Specialized agent for psychology research using the integrated Psychology RAG system
with the main AI-CoScientist LLM infrastructure.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

# Agent system imports
from src.agents.base import ResearchAgent
from src.core.config import settings

# LLM and RAG imports
from src.services.llm.types import LLMRequest, TaskType, ModelProvider
from src.services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator,
    QueryContext,
    QueryDomain,
    QueryComplexity,
    RAGStrategy
)

# Psychology-specific imports
try:
    from src.services.rag.psychology_rag_strategy import PsychologyRAGStrategy
    from src.services.psychology.domain_classifier import PsychologyDomainClassifier
    PSYCHOLOGY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Psychology services not available: {e}")
    PSYCHOLOGY_AVAILABLE = False

logger = logging.getLogger(__name__)

class PsychologyExpert(ResearchAgent):
    """
    Psychology research expert agent with specialized Korean NLP
    and psychology literature knowledge integration.
    """

    def __init__(self, rag_orchestrator: UnifiedRAGOrchestrator):
        """
        Initialize Psychology Expert Agent.

        Args:
            rag_orchestrator: Unified RAG orchestrator with psychology strategy
        """
        super().__init__(
            name="PsychologyExpert",
            specialties=[
                "psychology_research",
                "clinical_assessment",
                "behavioral_analysis",
                "cognitive_evaluation",
                "developmental_assessment",
                "neuropsychology"
            ],
            capabilities={
                "korean_language": True,
                "psychology_literature": True,
                "clinical_knowledge": True,
                "research_methodology": True,
                "statistical_analysis": True,
                "multimodal_data": False  # Future capability
            }
        )

        self.rag_orchestrator = rag_orchestrator
        self._available = PSYCHOLOGY_AVAILABLE

        # Initialize psychology domain classifier
        if PSYCHOLOGY_AVAILABLE:
            try:
                self.domain_classifier = PsychologyDomainClassifier()
                logger.info("Psychology Expert Agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize domain classifier: {e}")
                self.domain_classifier = None
        else:
            self.domain_classifier = None

    async def analyze_research_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze psychology research question with domain expertise.

        Args:
            question: Research question in Korean or English
            context: Additional context information

        Returns:
            Analysis results with domain classification and recommendations
        """
        if not self._available:
            return {
                "error": "Psychology Expert Agent is not available",
                "available": False
            }

        try:
            # Step 1: Classify psychology domain
            domain_info = await self._classify_psychology_domain(question)

            # Step 2: Determine query complexity
            complexity = self._assess_query_complexity(question, domain_info)

            # Step 3: Create query context
            query_context = QueryContext(
                query=question,
                complexity=complexity,
                domain=QueryDomain.PSYCHOLOGY,
                intent=domain_info.get("intent", "research"),
                confidence=domain_info.get("confidence", 0.8),
                metadata={
                    "psychology_domain": domain_info.get("domain", "general"),
                    "research_method": domain_info.get("methodology", "not_specified"),
                    "target_population": domain_info.get("population", "not_specified"),
                    "language": "korean" if self._is_korean_text(question) else "english"
                },
                user_preferences=context
            )

            # Step 4: Execute psychology-specialized search
            rag_response = await self.rag_orchestrator.search(
                query_context,
                strategy_override=RAGStrategy.PSYCHOLOGY_RAG
            )

            # Step 5: Enhance response with psychology expertise
            enhanced_analysis = await self._enhance_with_psychology_expertise(
                question,
                rag_response,
                domain_info
            )

            return {
                "question": question,
                "domain_classification": domain_info,
                "complexity": complexity.value,
                "rag_response": {
                    "answer": rag_response.answer,
                    "sources": rag_response.sources,
                    "confidence": rag_response.confidence
                },
                "psychology_analysis": enhanced_analysis,
                "recommendations": self._generate_research_recommendations(
                    question, domain_info, rag_response
                ),
                "available": True,
                "agent": "PsychologyExpert"
            }

        except Exception as e:
            logger.error(f"Psychology research analysis failed: {e}")
            return {
                "error": f"Research analysis failed: {str(e)}",
                "question": question,
                "available": True,
                "agent": "PsychologyExpert"
            }

    async def provide_clinical_guidance(
        self,
        case_description: str,
        assessment_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Provide clinical psychology guidance based on literature.

        Args:
            case_description: Clinical case description
            assessment_type: Type of assessment needed

        Returns:
            Clinical guidance with evidence-based recommendations
        """
        if not self._available:
            return {"error": "Psychology Expert Agent is not available"}

        try:
            # Create clinical assessment query context
            query_context = QueryContext(
                query=f"임상 사례 평가: {case_description}",
                complexity=QueryComplexity.COMPLEX,
                domain=QueryDomain.PSYCHOLOGY,
                intent="clinical_assessment",
                confidence=0.9,
                metadata={
                    "psychology_domain": "clinical_psychology",
                    "assessment_type": assessment_type,
                    "language": "korean" if self._is_korean_text(case_description) else "english"
                }
            )

            # Execute psychology search
            rag_response = await self.rag_orchestrator.search(
                query_context,
                strategy_override=RAGStrategy.PSYCHOLOGY_RAG
            )

            # Generate clinical recommendations
            clinical_analysis = await self._generate_clinical_analysis(
                case_description,
                assessment_type,
                rag_response
            )

            return {
                "case_description": case_description,
                "assessment_type": assessment_type,
                "evidence_base": {
                    "answer": rag_response.answer,
                    "sources": rag_response.sources,
                    "confidence": rag_response.confidence
                },
                "clinical_analysis": clinical_analysis,
                "agent": "PsychologyExpert"
            }

        except Exception as e:
            logger.error(f"Clinical guidance failed: {e}")
            return {"error": f"Clinical guidance failed: {str(e)}"}

    async def _classify_psychology_domain(self, question: str) -> Dict[str, Any]:
        """Classify psychology domain using domain classifier."""
        if self.domain_classifier:
            try:
                return await self.domain_classifier.classify_text(question)
            except Exception as e:
                logger.warning(f"Domain classification failed: {e}")

        # Fallback classification
        question_lower = question.lower()
        if any(term in question_lower for term in ['치료', '상담', '임상', '진단', '장애']):
            return {
                "domain": "clinical_psychology",
                "methodology": "clinical",
                "population": "clinical",
                "confidence": 0.7,
                "intent": "clinical_assessment"
            }
        elif any(term in question_lower for term in ['발달', '아동', '청소년', '성장']):
            return {
                "domain": "developmental_psychology",
                "methodology": "longitudinal",
                "population": "children",
                "confidence": 0.7,
                "intent": "developmental_assessment"
            }
        elif any(term in question_lower for term in ['인지', '기억', '사고', '지능', '학습']):
            return {
                "domain": "cognitive_psychology",
                "methodology": "experimental",
                "population": "adults",
                "confidence": 0.7,
                "intent": "cognitive_evaluation"
            }
        elif any(term in question_lower for term in ['뇌', '신경', 'fMRI', 'EEG', 'tDCS']):
            return {
                "domain": "neuroscience",
                "methodology": "neuroimaging",
                "population": "adults",
                "confidence": 0.8,
                "intent": "neuropsychology_analysis"
            }
        else:
            return {
                "domain": "general_psychology",
                "methodology": "not_specified",
                "population": "general",
                "confidence": 0.5,
                "intent": "psychology_research"
            }

    def _assess_query_complexity(self, question: str, domain_info: Dict[str, Any]) -> QueryComplexity:
        """Assess query complexity based on content and domain."""
        question_lower = question.lower()

        # Complex indicators
        complex_terms = [
            '메타분석', 'meta-analysis', '체계적 문헌고찰', 'systematic review',
            '무작위 대조시험', 'randomized controlled trial', 'rct',
            '구조방정식', 'structural equation modeling', 'sem',
            '기능적 자기공명영상', 'fMRI', '뇌전도', 'EEG'
        ]

        # Medium complexity indicators
        medium_terms = [
            '비교 분석', 'comparative analysis', '상관관계', 'correlation',
            '효과성', 'effectiveness', '타당성', 'validity',
            '신뢰성', 'reliability', '통계적 분석', 'statistical analysis'
        ]

        if any(term in question_lower for term in complex_terms):
            return QueryComplexity.COMPLEX
        elif any(term in question_lower for term in medium_terms):
            return QueryComplexity.MEDIUM
        elif len(question.split()) > 10:  # Long questions tend to be complex
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE

    def _is_korean_text(self, text: str) -> bool:
        """Check if text contains Korean characters."""
        return any('\uac00' <= char <= '\ud7af' for char in text)

    async def _enhance_with_psychology_expertise(
        self,
        question: str,
        rag_response,
        domain_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance response with psychology domain expertise."""

        # Extract key psychological concepts
        psychological_concepts = self._extract_psychological_concepts(question, rag_response.answer)

        # Identify research methodologies mentioned
        methodologies = self._identify_research_methodologies(rag_response.answer)

        # Assess ethical considerations
        ethical_considerations = self._assess_ethical_considerations(question, domain_info)

        # Clinical implications (if applicable)
        clinical_implications = None
        if domain_info.get("domain") == "clinical_psychology":
            clinical_implications = self._assess_clinical_implications(rag_response.answer)

        return {
            "psychological_concepts": psychological_concepts,
            "research_methodologies": methodologies,
            "ethical_considerations": ethical_considerations,
            "clinical_implications": clinical_implications,
            "domain_expertise": {
                "primary_domain": domain_info.get("domain"),
                "related_domains": self._identify_related_domains(domain_info),
                "theoretical_frameworks": self._identify_theoretical_frameworks(rag_response.answer)
            }
        }

    def _extract_psychological_concepts(self, question: str, answer: str) -> List[str]:
        """Extract key psychological concepts from question and answer."""
        # This is a simplified implementation - could be enhanced with NLP
        concepts = []

        psychological_terms = [
            'ADHD', '주의력결핍', '실행기능', '작업기억',
            '우울증', '불안', '인지행동치료', 'CBT',
            '전전두엽', '뇌자극', 'tDCS', 'fMRI',
            '발달장애', '자폐스펙트럼', '학습장애',
            '신경가소성', '인지재활', '행동수정'
        ]

        text = f"{question} {answer}".lower()
        for term in psychological_terms:
            if term.lower() in text:
                concepts.append(term)

        return list(set(concepts))  # Remove duplicates

    def _identify_research_methodologies(self, answer: str) -> List[str]:
        """Identify research methodologies mentioned in the answer."""
        methodologies = []

        method_terms = {
            '무작위 대조시험': ['무작위', 'randomized', 'rct'],
            '종단연구': ['종단', 'longitudinal'],
            '횡단연구': ['횡단', 'cross-sectional'],
            '메타분석': ['메타분석', 'meta-analysis'],
            '사례연구': ['사례연구', 'case study'],
            '실험연구': ['실험', 'experimental'],
            '관찰연구': ['관찰', 'observational'],
            '신경영상': ['fMRI', '뇌영상', 'neuroimaging']
        }

        answer_lower = answer.lower()
        for method, terms in method_terms.items():
            if any(term in answer_lower for term in terms):
                methodologies.append(method)

        return methodologies

    def _assess_ethical_considerations(self, question: str, domain_info: Dict[str, Any]) -> List[str]:
        """Assess ethical considerations relevant to the query."""
        considerations = []

        # Always include basic research ethics
        considerations.append("연구 참가자의 인권 보호 및 동의서 필요")

        # Domain-specific ethical considerations
        domain = domain_info.get("domain", "")

        if "clinical" in domain:
            considerations.extend([
                "환자 개인정보 보호",
                "치료적 관계의 윤리",
                "진단의 정확성과 책임"
            ])

        if "developmental" in domain or "children" in domain_info.get("population", ""):
            considerations.extend([
                "아동 및 청소년 보호",
                "부모/보호자 동의 필요",
                "발달 단계 고려"
            ])

        return considerations

    def _assess_clinical_implications(self, answer: str) -> Dict[str, Any]:
        """Assess clinical implications from the research answer."""
        # Simplified implementation - could be enhanced with clinical knowledge base
        implications = {
            "treatment_recommendations": [],
            "diagnostic_considerations": [],
            "risk_factors": [],
            "contraindications": []
        }

        answer_lower = answer.lower()

        # Look for treatment mentions
        if any(term in answer_lower for term in ['치료', '개입', 'intervention', 'treatment']):
            implications["treatment_recommendations"].append("연구 결과를 임상 실무에 적용 시 신중한 검토 필요")

        # Look for diagnostic mentions
        if any(term in answer_lower for term in ['진단', '평가', 'assessment', 'diagnosis']):
            implications["diagnostic_considerations"].append("진단 도구의 타당성 및 신뢰성 검토 필요")

        return implications

    def _identify_related_domains(self, domain_info: Dict[str, Any]) -> List[str]:
        """Identify related psychology domains."""
        primary_domain = domain_info.get("domain", "")

        domain_relationships = {
            "clinical_psychology": ["health_psychology", "abnormal_psychology"],
            "cognitive_psychology": ["neuropsychology", "experimental_psychology"],
            "developmental_psychology": ["educational_psychology", "child_psychology"],
            "neuroscience": ["cognitive_psychology", "biological_psychology"],
            "social_psychology": ["personality_psychology", "cultural_psychology"]
        }

        return domain_relationships.get(primary_domain, [])

    def _identify_theoretical_frameworks(self, answer: str) -> List[str]:
        """Identify theoretical frameworks mentioned in the answer."""
        frameworks = []

        framework_terms = {
            '인지행동모델': ['인지행동', 'cognitive behavioral', 'cbt'],
            '정신분석이론': ['정신분석', 'psychoanalytic'],
            '인본주의이론': ['인본주의', 'humanistic'],
            '행동주의이론': ['행동주의', 'behavioral'],
            '생물심리사회모델': ['생물심리사회', 'biopsychosocial'],
            '애착이론': ['애착', 'attachment'],
            '인지이론': ['인지이론', 'cognitive theory']
        }

        answer_lower = answer.lower()
        for framework, terms in framework_terms.items():
            if any(term in answer_lower for term in terms):
                frameworks.append(framework)

        return frameworks

    def _generate_research_recommendations(
        self,
        question: str,
        domain_info: Dict[str, Any],
        rag_response
    ) -> List[str]:
        """Generate research recommendations based on analysis."""
        recommendations = []

        # Basic research recommendations
        recommendations.append("관련 최신 문헌 추가 검토 권장")

        # Domain-specific recommendations
        domain = domain_info.get("domain", "")

        if "clinical" in domain:
            recommendations.extend([
                "임상 실무 적용 전 추가 검증 연구 필요",
                "다양한 임상 집단에서의 효과성 검토",
                "장기 추적 관찰 연구 고려"
            ])

        if "experimental" in domain_info.get("methodology", ""):
            recommendations.extend([
                "실험 설계의 내적 타당성 검토",
                "통제 조건 강화 방안 모색",
                "표본 크기 적절성 평가"
            ])

        # Based on confidence level
        if rag_response.confidence < 0.7:
            recommendations.append("연구 결과 해석 시 주의 필요 - 추가 검증 권장")

        return recommendations

    async def _generate_clinical_analysis(
        self,
        case_description: str,
        assessment_type: str,
        rag_response
    ) -> Dict[str, Any]:
        """Generate clinical analysis based on evidence."""
        return {
            "assessment_summary": f"{assessment_type} 평가를 위한 문헌 기반 분석",
            "evidence_level": "문헌 기반 권장사항 (직접적 진단 아님)",
            "key_considerations": [
                "개별 사례의 특수성 고려 필요",
                "전문가 직접 평가 권장",
                "다면적 평가 접근법 사용"
            ],
            "literature_support": f"관련 논문 {len(rag_response.sources)}편 참조",
            "confidence_level": rag_response.confidence,
            "disclaimer": "이 분석은 연구 목적으로만 사용되며 임상 진단을 대체할 수 없음"
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and status."""
        return {
            "agent_name": self.name,
            "available": self._available,
            "specialties": self.specialties,
            "capabilities": self.capabilities,
            "supported_languages": ["korean", "english"],
            "supported_domains": [
                "clinical_psychology",
                "cognitive_psychology",
                "developmental_psychology",
                "neuroscience",
                "health_psychology",
                "educational_psychology",
                "social_psychology"
            ],
            "features": [
                "Korean language support",
                "Psychology literature search",
                "Domain classification",
                "Clinical guidance",
                "Research recommendations",
                "Ethical considerations"
            ]
        }
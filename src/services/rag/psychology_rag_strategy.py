"""
Psychology RAG Strategy

Integration of the Psychology Vector Store with the main AI-CoScientist LLM system.
Provides specialized psychology research capabilities with Korean NLP support.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.services.rag.unified_rag_orchestrator import (
    RAGStrategyInterface,
    QueryContext,
    RAGResponse,
    RAGStrategy,
    QueryDomain,
    QueryComplexity
)
from src.monitoring.rag_metrics import RAGMetrics

# Psychology-specific imports
try:
    from src.services.psychology.psychology_vector_store import PsychologyVectorStore
    from src.services.psychology.korean_nlp_processor import KoreanNLPPipeline
    from src.services.psychology.domain_classifier import PsychologyDomainClassifier
    from src.services.psychology.query_enhancer import PsychologyQueryEnhancer
    PSYCHOLOGY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Psychology services not available: {e}")
    PSYCHOLOGY_AVAILABLE = False

# LLM service imports
from src.services.llm.interface import LLMServiceInterface
from src.services.llm.types import LLMRequest, TaskType, ModelProvider

logger = logging.getLogger(__name__)

class PsychologyRAGStrategy(RAGStrategyInterface):
    """
    Psychology-specialized RAG strategy integrating Korean NLP and domain expertise
    with the main AI-CoScientist LLM infrastructure.
    """

    def __init__(self, llm_service: LLMServiceInterface):
        """
        Initialize Psychology RAG Strategy.

        Args:
            llm_service: LLM service adapter (Anthropic, OpenAI, etc.)
        """
        self.llm_service = llm_service
        self._available = PSYCHOLOGY_AVAILABLE

        # Initialize psychology components
        if PSYCHOLOGY_AVAILABLE:
            try:
                self.vector_store = PsychologyVectorStore()
                self.korean_nlp = KoreanNLPPipeline()
                self.domain_classifier = PsychologyDomainClassifier()
                self.query_enhancer = PsychologyQueryEnhancer()
                self._available = True
                logger.info("Psychology RAG Strategy initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Psychology components: {e}")
                self._available = False
        else:
            self.vector_store = None
            self.korean_nlp = None
            self.domain_classifier = None
            self.query_enhancer = None

    async def search(self, query_context: QueryContext) -> RAGResponse:
        """
        Execute psychology-specialized search with LLM generation.

        Args:
            query_context: Query context with complexity, domain, intent info

        Returns:
            RAGResponse with psychology-specific results
        """
        start_time = time.time()

        if not self._available:
            raise RuntimeError("Psychology RAG Strategy is not available")

        try:
            # Step 1: Korean NLP Analysis
            nlp_result = await self._analyze_query(query_context.query)

            # Step 2: Query Enhancement
            enhanced_query = await self._enhance_query(query_context.query)

            # Step 3: Vector Search
            search_results = await self._vector_search(enhanced_query, limit=5)

            # Step 4: Context Preparation
            context = self._prepare_context(search_results, nlp_result)

            # Step 5: LLM Generation with Psychology Prompt
            llm_response = await self._generate_response(
                query_context,
                context,
                nlp_result
            )

            # Step 6: Response Assembly
            response = await self._assemble_response(
                llm_response,
                search_results,
                query_context,
                start_time
            )

            return response

        except Exception as e:
            logger.error(f"Psychology RAG search failed: {e}")
            # Fallback response
            return RAGResponse(
                answer=f"죄송합니다. 심리학 연구 검색 중 오류가 발생했습니다: {str(e)}",
                sources=[],
                confidence=0.0,
                strategy_used=RAGStrategy.PSYCHOLOGY_RAG,
                metadata={"error": str(e)}
            )

    async def _analyze_query(self, query: str) -> Optional[Any]:
        """Analyze query with Korean NLP pipeline."""
        if self.korean_nlp:
            try:
                return await self.korean_nlp.analyze_text(query)
            except Exception as e:
                logger.warning(f"Korean NLP analysis failed: {e}")
                return None
        return None

    async def _enhance_query(self, query: str) -> str:
        """Enhance query with psychology-specific terms."""
        if self.query_enhancer:
            try:
                result = await self.query_enhancer.enhance_query(query)
                return result.enhanced_query
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")
                return query
        return query

    async def _vector_search(self, query: str, limit: int = 5) -> List[Any]:
        """Search psychology papers vector store."""
        if self.vector_store:
            try:
                return await self.vector_store.search_papers(query, limit=limit)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                return []
        return []

    def _prepare_context(self, search_results: List[Any], nlp_result: Any) -> str:
        """Prepare context for LLM from search results."""
        if not search_results:
            return "관련 심리학 연구 논문을 찾을 수 없습니다."

        context_parts = ["다음은 관련 심리학 연구 논문들입니다:\n"]

        for i, result in enumerate(search_results, 1):
            try:
                context_parts.append(f"{i}. 논문 제목: {result.title}")
                context_parts.append(f"   저자: {', '.join(result.metadata.authors)}")
                context_parts.append(f"   연도: {result.metadata.year}")
                context_parts.append(f"   내용: {result.content[:300]}...")
                context_parts.append(f"   유사도: {result.similarity_score:.3f}\n")
            except Exception as e:
                logger.warning(f"Error formatting search result {i}: {e}")
                continue

        return "\n".join(context_parts)

    async def _generate_response(
        self,
        query_context: QueryContext,
        context: str,
        nlp_result: Any
    ) -> str:
        """Generate response using LLM with psychology-specific prompt."""

        # Determine task type based on query intent
        task_type = self._determine_task_type(query_context, nlp_result)

        # Build psychology-specialized prompt
        system_prompt = self._build_system_prompt(query_context, nlp_result)
        user_prompt = self._build_user_prompt(query_context.query, context)

        # Create LLM request
        llm_request = LLMRequest(
            prompt=user_prompt,
            task_type=task_type,
            system_message=system_prompt
        )

        try:
            # Generate response using the integrated LLM service
            response = await self.llm_service.complete(llm_request)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

    def _determine_task_type(self, query_context: QueryContext, nlp_result: Any) -> TaskType:
        """Determine appropriate task type for psychology queries."""
        query_lower = query_context.query.lower()

        # Psychology-specific task type mapping
        if any(term in query_lower for term in ['치료', '상담', '임상', '진단']):
            return TaskType.CLINICAL_ASSESSMENT
        elif any(term in query_lower for term in ['행동', '관찰', '분석']):
            return TaskType.BEHAVIORAL_ANALYSIS
        elif any(term in query_lower for term in ['인지', '기억', '사고', '지능']):
            return TaskType.COGNITIVE_EVALUATION
        elif any(term in query_lower for term in ['발달', '아동', '청소년']):
            return TaskType.DEVELOPMENTAL_ASSESSMENT
        elif any(term in query_lower for term in ['뇌', '신경', 'fMRI', 'EEG']):
            return TaskType.NEUROPSYCHOLOGY_ANALYSIS
        else:
            return TaskType.PSYCHOLOGY_RESEARCH

    def _build_system_prompt(self, query_context: QueryContext, nlp_result: Any) -> str:
        """Build psychology-specialized system prompt."""
        base_prompt = """당신은 서울대학교 심리학과의 전문 AI 연구 어시스턴트입니다.

심리학 연구 논문을 바탕으로 정확하고 전문적인 답변을 제공해야 합니다.
다음 원칙을 따르세요:

1. 과학적 근거: 제공된 논문 데이터를 기반으로 답변
2. 정확성: 불확실한 내용은 명시적으로 표시
3. 전문성: 심리학 전문 용어를 적절히 사용
4. 한국어 지원: 한국어와 영어 용어를 병행 설명
5. 윤리적 고려: 개인정보 보호 및 연구 윤리 준수"""

        # Add domain-specific guidance
        if query_context.domain == QueryDomain.PSYCHOLOGY:
            if nlp_result and hasattr(nlp_result, 'psychology_terms'):
                terms = [term.korean for term in nlp_result.psychology_terms[:3]]
                if terms:
                    base_prompt += f"\n\n주요 심리학 용어: {', '.join(terms)}"

        return base_prompt

    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build user prompt with query and context."""
        return f"""질문: {query}

참고 자료:
{context}

위 심리학 연구 논문들을 참고하여 질문에 대한 전문적인 답변을 제공해주세요.
답변에는 다음을 포함해야 합니다:
- 주요 연구 결과 요약
- 관련 논문 인용
- 실무 적용 방안 (해당되는 경우)
- 추가 연구 방향 제안

답변:"""

    async def _assemble_response(
        self,
        llm_response: str,
        search_results: List[Any],
        query_context: QueryContext,
        start_time: float
    ) -> RAGResponse:
        """Assemble final RAG response."""

        # Calculate performance metrics
        latency_ms = (time.time() - start_time) * 1000

        # Prepare sources
        sources = []
        for result in search_results:
            try:
                sources.append({
                    "title": result.title,
                    "authors": result.metadata.authors,
                    "year": result.metadata.year,
                    "similarity": result.similarity_score,
                    "url": getattr(result, 'url', None)
                })
            except Exception as e:
                logger.warning(f"Error preparing source: {e}")
                continue

        # Calculate confidence based on search results quality
        confidence = self._calculate_confidence(search_results, query_context)

        # Create performance metrics
        performance_metrics = RAGMetrics(
            latency=latency_ms / 1000,
            quality_score=confidence,
            tokens_processed=len(llm_response.split()),
            retrieval_time=0.1,  # Estimated
            generation_time=latency_ms / 1000 - 0.1,
            context_relevance=confidence,
            faithfulness=0.85,  # Estimated
            answer_relevancy=confidence,
            strategy="psychology_rag"
        )

        return RAGResponse(
            answer=llm_response,
            sources=sources,
            confidence=confidence,
            strategy_used=RAGStrategy.PSYCHOLOGY_RAG,
            performance_metrics=performance_metrics,
            metadata={
                "query_domain": query_context.domain.value,
                "query_complexity": query_context.complexity.value,
                "source_count": len(sources),
                "latency_ms": latency_ms
            }
        )

    def _calculate_confidence(
        self,
        search_results: List[Any],
        query_context: QueryContext
    ) -> float:
        """Calculate response confidence based on search quality."""
        if not search_results:
            return 0.1

        # Base confidence on similarity scores
        avg_similarity = sum(r.similarity_score for r in search_results) / len(search_results)

        # Adjust based on query complexity
        complexity_factor = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MEDIUM: 0.9,
            QueryComplexity.COMPLEX: 0.8
        }.get(query_context.complexity, 0.8)

        return min(avg_similarity * complexity_factor, 0.95)

    def is_available(self) -> bool:
        """Check if Psychology RAG strategy is available."""
        return self._available

    def get_strategy_name(self) -> RAGStrategy:
        """Get strategy identifier."""
        return RAGStrategy.PSYCHOLOGY_RAG

    def estimate_performance(self, query_context: QueryContext) -> float:
        """Estimate performance score for psychology queries."""
        if query_context.domain == QueryDomain.PSYCHOLOGY:
            return 0.95  # High performance for psychology domain
        elif any(term in query_context.query.lower() for term in
                ['심리', '인지', '행동', '뇌', '신경', '발달', '치료']):
            return 0.85  # Good performance for psychology-related queries
        else:
            return 0.3   # Lower performance for non-psychology queries
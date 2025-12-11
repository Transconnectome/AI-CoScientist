"""
Context Sufficiency Checker for RAG Systems

Implementation for: Context sufficiency validation and quality assessment
Created: 2025-12-05

Acceptance Criteria:
- Context quality scoring algorithms
- Sufficiency threshold determination
- Missing information detection
- Context enrichment recommendations

This module provides intelligent assessment of whether retrieved context
is sufficient to answer queries accurately, with recommendations for
context enrichment and quality improvement.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import json

# External dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core dependencies
from ..knowledge_base.vector_store import VectorStore
from ..rag.unified_rag_orchestrator import QueryContext, RAGResponse

logger = logging.getLogger(__name__)

class SufficiencyLevel(Enum):
    """Context sufficiency levels"""
    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    SUFFICIENT = "sufficient"
    COMPREHENSIVE = "comprehensive"

class QualityDimension(Enum):
    """Context quality dimensions"""
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    SPECIFICITY = "specificity"
    COHERENCE = "coherence"
    RECENCY = "recency"

@dataclass
class ContextQuality:
    """Context quality assessment"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    sufficiency_level: SufficiencyLevel
    confidence: float
    missing_aspects: List[str]
    improvement_suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextDocument:
    """Individual context document"""
    content: str
    source: str
    relevance_score: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnrichmentRecommendation:
    """Context enrichment recommendation"""
    type: str  # "additional_search", "domain_specific", "temporal", "methodological"
    description: str
    keywords: List[str]
    priority: float
    estimated_improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityScorer(ABC):
    """Abstract base class for quality scoring algorithms"""

    @abstractmethod
    async def score_relevance(self, query: str, context: str) -> float:
        """Score context relevance to query"""
        pass

    @abstractmethod
    async def score_completeness(self, query: str, context: str) -> float:
        """Score context completeness for query"""
        pass

    @abstractmethod
    async def score_accuracy(self, context: str) -> float:
        """Score context accuracy and reliability"""
        pass

class SemanticQualityScorer(QualityScorer):
    """Semantic similarity-based quality scorer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize scoring models"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized SentenceTransformer: {self.model_name}")

            if SKLEARN_AVAILABLE:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                logger.info("Initialized TF-IDF vectorizer")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic models: {e}")

    async def score_relevance(self, query: str, context: str) -> float:
        """Score semantic relevance using embeddings or TF-IDF"""
        try:
            if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
                return await self._semantic_relevance(query, context)
            elif self.tfidf_vectorizer and SKLEARN_AVAILABLE:
                return await self._tfidf_relevance(query, context)
            else:
                return await self._keyword_relevance(query, context)
        except Exception as e:
            logger.error(f"Error scoring relevance: {e}")
            return await self._keyword_relevance(query, context)

    async def _semantic_relevance(self, query: str, context: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            query_embedding = self.model.encode([query])
            context_embedding = self.model.encode([context])

            similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Semantic relevance error: {e}")
            return 0.5

    async def _tfidf_relevance(self, query: str, context: str) -> float:
        """Calculate TF-IDF based similarity"""
        try:
            # Fit on both texts and transform
            corpus = [query, context]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"TF-IDF relevance error: {e}")
            return 0.5

    async def _keyword_relevance(self, query: str, context: str) -> float:
        """Simple keyword-based relevance scoring"""
        try:
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())

            if not query_words:
                return 0.0

            overlap = len(query_words.intersection(context_words))
            return overlap / len(query_words)
        except Exception as e:
            logger.error(f"Keyword relevance error: {e}")
            return 0.0

    async def score_completeness(self, query: str, context: str) -> float:
        """Score completeness based on query coverage"""
        try:
            # Extract key concepts from query
            query_concepts = await self._extract_concepts(query)

            # Check coverage in context
            coverage_score = 0.0
            for concept in query_concepts:
                if concept.lower() in context.lower():
                    coverage_score += 1.0

            if query_concepts:
                coverage_score /= len(query_concepts)

            # Adjust for context length and detail
            context_length_factor = min(1.0, len(context) / 1000)  # Normalize around 1000 chars

            return (coverage_score * 0.7) + (context_length_factor * 0.3)
        except Exception as e:
            logger.error(f"Error scoring completeness: {e}")
            return 0.5

    async def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Simple concept extraction (could be enhanced with NER)
        words = query.split()
        # Filter out common words, keep important terms
        stopwords = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an'}
        concepts = [word.strip('?.,!') for word in words
                   if word.lower() not in stopwords and len(word) > 2]
        return concepts

    async def score_accuracy(self, context: str) -> float:
        """Score accuracy based on various heuristics"""
        try:
            score = 0.0

            # Check for citations/references (higher accuracy)
            if any(marker in context for marker in ['[', ']', 'et al.', 'doi:', 'http']):
                score += 0.3

            # Check for specific numbers/data (more concrete)
            import re
            if re.search(r'\d+\.?\d*%|\d+\.?\d*\s*(seconds?|minutes?|hours?|days?|years?)', context):
                score += 0.2

            # Check for hedging language (appropriate uncertainty)
            hedge_words = ['may', 'might', 'could', 'possibly', 'likely', 'suggests']
            if any(word in context.lower() for word in hedge_words):
                score += 0.2

            # Check for technical terminology (domain expertise)
            if len([word for word in context.split() if len(word) > 8]) > 3:
                score += 0.2

            # Base score for any coherent text
            score += 0.1

            return min(1.0, score)
        except Exception as e:
            logger.error(f"Error scoring accuracy: {e}")
            return 0.5

class RuleBasedQualityScorer(QualityScorer):
    """Rule-based quality scorer for fallback"""

    async def score_relevance(self, query: str, context: str) -> float:
        """Rule-based relevance scoring"""
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(context_terms))
        return overlap / len(query_terms)

    async def score_completeness(self, query: str, context: str) -> float:
        """Rule-based completeness scoring"""
        # Simple heuristic based on context length and query coverage
        min_length = max(100, len(query) * 3)  # Expect at least 3x query length
        length_score = min(1.0, len(context) / min_length)

        relevance_score = await self.score_relevance(query, context)

        return (length_score * 0.4) + (relevance_score * 0.6)

    async def score_accuracy(self, context: str) -> float:
        """Rule-based accuracy scoring"""
        # Simple heuristics
        score = 0.5  # Base score

        # Prefer longer, more detailed context
        if len(context) > 500:
            score += 0.1

        # Check for structured content
        if any(marker in context for marker in ['\n', '•', '-', '1.', '2.']):
            score += 0.1

        # Check for citations or references
        if any(ref in context for ref in ['[', ']', 'et al.', 'study']):
            score += 0.2

        return min(1.0, score)

class ContextSufficiencyChecker:
    """Main context sufficiency checker"""

    def __init__(
        self,
        quality_scorer: Optional[QualityScorer] = None,
        sufficiency_thresholds: Optional[Dict[SufficiencyLevel, float]] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.quality_scorer = quality_scorer or self._create_default_scorer()
        self.sufficiency_thresholds = sufficiency_thresholds or {
            SufficiencyLevel.INSUFFICIENT: 0.3,
            SufficiencyLevel.PARTIAL: 0.5,
            SufficiencyLevel.SUFFICIENT: 0.7,
            SufficiencyLevel.COMPREHENSIVE: 0.85
        }
        self.vector_store = vector_store

        # Quality assessment cache
        self._quality_cache: Dict[str, ContextQuality] = {}

        # Performance tracking
        self.assessment_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _create_default_scorer(self) -> QualityScorer:
        """Create default quality scorer"""
        if SENTENCE_TRANSFORMERS_AVAILABLE or SKLEARN_AVAILABLE:
            return SemanticQualityScorer()
        else:
            logger.info("Using rule-based quality scorer (semantic models unavailable)")
            return RuleBasedQualityScorer()

    async def assess_context_quality(
        self,
        query: str,
        context_documents: List[ContextDocument],
        query_context: Optional[QueryContext] = None
    ) -> ContextQuality:
        """Assess overall context quality and sufficiency"""
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._generate_cache_key(query, context_documents)
            if cache_key in self._quality_cache:
                self.cache_hits += 1
                return self._quality_cache[cache_key]

            self.cache_misses += 1

            # Combine all context
            combined_context = " ".join([doc.content for doc in context_documents])

            # Score each quality dimension
            dimension_scores = {}

            # Relevance scoring
            dimension_scores[QualityDimension.RELEVANCE] = await self.quality_scorer.score_relevance(
                query, combined_context
            )

            # Completeness scoring
            dimension_scores[QualityDimension.COMPLETENESS] = await self.quality_scorer.score_completeness(
                query, combined_context
            )

            # Accuracy scoring
            dimension_scores[QualityDimension.ACCURACY] = await self.quality_scorer.score_accuracy(
                combined_context
            )

            # Specificity scoring
            dimension_scores[QualityDimension.SPECIFICITY] = await self._score_specificity(
                query, combined_context
            )

            # Coherence scoring
            dimension_scores[QualityDimension.COHERENCE] = await self._score_coherence(
                context_documents
            )

            # Recency scoring
            dimension_scores[QualityDimension.RECENCY] = await self._score_recency(
                context_documents
            )

            # Calculate overall score (weighted average)
            weights = {
                QualityDimension.RELEVANCE: 0.25,
                QualityDimension.COMPLETENESS: 0.20,
                QualityDimension.ACCURACY: 0.20,
                QualityDimension.SPECIFICITY: 0.15,
                QualityDimension.COHERENCE: 0.10,
                QualityDimension.RECENCY: 0.10
            }

            overall_score = sum(
                score * weights[dimension]
                for dimension, score in dimension_scores.items()
            )

            # Determine sufficiency level
            sufficiency_level = self._determine_sufficiency_level(overall_score)

            # Calculate confidence based on score consistency
            confidence = self._calculate_confidence(dimension_scores)

            # Identify missing aspects and improvement suggestions
            missing_aspects = await self._identify_missing_aspects(
                query, context_documents, query_context
            )

            improvement_suggestions = await self._generate_improvement_suggestions(
                query, context_documents, dimension_scores, missing_aspects
            )

            # Create quality assessment
            quality = ContextQuality(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                sufficiency_level=sufficiency_level,
                confidence=confidence,
                missing_aspects=missing_aspects,
                improvement_suggestions=improvement_suggestions,
                metadata={
                    "num_documents": len(context_documents),
                    "total_length": len(combined_context),
                    "assessment_time": time.time() - start_time
                }
            )

            # Cache result
            self._quality_cache[cache_key] = quality
            self.assessment_times.append(time.time() - start_time)

            return quality

        except Exception as e:
            logger.error(f"Error assessing context quality: {e}")
            # Return default assessment
            return ContextQuality(
                overall_score=0.5,
                dimension_scores={dim: 0.5 for dim in QualityDimension},
                sufficiency_level=SufficiencyLevel.PARTIAL,
                confidence=0.0,
                missing_aspects=["Unable to assess - error occurred"],
                improvement_suggestions=["Retry assessment with different context"]
            )

    async def _score_specificity(self, query: str, context: str) -> float:
        """Score how specific the context is to the query"""
        try:
            # Check for specific terms, numbers, examples
            specificity_indicators = [
                r'\d+\.?\d*%',  # Percentages
                r'\d+\.?\d*\s*(mg|kg|ml|cm|mm)',  # Measurements
                r'\b(study|research|experiment|analysis)\b',  # Research terms
                r'\b(Figure|Table|Chart)\s+\d+',  # References to figures/tables
                r'\b\d{4}\b',  # Years
            ]

            import re
            specificity_score = 0.0
            for pattern in specificity_indicators:
                matches = len(re.findall(pattern, context, re.IGNORECASE))
                specificity_score += min(0.2, matches * 0.05)  # Cap contribution

            return min(1.0, specificity_score)
        except Exception as e:
            logger.error(f"Error scoring specificity: {e}")
            return 0.5

    async def _score_coherence(self, context_documents: List[ContextDocument]) -> float:
        """Score coherence across context documents"""
        try:
            if len(context_documents) <= 1:
                return 1.0  # Single document is coherent by definition

            # Simple coherence check based on vocabulary overlap
            all_words = set()
            doc_words = []

            for doc in context_documents:
                words = set(doc.content.lower().split())
                doc_words.append(words)
                all_words.update(words)

            # Calculate pairwise overlap
            total_overlap = 0.0
            pairs = 0

            for i in range(len(doc_words)):
                for j in range(i + 1, len(doc_words)):
                    overlap = len(doc_words[i].intersection(doc_words[j]))
                    union = len(doc_words[i].union(doc_words[j]))
                    if union > 0:
                        total_overlap += overlap / union
                    pairs += 1

            if pairs == 0:
                return 1.0

            return total_overlap / pairs
        except Exception as e:
            logger.error(f"Error scoring coherence: {e}")
            return 0.5

    async def _score_recency(self, context_documents: List[ContextDocument]) -> float:
        """Score recency of context documents"""
        try:
            # Simple heuristic based on year mentions
            current_year = 2025
            recent_years = 0
            total_years = 0

            import re
            for doc in context_documents:
                years = re.findall(r'\b(20\d{2})\b', doc.content)
                for year_str in years:
                    year = int(year_str)
                    if year >= current_year - 5:  # Last 5 years considered recent
                        recent_years += 1
                    total_years += 1

            if total_years == 0:
                return 0.5  # No temporal information

            return recent_years / total_years
        except Exception as e:
            logger.error(f"Error scoring recency: {e}")
            return 0.5

    def _determine_sufficiency_level(self, overall_score: float) -> SufficiencyLevel:
        """Determine sufficiency level based on overall score"""
        if overall_score >= self.sufficiency_thresholds[SufficiencyLevel.COMPREHENSIVE]:
            return SufficiencyLevel.COMPREHENSIVE
        elif overall_score >= self.sufficiency_thresholds[SufficiencyLevel.SUFFICIENT]:
            return SufficiencyLevel.SUFFICIENT
        elif overall_score >= self.sufficiency_thresholds[SufficiencyLevel.PARTIAL]:
            return SufficiencyLevel.PARTIAL
        else:
            return SufficiencyLevel.INSUFFICIENT

    def _calculate_confidence(self, dimension_scores: Dict[QualityDimension, float]) -> float:
        """Calculate confidence based on score consistency"""
        try:
            scores = list(dimension_scores.values())
            if not scores:
                return 0.0

            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

            # Lower variance = higher confidence
            confidence = 1.0 - min(1.0, variance)
            return confidence
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    async def _identify_missing_aspects(
        self,
        query: str,
        context_documents: List[ContextDocument],
        query_context: Optional[QueryContext]
    ) -> List[str]:
        """Identify missing aspects in context"""
        missing_aspects = []

        try:
            combined_context = " ".join([doc.content for doc in context_documents])

            # Check for common missing elements based on query type
            query_lower = query.lower()

            if any(word in query_lower for word in ['how', 'mechanism', 'process']):
                if 'step' not in combined_context.lower() and 'process' not in combined_context.lower():
                    missing_aspects.append("Step-by-step process explanation")

            if any(word in query_lower for word in ['why', 'cause', 'reason']):
                if 'because' not in combined_context.lower() and 'due to' not in combined_context.lower():
                    missing_aspects.append("Causal explanations")

            if any(word in query_lower for word in ['compare', 'difference', 'versus']):
                if 'however' not in combined_context.lower() and 'while' not in combined_context.lower():
                    missing_aspects.append("Comparative analysis")

            if any(word in query_lower for word in ['example', 'instance', 'case']):
                if 'for example' not in combined_context.lower() and 'such as' not in combined_context.lower():
                    missing_aspects.append("Concrete examples")

            # Domain-specific missing aspects
            if query_context and query_context.domain:
                domain_checks = {
                    "neuroscience": ["brain", "neural", "neuron"],
                    "quantum_ml": ["quantum", "qubit", "entanglement"],
                    "developmental_disorders": ["development", "disorder", "symptoms"]
                }

                domain_name = query_context.domain.value if hasattr(query_context.domain, 'value') else str(query_context.domain)
                if domain_name in domain_checks:
                    domain_terms = domain_checks[domain_name]
                    if not any(term in combined_context.lower() for term in domain_terms):
                        missing_aspects.append(f"Domain-specific {domain_name} terminology")

        except Exception as e:
            logger.error(f"Error identifying missing aspects: {e}")
            missing_aspects.append("Unable to identify missing aspects")

        return missing_aspects

    async def _generate_improvement_suggestions(
        self,
        query: str,
        context_documents: List[ContextDocument],
        dimension_scores: Dict[QualityDimension, float],
        missing_aspects: List[str]
    ) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []

        try:
            # Suggestions based on low dimension scores
            if dimension_scores.get(QualityDimension.RELEVANCE, 0) < 0.6:
                suggestions.append("Search for more directly relevant documents")

            if dimension_scores.get(QualityDimension.COMPLETENESS, 0) < 0.6:
                suggestions.append("Expand search to include more comprehensive sources")

            if dimension_scores.get(QualityDimension.ACCURACY, 0) < 0.6:
                suggestions.append("Include more authoritative sources with citations")

            if dimension_scores.get(QualityDimension.SPECIFICITY, 0) < 0.6:
                suggestions.append("Search for more specific and detailed information")

            if dimension_scores.get(QualityDimension.RECENCY, 0) < 0.6:
                suggestions.append("Include more recent publications and findings")

            # Suggestions based on missing aspects
            if "Step-by-step process explanation" in missing_aspects:
                suggestions.append("Add methodological or procedural documentation")

            if "Causal explanations" in missing_aspects:
                suggestions.append("Include sources explaining underlying mechanisms")

            if "Comparative analysis" in missing_aspects:
                suggestions.append("Add comparative studies or analysis documents")

            if "Concrete examples" in missing_aspects:
                suggestions.append("Include case studies and practical examples")

            # General suggestions based on context characteristics
            if len(context_documents) < 3:
                suggestions.append("Increase number of source documents")

            total_length = sum(len(doc.content) for doc in context_documents)
            if total_length < 500:
                suggestions.append("Include longer, more detailed sources")

        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            suggestions.append("Unable to generate specific suggestions")

        return suggestions[:5]  # Limit to top 5 suggestions

    async def get_enrichment_recommendations(
        self,
        query: str,
        current_context: List[ContextDocument],
        quality_assessment: ContextQuality
    ) -> List[EnrichmentRecommendation]:
        """Generate context enrichment recommendations"""
        recommendations = []

        try:
            # Analyze gaps and generate targeted recommendations
            if quality_assessment.sufficiency_level in [SufficiencyLevel.INSUFFICIENT, SufficiencyLevel.PARTIAL]:

                # Recommendation for additional search terms
                if quality_assessment.dimension_scores.get(QualityDimension.RELEVANCE, 0) < 0.6:
                    recommendations.append(EnrichmentRecommendation(
                        type="additional_search",
                        description="Expand search with related keywords",
                        keywords=await self._extract_expansion_keywords(query, current_context),
                        priority=0.8,
                        estimated_improvement=0.3
                    ))

                # Domain-specific recommendations
                domain_keywords = await self._get_domain_keywords(query)
                if domain_keywords:
                    recommendations.append(EnrichmentRecommendation(
                        type="domain_specific",
                        description="Include domain-specific sources",
                        keywords=domain_keywords,
                        priority=0.7,
                        estimated_improvement=0.25
                    ))

                # Temporal recommendations
                if quality_assessment.dimension_scores.get(QualityDimension.RECENCY, 0) < 0.5:
                    recommendations.append(EnrichmentRecommendation(
                        type="temporal",
                        description="Include more recent publications",
                        keywords=["2023", "2024", "2025", "recent", "latest"],
                        priority=0.6,
                        estimated_improvement=0.2
                    ))

                # Methodological recommendations
                if "methodological" in " ".join(quality_assessment.missing_aspects).lower():
                    recommendations.append(EnrichmentRecommendation(
                        type="methodological",
                        description="Add methodological documentation",
                        keywords=["method", "methodology", "approach", "technique", "protocol"],
                        priority=0.65,
                        estimated_improvement=0.22
                    ))

        except Exception as e:
            logger.error(f"Error generating enrichment recommendations: {e}")

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        return recommendations[:3]  # Return top 3 recommendations

    async def _extract_expansion_keywords(
        self,
        query: str,
        current_context: List[ContextDocument]
    ) -> List[str]:
        """Extract keywords for search expansion"""
        try:
            # Extract important terms from query
            query_terms = [word for word in query.split() if len(word) > 3]

            # Add synonyms and related terms (simplified)
            expansion_map = {
                'machine learning': ['ML', 'artificial intelligence', 'AI', 'deep learning'],
                'neural network': ['neuron', 'neural net', 'deep network', 'CNN', 'RNN'],
                'quantum': ['qubit', 'quantum computing', 'quantum mechanics'],
                'fMRI': ['functional MRI', 'brain imaging', 'neuroimaging'],
                'autism': ['ASD', 'autism spectrum disorder', 'developmental disorder']
            }

            expanded_terms = []
            for term in query_terms:
                expanded_terms.append(term)
                for key, expansions in expansion_map.items():
                    if key.lower() in query.lower():
                        expanded_terms.extend(expansions)

            return list(set(expanded_terms))[:10]  # Limit to 10 terms
        except Exception as e:
            logger.error(f"Error extracting expansion keywords: {e}")
            return []

    async def _get_domain_keywords(self, query: str) -> List[str]:
        """Get domain-specific keywords"""
        domain_maps = {
            'neuroscience': ['brain', 'neural', 'neuron', 'cortex', 'fMRI', 'EEG', 'synapse'],
            'quantum': ['quantum', 'qubit', 'entanglement', 'superposition', 'quantum computing'],
            'machine learning': ['ML', 'algorithm', 'model', 'training', 'neural network'],
            'autism': ['ASD', 'autism', 'developmental', 'behavior', 'intervention', 'spectrum']
        }

        query_lower = query.lower()
        for domain, keywords in domain_maps.items():
            if any(keyword in query_lower for keyword in keywords):
                return keywords

        return []

    def _generate_cache_key(self, query: str, context_documents: List[ContextDocument]) -> str:
        """Generate cache key for quality assessment"""
        content_hash = hashlib.md5()
        content_hash.update(query.encode())
        for doc in context_documents:
            content_hash.update(doc.content.encode())
        return content_hash.hexdigest()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_assessments = self.cache_hits + self.cache_misses

        return {
            "total_assessments": total_assessments,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, total_assessments),
            "avg_assessment_time": sum(self.assessment_times) / max(1, len(self.assessment_times)),
            "cached_qualities": len(self._quality_cache)
        }

    def clear_cache(self):
        """Clear quality assessment cache"""
        self._quality_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

def create_context_sufficiency_checker(
    vector_store: Optional[VectorStore] = None,
    quality_scorer: Optional[QualityScorer] = None
) -> ContextSufficiencyChecker:
    """Factory function to create context sufficiency checker"""
    return ContextSufficiencyChecker(
        quality_scorer=quality_scorer,
        vector_store=vector_store
    )

# Example usage
if __name__ == "__main__":
    async def test_context_sufficiency():
        """Test context sufficiency checker"""
        checker = create_context_sufficiency_checker()

        # Test documents
        documents = [
            ContextDocument(
                content="Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
                source="ML_basics.pdf",
                relevance_score=0.9,
                quality_score=0.8
            ),
            ContextDocument(
                content="Neural networks are computing systems inspired by biological neural networks. They learn patterns from data.",
                source="neural_networks.pdf",
                relevance_score=0.85,
                quality_score=0.75
            )
        ]

        # Assess quality
        quality = await checker.assess_context_quality(
            "What is machine learning?",
            documents
        )

        print(f"Overall Score: {quality.overall_score:.2f}")
        print(f"Sufficiency Level: {quality.sufficiency_level}")
        print(f"Missing Aspects: {quality.missing_aspects}")
        print(f"Suggestions: {quality.improvement_suggestions}")

        # Get enrichment recommendations
        recommendations = await checker.get_enrichment_recommendations(
            "What is machine learning?",
            documents,
            quality
        )

        print(f"Enrichment Recommendations: {len(recommendations)}")
        for rec in recommendations:
            print(f"  - {rec.description} (Priority: {rec.priority:.2f})")

    # Run test
    asyncio.run(test_context_sufficiency())
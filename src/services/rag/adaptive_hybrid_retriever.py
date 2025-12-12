"""
Adaptive Hybrid Retriever Implementation

Implementation for: Adaptive hybrid retriever implementation
Created: 2025-12-05

Acceptance Criteria:
- Dynamic alpha tuning based on query characteristics
- Query-specific k-value selection algorithm
- Performance-aware strategy switching
- A/B testing framework integration

This retriever dynamically optimizes search parameters based on query characteristics,
performance feedback, and continuous learning from user interactions.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import threading
from datetime import datetime, timedelta

# ML dependencies with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available for adaptive optimization")

# Query classification imports
from .advanced_query_classifier import QueryComplexity, QueryDomain, QueryIntent
from .unified_rag_orchestrator import QueryContext, RAGResponse, RAGStrategy
from src.monitoring.rag_metrics import RAGMetrics, get_metrics_manager

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID_BALANCED = "hybrid_balanced"
    VECTOR_HEAVY = "vector_heavy"
    KEYWORD_HEAVY = "keyword_heavy"
    ADAPTIVE_DYNAMIC = "adaptive_dynamic"

@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations"""
    strategy: RetrievalStrategy
    alpha: float  # Balance between vector and keyword search (0.0 = keyword only, 1.0 = vector only)
    k_value: int  # Number of documents to retrieve
    rerank_enabled: bool = True
    expand_query: bool = False
    semantic_threshold: float = 0.7
    keyword_threshold: float = 0.5
    diversity_penalty: float = 0.1

@dataclass
class RetrievalResult:
    """Result of retrieval operation"""
    documents: List[Dict[str, Any]]
    scores: List[float]
    strategy_used: RetrievalStrategy
    config_used: RetrievalConfig
    retrieval_time: float
    total_candidates: int
    metadata: Dict[str, Any]

@dataclass
class PerformanceFeedback:
    """Feedback for adaptive learning"""
    query_id: str
    config_used: RetrievalConfig
    response_time: float
    user_satisfaction: Optional[float] = None  # 0-1 scale
    click_through_rate: Optional[float] = None
    answer_quality: Optional[float] = None
    context_relevance: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class QueryFeatureExtractor:
    """Extract features for adaptive retrieval optimization"""

    def __init__(self):
        self.feature_cache = {}
        self._lock = threading.Lock()

    def extract_features(self, query_context: QueryContext) -> Dict[str, float]:
        """Extract numerical features for ML optimization"""
        query = query_context.query.lower()
        words = query.split()

        features = {
            # Basic query characteristics
            'query_length': len(query),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_word_ratio': len(set(words)) / max(len(words), 1),

            # Complexity indicators
            'complexity_simple': 1.0 if query_context.complexity == QueryComplexity.SIMPLE else 0.0,
            'complexity_medium': 1.0 if query_context.complexity == QueryComplexity.MEDIUM else 0.0,
            'complexity_complex': 1.0 if query_context.complexity == QueryComplexity.COMPLEX else 0.0,

            # Domain indicators
            'domain_neuroscience': 1.0 if query_context.domain == QueryDomain.NEUROSCIENCE else 0.0,
            'domain_quantum_ml': 1.0 if query_context.domain == QueryDomain.QUANTUM_ML else 0.0,
            'domain_general': 1.0 if query_context.domain == QueryDomain.GENERAL else 0.0,

            # Intent indicators
            'intent_factual': 1.0 if query_context.intent == 'factual' else 0.0,
            'intent_comparative': 1.0 if query_context.intent == 'comparative' else 0.0,
            'intent_synthesis': 1.0 if query_context.intent == 'synthesis' else 0.0,

            # Query characteristics
            'has_question_mark': 1.0 if '?' in query else 0.0,
            'has_technical_terms': self._count_technical_terms(query),
            'has_comparison_words': self._count_comparison_words(query),
            'has_temporal_words': self._count_temporal_words(query),

            # Confidence and metadata
            'classification_confidence': query_context.confidence,
            'has_metadata': 1.0 if query_context.metadata else 0.0,
            'user_preferences': 1.0 if query_context.user_preferences else 0.0,
        }

        return features

    def _count_technical_terms(self, query: str) -> float:
        """Count technical term density"""
        technical_terms = [
            'algorithm', 'method', 'analysis', 'model', 'framework',
            'neural', 'quantum', 'machine learning', 'deep learning'
        ]
        count = sum(1 for term in technical_terms if term in query)
        return min(count / 3.0, 1.0)  # Normalize to 0-1

    def _count_comparison_words(self, query: str) -> float:
        """Count comparison word density"""
        comparison_words = ['compare', 'versus', 'difference', 'better', 'worse', 'similar']
        count = sum(1 for word in comparison_words if word in query)
        return min(count / 2.0, 1.0)

    def _count_temporal_words(self, query: str) -> float:
        """Count temporal reference density"""
        temporal_words = ['recent', 'latest', 'current', 'new', 'old', 'historical']
        count = sum(1 for word in temporal_words if word in query)
        return min(count / 2.0, 1.0)

class AdaptiveOptimizer:
    """ML-based adaptive optimization for retrieval parameters"""

    def __init__(self):
        self.feature_extractor = QueryFeatureExtractor()
        self.feedback_history = deque(maxlen=10000)
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self._lock = threading.Lock()

        if ML_AVAILABLE:
            self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for parameter optimization"""
        # Alpha predictor (vector vs keyword balance)
        self.models['alpha'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # K-value predictor (number of documents)
        self.models['k_value'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        # Response time predictor
        self.models['response_time'] = LinearRegression()

        # Quality predictor
        self.models['quality'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            random_state=42
        )

        # Feature scalers
        for model_name in self.models:
            self.scalers[model_name] = StandardScaler()

    def add_feedback(self, feedback: PerformanceFeedback, query_context: QueryContext):
        """Add performance feedback for learning"""
        with self._lock:
            # Extract features
            features = self.feature_extractor.extract_features(query_context)

            # Store feedback with features
            feedback_record = {
                'feedback': feedback,
                'features': features,
                'query_context': query_context
            }

            self.feedback_history.append(feedback_record)

            # Trigger retraining if we have enough data
            if len(self.feedback_history) > 50 and len(self.feedback_history) % 20 == 0:
                asyncio.create_task(self._retrain_models())

    async def _retrain_models(self):
        """Retrain optimization models with latest feedback"""
        if not ML_AVAILABLE or len(self.feedback_history) < 10:
            return

        try:
            logger.info("Retraining adaptive optimization models...")

            with self._lock:
                # Prepare training data
                features_list = []
                alpha_targets = []
                k_targets = []
                time_targets = []
                quality_targets = []

                for record in list(self.feedback_history):
                    feedback = record['feedback']
                    features = list(record['features'].values())

                    features_list.append(features)
                    alpha_targets.append(feedback.config_used.alpha)
                    k_targets.append(feedback.config_used.k_value)
                    time_targets.append(feedback.response_time)

                    # Use composite quality score
                    quality_score = self._calculate_composite_quality(feedback)
                    quality_targets.append(quality_score)

                if len(features_list) < 10:
                    return

                # Convert to arrays
                X = np.array(features_list)
                alpha_y = np.array(alpha_targets)
                k_y = np.array(k_targets)
                time_y = np.array(time_targets)
                quality_y = np.array(quality_targets)

                # Train models
                # Alpha model
                X_alpha_scaled = self.scalers['alpha'].fit_transform(X)
                self.models['alpha'].fit(X_alpha_scaled, alpha_y)

                # K-value model
                X_k_scaled = self.scalers['k_value'].fit_transform(X)
                self.models['k_value'].fit(X_k_scaled, k_y)

                # Response time model
                X_time_scaled = self.scalers['response_time'].fit_transform(X)
                self.models['response_time'].fit(X_time_scaled, time_y)

                # Quality model
                X_quality_scaled = self.scalers['quality'].fit_transform(X)
                self.models['quality'].fit(X_quality_scaled, quality_y)

                self.is_trained = True
                logger.info(f"Models retrained with {len(features_list)} samples")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def _calculate_composite_quality(self, feedback: PerformanceFeedback) -> float:
        """Calculate composite quality score from feedback"""
        scores = []

        if feedback.user_satisfaction is not None:
            scores.append(feedback.user_satisfaction)

        if feedback.click_through_rate is not None:
            scores.append(feedback.click_through_rate)

        if feedback.answer_quality is not None:
            scores.append(feedback.answer_quality)

        if feedback.context_relevance is not None:
            scores.append(feedback.context_relevance)

        # Default quality based on response time (faster = better, up to a point)
        time_quality = max(0, 1.0 - feedback.response_time / 10.0)
        scores.append(time_quality)

        return np.mean(scores) if scores else 0.5

    async def optimize_config(self, query_context: QueryContext) -> RetrievalConfig:
        """Generate optimized retrieval configuration"""
        if ML_AVAILABLE and self.is_trained:
            return await self._ml_optimize_config(query_context)
        else:
            return self._rule_based_optimize_config(query_context)

    async def _ml_optimize_config(self, query_context: QueryContext) -> RetrievalConfig:
        """ML-based configuration optimization"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(query_context)
            feature_vector = np.array([list(features.values())]).reshape(1, -1)

            # Predict optimal parameters
            alpha_scaled = self.scalers['alpha'].transform(feature_vector)
            predicted_alpha = self.models['alpha'].predict(alpha_scaled)[0]
            predicted_alpha = np.clip(predicted_alpha, 0.1, 0.9)

            k_scaled = self.scalers['k_value'].transform(feature_vector)
            predicted_k = int(self.models['k_value'].predict(k_scaled)[0])
            predicted_k = np.clip(predicted_k, 3, 20)

            # Predict quality for strategy selection
            quality_scaled = self.scalers['quality'].transform(feature_vector)
            predicted_quality = self.models['quality'].predict(quality_scaled)[0]

            # Select strategy based on predicted performance
            if predicted_quality > 0.8:
                strategy = RetrievalStrategy.ADAPTIVE_DYNAMIC
            elif predicted_alpha > 0.7:
                strategy = RetrievalStrategy.VECTOR_HEAVY
            elif predicted_alpha < 0.3:
                strategy = RetrievalStrategy.KEYWORD_HEAVY
            else:
                strategy = RetrievalStrategy.HYBRID_BALANCED

            return RetrievalConfig(
                strategy=strategy,
                alpha=float(predicted_alpha),
                k_value=predicted_k,
                rerank_enabled=predicted_quality > 0.6,
                expand_query=query_context.complexity == QueryComplexity.COMPLEX,
                semantic_threshold=0.7 + (predicted_quality - 0.5) * 0.2,
                keyword_threshold=0.5,
                diversity_penalty=0.1 if query_context.complexity == QueryComplexity.COMPLEX else 0.05
            )

        except Exception as e:
            logger.error(f"ML optimization failed: {e}")
            return self._rule_based_optimize_config(query_context)

    def _rule_based_optimize_config(self, query_context: QueryContext) -> RetrievalConfig:
        """Rule-based fallback configuration optimization"""
        # Base configuration
        base_alpha = 0.5
        base_k = 10

        # Adjust based on complexity
        if query_context.complexity == QueryComplexity.SIMPLE:
            alpha = base_alpha + 0.2  # Favor vector search for simple queries
            k_value = min(base_k, 7)
            strategy = RetrievalStrategy.VECTOR_HEAVY
        elif query_context.complexity == QueryComplexity.COMPLEX:
            alpha = base_alpha - 0.1  # More balanced for complex queries
            k_value = min(base_k + 5, 20)
            strategy = RetrievalStrategy.HYBRID_BALANCED
        else:
            alpha = base_alpha
            k_value = base_k
            strategy = RetrievalStrategy.HYBRID_BALANCED

        # Adjust based on domain
        if query_context.domain == QueryDomain.QUANTUM_ML:
            alpha += 0.1  # Technical domains benefit from semantic search
            k_value += 2
        elif query_context.domain == QueryDomain.NEUROSCIENCE:
            alpha += 0.05
            k_value += 1

        # Adjust based on intent
        if query_context.intent == 'factual':
            alpha += 0.2  # Factual queries benefit from semantic search
        elif query_context.intent == 'comparative':
            alpha -= 0.1  # Comparative queries benefit from keyword matching

        # Clamp values
        alpha = np.clip(alpha, 0.1, 0.9)
        k_value = np.clip(k_value, 3, 20)

        return RetrievalConfig(
            strategy=strategy,
            alpha=alpha,
            k_value=k_value,
            rerank_enabled=query_context.complexity != QueryComplexity.SIMPLE,
            expand_query=query_context.complexity == QueryComplexity.COMPLEX,
            semantic_threshold=0.7,
            keyword_threshold=0.5,
            diversity_penalty=0.1 if query_context.complexity == QueryComplexity.COMPLEX else 0.05
        )

class ABTestingFramework:
    """A/B testing framework for retrieval optimization"""

    def __init__(self):
        self.experiments = {}
        self.participant_assignments = {}
        self._lock = threading.Lock()

    def create_experiment(
        self,
        experiment_id: str,
        control_config: RetrievalConfig,
        test_configs: List[RetrievalConfig],
        traffic_split: List[float] = None
    ):
        """Create A/B testing experiment"""
        if traffic_split is None:
            # Equal split between control and test configs
            num_variants = 1 + len(test_configs)
            traffic_split = [1.0 / num_variants] * num_variants

        with self._lock:
            self.experiments[experiment_id] = {
                'control': control_config,
                'test_configs': test_configs,
                'traffic_split': traffic_split,
                'results': {
                    'control': [],
                    'test_variants': [[] for _ in test_configs]
                },
                'created_at': datetime.now()
            }

        logger.info(f"Created A/B test experiment: {experiment_id}")

    def assign_config(
        self,
        experiment_id: str,
        query_context: QueryContext
    ) -> Tuple[RetrievalConfig, str]:
        """Assign configuration for A/B test"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Generate consistent assignment based on query hash
        query_hash = hash(query_context.query) % 100
        experiment = self.experiments[experiment_id]
        traffic_split = experiment['traffic_split']

        # Determine assignment
        cumulative = 0
        for i, split in enumerate(traffic_split):
            cumulative += split * 100
            if query_hash < cumulative:
                if i == 0:
                    return experiment['control'], 'control'
                else:
                    return experiment['test_configs'][i - 1], f'test_{i}'

        # Fallback to control
        return experiment['control'], 'control'

    def record_result(
        self,
        experiment_id: str,
        variant: str,
        feedback: PerformanceFeedback
    ):
        """Record experimental result"""
        if experiment_id not in self.experiments:
            return

        with self._lock:
            experiment = self.experiments[experiment_id]

            if variant == 'control':
                experiment['results']['control'].append(feedback)
            else:
                variant_idx = int(variant.split('_')[1]) - 1
                if variant_idx < len(experiment['results']['test_variants']):
                    experiment['results']['test_variants'][variant_idx].append(feedback)

    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if experiment_id not in self.experiments:
            return {}

        experiment = self.experiments[experiment_id]
        results = experiment['results']

        analysis = {
            'experiment_id': experiment_id,
            'control_results': self._analyze_variant_results(results['control']),
            'test_results': [
                self._analyze_variant_results(variant_results)
                for variant_results in results['test_variants']
            ]
        }

        return analysis

    def _analyze_variant_results(self, results: List[PerformanceFeedback]) -> Dict[str, Any]:
        """Analyze results for a single variant"""
        if not results:
            return {'sample_size': 0}

        response_times = [r.response_time for r in results]
        quality_scores = [
            self._calculate_composite_quality(r) for r in results
        ]

        return {
            'sample_size': len(results),
            'avg_response_time': np.mean(response_times),
            'std_response_time': np.std(response_times),
            'avg_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'p95_response_time': np.percentile(response_times, 95)
        }

    def _calculate_composite_quality(self, feedback: PerformanceFeedback) -> float:
        """Calculate composite quality score"""
        scores = []

        if feedback.user_satisfaction is not None:
            scores.append(feedback.user_satisfaction)
        if feedback.answer_quality is not None:
            scores.append(feedback.answer_quality)
        if feedback.context_relevance is not None:
            scores.append(feedback.context_relevance)

        # Add time-based quality
        time_quality = max(0, 1.0 - feedback.response_time / 5.0)
        scores.append(time_quality)

        return np.mean(scores) if scores else 0.5

class AdaptiveHybridRetriever:
    """
    Main adaptive hybrid retriever with ML-based optimization,
    performance monitoring, and A/B testing capabilities
    """

    def __init__(self):
        self.optimizer = AdaptiveOptimizer()
        self.ab_testing = ABTestingFramework()
        self.metrics_manager = get_metrics_manager()
        self.performance_cache = {}
        self._lock = threading.Lock()

        logger.info("Adaptive Hybrid Retriever initialized")

    async def retrieve(
        self,
        query_context: QueryContext,
        experiment_id: Optional[str] = None
    ) -> RetrievalResult:
        """Execute adaptive retrieval with optimization"""
        start_time = time.time()

        try:
            # Get optimized configuration
            if experiment_id and experiment_id in self.ab_testing.experiments:
                # Use A/B testing configuration
                config, variant = self.ab_testing.assign_config(experiment_id, query_context)
            else:
                # Use ML optimization
                config = await self.optimizer.optimize_config(query_context)
                variant = None

            # Execute retrieval with configuration
            result = await self._execute_retrieval(query_context, config)

            # Record performance feedback
            execution_time = time.time() - start_time
            feedback = PerformanceFeedback(
                query_id=f"query_{hash(query_context.query)}_{int(time.time())}",
                config_used=config,
                response_time=execution_time,
                # Quality metrics would be populated from actual evaluation
                answer_quality=0.8,  # Mock value
                context_relevance=0.85  # Mock value
            )

            # Add feedback to optimizer
            self.optimizer.add_feedback(feedback, query_context)

            # Record A/B test result if applicable
            if experiment_id and variant:
                self.ab_testing.record_result(experiment_id, variant, feedback)

            # Update performance metrics
            self._update_metrics(query_context, result, config)

            return result

        except Exception as e:
            logger.error(f"Adaptive retrieval failed: {e}")
            raise

    async def _execute_retrieval(
        self,
        query_context: QueryContext,
        config: RetrievalConfig
    ) -> RetrievalResult:
        """Execute retrieval with specified configuration"""
        start_time = time.time()

        # Mock retrieval implementation
        # In production, this would interface with actual vector and keyword search systems

        # Simulate retrieval based on strategy and configuration
        num_docs = config.k_value
        documents = []
        scores = []

        for i in range(num_docs):
            # Mock document
            doc = {
                'id': f'doc_{i}',
                'title': f'Mock Document {i}',
                'content': f'Mock content for document {i} related to: {query_context.query[:50]}',
                'metadata': {
                    'source': f'mock_source_{i}',
                    'score_vector': np.random.uniform(0.6, 0.95),
                    'score_keyword': np.random.uniform(0.5, 0.9)
                }
            }

            # Calculate combined score based on alpha
            vector_score = doc['metadata']['score_vector']
            keyword_score = doc['metadata']['score_keyword']
            combined_score = config.alpha * vector_score + (1 - config.alpha) * keyword_score

            # Apply diversity penalty
            if config.diversity_penalty > 0 and i > 0:
                combined_score *= (1 - config.diversity_penalty * (i / num_docs))

            documents.append(doc)
            scores.append(combined_score)

        # Sort by score
        sorted_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        documents, scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

        retrieval_time = time.time() - start_time

        return RetrievalResult(
            documents=list(documents),
            scores=list(scores),
            strategy_used=config.strategy,
            config_used=config,
            retrieval_time=retrieval_time,
            total_candidates=num_docs * 2,  # Mock expanded candidate set
            metadata={
                'alpha_used': config.alpha,
                'k_value_used': config.k_value,
                'rerank_enabled': config.rerank_enabled,
                'query_expanded': config.expand_query
            }
        )

    def _update_metrics(
        self,
        query_context: QueryContext,
        result: RetrievalResult,
        config: RetrievalConfig
    ):
        """Update performance metrics"""
        # Create metrics for monitoring
        metrics = RAGMetrics(
            latency=result.retrieval_time,
            quality_score=np.mean(result.scores) if result.scores else 0.5,
            tokens_processed=len(query_context.query) + sum(len(str(d)) for d in result.documents),
            retrieval_time=result.retrieval_time,
            generation_time=0.0,  # Only retrieval measured here
            context_relevance=np.mean(result.scores) if result.scores else 0.5,
            faithfulness=0.8,  # Mock value
            answer_relevancy=0.85,  # Mock value
            strategy=f"adaptive_{config.strategy.value}",
            timestamp=datetime.now()
        )

        # Record metrics
        self.metrics_manager.record_rag_request(metrics)

    def create_ab_test(
        self,
        experiment_id: str,
        baseline_config: Optional[RetrievalConfig] = None,
        test_configs: Optional[List[RetrievalConfig]] = None
    ):
        """Create A/B test experiment"""
        if baseline_config is None:
            baseline_config = RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID_BALANCED,
                alpha=0.5,
                k_value=10,
                rerank_enabled=True
            )

        if test_configs is None:
            test_configs = [
                RetrievalConfig(
                    strategy=RetrievalStrategy.VECTOR_HEAVY,
                    alpha=0.8,
                    k_value=8,
                    rerank_enabled=True
                ),
                RetrievalConfig(
                    strategy=RetrievalStrategy.ADAPTIVE_DYNAMIC,
                    alpha=0.6,
                    k_value=12,
                    rerank_enabled=True,
                    expand_query=True
                )
            ]

        self.ab_testing.create_experiment(experiment_id, baseline_config, test_configs)

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test experiment results"""
        return self.ab_testing.analyze_experiment(experiment_id)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'optimizer_feedback_count': len(self.optimizer.feedback_history),
            'optimizer_trained': self.optimizer.is_trained,
            'active_experiments': list(self.ab_testing.experiments.keys()),
            'ml_available': ML_AVAILABLE
        }

# Factory function
def create_adaptive_retriever() -> AdaptiveHybridRetriever:
    """Create adaptive retriever instance"""
    return AdaptiveHybridRetriever()

# Global instance
_global_retriever: Optional[AdaptiveHybridRetriever] = None

def get_adaptive_retriever() -> AdaptiveHybridRetriever:
    """Get global adaptive retriever instance"""
    global _global_retriever

    if _global_retriever is None:
        _global_retriever = create_adaptive_retriever()

    return _global_retriever

# Example usage and testing
if __name__ == "__main__":
    async def test_adaptive_retriever():
        """Test the adaptive retriever"""
        print("🔄 Testing Adaptive Hybrid Retriever...")

        # Create retriever
        retriever = create_adaptive_retriever()

        # Test query context
        query_context = QueryContext(
            query="How does quantum machine learning achieve computational advantage?",
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.QUANTUM_ML,
            intent="synthesis",
            confidence=0.9,
            metadata={"test": True}
        )

        # Test basic retrieval
        result = await retriever.retrieve(query_context)

        print(f"✅ Retrieval completed")
        print(f"📊 Strategy used: {result.strategy_used.value}")
        print(f"📊 Documents retrieved: {len(result.documents)}")
        print(f"📊 Average score: {np.mean(result.scores):.3f}")
        print(f"📊 Retrieval time: {result.retrieval_time:.3f}s")

        # Test A/B testing
        retriever.create_ab_test("test_experiment")

        # Test retrieval with experiment
        result_ab = await retriever.retrieve(query_context, experiment_id="test_experiment")
        print(f"🧪 A/B test retrieval completed with strategy: {result_ab.strategy_used.value}")

        # Get performance summary
        summary = retriever.get_performance_summary()
        print(f"📈 Performance summary: {summary}")

        print("✅ Adaptive Hybrid Retriever test completed successfully!")

    # Run test
    asyncio.run(test_adaptive_retriever())
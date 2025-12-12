"""
Adaptive Strategy Selection for Self-Learning RAG Systems

Implementation for: Dynamic strategy selection based on learning
Created: 2025-12-05

Acceptance Criteria:
- ML-based strategy selection with performance prediction
- Context-aware strategy routing with confidence scoring
- Real-time adaptation based on feedback loops
- Strategy ensemble optimization with dynamic weighting

This module provides intelligent strategy selection that adapts based on
historical performance, query characteristics, and continuous learning.
"""

import asyncio
import logging
import json
import time
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

# External dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Core dependencies
from datetime import datetime, timedelta
from ..rag.unified_rag_orchestrator import (
    RAGStrategy, QueryContext, RAGResponse, QueryComplexity, QueryDomain
)
from ..rag.feedback_loop_integration import (
    UserFeedback, FeedbackAnalysis, AdaptiveLearningEngine
)

logger = logging.getLogger(__name__)

class SelectionMode(Enum):
    """Strategy selection modes"""
    PERFORMANCE_BASED = "performance_based"
    CONTEXT_AWARE = "context_aware"
    ENSEMBLE = "ensemble"
    LEARNING_OPTIMIZED = "learning_optimized"

class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class StrategyPrediction:
    """Prediction for strategy performance"""
    strategy: RAGStrategy
    predicted_performance: float
    confidence: PredictionConfidence
    reasoning: List[str]
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SelectionResult:
    """Result of strategy selection"""
    selected_strategy: RAGStrategy
    selection_confidence: float
    alternative_strategies: List[Tuple[RAGStrategy, float]]
    reasoning_chain: List[str]
    selection_mode: SelectionMode
    feature_analysis: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceHistory:
    """Historical performance data for a strategy"""
    strategy: RAGStrategy
    performance_scores: List[float]
    query_contexts: List[QueryContext]
    timestamps: List[datetime]
    feedback_scores: List[float]
    avg_performance: float
    trend: str  # "improving", "declining", "stable"
    reliability: float  # Consistency of performance

class FeatureExtractor:
    """Extract features from query context for ML models"""

    def __init__(self):
        self.domain_encoder = LabelEncoder()
        self.complexity_encoder = LabelEncoder()
        self.intent_categories = ['factual', 'procedural', 'causal', 'comparative', 'synthesis']

        # Initialize encoders with known values
        try:
            self.domain_encoder.fit([domain.value for domain in QueryDomain])
            self.complexity_encoder.fit([complexity.value for complexity in QueryComplexity])
        except Exception as e:
            logger.warning(f"Feature extractor initialization warning: {e}")

    def extract_features(self, query_context: QueryContext) -> Dict[str, float]:
        """Extract numerical features from query context"""
        features = {}

        try:
            # Query text features
            query_text = query_context.query
            features['query_length'] = len(query_text)
            features['word_count'] = len(query_text.split())
            features['avg_word_length'] = sum(len(word) for word in query_text.split()) / max(1, len(query_text.split()))

            # Question type features
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            features['question_word_count'] = sum(1 for word in question_words if word in query_text.lower())

            # Complexity features
            try:
                features['complexity_encoded'] = self.complexity_encoder.transform([query_context.complexity.value])[0]
            except:
                features['complexity_encoded'] = 1  # Default to medium

            # Domain features
            try:
                features['domain_encoded'] = self.domain_encoder.transform([query_context.domain.value])[0]
            except:
                features['domain_encoded'] = 0  # Default to general

            # Intent features
            intent_vector = [1.0 if category in query_context.intent.lower() else 0.0
                           for category in self.intent_categories]
            for i, category in enumerate(self.intent_categories):
                features[f'intent_{category}'] = intent_vector[i]

            # Confidence feature
            features['context_confidence'] = query_context.confidence

            # Time-based features
            now = datetime.now()
            features['hour_of_day'] = now.hour
            features['day_of_week'] = now.weekday()

            # Special domain indicators
            features['is_neuroscience'] = 1.0 if query_context.domain == QueryDomain.NEUROSCIENCE else 0.0
            features['is_quantum_ml'] = 1.0 if query_context.domain == QueryDomain.QUANTUM_ML else 0.0
            features['is_dev_disorders'] = 1.0 if query_context.domain == QueryDomain.DEVELOPMENTAL_DISORDERS else 0.0

            # Query complexity indicators
            features['is_simple'] = 1.0 if query_context.complexity == QueryComplexity.SIMPLE else 0.0
            features['is_complex'] = 1.0 if query_context.complexity == QueryComplexity.COMPLEX else 0.0

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return basic features as fallback
            features = {
                'query_length': len(query_context.query),
                'word_count': len(query_context.query.split()),
                'complexity_encoded': 1,
                'domain_encoded': 0,
                'context_confidence': query_context.confidence
            }

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        base_features = [
            'query_length', 'word_count', 'avg_word_length', 'question_word_count',
            'complexity_encoded', 'domain_encoded', 'context_confidence',
            'hour_of_day', 'day_of_week', 'is_neuroscience', 'is_quantum_ml',
            'is_dev_disorders', 'is_simple', 'is_complex'
        ]

        intent_features = [f'intent_{category}' for category in self.intent_categories]

        return base_features + intent_features

class PerformancePredictor:
    """Predict strategy performance using ML models"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.performance_model = None
        self.strategy_classifier = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.training_data = []

    async def add_training_data(
        self,
        query_context: QueryContext,
        strategy_used: RAGStrategy,
        performance_score: float
    ):
        """Add training data for model learning"""
        try:
            features = self.feature_extractor.extract_features(query_context)
            self.training_data.append({
                'features': features,
                'strategy': strategy_used,
                'performance': performance_score,
                'timestamp': datetime.now()
            })

            # Retrain if we have enough data
            if len(self.training_data) >= 50 and len(self.training_data) % 10 == 0:
                await self.train_models()

        except Exception as e:
            logger.error(f"Training data addition error: {e}")

    async def train_models(self) -> bool:
        """Train performance prediction models"""
        try:
            if not SKLEARN_AVAILABLE or len(self.training_data) < 20:
                logger.warning("Insufficient data or sklearn not available for model training")
                return False

            # Prepare training data
            features_list = []
            strategies_list = []
            performances_list = []

            for data_point in self.training_data:
                feature_vector = [data_point['features'].get(fname, 0.0)
                                for fname in self.feature_extractor.get_feature_names()]
                features_list.append(feature_vector)
                strategies_list.append(data_point['strategy'].value)
                performances_list.append(data_point['performance'])

            X = np.array(features_list)
            y_strategy = np.array(strategies_list)
            y_performance = np.array(performances_list)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train performance predictor (regression)
            self.performance_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.performance_model.fit(X_scaled, y_performance)

            # Train strategy classifier
            self.strategy_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.strategy_classifier.fit(X_scaled, y_strategy)

            self.is_trained = True
            logger.info(f"Trained models with {len(self.training_data)} data points")

            # Evaluate model performance
            await self._evaluate_models(X_scaled, y_strategy, y_performance)

            return True

        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False

    async def _evaluate_models(self, X: np.ndarray, y_strategy: np.ndarray, y_performance: np.ndarray):
        """Evaluate trained models"""
        try:
            # Split data for evaluation
            X_train, X_test, y_strat_train, y_strat_test, y_perf_train, y_perf_test = train_test_split(
                X, y_strategy, y_performance, test_size=0.2, random_state=42
            )

            # Strategy classification accuracy
            if len(X_test) > 0:
                strategy_pred = self.strategy_classifier.predict(X_test)
                strategy_accuracy = accuracy_score(y_strat_test, strategy_pred)

                # Performance prediction error
                performance_pred = self.performance_model.predict(X_test)
                performance_mse = mean_squared_error(y_perf_test, performance_pred)

                logger.info(f"Model evaluation - Strategy accuracy: {strategy_accuracy:.3f}, "
                          f"Performance MSE: {performance_mse:.3f}")

        except Exception as e:
            logger.error(f"Model evaluation error: {e}")

    async def predict_strategy_performance(
        self,
        query_context: QueryContext,
        strategies: List[RAGStrategy]
    ) -> List[StrategyPrediction]:
        """Predict performance for given strategies"""
        predictions = []

        try:
            if not self.is_trained or not SKLEARN_AVAILABLE:
                # Fallback to rule-based predictions
                return await self._rule_based_predictions(query_context, strategies)

            # Extract features
            features = self.feature_extractor.extract_features(query_context)
            feature_vector = np.array([[features.get(fname, 0.0)
                                      for fname in self.feature_extractor.get_feature_names()]])

            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Get feature importance
            feature_importance = {}
            if hasattr(self.performance_model, 'feature_importances_'):
                feature_names = self.feature_extractor.get_feature_names()
                for i, importance in enumerate(self.performance_model.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(importance)

            # Predict for each strategy
            for strategy in strategies:
                # Predict performance
                predicted_performance = self.performance_model.predict(feature_vector_scaled)[0]

                # Predict strategy probability
                strategy_probs = self.strategy_classifier.predict_proba(feature_vector_scaled)[0]
                strategy_classes = self.strategy_classifier.classes_

                strategy_prob = 0.5  # Default probability
                if strategy.value in strategy_classes:
                    strategy_idx = list(strategy_classes).index(strategy.value)
                    strategy_prob = strategy_probs[strategy_idx]

                # Calculate confidence
                confidence = self._calculate_prediction_confidence(
                    predicted_performance, strategy_prob, len(self.training_data)
                )

                # Generate reasoning
                reasoning = await self._generate_prediction_reasoning(
                    strategy, predicted_performance, features, feature_importance
                )

                prediction = StrategyPrediction(
                    strategy=strategy,
                    predicted_performance=float(predicted_performance),
                    confidence=confidence,
                    reasoning=reasoning,
                    feature_importance=feature_importance,
                    metadata={
                        'strategy_probability': float(strategy_prob),
                        'model_trained': True,
                        'training_samples': len(self.training_data)
                    }
                )

                predictions.append(prediction)

        except Exception as e:
            logger.error(f"Performance prediction error: {e}")
            # Fallback to rule-based predictions
            predictions = await self._rule_based_predictions(query_context, strategies)

        return predictions

    async def _rule_based_predictions(
        self,
        query_context: QueryContext,
        strategies: List[RAGStrategy]
    ) -> List[StrategyPrediction]:
        """Fallback rule-based predictions when ML models unavailable"""
        predictions = []

        try:
            # Rule-based performance estimation
            strategy_scores = {
                RAGStrategy.SIMPLE_RAG: 0.6,
                RAGStrategy.HYBRID: 0.75,
                RAGStrategy.ENHANCED_DD_RAPTOR: 0.8,
                RAGStrategy.GRAPH_RAG: 0.7,
                RAGStrategy.GOLDEN_REFERENCE: 0.85,
                RAGStrategy.MULTIMODAL_RAG: 0.72
            }

            # Adjust scores based on query characteristics
            for strategy in strategies:
                base_score = strategy_scores.get(strategy, 0.6)

                # Domain-specific adjustments
                domain_bonus = 0.0
                if query_context.domain == QueryDomain.NEUROSCIENCE:
                    if strategy in [RAGStrategy.MULTIMODAL_RAG, RAGStrategy.ENHANCED_DD_RAPTOR]:
                        domain_bonus = 0.1
                elif query_context.domain == QueryDomain.QUANTUM_ML:
                    if strategy == RAGStrategy.GRAPH_RAG:
                        domain_bonus = 0.1

                # Complexity adjustments
                complexity_bonus = 0.0
                if query_context.complexity == QueryComplexity.COMPLEX:
                    if strategy in [RAGStrategy.HYBRID, RAGStrategy.GRAPH_RAG]:
                        complexity_bonus = 0.05
                elif query_context.complexity == QueryComplexity.SIMPLE:
                    if strategy == RAGStrategy.SIMPLE_RAG:
                        complexity_bonus = 0.1

                final_score = min(1.0, base_score + domain_bonus + complexity_bonus)

                prediction = StrategyPrediction(
                    strategy=strategy,
                    predicted_performance=final_score,
                    confidence=PredictionConfidence.MEDIUM,
                    reasoning=[f"Rule-based prediction for {strategy.value}"],
                    feature_importance={},
                    metadata={'model_trained': False}
                )

                predictions.append(prediction)

        except Exception as e:
            logger.error(f"Rule-based prediction error: {e}")

        return predictions

    def _calculate_prediction_confidence(
        self,
        predicted_performance: float,
        strategy_probability: float,
        training_samples: int
    ) -> PredictionConfidence:
        """Calculate prediction confidence level"""
        try:
            # Confidence factors
            performance_certainty = 1.0 - abs(predicted_performance - 0.5) * 2  # Higher for extreme values
            strategy_certainty = strategy_probability
            data_certainty = min(1.0, training_samples / 100)  # More data = higher confidence

            overall_confidence = (performance_certainty + strategy_certainty + data_certainty) / 3

            if overall_confidence >= 0.8:
                return PredictionConfidence.VERY_HIGH
            elif overall_confidence >= 0.65:
                return PredictionConfidence.HIGH
            elif overall_confidence >= 0.45:
                return PredictionConfidence.MEDIUM
            else:
                return PredictionConfidence.LOW

        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return PredictionConfidence.MEDIUM

    async def _generate_prediction_reasoning(
        self,
        strategy: RAGStrategy,
        predicted_performance: float,
        features: Dict[str, float],
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """Generate reasoning for prediction"""
        reasoning = []

        try:
            # Performance level reasoning
            if predicted_performance >= 0.8:
                reasoning.append(f"High performance predicted for {strategy.value}")
            elif predicted_performance >= 0.6:
                reasoning.append(f"Moderate performance predicted for {strategy.value}")
            else:
                reasoning.append(f"Lower performance predicted for {strategy.value}")

            # Feature-based reasoning
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                for feature, importance in top_features:
                    if importance > 0.1:  # Only mention significant features
                        reasoning.append(f"Key factor: {feature} (importance: {importance:.2f})")

            # Query characteristic reasoning
            complexity = features.get('complexity_encoded', 1)
            if complexity == 2:  # Complex
                reasoning.append("Complex query favors sophisticated strategies")
            elif complexity == 0:  # Simple
                reasoning.append("Simple query allows efficient basic strategies")

            # Domain-specific reasoning
            if features.get('is_neuroscience', 0) == 1:
                reasoning.append("Neuroscience domain benefits from specialized processing")
            elif features.get('is_quantum_ml', 0) == 1:
                reasoning.append("Quantum ML domain suits graph-based reasoning")

        except Exception as e:
            logger.error(f"Reasoning generation error: {e}")
            reasoning.append("Prediction based on available data")

        return reasoning[:5]  # Limit to top 5 reasons

class AdaptiveStrategySelector:
    """Main adaptive strategy selection system"""

    def __init__(self, learning_engine: Optional[AdaptiveLearningEngine] = None):
        self.learning_engine = learning_engine
        self.performance_predictor = PerformancePredictor()

        # Performance history tracking
        self.performance_history: Dict[RAGStrategy, PerformanceHistory] = {}
        self.selection_history: List[SelectionResult] = []

        # Selection parameters
        self.exploration_rate = 0.1  # Exploration vs exploitation balance
        self.ensemble_threshold = 0.8  # Threshold for ensemble selection
        self.confidence_threshold = 0.7  # Minimum confidence for predictions

        # Available strategies
        self.available_strategies = list(RAGStrategy)

    async def select_strategy(
        self,
        query_context: QueryContext,
        mode: SelectionMode = SelectionMode.LEARNING_OPTIMIZED
    ) -> SelectionResult:
        """Select optimal strategy for given query context"""
        try:
            if mode == SelectionMode.PERFORMANCE_BASED:
                return await self._performance_based_selection(query_context)
            elif mode == SelectionMode.CONTEXT_AWARE:
                return await self._context_aware_selection(query_context)
            elif mode == SelectionMode.ENSEMBLE:
                return await self._ensemble_selection(query_context)
            else:  # LEARNING_OPTIMIZED
                return await self._learning_optimized_selection(query_context)

        except Exception as e:
            logger.error(f"Strategy selection error: {e}")
            # Fallback to simple selection
            return SelectionResult(
                selected_strategy=RAGStrategy.HYBRID,
                selection_confidence=0.5,
                alternative_strategies=[],
                reasoning_chain=["Fallback selection due to error"],
                selection_mode=mode,
                feature_analysis={}
            )

    async def _performance_based_selection(self, query_context: QueryContext) -> SelectionResult:
        """Select strategy based on historical performance"""
        try:
            # Calculate average performance for each strategy
            strategy_performance = {}
            for strategy in self.available_strategies:
                if strategy in self.performance_history:
                    hist = self.performance_history[strategy]
                    strategy_performance[strategy] = hist.avg_performance
                else:
                    strategy_performance[strategy] = 0.5  # Default score

            # Select best performing strategy
            best_strategy = max(strategy_performance.keys(), key=lambda x: strategy_performance[x])

            # Create alternatives list
            sorted_strategies = sorted(strategy_performance.items(), key=lambda x: x[1], reverse=True)
            alternatives = [(strategy, score) for strategy, score in sorted_strategies[1:4]]

            reasoning = [
                f"Selected {best_strategy.value} based on historical performance",
                f"Average performance: {strategy_performance[best_strategy]:.3f}"
            ]

            return SelectionResult(
                selected_strategy=best_strategy,
                selection_confidence=min(1.0, strategy_performance[best_strategy] + 0.2),
                alternative_strategies=alternatives,
                reasoning_chain=reasoning,
                selection_mode=SelectionMode.PERFORMANCE_BASED,
                feature_analysis={'performance_scores': strategy_performance}
            )

        except Exception as e:
            logger.error(f"Performance-based selection error: {e}")
            return self._fallback_selection(query_context, SelectionMode.PERFORMANCE_BASED)

    async def _context_aware_selection(self, query_context: QueryContext) -> SelectionResult:
        """Select strategy based on query context characteristics"""
        try:
            # Extract query features for analysis
            features = self.performance_predictor.feature_extractor.extract_features(query_context)

            # Context-based strategy mapping
            selected_strategy = RAGStrategy.HYBRID  # Default

            reasoning = ["Context-aware strategy selection"]

            # Domain-based selection
            if query_context.domain == QueryDomain.NEUROSCIENCE:
                if features.get('word_count', 0) > 15:  # Complex neuroscience query
                    selected_strategy = RAGStrategy.MULTIMODAL_RAG
                    reasoning.append("Neuroscience domain with complex query → Multimodal RAG")
                else:
                    selected_strategy = RAGStrategy.ENHANCED_DD_RAPTOR
                    reasoning.append("Neuroscience domain → Enhanced DD-RAPTOR")

            elif query_context.domain == QueryDomain.QUANTUM_ML:
                selected_strategy = RAGStrategy.GRAPH_RAG
                reasoning.append("Quantum ML domain → Graph RAG for conceptual relationships")

            elif query_context.domain == QueryDomain.DEVELOPMENTAL_DISORDERS:
                selected_strategy = RAGStrategy.ENHANCED_DD_RAPTOR
                reasoning.append("Developmental disorders domain → Specialized DD-RAPTOR")

            # Complexity-based adjustments
            elif query_context.complexity == QueryComplexity.SIMPLE:
                selected_strategy = RAGStrategy.SIMPLE_RAG
                reasoning.append("Simple query → Simple RAG for efficiency")

            elif query_context.complexity == QueryComplexity.COMPLEX:
                selected_strategy = RAGStrategy.HYBRID
                reasoning.append("Complex query → Hybrid strategy for comprehensive analysis")

            # Intent-based adjustments
            if 'synthesis' in query_context.intent.lower():
                selected_strategy = RAGStrategy.GRAPH_RAG
                reasoning.append("Synthesis intent → Graph RAG for knowledge integration")

            confidence = 0.8 if len(reasoning) > 1 else 0.6

            return SelectionResult(
                selected_strategy=selected_strategy,
                selection_confidence=confidence,
                alternative_strategies=[(RAGStrategy.HYBRID, 0.7)],
                reasoning_chain=reasoning,
                selection_mode=SelectionMode.CONTEXT_AWARE,
                feature_analysis=features
            )

        except Exception as e:
            logger.error(f"Context-aware selection error: {e}")
            return self._fallback_selection(query_context, SelectionMode.CONTEXT_AWARE)

    async def _ensemble_selection(self, query_context: QueryContext) -> SelectionResult:
        """Select multiple strategies for ensemble execution"""
        try:
            # Get predictions for all strategies
            predictions = await self.performance_predictor.predict_strategy_performance(
                query_context, self.available_strategies
            )

            # Sort by predicted performance
            predictions.sort(key=lambda x: x.predicted_performance, reverse=True)

            # Select top strategies above threshold
            ensemble_strategies = []
            for pred in predictions:
                if (pred.predicted_performance >= self.ensemble_threshold and
                    pred.confidence != PredictionConfidence.LOW):
                    ensemble_strategies.append(pred.strategy)

                if len(ensemble_strategies) >= 3:  # Limit ensemble size
                    break

            if not ensemble_strategies:
                ensemble_strategies = [predictions[0].strategy]  # At least one strategy

            # Primary strategy is the top performer
            selected_strategy = ensemble_strategies[0]

            # Create alternatives from ensemble
            alternatives = [(strategy, predictions[i].predicted_performance)
                          for i, strategy in enumerate(ensemble_strategies[1:], 1)]

            reasoning = [
                f"Ensemble selection with {len(ensemble_strategies)} strategies",
                f"Primary strategy: {selected_strategy.value}",
                f"Ensemble threshold: {self.ensemble_threshold}"
            ]

            return SelectionResult(
                selected_strategy=selected_strategy,
                selection_confidence=predictions[0].predicted_performance,
                alternative_strategies=alternatives,
                reasoning_chain=reasoning,
                selection_mode=SelectionMode.ENSEMBLE,
                feature_analysis={
                    'ensemble_strategies': [s.value for s in ensemble_strategies],
                    'predictions': [p.predicted_performance for p in predictions[:3]]
                }
            )

        except Exception as e:
            logger.error(f"Ensemble selection error: {e}")
            return self._fallback_selection(query_context, SelectionMode.ENSEMBLE)

    async def _learning_optimized_selection(self, query_context: QueryContext) -> SelectionResult:
        """Select strategy optimized through learning with exploration"""
        try:
            # Get ML predictions
            predictions = await self.performance_predictor.predict_strategy_performance(
                query_context, self.available_strategies
            )

            # Sort by predicted performance
            predictions.sort(key=lambda x: x.predicted_performance, reverse=True)

            # Exploration vs exploitation decision
            import random
            if random.random() < self.exploration_rate:
                # Exploration: select a less certain strategy
                exploration_candidates = [p for p in predictions[1:]
                                        if p.confidence == PredictionConfidence.MEDIUM]
                if exploration_candidates:
                    selected_pred = random.choice(exploration_candidates)
                    reasoning_prefix = "Exploration selection: "
                else:
                    selected_pred = predictions[0]
                    reasoning_prefix = "Best prediction: "
            else:
                # Exploitation: select best predicted strategy
                selected_pred = predictions[0]
                reasoning_prefix = "Best prediction: "

            # Integrate learning engine feedback if available
            if self.learning_engine:
                learning_stats = self.learning_engine.get_learning_stats()
                if 'strategy_performance' in learning_stats:
                    # Adjust selection based on recent learning
                    strategy_learning = learning_stats['strategy_performance']
                    for pred in predictions:
                        if pred.strategy.value in strategy_learning:
                            trend = strategy_learning[pred.strategy.value].get('trend', 'stable')
                            if trend == 'improving':
                                pred.predicted_performance *= 1.1  # Boost improving strategies
                            elif trend == 'declining':
                                pred.predicted_performance *= 0.9  # Penalize declining strategies

            # Re-sort after learning adjustments
            predictions.sort(key=lambda x: x.predicted_performance, reverse=True)
            selected_pred = predictions[0]

            alternatives = [(pred.strategy, pred.predicted_performance)
                          for pred in predictions[1:4]]

            reasoning = [reasoning_prefix + selected_pred.strategy.value]
            reasoning.extend(selected_pred.reasoning[:3])

            return SelectionResult(
                selected_strategy=selected_pred.strategy,
                selection_confidence=selected_pred.predicted_performance,
                alternative_strategies=alternatives,
                reasoning_chain=reasoning,
                selection_mode=SelectionMode.LEARNING_OPTIMIZED,
                feature_analysis={
                    'ml_predictions': [(p.strategy.value, p.predicted_performance) for p in predictions],
                    'feature_importance': selected_pred.feature_importance,
                    'exploration_rate': self.exploration_rate
                }
            )

        except Exception as e:
            logger.error(f"Learning-optimized selection error: {e}")
            return self._fallback_selection(query_context, SelectionMode.LEARNING_OPTIMIZED)

    def _fallback_selection(self, query_context: QueryContext, mode: SelectionMode) -> SelectionResult:
        """Fallback strategy selection"""
        return SelectionResult(
            selected_strategy=RAGStrategy.HYBRID,
            selection_confidence=0.5,
            alternative_strategies=[(RAGStrategy.SIMPLE_RAG, 0.6)],
            reasoning_chain=["Fallback to hybrid strategy"],
            selection_mode=mode,
            feature_analysis={}
        )

    async def update_performance(
        self,
        query_context: QueryContext,
        strategy_used: RAGStrategy,
        performance_score: float,
        user_feedback: Optional[UserFeedback] = None
    ):
        """Update strategy performance based on results"""
        try:
            # Update ML model training data
            await self.performance_predictor.add_training_data(
                query_context, strategy_used, performance_score
            )

            # Update performance history
            if strategy_used not in self.performance_history:
                self.performance_history[strategy_used] = PerformanceHistory(
                    strategy=strategy_used,
                    performance_scores=[],
                    query_contexts=[],
                    timestamps=[],
                    feedback_scores=[],
                    avg_performance=0.0,
                    trend="stable",
                    reliability=0.0
                )

            hist = self.performance_history[strategy_used]
            hist.performance_scores.append(performance_score)
            hist.query_contexts.append(query_context)
            hist.timestamps.append(datetime.now())

            if user_feedback:
                feedback_score = user_feedback.user_rating / 5.0 if user_feedback.user_rating else 0.5
                hist.feedback_scores.append(feedback_score)

            # Update averages and trends
            await self._update_strategy_stats(hist)

        except Exception as e:
            logger.error(f"Performance update error: {e}")

    async def _update_strategy_stats(self, hist: PerformanceHistory):
        """Update strategy statistics"""
        try:
            # Update average performance
            if hist.performance_scores:
                hist.avg_performance = sum(hist.performance_scores) / len(hist.performance_scores)

            # Calculate trend
            if len(hist.performance_scores) >= 10:
                recent_scores = hist.performance_scores[-5:]
                older_scores = hist.performance_scores[-10:-5]

                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)

                if recent_avg > older_avg + 0.05:
                    hist.trend = "improving"
                elif recent_avg < older_avg - 0.05:
                    hist.trend = "declining"
                else:
                    hist.trend = "stable"

            # Calculate reliability (consistency)
            if len(hist.performance_scores) >= 5:
                variance = sum((score - hist.avg_performance) ** 2
                             for score in hist.performance_scores) / len(hist.performance_scores)
                hist.reliability = max(0.0, 1.0 - variance)

        except Exception as e:
            logger.error(f"Strategy stats update error: {e}")

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get strategy selection statistics"""
        try:
            stats = {
                'total_selections': len(self.selection_history),
                'strategy_usage': {},
                'performance_summary': {},
                'learning_status': {
                    'model_trained': self.performance_predictor.is_trained,
                    'training_samples': len(self.performance_predictor.training_data)
                }
            }

            # Strategy usage statistics
            for result in self.selection_history:
                strategy = result.selected_strategy.value
                if strategy not in stats['strategy_usage']:
                    stats['strategy_usage'][strategy] = 0
                stats['strategy_usage'][strategy] += 1

            # Performance summary
            for strategy, hist in self.performance_history.items():
                stats['performance_summary'][strategy.value] = {
                    'avg_performance': hist.avg_performance,
                    'sample_size': len(hist.performance_scores),
                    'trend': hist.trend,
                    'reliability': hist.reliability
                }

            return stats

        except Exception as e:
            logger.error(f"Selection stats error: {e}")
            return {'error': str(e)}

def create_adaptive_strategy_selector(
    learning_engine: Optional[AdaptiveLearningEngine] = None
) -> AdaptiveStrategySelector:
    """Factory function to create adaptive strategy selector"""
    return AdaptiveStrategySelector(learning_engine)

# Example usage
if __name__ == "__main__":
    async def test_adaptive_selection():
        """Test adaptive strategy selection"""
        selector = create_adaptive_strategy_selector()

        # Test query
        from ..rag.unified_rag_orchestrator import QueryContext, QueryComplexity, QueryDomain

        query_context = QueryContext(
            query="How does fMRI measure neural activity in autism research?",
            complexity=QueryComplexity.MEDIUM,
            domain=QueryDomain.NEUROSCIENCE,
            intent="procedural",
            confidence=0.9,
            metadata={}
        )

        # Test different selection modes
        for mode in SelectionMode:
            result = await selector.select_strategy(query_context, mode)
            print(f"{mode.value}: {result.selected_strategy.value} (confidence: {result.selection_confidence:.2f})")

        # Simulate performance update
        await selector.update_performance(query_context, RAGStrategy.MULTIMODAL_RAG, 0.85)

        # Get stats
        stats = selector.get_selection_stats()
        print(f"Selection stats: {stats}")

    # Run test
    asyncio.run(test_adaptive_selection())
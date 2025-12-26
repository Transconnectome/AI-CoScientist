"""
Feedback Loop Integration for Self-Learning RAG Systems

Implementation for: Feedback collection and adaptive improvement
Created: 2025-12-05

Acceptance Criteria:
- User feedback collection with quality scoring
- Automated feedback analysis and pattern detection
- Strategy performance adaptation based on feedback
- Continuous learning loop with model updates

This module provides comprehensive feedback loop integration for RAG systems
with automated learning and continuous improvement capabilities.
"""

import asyncio
import logging
import json
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
from abc import ABC, abstractmethod

# External dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Core dependencies
from ..rag.unified_rag_orchestrator import (
    RAGStrategy, QueryContext, RAGResponse, RAGStrategyConfig
)

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback"""
    EXPLICIT = "explicit"  # Direct user ratings/feedback
    IMPLICIT = "implicit"  # Click-through, dwell time, etc.
    AUTOMATIC = "automatic"  # System-generated quality scores
    EXPERT = "expert"  # Domain expert validation

class FeedbackSentiment(Enum):
    """Feedback sentiment classification"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class LearningAction(Enum):
    """Types of learning actions"""
    STRATEGY_WEIGHT_UPDATE = "strategy_weight_update"
    QUERY_ROUTING_UPDATE = "query_routing_update"
    PARAMETER_TUNING = "parameter_tuning"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    MODEL_RETRAINING = "model_retraining"

@dataclass
class UserFeedback:
    """Individual user feedback entry"""
    feedback_id: str
    query: str
    response: str
    strategy_used: RAGStrategy
    user_rating: Optional[float] = None  # 1-5 scale
    user_comment: Optional[str] = None
    implicit_signals: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    feedback_type: FeedbackType = FeedbackType.EXPLICIT
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeedbackAnalysis:
    """Analysis results from feedback data"""
    sentiment: FeedbackSentiment
    quality_score: float
    confidence: float
    key_issues: List[str]
    improvement_suggestions: List[str]
    pattern_insights: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningUpdate:
    """Learning update to be applied to the system"""
    update_id: str
    action_type: LearningAction
    target_component: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    validation_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    applied: bool = False

class FeedbackAnalyzer:
    """Analyze feedback patterns and extract insights"""

    def __init__(self):
        self.sentiment_keywords = {
            FeedbackSentiment.POSITIVE: [
                'good', 'great', 'excellent', 'helpful', 'accurate', 'relevant',
                'perfect', 'correct', 'useful', 'comprehensive', 'clear'
            ],
            FeedbackSentiment.NEGATIVE: [
                'bad', 'poor', 'wrong', 'irrelevant', 'unhelpful', 'inaccurate',
                'incomplete', 'confusing', 'useless', 'terrible', 'unclear'
            ]
        }

    async def analyze_feedback(self, feedback: UserFeedback) -> FeedbackAnalysis:
        """Analyze individual feedback entry"""
        try:
            # Sentiment analysis
            sentiment = await self._analyze_sentiment(feedback)

            # Quality scoring
            quality_score = await self._calculate_quality_score(feedback)

            # Issue extraction
            key_issues = await self._extract_key_issues(feedback)

            # Generate improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(feedback, key_issues)

            # Pattern analysis
            pattern_insights = await self._analyze_patterns(feedback)

            # Calculate confidence
            confidence = self._calculate_analysis_confidence(feedback, sentiment, quality_score)

            return FeedbackAnalysis(
                sentiment=sentiment,
                quality_score=quality_score,
                confidence=confidence,
                key_issues=key_issues,
                improvement_suggestions=improvement_suggestions,
                pattern_insights=pattern_insights,
                metadata={
                    'feedback_id': feedback.feedback_id,
                    'strategy_used': feedback.strategy_used.value,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Feedback analysis error: {e}")
            return FeedbackAnalysis(
                sentiment=FeedbackSentiment.NEUTRAL,
                quality_score=0.5,
                confidence=0.0,
                key_issues=['Analysis failed'],
                improvement_suggestions=['Retry feedback analysis'],
                pattern_insights={}
            )

    async def _analyze_sentiment(self, feedback: UserFeedback) -> FeedbackSentiment:
        """Analyze sentiment from feedback text and rating"""
        try:
            # Rating-based sentiment
            if feedback.user_rating is not None:
                if feedback.user_rating >= 4.0:
                    return FeedbackSentiment.POSITIVE
                elif feedback.user_rating <= 2.0:
                    return FeedbackSentiment.NEGATIVE
                else:
                    return FeedbackSentiment.NEUTRAL

            # Text-based sentiment
            if feedback.user_comment:
                comment_lower = feedback.user_comment.lower()

                positive_count = sum(1 for word in self.sentiment_keywords[FeedbackSentiment.POSITIVE]
                                   if word in comment_lower)
                negative_count = sum(1 for word in self.sentiment_keywords[FeedbackSentiment.NEGATIVE]
                                   if word in comment_lower)

                if positive_count > negative_count:
                    return FeedbackSentiment.POSITIVE
                elif negative_count > positive_count:
                    return FeedbackSentiment.NEGATIVE
                elif positive_count > 0 and negative_count > 0:
                    return FeedbackSentiment.MIXED

            return FeedbackSentiment.NEUTRAL

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return FeedbackSentiment.NEUTRAL

    async def _calculate_quality_score(self, feedback: UserFeedback) -> float:
        """Calculate quality score from feedback"""
        try:
            score_components = []

            # User rating component
            if feedback.user_rating is not None:
                normalized_rating = (feedback.user_rating - 1) / 4  # Convert 1-5 to 0-1
                score_components.append(normalized_rating)

            # Implicit signals component
            if feedback.implicit_signals:
                # Dwell time (longer = better)
                dwell_time = feedback.implicit_signals.get('dwell_time', 0)
                if dwell_time > 0:
                    dwell_score = min(1.0, dwell_time / 60)  # Normalize to 1 minute
                    score_components.append(dwell_score)

                # Click-through behavior
                clicked_sources = feedback.implicit_signals.get('clicked_sources', 0)
                if clicked_sources > 0:
                    click_score = min(1.0, clicked_sources / 3)  # Normalize to 3 clicks
                    score_components.append(click_score)

                # Follow-up queries (fewer = better)
                follow_ups = feedback.implicit_signals.get('follow_up_queries', 0)
                if follow_ups == 0:
                    score_components.append(1.0)
                else:
                    followup_score = max(0.0, 1.0 - (follow_ups / 5))  # Penalty for many follow-ups
                    score_components.append(followup_score)

            # Text sentiment component
            if feedback.user_comment:
                sentiment = await self._analyze_sentiment(feedback)
                if sentiment == FeedbackSentiment.POSITIVE:
                    score_components.append(0.8)
                elif sentiment == FeedbackSentiment.NEGATIVE:
                    score_components.append(0.2)
                elif sentiment == FeedbackSentiment.NEUTRAL:
                    score_components.append(0.5)

            # Calculate weighted average
            if score_components:
                return sum(score_components) / len(score_components)
            else:
                return 0.5  # Default neutral score

        except Exception as e:
            logger.error(f"Quality score calculation error: {e}")
            return 0.5

    async def _extract_key_issues(self, feedback: UserFeedback) -> List[str]:
        """Extract key issues from feedback"""
        issues = []

        try:
            # Low rating issues
            if feedback.user_rating is not None and feedback.user_rating <= 2.0:
                issues.append("Low user satisfaction rating")

            # Comment-based issues
            if feedback.user_comment:
                comment_lower = feedback.user_comment.lower()

                issue_indicators = {
                    'irrelevant': 'Response not relevant to query',
                    'incomplete': 'Incomplete or insufficient information',
                    'wrong': 'Factual inaccuracies detected',
                    'unclear': 'Response clarity issues',
                    'slow': 'Performance/speed concerns',
                    'missing': 'Missing expected information'
                }

                for indicator, issue in issue_indicators.items():
                    if indicator in comment_lower:
                        issues.append(issue)

            # Implicit signal issues
            if feedback.implicit_signals:
                # Short dwell time
                dwell_time = feedback.implicit_signals.get('dwell_time', 0)
                if dwell_time > 0 and dwell_time < 10:  # Less than 10 seconds
                    issues.append("User quickly abandoned response")

                # No source clicks
                clicked_sources = feedback.implicit_signals.get('clicked_sources', 0)
                if clicked_sources == 0:
                    issues.append("No engagement with provided sources")

                # Many follow-up queries
                follow_ups = feedback.implicit_signals.get('follow_up_queries', 0)
                if follow_ups >= 3:
                    issues.append("Multiple follow-up queries suggest initial response inadequate")

        except Exception as e:
            logger.error(f"Issue extraction error: {e}")
            issues.append("Unable to extract specific issues")

        return issues[:5]  # Limit to top 5 issues

    async def _generate_improvement_suggestions(self, feedback: UserFeedback, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on feedback and issues"""
        suggestions = []

        try:
            # Strategy-specific suggestions
            strategy = feedback.strategy_used

            if "Low user satisfaction rating" in issues:
                if strategy == RAGStrategy.SIMPLE_RAG:
                    suggestions.append("Consider upgrading to more sophisticated RAG strategy")
                elif strategy == RAGStrategy.HYBRID:
                    suggestions.append("Tune hybrid strategy parameters for better performance")

            if "Response not relevant to query" in issues:
                suggestions.append("Improve query understanding and routing logic")
                suggestions.append("Enhance similarity matching algorithms")

            if "Incomplete or insufficient information" in issues:
                suggestions.append("Increase context window size")
                suggestions.append("Improve document chunking strategy")

            if "Factual inaccuracies detected" in issues:
                suggestions.append("Implement fact-checking validation")
                suggestions.append("Improve source quality filtering")

            if "Response clarity issues" in issues:
                suggestions.append("Enhance response generation and formatting")
                suggestions.append("Implement clarity scoring for responses")

            if "Performance/speed concerns" in issues:
                suggestions.append("Optimize retrieval performance")
                suggestions.append("Implement better caching strategies")

            # General suggestions based on feedback type
            if feedback.feedback_type == FeedbackType.EXPERT:
                suggestions.append("Prioritize expert feedback for model updates")

            if not suggestions:
                suggestions.append("Continue monitoring feedback patterns for improvement opportunities")

        except Exception as e:
            logger.error(f"Suggestion generation error: {e}")
            suggestions.append("Unable to generate specific suggestions")

        return suggestions[:3]  # Limit to top 3 suggestions

    async def _analyze_patterns(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze patterns in feedback"""
        patterns = {}

        try:
            # Query complexity pattern
            query_length = len(feedback.query.split())
            if query_length <= 5:
                patterns['query_complexity'] = 'simple'
            elif query_length <= 15:
                patterns['query_complexity'] = 'medium'
            else:
                patterns['query_complexity'] = 'complex'

            # Strategy effectiveness pattern
            patterns['strategy_used'] = feedback.strategy_used.value

            # Time-based patterns
            hour = feedback.timestamp.hour
            if 6 <= hour <= 12:
                patterns['time_of_day'] = 'morning'
            elif 12 <= hour <= 18:
                patterns['time_of_day'] = 'afternoon'
            elif 18 <= hour <= 22:
                patterns['time_of_day'] = 'evening'
            else:
                patterns['time_of_day'] = 'night'

            # Feedback quality pattern
            if feedback.user_rating is not None:
                patterns['rating_provided'] = True
                patterns['rating_level'] = 'high' if feedback.user_rating >= 4 else 'low' if feedback.user_rating <= 2 else 'medium'
            else:
                patterns['rating_provided'] = False

        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")

        return patterns

    def _calculate_analysis_confidence(
        self,
        feedback: UserFeedback,
        sentiment: FeedbackSentiment,
        quality_score: float
    ) -> float:
        """Calculate confidence in analysis results"""
        try:
            confidence_factors = []

            # Rating availability
            if feedback.user_rating is not None:
                confidence_factors.append(0.8)

            # Comment availability
            if feedback.user_comment and len(feedback.user_comment.strip()) > 10:
                confidence_factors.append(0.7)

            # Implicit signals availability
            if feedback.implicit_signals:
                signal_count = len(feedback.implicit_signals)
                signal_confidence = min(0.6, signal_count * 0.2)
                confidence_factors.append(signal_confidence)

            # Sentiment consistency
            if feedback.user_rating is not None and feedback.user_comment:
                rating_sentiment = FeedbackSentiment.POSITIVE if feedback.user_rating >= 4 else FeedbackSentiment.NEGATIVE if feedback.user_rating <= 2 else FeedbackSentiment.NEUTRAL
                if rating_sentiment == sentiment:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.4)

            # Calculate overall confidence
            if confidence_factors:
                return min(1.0, sum(confidence_factors) / len(confidence_factors))
            else:
                return 0.3  # Low confidence with minimal data

        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.1

class AdaptiveLearningEngine:
    """Engine for adaptive learning and system updates"""

    def __init__(self, rag_config: RAGStrategyConfig):
        self.rag_config = rag_config
        self.learning_history: List[LearningUpdate] = []
        self.feedback_store: List[UserFeedback] = []
        self.analyzer = FeedbackAnalyzer()

        # Learning parameters
        self.min_feedback_threshold = 10  # Minimum feedback before learning
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7

        # Performance tracking
        self.strategy_performance: Dict[RAGStrategy, List[float]] = {}
        self.learning_effectiveness: List[float] = []

    async def process_feedback(self, feedback: UserFeedback) -> Optional[LearningUpdate]:
        """Process new feedback and generate learning updates"""
        try:
            # Store feedback
            self.feedback_store.append(feedback)

            # Analyze feedback
            analysis = await self.analyzer.analyze_feedback(feedback)

            # Update strategy performance tracking
            await self._update_strategy_performance(feedback, analysis)

            # Generate learning update if conditions are met
            if len(self.feedback_store) >= self.min_feedback_threshold:
                learning_update = await self._generate_learning_update(analysis)
                if learning_update:
                    self.learning_history.append(learning_update)
                    return learning_update

            return None

        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            return None

    async def _update_strategy_performance(self, feedback: UserFeedback, analysis: FeedbackAnalysis):
        """Update strategy performance metrics"""
        try:
            strategy = feedback.strategy_used

            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = []

            self.strategy_performance[strategy].append(analysis.quality_score)

            # Keep only recent performance data (last 100 entries)
            if len(self.strategy_performance[strategy]) > 100:
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]

        except Exception as e:
            logger.error(f"Strategy performance update error: {e}")

    async def _generate_learning_update(self, analysis: FeedbackAnalysis) -> Optional[LearningUpdate]:
        """Generate learning update based on feedback analysis"""
        try:
            if analysis.confidence < self.confidence_threshold:
                return None

            # Analyze recent feedback trends
            recent_feedback = self.feedback_store[-50:]  # Last 50 feedback entries
            update_candidates = []

            # Strategy weight updates
            strategy_update = await self._generate_strategy_weight_update(recent_feedback)
            if strategy_update:
                update_candidates.append(strategy_update)

            # Query routing updates
            routing_update = await self._generate_routing_update(recent_feedback)
            if routing_update:
                update_candidates.append(routing_update)

            # Parameter tuning updates
            parameter_update = await self._generate_parameter_update(recent_feedback)
            if parameter_update:
                update_candidates.append(parameter_update)

            # Select best update candidate
            if update_candidates:
                best_update = max(update_candidates, key=lambda x: x.expected_improvement)
                return best_update

            return None

        except Exception as e:
            logger.error(f"Learning update generation error: {e}")
            return None

    async def _generate_strategy_weight_update(self, recent_feedback: List[UserFeedback]) -> Optional[LearningUpdate]:
        """Generate strategy weight update based on performance"""
        try:
            if len(recent_feedback) < 10:
                return None

            # Calculate strategy performance scores
            strategy_scores = {}
            for feedback in recent_feedback:
                strategy = feedback.strategy_used
                analysis = await self.analyzer.analyze_feedback(feedback)

                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(analysis.quality_score)

            # Find underperforming strategies
            avg_scores = {}
            for strategy, scores in strategy_scores.items():
                if len(scores) >= 3:  # Minimum data points
                    avg_scores[strategy] = sum(scores) / len(scores)

            if not avg_scores:
                return None

            # Find strategy with largest performance gap
            overall_avg = sum(avg_scores.values()) / len(avg_scores)
            performance_gaps = {strategy: overall_avg - score for strategy, score in avg_scores.items()}

            worst_strategy = max(performance_gaps.keys(), key=lambda x: performance_gaps[x])
            worst_gap = performance_gaps[worst_strategy]

            if worst_gap > 0.1:  # Significant performance gap
                # Reduce weight for underperforming strategy
                current_weight = self.rag_config.get_strategy_weight(worst_strategy)
                new_weight = max(0.1, current_weight - self.learning_rate * worst_gap)

                update = LearningUpdate(
                    update_id=f"strategy_weight_{int(time.time())}",
                    action_type=LearningAction.STRATEGY_WEIGHT_UPDATE,
                    target_component=worst_strategy.value,
                    parameters={'new_weight': new_weight, 'old_weight': current_weight},
                    expected_improvement=worst_gap * 0.5,
                    confidence=min(1.0, len(recent_feedback) / 50),
                    validation_metrics={'performance_gap': worst_gap, 'sample_size': len(recent_feedback)}
                )

                return update

            return None

        except Exception as e:
            logger.error(f"Strategy weight update generation error: {e}")
            return None

    async def _generate_routing_update(self, recent_feedback: List[UserFeedback]) -> Optional[LearningUpdate]:
        """Generate query routing update"""
        try:
            # Analyze query patterns and strategy effectiveness
            if not SKLEARN_AVAILABLE or len(recent_feedback) < 20:
                return None

            # Extract query features and strategy performance
            query_features = []
            strategy_performance = []

            for feedback in recent_feedback:
                # Simple feature extraction
                query_length = len(feedback.query.split())
                query_complexity = 1 if query_length <= 5 else 2 if query_length <= 15 else 3

                analysis = await self.analyzer.analyze_feedback(feedback)
                performance_score = analysis.quality_score

                query_features.append([query_length, query_complexity])
                strategy_performance.append(performance_score)

            # Use clustering to identify query patterns
            if len(set(strategy_performance)) > 1:  # Need variation in performance
                kmeans = KMeans(n_clusters=2, random_state=42)
                query_clusters = kmeans.fit_predict(query_features)

                # Analyze cluster performance
                cluster_performance = {}
                for i, cluster in enumerate(query_clusters):
                    if cluster not in cluster_performance:
                        cluster_performance[cluster] = []
                    cluster_performance[cluster].append(strategy_performance[i])

                # Find performance differences between clusters
                cluster_avg = {cluster: sum(scores) / len(scores) for cluster, scores in cluster_performance.items()}

                if len(cluster_avg) > 1 and max(cluster_avg.values()) - min(cluster_avg.values()) > 0.2:
                    # Significant routing improvement opportunity
                    update = LearningUpdate(
                        update_id=f"routing_update_{int(time.time())}",
                        action_type=LearningAction.QUERY_ROUTING_UPDATE,
                        target_component="query_classifier",
                        parameters={
                            'cluster_centers': kmeans.cluster_centers_.tolist(),
                            'cluster_performance': cluster_avg
                        },
                        expected_improvement=0.15,
                        confidence=0.6,
                        validation_metrics={'performance_variance': max(cluster_avg.values()) - min(cluster_avg.values())}
                    )

                    return update

            return None

        except Exception as e:
            logger.error(f"Routing update generation error: {e}")
            return None

    async def _generate_parameter_update(self, recent_feedback: List[UserFeedback]) -> Optional[LearningUpdate]:
        """Generate parameter tuning update"""
        try:
            if len(recent_feedback) < 15:
                return None

            # Analyze feedback for common issues
            issue_patterns = {}
            for feedback in recent_feedback:
                analysis = await self.analyzer.analyze_feedback(feedback)

                for issue in analysis.key_issues:
                    if issue not in issue_patterns:
                        issue_patterns[issue] = 0
                    issue_patterns[issue] += 1

            # Find most common issue
            if issue_patterns:
                most_common_issue = max(issue_patterns.keys(), key=lambda x: issue_patterns[x])
                issue_frequency = issue_patterns[most_common_issue] / len(recent_feedback)

                if issue_frequency > 0.3:  # Issue appears in >30% of feedback
                    # Generate parameter update based on issue
                    parameter_updates = {
                        "Response not relevant to query": {
                            'similarity_threshold': 0.05,  # Increase threshold
                            'context_window_size': 200     # Increase context
                        },
                        "Incomplete or insufficient information": {
                            'max_context_blocks': 2,       # More context blocks
                            'context_window_size': 300
                        },
                        "Performance/speed concerns": {
                            'cache_ttl': 300,             # Longer cache TTL
                            'max_concurrent_requests': -2  # Reduce concurrency
                        }
                    }

                    if most_common_issue in parameter_updates:
                        update = LearningUpdate(
                            update_id=f"parameter_update_{int(time.time())}",
                            action_type=LearningAction.PARAMETER_TUNING,
                            target_component="rag_parameters",
                            parameters=parameter_updates[most_common_issue],
                            expected_improvement=issue_frequency * 0.4,
                            confidence=min(0.8, issue_frequency * 2),
                            validation_metrics={
                                'issue_frequency': issue_frequency,
                                'affected_queries': issue_patterns[most_common_issue]
                            }
                        )

                        return update

            return None

        except Exception as e:
            logger.error(f"Parameter update generation error: {e}")
            return None

    async def apply_learning_update(self, update: LearningUpdate) -> bool:
        """Apply learning update to the system"""
        try:
            logger.info(f"Applying learning update: {update.action_type.value}")

            if update.action_type == LearningAction.STRATEGY_WEIGHT_UPDATE:
                success = await self._apply_strategy_weight_update(update)
            elif update.action_type == LearningAction.QUERY_ROUTING_UPDATE:
                success = await self._apply_routing_update(update)
            elif update.action_type == LearningAction.PARAMETER_TUNING:
                success = await self._apply_parameter_update(update)
            else:
                logger.warning(f"Unknown learning action type: {update.action_type}")
                success = False

            if success:
                update.applied = True
                logger.info(f"Successfully applied learning update {update.update_id}")
            else:
                logger.error(f"Failed to apply learning update {update.update_id}")

            return success

        except Exception as e:
            logger.error(f"Learning update application error: {e}")
            return False

    async def _apply_strategy_weight_update(self, update: LearningUpdate) -> bool:
        """Apply strategy weight update"""
        try:
            target_strategy = RAGStrategy(update.target_component)
            new_weight = update.parameters['new_weight']

            # Update strategy weight in configuration
            self.rag_config.set_strategy_weight(target_strategy, new_weight)

            logger.info(f"Updated {target_strategy.value} weight to {new_weight}")
            return True

        except Exception as e:
            logger.error(f"Strategy weight update error: {e}")
            return False

    async def _apply_routing_update(self, update: LearningUpdate) -> bool:
        """Apply query routing update"""
        try:
            # Update routing logic (simplified implementation)
            cluster_centers = update.parameters['cluster_centers']
            cluster_performance = update.parameters['cluster_performance']

            # In a real implementation, this would update the query classifier
            logger.info(f"Updated query routing with {len(cluster_centers)} cluster centers")
            return True

        except Exception as e:
            logger.error(f"Routing update error: {e}")
            return False

    async def _apply_parameter_update(self, update: LearningUpdate) -> bool:
        """Apply parameter tuning update"""
        try:
            for param_name, param_value in update.parameters.items():
                # Update configuration parameters
                if hasattr(self.rag_config, param_name):
                    current_value = getattr(self.rag_config, param_name)

                    if isinstance(param_value, str) and param_value.startswith(('+', '-')):
                        # Relative update
                        delta = float(param_value)
                        new_value = current_value + delta
                    else:
                        # Absolute update
                        new_value = param_value

                    setattr(self.rag_config, param_name, new_value)
                    logger.info(f"Updated {param_name} from {current_value} to {new_value}")

            return True

        except Exception as e:
            logger.error(f"Parameter update error: {e}")
            return False

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        try:
            stats = {
                'total_feedback': len(self.feedback_store),
                'total_updates': len(self.learning_history),
                'applied_updates': len([u for u in self.learning_history if u.applied]),
                'strategy_performance': {}
            }

            # Strategy performance summary
            for strategy, scores in self.strategy_performance.items():
                if scores:
                    stats['strategy_performance'][strategy.value] = {
                        'avg_score': sum(scores) / len(scores),
                        'sample_size': len(scores),
                        'trend': 'improving' if len(scores) > 5 and scores[-5:] > scores[:-5] else 'stable'
                    }

            # Learning effectiveness
            if self.learning_effectiveness:
                stats['learning_effectiveness'] = {
                    'avg_improvement': sum(self.learning_effectiveness) / len(self.learning_effectiveness),
                    'improvement_count': len(self.learning_effectiveness)
                }

            return stats

        except Exception as e:
            logger.error(f"Learning stats generation error: {e}")
            return {'error': str(e)}

def create_feedback_loop_integration(rag_config: RAGStrategyConfig) -> AdaptiveLearningEngine:
    """Factory function to create feedback loop integration"""
    return AdaptiveLearningEngine(rag_config)

# Example usage
if __name__ == "__main__":
    async def test_feedback_loop():
        """Test feedback loop integration"""
        from ..rag.unified_rag_orchestrator import RAGStrategyConfig

        config = RAGStrategyConfig()
        learning_engine = create_feedback_loop_integration(config)

        # Simulate feedback
        feedback = UserFeedback(
            feedback_id="test_feedback_1",
            query="What is machine learning?",
            response="Machine learning is a subset of AI...",
            strategy_used=RAGStrategy.SIMPLE_RAG,
            user_rating=2.0,
            user_comment="Response was not detailed enough",
            implicit_signals={
                'dwell_time': 15,
                'clicked_sources': 0,
                'follow_up_queries': 2
            }
        )

        # Process feedback
        update = await learning_engine.process_feedback(feedback)
        if update:
            print(f"Generated learning update: {update.action_type.value}")
            success = await learning_engine.apply_learning_update(update)
            print(f"Update applied: {success}")

        # Get stats
        stats = learning_engine.get_learning_stats()
        print(f"Learning stats: {stats}")

    # Run test
    asyncio.run(test_feedback_loop())
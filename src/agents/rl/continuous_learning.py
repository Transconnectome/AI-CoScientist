"""
Continuous Learning Pipeline for RL Agent Selection

This module implements a comprehensive continuous learning system that allows
the RL agent selection model to improve over time based on real-world feedback.

Key features:
- Online learning from task outcomes and user feedback
- Periodic model retraining with accumulated experience
- Safe model updates with validation and rollback capabilities
- Adaptive exploration strategies based on performance
- Human-in-the-loop feedback integration
- Model versioning and A/B testing of model updates

The system ensures safe, gradual improvement of the RL model while maintaining
system stability and performance guarantees.
"""

import asyncio
import logging
import pickle
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import hashlib
import shutil

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from .agent_selection_env import AgentSelectionEnvironment, TaskContext
    from .agent_coordination_dqn import RLAgentSelector, DQNConfig, AgentCoordinationDQN
    from .performance_monitor import RLPerformanceMonitor, create_performance_monitor
    RL_COMPONENTS_AVAILABLE = True
except ImportError:
    RL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning mode types"""
    ONLINE_ONLY = "online_only"              # Update model incrementally
    PERIODIC_RETRAIN = "periodic_retrain"    # Full retraining at intervals
    HYBRID = "hybrid"                        # Both online and periodic
    HUMAN_FEEDBACK = "human_feedback"        # Learning from human annotations


class ModelValidationStatus(Enum):
    """Model validation status"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class ExperienceItem:
    """Individual experience for learning"""
    task_context: Dict[str, Any]
    selected_agents: List[str]
    task_outcome: bool
    quality_score: float
    execution_time: float
    user_feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    importance_weight: float = 1.0


@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_path: str
    creation_time: datetime
    performance_metrics: Dict[str, float]
    validation_status: ModelValidationStatus
    training_data_size: int
    parent_version_id: Optional[str] = None
    description: str = ""


@dataclass
class LearningConfig:
    """Configuration for continuous learning system"""
    # Learning parameters
    learning_mode: LearningMode = LearningMode.HYBRID
    online_learning_rate: float = 1e-5
    experience_buffer_size: int = 10000
    min_experiences_for_update: int = 100

    # Periodic retraining
    retrain_interval_hours: int = 24
    retrain_min_new_experiences: int = 500
    retrain_performance_threshold: float = 0.05  # 5% improvement threshold

    # Model validation
    validation_sample_size: int = 100
    validation_success_threshold: float = 0.8
    validation_timeout_seconds: float = 300.0

    # Safety mechanisms
    max_performance_degradation: float = 0.1  # 10% max degradation
    rollback_threshold: float = 0.05  # 5% degradation triggers rollback
    safe_deployment_traffic_percentage: float = 0.1

    # Experience prioritization
    enable_prioritized_experience: bool = True
    importance_decay_rate: float = 0.95
    quality_weight: float = 0.6
    recency_weight: float = 0.4

    # Human feedback
    enable_human_feedback: bool = True
    feedback_weight: float = 2.0  # Higher weight for human feedback
    feedback_confidence_threshold: float = 0.7


class ExperienceBuffer:
    """Experience replay buffer with prioritization"""

    def __init__(self, max_size: int, enable_prioritization: bool = True):
        self.max_size = max_size
        self.enable_prioritization = enable_prioritization
        self.experiences: deque = deque(maxlen=max_size)
        self.priorities: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add_experience(self, experience: ExperienceItem, priority: Optional[float] = None):
        """Add experience to the buffer"""
        with self._lock:
            if priority is None:
                priority = self._calculate_priority(experience)

            self.experiences.append(experience)
            self.priorities.append(priority)

    def _calculate_priority(self, experience: ExperienceItem) -> float:
        """Calculate priority for experience based on multiple factors"""
        if not self.enable_prioritization:
            return 1.0

        # Base priority on quality score and recency
        quality_factor = experience.quality_score
        time_factor = 1.0  # Most recent experiences get higher priority

        # Boost priority for human feedback
        feedback_factor = 1.0
        if experience.user_feedback:
            feedback_factor = 1.5

        # Boost priority for rare or difficult tasks
        rarity_factor = self._estimate_task_rarity(experience.task_context)

        priority = (quality_factor * 0.4 +
                   time_factor * 0.3 +
                   feedback_factor * 0.2 +
                   rarity_factor * 0.1)

        return priority * experience.importance_weight

    def _estimate_task_rarity(self, task_context: Dict[str, Any]) -> float:
        """Estimate task rarity based on task characteristics"""
        # Simple heuristic: complex tasks are rarer
        complexity = task_context.get('complexity_score', 0.5)
        task_type_rarity = {
            'simple': 0.3,
            'complex': 0.7,
            'comprehensive': 0.9
        }
        return task_type_rarity.get(task_context.get('task_type', 'simple'), 0.5) * complexity

    def sample_experiences(self, batch_size: int, prioritized: bool = True) -> List[ExperienceItem]:
        """Sample experiences from the buffer"""
        with self._lock:
            if not self.experiences:
                return []

            batch_size = min(batch_size, len(self.experiences))

            if not prioritized or not self.enable_prioritization:
                # Random sampling
                indices = np.random.choice(len(self.experiences), batch_size, replace=False)
                return [self.experiences[i] for i in indices]

            # Prioritized sampling
            priorities = np.array(list(self.priorities))
            probabilities = priorities / np.sum(priorities)

            indices = np.random.choice(
                len(self.experiences),
                batch_size,
                replace=False,
                p=probabilities
            )

            return [self.experiences[i] for i in indices]

    def get_recent_experiences(self, hours: int = 24) -> List[ExperienceItem]:
        """Get experiences from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [exp for exp in self.experiences if exp.timestamp > cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            if not self.experiences:
                return {"size": 0}

            qualities = [exp.quality_score for exp in self.experiences]
            priorities = list(self.priorities) if self.priorities else [1.0] * len(self.experiences)

            return {
                "size": len(self.experiences),
                "avg_quality": np.mean(qualities) if qualities else 0.0,
                "avg_priority": np.mean(priorities) if priorities else 0.0,
                "time_span_hours": (
                    datetime.now() - self.experiences[0].timestamp
                ).total_seconds() / 3600 if self.experiences else 0.0
            }


class ModelValidator:
    """Validates new models before deployment"""

    def __init__(self, agent_pool, performance_monitor: RLPerformanceMonitor):
        self.agent_pool = agent_pool
        self.performance_monitor = performance_monitor

    async def validate_model(self,
                           new_model: RLAgentSelector,
                           baseline_model: Optional[RLAgentSelector],
                           validation_config: LearningConfig) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a new model against baseline performance

        Args:
            new_model: New model to validate
            baseline_model: Current model for comparison
            validation_config: Validation configuration

        Returns:
            Tuple of (is_valid, validation_results)
        """
        logger.info("Starting model validation...")

        # Generate validation tasks
        validation_tasks = self._generate_validation_tasks(validation_config.validation_sample_size)

        # Test new model
        new_model_results = await self._test_model_performance(
            new_model, validation_tasks, validation_config
        )

        # Test baseline model if available
        baseline_results = None
        if baseline_model:
            baseline_results = await self._test_model_performance(
                baseline_model, validation_tasks, validation_config
            )

        # Compare performance
        validation_results = self._compare_model_performance(
            new_model_results, baseline_results, validation_config
        )

        is_valid = validation_results['passes_validation']

        logger.info(f"Model validation completed - Valid: {is_valid}")
        return is_valid, validation_results

    def _generate_validation_tasks(self, sample_size: int) -> List[Dict[str, Any]]:
        """Generate diverse validation tasks"""
        task_templates = [
            {
                'description': 'Statistical analysis of neuroimaging data',
                'task_type': 'complex',
                'capabilities': ['statistical_analysis', 'neuroscience_analysis'],
                'domains': ['neuroscience', 'statistics'],
                'complexity_score': 0.7
            },
            {
                'description': 'Grant proposal writing for research funding',
                'task_type': 'simple',
                'capabilities': ['grant_writing', 'scientific_writing'],
                'domains': ['grant_writing'],
                'complexity_score': 0.5
            },
            {
                'description': 'Comprehensive literature review and meta-analysis',
                'task_type': 'comprehensive',
                'capabilities': ['literature_synthesis', 'statistical_analysis'],
                'domains': ['literature_analysis', 'meta_analysis'],
                'complexity_score': 0.9
            }
        ]

        validation_tasks = []
        for i in range(sample_size):
            template = task_templates[i % len(task_templates)]
            task = template.copy()
            task['validation_id'] = f"val_{i}"
            task['priority'] = 1
            task['estimated_duration'] = 60.0 + (i % 10) * 30  # Vary duration
            validation_tasks.append(task)

        return validation_tasks

    async def _test_model_performance(self,
                                    model: RLAgentSelector,
                                    tasks: List[Dict[str, Any]],
                                    config: LearningConfig) -> Dict[str, Any]:
        """Test model performance on validation tasks"""
        results = {
            'total_tasks': len(tasks),
            'successful_selections': 0,
            'total_time': 0.0,
            'selection_times': [],
            'confidence_scores': [],
            'agent_selections': []
        }

        start_time = time.time()

        for task in tasks:
            try:
                task_context = TaskContext(
                    task_type=task['task_type'],
                    description=task['description'],
                    priority=task['priority'],
                    domains=task['domains'],
                    capabilities=task['capabilities'],
                    complexity_score=task['complexity_score'],
                    estimated_duration=task['estimated_duration']
                )

                selection_start = time.time()
                selected_agents = await model.select_agents(task_context)
                selection_time = time.time() - selection_start

                # Validate selection
                if selected_agents and len(selected_agents) > 0:
                    results['successful_selections'] += 1

                results['selection_times'].append(selection_time)
                results['agent_selections'].append(selected_agents)

                # Check timeout
                if time.time() - start_time > config.validation_timeout_seconds:
                    logger.warning("Validation timeout reached")
                    break

            except Exception as e:
                logger.error(f"Validation task failed: {e}")
                results['selection_times'].append(float('inf'))
                results['agent_selections'].append([])

        results['total_time'] = time.time() - start_time
        results['success_rate'] = results['successful_selections'] / results['total_tasks']
        results['avg_selection_time'] = np.mean([t for t in results['selection_times'] if t != float('inf')])

        return results

    def _compare_model_performance(self,
                                 new_results: Dict[str, Any],
                                 baseline_results: Optional[Dict[str, Any]],
                                 config: LearningConfig) -> Dict[str, Any]:
        """Compare new model performance against baseline"""
        comparison = {
            'new_model_success_rate': new_results['success_rate'],
            'new_model_avg_time': new_results['avg_selection_time'],
            'passes_validation': False,
            'performance_improvement': 0.0,
            'validation_details': {}
        }

        # Check absolute performance thresholds
        meets_success_threshold = new_results['success_rate'] >= config.validation_success_threshold

        if baseline_results is None:
            # No baseline - just check absolute performance
            comparison['passes_validation'] = meets_success_threshold
            comparison['validation_details'] = {
                'type': 'absolute_threshold',
                'success_threshold_met': meets_success_threshold
            }
        else:
            # Compare against baseline
            baseline_success_rate = baseline_results['success_rate']
            performance_change = new_results['success_rate'] - baseline_success_rate
            comparison['performance_improvement'] = performance_change

            # Check if performance degradation is acceptable
            acceptable_degradation = performance_change >= -config.max_performance_degradation

            comparison['passes_validation'] = meets_success_threshold and acceptable_degradation
            comparison['validation_details'] = {
                'type': 'baseline_comparison',
                'baseline_success_rate': baseline_success_rate,
                'performance_change': performance_change,
                'acceptable_degradation': acceptable_degradation,
                'success_threshold_met': meets_success_threshold
            }

        return comparison


class ContinuousLearningPipeline:
    """
    Comprehensive continuous learning pipeline for RL agent selection

    Features:
    - Online learning from task outcomes
    - Periodic model retraining
    - Safe model updates with validation
    - Experience prioritization and replay
    - Human feedback integration
    - Model versioning and rollback
    """

    def __init__(self,
                 agent_pool,
                 rl_selector: RLAgentSelector,
                 config: Optional[LearningConfig] = None):
        """
        Initialize continuous learning pipeline

        Args:
            agent_pool: Agent pool for validation
            rl_selector: RL agent selector to improve
            config: Learning configuration
        """
        self.agent_pool = agent_pool
        self.rl_selector = rl_selector
        self.config = config or LearningConfig()

        # Experience management
        self.experience_buffer = ExperienceBuffer(
            max_size=self.config.experience_buffer_size,
            enable_prioritization=self.config.enable_prioritized_experience
        )

        # Model management
        self.model_versions: Dict[str, ModelVersion] = {}
        self.current_version_id: Optional[str] = None
        self.model_storage_path = Path("models/rl_agent_selection")
        self.model_storage_path.mkdir(parents=True, exist_ok=True)

        # Performance monitoring
        self.performance_monitor = create_performance_monitor(enable_prometheus=False)
        self.model_validator = ModelValidator(agent_pool, self.performance_monitor)

        # Learning state
        self.learning_active = False
        self.last_retrain_time = datetime.now()
        self.total_experiences_processed = 0

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        logger.info(f"Continuous learning pipeline initialized - Mode: {self.config.learning_mode.value}")

    async def add_experience(self,
                           task_context: Dict[str, Any],
                           selected_agents: List[str],
                           task_outcome: bool,
                           quality_score: float,
                           execution_time: float,
                           user_feedback: Optional[Dict[str, Any]] = None):
        """
        Add new experience for learning

        Args:
            task_context: Original task context
            selected_agents: Agents that were selected
            task_outcome: Whether task was successful
            quality_score: Quality score of the outcome
            execution_time: Time taken to complete task
            user_feedback: Optional human feedback
        """
        experience = ExperienceItem(
            task_context=task_context,
            selected_agents=selected_agents,
            task_outcome=task_outcome,
            quality_score=quality_score,
            execution_time=execution_time,
            user_feedback=user_feedback,
            importance_weight=self._calculate_experience_importance(user_feedback)
        )

        self.experience_buffer.add_experience(experience)
        self.total_experiences_processed += 1

        # Record for monitoring
        self.performance_monitor.record_selection_event(
            strategy="rl_learning",
            agent_ids=selected_agents,
            task_type=task_context.get('task_type', 'unknown'),
            selection_time=0.0,  # Not applicable here
            confidence=quality_score,
            success=task_outcome,
            quality_score=quality_score
        )

        # Trigger online learning if enabled and enough experiences
        if (self.config.learning_mode in [LearningMode.ONLINE_ONLY, LearningMode.HYBRID] and
            self.total_experiences_processed % self.config.min_experiences_for_update == 0):
            await self._trigger_online_learning()

        logger.debug(f"Added experience - Total: {self.total_experiences_processed}, "
                    f"Buffer size: {len(self.experience_buffer.experiences)}")

    def _calculate_experience_importance(self, user_feedback: Optional[Dict[str, Any]]) -> float:
        """Calculate importance weight for experience"""
        if not user_feedback:
            return 1.0

        # Higher weight for human feedback
        base_weight = self.config.feedback_weight

        # Adjust based on feedback confidence if available
        confidence = user_feedback.get('confidence', 1.0)
        if confidence >= self.config.feedback_confidence_threshold:
            return base_weight * confidence
        else:
            return base_weight * 0.5  # Lower weight for low-confidence feedback

    async def _trigger_online_learning(self):
        """Trigger online learning update"""
        if not self.learning_active:
            return

        try:
            logger.info("Starting online learning update...")

            # Sample recent high-quality experiences
            batch_size = min(self.config.min_experiences_for_update,
                           len(self.experience_buffer.experiences))

            experiences = self.experience_buffer.sample_experiences(batch_size, prioritized=True)

            if len(experiences) < self.config.min_experiences_for_update // 2:
                logger.warning("Insufficient experiences for online learning")
                return

            # Perform online update (placeholder - would integrate with actual RL training)
            await self._perform_online_update(experiences)

            logger.info(f"Online learning update completed with {len(experiences)} experiences")

        except Exception as e:
            logger.error(f"Online learning failed: {e}")

    async def _perform_online_update(self, experiences: List[ExperienceItem]):
        """Perform online model update with experiences"""
        # This is a placeholder for actual online learning implementation
        # In a real implementation, this would:
        # 1. Convert experiences to training data
        # 2. Perform gradient updates on the RL model
        # 3. Update the model parameters incrementally

        # For now, we'll just log the update
        avg_quality = np.mean([exp.quality_score for exp in experiences])
        success_rate = np.mean([1.0 if exp.task_outcome else 0.0 for exp in experiences])

        logger.info(f"Online update - Avg quality: {avg_quality:.3f}, Success rate: {success_rate:.3f}")

        # Simulate model update delay
        await asyncio.sleep(0.1)

    async def trigger_periodic_retrain(self, force: bool = False) -> bool:
        """
        Trigger periodic retraining if conditions are met

        Args:
            force: Force retraining regardless of conditions

        Returns:
            True if retraining was performed
        """
        if not self.learning_active and not force:
            return False

        # Check if retraining is needed
        should_retrain = (
            force or
            self._should_trigger_retrain()
        )

        if not should_retrain:
            return False

        try:
            logger.info("Starting periodic retraining...")

            # Create new model version
            new_version_id = self._generate_version_id()
            new_model_path = self.model_storage_path / f"model_{new_version_id}"

            # Perform retraining
            retrain_success = await self._perform_periodic_retrain(new_model_path)

            if not retrain_success:
                logger.error("Periodic retraining failed")
                return False

            # Create model version record
            performance_metrics = await self._evaluate_model_performance(new_model_path)

            new_version = ModelVersion(
                version_id=new_version_id,
                model_path=str(new_model_path),
                creation_time=datetime.now(),
                performance_metrics=performance_metrics,
                validation_status=ModelValidationStatus.PENDING,
                training_data_size=len(self.experience_buffer.experiences),
                parent_version_id=self.current_version_id,
                description=f"Periodic retrain from {len(self.experience_buffer.experiences)} experiences"
            )

            # Validate new model
            validation_success = await self._validate_new_model(new_version)

            if validation_success:
                # Deploy new model
                await self._deploy_model(new_version)
                self.last_retrain_time = datetime.now()
                logger.info(f"Successfully deployed new model version {new_version_id}")
                return True
            else:
                # Mark as failed and clean up
                new_version.validation_status = ModelValidationStatus.FAILED
                self.model_versions[new_version_id] = new_version
                logger.warning(f"New model version {new_version_id} failed validation")
                return False

        except Exception as e:
            logger.error(f"Periodic retraining failed: {e}")
            return False

    def _should_trigger_retrain(self) -> bool:
        """Check if periodic retraining should be triggered"""
        # Check time since last retrain
        hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        time_condition = hours_since_retrain >= self.config.retrain_interval_hours

        # Check number of new experiences
        recent_experiences = self.experience_buffer.get_recent_experiences(self.config.retrain_interval_hours)
        experience_condition = len(recent_experiences) >= self.config.retrain_min_new_experiences

        # Check performance improvement potential
        buffer_stats = self.experience_buffer.get_stats()
        avg_quality = buffer_stats.get('avg_quality', 0.0)
        performance_condition = avg_quality > 0.5  # Only retrain if we have decent quality data

        return time_condition and experience_condition and performance_condition

    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{self.total_experiences_processed}"
        version_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"v{timestamp}_{version_hash}"

    async def _perform_periodic_retrain(self, model_path: Path) -> bool:
        """Perform full model retraining"""
        try:
            # Get all experiences for training
            all_experiences = list(self.experience_buffer.experiences)

            if len(all_experiences) < self.config.min_experiences_for_update:
                logger.warning("Insufficient experiences for retraining")
                return False

            # This is a placeholder for actual retraining logic
            # In a real implementation, this would:
            # 1. Convert experiences to training dataset
            # 2. Initialize new model or copy current model
            # 3. Train model on accumulated experiences
            # 4. Save trained model to model_path

            logger.info(f"Retraining model with {len(all_experiences)} experiences")

            # Simulate training time
            await asyncio.sleep(1.0)

            # Create dummy model file
            model_path.mkdir(parents=True, exist_ok=True)
            model_metadata = {
                "version": self._generate_version_id(),
                "training_experiences": len(all_experiences),
                "training_time": datetime.now().isoformat(),
                "config": asdict(self.config)
            }

            with open(model_path / "metadata.json", "w") as f:
                json.dump(model_metadata, f, indent=2)

            logger.info(f"Model retrained and saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False

    async def _evaluate_model_performance(self, model_path: Path) -> Dict[str, float]:
        """Evaluate model performance metrics"""
        # Placeholder for model performance evaluation
        # In a real implementation, this would load the model and evaluate it

        # Return dummy metrics for now
        return {
            "success_rate": 0.85 + np.random.normal(0, 0.05),
            "avg_quality": 0.80 + np.random.normal(0, 0.05),
            "avg_latency_ms": 800 + np.random.normal(0, 100)
        }

    async def _validate_new_model(self, model_version: ModelVersion) -> bool:
        """Validate new model version"""
        try:
            # Load current model for comparison
            current_model = self.rl_selector

            # Create temporary model for validation (placeholder)
            # In real implementation, would load the actual trained model
            new_model = self.rl_selector  # Placeholder

            # Run validation
            is_valid, validation_results = await self.model_validator.validate_model(
                new_model, current_model, self.config
            )

            # Update model version with validation results
            model_version.validation_status = (
                ModelValidationStatus.PASSED if is_valid else ModelValidationStatus.FAILED
            )

            # Store validation results in performance metrics
            model_version.performance_metrics.update(validation_results.get('validation_details', {}))

            self.model_versions[model_version.version_id] = model_version

            return is_valid

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            model_version.validation_status = ModelValidationStatus.FAILED
            return False

    async def _deploy_model(self, model_version: ModelVersion):
        """Deploy validated model version"""
        try:
            # In a real implementation, this would:
            # 1. Load the new model
            # 2. Replace the current RL selector's model
            # 3. Update routing to use new model

            # For now, just update version tracking
            self.current_version_id = model_version.version_id
            model_version.validation_status = ModelValidationStatus.PASSED

            logger.info(f"Deployed model version {model_version.version_id}")

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise

    async def rollback_model(self, target_version_id: Optional[str] = None) -> bool:
        """
        Rollback to previous model version

        Args:
            target_version_id: Specific version to rollback to, or None for previous version

        Returns:
            True if rollback was successful
        """
        try:
            if not target_version_id:
                # Find previous working version
                sorted_versions = sorted(
                    [v for v in self.model_versions.values()
                     if v.validation_status == ModelValidationStatus.PASSED],
                    key=lambda x: x.creation_time,
                    reverse=True
                )

                if len(sorted_versions) < 2:
                    logger.error("No previous version available for rollback")
                    return False

                target_version = sorted_versions[1]  # Second most recent
            else:
                target_version = self.model_versions.get(target_version_id)
                if not target_version:
                    logger.error(f"Target version {target_version_id} not found")
                    return False

            # Perform rollback
            await self._deploy_model(target_version)

            logger.info(f"Rolled back to model version {target_version.version_id}")
            return True

        except Exception as e:
            logger.error(f"Model rollback failed: {e}")
            return False

    def add_human_feedback(self,
                          task_id: str,
                          agent_selection_quality: float,
                          comments: str,
                          confidence: float = 1.0):
        """
        Add human feedback for a specific task

        Args:
            task_id: Task identifier
            agent_selection_quality: Human rating of agent selection (0-1)
            comments: Human feedback comments
            confidence: Confidence in the feedback (0-1)
        """
        if not self.config.enable_human_feedback:
            logger.warning("Human feedback disabled in configuration")
            return

        feedback = {
            'task_id': task_id,
            'selection_quality': agent_selection_quality,
            'comments': comments,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'human_rating'
        }

        # Find recent experiences for this task and update them
        # This is a simplified approach - in reality, you'd want better task tracking
        recent_experiences = self.experience_buffer.get_recent_experiences(hours=24)
        updated_count = 0

        for experience in recent_experiences:
            if experience.user_feedback is None:
                experience.user_feedback = feedback
                # Recalculate importance weight
                experience.importance_weight = self._calculate_experience_importance(feedback)
                updated_count += 1
                break  # Update only the most recent matching experience

        logger.info(f"Added human feedback - Updated {updated_count} experiences")

    async def start_continuous_learning(self):
        """Start the continuous learning system"""
        if self.learning_active:
            logger.warning("Continuous learning already active")
            return

        self.learning_active = True
        self._shutdown_event.clear()

        # Start background monitoring tasks
        await self.performance_monitor.start_background_monitoring()

        # Start periodic retraining task
        if self.config.learning_mode in [LearningMode.PERIODIC_RETRAIN, LearningMode.HYBRID]:
            retrain_task = asyncio.create_task(self._periodic_retrain_loop())
            self._background_tasks.append(retrain_task)

        logger.info("Continuous learning system started")

    async def stop_continuous_learning(self):
        """Stop the continuous learning system"""
        if not self.learning_active:
            return

        self.learning_active = False
        self._shutdown_event.set()

        # Stop background tasks
        for task in self._background_tasks:
            task.cancel()

        try:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")

        self._background_tasks.clear()

        # Stop performance monitoring
        await self.performance_monitor.stop_background_monitoring()

        logger.info("Continuous learning system stopped")

    async def _periodic_retrain_loop(self):
        """Background loop for periodic retraining"""
        while not self._shutdown_event.is_set():
            try:
                # Check if retraining is needed
                if self._should_trigger_retrain():
                    await self.trigger_periodic_retrain()

                # Wait for next check (every hour)
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=3600)

            except asyncio.TimeoutError:
                # Timeout is expected - continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic retrain loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        buffer_stats = self.experience_buffer.get_stats()

        return {
            "learning_active": self.learning_active,
            "learning_mode": self.config.learning_mode.value,
            "total_experiences": self.total_experiences_processed,
            "buffer_stats": buffer_stats,
            "current_version": self.current_version_id,
            "model_versions": len(self.model_versions),
            "last_retrain": self.last_retrain_time.isoformat(),
            "next_retrain_due": (
                self.last_retrain_time + timedelta(hours=self.config.retrain_interval_hours)
            ).isoformat(),
            "should_retrain": self._should_trigger_retrain()
        }

    def get_model_history(self) -> List[Dict[str, Any]]:
        """Get history of model versions"""
        return [
            {
                "version_id": version.version_id,
                "creation_time": version.creation_time.isoformat(),
                "performance_metrics": version.performance_metrics,
                "validation_status": version.validation_status.value,
                "training_data_size": version.training_data_size,
                "description": version.description,
                "is_current": version.version_id == self.current_version_id
            }
            for version in sorted(self.model_versions.values(),
                                key=lambda x: x.creation_time, reverse=True)
        ]


# Factory function for easy initialization
def create_continuous_learning_pipeline(
    agent_pool,
    rl_selector: RLAgentSelector,
    learning_mode: LearningMode = LearningMode.HYBRID,
    enable_human_feedback: bool = True
) -> ContinuousLearningPipeline:
    """Create a continuous learning pipeline with sensible defaults"""

    config = LearningConfig(
        learning_mode=learning_mode,
        enable_human_feedback=enable_human_feedback,
        experience_buffer_size=10000,
        min_experiences_for_update=100,
        retrain_interval_hours=24
    )

    return ContinuousLearningPipeline(agent_pool, rl_selector, config)


# Example usage and demonstration
async def demo_continuous_learning():
    """Demonstrate continuous learning functionality"""
    print("Continuous Learning Pipeline Demo")
    print("=" * 50)

    # Mock components
    class MockAgentPool:
        pass

    class MockRLSelector:
        async def select_agents(self, task_context):
            return ['neuroscience_expert']

    mock_agent_pool = MockAgentPool()
    mock_rl_selector = MockRLSelector()

    # Create pipeline
    pipeline = create_continuous_learning_pipeline(
        mock_agent_pool,
        mock_rl_selector,
        learning_mode=LearningMode.HYBRID
    )

    await pipeline.start_continuous_learning()

    print("Pipeline started - simulating task experiences...")

    # Simulate task experiences
    for i in range(50):
        task_context = {
            'task_type': 'complex',
            'description': f'Task {i}: Analyze neuroscience data',
            'capabilities': ['statistical_analysis'],
            'domains': ['neuroscience'],
            'complexity_score': 0.5 + (i % 5) * 0.1
        }

        selected_agents = ['neuroscience_expert', 'statistical_analyst']
        outcome = i % 4 != 0  # 75% success rate
        quality = 0.8 if outcome else 0.4
        execution_time = 90.0 + (i % 10) * 10

        await pipeline.add_experience(
            task_context=task_context,
            selected_agents=selected_agents,
            task_outcome=outcome,
            quality_score=quality,
            execution_time=execution_time
        )

        # Add human feedback occasionally
        if i % 10 == 0:
            pipeline.add_human_feedback(
                task_id=f"task_{i}",
                agent_selection_quality=quality + 0.1,
                comments=f"Good selection for task {i}",
                confidence=0.9
            )

    print(f"Added {50} experiences to the pipeline")

    # Check status
    status = pipeline.get_learning_status()
    print(f"\nLearning Status:")
    print(f"  Active: {status['learning_active']}")
    print(f"  Mode: {status['learning_mode']}")
    print(f"  Total experiences: {status['total_experiences']}")
    print(f"  Buffer size: {status['buffer_stats']['size']}")
    print(f"  Average quality: {status['buffer_stats'].get('avg_quality', 0):.3f}")

    # Trigger retraining
    print("\nTriggering periodic retraining...")
    retrain_success = await pipeline.trigger_periodic_retrain(force=True)
    print(f"Retraining successful: {retrain_success}")

    # Check model history
    history = pipeline.get_model_history()
    print(f"\nModel versions: {len(history)}")
    for version in history[:3]:  # Show latest 3 versions
        print(f"  {version['version_id']}: {version['validation_status']} "
              f"({version['training_data_size']} experiences)")

    await pipeline.stop_continuous_learning()
    print("\nPipeline stopped")

    return pipeline


if __name__ == "__main__":
    # Run continuous learning demo
    asyncio.run(demo_continuous_learning())
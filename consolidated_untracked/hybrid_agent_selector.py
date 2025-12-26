"""
Hybrid RL-Traditional Agent Selector

This module provides a production-ready agent selection system that combines:
1. Reinforcement Learning (RL) for intelligent agent coordination when available
2. Traditional keyword-based fallback for reliability
3. A/B testing framework for gradual rollout
4. Performance monitoring and safety mechanisms

The hybrid approach ensures system reliability while enabling gradual deployment
of RL-enhanced agent selection capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import random

# Import RL components with fallback handling
try:
    from .agent_selection_env import AgentSelectionEnvironment, TaskContext
    from .agent_coordination_dqn import RLAgentSelector, DQNConfig
    RL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RL components not available: {e}")
    RL_AVAILABLE = False

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Agent selection strategy types"""
    TRADITIONAL = "traditional"
    RL_ENABLED = "rl_enabled"
    HYBRID = "hybrid"
    A_B_TEST = "a_b_test"


@dataclass
class SelectionMetrics:
    """Metrics for tracking selection performance"""
    strategy: str
    selection_time: float
    agent_ids: List[str]
    confidence_score: float
    task_type: str
    success: bool = False
    quality_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HybridConfig:
    """Configuration for the hybrid selector"""
    # RL Configuration
    enable_rl: bool = True
    rl_confidence_threshold: float = 0.7
    rl_model_path: Optional[str] = None

    # A/B Testing Configuration
    enable_ab_testing: bool = False
    rl_traffic_percentage: float = 0.1  # Start with 10% traffic to RL
    ab_test_duration_hours: int = 24

    # Safety Configuration
    max_selection_time_seconds: float = 5.0
    fallback_on_error: bool = True
    enable_performance_monitoring: bool = True

    # Performance Thresholds
    min_success_rate: float = 0.8
    max_latency_p95_ms: float = 2000.0
    performance_window_minutes: int = 60


class PerformanceMonitor:
    """Monitors performance of different selection strategies"""

    def __init__(self):
        self.metrics: List[SelectionMetrics] = []
        self.strategy_stats: Dict[str, Dict[str, float]] = {}

    def record_selection(self, metrics: SelectionMetrics):
        """Record a selection event"""
        self.metrics.append(metrics)

        # Keep only recent metrics (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]

        self._update_strategy_stats()

    def _update_strategy_stats(self):
        """Update performance statistics for each strategy"""
        strategy_groups = {}

        for metric in self.metrics:
            strategy = metric.strategy
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(metric)

        self.strategy_stats = {}
        for strategy, metrics in strategy_groups.items():
            if not metrics:
                continue

            success_rate = sum(1 for m in metrics if m.success) / len(metrics)
            avg_latency = sum(m.selection_time for m in metrics) / len(metrics)
            p95_latency = sorted([m.selection_time for m in metrics])[int(len(metrics) * 0.95)]
            avg_confidence = sum(m.confidence_score for m in metrics) / len(metrics)

            quality_scores = [m.quality_score for m in metrics if m.quality_score is not None]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

            self.strategy_stats[strategy] = {
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency * 1000,
                'p95_latency_ms': p95_latency * 1000,
                'avg_confidence': avg_confidence,
                'avg_quality': avg_quality,
                'sample_count': len(metrics)
            }

    def get_strategy_performance(self, strategy: str) -> Dict[str, float]:
        """Get performance metrics for a specific strategy"""
        return self.strategy_stats.get(strategy, {})

    def compare_strategies(self) -> Dict[str, Any]:
        """Compare performance between strategies"""
        if len(self.strategy_stats) < 2:
            return {"status": "insufficient_data"}

        comparison = {}
        strategies = list(self.strategy_stats.keys())

        for i, strategy_a in enumerate(strategies):
            for strategy_b in strategies[i+1:]:
                stats_a = self.strategy_stats[strategy_a]
                stats_b = self.strategy_stats[strategy_b]

                comparison[f"{strategy_a}_vs_{strategy_b}"] = {
                    'success_rate_diff': stats_a.get('success_rate', 0) - stats_b.get('success_rate', 0),
                    'latency_diff_ms': stats_a.get('avg_latency_ms', 0) - stats_b.get('avg_latency_ms', 0),
                    'quality_diff': stats_a.get('avg_quality', 0) - stats_b.get('avg_quality', 0)
                }

        return comparison

    def should_increase_rl_traffic(self, config: HybridConfig) -> bool:
        """Determine if RL traffic should be increased based on performance"""
        if SelectionStrategy.RL_ENABLED.value not in self.strategy_stats:
            return False

        rl_stats = self.strategy_stats[SelectionStrategy.RL_ENABLED.value]

        # Check if RL meets minimum performance thresholds
        success_rate_ok = rl_stats.get('success_rate', 0) >= config.min_success_rate
        latency_ok = rl_stats.get('p95_latency_ms', float('inf')) <= config.max_latency_p95_ms
        sample_size_ok = rl_stats.get('sample_count', 0) >= 50

        return success_rate_ok and latency_ok and sample_size_ok


class HybridAgentSelector:
    """
    Hybrid agent selector that combines RL and traditional approaches

    Features:
    - Graceful fallback to traditional selection when RL unavailable
    - A/B testing framework for gradual RL rollout
    - Performance monitoring and automatic traffic adjustment
    - Safety mechanisms and timeout handling
    """

    def __init__(self, agent_pool, config: Optional[HybridConfig] = None):
        self.agent_pool = agent_pool
        self.config = config or HybridConfig()
        self.performance_monitor = PerformanceMonitor()

        # Initialize RL components
        self.rl_selector: Optional[RLAgentSelector] = None
        self._initialize_rl_selector()

        # A/B testing state
        self.ab_test_start_time = datetime.now()
        self.current_rl_percentage = self.config.rl_traffic_percentage

        logger.info(f"HybridAgentSelector initialized - RL Available: {RL_AVAILABLE}")

    def _initialize_rl_selector(self):
        """Initialize RL selector if available"""
        if not self.config.enable_rl or not RL_AVAILABLE:
            logger.info("RL selector disabled or unavailable")
            return

        try:
            dqn_config = DQNConfig()
            if self.config.rl_model_path:
                dqn_config.model_save_path = self.config.rl_model_path

            self.rl_selector = RLAgentSelector(
                agent_pool=self.agent_pool,
                config=dqn_config
            )

            # Load pre-trained model if available
            if self.config.rl_model_path:
                self.rl_selector.load_model(self.config.rl_model_path)
                logger.info(f"Loaded RL model from {self.config.rl_model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize RL selector: {e}")
            self.rl_selector = None

    def _determine_selection_strategy(self, task: Dict[str, Any]) -> SelectionStrategy:
        """Determine which selection strategy to use"""

        # If RL not available, use traditional
        if not self.rl_selector:
            return SelectionStrategy.TRADITIONAL

        # If A/B testing enabled, randomly assign based on traffic percentage
        if self.config.enable_ab_testing:
            if random.random() < self.current_rl_percentage:
                return SelectionStrategy.RL_ENABLED
            else:
                return SelectionStrategy.TRADITIONAL

        # Default: use RL if available and confidence is high enough
        return SelectionStrategy.RL_ENABLED

    async def select_agents(self, task: Dict[str, Any]) -> Tuple[List[str], SelectionMetrics]:
        """
        Select optimal agents for a task using hybrid approach

        Args:
            task: Task dictionary with description, type, requirements

        Returns:
            Tuple of (selected_agent_ids, selection_metrics)
        """
        start_time = time.time()
        strategy = self._determine_selection_strategy(task)

        try:
            if strategy == SelectionStrategy.RL_ENABLED:
                agent_ids, confidence = await self._select_with_rl(task)
            else:
                agent_ids, confidence = await self._select_traditional(task)

            selection_time = time.time() - start_time

            # Check for timeout
            if selection_time > self.config.max_selection_time_seconds:
                logger.warning(f"Selection timeout: {selection_time:.2f}s > {self.config.max_selection_time_seconds}s")
                if self.config.fallback_on_error and strategy == SelectionStrategy.RL_ENABLED:
                    agent_ids, confidence = await self._select_traditional(task)
                    strategy = SelectionStrategy.TRADITIONAL
                    selection_time = time.time() - start_time

            # Create metrics
            metrics = SelectionMetrics(
                strategy=strategy.value,
                selection_time=selection_time,
                agent_ids=agent_ids,
                confidence_score=confidence,
                task_type=task.get('task_type', 'unknown'),
                success=True
            )

            if self.config.enable_performance_monitoring:
                self.performance_monitor.record_selection(metrics)

            # Adjust A/B testing traffic if needed
            await self._adjust_ab_testing()

            return agent_ids, metrics

        except Exception as e:
            logger.error(f"Agent selection failed: {e}")

            # Fallback to traditional on error
            try:
                agent_ids, confidence = await self._select_traditional(task)
                selection_time = time.time() - start_time

                metrics = SelectionMetrics(
                    strategy=SelectionStrategy.TRADITIONAL.value,
                    selection_time=selection_time,
                    agent_ids=agent_ids,
                    confidence_score=confidence,
                    task_type=task.get('task_type', 'unknown'),
                    success=True
                )

                return agent_ids, metrics

            except Exception as fallback_error:
                logger.error(f"Fallback selection also failed: {fallback_error}")

                # Ultimate fallback: return default agent
                selection_time = time.time() - start_time
                default_agents = ["neuroscience_expert"]

                metrics = SelectionMetrics(
                    strategy="fallback",
                    selection_time=selection_time,
                    agent_ids=default_agents,
                    confidence_score=0.1,
                    task_type=task.get('task_type', 'unknown'),
                    success=False
                )

                return default_agents, metrics

    async def _select_with_rl(self, task: Dict[str, Any]) -> Tuple[List[str], float]:
        """Select agents using RL approach"""
        if not self.rl_selector:
            raise RuntimeError("RL selector not available")

        # Convert task to format expected by RL selector
        task_context = TaskContext(
            task_type=task.get('task_type', 'general'),
            description=task.get('description', ''),
            priority=task.get('priority', 1),
            domains=task.get('domains', []),
            capabilities=task.get('capabilities', []),
            complexity_score=task.get('complexity_score', 0.5),
            estimated_duration=task.get('estimated_duration', 60.0)
        )

        # Use RL selector
        agent_ids = await self.rl_selector.select_agents(task_context)

        # Calculate confidence based on model predictions
        confidence = min(0.9, 0.5 + len(agent_ids) * 0.1)  # Placeholder logic

        return agent_ids, confidence

    async def _select_traditional(self, task: Dict[str, Any]) -> Tuple[List[str], float]:
        """Select agents using traditional keyword-based approach"""

        # Use existing AgentPool smart task routing
        try:
            # For single agent selection
            if task.get('team_size', 1) == 1:
                best_agent = await self.agent_pool.smart_task_routing(task)
                return [best_agent], 0.6  # Fixed confidence for traditional

            # For team selection, use get_optimal_agent_team
            requirements = {
                "capabilities": task.get('capabilities', []),
                "domains": task.get('domains', []),
                "task_type": task.get('task_type', 'general')
            }

            # Extract requirements from task description if not provided
            if not requirements["capabilities"] and not requirements["domains"]:
                description = task.get('description', '').lower()

                if "statistical" in description:
                    requirements["capabilities"].append("statistical_analysis")
                if "grant" in description or "proposal" in description:
                    requirements["capabilities"].append("grant_writing")
                if "hypothesis" in description:
                    requirements["capabilities"].append("hypothesis_generation")
                if "clinical" in description:
                    requirements["capabilities"].append("clinical_validation")
                if "literature" in description:
                    requirements["capabilities"].append("literature_synthesis")
                if "neuroscience" in description:
                    requirements["domains"].append("neuroscience")

            optimal_agents = self.agent_pool.get_optimal_agent_team(requirements)
            return optimal_agents[:task.get('team_size', 2)], 0.7

        except Exception as e:
            logger.error(f"Traditional selection error: {e}")
            return ["neuroscience_expert"], 0.3  # Ultimate fallback

    async def _adjust_ab_testing(self):
        """Adjust A/B testing traffic based on performance"""
        if not self.config.enable_ab_testing:
            return

        # Check if it's time to adjust traffic
        hours_since_start = (datetime.now() - self.ab_test_start_time).total_seconds() / 3600
        if hours_since_start < self.config.ab_test_duration_hours:
            return

        # Evaluate if RL should get more traffic
        if self.performance_monitor.should_increase_rl_traffic(self.config):
            self.current_rl_percentage = min(1.0, self.current_rl_percentage + 0.1)
            logger.info(f"Increased RL traffic to {self.current_rl_percentage:.1%}")
        else:
            # If RL performance is poor, decrease traffic
            comparison = self.performance_monitor.compare_strategies()
            if comparison and comparison.get('rl_enabled_vs_traditional', {}).get('success_rate_diff', 0) < -0.05:
                self.current_rl_percentage = max(0.05, self.current_rl_percentage - 0.05)
                logger.info(f"Decreased RL traffic to {self.current_rl_percentage:.1%}")

        # Reset A/B test timer
        self.ab_test_start_time = datetime.now()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "configuration": {
                "rl_enabled": self.config.enable_rl,
                "rl_available": self.rl_selector is not None,
                "ab_testing_enabled": self.config.enable_ab_testing,
                "current_rl_traffic": self.current_rl_percentage
            },
            "strategy_performance": self.performance_monitor.strategy_stats,
            "strategy_comparison": self.performance_monitor.compare_strategies(),
            "total_selections": len(self.performance_monitor.metrics),
            "monitoring_enabled": self.config.enable_performance_monitoring
        }

    async def update_task_outcome(self, metrics: SelectionMetrics, success: bool, quality_score: Optional[float] = None):
        """Update the outcome of a task for learning purposes"""
        metrics.success = success
        metrics.quality_score = quality_score

        # Update RL model if available and this was an RL selection
        if (self.rl_selector and
            metrics.strategy == SelectionStrategy.RL_ENABLED.value and
            quality_score is not None):
            # Future: implement online learning update
            pass

    def save_rl_model(self, path: str):
        """Save the current RL model"""
        if self.rl_selector:
            self.rl_selector.save_model(path)
            logger.info(f"Saved RL model to {path}")

    def load_rl_model(self, path: str):
        """Load a pre-trained RL model"""
        if self.rl_selector:
            self.rl_selector.load_model(path)
            logger.info(f"Loaded RL model from {path}")


# Factory function for easy initialization
def create_hybrid_selector(agent_pool, enable_rl: bool = True, enable_ab_testing: bool = False) -> HybridAgentSelector:
    """Create a hybrid agent selector with sensible defaults"""

    config = HybridConfig(
        enable_rl=enable_rl and RL_AVAILABLE,
        enable_ab_testing=enable_ab_testing,
        rl_traffic_percentage=0.1 if enable_ab_testing else 1.0,
        enable_performance_monitoring=True
    )

    return HybridAgentSelector(agent_pool, config)


# Example usage and testing
async def demo_hybrid_selection():
    """Demonstrate the hybrid selector functionality"""

    # Mock agent pool for demonstration
    class MockAgentPool:
        async def smart_task_routing(self, task):
            return "neuroscience_expert"

        def get_optimal_agent_team(self, requirements):
            return ["neuroscience_expert", "statistical_analyst"]

    mock_pool = MockAgentPool()
    selector = create_hybrid_selector(mock_pool, enable_rl=True, enable_ab_testing=True)

    # Test various task types
    test_tasks = [
        {
            "description": "Analyze statistical significance of neuroscience data",
            "task_type": "complex",
            "capabilities": ["statistical_analysis", "neuroscience_analysis"],
            "domains": ["neuroscience", "statistics"]
        },
        {
            "description": "Write a grant proposal for autism research",
            "task_type": "simple",
            "capabilities": ["grant_writing"],
            "domains": ["grant_writing"]
        },
        {
            "description": "Generate hypothesis about brain connectivity",
            "task_type": "general",
            "capabilities": ["hypothesis_generation"],
            "domains": ["neuroscience"]
        }
    ]

    print("Hybrid Agent Selector Demo")
    print("=" * 50)

    for i, task in enumerate(test_tasks):
        print(f"\nTask {i+1}: {task['description']}")
        agents, metrics = await selector.select_agents(task)

        print(f"Selected agents: {agents}")
        print(f"Strategy: {metrics.strategy}")
        print(f"Selection time: {metrics.selection_time:.3f}s")
        print(f"Confidence: {metrics.confidence_score:.2f}")

        # Simulate task outcome
        success = random.random() > 0.1  # 90% success rate
        quality = random.uniform(0.6, 0.95) if success else random.uniform(0.1, 0.5)
        await selector.update_task_outcome(metrics, success, quality)

    # Show performance report
    print("\n" + "=" * 50)
    print("Performance Report:")
    report = selector.get_performance_report()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_hybrid_selection())
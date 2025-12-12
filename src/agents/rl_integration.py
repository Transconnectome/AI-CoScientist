"""
RL Integration Module

Integrates the RL-enhanced agent selection system with the existing AgentPool
in a backwards-compatible way. This module can be imported to upgrade agent
selection capabilities without breaking existing functionality.

Usage:
    from src.agents.rl_integration import enhance_agent_pool_with_rl

    # Enhance existing agent pool
    enhanced_pool = enhance_agent_pool_with_rl(agent_pool)

    # Use enhanced capabilities
    agents, metrics = await enhanced_pool.select_agents_smart(task)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps

try:
    from .rl import (
        HybridAgentSelector,
        HybridConfig,
        create_hybrid_selector,
        RL_COMPONENTS_AVAILABLE
    )
except ImportError:
    logging.warning("RL components not available, using fallback integration")
    RL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RLEnhancedAgentPool:
    """
    RL-Enhanced Agent Pool wrapper

    This class wraps the existing AgentPool and adds RL-enhanced selection
    capabilities while maintaining full backwards compatibility.
    """

    def __init__(self, original_pool, rl_config: Optional[Dict[str, Any]] = None):
        """
        Initialize RL-enhanced agent pool

        Args:
            original_pool: The original AgentPool instance
            rl_config: Optional RL configuration parameters
        """
        self.original_pool = original_pool
        self.rl_enabled = RL_COMPONENTS_AVAILABLE

        # Create hybrid selector if RL available
        if self.rl_enabled:
            config = HybridConfig()
            if rl_config:
                for key, value in rl_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            self.hybrid_selector = HybridAgentSelector(original_pool, config)
            logger.info("RL-enhanced agent selection initialized")
        else:
            self.hybrid_selector = None
            logger.info("RL not available, using traditional selection only")

    def __getattr__(self, name):
        """Delegate all unknown attributes to the original pool"""
        return getattr(self.original_pool, name)

    async def select_agents_smart(self, task: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Smart agent selection using hybrid RL-traditional approach

        Args:
            task: Task dictionary with description, requirements, etc.

        Returns:
            Tuple of (selected_agent_ids, selection_metadata)
        """
        if self.hybrid_selector:
            agent_ids, metrics = await self.hybrid_selector.select_agents(task)
            metadata = {
                'strategy': metrics.strategy,
                'selection_time': metrics.selection_time,
                'confidence_score': metrics.confidence_score,
                'task_type': metrics.task_type,
                'rl_enabled': True
            }
            return agent_ids, metadata
        else:
            # Fallback to traditional selection
            agent_id = await self.original_pool.smart_task_routing(task)
            metadata = {
                'strategy': 'traditional',
                'selection_time': 0.0,
                'confidence_score': 0.5,
                'task_type': task.get('task_type', 'general'),
                'rl_enabled': False
            }
            return [agent_id], metadata

    async def smart_task_routing_enhanced(self, task: Dict[str, Any]) -> str:
        """
        Enhanced version of smart_task_routing that returns the best single agent
        """
        agent_ids, _ = await self.select_agents_smart(task)
        return agent_ids[0] if agent_ids else "neuroscience_expert"

    def get_optimal_agent_team_enhanced(self, task_requirements: Dict[str, Any]) -> List[str]:
        """
        Enhanced version of get_optimal_agent_team using RL when available
        """
        if self.hybrid_selector:
            # Convert requirements to task format for RL selection
            task = {
                'description': task_requirements.get('description', ''),
                'task_type': task_requirements.get('task_type', 'general'),
                'capabilities': task_requirements.get('capabilities', []),
                'domains': task_requirements.get('domains', []),
                'team_size': task_requirements.get('team_size', 3)
            }

            # Use async wrapper for sync interface compatibility
            loop = asyncio.get_event_loop()
            agent_ids, _ = loop.run_until_complete(self.select_agents_smart(task))
            return agent_ids
        else:
            # Fallback to original implementation
            return self.original_pool.get_optimal_agent_team(task_requirements)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL performance metrics if available"""
        if self.hybrid_selector:
            return self.hybrid_selector.get_performance_report()
        else:
            return {
                "rl_available": False,
                "strategy": "traditional_only",
                "message": "RL components not available"
            }

    async def update_selection_outcome(self, task: Dict[str, Any],
                                     agent_ids: List[str],
                                     success: bool,
                                     quality_score: Optional[float] = None):
        """
        Update the outcome of an agent selection for learning purposes

        Args:
            task: Original task that was assigned
            agent_ids: Agents that were selected
            success: Whether the task was successful
            quality_score: Optional quality score (0.0 to 1.0)
        """
        if self.hybrid_selector:
            # Create mock metrics for the update
            from .rl import SelectionMetrics
            from datetime import datetime

            metrics = SelectionMetrics(
                strategy="rl_enabled",
                selection_time=0.0,
                agent_ids=agent_ids,
                confidence_score=0.0,
                task_type=task.get('task_type', 'general'),
                timestamp=datetime.now()
            )

            await self.hybrid_selector.update_task_outcome(metrics, success, quality_score)

    def enable_ab_testing(self, rl_traffic_percentage: float = 0.1):
        """Enable A/B testing with specified RL traffic percentage"""
        if self.hybrid_selector:
            self.hybrid_selector.config.enable_ab_testing = True
            self.hybrid_selector.config.rl_traffic_percentage = rl_traffic_percentage
            self.hybrid_selector.current_rl_percentage = rl_traffic_percentage
            logger.info(f"A/B testing enabled with {rl_traffic_percentage:.1%} RL traffic")

    def disable_ab_testing(self):
        """Disable A/B testing"""
        if self.hybrid_selector:
            self.hybrid_selector.config.enable_ab_testing = False
            logger.info("A/B testing disabled")

    def save_rl_model(self, path: str):
        """Save the current RL model"""
        if self.hybrid_selector:
            self.hybrid_selector.save_rl_model(path)

    def load_rl_model(self, path: str):
        """Load a pre-trained RL model"""
        if self.hybrid_selector:
            self.hybrid_selector.load_rl_model(path)


def enhance_agent_pool_with_rl(agent_pool, rl_config: Optional[Dict[str, Any]] = None) -> RLEnhancedAgentPool:
    """
    Enhance an existing AgentPool with RL capabilities

    Args:
        agent_pool: Existing AgentPool instance
        rl_config: Optional configuration for RL components

    Returns:
        Enhanced agent pool with RL capabilities
    """
    return RLEnhancedAgentPool(agent_pool, rl_config)


def add_rl_monitoring_decorator(func):
    """
    Decorator to add RL performance monitoring to agent pool methods
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Record selection performance for monitoring
        start_time = asyncio.get_event_loop().time()

        try:
            result = await func(self, *args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            success = False
            result = None

        end_time = asyncio.get_event_loop().time()
        selection_time = end_time - start_time

        # Log performance metrics
        if hasattr(self, 'hybrid_selector') and self.hybrid_selector:
            logger.debug(f"Agent selection: {selection_time:.3f}s, success: {success}")

        return result

    return wrapper


# Integration helper functions
def create_rl_config(
    enable_rl: bool = True,
    enable_ab_testing: bool = False,
    rl_traffic_percentage: float = 0.1,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """Create an RL configuration dictionary"""
    return {
        'enable_rl': enable_rl,
        'enable_ab_testing': enable_ab_testing,
        'rl_traffic_percentage': rl_traffic_percentage,
        'rl_confidence_threshold': confidence_threshold,
        'fallback_on_error': True,
        'enable_performance_monitoring': True
    }


async def test_rl_integration(agent_pool):
    """
    Test the RL integration with an existing agent pool

    Args:
        agent_pool: AgentPool instance to test with

    Returns:
        Test results dictionary
    """
    print("Testing RL Integration...")
    print("=" * 50)

    # Enhance the pool
    enhanced_pool = enhance_agent_pool_with_rl(
        agent_pool,
        create_rl_config(enable_ab_testing=True)
    )

    # Test tasks
    test_tasks = [
        {
            "description": "Analyze fMRI data for autism spectrum disorders",
            "task_type": "complex",
            "capabilities": ["statistical_analysis", "neuroscience_analysis"],
            "domains": ["neuroscience", "medical_imaging"]
        },
        {
            "description": "Write grant proposal for developmental disorders research",
            "task_type": "simple",
            "capabilities": ["grant_writing", "scientific_writing"],
            "domains": ["grant_writing", "developmental_disorders"]
        },
        {
            "description": "Systematic literature review on brain connectivity",
            "task_type": "comprehensive",
            "capabilities": ["literature_synthesis", "data_analysis"],
            "domains": ["neuroscience", "literature_analysis"]
        }
    ]

    results = []

    for i, task in enumerate(test_tasks):
        print(f"\nTask {i+1}: {task['description'][:50]}...")

        # Test enhanced selection
        agents, metadata = await enhanced_pool.select_agents_smart(task)

        print(f"Selected agents: {agents}")
        print(f"Strategy used: {metadata['strategy']}")
        print(f"Selection time: {metadata['selection_time']:.3f}s")
        print(f"Confidence: {metadata['confidence_score']:.2f}")
        print(f"RL enabled: {metadata['rl_enabled']}")

        # Simulate task execution and feedback
        import random
        success = random.random() > 0.1  # 90% success rate
        quality = random.uniform(0.7, 0.95) if success else random.uniform(0.2, 0.6)

        await enhanced_pool.update_selection_outcome(task, agents, success, quality)
        print(f"Task outcome: {'Success' if success else 'Failed'}, Quality: {quality:.2f}")

        results.append({
            'task': task['description'][:30] + "...",
            'agents': agents,
            'strategy': metadata['strategy'],
            'selection_time': metadata['selection_time'],
            'confidence': metadata['confidence_score'],
            'success': success,
            'quality': quality
        })

    # Show performance report
    print("\n" + "=" * 50)
    print("Performance Report:")
    performance_report = enhanced_pool.get_performance_metrics()

    if performance_report.get('rl_available', True):
        for strategy, stats in performance_report.get('strategy_performance', {}).items():
            print(f"\nStrategy: {strategy}")
            print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
            print(f"  Avg Latency: {stats.get('avg_latency_ms', 0):.0f}ms")
            print(f"  Avg Quality: {stats.get('avg_quality', 0):.2f}")
            print(f"  Sample Count: {stats.get('sample_count', 0)}")
    else:
        print("RL not available - using traditional selection only")

    return {
        'test_results': results,
        'performance_report': performance_report,
        'rl_available': RL_COMPONENTS_AVAILABLE
    }


# Example usage
async def demo_integration():
    """Demonstrate the integration functionality"""

    # Mock agent pool for demonstration
    class MockAgentPool:
        async def smart_task_routing(self, task):
            return "neuroscience_expert"

        def get_optimal_agent_team(self, requirements):
            return ["neuroscience_expert", "statistical_analyst"]

    mock_pool = MockAgentPool()

    # Test the integration
    test_results = await test_rl_integration(mock_pool)

    print(f"\nIntegration test completed. RL Available: {test_results['rl_available']}")
    return test_results


if __name__ == "__main__":
    # Run integration demo
    asyncio.run(demo_integration())
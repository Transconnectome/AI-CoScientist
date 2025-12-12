"""
Reinforcement Learning Enhanced Agent Coordination

This package provides RL-enhanced agent selection and coordination capabilities
for the AI-CoScientist system, with production-ready features including:

- Gymnasium-compatible RL environment for agent selection
- DQN model for intelligent agent coordination
- Hybrid RL-traditional selector with fallback mechanisms
- A/B testing framework for gradual rollout
- Performance monitoring and safety mechanisms

Components:
- agent_selection_env.py: RL environment for training agent selection policies
- agent_coordination_dqn.py: DQN model implementation with Stable-Baselines3
- hybrid_agent_selector.py: Production-ready hybrid selector with fallbacks
"""

# Import public interface with graceful fallback
try:
    from .hybrid_agent_selector import (
        HybridAgentSelector,
        HybridConfig,
        SelectionStrategy,
        SelectionMetrics,
        PerformanceMonitor,
        create_hybrid_selector
    )
    from .agent_selection_env import AgentSelectionEnvironment, TaskContext
    from .agent_coordination_dqn import RLAgentSelector, DQNConfig

    RL_COMPONENTS_AVAILABLE = True

except ImportError as e:
    # Create mock classes for graceful degradation
    import logging
    logging.warning(f"RL components not fully available: {e}")

    class MockHybridSelector:
        """Mock hybrid selector for fallback when RL unavailable"""
        def __init__(self, agent_pool, config=None):
            self.agent_pool = agent_pool

        async def select_agents(self, task):
            # Fall back to traditional selection
            agent_id = await self.agent_pool.smart_task_routing(task)
            return [agent_id], type('MockMetrics', (), {
                'strategy': 'traditional',
                'selection_time': 0.0,
                'confidence_score': 0.5
            })()

        def get_performance_report(self):
            return {"status": "rl_unavailable", "using_traditional": True}

    HybridAgentSelector = MockHybridSelector
    HybridConfig = type('MockConfig', (), {})()
    create_hybrid_selector = lambda agent_pool, **kwargs: MockHybridSelector(agent_pool)

    RL_COMPONENTS_AVAILABLE = False

__all__ = [
    'HybridAgentSelector',
    'HybridConfig',
    'SelectionStrategy',
    'SelectionMetrics',
    'PerformanceMonitor',
    'create_hybrid_selector',
    'RL_COMPONENTS_AVAILABLE'
]
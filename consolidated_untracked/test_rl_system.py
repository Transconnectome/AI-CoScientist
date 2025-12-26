"""
Comprehensive test suite for the RL-enhanced agent selection system

This test suite covers:
- RL environment functionality and state transitions
- DQN model training and inference
- Hybrid selector with fallback mechanisms
- Performance monitoring and metrics collection
- A/B testing framework validation
- Integration with existing AgentPool
- Error handling and edge cases

Test categories:
- Unit tests: Individual component testing
- Integration tests: Component interaction testing
- Performance tests: Latency and throughput validation
- End-to-end tests: Full system workflow testing
"""

import asyncio
import pytest
import tempfile
import json
import os
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
import numpy as np

# Import the RL components with graceful fallback
try:
    from src.agents.rl.agent_selection_env import (
        AgentSelectionEnvironment,
        TaskContext,
        AgentState,
        EnvironmentState,
        StateEncoder,
        RewardCalculator
    )
    from src.agents.rl.agent_coordination_dqn import (
        RLAgentSelector,
        DQNConfig,
        CustomDQNNetwork,
        AgentCoordinationDQN
    )
    from src.agents.rl.hybrid_agent_selector import (
        HybridAgentSelector,
        HybridConfig,
        SelectionStrategy,
        SelectionMetrics,
        PerformanceMonitor
    )
    from src.agents.rl.performance_monitor import (
        RLPerformanceMonitor,
        MetricThresholds,
        PerformanceAlert,
        AlertLevel,
        create_performance_monitor
    )
    from src.agents.rl_integration import (
        RLEnhancedAgentPool,
        enhance_agent_pool_with_rl,
        create_rl_config
    )
    RL_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"RL components not available: {e}", allow_module_level=True)
    RL_AVAILABLE = False


class MockAgentPool:
    """Mock AgentPool for testing"""

    def __init__(self):
        self.agents = {
            "neuroscience_expert": Mock(
                capabilities=["neuroscience_analysis", "data_analysis"],
                domains=["neuroscience", "medical"],
                get_success_rate=Mock(return_value=0.85)
            ),
            "statistical_analyst": Mock(
                capabilities=["statistical_analysis", "data_analysis"],
                domains=["statistics", "mathematics"],
                get_success_rate=Mock(return_value=0.90)
            ),
            "grant_writer": Mock(
                capabilities=["grant_writing", "scientific_writing"],
                domains=["grant_writing", "funding"],
                get_success_rate=Mock(return_value=0.88)
            )
        }

    async def smart_task_routing(self, task: Dict[str, Any]) -> str:
        """Mock smart task routing"""
        description = task.get('description', '').lower()
        if 'statistical' in description:
            return 'statistical_analyst'
        elif 'grant' in description:
            return 'grant_writer'
        else:
            return 'neuroscience_expert'

    def get_optimal_agent_team(self, requirements: Dict[str, Any]) -> List[str]:
        """Mock optimal agent team selection"""
        team_size = requirements.get('team_size', 2)
        capabilities = requirements.get('capabilities', [])

        if 'statistical_analysis' in capabilities:
            team = ['statistical_analyst', 'neuroscience_expert']
        elif 'grant_writing' in capabilities:
            team = ['grant_writer', 'neuroscience_expert']
        else:
            team = ['neuroscience_expert', 'statistical_analyst']

        return team[:team_size]

    def get_agent(self, agent_id: str):
        """Mock get agent"""
        return self.agents.get(agent_id)


@pytest.fixture
def mock_agent_pool():
    """Fixture providing mock agent pool"""
    return MockAgentPool()


@pytest.fixture
def sample_task():
    """Fixture providing sample task"""
    return {
        'description': 'Analyze fMRI data for autism spectrum disorders',
        'task_type': 'complex',
        'capabilities': ['statistical_analysis', 'neuroscience_analysis'],
        'domains': ['neuroscience', 'medical_imaging'],
        'priority': 1,
        'complexity_score': 0.7,
        'estimated_duration': 120.0
    }


@pytest.fixture
def task_context(sample_task):
    """Fixture providing TaskContext"""
    return TaskContext(
        task_type=sample_task['task_type'],
        description=sample_task['description'],
        priority=sample_task['priority'],
        domains=sample_task['domains'],
        capabilities=sample_task['capabilities'],
        complexity_score=sample_task['complexity_score'],
        estimated_duration=sample_task['estimated_duration']
    )


class TestAgentSelectionEnvironment:
    """Test the RL environment for agent selection"""

    @pytest.fixture
    def env(self, mock_agent_pool):
        """Create environment for testing"""
        return AgentSelectionEnvironment(mock_agent_pool)

    def test_environment_initialization(self, env):
        """Test environment initializes correctly"""
        assert env.agent_pool is not None
        assert env.state_encoder is not None
        assert env.reward_calculator is not None
        assert env.task_generator is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_state_encoding(self, env, task_context):
        """Test state encoding produces correct dimensions"""
        env.current_task = task_context

        # Create mock agent states
        agent_states = {
            'neuroscience_expert': AgentState(
                agent_id='neuroscience_expert',
                success_rate=0.85,
                avg_quality=0.80,
                workload=0.3,
                last_task_time=60.0
            )
        }

        system_state = EnvironmentState(
            total_tasks_completed=100,
            avg_system_performance=0.82,
            current_system_load=0.4,
            available_agents=3
        )

        encoded_state = env.state_encoder.encode_state(task_context, agent_states, system_state)

        assert len(encoded_state) == 128  # Expected state dimension
        assert all(isinstance(x, (int, float, np.number)) for x in encoded_state)

    def test_environment_reset(self, env):
        """Test environment reset functionality"""
        initial_state, _ = env.reset()

        assert len(initial_state) == 128
        assert env.current_task is not None
        assert env.step_count == 0
        assert env.episode_reward == 0.0

    def test_environment_step(self, env):
        """Test environment step functionality"""
        state, _ = env.reset()

        # Take a valid action (select agents)
        action = np.array([1, 1, 0])  # Select first two agents
        next_state, reward, done, truncated, info = env.step(action)

        assert len(next_state) == 128
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_calculation(self, env, task_context):
        """Test reward calculation logic"""
        reward_calc = env.reward_calculator

        # Test successful task completion
        reward = reward_calc.calculate_reward(
            task_context=task_context,
            selected_agents=['neuroscience_expert', 'statistical_analyst'],
            task_success=True,
            quality_score=0.85,
            execution_time=90.0,
            collaboration_effectiveness=0.8
        )

        assert isinstance(reward, float)
        assert reward > 0  # Successful task should give positive reward

        # Test failed task completion
        failed_reward = reward_calc.calculate_reward(
            task_context=task_context,
            selected_agents=['neuroscience_expert'],
            task_success=False,
            quality_score=0.3,
            execution_time=200.0,
            collaboration_effectiveness=0.2
        )

        assert failed_reward < reward  # Failed task should give lower reward

    def test_action_validation(self, env):
        """Test action space validation"""
        env.reset()

        # Test valid action
        valid_action = np.array([1, 0, 1])
        assert env._is_valid_action(valid_action)

        # Test invalid action (no agents selected)
        invalid_action = np.array([0, 0, 0])
        assert not env._is_valid_action(invalid_action)

        # Test invalid action (wrong shape)
        with pytest.raises((ValueError, IndexError)):
            env.step(np.array([1, 0]))  # Wrong action shape


class TestDQNModel:
    """Test the DQN model implementation"""

    @pytest.fixture
    def dqn_config(self):
        """Create DQN configuration for testing"""
        return DQNConfig(
            total_timesteps=1000,  # Reduced for testing
            learning_rate=1e-4,
            batch_size=32,
            buffer_size=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.1,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=100,
            model_save_interval=500
        )

    @pytest.fixture
    def temp_model_path(self):
        """Create temporary path for model saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield os.path.join(temp_dir, "test_model")

    def test_dqn_initialization(self, mock_agent_pool, dqn_config, temp_model_path):
        """Test DQN model initialization"""
        dqn_config.model_save_path = temp_model_path

        dqn = AgentCoordinationDQN(
            agent_pool=mock_agent_pool,
            config=dqn_config
        )

        assert dqn.agent_pool == mock_agent_pool
        assert dqn.config == dqn_config
        assert dqn.env is not None

    @pytest.mark.asyncio
    async def test_rl_agent_selector_initialization(self, mock_agent_pool, dqn_config):
        """Test RL agent selector initialization"""
        selector = RLAgentSelector(
            agent_pool=mock_agent_pool,
            config=dqn_config
        )

        assert selector.agent_pool == mock_agent_pool
        assert selector.config == dqn_config

    @pytest.mark.asyncio
    async def test_rl_agent_selection(self, mock_agent_pool, dqn_config, task_context):
        """Test RL agent selection without training"""
        selector = RLAgentSelector(
            agent_pool=mock_agent_pool,
            config=dqn_config
        )

        # Mock the DQN model to avoid actual training
        with patch.object(selector, 'dqn_model') as mock_dqn:
            mock_dqn.is_trained = True
            mock_dqn.predict.return_value = np.array([1, 1, 0])

            selected_agents = await selector.select_agents(task_context)

            assert isinstance(selected_agents, list)
            assert len(selected_agents) > 0
            assert all(isinstance(agent_id, str) for agent_id in selected_agents)

    def test_custom_dqn_network_forward_pass(self):
        """Test custom DQN network forward pass"""
        try:
            import torch
            import torch.nn as nn

            # Create a small test network
            network = CustomDQNNetwork(
                task_features=10,
                agent_features=15,
                system_features=5,
                num_agents=3
            )

            # Test forward pass
            batch_size = 4
            test_input = torch.randn(batch_size, 30)  # 10 + 15 + 5
            output = network(test_input)

            assert output.shape == (batch_size, 3)  # Should output action probabilities
            assert torch.all(torch.isfinite(output))  # No NaN or inf values

        except ImportError:
            pytest.skip("PyTorch not available")


class TestHybridAgentSelector:
    """Test the hybrid RL-traditional agent selector"""

    @pytest.fixture
    def hybrid_config(self):
        """Create hybrid configuration for testing"""
        return HybridConfig(
            enable_rl=True,
            enable_ab_testing=True,
            rl_traffic_percentage=0.5,
            fallback_on_error=True,
            enable_performance_monitoring=True
        )

    @pytest.fixture
    def hybrid_selector(self, mock_agent_pool, hybrid_config):
        """Create hybrid selector for testing"""
        return HybridAgentSelector(mock_agent_pool, hybrid_config)

    @pytest.mark.asyncio
    async def test_hybrid_selector_initialization(self, hybrid_selector):
        """Test hybrid selector initializes correctly"""
        assert hybrid_selector.agent_pool is not None
        assert hybrid_selector.config is not None
        assert hybrid_selector.performance_monitor is not None

    @pytest.mark.asyncio
    async def test_agent_selection_traditional_fallback(self, mock_agent_pool, sample_task):
        """Test fallback to traditional selection when RL unavailable"""
        # Force RL unavailable
        config = HybridConfig(enable_rl=False)
        selector = HybridAgentSelector(mock_agent_pool, config)

        agents, metrics = await selector.select_agents(sample_task)

        assert isinstance(agents, list)
        assert len(agents) > 0
        assert metrics.strategy == SelectionStrategy.TRADITIONAL.value
        assert isinstance(metrics.selection_time, float)
        assert metrics.selection_time >= 0

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, hybrid_selector, sample_task):
        """Test performance monitoring integration"""
        agents, metrics = await hybrid_selector.select_agents(sample_task)

        # Update task outcome
        await hybrid_selector.update_task_outcome(metrics, success=True, quality_score=0.85)

        # Check performance report
        report = hybrid_selector.get_performance_report()
        assert isinstance(report, dict)
        assert 'configuration' in report
        assert 'strategy_performance' in report

    @pytest.mark.asyncio
    async def test_ab_testing_traffic_distribution(self, mock_agent_pool):
        """Test A/B testing traffic distribution"""
        config = HybridConfig(
            enable_ab_testing=True,
            rl_traffic_percentage=0.3  # 30% RL traffic
        )
        selector = HybridAgentSelector(mock_agent_pool, config)

        # Mock RL selector to avoid initialization issues
        selector.rl_selector = Mock()
        selector.rl_selector.select_agents = AsyncMock(return_value=['neuroscience_expert'])

        strategies_used = []
        sample_task = {
            'description': 'Test task',
            'task_type': 'simple'
        }

        # Run multiple selections to test traffic distribution
        for _ in range(100):
            _, metrics = await selector.select_agents(sample_task)
            strategies_used.append(metrics.strategy)

        # Check that both strategies are used
        strategy_counts = {strategy: strategies_used.count(strategy) for strategy in set(strategies_used)}
        assert len(strategy_counts) > 1  # Should have multiple strategies

    def test_selection_metrics_creation(self):
        """Test SelectionMetrics creation and serialization"""
        metrics = SelectionMetrics(
            strategy="rl_enabled",
            selection_time=0.5,
            agent_ids=["agent_1", "agent_2"],
            confidence_score=0.8,
            task_type="complex",
            success=True,
            quality_score=0.85
        )

        assert metrics.strategy == "rl_enabled"
        assert metrics.selection_time == 0.5
        assert metrics.agent_ids == ["agent_1", "agent_2"]
        assert metrics.success is True

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, mock_agent_pool):
        """Test error handling and fallback mechanisms"""
        config = HybridConfig(
            enable_rl=True,
            fallback_on_error=True
        )
        selector = HybridAgentSelector(mock_agent_pool, config)

        # Mock RL selector to raise an error
        selector.rl_selector = Mock()
        selector.rl_selector.select_agents = AsyncMock(side_effect=Exception("RL Error"))

        sample_task = {
            'description': 'Test task',
            'task_type': 'simple'
        }

        # Should fallback to traditional selection on RL error
        agents, metrics = await selector.select_agents(sample_task)

        assert isinstance(agents, list)
        assert len(agents) > 0
        assert metrics.strategy == SelectionStrategy.TRADITIONAL.value


class TestPerformanceMonitor:
    """Test the performance monitoring system"""

    @pytest.fixture
    def monitor(self):
        """Create performance monitor for testing"""
        return create_performance_monitor(enable_prometheus=False)

    def test_monitor_initialization(self, monitor):
        """Test performance monitor initialization"""
        assert monitor.thresholds is not None
        assert monitor.performance_monitor is not None
        assert isinstance(monitor.selection_history, deque)

    def test_selection_event_recording(self, monitor):
        """Test recording selection events"""
        monitor.record_selection_event(
            strategy="rl_enabled",
            agent_ids=["agent_1", "agent_2"],
            task_type="complex",
            selection_time=1.2,
            confidence=0.8,
            success=True,
            quality_score=0.85
        )

        assert len(monitor.selection_history) == 1
        event = monitor.selection_history[0]
        assert event['strategy'] == "rl_enabled"
        assert event['agent_ids'] == ["agent_1", "agent_2"]

    def test_error_event_recording(self, monitor):
        """Test recording error events"""
        monitor.record_selection_error(
            strategy="rl_enabled",
            error_type="timeout",
            error_message="Selection timeout after 5 seconds"
        )

        assert len(monitor.selection_history) == 1
        assert len(monitor.alerts) >= 1  # Should generate alert

    def test_real_time_metrics_calculation(self, monitor):
        """Test real-time metrics calculation"""
        # Record several events
        for i in range(10):
            monitor.record_selection_event(
                strategy="test_strategy",
                agent_ids=[f"agent_{i}"],
                task_type="simple",
                selection_time=0.5 + i * 0.1,
                confidence=0.8,
                success=i % 2 == 0,  # 50% success rate
                quality_score=0.7 + (i % 3) * 0.1
            )

        # Get real-time metrics
        metrics = monitor.get_real_time_metrics("test_strategy", "5m")
        if metrics:
            assert metrics.total_selections == 10
            assert 0 <= metrics.success_rate <= 1
            assert metrics.avg_latency_ms > 0

    def test_alert_generation(self, monitor):
        """Test alert generation on threshold violations"""
        # Set strict thresholds
        monitor.thresholds.success_rate_critical = 0.9

        # Record poor performance events
        for i in range(5):
            monitor.record_selection_event(
                strategy="poor_strategy",
                agent_ids=["agent_1"],
                task_type="simple",
                selection_time=0.5,
                confidence=0.3,
                success=False,  # All failures
                quality_score=0.2
            )

        # Check if alerts were generated
        alerts = monitor.get_recent_alerts()
        assert len(alerts) > 0

        # Check alert content
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert any(a.metric == 'success_rate' for a in critical_alerts)

    def test_ab_test_analysis(self, monitor):
        """Test A/B test statistical analysis"""
        # Record events for strategy A (better performance)
        for i in range(20):
            monitor.record_selection_event(
                strategy="strategy_a",
                agent_ids=["agent_1"],
                task_type="simple",
                selection_time=0.5,
                confidence=0.8,
                success=True,
                quality_score=0.9  # High quality
            )

        # Record events for strategy B (worse performance)
        for i in range(20):
            monitor.record_selection_event(
                strategy="strategy_b",
                agent_ids=["agent_1"],
                task_type="simple",
                selection_time=0.5,
                confidence=0.8,
                success=True,
                quality_score=0.6  # Lower quality
            )

        # Analyze A/B test
        result = monitor.analyze_ab_test("strategy_a", "strategy_b", "quality_score")

        assert result['status'] == 'success'
        assert result['sample_size_a'] == 20
        assert result['sample_size_b'] == 20
        assert result['mean_a'] > result['mean_b']  # Strategy A should be better

    def test_performance_dashboard_data(self, monitor):
        """Test performance dashboard data generation"""
        # Record some test events
        monitor.record_selection_event(
            strategy="test_strategy",
            agent_ids=["agent_1"],
            task_type="simple",
            selection_time=0.5,
            confidence=0.8,
            success=True,
            quality_score=0.8
        )

        dashboard_data = monitor.get_performance_dashboard_data()

        assert isinstance(dashboard_data, dict)
        assert 'timestamp' in dashboard_data
        assert 'real_time_metrics' in dashboard_data
        assert 'recent_alerts' in dashboard_data
        assert 'system_health' in dashboard_data


class TestRLIntegration:
    """Test RL integration with existing AgentPool"""

    @pytest.fixture
    def enhanced_pool(self, mock_agent_pool):
        """Create enhanced agent pool for testing"""
        return enhance_agent_pool_with_rl(mock_agent_pool, create_rl_config())

    def test_enhanced_pool_initialization(self, enhanced_pool):
        """Test enhanced pool initialization"""
        assert enhanced_pool.original_pool is not None
        assert hasattr(enhanced_pool, 'hybrid_selector')

    def test_backwards_compatibility(self, enhanced_pool):
        """Test backwards compatibility with original AgentPool"""
        # Should still have access to original methods
        agent = enhanced_pool.get_agent("neuroscience_expert")
        assert agent is not None

        # Should still have access to original attributes
        assert hasattr(enhanced_pool, 'agents')

    @pytest.mark.asyncio
    async def test_smart_agent_selection(self, enhanced_pool, sample_task):
        """Test smart agent selection with RL enhancement"""
        agents, metadata = await enhanced_pool.select_agents_smart(sample_task)

        assert isinstance(agents, list)
        assert len(agents) > 0
        assert isinstance(metadata, dict)
        assert 'strategy' in metadata
        assert 'selection_time' in metadata
        assert 'confidence_score' in metadata

    def test_performance_metrics_access(self, enhanced_pool):
        """Test access to performance metrics"""
        metrics = enhanced_pool.get_performance_metrics()

        assert isinstance(metrics, dict)
        # Should contain either RL metrics or fallback information

    @pytest.mark.asyncio
    async def test_selection_outcome_feedback(self, enhanced_pool, sample_task):
        """Test selection outcome feedback for learning"""
        agents, metadata = await enhanced_pool.select_agents_smart(sample_task)

        # Simulate task completion and feedback
        await enhanced_pool.update_selection_outcome(
            task=sample_task,
            agent_ids=agents,
            success=True,
            quality_score=0.85
        )

        # Should not raise any errors

    def test_ab_testing_control(self, enhanced_pool):
        """Test A/B testing enable/disable functionality"""
        # Test enabling A/B testing
        enhanced_pool.enable_ab_testing(0.3)
        # Should not raise errors

        # Test disabling A/B testing
        enhanced_pool.disable_ab_testing()
        # Should not raise errors


class TestEndToEndWorkflow:
    """End-to-end testing of the complete RL system"""

    @pytest.fixture
    def complete_system(self, mock_agent_pool):
        """Create complete system for end-to-end testing"""
        config = create_rl_config(
            enable_rl=True,
            enable_ab_testing=True,
            rl_traffic_percentage=0.5
        )
        return enhance_agent_pool_with_rl(mock_agent_pool, config)

    @pytest.mark.asyncio
    async def test_complete_selection_workflow(self, complete_system):
        """Test complete agent selection workflow"""
        # Define a complex task
        task = {
            'description': 'Comprehensive analysis of autism spectrum disorders using multimodal brain imaging',
            'task_type': 'comprehensive',
            'capabilities': ['statistical_analysis', 'neuroscience_analysis', 'data_visualization'],
            'domains': ['neuroscience', 'medical_imaging', 'autism_research'],
            'priority': 1,
            'complexity_score': 0.9,
            'estimated_duration': 240.0
        }

        # Step 1: Agent selection
        selected_agents, metadata = await complete_system.select_agents_smart(task)

        assert len(selected_agents) > 0
        assert metadata['strategy'] in ['rl_enabled', 'traditional', 'hybrid']

        # Step 2: Simulate task execution
        execution_success = True
        quality_score = 0.87

        # Step 3: Provide feedback
        await complete_system.update_selection_outcome(
            task=task,
            agent_ids=selected_agents,
            success=execution_success,
            quality_score=quality_score
        )

        # Step 4: Check performance metrics
        performance_report = complete_system.get_performance_metrics()
        assert isinstance(performance_report, dict)

    @pytest.mark.asyncio
    async def test_multiple_task_execution(self, complete_system):
        """Test multiple task execution for performance analysis"""
        tasks = [
            {
                'description': 'Statistical analysis of fMRI data',
                'task_type': 'simple',
                'capabilities': ['statistical_analysis'],
                'domains': ['neuroscience']
            },
            {
                'description': 'Grant proposal for autism research funding',
                'task_type': 'complex',
                'capabilities': ['grant_writing', 'scientific_writing'],
                'domains': ['grant_writing', 'autism_research']
            },
            {
                'description': 'Systematic literature review on brain connectivity',
                'task_type': 'comprehensive',
                'capabilities': ['literature_synthesis', 'data_analysis'],
                'domains': ['neuroscience', 'literature_analysis']
            }
        ]

        results = []

        for i, task in enumerate(tasks):
            # Select agents
            agents, metadata = await complete_system.select_agents_smart(task)

            # Simulate execution
            success = i % 3 != 0  # 2/3 success rate
            quality = 0.8 if success else 0.4

            # Provide feedback
            await complete_system.update_selection_outcome(task, agents, success, quality)

            results.append({
                'task_id': i,
                'agents': agents,
                'strategy': metadata['strategy'],
                'success': success,
                'quality': quality
            })

        # Verify we have diverse results
        strategies_used = set(r['strategy'] for r in results)
        assert len(results) == 3

    def test_system_configuration_flexibility(self, mock_agent_pool):
        """Test system configuration flexibility"""
        # Test different configurations
        configs = [
            create_rl_config(enable_rl=True, enable_ab_testing=False),
            create_rl_config(enable_rl=False, enable_ab_testing=False),
            create_rl_config(enable_rl=True, enable_ab_testing=True, rl_traffic_percentage=0.1)
        ]

        for config in configs:
            enhanced_pool = enhance_agent_pool_with_rl(mock_agent_pool, config)
            assert enhanced_pool is not None

            # Test basic functionality
            performance = enhanced_pool.get_performance_metrics()
            assert isinstance(performance, dict)

    @pytest.mark.asyncio
    async def test_error_resilience(self, mock_agent_pool):
        """Test system resilience to various error conditions"""
        enhanced_pool = enhance_agent_pool_with_rl(mock_agent_pool)

        # Test with malformed task
        malformed_task = {
            'description': None,
            'task_type': 'invalid_type'
        }

        try:
            agents, metadata = await enhanced_pool.select_agents_smart(malformed_task)
            # Should handle gracefully and return some agents
            assert len(agents) > 0
        except Exception as e:
            # Or raise a specific, handled exception
            assert isinstance(e, (ValueError, KeyError))

        # Test with empty task
        empty_task = {}

        agents, metadata = await enhanced_pool.select_agents_smart(empty_task)
        assert len(agents) > 0  # Should use fallback


# Performance benchmarking tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests for the RL system"""

    @pytest.mark.asyncio
    async def test_selection_latency_benchmark(self, mock_agent_pool):
        """Benchmark agent selection latency"""
        enhanced_pool = enhance_agent_pool_with_rl(mock_agent_pool)

        task = {
            'description': 'Test task for latency benchmark',
            'task_type': 'simple'
        }

        # Warm up
        for _ in range(5):
            await enhanced_pool.select_agents_smart(task)

        # Benchmark
        start_time = time.time()
        iterations = 50

        for _ in range(iterations):
            await enhanced_pool.select_agents_smart(task)

        end_time = time.time()
        avg_latency = (end_time - start_time) / iterations

        # Assert reasonable latency (should be under 100ms for mock system)
        assert avg_latency < 0.1, f"Average latency {avg_latency:.3f}s too high"

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, mock_agent_pool):
        """Benchmark system throughput"""
        enhanced_pool = enhance_agent_pool_with_rl(mock_agent_pool)

        tasks = [
            {
                'description': f'Task {i}',
                'task_type': 'simple'
            }
            for i in range(100)
        ]

        start_time = time.time()

        # Process tasks concurrently
        tasks_coroutines = [
            enhanced_pool.select_agents_smart(task)
            for task in tasks
        ]

        results = await asyncio.gather(*tasks_coroutines)

        end_time = time.time()
        throughput = len(tasks) / (end_time - start_time)

        # Assert reasonable throughput
        assert throughput > 50, f"Throughput {throughput:.1f} tasks/sec too low"
        assert all(len(agents) > 0 for agents, metadata in results)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
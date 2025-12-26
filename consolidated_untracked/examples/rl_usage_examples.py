"""
RL System Usage Examples

Comprehensive examples demonstrating how to use the RL-enhanced agent selection
system in various scenarios. These examples show:

- Basic agent selection with RL enhancement
- Performance monitoring and metrics collection
- A/B testing setup and analysis
- Safety mechanism integration
- Continuous learning workflow
- Migration and rollback procedures
- Custom integration patterns

Use these examples as templates for integrating RL capabilities
into your AI-CoScientist workflows.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_rl_integration():
    """
    Example 1: Basic RL Integration with Existing Agent Pool

    Shows how to enhance an existing agent pool with RL capabilities
    and use smart agent selection.
    """
    print("=" * 60)
    print("Example 1: Basic RL Integration")
    print("=" * 60)

    try:
        from src.agents.rl_integration import enhance_agent_pool_with_rl, create_rl_config
        from src.agents.pool import AgentPool  # Assuming this exists

        # Mock agent pool for demonstration
        class MockAgentPool:
            def __init__(self):
                self.agents = {
                    "neuroscience_expert": type('Agent', (), {
                        'capabilities': ['neuroscience_analysis', 'data_analysis'],
                        'domains': ['neuroscience', 'medical'],
                        'get_success_rate': lambda: 0.85
                    })(),
                    "statistical_analyst": type('Agent', (), {
                        'capabilities': ['statistical_analysis', 'data_analysis'],
                        'domains': ['statistics', 'mathematics'],
                        'get_success_rate': lambda: 0.90
                    })(),
                    "grant_writer": type('Agent', (), {
                        'capabilities': ['grant_writing', 'scientific_writing'],
                        'domains': ['grant_writing', 'funding'],
                        'get_success_rate': lambda: 0.88
                    })()
                }

            async def smart_task_routing(self, task):
                # Simple routing logic
                if 'statistical' in task.get('description', '').lower():
                    return 'statistical_analyst'
                elif 'grant' in task.get('description', '').lower():
                    return 'grant_writer'
                else:
                    return 'neuroscience_expert'

        # Create original agent pool
        original_pool = MockAgentPool()
        print("✓ Created original agent pool")

        # Create RL configuration
        rl_config = create_rl_config(
            enable_rl=True,
            enable_ab_testing=True,
            rl_traffic_percentage=0.2,  # Start with 20% RL traffic
            confidence_threshold=0.7
        )
        print("✓ Created RL configuration")

        # Enhance agent pool with RL
        enhanced_pool = enhance_agent_pool_with_rl(original_pool, rl_config)
        print("✓ Enhanced agent pool with RL capabilities")

        # Example task
        task = {
            'description': 'Analyze fMRI data for autism spectrum disorders using advanced statistical methods',
            'task_type': 'complex',
            'capabilities': ['statistical_analysis', 'neuroscience_analysis'],
            'domains': ['neuroscience', 'medical_imaging'],
            'priority': 1,
            'complexity_score': 0.8
        }

        print(f"\nTask: {task['description']}")

        # Run agent selection asynchronously
        async def run_selection():
            agents, metadata = await enhanced_pool.select_agents_smart(task)
            return agents, metadata

        # Execute selection
        agents, metadata = asyncio.run(run_selection())

        print(f"Selected agents: {agents}")
        print(f"Selection strategy: {metadata['strategy']}")
        print(f"Selection time: {metadata['selection_time']:.3f}s")
        print(f"Confidence score: {metadata['confidence_score']:.2f}")
        print(f"RL enabled: {metadata['rl_enabled']}")

        print("✓ Example 1 completed successfully")

    except ImportError as e:
        print(f"✗ RL components not available: {e}")
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")

    print()


async def example_2_performance_monitoring():
    """
    Example 2: Performance Monitoring and Metrics Collection

    Demonstrates how to set up performance monitoring and collect
    metrics for RL-enhanced agent selection.
    """
    print("=" * 60)
    print("Example 2: Performance Monitoring")
    print("=" * 60)

    try:
        from src.agents.rl.performance_monitor import create_performance_monitor

        # Create performance monitor
        monitor = create_performance_monitor(enable_prometheus=False)
        print("✓ Created performance monitor")

        # Start background monitoring
        await monitor.start_background_monitoring()
        print("✓ Started background monitoring")

        # Simulate agent selection events
        print("\nSimulating agent selection events...")

        strategies = ['rl_enabled', 'traditional', 'hybrid']
        task_types = ['simple', 'complex', 'comprehensive']

        for i in range(20):
            # Simulate selection event
            strategy = strategies[i % 3]
            task_type = task_types[i % 3]
            agents = [f"agent_{(i % 3) + 1}"]

            # Simulate different performance characteristics
            if strategy == 'rl_enabled':
                latency = 0.8 + (i % 5) * 0.1
                quality = 0.85 + (i % 4) * 0.03
                success_rate = 0.92
            elif strategy == 'traditional':
                latency = 0.5 + (i % 3) * 0.1
                quality = 0.78 + (i % 3) * 0.02
                success_rate = 0.85
            else:  # hybrid
                latency = 0.6 + (i % 4) * 0.1
                quality = 0.82 + (i % 3) * 0.02
                success_rate = 0.88

            success = (i % 10) < (success_rate * 10)  # Probabilistic success

            monitor.record_selection_event(
                strategy=strategy,
                agent_ids=agents,
                task_type=task_type,
                selection_time=latency,
                confidence=0.7 + (i % 5) * 0.05,
                success=success,
                quality_score=quality if success else quality * 0.6
            )

        print(f"✓ Recorded {20} selection events")

        # Wait a moment for processing
        await asyncio.sleep(1)

        # Get performance metrics
        print("\nPerformance Metrics:")
        for strategy in strategies:
            metrics = monitor.get_real_time_metrics(strategy, '5m')
            if metrics and metrics.total_selections > 0:
                print(f"\n{strategy.upper()}:")
                print(f"  Success Rate: {metrics.success_rate:.1%}")
                print(f"  Avg Latency: {metrics.avg_latency_ms:.0f}ms")
                print(f"  P95 Latency: {metrics.p95_latency_ms:.0f}ms")
                print(f"  Avg Quality: {metrics.avg_quality_score:.2f}")
                print(f"  Total Selections: {metrics.total_selections}")

        # A/B test analysis
        print("\nA/B Test Analysis:")
        ab_result = monitor.analyze_ab_test('rl_enabled', 'traditional', 'quality_score')
        if ab_result['status'] == 'success':
            print(f"  RL Mean Quality: {ab_result['mean_a']:.3f}")
            print(f"  Traditional Mean Quality: {ab_result['mean_b']:.3f}")
            print(f"  Improvement: {ab_result['improvement_percentage']:.1f}%")
            print(f"  Statistically Significant: {ab_result['is_statistically_significant']}")
            print(f"  Recommendation: {ab_result['recommendation']}")

        # Dashboard data
        dashboard_data = monitor.get_performance_dashboard_data()
        print(f"\nDashboard Summary:")
        print(f"  Total Selections (1h): {dashboard_data['system_health']['total_selections_1h']}")
        print(f"  Error Rate (1h): {dashboard_data['system_health']['error_rate_1h']:.2%}")
        print(f"  Strategies Active: {dashboard_data['system_health']['strategies_active']}")

        # Stop monitoring
        await monitor.stop_background_monitoring()
        print("✓ Stopped background monitoring")

        print("✓ Example 2 completed successfully")

    except ImportError as e:
        print(f"✗ RL components not available: {e}")
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")

    print()


async def example_3_safety_mechanisms():
    """
    Example 3: Safety Mechanisms and Circuit Breakers

    Shows how to set up and use safety mechanisms including
    circuit breakers, rate limiting, and emergency procedures.
    """
    print("=" * 60)
    print("Example 3: Safety Mechanisms")
    print("=" * 60)

    try:
        from src.agents.rl.safety_mechanisms import create_safety_manager

        # Create safety manager with custom config
        custom_config = {
            'circuit_failure_threshold': 3,  # Trip after 3 failures
            'max_error_rate': 0.10,  # 10% error rate threshold
            'max_latency_p95_ms': 2000.0,  # 2 second latency threshold
            'max_concurrent_requests': 50
        }

        safety_manager = create_safety_manager(
            enable_circuit_breakers=True,
            enable_rate_limiting=True,
            custom_config=custom_config
        )
        print("✓ Created safety manager with custom configuration")

        # Start monitoring
        await safety_manager.start_monitoring()
        print("✓ Started safety monitoring")

        # Example 1: Successful operations
        print("\nTesting normal operations...")

        async def normal_operation():
            await asyncio.sleep(0.1)  # Simulate work
            return "Operation successful"

        for i in range(5):
            try:
                result = await safety_manager.safe_execute(
                    "agent_selection",
                    normal_operation
                )
                print(f"  Operation {i+1}: {result}")
            except Exception as e:
                print(f"  Operation {i+1}: Failed - {e}")

        # Example 2: Circuit breaker with failures
        print("\nTesting circuit breaker with failures...")

        async def failing_operation():
            await asyncio.sleep(0.05)
            raise Exception("Simulated failure")

        async def fallback_operation():
            return "Fallback response"

        # This should trip the circuit breaker after 3 failures
        for i in range(6):
            try:
                result = await safety_manager.safe_execute(
                    "failing_service",
                    failing_operation,
                    fallback_func=fallback_operation
                )
                print(f"  Failing operation {i+1}: {result}")
            except Exception as e:
                print(f"  Failing operation {i+1}: {e}")

        # Example 3: Rate limiting
        print("\nTesting rate limiting...")

        rate_limiter = safety_manager.rate_limiters.get("agent_selection")
        if rate_limiter:
            for i in range(10):
                acquired = await rate_limiter.acquire()
                status = "✓" if acquired else "✗ Rate limited"
                print(f"  Request {i+1}: {status}")
        else:
            print("  Rate limiter not available")

        # Safety status
        print("\nSafety Status:")
        status = safety_manager.get_safety_status()
        print(f"  Current Safety Level: {status['current_safety_level']}")
        print(f"  Resource Safe: {status['resource_safe']}")
        print(f"  Performance Healthy: {status['performance_healthy']}")
        print(f"  Total Incidents: {status['total_incidents']}")

        # Circuit breaker stats
        print("\nCircuit Breaker Stats:")
        for name, stats in status['circuit_breakers'].items():
            print(f"  {name}:")
            print(f"    State: {stats['state']}")
            print(f"    Error Rate: {stats['error_rate']:.2%}")
            print(f"    Total Requests: {stats['total_requests']}")

        # Stop monitoring
        await safety_manager.stop_monitoring()
        print("✓ Stopped safety monitoring")

        print("✓ Example 3 completed successfully")

    except ImportError as e:
        print(f"✗ RL components not available: {e}")
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")

    print()


async def example_4_continuous_learning():
    """
    Example 4: Continuous Learning Pipeline

    Demonstrates how to set up continuous learning to improve
    agent selection performance over time.
    """
    print("=" * 60)
    print("Example 4: Continuous Learning")
    print("=" * 60)

    try:
        from src.agents.rl.continuous_learning import create_continuous_learning_pipeline
        from src.agents.rl.continuous_learning import LearningMode

        # Mock agent pool and RL selector
        class MockAgentPool:
            pass

        class MockRLSelector:
            async def select_agents(self, task_context):
                return ['neuroscience_expert', 'statistical_analyst']

        agent_pool = MockAgentPool()
        rl_selector = MockRLSelector()

        # Create learning pipeline
        pipeline = create_continuous_learning_pipeline(
            agent_pool,
            rl_selector,
            learning_mode=LearningMode.HYBRID,
            enable_human_feedback=True
        )
        print("✓ Created continuous learning pipeline")

        # Start learning
        await pipeline.start_continuous_learning()
        print("✓ Started continuous learning")

        # Simulate task experiences
        print("\nSimulating task experiences...")

        for i in range(30):
            task_context = {
                'task_type': 'complex' if i % 3 == 0 else 'simple',
                'description': f'Task {i}: Neuroscience data analysis',
                'capabilities': ['statistical_analysis', 'neuroscience_analysis'],
                'domains': ['neuroscience', 'statistics'],
                'complexity_score': 0.3 + (i % 7) * 0.1,
                'priority': 1 + (i % 3)
            }

            selected_agents = ['neuroscience_expert', 'statistical_analyst']

            # Simulate task outcome (higher success rate for complex tasks)
            task_success = (i % 5 != 0) if task_context['task_type'] == 'simple' else (i % 4 != 0)
            quality_score = 0.8 + (i % 4) * 0.05 if task_success else 0.4 + (i % 3) * 0.1
            execution_time = 60.0 + (i % 10) * 15.0

            await pipeline.add_experience(
                task_context=task_context,
                selected_agents=selected_agents,
                task_outcome=task_success,
                quality_score=quality_score,
                execution_time=execution_time
            )

            # Add human feedback occasionally
            if i % 8 == 0:
                pipeline.add_human_feedback(
                    task_id=f"task_{i}",
                    agent_selection_quality=quality_score + 0.05,
                    comments=f"Good agent selection for task {i}",
                    confidence=0.8 + (i % 3) * 0.05
                )

        print(f"✓ Added {30} experiences to learning pipeline")

        # Get learning status
        status = pipeline.get_learning_status()
        print(f"\nLearning Status:")
        print(f"  Learning Active: {status['learning_active']}")
        print(f"  Learning Mode: {status['learning_mode']}")
        print(f"  Total Experiences: {status['total_experiences']}")
        print(f"  Buffer Size: {status['buffer_stats']['size']}")
        print(f"  Avg Quality: {status['buffer_stats'].get('avg_quality', 0):.3f}")
        print(f"  Should Retrain: {status['should_retrain']}")

        # Trigger retraining (forced for demo)
        print("\nTriggering model retraining...")
        retrain_success = await pipeline.trigger_periodic_retrain(force=True)
        print(f"✓ Retraining {'successful' if retrain_success else 'failed'}")

        # Get model history
        history = pipeline.get_model_history()
        print(f"\nModel History: {len(history)} versions")
        for version in history[:3]:  # Show first 3 versions
            print(f"  {version['version_id']}: {version['validation_status']} "
                  f"({version['training_data_size']} experiences)")

        # Stop learning
        await pipeline.stop_continuous_learning()
        print("✓ Stopped continuous learning")

        print("✓ Example 4 completed successfully")

    except ImportError as e:
        print(f"✗ RL components not available: {e}")
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")

    print()


async def example_5_traffic_migration():
    """
    Example 5: Gradual Traffic Migration

    Shows how to gradually migrate traffic from traditional to RL
    agent selection with safety monitoring and rollback capabilities.
    """
    print("=" * 60)
    print("Example 5: Traffic Migration")
    print("=" * 60)

    try:
        from src.agents.rl.traffic_migration import create_migration_controller
        from src.agents.rl.traffic_migration import MigrationSchedule, PerformanceThresholds
        from src.agents.rl.hybrid_agent_selector import create_hybrid_selector
        from src.agents.rl.performance_monitor import create_performance_monitor

        # Create mock components
        class MockAgentPool:
            pass

        class MockHybridSelector:
            def __init__(self):
                self.current_rl_percentage = 0.0
                self.rl_selector = True

        class MockPerformanceMonitor:
            def __init__(self):
                self.metrics_data = {
                    "traditional": {
                        "success_rate": 0.85,
                        "avg_latency_ms": 900,
                        "p95_latency_ms": 1300,
                        "avg_quality_score": 0.78,
                        "total_selections": 100
                    },
                    "rl_enabled": {
                        "success_rate": 0.88,
                        "avg_latency_ms": 850,
                        "p95_latency_ms": 1200,
                        "avg_quality_score": 0.82,
                        "total_selections": 0
                    }
                }

            def get_real_time_metrics(self, strategy, window):
                data = self.metrics_data.get(strategy, {})
                if not data:
                    return None
                return type('Metrics', (), data)()

        # Create components
        agent_pool = MockAgentPool()
        hybrid_selector = MockHybridSelector()
        performance_monitor = MockPerformanceMonitor()

        controller = create_migration_controller(
            hybrid_selector,
            performance_monitor
        )
        print("✓ Created migration controller")

        # Configure migration schedule
        schedule = MigrationSchedule(
            phase_duration_minutes=1,  # Short for demo
            canary_percentage=10.0,
            ramp_increments=[20.0, 40.0, 60.0, 80.0, 100.0],
            evaluation_window_minutes=0.5,
            min_samples_per_phase=5
        )

        thresholds = PerformanceThresholds(
            min_success_rate=0.80,
            max_latency_p95_ms=2000.0,
            min_quality_score=0.75
        )

        print("✓ Configured migration schedule and thresholds")

        # Plan migration
        migration_id = controller.plan_migration(
            target_strategy="rl_enabled",
            schedule=schedule,
            thresholds=thresholds
        )
        print(f"✓ Planned migration: {migration_id}")

        # Start migration
        success = await controller.start_migration(migration_id)
        print(f"✓ Started migration: {success}")

        # Simulate migration progress
        print("\nMigration Progress:")
        for i in range(8):
            await asyncio.sleep(0.3)  # Short intervals for demo

            status = controller.get_migration_status()
            print(f"  Step {i+1}: Phase: {status['current_phase']}, "
                  f"Progress: {status['progress_percentage']:.1f}%")

            # Simulate more requests being processed
            for j in range(15):
                strategy = 'rl_enabled' if j % 3 == 0 else 'traditional'
                performance_monitor.metrics_data[strategy]["total_selections"] += 1

            # Force phase advancement for demo
            if controller.automatic_mode:
                # Manually trigger phase check (normally done by background task)
                if await controller._should_advance_phase():
                    await controller._advance_to_next_phase()

            if status['current_phase'] in ['completed', 'failed']:
                break

        # Final status
        final_status = controller.get_migration_status()
        print(f"\nMigration Complete:")
        print(f"  Final Phase: {final_status['current_phase']}")
        print(f"  Duration: {final_status['duration_minutes']:.1f} minutes")
        print(f"  Final Progress: {final_status['progress_percentage']:.1f}%")

        # Show migration events
        history = controller.get_migration_history()
        print(f"\nMigration Events ({len(history)} total):")
        for event in history[-5:]:  # Show last 5 events
            timestamp = event['timestamp'][:19]  # Remove microseconds
            print(f"  {timestamp}: {event['phase']} - {event['action']}")

        print("✓ Example 5 completed successfully")

    except ImportError as e:
        print(f"✗ RL components not available: {e}")
    except Exception as e:
        print(f"✗ Example 5 failed: {e}")

    print()


def example_6_configuration_management():
    """
    Example 6: Configuration Management

    Shows how to manage RL system configuration, validation,
    and environment-specific settings.
    """
    print("=" * 60)
    print("Example 6: Configuration Management")
    print("=" * 60)

    try:
        from src.agents.rl.deployment import RLSystemConfig
        import tempfile
        import os

        # Example 1: Environment-based configuration
        print("Testing environment-based configuration...")

        # Set environment variables
        os.environ['RL_ENABLED'] = 'true'
        os.environ['AB_TESTING_ENABLED'] = 'true'
        os.environ['INITIAL_RL_TRAFFIC_PCT'] = '15'
        os.environ['SUCCESS_RATE_WARNING'] = '0.80'

        config = RLSystemConfig()
        print(f"✓ Loaded config from environment")
        print(f"  RL Enabled: {config.get('rl_enabled')}")
        print(f"  AB Testing: {config.get('ab_testing_enabled')}")
        print(f"  Initial Traffic: {config.get('initial_rl_traffic_pct')}%")

        # Example 2: File-based configuration
        print("\nTesting file-based configuration...")

        config_data = {
            "rl_enabled": True,
            "ab_testing_enabled": False,
            "initial_rl_traffic_pct": 25.0,
            "performance_thresholds": {
                "success_rate_warning": 0.90,
                "latency_p95_warning_ms": 1500.0
            },
            "continuous_learning_enabled": True,
            "learning_mode": "online_only"
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            temp_config_path = f.name

        try:
            file_config = RLSystemConfig(temp_config_path)
            print(f"✓ Loaded config from file: {temp_config_path}")
            print(f"  RL Enabled: {file_config.get('rl_enabled')}")
            print(f"  Learning Mode: {file_config.get('learning_mode')}")
            print(f"  Success Rate Warning: {file_config.get_nested('performance_thresholds', 'success_rate_warning')}")

        finally:
            os.unlink(temp_config_path)

        # Example 3: Configuration validation
        print("\nTesting configuration validation...")

        # Valid configuration
        try:
            valid_config = RLSystemConfig()
            print("✓ Configuration validation passed")
        except ValueError as e:
            print(f"✗ Configuration validation failed: {e}")

        # Example 4: Configuration export
        print("\nExporting configuration...")
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": config.config_data,
            "validation_status": "valid"
        }

        print("✓ Configuration exported")
        print("  Sample export:")
        print(f"    RL Enabled: {export_data['configuration']['rl_enabled']}")
        print(f"    Model Path: {export_data['configuration']['rl_model_path']}")
        print(f"    Dashboard Port: {export_data['configuration']['dashboard_port']}")

        print("✓ Example 6 completed successfully")

    except ImportError as e:
        print(f"✗ RL components not available: {e}")
    except Exception as e:
        print(f"✗ Example 6 failed: {e}")

    print()


async def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 Running RL System Usage Examples")
    print("=" * 80)

    start_time = datetime.now()

    # Run synchronous examples
    example_1_basic_rl_integration()
    example_6_configuration_management()

    # Run asynchronous examples
    await example_2_performance_monitoring()
    await example_3_safety_mechanisms()
    await example_4_continuous_learning()
    await example_5_traffic_migration()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("=" * 80)
    print(f"🎉 All examples completed successfully in {duration:.1f} seconds!")
    print("\nNext Steps:")
    print("1. Review the examples above to understand RL system integration")
    print("2. Adapt the patterns to your specific use cases")
    print("3. Set up monitoring and safety mechanisms in production")
    print("4. Start with A/B testing at low traffic percentages")
    print("5. Monitor performance and gradually increase RL usage")
    print("\nFor more information:")
    print("- Documentation: docs/RL_SYSTEM_USAGE.md")
    print("- Configuration: config/rl_system.yaml")
    print("- Health check: python -m src.agents.rl.deployment --health")
    print("- Dashboard: http://localhost:8001")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())
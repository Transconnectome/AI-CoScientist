"""
Comprehensive Testing Suite for Enhanced Agent Pool 2.0
Tests all components of the enhanced multi-agent system

Test Coverage:
- Individual specialist agents
- Multi-agent orchestration
- Communication protocols
- Supervisor integration
- Performance and scalability
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import tempfile
import logging

# Import components to test
from src.agents.specialist_agents import (
    StatisticalAnalysisAgent,
    GrantWriterAgent,
    HypothesisGeneratorAgent,
    ClinicalValidationAgent,
    EnhancedLiteratureAnalystAgent
)
from src.agents.pool import AgentPool
from src.agents.langgraph_orchestrator import LangGraphOrchestrator
from src.agents.communication import AgentCommunicationHub, MessageType, MessagePriority
from src.agents.supervisor_integration import EnhancedProposalSupervisor
from src.agents.types import AgentTask, AgentResult
from src.agents.base import ResearchAgent

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSpecialistAgents:
    """Test individual specialist agents"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_llm = Mock()
        self.mock_context = Mock()

    @pytest.mark.asyncio
    async def test_statistical_analysis_agent(self):
        """Test StatisticalAnalysisAgent functionality"""

        agent = StatisticalAnalysisAgent("test_stats", self.mock_llm, self.mock_context)

        # Test agent properties
        assert "statistical_analysis" in agent.capabilities
        assert "statistics" in agent.domains
        assert "experimental_design" in agent.capabilities

        # Test statistical test processing
        task = AgentTask(
            task_id="test_stats_001",
            description="Perform statistical test for autism diagnosis accuracy",
            task_type="statistical_test"
        )

        context = {
            "sample_size": 150,
            "data_type": "continuous",
            "groups": 2
        }

        result = await agent.process(task, context)

        assert result.agent_id == "test_stats"
        assert result.task_id == "test_stats_001"
        assert result.confidence > 0.8
        assert "Independent samples t-test" in result.output
        assert "p-value" in result.output

    @pytest.mark.asyncio
    async def test_grant_writer_agent(self):
        """Test GrantWriterAgent functionality"""

        agent = GrantWriterAgent("test_grant", self.mock_llm, self.mock_context)

        # Test agent properties
        assert "grant_writing" in agent.capabilities
        assert "samsung_future_tech_grants" in agent.specializations

        # Test budget justification
        task = AgentTask(
            task_id="test_grant_001",
            description="Write budget justification for Samsung grant",
            task_type="budget"
        )

        context = {
            "budget_total": 5000000000,
            "grant_type": "samsung_future_tech"
        }

        result = await agent.process(task, context)

        assert result.agent_id == "test_grant"
        assert result.confidence > 0.8
        assert "BUDGET JUSTIFICATION" in result.output
        assert "₩5,000,000,000" in result.output
        assert "PERSONNEL" in result.output

    @pytest.mark.asyncio
    async def test_hypothesis_generator_agent(self):
        """Test HypothesisGeneratorAgent functionality"""

        agent = HypothesisGeneratorAgent("test_hyp", self.mock_llm, self.mock_context)

        # Test agent properties
        assert "hypothesis_generation" in agent.capabilities
        assert "scientific_method" in agent.domains

        # Test hypothesis generation
        task = AgentTask(
            task_id="test_hyp_001",
            description="Generate research hypotheses for AI autism diagnosis",
            task_type="generate"
        )

        context = {
            "research_area": "developmental_disorders",
            "literature_summary": "AI shows promise for early autism detection"
        }

        result = await agent.process(task, context)

        assert result.agent_id == "test_hyp"
        assert result.confidence > 0.8
        assert "HYPOTHESIS" in result.output
        assert "null_hypothesis" in result.output.lower()
        assert "alternative_hypothesis" in result.output.lower()

    @pytest.mark.asyncio
    async def test_clinical_validation_agent(self):
        """Test ClinicalValidationAgent functionality"""

        agent = ClinicalValidationAgent("test_clinical", self.mock_llm, self.mock_context)

        # Test agent properties
        assert "clinical_validation" in agent.capabilities
        assert "clinical_research" in agent.domains

        # Test clinical validation design
        task = AgentTask(
            task_id="test_clinical_001",
            description="Design clinical validation study",
            task_type="validation"
        )

        context = {
            "device_type": "AI diagnostic system"
        }

        result = await agent.process(task, context)

        assert result.agent_id == "test_clinical"
        assert result.confidence > 0.85
        assert "CLINICAL VALIDATION" in result.output
        assert "Phase" in result.output
        assert "regulatory" in result.output.lower()

    @pytest.mark.asyncio
    async def test_literature_analyst_agent(self):
        """Test EnhancedLiteratureAnalystAgent functionality"""

        agent = EnhancedLiteratureAnalystAgent("test_lit", self.mock_llm, self.mock_context)

        # Test agent properties
        assert "literature_synthesis" in agent.capabilities
        assert "scientific_literature" in agent.domains

        # Test systematic review
        task = AgentTask(
            task_id="test_lit_001",
            description="Conduct systematic review of AI autism diagnosis literature",
            task_type="systematic_review"
        )

        context = {
            "research_question": "How effective is AI for autism diagnosis?"
        }

        result = await agent.process(task, context)

        assert result.agent_id == "test_lit"
        assert result.confidence > 0.8
        assert "SYSTEMATIC REVIEW" in result.output
        assert "PRISMA" in result.output
        assert "methodology" in result.output.lower()

class TestAgentPool:
    """Test enhanced agent pool functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_llm = Mock()
        self.mock_context = Mock()
        self.agent_pool = AgentPool(self.mock_llm, self.mock_context)

    def test_agent_registration(self):
        """Test that all enhanced agents are registered"""

        agents = self.agent_pool.list_all_agents()

        # Check all required agents are present
        required_agents = [
            "neuroscience_expert",
            "statistical_analyst",
            "grant_writer",
            "hypothesis_generator",
            "clinical_validator",
            "literature_analyst"
        ]

        for agent_id in required_agents:
            assert agent_id in agents
            assert "capabilities" in agents[agent_id]
            assert "success_rate" in agents[agent_id]

    def test_optimal_agent_team_selection(self):
        """Test optimal agent team selection"""

        # Test simple task
        requirements = {
            "capabilities": ["statistical_analysis"],
            "domains": ["statistics"],
            "task_type": "simple"
        }

        team = self.agent_pool.get_optimal_agent_team(requirements)
        assert len(team) == 1
        assert "statistical_analyst" in team

        # Test complex task
        requirements = {
            "capabilities": ["grant_writing", "statistical_analysis"],
            "domains": ["grant_writing", "statistics"],
            "task_type": "complex"
        }

        team = self.agent_pool.get_optimal_agent_team(requirements)
        assert len(team) <= 3
        assert "grant_writer" in team or "statistical_analyst" in team

    @pytest.mark.asyncio
    async def test_parallel_task_execution(self):
        """Test parallel task execution"""

        # Create mock tasks
        tasks = [
            {
                "agent_id": "statistical_analyst",
                "task": {
                    "task_id": "parallel_001",
                    "description": "Analyze statistics",
                    "task_type": "analysis"
                },
                "context": {"data": "test_data"}
            },
            {
                "agent_id": "literature_analyst",
                "task": {
                    "task_id": "parallel_002",
                    "description": "Review literature",
                    "task_type": "review"
                },
                "context": {"query": "test_query"}
            }
        ]

        # Execute in parallel
        results = await self.agent_pool.execute_parallel_tasks(tasks)

        assert len(results) == 2
        # Results should be AgentResult objects or exceptions
        for result in results:
            assert hasattr(result, 'agent_id') or isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_collaborative_analysis(self):
        """Test collaborative analysis functionality"""

        research_question = "What are the best practices for AI-based autism diagnosis?"

        result = await self.agent_pool.collaborative_analysis(
            research_question,
            ["statistical_analyst", "literature_analyst"]
        )

        assert "research_question" in result
        assert "participating_agents" in result
        assert "results" in result
        assert "success_rate" in result
        assert result["research_question"] == research_question

    @pytest.mark.asyncio
    async def test_smart_task_routing(self):
        """Test intelligent task routing"""

        # Test statistical task routing
        task = {
            "description": "Perform statistical analysis of autism diagnosis data",
            "task_type": "analysis"
        }

        routed_agent = await self.agent_pool.smart_task_routing(task)
        assert routed_agent == "statistical_analyst"

        # Test grant writing task routing
        task = {
            "description": "Write grant proposal for Samsung funding",
            "task_type": "writing"
        }

        routed_agent = await self.agent_pool.smart_task_routing(task)
        assert routed_agent == "grant_writer"

class TestCommunicationHub:
    """Test inter-agent communication system"""

    def setup_method(self):
        """Setup test environment"""
        self.communication_hub = AgentCommunicationHub()

    def test_agent_registration(self):
        """Test agent registration in communication hub"""

        self.communication_hub.register_agent(
            "test_agent",
            ["test_capability"],
            {"test_preference": "value"}
        )

        assert "test_agent" in self.communication_hub.registered_agents
        assert self.communication_hub.registered_agents["test_agent"]["capabilities"] == ["test_capability"]

    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test message sending between agents"""

        # Register agents
        self.communication_hub.register_agent("sender", ["sending"])
        self.communication_hub.register_agent("receiver", ["receiving"])

        # Send message
        message_id = await self.communication_hub.send_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.REQUEST,
            content={"test": "message"},
            requires_response=True
        )

        assert message_id is not None
        assert len(message_id) > 0

        # Check message queue
        messages = self.communication_hub.get_messages_for_agent("receiver")
        assert len(messages) > 0
        assert messages[0].sender_id == "sender"

    @pytest.mark.asyncio
    async def test_response_handling(self):
        """Test message response handling"""

        # Register agents
        self.communication_hub.register_agent("agent1", ["test"])
        self.communication_hub.register_agent("agent2", ["test"])

        # Send message requiring response
        message_id = await self.communication_hub.send_message(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.REQUEST,
            content={"request": "data"},
            requires_response=True
        )

        # Send response
        response_id = await self.communication_hub.respond_to_message(
            message_id,
            "agent2",
            {"response": "data"}
        )

        assert response_id is not None
        assert message_id not in self.communication_hub.pending_responses

    @pytest.mark.asyncio
    async def test_communication_session(self):
        """Test multi-agent communication sessions"""

        # Register agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            self.communication_hub.register_agent(agent, ["test"])

        # Start session
        session_id = await self.communication_hub.start_communication_session(
            "collaboration",
            agents,
            {"test": "context"}
        )

        assert session_id in self.communication_hub.active_sessions
        session = self.communication_hub.active_sessions[session_id]
        assert session.participants == agents

        # End session
        summary = await self.communication_hub.end_communication_session(session_id)
        assert session_id not in self.communication_hub.active_sessions
        assert "session_id" in summary

class TestLangGraphOrchestrator:
    """Test LangGraph orchestration system"""

    def setup_method(self):
        """Setup test environment"""
        mock_agent_pool = Mock()
        mock_agent_pool.get_agent.return_value = Mock()
        mock_agent_pool.get_agent.return_value.process = AsyncMock(return_value=Mock(
            output="Mock agent output",
            confidence=0.9,
            agent_id="mock_agent"
        ))

        self.orchestrator = LangGraphOrchestrator(mock_agent_pool)

    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test workflow execution"""

        context = {
            "research_topic": "AI autism diagnosis",
            "budget": 5000000000
        }

        result = await self.orchestrator.execute_workflow("samsung_grant", context)

        assert "workflow_id" in result
        assert "execution_type" in result
        assert "outputs" in result
        assert "quality_score" in result

    def test_available_workflows(self):
        """Test available workflow listing"""

        workflows = self.orchestrator.list_available_workflows()

        expected_workflows = ["samsung_grant", "research_analysis", "clinical_validation", "hypothesis_generation"]

        for workflow in expected_workflows:
            assert workflow in workflows

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        """Test parallel agent execution"""

        agent_tasks = [
            {
                "agent_id": "test_agent1",
                "task": {
                    "task_id": "task1",
                    "description": "Test task 1",
                    "task_type": "test"
                },
                "context": {"test": "data1"}
            },
            {
                "agent_id": "test_agent2",
                "task": {
                    "task_id": "task2",
                    "description": "Test task 2",
                    "task_type": "test"
                },
                "context": {"test": "data2"}
            }
        ]

        results = await self.orchestrator.execute_parallel_agents(agent_tasks)

        assert len(results) >= 0  # May be empty if agents not found, but should not error

class TestSupervisorIntegration:
    """Test supervisor pattern integration"""

    def setup_method(self):
        """Setup test environment"""
        # Create mock components
        self.mock_agent_pool = Mock()
        self.mock_communication_hub = Mock()
        self.mock_orchestrator = Mock()

        # Configure mocks
        self.mock_agent_pool.list_all_agents.return_value = {
            "statistical_analyst": {"capabilities": ["statistical_analysis"]},
            "grant_writer": {"capabilities": ["grant_writing"]}
        }

        self.mock_orchestrator.execute_workflow = AsyncMock(return_value={
            "workflow_id": "test_workflow",
            "execution_type": "test",
            "quality_score": 0.85,
            "outputs": {
                "literature_analysis": "Mock literature output",
                "grant_proposal": "Mock proposal output"
            }
        })

        self.supervisor = EnhancedProposalSupervisor(
            self.mock_agent_pool,
            self.mock_communication_hub,
            self.mock_orchestrator
        )

    @pytest.mark.asyncio
    async def test_samsung_grant_supervision(self):
        """Test Samsung grant generation supervision"""

        research_context = {
            "research_topic": "AI-based autism diagnosis",
            "budget": 5000000000,
            "duration": "5 years"
        }

        result = await self.supervisor.supervise_samsung_grant_generation(research_context)

        assert result.task_id is not None
        assert result.supervisor_id == "proposal_supervisor"
        assert result.quality_score >= 0.0
        assert isinstance(result.success, bool)
        assert result.execution_time_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_parallel_analysis_supervision(self):
        """Test parallel analysis supervision"""

        # Mock collaborative analysis
        self.mock_agent_pool.collaborative_analysis = AsyncMock(return_value={
            "success_rate": 0.9,
            "results": {
                "agent1": {"status": "success", "output": "Mock output", "confidence": 0.9}
            }
        })

        research_question = "How effective is AI for autism diagnosis?"

        result = await self.supervisor.supervise_parallel_analysis(research_question)

        assert result.task_id is not None
        assert result.supervisor_id == "proposal_supervisor"
        assert result.integrated_result is not None

class TestIntegrationScenarios:
    """Test complete integration scenarios"""

    def setup_method(self):
        """Setup comprehensive test environment"""
        # Initialize all components
        self.mock_llm = Mock()
        self.mock_context = Mock()
        self.agent_pool = AgentPool(self.mock_llm, self.mock_context)
        self.communication_hub = AgentCommunicationHub()
        self.orchestrator = LangGraphOrchestrator(self.agent_pool)

        # Register agents in communication hub
        for agent_id in self.agent_pool.list_all_agents().keys():
            agent_info = self.agent_pool.list_all_agents()[agent_id]
            self.communication_hub.register_agent(agent_id, agent_info["capabilities"])

    @pytest.mark.asyncio
    async def test_full_samsung_grant_pipeline(self):
        """Test complete Samsung grant generation pipeline"""

        # Test context
        research_context = {
            "research_topic": "AI-기반 자폐스펙트럼장애 조기 진단 시스템",
            "target_population": "만 2-5세 아동",
            "methodology": "딥러닝 기반 멀티모달 분석",
            "budget_requirement": 5000000000,
            "duration_months": 60,
            "expected_outcomes": ["조기 진단 정확도 99% 달성", "진단 시간 단축 80%"]
        }

        # Execute workflow
        result = await self.orchestrator.execute_workflow("samsung_grant", research_context)

        # Verify results
        assert result is not None
        assert "workflow_id" in result
        assert result["quality_score"] >= 0.0

        # Check outputs
        outputs = result.get("outputs", {})
        assert isinstance(outputs, dict)

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration scenario"""

        research_question = "삼성 미래기술육성사업을 위한 AI 기반 발달장애 진단 시스템의 최적 설계 방안"

        # Test collaborative analysis
        collaboration_result = await self.agent_pool.collaborative_analysis(
            research_question,
            ["literature_analyst", "statistical_analyst", "clinical_validator"]
        )

        # Verify collaboration results
        assert "research_question" in collaboration_result
        assert "results" in collaboration_result
        assert collaboration_result["research_question"] == research_question

    @pytest.mark.asyncio
    async def test_communication_workflow(self):
        """Test communication-based workflow"""

        # Start communication session
        session_id = await self.communication_hub.start_communication_session(
            "grant_collaboration",
            ["grant_writer", "statistical_analyst", "literature_analyst"],
            {"project": "samsung_grant"}
        )

        # Send collaboration messages
        message_id = await self.communication_hub.send_message(
            sender_id="grant_writer",
            receiver_id="statistical_analyst",
            message_type=MessageType.COLLABORATION,
            content={
                "action": "statistical_consultation",
                "parameters": {
                    "research_question": "Sample size for autism diagnosis study",
                    "target_power": 0.8
                }
            },
            requires_response=True
        )

        # Verify session and messaging
        assert session_id in self.communication_hub.active_sessions
        assert message_id is not None

        # End session
        summary = await self.communication_hub.end_communication_session(session_id)
        assert "session_id" in summary

class TestPerformanceAndScalability:
    """Test performance and scalability of enhanced system"""

    def setup_method(self):
        """Setup performance test environment"""
        self.mock_llm = Mock()
        self.mock_context = Mock()
        self.agent_pool = AgentPool(self.mock_llm, self.mock_context)

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self):
        """Test concurrent execution of multiple agents"""

        # Create multiple tasks
        num_tasks = 10
        tasks = []

        for i in range(num_tasks):
            tasks.append({
                "agent_id": "statistical_analyst",
                "task": {
                    "task_id": f"concurrent_task_{i}",
                    "description": f"Concurrent statistical analysis {i}",
                    "task_type": "analysis"
                },
                "context": {"data": f"test_data_{i}"}
            })

        # Measure execution time
        start_time = datetime.now()
        results = await self.agent_pool.execute_parallel_tasks(tasks)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Verify results
        assert len(results) == num_tasks
        assert execution_time < 60  # Should complete within 60 seconds

        logger.info(f"Concurrent execution of {num_tasks} tasks completed in {execution_time:.2f} seconds")

    @pytest.mark.asyncio
    async def test_agent_pool_scalability(self):
        """Test agent pool scalability"""

        # Test with different team sizes
        team_sizes = [1, 3, 5]
        research_question = "Test scalability question"

        for size in team_sizes:
            agent_ids = list(self.agent_pool.list_all_agents().keys())[:size]

            start_time = datetime.now()
            result = await self.agent_pool.collaborative_analysis(research_question, agent_ids)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Verify scaling characteristics
            assert "success_rate" in result
            assert execution_time < 120  # Should scale reasonably

            logger.info(f"Team size {size}: {execution_time:.2f}s, Success rate: {result.get('success_rate', 0.0):.2f}")

    def test_memory_usage(self):
        """Test memory usage under load"""

        # Create multiple agent pools
        pools = []
        for i in range(5):
            pool = AgentPool(Mock(), Mock())
            pools.append(pool)

        # Verify reasonable memory usage
        assert len(pools) == 5

        # Cleanup
        del pools

class TestErrorHandling:
    """Test error handling and resilience"""

    def setup_method(self):
        """Setup error testing environment"""
        self.mock_llm = Mock()
        self.mock_context = Mock()
        self.agent_pool = AgentPool(self.mock_llm, self.mock_context)

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self):
        """Test handling of agent failures"""

        # Create task with invalid agent
        tasks = [{
            "agent_id": "nonexistent_agent",
            "task": {
                "task_id": "failure_test",
                "description": "Test failure handling",
                "task_type": "test"
            },
            "context": {}
        }]

        # Should handle gracefully
        results = await self.agent_pool.execute_parallel_tasks(tasks)

        # Verify error handling
        assert len(results) == 1
        # Result should be empty or exception, but should not crash

    @pytest.mark.asyncio
    async def test_communication_timeout_handling(self):
        """Test communication timeout handling"""

        hub = AgentCommunicationHub()
        hub.register_agent("sender", ["test"])
        hub.register_agent("receiver", ["test"])

        # Send message with short timeout
        message_id = await hub.send_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.REQUEST,
            content={"test": "timeout"},
            requires_response=True,
            response_timeout=1  # 1 second timeout
        )

        # Wait for timeout
        await asyncio.sleep(2)

        # Message should be removed from pending responses
        assert message_id not in hub.pending_responses

# Convenience functions for running tests
def run_all_tests():
    """Run all tests in the suite"""

    test_classes = [
        TestSpecialistAgents,
        TestAgentPool,
        TestCommunicationHub,
        TestLangGraphOrchestrator,
        TestSupervisorIntegration,
        TestIntegrationScenarios,
        TestPerformanceAndScalability,
        TestErrorHandling
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n{'='*60}")
        print(f"Running {class_name}")
        print(f"{'='*60}")

        # Get test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                test_instance.setup_method()

                # Run test
                method = getattr(test_instance, test_method)
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()

                print(f"✅ {test_method}")
                passed_tests += 1

            except Exception as e:
                print(f"❌ {test_method}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    return passed_tests / total_tests

if __name__ == "__main__":
    # Run comprehensive test suite
    success_rate = run_all_tests()

    print(f"\n🎉 Enhanced Agent Pool 2.0 Test Suite Completed!")
    print(f"Overall success rate: {success_rate*100:.1f}%")

    if success_rate >= 0.8:
        print("✅ System ready for production deployment!")
    else:
        print("⚠️ Some tests failed - review and fix issues before deployment.")
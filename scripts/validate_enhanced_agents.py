#!/usr/bin/env python3
"""
Enhanced Agent Pool 2.0 Validation Script
Validates the implementation of the enhanced multi-agent system

Usage:
    python scripts/validate_enhanced_agents.py
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")

async def validate_specialist_agents():
    """Validate individual specialist agents"""
    print_header("SPECIALIST AGENTS VALIDATION")

    try:
        from agents.specialist_agents import (
            StatisticalAnalysisAgent,
            GrantWriterAgent,
            HypothesisGeneratorAgent,
            ClinicalValidationAgent,
            EnhancedLiteratureAnalystAgent
        )
        from agents.types import AgentTask, AgentResult

        print_success("All specialist agent imports successful")

        # Test agent initialization
        mock_llm = None
        mock_context = None

        agents = {
            "Statistical Analysis": StatisticalAnalysisAgent("test_stats", mock_llm, mock_context),
            "Grant Writer": GrantWriterAgent("test_grant", mock_llm, mock_context),
            "Hypothesis Generator": HypothesisGeneratorAgent("test_hyp", mock_llm, mock_context),
            "Clinical Validator": ClinicalValidationAgent("test_clinical", mock_llm, mock_context),
            "Literature Analyst": EnhancedLiteratureAnalystAgent("test_lit", mock_llm, mock_context)
        }

        for name, agent in agents.items():
            capabilities = len(agent.capabilities)
            domains = len(agent.domains)
            specializations = len(agent.specializations)
            print_success(f"{name}: {capabilities} capabilities, {domains} domains, {specializations} specializations")

        return True

    except Exception as e:
        print_error(f"Specialist agents validation failed: {e}")
        return False

async def validate_agent_pool():
    """Validate enhanced agent pool"""
    print_header("ENHANCED AGENT POOL VALIDATION")

    try:
        from agents.pool import AgentPool

        # Initialize agent pool
        mock_llm = None
        mock_context = None
        pool = AgentPool(mock_llm, mock_context)

        # Check registered agents
        agents = pool.list_all_agents()
        expected_agents = [
            "neuroscience_expert",
            "statistical_analyst",
            "grant_writer",
            "hypothesis_generator",
            "clinical_validator",
            "literature_analyst"
        ]

        print_success(f"Agent pool initialized with {len(agents)} agents")

        for agent_id in expected_agents:
            if agent_id in agents:
                print_success(f"✓ {agent_id} registered")
            else:
                print_error(f"✗ {agent_id} missing")
                return False

        # Test agent team selection
        requirements = {
            "capabilities": ["statistical_analysis", "grant_writing"],
            "domains": ["statistics", "grant_writing"],
            "task_type": "complex"
        }

        optimal_team = pool.get_optimal_agent_team(requirements)
        print_success(f"Optimal team selection: {optimal_team}")

        # Test smart routing
        task = {"description": "Perform statistical analysis", "task_type": "analysis"}
        routed_agent = await pool.smart_task_routing(task)
        print_success(f"Smart routing result: {routed_agent}")

        return True

    except Exception as e:
        print_error(f"Agent pool validation failed: {e}")
        return False

async def validate_orchestration():
    """Validate LangGraph orchestration"""
    print_header("ORCHESTRATION SYSTEM VALIDATION")

    try:
        from agents.langgraph_orchestrator import LangGraphOrchestrator
        from agents.pool import AgentPool

        # Initialize components
        mock_llm = None
        mock_context = None
        agent_pool = AgentPool(mock_llm, mock_context)
        orchestrator = LangGraphOrchestrator(agent_pool)

        # Check available workflows
        workflows = orchestrator.list_available_workflows()
        expected_workflows = ["samsung_grant", "research_analysis", "clinical_validation", "hypothesis_generation"]

        print_success(f"Orchestrator initialized with {len(workflows)} workflows")

        for workflow in expected_workflows:
            if workflow in workflows:
                print_success(f"✓ {workflow} workflow available")
            else:
                print_error(f"✗ {workflow} workflow missing")
                return False

        # Test workflow execution (simplified)
        context = {"test": "context"}
        result = await orchestrator.execute_workflow("samsung_grant", context)

        print_success(f"Workflow execution test: {result['execution_type']}")

        return True

    except Exception as e:
        print_error(f"Orchestration validation failed: {e}")
        return False

async def validate_communication():
    """Validate communication protocols"""
    print_header("COMMUNICATION SYSTEM VALIDATION")

    try:
        from agents.communication import AgentCommunicationHub, MessageType, MessagePriority

        # Initialize communication hub
        hub = AgentCommunicationHub()

        # Register test agents
        test_agents = ["agent1", "agent2", "agent3"]
        for agent_id in test_agents:
            hub.register_agent(agent_id, [f"capability_{agent_id}"])

        print_success(f"Communication hub initialized with {len(hub.registered_agents)} agents")

        # Test message sending
        message_id = await hub.send_message(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.REQUEST,
            content={"test": "message"},
            priority=MessagePriority.NORMAL
        )

        print_success(f"Message sent successfully: {message_id}")

        # Test communication session
        session_id = await hub.start_communication_session(
            "test_collaboration",
            test_agents,
            {"test": "context"}
        )

        print_success(f"Communication session started: {session_id}")

        # Get communication stats
        stats = hub.get_communication_statistics()
        print_success(f"Communication stats: {stats['total_agents']} agents, {stats['total_messages']} messages")

        return True

    except Exception as e:
        print_error(f"Communication validation failed: {e}")
        return False

async def validate_supervisor_integration():
    """Validate supervisor integration"""
    print_header("SUPERVISOR INTEGRATION VALIDATION")

    try:
        from agents.supervisor_integration import EnhancedProposalSupervisor
        from agents.pool import AgentPool
        from agents.communication import AgentCommunicationHub
        from agents.langgraph_orchestrator import LangGraphOrchestrator

        # Initialize all components
        mock_llm = None
        mock_context = None
        agent_pool = AgentPool(mock_llm, mock_context)
        communication_hub = AgentCommunicationHub()
        orchestrator = LangGraphOrchestrator(agent_pool)

        # Initialize supervisor
        supervisor = EnhancedProposalSupervisor(
            agent_pool, communication_hub, orchestrator
        )

        print_success("Supervisor integration initialized successfully")

        # Check supervisor configuration
        config_keys = ["quality_thresholds", "coordination_patterns", "escalation_rules"]
        for key in config_keys:
            if key in supervisor.supervision_config:
                print_success(f"✓ {key} configured")
            else:
                print_error(f"✗ {key} missing")
                return False

        return True

    except Exception as e:
        print_error(f"Supervisor integration validation failed: {e}")
        return False

async def validate_integration_scenario():
    """Validate end-to-end integration scenario"""
    print_header("INTEGRATION SCENARIO VALIDATION")

    try:
        from agents.pool import AgentPool
        from agents.langgraph_orchestrator import LangGraphOrchestrator
        from agents.communication import AgentCommunicationHub

        # Initialize system
        mock_llm = None
        mock_context = None
        agent_pool = AgentPool(mock_llm, mock_context)
        orchestrator = LangGraphOrchestrator(agent_pool)
        communication_hub = AgentCommunicationHub()

        # Register agents in communication system
        for agent_id in agent_pool.list_all_agents().keys():
            agent_info = agent_pool.list_all_agents()[agent_id]
            communication_hub.register_agent(agent_id, agent_info["capabilities"])

        print_success("Integration components initialized")

        # Test collaborative analysis
        research_question = "AI-based autism diagnosis for Samsung grant proposal"
        collaboration_result = await agent_pool.collaborative_analysis(
            research_question,
            ["literature_analyst", "statistical_analyst"]
        )

        print_success(f"Collaborative analysis: {collaboration_result['success_rate']:.2f} success rate")

        # Test workflow execution
        context = {
            "research_topic": "AI autism diagnosis",
            "budget": 5000000000,
            "duration": "5 years"
        }

        workflow_result = await orchestrator.execute_workflow("samsung_grant", context)
        print_success(f"Workflow execution: {workflow_result['quality_score']:.2f} quality score")

        return True

    except Exception as e:
        print_error(f"Integration scenario validation failed: {e}")
        return False

async def main():
    """Main validation function"""

    print_header("ENHANCED AGENT POOL 2.0 VALIDATION")
    print_info(f"Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all validation tests
    validation_tests = [
        ("Specialist Agents", validate_specialist_agents),
        ("Agent Pool", validate_agent_pool),
        ("Orchestration", validate_orchestration),
        ("Communication", validate_communication),
        ("Supervisor Integration", validate_supervisor_integration),
        ("Integration Scenario", validate_integration_scenario)
    ]

    results = {}

    for test_name, test_func in validation_tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print_error(f"{test_name} validation crashed: {e}")
            results[test_name] = False

    # Summary
    print_header("VALIDATION SUMMARY")

    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = passed_tests / total_tests

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")

    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed ({success_rate*100:.1f}%)")

    if success_rate == 1.0:
        print_success("🎉 ALL VALIDATIONS PASSED - ENHANCED AGENT POOL 2.0 READY FOR PRODUCTION!")
        return 0
    elif success_rate >= 0.8:
        print_info("⚠️ Most validations passed - minor issues to address")
        return 1
    else:
        print_error("❌ Multiple validation failures - significant issues to resolve")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
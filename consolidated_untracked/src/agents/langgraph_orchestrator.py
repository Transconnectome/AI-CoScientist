"""
LangGraph Multi-Agent Orchestration System
Coordination layer for Agent Pool Enhancement 2.0

Features:
- Parallel agent execution
- Supervisor pattern implementation
- Inter-agent communication
- Dynamic workflow adaptation
- State management across agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

try:
    from langgraph.graph import StateGraph, END
    from langgraph.pregel import Pregel
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback implementation for environments without LangGraph
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    MemorySaver = None

from .pool import AgentPool
from .types import AgentTask, AgentResult
from .base import ResearchAgent

logger = logging.getLogger(__name__)

class WorkflowType(str, Enum):
    """Multi-agent workflow patterns"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SUPERVISOR = "supervisor"
    COLLABORATIVE = "collaborative"
    HIERARCHICAL = "hierarchical"

class AgentRole(str, Enum):
    """Agent roles in multi-agent workflows"""
    SUPERVISOR = "supervisor"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"

@dataclass
class AgentState:
    """State for individual agents in workflow"""
    agent_id: str
    status: str = "pending"  # pending, running, completed, failed
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowState:
    """Overall workflow state management"""
    workflow_id: str
    workflow_type: WorkflowType
    agents: Dict[str, AgentState] = field(default_factory=dict)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    current_step: str = "start"
    completed_steps: List[str] = field(default_factory=list)
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)

class LangGraphOrchestrator:
    """LangGraph-powered multi-agent orchestration system"""

    def __init__(self, agent_pool: AgentPool):
        self.agent_pool = agent_pool
        self.workflows: Dict[str, Any] = {}

        # Initialize memory for state persistence
        if LANGGRAPH_AVAILABLE:
            self.memory = MemorySaver()
        else:
            self.memory = {}  # Simple fallback

        # Predefined workflow templates
        self._initialize_workflow_templates()

    def _initialize_workflow_templates(self):
        """Initialize common workflow patterns"""

        if LANGGRAPH_AVAILABLE:
            # Samsung Grant Generation Workflow
            self.workflows["samsung_grant"] = self._create_samsung_grant_workflow()

            # Research Analysis Workflow
            self.workflows["research_analysis"] = self._create_research_analysis_workflow()

            # Clinical Validation Workflow
            self.workflows["clinical_validation"] = self._create_clinical_validation_workflow()

            # Hypothesis Generation Workflow
            self.workflows["hypothesis_generation"] = self._create_hypothesis_workflow()
        else:
            # Fallback workflows without LangGraph
            self.workflows = {
                "samsung_grant": self._create_fallback_workflow("samsung_grant"),
                "research_analysis": self._create_fallback_workflow("research_analysis"),
                "clinical_validation": self._create_fallback_workflow("clinical_validation"),
                "hypothesis_generation": self._create_fallback_workflow("hypothesis_generation")
            }

    def _create_samsung_grant_workflow(self) -> StateGraph:
        """Create Samsung grant generation workflow using LangGraph"""

        if not LANGGRAPH_AVAILABLE:
            return self._create_fallback_workflow("samsung_grant")

        workflow = StateGraph(WorkflowState)

        # Define workflow nodes
        workflow.add_node("literature_analysis", self._literature_analysis_node)
        workflow.add_node("hypothesis_generation", self._hypothesis_generation_node)
        workflow.add_node("statistical_planning", self._statistical_planning_node)
        workflow.add_node("grant_writing", self._grant_writing_node)
        workflow.add_node("clinical_validation", self._clinical_validation_node)
        workflow.add_node("supervisor_review", self._supervisor_review_node)

        # Define workflow edges (dependencies)
        workflow.add_edge("literature_analysis", "hypothesis_generation")
        workflow.add_edge("hypothesis_generation", "statistical_planning")
        workflow.add_edge("statistical_planning", "grant_writing")
        workflow.add_edge("grant_writing", "clinical_validation")
        workflow.add_edge("clinical_validation", "supervisor_review")
        workflow.add_edge("supervisor_review", END)

        # Set entry point
        workflow.set_entry_point("literature_analysis")

        return workflow.compile(checkpointer=self.memory)

    def _create_research_analysis_workflow(self) -> StateGraph:
        """Create parallel research analysis workflow"""

        if not LANGGRAPH_AVAILABLE:
            return self._create_fallback_workflow("research_analysis")

        workflow = StateGraph(WorkflowState)

        # Parallel analysis nodes
        workflow.add_node("literature_analysis", self._literature_analysis_node)
        workflow.add_node("statistical_analysis", self._statistical_analysis_node)
        workflow.add_node("hypothesis_analysis", self._hypothesis_analysis_node)
        workflow.add_node("synthesis", self._synthesis_node)

        # Parallel execution pattern
        workflow.add_edge("literature_analysis", "synthesis")
        workflow.add_edge("statistical_analysis", "synthesis")
        workflow.add_edge("hypothesis_analysis", "synthesis")
        workflow.add_edge("synthesis", END)

        # Multiple entry points for parallel execution
        workflow.set_entry_point("literature_analysis")

        return workflow.compile(checkpointer=self.memory)

    def _create_clinical_validation_workflow(self) -> StateGraph:
        """Create clinical validation workflow with regulatory focus"""

        if not LANGGRAPH_AVAILABLE:
            return self._create_fallback_workflow("clinical_validation")

        workflow = StateGraph(WorkflowState)

        workflow.add_node("clinical_planning", self._clinical_planning_node)
        workflow.add_node("regulatory_assessment", self._regulatory_assessment_node)
        workflow.add_node("safety_analysis", self._safety_analysis_node)
        workflow.add_node("efficacy_design", self._efficacy_design_node)
        workflow.add_node("validation_synthesis", self._validation_synthesis_node)

        # Sequential with conditional paths
        workflow.add_edge("clinical_planning", "regulatory_assessment")
        workflow.add_edge("regulatory_assessment", "safety_analysis")
        workflow.add_edge("safety_analysis", "efficacy_design")
        workflow.add_edge("efficacy_design", "validation_synthesis")
        workflow.add_edge("validation_synthesis", END)

        workflow.set_entry_point("clinical_planning")

        return workflow.compile(checkpointer=self.memory)

    def _create_hypothesis_workflow(self) -> StateGraph:
        """Create hypothesis generation and testing workflow"""

        if not LANGGRAPH_AVAILABLE:
            return self._create_fallback_workflow("hypothesis_generation")

        workflow = StateGraph(WorkflowState)

        workflow.add_node("gap_analysis", self._gap_analysis_node)
        workflow.add_node("hypothesis_generation", self._hypothesis_generation_node)
        workflow.add_node("hypothesis_refinement", self._hypothesis_refinement_node)
        workflow.add_node("test_design", self._test_design_node)
        workflow.add_node("feasibility_check", self._feasibility_check_node)

        workflow.add_edge("gap_analysis", "hypothesis_generation")
        workflow.add_edge("hypothesis_generation", "hypothesis_refinement")
        workflow.add_edge("hypothesis_refinement", "test_design")
        workflow.add_edge("test_design", "feasibility_check")
        workflow.add_edge("feasibility_check", END)

        workflow.set_entry_point("gap_analysis")

        return workflow.compile(checkpointer=self.memory)

    # Workflow Node Implementations
    async def _literature_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Literature analysis node"""
        try:
            agent = self.agent_pool.get_agent("literature_analyst")
            if not agent:
                raise ValueError("Literature analyst agent not found")

            task = AgentTask(
                task_id=f"{state.workflow_id}_literature",
                description="Conduct comprehensive literature analysis for Samsung grant proposal",
                task_type="literature_synthesis",
                priority=1
            )

            result = await agent.process(task, state.shared_context)

            # Update state
            state.agents["literature_analyst"] = AgentState(
                agent_id="literature_analyst",
                status="completed",
                output_data={"analysis": result.output, "confidence": result.confidence}
            )

            # Add to shared context
            state.shared_context["literature_analysis"] = result.output
            state.completed_steps.append("literature_analysis")

        except Exception as e:
            logger.error(f"Literature analysis node error: {e}")
            state.agents["literature_analyst"] = AgentState(
                agent_id="literature_analyst",
                status="failed",
                metadata={"error": str(e)}
            )

        return state

    async def _hypothesis_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Hypothesis generation node"""
        try:
            agent = self.agent_pool.get_agent("hypothesis_generator")
            if not agent:
                raise ValueError("Hypothesis generator agent not found")

            task = AgentTask(
                task_id=f"{state.workflow_id}_hypothesis",
                description="Generate novel research hypotheses based on literature analysis",
                task_type="hypothesis_generation",
                priority=1
            )

            # Use literature analysis as context
            context = {
                "literature_summary": state.shared_context.get("literature_analysis", ""),
                "research_area": "developmental_disorders",
                **state.shared_context
            }

            result = await agent.process(task, context)

            state.agents["hypothesis_generator"] = AgentState(
                agent_id="hypothesis_generator",
                status="completed",
                output_data={"hypotheses": result.output, "confidence": result.confidence}
            )

            state.shared_context["research_hypotheses"] = result.output
            state.completed_steps.append("hypothesis_generation")

        except Exception as e:
            logger.error(f"Hypothesis generation node error: {e}")
            state.agents["hypothesis_generator"] = AgentState(
                agent_id="hypothesis_generator",
                status="failed",
                metadata={"error": str(e)}
            )

        return state

    async def _statistical_planning_node(self, state: WorkflowState) -> WorkflowState:
        """Statistical analysis planning node"""
        try:
            agent = self.agent_pool.get_agent("statistical_analyst")
            if not agent:
                raise ValueError("Statistical analyst agent not found")

            task = AgentTask(
                task_id=f"{state.workflow_id}_statistics",
                description="Design statistical analysis plan for research hypotheses",
                task_type="experimental_design",
                priority=1
            )

            context = {
                "research_hypotheses": state.shared_context.get("research_hypotheses", ""),
                "sample_size": 3000,
                "study_type": "prospective_cohort",
                **state.shared_context
            }

            result = await agent.process(task, context)

            state.agents["statistical_analyst"] = AgentState(
                agent_id="statistical_analyst",
                status="completed",
                output_data={"analysis_plan": result.output, "confidence": result.confidence}
            )

            state.shared_context["statistical_plan"] = result.output
            state.completed_steps.append("statistical_planning")

        except Exception as e:
            logger.error(f"Statistical planning node error: {e}")
            state.agents["statistical_analyst"] = AgentState(
                agent_id="statistical_analyst",
                status="failed",
                metadata={"error": str(e)}
            )

        return state

    async def _grant_writing_node(self, state: WorkflowState) -> WorkflowState:
        """Grant writing node"""
        try:
            agent = self.agent_pool.get_agent("grant_writer")
            if not agent:
                raise ValueError("Grant writer agent not found")

            task = AgentTask(
                task_id=f"{state.workflow_id}_grant_writing",
                description="Write Samsung grant proposal sections",
                task_type="grant_writing",
                priority=1
            )

            context = {
                "literature_analysis": state.shared_context.get("literature_analysis", ""),
                "research_hypotheses": state.shared_context.get("research_hypotheses", ""),
                "statistical_plan": state.shared_context.get("statistical_plan", ""),
                "grant_type": "samsung_future_tech",
                "budget_total": 5000000000,  # 5 billion won
                **state.shared_context
            }

            result = await agent.process(task, context)

            state.agents["grant_writer"] = AgentState(
                agent_id="grant_writer",
                status="completed",
                output_data={"grant_sections": result.output, "confidence": result.confidence}
            )

            state.shared_context["grant_proposal"] = result.output
            state.completed_steps.append("grant_writing")

        except Exception as e:
            logger.error(f"Grant writing node error: {e}")
            state.agents["grant_writer"] = AgentState(
                agent_id="grant_writer",
                status="failed",
                metadata={"error": str(e)}
            )

        return state

    async def _clinical_validation_node(self, state: WorkflowState) -> WorkflowState:
        """Clinical validation planning node"""
        try:
            agent = self.agent_pool.get_agent("clinical_validator")
            if not agent:
                raise ValueError("Clinical validator agent not found")

            task = AgentTask(
                task_id=f"{state.workflow_id}_clinical_validation",
                description="Design clinical validation strategy",
                task_type="clinical_validation",
                priority=1
            )

            context = {
                "research_hypotheses": state.shared_context.get("research_hypotheses", ""),
                "statistical_plan": state.shared_context.get("statistical_plan", ""),
                "device_type": "AI diagnostic system",
                **state.shared_context
            }

            result = await agent.process(task, context)

            state.agents["clinical_validator"] = AgentState(
                agent_id="clinical_validator",
                status="completed",
                output_data={"validation_plan": result.output, "confidence": result.confidence}
            )

            state.shared_context["clinical_validation"] = result.output
            state.completed_steps.append("clinical_validation")

        except Exception as e:
            logger.error(f"Clinical validation node error: {e}")
            state.agents["clinical_validator"] = AgentState(
                agent_id="clinical_validator",
                status="failed",
                metadata={"error": str(e)}
            )

        return state

    async def _supervisor_review_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor review and integration node"""
        try:
            # Use existing proposal generation agent as supervisor
            # This leverages the existing sophisticated proposal agent

            # Integrate all agent outputs
            integrated_content = {
                "literature_analysis": state.shared_context.get("literature_analysis", ""),
                "research_hypotheses": state.shared_context.get("research_hypotheses", ""),
                "statistical_plan": state.shared_context.get("statistical_plan", ""),
                "grant_proposal": state.shared_context.get("grant_proposal", ""),
                "clinical_validation": state.shared_context.get("clinical_validation", "")
            }

            # Quality assessment and integration
            quality_score = self._assess_workflow_quality(state)

            state.agents["supervisor"] = AgentState(
                agent_id="supervisor",
                status="completed",
                output_data={
                    "integrated_proposal": integrated_content,
                    "quality_score": quality_score,
                    "review_status": "approved" if quality_score > 0.8 else "needs_revision"
                }
            )

            state.shared_context["final_proposal"] = integrated_content
            state.shared_context["quality_assessment"] = quality_score
            state.completed_steps.append("supervisor_review")

        except Exception as e:
            logger.error(f"Supervisor review node error: {e}")
            state.agents["supervisor"] = AgentState(
                agent_id="supervisor",
                status="failed",
                metadata={"error": str(e)}
            )

        return state

    # Additional node implementations for other workflows
    async def _statistical_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Statistical analysis node for parallel workflows"""
        agent = self.agent_pool.get_agent("statistical_analyst")
        task = AgentTask(
            task_id=f"{state.workflow_id}_stats",
            description="Perform statistical analysis",
            task_type="statistical_analysis"
        )
        result = await agent.process(task, state.shared_context)

        state.agents["statistical_analyst"] = AgentState(
            agent_id="statistical_analyst",
            status="completed",
            output_data={"analysis": result.output}
        )
        return state

    async def _hypothesis_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Hypothesis analysis node for parallel workflows"""
        agent = self.agent_pool.get_agent("hypothesis_generator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_hyp_analysis",
            description="Analyze hypotheses",
            task_type="hypothesis_analysis"
        )
        result = await agent.process(task, state.shared_context)

        state.agents["hypothesis_generator"] = AgentState(
            agent_id="hypothesis_generator",
            status="completed",
            output_data={"analysis": result.output}
        )
        return state

    async def _synthesis_node(self, state: WorkflowState) -> WorkflowState:
        """Synthesis node for combining parallel outputs"""
        # Combine outputs from parallel agents
        synthesis_content = {
            "literature": state.agents.get("literature_analyst", {}).get("output_data", {}),
            "statistics": state.agents.get("statistical_analyst", {}).get("output_data", {}),
            "hypotheses": state.agents.get("hypothesis_generator", {}).get("output_data", {})
        }

        state.shared_context["synthesis"] = synthesis_content
        state.completed_steps.append("synthesis")
        return state

    # Additional node implementations for clinical validation workflow
    async def _clinical_planning_node(self, state: WorkflowState) -> WorkflowState:
        """Clinical planning node"""
        agent = self.agent_pool.get_agent("clinical_validator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_clinical_planning",
            description="Design clinical validation study",
            task_type="validation"
        )
        result = await agent.process(task, state.shared_context)

        state.agents["clinical_validator"] = AgentState(
            agent_id="clinical_validator",
            status="completed",
            output_data={"plan": result.output}
        )
        state.shared_context["clinical_plan"] = result.output
        return state

    async def _regulatory_assessment_node(self, state: WorkflowState) -> WorkflowState:
        """Regulatory assessment node"""
        agent = self.agent_pool.get_agent("clinical_validator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_regulatory",
            description="Assess regulatory requirements",
            task_type="regulatory"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["regulatory_assessment"] = result.output
        return state

    async def _safety_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Safety analysis node"""
        agent = self.agent_pool.get_agent("clinical_validator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_safety",
            description="Evaluate safety profile",
            task_type="safety"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["safety_analysis"] = result.output
        return state

    async def _efficacy_design_node(self, state: WorkflowState) -> WorkflowState:
        """Efficacy study design node"""
        agent = self.agent_pool.get_agent("clinical_validator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_efficacy",
            description="Design efficacy study",
            task_type="efficacy"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["efficacy_design"] = result.output
        return state

    async def _validation_synthesis_node(self, state: WorkflowState) -> WorkflowState:
        """Validation synthesis node"""
        synthesis = {
            "clinical_plan": state.shared_context.get("clinical_plan", ""),
            "regulatory_assessment": state.shared_context.get("regulatory_assessment", ""),
            "safety_analysis": state.shared_context.get("safety_analysis", ""),
            "efficacy_design": state.shared_context.get("efficacy_design", "")
        }

        state.shared_context["validation_synthesis"] = synthesis
        state.completed_steps.append("validation_synthesis")
        return state

    # Additional node implementations for hypothesis workflow
    async def _gap_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Research gap analysis node"""
        agent = self.agent_pool.get_agent("literature_analyst")
        task = AgentTask(
            task_id=f"{state.workflow_id}_gaps",
            description="Identify research gaps",
            task_type="gaps"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["gap_analysis"] = result.output
        return state

    async def _hypothesis_refinement_node(self, state: WorkflowState) -> WorkflowState:
        """Hypothesis refinement node"""
        agent = self.agent_pool.get_agent("hypothesis_generator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_refinement",
            description="Refine research hypotheses",
            task_type="refine"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["refined_hypotheses"] = result.output
        return state

    async def _test_design_node(self, state: WorkflowState) -> WorkflowState:
        """Test design node"""
        agent = self.agent_pool.get_agent("hypothesis_generator")
        task = AgentTask(
            task_id=f"{state.workflow_id}_test_design",
            description="Design hypothesis tests",
            task_type="test"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["test_design"] = result.output
        return state

    async def _feasibility_check_node(self, state: WorkflowState) -> WorkflowState:
        """Feasibility check node"""
        agent = self.agent_pool.get_agent("statistical_analyst")
        task = AgentTask(
            task_id=f"{state.workflow_id}_feasibility",
            description="Check study feasibility",
            task_type="feasibility"
        )
        result = await agent.process(task, state.shared_context)

        state.shared_context["feasibility_assessment"] = result.output
        state.completed_steps.append("feasibility_check")
        return state

    def _assess_workflow_quality(self, state: WorkflowState) -> float:
        """Assess overall quality of workflow output"""
        completed_agents = len([a for a in state.agents.values() if a.status == "completed"])
        total_agents = len(state.agents)

        if total_agents == 0:
            return 0.0

        completion_score = completed_agents / total_agents

        # Quality indicators
        has_literature = bool(state.shared_context.get("literature_analysis"))
        has_hypotheses = bool(state.shared_context.get("research_hypotheses"))
        has_statistics = bool(state.shared_context.get("statistical_plan"))
        has_grant = bool(state.shared_context.get("grant_proposal"))
        has_validation = bool(state.shared_context.get("clinical_validation"))

        content_score = sum([has_literature, has_hypotheses, has_statistics, has_grant, has_validation]) / 5

        return (completion_score * 0.6 + content_score * 0.4)

    # Fallback implementations for environments without LangGraph
    def _create_fallback_workflow(self, workflow_type: str) -> Dict[str, Any]:
        """Create fallback workflow implementation"""
        return {
            "type": workflow_type,
            "implementation": "fallback",
            "execute": self._execute_fallback_workflow
        }

    async def _execute_fallback_workflow(self, workflow_type: str, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow without LangGraph"""

        workflow_id = f"fallback_{workflow_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=WorkflowType(workflow_type.replace("_", "")),
            shared_context=initial_context
        )

        try:
            if workflow_type == "samsung_grant":
                # Sequential execution
                await self._literature_analysis_node(state)
                await self._hypothesis_generation_node(state)
                await self._statistical_planning_node(state)
                await self._grant_writing_node(state)
                await self._clinical_validation_node(state)
                await self._supervisor_review_node(state)

            elif workflow_type == "research_analysis":
                # Parallel execution simulation
                tasks = [
                    self._literature_analysis_node(state),
                    self._statistical_analysis_node(state),
                    self._hypothesis_analysis_node(state)
                ]
                await asyncio.gather(*tasks)
                await self._synthesis_node(state)

            elif workflow_type == "clinical_validation":
                # Sequential clinical workflow
                await self._clinical_planning_node(state)
                await self._regulatory_assessment_node(state)
                await self._safety_analysis_node(state)
                await self._efficacy_design_node(state)
                await self._validation_synthesis_node(state)

            elif workflow_type == "hypothesis_generation":
                # Hypothesis workflow
                await self._gap_analysis_node(state)
                await self._hypothesis_generation_node(state)
                await self._hypothesis_refinement_node(state)
                await self._test_design_node(state)
                await self._feasibility_check_node(state)

        except Exception as e:
            logger.error(f"Fallback workflow execution error: {e}")
            state.shared_context["error"] = str(e)

        return {
            "workflow_id": workflow_id,
            "state": state,
            "outputs": state.shared_context,
            "quality_score": self._assess_workflow_quality(state)
        }

    # Public Interface Methods
    async def execute_workflow(self,
                             workflow_type: str,
                             initial_context: Dict[str, Any],
                             workflow_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a multi-agent workflow"""

        if workflow_type not in self.workflows:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        workflow_config = workflow_config or {}
        workflow_id = f"{workflow_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            if LANGGRAPH_AVAILABLE and hasattr(self.workflows[workflow_type], 'invoke'):
                # Use LangGraph execution
                initial_state = WorkflowState(
                    workflow_id=workflow_id,
                    workflow_type=WorkflowType(workflow_type),
                    shared_context=initial_context
                )

                result = await self.workflows[workflow_type].ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": workflow_id}}
                )

                return {
                    "workflow_id": workflow_id,
                    "execution_type": "langgraph",
                    "state": result,
                    "outputs": result.shared_context,
                    "quality_score": self._assess_workflow_quality(result)
                }
            else:
                # Use fallback execution
                result = await self._execute_fallback_workflow(workflow_type, initial_context)
                result["execution_type"] = "fallback"
                return result

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "workflow_id": workflow_id,
                "execution_type": "error",
                "error": str(e),
                "outputs": {},
                "quality_score": 0.0
            }

    async def execute_parallel_agents(self,
                                    agent_tasks: List[Dict[str, Any]]) -> List[AgentResult]:
        """Execute multiple agents in parallel"""

        tasks = []
        for task_config in agent_tasks:
            agent_id = task_config["agent_id"]
            task = AgentTask(**task_config["task"])
            context = task_config.get("context", {})

            agent = self.agent_pool.get_agent(agent_id)
            if agent:
                tasks.append(agent.process(task, context))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Parallel agent execution error: {result}")
                    processed_results.append(AgentResult(
                        agent_id="unknown",
                        task_id="error",
                        output=f"Error: {str(result)}",
                        confidence=0.0
                    ))
                else:
                    processed_results.append(result)

            return processed_results
        else:
            return []

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of running workflow"""

        # In production, this would query the memory/checkpoint system
        # For now, return basic status information
        return {
            "workflow_id": workflow_id,
            "status": "completed",  # Would be dynamic in production
            "available_workflows": list(self.workflows.keys()),
            "langgraph_available": LANGGRAPH_AVAILABLE
        }

    def list_available_workflows(self) -> List[str]:
        """List available workflow types"""
        return list(self.workflows.keys())

# Convenience functions for common patterns
async def create_samsung_grant_proposal(agent_pool: AgentPool,
                                      research_context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for Samsung grant proposal generation"""

    orchestrator = LangGraphOrchestrator(agent_pool)
    return await orchestrator.execute_workflow("samsung_grant", research_context)

async def parallel_research_analysis(agent_pool: AgentPool,
                                   research_question: str) -> Dict[str, Any]:
    """Convenience function for parallel research analysis"""

    orchestrator = LangGraphOrchestrator(agent_pool)
    context = {"research_question": research_question}
    return await orchestrator.execute_workflow("research_analysis", context)

async def validate_clinical_approach(agent_pool: AgentPool,
                                   research_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for clinical validation workflow"""

    orchestrator = LangGraphOrchestrator(agent_pool)
    return await orchestrator.execute_workflow("clinical_validation", research_plan)

# Usage example and testing
if __name__ == "__main__":
    async def test_orchestrator():
        """Test the orchestration system"""

        # Mock agent pool for testing
        class MockAgentPool:
            def get_agent(self, agent_id: str):
                return MockAgent(agent_id)

        class MockAgent:
            def __init__(self, agent_id: str):
                self.agent_id = agent_id

            async def process(self, task, context):
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    output=f"Mock output from {self.agent_id}",
                    confidence=0.9
                )

        # Test orchestrator
        agent_pool = MockAgentPool()
        orchestrator = LangGraphOrchestrator(agent_pool)

        # Test Samsung grant workflow
        result = await orchestrator.execute_workflow(
            "samsung_grant",
            {
                "research_topic": "AI-based autism diagnosis",
                "budget": 5000000000,
                "duration": "5 years"
            }
        )

        print("Orchestration Test Results:")
        print(f"Workflow ID: {result['workflow_id']}")
        print(f"Execution Type: {result['execution_type']}")
        print(f"Quality Score: {result['quality_score']:.2f}")
        print(f"Available Workflows: {orchestrator.list_available_workflows()}")

    # Run test if module executed directly
    asyncio.run(test_orchestrator())
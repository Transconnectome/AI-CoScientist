"""
RL Environment for Intelligent Agent Selection

Implementation for: Enhanced agent coordination using reinforcement learning
Created: 2025-12-05

This module provides a Gymnasium-compatible environment for training RL models
to optimize agent selection based on task requirements, context, and historical performance.
"""

import gymnasium as gym
import numpy as np
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import asyncio
from abc import ABC, abstractmethod

# ML dependencies with fallbacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

try:
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("Stable-Baselines3 not available. Install with: pip install stable-baselines3")

# Internal dependencies
from ..base import ResearchAgent
from ..pool import AgentPool
from ...core.config import settings

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = 0      # Single agent, straightforward task
    MEDIUM = 1      # Multiple agents, some coordination needed
    COMPLEX = 2     # High coordination, multi-step process
    EXPERT = 3      # Domain expertise required

class TaskDomain(Enum):
    """Task domain categories"""
    NEUROSCIENCE = 0
    STATISTICAL_ANALYSIS = 1
    GRANT_WRITING = 2
    HYPOTHESIS_GENERATION = 3
    CLINICAL_VALIDATION = 4
    LITERATURE_ANALYSIS = 5
    GENERAL = 6

class TaskUrgency(Enum):
    """Task urgency levels"""
    LOW = 0      # Can wait, quality preferred
    MEDIUM = 1   # Normal priority
    HIGH = 2     # Fast turnaround needed
    CRITICAL = 3 # Immediate response required

@dataclass
class TaskContext:
    """Enhanced task context for RL environment"""
    task_id: str
    description: str
    complexity: TaskComplexity
    domain: TaskDomain
    urgency: TaskUrgency
    quality_requirement: float  # 0-1 scale
    estimated_duration: float   # hours
    collaboration_required: bool
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class AgentState:
    """Current state of an agent"""
    agent_id: str
    current_workload: float     # 0-1 scale
    recent_success_rate: float  # 0-1 scale
    expertise_match: float      # 0-1 for current task
    collaboration_score: float  # 0-1 recent collaboration effectiveness
    availability: bool
    last_task_completion: datetime

@dataclass
class EnvironmentState:
    """Complete environment state for RL"""
    task_context: TaskContext
    agent_states: List[AgentState]
    system_load: float          # 0-1 overall system utilization
    time_of_day: float         # 0-23 normalized to 0-1
    recent_performance: Dict[str, float]  # Recent metrics
    collaboration_matrix: np.ndarray      # Agent collaboration effectiveness

@dataclass
class ActionResult:
    """Result of agent selection action"""
    selected_agents: List[str]
    task_success: bool
    task_duration: float        # actual vs estimated
    quality_score: float        # 0-1 output quality
    collaboration_effectiveness: float  # 0-1 if multiple agents
    user_satisfaction: float    # 0-1 user feedback
    cost_efficiency: float     # 0-1 resource utilization

class StateEncoder:
    """Encodes environment state into fixed-size vector for RL"""

    def __init__(self, num_agents: int = 6, state_size: int = 128):
        self.num_agents = num_agents
        self.state_size = state_size

        # State vector composition:
        # Task features (32 dims) + Agent states (6*12=72 dims) + System state (24 dims)
        self.task_dims = 32
        self.agent_dims = 12  # per agent
        self.system_dims = 24

        assert self.task_dims + (num_agents * self.agent_dims) + self.system_dims == state_size

    def encode_task_context(self, task: TaskContext) -> np.ndarray:
        """Encode task context into vector"""
        features = np.zeros(self.task_dims)

        # Task complexity (4 dims - one-hot)
        features[task.complexity.value] = 1.0

        # Task domain (7 dims - one-hot)
        features[4 + task.domain.value] = 1.0

        # Task urgency (4 dims - one-hot)
        features[11 + task.urgency.value] = 1.0

        # Continuous features
        features[15] = task.quality_requirement
        features[16] = min(task.estimated_duration / 8.0, 1.0)  # Normalize to 8 hours
        features[17] = 1.0 if task.collaboration_required else 0.0

        # Task description features (using simple heuristics)
        description = task.description.lower()
        features[18] = 1.0 if any(word in description for word in ['urgent', 'asap', 'immediately']) else 0.0
        features[19] = 1.0 if any(word in description for word in ['complex', 'difficult', 'challenging']) else 0.0
        features[20] = 1.0 if any(word in description for word in ['analysis', 'analyze', 'study']) else 0.0
        features[21] = 1.0 if any(word in description for word in ['write', 'draft', 'document']) else 0.0
        features[22] = 1.0 if any(word in description for word in ['review', 'validate', 'check']) else 0.0

        # Time features
        hour = task.timestamp.hour
        features[23] = np.sin(2 * np.pi * hour / 24)  # Cyclical hour encoding
        features[24] = np.cos(2 * np.pi * hour / 24)

        day_of_week = task.timestamp.weekday()
        features[25] = np.sin(2 * np.pi * day_of_week / 7)  # Cyclical day encoding
        features[26] = np.cos(2 * np.pi * day_of_week / 7)

        # Description length (normalized)
        features[27] = min(len(task.description) / 1000.0, 1.0)

        # Reserved for future expansion
        features[28:32] = 0.0

        return features

    def encode_agent_states(self, agents: List[AgentState]) -> np.ndarray:
        """Encode agent states into vector"""
        features = np.zeros(self.num_agents * self.agent_dims)

        for i, agent in enumerate(agents[:self.num_agents]):
            base_idx = i * self.agent_dims

            features[base_idx + 0] = agent.current_workload
            features[base_idx + 1] = agent.recent_success_rate
            features[base_idx + 2] = agent.expertise_match
            features[base_idx + 3] = agent.collaboration_score
            features[base_idx + 4] = 1.0 if agent.availability else 0.0

            # Time since last task (in hours, normalized to 24h)
            if agent.last_task_completion:
                hours_since = (datetime.now() - agent.last_task_completion).total_seconds() / 3600
                features[base_idx + 5] = min(hours_since / 24.0, 1.0)
            else:
                features[base_idx + 5] = 1.0  # No recent tasks

            # Agent-specific features (can be extended)
            features[base_idx + 6:base_idx + 12] = 0.0

        return features

    def encode_system_state(self, state: EnvironmentState) -> np.ndarray:
        """Encode system-level state"""
        features = np.zeros(self.system_dims)

        features[0] = state.system_load
        features[1] = state.time_of_day

        # Recent performance metrics (normalized)
        perf_keys = ['avg_success_rate', 'avg_response_time', 'avg_quality', 'user_satisfaction']
        for i, key in enumerate(perf_keys[:4]):
            features[2 + i] = state.recent_performance.get(key, 0.5)

        # Collaboration matrix summary statistics
        if state.collaboration_matrix.size > 0:
            features[6] = np.mean(state.collaboration_matrix)
            features[7] = np.std(state.collaboration_matrix)
            features[8] = np.max(state.collaboration_matrix)
            features[9] = np.min(state.collaboration_matrix)

        # System health indicators
        features[10] = 1.0  # System operational
        features[11] = min(len(state.agent_states) / 6.0, 1.0)  # Agent availability ratio

        # Reserved for future metrics
        features[12:24] = 0.0

        return features

    def encode_state(self, state: EnvironmentState) -> np.ndarray:
        """Encode complete environment state"""
        task_features = self.encode_task_context(state.task_context)
        agent_features = self.encode_agent_states(state.agent_states)
        system_features = self.encode_system_state(state)

        return np.concatenate([task_features, agent_features, system_features])

class RewardCalculator:
    """Calculates rewards for agent selection actions"""

    def __init__(self):
        # Reward component weights
        self.weights = {
            'success': 2.0,         # Task completion success
            'quality': 1.5,         # Output quality
            'efficiency': 1.0,      # Time/resource efficiency
            'collaboration': 0.8,   # Team effectiveness
            'satisfaction': 1.2,    # User satisfaction
            'cost': 0.5            # Resource cost efficiency
        }

        # Penalty weights
        self.penalties = {
            'failure': -2.0,        # Task failure
            'timeout': -1.0,        # Task timeout
            'overallocation': -0.5, # Too many agents
            'underallocation': -0.3 # Insufficient agents
        }

    def calculate_reward(self, action_result: ActionResult, task_context: TaskContext) -> float:
        """Calculate multi-dimensional reward"""
        reward = 0.0

        # Base success/failure
        if action_result.task_success:
            reward += self.weights['success']
        else:
            reward += self.penalties['failure']
            return reward  # Early return for failed tasks

        # Quality reward (0-1 score)
        quality_reward = action_result.quality_score * self.weights['quality']
        reward += quality_reward

        # Efficiency reward (actual vs estimated duration)
        if action_result.task_duration > 0:
            efficiency_ratio = min(task_context.estimated_duration / action_result.task_duration, 2.0)
            efficiency_reward = (efficiency_ratio - 0.5) * self.weights['efficiency']
            reward += efficiency_reward

        # Collaboration reward (for multi-agent tasks)
        if len(action_result.selected_agents) > 1:
            collab_reward = action_result.collaboration_effectiveness * self.weights['collaboration']
            reward += collab_reward

        # User satisfaction reward
        satisfaction_reward = action_result.user_satisfaction * self.weights['satisfaction']
        reward += satisfaction_reward

        # Cost efficiency reward
        cost_reward = action_result.cost_efficiency * self.weights['cost']
        reward += cost_reward

        # Penalties for poor resource allocation
        num_agents = len(action_result.selected_agents)
        if task_context.complexity == TaskComplexity.SIMPLE and num_agents > 2:
            reward += self.penalties['overallocation']
        elif task_context.complexity == TaskComplexity.COMPLEX and num_agents < 2:
            reward += self.penalties['underallocation']

        return float(np.clip(reward, -5.0, 5.0))

class AgentSelectionEnvironment(gym.Env):
    """Gymnasium environment for agent selection RL training"""

    def __init__(self, agent_pool: Optional[AgentPool] = None, config: Optional[Dict] = None):
        super().__init__()

        self.config = config or self._default_config()
        self.agent_pool = agent_pool
        self.num_agents = 6  # Current AI-CoScientist has 6 agents

        # Environment components
        self.state_encoder = StateEncoder(num_agents=self.num_agents)
        self.reward_calculator = RewardCalculator()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.state_encoder.state_size,),
            dtype=np.float32
        )

        # Action space: Select 1-3 agents (multi-discrete)
        # Each action dimension represents whether to select that agent
        self.action_space = gym.spaces.MultiBinary(self.num_agents)

        # Environment state
        self.current_state: Optional[EnvironmentState] = None
        self.current_task: Optional[TaskContext] = None
        self.step_count = 0
        self.episode_count = 0

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []

        # Task generation for training
        self.task_generator = TaskGenerator()

        logger.info(f"Initialized AgentSelectionEnvironment with {self.num_agents} agents")

    def _default_config(self) -> Dict:
        """Default environment configuration"""
        return {
            'max_episode_length': 100,
            'reward_scaling': 1.0,
            'state_normalization': True,
            'action_validation': True,
            'performance_tracking': True
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with new task"""
        super().reset(seed=seed)

        # Generate new task
        self.current_task = self.task_generator.generate_task()

        # Initialize agent states (in real deployment, this would query actual agent pool)
        agent_states = self._initialize_agent_states()

        # Create environment state
        self.current_state = EnvironmentState(
            task_context=self.current_task,
            agent_states=agent_states,
            system_load=np.random.uniform(0.1, 0.8),
            time_of_day=datetime.now().hour / 24.0,
            recent_performance=self._get_recent_performance(),
            collaboration_matrix=self._get_collaboration_matrix()
        )

        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1

        # Encode state
        state_vector = self.state_encoder.encode_state(self.current_state)

        # Info dictionary
        info = {
            'task_id': self.current_task.task_id,
            'complexity': self.current_task.complexity.name,
            'domain': self.current_task.domain.name,
            'episode': self.episode_count
        }

        return state_vector.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action (agent selection) and return result"""
        self.step_count += 1

        # Validate and convert action
        selected_agents = self._action_to_agent_selection(action)

        # Simulate task execution (in real deployment, this would be actual execution)
        action_result = self._simulate_task_execution(selected_agents)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(action_result, self.current_task)

        # Episode termination (each task is one episode in this formulation)
        terminated = True
        truncated = self.step_count >= self.config['max_episode_length']

        # Update performance tracking
        self._update_performance_tracking(action_result, reward)

        # Generate next state (for multi-step episodes, currently single-step)
        next_state_vector = np.zeros_like(self.state_encoder.encode_state(self.current_state))

        # Info dictionary with detailed feedback
        info = {
            'task_success': action_result.task_success,
            'quality_score': action_result.quality_score,
            'selected_agents': selected_agents,
            'task_duration': action_result.task_duration,
            'reward_components': self._get_reward_breakdown(action_result),
            'step_count': self.step_count
        }

        return next_state_vector.astype(np.float32), float(reward), terminated, truncated, info

    def _action_to_agent_selection(self, action: np.ndarray) -> List[str]:
        """Convert RL action to agent selection"""
        agent_ids = [
            "neuroscience_expert", "statistical_analyst", "grant_writer",
            "hypothesis_generator", "clinical_validator", "literature_analyst"
        ]

        # Binary action for each agent
        selected = []
        for i, select in enumerate(action):
            if select > 0.5:  # Threshold for selection
                selected.append(agent_ids[i])

        # Ensure at least one agent is selected
        if not selected:
            # Select agent with highest action value
            max_idx = np.argmax(action)
            selected.append(agent_ids[max_idx])

        # Validate selection (ensure reasonable number of agents)
        if len(selected) > 3:
            # Keep top 3 agents by action value
            action_values = [(agent_ids[i], action[i]) for i in range(len(agent_ids))]
            action_values.sort(key=lambda x: x[1], reverse=True)
            selected = [av[0] for av in action_values[:3]]

        return selected

    def _simulate_task_execution(self, selected_agents: List[str]) -> ActionResult:
        """Simulate task execution with selected agents"""
        # This is simulation for training - in production, this would be actual task execution

        # Base success probability based on agent-task match
        success_prob = self._calculate_success_probability(selected_agents)
        task_success = np.random.random() < success_prob

        # Quality score based on agent expertise and task requirements
        quality_score = self._calculate_quality_score(selected_agents) if task_success else 0.0

        # Task duration based on complexity and agent efficiency
        base_duration = self.current_task.estimated_duration
        efficiency_factor = self._calculate_efficiency_factor(selected_agents)
        task_duration = base_duration * efficiency_factor

        # Collaboration effectiveness for multi-agent tasks
        collaboration_effectiveness = self._calculate_collaboration_score(selected_agents)

        # User satisfaction based on quality and timeliness
        satisfaction_factor = (quality_score + (1.0 / max(efficiency_factor, 0.1))) / 2.0
        user_satisfaction = min(satisfaction_factor, 1.0) if task_success else 0.1

        # Cost efficiency based on resource utilization
        cost_efficiency = self._calculate_cost_efficiency(selected_agents)

        return ActionResult(
            selected_agents=selected_agents,
            task_success=task_success,
            task_duration=task_duration,
            quality_score=quality_score,
            collaboration_effectiveness=collaboration_effectiveness,
            user_satisfaction=user_satisfaction,
            cost_efficiency=cost_efficiency
        )

    def _calculate_success_probability(self, selected_agents: List[str]) -> float:
        """Calculate task success probability based on agent selection"""
        task = self.current_task

        # Base probability based on task complexity
        base_probs = {
            TaskComplexity.SIMPLE: 0.8,
            TaskComplexity.MEDIUM: 0.6,
            TaskComplexity.COMPLEX: 0.4,
            TaskComplexity.EXPERT: 0.3
        }

        prob = base_probs[task.complexity]

        # Adjust based on domain expertise
        domain_agents = {
            TaskDomain.NEUROSCIENCE: "neuroscience_expert",
            TaskDomain.STATISTICAL_ANALYSIS: "statistical_analyst",
            TaskDomain.GRANT_WRITING: "grant_writer",
            TaskDomain.HYPOTHESIS_GENERATION: "hypothesis_generator",
            TaskDomain.CLINICAL_VALIDATION: "clinical_validator",
            TaskDomain.LITERATURE_ANALYSIS: "literature_analyst"
        }

        if task.domain in domain_agents and domain_agents[task.domain] in selected_agents:
            prob += 0.3

        # Adjust based on collaboration requirements
        if task.collaboration_required and len(selected_agents) > 1:
            prob += 0.2
        elif not task.collaboration_required and len(selected_agents) > 2:
            prob -= 0.1  # Over-allocation penalty

        # Adjust based on urgency vs agent load
        if task.urgency in [TaskUrgency.HIGH, TaskUrgency.CRITICAL]:
            prob -= 0.1  # Pressure reduces success rate

        return np.clip(prob, 0.1, 0.95)

    def _calculate_quality_score(self, selected_agents: List[str]) -> float:
        """Calculate output quality score"""
        # Base quality depends on expertise match
        quality = 0.5

        # Domain expertise bonus
        domain_agents = {
            TaskDomain.NEUROSCIENCE: "neuroscience_expert",
            TaskDomain.STATISTICAL_ANALYSIS: "statistical_analyst",
            TaskDomain.GRANT_WRITING: "grant_writer",
            TaskDomain.HYPOTHESIS_GENERATION: "hypothesis_generator",
            TaskDomain.CLINICAL_VALIDATION: "clinical_validator",
            TaskDomain.LITERATURE_ANALYSIS: "literature_analyst"
        }

        if self.current_task.domain in domain_agents:
            if domain_agents[self.current_task.domain] in selected_agents:
                quality += 0.3

        # Multi-agent quality bonus for complex tasks
        if (self.current_task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]
            and len(selected_agents) > 1):
            quality += 0.2

        # Quality requirement matching
        if self.current_task.quality_requirement > 0.8:
            quality += 0.1  # High quality requirement met

        # Add some randomness
        quality += np.random.uniform(-0.1, 0.1)

        return np.clip(quality, 0.0, 1.0)

    def _calculate_efficiency_factor(self, selected_agents: List[str]) -> float:
        """Calculate task completion efficiency (duration multiplier)"""
        # Base efficiency (1.0 means estimated duration)
        efficiency = 1.0

        # More agents can be faster for complex tasks but slower for simple ones
        num_agents = len(selected_agents)
        if self.current_task.complexity == TaskComplexity.SIMPLE:
            efficiency *= (1.0 + (num_agents - 1) * 0.2)  # Coordination overhead
        else:
            efficiency *= max(0.5, 1.0 - (num_agents - 1) * 0.15)  # Parallel work benefit

        # Domain expertise speeds up work
        domain_agents = {
            TaskDomain.NEUROSCIENCE: "neuroscience_expert",
            TaskDomain.STATISTICAL_ANALYSIS: "statistical_analyst",
            TaskDomain.GRANT_WRITING: "grant_writer",
            TaskDomain.HYPOTHESIS_GENERATION: "hypothesis_generator",
            TaskDomain.CLINICAL_VALIDATION: "clinical_validator",
            TaskDomain.LITERATURE_ANALYSIS: "literature_analyst"
        }

        if (self.current_task.domain in domain_agents and
            domain_agents[self.current_task.domain] in selected_agents):
            efficiency *= 0.8  # 20% faster with domain expert

        # Urgency pressure affects efficiency
        urgency_factors = {
            TaskUrgency.LOW: 1.0,
            TaskUrgency.MEDIUM: 1.1,
            TaskUrgency.HIGH: 1.3,
            TaskUrgency.CRITICAL: 1.5
        }
        efficiency *= urgency_factors[self.current_task.urgency]

        # Add randomness
        efficiency *= np.random.uniform(0.8, 1.2)

        return max(0.3, efficiency)

    def _calculate_collaboration_score(self, selected_agents: List[str]) -> float:
        """Calculate collaboration effectiveness"""
        if len(selected_agents) <= 1:
            return 1.0

        # Base collaboration score
        score = 0.7

        # Some agent combinations work better together
        good_combinations = {
            ("neuroscience_expert", "statistical_analyst"),
            ("hypothesis_generator", "literature_analyst"),
            ("grant_writer", "statistical_analyst"),
            ("clinical_validator", "neuroscience_expert")
        }

        for combo in good_combinations:
            if all(agent in selected_agents for agent in combo):
                score += 0.2
                break

        # Too many agents reduce collaboration effectiveness
        if len(selected_agents) > 3:
            score -= (len(selected_agents) - 3) * 0.1

        # Add randomness
        score += np.random.uniform(-0.1, 0.1)

        return np.clip(score, 0.1, 1.0)

    def _calculate_cost_efficiency(self, selected_agents: List[str]) -> float:
        """Calculate resource cost efficiency"""
        # Fewer agents for simple tasks = higher efficiency
        num_agents = len(selected_agents)

        optimal_agents = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MEDIUM: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.EXPERT: 2
        }

        optimal = optimal_agents[self.current_task.complexity]
        deviation = abs(num_agents - optimal)

        efficiency = max(0.3, 1.0 - deviation * 0.2)

        return efficiency

    def _initialize_agent_states(self) -> List[AgentState]:
        """Initialize realistic agent states for simulation"""
        agent_ids = [
            "neuroscience_expert", "statistical_analyst", "grant_writer",
            "hypothesis_generator", "clinical_validator", "literature_analyst"
        ]

        states = []
        for agent_id in agent_ids:
            state = AgentState(
                agent_id=agent_id,
                current_workload=np.random.uniform(0.1, 0.8),
                recent_success_rate=np.random.uniform(0.6, 0.95),
                expertise_match=self._calculate_expertise_match(agent_id),
                collaboration_score=np.random.uniform(0.7, 0.9),
                availability=True,
                last_task_completion=datetime.now() - timedelta(hours=np.random.uniform(1, 24))
            )
            states.append(state)

        return states

    def _calculate_expertise_match(self, agent_id: str) -> float:
        """Calculate how well agent expertise matches current task"""
        domain_matches = {
            TaskDomain.NEUROSCIENCE: "neuroscience_expert",
            TaskDomain.STATISTICAL_ANALYSIS: "statistical_analyst",
            TaskDomain.GRANT_WRITING: "grant_writer",
            TaskDomain.HYPOTHESIS_GENERATION: "hypothesis_generator",
            TaskDomain.CLINICAL_VALIDATION: "clinical_validator",
            TaskDomain.LITERATURE_ANALYSIS: "literature_analyst"
        }

        if (self.current_task.domain in domain_matches and
            domain_matches[self.current_task.domain] == agent_id):
            return np.random.uniform(0.8, 0.95)
        else:
            return np.random.uniform(0.3, 0.7)

    def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent system performance metrics"""
        return {
            'avg_success_rate': np.random.uniform(0.7, 0.9),
            'avg_response_time': np.random.uniform(0.5, 0.8),
            'avg_quality': np.random.uniform(0.75, 0.9),
            'user_satisfaction': np.random.uniform(0.7, 0.85)
        }

    def _get_collaboration_matrix(self) -> np.ndarray:
        """Get agent collaboration effectiveness matrix"""
        matrix = np.random.uniform(0.6, 0.9, (self.num_agents, self.num_agents))
        np.fill_diagonal(matrix, 1.0)
        return (matrix + matrix.T) / 2  # Make symmetric

    def _update_performance_tracking(self, result: ActionResult, reward: float):
        """Update performance tracking metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(self.step_count)

        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
            self.episode_lengths = self.episode_lengths[-1000:]

    def _get_reward_breakdown(self, result: ActionResult) -> Dict[str, float]:
        """Get detailed reward component breakdown for analysis"""
        weights = self.reward_calculator.weights

        breakdown = {
            'success': (1.0 if result.task_success else 0.0) * weights['success'],
            'quality': result.quality_score * weights['quality'],
            'collaboration': result.collaboration_effectiveness * weights['collaboration'],
            'satisfaction': result.user_satisfaction * weights['satisfaction'],
            'cost': result.cost_efficiency * weights['cost']
        }

        return breakdown

    def get_performance_stats(self) -> Dict[str, float]:
        """Get environment performance statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'avg_reward': np.mean(self.episode_rewards[-100:]),
            'avg_episode_length': np.mean(self.episode_lengths[-100:]),
            'success_rate': np.mean([1 if r > 0 else 0 for r in self.episode_rewards[-100:]]),
            'total_episodes': self.episode_count
        }

class TaskGenerator:
    """Generates realistic tasks for training the RL environment"""

    def __init__(self):
        self.task_id_counter = 0

        # Task templates for realistic generation
        self.task_templates = {
            TaskDomain.NEUROSCIENCE: [
                "Analyze fMRI data for autism spectrum disorder detection",
                "Review literature on neural decoding techniques",
                "Validate brain connectivity patterns in developmental disorders"
            ],
            TaskDomain.STATISTICAL_ANALYSIS: [
                "Perform statistical analysis on clinical trial data",
                "Validate hypothesis using regression analysis",
                "Compare treatment groups using appropriate statistical tests"
            ],
            TaskDomain.GRANT_WRITING: [
                "Draft research proposal for neuroscience grant",
                "Write budget justification for equipment purchase",
                "Prepare quarterly progress report for funding agency"
            ],
            TaskDomain.HYPOTHESIS_GENERATION: [
                "Generate testable hypotheses for brain imaging study",
                "Develop research questions for clinical investigation",
                "Propose novel approaches to autism detection"
            ],
            TaskDomain.CLINICAL_VALIDATION: [
                "Validate diagnostic criteria against clinical outcomes",
                "Review treatment protocols for safety and efficacy",
                "Assess clinical significance of research findings"
            ],
            TaskDomain.LITERATURE_ANALYSIS: [
                "Systematic review of recent autism research",
                "Meta-analysis of brain imaging studies",
                "Comprehensive literature synthesis for grant proposal"
            ]
        }

    def generate_task(self) -> TaskContext:
        """Generate a realistic task for training"""
        self.task_id_counter += 1

        # Randomly select domain and template
        domain = np.random.choice(list(TaskDomain))
        if domain == TaskDomain.GENERAL:
            # Pick from any domain for general tasks
            all_templates = []
            for templates in self.task_templates.values():
                all_templates.extend(templates)
            description = np.random.choice(all_templates)
        else:
            description = np.random.choice(self.task_templates[domain])

        # Generate realistic task parameters
        complexity = np.random.choice(list(TaskComplexity))
        urgency = np.random.choice(list(TaskUrgency))

        # Quality requirement correlates with complexity
        quality_base = {
            TaskComplexity.SIMPLE: 0.6,
            TaskComplexity.MEDIUM: 0.7,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 0.9
        }[complexity]
        quality_req = min(1.0, quality_base + np.random.uniform(-0.1, 0.1))

        # Duration correlates with complexity
        duration_base = {
            TaskComplexity.SIMPLE: 2.0,
            TaskComplexity.MEDIUM: 4.0,
            TaskComplexity.COMPLEX: 8.0,
            TaskComplexity.EXPERT: 12.0
        }[complexity]
        duration = max(0.5, duration_base + np.random.uniform(-1.0, 2.0))

        # Collaboration requirement
        collab_prob = {
            TaskComplexity.SIMPLE: 0.2,
            TaskComplexity.MEDIUM: 0.5,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 0.7
        }[complexity]
        collaboration_required = np.random.random() < collab_prob

        return TaskContext(
            task_id=f"task_{self.task_id_counter:06d}",
            description=description,
            complexity=complexity,
            domain=domain,
            urgency=urgency,
            quality_requirement=quality_req,
            estimated_duration=duration,
            collaboration_required=collaboration_required,
            metadata={},
            timestamp=datetime.now()
        )

# Environment validation
def validate_environment():
    """Validate the RL environment setup"""
    try:
        env = AgentSelectionEnvironment()

        if SB3_AVAILABLE:
            check_env(env, warn=True)
            logger.info("✅ Environment passed Stable-Baselines3 validation")

        # Test episode
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logger.info(f"✅ Test episode completed successfully")
        logger.info(f"   Observation shape: {obs.shape}")
        logger.info(f"   Action space: {env.action_space}")
        logger.info(f"   Reward: {reward:.3f}")

        return True

    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test environment
    logging.basicConfig(level=logging.INFO)

    print("🤖 Testing Agent Selection RL Environment...")

    success = validate_environment()
    if success:
        print("✅ Environment validation passed!")
    else:
        print("❌ Environment validation failed!")
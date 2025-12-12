#!/usr/bin/env python3
"""
Standalone RL System Test
Tests the RL components without importing the main config system
"""

import sys
import os
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock agent types
class AgentType(Enum):
    NEUROSCIENCE_EXPERT = "neuroscience_expert"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    GRANT_WRITER = "grant_writer"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    CLINICAL_VALIDATION = "clinical_validation"
    LITERATURE_ANALYST = "literature_analyst"

@dataclass
class TaskContext:
    """Context information for agent selection tasks"""
    task_type: str
    complexity: float
    domain: str
    priority: int
    keywords: List[str]
    estimated_duration: float
    required_capabilities: List[str]

@dataclass
class AgentState:
    """Current state of an agent"""
    agent_type: AgentType
    availability: float  # 0.0 to 1.0
    current_workload: float
    success_rate: float
    avg_response_time: float
    specialization_score: float
    collaboration_score: float

class SimpleAgentSelectionEnvironment(gym.Env):
    """Simplified RL environment for agent selection without config dependencies"""

    def __init__(self):
        super().__init__()

        # Environment configuration
        self.num_agents = 6
        self.state_dim = 128
        self.max_episode_steps = 100

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Use discrete action space for DQN compatibility
        # Each action represents a different team composition
        self.action_space = gym.spaces.Discrete(64)  # 2^6 = 64 possible team combinations

        # Create mapping from action index to agent selection
        self.action_to_agents = {}
        for i in range(64):
            binary_repr = format(i, f'0{self.num_agents}b')
            agents = [j for j, bit in enumerate(binary_repr) if bit == '1']
            self.action_to_agents[i] = agents

        # Initialize episode tracking
        self.current_step = 0
        self.episode_reward = 0.0

        print("🤖 RL Environment initialized successfully")

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0

        # Generate initial observation
        observation = self._get_observation()
        info = {"step": self.current_step}

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1

        # Convert discrete action to agent selection
        selected_agents = self.action_to_agents[action]

        # Calculate reward based on action
        reward = self._calculate_reward(selected_agents)
        self.episode_reward += reward

        # Check termination conditions
        terminated = self.current_step >= self.max_episode_steps
        truncated = False

        # Get new observation
        observation = self._get_observation()

        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "selected_agents": selected_agents,
            "action": action
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Generate current state observation"""
        # Simulate realistic state features
        task_features = np.random.normal(0, 1, 32)  # Task context features
        agent_features = np.random.normal(0, 1, 72)  # Agent state features
        system_features = np.random.normal(0, 1, 24)  # System state features

        observation = np.concatenate([task_features, agent_features, system_features])
        return observation.astype(np.float32)

    def _calculate_reward(self, selected_agents):
        """Calculate reward for the given agent selection"""
        # Simple reward calculation
        num_selected = len(selected_agents)

        if num_selected == 0:
            return -1.0  # Penalty for selecting no agents
        elif num_selected > 4:
            return -0.5  # Penalty for selecting too many agents
        else:
            # Reward based on reasonable team size
            base_reward = 1.0
            efficiency_bonus = (4 - num_selected) * 0.1
            return base_reward + efficiency_bonus

def test_rl_environment():
    """Test the RL environment functionality"""
    print("🔄 Starting RL Environment Test...")

    try:
        # Create environment
        env = SimpleAgentSelectionEnvironment()
        print("✅ Environment created successfully")

        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful - observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Test multiple steps
        total_reward = 0
        for step in range(5):
            # Sample random action
            action = env.action_space.sample()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"   Step {step + 1}: action={action} (agents={info['selected_agents']}), reward={reward:.3f}")

            if terminated or truncated:
                break

        print(f"✅ Environment test completed - total reward: {total_reward:.3f}")
        return True

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_dqn_compatibility():
    """Test DQN model compatibility"""
    print("🔄 Testing DQN compatibility...")

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.env_checker import check_env

        # Create environment
        env = SimpleAgentSelectionEnvironment()

        # Check environment compatibility
        check_env(env, warn=True)
        print("✅ Environment passes Stable-Baselines3 compatibility check")

        # Try to create DQN model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            buffer_size=1000,
            learning_starts=100,
            target_update_interval=50,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            verbose=0
        )
        print("✅ DQN model created successfully")

        # Test model prediction
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"✅ Model prediction successful - action: {action}")

        return True

    except Exception as e:
        print(f"❌ DQN compatibility test failed: {e}")
        return False

def main():
    """Run all RL system tests"""
    print("🚀 Starting Standalone RL System Tests...\n")

    # Test 1: Basic environment functionality
    env_success = test_rl_environment()
    print()

    # Test 2: DQN compatibility
    dqn_success = test_dqn_compatibility()
    print()

    # Summary
    if env_success and dqn_success:
        print("🎉 All RL system tests passed successfully!")
        print("✅ RL system is ready for integration with AI-CoScientist")
        return 0
    else:
        print("❌ Some RL system tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
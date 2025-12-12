"""
Deep Q-Network for Agent Coordination

Implementation for: RL-enhanced agent selection using DQN
Created: 2025-12-05

This module implements a sophisticated DQN model for optimizing agent selection
and coordination in the AI-CoScientist system using Stable-Baselines3.
"""

import logging
import numpy as np
import json
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

# ML dependencies with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.dqn import MlpPolicy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("Stable-Baselines3 not available. Install with: pip install stable-baselines3")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Install with: pip install wandb")

# Internal dependencies
from .agent_selection_env import AgentSelectionEnvironment, TaskContext, ActionResult
from ..base import ResearchAgent
from ..pool import AgentPool
from ...core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class DQNConfig:
    """Configuration for DQN training"""
    # Model architecture
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 32
    target_update_interval: int = 1000
    gradient_steps: int = 1

    # Exploration
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # Training
    train_freq: int = 4
    gamma: float = 0.99
    tau: float = 1.0  # Hard update

    # Network architecture
    net_arch: List[int] = None
    activation_fn: str = "relu"

    # Training parameters
    total_timesteps: int = 50000
    eval_episodes: int = 10
    eval_freq: int = 1000

    # Safety and quality
    min_success_rate: float = 0.6
    reward_threshold: float = 2.0

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256, 128]

class CustomDQNNetwork(BaseFeaturesExtractor):
    """Custom neural network architecture for agent coordination"""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]  # 128 from StateEncoder

        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            # Task context processing (first 32 dims)
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Agent state processing (next 72 dims = 6 agents * 12 features)
        self.agent_layers = nn.Sequential(
            nn.Linear(72, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # System state processing (last 24 dims)
        self.system_layers = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
        )

        # Feature fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(64 + 64 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations into components
        task_features = observations[:, :32]
        agent_features = observations[:, 32:104]
        system_features = observations[:, 104:128]

        # Process each component
        task_processed = self.feature_layers(task_features)
        agent_processed = self.agent_layers(agent_features)
        system_processed = self.system_layers(system_features)

        # Fuse features
        combined = torch.cat([task_processed, agent_processed, system_processed], dim=1)
        output = self.fusion_layers(combined)

        return output

class AgentCoordinationDQN:
    """DQN-based agent coordination system"""

    def __init__(self,
                 environment: AgentSelectionEnvironment,
                 config: Optional[DQNConfig] = None,
                 model_path: Optional[str] = None):

        self.config = config or DQNConfig()
        self.environment = environment
        self.model_path = model_path or "models/agent_coordination_dqn"

        # Create model directory
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model: Optional[DQN] = None
        self.training_stats = {
            'episodes': 0,
            'total_timesteps': 0,
            'best_mean_reward': float('-inf'),
            'training_history': [],
            'evaluation_history': []
        }

        # Performance tracking
        self.performance_buffer = []
        self.success_rate_window = 100

        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 required for DQN training")

        logger.info("Initialized AgentCoordinationDQN")

    def create_model(self, env) -> DQN:
        """Create DQN model with custom architecture"""

        # Custom policy with our network
        policy_kwargs = dict(
            features_extractor_class=CustomDQNNetwork,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=self.config.net_arch,
            activation_fn=getattr(nn, self.config.activation_fn.upper())
        )

        model = DQN(
            policy=MlpPolicy,
            env=env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_initial_eps=self.config.exploration_initial_eps,
            exploration_final_eps=self.config.exploration_final_eps,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard/",
            verbose=1,
            device='auto'  # Will use CUDA if available
        )

        return model

    def train(self,
              total_timesteps: Optional[int] = None,
              eval_callback: bool = True,
              wandb_tracking: bool = False) -> Dict[str, Any]:
        """Train the DQN model"""

        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 required for training")

        timesteps = total_timesteps or self.config.total_timesteps

        logger.info(f"Starting DQN training for {timesteps} timesteps...")

        # Setup environment monitoring
        monitor_env = Monitor(self.environment)
        vec_env = DummyVecEnv([lambda: monitor_env])

        # Create model
        self.model = self.create_model(vec_env)

        # Setup callbacks
        callbacks = []

        if eval_callback:
            eval_env = DummyVecEnv([lambda: Monitor(AgentSelectionEnvironment())])
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{self.model_path}/best_model",
                log_path=f"{self.model_path}/eval_logs",
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # Early stopping on good performance
        if self.config.reward_threshold > 0:
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=self.config.reward_threshold,
                verbose=1
            )
            callbacks.append(stop_callback)

        # Weights & Biases tracking
        if wandb_tracking and WANDB_AVAILABLE:
            wandb.init(
                project="ai-coscientist-agent-coordination",
                config=asdict(self.config),
                monitor_gym=True
            )

        try:
            # Train model
            start_time = datetime.now()

            self.model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                log_interval=100,
                tb_log_name="agent_coordination_dqn",
                reset_num_timesteps=False,
                progress_bar=True
            )

            training_time = datetime.now() - start_time

            # Save final model
            self.model.save(f"{self.model_path}/final_model")

            # Evaluate trained model
            eval_results = self.evaluate_model()

            # Update training stats
            self.training_stats.update({
                'episodes': self.model.num_timesteps,
                'total_timesteps': self.model.num_timesteps,
                'training_time': training_time.total_seconds(),
                'final_evaluation': eval_results
            })

            # Save training stats
            self._save_training_stats()

            logger.info(f"Training completed in {training_time}")
            logger.info(f"Final evaluation - Mean reward: {eval_results['mean_reward']:.3f}")

            return {
                'success': True,
                'training_time': training_time.total_seconds(),
                'final_mean_reward': eval_results['mean_reward'],
                'total_timesteps': self.model.num_timesteps,
                'model_path': f"{self.model_path}/final_model"
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

        finally:
            if wandb_tracking and WANDB_AVAILABLE:
                wandb.finish()

    def evaluate_model(self,
                      n_eval_episodes: Optional[int] = None,
                      deterministic: bool = True) -> Dict[str, float]:
        """Evaluate trained model performance"""

        if self.model is None:
            raise ValueError("Model not trained or loaded")

        episodes = n_eval_episodes or self.config.eval_episodes

        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: AgentSelectionEnvironment()])

        # Evaluate policy
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=episodes,
            deterministic=deterministic,
            render=False,
            return_episode_rewards=False
        )

        # Collect detailed statistics
        episode_rewards = []
        episode_successes = []
        episode_lengths = []

        for _ in range(episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1

                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_successes.append(1 if reward[0] > 0 else 0)
                    break

        results = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'success_rate': np.mean(episode_successes),
            'mean_episode_length': np.mean(episode_lengths),
            'evaluation_episodes': episodes
        }

        # Update evaluation history
        self.training_stats['evaluation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })

        logger.info(f"Evaluation results: {results}")

        return results

    def predict_agent_selection(self,
                               task_context: TaskContext,
                               deterministic: bool = True) -> Tuple[List[str], float]:
        """Predict optimal agent selection for given task"""

        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Create temporary environment state
        env = AgentSelectionEnvironment()
        env.current_task = task_context
        env.current_state = env._create_state_for_task(task_context)

        # Encode state
        state_vector = env.state_encoder.encode_state(env.current_state)

        # Predict action
        action, _ = self.model.predict(
            state_vector.reshape(1, -1),
            deterministic=deterministic
        )

        # Convert action to agent selection
        selected_agents = env._action_to_agent_selection(action[0])

        # Calculate confidence (based on Q-values if available)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            if hasattr(self.model.policy, 'q_net'):
                q_values = self.model.policy.q_net(state_tensor)
                confidence = torch.softmax(q_values, dim=-1).max().item()
            else:
                confidence = 0.8  # Default confidence

        return selected_agents, confidence

    def update_performance(self,
                          task_context: TaskContext,
                          selected_agents: List[str],
                          result: ActionResult):
        """Update performance tracking with real execution results"""

        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_context.task_id,
            'complexity': task_context.complexity.name,
            'domain': task_context.domain.name,
            'selected_agents': selected_agents,
            'success': result.task_success,
            'quality_score': result.quality_score,
            'duration_ratio': result.task_duration / task_context.estimated_duration,
            'user_satisfaction': result.user_satisfaction
        }

        self.performance_buffer.append(performance_entry)

        # Keep buffer size manageable
        if len(self.performance_buffer) > 1000:
            self.performance_buffer = self.performance_buffer[-1000:]

        # Log performance trends
        recent_success_rate = self._calculate_recent_success_rate()
        if recent_success_rate < self.config.min_success_rate:
            logger.warning(f"Success rate dropped to {recent_success_rate:.3f}")

    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate over recent episodes"""
        if len(self.performance_buffer) < 10:
            return 1.0  # Not enough data

        recent_entries = self.performance_buffer[-self.success_rate_window:]
        successes = sum(1 for entry in recent_entries if entry['success'])
        return successes / len(recent_entries)

    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        save_path = path or f"{self.model_path}/model"
        self.model.save(save_path)

        # Save training stats
        self._save_training_stats()

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None) -> bool:
        """Load trained model"""
        load_path = path or f"{self.model_path}/model"

        try:
            if not os.path.exists(f"{load_path}.zip"):
                logger.warning(f"Model file not found: {load_path}.zip")
                return False

            # Create dummy environment for loading
            dummy_env = DummyVecEnv([lambda: AgentSelectionEnvironment()])
            self.model = DQN.load(load_path, env=dummy_env)

            # Load training stats if available
            self._load_training_stats()

            logger.info(f"Model loaded from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _save_training_stats(self):
        """Save training statistics"""
        stats_path = f"{self.model_path}/training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)

    def _load_training_stats(self):
        """Load training statistics"""
        stats_path = f"{self.model_path}/training_stats.json"
        try:
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training stats: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        info = {
            'model_loaded': self.model is not None,
            'config': asdict(self.config),
            'training_stats': self.training_stats,
            'recent_performance': {
                'buffer_size': len(self.performance_buffer),
                'recent_success_rate': self._calculate_recent_success_rate() if self.performance_buffer else 0.0
            }
        }

        if self.model is not None:
            info['model_details'] = {
                'policy_type': str(type(self.model.policy)),
                'device': str(self.model.device),
                'num_timesteps': self.model.num_timesteps
            }

        return info

class RLAgentSelector:
    """High-level interface for RL-based agent selection"""

    def __init__(self,
                 agent_pool: AgentPool,
                 config: Optional[DQNConfig] = None,
                 model_path: Optional[str] = None):

        self.agent_pool = agent_pool
        self.environment = AgentSelectionEnvironment(agent_pool=agent_pool)
        self.dqn = AgentCoordinationDQN(
            environment=self.environment,
            config=config,
            model_path=model_path
        )

        # Load existing model if available
        self.model_loaded = self.dqn.load_model()

        logger.info(f"RLAgentSelector initialized (model_loaded: {self.model_loaded})")

    async def select_agents(self,
                           task_context: TaskContext,
                           fallback_to_traditional: bool = True) -> Tuple[List[str], float]:
        """Select optimal agents for task using RL"""

        try:
            if self.model_loaded:
                selected_agents, confidence = self.dqn.predict_agent_selection(task_context)
                return selected_agents, confidence
            else:
                if fallback_to_traditional:
                    # Fall back to traditional selection
                    logger.warning("RL model not available, using traditional selection")
                    traditional_agents = self._traditional_agent_selection(task_context)
                    return traditional_agents, 0.5  # Medium confidence
                else:
                    raise ValueError("RL model not trained and fallback disabled")

        except Exception as e:
            logger.error(f"RL agent selection failed: {e}")
            if fallback_to_traditional:
                traditional_agents = self._traditional_agent_selection(task_context)
                return traditional_agents, 0.3  # Low confidence due to error
            else:
                raise

    def _traditional_agent_selection(self, task_context: TaskContext) -> List[str]:
        """Traditional agent selection as fallback"""
        # Use existing agent pool logic
        task_requirements = {
            "capabilities": self._extract_capabilities(task_context),
            "domains": [task_context.domain.name.lower()],
            "task_type": task_context.complexity.name.lower()
        }

        return self.agent_pool.get_optimal_agent_team(task_requirements)

    def _extract_capabilities(self, task_context: TaskContext) -> List[str]:
        """Extract required capabilities from task context"""
        capabilities = []

        domain_capabilities = {
            "neuroscience": ["brain_imaging", "neural_analysis"],
            "statistical_analysis": ["statistics", "data_analysis"],
            "grant_writing": ["writing", "proposal_development"],
            "hypothesis_generation": ["research_design", "hypothesis_formation"],
            "clinical_validation": ["clinical_assessment", "validation"],
            "literature_analysis": ["literature_review", "synthesis"]
        }

        domain_name = task_context.domain.name.lower()
        if domain_name in domain_capabilities:
            capabilities.extend(domain_capabilities[domain_name])

        return capabilities

    def train_model(self,
                   total_timesteps: int = 50000,
                   eval_freq: int = 1000) -> Dict[str, Any]:
        """Train the RL model"""

        # Update config with parameters
        self.dqn.config.total_timesteps = total_timesteps
        self.dqn.config.eval_freq = eval_freq

        # Train model
        results = self.dqn.train()

        if results['success']:
            self.model_loaded = True
            logger.info("Model training completed successfully")
        else:
            logger.error(f"Model training failed: {results.get('error', 'Unknown error')}")

        return results

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate current model performance"""
        if not self.model_loaded:
            raise ValueError("No model loaded for evaluation")

        return self.dqn.evaluate_model()

    def update_performance(self,
                          task_context: TaskContext,
                          selected_agents: List[str],
                          result: ActionResult):
        """Update model with real performance data"""
        self.dqn.update_performance(task_context, selected_agents, result)

# Training utilities
def train_agent_coordination_model(
    config: Optional[DQNConfig] = None,
    model_path: str = "models/agent_coordination_dqn",
    wandb_tracking: bool = False
) -> Dict[str, Any]:
    """Standalone function to train agent coordination model"""

    logger.info("Starting agent coordination model training...")

    # Create environment and DQN
    env = AgentSelectionEnvironment()
    dqn = AgentCoordinationDQN(env, config, model_path)

    # Train model
    results = dqn.train(wandb_tracking=wandb_tracking)

    if results['success']:
        # Evaluate final model
        eval_results = dqn.evaluate_model()
        results['evaluation'] = eval_results

        logger.info(f"Training completed successfully!")
        logger.info(f"Final performance - Mean reward: {eval_results['mean_reward']:.3f}")
        logger.info(f"Success rate: {eval_results['success_rate']:.3f}")

    return results

# Testing and validation
def validate_rl_system():
    """Validate the complete RL system"""
    logger.info("🧪 Validating RL Agent Coordination System...")

    try:
        # Test environment
        env = AgentSelectionEnvironment()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logger.info("✅ Environment validation passed")

        # Test DQN creation (without training)
        if SB3_AVAILABLE:
            dqn = AgentCoordinationDQN(env)
            logger.info("✅ DQN model creation passed")

        # Test task generation
        task = env.task_generator.generate_task()
        logger.info(f"✅ Task generation passed: {task.description[:50]}...")

        return True

    except Exception as e:
        logger.error(f"❌ RL system validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test the RL system
    logging.basicConfig(level=logging.INFO)

    print("🤖 Testing RL Agent Coordination System...")

    success = validate_rl_system()
    if success:
        print("✅ RL system validation passed!")

        # Optionally run quick training test
        if SB3_AVAILABLE:
            print("\n🏃 Running quick training test...")
            results = train_agent_coordination_model(
                config=DQNConfig(total_timesteps=1000),  # Quick test
                model_path="models/test_model"
            )
            if results['success']:
                print("✅ Training test completed!")
            else:
                print("❌ Training test failed!")
    else:
        print("❌ RL system validation failed!")
"""
RL Ensemble System with Dynamic Weight Adjustment and Meta-Learning.

This module implements an ensemble of RL agents with Thompson sampling for
exploration-exploitation balance and meta-learning for ensemble optimization.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim

from .base_models import BaseRLAgent


class ThompsonSampler:
    """Thompson sampling for exploration-exploitation balance in ensemble weights"""
    
    def __init__(self, n_agents: int, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Initialize Thompson sampler
        
        Args:
            n_agents: Number of agents in ensemble
            alpha_prior: Prior alpha parameter for Beta distribution
            beta_prior: Prior beta parameter for Beta distribution
        """
        self.n_agents = n_agents
        self.alpha = np.full(n_agents, alpha_prior)
        self.beta = np.full(n_agents, beta_prior)
        self.rewards_history: List[List[float]] = [[] for _ in range(n_agents)]
        
    def sample_weights(self) -> np.ndarray:
        """Sample ensemble weights using Thompson sampling
        
        Returns:
            Normalized weights for each agent
        """
        # Sample from Beta distributions
        samples = np.array([
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_agents)
        ])
        
        # Normalize to sum to 1
        weights = samples / np.sum(samples)
        return weights
    
    def update(self, agent_idx: int, reward: float, baseline_reward: float = 0.0) -> None:
        """Update Thompson sampler with reward feedback
        
        Args:
            agent_idx: Index of the agent that received reward
            reward: Reward received
            baseline_reward: Baseline reward for comparison
            
        Raises:
            ValueError: If agent_idx is out of bounds or reward is invalid
        """
        # Input validation
        if not 0 <= agent_idx < self.n_agents:
            raise ValueError(f"agent_idx {agent_idx} out of bounds [0, {self.n_agents})")
        
        if not np.isfinite(reward):
            raise ValueError(f"Invalid reward value: {reward}")
        
        if not np.isfinite(baseline_reward):
            raise ValueError(f"Invalid baseline_reward value: {baseline_reward}")
        
        # Convert reward to success/failure for Beta update
        success = reward > baseline_reward
        
        if success:
            self.alpha[agent_idx] += 1
        else:
            self.beta[agent_idx] += 1
        
        # Store reward history
        self.rewards_history[agent_idx].append(reward)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Thompson sampler statistics
        
        Returns:
            Dictionary with sampler statistics
        """
        return {
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'mean_rewards': [
                np.mean(rewards) if rewards else 0.0
                for rewards in self.rewards_history
            ],
            'reward_counts': [len(rewards) for rewards in self.rewards_history]
        }


class MetaLearner(nn.Module):
    """Meta-learning network for ensemble weight optimization"""
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3
    ):
        """Initialize meta-learner
        
        Args:
            n_agents: Number of agents in ensemble
            state_dim: Dimension of state/context features
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for meta-learner
        """
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        
        # Network architecture (state_dim + 2 features per agent)
        input_dim = state_dim + n_agents * 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, state: torch.Tensor, agent_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through meta-learner
        
        Args:
            state: Current state/context features
            agent_features: Features describing agent performance
            
        Returns:
            Ensemble weights
        """
        # Concatenate state and agent features
        input_features = torch.cat([state, agent_features], dim=-1)
        weights = self.network(input_features)
        return weights
    
    def update(
        self,
        state: torch.Tensor,
        agent_features: torch.Tensor,
        target_weights: torch.Tensor,
        rewards: torch.Tensor
    ) -> float:
        """Update meta-learner with experience
        
        Args:
            state: State features
            agent_features: Agent performance features
            target_weights: Target ensemble weights
            rewards: Rewards received
            
        Returns:
            Training loss
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_weights = self.forward(state, agent_features)
        
        # Compute loss (weighted by rewards)
        loss = self.loss_fn(predicted_weights, target_weights)
        reward_weight = torch.mean(rewards)
        weighted_loss = loss * reward_weight
        
        # Backward pass
        weighted_loss.backward()
        self.optimizer.step()
        
        return loss.item()


class EnsembleManager:
    """Manages ensemble of RL agents with dynamic weight adjustment"""
    
    def __init__(
        self,
        agents: List[BaseRLAgent],
        use_thompson_sampling: bool = True,
        use_meta_learning: bool = True,
        state_dim: Optional[int] = None,
        meta_learning_rate: float = 1e-3,
        weight_update_frequency: int = 100,
        performance_window: int = 1000
    ):
        """Initialize ensemble manager
        
        Args:
            agents: List of RL agents
            use_thompson_sampling: Whether to use Thompson sampling
            use_meta_learning: Whether to use meta-learning
            state_dim: State dimension for meta-learning
            meta_learning_rate: Learning rate for meta-learner
            weight_update_frequency: How often to update weights
            performance_window: Window size for performance tracking
        """
        self.agents = agents
        self.n_agents = len(agents)
        self.use_thompson_sampling = use_thompson_sampling
        self.use_meta_learning = use_meta_learning
        self.weight_update_frequency = weight_update_frequency
        self.performance_window = performance_window
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_agents) / self.n_agents
        
        # Thompson sampler
        if use_thompson_sampling:
            self.thompson_sampler = ThompsonSampler(self.n_agents)
        
        # Meta-learner
        if use_meta_learning and state_dim is not None:
            self.meta_learner = MetaLearner(
                n_agents=self.n_agents,
                state_dim=state_dim,
                learning_rate=meta_learning_rate
            )
        else:
            self.meta_learner = None
        
        # Performance tracking
        self.agent_rewards: List[List[float]] = [[] for _ in range(self.n_agents)]
        self.ensemble_rewards: List[float] = []
        self.weight_history: List[np.ndarray] = []
        self.step_count = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """Make ensemble prediction
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policies
            return_individual: Whether to return individual agent predictions
            
        Returns:
            Ensemble action or tuple of (ensemble_action, individual_actions)
        """
        # Get predictions from all agents
        individual_actions = []
        individual_probs = []
        
        for agent in self.agents:
            if not agent.is_trained:
                # Use random action for untrained agents
                if hasattr(agent, 'env') and hasattr(agent.env, 'action_space'):
                    action = agent.env.action_space.sample()
                else:
                    action = np.random.randn(observation.shape[0])  # Assume continuous
                prob = None
            else:
                action, prob = agent.predict(observation, deterministic=deterministic)
            
            individual_actions.append(action)
            individual_probs.append(prob)
        
        # Compute ensemble action using weighted average
        individual_actions = np.array(individual_actions)
        ensemble_action = np.average(individual_actions, weights=self.weights, axis=0)
        
        if return_individual:
            return ensemble_action, individual_actions
        return ensemble_action
    
    def update_weights(
        self,
        rewards: List[float],
        state_features: Optional[np.ndarray] = None
    ) -> None:
        """Update ensemble weights based on performance
        
        Args:
            rewards: Rewards for each agent
            state_features: State features for meta-learning
        """
        # Update Thompson sampler
        if self.use_thompson_sampling:
            baseline_reward = np.mean(rewards)
            for i, reward in enumerate(rewards):
                self.thompson_sampler.update(i, reward, baseline_reward)
            
            # Sample new weights
            thompson_weights = self.thompson_sampler.sample_weights()
        else:
            thompson_weights = self.weights
        
        # Update meta-learner
        if self.use_meta_learning and self.meta_learner is not None and state_features is not None:
            # Prepare agent features (recent performance)
            agent_features = []
            for i in range(self.n_agents):
                recent_rewards = self.agent_rewards[i][-self.performance_window:]
                if recent_rewards:
                    mean_reward = np.mean(recent_rewards)
                    std_reward = np.std(recent_rewards)
                    sharpe_ratio = mean_reward / (std_reward + 1e-8)
                else:
                    mean_reward = 0.0
                    sharpe_ratio = 0.0
                
                agent_features.extend([mean_reward, sharpe_ratio])
            
            # Convert to tensors
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
            agent_features_tensor = torch.FloatTensor(agent_features).unsqueeze(0)
            target_weights_tensor = torch.FloatTensor(thompson_weights).unsqueeze(0)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(0)
            
            # Update meta-learner
            self.meta_learner.update(
                state_tensor,
                agent_features_tensor,
                target_weights_tensor,
                rewards_tensor
            )
            
            # Get meta-learned weights
            with torch.no_grad():
                meta_weights = self.meta_learner(state_tensor, agent_features_tensor)
                meta_weights = meta_weights.squeeze(0).numpy()
        else:
            meta_weights = thompson_weights
        
        # Combine Thompson sampling and meta-learning weights
        if (self.use_thompson_sampling and self.use_meta_learning and 
            self.meta_learner is not None):
            # Weighted combination
            alpha = 0.7  # Weight for meta-learning
            self.weights = alpha * meta_weights + (1 - alpha) * thompson_weights
        elif self.use_meta_learning and self.meta_learner is not None:
            self.weights = meta_weights
        elif self.use_thompson_sampling:
            self.weights = thompson_weights
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        # Store weight history
        self.weight_history.append(self.weights.copy())
        
        self.logger.debug("Updated ensemble weights: %s", self.weights)
    
    def train_ensemble(
        self,
        env,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5
    ) -> Dict[str, Any]:
        """Train all agents in the ensemble
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps per agent
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            
        Returns:
            Training results for all agents
        """
        training_results = {}
        
        for i, agent in enumerate(self.agents):
            self.logger.info("Training agent %d/%d", i+1, self.n_agents)
            
            # Train individual agent
            agent_results = agent.train(
                env=env,
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes
            )
            
            training_results[f'agent_{i}'] = agent_results
        
        return training_results
    
    def evaluate_ensemble(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Evaluate ensemble performance
        
        Args:
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policies
            
        Returns:
            Evaluation results
        """
        episode_rewards = []
        individual_rewards = [[] for _ in range(self.n_agents)]
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_individual_rewards = [0] * self.n_agents
            done = False
            truncated = False
            
            while not (done or truncated):
                # Get ensemble and individual actions
                ensemble_action, individual_actions = self.predict(
                    obs, deterministic=deterministic, return_individual=True
                )
                
                # Step with ensemble action
                obs, reward, done, truncated, _ = env.step(ensemble_action)
                episode_reward += reward
                
                # Simulate individual agent rewards (approximate)
                for i in range(self.n_agents):
                    # Weight individual reward by how close their action was to ensemble
                    action_similarity = 1.0 - np.linalg.norm(
                        individual_actions[i] - ensemble_action
                    ) / (np.linalg.norm(ensemble_action) + 1e-8)
                    episode_individual_rewards[i] += reward * action_similarity
            
            episode_rewards.append(episode_reward)
            for i in range(self.n_agents):
                individual_rewards[i].append(episode_individual_rewards[i])
        
        # Calculate statistics
        results = {
            'ensemble': {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards)
            },
            'individual_agents': {},
            'weights': self.weights.tolist(),
            'n_episodes': n_episodes
        }
        
        for i in range(self.n_agents):
            results['individual_agents'][f'agent_{i}'] = {
                'mean_reward': np.mean(individual_rewards[i]),
                'std_reward': np.std(individual_rewards[i]),
                'weight': self.weights[i]
            }
        
        return results
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics
        
        Returns:
            Dictionary with ensemble statistics
        """
        stats = {
            'n_agents': self.n_agents,
            'current_weights': self.weights.tolist(),
            'weight_history': self.weight_history,
            'step_count': self.step_count,
            'performance_window': self.performance_window
        }
        
        # Thompson sampler statistics
        if self.use_thompson_sampling:
            stats['thompson_sampling'] = self.thompson_sampler.get_statistics()
        
        # Agent performance statistics
        agent_stats = {}
        for i in range(self.n_agents):
            rewards = self.agent_rewards[i]
            if rewards:
                agent_stats[f'agent_{i}'] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'total_episodes': len(rewards),
                    'recent_performance': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                }
            else:
                agent_stats[f'agent_{i}'] = {
                    'mean_reward': 0.0,
                    'std_reward': 0.0,
                    'total_episodes': 0,
                    'recent_performance': 0.0
                }
        
        stats['agent_performance'] = agent_stats
        
        return stats
    
    def save_ensemble(self, filepath: str) -> None:
        """Save ensemble configuration and weights
        
        Args:
            filepath: Path to save ensemble
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ensemble metadata
        ensemble_data = {
            'n_agents': self.n_agents,
            'weights': self.weights.tolist(),
            'weight_history': [w.tolist() for w in self.weight_history],
            'use_thompson_sampling': self.use_thompson_sampling,
            'use_meta_learning': self.use_meta_learning,
            'step_count': self.step_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add Thompson sampler data
        if self.use_thompson_sampling:
            ensemble_data['thompson_sampler'] = self.thompson_sampler.get_statistics()
        
        # Save ensemble data
        with open(f"{filepath}_ensemble.json", 'w', encoding='utf-8') as f:
            json.dump(ensemble_data, f, indent=2)
        
        # Save individual agents
        for i, agent in enumerate(self.agents):
            agent_path = f"{filepath}_agent_{i}"
            agent.save_model(agent_path)
        
        # Save meta-learner if available
        if self.meta_learner is not None:
            torch.save(self.meta_learner.state_dict(), f"{filepath}_meta_learner.pth")
        
        self.logger.info("Ensemble saved to %s", filepath)
    
    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble configuration and weights
        
        Args:
            filepath: Path to load ensemble from
        """
        # Load ensemble metadata
        with open(f"{filepath}_ensemble.json", 'r', encoding='utf-8') as f:
            ensemble_data = json.load(f)
        
        # Restore ensemble state
        self.weights = np.array(ensemble_data['weights'])
        self.weight_history = [np.array(w) for w in ensemble_data['weight_history']]
        self.step_count = ensemble_data['step_count']
        
        # Restore Thompson sampler
        if self.use_thompson_sampling and 'thompson_sampler' in ensemble_data:
            ts_data = ensemble_data['thompson_sampler']
            self.thompson_sampler.alpha = np.array(ts_data['alpha'])
            self.thompson_sampler.beta = np.array(ts_data['beta'])
            self.thompson_sampler.rewards_history = [
                [] for _ in range(self.n_agents)
            ]  # Reset history
        
        # Load individual agents
        for i, agent in enumerate(self.agents):
            agent_path = f"{filepath}_agent_{i}"
            if os.path.exists(f"{agent_path}.zip"):
                agent.load_model(agent_path)
        
        # Load meta-learner if available
        meta_learner_path = f"{filepath}_meta_learner.pth"
        if self.meta_learner is not None and os.path.exists(meta_learner_path):
            self.meta_learner.load_state_dict(torch.load(meta_learner_path))
        
        self.logger.info("Ensemble loaded from %s", filepath)


class EnsembleFactory:
    """Factory for creating RL ensembles"""
    
    @staticmethod
    def create_diverse_ensemble(
        env,
        agent_configs: List[Dict[str, Any]],
        use_thompson_sampling: bool = True,
        use_meta_learning: bool = True,
        state_dim: Optional[int] = None
    ) -> EnsembleManager:
        """Create ensemble with diverse agent configurations
        
        Args:
            env: Training environment
            agent_configs: List of agent configurations
            use_thompson_sampling: Whether to use Thompson sampling
            use_meta_learning: Whether to use meta-learning
            state_dim: State dimension for meta-learning
            
        Returns:
            Configured ensemble manager
        """
        from .rl_agents import RLAgentFactory
        
        agents = []
        for config in agent_configs:
            agent = RLAgentFactory.create_agent(env=env, **config)
            agents.append(agent)
        
        return EnsembleManager(
            agents=agents,
            use_thompson_sampling=use_thompson_sampling,
            use_meta_learning=use_meta_learning,
            state_dim=state_dim
        )
    
    @staticmethod
    def create_standard_ensemble(
        env,
        state_dim: Optional[int] = None,
        use_thompson_sampling: bool = True,
        use_meta_learning: bool = True
    ) -> EnsembleManager:
        """Create standard ensemble with PPO, SAC, TD3, and DQN agents
        
        Args:
            env: Training environment
            state_dim: State dimension for meta-learning
            use_thompson_sampling: Whether to use Thompson sampling
            use_meta_learning: Whether to use meta-learning
            
        Returns:
            Standard ensemble manager
        """
        from .rl_agents import RLAgentFactory
        
        # Create diverse agents with different configurations
        agents = []
        
        # PPO agent
        ppo_agent = RLAgentFactory.create_ppo_agent(
            env=env,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048
        )
        agents.append(ppo_agent)
        
        # SAC agent
        sac_agent = RLAgentFactory.create_sac_agent(
            env=env,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256
        )
        agents.append(sac_agent)
        
        # TD3 agent
        td3_agent = RLAgentFactory.create_td3_agent(
            env=env,
            learning_rate=1e-3,
            buffer_size=1000000,
            batch_size=100
        )
        agents.append(td3_agent)
        
        # Only add DQN if action space is discrete
        import gymnasium as gym
        if isinstance(env.action_space, gym.spaces.Discrete):
            dqn_agent = RLAgentFactory.create_dqn_agent(
                env=env,
                learning_rate=1e-4,
                buffer_size=1000000,
                batch_size=32
            )
            agents.append(dqn_agent)
        
        return EnsembleManager(
            agents=agents,
            use_thompson_sampling=use_thompson_sampling,
            use_meta_learning=use_meta_learning,
            state_dim=state_dim
        )
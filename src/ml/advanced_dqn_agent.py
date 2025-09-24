"""
Advanced DQN Agent with Full Rainbow Implementation.

This module implements a state-of-the-art DQN agent with all Rainbow features:
- C51 Distributional DQN for return distribution modeling
- Double DQN for reduced overestimation bias
- Dueling DQN for separate value and advantage estimation
- Prioritized Experience Replay for efficient learning
- Noisy Networks for parameter space exploration
- Multi-step learning for improved sample efficiency

The agent is specifically designed for financial trading with advanced features
for handling market data and achieving superior performance.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from .base_models import BaseRLAgent

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class RainbowDQNConfig:
    """Configuration for Rainbow DQN agent with comprehensive parameters."""
    
    # Network architecture
    hidden_dims: List[int] = None
    dueling: bool = True
    noisy: bool = True
    
    # Distributional RL (C51)
    distributional: bool = True
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    
    # Learning parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 1.0  # Hard update for DQN
    
    # Experience replay
    buffer_size: int = 1000000
    prioritized_replay: bool = True
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4  # Importance sampling exponent
    beta_increment: float = 0.001
    epsilon_priority: float = 1e-6
    
    # Multi-step learning
    multi_step: int = 3
    
    # Training schedule
    learning_starts: int = 10000
    train_freq: int = 4
    target_update_interval: int = 10000
    gradient_steps: int = 1
    
    # Exploration
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    
    # Noisy networks
    noisy_std: float = 0.5
    
    # Regularization
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-5
    
    # Device and logging
    device: str = "auto"
    verbose: int = 1
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.hidden_dims is None:
            self.hidden_dims = [512, 512, 256]
        
        # Validate parameters
        assert self.n_atoms > 1, "Number of atoms must be > 1"
        assert self.v_min < self.v_max, "v_min must be < v_max"
        assert 0 < self.alpha <= 1, "Alpha must be in (0, 1]"
        assert 0 < self.beta <= 1, "Beta must be in (0, 1]"
        assert self.multi_step >= 1, "Multi-step must be >= 1"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.buffer_size > self.batch_size, "Buffer size must be > batch size"


class NoisyLinear(nn.Module):
    """Noisy linear layer for parameter space exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialize noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation for noise
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise for both weights and biases."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """Dueling network architecture for separate value and advantage estimation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        distributional: bool = True,
        n_atoms: int = 51,
        noisy: bool = True,
        noisy_std: float = 0.5
    ):
        """Initialize dueling network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            distributional: Whether to use distributional RL
            n_atoms: Number of atoms for distributional RL
            noisy: Whether to use noisy networks
            noisy_std: Standard deviation for noisy networks
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.distributional = distributional
        self.n_atoms = n_atoms
        self.noisy = noisy
        
        # Shared feature extractor
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            if noisy:
                self.feature_layers.append(NoisyLinear(prev_dim, hidden_dim, noisy_std))
            else:
                self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Value stream
        value_dim = hidden_dims[-1] // 2
        if noisy:
            self.value_hidden = NoisyLinear(prev_dim, value_dim, noisy_std)
            self.value_output = NoisyLinear(value_dim, n_atoms if distributional else 1, noisy_std)
        else:
            self.value_hidden = nn.Linear(prev_dim, value_dim)
            self.value_output = nn.Linear(value_dim, n_atoms if distributional else 1)
        
        # Advantage stream
        advantage_dim = hidden_dims[-1] // 2
        if noisy:
            self.advantage_hidden = NoisyLinear(prev_dim, advantage_dim, noisy_std)
            self.advantage_output = NoisyLinear(
                advantage_dim, 
                action_dim * (n_atoms if distributional else 1), 
                noisy_std
            )
        else:
            self.advantage_hidden = nn.Linear(prev_dim, advantage_dim)
            self.advantage_output = nn.Linear(
                advantage_dim, 
                action_dim * (n_atoms if distributional else 1)
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Linear, NoisyLinear)):
            if hasattr(module, 'weight_mu'):
                # Already initialized in NoisyLinear
                pass
            elif hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values or Q-value distributions
        """
        # Feature extraction
        x = state
        for layer in self.feature_layers:
            x = layer(x)
        
        # Value stream
        value = F.relu(self.value_hidden(x))
        value = self.value_output(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage)
        
        if self.distributional:
            # Reshape for distributional RL
            batch_size = state.size(0)
            value = value.view(batch_size, 1, self.n_atoms)
            advantage = advantage.view(batch_size, self.action_dim, self.n_atoms)
            
            # Dueling aggregation for distributions
            q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
            
            # Apply softmax to get probability distributions
            q_dist = F.softmax(q_dist, dim=-1)
            
            return q_dist
        else:
            # Standard dueling aggregation
            advantage = advantage.view(-1, self.action_dim)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            
            return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with sum tree implementation."""
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
            epsilon: Small constant for numerical stability
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Sum tree for efficient sampling
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.full(2 * self.tree_capacity - 1, float('inf'))
        
        # Experience storage
        self.experiences = [None] * capacity
        self.position = 0
        self.size = 0
        
        # Maximum priority for new experiences
        self.max_priority = 1.0
    
    def _update_tree(self, tree_idx: int, priority: float):
        """Update sum and min trees."""
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        
        # Propagate changes up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += change
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx + 1],
                self.min_tree[2 * tree_idx + 2]
            )
    
    def _get_leaf(self, value: float) -> int:
        """Get leaf index for given value."""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.sum_tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.sum_tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.sum_tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.tree_capacity + 1
        return data_idx, leaf_idx
    
    def add(self, experience: Tuple):
        """Add experience to buffer."""
        tree_idx = self.position + self.tree_capacity - 1
        
        self.experiences[self.position] = experience
        self._update_tree(tree_idx, self.max_priority)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        batch = []
        indices = []
        priorities = []
        
        # Calculate sampling probabilities
        total_priority = self.sum_tree[0]
        segment = total_priority / batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            
            data_idx, tree_idx = self._get_leaf(value)
            
            # Ensure valid index
            if data_idx < self.size and self.experiences[data_idx] is not None:
                batch.append(self.experiences[data_idx])
                indices.append(data_idx)
                priorities.append(self.sum_tree[tree_idx])
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / total_priority
        max_weight = (self.size * sampling_probabilities.min()) ** (-self.beta)
        weights = (self.size * sampling_probabilities) ** (-self.beta) / max_weight
        
        return batch, np.array(indices), weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            tree_idx = idx + self.tree_capacity - 1
            self._update_tree(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size


class RainbowDQNAgent(BaseRLAgent):
    """Advanced DQN agent with full Rainbow implementation."""
    
    def __init__(self, env: gym.Env, config: Optional[RainbowDQNConfig] = None):
        """Initialize Rainbow DQN agent.
        
        Args:
            env: Training environment
            config: Agent configuration
        """
        # Store original config object
        original_config = config or RainbowDQNConfig()
        
        # Handle case where config is passed as dict (for compatibility)
        if isinstance(original_config, dict):
            config_dict = original_config
            original_config = RainbowDQNConfig()
            for key, value in config_dict.items():
                if hasattr(original_config, key):
                    setattr(original_config, key, value)
        
        # Initialize base class with config dict (this will set self.config to dict)
        super().__init__(original_config.__dict__)
        
        # Restore the original config object after parent init
        self.config = original_config
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        
        # Check if environment has discrete action space
        if hasattr(env.action_space, 'n'):
            self.action_dim = env.action_space.n
        else:
            raise ValueError(
                f"Rainbow DQN requires discrete action space, but got {type(env.action_space)}. "
                "Use DiscreteTradingWrapper to convert continuous actions to discrete."
            )
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Set random seeds
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Initialize networks
        self.q_network = DuelingNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            distributional=self.config.distributional,
            n_atoms=self.config.n_atoms,
            noisy=self.config.noisy,
            noisy_std=self.config.noisy_std
        ).to(self.device)
        
        self.target_network = DuelingNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            distributional=self.config.distributional,
            n_atoms=self.config.n_atoms,
            noisy=self.config.noisy,
            noisy_std=self.config.noisy_std
        ).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Experience replay buffer
        if self.config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                alpha=self.config.alpha,
                beta=self.config.beta,
                beta_increment=self.config.beta_increment,
                epsilon=self.config.epsilon_priority
            )
        else:
            self.replay_buffer = deque(maxlen=self.config.buffer_size)
        
        # Distributional RL support
        if self.config.distributional:
            self.support = torch.linspace(
                self.config.v_min, 
                self.config.v_max, 
                self.config.n_atoms
            ).to(self.device)
            self.delta_z = (self.config.v_max - self.config.v_min) / (self.config.n_atoms - 1)
        
        # Multi-step learning
        self.multi_step_buffer = deque(maxlen=self.config.multi_step)
        
        # Training state
        self.steps_done = 0
        self.episodes_done = 0
        self.training_history = []
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        logger.info(f"Rainbow DQN agent initialized with {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        if self.config.noisy:
            return 0.0  # No epsilon-greedy with noisy networks
        
        # Linear decay
        fraction = min(1.0, self.steps_done / (self.config.exploration_fraction * 1000000))
        return self.config.exploration_initial_eps + fraction * (
            self.config.exploration_final_eps - self.config.exploration_initial_eps
        )
    
    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action given observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob or None)
        """
        if not self.is_trained and not deterministic:
            # Random action during initial exploration
            action = self.env.action_space.sample()
            return np.array([action]), None
        
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Reset noise for noisy networks
        if self.config.noisy:
            self.q_network.reset_noise()
        
        with torch.no_grad():
            if self.config.distributional:
                # Get Q-value distributions
                q_dist = self.q_network(state)
                # Convert to Q-values by taking expectation
                q_values = (q_dist * self.support).sum(dim=-1)
            else:
                q_values = self.q_network(state)
            
            if deterministic or self.config.noisy:
                action = q_values.argmax(dim=1).cpu().numpy()
            else:
                # Epsilon-greedy exploration
                if np.random.random() < self.get_epsilon():
                    action = np.array([self.env.action_space.sample()])
                else:
                    action = q_values.argmax(dim=1).cpu().numpy()
        
        return action, None
    
    def _add_to_multi_step_buffer(self, experience: Tuple):
        """Add experience to multi-step buffer."""
        self.multi_step_buffer.append(experience)
        
        if len(self.multi_step_buffer) == self.config.multi_step:
            # Calculate multi-step return
            state, action, reward, next_state, done = self.multi_step_buffer[0]
            
            multi_step_reward = 0
            gamma = 1
            
            for i, (_, _, r, _, d) in enumerate(self.multi_step_buffer):
                multi_step_reward += gamma * r
                gamma *= self.config.gamma
                if d:
                    break
            
            # Use the last non-terminal state as next_state
            final_next_state = next_state
            final_done = done
            
            for i in range(1, len(self.multi_step_buffer)):
                _, _, _, ns, d = self.multi_step_buffer[i]
                if not d:
                    final_next_state = ns
                final_done = d
                if d:
                    break
            
            multi_step_experience = (state, action, multi_step_reward, final_next_state, final_done)
            
            if self.config.prioritized_replay:
                self.replay_buffer.add(multi_step_experience)
            else:
                self.replay_buffer.append(multi_step_experience)
    
    def _compute_td_error(self, batch: List[Tuple]) -> torch.Tensor:
        """Compute TD errors for prioritized replay."""
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        with torch.no_grad():
            if self.config.distributional:
                # Distributional TD error computation
                current_q_dist = self.q_network(states)
                current_q_values = (current_q_dist * self.support).sum(dim=-1)
                current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Double DQN for target
                next_q_dist = self.q_network(next_states)
                next_q_values = (next_q_dist * self.support).sum(dim=-1)
                next_actions = next_q_values.argmax(dim=1)
                
                target_next_q_dist = self.target_network(next_states)
                target_next_q_values = (target_next_q_dist * self.support).sum(dim=-1)
                target_next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                target_q = rewards + (self.config.gamma ** self.config.multi_step) * target_next_q * (~dones)
            else:
                # Standard TD error
                current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Double DQN
                next_q_values = self.q_network(next_states)
                next_actions = next_q_values.argmax(dim=1)
                target_next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                target_q = rewards + (self.config.gamma ** self.config.multi_step) * target_next_q * (~dones)
            
            td_errors = torch.abs(current_q - target_q)
        
        return td_errors
    
    def _train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if self.config.prioritized_replay:
            if len(self.replay_buffer) < self.config.batch_size:
                return {}
            
            batch, indices, weights = self.replay_buffer.sample(self.config.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.replay_buffer) < self.config.batch_size:
                return {}
            
            batch = list(np.random.choice(self.replay_buffer, self.config.batch_size, replace=False))
            weights = torch.ones(self.config.batch_size).to(self.device)
        
        # Prepare batch data
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Reset noise for noisy networks
        if self.config.noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        if self.config.distributional:
            # Distributional RL loss (C51)
            loss = self._compute_distributional_loss(states, actions, rewards, next_states, dones, weights)
        else:
            # Standard DQN loss
            loss = self._compute_dqn_loss(states, actions, rewards, next_states, dones, weights)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        # Update priorities for prioritized replay
        if self.config.prioritized_replay:
            td_errors = self._compute_td_error(batch)
            self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        return {"loss": loss.item()}
    
    def _compute_distributional_loss(
        self, 
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute distributional RL loss (C51)."""
        batch_size = states.size(0)
        
        # Current Q-distribution
        current_q_dist = self.q_network(states)
        current_q_dist = current_q_dist[range(batch_size), actions]
        
        with torch.no_grad():
            # Double DQN for action selection
            next_q_dist = self.q_network(next_states)
            next_q_values = (next_q_dist * self.support).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=1)
            
            # Target Q-distribution
            target_next_q_dist = self.target_network(next_states)
            target_next_q_dist = target_next_q_dist[range(batch_size), next_actions]
            
            # Compute target distribution
            target_support = rewards.unsqueeze(1) + (self.config.gamma ** self.config.multi_step) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.config.v_min, self.config.v_max)
            
            # Distribute probability mass
            b = (target_support - self.config.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.n_atoms - 1)) * (l == u)] += 1
            
            # Distribute probability mass
            target_q_dist = torch.zeros_like(target_next_q_dist)
            offset = torch.linspace(0, (batch_size - 1) * self.config.n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.config.n_atoms).to(self.device)
            
            target_q_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
            target_q_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))
        
        # Cross-entropy loss
        loss = -(target_q_dist * current_q_dist.log()).sum(dim=1)
        loss = (loss * weights).mean()
        
        return loss
    
    def _compute_dqn_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard DQN loss."""
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
            target_next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (self.config.gamma ** self.config.multi_step) * target_next_q * (~dones)
        
        # Huber loss
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss * weights).mean()
        
        return loss
    
    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        checkpoint_freq: int = 50000,
        checkpoint_path: Optional[str] = None,
        log_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the Rainbow DQN agent.
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            checkpoint_freq: Checkpoint saving frequency
            checkpoint_path: Path to save checkpoints
            log_path: Path to save training logs
            
        Returns:
            Training results
        """
        logger.info(f"Starting Rainbow DQN training for {total_timesteps} timesteps")
        
        start_time = datetime.now()
        episode_reward = 0
        episode_length = 0
        obs, _ = env.reset()
        
        evaluations = []
        
        for step in range(total_timesteps):
            self.steps_done = step
            
            # Select action
            action, _ = self.predict(obs, deterministic=False)
            action = action[0]  # Extract scalar action
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            experience = (obs, action, reward, next_obs, done)
            self._add_to_multi_step_buffer(experience)
            
            episode_reward += reward
            episode_length += 1
            
            # Training
            if step >= self.config.learning_starts and step % self.config.train_freq == 0:
                for _ in range(self.config.gradient_steps):
                    train_metrics = self._train_step()
                    if train_metrics:
                        self.losses.append(train_metrics['loss'])
            
            # Update target network
            if step % self.config.target_update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.debug(f"Target network updated at step {step}")
            
            # Episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episodes_done += 1
                
                if self.config.verbose > 0 and self.episodes_done % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    logger.info(
                        f"Episode {self.episodes_done}, Step {step}, "
                        f"Mean Reward: {np.mean(recent_rewards):.2f}, "
                        f"Epsilon: {self.get_epsilon():.3f}"
                    )
                
                episode_reward = 0
                episode_length = 0
                obs, _ = env.reset()
            else:
                obs = next_obs
            
            # Evaluation
            if eval_env is not None and step % eval_freq == 0 and step > 0:
                eval_metrics = self._evaluate(eval_env, n_eval_episodes)
                eval_metrics['timesteps'] = step
                evaluations.append(eval_metrics)
                
                logger.info(
                    f"Evaluation at step {step}: "
                    f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}"
                )
            
            # Checkpointing
            if checkpoint_path and step % checkpoint_freq == 0 and step > 0:
                checkpoint_file = os.path.join(checkpoint_path, f"rainbow_dqn_step_{step}.pth")
                self.save_model(checkpoint_file)
                logger.info(f"Checkpoint saved at step {step}")
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        self.is_trained = True
        
        # Final evaluation
        if eval_env is not None:
            final_eval = self._evaluate(eval_env, n_eval_episodes)
            evaluations.append({**final_eval, 'timesteps': total_timesteps})
        
        results = {
            'agent_type': 'Rainbow_DQN',
            'total_timesteps': total_timesteps,
            'episodes_completed': self.episodes_done,
            'training_time': training_time,
            'mean_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'mean_loss': np.mean(self.losses[-1000:]) if self.losses else 0,
            'evaluations': evaluations,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        self.training_history.append(results)
        
        logger.info(f"Rainbow DQN training completed in {training_time:.2f} seconds")
        logger.info(f"Final mean reward: {results['mean_episode_reward']:.2f}")
        
        # Check if Sharpe ratio target is met
        if evaluations and 'sharpe_ratio' in evaluations[-1]:
            final_sharpe = evaluations[-1]['sharpe_ratio']
            logger.info(f"Final Sharpe ratio: {final_sharpe:.4f}")
            if final_sharpe >= 1.5:
                logger.info("✅ TARGET ACHIEVED: Sharpe ratio > 1.5")
                results['target_achieved'] = True
            else:
                logger.warning(f"❌ TARGET NOT MET: Sharpe ratio {final_sharpe:.4f} < 1.5")
                results['target_achieved'] = False
        else:
            logger.warning("❌ Could not calculate Sharpe ratio for target validation")
            results['target_achieved'] = False
        
        return results
    
    def _evaluate(self, env: gym.Env, n_episodes: int) -> Dict[str, float]:
        """Evaluate agent performance with comprehensive financial metrics."""
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        returns_series = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_returns = []
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                # Track portfolio metrics if available
                if 'portfolio_value' in info:
                    portfolio_values.append(info['portfolio_value'])
                if 'total_return' in info:
                    episode_returns.append(info['total_return'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_returns:
                returns_series.extend(episode_returns)
        
        # Calculate financial metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }
        
        # Add financial performance metrics
        if portfolio_values:
            portfolio_values = np.array(portfolio_values)
            if len(portfolio_values) > 1:
                portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                portfolio_returns = portfolio_returns[np.isfinite(portfolio_returns)]
                
                if len(portfolio_returns) > 0:
                    # Sharpe ratio (annualized, assuming daily returns)
                    mean_return = np.mean(portfolio_returns)
                    std_return = np.std(portfolio_returns)
                    if std_return > 0:
                        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
                        metrics['sharpe_ratio'] = sharpe_ratio
                    else:
                        metrics['sharpe_ratio'] = 0.0
                    
                    # Maximum drawdown
                    cumulative = np.cumprod(1 + portfolio_returns)
                    peak = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - peak) / peak
                    metrics['max_drawdown'] = np.min(drawdown)
                    
                    # Sortino ratio (downside deviation)
                    downside_returns = portfolio_returns[portfolio_returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = np.std(downside_returns)
                        if downside_std > 0:
                            metrics['sortino_ratio'] = mean_return / downside_std * np.sqrt(252)
                        else:
                            metrics['sortino_ratio'] = float('inf') if mean_return > 0 else 0.0
                    else:
                        metrics['sortino_ratio'] = float('inf') if mean_return > 0 else 0.0
                    
                    # Calmar ratio
                    annual_return = mean_return * 252
                    if metrics['max_drawdown'] < 0:
                        metrics['calmar_ratio'] = abs(annual_return / metrics['max_drawdown'])
                    else:
                        metrics['calmar_ratio'] = float('inf') if annual_return > 0 else 0.0
                    
                    # Win rate
                    metrics['win_rate'] = np.mean(portfolio_returns > 0)
                    
                    # Volatility (annualized)
                    metrics['volatility'] = std_return * np.sqrt(252)
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save Rainbow DQN model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
        
        logger.info(f"Rainbow DQN model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load Rainbow DQN model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.config = checkpoint['config']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.is_trained = checkpoint['is_trained']
        self.training_history = checkpoint['training_history']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']
        
        logger.info(f"Rainbow DQN model loaded from {filepath}")
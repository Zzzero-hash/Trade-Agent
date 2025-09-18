"""
RL Ensemble System Demo.

This example demonstrates the RL ensemble system with Thompson sampling
and meta-learning for dynamic weight adjustment.
"""

import numpy as np
import torch
import gymnasium as gym
from typing import List

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.rl_ensemble import (
    ThompsonSampler,
    MetaLearner,
    EnsembleManager,
    EnsembleFactory
)
from src.ml.base_models import BaseRLAgent


class DemoRLAgent(BaseRLAgent):
    """Demo RL agent with configurable performance characteristics"""
    
    def __init__(self, agent_id: int, performance_level: float = 0.5, volatility: float = 0.1):
        """Initialize demo agent
        
        Args:
            agent_id: Unique identifier for the agent
            performance_level: Base performance level (0.0 to 1.0)
            volatility: Performance volatility
        """
        super().__init__({'agent_id': agent_id})
        self.agent_id = agent_id
        self.performance_level = performance_level
        self.volatility = volatility
        self.is_trained = True
        self.episode_count = 0
    
    def train(self, env, total_timesteps: int):
        """Simulate training"""
        return {
            'agent_id': self.agent_id,
            'timesteps': total_timesteps,
            'final_reward': self.performance_level + np.random.normal(0, self.volatility)
        }
    
    def predict(self, observation, deterministic=True):
        """Predict action with performance-based noise"""
        # Base action with performance-dependent quality
        base_action = np.random.randn(*observation.shape)
        
        # Add performance-based bias
        performance_noise = np.random.normal(0, 1 - self.performance_level)
        action = base_action + performance_noise
        
        return action, None
    
    def simulate_episode_reward(self) -> float:
        """Simulate episode reward based on agent characteristics"""
        self.episode_count += 1
        
        # Performance can change over time (some agents improve, others degrade)
        time_factor = self.episode_count / 100.0
        
        if self.agent_id == 0:
            # Agent 0 starts poor but improves over time
            current_performance = self.performance_level + time_factor * 0.3
        elif self.agent_id == 1:
            # Agent 1 starts good but degrades slightly
            current_performance = self.performance_level - time_factor * 0.1
        else:
            # Other agents remain relatively stable
            current_performance = self.performance_level
        
        # Add noise
        reward = current_performance + np.random.normal(0, self.volatility)
        return reward
    
    def save_model(self, filepath: str):
        """Save model (demo implementation)"""
        import json
        with open(f"{filepath}.json", 'w') as f:
            json.dump({
                'agent_id': self.agent_id,
                'performance_level': self.performance_level,
                'volatility': self.volatility
            }, f)
    
    def load_model(self, filepath: str):
        """Load model (demo implementation)"""
        import json
        with open(f"{filepath}.json", 'r') as f:
            data = json.load(f)
            self.agent_id = data['agent_id']
            self.performance_level = data['performance_level']
            self.volatility = data['volatility']


def demo_thompson_sampling():
    """Demonstrate Thompson sampling for exploration-exploitation"""
    print("=== Thompson Sampling Demo ===")
    
    n_agents = 3
    sampler = ThompsonSampler(n_agents)
    
    print(f"Initial weights: {sampler.sample_weights()}")
    
    # Simulate episodes where agent 2 is consistently better
    for episode in range(50):
        # Sample weights
        weights = sampler.sample_weights()
        
        # Simulate rewards (agent 2 is best)
        rewards = [
            0.3 + np.random.normal(0, 0.2),  # Agent 0: poor
            0.5 + np.random.normal(0, 0.1),  # Agent 1: medium
            0.8 + np.random.normal(0, 0.1)   # Agent 2: good
        ]
        
        # Update sampler with rewards
        baseline = np.mean(rewards)
        for i, reward in enumerate(rewards):
            sampler.update(i, reward, baseline)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: weights = {weights}, rewards = {rewards}")
    
    # Final statistics
    stats = sampler.get_statistics()
    print(f"Final statistics: {stats}")
    print(f"Final weights: {sampler.sample_weights()}")
    print()


def demo_meta_learning():
    """Demonstrate meta-learning for ensemble weight optimization"""
    print("=== Meta-Learning Demo ===")
    
    n_agents = 3
    state_dim = 5
    meta_learner = MetaLearner(n_agents, state_dim, hidden_dim=32)
    
    print("Training meta-learner...")
    
    # Training data: state features and optimal weights
    for epoch in range(100):
        # Random state
        state = torch.randn(1, state_dim)
        
        # Agent features (performance metrics)
        agent_features = torch.randn(1, n_agents * 2)
        
        # Target weights (prefer agent with higher performance)
        target_weights = torch.softmax(torch.tensor([[0.2, 0.3, 0.8]]), dim=1)
        
        # Rewards
        rewards = torch.tensor([[0.2, 0.3, 0.8]])
        
        # Update meta-learner
        loss = meta_learner.update(state, agent_features, target_weights, rewards)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    # Test prediction
    test_state = torch.randn(1, state_dim)
    test_agent_features = torch.randn(1, n_agents * 2)
    
    with torch.no_grad():
        predicted_weights = meta_learner(test_state, test_agent_features)
        print(f"Predicted weights: {predicted_weights.numpy()}")
    
    print()


def demo_ensemble_manager():
    """Demonstrate complete ensemble manager functionality"""
    print("=== Ensemble Manager Demo ===")
    
    # Create diverse agents with different performance characteristics
    agents = [
        DemoRLAgent(agent_id=0, performance_level=0.3, volatility=0.2),  # Poor, improving
        DemoRLAgent(agent_id=1, performance_level=0.7, volatility=0.1),  # Good, stable
        DemoRLAgent(agent_id=2, performance_level=0.5, volatility=0.3),  # Medium, volatile
    ]
    
    # Create ensemble manager
    ensemble = EnsembleManager(
        agents=agents,
        use_thompson_sampling=True,
        use_meta_learning=True,
        state_dim=5,
        weight_update_frequency=10
    )
    
    print(f"Initial weights: {ensemble.weights}")
    
    # Simulate trading episodes
    observation = np.array([1.0, 0.5, -0.2, 0.8, -0.1])
    
    for episode in range(100):
        # Make ensemble prediction
        action = ensemble.predict(observation)
        
        # Simulate individual agent rewards
        rewards = [agent.simulate_episode_reward() for agent in agents]
        
        # Update ensemble weights periodically
        if episode % 10 == 0:
            state_features = np.random.randn(5)  # Market state features
            ensemble.update_weights(rewards, state_features)
            
            print(f"Episode {episode}:")
            print(f"  Rewards: {[f'{r:.3f}' for r in rewards]}")
            print(f"  Weights: {[f'{w:.3f}' for w in ensemble.weights]}")
            print(f"  Action: {action[:3]}...")  # Show first 3 elements
        
        # Store rewards for tracking
        for i, reward in enumerate(rewards):
            ensemble.agent_rewards[i].append(reward)
    
    # Final statistics
    stats = ensemble.get_ensemble_statistics()
    print("\nFinal Ensemble Statistics:")
    for agent_key, agent_stats in stats['agent_performance'].items():
        print(f"  {agent_key}: mean_reward = {agent_stats['mean_reward']:.3f}, "
              f"weight = {ensemble.weights[int(agent_key.split('_')[1])]:.3f}")
    
    print()


def demo_ensemble_evaluation():
    """Demonstrate ensemble evaluation"""
    print("=== Ensemble Evaluation Demo ===")
    
    # Create agents
    agents = [
        DemoRLAgent(agent_id=0, performance_level=0.4, volatility=0.1),
        DemoRLAgent(agent_id=1, performance_level=0.8, volatility=0.1),
    ]
    
    ensemble = EnsembleManager(
        agents=agents,
        use_thompson_sampling=True,
        use_meta_learning=False  # Disable for faster demo
    )
    
    # Mock environment for evaluation
    class MockEnv:
        def __init__(self):
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return np.random.randn(5), {}
        
        def step(self, action):
            self.step_count += 1
            # Reward based on action quality (simulated)
            reward = np.mean(action) + np.random.normal(0, 0.1)
            done = self.step_count >= 10
            truncated = False
            return np.random.randn(5), reward, done, truncated, {}
    
    mock_env = MockEnv()
    
    # Evaluate ensemble
    results = ensemble.evaluate_ensemble(mock_env, n_episodes=5)
    
    print("Evaluation Results:")
    print(f"  Ensemble mean reward: {results['ensemble']['mean_reward']:.3f}")
    print(f"  Ensemble std reward: {results['ensemble']['std_reward']:.3f}")
    
    for agent_key, agent_results in results['individual_agents'].items():
        print(f"  {agent_key}: mean_reward = {agent_results['mean_reward']:.3f}, "
              f"weight = {agent_results['weight']:.3f}")
    
    print()


def main():
    """Run all demos"""
    print("RL Ensemble System Demo")
    print("=" * 50)
    
    demo_thompson_sampling()
    demo_meta_learning()
    demo_ensemble_manager()
    demo_ensemble_evaluation()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
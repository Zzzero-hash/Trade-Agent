"""
Comprehensive Test Suite for Rainbow DQN Training.

This test suite validates the Rainbow DQN implementation including:
- Agent initialization and configuration
- Network architectures (Dueling, Noisy, Distributional)
- Experience replay (Prioritized and standard)
- Training procedures and convergence
- Evaluation and performance metrics
- Model saving and loading
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch
import warnings

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.advanced_dqn_agent import (
    RainbowDQNAgent, RainbowDQNConfig, NoisyLinear, 
    DuelingNetwork, PrioritizedReplayBuffer
)
from src.ml.discrete_trading_wrapper import DiscreteTradingWrapper, create_discrete_trading_env
from src.ml.train_rainbow_dqn import RainbowDQNTrainer
from src.ml.yfinance_trading_environment import YFinanceTradingEnvironment, YFinanceConfig

warnings.filterwarnings('ignore', category=UserWarning)


class MockTradingEnvironment(gym.Env):
    """Mock trading environment for testing."""
    
    def __init__(self, n_symbols=3, obs_dim=50):
        super().__init__()
        self.n_symbols = n_symbols
        self.symbols = [f'SYMBOL_{i}' for i in range(n_symbols)]
        self.obs_dim = obs_dim
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=2, shape=(n_symbols * 2,), dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.max_steps = 1000
        self.portfolio_value = 100000.0
        
    def reset(self, **kwargs):
        self.current_step = 0
        self.portfolio_value = 100000.0
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Mock reward based on action
        reward = np.random.randn() * 0.1
        
        # Mock portfolio value change
        self.portfolio_value *= (1 + reward / 1000)
        
        # Generate next observation
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        
        # Episode termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': reward / 1000,
            'drawdown': max(0, (110000 - self.portfolio_value) / 110000)
        }
        
        return obs, reward, terminated, truncated, info


class TestRainbowDQNConfig(unittest.TestCase):
    """Test Rainbow DQN configuration."""
    
    def test_default_config(self):
        """Test default configuration initialization."""
        config = RainbowDQNConfig()
        
        self.assertEqual(config.n_atoms, 51)
        self.assertEqual(config.v_min, -10.0)
        self.assertEqual(config.v_max, 10.0)
        self.assertTrue(config.distributional)
        self.assertTrue(config.prioritized_replay)
        self.assertTrue(config.dueling)
        self.assertTrue(config.noisy)
        self.assertEqual(config.multi_step, 3)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid n_atoms
        with self.assertRaises(AssertionError):
            RainbowDQNConfig(n_atoms=1)
        
        # Test invalid v_min/v_max
        with self.assertRaises(AssertionError):
            RainbowDQNConfig(v_min=10.0, v_max=5.0)
        
        # Test invalid alpha
        with self.assertRaises(AssertionError):
            RainbowDQNConfig(alpha=1.5)
        
        # Test invalid multi_step
        with self.assertRaises(AssertionError):
            RainbowDQNConfig(multi_step=0)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RainbowDQNConfig(
            learning_rate=5e-4,
            batch_size=64,
            n_atoms=21,
            multi_step=5,
            noisy=False
        )
        
        self.assertEqual(config.learning_rate, 5e-4)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.n_atoms, 21)
        self.assertEqual(config.multi_step, 5)
        self.assertFalse(config.noisy)


class TestNoisyLinear(unittest.TestCase):
    """Test Noisy Linear layer."""
    
    def test_initialization(self):
        """Test noisy linear layer initialization."""
        layer = NoisyLinear(10, 5, std_init=0.5)
        
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.std_init, 0.5)
        
        # Check parameter shapes
        self.assertEqual(layer.weight_mu.shape, (5, 10))
        self.assertEqual(layer.weight_sigma.shape, (5, 10))
        self.assertEqual(layer.bias_mu.shape, (5,))
        self.assertEqual(layer.bias_sigma.shape, (5,))
    
    def test_forward_pass(self):
        """Test forward pass through noisy layer."""
        layer = NoisyLinear(10, 5)
        x = torch.randn(3, 10)
        
        # Training mode (with noise)
        layer.train()
        output_train = layer(x)
        self.assertEqual(output_train.shape, (3, 5))
        
        # Eval mode (without noise)
        layer.eval()
        output_eval = layer(x)
        self.assertEqual(output_eval.shape, (3, 5))
        
        # Outputs should be different due to noise (reset noise first)
        layer.train()
        layer.reset_noise()  # Reset noise to get different output
        output_train2 = layer(x)
        self.assertFalse(torch.allclose(output_train, output_train2, atol=1e-6))
    
    def test_noise_reset(self):
        """Test noise reset functionality."""
        layer = NoisyLinear(10, 5)
        x = torch.randn(3, 10)
        
        layer.train()
        output1 = layer(x)
        
        # Reset noise
        layer.reset_noise()
        output2 = layer(x)
        
        # Outputs should be different after noise reset
        self.assertFalse(torch.allclose(output1, output2))


class TestDuelingNetwork(unittest.TestCase):
    """Test Dueling Network architecture."""
    
    def test_initialization(self):
        """Test dueling network initialization."""
        network = DuelingNetwork(
            state_dim=50,
            action_dim=10,
            hidden_dims=[128, 64],
            distributional=True,
            n_atoms=51,
            noisy=True
        )
        
        self.assertEqual(network.state_dim, 50)
        self.assertEqual(network.action_dim, 10)
        self.assertEqual(network.n_atoms, 51)
        self.assertTrue(network.distributional)
        self.assertTrue(network.noisy)
    
    def test_forward_pass_distributional(self):
        """Test forward pass with distributional RL."""
        network = DuelingNetwork(
            state_dim=50,
            action_dim=10,
            hidden_dims=[128, 64],
            distributional=True,
            n_atoms=51
        )
        
        batch_size = 4
        state = torch.randn(batch_size, 50)
        
        output = network(state)
        
        # Should output probability distributions
        self.assertEqual(output.shape, (batch_size, 10, 51))
        
        # Probabilities should sum to 1
        prob_sums = output.sum(dim=-1)
        expected_sums = torch.ones(batch_size, 10)
        self.assertTrue(torch.allclose(prob_sums, expected_sums, atol=1e-6))
    
    def test_forward_pass_standard(self):
        """Test forward pass with standard DQN."""
        network = DuelingNetwork(
            state_dim=50,
            action_dim=10,
            hidden_dims=[128, 64],
            distributional=False
        )
        
        batch_size = 4
        state = torch.randn(batch_size, 50)
        
        output = network(state)
        
        # Should output Q-values
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_noise_reset(self):
        """Test noise reset in dueling network."""
        network = DuelingNetwork(
            state_dim=50,
            action_dim=10,
            hidden_dims=[128, 64],
            noisy=True
        )
        
        state = torch.randn(4, 50)
        
        network.train()
        output1 = network(state)
        
        # Reset noise
        network.reset_noise()
        output2 = network(state)
        
        # Outputs should be different after noise reset
        self.assertFalse(torch.allclose(output1, output2))


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test Prioritized Experience Replay buffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
        
        self.assertEqual(buffer.capacity, 1000)
        self.assertEqual(buffer.alpha, 0.6)
        self.assertEqual(buffer.beta, 0.4)
        self.assertEqual(len(buffer), 0)
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(50):
            experience = (
                np.random.randn(10),  # state
                np.random.randint(0, 5),  # action
                np.random.randn(),  # reward
                np.random.randn(10),  # next_state
                np.random.choice([True, False])  # done
            )
            buffer.add(experience)
        
        self.assertEqual(len(buffer), 50)
    
    def test_sampling(self):
        """Test sampling from buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(50):
            experience = (
                np.random.randn(10),
                np.random.randint(0, 5),
                np.random.randn(),
                np.random.randn(10),
                False
            )
            buffer.add(experience)
        
        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=16)
        
        self.assertEqual(len(batch), 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(weights), 16)
        
        # Check batch structure
        for experience in batch:
            self.assertEqual(len(experience), 5)  # state, action, reward, next_state, done
    
    def test_priority_update(self):
        """Test priority updates."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(20):
            experience = (np.random.randn(10), 0, 0.0, np.random.randn(10), False)
            buffer.add(experience)
        
        # Sample and update priorities
        batch, indices, weights = buffer.sample(batch_size=10)
        new_priorities = np.random.rand(10)
        buffer.update_priorities(indices, new_priorities)
        
        # Should not raise any errors
        self.assertEqual(len(buffer), 20)


class TestDiscreteTradingWrapper(unittest.TestCase):
    """Test Discrete Trading Wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_env = MockTradingEnvironment(n_symbols=3)
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        wrapper = DiscreteTradingWrapper(self.base_env, action_strategy="single_asset")
        
        self.assertIsInstance(wrapper.action_space, spaces.Discrete)
        self.assertEqual(wrapper.n_symbols, 3)
        self.assertEqual(len(wrapper.symbols), 3)
        self.assertGreater(wrapper.action_space.n, 1)
    
    def test_action_mapping(self):
        """Test action mapping creation."""
        wrapper = DiscreteTradingWrapper(self.base_env, action_strategy="single_asset")
        
        # Should have HOLD + BUY actions + SELL actions
        expected_actions = 1 + (3 * 3) + (3 * 4)  # HOLD + (symbols * buy_sizes) + (symbols * sell_fractions)
        self.assertEqual(wrapper.action_space.n, expected_actions)
        
        # Test action info
        action_info = wrapper.get_action_info(0)
        self.assertEqual(action_info['type'], 'HOLD')
        
        # Test action meanings
        meanings = wrapper.get_action_meanings()
        self.assertEqual(len(meanings), wrapper.action_space.n)
    
    def test_step_execution(self):
        """Test step execution with discrete actions."""
        wrapper = DiscreteTradingWrapper(self.base_env)
        
        obs, _ = wrapper.reset()
        
        # Test HOLD action
        obs, reward, terminated, truncated, info = wrapper.step(0)
        self.assertIn('discrete_action', info)
        self.assertIn('action_type', info)
        self.assertIn('action_description', info)
        
        # Test other actions
        for action in range(min(5, wrapper.action_space.n)):
            obs, reward, terminated, truncated, info = wrapper.step(action)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
    
    def test_portfolio_strategy(self):
        """Test portfolio-level action strategy."""
        wrapper = DiscreteTradingWrapper(self.base_env, action_strategy="portfolio")
        
        self.assertGreater(wrapper.action_space.n, 1)
        
        # Test portfolio actions
        obs, _ = wrapper.reset()
        for action in range(min(3, wrapper.action_space.n)):
            obs, reward, terminated, truncated, info = wrapper.step(action)
            self.assertIn('action_type', info)


class TestRainbowDQNAgent(unittest.TestCase):
    """Test Rainbow DQN Agent."""
    
    def setUp(self):
        """Set up test environment and agent."""
        self.base_env = MockTradingEnvironment(n_symbols=2, obs_dim=20)
        self.discrete_env = DiscreteTradingWrapper(self.base_env)
        self.config = RainbowDQNConfig(
            learning_rate=1e-3,
            batch_size=16,
            buffer_size=1000,
            learning_starts=100,
            target_update_interval=100,
            verbose=0
        )
        self.agent = RainbowDQNAgent(self.discrete_env, self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, 20)
        self.assertEqual(self.agent.action_dim, self.discrete_env.action_space.n)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertFalse(self.agent.is_trained)
    
    def test_prediction(self):
        """Test action prediction."""
        obs = np.random.randn(20).astype(np.float32)
        
        # Test random prediction (untrained)
        action, _ = self.agent.predict(obs, deterministic=False)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), 1)
        self.assertTrue(0 <= action[0] < self.agent.action_dim)
        
        # Test deterministic prediction
        action, _ = self.agent.predict(obs, deterministic=True)
        self.assertIsInstance(action, np.ndarray)
    
    def test_training_step(self):
        """Test single training step."""
        # Fill replay buffer with some experiences
        for _ in range(200):
            experience = (
                np.random.randn(20).astype(np.float32),
                np.random.randint(0, self.agent.action_dim),
                np.random.randn(),
                np.random.randn(20).astype(np.float32),
                np.random.choice([True, False])
            )
            self.agent._add_to_multi_step_buffer(experience)
        
        # Perform training step
        metrics = self.agent._train_step()
        
        if metrics:  # If training actually occurred
            self.assertIn('loss', metrics)
            self.assertIsInstance(metrics['loss'], float)
    
    def test_short_training(self):
        """Test short training run."""
        # Very short training for testing
        results = self.agent.train(
            env=self.discrete_env,
            total_timesteps=500,
            eval_env=None,
            eval_freq=200,
            n_eval_episodes=2
        )
        
        self.assertIn('agent_type', results)
        self.assertIn('total_timesteps', results)
        self.assertIn('training_time', results)
        self.assertEqual(results['agent_type'], 'Rainbow_DQN')
        self.assertEqual(results['total_timesteps'], 500)
        self.assertTrue(self.agent.is_trained)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            
            # Train briefly
            self.agent.train(self.discrete_env, total_timesteps=100)
            
            # Save model
            self.agent.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new agent and load model
            new_agent = RainbowDQNAgent(self.discrete_env, self.config)
            new_agent.load_model(model_path)
            
            self.assertTrue(new_agent.is_trained)
            self.assertEqual(new_agent.steps_done, self.agent.steps_done)


class TestRainbowDQNTrainer(unittest.TestCase):
    """Test Rainbow DQN Trainer."""
    
    def setUp(self):
        """Set up trainer for testing."""
        self.config = RainbowDQNConfig(
            learning_rate=1e-3,
            batch_size=16,
            buffer_size=500,
            learning_starts=50,
            verbose=0
        )
        self.env_config = YFinanceConfig(
            initial_balance=100000.0,
            lookback_window=20
        )
    
    @patch('src.ml.train_rainbow_dqn.YFinanceTradingEnvironment')
    def test_trainer_initialization(self, mock_env_class):
        """Test trainer initialization with mocked environment."""
        # Mock the environment creation
        mock_env = MockTradingEnvironment(n_symbols=2, obs_dim=50)
        mock_env_class.return_value = mock_env
        
        trainer = RainbowDQNTrainer(
            config=self.config,
            env_config=self.env_config,
            symbols=['AAPL', 'GOOGL'],
            start_date="2020-01-01",
            end_date="2022-12-31"
        )
        
        self.assertIsNotNone(trainer.agent)
        self.assertEqual(len(trainer.symbols), 2)
    
    def test_financial_metrics_calculation(self):
        """Test financial metrics calculation."""
        trainer = RainbowDQNTrainer(
            config=self.config,
            env_config=self.env_config
        )
        
        # Mock test results
        test_results = {
            'portfolio_values': [100000, 101000, 99000, 102000, 98000],
            'episode_rewards': [10, -20, 30, -40, 15]
        }
        
        metrics = trainer._calculate_financial_metrics(test_results)
        
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # Check win rate calculation
        expected_win_rate = 3/5  # 3 positive out of 5 episodes
        self.assertAlmostEqual(metrics['win_rate'], expected_win_rate)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete Rainbow DQN system."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create mock environment
        base_env = MockTradingEnvironment(n_symbols=2, obs_dim=30)
        discrete_env = DiscreteTradingWrapper(base_env, action_strategy="single_asset")
        
        # Create agent with minimal configuration
        config = RainbowDQNConfig(
            learning_rate=1e-3,
            batch_size=8,
            buffer_size=200,
            learning_starts=50,
            target_update_interval=50,
            verbose=0
        )
        
        agent = RainbowDQNAgent(discrete_env, config)
        
        # Train for short period
        results = agent.train(
            env=discrete_env,
            total_timesteps=200,
            eval_env=discrete_env,
            eval_freq=100,
            n_eval_episodes=2
        )
        
        # Verify training completed
        self.assertTrue(agent.is_trained)
        self.assertGreater(agent.steps_done, 0)
        self.assertGreater(agent.episodes_done, 0)
        
        # Test evaluation
        eval_results = agent._evaluate(discrete_env, n_episodes=3)
        self.assertIn('mean_reward', eval_results)
        self.assertIn('std_reward', eval_results)
    
    def test_different_configurations(self):
        """Test different Rainbow DQN configurations."""
        base_env = MockTradingEnvironment(n_symbols=1, obs_dim=20)
        discrete_env = DiscreteTradingWrapper(base_env)
        
        configurations = [
            # Standard DQN (no Rainbow features)
            RainbowDQNConfig(
                distributional=False,
                prioritized_replay=False,
                noisy=False,
                dueling=False,
                multi_step=1,
                verbose=0
            ),
            # Partial Rainbow
            RainbowDQNConfig(
                distributional=True,
                prioritized_replay=True,
                noisy=False,
                dueling=True,
                multi_step=1,
                verbose=0
            ),
            # Full Rainbow
            RainbowDQNConfig(
                distributional=True,
                prioritized_replay=True,
                noisy=True,
                dueling=True,
                multi_step=3,
                verbose=0
            )
        ]
        
        for i, config in enumerate(configurations):
            with self.subTest(config_id=i):
                agent = RainbowDQNAgent(discrete_env, config)
                
                # Short training
                results = agent.train(discrete_env, total_timesteps=100)
                
                self.assertTrue(agent.is_trained)
                self.assertIn('agent_type', results)


def run_performance_test():
    """Run performance test to validate training speed and convergence."""
    print("Running Rainbow DQN Performance Test...")
    
    # Create environment
    base_env = MockTradingEnvironment(n_symbols=3, obs_dim=50)
    discrete_env = DiscreteTradingWrapper(base_env, action_strategy="single_asset")
    
    # Create agent
    config = RainbowDQNConfig(
        learning_rate=1e-4,
        batch_size=32,
        buffer_size=10000,
        learning_starts=1000,
        target_update_interval=1000,
        verbose=1
    )
    
    agent = RainbowDQNAgent(discrete_env, config)
    
    # Train and measure performance
    import time
    start_time = time.time()
    
    results = agent.train(
        env=discrete_env,
        total_timesteps=10000,
        eval_env=discrete_env,
        eval_freq=2000,
        n_eval_episodes=5
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Steps per second: {10000 / training_time:.2f}")
    print(f"Episodes completed: {agent.episodes_done}")
    print(f"Final mean reward: {results.get('mean_episode_reward', 'N/A')}")
    
    if results.get('evaluations'):
        final_eval = results['evaluations'][-1]
        print(f"Final evaluation reward: {final_eval['mean_reward']:.4f} ± {final_eval['std_reward']:.4f}")
    
    # Test model persistence
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "performance_test_model.pth")
        agent.save_model(model_path)
        
        # Load and verify
        new_agent = RainbowDQNAgent(discrete_env, config)
        new_agent.load_model(model_path)
        
        print(f"Model saved and loaded successfully")
        print(f"Loaded agent steps: {new_agent.steps_done}")
    
    print("Performance test completed successfully!")


if __name__ == '__main__':
    # Run unit tests
    print("Running Rainbow DQN Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    run_performance_test()
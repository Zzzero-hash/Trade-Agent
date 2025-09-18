"""
Tests for RL Agent Implementations.

This module contains comprehensive tests for RL agents including training
convergence, policy evaluation, and ensemble functionality.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from unittest.mock import Mock, patch

from src.ml.rl_agents import (
    RLAgentConfig, StableBaselinesRLAgent, RLAgentFactory,
    RLAgentEnsemble, TradingCallback, create_rl_ensemble
)
from src.ml.trading_environment import TradingEnvironment, TradingConfig
from src.ml.rl_hyperopt import HyperparameterOptimizer, optimize_agent_hyperparameters


class TestRLAgentConfig:
    """Test RL agent configuration"""
    
    def test_config_initialization(self):
        """Test basic configuration initialization"""
        config = RLAgentConfig(
            agent_type="PPO",
            learning_rate=1e-3,
            batch_size=64
        )
        
        assert config.agent_type == "PPO"
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64
        assert config.policy == "MlpPolicy"
    
    def test_invalid_agent_type(self):
        """Test invalid agent type raises error"""
        with pytest.raises(ValueError, match="Unsupported agent type"):
            RLAgentConfig(agent_type="INVALID")
    
    def test_additional_params(self):
        """Test additional parameters storage"""
        config = RLAgentConfig(
            agent_type="SAC",
            custom_param=42,
            another_param="test"
        )
        
        assert config.additional_params["custom_param"] == 42
        assert config.additional_params["another_param"] == "test"


class TestTradingCallback:
    """Test trading callback functionality"""
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment"""
        env = Mock()
        env.get_portfolio_metrics.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05
        }
        return env
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_callback_initialization(self, mock_env, temp_dir):
        """Test callback initialization"""
        log_path = os.path.join(temp_dir, "eval_log.json")
        
        callback = TradingCallback(
            eval_env=mock_env,
            eval_freq=1000,
            n_eval_episodes=3,
            log_path=log_path
        )
        
        assert callback.eval_env == mock_env
        assert callback.eval_freq == 1000
        assert callback.n_eval_episodes == 3
        assert callback.log_path == log_path
        assert len(callback.evaluations) == 0
    
    @patch('src.ml.rl_agents.evaluate_policy')
    def test_callback_evaluation(self, mock_evaluate, mock_env, temp_dir):
        """Test callback evaluation functionality"""
        mock_evaluate.return_value = (100.0, 10.0)
        
        log_path = os.path.join(temp_dir, "eval_log.json")
        callback = TradingCallback(
            eval_env=mock_env,
            eval_freq=10,
            n_eval_episodes=3,
            log_path=log_path
        )
        
        # Mock model and logger
        callback.model = Mock()
        callback.logger = Mock()
        callback.n_calls = 10
        
        # Trigger evaluation
        result = callback._on_step()
        
        assert result is True
        assert len(callback.evaluations) == 1
        
        evaluation = callback.evaluations[0]
        assert evaluation['timesteps'] == 10
        assert evaluation['mean_reward'] == 100.0
        assert evaluation['std_reward'] == 10.0
        assert 'portfolio_metrics' in evaluation
        assert evaluation['portfolio_metrics']['total_return'] == 0.15


class TestStableBaselinesRLAgent:
    """Test Stable-Baselines3 RL agent wrapper"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        
        data = []
        for symbol in ['AAPL', 'GOOGL']:
            prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
            for i, (date, price) in enumerate(zip(dates, prices)):
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * (1 + np.random.randn() * 0.001),
                    'high': price * (1 + abs(np.random.randn()) * 0.002),
                    'low': price * (1 - abs(np.random.randn()) * 0.002),
                    'close': price,
                    'volume': np.random.randint(1000, 10000)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def trading_env(self, sample_market_data):
        """Create trading environment"""
        config = TradingConfig(
            initial_balance=10000,
            lookback_window=20,
            max_drawdown_limit=0.5
        )
        return TradingEnvironment(sample_market_data, config)
    
    def test_ppo_agent_creation(self, trading_env):
        """Test PPO agent creation"""
        config = RLAgentConfig(
            agent_type="PPO",
            learning_rate=3e-4,
            batch_size=32,
            n_steps=64,
            verbose=0
        )
        
        agent = StableBaselinesRLAgent(config, trading_env)
        
        assert agent.config.agent_type == "PPO"
        assert agent.model is not None
        assert not agent.is_trained
    
    def test_sac_agent_creation(self, trading_env):
        """Test SAC agent creation"""
        config = RLAgentConfig(
            agent_type="SAC",
            learning_rate=3e-4,
            buffer_size=10000,
            verbose=0
        )
        
        agent = StableBaselinesRLAgent(config, trading_env)
        
        assert agent.config.agent_type == "SAC"
        assert agent.model is not None
        assert not agent.is_trained
    
    def test_td3_agent_creation(self, trading_env):
        """Test TD3 agent creation"""
        config = RLAgentConfig(
            agent_type="TD3",
            learning_rate=1e-3,
            buffer_size=10000,
            verbose=0
        )
        
        agent = StableBaselinesRLAgent(config, trading_env)
        
        assert agent.config.agent_type == "TD3"
        assert agent.model is not None
        assert not agent.is_trained
    
    def test_dqn_agent_creation(self, trading_env):
        """Test DQN agent creation"""
        config = RLAgentConfig(
            agent_type="DQN",
            learning_rate=1e-4,
            buffer_size=10000,
            verbose=0
        )
        
        agent = StableBaselinesRLAgent(config, trading_env)
        
        assert agent.config.agent_type == "DQN"
        assert agent.model is not None
        assert not agent.is_trained
    
    def test_agent_training_convergence(self, trading_env):
        """Test agent training convergence (short training)"""
        config = RLAgentConfig(
            agent_type="PPO",
            learning_rate=3e-4,
            batch_size=32,
            n_steps=64,
            verbose=0
        )
        
        agent = StableBaselinesRLAgent(config, trading_env)
        
        # Short training for testing
        results = agent.train(
            env=trading_env,
            total_timesteps=1000,
            eval_freq=500,
            n_eval_episodes=2
        )
        
        assert agent.is_trained
        assert 'agent_type' in results
        assert 'total_timesteps' in results
        assert 'training_time' in results
        assert results['total_timesteps'] == 1000
    
    def test_prediction_before_training(self, trading_env):
        """Test prediction fails before training"""
        config = RLAgentConfig(agent_type="PPO", verbose=0)
        agent = StableBaselinesRLAgent(config, trading_env)
        
        obs = trading_env.reset()[0]
        
        with pytest.raises(ValueError, match="Agent must be trained"):
            agent.predict(obs)
    
    def test_prediction_after_training(self, trading_env):
        """Test prediction works after training"""
        config = RLAgentConfig(
            agent_type="PPO",
            batch_size=32,
            n_steps=64,
            verbose=0
        )
        agent = StableBaselinesRLAgent(config, trading_env)
        
        # Quick training
        agent.train(env=trading_env, total_timesteps=500)
        
        obs = trading_env.reset()[0]
        action, _ = agent.predict(obs)
        
        assert action is not None
        assert len(action) == trading_env.action_space.shape[0]
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_save_load(self, trading_env, temp_dir):
        """Test model saving and loading"""
        config = RLAgentConfig(
            agent_type="PPO",
            batch_size=32,
            n_steps=64,
            verbose=0
        )
        agent = StableBaselinesRLAgent(config, trading_env)
        
        # Train briefly
        agent.train(env=trading_env, total_timesteps=500)
        
        # Save model
        model_path = os.path.join(temp_dir, "test_model")
        agent.save_model(model_path)
        
        assert os.path.exists(model_path + ".zip")
        assert os.path.exists(model_path + "_metadata.json")
        
        # Create new agent and load model
        new_agent = StableBaselinesRLAgent(config, trading_env)
        new_agent.load_model(model_path)
        
        assert new_agent.is_trained
        
        # Test prediction works
        obs = trading_env.reset()[0]
        action, _ = new_agent.predict(obs)
        assert action is not None


class TestRLAgentFactory:
    """Test RL agent factory"""
    
    @pytest.fixture
    def sample_env(self):
        """Create simple mock environment"""
        env = Mock()
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        return env
    
    def test_create_ppo_agent(self, sample_env):
        """Test PPO agent creation via factory"""
        agent = RLAgentFactory.create_ppo_agent(
            env=sample_env,
            learning_rate=1e-3,
            batch_size=64,
            verbose=0
        )
        
        assert agent.config.agent_type == "PPO"
        assert agent.config.learning_rate == 1e-3
        assert agent.config.batch_size == 64
    
    def test_create_sac_agent(self, sample_env):
        """Test SAC agent creation via factory"""
        agent = RLAgentFactory.create_sac_agent(
            env=sample_env,
            learning_rate=3e-4,
            buffer_size=50000,
            verbose=0
        )
        
        assert agent.config.agent_type == "SAC"
        assert agent.config.learning_rate == 3e-4
        assert agent.config.buffer_size == 50000
    
    def test_create_td3_agent(self, sample_env):
        """Test TD3 agent creation via factory"""
        agent = RLAgentFactory.create_td3_agent(
            env=sample_env,
            learning_rate=1e-3,
            policy_delay=2,
            verbose=0
        )
        
        assert agent.config.agent_type == "TD3"
        assert agent.config.learning_rate == 1e-3
        assert agent.config.additional_params["policy_delay"] == 2
    
    def test_create_dqn_agent(self, sample_env):
        """Test DQN agent creation via factory"""
        agent = RLAgentFactory.create_dqn_agent(
            env=sample_env,
            learning_rate=1e-4,
            exploration_fraction=0.2,
            verbose=0
        )
        
        assert agent.config.agent_type == "DQN"
        assert agent.config.learning_rate == 1e-4
        assert agent.config.exploration_fraction == 0.2


class TestRLAgentEnsemble:
    """Test RL agent ensemble functionality"""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock trained agents"""
        agents = []
        for agent_type in ['PPO', 'SAC', 'TD3']:
            agent = Mock()
            agent.config = Mock()
            agent.config.agent_type = agent_type
            agent.is_trained = True
            agents.append(agent)
        return agents
    
    def test_ensemble_initialization(self, mock_agents):
        """Test ensemble initialization"""
        ensemble = RLAgentEnsemble(
            agents=mock_agents,
            weighting_method="equal",
            performance_window=50
        )
        
        assert len(ensemble.agents) == 3
        assert ensemble.weighting_method == "equal"
        assert len(ensemble.weights) == 3
        assert np.allclose(ensemble.weights, 1/3)
    
    def test_equal_weighting_prediction(self, mock_agents):
        """Test prediction with equal weighting"""
        # Setup mock predictions
        mock_agents[0].predict.return_value = (np.array([1.0, 0.5]), None)
        mock_agents[1].predict.return_value = (np.array([0.0, 1.0]), None)
        mock_agents[2].predict.return_value = (np.array([-1.0, 0.0]), None)
        
        ensemble = RLAgentEnsemble(
            agents=mock_agents,
            weighting_method="equal"
        )
        
        obs = np.array([1, 2, 3, 4, 5])
        action, info = ensemble.predict(obs)
        
        expected_action = np.array([0.0, 0.5])  # Average of predictions
        assert np.allclose(action, expected_action)
        assert 'individual_actions' in info
        assert 'weights' in info
        assert len(info['individual_actions']) == 3
    
    def test_performance_weighting_update(self, mock_agents):
        """Test performance-based weight updates"""
        ensemble = RLAgentEnsemble(
            agents=mock_agents,
            weighting_method="performance"
        )
        
        # Update with different performance
        agent_rewards = [10.0, 5.0, 15.0]  # Third agent performs best
        ensemble.update_performance(agent_rewards)
        
        # Check that weights favor better performing agent
        assert ensemble.weights[2] > ensemble.weights[0]
        assert ensemble.weights[2] > ensemble.weights[1]
        assert np.isclose(np.sum(ensemble.weights), 1.0)
    
    def test_ensemble_metrics(self, mock_agents):
        """Test ensemble metrics calculation"""
        ensemble = RLAgentEnsemble(agents=mock_agents)
        
        # Add some performance data
        ensemble.agent_rewards[0] = [10.0, 12.0, 8.0]
        ensemble.agent_rewards[1] = [5.0, 7.0, 6.0]
        ensemble.agent_rewards[2] = [15.0, 18.0, 12.0]
        
        metrics = ensemble.get_ensemble_metrics()
        
        assert metrics['num_agents'] == 3
        assert 'current_weights' in metrics
        assert 'agent_performance' in metrics
        assert len(metrics['agent_performance']) == 3
        
        # Check performance calculations
        perf = metrics['agent_performance']
        assert perf[0]['mean_reward'] == 10.0
        assert perf[1]['mean_reward'] == 6.0
        assert perf[2]['mean_reward'] == 15.0
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_ensemble_save_load(self, mock_agents, temp_dir):
        """Test ensemble saving and loading"""
        ensemble = RLAgentEnsemble(agents=mock_agents)
        
        # Add some data
        ensemble.agent_rewards[0] = [10.0, 12.0]
        ensemble.weights = np.array([0.4, 0.3, 0.3])
        
        # Save ensemble
        save_path = os.path.join(temp_dir, "ensemble.json")
        ensemble.save_ensemble(save_path)
        
        assert os.path.exists(save_path)
        
        # Load into new ensemble
        new_ensemble = RLAgentEnsemble(agents=mock_agents)
        new_ensemble.load_ensemble(save_path)
        
        assert np.allclose(new_ensemble.weights, [0.4, 0.3, 0.3])
        assert new_ensemble.agent_rewards[0] == [10.0, 12.0]


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality"""
    
    @pytest.fixture
    def simple_env_factory(self):
        """Create simple environment factory"""
        def factory():
            env = Mock()
            env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
            env.reset.return_value = (np.zeros(5), {})
            env.step.return_value = (np.zeros(5), 1.0, False, False, {})
            return env
        return factory
    
    def test_optimizer_initialization(self, simple_env_factory):
        """Test hyperparameter optimizer initialization"""
        optimizer = HyperparameterOptimizer(
            env_factory=simple_env_factory,
            agent_type="PPO",
            num_samples=5
        )
        
        assert optimizer.agent_type == "PPO"
        assert optimizer.num_samples == 5
        assert optimizer.optimization_metric == "mean_reward"
    
    def test_search_space_generation(self, simple_env_factory):
        """Test search space generation for different agents"""
        optimizer = HyperparameterOptimizer(
            env_factory=simple_env_factory,
            agent_type="PPO"
        )
        
        search_space = optimizer.get_search_space()
        
        assert 'learning_rate' in search_space
        assert 'batch_size' in search_space
        assert 'n_steps' in search_space
        assert 'clip_range' in search_space
    
    @patch('src.ml.rl_hyperopt.tune.run')
    def test_optimization_run(self, mock_tune_run, simple_env_factory):
        """Test optimization execution (mocked)"""
        # Mock tune.run return value
        mock_analysis = Mock()
        mock_trial = Mock()
        mock_trial.config = {'learning_rate': 3e-4, 'batch_size': 64}
        mock_trial.last_result = {'mean_reward': 100.0}
        mock_trial.trial_id = 'test_trial'
        mock_analysis.get_best_trial.return_value = mock_trial
        mock_analysis.trials = [mock_trial]
        mock_tune_run.return_value = mock_analysis
        
        optimizer = HyperparameterOptimizer(
            env_factory=simple_env_factory,
            agent_type="PPO",
            num_samples=2
        )
        
        results = optimizer.optimize()
        
        assert 'best_config' in results
        assert 'best_metrics' in results
        assert results['best_config']['learning_rate'] == 3e-4
        assert results['best_metrics']['mean_reward'] == 100.0
        
        # Verify tune.run was called
        mock_tune_run.assert_called_once()


class TestIntegration:
    """Integration tests for RL agents"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for integration tests"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        
        data = []
        for symbol in ['AAPL']:
            prices = 100 + np.cumsum(np.random.randn(200) * 0.01)
            for i, (date, price) in enumerate(zip(dates, prices)):
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * (1 + np.random.randn() * 0.001),
                    'high': price * (1 + abs(np.random.randn()) * 0.002),
                    'low': price * (1 - abs(np.random.randn()) * 0.002),
                    'close': price,
                    'volume': np.random.randint(1000, 10000)
                })
        
        return pd.DataFrame(data)
    
    def test_end_to_end_training_evaluation(self, sample_market_data):
        """Test end-to-end training and evaluation"""
        # Create environment
        config = TradingConfig(
            initial_balance=10000,
            lookback_window=10,
            max_drawdown_limit=0.5
        )
        env = TradingEnvironment(sample_market_data, config)
        
        # Create agent
        agent = RLAgentFactory.create_ppo_agent(
            env=env,
            learning_rate=3e-4,
            batch_size=32,
            n_steps=64,
            verbose=0
        )
        
        # Train agent (very short for testing)
        results = agent.train(
            env=env,
            total_timesteps=500,
            eval_freq=250,
            n_eval_episodes=2
        )
        
        assert agent.is_trained
        assert 'training_time' in results
        
        # Test prediction
        obs = env.reset()[0]
        action, _ = agent.predict(obs)
        
        assert action is not None
        assert len(action) == env.action_space.shape[0]
        
        # Test evaluation
        metrics = agent.evaluate(env, n_episodes=2)
        assert 'mean_reward' in metrics
        assert 'std_reward' in metrics
    
    def test_ensemble_creation_and_prediction(self, sample_market_data):
        """Test ensemble creation and prediction"""
        # Create environment
        config = TradingConfig(
            initial_balance=10000,
            lookback_window=10
        )
        env = TradingEnvironment(sample_market_data, config)
        
        # Create ensemble with different agent configs
        agent_configs = [
            {'agent_type': 'PPO', 'learning_rate': 3e-4, 'verbose': 0},
            {'agent_type': 'SAC', 'learning_rate': 3e-4, 'verbose': 0}
        ]
        
        ensemble = create_rl_ensemble(
            env=env,
            agent_configs=agent_configs,
            weighting_method="equal"
        )
        
        assert len(ensemble.agents) == 2
        assert ensemble.agents[0].config.agent_type == "PPO"
        assert ensemble.agents[1].config.agent_type == "SAC"
        
        # Train agents briefly
        for agent in ensemble.agents:
            agent.train(env=env, total_timesteps=200)
        
        # Test ensemble prediction
        obs = env.reset()[0]
        action, info = ensemble.predict(obs)
        
        assert action is not None
        assert 'individual_actions' in info
        assert 'weights' in info
        assert len(info['individual_actions']) == 2


if __name__ == "__main__":
    pytest.main([__file__])
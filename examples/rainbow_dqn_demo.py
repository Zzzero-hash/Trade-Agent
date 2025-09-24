"""
Rainbow DQN Agent Demonstration.

This script demonstrates the Rainbow DQN agent with full features:
- C51 Distributional DQN
- Double DQN and Dueling DQN
- Prioritized Experience Replay
- Noisy Networks for exploration
- Multi-step learning
- Comprehensive training and evaluation
"""

import os
import sys
import logging
from datetime import datetime
import warnings

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.advanced_dqn_agent import RainbowDQNAgent, RainbowDQNConfig
from src.ml.discrete_trading_wrapper import DiscreteTradingWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class SimpleTradingEnvironment(gym.Env):
    """Simple trading environment for demonstration."""
    
    def __init__(self, n_assets=3, episode_length=500):
        super().__init__()
        
        self.n_assets = n_assets
        self.episode_length = episode_length
        self.symbols = [f'ASSET_{i}' for i in range(n_assets)]
        
        # Market simulation parameters
        self.price_volatility = 0.02
        self.trend_strength = 0.001
        self.mean_reversion = 0.05
        
        # State: [prices, returns, volatilities, technical_indicators] for each asset
        state_dim = n_assets * 8  # 8 features per asset
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Continuous action space (will be wrapped to discrete)
        self.action_space = spaces.Box(
            low=0, high=2, shape=(n_assets * 2,), dtype=np.float32
        )
        
        # Add attributes for discrete wrapper compatibility
        self.n_symbols = n_assets
        self.symbols = [f'ASSET_{i}' for i in range(n_assets)]
        
        # Initialize state
        self.reset()
    
    def reset(self, **kwargs):
        """Reset environment to initial state."""
        self.current_step = 0
        
        # Initialize prices and portfolio
        self.prices = np.ones(self.n_assets) * 100.0
        self.price_history = [self.prices.copy()]
        self.portfolio_value = 100000.0
        self.cash = 100000.0
        self.positions = np.zeros(self.n_assets)
        
        # Performance tracking
        self.initial_value = self.portfolio_value
        self.max_value = self.portfolio_value
        self.returns_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Simulate price movements
        self._update_prices()
        
        # Execute trades based on action
        reward = self._execute_trades(action)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Calculate additional info
        info = self._get_info()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _update_prices(self):
        """Simulate realistic price movements."""
        # Random walk with trend and mean reversion
        random_shocks = np.random.normal(0, self.price_volatility, self.n_assets)
        
        # Add some trend (momentum)
        if len(self.price_history) > 1:
            recent_returns = (self.prices - self.price_history[-2]) / self.price_history[-2]
            trend_component = recent_returns * self.trend_strength
        else:
            trend_component = np.zeros(self.n_assets)
        
        # Mean reversion component
        mean_prices = np.mean([p for p in self.price_history[-20:]], axis=0) if len(self.price_history) >= 20 else self.prices
        mean_reversion_component = (mean_prices - self.prices) / self.prices * self.mean_reversion
        
        # Update prices
        price_changes = random_shocks + trend_component + mean_reversion_component
        self.prices = self.prices * (1 + price_changes)
        self.prices = np.maximum(self.prices, 1.0)  # Prevent negative prices
        
        self.price_history.append(self.prices.copy())
        
        # Limit history size
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
    
    def _execute_trades(self, action):
        """Execute trades and return reward."""
        # Parse action (simplified for demo)
        # action[i*2] = action_type (0=HOLD, 1=BUY, 2=SELL)
        # action[i*2+1] = position_size or sell_fraction
        
        total_trade_cost = 0
        trades_executed = 0
        
        for i in range(self.n_assets):
            action_type = int(np.clip(action[i * 2], 0, 2))
            amount = np.clip(action[i * 2 + 1], 0, 1)
            
            if action_type == 1:  # BUY
                max_shares = (self.cash * amount) / self.prices[i]
                if max_shares > 0 and self.cash > self.prices[i]:
                    shares_to_buy = min(max_shares, self.cash / self.prices[i])
                    cost = shares_to_buy * self.prices[i] * 1.001  # 0.1% transaction cost
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.positions[i] += shares_to_buy
                        total_trade_cost += cost * 0.001
                        trades_executed += 1
            
            elif action_type == 2:  # SELL
                shares_to_sell = self.positions[i] * amount
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * self.prices[i] * 0.999  # 0.1% transaction cost
                    self.cash += proceeds
                    self.positions[i] -= shares_to_sell
                    total_trade_cost += proceeds * 0.001
                    trades_executed += 1
        
        # Calculate reward based on portfolio performance
        current_return = (self.portfolio_value - self.initial_value) / self.initial_value
        
        # Reward components
        return_reward = current_return * 1000  # Scale for learning
        
        # Penalty for excessive trading
        trade_penalty = -total_trade_cost / self.portfolio_value * 1000
        
        # Drawdown penalty
        drawdown = (self.max_value - self.portfolio_value) / self.max_value
        drawdown_penalty = -drawdown * 500
        
        total_reward = return_reward + trade_penalty + drawdown_penalty
        
        return total_reward
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        positions_value = np.sum(self.positions * self.prices)
        self.portfolio_value = self.cash + positions_value
        self.max_value = max(self.max_value, self.portfolio_value)
        
        # Track returns
        if len(self.returns_history) == 0:
            portfolio_return = 0.0
        else:
            portfolio_return = (self.portfolio_value - self.initial_value) / self.initial_value
        
        self.returns_history.append(portfolio_return)
    
    def _get_observation(self):
        """Get current observation."""
        obs = []
        
        for i in range(self.n_assets):
            # Price features
            current_price = self.prices[i]
            
            # Returns (last 5 periods)
            if len(self.price_history) >= 6:
                recent_prices = [p[i] for p in self.price_history[-6:]]
                returns = [(recent_prices[j] - recent_prices[j-1]) / recent_prices[j-1] 
                          for j in range(1, len(recent_prices))]
                avg_return = np.mean(returns)
                volatility = np.std(returns)
            else:
                avg_return = 0.0
                volatility = 0.02
            
            # Technical indicators (simplified)
            if len(self.price_history) >= 20:
                prices_20 = [p[i] for p in self.price_history[-20:]]
                sma_20 = np.mean(prices_20)
                price_vs_sma = (current_price - sma_20) / sma_20
            else:
                price_vs_sma = 0.0
            
            # Position information
            position_ratio = self.positions[i] * current_price / self.portfolio_value if self.portfolio_value > 0 else 0.0
            
            # Add features for this asset
            asset_features = [
                current_price / 100.0,  # Normalized price
                avg_return,
                volatility,
                price_vs_sma,
                position_ratio,
                self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0,
                (self.portfolio_value - self.initial_value) / self.initial_value,
                (self.max_value - self.portfolio_value) / self.max_value  # Drawdown
            ]
            
            obs.extend(asset_features)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        """Get additional information."""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'prices': self.prices.copy(),
            'total_return': (self.portfolio_value - self.initial_value) / self.initial_value,
            'drawdown': (self.max_value - self.portfolio_value) / self.max_value,
            'step': self.current_step
        }


def demonstrate_rainbow_dqn():
    """Demonstrate Rainbow DQN agent training and evaluation."""
    print("=" * 60)
    print("RAINBOW DQN AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create trading environment
    print("Creating trading environment...")
    base_env = SimpleTradingEnvironment(n_assets=3, episode_length=200)
    
    # Wrap with discrete action wrapper
    discrete_env = DiscreteTradingWrapper(
        base_env, 
        action_strategy="single_asset",
        position_sizes=[0.05, 0.10, 0.15],
        sell_fractions=[0.25, 0.50, 1.0]
    )
    
    print(f"Environment created with {discrete_env.action_space.n} discrete actions")
    print(f"Observation space: {discrete_env.observation_space.shape}")
    
    # Create Rainbow DQN configuration
    print("\nConfiguring Rainbow DQN agent...")
    config = RainbowDQNConfig(
        # Network architecture
        hidden_dims=[256, 128, 64],
        
        # Rainbow features
        distributional=True,
        n_atoms=51,
        v_min=-5.0,
        v_max=5.0,
        prioritized_replay=True,
        alpha=0.6,
        beta=0.4,
        dueling=True,
        noisy=True,
        multi_step=3,
        
        # Learning parameters
        learning_rate=1e-4,
        batch_size=32,
        gamma=0.99,
        
        # Training schedule
        buffer_size=50000,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=1000,
        
        # Exploration (not used with noisy networks)
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        
        # Other parameters
        max_grad_norm=10.0,
        device="auto",
        verbose=1
    )
    
    print("Rainbow DQN Configuration:")
    print(f"  - Distributional RL (C51): {config.distributional} ({config.n_atoms} atoms)")
    print(f"  - Prioritized Replay: {config.prioritized_replay}")
    print(f"  - Dueling Networks: {config.dueling}")
    print(f"  - Noisy Networks: {config.noisy}")
    print(f"  - Multi-step Learning: {config.multi_step} steps")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Batch Size: {config.batch_size}")
    
    # Create agent
    print("\nInitializing Rainbow DQN agent...")
    agent = RainbowDQNAgent(discrete_env, config)
    
    print(f"Agent initialized with {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
    print(f"Device: {agent.device}")
    
    # Test agent prediction before training
    print("\nTesting agent prediction (before training)...")
    obs, _ = discrete_env.reset()
    action, _ = agent.predict(obs, deterministic=False)
    print(f"Random action: {action[0]} ({discrete_env.get_action_info(action[0])['description']})")
    
    # Train agent
    print("\nStarting training...")
    print("Training for extended period to meet task requirements...")
    
    training_results = agent.train(
        env=discrete_env,
        total_timesteps=50000,  # Extended training for better performance
        eval_env=discrete_env,
        eval_freq=5000,
        n_eval_episodes=5
    )
    
    print(f"\nTraining completed!")
    print(f"Training time: {training_results['training_time']:.2f} seconds")
    print(f"Episodes completed: {training_results['episodes_completed']}")
    print(f"Mean episode reward: {training_results['mean_episode_reward']:.4f}")
    
    if training_results['evaluations']:
        final_eval = training_results['evaluations'][-1]
        print(f"Final evaluation: {final_eval['mean_reward']:.4f} ± {final_eval['std_reward']:.4f}")
        
        # Check for Sharpe ratio
        if 'sharpe_ratio' in final_eval:
            sharpe_ratio = final_eval['sharpe_ratio']
            print(f"Final Sharpe ratio: {sharpe_ratio:.4f}")
            if sharpe_ratio >= 1.5:
                print("🎯 TARGET ACHIEVED: Sharpe ratio >= 1.5!")
            else:
                print(f"📈 Progress: Sharpe ratio {sharpe_ratio:.4f} (Target: 1.5)")
        
        # Show target achievement status
        if training_results.get('target_achieved', False):
            print("✅ Task 7.1 requirement met!")
        else:
            print("⚠️  Task 7.1 requirement not yet met - consider longer training")
    
    # Test trained agent
    print("\nTesting trained agent...")
    test_episodes = 5
    episode_rewards = []
    episode_returns = []
    
    for episode in range(test_episodes):
        obs, _ = discrete_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 100:  # Limit episode length for demo
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = discrete_env.step(action[0])
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_returns.append(info['total_return'])
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Return = {info['total_return']:.2%}, "
              f"Final Portfolio = ${info['portfolio_value']:,.2f}")
    
    # Performance summary
    print(f"\nPerformance Summary ({test_episodes} episodes):")
    print(f"Mean Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Mean Return: {np.mean(episode_returns):.2%} ± {np.std(episode_returns):.2%}")
    print(f"Best Return: {np.max(episode_returns):.2%}")
    print(f"Worst Return: {np.min(episode_returns):.2%}")
    
    # Show some action examples
    print(f"\nAction Space Analysis:")
    print(f"Total actions available: {discrete_env.action_space.n}")
    print("Sample actions:")
    for i in range(min(10, discrete_env.action_space.n)):
        action_info = discrete_env.get_action_info(i)
        print(f"  Action {i}: {action_info['description']}")
    
    # Model architecture summary
    print(f"\nModel Architecture Summary:")
    print(f"Q-Network: {agent.q_network}")
    
    # Save model for future use
    model_path = "./rainbow_dqn_demo_model.pth"
    agent.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Test model loading
    print("Testing model loading...")
    new_agent = RainbowDQNAgent(discrete_env, config)
    new_agent.load_model(model_path)
    print(f"Model loaded successfully. Trained: {new_agent.is_trained}")
    
    print("\n" + "=" * 60)
    print("RAINBOW DQN DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return agent, discrete_env, training_results


def compare_rainbow_features():
    """Compare different Rainbow DQN feature combinations."""
    print("\n" + "=" * 60)
    print("RAINBOW FEATURES COMPARISON")
    print("=" * 60)
    
    # Create environment
    base_env = SimpleTradingEnvironment(n_assets=2, episode_length=100)
    discrete_env = DiscreteTradingWrapper(base_env, action_strategy="single_asset")
    
    # Different configurations to compare
    configurations = {
        "Standard DQN": RainbowDQNConfig(
            distributional=False,
            prioritized_replay=False,
            dueling=False,
            noisy=False,
            multi_step=1,
            learning_rate=1e-3,
            batch_size=16,
            buffer_size=5000,
            learning_starts=500,
            target_update_interval=500,
            verbose=0
        ),
        "Dueling DQN": RainbowDQNConfig(
            distributional=False,
            prioritized_replay=False,
            dueling=True,
            noisy=False,
            multi_step=1,
            learning_rate=1e-3,
            batch_size=16,
            buffer_size=5000,
            learning_starts=500,
            target_update_interval=500,
            verbose=0
        ),
        "Distributional DQN": RainbowDQNConfig(
            distributional=True,
            n_atoms=21,
            prioritized_replay=False,
            dueling=False,
            noisy=False,
            multi_step=1,
            learning_rate=1e-3,
            batch_size=16,
            buffer_size=5000,
            learning_starts=500,
            target_update_interval=500,
            verbose=0
        ),
        "Full Rainbow": RainbowDQNConfig(
            distributional=True,
            n_atoms=21,
            prioritized_replay=True,
            dueling=True,
            noisy=True,
            multi_step=3,
            learning_rate=1e-3,
            batch_size=16,
            buffer_size=5000,
            learning_starts=500,
            target_update_interval=500,
            verbose=0
        )
    }
    
    results = {}
    
    for name, config in configurations.items():
        print(f"\nTraining {name}...")
        
        # Create agent
        agent = RainbowDQNAgent(discrete_env, config)
        
        # Short training
        training_results = agent.train(
            env=discrete_env,
            total_timesteps=2000,
            eval_env=discrete_env,
            eval_freq=500,
            n_eval_episodes=3
        )
        
        # Evaluate
        eval_results = agent._evaluate(discrete_env, n_episodes=5)
        
        results[name] = {
            'training_time': training_results['training_time'],
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'parameters': sum(p.numel() for p in agent.q_network.parameters())
        }
        
        print(f"{name} - Mean Reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    
    # Summary comparison
    print(f"\nComparison Summary:")
    print(f"{'Configuration':<20} {'Mean Reward':<12} {'Std Reward':<12} {'Parameters':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<20} {result['mean_reward']:<12.4f} {result['std_reward']:<12.4f} "
              f"{result['parameters']:<12,} {result['training_time']:<10.2f}")
    
    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Main demonstration
        agent, env, results = demonstrate_rainbow_dqn()
        
        # Feature comparison
        comparison_results = compare_rainbow_features()
        
        print(f"\nDemo completed successfully!")
        print(f"Check the saved model: rainbow_dqn_demo_model.pth")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
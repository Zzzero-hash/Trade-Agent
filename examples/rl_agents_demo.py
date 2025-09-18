"""
RL Agents Demo Script.

This script demonstrates how to use the RL agents for trading, including:
- Creating and training individual agents (PPO, SAC, TD3, DQN)
- Hyperparameter optimization
- Ensemble creation and usage
- Model saving and loading
- Performance evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.rl_agents import (
    RLAgentFactory, RLAgentEnsemble, create_rl_ensemble
)
from src.ml.rl_hyperopt import optimize_agent_hyperparameters
from src.ml.trading_environment import TradingEnvironment, TradingConfig


def generate_sample_market_data(
    symbols: list = ['AAPL', 'GOOGL'],
    start_date: str = '2023-01-01',
    periods: int = 2000,
    freq: str = '1H'
) -> pd.DataFrame:
    """Generate sample market data for demonstration"""
    print("Generating sample market data...")
    
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=periods, freq=freq)
    
    data = []
    for symbol in symbols:
        # Generate realistic price movements
        initial_price = np.random.uniform(50, 200)
        returns = np.random.normal(0, 0.02, periods)  # 2% volatility
        prices = initial_price * np.exp(np.cumsum(returns))
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add some intraday volatility
            open_price = price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(10000, 100000)
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} data points for {len(symbols)} symbols")
    return df


def create_trading_environment(market_data: pd.DataFrame) -> TradingEnvironment:
    """Create trading environment with sample data"""
    print("Creating trading environment...")
    
    config = TradingConfig(
        initial_balance=100000.0,
        max_position_size=0.3,  # Max 30% per position
        transaction_cost=0.001,  # 0.1% transaction cost
        slippage=0.0005,  # 0.05% slippage
        lookback_window=50,  # 50 periods lookback
        max_drawdown_limit=0.25,  # 25% max drawdown
        reward_scaling=1000.0
    )
    
    env = TradingEnvironment(
        market_data=market_data,
        config=config,
        symbols=['AAPL', 'GOOGL']
    )
    
    print(f"Environment created with {len(market_data)} data points")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    return env


def demo_individual_agents(env: TradingEnvironment):
    """Demonstrate individual RL agents"""
    print("\n" + "="*60)
    print("INDIVIDUAL AGENT DEMONSTRATION")
    print("="*60)
    
    agents = {}
    training_timesteps = 10000  # Reduced for demo
    
    # Create different types of agents
    agent_configs = {
        'PPO': {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 1024,
            'n_epochs': 10,
            'clip_range': 0.2,
            'verbose': 1
        },
        'SAC': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'verbose': 1
        },
        'TD3': {
            'learning_rate': 1e-3,
            'buffer_size': 100000,
            'batch_size': 100,
            'tau': 0.005,
            'policy_delay': 2,
            'verbose': 1
        }
    }
    
    # Train each agent
    for agent_type, config in agent_configs.items():
        print(f"\nTraining {agent_type} agent...")
        
        # Create agent
        if agent_type == 'PPO':
            agent = RLAgentFactory.create_ppo_agent(env, **config)
        elif agent_type == 'SAC':
            agent = RLAgentFactory.create_sac_agent(env, **config)
        elif agent_type == 'TD3':
            agent = RLAgentFactory.create_td3_agent(env, **config)
        
        # Train agent
        results = agent.train(
            env=env,
            total_timesteps=training_timesteps,
            eval_freq=2000,
            n_eval_episodes=3
        )
        
        agents[agent_type] = agent
        
        print(f"{agent_type} training completed in {results['training_time']:.2f} seconds")
        
        # Evaluate agent
        metrics = agent.evaluate(env, n_episodes=5)
        print(f"{agent_type} evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return agents


def demo_hyperparameter_optimization(env_factory):
    """Demonstrate hyperparameter optimization"""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    print("Running hyperparameter optimization for PPO agent...")
    print("(This is a simplified demo with few samples)")
    
    # Note: This would normally use Ray Tune, but for demo we'll simulate
    try:
        results = optimize_agent_hyperparameters(
            env_factory=env_factory,
            agent_type='PPO',
            num_samples=3,  # Very small for demo
            optimization_metric='mean_reward'
        )
        
        print("Optimization completed!")
        print(f"Best configuration: {results['best_config']}")
        print(f"Best performance: {results['best_metrics']['mean_reward']:.4f}")
        
        return results['best_config']
        
    except Exception as e:
        print(f"Hyperparameter optimization failed (expected in demo): {e}")
        print("Using default configuration instead...")
        
        # Return default config
        return {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 1024,
            'n_epochs': 10,
            'clip_range': 0.2
        }


def demo_ensemble_agents(env: TradingEnvironment, trained_agents: dict):
    """Demonstrate ensemble of RL agents"""
    print("\n" + "="*60)
    print("ENSEMBLE AGENT DEMONSTRATION")
    print("="*60)
    
    # Create ensemble from trained agents
    agent_list = list(trained_agents.values())
    
    ensemble = RLAgentEnsemble(
        agents=agent_list,
        weighting_method="performance",
        performance_window=100,
        min_weight=0.1
    )
    
    print(f"Created ensemble with {len(agent_list)} agents")
    print(f"Agent types: {[agent.config.agent_type for agent in agent_list]}")
    print(f"Initial weights: {ensemble.weights}")
    
    # Test ensemble predictions
    print("\nTesting ensemble predictions...")
    
    obs = env.reset()[0]
    episode_rewards = []
    
    for episode in range(5):
        obs = env.reset()[0]
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:  # Limit steps for demo
            # Get ensemble prediction
            action, info = ensemble.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        
        # Update ensemble performance (simplified)
        individual_rewards = [episode_reward] * len(agent_list)  # Simplified
        ensemble.update_performance(individual_rewards)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nEnsemble performance:")
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Std episode reward: {np.std(episode_rewards):.2f}")
    print(f"Final weights: {ensemble.weights}")
    
    # Get ensemble metrics
    metrics = ensemble.get_ensemble_metrics()
    print(f"\nEnsemble metrics:")
    for key, value in metrics.items():
        if key != 'agent_performance':
            print(f"  {key}: {value}")
    
    return ensemble


def demo_model_persistence(agent, ensemble, save_dir: str = "demo_models"):
    """Demonstrate model saving and loading"""
    print("\n" + "="*60)
    print("MODEL PERSISTENCE DEMONSTRATION")
    print("="*60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual agent
    agent_path = os.path.join(save_dir, "demo_ppo_agent")
    print(f"Saving agent to {agent_path}...")
    agent.save_model(agent_path)
    print("Agent saved successfully!")
    
    # Save ensemble
    ensemble_path = os.path.join(save_dir, "demo_ensemble.json")
    print(f"Saving ensemble to {ensemble_path}...")
    ensemble.save_ensemble(ensemble_path)
    print("Ensemble saved successfully!")
    
    # Demonstrate loading (create new instances)
    print("\nTesting model loading...")
    
    # Load agent
    new_agent = RLAgentFactory.create_ppo_agent(
        env=agent.env,
        verbose=0
    )
    new_agent.load_model(agent_path)
    print("Agent loaded successfully!")
    
    # Load ensemble
    new_ensemble = RLAgentEnsemble(
        agents=ensemble.agents,
        weighting_method="performance"
    )
    new_ensemble.load_ensemble(ensemble_path)
    print("Ensemble loaded successfully!")
    
    print(f"Loaded ensemble weights: {new_ensemble.weights}")
    
    return new_agent, new_ensemble


def demo_performance_analysis(agents: dict, env: TradingEnvironment):
    """Demonstrate performance analysis and comparison"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Evaluate all agents
    results = {}
    
    for agent_type, agent in agents.items():
        print(f"\nEvaluating {agent_type} agent...")
        
        # Run evaluation episodes
        episode_rewards = []
        portfolio_values = []
        
        for episode in range(3):  # Limited for demo
            obs = env.reset()[0]
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 50:  # Limited steps for demo
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
            portfolio_values.append(info.get('portfolio_value', 100000))
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        final_portfolio = np.mean(portfolio_values)
        
        results[agent_type] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'final_portfolio_value': final_portfolio,
            'total_return': (final_portfolio - 100000) / 100000
        }
        
        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Std reward: {std_reward:.2f}")
        print(f"  Final portfolio value: ${final_portfolio:,.2f}")
        print(f"  Total return: {results[agent_type]['total_return']:.2%}")
    
    # Compare agents
    print(f"\nAgent Comparison:")
    print(f"{'Agent':<8} {'Mean Reward':<12} {'Std Reward':<12} {'Total Return':<12}")
    print("-" * 50)
    
    for agent_type, metrics in results.items():
        print(f"{agent_type:<8} {metrics['mean_reward']:<12.2f} "
              f"{metrics['std_reward']:<12.2f} {metrics['total_return']:<12.2%}")
    
    return results


def main():
    """Main demonstration function"""
    print("RL AGENTS DEMONSTRATION")
    print("="*60)
    print("This demo shows how to use RL agents for trading")
    print("Note: Training times are reduced for demonstration purposes")
    
    # Generate sample data
    market_data = generate_sample_market_data(
        symbols=['AAPL', 'GOOGL'],
        periods=1000  # Reduced for demo
    )
    
    # Create environment
    env = create_trading_environment(market_data)
    
    # Environment factory for hyperparameter optimization
    def env_factory():
        return create_trading_environment(market_data)
    
    try:
        # Demo 1: Individual agents
        trained_agents = demo_individual_agents(env)
        
        # Demo 2: Hyperparameter optimization (simplified)
        best_config = demo_hyperparameter_optimization(env_factory)
        
        # Demo 3: Ensemble agents
        ensemble = demo_ensemble_agents(env, trained_agents)
        
        # Demo 4: Model persistence
        if trained_agents:
            first_agent = list(trained_agents.values())[0]
            demo_model_persistence(first_agent, ensemble)
        
        # Demo 5: Performance analysis
        performance_results = demo_performance_analysis(trained_agents, env)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. Multiple RL algorithms can be used for trading")
        print("2. Hyperparameter optimization improves performance")
        print("3. Ensembles can combine multiple agents effectively")
        print("4. Models can be saved and loaded for persistence")
        print("5. Performance analysis helps compare different approaches")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This might be due to missing dependencies or environment issues")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
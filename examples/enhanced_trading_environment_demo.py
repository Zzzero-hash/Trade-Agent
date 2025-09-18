"""Enhanced Trading Environment Demo

This script demonstrates the enhanced trading environment with CNN+LSTM
feature integration, showing how to use the environment for RL training
with rich learned features instead of basic technical indicators.

Requirements: 1.4, 2.4, 9.1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.enhanced_trading_environment import (
    EnhancedTradingEnvironment,
    create_enhanced_trading_config
)
from src.ml.cnn_lstm_feature_extractor import create_feature_extraction_config


def create_sample_market_data(num_days: int = 100, symbols: list = None) -> pd.DataFrame:
    """Create sample market data for demonstration"""
    if symbols is None:
        symbols = ['AAPL']
    
    np.random.seed(42)
    
    # Create timestamps (hourly data)
    start_date = datetime.now() - timedelta(days=num_days)
    timestamps = pd.date_range(start_date, periods=num_days * 24, freq='1H')
    
    data = []
    
    for symbol in symbols:
        # Initialize price
        base_price = np.random.uniform(50, 200)
        price = base_price
        
        for i, timestamp in enumerate(timestamps):
            # Random walk with trend
            price_change = np.random.randn() * 0.02 + 0.0001  # Small upward trend
            price = max(price * (1 + price_change), 1.0)  # Prevent negative prices
            
            # OHLCV data
            high = price * np.random.uniform(1.0, 1.02)
            low = price * np.random.uniform(0.98, 1.0)
            open_price = price * np.random.uniform(0.99, 1.01)
            volume = np.random.randint(1000, 100000)
            
            # Technical indicators (simplified)
            returns = price_change
            volatility = abs(np.random.randn() * 0.02)
            rsi = np.clip(50 + np.random.randn() * 15, 0, 100)
            macd = np.random.randn() * 0.1
            macd_signal = macd + np.random.randn() * 0.05
            bb_position = np.clip(np.random.uniform(0, 1), 0, 1)
            volume_ratio = np.random.uniform(0.5, 2.0)
            
            # Moving averages
            sma_5 = price * np.random.uniform(0.98, 1.02)
            sma_20 = price * np.random.uniform(0.95, 1.05)
            ema_12 = price * np.random.uniform(0.97, 1.03)
            
            row = {
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'returns': returns,
                'volatility': volatility,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'sma_5': sma_5,
                'sma_20': sma_20,
                'ema_12': ema_12
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)
    df.index = range(len(df))
    
    return df


def demonstrate_basic_usage():
    """Demonstrate basic usage of enhanced trading environment"""
    print("=== Enhanced Trading Environment Demo ===")
    print()
    
    # Create sample market data
    print("Creating sample market data...")
    market_data = create_sample_market_data(num_days=30, symbols=['AAPL'])
    print(f"Created {len(market_data)} data points")
    print(f"Data shape: {market_data.shape}")
    print(f"Date range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
    print()
    
    # Create enhanced trading configuration
    print("Creating enhanced trading configuration...")
    config = create_enhanced_trading_config(
        initial_balance=100000.0,
        lookback_window=60,
        fused_feature_dim=256,
        enable_feature_caching=True,
        enable_fallback=True,
        fallback_feature_dim=15,
        include_uncertainty=True,
        include_ensemble_weights=False
    )
    print(f"Configuration created:")
    print(f"  Initial balance: ${config.initial_balance:,.2f}")
    print(f"  Lookback window: {config.lookback_window}")
    print(f"  Feature dimension: {config.fused_feature_dim}")
    print(f"  Fallback enabled: {config.enable_fallback}")
    print()
    
    # Create enhanced trading environment
    print("Creating enhanced trading environment...")
    env = EnhancedTradingEnvironment(
        market_data=market_data,
        config=config,
        symbols=['AAPL']
    )
    
    print(f"Environment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Observation dimension: {env.observation_space.shape[0]}")
    print()
    
    # Reset environment and get initial observation
    print("Resetting environment...")
    observation, info = env.reset(seed=42)
    
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation stats:")
    print(f"  Min: {observation.min():.4f}")
    print(f"  Max: {observation.max():.4f}")
    print(f"  Mean: {observation.mean():.4f}")
    print(f"  Std: {observation.std():.4f}")
    print()
    
    print(f"Initial info keys: {list(info.keys())}")
    if 'feature_extractor_status' in info:
        status = info['feature_extractor_status']
        print(f"Feature extractor status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    print()
    
    return env, observation, info


def demonstrate_environment_interaction(env, num_steps: int = 50):
    """Demonstrate environment interaction"""
    print(f"=== Environment Interaction Demo ({num_steps} steps) ===")
    print()
    
    # Track performance
    observations = []
    rewards = []
    actions = []
    portfolio_values = []
    feature_stats = []
    
    observation, _ = env.reset(seed=42)
    
    for step in range(num_steps):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        observations.append(observation.copy())
        rewards.append(reward)
        actions.append(action.copy())
        portfolio_values.append(info.get('portfolio_value', 0))
        
        # Store feature extraction stats
        feature_stats.append({
            'step': step,
            'extraction_count': info.get('feature_extraction_count', 0),
            'fallback_count': info.get('fallback_usage_count', 0),
            'fallback_rate': info.get('fallback_rate', 0)
        })
        
        # Print progress
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: "
                  f"Reward: {reward:8.4f}, "
                  f"Portfolio: ${info.get('portfolio_value', 0):10,.2f}, "
                  f"Fallback rate: {info.get('fallback_rate', 0):6.2%}")
        
        # Check if episode ended
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
        
        observation = next_observation
    
    print()
    
    # Print summary statistics
    print("=== Episode Summary ===")
    print(f"Total steps: {len(rewards)}")
    print(f"Total reward: {sum(rewards):.4f}")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
    print(f"Total return: {(portfolio_values[-1] / env.initial_balance - 1) * 100:.2f}%")
    print()
    
    # Feature extraction statistics
    if feature_stats:
        final_stats = feature_stats[-1]
        print("=== Feature Extraction Statistics ===")
        print(f"Total extractions: {final_stats['extraction_count']}")
        print(f"Fallback usage: {final_stats['fallback_count']}")
        print(f"Fallback rate: {final_stats['fallback_rate']:.2%}")
        print()
    
    return {
        'observations': observations,
        'rewards': rewards,
        'actions': actions,
        'portfolio_values': portfolio_values,
        'feature_stats': feature_stats
    }


def demonstrate_feature_extraction_modes(env):
    """Demonstrate different feature extraction modes"""
    print("=== Feature Extraction Modes Demo ===")
    print()
    
    # Test normal mode (fallback since no model loaded)
    print("Testing normal mode (fallback)...")
    env.enable_fallback_mode(False)  # Try to disable fallback
    
    observation, info = env.reset()
    status = env.get_feature_extractor_status()
    
    print(f"Feature extractor status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        if i == 0:
            print(f"First observation shape: {obs.shape}")
            print(f"Fallback rate: {step_info.get('fallback_rate', 0):.2%}")
        
        if terminated or truncated:
            break
    
    print()
    
    # Test explicit fallback mode
    print("Testing explicit fallback mode...")
    env.enable_fallback_mode(True)
    
    observation, info = env.reset()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        if i == 0:
            print(f"Fallback observation shape: {obs.shape}")
            print(f"Fallback rate: {step_info.get('fallback_rate', 0):.2%}")
        
        if terminated or truncated:
            break
    
    print()


def demonstrate_enhanced_metrics(env):
    """Demonstrate enhanced metrics functionality"""
    print("=== Enhanced Metrics Demo ===")
    print()
    
    # Reset and run for a few steps
    env.reset()
    
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Get enhanced metrics
    metrics = env.get_enhanced_metrics()
    
    print("Enhanced metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print()


def plot_results(results):
    """Plot results from environment interaction"""
    print("=== Plotting Results ===")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Enhanced Trading Environment Results')
        
        # Portfolio value over time
        axes[0, 0].plot(results['portfolio_values'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Rewards over time
        axes[0, 1].plot(results['rewards'])
        axes[0, 1].set_title('Rewards Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
        
        # Cumulative rewards
        cumulative_rewards = np.cumsum(results['rewards'])
        axes[1, 0].plot(cumulative_rewards)
        axes[1, 0].set_title('Cumulative Rewards')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].grid(True)
        
        # Feature extraction statistics
        if results['feature_stats']:
            fallback_rates = [stat['fallback_rate'] for stat in results['feature_stats']]
            axes[1, 1].plot(fallback_rates)
            axes[1, 1].set_title('Fallback Usage Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Fallback Rate')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = 'enhanced_trading_environment_demo.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            print("Could not display plot (no GUI available)")
        
    except Exception as e:
        print(f"Could not create plots: {e}")


def main():
    """Main demonstration function"""
    print("Enhanced Trading Environment with CNN+LSTM Integration Demo")
    print("=" * 60)
    print()
    
    try:
        # Basic usage demonstration
        env, observation, info = demonstrate_basic_usage()
        
        # Environment interaction demonstration
        results = demonstrate_environment_interaction(env, num_steps=100)
        
        # Feature extraction modes demonstration
        demonstrate_feature_extraction_modes(env)
        
        # Enhanced metrics demonstration
        demonstrate_enhanced_metrics(env)
        
        # Plot results
        plot_results(results)
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
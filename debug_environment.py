#!/usr/bin/env python3
"""Debug script to test the trading environment and identify reward issues."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

# Import directly to avoid module loading issues
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import gymnasium as gym

def debug_environment():
    """Debug the trading environment to understand reward calculation."""
    
    # Create environment config
    config = YFinanceConfig(
        initial_balance=100000.0,
        max_position_size=0.1,
        transaction_cost=0.001,
        slippage_factor=0.0005,
        lookback_window=60,
        min_episode_length=100,
        max_drawdown_limit=0.2,
        reward_scaling=10.0
    )
    
    # Create environment
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    env = YFinanceTradingEnvironment(
        symbols=symbols,
        start_date="2022-01-01",
        end_date="2023-01-01",
        config=config
    )
    
    print("=== Environment Debug ===")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Symbols: {symbols}")
    print(f"Initial balance: ${config.initial_balance:,.2f}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial state:")
    print(f"Portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"Cash balance: ${info['cash_balance']:,.2f}")
    print(f"Current step: {env.current_step}")
    
    # Test a few steps with different actions
    print(f"\n=== Testing Actions ===")
    
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Create a simple buy action for first symbol
        action = np.zeros(env.action_space.shape[0])
        if step == 0:
            # Buy AAPL
            action[0] = 1  # BUY action
            action[1] = 0.05  # 5% position size
            print("Action: BUY AAPL 5%")
        elif step == 2:
            # Buy GOOGL
            action[2] = 1  # BUY action  
            action[3] = 0.03  # 3% position size
            print("Action: BUY GOOGL 3%")
        else:
            print("Action: HOLD all")
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.6f}")
        print(f"Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"Cash balance: ${info['cash_balance']:,.2f}")
        print(f"Positions: {info['positions']}")
        print(f"Portfolio return: {info.get('portfolio_return', 0):.6f}")
        
        if terminated or truncated:
            print("Episode terminated")
            break
    
    print(f"\n=== Portfolio History ===")
    print(f"Portfolio history length: {len(env.portfolio_history)}")
    if len(env.portfolio_history) > 0:
        print(f"Portfolio values: {env.portfolio_history}")
        
    print(f"\n=== Returns History ===")
    print(f"Returns history length: {len(env.returns_history)}")
    if len(env.returns_history) > 0:
        print(f"Returns: {env.returns_history}")

if __name__ == "__main__":
    debug_environment()
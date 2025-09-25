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

from config.trading_configs import TradingConfigFactory


def debug_environment():
    """Debug the trading environment to understand reward calculation."""
    
    # Use centralized configuration factory
    config = TradingConfigFactory.create_debug_config(
        max_position_size=0.1,
        reward_scaling=10.0
    )
    
    # Create environment with default symbols subset
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
    
    # Use centralized portfolio utilities
    from utils.portfolio_utils import format_portfolio_status
    print(f"\nInitial state:")
    print(format_portfolio_status(info))
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
        
        # Use centralized portfolio utilities
        from utils.portfolio_utils import log_action_result
        print(log_action_result(action, reward, info, step + 1))
        
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
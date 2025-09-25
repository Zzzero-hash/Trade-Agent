#!/usr/bin/env python3
"""
Simple runner for Task 7.2 - Train sophisticated PPO agent with centralized config.
"""

import sys
sys.path.append('src')

from ml.sophisticated_ppo_trainer import SophisticatedPPOTrainer
from config.trading_configs import (
    TradingConfigFactory, 
    get_default_symbols, 
    validate_performance,
    format_performance_report
)


def main():
    """Train sophisticated PPO agent for Task 7.2"""
    print("Starting Task 7.2: Train sophisticated PPO agent")
    
    # Use centralized configuration factory
    env_config = TradingConfigFactory.create_training_config(
        max_position_size=0.2,
        reward_scaling=10.0
    )
    
    trainer = SophisticatedPPOTrainer(
        env_config=env_config,
        symbols=get_default_symbols(),
        start_date="2020-01-01", 
        end_date="2023-12-31",
        n_envs=8,
        log_dir="logs/task_7_2"
    )
    
    # Train until benchmarks are met (quick test mode)
    results = trainer.train(
        total_timesteps=50000,  # Reduced for quick demonstration
        target_kl=0.01,
        verbose=1
    )
    
    # Use centralized performance validation
    validation_results = validate_performance(results)
    
    # Print formatted report
    print("\nTask 7.2 Results:")
    print(format_performance_report(validation_results))
    
    return results


if __name__ == "__main__":
    main()
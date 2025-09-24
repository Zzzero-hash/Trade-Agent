#!/usr/bin/env python3
"""
Task 7.1 Execution: Train advanced DQN agent with full Rainbow implementation.

This script executes the task requirements:
- Implement and train C51 distributional DQN for 2000+ episodes until convergence
- Train Double DQN, Dueling DQN with prioritized experience replay for stable learning
- Add Noisy Networks training with parameter space exploration over 1000+ episodes
- Validate DQN performance achieving >1.5 Sharpe ratio on training environment
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.rainbow_dqn_performance_trainer import RainbowDQNTask71Trainer, Task71TrainingConfig

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def execute_task_7_1():
    """Execute Task 7.1 requirements."""
    logger.info("Starting Task 7.1: Advanced Rainbow DQN Training")
    
    # Create configuration for task requirements
    config = Task71TrainingConfig(
        target_episodes=2000,  # Meet 2000+ episodes requirement
        target_sharpe_ratio=1.5,  # Meet >1.5 Sharpe ratio requirement
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    # Create trainer (includes all Rainbow features)
    trainer = RainbowDQNTask71Trainer(config)
    
    # Execute training
    logger.info("Training Rainbow DQN with all components enabled...")
    results = trainer.train_rainbow_dqn()
    
    # Validate results
    requirements_met = results['task_7_1_requirements']
    
    logger.info("=" * 50)
    logger.info("TASK 7.1 RESULTS")
    logger.info("=" * 50)
    logger.info("Episodes completed: %d (target: %d+)", 
                requirements_met['episodes_completed'], config.target_episodes)
    logger.info("Final Sharpe ratio: %.4f (target: >%.1f)", 
                requirements_met['final_sharpe_ratio'], config.target_sharpe_ratio)
    logger.info("Target achieved: %s", requirements_met['target_achieved'])
    
    # Check if all requirements met
    success = (requirements_met['target_episodes_met'] and 
               requirements_met['target_sharpe_met'])
    
    if success:
        logger.info("SUCCESS: Task 7.1 requirements met!")
        return 0
    else:
        logger.error("FAILED: Task 7.1 requirements not met")
        return 1


if __name__ == "__main__":
    exit_code = execute_task_7_1()
    sys.exit(exit_code)
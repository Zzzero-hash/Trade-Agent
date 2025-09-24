#!/usr/bin/env python3
"""
Execute Task 7.1: Train advanced DQN agent with full Rainbow implementation.

This script executes the complete Task 7.1 requirements:
- Implement and train C51 distributional DQN for 2000+ episodes until convergence
- Train Double DQN, Dueling DQN with prioritized experience replay for stable learning
- Add Noisy Networks training with parameter space exploration over 1000+ episodes
- Validate DQN performance achieving >1.5 Sharpe ratio on training environment

Requirements: 2.1, 3.1, 9.2
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.rainbow_dqn_task_7_1_trainer import RainbowDQNTask71Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/task_7_1_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Execute complete Task 7.1 training pipeline."""
    logger.info("Starting Task 7.1: Advanced DQN Agent with Full Rainbow Implementation")
    logger.info("=" * 80)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/rainbow_dqn_task_7_1", exist_ok=True)
    
    try:
        # Initialize trainer with high-quality financial symbols
        symbols = [
            "SPY",   # S&P 500 ETF
            "QQQ",   # NASDAQ 100 ETF
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "GOOGL", # Google
            "AMZN",  # Amazon
            "TSLA",  # Tesla
            "NVDA",  # NVIDIA
            "META",  # Meta
            "NFLX"   # Netflix
        ]
        
        trainer = RainbowDQNTask71Trainer(
            symbols=symbols,
            start_date="2018-01-01",
            end_date="2023-12-31",
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        
        # Execute complete Task 7.1
        logger.info("Executing complete Task 7.1 training pipeline...")
        results = trainer.run_complete_task_7_1(
            save_dir="models/rainbow_dqn_task_7_1",
            target_sharpe=1.5
        )
        
        # Log final results
        logger.info("=" * 80)
        logger.info("TASK 7.1 EXECUTION COMPLETED!")
        logger.info("=" * 80)
        
        if results.get('task_completed', False):
            logger.info("SUCCESS: All Task 7.1 requirements have been met!")
            logger.info("C51 distributional DQN trained for 2000+ episodes")
            logger.info("Double DQN, Dueling DQN with prioritized experience replay")
            logger.info("Noisy Networks training with parameter space exploration")
            logger.info("DQN performance validated with >1.5 Sharpe ratio")
            
            # Log performance metrics
            if 'validation_results' in results:
                val_results = results['validation_results']
                logger.info(f"Final Sharpe Ratio: {val_results.get('achieved_sharpe', 0):.4f}")
                logger.info(f"Target Met: {val_results.get('target_met', False)}")
        else:
            logger.warning("PARTIAL SUCCESS: Some requirements may not have been fully met")
            logger.warning("Check individual training results for details")
        
        # Log save locations
        logger.info(f"Models saved to: models/rainbow_dqn_task_7_1/")
        logger.info(f"Results saved to: models/rainbow_dqn_task_7_1/complete_task_7_1_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"ERROR during Task 7.1 execution: {e}")
        logger.error("Check logs for detailed error information")
        raise


if __name__ == "__main__":
    main()
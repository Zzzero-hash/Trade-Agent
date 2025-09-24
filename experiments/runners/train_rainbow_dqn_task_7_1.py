#!/usr/bin/env python3
"""
Rainbow DQN Task 7.1 Training Runner.

This experiment runner executes the complete Task 7.1 requirements:
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.rainbow_dqn_task_7_1_trainer import RainbowDQNTask71Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/rainbow_dqn_task_7_1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Execute complete Rainbow DQN Task 7.1 training pipeline."""
    logger.info("Starting Rainbow DQN Task 7.1 Training")
    logger.info("=" * 80)
    logger.info("Training Components:")
    logger.info("1. C51 Distributional DQN (2000+ episodes)")
    logger.info("2. Double DQN + Dueling DQN + Prioritized Replay (1500+ episodes)")
    logger.info("3. Noisy Networks Exploration (1000+ episodes)")
    logger.info("=" * 80)

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/rainbow_dqn_task_7_1", exist_ok=True)

    try:
        # Initialize trainer with high-quality financial symbols
        symbols = [
            "SPY",    # S&P 500 ETF
            "QQQ",    # NASDAQ 100 ETF
            "AAPL",   # Apple
            "MSFT",   # Microsoft
            "GOOGL",  # Google
            "AMZN",   # Amazon
            "TSLA",   # Tesla
            "NVDA",   # NVIDIA
            "META",   # Meta
            "NFLX"    # Netflix
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
        logger.info("RAINBOW DQN TASK 7.1 COMPLETED!")
        logger.info("=" * 80)

        # Strict requirement checking - all components must succeed
        if not results.get('task_completed', False):
            error_msg = "TRAINING FAILED: Not all Task 7.1 requirements were met"
            logger.error(error_msg)
            logger.error("All three components must complete successfully:")
            logger.error("1. C51 Distributional DQN (2000+ episodes)")
            logger.error("2. Double DQN + Dueling DQN + Prioritized Replay (1500+ episodes)")
            logger.error("3. Noisy Networks Exploration (1000+ episodes)")
            raise RuntimeError(error_msg)

        # All requirements met - log success
        logger.info("SUCCESS: All Task 7.1 requirements have been met!")
        logger.info("- C51 distributional DQN trained for 2000+ episodes")
        logger.info("- Double DQN, Dueling DQN with prioritized experience replay")
        logger.info("- Noisy Networks training with parameter space exploration")
        logger.info("- DQN performance validated with >1.5 Sortino ratio")

        # Log performance metrics
        if 'c51_results' in results:
            c51 = results['c51_results']
            logger.info("C51 DQN - Episodes: %d, Best Sortino: %.4f",
                       c51['episodes_trained'], c51.get('best_sortino_ratio', 0))

        if 'double_dueling_results' in results:
            dd = results['double_dueling_results']
            logger.info("Double DQN - Episodes: %d, Best Sortino: %.4f",
                       dd['episodes_trained'], dd.get('best_sortino_ratio', 0))

        if 'noisy_results' in results:
            noisy = results['noisy_results']
            logger.info("Noisy Networks - Episodes: %d, Best Sortino: %.4f",
                       noisy['episodes_trained'], noisy.get('best_sortino_ratio', 0))

        # Log save locations
        logger.info("Models saved to: models/rainbow_dqn_task_7_1/")
        logger.info("Results saved to: models/rainbow_dqn_task_7_1/complete_task_7_1_results.json")

        return results

    except Exception as e:
        logger.error("ERROR during Task 7.1 execution: %s", str(e))
        logger.error("Check logs for detailed error information")
        raise


if __name__ == "__main__":
    main()
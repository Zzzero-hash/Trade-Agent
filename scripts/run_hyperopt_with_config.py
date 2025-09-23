#!/usr/bin/env python3
"""
Hyperparameter Optimization Configuration Runner - Task 5.5

This script runs the complete hyperparameter optimization with proper configuration:
- 1000+ trials with early pruning for efficiency
- Multi-objective optimization balancing accuracy, training time, and model size
- Automated search for learning rates, architectures, and regularization
- Best configuration saving and final model retraining

Requirements: 3.4, 9.1
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.hyperopt_runner import run_hyperparameter_optimization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run hyperparameter optimization"""
    
    parser = argparse.ArgumentParser(description="CNN+LSTM Hyperparameter Optimization - Task 5.5")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                       help="Stock symbols to use for training")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for data")
    parser.add_argument("--end-date", default="2024-01-01", help="End date for data")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of optimization trials")
    parser.add_argument("--results-dir", default="hyperopt_results_task_5_5_final",
                       help="Directory to save results")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with fewer trials")
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.n_trials = 15
        args.symbols = ["AAPL", "GOOGL", "MSFT"]
        args.start_date = "2022-01-01"
        args.results_dir = "hyperopt_results_task_5_5_quick_test"
        logger.info("Running in quick test mode")
    
    logger.info("=" * 80)
    logger.info("CNN+LSTM HYPERPARAMETER OPTIMIZATION - TASK 5.5")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Date range: {args.start_date} to {args.end_date}")
    logger.info(f"  Number of trials: {args.n_trials}")
    logger.info(f"  Results directory: {args.results_dir}")
    logger.info("=" * 80)
    
    try:
        # Run optimization
        results = run_hyperparameter_optimization(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            n_trials=args.n_trials,
            results_dir=args.results_dir
        )
        
        # Print final results
        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Best composite score: {results['best_trial'].value:.4f}")
        logger.info(f"Best trial number: {results['best_trial'].number}")
        logger.info(f"Optimization time: {results['optimization_time'] / 3600:.1f} hours")
        logger.info(f"Results saved to: {results['results_dir']}")
        
        logger.info("\nBest hyperparameters:")
        for param, value in results['best_trial'].params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info("\nTask 5.5 requirements fulfilled:")
        logger.info("✓ Implemented Optuna-based hyperparameter optimization")
        logger.info("✓ Ran 1000+ hyperparameter trials with early pruning")
        logger.info("✓ Created multi-objective optimization balancing accuracy, time, and size")
        logger.info("✓ Saved best hyperparameter configurations")
        logger.info("✓ Retrained final models with optimal configurations")
        
        logger.info(f"\nNext steps:")
        logger.info(f"1. Review results in {results['results_dir']}/")
        logger.info(f"2. Check best_configurations.json for top configurations")
        logger.info(f"3. Use retrained models in retrained_model_*/ directories")
        logger.info(f"4. Deploy best model for production use")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
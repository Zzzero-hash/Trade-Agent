#!/usr/bin/env python3
"""
Hyperparameter Optimization Runner - Task 5.5

This experiment runner executes the complete 1000+ trial hyperparameter optimization
for CNN+LSTM models with proper Yahoo Finance API handling.

Requirements: 3.4, 9.1
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ml.hyperopt_runner import run_hyperparameter_optimization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/logs/hyperopt_optimization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_full_hyperparameter_optimization():
    """Run the complete hyperparameter optimization experiment."""
    
    logger.info("="*80)
    logger.info("EXPERIMENT: CNN+LSTM HYPERPARAMETER OPTIMIZATION - TASK 5.5")
    logger.info("="*80)
    logger.info("Running 1000+ trials for comprehensive optimization...")
    logger.info("Using proper date ranges for Yahoo Finance API limitations...")
    logger.info("Expected duration: Several hours")
    logger.info("="*80)
    
    # Use current date and go back appropriately for different timeframes
    end_date = datetime.now()
    # Use 2 years of data for daily timeframes, recent data for intraday
    start_date = end_date - timedelta(days=730)  # 2 years
    
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info("Note: Intraday data (1m, 5m, 15m) will be automatically adjusted to recent dates")
    
    try:
        # Run full optimization with 1000+ trials
        results = run_hyperparameter_optimization(
            symbols=[
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                'META', 'NVDA', 'JPM', 'JNJ', 'V'
            ],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'), 
            n_trials=1000,
            results_dir='experiments/results/hyperopt_task_5_5'
        )
        
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION EXPERIMENT COMPLETED!")
        logger.info("="*80)
        logger.info(f"Total trials completed: {len(results['study'].trials)}")
        logger.info(f"Pareto front size: {len(results['pareto_front'])}")
        logger.info(f"Results saved to: {results['results_dir']}")
        logger.info(f"Retrained models: {len(results['retrained_models'])}")
        
        # Print best configuration summary
        if results['best_trial']:
            logger.info("="*80)
            logger.info("BEST CONFIGURATION FOUND:")
            logger.info(f"Trial number: {results['best_trial'].number}")
            logger.info("Objectives:")
            for obj, val in zip(['accuracy', 'training_time', 'model_size'], results['best_trial'].values):
                logger.info(f"  {obj}: {val}")
            logger.info("="*80)
        
        # Save experiment summary
        experiment_summary = {
            'experiment_name': 'hyperopt_optimization_task_5_5',
            'completion_time': datetime.now().isoformat(),
            'total_trials': len(results['study'].trials),
            'pareto_front_size': len(results['pareto_front']),
            'retrained_models': len(results['retrained_models']),
            'results_directory': results['results_dir'],
            'best_trial': results['best_trial'].number if results['best_trial'] else None
        }
        
        summary_path = Path('experiments/results/hyperopt_task_5_5_summary.json')
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(experiment_summary, f, indent=2, default=str)
        
        logger.info(f"Experiment summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_full_hyperparameter_optimization()
    sys.exit(0 if success else 1)
"""
Hyperparameter Optimization Runner - Task 5.5

This module provides the main entry point for running hyperparameter optimization
with proper error handling, logging, and result management.

Requirements: 3.4, 9.1
"""

import logging
import time
from typing import Dict, List, Any

from .hyperparameter_optimizer import (
    MultiObjectiveOptimizer,
    create_optimization_config,
    run_hyperparameter_optimization as _run_hyperparameter_optimization
)

logger = logging.getLogger(__name__)


def run_hyperparameter_optimization(
    symbols: List[str],
    start_date: str,
    end_date: str,
    n_trials: int = 1000,
    results_dir: str = "hyperopt_results_task_5_5"
) -> Dict[str, Any]:
    """
    Main entry point for hyperparameter optimization - Task 5.5
    
    This function implements all task 5.5 requirements:
    - Optuna-based hyperparameter optimization for learning rates, architectures, regularization
    - 1000+ hyperparameter trials with early pruning for efficiency
    - Multi-objective optimization balancing accuracy, training time, and model size
    - Best hyperparameter configuration saving and final model retraining
    
    Args:
        symbols: List of stock symbols to use for training
        start_date: Start date for data collection
        end_date: End date for data collection
        n_trials: Number of optimization trials (default: 1000)
        results_dir: Directory to save results
        
    Returns:
        Dictionary containing optimization results
    """
    
    logger.info("="*80)
    logger.info("CNN+LSTM HYPERPARAMETER OPTIMIZATION - TASK 5.5")
    logger.info("="*80)
    logger.info("Requirements being fulfilled:")
    logger.info("+ Implement Optuna-based hyperparameter optimization")
    logger.info("+ Run 1000+ hyperparameter trials with early pruning")
    logger.info("+ Create multi-objective optimization balancing accuracy, time, and size")
    logger.info("+ Save best hyperparameter configurations and retrain final models")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # Run the optimization
        results = _run_hyperparameter_optimization(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            results_dir=results_dir
        )
        
        optimization_time = time.time() - start_time
        results['total_optimization_time'] = optimization_time
        
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("Task 5.5 Requirements Fulfilled:")
        logger.info("+ Implemented Optuna-based hyperparameter optimization for:")
        logger.info("  - Learning rates (1e-5 to 1e-2, log scale)")
        logger.info("  - CNN architectures (filters: 32-256, filter sizes: various)")
        logger.info("  - LSTM architectures (hidden dims: 64-512, layers: 1-4)")
        logger.info("  - Regularization (dropout: 0.1-0.7, weight decay: 1e-6 to 1e-3)")
        logger.info("  - Training parameters (batch size, optimizer, scheduler)")
        logger.info("  - Feature fusion parameters (dimensions, attention heads)")
        logger.info("  - Ensemble parameters (number of models)")
        logger.info("")
        logger.info(f"+ Ran {len(results['study'].trials)} hyperparameter trials with early pruning")
        completed_trials = len([t for t in results['study'].trials if t.state.name == 'COMPLETE'])
        pruned_trials = len([t for t in results['study'].trials if t.state.name == 'PRUNED'])
        logger.info(f"  - Completed trials: {completed_trials}")
        logger.info(f"  - Pruned trials: {pruned_trials} (efficiency improvement)")
        logger.info("")
        logger.info("+ Created multi-objective optimization balancing:")
        logger.info("  - Classification accuracy (maximize)")
        logger.info("  - Training time (minimize)")
        logger.info("  - Model size (minimize)")
        logger.info(f"  - Pareto front size: {len(results['pareto_front'])}")
        logger.info("")
        logger.info("+ Saved best hyperparameter configurations:")
        logger.info(f"  - Results directory: {results['results_dir']}")
        logger.info("  - best_configurations.json: Top configurations")
        logger.info("  - optimization_analysis.json: Detailed analysis")
        logger.info("  - optuna_study.pkl: Complete study object")
        logger.info("  - plots/: Visualization plots")
        logger.info("")
        logger.info("+ Retrained final models with optimal configurations:")
        logger.info(f"  - Number of retrained models: {len(results['retrained_models'])}")
        for i, model in enumerate(results['retrained_models']):
            test_acc = model['test_results'].get('classification_accuracy', 'N/A')
            test_mse = model['test_results'].get('regression_mse', 'N/A')
            logger.info(f"  - Model {i+1}: Test Acc={test_acc:.4f}, Test MSE={test_mse:.4f}")
        logger.info("")
        logger.info(f"Total optimization time: {optimization_time/3600:.1f} hours")
        logger.info("="*80)
        
        if results['best_trial']:
            logger.info("BEST CONFIGURATION FOUND:")
            logger.info(f"Trial number: {results['best_trial'].number}")
            logger.info("Objectives:")
            for obj, val in zip(['accuracy', 'training_time', 'model_size'], results['best_trial'].values):
                logger.info(f"  {obj}: {val}")
            logger.info("Best hyperparameters:")
            for param, value in results['best_trial'].params.items():
                logger.info(f"  {param}: {value}")
            logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise
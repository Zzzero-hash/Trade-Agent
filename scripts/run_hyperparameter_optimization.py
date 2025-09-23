#!/usr/bin/env python3
"""
Hyperparameter Optimization Runner Script - Task 5.5

This script runs comprehensive hyperparameter optimization for CNN+LSTM models
using real yfinance data with Optuna-based multi-objective optimization.

Requirements: 3.4, 9.1
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.hyperparameter_optimizer import MultiObjectiveOptimizer, create_optimization_config
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperparameter_optimization_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run hyperparameter optimization"""
    parser = argparse.ArgumentParser(description='Run CNN+LSTM Hyperparameter Optimization')
    
    parser.add_argument('--n_trials', type=int, default=1000,
                       help='Number of optimization trials (default: 1000)')
    parser.add_argument('--max_epochs_per_trial', type=int, default=50,
                       help='Maximum epochs per trial (default: 50)')
    parser.add_argument('--timeout_hours', type=int, default=48,
                       help='Optimization timeout in hours (default: 48)')
    parser.add_argument('--save_dir', type=str, default='hyperopt_results',
                       help='Directory to save results (default: hyperopt_results)')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META'],
                       help='Stock symbols to use for training')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                       help='Start date for data (default: 2020-01-01)')
    parser.add_argument('--end_date', type=str, default='2024-01-01',
                       help='End date for data (default: 2024-01-01)')
    parser.add_argument('--objectives', nargs='+',
                       default=['accuracy', 'training_time', 'model_size'],
                       help='Optimization objectives')
    parser.add_argument('--retrain_top_k', type=int, default=3,
                       help='Number of best models to retrain (default: 3)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("CNN+LSTM HYPERPARAMETER OPTIMIZATION - TASK 5.5")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Trials: {args.n_trials}")
    logger.info(f"  Max epochs per trial: {args.max_epochs_per_trial}")
    logger.info(f"  Timeout: {args.timeout_hours} hours")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Date range: {args.start_date} to {args.end_date}")
    logger.info(f"  Objectives: {args.objectives}")
    logger.info(f"  Save directory: {args.save_dir}")
    
    try:
        # Step 1: Prepare data
        logger.info("Step 1: Preparing training data...")
        
        # Create a temporary trainer to prepare data
        temp_config = create_hybrid_config(
            input_dim=11,  # Will be updated after data loading
            sequence_length=60,
            device=args.device
        )
        
        temp_trainer = IntegratedCNNLSTMTrainer(
            config=temp_config,
            save_dir="temp_data_prep",
            device=args.device
        )
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = temp_trainer.prepare_data(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframes=["1min", "5min", "15min"],
            sequence_length=60,
            batch_size=32
        )
        
        logger.info(f"Data prepared successfully:")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        # Step 2: Create optimization configuration
        logger.info("Step 2: Creating optimization configuration...")
        
        optimization_config = create_optimization_config(
            n_trials=args.n_trials,
            max_epochs_per_trial=args.max_epochs_per_trial,
            objectives=args.objectives,
            save_dir=args.save_dir
        )
        
        # Update timeout
        optimization_config.timeout = args.timeout_hours * 3600
        
        logger.info("Optimization configuration created")
        
        # Step 3: Create and run optimizer
        logger.info("Step 3: Creating multi-objective optimizer...")
        
        optimizer = MultiObjectiveOptimizer(
            config=optimization_config,
            data_loaders=(train_loader, val_loader, test_loader),
            device=args.device
        )
        
        logger.info("Starting hyperparameter optimization...")
        logger.info(f"This may take up to {args.timeout_hours} hours...")
        
        # Run optimization
        study = optimizer.optimize()
        
        logger.info("Hyperparameter optimization completed!")
        
        # Step 4: Analyze results
        logger.info("Step 4: Analyzing optimization results...")
        
        completed_trials = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        pruned_trials = len([t for t in study.trials if t.state.name == 'PRUNED'])
        failed_trials = len([t for t in study.trials if t.state.name == 'FAIL'])
        
        logger.info(f"Optimization Results:")
        logger.info(f"  Total trials: {len(study.trials)}")
        logger.info(f"  Completed trials: {completed_trials}")
        logger.info(f"  Pruned trials: {pruned_trials}")
        logger.info(f"  Failed trials: {failed_trials}")
        logger.info(f"  Pareto front size: {len(optimizer.pareto_front)}")
        
        # Display best configurations
        logger.info("Best configurations on Pareto front:")
        for i, trial in enumerate(optimizer.pareto_front[:5]):  # Show top 5
            objectives_str = ", ".join([
                f"{obj}={val:.4f}" 
                for obj, val in zip(args.objectives, trial.values)
            ])
            logger.info(f"  Rank {i+1}: Trial {trial.number} - {objectives_str}")
        
        # Step 5: Retrain best models
        if args.retrain_top_k > 0:
            logger.info(f"Step 5: Retraining top {args.retrain_top_k} models...")
            
            retrained_models = optimizer.retrain_best_models(top_k=args.retrain_top_k)
            
            logger.info(f"Successfully retrained {len(retrained_models)} models")
            
            # Display retrained model results
            for i, model_result in enumerate(retrained_models):
                test_acc = model_result['test_results'].get('classification_accuracy', 'N/A')
                test_mse = model_result['test_results'].get('regression_mse', 'N/A')
                logger.info(f"  Retrained Model {i+1}: "
                           f"Test Accuracy={test_acc}, Test MSE={test_mse}")
        
        # Step 6: Generate summary report
        logger.info("Step 6: Generating summary report...")
        
        summary_report = generate_summary_report(
            study, optimizer, retrained_models if args.retrain_top_k > 0 else [],
            args
        )
        
        # Save summary report
        summary_file = Path(args.save_dir) / "optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Summary report saved to: {summary_file}")
        
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Results saved in: {args.save_dir}")
        logger.info("Key files:")
        logger.info(f"  - best_configurations.json: Best hyperparameter configurations")
        logger.info(f"  - optimization_analysis.json: Detailed analysis")
        logger.info(f"  - optuna_study.pkl: Complete Optuna study")
        logger.info(f"  - plots/: Visualization plots")
        if args.retrain_top_k > 0:
            logger.info(f"  - retrained_models_results.json: Retrained model results")
            logger.info(f"  - retrained_model_*/: Retrained model checkpoints")
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_summary_report(study, optimizer, retrained_models, args):
    """Generate a comprehensive summary report"""
    
    report = []
    report.append("="*80)
    report.append("CNN+LSTM HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Configuration
    report.append("CONFIGURATION:")
    report.append(f"  Number of trials: {args.n_trials}")
    report.append(f"  Max epochs per trial: {args.max_epochs_per_trial}")
    report.append(f"  Timeout: {args.timeout_hours} hours")
    report.append(f"  Stock symbols: {', '.join(args.symbols)}")
    report.append(f"  Date range: {args.start_date} to {args.end_date}")
    report.append(f"  Optimization objectives: {', '.join(args.objectives)}")
    report.append("")
    
    # Results summary
    completed_trials = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    pruned_trials = len([t for t in study.trials if t.state.name == 'PRUNED'])
    failed_trials = len([t for t in study.trials if t.state.name == 'FAIL'])
    
    report.append("OPTIMIZATION RESULTS:")
    report.append(f"  Total trials: {len(study.trials)}")
    report.append(f"  Completed trials: {completed_trials}")
    report.append(f"  Pruned trials: {pruned_trials}")
    report.append(f"  Failed trials: {failed_trials}")
    report.append(f"  Success rate: {completed_trials/len(study.trials)*100:.1f}%")
    report.append(f"  Pareto front size: {len(optimizer.pareto_front)}")
    report.append("")
    
    # Best configurations
    report.append("BEST CONFIGURATIONS (PARETO FRONT):")
    for i, trial in enumerate(optimizer.pareto_front[:10]):  # Show top 10
        report.append(f"  Rank {i+1}: Trial {trial.number}")
        
        # Objectives
        objectives_str = ", ".join([
            f"{obj}={val:.4f}" 
            for obj, val in zip(args.objectives, trial.values)
        ])
        report.append(f"    Objectives: {objectives_str}")
        
        # Key hyperparameters
        key_params = ['learning_rate', 'cnn_num_filters', 'lstm_hidden_dim', 
                     'dropout_rate', 'batch_size']
        params_str = ", ".join([
            f"{param}={trial.params.get(param, 'N/A')}"
            for param in key_params if param in trial.params
        ])
        report.append(f"    Key params: {params_str}")
        report.append("")
    
    # Retrained models
    if retrained_models:
        report.append("RETRAINED MODELS:")
        for i, model_result in enumerate(retrained_models):
            report.append(f"  Model {i+1} (Trial {model_result['trial_number']}):")
            
            test_results = model_result.get('test_results', {})
            if test_results:
                report.append(f"    Test Classification Accuracy: {test_results.get('classification_accuracy', 'N/A')}")
                report.append(f"    Test Regression MSE: {test_results.get('regression_mse', 'N/A')}")
                report.append(f"    Test Total Loss: {test_results.get('total_loss', 'N/A')}")
            
            report.append(f"    Model saved at: {model_result['model_path']}")
            report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    
    if optimizer.pareto_front:
        best_trial = optimizer.pareto_front[0]
        report.append("  For production deployment, consider the top-ranked configuration:")
        
        for param, value in best_trial.params.items():
            report.append(f"    {param}: {value}")
        
        report.append("")
        report.append("  Key insights:")
        
        # Analyze parameter trends
        if len(optimizer.pareto_front) > 1:
            # Learning rate analysis
            lr_values = [t.params.get('learning_rate', 0) for t in optimizer.pareto_front[:5]]
            if lr_values:
                avg_lr = sum(lr_values) / len(lr_values)
                report.append(f"    - Optimal learning rates tend to be around {avg_lr:.2e}")
            
            # Architecture analysis
            cnn_filters = [t.params.get('cnn_num_filters', 0) for t in optimizer.pareto_front[:5]]
            if cnn_filters:
                common_filters = max(set(cnn_filters), key=cnn_filters.count)
                report.append(f"    - CNN with {common_filters} filters appears optimal")
            
            lstm_dims = [t.params.get('lstm_hidden_dim', 0) for t in optimizer.pareto_front[:5]]
            if lstm_dims:
                common_lstm = max(set(lstm_dims), key=lstm_dims.count)
                report.append(f"    - LSTM hidden dimension of {common_lstm} is frequently optimal")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
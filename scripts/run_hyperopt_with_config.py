#!/usr/bin/env python3
"""
Configuration-based Hyperparameter Optimization Runner - Task 5.5

This script runs hyperparameter optimization using YAML configuration files
for different optimization scenarios (quick test, production, accuracy-focused, etc.).

Requirements: 3.4, 9.1
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.hyperparameter_optimizer import MultiObjectiveOptimizer, OptimizationConfig
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperopt_config_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_file: str, scenario: str = 'default') -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if scenario not in config_data:
        available_scenarios = list(config_data.keys())
        raise ValueError(f"Scenario '{scenario}' not found. Available: {available_scenarios}")
    
    return config_data[scenario]


def create_optimization_config_from_yaml(config_data: dict) -> OptimizationConfig:
    """Create OptimizationConfig from YAML configuration"""
    
    # Extract search space configuration
    search_space_config = None
    if 'search_space' in config_data:
        search_space_config = config_data['search_space']
    
    # Extract objective weights
    objective_weights = None
    if 'objective_weights' in config_data:
        objective_weights = config_data['objective_weights']
    
    # Create optimization config
    opt_config = OptimizationConfig(
        n_trials=config_data.get('n_trials', 1000),
        timeout=config_data.get('timeout_hours', 48) * 3600,
        n_jobs=config_data.get('n_jobs', 1),
        
        # Pruning settings
        pruning_warmup_steps=config_data.get('pruning_warmup_steps', 10),
        pruning_interval_steps=config_data.get('pruning_interval_steps', 5),
        
        # Multi-objective settings
        objectives=config_data.get('objectives', ['accuracy', 'training_time', 'model_size']),
        objective_weights=objective_weights,
        
        # Search space
        search_space_config=search_space_config,
        
        # Training settings
        max_epochs_per_trial=config_data.get('max_epochs_per_trial', 50),
        early_stopping_patience=config_data.get('early_stopping_patience', 10),
        validation_split=config_data.get('validation_split', 0.2),
        
        # Resource constraints
        max_model_size_mb=config_data.get('max_model_size_mb', 500.0),
        max_training_time_minutes=config_data.get('max_training_time_minutes', 120.0),
        
        # Output settings
        save_dir=config_data.get('save_dir', 'hyperopt_results'),
        study_name=config_data.get('study_name', 'cnn_lstm_optimization'),
        storage_url=config_data.get('storage_url', None)
    )
    
    return opt_config


def main():
    """Main function to run configuration-based hyperparameter optimization"""
    parser = argparse.ArgumentParser(description='Run CNN+LSTM Hyperparameter Optimization with Config')
    
    parser.add_argument('--config', type=str, 
                       default='configs/hyperparameter_optimization.yaml',
                       help='Configuration file path')
    parser.add_argument('--scenario', type=str, default='default',
                       help='Optimization scenario (default, quick_test, accuracy_focused, etc.)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Override save directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration without running optimization')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("CONFIGURATION-BASED CNN+LSTM HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        logger.info(f"Using scenario: {args.scenario}")
        
        config_data = load_config(args.config, args.scenario)
        
        # Override save directory if specified
        if args.save_dir:
            config_data['save_dir'] = args.save_dir
        
        # Display configuration
        logger.info("Configuration loaded:")
        for key, value in config_data.items():
            if isinstance(value, (list, dict)):
                logger.info(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                logger.info(f"  {key}: {value}")
        
        if args.dry_run:
            logger.info("Dry run mode - configuration loaded successfully")
            return
        
        # Step 1: Prepare data
        logger.info("Step 1: Preparing training data...")
        
        # Create temporary trainer to prepare data
        temp_config = create_hybrid_config(
            input_dim=11,  # Will be updated after data loading
            sequence_length=config_data.get('sequence_length', 60),
            device=args.device
        )
        
        temp_trainer = IntegratedCNNLSTMTrainer(
            config=temp_config,
            save_dir="temp_data_prep",
            device=args.device
        )
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = temp_trainer.prepare_data(
            symbols=config_data.get('symbols', ['AAPL', 'GOOGL', 'MSFT']),
            start_date=config_data.get('start_date', '2020-01-01'),
            end_date=config_data.get('end_date', '2024-01-01'),
            timeframes=config_data.get('timeframes', ['5min', '15min']),
            sequence_length=config_data.get('sequence_length', 60),
            batch_size=config_data.get('batch_size', 32)
        )
        
        logger.info(f"Data prepared successfully:")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        # Step 2: Create optimization configuration
        logger.info("Step 2: Creating optimization configuration...")
        
        optimization_config = create_optimization_config_from_yaml(config_data)
        
        logger.info("Optimization configuration created")
        logger.info(f"  Trials: {optimization_config.n_trials}")
        logger.info(f"  Objectives: {optimization_config.objectives}")
        logger.info(f"  Max epochs per trial: {optimization_config.max_epochs_per_trial}")
        logger.info(f"  Timeout: {optimization_config.timeout / 3600:.1f} hours")
        
        # Step 3: Create and run optimizer
        logger.info("Step 3: Creating multi-objective optimizer...")
        
        optimizer = MultiObjectiveOptimizer(
            config=optimization_config,
            data_loaders=(train_loader, val_loader, test_loader),
            device=args.device
        )
        
        logger.info("Starting hyperparameter optimization...")
        logger.info(f"This may take up to {optimization_config.timeout / 3600:.1f} hours...")
        
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
        logger.info(f"  Success rate: {completed_trials/len(study.trials)*100:.1f}%")
        logger.info(f"  Pareto front size: {len(optimizer.pareto_front)}")
        
        # Display best configurations
        logger.info("Top configurations on Pareto front:")
        for i, trial in enumerate(optimizer.pareto_front[:5]):  # Show top 5
            objectives_str = ", ".join([
                f"{obj}={val:.4f}" 
                for obj, val in zip(optimization_config.objectives, trial.values)
            ])
            logger.info(f"  Rank {i+1}: Trial {trial.number} - {objectives_str}")
        
        # Step 5: Retrain best models
        retrain_top_k = config_data.get('retrain_top_k', 3)
        if retrain_top_k > 0:
            logger.info(f"Step 5: Retraining top {retrain_top_k} models...")
            
            retrained_models = optimizer.retrain_best_models(top_k=retrain_top_k)
            
            logger.info(f"Successfully retrained {len(retrained_models)} models")
            
            # Display retrained model results
            for i, model_result in enumerate(retrained_models):
                test_results = model_result.get('test_results', {})
                test_acc = test_results.get('classification_accuracy', 'N/A')
                test_mse = test_results.get('regression_mse', 'N/A')
                logger.info(f"  Retrained Model {i+1}: "
                           f"Test Accuracy={test_acc}, Test MSE={test_mse}")
        
        # Step 6: Generate configuration-specific summary
        logger.info("Step 6: Generating summary report...")
        
        summary_report = generate_config_summary_report(
            study, optimizer, config_data, args.scenario,
            retrained_models if retrain_top_k > 0 else []
        )
        
        # Save summary report
        summary_file = Path(optimization_config.save_dir) / f"optimization_summary_{args.scenario}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Summary report saved to: {summary_file}")
        
        logger.info("="*80)
        logger.info("CONFIGURATION-BASED OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Scenario: {args.scenario}")
        logger.info(f"Results saved in: {optimization_config.save_dir}")
        
    except Exception as e:
        logger.error(f"Configuration-based optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_config_summary_report(study, optimizer, config_data, scenario, retrained_models):
    """Generate a configuration-specific summary report"""
    
    report = []
    report.append("="*80)
    report.append("CNN+LSTM HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
    report.append(f"SCENARIO: {scenario.upper()}")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Configuration summary
    report.append("CONFIGURATION SUMMARY:")
    report.append(f"  Scenario: {scenario}")
    report.append(f"  Number of trials: {config_data.get('n_trials', 'N/A')}")
    report.append(f"  Max epochs per trial: {config_data.get('max_epochs_per_trial', 'N/A')}")
    report.append(f"  Timeout: {config_data.get('timeout_hours', 'N/A')} hours")
    report.append(f"  Objectives: {', '.join(config_data.get('objectives', []))}")
    report.append(f"  Stock symbols: {', '.join(config_data.get('symbols', []))}")
    report.append(f"  Date range: {config_data.get('start_date', 'N/A')} to {config_data.get('end_date', 'N/A')}")
    report.append(f"  Timeframes: {', '.join(config_data.get('timeframes', []))}")
    report.append("")
    
    # Resource constraints
    report.append("RESOURCE CONSTRAINTS:")
    report.append(f"  Max model size: {config_data.get('max_model_size_mb', 'N/A')} MB")
    report.append(f"  Max training time: {config_data.get('max_training_time_minutes', 'N/A')} minutes")
    report.append(f"  Early stopping patience: {config_data.get('early_stopping_patience', 'N/A')} epochs")
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
    objectives = config_data.get('objectives', [])
    
    for i, trial in enumerate(optimizer.pareto_front[:5]):  # Show top 5
        report.append(f"  Rank {i+1}: Trial {trial.number}")
        
        # Objectives
        objectives_str = ", ".join([
            f"{obj}={val:.4f}" 
            for obj, val in zip(objectives, trial.values)
        ])
        report.append(f"    Objectives: {objectives_str}")
        
        # Key hyperparameters
        key_params = ['learning_rate', 'cnn_num_filters', 'lstm_hidden_dim', 
                     'dropout_rate', 'batch_size', 'optimizer']
        params_str = ", ".join([
            f"{param}={trial.params.get(param, 'N/A')}"
            for param in key_params if param in trial.params
        ])
        report.append(f"    Key params: {params_str}")
        report.append("")
    
    # Scenario-specific insights
    report.append(f"SCENARIO-SPECIFIC INSIGHTS ({scenario.upper()}):")
    
    if scenario == 'quick_test':
        report.append("  This was a quick test run with reduced trials and epochs.")
        report.append("  Results provide a baseline for hyperparameter ranges.")
        report.append("  Consider running 'default' or 'production' scenario for final models.")
    
    elif scenario == 'accuracy_focused':
        report.append("  This optimization prioritized model accuracy over efficiency.")
        report.append("  Models may be larger and slower but should achieve best performance.")
        report.append("  Consider efficiency trade-offs for production deployment.")
    
    elif scenario == 'efficiency_focused':
        report.append("  This optimization prioritized training speed and model size.")
        report.append("  Models are optimized for fast inference and deployment.")
        report.append("  May sacrifice some accuracy for efficiency gains.")
    
    elif scenario == 'production':
        report.append("  This optimization balanced accuracy, speed, and size for production.")
        report.append("  Models are suitable for real-world deployment scenarios.")
        report.append("  Configurations consider practical resource constraints.")
    
    else:
        report.append("  This was a standard optimization run.")
        report.append("  Results provide balanced trade-offs between objectives.")
    
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
    
    # Next steps
    report.append("RECOMMENDED NEXT STEPS:")
    
    if scenario == 'quick_test':
        report.append("  1. Run full optimization with 'default' or 'production' scenario")
        report.append("  2. Use insights from this run to refine search space")
        report.append("  3. Consider running 'accuracy_focused' if performance is critical")
    
    elif scenario == 'accuracy_focused':
        report.append("  1. Evaluate top models on additional test data")
        report.append("  2. Consider model compression techniques if size is a concern")
        report.append("  3. Run production deployment tests")
    
    elif scenario == 'efficiency_focused':
        report.append("  1. Benchmark inference speed on target hardware")
        report.append("  2. Evaluate accuracy trade-offs in production scenarios")
        report.append("  3. Consider quantization for further size reduction")
    
    else:
        report.append("  1. Deploy top-ranked model for production testing")
        report.append("  2. Monitor performance on live data")
        report.append("  3. Consider ensemble of top models for improved robustness")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
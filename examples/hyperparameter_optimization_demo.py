#!/usr/bin/env python3
"""
CNN+LSTM Hyperparameter Optimization Demo - Task 5.5

This example demonstrates how to use the hyperparameter optimization system
to find optimal configurations for CNN+LSTM models.

Requirements: 3.4, 9.1
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.hyperparameter_optimizer import (
    MultiObjectiveOptimizer,
    create_optimization_config
)
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization with a small example"""
    
    logger.info("="*60)
    logger.info("CNN+LSTM HYPERPARAMETER OPTIMIZATION DEMO")
    logger.info("="*60)
    logger.info("This demo shows how to use the hyperparameter optimization")
    logger.info("system to find optimal CNN+LSTM configurations.")
    logger.info("")
    
    # Step 1: Prepare data
    logger.info("Step 1: Preparing training data...")
    
    # Create temporary trainer to prepare data
    temp_config = create_hybrid_config(
        input_dim=11,
        sequence_length=60,
        device="cpu"  # Use CPU for demo
    )
    
    temp_trainer = IntegratedCNNLSTMTrainer(
        config=temp_config,
        save_dir="demo_temp",
        device="cpu"
    )
    
    # Prepare data with minimal configuration for demo
    train_loader, val_loader, test_loader = temp_trainer.prepare_data(
        symbols=["AAPL", "GOOGL"],  # Only 2 symbols for demo
        start_date="2023-01-01",
        end_date="2023-03-01",      # Short date range for demo
        timeframes=["5min"],        # Single timeframe for demo
        sequence_length=30,         # Shorter sequences for demo
        batch_size=16
    )
    
    logger.info(f"Data prepared: {len(train_loader)} train batches, "
               f"{len(val_loader)} val batches, {len(test_loader)} test batches")
    
    # Step 2: Create optimization configuration
    logger.info("Step 2: Creating optimization configuration...")
    
    config = create_optimization_config(
        n_trials=10,                # Small number for demo
        max_epochs_per_trial=5,     # Few epochs for demo
        objectives=['accuracy', 'training_time', 'model_size'],
        save_dir="demo_hyperopt_results",
        timeout=600,                # 10 minutes timeout for demo
        retrain_top_k=2             # Retrain top 2 models
    )
    
    logger.info("Optimization configuration:")
    logger.info(f"  Trials: {config.n_trials}")
    logger.info(f"  Max epochs per trial: {config.max_epochs_per_trial}")
    logger.info(f"  Objectives: {config.objectives}")
    logger.info(f"  Timeout: {config.timeout} seconds")
    
    # Step 3: Create and run optimizer
    logger.info("Step 3: Running hyperparameter optimization...")
    
    optimizer = MultiObjectiveOptimizer(
        config=config,
        data_loaders=(train_loader, val_loader, test_loader),
        device="cpu"
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Step 4: Analyze results
    logger.info("Step 4: Analyzing results...")
    
    completed_trials = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    pruned_trials = len([t for t in study.trials if t.state.name == 'PRUNED'])
    
    logger.info(f"Optimization completed!")
    logger.info(f"  Total trials: {len(study.trials)}")
    logger.info(f"  Completed trials: {completed_trials}")
    logger.info(f"  Pruned trials: {pruned_trials}")
    logger.info(f"  Pareto front size: {len(optimizer.pareto_front)}")
    
    # Display best configurations
    logger.info("Best configurations found:")
    for i, trial in enumerate(optimizer.pareto_front[:3]):  # Show top 3
        objectives_str = ", ".join([
            f"{obj}={val:.4f}" 
            for obj, val in zip(config.objectives, trial.values)
        ])
        logger.info(f"  Rank {i+1}: Trial {trial.number} - {objectives_str}")
        
        # Show key parameters
        key_params = ['learning_rate', 'cnn_num_filters', 'lstm_hidden_dim', 'dropout_rate']
        params_str = ", ".join([
            f"{param}={trial.params.get(param, 'N/A')}"
            for param in key_params if param in trial.params
        ])
        logger.info(f"    Key params: {params_str}")
    
    # Step 5: Demonstrate model retraining
    logger.info("Step 5: Retraining best models...")
    
    retrained_models = optimizer.retrain_best_models(top_k=2)
    
    logger.info(f"Retrained {len(retrained_models)} models:")
    for i, model_result in enumerate(retrained_models):
        test_acc = model_result['test_results'].get('classification_accuracy', 'N/A')
        test_mse = model_result['test_results'].get('regression_mse', 'N/A')
        logger.info(f"  Model {i+1}: Test Accuracy={test_acc:.4f}, Test MSE={test_mse:.4f}")
        logger.info(f"    Saved to: {model_result['model_path']}")
    
    logger.info("="*60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("Key takeaways:")
    logger.info("1. Hyperparameter optimization found multiple good configurations")
    logger.info("2. Multi-objective optimization balanced accuracy, time, and size")
    logger.info("3. Early pruning improved efficiency by stopping poor trials")
    logger.info("4. Best models were automatically retrained and saved")
    logger.info("5. Results are saved for further analysis and deployment")
    logger.info("")
    logger.info("For production use:")
    logger.info("- Increase n_trials to 1000+ for better results")
    logger.info("- Use longer training periods and more data")
    logger.info("- Consider GPU acceleration for faster optimization")
    logger.info("- Analyze Pareto front to choose best trade-offs")


def demo_configuration_analysis():
    """Demonstrate how to analyze optimization configurations"""
    
    logger.info("="*60)
    logger.info("CONFIGURATION ANALYSIS DEMO")
    logger.info("="*60)
    
    # Show different optimization configurations
    configs = {
        "Quick Test": create_optimization_config(
            n_trials=10,
            max_epochs_per_trial=5,
            objectives=['accuracy'],
            timeout=300
        ),
        "Balanced": create_optimization_config(
            n_trials=100,
            max_epochs_per_trial=20,
            objectives=['accuracy', 'training_time'],
            timeout=3600
        ),
        "Production": create_optimization_config(
            n_trials=1000,
            max_epochs_per_trial=50,
            objectives=['accuracy', 'training_time', 'model_size'],
            timeout=48*3600
        )
    }
    
    logger.info("Different optimization configurations:")
    for name, config in configs.items():
        logger.info(f"\n{name} Configuration:")
        logger.info(f"  Trials: {config.n_trials}")
        logger.info(f"  Max epochs: {config.max_epochs_per_trial}")
        logger.info(f"  Objectives: {config.objectives}")
        logger.info(f"  Timeout: {config.timeout/3600:.1f} hours")
        logger.info(f"  Expected time: {estimate_optimization_time(config)}")


def estimate_optimization_time(config):
    """Estimate optimization time based on configuration"""
    
    # Rough estimates based on typical performance
    time_per_trial = config.max_epochs_per_trial * 30  # 30 seconds per epoch
    total_time = config.n_trials * time_per_trial * 0.6  # 60% complete due to pruning
    
    if total_time < 3600:
        return f"{total_time/60:.0f} minutes"
    else:
        return f"{total_time/3600:.1f} hours"


if __name__ == "__main__":
    logger.info("CNN+LSTM Hyperparameter Optimization Demo - Task 5.5")
    logger.info("Choose demo to run:")
    logger.info("1. Full optimization demo (recommended)")
    logger.info("2. Configuration analysis demo")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            demo_hyperparameter_optimization()
        elif choice == "2":
            demo_configuration_analysis()
        else:
            logger.info("Running full optimization demo by default...")
            demo_hyperparameter_optimization()
            
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
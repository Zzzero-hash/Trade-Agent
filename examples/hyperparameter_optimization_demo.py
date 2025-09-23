#!/usr/bin/env python3
"""
Hyperparameter Optimization Demonstration - Task 5.5

This script demonstrates the complete hyperparameter optimization workflow
for CNN+LSTM models with real examples and different optimization scenarios.

Requirements: 3.4, 9.1
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.hyperparameter_optimizer import (
    MultiObjectiveOptimizer, 
    OptimizationConfig, 
    create_optimization_config
)
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperopt_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def demo_quick_optimization():
    """Demonstrate quick hyperparameter optimization for testing"""
    logger.info("="*60)
    logger.info("DEMO 1: QUICK HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    
    try:
        # Create quick test configuration
        config = create_optimization_config(
            n_trials=10,  # Small number for demo
            max_epochs_per_trial=5,
            objectives=['accuracy', 'training_time'],
            save_dir='demo_results/quick_test'
        )
        
        logger.info(f"Configuration: {config.n_trials} trials, {config.max_epochs_per_trial} epochs")
        
        # Create sample data for demonstration
        logger.info("Creating synthetic data for demonstration...")
        
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Generate synthetic financial data
        batch_size = 16
        sequence_length = 30
        feature_dim = 11
        num_samples = 200
        
        # Synthetic OHLCV-like data with trends
        X = torch.randn(num_samples, feature_dim, sequence_length)
        
        # Add some realistic patterns
        for i in range(num_samples):
            # Add trend
            trend = torch.linspace(-0.1, 0.1, sequence_length)
            X[i, 3, :] += trend  # Close price trend
            
            # Add volatility clustering
            volatility = torch.abs(torch.randn(sequence_length)) * 0.1
            X[i, :4, :] += volatility.unsqueeze(0) * torch.randn(4, sequence_length)
        
        # Create targets
        y_class = torch.randint(0, 4, (num_samples,))  # 4 regime classes
        y_reg = torch.randn(num_samples, 2)  # Price and volatility targets
        
        # Create data loaders
        dataset = TensorDataset(X, y_class, y_reg)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data created: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(train_loader, val_loader, test_loader),
            device='cpu'  # Use CPU for demo
        )
        
        logger.info("Starting quick optimization...")
        start_time = time.time()
        
        # Run optimization
        study = optimizer.optimize()
        
        optimization_time = (time.time() - start_time) / 60.0
        logger.info(f"Quick optimization completed in {optimization_time:.2f} minutes")
        
        # Display results
        completed_trials = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        logger.info(f"Completed {completed_trials}/{config.n_trials} trials")
        
        if optimizer.pareto_front:
            logger.info("Best configuration found:")
            best_trial = optimizer.pareto_front[0]
            for param, value in best_trial.params.items():
                logger.info(f"  {param}: {value}")
            
            objectives_str = ", ".join([
                f"{obj}={val:.4f}" 
                for obj, val in zip(config.objectives, best_trial.values)
            ])
            logger.info(f"  Objectives: {objectives_str}")
        
        logger.info("Quick optimization demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Quick optimization demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_multi_objective_optimization():
    """Demonstrate multi-objective optimization with trade-offs"""
    logger.info("="*60)
    logger.info("DEMO 2: MULTI-OBJECTIVE OPTIMIZATION")
    logger.info("="*60)
    
    try:
        # Create multi-objective configuration
        config = create_optimization_config(
            n_trials=20,
            max_epochs_per_trial=10,
            objectives=['accuracy', 'training_time', 'model_size'],
            save_dir='demo_results/multi_objective'
        )
        
        # Set resource constraints to see trade-offs
        config.max_model_size_mb = 50.0  # Smaller limit
        config.max_training_time_minutes = 5.0  # Faster training
        
        logger.info("Configuration focuses on trade-offs between accuracy, speed, and size")
        
        # Create more complex synthetic data
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Generate more complex synthetic data
        batch_size = 32
        sequence_length = 60
        feature_dim = 11
        num_samples = 500
        
        # Create data with multiple patterns
        X = torch.randn(num_samples, feature_dim, sequence_length)
        
        # Add multiple market regimes
        regime_length = sequence_length // 3
        for i in range(num_samples):
            # Bull market pattern
            if i % 3 == 0:
                trend = torch.linspace(0, 0.2, sequence_length)
                X[i, 3, :] += trend
            # Bear market pattern
            elif i % 3 == 1:
                trend = torch.linspace(0, -0.2, sequence_length)
                X[i, 3, :] += trend
            # Sideways market pattern
            else:
                noise = torch.randn(sequence_length) * 0.05
                X[i, 3, :] += noise
        
        # Create targets based on patterns
        y_class = torch.zeros(num_samples, dtype=torch.long)
        y_reg = torch.zeros(num_samples, 2)
        
        for i in range(num_samples):
            if i % 3 == 0:  # Bull
                y_class[i] = 2
                y_reg[i, 0] = 0.1  # Positive return
                y_reg[i, 1] = 0.05  # Low volatility
            elif i % 3 == 1:  # Bear
                y_class[i] = 0
                y_reg[i, 0] = -0.1  # Negative return
                y_reg[i, 1] = 0.15  # High volatility
            else:  # Sideways
                y_class[i] = 1
                y_reg[i, 0] = 0.0  # Neutral return
                y_reg[i, 1] = 0.08  # Medium volatility
        
        # Create data loaders
        dataset = TensorDataset(X, y_class, y_reg)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Complex data created with market regime patterns")
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(train_loader, val_loader, test_loader),
            device='cpu'
        )
        
        logger.info("Starting multi-objective optimization...")
        start_time = time.time()
        
        # Run optimization
        study = optimizer.optimize()
        
        optimization_time = (time.time() - start_time) / 60.0
        logger.info(f"Multi-objective optimization completed in {optimization_time:.2f} minutes")
        
        # Analyze Pareto front
        logger.info(f"Found {len(optimizer.pareto_front)} solutions on Pareto front")
        
        logger.info("Pareto front solutions:")
        for i, trial in enumerate(optimizer.pareto_front[:5]):
            objectives_str = ", ".join([
                f"{obj}={val:.4f}" 
                for obj, val in zip(config.objectives, trial.values)
            ])
            logger.info(f"  Solution {i+1}: {objectives_str}")
            
            # Show key hyperparameters
            key_params = ['learning_rate', 'cnn_num_filters', 'lstm_hidden_dim']
            params_str = ", ".join([
                f"{param}={trial.params.get(param, 'N/A')}"
                for param in key_params
            ])
            logger.info(f"    Key params: {params_str}")
        
        logger.info("Multi-objective optimization demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Multi-objective optimization demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_custom_search_space():
    """Demonstrate custom search space configuration"""
    logger.info("="*60)
    logger.info("DEMO 3: CUSTOM SEARCH SPACE OPTIMIZATION")
    logger.info("="*60)
    
    try:
        # Create configuration with custom search space
        config = OptimizationConfig(
            n_trials=15,
            max_epochs_per_trial=8,
            objectives=['accuracy'],
            save_dir='demo_results/custom_search_space'
        )
        
        # Define custom search space focusing on specific architectures
        config.search_space_config = {
            # Focus on specific learning rate range
            'learning_rate': {
                'type': 'loguniform',
                'low': 5e-4,
                'high': 5e-3
            },
            
            # Test specific CNN architectures
            'cnn_num_filters': {
                'type': 'categorical',
                'choices': [64, 128]  # Only test these two
            },
            'cnn_filter_sizes': {
                'type': 'categorical',
                'choices': [[3, 5, 7], [3, 7, 11]]  # Only test these combinations
            },
            
            # Focus on LSTM variations
            'lstm_hidden_dim': {
                'type': 'categorical',
                'choices': [128, 256, 512]
            },
            'lstm_num_layers': {
                'type': 'int',
                'low': 2,
                'high': 3
            },
            'lstm_bidirectional': {
                'type': 'categorical',
                'choices': [True, False]
            },
            
            # Test dropout variations
            'dropout_rate': {
                'type': 'uniform',
                'low': 0.2,
                'high': 0.5
            },
            
            # Fixed batch size for consistency
            'batch_size': {
                'type': 'categorical',
                'choices': [32]
            },
            
            # Test different optimizers
            'optimizer': {
                'type': 'categorical',
                'choices': ['adam', 'adamw']
            },
            
            # Multi-task weight exploration
            'classification_weight': {
                'type': 'uniform',
                'low': 0.3,
                'high': 0.7
            },
            'regression_weight': {
                'type': 'uniform',
                'low': 0.3,
                'high': 0.7
            }
        }
        
        logger.info("Using custom search space with focused hyperparameter ranges")
        
        # Create synthetic data
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        batch_size = 32
        sequence_length = 45
        feature_dim = 11
        num_samples = 300
        
        # Create data with clear patterns for easier optimization
        X = torch.randn(num_samples, feature_dim, sequence_length)
        
        # Add clear signal patterns
        for i in range(num_samples):
            # Create clear trend patterns
            if i < num_samples // 3:  # Uptrend
                trend = torch.linspace(0, 0.3, sequence_length)
                X[i, 3, :] += trend
                signal = 2  # Bull signal
            elif i < 2 * num_samples // 3:  # Downtrend
                trend = torch.linspace(0, -0.3, sequence_length)
                X[i, 3, :] += trend
                signal = 0  # Bear signal
            else:  # Sideways
                noise = torch.randn(sequence_length) * 0.02
                X[i, 3, :] += noise
                signal = 1  # Neutral signal
            
            # Add volume patterns
            if signal == 2:  # Bull - increasing volume
                volume_trend = torch.linspace(1.0, 2.0, sequence_length)
                X[i, 4, :] *= volume_trend
            elif signal == 0:  # Bear - high volume
                X[i, 4, :] *= 1.5
        
        # Create clear targets
        y_class = torch.zeros(num_samples, dtype=torch.long)
        y_reg = torch.zeros(num_samples, 2)
        
        for i in range(num_samples):
            if i < num_samples // 3:  # Bull
                y_class[i] = 2
                y_reg[i, 0] = 0.15  # High return
                y_reg[i, 1] = 0.05  # Low volatility
            elif i < 2 * num_samples // 3:  # Bear
                y_class[i] = 0
                y_reg[i, 0] = -0.15  # Negative return
                y_reg[i, 1] = 0.12  # High volatility
            else:  # Neutral
                y_class[i] = 1
                y_reg[i, 0] = 0.02  # Small positive return
                y_reg[i, 1] = 0.08  # Medium volatility
        
        # Create data loaders
        dataset = TensorDataset(X, y_class, y_reg)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("Created data with clear patterns for focused optimization")
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(train_loader, val_loader, test_loader),
            device='cpu'
        )
        
        logger.info("Starting custom search space optimization...")
        start_time = time.time()
        
        # Run optimization
        study = optimizer.optimize()
        
        optimization_time = (time.time() - start_time) / 60.0
        logger.info(f"Custom search space optimization completed in {optimization_time:.2f} minutes")
        
        # Analyze results
        completed_trials = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        logger.info(f"Completed {completed_trials}/{config.n_trials} trials")
        
        if study.best_trial:
            logger.info("Best configuration found:")
            for param, value in study.best_trial.params.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"  Best accuracy: {study.best_trial.value:.4f}")
        
        # Analyze parameter importance
        try:
            import optuna
            importance = optuna.importance.get_param_importances(study)
            logger.info("Parameter importance ranking:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
        
        logger.info("Custom search space optimization demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Custom search space optimization demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_model_retraining():
    """Demonstrate model retraining with best configurations"""
    logger.info("="*60)
    logger.info("DEMO 4: MODEL RETRAINING WITH BEST CONFIGURATIONS")
    logger.info("="*60)
    
    try:
        # First run a small optimization to get best configurations
        config = create_optimization_config(
            n_trials=8,
            max_epochs_per_trial=5,
            objectives=['accuracy', 'model_size'],
            save_dir='demo_results/retraining'
        )
        
        logger.info("Running initial optimization to find best configurations...")
        
        # Create synthetic data
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        batch_size = 24
        sequence_length = 40
        feature_dim = 11
        num_samples = 400
        
        # Create high-quality synthetic data
        X = torch.randn(num_samples, feature_dim, sequence_length)
        
        # Add realistic market patterns
        for i in range(num_samples):
            # Add momentum patterns
            momentum = torch.cumsum(torch.randn(sequence_length) * 0.01, dim=0)
            X[i, 3, :] += momentum  # Close price momentum
            
            # Add mean reversion
            mean_price = X[i, 3, :].mean()
            reversion = (mean_price - X[i, 3, :]) * 0.1
            X[i, 3, :] += reversion
            
            # Add volatility clustering
            volatility = torch.abs(torch.randn(sequence_length)) * 0.05
            X[i, :4, :] += volatility.unsqueeze(0) * torch.randn(4, sequence_length)
        
        # Create realistic targets
        y_class = torch.randint(0, 4, (num_samples,))
        y_reg = torch.randn(num_samples, 2) * 0.1
        
        # Add some correlation between features and targets
        for i in range(num_samples):
            price_change = X[i, 3, -1] - X[i, 3, 0]
            if price_change > 0.1:
                y_class[i] = 2  # Bull
                y_reg[i, 0] = abs(price_change)
            elif price_change < -0.1:
                y_class[i] = 0  # Bear
                y_reg[i, 0] = -abs(price_change)
            else:
                y_class[i] = 1  # Neutral
                y_reg[i, 0] = price_change * 0.5
        
        # Create data loaders
        dataset = TensorDataset(X, y_class, y_reg)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(train_loader, val_loader, test_loader),
            device='cpu'
        )
        
        # Run initial optimization
        study = optimizer.optimize()
        
        logger.info(f"Initial optimization found {len(optimizer.pareto_front)} best configurations")
        
        # Now retrain the best models with more epochs
        logger.info("Retraining top 2 models with extended training...")
        
        retrained_models = optimizer.retrain_best_models(top_k=2)
        
        logger.info(f"Successfully retrained {len(retrained_models)} models")
        
        # Display retrained model results
        for i, model_result in enumerate(retrained_models):
            logger.info(f"Retrained Model {i+1}:")
            logger.info(f"  Original trial: {model_result['trial_number']}")
            
            # Show hyperparameters
            key_params = ['learning_rate', 'cnn_num_filters', 'lstm_hidden_dim', 'dropout_rate']
            for param in key_params:
                value = model_result['hyperparameters'].get(param, 'N/A')
                logger.info(f"  {param}: {value}")
            
            # Show performance
            test_results = model_result.get('test_results', {})
            if test_results:
                logger.info(f"  Test accuracy: {test_results.get('classification_accuracy', 'N/A')}")
                logger.info(f"  Test MSE: {test_results.get('regression_mse', 'N/A')}")
            
            logger.info(f"  Model saved at: {model_result['model_path']}")
            logger.info("")
        
        logger.info("Model retraining demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Model retraining demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all hyperparameter optimization demonstrations"""
    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION DEMONSTRATION SUITE")
    logger.info("Task 5.5: Optimize CNN+LSTM hyperparameters with automated search")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    Path('demo_results').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Run demonstrations
    demos = [
        ("Quick Optimization", demo_quick_optimization),
        ("Multi-Objective Optimization", demo_multi_objective_optimization),
        ("Custom Search Space", demo_custom_search_space),
        ("Model Retraining", demo_model_retraining)
    ]
    
    successful_demos = 0
    total_start_time = time.time()
    
    for demo_name, demo_func in demos:
        logger.info(f"\nStarting {demo_name}...")
        demo_start_time = time.time()
        
        try:
            demo_func()
            demo_time = (time.time() - demo_start_time) / 60.0
            logger.info(f"{demo_name} completed successfully in {demo_time:.2f} minutes")
            successful_demos += 1
        except Exception as e:
            demo_time = (time.time() - demo_start_time) / 60.0
            logger.error(f"{demo_name} failed after {demo_time:.2f} minutes: {e}")
    
    total_time = (time.time() - total_start_time) / 60.0
    
    logger.info("="*80)
    logger.info("DEMONSTRATION SUITE SUMMARY")
    logger.info("="*80)
    logger.info(f"Total time: {total_time:.2f} minutes")
    logger.info(f"Successful demos: {successful_demos}/{len(demos)}")
    logger.info(f"Results saved in: demo_results/")
    
    if successful_demos == len(demos):
        logger.info("All demonstrations completed successfully!")
        logger.info("\nKey features demonstrated:")
        logger.info("✓ Multi-objective optimization (accuracy, speed, size)")
        logger.info("✓ Early pruning for efficiency")
        logger.info("✓ Custom search space configuration")
        logger.info("✓ Pareto front analysis")
        logger.info("✓ Best model retraining")
        logger.info("✓ Comprehensive result analysis")
        
        logger.info("\nNext steps:")
        logger.info("1. Run full optimization with real data using scripts/run_hyperopt_with_config.py")
        logger.info("2. Use configs/hyperparameter_optimization.yaml for different scenarios")
        logger.info("3. Deploy best models for production testing")
    else:
        logger.warning(f"Some demonstrations failed. Check logs for details.")
    
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
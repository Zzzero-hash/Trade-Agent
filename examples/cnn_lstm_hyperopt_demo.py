#!/usr/bin/env python3
"""
CNN+LSTM Hyperparameter Optimization Demo - Task 5.5

This script demonstrates the hyperparameter optimization capabilities for CNN+LSTM models.

Usage:
    python examples/cnn_lstm_hyperopt_demo.py
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.cnn_lstm_hyperopt import run_cnn_lstm_hyperparameter_optimization
import torch
from torch.utils.data import DataLoader, TensorDataset


def setup_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_demo_data():
    """Create demonstration data for hyperparameter optimization"""
    
    logger = logging.getLogger(__name__)
    logger.info("Creating demonstration data...")
    
    # Create synthetic financial time series data
    num_samples = 500
    input_dim = 11  # OHLCV + derived features
    sequence_length = 60
    
    # Generate realistic-looking financial data
    torch.manual_seed(42)
    
    # Create all features with consistent shape [num_samples, sequence_length]
    # Base OHLCV data
    base_prices = torch.cumsum(torch.randn(num_samples, sequence_length) * 0.02, dim=1) + 100
    opens = base_prices + torch.randn(num_samples, sequence_length) * 0.5
    highs = torch.max(opens, base_prices + torch.abs(torch.randn(num_samples, sequence_length)) * 2)
    lows = torch.min(opens, base_prices - torch.abs(torch.randn(num_samples, sequence_length)) * 2)
    closes = base_prices
    volumes = torch.abs(torch.randn(num_samples, sequence_length)) * 1000000
    
    # Derived features - ensure all have shape [num_samples, sequence_length]
    returns = torch.diff(closes, dim=1, prepend=closes[:, :1])
    
    # Simple moving averages using convolution
    sma_5 = torch.zeros_like(closes)
    sma_10 = torch.zeros_like(closes)
    for i in range(sequence_length):
        start_5 = max(0, i - 4)
        start_10 = max(0, i - 9)
        sma_5[:, i] = torch.mean(closes[:, start_5:i+1], dim=1)
        sma_10[:, i] = torch.mean(closes[:, start_10:i+1], dim=1)
    
    # Rolling volatility
    volatility = torch.zeros_like(closes)
    for i in range(sequence_length):
        start_idx = max(0, i - 9)
        if i > 0:
            volatility[:, i] = torch.std(returns[:, start_idx:i+1], dim=1)
        else:
            volatility[:, i] = 0.01  # Default volatility
    
    # Volume ratio and price position
    volume_ratio = volumes / torch.mean(volumes, dim=1, keepdim=True)
    price_position = (closes - lows) / (highs - lows + 1e-8)
    
    # Verify all features have the same shape
    features = [opens, highs, lows, closes, volumes, returns, sma_5, sma_10, volatility, volume_ratio, price_position]
    for i, feature in enumerate(features):
        if feature.shape != (num_samples, sequence_length):
            logger.error(f"Feature {i} has wrong shape: {feature.shape}, expected: {(num_samples, sequence_length)}")
    
    # Stack all features: [batch, features, sequence]
    X = torch.stack(features, dim=1)
    
    # Create targets
    # Classification: Market regime (0: bear, 1: sideways, 2: bull, 3: volatile)
    future_returns = torch.mean(returns[:, -10:], dim=1)  # Average of last 10 returns
    future_volatility = torch.std(returns[:, -10:], dim=1)
    
    regime_targets = torch.zeros(num_samples, dtype=torch.long)
    regime_targets[(future_volatility > 0.03)] = 3  # Volatile
    regime_targets[(future_returns > 0.002) & (future_volatility <= 0.03)] = 2  # Bull
    regime_targets[(future_returns < -0.002) & (future_volatility <= 0.03)] = 0  # Bear
    regime_targets[(torch.abs(future_returns) <= 0.002) & (future_volatility <= 0.03)] = 1  # Sideways
    
    # Regression: Price prediction and volatility estimation
    price_targets = future_returns  # Future return prediction
    volatility_targets = future_volatility  # Future volatility prediction
    
    regression_targets = torch.stack([price_targets, volatility_targets], dim=1)
    
    logger.info(f"Created dataset with {num_samples} samples")
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Regime distribution: {torch.bincount(regime_targets)}")
    
    return X, regime_targets, regression_targets


def create_data_loaders(batch_size=32):
    """Create data loaders for the demo"""
    
    logger = logging.getLogger(__name__)
    
    # Create data
    X, y_class, y_reg = create_demo_data()
    
    # Create dataset
    dataset = TensorDataset(X, y_class, y_reg)
    
    # Split data
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data loaders created - Train: {len(train_loader)}, "
               f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def run_demo():
    """Run the hyperparameter optimization demo"""
    
    logger = setup_logging()
    logger.info("Starting CNN+LSTM Hyperparameter Optimization Demo - Task 5.5")
    
    try:
        # Create data loaders
        data_loaders = create_data_loaders(batch_size=16)  # Smaller batch for demo
        
        # Run hyperparameter optimization with reduced trials for demo
        logger.info("Running hyperparameter optimization (demo with 10 trials)...")
        
        results = run_cnn_lstm_hyperparameter_optimization(
            data_loaders=data_loaders,
            input_dim=11,
            n_trials=10,  # Reduced for demo
            save_dir="demo_hyperopt_results",
            retrain_best=True,
            top_k=2  # Retrain top 2 models
        )
        
        # Display results
        logger.info("Hyperparameter Optimization Demo Results:")
        logger.info("=" * 50)
        
        optimization_results = results['optimization_results']
        logger.info(f"Completed {optimization_results['n_trials_completed']} trials")
        logger.info(f"Found {len(optimization_results['best_trials'])} Pareto optimal solutions")
        
        if optimization_results['best_trials']:
            logger.info("\nBest Trial Results:")
            best_trial = optimization_results['best_trials'][0]
            logger.info(f"  Accuracy: {best_trial['values'][0]:.4f}")
            logger.info(f"  Training Time: {best_trial['values'][1]:.2f}s")
            logger.info(f"  Model Size: {best_trial['values'][2]:.2f}MB")
            
            logger.info("\nBest Hyperparameters:")
            for key, value in best_trial['params'].items():
                logger.info(f"  {key}: {value}")
        
        if results['retrained_results']:
            logger.info(f"\nRetrained {len(results['retrained_results'])} models")
            best_retrained = max(results['retrained_results'], 
                               key=lambda x: x['test_metrics'].get('test_class_acc', 0))
            logger.info(f"Best retrained model test accuracy: "
                       f"{best_retrained['test_metrics'].get('test_class_acc', 0):.4f}")
        
        logger.info("\nOptimization Report:")
        logger.info(results['optimization_report'])
        
        logger.info("\nDemo completed successfully!")
        logger.info("Check 'demo_hyperopt_results' directory for detailed results")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
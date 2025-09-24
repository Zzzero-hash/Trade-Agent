#!/usr/bin/env python3
"""
CNN+LSTM Hyperparameter Optimization Runner - Task 5.5

This script runs the complete hyperparameter optimization pipeline for CNN+LSTM models.

Usage:
    python scripts/run_cnn_lstm_hyperopt.py --n-trials 1000 --retrain-best --top-k 5

Requirements: 3.4, 9.1
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.cnn_lstm_hyperopt import CNNLSTMHyperparameterOptimizer, run_cnn_lstm_hyperparameter_optimization
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/cnn_lstm_hyperopt.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_sample_data_loaders(batch_size: int = 32, input_dim: int = 11):
    """Create sample data loaders for testing hyperparameter optimization"""
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create synthetic data for testing
    num_samples = 1000
    sequence_length = 60
    
    # Generate synthetic multi-timeframe data
    X = torch.randn(num_samples, input_dim, sequence_length)
    
    # Generate synthetic targets
    y_class = torch.randint(0, 4, (num_samples,))  # 4 regime classes
    y_reg = torch.randn(num_samples, 2)  # Price and volatility targets
    
    # Create dataset
    dataset = TensorDataset(X, y_class, y_reg)
    
    # Split data
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_real_data_loaders(
    symbols: list = ["AAPL", "GOOGL", "MSFT"],
    batch_size: int = 32,
    start_date: str = "2022-01-01",
    end_date: str = "2024-01-01"
):
    """Create real data loaders using yfinance data"""
    
    logger = logging.getLogger(__name__)
    logger.info("Creating real data loaders with yfinance data...")
    
    try:
        # Create a temporary trainer to prepare data
        config = create_hybrid_config(
            input_dim=11,  # Will be updated after data preparation
            sequence_length=60,
            num_classes=4,
            regression_targets=2
        )
        
        trainer = IntegratedCNNLSTMTrainer(config)
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size
        )
        
        logger.info("Real data loaders created successfully")
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Failed to create real data loaders: {e}")
        logger.info("Falling back to synthetic data loaders")
        return create_sample_data_loaders(batch_size)


def main():
    """Main hyperparameter optimization pipeline"""
    
    parser = argparse.ArgumentParser(description="CNN+LSTM Hyperparameter Optimization")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="Number of hyperparameter trials to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for optimization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=11,
        help="Input feature dimension"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="hyperopt_results",
        help="Directory to save optimization results"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="cnn_lstm_optimization",
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["median", "successive_halving"],
        default="median",
        help="Type of pruner to use"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Type of sampler to use"
    )
    parser.add_argument(
        "--retrain-best",
        action="store_true",
        help="Retrain best models with full training"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best models to retrain"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real yfinance data instead of synthetic data"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        help="Stock symbols to use for real data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Start date for real data"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-01",
        help="End date for real data"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting CNN+LSTM Hyperparameter Optimization - Task 5.5")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Create data loaders
        if args.use_real_data:
            logger.info("Using real yfinance data")
            data_loaders = create_real_data_loaders(
                symbols=args.symbols,
                batch_size=args.batch_size,
                start_date=args.start_date,
                end_date=args.end_date
            )
        else:
            logger.info("Using synthetic data for testing")
            data_loaders = create_sample_data_loaders(
                batch_size=args.batch_size,
                input_dim=args.input_dim
            )
        
        train_loader, val_loader, test_loader = data_loaders
        
        logger.info(f"Data loaders created - Train: {len(train_loader)}, "
                   f"Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Create hyperparameter optimizer
        optimizer = CNNLSTMHyperparameterOptimizer(
            data_loaders=data_loaders,
            input_dim=args.input_dim,
            save_dir=args.save_dir,
            study_name=args.study_name,
            device=args.device
        )
        
        # Run optimization
        logger.info(f"Starting optimization with {args.n_trials} trials...")
        optimization_results = optimizer.optimize(
            n_trials=args.n_trials,
            timeout=args.timeout,
            pruner_type=args.pruner,
            sampler_type=args.sampler
        )
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Found {len(optimization_results['best_trials'])} Pareto optimal solutions")
        
        # Retrain best models if requested
        retrained_results = []
        if args.retrain_best:
            logger.info(f"Retraining top {args.top_k} models...")
            retrained_results = optimizer.retrain_best_models(
                top_k=args.top_k,
                full_epochs=200,
                save_models=True
            )
            logger.info(f"Successfully retrained {len(retrained_results)} models")
        
        # Generate and display report
        report = optimizer.generate_optimization_report()
        logger.info("Optimization Report:")
        logger.info(report)
        
        # Save final summary
        summary = {
            'optimization_results': optimization_results,
            'retrained_results': retrained_results,
            'report': report,
            'configuration': vars(args)
        }
        
        summary_path = Path(args.save_dir) / "optimization_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Complete summary saved to {summary_path}")
        logger.info("CNN+LSTM Hyperparameter Optimization completed successfully!")
        
        # Print key results
        if optimization_results['best_trials']:
            best_trial = optimization_results['best_trials'][0]
            logger.info(f"Best trial results:")
            logger.info(f"  Accuracy: {best_trial['values'][0]:.4f}")
            logger.info(f"  Training Time: {best_trial['values'][1]:.2f}s")
            logger.info(f"  Model Size: {best_trial['values'][2]:.2f}MB")
        
        if retrained_results:
            best_retrained = max(retrained_results, 
                               key=lambda x: x['test_metrics'].get('test_class_acc', 0))
            logger.info(f"Best retrained model test accuracy: "
                       f"{best_retrained['test_metrics'].get('test_class_acc', 0):.4f}")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
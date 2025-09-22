#!/usr/bin/env python3
"""
Test script for CNN+LSTM training pipeline.

This script tests the basic functionality of the training pipeline
with minimal configuration to ensure everything works correctly.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.cnn_lstm.trainer import CNNLSTMTrainer, TrainingConfig
from data.pipeline import create_data_loaders


def test_basic_training():
    """Test basic training functionality."""
    print("Testing CNN+LSTM training pipeline...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create minimal training configuration
    config = TrainingConfig(
        model_type="hybrid",
        model_config={
            "input_dim": 5,
            "sequence_length": 100,
            "fusion_dim": 256,  # Smaller for testing
            "use_adaptive_selection": False,  # Disable for simplicity
            "output_dim": 128
        },
        num_epochs=2,  # Very short for testing
        batch_size=8,  # Small batch size
        learning_rate=1e-3,
        use_mixed_precision=False,  # Disable for CPU testing
        scheduler=None,  # Disable scheduler for testing
        early_stopping_patience=5,
        save_every_n_epochs=1,
        checkpoint_dir="test_checkpoints",
        track_feature_quality=True,
        experiment_name="test_training",
        device="cpu"  # Force CPU for testing
    )
    
    logger.info(f"Created training config: {config.model_type}")
    
    # Create trainer
    trainer = CNNLSTMTrainer(config)
    logger.info(f"Created trainer with device: {trainer.device}")
    
    # Create simple data loaders
    try:
        train_loader, val_loader, _ = create_data_loaders(
            timeframes=["1min", "5min", "15min"],
            sequence_length=100,
            target_columns=["price_prediction"],
            batch_size=config.batch_size,
            validation_split=0.2,
            num_workers=0,  # No multiprocessing for testing
            pin_memory=False
        )
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return False
    
    # Test a single batch
    try:
        for batch_idx, (data, targets) in enumerate(train_loader):
            logger.info(f"Testing batch {batch_idx}")
            logger.info(f"Data keys: {list(data.keys())}")
            for key, tensor in data.items():
                logger.info(f"  {key}: {tensor.shape}")
            
            if isinstance(targets, dict):
                logger.info(f"Target keys: {list(targets.keys())}")
                for key, tensor in targets.items():
                    logger.info(f"  {key}: {tensor.shape}")
            else:
                logger.info(f"Targets: {targets.shape}")
            
            break
    except Exception as e:
        logger.error(f"Failed to process batch: {e}")
        return False
    
    # Test training for a few steps
    try:
        logger.info("Starting training test...")
        results = trainer.train(train_loader, val_loader)
        logger.info(f"Training completed successfully!")
        logger.info(f"Final metrics: {results.get('best_metrics', {})}")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_models():
    """Test individual CNN and LSTM models."""
    print("\nTesting individual models...")
    
    # Test CNN model
    print("Testing CNN model...")
    cnn_config = TrainingConfig(
        model_type="cnn",
        model_config={
            "timeframes": ["1min", "5min", "15min"],
            "sequence_length": 100,
            "num_features": 5,
            "cnn_filters": [32, 64],
            "kernel_sizes": [3, 5],
            "output_dim": 128
        },
        num_epochs=1,
        batch_size=4,
        scheduler=None,  # Disable scheduler
        device="cpu"
    )
    
    try:
        cnn_trainer = CNNLSTMTrainer(cnn_config)
        print("CNN trainer created successfully")
    except Exception as e:
        print(f"CNN trainer creation failed: {e}")
        return False
    
    # Test LSTM model
    print("Testing LSTM model...")
    lstm_config = TrainingConfig(
        model_type="lstm",
        model_config={
            "input_dim": 5,
            "sequence_length": 100,
            "lstm_hidden_dim": 64,
            "num_lstm_layers": 1,
            "output_dim": 128
        },
        num_epochs=1,
        batch_size=4,
        scheduler=None,  # Disable scheduler
        device="cpu"
    )
    
    try:
        lstm_trainer = CNNLSTMTrainer(lstm_config)
        print("LSTM trainer created successfully")
    except Exception as e:
        print(f"LSTM trainer creation failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CNN+LSTM Training Pipeline Test")
    print("=" * 60)
    
    # Test individual models first
    if not test_individual_models():
        print("Individual model tests failed!")
        sys.exit(1)
    
    # Test basic training
    if test_basic_training():
        print("\n" + "=" * 60)
        print("All tests passed! Training pipeline is working correctly.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training test failed!")
        print("=" * 60)
        sys.exit(1)
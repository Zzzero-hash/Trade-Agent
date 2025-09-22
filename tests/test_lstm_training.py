"""
Test script for LSTM temporal sequence modeling training.

This script tests the LSTM training pipeline to ensure it works correctly
and meets the requirements for task 5.3.
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from experiments.runners.train_lstm_temporal import LSTMTrainingConfig, LSTMTemporalTrainer
from data.pipeline import create_data_loaders


def test_lstm_training():
    """Test LSTM temporal training pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Testing LSTM temporal training pipeline...")
    
    # Create test configuration
    config = LSTMTrainingConfig(
        # Model configuration
        input_dim=5,
        sequence_length=50,  # Shorter for testing
        lstm_hidden_dim=128,  # Smaller for testing
        num_lstm_layers=2,
        lstm_dropout=0.2,
        
        # Training configuration
        num_epochs=3,  # Just a few epochs for testing
        batch_size=16,  # Smaller batch for testing
        learning_rate=1e-3,
        weight_decay=1e-4,
        
        # LSTM-specific regularization
        gradient_clip_val=1.0,
        recurrent_dropout=0.1,
        layer_norm=True,
        
        # Device configuration
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=torch.cuda.is_available(),
        
        # Paths
        checkpoint_dir="test_checkpoints",
        data_path="data/processed",
        
        # Validation
        prediction_horizons=[1, 5, 10],
        
        # Experiment tracking
        experiment_name="test_lstm_temporal",
        use_wandb=False
    )
    
    logger.info(f"Using device: {config.device}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_l
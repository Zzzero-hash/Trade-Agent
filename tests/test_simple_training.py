#!/usr/bin/env python3
"""
Simple test script to verify the training pipeline fixes with minimal complexity.
"""

import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.cnn_lstm.bidirectional_lstm_attention import BidirectionalLSTMWithAttention, BidirectionalLSTMConfig
from data.pipeline import create_data_loaders


def test_simple_lstm():
    """Test simple LSTM model with fixed prediction heads."""
    print("Testing simple LSTM model with prediction heads...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create LSTM model
    config = BidirectionalLSTMConfig(
        input_dim=5,
        sequence_length=100,
        lstm_hidden_dim=32,
        num_lstm_layers=1,
        attention_dim=64,
        output_dim=32,
        dropout_rate=0.1
    )
    
    model = BidirectionalLSTMWithAttention(config)
    logger.info(f"Created LSTM model with output_dim: {config.output_dim}")
    
    # Create simple data loaders
    try:
        train_loader, val_loader, _ = create_data_loaders(
            timeframes=["1min", "5min", "15min"],
            sequence_length=100,
            target_columns=["price_prediction"],
            batch_size=4,
            validation_split=0.2,
            num_workers=0,
            pin_memory=False
        )
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return False
    
    # Test model forward pass and training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    total_direction_correct = 0
    total_samples = 0
    
    for epoch in range(3):
        epoch_losses = []
        epoch_direction_acc = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            if batch_idx >= 5:  # Only test a few batches
                break
                
            # Get sequence data for LSTM
            sequence_data = data["sequence_data"]  # (batch_size, seq_len, features)
            
            logger.info(f"Epoch {epoch}, Batch {batch_idx}")
            logger.info(f"Sequence data shape: {sequence_data.shape}")
            logger.info(f"Targets shape: {targets.shape}")
            logger.info(f"Target range: [{targets.min():.6f}, {targets.max():.6f}]")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequence_data)
            
            logger.info(f"Model outputs keys: {list(outputs.keys())}")
            
            # Use price prediction output
            predictions = outputs["price_prediction"]
            logger.info(f"Predictions shape: {predictions.shape}")
            logger.info(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute direction accuracy
            pred_direction = torch.where(predictions > 0, 1, -1)
            true_direction = torch.where(targets > 0, 1, -1)
            
            # Only count non-zero targets
            non_zero_mask = torch.abs(targets) > 1e-6
            if torch.sum(non_zero_mask) > 0:
                direction_correct = torch.sum(
                    pred_direction[non_zero_mask] == true_direction[non_zero_mask]
                ).item()
                direction_total = torch.sum(non_zero_mask).item()
                direction_acc = direction_correct / direction_total
                
                total_direction_correct += direction_correct
                total_samples += direction_total
                
                epoch_direction_acc.append(direction_acc)
                logger.info(f"Batch direction accuracy: {direction_acc:.4f} ({direction_correct}/{direction_total})")
            
            epoch_losses.append(loss.item())
            logger.info(f"Batch loss: {loss.item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        avg_direction_acc = np.mean(epoch_direction_acc) if epoch_direction_acc else 0.0
        
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.6f}, Direction Acc={avg_direction_acc:.4f}")
    
    # Final results
    overall_direction_acc = total_direction_correct / total_samples if total_samples > 0 else 0.0
    logger.info(f"Overall direction accuracy: {overall_direction_acc:.4f} ({total_direction_correct}/{total_samples})")
    
    if overall_direction_acc > 0.0:
        logger.info("SUCCESS: Direction accuracy is non-zero!")
        return True
    else:
        logger.warning("Direction accuracy is still zero")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Simple LSTM with Fixed Prediction Heads")
    print("=" * 60)
    
    if test_simple_lstm():
        print("\n" + "=" * 60)
        print("SUCCESS: Simple LSTM training is working!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED: Simple LSTM training has issues")
        print("=" * 60)
        sys.exit(1)
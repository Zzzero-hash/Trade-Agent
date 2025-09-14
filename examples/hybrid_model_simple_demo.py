"""Simple CNN+LSTM Hybrid Model Demo

This demo showcases the CNN+LSTM hybrid model capabilities without visualization:
- Multi-task learning (classification and regression)
- Ensemble predictions
- Uncertainty quantification
- Model training and evaluation
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.hybrid_model import (
    CNNLSTMHybridModel,
    create_hybrid_config,
    create_hybrid_data_loader
)


def generate_synthetic_data(num_samples=500, sequence_length=30, num_features=8):
    """Generate synthetic market data for demonstration"""
    
    print(f"Generating {num_samples} samples of synthetic market data...")
    
    # Generate base time series with trends and patterns
    time_steps = np.arange(num_samples + sequence_length)
    
    # Create multiple correlated time series
    features = []
    
    for i in range(num_features):
        # Base trend with different frequencies
        trend = np.sin(time_steps * 0.02 * (i + 1)) + np.cos(time_steps * 0.01 * (i + 1))
        
        # Add random walk component
        random_walk = np.cumsum(np.random.randn(len(time_steps)) * 0.01)
        
        # Combine with noise
        feature = trend + random_walk + np.random.randn(len(time_steps)) * 0.1
        features.append(feature)
    
    features = np.array(features).T  # Shape: (time_steps, num_features)
    
    # Create sequences
    X = []
    y_prices = []
    y_returns = []
    
    for i in range(num_samples):
        # Input sequence
        seq = features[i:i + sequence_length]
        X.append(seq.T)  # Transpose for CNN: (features, sequence_length)
        
        # Future price (regression target)
        future_price = features[i + sequence_length, 0]  # Use first feature as price
        y_prices.append([future_price])
        
        # Future return
        current_price = features[i + sequence_length - 1, 0]
        future_return = (future_price - current_price) / current_price
        y_returns.append(future_return)
    
    X = np.array(X)
    y_prices = np.array(y_prices)
    y_returns = np.array(y_returns)
    
    # Create classification targets based on returns
    # 0: Sell (return < -0.01), 1: Hold (-0.01 <= return <= 0.01), 2: Buy (return > 0.01)
    y_class = np.zeros(len(y_returns), dtype=int)
    y_class[y_returns > 0.01] = 2  # Buy
    y_class[y_returns < -0.01] = 0  # Sell
    y_class[(y_returns >= -0.01) & (y_returns <= 0.01)] = 1  # Hold
    
    print(f"Data shape: X={X.shape}, y_class={y_class.shape}, y_prices={y_prices.shape}")
    print(f"Class distribution: Sell={np.sum(y_class == 0)}, Hold={np.sum(y_class == 1)}, Buy={np.sum(y_class == 2)}")
    
    return X, y_class, y_prices


def main():
    """Main demonstration function"""
    
    print("CNN+LSTM HYBRID MODEL SIMPLE DEMO")
    print("="*50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    X, y_class, y_prices = generate_synthetic_data(
        num_samples=800,
        sequence_length=30,
        num_features=8
    )
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_class_train = y_class[:train_size]
    y_reg_train = y_prices[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_class_val = y_class[train_size:train_size + val_size]
    y_reg_val = y_prices[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_class_test = y_class[train_size + val_size:]
    y_reg_test = y_prices[train_size + val_size:]
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Create model configuration
    print(f"\nCreating hybrid model...")
    config = create_hybrid_config(
        input_dim=X_train.shape[1],
        sequence_length=X_train.shape[2],
        prediction_horizon=1,
        num_classes=3,
        regression_targets=1,
        feature_fusion_dim=128,
        cnn_num_filters=32,
        lstm_hidden_dim=64,
        num_ensemble_models=3,
        batch_size=16,
        epochs=20,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Model configuration:")
    print(f"  Input dimension: {config.input_dim}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Feature fusion dimension: {config.feature_fusion_dim}")
    print(f"  Number of ensemble models: {config.num_ensemble_models}")
    print(f"  Device: {config.device}")
    
    # Create model
    model = CNNLSTMHybridModel(config)
    
    print(f"\nModel architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    print(f"\nStarting training...")
    start_time = datetime.now()
    
    training_result = model.fit(
        X_train, y_class_train, y_reg_train,
        X_val, y_class_val, y_reg_val
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    print(f"\nTraining completed!")
    print(f"  Duration: {training_duration:.2f} seconds")
    print(f"  Epochs trained: {training_result.epochs_trained}")
    print(f"  Final train loss: {training_result.train_loss:.6f}")
    print(f"  Final validation loss: {training_result.val_loss:.6f}")
    print(f"  Best epoch: {training_result.best_epoch}")
    
    # Evaluate model
    print(f"\nEvaluating model...")
    predictions = model.predict(X_test, return_uncertainty=True, use_ensemble=True)
    
    # Classification evaluation
    class_pred = predictions['classification_pred']
    class_probs = predictions['classification_probs']
    
    # Classification accuracy
    accuracy = np.mean(class_pred == y_class_test)
    print(f"\nClassification Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    for class_idx in range(3):
        class_mask = y_class_test == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(class_pred[class_mask] == class_idx)
            class_name = ['Sell', 'Hold', 'Buy'][class_idx]
            print(f"  {class_name} accuracy: {class_acc:.4f}")
    
    # Regression evaluation
    reg_pred = predictions['regression_pred'].flatten()
    reg_uncertainty = predictions['regression_uncertainty'].flatten()
    ensemble_reg = predictions['ensemble_regression'].flatten()
    y_reg_flat = y_reg_test.flatten()
    
    # Regression metrics
    mse = np.mean((reg_pred - y_reg_flat) ** 2)
    mae = np.mean(np.abs(reg_pred - y_reg_flat))
    rmse = np.sqrt(mse)
    
    print(f"\nRegression Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Ensemble comparison
    ensemble_mse = np.mean((ensemble_reg - y_reg_flat) ** 2)
    
    print(f"\nEnsemble vs Individual:")
    print(f"  Individual MSE: {mse:.6f}")
    print(f"  Ensemble MSE: {ensemble_mse:.6f}")
    print(f"  Improvement: {((mse - ensemble_mse) / mse * 100):.2f}%")
    
    # Uncertainty analysis
    print(f"\nUncertainty Analysis:")
    print(f"  Mean uncertainty: {np.mean(reg_uncertainty):.6f}")
    print(f"  Std uncertainty: {np.std(reg_uncertainty):.6f}")
    print(f"  Min uncertainty: {np.min(reg_uncertainty):.6f}")
    print(f"  Max uncertainty: {np.max(reg_uncertainty):.6f}")
    
    # Feature extraction demo
    print(f"\nFeature Extraction Demo:")
    sample_input = X_test[:3]  # Use first 3 test samples
    
    with torch.no_grad():
        model.eval()
        x_tensor = torch.FloatTensor(sample_input).to(model.device)
        output = model.forward(x_tensor, return_features=True, use_ensemble=False)
    
    print(f"  Input shape: {sample_input.shape}")
    print(f"  CNN features shape: {output['cnn_features'].shape}")
    print(f"  LSTM features shape: {output['lstm_features'].shape}")
    print(f"  Fused features shape: {output['fused_features'].shape}")
    
    # Save model
    model_path = os.path.join(".", "hybrid_model_demo.pth")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Key achievements:")
    print("✓ Successfully trained CNN+LSTM hybrid model")
    print("✓ Demonstrated multi-task learning capabilities")
    print("✓ Showed ensemble prediction improvements")
    print("✓ Quantified prediction uncertainty")
    print("✓ Extracted and analyzed intermediate features")
    print("="*50)


if __name__ == "__main__":
    main()
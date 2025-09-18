"""CNN+LSTM Hybrid Model Demo

This demo showcases the CNN+LSTM hybrid model capabilities including:
- Multi-task learning (classification and regression)
- Ensemble predictions
- Uncertainty quantification
- Feature extraction and fusion
- Model training and evaluation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.hybrid_model import (
    CNNLSTMHybridModel,
    create_hybrid_config,
    create_hybrid_data_loader
)
from ml.feature_engineering import TechnicalIndicators


def generate_synthetic_market_data(
    num_samples: int = 1000,
    sequence_length: int = 60,
    num_features: int = 10,
    noise_level: float = 0.1
) -> tuple:
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
        feature = trend + random_walk + np.random.randn(len(time_steps)) * noise_level
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
    
    return X, y_class, y_prices, y_returns


def train_hybrid_model(X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val):
    """Train the CNN+LSTM hybrid model"""
    
    print("\n" + "="*50)
    print("TRAINING CNN+LSTM HYBRID MODEL")
    print("="*50)
    
    # Create model configuration
    config = create_hybrid_config(
        input_dim=X_train.shape[1],
        sequence_length=X_train.shape[2],
        prediction_horizon=1,
        num_classes=3,
        regression_targets=1,
        feature_fusion_dim=256,
        cnn_filter_sizes=[3, 5, 7, 11],
        cnn_num_filters=64,
        cnn_use_attention=True,
        lstm_hidden_dim=128,
        lstm_num_layers=3,
        lstm_bidirectional=True,
        lstm_use_attention=True,
        num_ensemble_models=5,
        use_monte_carlo_dropout=True,
        mc_dropout_samples=50,
        classification_weight=0.4,
        regression_weight=0.6,
        batch_size=32,
        epochs=50,
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
    
    return model, training_result


def evaluate_model(model, X_test, y_class_test, y_reg_test):
    """Evaluate the trained model"""
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test, return_uncertainty=True, use_ensemble=True)
    
    # Classification evaluation
    class_pred = predictions['classification_pred']
    class_probs = predictions['classification_probs']
    ensemble_class = predictions['ensemble_classification']
    
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
    ensemble_mae = np.mean(np.abs(ensemble_reg - y_reg_flat))
    
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
    
    return predictions


def visualize_results(predictions, y_class_test, y_reg_test, save_path="hybrid_model_results.png"):
    """Visualize model predictions and results"""
    
    print(f"\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CNN+LSTM Hybrid Model Results', fontsize=16, fontweight='bold')
    
    # 1. Classification confusion matrix
    ax = axes[0, 0]
    class_names = ['Sell', 'Hold', 'Buy']
    class_pred = predictions['classification_pred']
    
    # Create confusion matrix
    confusion_matrix = np.zeros((3, 3))
    for true_class in range(3):
        for pred_class in range(3):
            confusion_matrix[true_class, pred_class] = np.sum(
                (y_class_test == true_class) & (class_pred == pred_class)
            )
    
    im = ax.imshow(confusion_matrix, cmap='Blues')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Classification Confusion Matrix')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{int(confusion_matrix[i, j])}', 
                   ha='center', va='center', fontweight='bold')
    
    # 2. Classification probabilities
    ax = axes[0, 1]
    class_probs = predictions['classification_probs']
    
    for class_idx in range(3):
        class_mask = y_class_test == class_idx
        if np.sum(class_mask) > 0:
            probs = class_probs[class_mask, class_idx]
            ax.hist(probs, bins=20, alpha=0.7, label=class_names[class_idx])
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Classification Probability Distributions')
    ax.legend()
    
    # 3. Ensemble weights
    ax = axes[0, 2]
    ensemble_weights = predictions['ensemble_weights']
    if ensemble_weights is not None:
        ax.bar(range(len(ensemble_weights)), ensemble_weights)
        ax.set_xlabel('Ensemble Model')
        ax.set_ylabel('Weight')
        ax.set_title('Ensemble Model Weights')
    else:
        ax.text(0.5, 0.5, 'No Ensemble Weights', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Ensemble Weights (Not Available)')
    
    # 4. Regression predictions vs actual
    ax = axes[1, 0]
    reg_pred = predictions['regression_pred'].flatten()
    y_reg_flat = y_reg_test.flatten()
    
    ax.scatter(y_reg_flat, reg_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(np.min(y_reg_flat), np.min(reg_pred))
    max_val = max(np.max(y_reg_flat), np.max(reg_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Regression: Predicted vs Actual')
    ax.legend()
    
    # 5. Prediction uncertainty
    ax = axes[1, 1]
    reg_uncertainty = predictions['regression_uncertainty'].flatten()
    
    # Scatter plot of uncertainty vs prediction error
    prediction_error = np.abs(reg_pred - y_reg_flat)
    ax.scatter(reg_uncertainty, prediction_error, alpha=0.6, s=20)
    
    ax.set_xlabel('Prediction Uncertainty')
    ax.set_ylabel('Prediction Error (Absolute)')
    ax.set_title('Uncertainty vs Prediction Error')
    
    # 6. Time series of predictions
    ax = axes[1, 2]
    
    # Show first 100 samples
    n_show = min(100, len(y_reg_flat))
    indices = range(n_show)
    
    ax.plot(indices, y_reg_flat[:n_show], 'b-', label='Actual', linewidth=2)
    ax.plot(indices, reg_pred[:n_show], 'r-', label='Predicted', linewidth=2)
    
    # Add uncertainty bands
    ax.fill_between(
        indices,
        reg_pred[:n_show] - reg_uncertainty[:n_show],
        reg_pred[:n_show] + reg_uncertainty[:n_show],
        alpha=0.3, color='red', label='Uncertainty'
    )
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.set_title('Time Series: Predictions with Uncertainty')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results visualization saved to: {save_path}")
    
    return fig


def demonstrate_feature_extraction(model, X_sample):
    """Demonstrate feature extraction capabilities"""
    
    print("\n" + "="*50)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("="*50)
    
    # Extract features from a sample
    sample_input = X_sample[:5]  # Use first 5 samples
    
    print(f"Extracting features from {sample_input.shape[0]} samples...")
    
    # Get model predictions with features
    with torch.no_grad():
        model.eval()
        x_tensor = torch.FloatTensor(sample_input).to(model.device)
        output = model.forward(x_tensor, return_features=True, use_ensemble=False)
    
    # Print feature shapes
    print(f"\nFeature extraction results:")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  CNN features shape: {output['cnn_features'].shape}")
    print(f"  LSTM features shape: {output['lstm_features'].shape}")
    print(f"  LSTM context shape: {output['lstm_context'].shape}")
    print(f"  Fused features shape: {output['fused_features'].shape}")
    
    # Analyze feature statistics
    cnn_features = output['cnn_features'].cpu().numpy()
    lstm_features = output['lstm_features'].cpu().numpy()
    fused_features = output['fused_features'].cpu().numpy()
    
    print(f"\nFeature statistics:")
    print(f"  CNN features - Mean: {np.mean(cnn_features):.4f}, Std: {np.std(cnn_features):.4f}")
    print(f"  LSTM features - Mean: {np.mean(lstm_features):.4f}, Std: {np.std(lstm_features):.4f}")
    print(f"  Fused features - Mean: {np.mean(fused_features):.4f}, Std: {np.std(fused_features):.4f}")
    
    return output


def main():
    """Main demonstration function"""
    
    print("CNN+LSTM HYBRID MODEL DEMONSTRATION")
    print("="*60)
    print("This demo showcases the hybrid model's capabilities:")
    print("- Multi-task learning (classification + regression)")
    print("- Ensemble predictions")
    print("- Uncertainty quantification")
    print("- Feature extraction and fusion")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    X, y_class, y_prices, y_returns = generate_synthetic_market_data(
        num_samples=2000,
        sequence_length=60,
        num_features=12,
        noise_level=0.1
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
    
    # Train model
    model, training_result = train_hybrid_model(
        X_train, y_class_train, y_reg_train,
        X_val, y_class_val, y_reg_val
    )
    
    # Evaluate model
    predictions = evaluate_model(model, X_test, y_class_test, y_reg_test)
    
    # Demonstrate feature extraction
    feature_output = demonstrate_feature_extraction(model, X_test)
    
    # Create visualizations
    fig = visualize_results(predictions, y_class_test, y_reg_test)
    
    # Save model
    model_path = "hybrid_model_demo.pth"
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key achievements:")
    print("✓ Successfully trained CNN+LSTM hybrid model")
    print("✓ Demonstrated multi-task learning capabilities")
    print("✓ Showed ensemble prediction improvements")
    print("✓ Quantified prediction uncertainty")
    print("✓ Extracted and analyzed intermediate features")
    print("✓ Created comprehensive visualizations")
    print("="*60)


if __name__ == "__main__":
    main()
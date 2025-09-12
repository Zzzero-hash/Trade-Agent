"""CNN Feature Extraction Demo

This script demonstrates how to use the CNN feature extraction model
for processing market data and extracting meaningful features.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.cnn_model import CNNFeatureExtractor, create_cnn_config
from src.ml.feature_engineering import TechnicalIndicators
from src.ml.training_pipeline import create_training_pipeline


def generate_sample_market_data(num_days: int = 200) -> pd.DataFrame:
    """Generate realistic sample market data for demonstration"""
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range("2023-01-01", periods=num_days, freq="D")

    # Generate realistic price data with trends and volatility
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, num_days)  # Daily returns

    # Add some trend and mean reversion
    trend = np.sin(np.arange(num_days) * 2 * np.pi / 50) * 0.005
    returns += trend

    # Calculate prices
    close_prices = [base_price]
    for i in range(1, num_days):
        close_prices.append(close_prices[-1] * (1 + returns[i]))

    close_prices = np.array(close_prices)

    # Generate OHLV data based on close prices
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, num_days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, num_days)))
    open_prices = close_prices + np.random.normal(0, 0.005, num_days) * close_prices
    volumes = np.random.lognormal(10, 0.5, num_days).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )


def prepare_cnn_data(features: np.ndarray, sequence_length: int = 30) -> tuple:
    """Prepare feature data for CNN input"""
    num_samples = len(features) - sequence_length + 1
    num_features = features.shape[1]

    # Create sequences for CNN input (batch_size, channels, sequence_length)
    X = np.zeros((num_samples, num_features, sequence_length))

    for i in range(num_samples):
        X[i] = features[i : i + sequence_length].T

    # Create dummy targets (in real scenario, these would be future returns or signals)
    y = np.random.randn(num_samples, sequence_length, 16)  # 16 output features

    return X.astype(np.float32), y.astype(np.float32)


def main():
    """Main demonstration function"""
    print("CNN Feature Extraction Demo")
    print("=" * 50)

    # 1. Generate sample market data
    print("\n1. Generating sample market data...")
    market_data = generate_sample_market_data(200)
    print(f"Generated {len(market_data)} days of market data")
    print(
        f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}"
    )

    # 2. Apply feature engineering
    print("\n2. Applying technical indicators...")
    tech_indicators = TechnicalIndicators()
    features = tech_indicators.fit_transform(market_data)
    print(f"Generated {features.shape[1]} technical features")
    print(f"Feature names: {tech_indicators.get_feature_names()}")

    # 3. Prepare data for CNN
    print("\n3. Preparing data for CNN...")
    sequence_length = 30
    X, y = prepare_cnn_data(features, sequence_length)
    print(f"CNN input shape: {X.shape}")
    print(f"CNN target shape: {y.shape}")

    # 4. Create CNN model
    print("\n4. Creating CNN model...")
    config = create_cnn_config(
        input_dim=X.shape[1],  # Number of features
        output_dim=16,  # Number of output features
        filter_sizes=[3, 5, 7, 11],
        num_filters=32,
        use_attention=True,
        num_attention_heads=4,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=16,
        epochs=5,
        device="cpu",
    )

    model = CNNFeatureExtractor(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 5. Test forward pass
    print("\n5. Testing forward pass...")
    sample_input = torch.FloatTensor(X[:4])  # Use first 4 samples
    model.eval()

    with torch.no_grad():
        output = model.forward(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")

    # 6. Extract features using convenience method
    print("\n6. Extracting features...")
    extracted_features = model.extract_features(X[:10])
    print(f"Extracted features shape: {extracted_features.shape}")

    # 7. Demonstrate training pipeline (short training for demo)
    print("\n7. Demonstrating training pipeline...")

    # Create training pipeline
    pipeline = create_training_pipeline(
        input_dim=X.shape[1],
        output_dim=16,
        checkpoint_dir="checkpoints/cnn_demo",
        log_dir="logs/cnn_demo",
        epochs=3,
        batch_size=8,
        learning_rate=0.001,
    )

    # Prepare data loaders (use original 2D features for the pipeline)
    train_loader, val_loader, test_loader = pipeline.prepare_data(
        features=features,  # Use original 2D features
        targets=np.random.randn(len(features), 16),  # Dummy targets
        sequence_length=sequence_length,
        train_split=0.7,
        val_split=0.2,
        batch_size=8,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Train model (short training for demo)
    print("\nTraining model (short demo)...")
    result = pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        early_stopping_patience=5,
    )

    print(f"Training completed:")
    print(f"  Final train loss: {result.train_loss:.6f}")
    print(f"  Final val loss: {result.val_loss:.6f}")
    print(f"  Epochs trained: {result.epochs_trained}")
    print(f"  Best epoch: {result.best_epoch}")

    # 8. Evaluate on test data
    print("\n8. Evaluating on test data...")
    test_metrics = pipeline.evaluate(test_loader)

    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # 9. Save model
    print("\n9. Saving model...")
    model_path = "checkpoints/cnn_demo/final_model.pth"
    pipeline.model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # 10. Load and test saved model
    print("\n10. Testing model loading...")
    new_model = CNNFeatureExtractor(config)
    new_model.load_model(model_path)

    # Test that loaded model produces same output
    with torch.no_grad():
        original_output = pipeline.model.extract_features(X[:2])
        loaded_output = new_model.extract_features(X[:2])

    if np.allclose(original_output, loaded_output, atol=1e-6):
        print("✓ Model loading successful - outputs match!")
    else:
        print("✗ Model loading issue - outputs don't match")

    # Cleanup
    pipeline.close()

    print("\n" + "=" * 50)
    print("CNN Feature Extraction Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("- Multi-filter CNN architecture with attention")
    print("- Feature extraction from market data")
    print("- Model training with validation")
    print("- Model checkpointing and versioning")
    print("- Complete training pipeline")


if __name__ == "__main__":
    main()

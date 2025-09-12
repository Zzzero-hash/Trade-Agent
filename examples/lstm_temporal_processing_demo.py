"""LSTM Temporal Processing Demo

This demo showcases the LSTM temporal processing model for time series prediction.
It demonstrates:
- Creating synthetic time series data
- Training the LSTM model with attention and skip connections
- Making sequence-to-sequence predictions
- Feature extraction capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.lstm_model import (
    LSTMTemporalProcessor,
    create_lstm_config,
    create_lstm_data_loader,
    create_sequence_data
)


def generate_synthetic_time_series(num_timesteps: int = 1000, num_features: int = 5) -> np.ndarray:
    """Generate synthetic time series data with multiple patterns"""
    t = np.linspace(0, 10 * np.pi, num_timesteps)
    
    # Create multiple time series with different patterns
    data = np.column_stack([
        np.sin(t) + 0.1 * np.random.randn(num_timesteps),  # Sine wave with noise
        np.cos(t) + 0.1 * np.random.randn(num_timesteps),  # Cosine wave with noise
        np.sin(2 * t) + 0.1 * np.random.randn(num_timesteps),  # Higher frequency sine
        np.sin(t/2) + 0.1 * np.random.randn(num_timesteps),  # Lower frequency sine
        0.1 * np.random.randn(num_timesteps)  # Pure noise
    ])
    
    # Add trend to first feature
    trend = np.linspace(0, 2, num_timesteps)
    data[:, 0] += trend
    
    return data


def demonstrate_lstm_training():
    """Demonstrate LSTM model training and prediction"""
    print("=== LSTM Temporal Processing Demo ===\n")
    
    # Generate synthetic data
    print("1. Generating synthetic time series data...")
    data = generate_synthetic_time_series(num_timesteps=2000, num_features=5)
    print(f"   Generated data shape: {data.shape}")
    
    # Create sequences for training
    print("\n2. Creating training sequences...")
    sequence_length = 60
    prediction_length = 20
    
    sequences, targets = create_sequence_data(
        data, 
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        stride=5
    )
    
    print(f"   Input sequences shape: {sequences.shape}")
    print(f"   Target sequences shape: {targets.shape}")
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
    train_target, val_target = targets[:split_idx], targets[split_idx:]
    
    print(f"   Training samples: {len(train_seq)}")
    print(f"   Validation samples: {len(val_seq)}")
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader = create_lstm_data_loader(train_seq, train_target, batch_size=32, shuffle=True)
    val_loader = create_lstm_data_loader(val_seq, val_target, batch_size=32, shuffle=False)
    
    # Create LSTM model
    print("\n4. Creating LSTM model...")
    config = create_lstm_config(
        input_dim=5,
        output_dim=5,
        hidden_dim=64,
        num_layers=3,
        sequence_length=sequence_length,
        bidirectional=True,
        use_attention=True,
        use_skip_connections=True,
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=20,
        device="cpu"
    )
    
    model = LSTMTemporalProcessor(config)
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\n5. Training LSTM model...")
    training_result = model.train_model(train_loader, val_loader, num_epochs=15)
    
    print(f"   Training completed!")
    print(f"   Final training loss: {training_result.train_loss:.6f}")
    print(f"   Final validation loss: {training_result.val_loss:.6f}")
    print(f"   Best epoch: {training_result.best_epoch}")
    
    # Make predictions
    print("\n6. Making predictions...")
    model.eval()
    
    # Use first few validation samples for prediction
    test_sequences = val_seq[:5]
    test_targets = val_target[:5]
    
    predictions = model.predict_sequence(test_sequences, target_length=prediction_length)
    print(f"   Prediction shape: {predictions.shape}")
    
    # Calculate prediction accuracy
    mse = np.mean((predictions - test_targets) ** 2)
    mae = np.mean(np.abs(predictions - test_targets))
    print(f"   Prediction MSE: {mse:.6f}")
    print(f"   Prediction MAE: {mae:.6f}")
    
    # Feature extraction
    print("\n7. Extracting temporal features...")
    encoded_sequences, context_vectors = model.extract_features(test_sequences)
    print(f"   Encoded sequences shape: {encoded_sequences.shape}")
    print(f"   Context vectors shape: {context_vectors.shape}")
    
    # Demonstrate different configurations
    print("\n8. Testing different model configurations...")
    
    # Test unidirectional LSTM
    config_uni = create_lstm_config(
        input_dim=5, output_dim=5, hidden_dim=32, num_layers=2,
        bidirectional=False, use_attention=False, device="cpu"
    )
    model_uni = LSTMTemporalProcessor(config_uni)
    print(f"   Unidirectional LSTM parameters: {sum(p.numel() for p in model_uni.parameters())}")
    
    # Test without skip connections
    config_no_skip = create_lstm_config(
        input_dim=5, output_dim=5, hidden_dim=32, num_layers=2,
        use_skip_connections=False, device="cpu"
    )
    model_no_skip = LSTMTemporalProcessor(config_no_skip)
    print(f"   LSTM without skip connections parameters: {sum(p.numel() for p in model_no_skip.parameters())}")
    
    # Test model outputs
    test_input = torch.randn(2, 30, 5)
    
    with torch.no_grad():
        output_bi = model.forward(test_input)
        output_uni = model_uni.forward(test_input)
        output_no_skip = model_no_skip.forward(test_input)
    
    print(f"   Bidirectional output shape: {output_bi.shape}")
    print(f"   Unidirectional output shape: {output_uni.shape}")
    print(f"   No skip connections output shape: {output_no_skip.shape}")
    
    # Save model
    print("\n9. Saving trained model...")
    model_path = "checkpoints/lstm_demo/lstm_temporal_model.pth"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Load model
    print("\n10. Loading saved model...")
    new_model = LSTMTemporalProcessor(config)
    new_model.load_model(model_path)
    
    # Verify loaded model works
    with torch.no_grad():
        original_output = model.forward(test_input)
        loaded_output = new_model.forward(test_input)
    
    if torch.allclose(original_output, loaded_output, atol=1e-6):
        print("    ✓ Model loaded successfully - outputs match!")
    else:
        print("    ✗ Model loading failed - outputs don't match!")
    
    print("\n=== Demo completed successfully! ===")
    
    return model, training_result, predictions, test_targets


def plot_predictions(predictions: np.ndarray, targets: np.ndarray, sample_idx: int = 0):
    """Plot prediction vs target for visualization"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot first 3 features for the selected sample
        for i in range(min(3, predictions.shape[-1])):
            plt.subplot(3, 1, i + 1)
            plt.plot(predictions[sample_idx, :, i], label=f'Prediction Feature {i+1}', linestyle='--')
            plt.plot(targets[sample_idx, :, i], label=f'Target Feature {i+1}', linestyle='-')
            plt.title(f'Feature {i+1} - Prediction vs Target')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('lstm_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Prediction plot saved as 'lstm_predictions.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")


if __name__ == "__main__":
    # Run the demonstration
    model, training_result, predictions, targets = demonstrate_lstm_training()
    
    # Create visualization if matplotlib is available
    plot_predictions(predictions, targets, sample_idx=0)
    
    print("\nLSTM Temporal Processing Demo completed!")
    print("Key features demonstrated:")
    print("- Bidirectional LSTM with attention mechanism")
    print("- Skip connections for improved gradient flow")
    print("- Sequence-to-sequence prediction")
    print("- Feature extraction capabilities")
    print("- Model persistence (save/load)")
    print("- Dropout and regularization for overfitting prevention")
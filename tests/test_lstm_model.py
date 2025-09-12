"""Tests for LSTM Temporal Processing Model

This module contains comprehensive tests for the LSTM model including:
- Output shape validation
- Gradient flow verification
- Training functionality
- Attention mechanism testing
- Skip connections validation
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ml.lstm_model import (
    LSTMTemporalProcessor,
    LSTMAttention,
    create_lstm_config,
    create_lstm_data_loader,
    create_sequence_data
)
from src.ml.base_models import ModelConfig


class TestLSTMAttention:
    """Test cases for LSTM attention mechanism"""
    
    def test_attention_initialization(self):
        """Test attention mechanism initialization"""
        hidden_dim = 128
        attention_dim = 64
        
        attention = LSTMAttention(hidden_dim, attention_dim)
        
        assert attention.hidden_dim == hidden_dim
        assert attention.attention_dim == attention_dim
        assert isinstance(attention.attention_linear, nn.Linear)
        assert isinstance(attention.context_vector, nn.Linear)
        assert isinstance(attention.dropout, nn.Dropout)
    
    def test_attention_forward_shape(self):
        """Test attention mechanism output shapes"""
        batch_size = 16
        seq_len = 60
        hidden_dim = 128
        attention_dim = 64
        
        attention = LSTMAttention(hidden_dim, attention_dim)
        
        # Create dummy LSTM outputs
        lstm_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        context, attention_weights = attention(lstm_outputs)
        
        # Check output shapes
        assert context.shape == (batch_size, hidden_dim)
        assert attention_weights.shape == (batch_size, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_attention_with_mask(self):
        """Test attention mechanism with padding mask"""
        batch_size = 8
        seq_len = 30
        hidden_dim = 64
        
        attention = LSTMAttention(hidden_dim)
        lstm_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Create mask (first half of sequence is valid)
        mask = torch.ones(batch_size, seq_len)
        mask[:, seq_len//2:] = 0  # Mask second half
        
        context, attention_weights = attention(lstm_outputs, mask)
        
        # Check that masked positions have near-zero attention
        assert torch.allclose(attention_weights[:, seq_len//2:], torch.zeros(batch_size, seq_len//2), atol=1e-6)
        
        # Check that valid positions have positive attention
        assert torch.all(attention_weights[:, :seq_len//2] > 0)


class TestLSTMTemporalProcessor:
    """Test cases for LSTM temporal processor"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic LSTM configuration for testing"""
        return create_lstm_config(
            input_dim=10,
            output_dim=5,
            hidden_dim=32,
            num_layers=2,
            sequence_length=20,
            epochs=5,
            device="cpu"
        )
    
    @pytest.fixture
    def lstm_model(self, basic_config):
        """LSTM model instance for testing"""
        return LSTMTemporalProcessor(basic_config)
    
    def test_model_initialization(self, basic_config):
        """Test LSTM model initialization"""
        model = LSTMTemporalProcessor(basic_config)
        
        assert model.input_dim == 10
        assert model.output_dim == 5
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert model.bidirectional is True
        assert model.use_attention is True
        assert model.use_skip_connections is True
        
        # Check that model components are created
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'lstm_layers')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'decoder_lstm')
    
    def test_forward_output_shapes(self, lstm_model):
        """Test forward pass output shapes"""
        batch_size = 8
        seq_len = 20
        input_dim = 10
        output_dim = 5
        target_length = 15
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = lstm_model.forward(x, target_length)
        
        # Check output shape
        expected_shape = (batch_size, target_length, output_dim)
        assert output.shape == expected_shape
        
        # Test with default target length (same as input)
        output_default = lstm_model.forward(x)
        expected_default_shape = (batch_size, seq_len, output_dim)
        assert output_default.shape == expected_default_shape
    
    def test_encoder_only_shapes(self, lstm_model):
        """Test encoder-only forward pass shapes"""
        batch_size = 4
        seq_len = 25
        input_dim = 10
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        encoded_seq, context_vec = lstm_model.forward_encoder_only(x)
        
        # Check shapes
        expected_seq_shape = (batch_size, seq_len, lstm_model.hidden_dim * 2)  # Bidirectional
        expected_context_shape = (batch_size, lstm_model.hidden_dim * 2)
        
        assert encoded_seq.shape == expected_seq_shape
        assert context_vec.shape == expected_context_shape
    
    def test_gradient_flow(self, lstm_model):
        """Test gradient flow through the model"""
        batch_size = 4
        seq_len = 15
        input_dim = 10
        output_dim = 5
        
        # Create dummy data
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        target = torch.randn(batch_size, seq_len, output_dim)
        
        # Forward pass
        output = lstm_model.forward(x)
        
        # Calculate loss
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check that model parameters have gradients
        for name, param in lstm_model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient for parameter: {name}"
    
    def test_skip_connections(self):
        """Test skip connections functionality"""
        config = create_lstm_config(
            input_dim=8,
            output_dim=4,
            hidden_dim=16,
            num_layers=3,
            use_skip_connections=True,
            device="cpu"
        )
        
        model_with_skip = LSTMTemporalProcessor(config)
        
        # Test that skip projections are created
        assert len(model_with_skip.skip_projections) == model_with_skip.num_layers
        
        # Test without skip connections
        config.use_skip_connections = False
        model_without_skip = LSTMTemporalProcessor(config)
        
        # Both models should work but may have different performance
        x = torch.randn(2, 10, 8)
        
        output_with_skip = model_with_skip.forward(x)
        output_without_skip = model_without_skip.forward(x)
        
        assert output_with_skip.shape == output_without_skip.shape
    
    def test_bidirectional_vs_unidirectional(self):
        """Test bidirectional vs unidirectional LSTM"""
        # Bidirectional config
        config_bi = create_lstm_config(
            input_dim=6,
            output_dim=3,
            hidden_dim=20,
            bidirectional=True,
            device="cpu"
        )
        
        # Unidirectional config
        config_uni = create_lstm_config(
            input_dim=6,
            output_dim=3,
            hidden_dim=20,
            bidirectional=False,
            device="cpu"
        )
        
        model_bi = LSTMTemporalProcessor(config_bi)
        model_uni = LSTMTemporalProcessor(config_uni)
        
        x = torch.randn(3, 12, 6)
        
        output_bi = model_bi.forward(x)
        output_uni = model_uni.forward(x)
        
        # Both should have same output shape
        assert output_bi.shape == output_uni.shape
        
        # But different internal representations
        encoded_bi, _ = model_bi.forward_encoder_only(x)
        encoded_uni, _ = model_uni.forward_encoder_only(x)
        
        # Bidirectional should have 2x hidden dimension
        assert encoded_bi.shape[-1] == 2 * encoded_uni.shape[-1]
    
    def test_attention_mechanism(self):
        """Test attention mechanism integration"""
        # With attention
        config_att = create_lstm_config(
            input_dim=5,
            output_dim=2,
            use_attention=True,
            device="cpu"
        )
        
        # Without attention
        config_no_att = create_lstm_config(
            input_dim=5,
            output_dim=2,
            use_attention=False,
            device="cpu"
        )
        
        model_att = LSTMTemporalProcessor(config_att)
        model_no_att = LSTMTemporalProcessor(config_no_att)
        
        x = torch.randn(2, 8, 5)
        
        output_att = model_att.forward(x)
        output_no_att = model_no_att.forward(x)
        
        # Both should work and have same output shape
        assert output_att.shape == output_no_att.shape
        
        # Check that attention model has attention component
        assert hasattr(model_att, 'attention')
        assert model_att.attention is not None
        
        # No attention model should not have attention
        assert not hasattr(model_no_att, 'attention') or model_no_att.attention is None
    
    def test_different_sequence_lengths(self, lstm_model):
        """Test model with different sequence lengths"""
        batch_size = 3
        input_dim = 10
        
        # Test different input sequence lengths
        for seq_len in [5, 15, 30, 50]:
            x = torch.randn(batch_size, seq_len, input_dim)
            output = lstm_model.forward(x)
            
            # Output should match input sequence length by default
            assert output.shape == (batch_size, seq_len, lstm_model.output_dim)
        
        # Test different target lengths
        x = torch.randn(batch_size, 20, input_dim)
        for target_len in [5, 10, 25, 40]:
            output = lstm_model.forward(x, target_length=target_len)
            assert output.shape == (batch_size, target_len, lstm_model.output_dim)
    
    def test_model_training_step(self, lstm_model):
        """Test single training step"""
        batch_size = 4
        seq_len = 15
        input_dim = 10
        output_dim = 5
        
        # Create dummy data
        x = torch.randn(batch_size, seq_len, input_dim)
        y = torch.randn(batch_size, seq_len, output_dim)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training step
        lstm_model.train()
        optimizer.zero_grad()
        
        output = lstm_model.forward(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_model_evaluation_mode(self, lstm_model):
        """Test model in evaluation mode"""
        batch_size = 2
        seq_len = 10
        input_dim = 10
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Test in training mode
        lstm_model.train()
        output_train = lstm_model.forward(x)
        
        # Test in evaluation mode
        lstm_model.eval()
        with torch.no_grad():
            output_eval = lstm_model.forward(x)
        
        # Outputs should have same shape
        assert output_train.shape == output_eval.shape
        
        # Due to dropout, outputs might be different
        # But in eval mode, output should be deterministic
        with torch.no_grad():
            output_eval2 = lstm_model.forward(x)
        
        assert torch.allclose(output_eval, output_eval2)


class TestLSTMUtilityFunctions:
    """Test utility functions for LSTM model"""
    
    def test_create_lstm_config(self):
        """Test LSTM configuration creation"""
        config = create_lstm_config(
            input_dim=15,
            output_dim=8,
            hidden_dim=64,
            num_layers=4,
            bidirectional=False,
            use_attention=False,
            dropout_rate=0.2,
            learning_rate=0.002
        )
        
        assert config.input_dim == 15
        assert config.output_dim == 8
        assert config.hidden_dim == 64
        assert config.num_layers == 4
        assert config.bidirectional is False
        assert config.use_attention is False
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 0.002
    
    def test_create_sequence_data(self):
        """Test sequence data creation"""
        # Create dummy time series data
        timesteps = 1000
        features = 5
        data = np.random.randn(timesteps, features)
        
        sequence_length = 60
        prediction_length = 10
        stride = 5
        
        sequences, targets = create_sequence_data(
            data, sequence_length, prediction_length, stride
        )
        
        # Check shapes
        expected_num_sequences = (timesteps - sequence_length - prediction_length + 1) // stride
        assert sequences.shape == (expected_num_sequences, sequence_length, features)
        assert targets.shape == (expected_num_sequences, prediction_length, features)
        
        # Check data consistency
        for i in range(min(5, expected_num_sequences)):  # Check first 5 sequences
            seq_start = i * stride
            seq_end = seq_start + sequence_length
            target_start = seq_end
            target_end = target_start + prediction_length
            
            np.testing.assert_array_equal(sequences[i], data[seq_start:seq_end])
            np.testing.assert_array_equal(targets[i], data[target_start:target_end])
    
    def test_create_lstm_data_loader(self):
        """Test LSTM data loader creation"""
        num_samples = 100
        seq_len = 30
        input_dim = 8
        output_dim = 4
        batch_size = 16
        
        sequences = np.random.randn(num_samples, seq_len, input_dim)
        targets = np.random.randn(num_samples, seq_len, output_dim)
        
        dataloader = create_lstm_data_loader(
            sequences, targets, batch_size=batch_size, shuffle=True
        )
        
        # Check dataloader properties
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == batch_size
        
        # Check batch shapes
        for batch_x, batch_y in dataloader:
            assert batch_x.shape[1:] == (seq_len, input_dim)
            assert batch_y.shape[1:] == (seq_len, output_dim)
            assert batch_x.shape[0] <= batch_size  # Last batch might be smaller
            assert batch_x.shape[0] == batch_y.shape[0]
            break  # Just check first batch


class TestLSTMModelPersistence:
    """Test model saving and loading"""
    
    def test_save_and_load_model(self, tmp_path):
        """Test model saving and loading"""
        config = create_lstm_config(
            input_dim=6,
            output_dim=3,
            hidden_dim=24,
            num_layers=2,
            device="cpu"
        )
        
        # Create and train model briefly
        model = LSTMTemporalProcessor(config)
        model.is_trained = True
        
        # Save model
        model_path = tmp_path / "test_lstm_model.pth"
        model.save_model(str(model_path))
        
        # Check that files are created
        assert model_path.exists()
        config_path = tmp_path / "test_lstm_model_config.json"
        assert config_path.exists()
        
        # Create new model and load
        new_model = LSTMTemporalProcessor(config)
        new_model.load_model(str(model_path))
        
        # Check that models are equivalent
        assert new_model.is_trained
        assert new_model.input_dim == model.input_dim
        assert new_model.output_dim == model.output_dim
        assert new_model.hidden_dim == model.hidden_dim
        
        # Test that loaded model produces same output
        x = torch.randn(2, 10, 6)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model.forward(x)
            output2 = new_model.forward(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)


class TestLSTMIntegration:
    """Integration tests for LSTM model"""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline"""
        # Create synthetic data
        num_samples = 200
        seq_len = 20
        input_dim = 5
        output_dim = 2
        
        # Generate sequences with some pattern
        sequences = []
        targets = []
        
        for _ in range(num_samples):
            # Create sequence with trend
            t = np.linspace(0, 4*np.pi, seq_len)
            seq = np.column_stack([
                np.sin(t + np.random.normal(0, 0.1)),
                np.cos(t + np.random.normal(0, 0.1)),
                np.sin(2*t + np.random.normal(0, 0.1)),
                np.cos(2*t + np.random.normal(0, 0.1)),
                np.random.normal(0, 0.1, seq_len)
            ])
            
            # Target is next few values of first two features
            target = seq[-10:, :output_dim]  # Last 10 timesteps, first 2 features
            target = np.pad(target, ((0, seq_len-10), (0, 0)), mode='constant')
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split data
        split_idx = int(0.8 * num_samples)
        train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
        train_target, val_target = targets[:split_idx], targets[split_idx:]
        
        # Create data loaders
        train_loader = create_lstm_data_loader(train_seq, train_target, batch_size=16)
        val_loader = create_lstm_data_loader(val_seq, val_target, batch_size=16, shuffle=False)
        
        # Create model
        config = create_lstm_config(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=2,
            epochs=10,
            learning_rate=0.01,
            device="cpu"
        )
        
        model = LSTMTemporalProcessor(config)
        
        # Train model
        result = model.train_model(train_loader, val_loader, num_epochs=5)
        
        # Check training results
        assert isinstance(result.train_loss, float)
        assert isinstance(result.val_loss, float)
        assert result.epochs_trained > 0
        assert model.is_trained
        
        # Test prediction
        test_input = val_seq[:5]  # First 5 validation samples
        predictions = model.predict_sequence(test_input)
        
        assert predictions.shape == (5, seq_len, output_dim)
        
        # Test feature extraction
        encoded_seq, context_vec = model.extract_features(test_input)
        
        expected_hidden_dim = config.hidden_dim * 2  # Bidirectional
        assert encoded_seq.shape == (5, seq_len, expected_hidden_dim)
        assert context_vec.shape == (5, expected_hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__])
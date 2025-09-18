"""Tests for CNN Feature Extraction Model

This module contains comprehensive tests for the CNN feature extractor,
including architecture validation, forward pass testing, and training pipeline.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import os
import json

from src.ml.cnn_model import (
    CNNFeatureExtractor,
    MultiHeadAttention,
    create_cnn_config,
    create_cnn_data_loader
)
from src.ml.base_models import ModelConfig


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention module"""
    
    def test_attention_initialization(self):
        """Test attention module initialization"""
        d_model = 64
        num_heads = 8
        attention = MultiHeadAttention(d_model, num_heads)
        
        assert attention.d_model == d_model
        assert attention.num_heads == num_heads
        assert attention.d_k == d_model // num_heads
        
        # Check layer dimensions
        assert attention.w_q.in_features == d_model
        assert attention.w_q.out_features == d_model
        assert attention.w_k.in_features == d_model
        assert attention.w_k.out_features == d_model
        assert attention.w_v.in_features == d_model
        assert attention.w_v.out_features == d_model
        assert attention.w_o.in_features == d_model
        assert attention.w_o.out_features == d_model
    
    def test_attention_forward_pass(self):
        """Test attention forward pass with valid inputs"""
        d_model = 64
        num_heads = 8
        seq_len = 20
        batch_size = 4
        
        attention = MultiHeadAttention(d_model, num_heads)
        
        # Create test input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output, weights = attention(x, x, x)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check attention weights sum to approximately 1 (with tolerance for numerical precision)
        weight_sums = weights.sum(dim=-1)
        expected_sums = torch.ones_like(weight_sums)
        # Use more lenient tolerance due to dropout and numerical precision
        assert torch.allclose(weight_sums, expected_sums, atol=0.15, rtol=0.15)
    
    def test_attention_with_mask(self):
        """Test attention with masking"""
        d_model = 32
        num_heads = 4
        seq_len = 10
        batch_size = 2
        
        attention = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask (mask out last 3 positions)
        mask = torch.ones(batch_size, num_heads, seq_len, seq_len)
        mask[:, :, :, -3:] = 0
        
        output, weights = attention(x, x, x, mask)
        
        # Check that masked positions have near-zero attention
        assert torch.all(weights[:, :, :, -3:] < 1e-6)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)


class TestCNNFeatureExtractor:
    """Test cases for CNN Feature Extractor"""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return create_cnn_config(
            input_dim=10,
            output_dim=32,
            filter_sizes=[3, 5, 7],
            num_filters=16,
            use_attention=True,
            num_attention_heads=4,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=8,
            epochs=5,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 8
        input_channels = 10
        sequence_length = 50
        
        # Generate synthetic market-like data
        np.random.seed(42)
        features = np.random.randn(batch_size, input_channels, sequence_length)
        
        # Add some trend and patterns
        for i in range(batch_size):
            trend = np.linspace(0, 1, sequence_length)
            for j in range(input_channels):
                features[i, j, :] += trend * (j + 1) * 0.1
        
        # Target data (e.g., future returns)
        targets = np.random.randn(batch_size, sequence_length, 32)
        
        return features.astype(np.float32), targets.astype(np.float32)
    
    def test_model_initialization(self, sample_config):
        """Test CNN model initialization"""
        model = CNNFeatureExtractor(sample_config)
        
        # Check configuration
        assert model.input_channels == 10
        assert model.filter_sizes == [3, 5, 7]
        assert model.num_filters == 16
        assert model.use_attention is True
        assert model.num_attention_heads == 4
        
        # Check model components
        assert len(model.conv_layers) == 3  # One for each filter size
        assert len(model.batch_norms) == 3
        assert isinstance(model.attention, MultiHeadAttention)
        assert isinstance(model.residual_projection, nn.Linear)
        assert isinstance(model.feature_projection, nn.Linear)
    
    def test_model_forward_pass(self, sample_config, sample_data):
        """Test CNN forward pass with valid shapes"""
        model = CNNFeatureExtractor(sample_config)
        features, _ = sample_data
        
        # Convert to tensor
        x = torch.FloatTensor(features)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model.forward(x)
        
        # Check output shape
        batch_size, input_channels, seq_len = x.shape
        expected_shape = (batch_size, seq_len, sample_config.output_dim)
        assert output.shape == expected_shape
        
        # Check output is not all zeros or NaN
        assert not torch.all(output == 0)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_model_without_attention(self, sample_data):
        """Test CNN model without attention mechanism"""
        config = create_cnn_config(
            input_dim=10,
            output_dim=32,
            use_attention=False,
            device="cpu"
        )
        
        model = CNNFeatureExtractor(config)
        features, _ = sample_data
        
        x = torch.FloatTensor(features)
        
        model.eval()
        with torch.no_grad():
            output = model.forward(x)
        
        # Check output shape
        batch_size, _, seq_len = x.shape
        assert output.shape == (batch_size, seq_len, config.output_dim)
    
    def test_extract_features_method(self, sample_config, sample_data):
        """Test the extract_features convenience method"""
        model = CNNFeatureExtractor(sample_config)
        features, _ = sample_data
        
        # Extract features
        extracted = model.extract_features(features)
        
        # Check output type and shape
        assert isinstance(extracted, np.ndarray)
        batch_size, _, seq_len = features.shape
        expected_shape = (batch_size, seq_len, sample_config.output_dim)
        assert extracted.shape == expected_shape
    
    def test_model_training_pipeline(self, sample_config, sample_data):
        """Test complete training pipeline"""
        model = CNNFeatureExtractor(sample_config)
        features, targets = sample_data
        
        # Create data loaders
        train_loader = create_cnn_data_loader(
            features, targets, batch_size=4, shuffle=True
        )
        val_loader = create_cnn_data_loader(
            features, targets, batch_size=4, shuffle=False
        )
        
        # Train model (short training for testing)
        result = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2
        )
        
        # Check training result
        assert result.epochs_trained == 2
        assert result.train_loss > 0
        assert result.val_loss > 0
        assert model.is_trained is True
    
    def test_model_save_and_load(self, sample_config):
        """Test model saving and loading"""
        model = CNNFeatureExtractor(sample_config)
        
        # Mark as trained
        model.is_trained = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, "test_cnn_model.pth")
            model.save_model(model_path)
            
            # Check files exist
            assert os.path.exists(model_path)
            config_path = model_path.replace('.pth', '_config.json')
            assert os.path.exists(config_path)
            
            # Check config file content
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config['model_type'] == 'CNNFeatureExtractor'
            assert saved_config['input_dim'] == sample_config.input_dim
            assert saved_config['output_dim'] == sample_config.output_dim
            
            # Create new model and load
            new_model = CNNFeatureExtractor(sample_config)
            new_model.load_model(model_path)
            
            # Check loaded model
            assert new_model.is_trained is True
            assert new_model.filter_sizes == model.filter_sizes
            assert new_model.num_filters == model.num_filters
    
    def test_gradient_flow(self, sample_config, sample_data):
        """Test that gradients flow properly through the model"""
        model = CNNFeatureExtractor(sample_config)
        features, targets = sample_data
        
        # Convert to tensors
        x = torch.FloatTensor(features[:2])  # Use small batch
        y = torch.FloatTensor(targets[:2])
        
        # Forward pass
        output = model.forward(x)
        
        # Compute loss
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"
    
    def test_different_filter_sizes(self):
        """Test model with different filter size configurations"""
        test_configs = [
            [3, 5],
            [3, 5, 7, 11, 15],
            [1, 3, 5],
            [7]
        ]
        
        for filter_sizes in test_configs:
            config = create_cnn_config(
                input_dim=5,
                output_dim=16,
                filter_sizes=filter_sizes,
                device="cpu"
            )
            
            model = CNNFeatureExtractor(config)
            
            # Test forward pass
            x = torch.randn(2, 5, 30)
            output = model.forward(x)
            
            assert output.shape == (2, 30, 16)
            assert len(model.conv_layers) == len(filter_sizes)
    
    def test_model_with_different_dimensions(self):
        """Test model with various input/output dimensions"""
        test_cases = [
            (5, 10),   # Small dimensions
            (20, 64),  # Medium dimensions
            (50, 128), # Large dimensions
        ]
        
        for input_dim, output_dim in test_cases:
            config = create_cnn_config(
                input_dim=input_dim,
                output_dim=output_dim,
                device="cpu"
            )
            
            model = CNNFeatureExtractor(config)
            
            # Test with random data
            batch_size = 3
            seq_len = 25
            x = torch.randn(batch_size, input_dim, seq_len)
            
            output = model.forward(x)
            assert output.shape == (batch_size, seq_len, output_dim)


class TestDataLoader:
    """Test cases for CNN data loader utilities"""
    
    def test_create_cnn_data_loader(self):
        """Test CNN data loader creation"""
        # Create sample data
        batch_size = 4
        features = np.random.randn(20, 8, 50).astype(np.float32)
        targets = np.random.randn(20, 50, 16).astype(np.float32)
        
        # Create data loader
        loader = create_cnn_data_loader(
            features, targets, batch_size=batch_size, shuffle=True
        )
        
        # Test data loader
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == batch_size
        
        # Test iteration
        for batch_x, batch_y in loader:
            assert batch_x.shape[0] <= batch_size  # Last batch might be smaller
            assert batch_x.shape[1:] == (8, 50)
            assert batch_y.shape[1:] == (50, 16)
            break  # Just test first batch
    
    def test_data_loader_with_different_shapes(self):
        """Test data loader with various input shapes"""
        test_cases = [
            ((10, 5, 30), (10, 30, 8)),
            ((15, 12, 100), (15, 100, 32)),
            ((8, 20, 75), (8, 75, 64)),
        ]
        
        for feature_shape, target_shape in test_cases:
            features = np.random.randn(*feature_shape).astype(np.float32)
            targets = np.random.randn(*target_shape).astype(np.float32)
            
            loader = create_cnn_data_loader(features, targets, batch_size=3)
            
            # Test first batch
            for batch_x, batch_y in loader:
                assert batch_x.shape[1:] == feature_shape[1:]
                assert batch_y.shape[1:] == target_shape[1:]
                break


class TestCNNConfig:
    """Test cases for CNN configuration utilities"""
    
    def test_create_cnn_config_defaults(self):
        """Test CNN config creation with default values"""
        config = create_cnn_config(input_dim=10, output_dim=32)
        
        # Check basic config
        assert config.model_type == "CNNFeatureExtractor"
        assert config.input_dim == 10
        assert config.output_dim == 32
        
        # Check CNN-specific defaults
        assert config.filter_sizes == [3, 5, 7, 11]
        assert config.num_filters == 64
        assert config.use_attention is True
        assert config.num_attention_heads == 8
        assert config.dropout_rate == 0.3
    
    def test_create_cnn_config_custom(self):
        """Test CNN config creation with custom values"""
        config = create_cnn_config(
            input_dim=15,
            output_dim=48,
            filter_sizes=[2, 4, 6],
            num_filters=32,
            use_attention=False,
            num_attention_heads=4,
            dropout_rate=0.5,
            learning_rate=0.01,
            batch_size=16,
            epochs=50,
            device="cuda"
        )
        
        # Check all custom values
        assert config.input_dim == 15
        assert config.output_dim == 48
        assert config.filter_sizes == [2, 4, 6]
        assert config.num_filters == 32
        assert config.use_attention is False
        assert config.num_attention_heads == 4
        assert config.dropout_rate == 0.5
        assert config.learning_rate == 0.01
        assert config.batch_size == 16
        assert config.epochs == 50
        assert config.device == "cuda"


class TestModelIntegration:
    """Integration tests for CNN model with feature engineering"""
    
    def test_cnn_with_feature_pipeline(self):
        """Test CNN integration with feature engineering pipeline"""
        from src.ml.feature_engineering import TechnicalIndicators
        import pandas as pd
        
        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        high_prices = close_prices + np.abs(np.random.randn(100) * 0.5)
        low_prices = close_prices - np.abs(np.random.randn(100) * 0.5)
        open_prices = close_prices + np.random.randn(100) * 0.1
        volumes = np.random.randint(1000, 10000, 100)
        
        market_data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        # Apply feature engineering
        tech_indicators = TechnicalIndicators()
        features = tech_indicators.fit_transform(market_data)
        
        # Prepare data for CNN (add batch and sequence dimensions)
        # Reshape to (batch_size, channels, sequence_length)
        sequence_length = 20
        num_samples = len(features) - sequence_length + 1
        num_features = features.shape[1]
        
        cnn_input = np.zeros((num_samples, num_features, sequence_length))
        for i in range(num_samples):
            cnn_input[i] = features[i:i+sequence_length].T
        
        # Create CNN model
        config = create_cnn_config(
            input_dim=num_features,
            output_dim=16,
            epochs=2,
            device="cpu"
        )
        
        model = CNNFeatureExtractor(config)
        
        # Test forward pass
        output = model.extract_features(cnn_input)
        
        # Check output shape
        assert output.shape == (num_samples, sequence_length, 16)
        assert not np.any(np.isnan(output))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
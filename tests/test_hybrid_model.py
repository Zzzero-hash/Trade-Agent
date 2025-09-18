"""Tests for CNN+LSTM Hybrid Model

This module contains comprehensive tests for the hybrid model including:
- Model architecture validation
- Multi-task learning functionality
- Ensemble capabilities
- Uncertainty quantification
- End-to-end training and inference
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the hybrid model components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.hybrid_model import (
    CNNLSTMHybridModel,
    HybridModelConfig,
    FeatureFusion,
    UncertaintyQuantification,
    create_hybrid_config,
    create_hybrid_data_loader
)
from ml.base_models import TrainingResult


class TestHybridModelConfig:
    """Test hybrid model configuration"""
    
    def test_config_creation(self):
        """Test creating hybrid model configuration"""
        config = create_hybrid_config(
            input_dim=10,
            sequence_length=50,
            prediction_horizon=5,
            num_classes=3,
            regression_targets=2
        )
        
        assert config.input_dim == 10
        assert config.sequence_length == 50
        assert config.prediction_horizon == 5
        assert config.num_classes == 3
        assert config.regression_targets == 2
        assert config.output_dim == 5  # num_classes + regression_targets
        
    def test_config_defaults(self):
        """Test default configuration values"""
        config = HybridModelConfig(
            model_type="CNNLSTMHybridModel",
            input_dim=10,
            output_dim=4,
            hidden_dims=[256]
        )
        
        assert config.cnn_filter_sizes == [3, 5, 7, 11]
        assert config.cnn_num_filters == 64
        assert config.lstm_hidden_dim == 128
        assert config.num_ensemble_models == 5
        assert config.use_monte_carlo_dropout == True


class TestFeatureFusion:
    """Test feature fusion module"""
    
    def test_feature_fusion_forward(self):
        """Test feature fusion forward pass"""
        batch_size, seq_len = 4, 50
        cnn_dim, lstm_dim, fusion_dim = 64, 128, 256
        
        fusion = FeatureFusion(cnn_dim, lstm_dim, fusion_dim)
        
        cnn_features = torch.randn(batch_size, seq_len, cnn_dim)
        lstm_features = torch.randn(batch_size, seq_len, lstm_dim)
        
        fused = fusion(cnn_features, lstm_features)
        
        assert fused.shape == (batch_size, seq_len, fusion_dim)
        assert not torch.isnan(fused).any()
        assert not torch.isinf(fused).any()
    
    def test_feature_fusion_dimensions(self):
        """Test feature fusion with different dimensions"""
        fusion = FeatureFusion(32, 64, 128)
        
        cnn_features = torch.randn(2, 30, 32)
        lstm_features = torch.randn(2, 30, 64)
        
        fused = fusion(cnn_features, lstm_features)
        
        assert fused.shape == (2, 30, 128)


class TestUncertaintyQuantification:
    """Test uncertainty quantification module"""
    
    def test_uncertainty_forward(self):
        """Test uncertainty quantification forward pass"""
        input_dim, output_dim = 256, 1
        batch_size = 4
        
        uq = UncertaintyQuantification(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = uq(x)
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
    
    def test_uncertainty_prediction(self):
        """Test Monte Carlo dropout uncertainty prediction"""
        input_dim, output_dim = 128, 2
        batch_size = 3
        num_samples = 50
        
        uq = UncertaintyQuantification(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        
        mean_pred, uncertainty = uq.predict_with_uncertainty(x, num_samples)
        
        assert mean_pred.shape == (batch_size, output_dim)
        assert uncertainty.shape == (batch_size, output_dim)
        assert torch.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_uncertainty_reproducibility(self):
        """Test that uncertainty prediction is consistent"""
        torch.manual_seed(42)
        
        input_dim, output_dim = 64, 1
        uq = UncertaintyQuantification(input_dim, output_dim)
        x = torch.randn(2, input_dim)
        
        # Run twice with same seed
        torch.manual_seed(123)
        mean1, unc1 = uq.predict_with_uncertainty(x, 10)
        
        torch.manual_seed(123)
        mean2, unc2 = uq.predict_with_uncertainty(x, 10)
        
        # Should be approximately equal (some randomness from dropout)
        assert torch.allclose(mean1, mean2, atol=1e-2)


class TestCNNLSTMHybridModel:
    """Test CNN+LSTM hybrid model"""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return create_hybrid_config(
            input_dim=8,
            sequence_length=30,
            prediction_horizon=5,
            num_classes=3,
            regression_targets=1,
            feature_fusion_dim=128,
            cnn_num_filters=32,
            lstm_hidden_dim=64,
            num_ensemble_models=3,
            batch_size=4,
            epochs=2,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_model(self, sample_config):
        """Create sample hybrid model for testing"""
        return CNNLSTMHybridModel(sample_config)
    
    def test_model_initialization(self, sample_config):
        """Test model initialization"""
        model = CNNLSTMHybridModel(sample_config)
        
        assert model.config == sample_config
        assert hasattr(model, 'cnn_extractor')
        assert hasattr(model, 'lstm_processor')
        assert hasattr(model, 'feature_fusion')
        assert hasattr(model, 'classification_head')
        assert hasattr(model, 'regression_head')
        assert len(model.ensemble_models) == sample_config.num_ensemble_models
    
    def test_model_forward_pass(self, sample_model, sample_config):
        """Test model forward pass"""
        batch_size = 2
        input_channels = sample_config.input_dim
        seq_len = sample_config.sequence_length
        
        x = torch.randn(batch_size, input_channels, seq_len)
        
        # Test forward pass
        output = sample_model.forward(x)
        
        # Check output structure
        assert isinstance(output, dict)
        assert 'classification_logits' in output
        assert 'classification_probs' in output
        assert 'regression_mean' in output
        assert 'regression_uncertainty' in output
        assert 'ensemble_classification' in output
        assert 'ensemble_regression' in output
        
        # Check output shapes
        assert output['classification_logits'].shape == (batch_size, sample_config.num_classes)
        assert output['classification_probs'].shape == (batch_size, sample_config.num_classes)
        assert output['regression_mean'].shape == (batch_size, sample_config.regression_targets)
        assert output['regression_uncertainty'].shape == (batch_size, sample_config.regression_targets)
        
        # Check probability constraints
        probs = output['classification_probs']
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_model_forward_with_features(self, sample_model, sample_config):
        """Test model forward pass with feature return"""
        batch_size = 2
        x = torch.randn(batch_size, sample_config.input_dim, sample_config.sequence_length)
        
        output = sample_model.forward(x, return_features=True)
        
        # Check that features are returned
        assert 'cnn_features' in output
        assert 'lstm_features' in output
        assert 'lstm_context' in output
        assert 'fused_features' in output
        
        # Check feature shapes
        assert output['cnn_features'].shape[0] == batch_size
        assert output['lstm_features'].shape[0] == batch_size
        assert output['fused_features'].shape[0] == batch_size
    
    def test_model_without_ensemble(self, sample_model, sample_config):
        """Test model forward pass without ensemble"""
        batch_size = 2
        x = torch.randn(batch_size, sample_config.input_dim, sample_config.sequence_length)
        
        output = sample_model.forward(x, use_ensemble=False)
        
        # Ensemble outputs should be same as individual outputs
        assert torch.allclose(output['ensemble_classification'], output['classification_probs'])
        assert torch.allclose(output['ensemble_regression'], output['regression_mean'])
        assert output['ensemble_weights'] is None
    
    def test_loss_computation(self, sample_model, sample_config):
        """Test loss computation"""
        batch_size = 3
        x = torch.randn(batch_size, sample_config.input_dim, sample_config.sequence_length)
        
        # Create targets
        class_targets = torch.randint(0, sample_config.num_classes, (batch_size,))
        reg_targets = torch.randn(batch_size, sample_config.regression_targets)
        
        # Forward pass
        predictions = sample_model.forward(x)
        
        # Compute losses
        losses = sample_model.compute_loss(predictions, class_targets, reg_targets)
        
        # Check loss structure
        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert 'classification_loss' in losses
        assert 'regression_loss' in losses
        assert 'uncertainty_loss' in losses
        assert 'ensemble_class_loss' in losses
        assert 'ensemble_regression_loss' in losses
        
        # Check that losses are scalars and finite
        for loss_name, loss_value in losses.items():
            assert loss_value.dim() == 0  # Scalar
            assert torch.isfinite(loss_value)
            assert loss_value >= 0  # Non-negative
    
    def test_prediction_method(self, sample_model, sample_config):
        """Test prediction method"""
        # Create sample data
        batch_size = 2
        X = np.random.randn(batch_size, sample_config.input_dim, sample_config.sequence_length)
        
        # Make predictions
        predictions = sample_model.predict(X, return_uncertainty=True, use_ensemble=True)
        
        # Check prediction structure
        assert isinstance(predictions, dict)
        assert 'classification_probs' in predictions
        assert 'classification_pred' in predictions
        assert 'regression_pred' in predictions
        assert 'regression_uncertainty' in predictions
        assert 'ensemble_classification' in predictions
        assert 'ensemble_regression' in predictions
        
        # Check shapes
        assert predictions['classification_probs'].shape == (batch_size, sample_config.num_classes)
        assert predictions['classification_pred'].shape == (batch_size,)
        assert predictions['regression_pred'].shape == (batch_size, sample_config.regression_targets)
        
        # Check classification predictions are valid class indices
        class_preds = predictions['classification_pred']
        assert np.all(class_preds >= 0) and np.all(class_preds < sample_config.num_classes)
    
    def test_data_loader_creation(self, sample_config):
        """Test data loader creation"""
        batch_size = 4
        num_samples = 10
        
        # Create sample data
        features = np.random.randn(num_samples, sample_config.input_dim, sample_config.sequence_length)
        class_targets = np.random.randint(0, sample_config.num_classes, num_samples)
        reg_targets = np.random.randn(num_samples, sample_config.regression_targets)
        
        # Create data loader
        dataloader = create_hybrid_data_loader(
            features, class_targets, reg_targets, batch_size=batch_size
        )
        
        # Test data loader
        for batch_x, batch_y_class, batch_y_reg in dataloader:
            assert batch_x.shape[0] <= batch_size
            assert batch_x.shape[1:] == (sample_config.input_dim, sample_config.sequence_length)
            assert batch_y_class.shape[0] <= batch_size
            assert batch_y_reg.shape == (batch_x.shape[0], sample_config.regression_targets)
            break  # Just test first batch
    
    def test_model_training(self, sample_model, sample_config):
        """Test model training"""
        # Create sample training data
        num_samples = 20
        X_train = np.random.randn(num_samples, sample_config.input_dim, sample_config.sequence_length)
        y_class_train = np.random.randint(0, sample_config.num_classes, num_samples)
        y_reg_train = np.random.randn(num_samples, sample_config.regression_targets)
        
        # Create validation data
        num_val = 8
        X_val = np.random.randn(num_val, sample_config.input_dim, sample_config.sequence_length)
        y_class_val = np.random.randint(0, sample_config.num_classes, num_val)
        y_reg_val = np.random.randn(num_val, sample_config.regression_targets)
        
        # Train model
        result = sample_model.fit(
            X_train, y_class_train, y_reg_train,
            X_val, y_class_val, y_reg_val
        )
        
        # Check training result
        assert isinstance(result, TrainingResult)
        assert result.epochs_trained > 0
        assert result.train_loss >= 0
        assert result.val_loss >= 0
        assert sample_model.is_trained
    
    def test_model_save_load(self, sample_model, sample_config):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, "hybrid_model.pth")
            sample_model.save_model(model_path)
            
            # Check files exist
            assert os.path.exists(model_path)
            assert os.path.exists(model_path.replace('.pth', '_config.json'))
            
            # Create new model and load
            new_model = CNNLSTMHybridModel(sample_config)
            new_model.load_model(model_path)
            
            # Test that the loaded model has the same architecture
            assert new_model.config.input_dim == sample_model.config.input_dim
            assert new_model.config.output_dim == sample_model.config.output_dim
            assert new_model.config.num_classes == sample_model.config.num_classes
            assert new_model.config.regression_targets == sample_model.config.regression_targets
            
            # Test that the model can make predictions (shape consistency)
            x = torch.randn(2, sample_config.input_dim, sample_config.sequence_length)
            
            with torch.no_grad():
                output = new_model.forward(x, use_ensemble=False)
                
                # Check output shapes are correct
                assert output['classification_logits'].shape == (2, sample_config.num_classes)
                assert output['regression_mean'].shape == (2, sample_config.regression_targets)
                
                # Check outputs are finite
                assert torch.isfinite(output['classification_logits']).all()
                assert torch.isfinite(output['regression_mean']).all()


class TestHybridModelIntegration:
    """Integration tests for the hybrid model"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Configuration
        config = create_hybrid_config(
            input_dim=12,
            sequence_length=40,
            prediction_horizon=3,
            num_classes=3,
            regression_targets=2,
            batch_size=8,
            epochs=3,
            device="cpu"
        )
        
        # Create model
        model = CNNLSTMHybridModel(config)
        
        # Generate synthetic data
        num_train = 50
        num_test = 20
        
        X_train = np.random.randn(num_train, config.input_dim, config.sequence_length)
        y_class_train = np.random.randint(0, config.num_classes, num_train)
        y_reg_train = np.random.randn(num_train, config.regression_targets)
        
        X_test = np.random.randn(num_test, config.input_dim, config.sequence_length)
        y_class_test = np.random.randint(0, config.num_classes, num_test)
        y_reg_test = np.random.randn(num_test, config.regression_targets)
        
        # Train model
        training_result = model.fit(X_train, y_class_train, y_reg_train)
        
        assert training_result.epochs_trained > 0
        assert model.is_trained
        
        # Make predictions
        predictions = model.predict(X_test, return_uncertainty=True, use_ensemble=True)
        
        # Validate predictions
        assert predictions['classification_probs'].shape == (num_test, config.num_classes)
        assert predictions['regression_pred'].shape == (num_test, config.regression_targets)
        assert predictions['regression_uncertainty'].shape == (num_test, config.regression_targets)
        
        # Check that predictions are reasonable
        class_probs = predictions['classification_probs']
        assert np.all(class_probs >= 0) and np.all(class_probs <= 1)
        assert np.allclose(np.sum(class_probs, axis=1), 1.0, atol=1e-5)
        
        # Check uncertainty is non-negative
        uncertainty = predictions['regression_uncertainty']
        assert np.all(uncertainty >= 0)
    
    def test_ensemble_consistency(self):
        """Test that ensemble predictions are consistent"""
        config = create_hybrid_config(
            input_dim=6,
            sequence_length=20,
            num_ensemble_models=4,
            batch_size=4,
            device="cpu"
        )
        
        model = CNNLSTMHybridModel(config)
        
        # Test data
        X = np.random.randn(5, config.input_dim, config.sequence_length)
        
        # Make predictions multiple times
        pred1 = model.predict(X, use_ensemble=True)
        pred2 = model.predict(X, use_ensemble=True)
        
        # Ensemble predictions should be identical (no randomness in inference)
        np.testing.assert_allclose(
            pred1['ensemble_classification'],
            pred2['ensemble_classification'],
            rtol=1e-5
        )
        np.testing.assert_allclose(
            pred1['ensemble_regression'],
            pred2['ensemble_regression'],
            rtol=1e-5
        )
    
    def test_uncertainty_calibration(self):
        """Test that uncertainty estimates are reasonable"""
        config = create_hybrid_config(
            input_dim=8,
            sequence_length=25,
            use_monte_carlo_dropout=True,
            mc_dropout_samples=20,
            batch_size=4,
            device="cpu"
        )
        
        model = CNNLSTMHybridModel(config)
        
        # Test with same input multiple times
        X = np.random.randn(3, config.input_dim, config.sequence_length)
        
        predictions = model.predict(X, return_uncertainty=True)
        uncertainty = predictions['regression_uncertainty']
        
        # Uncertainty should be positive
        assert np.all(uncertainty > 0)
        
        # Uncertainty should be reasonable (not too large or too small)
        assert np.all(uncertainty < 10.0)  # Not unreasonably large
        assert np.all(uncertainty > 1e-6)  # Not unreasonably small


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
"""
Unit tests for the model factory implementation.

Tests cover model creation, builder registration, checkpoint loading,
and error handling scenarios.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.ml.refactored.model_factory import (
    ModelInterface, ModelBuilder, CNNModelBuilder, LSTMModelBuilder,
    HybridModelBuilder, CNNModelWrapper, LSTMModelWrapper, HybridModelWrapper,
    ModelFactory, create_model, create_from_checkpoint, register_custom_builder,
    get_supported_model_types, create_hybrid_model, create_cnn_model, create_lstm_model
)
from src.ml.refactored.config import (
    ModelType, BaseModelConfig, CNNConfig, LSTMConfig, HybridModelConfig,
    create_hybrid_config, create_cnn_config, create_lstm_config
)
from src.ml.refactored.exceptions import (
    ConfigurationError, ModelLoadError, ValidationError
)


class MockModel:
    """Mock model for testing"""
    
    def __init__(self, config):
        self.config = config
        self.is_trained = False
    
    def build_model(self):
        pass
    
    def fit(self, *args, **kwargs):
        self.is_trained = True
        return {"loss": 0.1}
    
    def predict(self, *args, **kwargs):
        return [0.5, 0.3, 0.2]
    
    def save_model(self, filepath):
        pass
    
    def load_model(self, filepath):
        pass
    
    def load_state_dict(self, state_dict):
        pass


class TestModelBuilder:
    """Test abstract model builder"""
    
    def test_builder_interface(self):
        """Test that ModelBuilder is abstract"""
        with pytest.raises(TypeError):
            ModelBuilder()


class TestCNNModelBuilder:
    """Test CNN model builder"""
    
    def test_can_build_cnn(self):
        """Test CNN builder can build CNN models"""
        builder = CNNModelBuilder()
        assert builder.can_build(ModelType.CNN_ONLY) is True
        assert builder.can_build(ModelType.LSTM_ONLY) is False
        assert builder.can_build(ModelType.CNN_LSTM_HYBRID) is False
    
    def test_get_supported_types(self):
        """Test getting supported types"""
        builder = CNNModelBuilder()
        supported = builder.get_supported_types()
        assert ModelType.CNN_ONLY in supported
        assert len(supported) == 1
    
    @patch('src.ml.refactored.model_factory.CNNFeatureExtractor')
    def test_build_success(self, mock_cnn_class):
        """Test successful CNN model building"""
        mock_model = MockModel(None)
        mock_cnn_class.return_value = mock_model
        
        builder = CNNModelBuilder()
        config = create_cnn_config(input_dim=128)
        
        model = builder.build(config)
        
        assert isinstance(model, CNNModelWrapper)
        assert model.config == config
        mock_cnn_class.assert_called_once()
    
    def test_build_wrong_config_type(self):
        """Test building with wrong config type"""
        builder = CNNModelBuilder()
        config = create_lstm_config(input_dim=128)  # Wrong type
        
        with pytest.raises(ConfigurationError, match="CNNModelBuilder requires CNNConfig"):
            builder.build(config)


class TestLSTMModelBuilder:
    """Test LSTM model builder"""
    
    def test_can_build_lstm(self):
        """Test LSTM builder can build LSTM models"""
        builder = LSTMModelBuilder()
        assert builder.can_build(ModelType.LSTM_ONLY) is True
        assert builder.can_build(ModelType.CNN_ONLY) is False
        assert builder.can_build(ModelType.CNN_LSTM_HYBRID) is False
    
    @patch('src.ml.refactored.model_factory.LSTMTemporalProcessor')
    def test_build_success(self, mock_lstm_class):
        """Test successful LSTM model building"""
        mock_model = MockModel(None)
        mock_lstm_class.return_value = mock_model
        
        builder = LSTMModelBuilder()
        config = create_lstm_config(input_dim=128)
        
        model = builder.build(config)
        
        assert isinstance(model, LSTMModelWrapper)
        assert model.config == config
        mock_lstm_class.assert_called_once()


class TestHybridModelBuilder:
    """Test hybrid model builder"""
    
    def test_can_build_hybrid(self):
        """Test hybrid builder can build hybrid models"""
        builder = HybridModelBuilder()
        assert builder.can_build(ModelType.CNN_LSTM_HYBRID) is True
        assert builder.can_build(ModelType.CNN_ONLY) is False
        assert builder.can_build(ModelType.LSTM_ONLY) is False
    
    @patch('src.ml.refactored.model_factory.CNNLSTMHybridModel')
    def test_build_success(self, mock_hybrid_class):
        """Test successful hybrid model building"""
        mock_model = MockModel(None)
        mock_hybrid_class.return_value = mock_model
        
        builder = HybridModelBuilder()
        config = create_hybrid_config(input_dim=128)
        
        model = builder.build(config)
        
        assert isinstance(model, HybridModelWrapper)
        assert model.config == config
        mock_hybrid_class.assert_called_once_with(config)


class TestModelWrappers:
    """Test model wrapper classes"""
    
    def test_cnn_wrapper_interface(self):
        """Test CNN wrapper implements ModelInterface"""
        mock_model = MockModel(None)
        config = create_cnn_config(input_dim=128)
        wrapper = CNNModelWrapper(mock_model, config)
        
        assert isinstance(wrapper, ModelInterface)
        assert wrapper.get_config() == config
        
        # Test method delegation
        wrapper.build_model()
        result = wrapper.fit()
        assert result == {"loss": 0.1}
        
        prediction = wrapper.predict()
        assert prediction == [0.5, 0.3, 0.2]
    
    def test_lstm_wrapper_interface(self):
        """Test LSTM wrapper implements ModelInterface"""
        mock_model = MockModel(None)
        config = create_lstm_config(input_dim=128)
        wrapper = LSTMModelWrapper(mock_model, config)
        
        assert isinstance(wrapper, ModelInterface)
        assert wrapper.get_config() == config
    
    def test_hybrid_wrapper_interface(self):
        """Test hybrid wrapper implements ModelInterface"""
        mock_model = MockModel(None)
        config = create_hybrid_config(input_dim=128)
        wrapper = HybridModelWrapper(mock_model, config)
        
        assert isinstance(wrapper, ModelInterface)
        assert wrapper.get_config() == config


class TestModelFactory:
    """Test model factory"""
    
    def test_factory_initialization(self):
        """Test factory initializes with default builders"""
        factory = ModelFactory()
        supported_types = factory.get_supported_types()
        
        assert ModelType.CNN_ONLY in supported_types
        assert ModelType.LSTM_ONLY in supported_types
        assert ModelType.CNN_LSTM_HYBRID in supported_types
    
    def test_register_builder(self):
        """Test registering custom builder"""
        factory = ModelFactory()
        
        # Create mock builder
        mock_builder = Mock(spec=ModelBuilder)
        mock_builder.get_supported_types.return_value = [ModelType.TRANSFORMER]
        
        factory.register_builder(mock_builder)
        
        assert ModelType.TRANSFORMER in factory.get_supported_types()
    
    def test_register_builder_override_warning(self, caplog):
        """Test warning when overriding existing builder"""
        factory = ModelFactory()
        
        # Create mock builder for existing type
        mock_builder = Mock(spec=ModelBuilder)
        mock_builder.get_supported_types.return_value = [ModelType.CNN_ONLY]
        
        factory.register_builder(mock_builder)
        
        assert "Overriding existing builder" in caplog.text
    
    @patch('src.ml.refactored.model_factory.CNNFeatureExtractor')
    def test_create_model_with_config_object(self, mock_cnn_class):
        """Test creating model with config object"""
        mock_model = MockModel(None)
        mock_cnn_class.return_value = mock_model
        
        factory = ModelFactory()
        config = create_cnn_config(input_dim=128)
        
        model = factory.create_model(ModelType.CNN_ONLY, config)
        
        assert isinstance(model, CNNModelWrapper)
    
    @patch('src.ml.refactored.model_factory.CNNFeatureExtractor')
    def test_create_model_with_dict_config(self, mock_cnn_class):
        """Test creating model with dictionary config"""
        mock_model = MockModel(None)
        mock_cnn_class.return_value = mock_model
        
        factory = ModelFactory()
        config_dict = {'input_dim': 128, 'num_filters': 64}
        
        model = factory.create_model(ModelType.CNN_ONLY, config_dict)
        
        assert isinstance(model, CNNModelWrapper)
    
    def test_create_model_string_type(self):
        """Test creating model with string model type"""
        factory = ModelFactory()
        
        with patch('src.ml.refactored.model_factory.CNNFeatureExtractor') as mock_cnn:
            mock_cnn.return_value = MockModel(None)
            
            model = factory.create_model("cnn_only", {'input_dim': 128})
            assert isinstance(model, CNNModelWrapper)
    
    def test_create_model_unknown_type(self):
        """Test error with unknown model type"""
        factory = ModelFactory()
        
        with pytest.raises(ConfigurationError, match="Unknown model type"):
            factory.create_model("unknown_type", {'input_dim': 128})
    
    def test_create_model_no_builder(self):
        """Test error when no builder is registered"""
        factory = ModelFactory()
        factory._builders.clear()  # Remove all builders
        
        with pytest.raises(ConfigurationError, match="No builder registered"):
            factory.create_model(ModelType.CNN_ONLY, {'input_dim': 128})
    
    def test_create_from_checkpoint_success(self):
        """Test successful checkpoint loading"""
        # Create temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        # Mock checkpoint data
        checkpoint_data = {
            'model_type': 'cnn_only',
            'config': create_cnn_config(input_dim=128),
            'model_state_dict': {'layer.weight': torch.randn(10, 128)},
            'is_trained': True
        }
        
        with patch('torch.load') as mock_load:
            mock_load.return_value = checkpoint_data
            
            with patch('src.ml.refactored.model_factory.CNNFeatureExtractor') as mock_cnn:
                mock_model = MockModel(None)
                mock_model.load_state_dict = Mock()
                mock_cnn.return_value = mock_model
                
                factory = ModelFactory()
                model = factory.create_from_checkpoint(checkpoint_path)
                
                assert isinstance(model, CNNModelWrapper)
                mock_model.load_state_dict.assert_called_once()
    
    def test_create_from_checkpoint_file_not_found(self):
        """Test error when checkpoint file doesn't exist"""
        factory = ModelFactory()
        
        with pytest.raises(ModelLoadError, match="Checkpoint file not found"):
            factory.create_from_checkpoint("nonexistent_file.pth")
    
    def test_create_from_checkpoint_missing_model_type(self):
        """Test error when checkpoint missing model_type"""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        checkpoint_data = {
            'config': create_cnn_config(input_dim=128),
            'model_state_dict': {}
        }
        
        with patch('torch.load') as mock_load:
            mock_load.return_value = checkpoint_data
            
            factory = ModelFactory()
            
            with pytest.raises(ModelLoadError, match="missing 'model_type' field"):
                factory.create_from_checkpoint(checkpoint_path)
    
    def test_create_from_checkpoint_missing_config(self):
        """Test error when checkpoint missing config"""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        checkpoint_data = {
            'model_type': 'cnn_only',
            'model_state_dict': {}
        }
        
        with patch('torch.load') as mock_load:
            mock_load.return_value = checkpoint_data
            
            factory = ModelFactory()
            
            with pytest.raises(ModelLoadError, match="missing 'config' field"):
                factory.create_from_checkpoint(checkpoint_path)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('src.ml.refactored.model_factory._global_factory')
    def test_create_model_function(self, mock_factory):
        """Test global create_model function"""
        mock_model = Mock()
        mock_factory.create_model.return_value = mock_model
        
        result = create_model(ModelType.CNN_ONLY, {'input_dim': 128})
        
        assert result == mock_model
        mock_factory.create_model.assert_called_once_with(ModelType.CNN_ONLY, {'input_dim': 128})
    
    @patch('src.ml.refactored.model_factory._global_factory')
    def test_create_from_checkpoint_function(self, mock_factory):
        """Test global create_from_checkpoint function"""
        mock_model = Mock()
        mock_factory.create_from_checkpoint.return_value = mock_model
        
        result = create_from_checkpoint("model.pth")
        
        assert result == mock_model
        mock_factory.create_from_checkpoint.assert_called_once_with("model.pth", None)
    
    @patch('src.ml.refactored.model_factory._global_factory')
    def test_register_custom_builder_function(self, mock_factory):
        """Test global register_custom_builder function"""
        mock_builder = Mock(spec=ModelBuilder)
        
        register_custom_builder(mock_builder)
        
        mock_factory.register_builder.assert_called_once_with(mock_builder)
    
    @patch('src.ml.refactored.model_factory._global_factory')
    def test_get_supported_model_types_function(self, mock_factory):
        """Test global get_supported_model_types function"""
        mock_types = [ModelType.CNN_ONLY, ModelType.LSTM_ONLY]
        mock_factory.get_supported_types.return_value = mock_types
        
        result = get_supported_model_types()
        
        assert result == mock_types
        mock_factory.get_supported_types.assert_called_once()
    
    @patch('src.ml.refactored.model_factory.create_model')
    def test_create_hybrid_model_function(self, mock_create):
        """Test create_hybrid_model convenience function"""
        mock_model = Mock()
        mock_create.return_value = mock_model
        
        result = create_hybrid_model(input_dim=128, feature_fusion_dim=512)
        
        assert result == mock_model
        mock_create.assert_called_once()
        
        # Check that the correct model type was used
        call_args = mock_create.call_args
        assert call_args[0][0] == ModelType.CNN_LSTM_HYBRID
        assert isinstance(call_args[0][1], HybridModelConfig)
    
    @patch('src.ml.refactored.model_factory.create_model')
    def test_create_cnn_model_function(self, mock_create):
        """Test create_cnn_model convenience function"""
        mock_model = Mock()
        mock_create.return_value = mock_model
        
        result = create_cnn_model(input_dim=128, num_filters=64)
        
        assert result == mock_model
        mock_create.assert_called_once()
        
        # Check that the correct model type was used
        call_args = mock_create.call_args
        assert call_args[0][0] == ModelType.CNN_ONLY
        assert isinstance(call_args[0][1], CNNConfig)
    
    @patch('src.ml.refactored.model_factory.create_model')
    def test_create_lstm_model_function(self, mock_create):
        """Test create_lstm_model convenience function"""
        mock_model = Mock()
        mock_create.return_value = mock_model
        
        result = create_lstm_model(input_dim=128, hidden_dim=256)
        
        assert result == mock_model
        mock_create.assert_called_once()
        
        # Check that the correct model type was used
        call_args = mock_create.call_args
        assert call_args[0][0] == ModelType.LSTM_ONLY
        assert isinstance(call_args[0][1], LSTMConfig)


class TestValidation:
    """Test input validation"""
    
    def test_invalid_builder_type(self):
        """Test registering invalid builder type"""
        factory = ModelFactory()
        
        with pytest.raises(ValidationError, match="must be of type ModelBuilder"):
            factory.register_builder("not_a_builder")
    
    def test_invalid_model_type(self):
        """Test creating model with invalid type"""
        factory = ModelFactory()
        
        with pytest.raises(ValidationError, match="must be of type ModelType"):
            factory.create_model(123, {'input_dim': 128})
    
    def test_invalid_config_type(self):
        """Test creating model with invalid config"""
        factory = ModelFactory()
        
        with pytest.raises(ValidationError, match="must be of type BaseModelConfig"):
            factory.create_model(ModelType.CNN_ONLY, "invalid_config")


if __name__ == "__main__":
    pytest.main([__file__])
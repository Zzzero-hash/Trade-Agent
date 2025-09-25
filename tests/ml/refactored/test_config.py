"""
Unit tests for the improved configuration management module.

Tests cover validation, error handling, file I/O, and type safety.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.ml.refactored.config import (
    ModelType, OptimizerType, SchedulerType,
    ConfigurationError, ValidationError,
    BaseModelConfig, CNNConfig, LSTMConfig, HybridModelConfig,
    ConfigManager, ModelDefaults,
    create_default_config, create_hybrid_config, create_cnn_config, create_lstm_config
)


class TestModelDefaults:
    """Test model defaults class"""
    
    def test_defaults_are_reasonable(self):
        """Test that default values are reasonable"""
        assert 0 < ModelDefaults.DROPOUT_RATE < 1
        assert 0 < ModelDefaults.LEARNING_RATE < 1
        assert ModelDefaults.BATCH_SIZE > 0
        assert ModelDefaults.SEQUENCE_LENGTH > 0
        assert ModelDefaults.EPOCHS > 0
        assert len(ModelDefaults.CNN_FILTER_SIZES) > 0
        assert all(size > 0 for size in ModelDefaults.CNN_FILTER_SIZES)


class TestBaseModelConfig:
    """Test base model configuration"""
    
    def test_valid_config_creation(self):
        """Test creating valid configuration"""
        config = BaseModelConfig(
            model_type=ModelType.CNN_ONLY,
            input_dim=128
        )
        
        assert config.model_type == ModelType.CNN_ONLY
        assert config.input_dim == 128
        assert config.learning_rate == ModelDefaults.LEARNING_RATE
        assert config.batch_size == ModelDefaults.BATCH_SIZE
    
    def test_invalid_input_dim(self):
        """Test validation of input_dim"""
        with pytest.raises(ValidationError, match="input_dim must be positive"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=0
            )
        
        with pytest.raises(ValidationError, match="input_dim must be positive"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=-10
            )
    
    def test_invalid_learning_rate(self):
        """Test validation of learning_rate"""
        with pytest.raises(ValidationError, match="learning_rate must be between 0 and 1"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                learning_rate=0
            )
        
        with pytest.raises(ValidationError, match="learning_rate must be between 0 and 1"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                learning_rate=1.5
            )
    
    def test_invalid_batch_size(self):
        """Test validation of batch_size"""
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                batch_size=0
            )
    
    def test_invalid_dropout_rate(self):
        """Test validation of dropout_rate"""
        with pytest.raises(ValidationError, match="dropout_rate must be between 0 and 1"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                dropout_rate=-0.1
            )
        
        with pytest.raises(ValidationError, match="dropout_rate must be between 0 and 1"):
            BaseModelConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                dropout_rate=1.0
            )


class TestCNNConfig:
    """Test CNN configuration"""
    
    def test_valid_cnn_config(self):
        """Test creating valid CNN configuration"""
        config = CNNConfig(
            model_type=ModelType.CNN_ONLY,
            input_dim=128,
            num_filters=64,
            filter_sizes=[3, 5, 7]
        )
        
        assert config.num_filters == 64
        assert config.filter_sizes == [3, 5, 7]
        assert config.use_attention is True
    
    def test_invalid_num_filters(self):
        """Test validation of num_filters"""
        with pytest.raises(ValidationError, match="num_filters must be positive"):
            CNNConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                num_filters=0
            )
    
    def test_empty_filter_sizes(self):
        """Test validation of filter_sizes"""
        with pytest.raises(ValidationError, match="filter_sizes cannot be empty"):
            CNNConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                filter_sizes=[]
            )
    
    def test_invalid_filter_sizes(self):
        """Test validation of individual filter sizes"""
        with pytest.raises(ValidationError, match="All filter sizes must be positive"):
            CNNConfig(
                model_type=ModelType.CNN_ONLY,
                input_dim=128,
                filter_sizes=[3, 0, 7]
            )


class TestLSTMConfig:
    """Test LSTM configuration"""
    
    def test_valid_lstm_config(self):
        """Test creating valid LSTM configuration"""
        config = LSTMConfig(
            model_type=ModelType.LSTM_ONLY,
            input_dim=128,
            hidden_dim=256,
            num_layers=2
        )
        
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.bidirectional is True
    
    def test_invalid_hidden_dim(self):
        """Test validation of hidden_dim"""
        with pytest.raises(ValidationError, match="hidden_dim must be positive"):
            LSTMConfig(
                model_type=ModelType.LSTM_ONLY,
                input_dim=128,
                hidden_dim=0
            )
    
    def test_invalid_num_layers(self):
        """Test validation of num_layers"""
        with pytest.raises(ValidationError, match="num_layers must be positive"):
            LSTMConfig(
                model_type=ModelType.LSTM_ONLY,
                input_dim=128,
                num_layers=0
            )


class TestHybridModelConfig:
    """Test hybrid model configuration"""
    
    def test_valid_hybrid_config(self):
        """Test creating valid hybrid configuration"""
        config = HybridModelConfig(
            model_type=ModelType.CNN_LSTM_HYBRID,
            input_dim=128,
            feature_fusion_dim=512,
            num_classes=3,
            regression_targets=2
        )
        
        assert config.feature_fusion_dim == 512
        assert config.num_classes == 3
        assert config.regression_targets == 2
        assert config.classification_weight == ModelDefaults.CLASSIFICATION_WEIGHT
        assert config.regression_weight == ModelDefaults.REGRESSION_WEIGHT
    
    def test_invalid_feature_fusion_dim(self):
        """Test validation of feature_fusion_dim"""
        with pytest.raises(ValidationError, match="feature_fusion_dim must be positive"):
            HybridModelConfig(
                model_type=ModelType.CNN_LSTM_HYBRID,
                input_dim=128,
                feature_fusion_dim=0
            )
    
    def test_invalid_num_classes(self):
        """Test validation of num_classes"""
        with pytest.raises(ValidationError, match="num_classes must be positive"):
            HybridModelConfig(
                model_type=ModelType.CNN_LSTM_HYBRID,
                input_dim=128,
                num_classes=0
            )
    
    def test_invalid_weights(self):
        """Test validation of classification and regression weights"""
        with pytest.raises(ValidationError, match="classification_weight must be between 0 and 1"):
            HybridModelConfig(
                model_type=ModelType.CNN_LSTM_HYBRID,
                input_dim=128,
                classification_weight=-0.1
            )
        
        with pytest.raises(ValidationError, match="regression_weight must be between 0 and 1"):
            HybridModelConfig(
                model_type=ModelType.CNN_LSTM_HYBRID,
                input_dim=128,
                regression_weight=1.5
            )
    
    def test_weight_sum_warning(self, caplog):
        """Test warning when weights don't sum to 1"""
        HybridModelConfig(
            model_type=ModelType.CNN_LSTM_HYBRID,
            input_dim=128,
            classification_weight=0.3,
            regression_weight=0.5
        )
        
        assert "Classification weight" in caplog.text
        assert "regression weight" in caplog.text


class TestConfigManager:
    """Test configuration manager"""
    
    def test_load_from_yaml_success(self):
        """Test successful YAML loading"""
        yaml_content = """
        model_type: cnn_lstm_hybrid
        input_dim: 128
        learning_rate: 0.001
        batch_size: 32
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config_dict = ConfigManager.load_from_yaml(f.name)
            
            assert config_dict['model_type'] == 'cnn_lstm_hybrid'
            assert config_dict['input_dim'] == 128
            assert config_dict['learning_rate'] == 0.001
            assert config_dict['batch_size'] == 32
    
    def test_load_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist"""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            ConfigManager.load_from_yaml("nonexistent_file.yaml")
    
    def test_load_from_yaml_invalid_yaml(self):
        """Test error with invalid YAML"""
        invalid_yaml = """
        model_type: cnn_lstm_hybrid
        input_dim: 128
        invalid_yaml: [unclosed bracket
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                ConfigManager.load_from_yaml(f.name)
    
    def test_load_from_yaml_empty_file(self):
        """Test error with empty YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            
            with pytest.raises(ConfigurationError, match="Empty configuration file"):
                ConfigManager.load_from_yaml(f.name)
    
    def test_save_to_yaml_success(self):
        """Test successful YAML saving"""
        config = BaseModelConfig(
            model_type=ModelType.CNN_ONLY,
            input_dim=128,
            learning_rate=0.001
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            ConfigManager.save_to_yaml(config, f.name)
            
            # Load back and verify
            with open(f.name, 'r') as read_f:
                saved_data = yaml.safe_load(read_f)
            
            assert saved_data['model_type'] == 'cnn_only'
            assert saved_data['input_dim'] == 128
            assert saved_data['learning_rate'] == 0.001
    
    def test_create_config_hybrid(self):
        """Test creating hybrid config from dictionary"""
        config_dict = {
            'input_dim': 128,
            'feature_fusion_dim': 512,
            'num_classes': 3
        }
        
        config = ConfigManager.create_config('cnn_lstm_hybrid', config_dict)
        
        assert isinstance(config, HybridModelConfig)
        assert config.model_type == ModelType.CNN_LSTM_HYBRID
        assert config.input_dim == 128
        assert config.feature_fusion_dim == 512
    
    def test_create_config_unknown_type(self):
        """Test error with unknown model type"""
        with pytest.raises(ConfigurationError, match="Unknown model type"):
            ConfigManager.create_config('unknown_model', {'input_dim': 128})
    
    def test_create_config_invalid_parameters(self):
        """Test error with invalid parameters"""
        with pytest.raises(ConfigurationError, match="Invalid configuration parameters"):
            ConfigManager.create_config('cnn_only', {'invalid_param': 'value'})


class TestConvenienceFunctions:
    """Test convenience functions for config creation"""
    
    def test_create_default_config(self):
        """Test creating default configuration"""
        config = create_default_config(ModelType.CNN_ONLY, input_dim=128)
        
        assert isinstance(config, CNNConfig)
        assert config.model_type == ModelType.CNN_ONLY
        assert config.input_dim == 128
    
    def test_create_hybrid_config(self):
        """Test creating hybrid configuration"""
        config = create_hybrid_config(input_dim=128, feature_fusion_dim=512)
        
        assert isinstance(config, HybridModelConfig)
        assert config.input_dim == 128
        assert config.feature_fusion_dim == 512
    
    def test_create_cnn_config(self):
        """Test creating CNN configuration"""
        config = create_cnn_config(input_dim=128, num_filters=128)
        
        assert isinstance(config, CNNConfig)
        assert config.input_dim == 128
        assert config.num_filters == 128
    
    def test_create_lstm_config(self):
        """Test creating LSTM configuration"""
        config = create_lstm_config(input_dim=128, hidden_dim=256)
        
        assert isinstance(config, LSTMConfig)
        assert config.input_dim == 128
        assert config.hidden_dim == 256


class TestEnumTypes:
    """Test enum type definitions"""
    
    def test_model_type_enum(self):
        """Test ModelType enum"""
        assert ModelType.CNN_LSTM_HYBRID.value == "cnn_lstm_hybrid"
        assert ModelType.CNN_ONLY.value == "cnn_only"
        assert ModelType.LSTM_ONLY.value == "lstm_only"
    
    def test_optimizer_type_enum(self):
        """Test OptimizerType enum"""
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.SGD.value == "sgd"
    
    def test_scheduler_type_enum(self):
        """Test SchedulerType enum"""
        assert SchedulerType.COSINE.value == "cosine"
        assert SchedulerType.STEP.value == "step"
        assert SchedulerType.NONE.value == "none"


if __name__ == "__main__":
    pytest.main([__file__])
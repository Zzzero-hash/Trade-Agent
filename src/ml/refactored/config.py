"""
Improved Configuration Management with Type Hints and Validation

This module provides centralized configuration management with proper type hints,
validation, and error handling for the ML trading platform.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types"""
    CNN_LSTM_HYBRID = "cnn_lstm_hybrid"
    CNN_ONLY = "cnn_only"
    LSTM_ONLY = "lstm_only"
    TRANSFORMER = "transformer"


class OptimizerType(Enum):
    """Supported optimizer types"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Supported scheduler types"""
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    NONE = "none"


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class ValidationError(ConfigurationError):
    """Custom exception for validation errors"""
    pass


@dataclass
class ModelDefaults:
    """Centralized default values with clear documentation"""
    
    # Core architecture defaults
    DROPOUT_RATE: float = 0.3
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    SEQUENCE_LENGTH: int = 60
    PREDICTION_HORIZON: int = 10
    
    # Training defaults
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 25
    GRADIENT_CLIP_NORM: float = 1.0
    
    # CNN defaults
    CNN_NUM_FILTERS: int = 64
    CNN_FILTER_SIZES: List[int] = field(default_factory=lambda: [3, 5, 7, 11])
    CNN_ATTENTION_HEADS: int = 8
    
    # LSTM defaults
    LSTM_HIDDEN_DIM: int = 128
    LSTM_NUM_LAYERS: int = 3
    
    # Hybrid model defaults
    FEATURE_FUSION_DIM: int = 256
    NUM_CLASSES: int = 3  # Buy, Hold, Sell
    REGRESSION_TARGETS: int = 2  # Price prediction + volatility
    
    # Ensemble defaults
    NUM_ENSEMBLE_MODELS: int = 5
    ENSEMBLE_DROPOUT_RATE: float = 0.1
    
    # Uncertainty quantification defaults
    MC_DROPOUT_SAMPLES: int = 100
    
    # Multi-task learning weights
    CLASSIFICATION_WEIGHT: float = 0.4
    REGRESSION_WEIGHT: float = 0.6


@dataclass
class BaseModelConfig:
    """Base configuration class with common parameters"""
    
    # Required parameters
    model_type: ModelType
    input_dim: int
    
    # Core architecture
    sequence_length: int = ModelDefaults.SEQUENCE_LENGTH
    prediction_horizon: int = ModelDefaults.PREDICTION_HORIZON
    
    # Training parameters
    learning_rate: float = ModelDefaults.LEARNING_RATE
    batch_size: int = ModelDefaults.BATCH_SIZE
    epochs: int = ModelDefaults.EPOCHS
    dropout_rate: float = ModelDefaults.DROPOUT_RATE
    
    # Optimization
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    scheduler_type: SchedulerType = SchedulerType.COSINE
    gradient_clip_norm: float = ModelDefaults.GRADIENT_CLIP_NORM
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    random_seed: int = 42
    
    # Early stopping
    early_stopping_patience: int = ModelDefaults.EARLY_STOPPING_PATIENCE
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.input_dim <= 0:
            raise ValidationError("input_dim must be positive")
        
        if self.sequence_length <= 0:
            raise ValidationError("sequence_length must be positive")
        
        if self.prediction_horizon <= 0:
            raise ValidationError("prediction_horizon must be positive")
        
        if not 0 < self.learning_rate < 1:
            raise ValidationError("learning_rate must be between 0 and 1")
        
        if self.batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        
        if self.epochs <= 0:
            raise ValidationError("epochs must be positive")
        
        if not 0 <= self.dropout_rate < 1:
            raise ValidationError("dropout_rate must be between 0 and 1")
        
        if self.gradient_clip_norm <= 0:
            raise ValidationError("gradient_clip_norm must be positive")


@dataclass
class CNNConfig(BaseModelConfig):
    """Configuration for CNN models"""
    
    # CNN-specific parameters
    num_filters: int = ModelDefaults.CNN_NUM_FILTERS
    filter_sizes: List[int] = field(default_factory=lambda: ModelDefaults.CNN_FILTER_SIZES.copy())
    use_attention: bool = True
    num_attention_heads: int = ModelDefaults.CNN_ATTENTION_HEADS
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_cnn_config()
    
    def _validate_cnn_config(self) -> None:
        """Validate CNN-specific configuration"""
        if self.num_filters <= 0:
            raise ValidationError("num_filters must be positive")
        
        if not self.filter_sizes:
            raise ValidationError("filter_sizes cannot be empty")
        
        if any(size <= 0 for size in self.filter_sizes):
            raise ValidationError("All filter sizes must be positive")
        
        if self.num_attention_heads <= 0:
            raise ValidationError("num_attention_heads must be positive")


@dataclass
class LSTMConfig(BaseModelConfig):
    """Configuration for LSTM models"""
    
    # LSTM-specific parameters
    hidden_dim: int = ModelDefaults.LSTM_HIDDEN_DIM
    num_layers: int = ModelDefaults.LSTM_NUM_LAYERS
    bidirectional: bool = True
    use_attention: bool = True
    use_skip_connections: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_lstm_config()
    
    def _validate_lstm_config(self) -> None:
        """Validate LSTM-specific configuration"""
        if self.hidden_dim <= 0:
            raise ValidationError("hidden_dim must be positive")
        
        if self.num_layers <= 0:
            raise ValidationError("num_layers must be positive")


@dataclass
class HybridModelConfig(BaseModelConfig):
    """Configuration for CNN+LSTM hybrid models"""
    
    # CNN configuration
    cnn_num_filters: int = ModelDefaults.CNN_NUM_FILTERS
    cnn_filter_sizes: List[int] = field(default_factory=lambda: ModelDefaults.CNN_FILTER_SIZES.copy())
    cnn_use_attention: bool = True
    cnn_attention_heads: int = ModelDefaults.CNN_ATTENTION_HEADS
    
    # LSTM configuration
    lstm_hidden_dim: int = ModelDefaults.LSTM_HIDDEN_DIM
    lstm_num_layers: int = ModelDefaults.LSTM_NUM_LAYERS
    lstm_bidirectional: bool = True
    lstm_use_attention: bool = True
    lstm_use_skip_connections: bool = True
    
    # Hybrid configuration
    feature_fusion_dim: int = ModelDefaults.FEATURE_FUSION_DIM
    
    # Multi-task configuration
    num_classes: int = ModelDefaults.NUM_CLASSES
    regression_targets: int = ModelDefaults.REGRESSION_TARGETS
    
    # Ensemble configuration
    num_ensemble_models: int = ModelDefaults.NUM_ENSEMBLE_MODELS
    ensemble_dropout_rate: float = ModelDefaults.ENSEMBLE_DROPOUT_RATE
    
    # Uncertainty quantification
    use_monte_carlo_dropout: bool = True
    mc_dropout_samples: int = ModelDefaults.MC_DROPOUT_SAMPLES
    
    # Multi-task learning weights
    classification_weight: float = ModelDefaults.CLASSIFICATION_WEIGHT
    regression_weight: float = ModelDefaults.REGRESSION_WEIGHT
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_hybrid_config()
    
    def _validate_hybrid_config(self) -> None:
        """Validate hybrid model configuration"""
        # CNN validation
        if self.cnn_num_filters <= 0:
            raise ValidationError("cnn_num_filters must be positive")
        
        if not self.cnn_filter_sizes:
            raise ValidationError("cnn_filter_sizes cannot be empty")
        
        if any(size <= 0 for size in self.cnn_filter_sizes):
            raise ValidationError("All CNN filter sizes must be positive")
        
        # LSTM validation
        if self.lstm_hidden_dim <= 0:
            raise ValidationError("lstm_hidden_dim must be positive")
        
        if self.lstm_num_layers <= 0:
            raise ValidationError("lstm_num_layers must be positive")
        
        # Hybrid validation
        if self.feature_fusion_dim <= 0:
            raise ValidationError("feature_fusion_dim must be positive")
        
        if self.num_classes <= 0:
            raise ValidationError("num_classes must be positive")
        
        if self.regression_targets <= 0:
            raise ValidationError("regression_targets must be positive")
        
        if self.num_ensemble_models <= 0:
            raise ValidationError("num_ensemble_models must be positive")
        
        if not 0 <= self.ensemble_dropout_rate < 1:
            raise ValidationError("ensemble_dropout_rate must be between 0 and 1")
        
        if self.mc_dropout_samples <= 0:
            raise ValidationError("mc_dropout_samples must be positive")
        
        # Weight validation
        if not 0 <= self.classification_weight <= 1:
            raise ValidationError("classification_weight must be between 0 and 1")
        
        if not 0 <= self.regression_weight <= 1:
            raise ValidationError("regression_weight must be between 0 and 1")
        
        if abs(self.classification_weight + self.regression_weight - 1.0) > 1e-6:
            logger.warning(
                f"Classification weight ({self.classification_weight}) + "
                f"regression weight ({self.regression_weight}) != 1.0"
            )


class ConfigManager:
    """Configuration manager with file I/O and validation"""
    
    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with error handling"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                raise ConfigurationError(f"Empty configuration file: {config_path}")
            
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config_dict
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}") from e
    
    @staticmethod
    def save_to_yaml(config: BaseModelConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file with error handling"""
        config_path = Path(config_path)
        
        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = {
                'model_type': config.model_type.value,
                **{k: v for k, v in config.__dict__.items() if k != 'model_type'}
            }
            
            # Handle enum values
            for key, value in config_dict.items():
                if isinstance(value, Enum):
                    config_dict[key] = value.value
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Successfully saved configuration to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}") from e
    
    @staticmethod
    def create_config(model_type: str, config_dict: Dict[str, Any]) -> BaseModelConfig:
        """Create configuration object from dictionary with validation"""
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            raise ConfigurationError(f"Unknown model type: {model_type}")
        
        # Convert string enums back to enum objects
        if 'optimizer_type' in config_dict:
            config_dict['optimizer_type'] = OptimizerType(config_dict['optimizer_type'])
        
        if 'scheduler_type' in config_dict:
            config_dict['scheduler_type'] = SchedulerType(config_dict['scheduler_type'])
        
        # Create appropriate config class
        config_dict['model_type'] = model_type_enum
        
        try:
            if model_type_enum == ModelType.CNN_LSTM_HYBRID:
                return HybridModelConfig(**config_dict)
            elif model_type_enum == ModelType.CNN_ONLY:
                return CNNConfig(**config_dict)
            elif model_type_enum == ModelType.LSTM_ONLY:
                return LSTMConfig(**config_dict)
            else:
                return BaseModelConfig(**config_dict)
                
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration parameters: {e}") from e


def create_default_config(
    model_type: ModelType,
    input_dim: int,
    **overrides
) -> BaseModelConfig:
    """Create default configuration with optional overrides"""
    
    base_config = {
        'model_type': model_type,
        'input_dim': input_dim,
        **overrides
    }
    
    return ConfigManager.create_config(model_type.value, base_config)


# Convenience functions for common configurations
def create_hybrid_config(input_dim: int, **kwargs) -> HybridModelConfig:
    """Create hybrid model configuration with defaults"""
    return create_default_config(ModelType.CNN_LSTM_HYBRID, input_dim, **kwargs)


def create_cnn_config(input_dim: int, **kwargs) -> CNNConfig:
    """Create CNN model configuration with defaults"""
    return create_default_config(ModelType.CNN_ONLY, input_dim, **kwargs)


def create_lstm_config(input_dim: int, **kwargs) -> LSTMConfig:
    """Create LSTM model configuration with defaults"""
    return create_default_config(ModelType.LSTM_ONLY, input_dim, **kwargs)
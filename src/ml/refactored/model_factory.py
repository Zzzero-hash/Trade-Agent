"""
Model Factory for creating ML models with proper abstraction and extensibility.

This module implements the Factory pattern to centralize model creation,
making it easy to add new model types and manage model instantiation.
"""

from typing import Dict, Any, Type, Optional, Callable, Union
from pathlib import Path
import torch
import logging
from abc import ABC, abstractmethod

from .config import (
    BaseModelConfig, CNNConfig, LSTMConfig, HybridModelConfig,
    ModelType, create_hybrid_config, create_cnn_config, create_lstm_config
)
from .exceptions import (
    ModelError, ModelLoadError, ConfigurationError, ValidationError,
    validate_input, handle_and_reraise
)

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """Abstract interface for all models"""
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        pass
    
    @abstractmethod
    def get_config(self) -> BaseModelConfig:
        """Get model configuration"""
        pass


class ModelBuilder(ABC):
    """Abstract builder for creating models"""
    
    @abstractmethod
    def can_build(self, model_type: ModelType) -> bool:
        """Check if this builder can create the specified model type"""
        pass
    
    @abstractmethod
    def build(self, config: BaseModelConfig) -> ModelInterface:
        """Build a model with the given configuration"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> list[ModelType]:
        """Get list of supported model types"""
        pass


class CNNModelBuilder(ModelBuilder):
    """Builder for CNN models"""
    
    def can_build(self, model_type: ModelType) -> bool:
        """Check if this builder can create CNN models"""
        return model_type == ModelType.CNN_ONLY
    
    def build(self, config: BaseModelConfig) -> ModelInterface:
        """Build a CNN model"""
        if not isinstance(config, CNNConfig):
            raise ConfigurationError(
                f"CNNModelBuilder requires CNNConfig, got {type(config).__name__}",
                context={'expected_type': 'CNNConfig', 'actual_type': type(config).__name__}
            )
        
        try:
            # Import here to avoid circular imports
            from ..cnn_model import CNNFeatureExtractor
            
            # Convert config to the format expected by CNNFeatureExtractor
            legacy_config = self._convert_to_legacy_config(config)
            model = CNNFeatureExtractor(legacy_config)
            
            return CNNModelWrapper(model, config)
            
        except Exception as e:
            handle_and_reraise("CNNModelBuilder.build", e, {'model_type': 'CNN'})
    
    def get_supported_types(self) -> list[ModelType]:
        """Get supported model types"""
        return [ModelType.CNN_ONLY]
    
    def _convert_to_legacy_config(self, config: CNNConfig) -> Any:
        """Convert new config format to legacy format"""
        # This would convert the new config to whatever format the existing CNN model expects
        # For now, return the config as-is
        return config


class LSTMModelBuilder(ModelBuilder):
    """Builder for LSTM models"""
    
    def can_build(self, model_type: ModelType) -> bool:
        """Check if this builder can create LSTM models"""
        return model_type == ModelType.LSTM_ONLY
    
    def build(self, config: BaseModelConfig) -> ModelInterface:
        """Build an LSTM model"""
        if not isinstance(config, LSTMConfig):
            raise ConfigurationError(
                f"LSTMModelBuilder requires LSTMConfig, got {type(config).__name__}",
                context={'expected_type': 'LSTMConfig', 'actual_type': type(config).__name__}
            )
        
        try:
            # Import here to avoid circular imports
            from ..lstm_model import LSTMTemporalProcessor
            
            # Convert config to the format expected by LSTMTemporalProcessor
            legacy_config = self._convert_to_legacy_config(config)
            model = LSTMTemporalProcessor(legacy_config)
            
            return LSTMModelWrapper(model, config)
            
        except Exception as e:
            handle_and_reraise("LSTMModelBuilder.build", e, {'model_type': 'LSTM'})
    
    def get_supported_types(self) -> list[ModelType]:
        """Get supported model types"""
        return [ModelType.LSTM_ONLY]
    
    def _convert_to_legacy_config(self, config: LSTMConfig) -> Any:
        """Convert new config format to legacy format"""
        return config


class HybridModelBuilder(ModelBuilder):
    """Builder for CNN+LSTM hybrid models"""
    
    def can_build(self, model_type: ModelType) -> bool:
        """Check if this builder can create hybrid models"""
        return model_type == ModelType.CNN_LSTM_HYBRID
    
    def build(self, config: BaseModelConfig) -> ModelInterface:
        """Build a hybrid model"""
        if not isinstance(config, HybridModelConfig):
            raise ConfigurationError(
                f"HybridModelBuilder requires HybridModelConfig, got {type(config).__name__}",
                context={'expected_type': 'HybridModelConfig', 'actual_type': type(config).__name__}
            )
        
        try:
            # Import here to avoid circular imports
            from ..hybrid_model import CNNLSTMHybridModel
            
            model = CNNLSTMHybridModel(config)
            return HybridModelWrapper(model, config)
            
        except Exception as e:
            handle_and_reraise("HybridModelBuilder.build", e, {'model_type': 'Hybrid'})
    
    def get_supported_types(self) -> list[ModelType]:
        """Get supported model types"""
        return [ModelType.CNN_LSTM_HYBRID]


# Model wrappers to provide consistent interface
class CNNModelWrapper(ModelInterface):
    """Wrapper for CNN models to provide consistent interface"""
    
    def __init__(self, model: Any, config: CNNConfig):
        self.model = model
        self.config = config
    
    def build_model(self) -> None:
        """Build the model architecture"""
        if hasattr(self.model, 'build_model'):
            self.model.build_model()
    
    def fit(self, *args, **kwargs) -> Any:
        """Train the model"""
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs) -> Any:
        """Make predictions"""
        return self.model.predict(*args, **kwargs)
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        self.model.load_model(filepath)
    
    def get_config(self) -> CNNConfig:
        """Get model configuration"""
        return self.config


class LSTMModelWrapper(ModelInterface):
    """Wrapper for LSTM models to provide consistent interface"""
    
    def __init__(self, model: Any, config: LSTMConfig):
        self.model = model
        self.config = config
    
    def build_model(self) -> None:
        """Build the model architecture"""
        if hasattr(self.model, 'build_model'):
            self.model.build_model()
    
    def fit(self, *args, **kwargs) -> Any:
        """Train the model"""
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs) -> Any:
        """Make predictions"""
        return self.model.predict(*args, **kwargs)
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        self.model.load_model(filepath)
    
    def get_config(self) -> LSTMConfig:
        """Get model configuration"""
        return self.config


class HybridModelWrapper(ModelInterface):
    """Wrapper for hybrid models to provide consistent interface"""
    
    def __init__(self, model: Any, config: HybridModelConfig):
        self.model = model
        self.config = config
    
    def build_model(self) -> None:
        """Build the model architecture"""
        if hasattr(self.model, 'build_model'):
            self.model.build_model()
    
    def fit(self, *args, **kwargs) -> Any:
        """Train the model"""
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs) -> Any:
        """Make predictions"""
        return self.model.predict(*args, **kwargs)
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        self.model.load_model(filepath)
    
    def get_config(self) -> HybridModelConfig:
        """Get model configuration"""
        return self.config


class ModelFactory:
    """
    Factory for creating ML models with extensible builder pattern.
    
    This factory supports:
    - Multiple model types (CNN, LSTM, Hybrid)
    - Extensible builder registration
    - Model loading from checkpoints
    - Proper error handling and validation
    """
    
    def __init__(self):
        self._builders: Dict[ModelType, ModelBuilder] = {}
        self._register_default_builders()
    
    def _register_default_builders(self) -> None:
        """Register default model builders"""
        self.register_builder(CNNModelBuilder())
        self.register_builder(LSTMModelBuilder())
        self.register_builder(HybridModelBuilder())
    
    def register_builder(self, builder: ModelBuilder) -> None:
        """
        Register a model builder for specific model types
        
        Args:
            builder: Model builder instance
        """
        validate_input(builder, "builder", expected_type=ModelBuilder)
        
        for model_type in builder.get_supported_types():
            if model_type in self._builders:
                logger.warning(f"Overriding existing builder for {model_type}")
            
            self._builders[model_type] = builder
            logger.info(f"Registered builder for {model_type}")
    
    def create_model(
        self,
        model_type: Union[str, ModelType],
        config: Union[Dict[str, Any], BaseModelConfig]
    ) -> ModelInterface:
        """
        Create a model of the specified type with given configuration
        
        Args:
            model_type: Type of model to create
            config: Model configuration (dict or config object)
            
        Returns:
            Created model instance
            
        Raises:
            ConfigurationError: If model type is not supported or config is invalid
            ModelError: If model creation fails
        """
        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                raise ConfigurationError(
                    f"Unknown model type: {model_type}",
                    context={'model_type': model_type, 'supported_types': list(ModelType)}
                )
        
        validate_input(model_type, "model_type", expected_type=ModelType)
        
        # Convert dict config to config object if needed
        if isinstance(config, dict):
            config = self._create_config_from_dict(model_type, config)
        
        validate_input(config, "config", expected_type=BaseModelConfig)
        
        # Find appropriate builder
        if model_type not in self._builders:
            raise ConfigurationError(
                f"No builder registered for model type: {model_type}",
                context={'model_type': model_type, 'available_builders': list(self._builders.keys())}
            )
        
        builder = self._builders[model_type]
        
        try:
            model = builder.build(config)
            logger.info(f"Successfully created {model_type} model")
            return model
            
        except Exception as e:
            handle_and_reraise("ModelFactory.create_model", e, {'model_type': model_type})
    
    def create_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None
    ) -> ModelInterface:
        """
        Create model from saved checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on (optional)
            
        Returns:
            Loaded model instance
            
        Raises:
            ModelLoadError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ModelLoadError(
                str(checkpoint_path),
                cause=FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            )
        
        try:
            # Load checkpoint
            if device:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            # Extract model type and config
            if 'model_type' not in checkpoint:
                raise ModelLoadError(
                    str(checkpoint_path),
                    cause=KeyError("Checkpoint missing 'model_type' field")
                )
            
            model_type = checkpoint['model_type']
            if isinstance(model_type, str):
                model_type = ModelType(model_type)
            
            config = checkpoint.get('config')
            if not config:
                raise ModelLoadError(
                    str(checkpoint_path),
                    cause=KeyError("Checkpoint missing 'config' field")
                )
            
            # Create model
            model = self.create_model(model_type, config)
            
            # Load state dict
            if hasattr(model.model, 'load_state_dict'):
                model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_model(str(checkpoint_path))
            
            # Set training status
            if hasattr(model.model, 'is_trained'):
                model.model.is_trained = checkpoint.get('is_trained', False)
            
            logger.info(f"Successfully loaded model from {checkpoint_path}")
            return model
            
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            handle_and_reraise("ModelFactory.create_from_checkpoint", e, {'checkpoint_path': str(checkpoint_path)})
    
    def get_supported_types(self) -> list[ModelType]:
        """Get list of supported model types"""
        return list(self._builders.keys())
    
    def _create_config_from_dict(self, model_type: ModelType, config_dict: Dict[str, Any]) -> BaseModelConfig:
        """Create config object from dictionary"""
        try:
            if model_type == ModelType.CNN_LSTM_HYBRID:
                return create_hybrid_config(**config_dict)
            elif model_type == ModelType.CNN_ONLY:
                return create_cnn_config(**config_dict)
            elif model_type == ModelType.LSTM_ONLY:
                return create_lstm_config(**config_dict)
            else:
                # For custom model types, create base config
                config_dict['model_type'] = model_type
                return BaseModelConfig(**config_dict)
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create config for {model_type}: {e}",
                context={'model_type': model_type, 'config_dict': config_dict},
                cause=e
            )


# Global factory instance
_global_factory = ModelFactory()


def create_model(
    model_type: Union[str, ModelType],
    config: Union[Dict[str, Any], BaseModelConfig]
) -> ModelInterface:
    """
    Convenience function to create a model using the global factory
    
    Args:
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Created model instance
    """
    return _global_factory.create_model(model_type, config)


def create_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None
) -> ModelInterface:
    """
    Convenience function to create model from checkpoint using global factory
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model instance
    """
    return _global_factory.create_from_checkpoint(checkpoint_path, device)


def register_custom_builder(builder: ModelBuilder) -> None:
    """
    Register a custom model builder with the global factory
    
    Args:
        builder: Custom model builder
    """
    _global_factory.register_builder(builder)


def get_supported_model_types() -> list[ModelType]:
    """Get list of supported model types from global factory"""
    return _global_factory.get_supported_types()


# Convenience functions for specific model types
def create_hybrid_model(input_dim: int, **kwargs) -> ModelInterface:
    """Create a CNN+LSTM hybrid model with default configuration"""
    config = create_hybrid_config(input_dim=input_dim, **kwargs)
    return create_model(ModelType.CNN_LSTM_HYBRID, config)


def create_cnn_model(input_dim: int, **kwargs) -> ModelInterface:
    """Create a CNN model with default configuration"""
    config = create_cnn_config(input_dim=input_dim, **kwargs)
    return create_model(ModelType.CNN_ONLY, config)


def create_lstm_model(input_dim: int, **kwargs) -> ModelInterface:
    """Create an LSTM model with default configuration"""
    config = create_lstm_config(input_dim=input_dim, **kwargs)
    return create_model(ModelType.LSTM_ONLY, config)
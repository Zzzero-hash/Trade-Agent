"""
Custom exceptions for the ML trading platform.

This module provides a hierarchy of custom exceptions with proper error handling,
logging, and context information for better debugging and user experience.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TradingPlatformError(Exception):
    """Base exception for all trading platform errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        
        # Log the error with context
        self._log_error()
    
    def _log_error(self) -> None:
        """Log the error with appropriate level and context"""
        log_message = f"{self.error_code}: {self.message}"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            log_message += f" (Context: {context_str})"
        
        if self.cause:
            log_message += f" (Caused by: {self.cause})"
        
        logger.error(log_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'cause': str(self.cause) if self.cause else None
        }


class ConfigurationError(TradingPlatformError):
    """Raised when there are configuration-related errors"""
    pass


class ValidationError(ConfigurationError):
    """Raised when validation fails"""
    pass


class ModelError(TradingPlatformError):
    """Base class for model-related errors"""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when attempting to use an untrained model"""
    
    def __init__(self, model_name: str, operation: str):
        super().__init__(
            f"Model '{model_name}' must be trained before {operation}",
            error_code="MODEL_NOT_TRAINED",
            context={'model_name': model_name, 'operation': operation}
        )


class ModelLoadError(ModelError):
    """Raised when model loading fails"""
    
    def __init__(self, model_path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to load model from '{model_path}'",
            error_code="MODEL_LOAD_FAILED",
            context={'model_path': model_path},
            cause=cause
        )


class ModelSaveError(ModelError):
    """Raised when model saving fails"""
    
    def __init__(self, model_path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to save model to '{model_path}'",
            error_code="MODEL_SAVE_FAILED",
            context={'model_path': model_path},
            cause=cause
        )


class ModelArchitectureError(ModelError):
    """Raised when there are model architecture issues"""
    
    def __init__(self, model_type: str, issue: str):
        super().__init__(
            f"Architecture error in {model_type}: {issue}",
            error_code="MODEL_ARCHITECTURE_ERROR",
            context={'model_type': model_type, 'issue': issue}
        )


class TrainingError(TradingPlatformError):
    """Base class for training-related errors"""
    pass


class TrainingDataError(TrainingError):
    """Raised when there are issues with training data"""
    
    def __init__(self, issue: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Training data error: {issue}",
            error_code="TRAINING_DATA_ERROR",
            context=data_info or {}
        )


class TrainingConvergenceError(TrainingError):
    """Raised when training fails to converge"""
    
    def __init__(self, epochs_trained: int, final_loss: float):
        super().__init__(
            f"Training failed to converge after {epochs_trained} epochs (final loss: {final_loss:.6f})",
            error_code="TRAINING_CONVERGENCE_FAILED",
            context={'epochs_trained': epochs_trained, 'final_loss': final_loss}
        )


class DataError(TradingPlatformError):
    """Base class for data-related errors"""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails"""
    
    def __init__(self, validation_issue: str, data_shape: Optional[tuple] = None):
        context = {}
        if data_shape:
            context['data_shape'] = data_shape
        
        super().__init__(
            f"Data validation failed: {validation_issue}",
            error_code="DATA_VALIDATION_FAILED",
            context=context
        )


class DataLoadError(DataError):
    """Raised when data loading fails"""
    
    def __init__(self, data_source: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to load data from '{data_source}'",
            error_code="DATA_LOAD_FAILED",
            context={'data_source': data_source},
            cause=cause
        )


class DataPreprocessingError(DataError):
    """Raised when data preprocessing fails"""
    
    def __init__(self, preprocessing_step: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Data preprocessing failed at step: {preprocessing_step}",
            error_code="DATA_PREPROCESSING_FAILED",
            context={'preprocessing_step': preprocessing_step},
            cause=cause
        )


class FeatureExtractionError(DataError):
    """Raised when feature extraction fails"""
    
    def __init__(self, feature_type: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Feature extraction failed for {feature_type}",
            error_code="FEATURE_EXTRACTION_FAILED",
            context={'feature_type': feature_type},
            cause=cause
        )


class TradingEnvironmentError(TradingPlatformError):
    """Base class for trading environment errors"""
    pass


class InvalidActionError(TradingEnvironmentError):
    """Raised when an invalid trading action is attempted"""
    
    def __init__(self, action: str, reason: str):
        super().__init__(
            f"Invalid trading action '{action}': {reason}",
            error_code="INVALID_TRADING_ACTION",
            context={'action': action, 'reason': reason}
        )


class InsufficientFundsError(TradingEnvironmentError):
    """Raised when there are insufficient funds for a trade"""
    
    def __init__(self, required_amount: float, available_amount: float):
        super().__init__(
            f"Insufficient funds: required ${required_amount:.2f}, available ${available_amount:.2f}",
            error_code="INSUFFICIENT_FUNDS",
            context={'required_amount': required_amount, 'available_amount': available_amount}
        )


class MarketDataError(TradingEnvironmentError):
    """Raised when there are market data issues"""
    
    def __init__(self, symbol: str, issue: str):
        super().__init__(
            f"Market data error for {symbol}: {issue}",
            error_code="MARKET_DATA_ERROR",
            context={'symbol': symbol, 'issue': issue}
        )


class DeviceError(TradingPlatformError):
    """Raised when there are device-related issues (GPU/CPU)"""
    
    def __init__(self, device: str, issue: str):
        super().__init__(
            f"Device error on {device}: {issue}",
            error_code="DEVICE_ERROR",
            context={'device': device, 'issue': issue}
        )


class OptimizationError(TradingPlatformError):
    """Raised when optimization fails"""
    
    def __init__(self, optimizer_type: str, issue: str):
        super().__init__(
            f"Optimization error with {optimizer_type}: {issue}",
            error_code="OPTIMIZATION_ERROR",
            context={'optimizer_type': optimizer_type, 'issue': issue}
        )


# Utility functions for error handling
def handle_and_reraise(
    func_name: str,
    original_exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Handle an exception and re-raise as appropriate platform exception
    
    Args:
        func_name: Name of the function where error occurred
        original_exception: The original exception
        context: Additional context information
    """
    context = context or {}
    context['function'] = func_name
    
    # Map common exceptions to platform exceptions
    if isinstance(original_exception, FileNotFoundError):
        if 'model' in func_name.lower():
            raise ModelLoadError(str(original_exception), cause=original_exception)
        else:
            raise DataLoadError(str(original_exception), cause=original_exception)
    
    elif isinstance(original_exception, ValueError):
        if 'config' in func_name.lower():
            raise ValidationError(str(original_exception), context=context, cause=original_exception)
        else:
            raise DataValidationError(str(original_exception), cause=original_exception)
    
    elif isinstance(original_exception, RuntimeError):
        if 'cuda' in str(original_exception).lower() or 'gpu' in str(original_exception).lower():
            raise DeviceError('GPU', str(original_exception))
        else:
            raise TradingPlatformError(
                f"Runtime error in {func_name}: {original_exception}",
                context=context,
                cause=original_exception
            )
    
    else:
        # Re-raise as generic platform error
        raise TradingPlatformError(
            f"Unexpected error in {func_name}: {original_exception}",
            context=context,
            cause=original_exception
        )


def validate_input(
    value: Any,
    name: str,
    expected_type: type = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allowed_values: Optional[list] = None
) -> None:
    """
    Validate input parameters with detailed error messages
    
    Args:
        value: Value to validate
        name: Name of the parameter
        expected_type: Expected type
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allowed_values: List of allowed values
    
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError(f"Parameter '{name}' cannot be None")
    
    if expected_type and not isinstance(value, expected_type):
        raise ValidationError(
            f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
            context={'parameter': name, 'expected_type': expected_type.__name__, 'actual_type': type(value).__name__}
        )
    
    if min_value is not None and hasattr(value, '__lt__') and value < min_value:
        raise ValidationError(
            f"Parameter '{name}' must be >= {min_value}, got {value}",
            context={'parameter': name, 'min_value': min_value, 'actual_value': value}
        )
    
    if max_value is not None and hasattr(value, '__gt__') and value > max_value:
        raise ValidationError(
            f"Parameter '{name}' must be <= {max_value}, got {value}",
            context={'parameter': name, 'max_value': max_value, 'actual_value': value}
        )
    
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"Parameter '{name}' must be one of {allowed_values}, got {value}",
            context={'parameter': name, 'allowed_values': allowed_values, 'actual_value': value}
        )


def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with proper error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    
    Raises:
        TradingPlatformError: If execution fails
    """
    try:
        return func(*args, **kwargs)
    except TradingPlatformError:
        # Re-raise platform exceptions as-is
        raise
    except Exception as e:
        # Convert other exceptions to platform exceptions
        handle_and_reraise(func.__name__, e)
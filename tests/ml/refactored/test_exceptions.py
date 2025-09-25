"""
Unit tests for custom exceptions and error handling.

Tests cover exception hierarchy, error logging, context handling, and utility functions.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from src.ml.refactored.exceptions import (
    TradingPlatformError, ConfigurationError, ValidationError,
    ModelError, ModelNotTrainedError, ModelLoadError, ModelSaveError,
    ModelArchitectureError, TrainingError, TrainingDataError,
    TrainingConvergenceError, DataError, DataValidationError,
    DataLoadError, DataPreprocessingError, FeatureExtractionError,
    TradingEnvironmentError, InvalidActionError, InsufficientFundsError,
    MarketDataError, DeviceError, OptimizationError,
    handle_and_reraise, validate_input, safe_execute
)


class TestTradingPlatformError:
    """Test base exception class"""
    
    def test_basic_exception_creation(self):
        """Test creating basic exception"""
        error = TradingPlatformError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code == "TradingPlatformError"
        assert error.context == {}
        assert error.cause is None
    
    def test_exception_with_all_parameters(self):
        """Test creating exception with all parameters"""
        cause = ValueError("Original error")
        context = {"param1": "value1", "param2": 42}
        
        error = TradingPlatformError(
            "Test error",
            error_code="CUSTOM_ERROR",
            context=context,
            cause=cause
        )
        
        assert error.message == "Test error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.context == context
        assert error.cause == cause
    
    def test_to_dict_method(self):
        """Test converting exception to dictionary"""
        cause = ValueError("Original error")
        context = {"param": "value"}
        
        error = TradingPlatformError(
            "Test error",
            error_code="TEST_ERROR",
            context=context,
            cause=cause
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['error_type'] == 'TradingPlatformError'
        assert error_dict['error_code'] == 'TEST_ERROR'
        assert error_dict['message'] == 'Test error'
        assert error_dict['context'] == context
        assert error_dict['cause'] == str(cause)
    
    @patch('src.ml.refactored.exceptions.logger')
    def test_error_logging(self, mock_logger):
        """Test that errors are logged properly"""
        context = {"param": "value"}
        cause = ValueError("Original error")
        
        TradingPlatformError(
            "Test error",
            error_code="TEST_ERROR",
            context=context,
            cause=cause
        )
        
        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args[0][0]
        
        assert "TEST_ERROR: Test error" in log_call
        assert "Context: param=value" in log_call
        assert "Caused by: Original error" in log_call


class TestSpecificExceptions:
    """Test specific exception types"""
    
    def test_model_not_trained_error(self):
        """Test ModelNotTrainedError"""
        error = ModelNotTrainedError("TestModel", "prediction")
        
        assert "TestModel" in str(error)
        assert "prediction" in str(error)
        assert error.error_code == "MODEL_NOT_TRAINED"
        assert error.context['model_name'] == "TestModel"
        assert error.context['operation'] == "prediction"
    
    def test_model_load_error(self):
        """Test ModelLoadError"""
        cause = FileNotFoundError("File not found")
        error = ModelLoadError("/path/to/model.pth", cause=cause)
        
        assert "/path/to/model.pth" in str(error)
        assert error.error_code == "MODEL_LOAD_FAILED"
        assert error.context['model_path'] == "/path/to/model.pth"
        assert error.cause == cause
    
    def test_model_save_error(self):
        """Test ModelSaveError"""
        cause = PermissionError("Permission denied")
        error = ModelSaveError("/path/to/model.pth", cause=cause)
        
        assert "/path/to/model.pth" in str(error)
        assert error.error_code == "MODEL_SAVE_FAILED"
        assert error.cause == cause
    
    def test_model_architecture_error(self):
        """Test ModelArchitectureError"""
        error = ModelArchitectureError("CNN", "Invalid filter size")
        
        assert "CNN" in str(error)
        assert "Invalid filter size" in str(error)
        assert error.error_code == "MODEL_ARCHITECTURE_ERROR"
    
    def test_training_data_error(self):
        """Test TrainingDataError"""
        data_info = {"shape": (100, 10), "dtype": "float32"}
        error = TrainingDataError("Missing labels", data_info)
        
        assert "Missing labels" in str(error)
        assert error.error_code == "TRAINING_DATA_ERROR"
        assert error.context == data_info
    
    def test_training_convergence_error(self):
        """Test TrainingConvergenceError"""
        error = TrainingConvergenceError(100, 0.5)
        
        assert "100 epochs" in str(error)
        assert "0.500000" in str(error)
        assert error.error_code == "TRAINING_CONVERGENCE_FAILED"
        assert error.context['epochs_trained'] == 100
        assert error.context['final_loss'] == 0.5
    
    def test_data_validation_error(self):
        """Test DataValidationError"""
        error = DataValidationError("Invalid shape", data_shape=(10, 5))
        
        assert "Invalid shape" in str(error)
        assert error.error_code == "DATA_VALIDATION_FAILED"
        assert error.context['data_shape'] == (10, 5)
    
    def test_invalid_action_error(self):
        """Test InvalidActionError"""
        error = InvalidActionError("BUY", "Insufficient funds")
        
        assert "BUY" in str(error)
        assert "Insufficient funds" in str(error)
        assert error.error_code == "INVALID_TRADING_ACTION"
    
    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError"""
        error = InsufficientFundsError(1000.0, 500.0)
        
        assert "$1000.00" in str(error)
        assert "$500.00" in str(error)
        assert error.error_code == "INSUFFICIENT_FUNDS"
        assert error.context['required_amount'] == 1000.0
        assert error.context['available_amount'] == 500.0
    
    def test_market_data_error(self):
        """Test MarketDataError"""
        error = MarketDataError("AAPL", "Price data missing")
        
        assert "AAPL" in str(error)
        assert "Price data missing" in str(error)
        assert error.error_code == "MARKET_DATA_ERROR"
    
    def test_device_error(self):
        """Test DeviceError"""
        error = DeviceError("cuda:0", "Out of memory")
        
        assert "cuda:0" in str(error)
        assert "Out of memory" in str(error)
        assert error.error_code == "DEVICE_ERROR"


class TestHandleAndReraise:
    """Test handle_and_reraise utility function"""
    
    def test_file_not_found_model_function(self):
        """Test FileNotFoundError in model function"""
        original_error = FileNotFoundError("Model file not found")
        
        with pytest.raises(ModelLoadError):
            handle_and_reraise("load_model", original_error)
    
    def test_file_not_found_data_function(self):
        """Test FileNotFoundError in data function"""
        original_error = FileNotFoundError("Data file not found")
        
        with pytest.raises(DataLoadError):
            handle_and_reraise("load_data", original_error)
    
    def test_value_error_config_function(self):
        """Test ValueError in config function"""
        original_error = ValueError("Invalid configuration")
        
        with pytest.raises(ValidationError):
            handle_and_reraise("validate_config", original_error)
    
    def test_value_error_data_function(self):
        """Test ValueError in data function"""
        original_error = ValueError("Invalid data format")
        
        with pytest.raises(DataValidationError):
            handle_and_reraise("process_data", original_error)
    
    def test_runtime_error_cuda(self):
        """Test RuntimeError with CUDA"""
        original_error = RuntimeError("CUDA out of memory")
        
        with pytest.raises(DeviceError) as exc_info:
            handle_and_reraise("train_model", original_error)
        
        assert exc_info.value.context['device'] == 'GPU'
    
    def test_generic_runtime_error(self):
        """Test generic RuntimeError"""
        original_error = RuntimeError("Generic runtime error")
        
        with pytest.raises(TradingPlatformError):
            handle_and_reraise("some_function", original_error)
    
    def test_unknown_exception(self):
        """Test unknown exception type"""
        original_error = TypeError("Unknown error")
        
        with pytest.raises(TradingPlatformError) as exc_info:
            handle_and_reraise("test_function", original_error)
        
        assert "test_function" in str(exc_info.value)
        assert exc_info.value.cause == original_error


class TestValidateInput:
    """Test validate_input utility function"""
    
    def test_none_value(self):
        """Test validation of None value"""
        with pytest.raises(ValidationError, match="cannot be None"):
            validate_input(None, "test_param")
    
    def test_type_validation_success(self):
        """Test successful type validation"""
        validate_input(42, "test_param", expected_type=int)
        # Should not raise
    
    def test_type_validation_failure(self):
        """Test failed type validation"""
        with pytest.raises(ValidationError, match="must be of type int"):
            validate_input("42", "test_param", expected_type=int)
    
    def test_min_value_validation_success(self):
        """Test successful min value validation"""
        validate_input(10, "test_param", min_value=5)
        # Should not raise
    
    def test_min_value_validation_failure(self):
        """Test failed min value validation"""
        with pytest.raises(ValidationError, match="must be >= 10"):
            validate_input(5, "test_param", min_value=10)
    
    def test_max_value_validation_success(self):
        """Test successful max value validation"""
        validate_input(5, "test_param", max_value=10)
        # Should not raise
    
    def test_max_value_validation_failure(self):
        """Test failed max value validation"""
        with pytest.raises(ValidationError, match="must be <= 5"):
            validate_input(10, "test_param", max_value=5)
    
    def test_allowed_values_validation_success(self):
        """Test successful allowed values validation"""
        validate_input("option1", "test_param", allowed_values=["option1", "option2"])
        # Should not raise
    
    def test_allowed_values_validation_failure(self):
        """Test failed allowed values validation"""
        with pytest.raises(ValidationError, match="must be one of"):
            validate_input("option3", "test_param", allowed_values=["option1", "option2"])
    
    def test_combined_validation(self):
        """Test combined validation rules"""
        validate_input(
            7,
            "test_param",
            expected_type=int,
            min_value=5,
            max_value=10,
            allowed_values=[5, 6, 7, 8, 9, 10]
        )
        # Should not raise


class TestSafeExecute:
    """Test safe_execute utility function"""
    
    def test_successful_execution(self):
        """Test successful function execution"""
        def test_func(x, y):
            return x + y
        
        result = safe_execute(test_func, 2, 3)
        assert result == 5
    
    def test_platform_exception_passthrough(self):
        """Test that platform exceptions are passed through"""
        def test_func():
            raise ValidationError("Test validation error")
        
        with pytest.raises(ValidationError):
            safe_execute(test_func)
    
    def test_generic_exception_conversion(self):
        """Test that generic exceptions are converted"""
        def test_func():
            raise ValueError("Generic error")
        
        with pytest.raises(TradingPlatformError):
            safe_execute(test_func)
    
    def test_function_with_kwargs(self):
        """Test function execution with keyword arguments"""
        def test_func(x, y=10):
            return x * y
        
        result = safe_execute(test_func, 5, y=3)
        assert result == 15


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy"""
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from TradingPlatformError"""
        error = ConfigurationError("Config error")
        assert isinstance(error, TradingPlatformError)
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from ConfigurationError"""
        error = ValidationError("Validation error")
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, TradingPlatformError)
    
    def test_model_error_inheritance(self):
        """Test ModelError inherits from TradingPlatformError"""
        error = ModelError("Model error")
        assert isinstance(error, TradingPlatformError)
    
    def test_specific_model_error_inheritance(self):
        """Test specific model errors inherit correctly"""
        error = ModelNotTrainedError("TestModel", "prediction")
        assert isinstance(error, ModelError)
        assert isinstance(error, TradingPlatformError)
    
    def test_data_error_inheritance(self):
        """Test DataError inherits from TradingPlatformError"""
        error = DataError("Data error")
        assert isinstance(error, TradingPlatformError)
    
    def test_trading_environment_error_inheritance(self):
        """Test TradingEnvironmentError inherits from TradingPlatformError"""
        error = TradingEnvironmentError("Environment error")
        assert isinstance(error, TradingPlatformError)


if __name__ == "__main__":
    pytest.main([__file__])
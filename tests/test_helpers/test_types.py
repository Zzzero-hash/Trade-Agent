"""
Type definitions and error handling for comprehensive integration tests.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from unittest.mock import Mock

from src.models.market_data import MarketData, ExchangeType
from src.models.trading_signal import TradingSignal


# Type aliases for better readability
MockExchangeDict = Dict[str, Mock]
MarketDataList = List[MarketData]
TestResultDict = Dict[str, Any]
PerformanceMetrics = Dict[str, Union[float, int]]


class TestCategory(Enum):
    """Enumeration of test categories."""
    WORKFLOW = "workflow"
    PERFORMANCE = "performance"
    CHAOS = "chaos"
    INTEGRATION = "integration"


class FailureType(Enum):
    """Enumeration of failure types for chaos testing."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONNECTION_ERROR = "connection_error"
    MODEL_ERROR = "model_error"
    DATABASE_ERROR = "database_error"


@dataclass
class TestScenario:
    """Data class for test scenarios."""
    name: str
    category: TestCategory
    exchange: Optional[str] = None
    failure_type: Optional[FailureType] = None
    expected_result: Optional[bool] = None
    timeout_seconds: Optional[float] = None


@dataclass
class PerformanceTestResult:
    """Data class for performance test results."""
    test_name: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_sec: float
    memory_usage_mb: float
    success_rate: float


class TestDataProvider(Protocol):
    """Protocol for test data providers."""
    
    def generate_market_data(self, symbol: str, hours: int) -> MarketDataList:
        """Generate market data for testing."""
        ...
    
    def create_mock_exchanges(self) -> MockExchangeDict:
        """Create mock exchange connectors."""
        ...


class TestResultValidator(Protocol):
    """Protocol for test result validation."""
    
    def validate_workflow_result(self, result: TestResultDict) -> bool:
        """Validate workflow test results."""
        ...
    
    def validate_performance_result(self, result: PerformanceTestResult) -> bool:
        """Validate performance test results."""
        ...


T = TypeVar('T')


class TestExecutionContext(Generic[T]):
    """Generic context for test execution."""
    
    def __init__(self, test_data: T, scenario: TestScenario):
        self.test_data = test_data
        self.scenario = scenario
        self.results: List[Any] = []
        self.errors: List[Exception] = []
    
    def add_result(self, result: Any) -> None:
        """Add a test result."""
        self.results.append(result)
    
    def add_error(self, error: Exception) -> None:
        """Add a test error."""
        self.errors.append(error)
    
    @property
    def has_errors(self) -> bool:
        """Check if context has errors."""
        return len(self.errors) > 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = len(self.results) + len(self.errors)
        if total == 0:
            return 0.0
        return len(self.results) / total


class TestException(Exception):
    """Base exception for test-related errors."""
    pass


class TestSetupError(TestException):
    """Exception raised during test setup."""
    pass


class TestExecutionError(TestException):
    """Exception raised during test execution."""
    pass


class TestValidationError(TestException):
    """Exception raised during test result validation."""
    pass


class PerformanceThresholdError(TestException):
    """Exception raised when performance thresholds are exceeded."""
    
    def __init__(self, metric: str, actual: float, threshold: float):
        self.metric = metric
        self.actual = actual
        self.threshold = threshold
        super().__init__(
            f"Performance threshold exceeded for {metric}: "
            f"actual={actual}, threshold={threshold}"
        )


def safe_test_execution(func):
    """Decorator for safe test execution with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TestException:
            raise  # Re-raise test-specific exceptions
        except Exception as e:
            raise TestExecutionError(f"Unexpected error in {func.__name__}: {e}") from e
    return wrapper


def validate_test_data(data: Any, expected_type: type) -> None:
    """Validate test data type and structure."""
    if not isinstance(data, expected_type):
        raise TestValidationError(
            f"Expected {expected_type.__name__}, got {type(data).__name__}"
        )


def assert_performance_threshold(metric_name: str, actual: float, 
                                threshold: float, lower_is_better: bool = True) -> None:
    """Assert that performance metric meets threshold."""
    if lower_is_better and actual > threshold:
        raise PerformanceThresholdError(metric_name, actual, threshold)
    elif not lower_is_better and actual < threshold:
        raise PerformanceThresholdError(metric_name, actual, threshold)
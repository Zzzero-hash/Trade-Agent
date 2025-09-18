"""
Test configuration and constants for comprehensive integration tests.
Centralizes test parameters and thresholds for better maintainability.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class TestType(Enum):
    """Enumeration of test types for timeout configuration."""
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


@dataclass(frozen=True)
class PerformanceThresholds:
    """Performance thresholds for test validation."""
    max_latency_ms: float = 1000.0
    min_throughput_per_sec: float = 100.0
    max_memory_growth_mb: float = 500.0
    max_signal_generation_avg_ms: float = 100.0
    max_signal_generation_p95_ms: float = 200.0
    max_signal_generation_p99_ms: float = 500.0

    def validate_latency(self, actual_ms: float) -> bool:
        """Validate latency against threshold."""
        return actual_ms <= self.max_latency_ms

    def validate_throughput(self, actual_tps: float) -> bool:
        """Validate throughput against threshold."""
        return actual_tps >= self.min_throughput_per_sec


@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    default_hours: int = 720  # 30 days
    performance_test_records: int = 1000
    batch_sizes: List[int] = field(
        default_factory=lambda: [10, 50, 100, 500, 1000]
    )
    concurrency_levels: List[int] = field(
        default_factory=lambda: [1, 5, 10, 20, 50]
    )


@dataclass
class MockDataConfig:
    """Configuration for mock data responses."""
    robinhood_account: Optional[Dict[str, float]] = None
    oanda_account: Optional[Dict[str, Any]] = None
    coinbase_account: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default mock data configurations."""
        if self.robinhood_account is None:
            self.robinhood_account = self._get_default_robinhood_account()

        if self.oanda_account is None:
            self.oanda_account = self._get_default_oanda_account()

        if self.coinbase_account is None:
            self.coinbase_account = self._get_default_coinbase_account()

    @staticmethod
    def _get_default_robinhood_account() -> Dict[str, float]:
        """Get default Robinhood account configuration."""
        return {
            'buying_power': 50000.0,
            'total_equity': 100000.0
        }

    @staticmethod
    def _get_default_oanda_account() -> Dict[str, Any]:
        """Get default OANDA account configuration."""
        return {
            'balance': 25000.0,
            'margin_available': 20000.0,
            'currency': 'USD'
        }

    @staticmethod
    def _get_default_coinbase_account() -> Dict[str, Any]:
        """Get default Coinbase account configuration."""
        return {
            'available_balance': {'USD': 15000.0, 'BTC': 0.0},
            'total_balance': {'USD': 15000.0, 'BTC': 0.0}
        }


@dataclass(frozen=True)
class TestConstants:
    """Constants used across integration tests."""

    # Time constants
    DEFAULT_LOOKBACK_DAYS: int = 7
    MARKET_DATA_GENERATION_DAYS: int = 30

    # Symbol constants
    DEFAULT_STOCK_SYMBOLS: List[str] = field(default_factory=lambda: [
        'AAPL', 'GOOGL', 'MSFT', 'TSLA'
    ])
    DEFAULT_FOREX_SYMBOLS: List[str] = field(default_factory=lambda: [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'
    ])
    DEFAULT_CRYPTO_SYMBOLS: List[str] = field(default_factory=lambda: [
        'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD'
    ])

    # Test execution constants
    SLOW_TEST_TIMEOUT: float = 300.0  # 5 minutes
    NORMAL_TEST_TIMEOUT: float = 60.0  # 1 minute
    FAST_TEST_TIMEOUT: float = 10.0   # 10 seconds

    # Performance constants
    MEMORY_CHECK_INTERVAL: int = 10  # Check memory every 10 batches
    MAX_MEMORY_GROWTH_GB: float = 1.0  # 1GB max growth

    # Chaos engineering constants
    PARTITION_SIMULATION_DURATION: float = 0.1  # 100ms for tests
    FAILURE_RECOVERY_TIMEOUT: float = 5.0  # 5 seconds

    # CI environment multiplier
    CI_TIMEOUT_MULTIPLIER: float = 2.0


class CIEnvironmentDetector:
    """Utility class for detecting CI environments."""

    CI_INDICATORS = [
        'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS',
        'JENKINS_URL', 'TRAVIS', 'CIRCLECI'
    ]

    @classmethod
    def is_ci_environment(cls) -> bool:
        """Detect if running in CI environment."""
        return any(os.getenv(indicator) for indicator in cls.CI_INDICATORS)


class TestEnvironment:
    """Test environment configuration with improved design patterns."""

    def __init__(
        self,
        performance_thresholds: Optional[PerformanceThresholds] = None,
        data_config: Optional[TestDataConfig] = None,
        mock_config: Optional[MockDataConfig] = None,
        constants: Optional[TestConstants] = None
    ):
        """Initialize test environment with dependency injection."""
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.data_config = data_config or TestDataConfig()
        self.mock_config = mock_config or MockDataConfig()
        self.constants = constants or TestConstants()
        self.is_ci_environment = CIEnvironmentDetector.is_ci_environment()
        self.enable_slow_tests = not self.is_ci_environment

    def get_timeout_for_test(self, test_type: TestType) -> float:
        """Get appropriate timeout for test type using enum."""
        timeout_mapping = {
            TestType.SLOW: self.constants.SLOW_TEST_TIMEOUT,
            TestType.NORMAL: self.constants.NORMAL_TEST_TIMEOUT,
            TestType.FAST: self.constants.FAST_TEST_TIMEOUT
        }

        timeout = timeout_mapping.get(test_type, self.constants.NORMAL_TEST_TIMEOUT)

        # Apply CI environment multiplier
        if self.is_ci_environment:
            timeout *= self.constants.CI_TIMEOUT_MULTIPLIER

        return timeout

    def should_run_slow_tests(self) -> bool:
        """Determine if slow tests should be executed."""
        force_slow = os.getenv('FORCE_SLOW_TESTS', '').lower() == 'true'
        return self.enable_slow_tests or force_slow

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration as dictionary."""
        return {
            'thresholds': self.performance_thresholds,
            'data_config': self.data_config,
            'is_ci': self.is_ci_environment
        }


# Factory function for creating test environment
def create_test_environment(**kwargs) -> TestEnvironment:
    """Factory function for creating test environment with custom config."""
    return TestEnvironment(**kwargs)


# Global test environment instance
TEST_ENV = create_test_environment()
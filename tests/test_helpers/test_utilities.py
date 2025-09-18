"""
Utility functions for comprehensive integration tests.
"""

import functools
import time
import statistics
from typing import List, Dict, Any, Callable
from contextlib import contextmanager
from unittest.mock import patch
import pandas as pd

from src.models.trading_signal import TradingSignal, TradingAction
from .test_types import PerformanceTestResult, TestException


def retry_on_failure(max_attempts: int = 3, delay: float = 0.1):
    """Decorator to retry test operations on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
                    continue
            
            raise TestException(f"Function {func.__name__} failed after {max_attempts} attempts") from last_exception
        return wrapper
    return decorator


@contextmanager
def mock_exchange_failure(exchange_name: str, failure_type: str):
    """Context manager for mocking exchange failures."""
    with patch(f'src.exchanges.{exchange_name}_connector') as mock_exchange:
        if failure_type == 'timeout':
            mock_exchange.get_historical_data.side_effect = TimeoutError("Connection timeout")
        elif failure_type == 'rate_limit':
            mock_exchange.get_historical_data.side_effect = Exception("Rate limit exceeded")
        elif failure_type == 'connection_error':
            mock_exchange.get_historical_data.side_effect = ConnectionError("Connection failed")
        
        yield mock_exchange


def calculate_performance_metrics(latencies: List[float]) -> PerformanceTestResult:
    """Calculate performance metrics from latency measurements."""
    if not latencies:
        raise ValueError("No latency measurements provided")
    
    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
    
    return PerformanceTestResult(
        test_name="performance_test",
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        throughput_per_sec=1000.0 / avg_latency if avg_latency > 0 else 0.0,
        memory_usage_mb=0.0,
        success_rate=1.0
    )


def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required structure."""
    if df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True
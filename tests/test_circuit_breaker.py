"""
Tests for Circuit Breaker Service

Tests circuit breaker functionality, graceful degradation,
and automatic recovery mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.services.circuit_breaker_service import (
    CircuitBreaker, CircuitBreakerService, CircuitBreakerConfig,
    CircuitState, CircuitBreakerException, circuit_breaker
)
from src.services.monitoring_service import MonitoringService
from src.services.alert_service import AlertService


@pytest.fixture
def mock_monitoring_service():
    service = Mock(spec=MonitoringService)
    service.record_metric = AsyncMock()
    return service


@pytest.fixture
def mock_alert_service():
    service = Mock(spec=AlertService)
    service.send_error_alert = AsyncMock()
    service.send_warning_alert = AsyncMock()
    return service


@pytest.fixture
def circuit_breaker_config():
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5,  # 5 seconds for testing
        success_threshold=2,
        timeout=1.0,
        expected_exception=ValueError
    )


@pytest.fixture
def circuit_breaker(mock_monitoring_service, mock_alert_service, circuit_breaker_config):
    return CircuitBreaker(
        name="test_service",
        config=circuit_breaker_config,
        monitoring_service=mock_monitoring_service,
        alert_service=mock_alert_service
    )


@pytest.fixture
def circuit_breaker_service(mock_monitoring_service, mock_alert_service):
    return CircuitBreakerService(
        monitoring_service=mock_monitoring_service,
        alert_service=mock_alert_service
    )


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.name == "test_service"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.failure_count == 0
        assert circuit_breaker.stats.success_count == 0
    
    def test_circuit_breaker_config(self, circuit_breaker, circuit_breaker_config):
        """Test circuit breaker configuration."""
        assert circuit_breaker.config.failure_threshold == 3
        assert circuit_breaker.config.recovery_timeout == 5
        assert circuit_breaker.config.success_threshold == 2
        assert circuit_breaker.config.timeout == 1.0
        assert circuit_breaker.config.expected_exception == ValueError


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""
    
    @pytest.mark.asyncio
    async def test_closed_state_success(self, circuit_breaker, mock_monitoring_service):
        """Test successful calls in closed state."""
        async def successful_function():
            return "success"
        
        result = await circuit_breaker.call(successful_function)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.success_count == 1
        assert circuit_breaker.stats.failure_count == 0
        
        # Should record success metric
        mock_monitoring_service.record_metric.assert_called()
    
    @pytest.mark.asyncio
    async def test_closed_state_failure(self, circuit_breaker, mock_monitoring_service):
        """Test failed calls in closed state."""
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.failure_count == 1
        assert circuit_breaker.stats.success_count == 0
        
        # Should record failure metric
        mock_monitoring_service.record_metric.assert_called()
    
    @pytest.mark.asyncio
    async def test_transition_to_open(self, circuit_breaker, mock_alert_service):
        """Test transition from closed to open state."""
        async def failing_function():
            raise ValueError("Test error")
        
        # Fail enough times to open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.stats.failure_count == 3
        
        # Should send alert
        mock_alert_service.send_error_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_open_state_blocks_calls(self, circuit_breaker):
        """Test that open state blocks calls."""
        # Force circuit to open
        await circuit_breaker._transition_to_open()
        
        async def any_function():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerException):
            await circuit_breaker.call(any_function)
    
    @pytest.mark.asyncio
    async def test_transition_to_half_open(self, circuit_breaker):
        """Test transition from open to half-open state."""
        # Force circuit to open
        await circuit_breaker._transition_to_open()
        
        # Simulate recovery timeout passing
        circuit_breaker.stats.last_failure_time = datetime.utcnow() - timedelta(seconds=10)
        
        async def test_function():
            return "test"
        
        # Should transition to half-open and allow call
        result = await circuit_breaker.call(test_function)
        assert result == "test"
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, circuit_breaker, mock_alert_service):
        """Test that successful calls in half-open state close the circuit."""
        # Force circuit to half-open
        await circuit_breaker._transition_to_half_open()
        
        async def successful_function():
            return "success"
        
        # Make enough successful calls to close circuit
        for i in range(2):  # success_threshold = 2
            result = await circuit_breaker.call(successful_function)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Should send recovery alert
        mock_alert_service.send_warning_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test that failure in half-open state reopens the circuit."""
        # Force circuit to half-open
        await circuit_breaker._transition_to_half_open()
        
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == CircuitState.OPEN


class TestCircuitBreakerTimeout:
    """Test circuit breaker timeout functionality."""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test that timeouts are handled as failures."""
        async def slow_function():
            await asyncio.sleep(2)  # Longer than timeout (1 second)
            return "should not reach here"
        
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_function)
        
        assert circuit_breaker.stats.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_timeout_opens_circuit(self, circuit_breaker):
        """Test that repeated timeouts open the circuit."""
        async def slow_function():
            await asyncio.sleep(2)
            return "timeout"
        
        # Timeout enough times to open circuit
        for i in range(3):
            with pytest.raises(asyncio.TimeoutError):
                await circuit_breaker.call(slow_function)
        
        assert circuit_breaker.state == CircuitState.OPEN


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics and statistics."""
    
    @pytest.mark.asyncio
    async def test_latency_recording(self, circuit_breaker, mock_monitoring_service):
        """Test that latency is recorded for successful calls."""
        async def timed_function():
            await asyncio.sleep(0.01)
            return "success"
        
        await circuit_breaker.call(timed_function)
        
        # Should record latency metric
        latency_calls = [
            call for call in mock_monitoring_service.record_metric.call_args_list
            if "latency" in str(call)
        ]
        assert len(latency_calls) > 0
    
    @pytest.mark.asyncio
    async def test_success_failure_metrics(self, circuit_breaker, mock_monitoring_service):
        """Test success and failure metrics recording."""
        async def successful_function():
            return "success"
        
        async def failing_function():
            raise ValueError("error")
        
        # Record success
        await circuit_breaker.call(successful_function)
        
        # Record failure
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        
        # Should record both success and failure metrics
        metric_calls = mock_monitoring_service.record_metric.call_args_list
        success_calls = [call for call in metric_calls if "successes" in str(call)]
        failure_calls = [call for call in metric_calls if "failures" in str(call)]
        
        assert len(success_calls) > 0
        assert len(failure_calls) > 0
    
    def test_get_stats(self, circuit_breaker):
        """Test getting circuit breaker statistics."""
        stats = circuit_breaker.get_stats()
        
        assert "name" in stats
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_requests" in stats
        assert "failure_rate" in stats
        assert "config" in stats
        
        assert stats["name"] == "test_service"
        assert stats["state"] == "closed"
        assert stats["failure_rate"] == 0.0  # No requests yet


class TestCircuitBreakerService:
    """Test circuit breaker service functionality."""
    
    def test_create_circuit_breaker(self, circuit_breaker_service):
        """Test creating circuit breakers."""
        cb = circuit_breaker_service.create_circuit_breaker("test_service")
        
        assert cb.name == "test_service"
        assert "test_service" in circuit_breaker_service.circuit_breakers
    
    def test_get_circuit_breaker(self, circuit_breaker_service):
        """Test getting existing circuit breakers."""
        # Create circuit breaker
        cb1 = circuit_breaker_service.create_circuit_breaker("service1")
        
        # Get existing circuit breaker
        cb2 = circuit_breaker_service.get_circuit_breaker("service1")
        assert cb1 is cb2
        
        # Get non-existent circuit breaker
        cb3 = circuit_breaker_service.get_circuit_breaker("nonexistent")
        assert cb3 is None
    
    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker(self, circuit_breaker_service):
        """Test calling function with circuit breaker protection."""
        async def test_function(value):
            return f"result: {value}"
        
        result = await circuit_breaker_service.call_with_circuit_breaker(
            "test_service", test_function, "test_value"
        )
        
        assert result == "result: test_value"
        assert "test_service" in circuit_breaker_service.circuit_breakers
    
    def test_get_all_stats(self, circuit_breaker_service):
        """Test getting statistics for all circuit breakers."""
        # Create some circuit breakers
        circuit_breaker_service.create_circuit_breaker("service1")
        circuit_breaker_service.create_circuit_breaker("service2")
        
        stats = circuit_breaker_service.get_all_stats()
        
        assert "service1" in stats
        assert "service2" in stats
        assert len(stats) == 2
    
    @pytest.mark.asyncio
    async def test_health_check(self, circuit_breaker_service):
        """Test circuit breaker service health check."""
        # Create circuit breakers in different states
        cb1 = circuit_breaker_service.create_circuit_breaker("healthy_service")
        cb2 = circuit_breaker_service.create_circuit_breaker("unhealthy_service")
        
        # Force one to open state
        await cb2._transition_to_open()
        
        health = await circuit_breaker_service.health_check()
        
        assert health["total_circuit_breakers"] == 2
        assert health["open_circuit_breakers"] == 1
        assert health["closed_circuit_breakers"] == 1
        assert health["health_score"] == 0.5  # 1 out of 2 healthy
        assert health["status"] == "degraded"
    
    @pytest.mark.asyncio
    async def test_force_open_all(self, circuit_breaker_service, mock_alert_service):
        """Test forcing all circuit breakers open."""
        # Create some circuit breakers
        circuit_breaker_service.create_circuit_breaker("service1")
        circuit_breaker_service.create_circuit_breaker("service2")
        
        await circuit_breaker_service.force_open_all()
        
        # All should be open
        for cb in circuit_breaker_service.circuit_breakers.values():
            assert cb.state == CircuitState.OPEN
        
        # Should send critical alert
        mock_alert_service.send_critical_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_force_close_all(self, circuit_breaker_service, mock_alert_service):
        """Test forcing all circuit breakers closed."""
        # Create and open circuit breakers
        cb1 = circuit_breaker_service.create_circuit_breaker("service1")
        cb2 = circuit_breaker_service.create_circuit_breaker("service2")
        await cb1._transition_to_open()
        await cb2._transition_to_open()
        
        await circuit_breaker_service.force_close_all()
        
        # All should be closed
        for cb in circuit_breaker_service.circuit_breakers.values():
            assert cb.state == CircuitState.CLOSED
        
        # Should send warning alert
        mock_alert_service.send_warning_alert.assert_called_once()


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_decorator_basic_usage(self, circuit_breaker_service):
        """Test basic decorator usage."""
        @circuit_breaker("decorated_service", circuit_breaker_service)
        async def decorated_function(value):
            return f"decorated: {value}"
        
        result = await decorated_function("test")
        assert result == "decorated: test"
        
        # Should create circuit breaker
        assert "decorated_service" in circuit_breaker_service.circuit_breakers
    
    @pytest.mark.asyncio
    async def test_decorator_with_config(self, circuit_breaker_service):
        """Test decorator with custom configuration."""
        config = CircuitBreakerConfig(failure_threshold=5)
        
        @circuit_breaker("configured_service", circuit_breaker_service, config)
        async def configured_function():
            return "configured"
        
        result = await configured_function()
        assert result == "configured"
        
        # Should use custom config
        cb = circuit_breaker_service.get_circuit_breaker("configured_service")
        assert cb.config.failure_threshold == 5
    
    @pytest.mark.asyncio
    async def test_decorator_failure_handling(self, circuit_breaker_service):
        """Test decorator failure handling."""
        @circuit_breaker("failing_service", circuit_breaker_service)
        async def failing_function():
            raise ValueError("decorator test error")
        
        with pytest.raises(ValueError):
            await failing_function()
        
        # Should record failure
        cb = circuit_breaker_service.get_circuit_breaker("failing_service")
        assert cb.stats.failure_count == 1


class TestCircuitBreakerEdgeCases:
    """Test circuit breaker edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(self, circuit_breaker):
        """Test handling of unexpected exceptions."""
        async def unexpected_error_function():
            raise RuntimeError("Unexpected error")  # Not the expected ValueError
        
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(unexpected_error_function)
        
        # Should not count as circuit breaker failure
        assert circuit_breaker.stats.failure_count == 0
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self, circuit_breaker):
        """Test concurrent calls to circuit breaker."""
        call_count = 0
        
        async def concurrent_function():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"call_{call_count}"
        
        # Make concurrent calls
        tasks = [circuit_breaker.call(concurrent_function) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert circuit_breaker.stats.success_count == 5
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_precision(self, circuit_breaker):
        """Test recovery timeout precision."""
        # Force circuit to open
        await circuit_breaker._transition_to_open()
        
        # Set precise failure time
        circuit_breaker.stats.last_failure_time = datetime.utcnow() - timedelta(seconds=4.9)
        
        async def test_function():
            return "test"
        
        # Should still be blocked (not enough time passed)
        with pytest.raises(CircuitBreakerException):
            await circuit_breaker.call(test_function)
        
        # Wait a bit more
        circuit_breaker.stats.last_failure_time = datetime.utcnow() - timedelta(seconds=5.1)
        
        # Should now allow call
        result = await circuit_breaker.call(test_function)
        assert result == "test"
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_manual_state_transitions(self, circuit_breaker, mock_alert_service):
        """Test manual state transitions."""
        # Manual force open
        await circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Manual force close
        await circuit_breaker.force_close()
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Should send appropriate alerts
        assert mock_alert_service.send_error_alert.call_count >= 1
        assert mock_alert_service.send_warning_alert.call_count >= 1


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with other services."""
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, circuit_breaker, mock_monitoring_service):
        """Test integration with monitoring service."""
        async def test_function():
            return "success"
        
        await circuit_breaker.call(test_function)
        
        # Should record metrics
        mock_monitoring_service.record_metric.assert_called()
        
        # Check metric names
        metric_calls = mock_monitoring_service.record_metric.call_args_list
        metric_names = [call[0][0] for call in metric_calls]
        
        expected_metrics = [
            "circuit_breaker_test_service_successes_total",
            "circuit_breaker_test_service_latency_seconds"
        ]
        
        for expected in expected_metrics:
            assert any(expected in name for name in metric_names)
    
    @pytest.mark.asyncio
    async def test_alert_integration(self, circuit_breaker, mock_alert_service):
        """Test integration with alert service."""
        async def failing_function():
            raise ValueError("Test error")
        
        # Fail enough times to open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_function)
        
        # Should send error alert when opening
        mock_alert_service.send_error_alert.assert_called_once()
        
        # Test recovery alert
        await circuit_breaker._transition_to_closed()
        mock_alert_service.send_warning_alert.assert_called_once()


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration options."""
    
    def test_custom_configuration(self, mock_monitoring_service, mock_alert_service):
        """Test circuit breaker with custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=30,
            success_threshold=5,
            timeout=2.0,
            expected_exception=ConnectionError
        )
        
        cb = CircuitBreaker(
            name="custom_service",
            config=config,
            monitoring_service=mock_monitoring_service,
            alert_service=mock_alert_service
        )
        
        assert cb.config.failure_threshold == 10
        assert cb.config.recovery_timeout == 30
        assert cb.config.success_threshold == 5
        assert cb.config.timeout == 2.0
        assert cb.config.expected_exception == ConnectionError
    
    def test_default_configuration(self, circuit_breaker_service):
        """Test circuit breaker with default configuration."""
        cb = circuit_breaker_service.create_circuit_breaker("default_service")
        
        # Should use default configuration
        assert cb.config.failure_threshold == 5
        assert cb.config.recovery_timeout == 60
        assert cb.config.success_threshold == 3
        assert cb.config.timeout == 30.0
        assert cb.config.expected_exception == Exception
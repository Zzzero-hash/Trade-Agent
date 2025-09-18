"""
Circuit Breaker Service

Implements circuit breaker pattern for external service failures,
providing graceful degradation and automatic recovery.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from src.services.monitoring_service import MonitoringService
from src.services.alert_service import ProductionAlertService


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker active, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0               # Request timeout in seconds
    expected_exception: type = Exception # Exception type that counts as failure


class CircuitBreakerStats:
    def __init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.state_changed_at = datetime.utcnow()
        self.total_requests = 0
        self.total_failures = 0


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Individual circuit breaker for a specific service/operation."""
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        monitoring_service: MonitoringService,
        alert_service: ProductionAlertService
    ):
        self.name = name
        self.config = config
        self.monitoring_service = monitoring_service
        self.alert_service = alert_service
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if we should attempt the call
            if not await self._should_attempt_call():
                await self._record_blocked_call()
                raise CircuitBreakerException(f"Circuit breaker {self.name} is OPEN")
            
            # If half-open, we're testing - only allow one call at a time
            if self.state == CircuitState.HALF_OPEN:
                return await self._attempt_call_half_open(func, *args, **kwargs)
            
            # Normal call (circuit closed)
            return await self._attempt_call_normal(func, *args, **kwargs)
    
    async def _should_attempt_call(self) -> bool:
        """Determine if we should attempt the call based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.stats.last_failure_time and 
                datetime.utcnow() - self.stats.last_failure_time >= 
                timedelta(seconds=self.config.recovery_timeout)):
                await self._transition_to_half_open()
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    async def _attempt_call_normal(self, func: Callable, *args, **kwargs) -> Any:
        """Attempt call in normal (closed) state."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Success
            await self._record_success()
            
            # Record latency
            latency = time.time() - start_time
            await self.monitoring_service.record_metric(
                f"circuit_breaker_{self.name}_latency_seconds",
                latency
            )
            
            return result
            
        except asyncio.TimeoutError as e:
            await self._record_failure(e)
            raise
        except self.config.expected_exception as e:
            await self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exception - don't count as circuit breaker failure
            self.logger.warning(f"Unexpected exception in {self.name}: {e}")
            raise
    
    async def _attempt_call_half_open(self, func: Callable, *args, **kwargs) -> Any:
        """Attempt call in half-open state (testing recovery)."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Success in half-open state
            await self._record_success()
            
            # Check if we should close the circuit
            if self.stats.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
            
            # Record latency
            latency = time.time() - start_time
            await self.monitoring_service.record_metric(
                f"circuit_breaker_{self.name}_latency_seconds",
                latency
            )
            
            return result
            
        except (asyncio.TimeoutError, self.config.expected_exception) as e:
            # Failure in half-open state - go back to open
            await self._record_failure(e)
            await self._transition_to_open()
            raise
    
    async def _record_success(self):
        """Record a successful call."""
        self.stats.success_count += 1
        self.stats.failure_count = 0  # Reset failure count on success
        self.stats.last_success_time = datetime.utcnow()
        
        await self.monitoring_service.record_metric(
            f"circuit_breaker_{self.name}_successes_total",
            1
        )
    
    async def _record_failure(self, exception: Exception):
        """Record a failed call."""
        self.stats.failure_count += 1
        self.stats.success_count = 0  # Reset success count on failure
        self.stats.last_failure_time = datetime.utcnow()
        self.stats.total_failures += 1
        
        await self.monitoring_service.record_metric(
            f"circuit_breaker_{self.name}_failures_total",
            1
        )
        
        self.logger.warning(f"Circuit breaker {self.name} recorded failure: {exception}")
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.stats.failure_count >= self.config.failure_threshold):
            await self._transition_to_open()
    
    async def _record_blocked_call(self):
        """Record a call that was blocked by the circuit breaker."""
        await self.monitoring_service.record_metric(
            f"circuit_breaker_{self.name}_blocked_calls_total",
            1
        )
    
    async def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.stats.state_changed_at = datetime.utcnow()
        
        self.logger.error(f"Circuit breaker {self.name} opened due to failures")
        
        await self.alert_service.send_error_alert(
            f"Circuit breaker opened: {self.name}",
            f"Circuit breaker {self.name} opened after {self.stats.failure_count} failures. "
            f"Service calls will fail fast for {self.config.recovery_timeout} seconds."
        )
        
        await self.monitoring_service.record_metric(
            f"circuit_breaker_{self.name}_state_changes_total",
            1,
            labels={"from_state": old_state.value, "to_state": self.state.value}
        )
    
    async def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.stats.state_changed_at = datetime.utcnow()
        self.stats.success_count = 0
        self.stats.failure_count = 0
        
        self.logger.info(f"Circuit breaker {self.name} transitioning to half-open for testing")
        
        await self.monitoring_service.record_metric(
            f"circuit_breaker_{self.name}_state_changes_total",
            1,
            labels={"from_state": old_state.value, "to_state": self.state.value}
        )
    
    async def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.stats.state_changed_at = datetime.utcnow()
        self.stats.success_count = 0
        self.stats.failure_count = 0
        
        self.logger.info(f"Circuit breaker {self.name} closed - service recovered")
        
        await self.alert_service.send_warning_alert(
            f"Circuit breaker recovered: {self.name}",
            f"Circuit breaker {self.name} closed after successful recovery test. "
            f"Service is now available again."
        )
        
        await self.monitoring_service.record_metric(
            f"circuit_breaker_{self.name}_state_changes_total",
            1,
            labels={"from_state": old_state.value, "to_state": self.state.value}
        )
    
    async def force_open(self):
        """Manually force circuit breaker to open."""
        await self._transition_to_open()
        self.logger.warning(f"Circuit breaker {self.name} manually forced open")
    
    async def force_close(self):
        """Manually force circuit breaker to close."""
        await self._transition_to_closed()
        self.logger.info(f"Circuit breaker {self.name} manually forced closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
            "state_changed_at": self.stats.state_changed_at.isoformat(),
            "failure_rate": self.stats.total_failures / max(self.stats.total_requests, 1),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }


class CircuitBreakerService:
    """Service to manage multiple circuit breakers."""
    
    def __init__(
        self,
        monitoring_service: MonitoringService,
        alert_service: ProductionAlertService
    ):
        self.monitoring_service = monitoring_service
        self.alert_service = alert_service
        self.logger = logging.getLogger(__name__)
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
        
    def create_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Create a new circuit breaker."""
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
        
        config = config or self._default_config
        
        circuit_breaker = CircuitBreaker(
            name=name,
            config=config,
            monitoring_service=self.monitoring_service,
            alert_service=self.alert_service
        )
        
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Created circuit breaker: {name}")
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get an existing circuit breaker."""
        return self.circuit_breakers.get(name)
    
    async def call_with_circuit_breaker(
        self,
        name: str,
        func: Callable,
        *args,
        config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        circuit_breaker = self.create_circuit_breaker(name, config)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: cb.get_stats()
            for name, cb in self.circuit_breakers.items()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all circuit breakers."""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.HALF_OPEN
        )
        
        health_score = 1.0
        if total_breakers > 0:
            health_score = (total_breakers - open_breakers) / total_breakers
        
        return {
            "total_circuit_breakers": total_breakers,
            "open_circuit_breakers": open_breakers,
            "half_open_circuit_breakers": half_open_breakers,
            "closed_circuit_breakers": total_breakers - open_breakers - half_open_breakers,
            "health_score": health_score,
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy"
        }
    
    async def force_open_all(self):
        """Force all circuit breakers to open (emergency stop)."""
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.force_open()
        
        await self.alert_service.send_critical_alert(
            "All circuit breakers forced open",
            "Emergency stop activated - all external service calls are blocked"
        )
    
    async def force_close_all(self):
        """Force all circuit breakers to close (recovery)."""
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.force_close()
        
        await self.alert_service.send_warning_alert(
            "All circuit breakers forced closed",
            "Manual recovery activated - all external service calls are enabled"
        )


# Decorator for easy circuit breaker usage
def circuit_breaker(
    name: str,
    service: CircuitBreakerService,
    config: Optional[CircuitBreakerConfig] = None
):
    """Decorator to add circuit breaker protection to a function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await service.call_with_circuit_breaker(
                name, func, *args, config=config, **kwargs
            )
        return wrapper
    return decorator
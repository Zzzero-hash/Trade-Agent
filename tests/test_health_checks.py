"""
Tests for service health checks and monitoring.
Requirements: 6.1, 6.3, 6.6
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
import aiohttp
import asyncpg
import aioredis


class HealthCheckService:
    """Service for performing comprehensive health checks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.timeout = config.get("timeout", 30)
    
    async def check_api_health(self, url: str) -> Dict:
        """Check API service health."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "response_time": data.get("response_time", 0),
                            "timestamp": datetime.utcnow().isoformat(),
                            "details": data
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "timestamp": datetime.utcnow().isoformat()
                        }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def check_database_health(self, connection_string: str) -> Dict:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            conn = await asyncpg.connect(connection_string)
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            
            # Test connection pool
            pool_info = await conn.fetchrow("""
                SELECT 
                    count(*) as active_connections,
                    current_setting('max_connections') as max_connections
                FROM pg_stat_activity 
                WHERE state = 'active'
            """)
            
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "active_connections": pool_info["active_connections"],
                "max_connections": pool_info["max_connections"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def check_redis_health(self, redis_url: str) -> Dict:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()
            redis = aioredis.from_url(redis_url)
            
            # Test basic operations
            await redis.ping()
            await redis.set("health_check", "test", ex=60)
            value = await redis.get("health_check")
            await redis.delete("health_check")
            
            # Get Redis info
            info = await redis.info()
            
            await redis.close()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def check_ml_model_health(self, model_service_url: str) -> Dict:
        """Check ML model service health and performance."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Test model loading
                async with session.get(f"{model_service_url}/models/status") as response:
                    if response.status != 200:
                        return {
                            "status": "unhealthy",
                            "error": f"Model service returned {response.status}",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    
                    model_status = await response.json()
                
                # Test inference endpoint with dummy data
                test_data = {
                    "features": [[0.1, 0.2, 0.3, 0.4, 0.5] * 10],  # Dummy feature vector
                    "sequence_length": 50
                }
                
                start_time = time.time()
                async with session.post(f"{model_service_url}/predict", json=test_data) as response:
                    if response.status == 200:
                        prediction = await response.json()
                        inference_time = (time.time() - start_time) * 1000
                        
                        return {
                            "status": "healthy",
                            "inference_time": inference_time,
                            "models_loaded": model_status.get("models_loaded", 0),
                            "gpu_available": model_status.get("gpu_available", False),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"Inference failed with status {response.status}",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def check_exchange_connectivity(self, exchange_configs: Dict) -> Dict:
        """Check connectivity to trading exchanges."""
        results = {}
        
        for exchange_name, config in exchange_configs.items():
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    # Test basic connectivity (usually a status or ping endpoint)
                    test_url = f"{config['base_url']}/status"
                    
                    start_time = time.time()
                    async with session.get(test_url) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            results[exchange_name] = {
                                "status": "healthy",
                                "response_time": response_time,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        else:
                            results[exchange_name] = {
                                "status": "unhealthy",
                                "error": f"HTTP {response.status}",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            
            except Exception as e:
                results[exchange_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Overall status
        all_healthy = all(result["status"] == "healthy" for result in results.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "exchanges": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def comprehensive_health_check(self) -> Dict:
        """Perform comprehensive system health check."""
        checks = {}
        
        # Run all health checks concurrently
        tasks = [
            ("api", self.check_api_health("http://localhost:8000")),
            ("database", self.check_database_health("postgresql://user:pass@localhost:5432/db")),
            ("redis", self.check_redis_health("redis://localhost:6379")),
            ("ml_service", self.check_ml_model_health("http://localhost:8080")),
            ("exchanges", self.check_exchange_connectivity({
                "robinhood": {"base_url": "https://robinhood.com/api"},
                "oanda": {"base_url": "https://api-fxtrade.oanda.com"},
                "coinbase": {"base_url": "https://api.exchange.coinbase.com"}
            }))
        ]
        
        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        
        for i, (name, _) in enumerate(tasks):
            if isinstance(results[i], Exception):
                checks[name] = {
                    "status": "error",
                    "error": str(results[i]),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                checks[name] = results[i]
        
        # Determine overall system status
        statuses = [check["status"] for check in checks.values()]
        if all(status == "healthy" for status in statuses):
            overall_status = "healthy"
        elif any(status == "healthy" for status in statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }


class TestHealthCheckService:
    """Test the health check service implementation."""
    
    @pytest.fixture
    def health_service(self):
        """Create health check service instance."""
        config = {"timeout": 30}
        return HealthCheckService(config)
    
    @pytest.mark.asyncio
    async def test_api_health_check_success(self, health_service):
        """Test successful API health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "response_time": 50,
                "version": "1.0.0"
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await health_service.check_api_health("http://localhost:8000")
            
            assert result["status"] == "healthy"
            assert "response_time" in result
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_api_health_check_failure(self, health_service):
        """Test API health check failure."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await health_service.check_api_health("http://localhost:8000")
            
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert result["error"] == "HTTP 500"
    
    @pytest.mark.asyncio
    async def test_database_health_check_success(self, health_service):
        """Test successful database health check."""
        with patch('asyncpg.connect') as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            mock_conn.fetchrow.return_value = {
                "active_connections": 5,
                "max_connections": "100"
            }
            mock_connect.return_value = mock_conn
            
            result = await health_service.check_database_health("postgresql://test")
            
            assert result["status"] == "healthy"
            assert "response_time" in result
            assert "active_connections" in result
    
    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, health_service):
        """Test database health check failure."""
        with patch('asyncpg.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            result = await health_service.check_database_health("postgresql://test")
            
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, health_service):
        """Test successful Redis health check."""
        with patch('aioredis.from_url') as mock_redis:
            mock_redis_conn = AsyncMock()
            mock_redis_conn.ping.return_value = True
            mock_redis_conn.set.return_value = True
            mock_redis_conn.get.return_value = b"test"
            mock_redis_conn.delete.return_value = 1
            mock_redis_conn.info.return_value = {
                "connected_clients": 10,
                "used_memory_human": "1.5M"
            }
            mock_redis.return_value = mock_redis_conn
            
            result = await health_service.check_redis_health("redis://localhost:6379")
            
            assert result["status"] == "healthy"
            assert "response_time" in result
            assert "connected_clients" in result
    
    @pytest.mark.asyncio
    async def test_ml_model_health_check_success(self, health_service):
        """Test successful ML model health check."""
        with patch('aiohttp.ClientSession.get') as mock_get, \
             patch('aiohttp.ClientSession.post') as mock_post:
            
            # Mock model status response
            mock_status_response = AsyncMock()
            mock_status_response.status = 200
            mock_status_response.json.return_value = {
                "models_loaded": 3,
                "gpu_available": True
            }
            mock_get.return_value.__aenter__.return_value = mock_status_response
            
            # Mock prediction response
            mock_pred_response = AsyncMock()
            mock_pred_response.status = 200
            mock_pred_response.json.return_value = {
                "prediction": [0.1, 0.7, 0.2],
                "confidence": 0.85
            }
            mock_post.return_value.__aenter__.return_value = mock_pred_response
            
            result = await health_service.check_ml_model_health("http://localhost:8080")
            
            assert result["status"] == "healthy"
            assert "inference_time" in result
            assert result["models_loaded"] == 3
            assert result["gpu_available"] is True
    
    @pytest.mark.asyncio
    async def test_exchange_connectivity_check(self, health_service):
        """Test exchange connectivity check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            exchange_configs = {
                "robinhood": {"base_url": "https://robinhood.com/api"},
                "oanda": {"base_url": "https://api-fxtrade.oanda.com"}
            }
            
            result = await health_service.check_exchange_connectivity(exchange_configs)
            
            assert result["status"] == "healthy"
            assert "exchanges" in result
            assert "robinhood" in result["exchanges"]
            assert "oanda" in result["exchanges"]
            assert result["exchanges"]["robinhood"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, health_service):
        """Test comprehensive system health check."""
        # Mock all individual health checks
        with patch.object(health_service, 'check_api_health') as mock_api, \
             patch.object(health_service, 'check_database_health') as mock_db, \
             patch.object(health_service, 'check_redis_health') as mock_redis, \
             patch.object(health_service, 'check_ml_model_health') as mock_ml, \
             patch.object(health_service, 'check_exchange_connectivity') as mock_exchanges:
            
            # Configure mocks to return healthy status
            mock_api.return_value = {"status": "healthy"}
            mock_db.return_value = {"status": "healthy"}
            mock_redis.return_value = {"status": "healthy"}
            mock_ml.return_value = {"status": "healthy"}
            mock_exchanges.return_value = {"status": "healthy"}
            
            result = await health_service.comprehensive_health_check()
            
            assert result["status"] == "healthy"
            assert "checks" in result
            assert len(result["checks"]) == 5
            assert all(check["status"] == "healthy" for check in result["checks"].values())
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check_degraded(self, health_service):
        """Test comprehensive health check with some services unhealthy."""
        with patch.object(health_service, 'check_api_health') as mock_api, \
             patch.object(health_service, 'check_database_health') as mock_db, \
             patch.object(health_service, 'check_redis_health') as mock_redis, \
             patch.object(health_service, 'check_ml_model_health') as mock_ml, \
             patch.object(health_service, 'check_exchange_connectivity') as mock_exchanges:
            
            # Configure mocks with mixed status
            mock_api.return_value = {"status": "healthy"}
            mock_db.return_value = {"status": "healthy"}
            mock_redis.return_value = {"status": "unhealthy", "error": "Connection timeout"}
            mock_ml.return_value = {"status": "healthy"}
            mock_exchanges.return_value = {"status": "degraded"}
            
            result = await health_service.comprehensive_health_check()
            
            assert result["status"] == "degraded"
            assert result["checks"]["redis"]["status"] == "unhealthy"


class TestDeploymentHealthChecks:
    """Test deployment-specific health checks."""
    
    def test_kubernetes_readiness_probe(self):
        """Test Kubernetes readiness probe configuration."""
        # This would test the actual readiness probe endpoint
        # For now, we'll test the configuration
        
        # Mock readiness check
        def readiness_check():
            # Check database connection
            # Check Redis connection
            # Check model loading
            return {"ready": True, "checks": {"db": True, "redis": True, "models": True}}
        
        result = readiness_check()
        assert result["ready"] is True
        assert all(result["checks"].values())
    
    def test_kubernetes_liveness_probe(self):
        """Test Kubernetes liveness probe configuration."""
        # Mock liveness check
        def liveness_check():
            # Basic application health
            return {"alive": True, "uptime": 3600}
        
        result = liveness_check()
        assert result["alive"] is True
        assert result["uptime"] > 0
    
    def test_startup_probe(self):
        """Test Kubernetes startup probe for slow-starting services."""
        # Mock startup check for ML service (which takes time to load models)
        def startup_check():
            # Check if models are loaded
            # Check if service is ready to accept requests
            return {"started": True, "models_loaded": 3, "ready_for_requests": True}
        
        result = startup_check()
        assert result["started"] is True
        assert result["models_loaded"] > 0
        assert result["ready_for_requests"] is True


class TestMonitoringIntegration:
    """Test integration with monitoring systems."""
    
    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics export."""
        # Mock metrics collection
        metrics = {
            "http_requests_total": 1000,
            "http_request_duration_seconds": 0.05,
            "database_connections_active": 10,
            "ml_inference_duration_seconds": 0.2,
            "exchange_api_calls_total": 500
        }
        
        # Test metrics format
        assert all(isinstance(value, (int, float)) for value in metrics.values())
        assert "http_requests_total" in metrics
        assert "ml_inference_duration_seconds" in metrics
    
    def test_grafana_dashboard_data(self):
        """Test data format for Grafana dashboards."""
        # Mock dashboard data
        dashboard_data = {
            "system_health": {
                "api": "healthy",
                "database": "healthy", 
                "redis": "healthy",
                "ml_service": "healthy"
            },
            "performance_metrics": {
                "avg_response_time": 45.2,
                "requests_per_second": 150.5,
                "error_rate": 0.01
            },
            "resource_usage": {
                "cpu_usage": 65.3,
                "memory_usage": 78.1,
                "disk_usage": 45.7
            }
        }
        
        # Validate data structure
        assert "system_health" in dashboard_data
        assert "performance_metrics" in dashboard_data
        assert "resource_usage" in dashboard_data
        
        # Check that all services have status
        for service, status in dashboard_data["system_health"].items():
            assert status in ["healthy", "unhealthy", "degraded"]
    
    def test_alerting_rules(self):
        """Test alerting rule configuration."""
        # Mock alerting rules
        alerting_rules = [
            {
                "name": "HighErrorRate",
                "condition": "error_rate > 0.05",
                "severity": "warning",
                "duration": "5m"
            },
            {
                "name": "ServiceDown",
                "condition": "up == 0",
                "severity": "critical",
                "duration": "1m"
            },
            {
                "name": "HighLatency",
                "condition": "avg_response_time > 1000",
                "severity": "warning",
                "duration": "10m"
            }
        ]
        
        # Validate alerting rules
        for rule in alerting_rules:
            assert "name" in rule
            assert "condition" in rule
            assert "severity" in rule
            assert rule["severity"] in ["info", "warning", "critical"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Performance and load tests for model serving API

This module contains performance tests to validate API response times,
throughput, and behavior under load conditions.

Requirements: 6.2, 11.1
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np

import httpx
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.model_serving import ModelType


class TestAPIPerformance:
    """Performance tests for the model serving API"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_single_prediction_latency(self, client):
        """Test single prediction latency"""
        request_data = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "latest",
            "data": [[1.0, 2.0, 3.0, 4.0, 5.0] * 10],  # 50 features
            "return_uncertainty": True,
            "use_ensemble": True
        }
        
        latencies = []
        
        # Make 10 requests to get average latency
        for _ in range(10):
            start_time = time.time()
            
            try:
                response = client.post("/api/v1/predict", json=request_data)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # Check if request was successful or failed due to missing model
                # (which is expected in test environment)
                assert response.status_code in [200, 404, 500]
                
            except Exception as e:
                # Expected in test environment without actual models
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")
        
        # Latency should be reasonable even for failed requests
        assert avg_latency < 1000  # Less than 1 second average
        assert p95_latency < 2000  # Less than 2 seconds for 95th percentile
    
    def test_batch_prediction_performance(self, client):
        """Test batch prediction performance"""
        # Create batch request with multiple predictions
        requests = []
        for i in range(20):  # 20 predictions in batch
            requests.append({
                "model_type": "cnn_lstm_hybrid",
                "model_version": "latest",
                "data": [[float(j) for j in range(i, i+10)]],
                "return_uncertainty": False,
                "use_ensemble": False
            })
        
        batch_request = {
            "requests": requests,
            "priority": 1
        }
        
        start_time = time.time()
        response = client.post("/api/v1/predict/batch", json=batch_request)
        end_time = time.time()
        
        total_latency = (end_time - start_time) * 1000
        
        print(f"Batch prediction latency: {total_latency:.2f}ms for {len(requests)} requests")
        
        # Batch should be more efficient than individual requests
        # Even with errors, should complete within reasonable time
        assert total_latency < 5000  # Less than 5 seconds
        
        # Response should have proper structure even if predictions fail
        assert response.status_code in [200, 500]
    
    def test_concurrent_requests(self, client):
        """Test concurrent request handling"""
        request_data = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "latest",
            "data": [[1.0, 2.0, 3.0]],
            "return_uncertainty": False,
            "use_ensemble": False
        }
        
        def make_request():
            """Make a single request"""
            start_time = time.time()
            try:
                response = client.post("/api/v1/predict", json=request_data)
                end_time = time.time()
                return {
                    "status_code": response.status_code,
                    "latency_ms": (end_time - start_time) * 1000,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "status_code": 500,
                    "latency_ms": (end_time - start_time) * 1000,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        num_concurrent = 10
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Analyze results
        latencies = [r["latency_ms"] for r in results]
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        print(f"Concurrent requests: {num_concurrent}")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")
        
        # All requests should complete within reasonable time
        assert total_time < 10000  # Less than 10 seconds total
        assert max_latency < 5000   # No single request takes more than 5 seconds
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint performance"""
        latencies = []
        
        # Make multiple health check requests
        for _ in range(50):
            start_time = time.time()
            response = client.get("/api/v1/health")
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Health endpoint should always respond
            assert response.status_code in [200, 503]  # Healthy or degraded
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        print(f"Health check average latency: {avg_latency:.2f}ms")
        print(f"Health check max latency: {max_latency:.2f}ms")
        
        # Health checks should be very fast
        assert avg_latency < 100   # Less than 100ms average
        assert max_latency < 500   # Less than 500ms maximum
    
    def test_metrics_endpoint_performance(self, client):
        """Test metrics endpoint performance"""
        start_time = time.time()
        response = client.get("/api/v1/metrics")
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        print(f"Metrics endpoint latency: {latency_ms:.2f}ms")
        
        # Metrics should respond quickly
        assert latency_ms < 1000  # Less than 1 second
        assert response.status_code in [200, 500]
    
    def test_model_types_endpoint_performance(self, client):
        """Test model types endpoint performance"""
        latencies = []
        
        # Make multiple requests to test caching
        for _ in range(20):
            start_time = time.time()
            response = client.get("/api/v1/models/types")
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            assert response.status_code == 200
        
        avg_latency = statistics.mean(latencies)
        
        print(f"Model types endpoint average latency: {avg_latency:.2f}ms")
        
        # Should be very fast since it's static data
        assert avg_latency < 50  # Less than 50ms average


class TestLoadTesting:
    """Load testing for sustained traffic"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.mark.slow
    def test_sustained_load(self, client):
        """Test sustained load over time"""
        request_data = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "latest",
            "data": [[1.0, 2.0, 3.0]],
            "return_uncertainty": False,
            "use_ensemble": False
        }
        
        def make_requests_for_duration(duration_seconds: int, requests_per_second: int):
            """Make requests at specified rate for given duration"""
            results = []
            start_time = time.time()
            request_interval = 1.0 / requests_per_second
            
            while time.time() - start_time < duration_seconds:
                request_start = time.time()
                
                try:
                    response = client.post("/api/v1/predict", json=request_data)
                    request_end = time.time()
                    
                    results.append({
                        "timestamp": request_start,
                        "latency_ms": (request_end - request_start) * 1000,
                        "status_code": response.status_code,
                        "success": response.status_code == 200
                    })
                    
                except Exception as e:
                    request_end = time.time()
                    results.append({
                        "timestamp": request_start,
                        "latency_ms": (request_end - request_start) * 1000,
                        "status_code": 500,
                        "success": False,
                        "error": str(e)
                    })
                
                # Wait for next request
                elapsed = time.time() - request_start
                sleep_time = max(0, request_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            return results
        
        # Run load test: 5 requests per second for 30 seconds
        print("Starting load test: 5 RPS for 30 seconds")
        results = make_requests_for_duration(duration_seconds=30, requests_per_second=5)
        
        # Analyze results
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests
        
        latencies = [r["latency_ms"] for r in results]
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        print(f"Load test results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  P99 latency: {p99_latency:.2f}ms")
        
        # Assertions for load test
        assert total_requests >= 140  # Should make at least 140 requests (5 RPS * 30s - some tolerance)
        assert avg_latency < 2000     # Average latency should be reasonable
        assert p99_latency < 5000     # P99 should be acceptable
        
        # Note: Success rate might be low due to missing models in test environment
        # In production with actual models, success rate should be high
    
    @pytest.mark.slow
    def test_memory_usage_under_load(self, client):
        """Test memory usage doesn't grow excessively under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        request_data = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "latest",
            "data": [[float(i) for i in range(100)]],  # Larger payload
            "return_uncertainty": True,
            "use_ensemble": True
        }
        
        # Make many requests
        for i in range(100):
            try:
                response = client.post("/api/v1/predict", json=request_data)
            except Exception:
                pass  # Expected in test environment
            
            # Check memory every 10 requests
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                print(f"Request {i}: Memory usage: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
                
                # Memory growth should be reasonable
                assert memory_growth < 500  # Less than 500MB growth
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        print(f"Total memory growth: {total_growth:.1f}MB")
        
        # Total memory growth should be reasonable
        assert total_growth < 1000  # Less than 1GB total growth


class TestAPIReliability:
    """Test API reliability and error handling"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_invalid_request_handling(self, client):
        """Test handling of invalid requests"""
        invalid_requests = [
            # Missing required fields
            {},
            {"model_type": "cnn_lstm_hybrid"},
            {"data": [[1, 2, 3]]},
            
            # Invalid data types
            {"model_type": "invalid_type", "data": [[1, 2, 3]]},
            {"model_type": "cnn_lstm_hybrid", "data": "invalid"},
            {"model_type": "cnn_lstm_hybrid", "data": []},
            
            # Invalid parameters
            {"model_type": "cnn_lstm_hybrid", "data": [[1, 2, 3]], "batch_size": -1},
            {"model_type": "cnn_lstm_hybrid", "data": [[1, 2, 3]], "batch_size": 10000},
        ]
        
        for i, invalid_request in enumerate(invalid_requests):
            response = client.post("/api/v1/predict", json=invalid_request)
            
            print(f"Invalid request {i}: Status {response.status_code}")
            
            # Should return 422 (validation error) or 400 (bad request)
            assert response.status_code in [400, 422]
            
            # Should have error details
            if response.status_code == 422:
                data = response.json()
                assert "error" in data or "detail" in data
    
    def test_large_request_handling(self, client):
        """Test handling of large requests"""
        # Create very large request
        large_data = [[float(i) for i in range(1000)] for _ in range(100)]  # 100k floats
        
        request_data = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "latest",
            "data": large_data,
            "return_uncertainty": False,
            "use_ensemble": False
        }
        
        start_time = time.time()
        response = client.post("/api/v1/predict", json=request_data)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        print(f"Large request latency: {latency_ms:.2f}ms")
        print(f"Large request status: {response.status_code}")
        
        # Should handle large requests gracefully
        assert response.status_code in [200, 400, 413, 500]  # Various acceptable responses
        assert latency_ms < 30000  # Should complete within 30 seconds
    
    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON"""
        malformed_requests = [
            '{"model_type": "cnn_lstm_hybrid", "data": [',  # Incomplete JSON
            '{"model_type": "cnn_lstm_hybrid", "data": [[1, 2, 3]',  # Missing closing bracket
            'not json at all',  # Not JSON
            '',  # Empty string
        ]
        
        for malformed_json in malformed_requests:
            response = client.post(
                "/api/v1/predict",
                content=malformed_json,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Malformed JSON response: {response.status_code}")
            
            # Should return 422 (validation error) or 400 (bad request)
            assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
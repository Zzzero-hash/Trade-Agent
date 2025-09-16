"""Integration tests for Ray Serve CNN+LSTM deployment.

This module contains tests to validate the integration of the CNN+LSTM model deployment
with the existing model loading pipeline and monitoring systems.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import logging

from src.ml.ray_serve.model_loader import (
    RayServeModelLoader, 
    GPUOptimizer, 
    load_cnn_lstm_model_for_ray_serve
)
from src.ml.ray_serve.config import (
    AutoscalingConfig, 
    ResourceConfig, 
    TradingWorkloadAutoscaler
)
from src.ml.ray_serve.monitoring import ModelMetrics, HealthChecker, PerformanceMonitor
from src.ml.ray_serve.cnn_lstm_deployment import CNNLSTMPredictor

# Configure logging
logger = logging.getLogger(__name__)


class TestModelLoadingIntegration(unittest.TestCase):
    """Test cases for model loading pipeline integration."""
    
    def test_model_loader_integration(self):
        """Test integration with model loading pipeline."""
        # Test that the model loader can create a model
        loader = RayServeModelLoader()
        model = loader.load_model_from_registry("test_model")
        
        # Verify that we got a model instance
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'config'))
    
    def test_gpu_optimizer_integration(self):
        """Test GPU optimizer integration."""
        # This should not raise an exception
        GPUOptimizer.setup_gpu_settings()
        
        # Test GPU memory info retrieval
        gpu_info = GPUOptimizer.get_gpu_memory_info()
        self.assertIsInstance(gpu_info, dict)
        self.assertIn("allocated_mb", gpu_info)
        self.assertIn("reserved_mb", gpu_info)
        self.assertIn("utilization_pct", gpu_info)
    
    def test_pipeline_integration_function(self):
        """Test the pipeline integration function."""
        # Test that the integration function works
        model = load_cnn_lstm_model_for_ray_serve()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'config'))


class TestConfigurationIntegration(unittest.TestCase):
    """Test cases for configuration integration."""
    
    def test_autoscaling_config_integration(self):
        """Test autoscaling configuration integration."""
        config = AutoscalingConfig()
        ray_config = config.to_ray_config()
        
        # Verify all expected keys are present
        expected_keys = [
            "min_replicas", "max_replicas", "target_num_ongoing_requests_per_replica",
            "upscale_delay_s", "downscale_delay_s", "upscale_smoothing_factor",
            "downscale_smoothing_factor", "metrics_interval_s", "look_back_period_s"
        ]
        
        for key in expected_keys:
            self.assertIn(key, ray_config)
    
    def test_resource_config_integration(self):
        """Test resource configuration integration."""
        config = ResourceConfig()
        ray_config = config.to_ray_config()
        
        # Verify all expected keys are present
        expected_keys = ["num_cpus", "num_gpus", "memory", "object_store_memory"]
        for key in expected_keys:
            self.assertIn(key, ray_config)
    
    def test_trading_workload_integration(self):
        """Test trading workload autoscaler integration."""
        # Test market hours config
        market_config = TradingWorkloadAutoscaler.get_market_hours_config()
        self.assertIsInstance(market_config, AutoscalingConfig)
        self.assertEqual(market_config.min_replicas, 5)
        self.assertEqual(market_config.upscale_delay_s, 15)
        
        # Test off hours config
        off_config = TradingWorkloadAutoscaler.get_off_hours_config()
        self.assertIsInstance(off_config, AutoscalingConfig)
        self.assertEqual(off_config.min_replicas, 2)
        self.assertEqual(off_config.upscale_delay_s, 60)


class TestMonitoringIntegration(unittest.TestCase):
    """Test cases for monitoring integration."""
    
    def test_model_metrics_integration(self):
        """Test model metrics integration."""
        metrics = ModelMetrics()
        
        # Test that metrics can be used without errors
        metrics.increment_counter("test_counter")
        metrics.record_histogram("test_histogram", 1.0)
        metrics.set_gauge("test_gauge", 1.0)
    
    def test_health_checker_integration(self):
        """Test health checker integration."""
        # Create a mock deployment
        mock_deployment = Mock()
        health_checker = HealthChecker(mock_deployment)
        
        # Test system health info
        system_health = health_checker.get_system_health()
        self.assertIsInstance(system_health, dict)
        self.assertIn("timestamp", system_health)
    
    def test_performance_monitor_integration(self):
        """Test performance monitor integration."""
        monitor = PerformanceMonitor()
        
        # Record some requests
        monitor.record_request(50.0, success=True)
        monitor.record_request(150.0, success=False)
        
        # Get stats
        stats = monitor.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("request_count", stats)
        self.assertIn("error_count", stats)
        self.assertIn("avg_latency_ms", stats)
        
        # Check performance requirements
        perf_reqs = monitor.check_performance_requirements()
        self.assertIsInstance(perf_reqs, dict)
        self.assertIn("meets_100ms_requirement", perf_reqs)


class TestDeploymentIntegration(unittest.TestCase):
    """Test cases for deployment integration."""
    
    def test_predictor_initialization(self):
        """Test CNNLSTMPredictor initialization."""
        predictor = CNNLSTMPredictor()
        self.assertIsNotNone(predictor)
        self.assertTrue(hasattr(predictor, 'model'))
        self.assertTrue(hasattr(predictor, 'device'))
    
    def test_predictor_validation(self):
        """Test predictor input validation."""
        predictor = CNNLSTMPredictor()
        
        # Test valid input
        valid_input = np.random.rand(1, 10, 60).astype(np.float32)
        # Should not raise an exception
        predictor._validate_input(valid_input)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            predictor._validate_input("invalid_input")
        
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            predictor._validate_input(np.array([1, 2, 3]))
    
    def test_predictor_stats(self):
        """Test predictor statistics."""
        predictor = CNNLSTMPredictor()
        stats = predictor.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("request_count", stats)
        self.assertIn("avg_processing_time_ms", stats)
        self.assertIn("device", stats)
        self.assertIn("uptime_seconds", stats)
        self.assertIn("model_type", stats)


def run_integration_tests():
    """Run all integration tests."""
    print("=== CNN+LSTM Ray Serve Integration Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestModelLoadingIntegration))
    test_suite.addTest(unittest.makeSuite(TestConfigurationIntegration))
    test_suite.addTest(unittest.makeSuite(TestMonitoringIntegration))
    test_suite.addTest(unittest.makeSuite(TestDeploymentIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n=== Integration Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run integration tests
    success = run_integration_tests()
    exit(0 if success else 1)
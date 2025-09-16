"""Tests for the A/B testing framework for Ray Serve deployments."""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.ml.ray_serve.ab_testing import (
    VariantConfig, 
    VariantStatus, 
    VariantMetrics, 
    StatisticalTestResult, 
    ABTestExperiment, 
    RayServeABTestManager
)


class TestVariantConfig(unittest.TestCase):
    """Test cases for VariantConfig."""

    def test_variant_config_creation(self):
        """Test creating a VariantConfig instance."""
        config = VariantConfig(
            name="test_variant",
            model_path="models/test_model.pth",
            weight=0.5,
            metadata={"description": "Test variant"}
        )
        
        self.assertEqual(config.name, "test_variant")
        self.assertEqual(config.model_path, "models/test_model.pth")
        self.assertEqual(config.weight, 0.5)
        self.assertEqual(config.status, VariantStatus.ACTIVE)
        self.assertEqual(config.metadata["description"], "Test variant")

    def test_variant_config_default_metadata(self):
        """Test that VariantConfig creates default metadata."""
        config = VariantConfig(
            name="test_variant",
            model_path="models/test_model.pth",
            weight=0.5
        )
        
        self.assertIsNotNone(config.metadata)
        self.assertIsInstance(config.metadata, dict)
        self.assertEqual(len(config.metadata), 0)


class TestVariantMetrics(unittest.TestCase):
    """Test cases for VariantMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = VariantMetrics()

    def test_initial_state(self):
        """Test initial state of metrics."""
        self.assertEqual(self.metrics.requests, 0)
        self.assertEqual(self.metrics.errors, 0)
        self.assertEqual(self.metrics.total_latency_ms, 0.0)
        self.assertEqual(self.metrics.total_processing_time_ms, 0.0)
        self.assertEqual(self.metrics.successful_predictions, 0)
        self.assertEqual(len(self.metrics.confidence_scores), 0)
        self.assertEqual(len(self.metrics.prediction_latencies), 0)

    def test_add_request_success(self):
        """Test adding a successful request."""
        self.metrics.add_request(
            latency_ms=50.0,
            processing_time_ms=45.0,
            confidence_score=0.85
        )
        
        self.assertEqual(self.metrics.requests, 1)
        self.assertEqual(self.metrics.errors, 0)
        self.assertEqual(self.metrics.successful_predictions, 1)
        self.assertEqual(self.metrics.total_latency_ms, 50.0)
        self.assertEqual(self.metrics.total_processing_time_ms, 45.0)
        self.assertEqual(len(self.metrics.confidence_scores), 1)
        self.assertEqual(self.metrics.confidence_scores[0], 0.85)
        self.assertEqual(len(self.metrics.prediction_latencies), 1)
        self.assertEqual(self.metrics.prediction_latencies[0], 50.0)

    def test_add_request_error(self):
        """Test adding a failed request."""
        self.metrics.add_request(
            latency_ms=100.0,
            processing_time_ms=95.0,
            error=True
        )
        
        self.assertEqual(self.metrics.requests, 1)
        self.assertEqual(self.metrics.errors, 1)
        self.assertEqual(self.metrics.successful_predictions, 0)
        self.assertEqual(self.metrics.total_latency_ms, 100.0)
        self.assertEqual(self.metrics.total_processing_time_ms, 95.0)
        self.assertEqual(len(self.metrics.confidence_scores), 0)
        self.assertEqual(len(self.metrics.prediction_latencies), 1)

    def test_avg_latency_ms(self):
        """Test average latency calculation."""
        # No requests
        self.assertEqual(self.metrics.avg_latency_ms, 0.0)
        
        # Add requests
        self.metrics.add_request(latency_ms=50.0, processing_time_ms=45.0)
        self.metrics.add_request(latency_ms=60.0, processing_time_ms=55.0)
        
        self.assertEqual(self.metrics.avg_latency_ms, 55.0)

    def test_error_rate(self):
        """Test error rate calculation."""
        # No requests
        self.assertEqual(self.metrics.error_rate, 0.0)
        
        # Add successful request
        self.metrics.add_request(latency_ms=50.0, processing_time_ms=45.0)
        self.assertEqual(self.metrics.error_rate, 0.0)
        
        # Add failed request
        self.metrics.add_request(latency_ms=100.0, processing_time_ms=95.0, error=True)
        self.assertEqual(self.metrics.error_rate, 0.5)

    def test_avg_confidence(self):
        """Test average confidence calculation."""
        # No confidence scores
        self.assertEqual(self.metrics.avg_confidence, 0.0)
        
        # Add requests with confidence scores
        self.metrics.add_request(latency_ms=50.0, processing_time_ms=45.0, confidence_score=0.8)
        self.metrics.add_request(latency_ms=60.0, processing_time_ms=55.0, confidence_score=0.9)
        
        self.assertEqual(self.metrics.avg_confidence, 0.85)


class TestStatisticalTestResult(unittest.TestCase):
    """Test cases for StatisticalTestResult."""

    def test_creation(self):
        """Test creating a StatisticalTestResult."""
        result = StatisticalTestResult(
            test_name="test_ttest",
            statistic=2.5,
            p_value=0.01,
            significant=True,
            confidence_level=0.95,
            effect_size=0.3,
            description="Test t-test"
        )
        
        self.assertEqual(result.test_name, "test_ttest")
        self.assertEqual(result.statistic, 2.5)
        self.assertEqual(result.p_value, 0.01)
        self.assertTrue(result.significant)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.effect_size, 0.3)
        self.assertEqual(result.description, "Test t-test")

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = StatisticalTestResult(
            test_name="test_ttest",
            statistic=2.5,
            p_value=0.01,
            significant=True,
            confidence_level=0.95
        )
        
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["test_name"], "test_ttest")
        self.assertEqual(result_dict["statistic"], 2.5)
        self.assertEqual(result_dict["p_value"], 0.01)
        self.assertTrue(result_dict["significant"])
        self.assertEqual(result_dict["confidence_level"], 0.95)


class TestABTestExperiment(unittest.TestCase):
    """Test cases for ABTestExperiment."""

    def setUp(self):
        """Set up test fixtures."""
        self.variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.3),
            VariantConfig(name="variant_b", model_path="models/variant_b.pth", weight=0.2)
        ]
        self.experiment = ABTestExperiment(
            experiment_id="test_exp",
            variants=self.variants,
            duration_hours=24,
            confidence_level=0.95
        )

    def test_experiment_creation(self):
        """Test creating an ABTestExperiment."""
        self.assertEqual(self.experiment.experiment_id, "test_exp")
        self.assertEqual(len(self.experiment.variants), 3)
        self.assertEqual(self.experiment.confidence_level, 0.95)
        self.assertEqual(self.experiment.status, "active")
        
        # Check that variant metrics were created
        self.assertEqual(len(self.experiment.variant_metrics), 3)
        for variant_name in ["control", "variant_a", "variant_b"]:
            self.assertIn(variant_name, self.experiment.variant_metrics)
            self.assertIsInstance(self.experiment.variant_metrics[variant_name], VariantMetrics)

    def test_invalid_weights(self):
        """Test that invalid weights raise an error."""
        invalid_variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.6),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.5)
        ]
        
        with self.assertRaises(ValueError):
            ABTestExperiment("invalid_exp", invalid_variants)

    def test_get_variant_for_request(self):
        """Test getting variant for a request."""
        # Test with active experiment
        variant = self.experiment.get_variant_for_request("test_request_1")
        self.assertIn(variant, ["control", "variant_a", "variant_b"])
        
        # Test consistent assignment
        variant1 = self.experiment.get_variant_for_request("test_request_1")
        variant2 = self.experiment.get_variant_for_request("test_request_1")
        self.assertEqual(variant1, variant2)

    def test_record_metrics(self):
        """Test recording metrics."""
        self.experiment.record_metrics(
            variant_name="control",
            latency_ms=50.0,
            processing_time_ms=45.0,
            confidence_score=0.85
        )
        
        metrics = self.experiment.variant_metrics["control"]
        self.assertEqual(metrics.requests, 1)
        self.assertEqual(metrics.successful_predictions, 1)
        self.assertEqual(metrics.total_latency_ms, 50.0)

    def test_get_results(self):
        """Test getting experiment results."""
        # Add some metrics
        self.experiment.record_metrics(
            variant_name="control",
            latency_ms=50.0,
            processing_time_ms=45.0,
            confidence_score=0.85
        )
        
        results = self.experiment.get_results()
        self.assertIsInstance(results, dict)
        self.assertEqual(results["experiment_id"], "test_exp")
        self.assertIn("variants", results)
        self.assertIn("control", results["variants"])
        
        control_results = results["variants"]["control"]
        self.assertIn("config", control_results)
        self.assertIn("metrics", control_results)
        self.assertEqual(control_results["metrics"]["requests"], 1)

    def test_perform_statistical_tests(self):
        """Test performing statistical tests."""
        # Add sufficient data for statistical testing
        np.random.seed(42)  # For reproducible results
        
        # Add data for control variant
        for i in range(50):
            latency = np.random.normal(50, 10)  # Mean 50ms, std 10ms
            self.experiment.record_metrics(
                variant_name="control",
                latency_ms=latency,
                processing_time_ms=latency * 0.9,
                confidence_score=np.random.uniform(0.7, 0.9)
            )
        
        # Add data for variant_a
        for i in range(50):
            latency = np.random.normal(45, 8)  # Mean 45ms, std 8ms (faster)
            self.experiment.record_metrics(
                variant_name="variant_a",
                latency_ms=latency,
                processing_time_ms=latency * 0.9,
                confidence_score=np.random.uniform(0.75, 0.95)
            )
        
        # Perform tests
        tests = self.experiment.perform_statistical_tests()
        self.assertIsInstance(tests, list)
        
        # Check that we have test results
        if len(tests) > 0:
            for test in tests:
                self.assertIsInstance(test, StatisticalTestResult)
                self.assertIsInstance(test.test_name, str)
                self.assertIsInstance(test.statistic, (int, float))
                self.assertIsInstance(test.p_value, (int, float))
                self.assertIsInstance(test.significant, bool)

    def test_get_statistical_summary(self):
        """Test getting statistical summary."""
        # Add some data
        self.experiment.record_metrics(
            variant_name="control",
            latency_ms=50.0,
            processing_time_ms=45.0
        )
        
        summary = self.experiment.get_statistical_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["experiment_id"], "test_exp")
        self.assertIn("total_tests", summary)
        self.assertIn("significant_tests", summary)
        self.assertIn("confidence_level", summary)


class TestRayServeABTestManager(unittest.TestCase):
    """Test cases for RayServeABTestManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()

    def test_initial_state(self):
        """Test initial state of manager."""
        self.assertEqual(len(self.manager.experiments), 0)

    def test_create_experiment(self):
        """Test creating an experiment."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant", model_path="models/variant.pth", weight=0.5)
        ]
        
        experiment = self.manager.create_experiment(
            experiment_id="test_exp",
            variants=variants
        )
        
        self.assertIsInstance(experiment, ABTestExperiment)
        self.assertEqual(len(self.manager.experiments), 1)
        self.assertIn("test_exp", self.manager.experiments)

    def test_create_duplicate_experiment(self):
        """Test creating a duplicate experiment raises an error."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        self.manager.create_experiment("test_exp", variants)
        
        with self.assertRaises(ValueError):
            self.manager.create_experiment("test_exp", variants)

    def test_get_experiment(self):
        """Test getting an experiment."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        self.manager.create_experiment("test_exp", variants)
        experiment = self.manager.get_experiment("test_exp")
        
        self.assertIsInstance(experiment, ABTestExperiment)
        self.assertEqual(experiment.experiment_id, "test_exp")

    def test_get_nonexistent_experiment(self):
        """Test getting a non-existent experiment returns None."""
        experiment = self.manager.get_experiment("nonexistent")
        self.assertIsNone(experiment)

    def test_get_variant_for_request(self):
        """Test getting variant for a request."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        self.manager.create_experiment("test_exp", variants)
        variant = self.manager.get_variant_for_request("test_exp", "test_request")
        
        # Should return the control variant
        self.assertEqual(variant, "control")

    def test_record_metrics(self):
        """Test recording metrics."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        self.manager.create_experiment("test_exp", variants)
        self.manager.record_metrics(
            experiment_id="test_exp",
            variant_name="control",
            latency_ms=50.0,
            processing_time_ms=45.0
        )
        
        # Check that metrics were recorded
        experiment = self.manager.get_experiment("test_exp")
        metrics = experiment.variant_metrics["control"]
        self.assertEqual(metrics.requests, 1)

    def test_get_experiment_results(self):
        """Test getting experiment results."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        self.manager.create_experiment("test_exp", variants)
        results = self.manager.get_experiment_results("test_exp")
        
        self.assertIsInstance(results, dict)
        self.assertEqual(results["experiment_id"], "test_exp")

    def test_list_experiments(self):
        """Test listing experiments."""
        # No experiments initially
        experiments = self.manager.list_experiments()
        self.assertEqual(len(experiments), 0)
        
        # Add an experiment
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        self.manager.create_experiment("test_exp", variants)
        
        # Check that we can list experiments
        experiments = self.manager.list_experiments()
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["experiment_id"], "test_exp")

    def test_stop_experiment(self):
        """Test stopping an experiment."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        self.manager.create_experiment("test_exp", variants)
        
        # Stop the experiment
        result = self.manager.stop_experiment("test_exp")
        self.assertTrue(result)
        
        # Check that experiment is stopped
        experiment = self.manager.get_experiment("test_exp")
        self.assertEqual(experiment.status, "stopped")

    def test_stop_nonexistent_experiment(self):
        """Test stopping a non-existent experiment."""
        result = self.manager.stop_experiment("nonexistent")
        self.assertFalse(result)

    @patch('src.ml.ray_serve.ab_testing.datetime')
    def test_cleanup_completed_experiments(self, mock_datetime):
        """Test cleaning up completed experiments."""
        # Create a mock current time
        current_time = datetime.now()
        mock_datetime.now.return_value = current_time
        
        # Create an experiment that ended in the past
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=1.0)
        ]
        
        experiment = self.manager.create_experiment(
            experiment_id="completed_exp",
            variants=variants,
            duration_hours=1  # 1 hour duration
        )
        
        # Set the end time to be in the past
        experiment.end_time = current_time - timedelta(hours=2)
        
        # Clean up completed experiments
        count = self.manager.cleanup_completed_experiments()
        self.assertEqual(count, 1)
        self.assertEqual(len(self.manager.experiments), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestVariantConfig))
    test_suite.addTest(unittest.makeSuite(TestVariantMetrics))
    test_suite.addTest(unittest.makeSuite(TestStatisticalTestResult))
    test_suite.addTest(unittest.makeSuite(TestABTestExperiment))
    test_suite.addTest(unittest.makeSuite(TestRayServeABTestManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_tests()
    print(f"Tests completed {'successfully' if success else 'with failures'}")
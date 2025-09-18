"""Comprehensive tests for A/B testing framework including statistical validity and safety mechanisms.

This test suite covers:
- Traffic splitting infrastructure
- Statistical significance testing
- Automated winner selection
- Gradual rollout capabilities
- Safety controls and monitoring
- API endpoint functionality
"""

import unittest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.ml.ray_serve.ab_testing import (
    VariantConfig, 
    VariantStatus, 
    VariantMetrics, 
    StatisticalTestResult, 
    ABTestExperiment, 
    RayServeABTestManager
)
from src.api.ab_testing_endpoints import router
from src.models.user import User, UserRole


class TestTrafficSplittingInfrastructure(unittest.TestCase):
    """Test traffic splitting and variant assignment."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()
        self.variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.3),
            VariantConfig(name="variant_b", model_path="models/variant_b.pth", weight=0.2)
        ]

    def test_consistent_traffic_splitting(self):
        """Test that traffic splitting is consistent for the same request ID."""
        experiment = self.manager.create_experiment(
            experiment_id="test_traffic_split",
            variants=self.variants
        )
        
        # Test consistency - same request ID should always get same variant
        request_id = "test_request_123"
        variant1 = experiment.get_variant_for_request(request_id)
        variant2 = experiment.get_variant_for_request(request_id)
        variant3 = experiment.get_variant_for_request(request_id)
        
        self.assertEqual(variant1, variant2)
        self.assertEqual(variant2, variant3)
        self.assertIn(variant1, ["control", "variant_a", "variant_b"])

    def test_traffic_distribution(self):
        """Test that traffic is distributed according to weights."""
        experiment = self.manager.create_experiment(
            experiment_id="test_distribution",
            variants=self.variants
        )
        
        # Generate many request IDs and check distribution
        num_requests = 10000
        variant_counts = {"control": 0, "variant_a": 0, "variant_b": 0}
        
        for i in range(num_requests):
            request_id = f"request_{i}"
            variant = experiment.get_variant_for_request(request_id)
            if variant:
                variant_counts[variant] += 1
        
        total_assigned = sum(variant_counts.values())
        
        # Check that distribution is approximately correct (within 5% tolerance)
        control_ratio = variant_counts["control"] / total_assigned
        variant_a_ratio = variant_counts["variant_a"] / total_assigned
        variant_b_ratio = variant_counts["variant_b"] / total_assigned
        
        self.assertAlmostEqual(control_ratio, 0.5, delta=0.05)
        self.assertAlmostEqual(variant_a_ratio, 0.3, delta=0.05)
        self.assertAlmostEqual(variant_b_ratio, 0.2, delta=0.05)

    def test_inactive_variant_exclusion(self):
        """Test that inactive variants are excluded from traffic splitting."""
        # Make variant_b inactive
        self.variants[2].status = VariantStatus.INACTIVE
        
        experiment = self.manager.create_experiment(
            experiment_id="test_inactive",
            variants=self.variants
        )
        
        # Generate requests and ensure variant_b is never assigned
        assigned_variants = set()
        for i in range(1000):
            request_id = f"request_{i}"
            variant = experiment.get_variant_for_request(request_id)
            if variant:
                assigned_variants.add(variant)
        
        self.assertNotIn("variant_b", assigned_variants)
        self.assertIn("control", assigned_variants)
        self.assertIn("variant_a", assigned_variants)


class TestStatisticalSignificanceTesting(unittest.TestCase):
    """Test statistical significance testing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()
        self.variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.5)
        ]
        self.experiment = self.manager.create_experiment(
            experiment_id="test_stats",
            variants=self.variants,
            confidence_level=0.95
        )

    def test_statistical_tests_with_sufficient_data(self):
        """Test statistical tests with sufficient sample size."""
        np.random.seed(42)  # For reproducible results
        
        # Add data for control (higher latency)
        for i in range(100):
            latency = np.random.normal(100, 15)  # Mean 100ms, std 15ms
            self.experiment.record_metrics(
                variant_name="control",
                latency_ms=latency,
                processing_time_ms=latency * 0.9,
                confidence_score=np.random.uniform(0.7, 0.9)
            )
        
        # Add data for variant_a (lower latency)
        for i in range(100):
            latency = np.random.normal(80, 12)  # Mean 80ms, std 12ms (better)
            self.experiment.record_metrics(
                variant_name="variant_a",
                latency_ms=latency,
                processing_time_ms=latency * 0.9,
                confidence_score=np.random.uniform(0.75, 0.95)
            )
        
        # Perform statistical tests
        tests = self.experiment.perform_statistical_tests()
        
        # Should have tests for latency and error rates
        self.assertGreater(len(tests), 0)
        
        # Check that we have latency comparison
        latency_tests = [t for t in tests if "latency" in t.test_name]
        self.assertGreater(len(latency_tests), 0)
        
        # Verify test structure
        for test in tests:
            self.assertIsInstance(test, StatisticalTestResult)
            self.assertIsInstance(test.test_name, str)
            self.assertIsInstance(test.statistic, (int, float))
            self.assertIsInstance(test.p_value, (int, float))
            self.assertIsInstance(test.significant, bool)
            self.assertEqual(test.confidence_level, 0.95)

    def test_statistical_tests_insufficient_data(self):
        """Test that statistical tests handle insufficient data gracefully."""
        # Add minimal data
        self.experiment.record_metrics("control", 50.0, 45.0, 0.8)
        self.experiment.record_metrics("variant_a", 60.0, 55.0, 0.85)
        
        # Should return empty list due to insufficient data
        tests = self.experiment.perform_statistical_tests()
        self.assertEqual(len(tests), 0)

    def test_effect_size_calculation(self):
        """Test that effect size is calculated correctly."""
        np.random.seed(123)
        
        # Add data with known effect size
        control_data = np.random.normal(100, 10, 50)
        variant_data = np.random.normal(90, 10, 50)  # 1 standard deviation difference
        
        for latency in control_data:
            self.experiment.record_metrics("control", latency, latency * 0.9, 0.8)
        
        for latency in variant_data:
            self.experiment.record_metrics("variant_a", latency, latency * 0.9, 0.8)
        
        tests = self.experiment.perform_statistical_tests()
        latency_tests = [t for t in tests if "latency" in t.test_name]
        
        if latency_tests:
            # Effect size should be approximately 1.0 (Cohen's d)
            effect_size = latency_tests[0].effect_size
            self.assertIsNotNone(effect_size)
            self.assertAlmostEqual(abs(effect_size), 1.0, delta=0.3)


class TestAutomatedWinnerSelection(unittest.TestCase):
    """Test automated winner selection based on risk-adjusted metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()
        self.variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.5)
        ]
        self.experiment = self.manager.create_experiment(
            experiment_id="test_winner",
            variants=self.variants
        )

    def test_winner_selection_with_clear_winner(self):
        """Test winner selection when one variant is clearly better."""
        # Control: higher error rate, higher latency
        for i in range(150):
            error = i < 15  # 10% error rate
            latency = np.random.normal(100, 10)
            confidence = np.random.uniform(0.6, 0.8)
            self.experiment.record_metrics("control", latency, latency * 0.9, confidence, error)
        
        # Variant A: lower error rate, lower latency, higher confidence
        for i in range(150):
            error = i < 3  # 2% error rate
            latency = np.random.normal(70, 8)
            confidence = np.random.uniform(0.8, 0.95)
            self.experiment.record_metrics("variant_a", latency, latency * 0.9, confidence, error)
        
        recommendation = self.manager.get_winner_recommendation("test_winner")
        
        self.assertEqual(recommendation["winner"], "variant_a")
        self.assertIn("variant_a", recommendation["variant_scores"])
        self.assertIn("control", recommendation["variant_scores"])
        
        # Variant A should have better (lower) composite score
        variant_a_score = recommendation["variant_scores"]["variant_a"]["composite_score"]
        control_score = recommendation["variant_scores"]["control"]["composite_score"]
        self.assertLess(variant_a_score, control_score)

    def test_winner_selection_insufficient_data(self):
        """Test winner selection with insufficient data."""
        # Add minimal data
        self.experiment.record_metrics("control", 50.0, 45.0, 0.8)
        self.experiment.record_metrics("variant_a", 60.0, 55.0, 0.85)
        
        recommendation = self.manager.get_winner_recommendation("test_winner")
        
        self.assertIsNone(recommendation["winner"])
        self.assertIn("Insufficient data", recommendation["reason"])

    def test_risk_adjusted_scoring(self):
        """Test that risk-adjusted scoring considers multiple factors."""
        # Variant with low latency but high error rate
        for i in range(150):
            error = i < 30  # 20% error rate (high)
            latency = 50  # Low latency
            confidence = 0.9  # High confidence
            self.experiment.record_metrics("control", latency, latency * 0.9, confidence, error)
        
        # Variant with higher latency but low error rate
        for i in range(150):
            error = i < 3  # 2% error rate (low)
            latency = 80  # Higher latency
            confidence = 0.85  # Good confidence
            self.experiment.record_metrics("variant_a", latency, latency * 0.9, confidence, error)
        
        recommendation = self.manager.get_winner_recommendation("test_winner")
        
        # Variant A should win due to much lower error rate despite higher latency
        self.assertEqual(recommendation["winner"], "variant_a")


class TestGradualRolloutCapabilities(unittest.TestCase):
    """Test gradual rollout functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()
        self.variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.7),
            VariantConfig(name="winner", model_path="models/winner.pth", weight=0.3)
        ]
        self.experiment = self.manager.create_experiment(
            experiment_id="test_rollout",
            variants=self.variants
        )

    def test_gradual_rollout_plan_creation(self):
        """Test creation of gradual rollout plan."""
        rollout_plan = self.manager.implement_gradual_rollout(
            experiment_id="test_rollout",
            winning_variant="winner",
            rollout_steps=[0.2, 0.5, 0.8, 1.0]
        )
        
        self.assertEqual(rollout_plan["winning_variant"], "winner")
        self.assertEqual(len(rollout_plan["rollout_steps"]), 4)
        self.assertEqual(rollout_plan["current_step"], 0)
        self.assertEqual(rollout_plan["status"], "planned")
        
        # Check step configurations
        for i, step in enumerate(rollout_plan["rollout_steps"]):
            expected_percentage = [0.2, 0.5, 0.8, 1.0][i]
            self.assertEqual(step["target_percentage"], expected_percentage)
            self.assertEqual(step["winner_weight"], expected_percentage)
            self.assertFalse(step["completed"])

    def test_rollout_step_execution(self):
        """Test execution of rollout steps."""
        rollout_plan = self.manager.implement_gradual_rollout(
            experiment_id="test_rollout",
            winning_variant="winner"
        )
        
        # Execute first step
        result = self.manager.execute_rollout_step(rollout_plan)
        
        self.assertEqual(result["step_executed"], 1)
        self.assertIn("winner", result["new_weights"])
        self.assertIn("control", result["new_weights"])
        
        # Check that weights were updated
        winner_weight = self.experiment.variants["winner"].weight
        control_weight = self.experiment.variants["control"].weight
        
        self.assertAlmostEqual(winner_weight + control_weight, 1.0, places=3)
        self.assertEqual(rollout_plan["current_step"], 1)

    def test_rollout_safety_check_integration(self):
        """Test that rollout respects safety checks."""
        # Add data that would trigger safety violations
        for i in range(150):
            error = i < 75  # 50% error rate (very high)
            self.experiment.record_metrics("winner", 100, 90, 0.5, error)
        
        rollout_plan = self.manager.implement_gradual_rollout(
            experiment_id="test_rollout",
            winning_variant="winner"
        )
        
        # Execution should fail due to safety violations
        result = self.manager.execute_rollout_step(rollout_plan)
        
        self.assertIn("error", result)
        self.assertIn("safety violations", result["error"])

    def test_default_rollout_steps(self):
        """Test default rollout steps when none provided."""
        rollout_plan = self.manager.implement_gradual_rollout(
            experiment_id="test_rollout",
            winning_variant="winner"
        )
        
        # Should use default steps: [0.1, 0.25, 0.5, 0.75, 1.0]
        expected_steps = [0.1, 0.25, 0.5, 0.75, 1.0]
        actual_steps = [step["target_percentage"] for step in rollout_plan["rollout_steps"]]
        
        self.assertEqual(actual_steps, expected_steps)


class TestSafetyControlsAndMonitoring(unittest.TestCase):
    """Test safety controls and monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()
        self.variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.5)
        ]
        self.experiment = self.manager.create_experiment(
            experiment_id="test_safety",
            variants=self.variants
        )

    def test_error_rate_safety_check(self):
        """Test safety check for high error rates."""
        # Add data with high error rate for variant_a
        for i in range(100):
            error = i < 20  # 20% error rate (above 10% threshold)
            self.experiment.record_metrics("variant_a", 50, 45, 0.8, error)
        
        # Add normal data for control
        for i in range(100):
            error = i < 5  # 5% error rate (normal)
            self.experiment.record_metrics("control", 55, 50, 0.8, error)
        
        safety_results = self.manager.check_safety_controls("test_safety")
        
        self.assertTrue(safety_results["should_stop"])
        self.assertGreater(len(safety_results["safety_violations"]), 0)
        
        # Check that high error rate violation is detected
        error_violations = [v for v in safety_results["safety_violations"] 
                          if v["violation"] == "high_error_rate"]
        self.assertGreater(len(error_violations), 0)

    def test_confidence_score_safety_check(self):
        """Test safety check for low confidence scores."""
        # Add data with low confidence for variant_a
        for i in range(100):
            confidence = 0.5  # Below 0.7 threshold
            self.experiment.record_metrics("variant_a", 50, 45, confidence)
        
        safety_results = self.manager.check_safety_controls("test_safety")
        
        confidence_violations = [v for v in safety_results["safety_violations"] 
                               if v["violation"] == "low_confidence"]
        self.assertGreater(len(confidence_violations), 0)

    def test_latency_safety_check(self):
        """Test safety check for high latency."""
        # Add data with high latency for variant_a
        for i in range(100):
            latency = 1500  # Above 1000ms threshold
            self.experiment.record_metrics("variant_a", latency, latency * 0.9, 0.8)
        
        safety_results = self.manager.check_safety_controls("test_safety")
        
        latency_violations = [v for v in safety_results["safety_violations"] 
                            if v["violation"] == "high_latency"]
        self.assertGreater(len(latency_violations), 0)

    def test_safety_recommendations(self):
        """Test that safety recommendations are generated."""
        # Add problematic data
        for i in range(100):
            error = i < 15  # 15% error rate
            latency = 1200  # High latency
            confidence = 0.6  # Low confidence
            self.experiment.record_metrics("variant_a", latency, latency * 0.9, confidence, error)
        
        safety_results = self.manager.check_safety_controls("test_safety")
        
        self.assertGreater(len(safety_results["recommendations"]), 0)
        self.assertIsInstance(safety_results["recommendations"][0], str)

    def test_safety_controls_with_insufficient_data(self):
        """Test safety controls with insufficient data."""
        # Add minimal data (less than 10 requests)
        self.experiment.record_metrics("variant_a", 50, 45, 0.8)
        
        safety_results = self.manager.check_safety_controls("test_safety")
        
        # Should not trigger violations due to insufficient data
        self.assertEqual(len(safety_results["safety_violations"]), 0)
        self.assertFalse(safety_results["should_stop"])


class TestAPIEndpointFunctionality(unittest.TestCase):
    """Test A/B testing API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test FastAPI app
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        
        # Mock authentication
        self.mock_user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role=UserRole.ADMIN,
            is_active=True
        )

    @patch('src.api.ab_testing_endpoints.get_current_user')
    @patch('src.api.ab_testing_endpoints.require_role')
    def test_create_experiment_endpoint(self, mock_require_role, mock_get_user):
        """Test experiment creation endpoint."""
        mock_require_role.return_value = self.mock_user
        mock_get_user.return_value = self.mock_user
        
        experiment_data = {
            "experiment_name": "Test Experiment",
            "description": "Test description",
            "variants": [
                {"name": "control", "model_path": "models/control.pth", "weight": 0.6},
                {"name": "variant_a", "model_path": "models/variant_a.pth", "weight": 0.4}
            ],
            "duration_hours": 48,
            "confidence_level": 0.95
        }
        
        response = self.client.post("/api/v1/ab-testing/experiments", json=experiment_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("experiment_id", data)
        self.assertEqual(data["experiment_name"], "Test Experiment")
        self.assertEqual(len(data["variants"]), 2)

    @patch('src.api.ab_testing_endpoints.get_current_user')
    def test_list_experiments_endpoint(self, mock_get_user):
        """Test experiment listing endpoint."""
        mock_get_user.return_value = self.mock_user
        
        response = self.client.get("/api/v1/ab-testing/experiments")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_invalid_experiment_creation(self):
        """Test validation for invalid experiment creation."""
        # Test with invalid weights (don't sum to 1.0)
        invalid_data = {
            "experiment_name": "Invalid Experiment",
            "variants": [
                {"name": "control", "model_path": "models/control.pth", "weight": 0.6},
                {"name": "variant_a", "model_path": "models/variant_a.pth", "weight": 0.6}
            ]
        }
        
        response = self.client.post("/api/v1/ab-testing/experiments", json=invalid_data)
        
        self.assertEqual(response.status_code, 422)  # Validation error

    @patch('src.api.ab_testing_endpoints.get_current_user')
    def test_experiment_results_endpoint(self, mock_get_user):
        """Test experiment results endpoint."""
        mock_get_user.return_value = self.mock_user
        
        # This would normally require an existing experiment
        response = self.client.get("/api/v1/ab-testing/experiments/nonexistent/results")
        
        self.assertEqual(response.status_code, 404)

    @patch('src.api.ab_testing_endpoints.get_current_user')
    def test_dashboard_endpoint(self, mock_get_user):
        """Test experiments dashboard endpoint."""
        mock_get_user.return_value = self.mock_user
        
        response = self.client.get("/api/v1/ab-testing/experiments/dashboard")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_experiments", data)
        self.assertIn("active_experiments", data)
        self.assertIn("recent_experiments", data)


class TestIntegrationScenarios(unittest.TestCase):
    """Test end-to-end integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RayServeABTestManager()

    def test_complete_ab_test_lifecycle(self):
        """Test complete A/B test lifecycle from creation to winner selection."""
        # 1. Create experiment
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="variant_a", model_path="models/variant_a.pth", weight=0.5)
        ]
        experiment = self.manager.create_experiment(
            experiment_id="lifecycle_test",
            variants=variants,
            duration_hours=24
        )
        
        # 2. Simulate traffic and collect metrics
        np.random.seed(42)
        for i in range(200):
            request_id = f"request_{i}"
            variant = experiment.get_variant_for_request(request_id)
            
            if variant == "control":
                # Control: slightly worse performance
                latency = np.random.normal(90, 12)
                error = np.random.random() < 0.08  # 8% error rate
                confidence = np.random.uniform(0.7, 0.85)
            else:
                # Variant A: better performance
                latency = np.random.normal(75, 10)
                error = np.random.random() < 0.04  # 4% error rate
                confidence = np.random.uniform(0.8, 0.95)
            
            experiment.record_metrics(variant, latency, latency * 0.9, confidence, error)
        
        # 3. Check safety controls
        safety_results = self.manager.check_safety_controls("lifecycle_test")
        self.assertFalse(safety_results["should_stop"])
        
        # 4. Perform statistical analysis
        statistical_summary = self.manager.get_statistical_summary("lifecycle_test")
        self.assertGreater(statistical_summary["total_tests"], 0)
        
        # 5. Get winner recommendation
        winner_rec = self.manager.get_winner_recommendation("lifecycle_test")
        self.assertEqual(winner_rec["winner"], "variant_a")
        
        # 6. Create gradual rollout plan
        rollout_plan = self.manager.implement_gradual_rollout(
            experiment_id="lifecycle_test",
            winning_variant="variant_a"
        )
        self.assertEqual(rollout_plan["winning_variant"], "variant_a")
        
        # 7. Execute first rollout step
        result = self.manager.execute_rollout_step(rollout_plan)
        self.assertEqual(result["step_executed"], 1)

    def test_safety_triggered_experiment_stop(self):
        """Test scenario where safety controls trigger experiment stop."""
        variants = [
            VariantConfig(name="control", model_path="models/control.pth", weight=0.5),
            VariantConfig(name="bad_variant", model_path="models/bad.pth", weight=0.5)
        ]
        experiment = self.manager.create_experiment(
            experiment_id="safety_test",
            variants=variants
        )
        
        # Simulate bad performance for bad_variant
        for i in range(100):
            # Control: normal performance
            experiment.record_metrics("control", 80, 75, 0.85, False)
            
            # Bad variant: terrible performance
            error = i < 50  # 50% error rate
            experiment.record_metrics("bad_variant", 2000, 1900, 0.3, error)
        
        # Safety check should recommend stopping
        safety_results = self.manager.check_safety_controls("safety_test")
        self.assertTrue(safety_results["should_stop"])
        
        # Stop experiment
        success = self.manager.stop_experiment("safety_test")
        self.assertTrue(success)
        self.assertEqual(experiment.status, "stopped")


def run_comprehensive_tests():
    """Run all comprehensive A/B testing tests."""
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTrafficSplittingInfrastructure,
        TestStatisticalSignificanceTesting,
        TestAutomatedWinnerSelection,
        TestGradualRolloutCapabilities,
        TestSafetyControlsAndMonitoring,
        TestAPIEndpointFunctionality,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        test_suite.addTest(unittest.makeSuite(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    print(f"\nComprehensive A/B testing tests completed {'successfully' if success else 'with failures'}")
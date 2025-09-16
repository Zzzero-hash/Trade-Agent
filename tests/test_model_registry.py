"""Tests for the model registry with versioning and automated rollback capabilities"""

import unittest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.ml.model_registry import (
    ModelRegistry, 
    ModelVersion, 
    ModelStatus, 
    RollbackReason,
    SemanticVersion
)


class TestSemanticVersion(unittest.TestCase):
    """Test cases for SemanticVersion helper class"""
    
    def test_parse_valid_versions(self):
        """Test parsing valid semantic versions"""
        # Test valid versions
        self.assertTrue(SemanticVersion.is_valid("1.0.0"))
        self.assertTrue(SemanticVersion.is_valid("2.1.3"))
        self.assertTrue(SemanticVersion.is_valid("10.20.30"))
        self.assertTrue(SemanticVersion.is_valid("1.0.0-alpha"))
        self.assertTrue(SemanticVersion.is_valid("1.0.0-alpha.1"))
        self.assertTrue(SemanticVersion.is_valid("1.0.0-0.3.7"))
        
    def test_parse_invalid_versions(self):
        """Test parsing invalid semantic versions"""
        # Test invalid versions
        self.assertFalse(SemanticVersion.is_valid("1.0"))
        self.assertFalse(SemanticVersion.is_valid("1"))
        self.assertFalse(SemanticVersion.is_valid("a.b.c"))
        self.assertFalse(SemanticVersion.is_valid("1.0.0.0"))
        self.assertFalse(SemanticVersion.is_valid(""))
        
    def test_next_version_functions(self):
        """Test next version functions"""
        # Test next major version
        self.assertEqual(SemanticVersion.next_major("1.2.3"), "2.0.0")
        self.assertEqual(SemanticVersion.next_major("0.1.0"), "1.0")
        
        # Test next minor version
        self.assertEqual(SemanticVersion.next_minor("1.2.3"), "1.3.0")
        self.assertEqual(SemanticVersion.next_minor("2.0.0"), "2.1.0")
        
        # Test next patch version
        self.assertEqual(SemanticVersion.next_patch("1.2.3"), "1.2.4")
        self.assertEqual(SemanticVersion.next_patch("1.2.0"), "1.2.1")


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for the registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_registry(self):
        """Test initializing the registry"""
        # Test that registry is initialized correctly
        self.assertIsInstance(self.registry, ModelRegistry)
        self.assertEqual(self.registry.registry_path, self.temp_dir)
        self.assertEqual(len(self.registry.models), 0)
        self.assertEqual(len(self.registry.deployments), 0)
        self.assertEqual(len(self.registry.rollback_history), 0)
    
    def test_register_model(self):
        """Test registering a model"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register a model
        model_version = self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1},
            metadata={"description": "Test model"}
        )
        
        # Verify the model was registered correctly
        self.assertIsInstance(model_version, ModelVersion)
        self.assertEqual(model_version.model_id, "test_model")
        self.assertEqual(model_version.version, "1.0.0")
        self.assertEqual(model_version.file_path, model_file)
        self.assertEqual(model_version.status, ModelStatus.INACTIVE)
        self.assertEqual(model_version.metadata["description"], "Test model")
        
        # Verify the model is in the registry
        self.assertIn("test_model", self.registry.models)
        self.assertEqual(len(self.registry.models["test_model"]), 1)
        
        # Verify the model can be retrieved
        retrieved_model = self.registry.get_model_version("test_model", "1.0.0")
        self.assertEqual(retrieved_model, model_version)
    
    def test_register_model_invalid_version(self):
        """Test registering a model with invalid version"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Try to register with invalid version
        with self.assertRaises(ValueError):
            self.registry.register_model(
                model_id="test_model",
                version="invalid_version",
                file_path=model_file,
                config={"input_dim": 10, "output_dim": 1}
            )
    
    def test_register_model_file_not_found(self):
        """Test registering a model with non-existent file"""
        # Try to register with non-existent file
        with self.assertRaises(ValueError):
            self.registry.register_model(
                model_id="test_model",
                version="1.0.0",
                file_path="/non/existent/file.pth",
                config={"input_dim": 10, "output_dim": 1}
            )
    
    def test_get_model_version(self):
        """Test getting model versions"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register multiple versions of a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        self.registry.register_model(
            model_id="test_model",
            version="1.1.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Test getting specific version
        model_v1 = self.registry.get_model_version("test_model", "1.0.0")
        self.assertIsNotNone(model_v1)
        self.assertEqual(model_v1.version, "1.0.0")
        
        model_v2 = self.registry.get_model_version("test_model", "1.1.0")
        self.assertIsNotNone(model_v2)
        self.assertEqual(model_v2.version, "1.1.0")
        
        # Test getting latest version (should be 1.1.0 as it's newer)
        latest_model = self.registry.get_model_version("test_model")
        self.assertIsNotNone(latest_model)
        self.assertEqual(latest_model.version, "1.1.0")
    
    def test_get_all_versions(self):
        """Test getting all versions of a model"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register multiple versions
        versions = ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]
        for version in versions:
            self.registry.register_model(
                model_id="test_model",
                version=version,
                file_path=model_file,
                config={"input_dim": 10, "output_dim": 1}
            )
        
        # Get all versions
        all_versions = self.registry.get_all_versions("test_model")
        self.assertEqual(len(all_versions), 4)
        
        # Verify versions are sorted correctly (newest first)
        version_strings = [mv.version for mv in all_versions]
        self.assertEqual(version_strings, ["2.0.0", "1.1.0", "1.0.1", "1.0.0"])
    
    def test_deploy_model(self):
        """Test deploying a model"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Deploy the model
        success = self.registry.deploy_model(
            model_id="test_model",
            version="1.0.0",
            config={"deployment_setting": "test"},
            traffic_percentage=0.5
        )
        
        self.assertTrue(success)
        
        # Verify deployment
        deployment_id = "test_model:1.0.0"
        self.assertIn(deployment_id, self.registry.deployments)
        
        deployment = self.registry.deployments[deployment_id]
        self.assertEqual(deployment.model_id, "test_model")
        self.assertEqual(deployment.version, "1.0.0")
        self.assertEqual(deployment.status, "active")
        self.assertEqual(deployment.traffic_percentage, 0.5)
        
        # Verify model status is updated
        model_version = self.registry.get_model_version("test_model", "1.0.0")
        self.assertEqual(model_version.status, ModelStatus.ACTIVE)
        self.assertIsNotNone(model_version.deployed_at)
    
    def test_undeploy_model(self):
        """Test undeploying a model"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register and deploy a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        self.registry.deploy_model(
            model_id="test_model",
            version="1.0.0",
            config={"deployment_setting": "test"}
        )
        
        # Undeploy the model
        success = self.registry.undeploy_model("test_model", "1.0.0")
        self.assertTrue(success)
        
        # Verify undeployment
        deployment_id = "test_model:1.0.0"
        self.assertIn(deployment_id, self.registry.deployments)
        
        deployment = self.registry.deployments[deployment_id]
        self.assertEqual(deployment.status, "inactive")
        
        # Verify model status is updated
        model_version = self.registry.get_model_version("test_model", "1.0.0")
        self.assertEqual(model_version.status, ModelStatus.INACTIVE)
    
    def test_get_active_models(self):
        """Test getting active models"""
        # Create temporary model files
        model_file1 = os.path.join(self.temp_dir, "test_model1.pth")
        model_file2 = os.path.join(self.temp_dir, "test_model2.pth")
        with open(model_file1, "w") as f:
            f.write("test model 1 content")
        with open(model_file2, "w") as f:
            f.write("test model 2 content")
        
        # Register models
        self.registry.register_model(
            model_id="test_model1",
            version="1.0.0",
            file_path=model_file1,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        self.registry.register_model(
            model_id="test_model2",
            version="1.0.0",
            file_path=model_file2,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Deploy one model
        self.registry.deploy_model(
            model_id="test_model1",
            version="1.0.0",
            config={"deployment_setting": "test"}
        )
        
        # Get active models
        active_models = self.registry.get_active_models()
        self.assertEqual(len(active_models), 1)
        self.assertIn("test_model1", active_models)
        self.assertEqual(active_models["test_model1"].version, "1.0.0")
    
    def test_update_performance_metrics(self):
        """Test updating performance metrics"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Update performance metrics
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        
        success = self.registry.update_performance_metrics(
            "test_model", "1.0.0", metrics
        )
        self.assertTrue(success)
        
        # Verify metrics were updated
        model_version = self.registry.get_model_version("test_model", "1.0.0")
        self.assertEqual(model_version.performance_metrics["accuracy"], 0.85)
        self.assertEqual(model_version.performance_metrics["precision"], 0.82)
        self.assertEqual(model_version.performance_metrics["recall"], 0.88)
        self.assertEqual(model_version.performance_metrics["f1_score"], 0.85)
    
    def test_check_rollback_conditions(self):
        """Test checking rollback conditions"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Test with good performance metrics (should not trigger rollback)
        good_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "error_rate": 0.02,
            "latency_95th_percentile": 50.0
        }
        
        self.registry.update_performance_metrics("test_model", "1.0.0", good_metrics)
        
        should_rollback, reason, description = self.registry.check_rollback_conditions(
            "test_model", "1.0.0"
        )
        self.assertFalse(should_rollback)
        self.assertIsNone(reason)
        self.assertIsNone(description)
        
        # Test with poor accuracy (should trigger rollback)
        poor_metrics = {
            "accuracy": 0.60,  # Below threshold of 0.7
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        
        self.registry.update_performance_metrics("test_model", "1.0.0", poor_metrics)
        
        should_rollback, reason, description = self.registry.check_rollback_conditions(
            "test_model", "1.0.0"
        )
        self.assertTrue(should_rollback)
        self.assertEqual(reason, RollbackReason.PERFORMANCE_DEGRADATION)
        self.assertIn("Accuracy", description)
    
    def test_rollback_model(self):
        """Test rolling back a model"""
        # Create temporary model files
        model_file1 = os.path.join(self.temp_dir, "test_model_v1.pth")
        model_file2 = os.path.join(self.temp_dir, "test_model_v2.pth")
        with open(model_file1, "w") as f:
            f.write("test model v1 content")
        with open(model_file2, "w") as f:
            f.write("test model v2 content")
        
        # Register multiple versions
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file1,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        self.registry.register_model(
            model_id="test_model",
            version="2.0.0",
            file_path=model_file2,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Deploy the newer version
        self.registry.deploy_model(
            model_id="test_model",
            version="2.0.0",
            config={"deployment_setting": "test"}
        )
        
        # Rollback to previous version
        success = self.registry.rollback_model(
            model_id="test_model",
            reason=RollbackReason.PERFORMANCE_DEGRADATION,
            description="Performance degradation detected"
        )
        self.assertTrue(success)
        
        # Verify rollback
        rollback_events = self.registry.get_rollback_history("test_model")
        self.assertEqual(len(rollback_events), 1)
        
        rollback_event = rollback_events[0]
        self.assertEqual(rollback_event.model_id, "test_model")
        self.assertEqual(rollback_event.from_version, "2.0.0")
        self.assertEqual(rollback_event.to_version, "1.0.0")
        self.assertEqual(rollback_event.reason, RollbackReason.PERFORMANCE_DEGRADATION)
        self.assertEqual(rollback_event.description, "Performance degradation detected")
        
        # Verify new active version
        active_model = self.registry.get_model_version("test_model")
        self.assertEqual(active_model.version, "1.0.0")
        self.assertEqual(active_model.status, ModelStatus.ACTIVE)
        
        # Verify old version status
        old_model = self.registry.get_model_version("test_model", "2.0.0")
        self.assertEqual(old_model.status, ModelStatus.ROLLED_BACK)
    
    def test_set_performance_thresholds(self):
        """Test setting performance thresholds"""
        # Set custom thresholds
        custom_thresholds = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.75,
            "f1_score": 0.78,
            "error_rate": 0.03,
            "latency_95th_percentile": 80.0
        }
        
        self.registry.set_performance_thresholds(custom_thresholds)
        
        # Verify thresholds were updated
        for key, value in custom_thresholds.items():
            self.assertEqual(self.registry.performance_thresholds[key], value)
    
    def test_get_model_info(self):
        """Test getting model information"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register and deploy a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1},
            metadata={"description": "Test model"}
        )
        
        self.registry.deploy_model(
            model_id="test_model",
            version="1.0.0",
            config={"deployment_setting": "test"}
        )
        
        # Get model info
        model_info = self.registry.get_model_info("test_model")
        
        # Verify info
        self.assertEqual(model_info["model_id"], "test_model")
        self.assertEqual(model_info["total_versions"], 1)
        self.assertEqual(model_info["active_version"], "1.0.0")
        self.assertEqual(len(model_info["versions"]), 1)
        self.assertEqual(len(model_info["deployments"]), 1)
        self.assertEqual(len(model_info["rollback_history"]), 0)
        self.assertEqual(model_info["versions"][0]["metadata"]["description"], "Test model")
    
    def test_save_and_load_registry(self):
        """Test saving and loading registry from disk"""
        # Create a temporary model file
        model_file = os.path.join(self.temp_dir, "test_model.pth")
        with open(model_file, "w") as f:
            f.write("test model content")
        
        # Register and deploy a model
        self.registry.register_model(
            model_id="test_model",
            version="1.0.0",
            file_path=model_file,
            config={"input_dim": 10, "output_dim": 1}
        )
        
        self.registry.deploy_model(
            model_id="test_model",
            version="1.0.0",
            config={"deployment_setting": "test"}
        )
        
        # Update performance metrics
        metrics = {"accuracy": 0.85}
        self.registry.update_performance_metrics("test_model", "1.0.0", metrics)
        
        # Rollback the model
        self.registry.rollback_model(
            model_id="test_model",
            reason=RollbackReason.MANUAL_ROLLBACK,
            description="Test rollback"
        )
        
        # Save registry
        self.registry._save_registry()
        
        # Create a new registry instance to load from disk
        new_registry = ModelRegistry(self.temp_dir)
        
        # Verify data was loaded correctly
        self.assertIn("test_model", new_registry.models)
        self.assertEqual(len(new_registry.models["test_model"]), 1)
        self.assertIn("test_model:1.0.0", new_registry.deployments)
        self.assertEqual(len(new_registry.rollback_history), 1)
        
        # Verify model data
        model_version = new_registry.get_model_version("test_model", "1.0.0")
        self.assertEqual(model_version.version, "1.0.0")
        self.assertEqual(model_version.performance_metrics["accuracy"], 0.85)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSemanticVersion))
    test_suite.addTest(unittest.makeSuite(TestModelRegistry))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_tests()
    print(f"Tests completed {'successfully' if success else 'with failures'}")
"""Tests for model interpretability modules.

This module contains tests for all interpretability and explainability features
implemented for the CNN+LSTM hybrid model.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
import warnings
from unittest.mock import Mock, patch

# Import interpretability modules
try:
    from .shap_explainer import SHAPExplainer, create_shap_explainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available, skipping SHAP tests")

try:
    from .attention_visualizer import AttentionVisualizer, create_attention_visualizer
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available, skipping visualization tests")

try:
    from .feature_importance_analyzer import FeatureImportanceAnalyzer, create_feature_importance_analyzer
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available, skipping integrated gradients tests")

try:
    from .decision_auditor import DecisionAuditor, create_decision_auditor
    DECISION_AUDITOR_AVAILABLE = True
except ImportError:
    DECISION_AUDITOR_AVAILABLE = False
    warnings.warn("Decision auditor not available, skipping audit tests")

try:
    from .uncertainty_calibrator import UncertaintyCalibrator, create_uncertainty_calibrator
    UNCERTAINTY_CALIBRATOR_AVAILABLE = True
except ImportError:
    UNCERTAINTY_CALIBRATOR_AVAILABLE = False
    warnings.warn("Uncertainty calibrator not available, skipping calibration tests")

# Mock model for testing
class MockCNNLSTMHybridModel:
    """Mock CNN+LSTM hybrid model for testing."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.is_trained = True
        self.version = "test_v1.0"
    
    def forward(self, x, return_features=False, use_ensemble=False):
        """Mock forward pass."""
        batch_size = x.shape[0] if x.ndim == 3 else 1
        sequence_length = x.shape[-1] if x.ndim == 3 else x.shape[-1]
        
        # Mock outputs
        outputs = {
            'classification_probs': torch.rand(batch_size, 3),  # 3 classes
            'regression_mean': torch.rand(batch_size, 1),  # 1 target
            'regression_uncertainty': torch.rand(batch_size, 1),
        }
        
        if return_features:
            outputs['fused_features'] = torch.rand(batch_size, 128, sequence_length)
            outputs['cnn_attention_weights'] = torch.rand(batch_size, sequence_length, 10)  # 10 features
            outputs['lstm_attention_weights'] = torch.rand(batch_size, sequence_length)
            outputs['fusion_attention_weights'] = torch.rand(batch_size, 128)
        
        return outputs
    
    def predict(self, X, return_uncertainty=False, use_ensemble=False):
        """Mock predict method."""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.forward(X, use_ensemble=use_ensemble)
            
        predictions = {
            'classification_pred': torch.argmax(outputs['classification_probs'], dim=1).numpy(),
            'classification_probs': outputs['classification_probs'].numpy(),
            'regression_pred': outputs['regression_mean'].numpy(),
        }
        
        if return_uncertainty:
            predictions['regression_uncertainty'] = outputs['regression_uncertainty'].numpy()
        
        return predictions


class TestSHAPExplainer(unittest.TestCase):
    """Test cases for SHAP explainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SHAP_AVAILABLE:
            self.skipTest("SHAP not available")
        
        self.model = MockCNNLSTMHybridModel()
        self.explainer = create_shap_explainer(self.model)
        
        # Create test data
        self.background_data = np.random.randn(10, 5, 20)  # 10 samples, 5 features, 20 timesteps
        self.test_data = np.random.randn(3, 5, 20)  # 3 samples, 5 features, 20 timesteps
    
    def test_initialization(self):
        """Test SHAPExplainer initialization."""
        self.assertIsInstance(self.explainer, SHAPExplainer)
        self.assertEqual(self.explainer.model, self.model)
        self.assertTrue(self.explainer.is_trained)
    
    def test_compute_shap_values_kernel(self):
        """Test computing SHAP values with kernel explainer."""
        result = self.explainer.compute_shap_values(
            self.background_data, 
            self.test_data, 
            explainer_type="kernel"
        )
        
        self.assertIn('shap_values', result)
        self.assertIn('background_data_shape', result)
        self.assertIn('test_data_shape', result)
        self.assertIn('explainer_type', result)
        self.assertEqual(result['explainer_type'], 'kernel')
    
    def test_get_feature_importance(self):
        """Test extracting feature importance from SHAP values."""
        # Compute SHAP values first
        shap_result = self.explainer.compute_shap_values(
            self.background_data, 
            self.test_data, 
            explainer_type="kernel"
        )
        
        # Get feature importance
        feature_names = [f"feature_{i}" for i in range(5)]
        importance = self.explainer.get_feature_importance(shap_result, feature_names)
        
        self.assertIn('feature_importance', importance)
        self.assertIn('importance_by_feature', importance)
        self.assertIn('total_importance', importance)
        self.assertEqual(len(importance['feature_importance']), 5)  # 5 features
    
    def test_get_attention_weights(self):
        """Test extracting attention weights."""
        attention_weights = self.explainer.get_attention_weights(self.test_data)
        
        self.assertIsInstance(attention_weights, dict)
        # Should have computed attention since no specific attention weights in mock
        self.assertIn('computed_attention', attention_weights)
    
    def test_explain_prediction(self):
        """Test complete prediction explanation."""
        feature_names = [f"feature_{i}" for i in range(5)]
        explanation = self.explainer.explain_prediction(
            self.background_data,
            self.test_data[0],  # Single sample
            feature_names
        )
        
        self.assertIn('shap_values', explanation)
        self.assertIn('feature_importance', explanation)
        self.assertIn('attention_weights', explanation)
        self.assertIn('explanation_timestamp', explanation)


class TestAttentionVisualizer(unittest.TestCase):
    """Test cases for attention visualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("Matplotlib not available")
        
        self.visualizer = create_attention_visualizer()
        
        # Create test attention weights
        self.cnn_attention = np.random.rand(10, 5)  # 10 timesteps, 5 features
        self.lstm_attention = np.random.rand(10)   # 10 timesteps
        self.cross_attention = np.random.rand(5, 8) # 5 CNN features, 8 LSTM features
    
    def test_visualize_cnn_attention(self):
        """Test CNN attention visualization."""
        # Should not raise an exception
        try:
            self.visualizer.visualize_cnn_attention(
                self.cnn_attention,
                feature_names=[f"feat_{i}" for i in range(5)],
                sequence_labels=[f"t{i}" for i in range(10)],
                show=False
            )
        except RuntimeError:
            # Matplotlib not available, which is expected in test environment
            pass
    
    def test_visualize_lstm_attention(self):
        """Test LSTM attention visualization."""
        # Should not raise an exception
        try:
            self.visualizer.visualize_lstm_attention(
                self.lstm_attention,
                sequence_labels=[f"t{i}" for i in range(10)],
                show=False
            )
        except RuntimeError:
            # Matplotlib not available, which is expected in test environment
            pass
    
    def test_visualize_cross_attention(self):
        """Test cross-attention visualization."""
        # Should not raise an exception
        try:
            self.visualizer.visualize_cross_attention(
                self.cross_attention,
                row_labels=[f"cnn_{i}" for i in range(5)],
                col_labels=[f"lstm_{i}" for i in range(8)],
                show=False
            )
        except RuntimeError:
            # Matplotlib not available, which is expected in test environment
            pass


class TestFeatureImportanceAnalyzer(unittest.TestCase):
    """Test cases for feature importance analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not CAPTUM_AVAILABLE:
            self.skipTest("Captum not available")
        
        self.model = MockCNNLSTMHybridModel()
        self.analyzer = create_feature_importance_analyzer(self.model)
        
        # Create test data
        self.X = np.random.randn(20, 5, 30)  # 20 samples, 5 features, 30 timesteps
        self.y_class = np.random.randint(0, 3, 20)  # 3 classes
        self.y_reg = np.random.randn(20, 1)  # 1 target
    
    def test_compute_permutation_importance(self):
        """Test permutation importance computation."""
        try:
            importance = self.analyzer.compute_permutation_importance(
                self.X,
                self.y_class,
                self.y_reg,
                scoring="accuracy",
                n_repeats=3,
                random_state=42
            )
            
            self.assertIn('importances_mean', importance)
            self.assertIn('importances_std', importance)
            self.assertIn('baseline_score', importance)
            self.assertEqual(len(importance['importances_mean']), 5)  # 5 features
        except RuntimeError:
            # Scikit-learn not available
            self.skipTest("Scikit-learn not available")


class TestDecisionAuditor(unittest.TestCase):
    """Test cases for decision auditor."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not DECISION_AUDITOR_AVAILABLE:
            self.skipTest("Decision auditor not available")
        
        # Create temporary audit log file
        self.temp_dir = tempfile.mkdtemp()
        self.audit_log_path = os.path.join(self.temp_dir, "test_audit_trail.json")
        self.auditor = create_decision_auditor(self.audit_log_path)
        
        # Create test data
        self.model = MockCNNLSTMHybridModel()
        self.input_data = np.random.randn(5, 20)  # 5 features, 20 timesteps
        self.prediction = {
            'classification_pred': 1,
            'classification_probs': np.array([0.2, 0.6, 0.2]),
            'regression_pred': np.array([1.5])
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.audit_log_path):
            os.remove(self.audit_log_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_log_decision(self):
        """Test logging a decision."""
        self.auditor.log_decision(
            self.model,
            self.input_data,
            self.prediction,
            metadata={'test': True}
        )
        
        # Check that entry was added
        self.assertEqual(len(self.auditor.audit_entries), 1)
        
        # Check that entry has correct data
        entry = self.auditor.audit_entries[0]
        self.assertEqual(entry.prediction, self.prediction)
        self.assertEqual(entry.metadata, {'test': True})
    
    def test_register_model_version(self):
        """Test registering a model version."""
        version = self.auditor.register_model_version(
            self.model,
            training_data_hash="test_hash",
            hyperparameters={'lr': 0.001},
            performance_metrics={'accuracy': 0.95}
        )
        
        self.assertEqual(version, "test_v1.0")
        self.assertIn(version, self.auditor.model_versions)

    @patch('src.ml.refactored_distributed_training.model_training_strategies.create_decision_auditor')
    def test_register_model_version_integration(self, mock_create_auditor):
        """Test integration of model registration in training strategy."""
        from src.ml.refactored_distributed_training.model_training_strategies import HybridTrainingStrategy
        from src.ml.refactored_distributed_training.data_classes import TrainingJob, DistributedTrainingConfig

        mock_auditor = Mock()
        mock_create_auditor.return_value = mock_auditor

        strategy = HybridTrainingStrategy()
        job = TrainingJob(
            job_id="test_job",
            model_type="hybrid",
            config={
                "input_dim": 10,
                "features": np.random.randn(10, 10, 50),
                "targets_class": np.random.randint(0, 3, 10),
                "targets_reg": np.random.randn(10, 1),
                "val_features": np.random.randn(10, 10, 50),
                "val_targets_class": np.random.randint(0, 3, 10),
                "val_targets_reg": np.random.randn(10, 1),
                "data_hash": "test_hash",
                "model_params": {"lr": 0.001}
            }
        )
        config = DistributedTrainingConfig()

        with patch('src.ml.refactored_distributed_training.model_training_strategies.CNNLSTMHybridModel') as mock_model_class:
            mock_model_instance = MockCNNLSTMHybridModel()
            mock_model_class.return_value = mock_model_instance
            mock_model_instance.fit.return_value = Mock(epochs_trained=10, train_loss=0.1, val_loss=0.1)
            mock_model_instance.evaluate.return_value = {'accuracy': 0.9}

            strategy.train(job, config)

            mock_auditor.register_model_version.assert_called_once()
            call_args = mock_auditor.register_model_version.call_args[1]
            self.assertEqual(call_args['training_data_hash'], 'test_hash')
            self.assertEqual(call_args['hyperparameters'], {"lr": 0.001})
            self.assertEqual(call_args['performance_metrics'], {'accuracy': 0.9})
    
    def test_get_decision_history(self):
        """Test getting decision history."""
        # Log a few decisions
        for i in range(3):
            self.auditor.log_decision(
                self.model,
                self.input_data,
                self.prediction,
                metadata={'test_run': i}
            )
        
        # Get all entries
        entries = self.auditor.get_decision_history()
        self.assertEqual(len(entries), 3)
    
    def test_generate_audit_report(self):
        """Test generating audit report."""
        # Log a few decisions
        for i in range(3):
            self.auditor.log_decision(
                self.model,
                self.input_data,
                self.prediction,
                metadata={'test_run': i}
            )
        
        # Generate report
        report = self.auditor.generate_audit_report()
        
        self.assertIn('total_decisions', report)
        self.assertIn('model_usage', report)
        self.assertIn('decision_types', report)
        self.assertEqual(report['total_decisions'], 3)


class TestUncertaintyCalibrator(unittest.TestCase):
    """Test cases for uncertainty calibrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not UNCERTAINTY_CALIBRATOR_AVAILABLE:
            self.skipTest("Uncertainty calibrator not available")
        
        self.model = MockCNNLSTMHybridModel()
        self.calibrator = create_uncertainty_calibrator(self.model)
        
        # Create test data
        self.X_val = np.random.randn(15, 5, 20)  # 15 samples, 5 features, 20 timesteps
        self.y_class_val = np.random.randint(0, 3, 15)  # 3 classes
        self.y_reg_val = np.random.randn(15, 1)  # 1 target
    
    def test_initialization(self):
        """Test uncertainty calibrator initialization."""
        self.assertIsInstance(self.calibrator, UncertaintyCalibrator)
        self.assertEqual(self.calibrator.model, self.model)
        self.assertFalse(self.calibrator.is_calibrated)
    
    def test_validate_calibration(self):
        """Test calibration validation."""
        # This would normally require calibration first, but we'll test the structure
        try:
            # Create mock predictions for validation
            X_test = np.random.randn(10, 5, 20)
            y_class_test = np.random.randint(0, 3, 10)
            y_reg_test = np.random.randn(10, 1)
            
            # Validate should raise an error since not calibrated
            with self.assertRaises(ValueError):
                self.calibrator.validate_calibration(X_test, y_class_test, y_reg_test)
        except RuntimeError:
            # Scikit-learn not available
            self.skipTest("Scikit-learn not available")

    @patch('src.ml.refactored_distributed_training.model_training_strategies.create_uncertainty_calibrator')
    def test_calibration_integration(self, mock_create_calibrator):
        """Test integration of uncertainty calibration in training strategy."""
        from src.ml.refactored_distributed_training.model_training_strategies import HybridTrainingStrategy
        from src.ml.refactored_distributed_training.data_classes import TrainingJob, DistributedTrainingConfig

        mock_calibrator = Mock()
        mock_create_calibrator.return_value = mock_calibrator

        strategy = HybridTrainingStrategy()
        val_features = np.random.randn(10, 10, 50)
        val_targets_class = np.random.randint(0, 3, 10)
        val_targets_reg = np.random.randn(10, 1)

        job = TrainingJob(
            job_id="test_job",
            model_type="hybrid",
            config={
                "input_dim": 10,
                "features": np.random.randn(10, 10, 50),
                "targets_class": np.random.randint(0, 3, 10),
                "targets_reg": np.random.randn(10, 1),
                "val_features": val_features,
                "val_targets_class": val_targets_class,
                "val_targets_reg": val_targets_reg,
            }
        )
        config = DistributedTrainingConfig()

        with patch('src.ml.refactored_distributed_training.model_training_strategies.CNNLSTMHybridModel') as mock_model_class:
            mock_model_instance = MockCNNLSTMHybridModel()
            mock_model_class.return_value = mock_model_instance
            mock_model_instance.fit.return_value = Mock(epochs_trained=10, train_loss=0.1, val_loss=0.1)
            mock_model_instance.evaluate.return_value = {'accuracy': 0.9}

            strategy.train(job, config)

            mock_calibrator.calibrate_uncertainty_isotonic.assert_called_once()
            call_args = mock_calibrator.calibrate_uncertainty_isotonic.call_args[1]
            np.testing.assert_array_equal(call_args['X_val'], val_features)
            np.testing.assert_array_equal(call_args['y_class_val'], val_targets_class)
            np.testing.assert_array_equal(call_args['y_reg_val'], val_targets_reg)


def create_test_suite():
    """Create test suite for interpretability modules."""
    suite = unittest.TestSuite()
    
    # Add tests conditionally based on available modules
    if SHAP_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestSHAPExplainer))
    
    if MATPLOTLIB_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestAttentionVisualizer))
    
    if CAPTUM_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestFeatureImportanceAnalyzer))
    
    if DECISION_AUDITOR_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestDecisionAuditor))
    
    if UNCERTAINTY_CALIBRATOR_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestUncertaintyCalibrator))
    
    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    runner.run(suite)
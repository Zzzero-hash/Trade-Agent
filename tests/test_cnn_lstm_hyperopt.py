"""
Tests for CNN+LSTM Hyperparameter Optimization - Task 5.5

This module tests the hyperparameter optimization implementation.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.cnn_lstm_hyperopt import CNNLSTMHyperparameterOptimizer
from src.ml.hybrid_model import create_hybrid_config


@pytest.fixture
def sample_data_loaders():
    """Create sample data loaders for testing"""
    
    # Create synthetic data
    num_samples = 100
    input_dim = 11
    sequence_length = 60
    
    X = torch.randn(num_samples, input_dim, sequence_length)
    y_class = torch.randint(0, 4, (num_samples,))
    y_reg = torch.randn(num_samples, 2)
    
    dataset = TensorDataset(X, y_class, y_reg)
    
    # Split data
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader, test_loader


@pytest.fixture
def temp_save_dir():
    """Create temporary directory for saving results"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def hyperopt_optimizer(sample_data_loaders, temp_save_dir):
    """Create hyperparameter optimizer for testing"""
    
    optimizer = CNNLSTMHyperparameterOptimizer(
        data_loaders=sample_data_loaders,
        input_dim=11,
        save_dir=temp_save_dir,
        study_name="test_study",
        device="cpu"
    )
    
    return optimizer


class TestCNNLSTMHyperparameterOptimizer:
    """Test CNN+LSTM hyperparameter optimizer"""
    
    def test_initialization(self, sample_data_loaders, temp_save_dir):
        """Test optimizer initialization"""
        
        optimizer = CNNLSTMHyperparameterOptimizer(
            data_loaders=sample_data_loaders,
            input_dim=11,
            save_dir=temp_save_dir,
            study_name="test_study"
        )
        
        assert optimizer.input_dim == 11
        assert optimizer.study_name == "test_study"
        assert optimizer.save_dir == Path(temp_save_dir)
        assert optimizer.device.type in ["cuda", "cpu"]
    
    def test_create_study(self, hyperopt_optimizer):
        """Test study creation"""
        
        study = hyperopt_optimizer.create_study(
            n_trials=10,
            pruner_type="median",
            sampler_type="tpe"
        )
        
        assert study is not None
        assert len(study.directions) == 3  # Multi-objective
    
    def test_suggest_hyperparameters(self, hyperopt_optimizer):
        """Test hyperparameter suggestion"""
        
        # Mock trial object
        trial = Mock()
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
        trial.suggest_float = Mock(return_value=0.1)
        trial.suggest_int = Mock(return_value=2)
        
        hyperparams = hyperopt_optimizer.suggest_hyperparameters(trial)
        
        assert isinstance(hyperparams, dict)
        assert "cnn_num_filters" in hyperparams
        assert "lstm_hidden_dim" in hyperparams
        assert "fusion_method" in hyperparams
        assert "learning_rate" in hyperparams
        assert "classification_weight" in hyperparams
    
    def test_create_config_from_hyperparams(self, hyperopt_optimizer):
        """Test configuration creation from hyperparameters"""
        
        hyperparams = {
            "cnn_num_filters": 64,
            "cnn_filter_sizes": [3, 5, 7],
            "cnn_dropout_rate": 0.2,
            "cnn_use_attention": True,
            "cnn_attention_heads": 4,
            "lstm_hidden_dim": 128,
            "lstm_num_layers": 2,
            "lstm_dropout_rate": 0.3,
            "lstm_bidirectional": True,
            "lstm_use_attention": True,
            "lstm_attention_heads": 8,
            "lstm_use_skip_connections": False,
            "fusion_method": "attention",
            "fusion_dim": 256,
            "fusion_dropout_rate": 0.2,
            "fusion_num_heads": 8,
            "learning_rate": 0.001,
            "batch_size": 32,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "gradient_clip_norm": 1.0,
            "l1_reg_weight": 0.0,
            "l2_reg_weight": 1e-5,
            "label_smoothing": 0.1,
            "classification_weight": 0.6,
            "regression_weight": 0.4
        }
        
        config = hyperopt_optimizer._create_config_from_hyperparams(hyperparams)
        
        assert config.cnn_num_filters == 64
        assert config.lstm_hidden_dim == 128
        assert config.fusion_method == "attention"
        assert config.learning_rate == 0.001
    
    def test_calculate_model_size(self, hyperopt_optimizer):
        """Test model size calculation"""
        
        config = create_hybrid_config(
            input_dim=11,
            sequence_length=60,
            num_classes=4,
            regression_targets=2
        )
        
        model_size = hyperopt_optimizer._calculate_model_size(config)
        
        assert isinstance(model_size, float)
        assert model_size > 0
    
    def test_train_trial_model(self, hyperopt_optimizer):
        """Test trial model training"""
        
        # Create simple configuration
        config = create_hybrid_config(
            input_dim=11,
            sequence_length=60,
            num_classes=4,
            regression_targets=2,
            device="cpu"
        )
        
        # Mock trial object
        trial = Mock()
        trial.number = 1
        trial.report = Mock()
        trial.should_prune = Mock(return_value=False)
        
        # Train model (with reduced epochs for testing)
        metrics = hyperopt_optimizer._train_trial_model(
            config=config,
            trial=trial,
            max_epochs=2
        )
        
        assert isinstance(metrics, dict)
        assert "val_class_acc" in metrics
        assert "val_total_loss" in metrics
        assert "epochs_trained" in metrics
    
    def test_evaluate_on_test_set(self, hyperopt_optimizer):
        """Test test set evaluation"""
        
        # Create simple model
        config = create_hybrid_config(
            input_dim=11,
            sequence_length=60,
            num_classes=4,
            regression_targets=2,
            device="cpu"
        )
        
        from src.ml.hybrid_model import CNNLSTMHybridModel
        model = CNNLSTMHybridModel(config)
        
        # Evaluate model
        test_metrics = hyperopt_optimizer._evaluate_on_test_set(model)
        
        assert isinstance(test_metrics, dict)
        assert "test_class_acc" in test_metrics
        assert "test_total_loss" in test_metrics
        assert "test_reg_mse" in test_metrics
    
    @patch('optuna.create_study')
    def test_optimize_small_scale(self, mock_create_study, hyperopt_optimizer):
        """Test optimization with small number of trials"""
        
        # Mock study
        mock_study = Mock()
        mock_study.best_trials = []
        mock_study.trials = []
        mock_create_study.return_value = mock_study
        
        # Run optimization with minimal trials
        results = hyperopt_optimizer.optimize(n_trials=2, timeout=10)
        
        assert isinstance(results, dict)
        assert "study_name" in results
        assert "n_trials_completed" in results
        assert "best_trials" in results
    
    def test_generate_optimization_report(self, hyperopt_optimizer):
        """Test optimization report generation"""
        
        # Add some mock history
        hyperopt_optimizer.optimization_history = [
            {
                'accuracy': 0.8,
                'training_time': 10.0,
                'model_size': 5.0,
                'trial_number': 1
            },
            {
                'accuracy': 0.85,
                'training_time': 12.0,
                'model_size': 6.0,
                'trial_number': 2
            }
        ]
        
        hyperopt_optimizer.best_configs = [
            {'learning_rate': 0.001, 'batch_size': 32},
            {'learning_rate': 0.0005, 'batch_size': 64}
        ]
        
        report = hyperopt_optimizer.generate_optimization_report()
        
        assert isinstance(report, str)
        assert "Optimization Summary" in report
        assert "Performance Statistics" in report
        assert "Best Configurations" in report


class TestHyperoptIntegration:
    """Test integration with existing components"""
    
    def test_integration_with_trainer(self, sample_data_loaders, temp_save_dir):
        """Test integration with IntegratedCNNLSTMTrainer"""
        
        from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
        
        # Create configuration
        config = create_hybrid_config(
            input_dim=11,
            sequence_length=60,
            num_classes=4,
            regression_targets=2,
            device="cpu"
        )
        
        # Create trainer
        trainer = IntegratedCNNLSTMTrainer(
            config=config,
            save_dir=temp_save_dir,
            device="cpu"
        )
        
        # Build models
        trainer.build_models(input_dim=11)
        
        assert trainer.hybrid_model is not None
        assert trainer.cnn_baseline is not None
        assert trainer.lstm_baseline is not None
    
    def test_config_compatibility(self):
        """Test configuration compatibility"""
        
        # Test that hyperparameter configurations are compatible with model configs
        hyperparams = {
            "cnn_num_filters": 64,
            "cnn_filter_sizes": [3, 5, 7],
            "lstm_hidden_dim": 128,
            "fusion_method": "attention",
            "learning_rate": 0.001
        }
        
        optimizer = CNNLSTMHyperparameterOptimizer(
            data_loaders=(Mock(), Mock(), Mock()),
            input_dim=11,
            save_dir="temp"
        )
        
        # Should not raise any errors
        config = optimizer._create_config_from_hyperparams(hyperparams)
        assert config is not None


class TestHyperoptUtilities:
    """Test utility functions"""
    
    def test_run_cnn_lstm_hyperparameter_optimization(self, sample_data_loaders, temp_save_dir):
        """Test convenience function"""
        
        from src.ml.cnn_lstm_hyperopt import run_cnn_lstm_hyperparameter_optimization
        
        # Mock the optimization to avoid long running tests
        with patch('src.ml.cnn_lstm_hyperopt.CNNLSTMHyperparameterOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = {
                'best_trials': [],
                'n_trials_completed': 2
            }
            mock_optimizer.retrain_best_models.return_value = []
            mock_optimizer.generate_optimization_report.return_value = "Test report"
            mock_optimizer_class.return_value = mock_optimizer
            
            results = run_cnn_lstm_hyperparameter_optimization(
                data_loaders=sample_data_loaders,
                input_dim=11,
                n_trials=2,
                save_dir=temp_save_dir,
                retrain_best=False
            )
            
            assert isinstance(results, dict)
            assert "optimization_results" in results
            assert "optimization_report" in results


if __name__ == "__main__":
    pytest.main([__file__])
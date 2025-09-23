"""
Test Suite for Hyperparameter Optimization - Task 5.5

This module contains comprehensive tests for the hyperparameter optimization system
including unit tests, integration tests, and validation tests.

Requirements: 3.4, 9.1
"""

import os
import sys
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.hyperparameter_optimizer import (
    MultiObjectiveOptimizer, 
    OptimizationConfig, 
    create_optimization_config
)
from src.ml.hybrid_model import HybridModelConfig, create_hybrid_config
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer


class TestOptimizationConfig:
    """Test OptimizationConfig class"""
    
    def test_default_config_creation(self):
        """Test creating default optimization configuration"""
        config = OptimizationConfig()
        
        assert config.n_trials == 1000
        assert config.timeout == 48 * 3600
        assert config.objectives == ['accuracy', 'training_time', 'model_size']
        assert config.max_epochs_per_trial == 50
        assert config.early_stopping_patience == 10
        assert config.max_model_size_mb == 500.0
        assert config.max_training_time_minutes == 120.0
    
    def test_custom_config_creation(self):
        """Test creating custom optimization configuration"""
        config = OptimizationConfig(
            n_trials=500,
            timeout=24 * 3600,
            objectives=['accuracy'],
            max_epochs_per_trial=30
        )
        
        assert config.n_trials == 500
        assert config.timeout == 24 * 3600
        assert config.objectives == ['accuracy']
        assert config.max_epochs_per_trial == 30
    
    def test_search_space_config(self):
        """Test search space configuration"""
        config = OptimizationConfig()
        
        assert config.search_space_config is not None
        assert 'learning_rate' in config.search_space_config
        assert 'cnn_num_filters' in config.search_space_config
        assert 'lstm_hidden_dim' in config.search_space_config
        
        # Test learning rate configuration
        lr_config = config.search_space_config['learning_rate']
        assert lr_config['type'] == 'loguniform'
        assert lr_config['low'] == 1e-5
        assert lr_config['high'] == 1e-2
    
    def test_objective_weights(self):
        """Test objective weights configuration"""
        config = OptimizationConfig()
        
        assert config.objective_weights is not None
        assert 'accuracy' in config.objective_weights
        assert 'training_time' in config.objective_weights
        assert 'model_size' in config.objective_weights
        
        # Weights should sum to 1.0
        total_weight = sum(config.objective_weights.values())
        assert abs(total_weight - 1.0) < 1e-6


class TestMultiObjectiveOptimizer:
    """Test MultiObjectiveOptimizer class"""
    
    @pytest.fixture
    def sample_data_loaders(self):
        """Create sample data loaders for testing"""
        # Create synthetic data
        batch_size = 16
        sequence_length = 30
        feature_dim = 11
        num_samples = 100
        
        # Generate synthetic data
        X = torch.randn(num_samples, feature_dim, sequence_length)
        y_class = torch.randint(0, 4, (num_samples,))
        y_reg = torch.randn(num_samples, 2)
        
        dataset = TensorDataset(X, y_class, y_reg)
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    @pytest.fixture
    def temp_save_dir(self):
        """Create temporary directory for saving results"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_optimizer_initialization(self, sample_data_loaders, temp_save_dir):
        """Test optimizer initialization"""
        config = OptimizationConfig(
            n_trials=10,
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        assert optimizer.config == config
        assert optimizer.device == torch.device('cpu')
        assert optimizer.save_dir == Path(temp_save_dir)
        assert optimizer.study is None
        assert optimizer.best_trials == []
    
    def test_create_study(self, sample_data_loaders, temp_save_dir):
        """Test study creation"""
        config = OptimizationConfig(
            n_trials=10,
            objectives=['accuracy', 'training_time'],
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        study = optimizer.create_study()
        
        assert study is not None
        assert len(study.directions) == 2
        assert study.directions[0] == optuna.study.StudyDirection.MAXIMIZE  # accuracy
        assert study.directions[1] == optuna.study.StudyDirection.MINIMIZE  # training_time
    
    def test_suggest_hyperparameters(self, sample_data_loaders, temp_save_dir):
        """Test hyperparameter suggestion"""
        config = OptimizationConfig(
            n_trials=10,
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        # Create a mock trial
        trial = Mock()
        trial.suggest_uniform = Mock(return_value=0.3)
        trial.suggest_loguniform = Mock(return_value=1e-3)
        trial.suggest_int = Mock(return_value=2)
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
        
        params = optimizer.suggest_hyperparameters(trial)
        
        assert 'learning_rate' in params
        assert 'cnn_num_filters' in params
        assert 'lstm_hidden_dim' in params
        assert 'dropout_rate' in params
        assert 'batch_size' in params
    
    def test_create_model_config(self, sample_data_loaders, temp_save_dir):
        """Test model configuration creation from hyperparameters"""
        config = OptimizationConfig(
            n_trials=10,
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        # Sample hyperparameters
        params = {
            'learning_rate': 1e-3,
            'cnn_num_filters': 64,
            'cnn_filter_sizes': [3, 5, 7],
            'cnn_attention_heads': 8,
            'lstm_hidden_dim': 128,
            'lstm_num_layers': 2,
            'lstm_bidirectional': True,
            'feature_fusion_dim': 256,
            'num_ensemble_models': 3,
            'dropout_rate': 0.3,
            'batch_size': 32,
            'classification_weight': 0.4,
            'regression_weight': 0.6
        }
        
        model_config = optimizer.create_model_config(params)
        
        assert isinstance(model_config, HybridModelConfig)
        assert model_config.learning_rate == 1e-3
        assert model_config.cnn_num_filters == 64
        assert model_config.lstm_hidden_dim == 128
        assert model_config.dropout_rate == 0.3
    
    def test_calculate_model_size(self, sample_data_loaders, temp_save_dir):
        """Test model size calculation"""
        config = OptimizationConfig(
            n_trials=10,
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        size_mb = optimizer._calculate_model_size(model)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    @patch('src.ml.hyperparameter_optimizer.IntegratedCNNLSTMTrainer')
    def test_objective_function_success(self, mock_trainer_class, sample_data_loaders, temp_save_dir):
        """Test successful objective function execution"""
        config = OptimizationConfig(
            n_trials=10,
            max_epochs_per_trial=5,
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.build_models = Mock()
        mock_trainer.hybrid_model = Mock()
        mock_trainer.hybrid_model.parameters = Mock(return_value=[torch.randn(10, 10)])
        mock_trainer.hybrid_model.buffers = Mock(return_value=[])
        
        # Mock training results
        mock_training_results = {
            'history': [
                {
                    'val_class_acc': 0.7,
                    'val_reg_mse': 0.1,
                    'val_total_loss': 0.5
                }
            ]
        }
        mock_trainer.train_integrated_model = Mock(return_value=mock_training_results)
        
        mock_trainer_class.return_value = mock_trainer
        
        # Create mock trial
        trial = Mock()
        trial.number = 1
        trial.suggest_uniform = Mock(return_value=0.3)
        trial.suggest_loguniform = Mock(return_value=1e-3)
        trial.suggest_int = Mock(return_value=2)
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
        trial.report = Mock()
        trial.should_prune = Mock(return_value=False)
        
        objectives = optimizer.objective_function(trial)
        
        assert len(objectives) == len(config.objectives)
        assert all(isinstance(obj, (int, float)) for obj in objectives)
    
    @patch('src.ml.hyperparameter_optimizer.IntegratedCNNLSTMTrainer')
    def test_objective_function_pruning(self, mock_trainer_class, sample_data_loaders, temp_save_dir):
        """Test objective function with pruning"""
        config = OptimizationConfig(
            n_trials=10,
            max_epochs_per_trial=5,
            max_model_size_mb=0.001,  # Very small limit to trigger pruning
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        # Mock trainer with large model
        mock_trainer = Mock()
        mock_trainer.build_models = Mock()
        mock_trainer.hybrid_model = Mock()
        # Large model parameters to exceed size limit
        mock_trainer.hybrid_model.parameters = Mock(return_value=[torch.randn(1000, 1000)])
        mock_trainer.hybrid_model.buffers = Mock(return_value=[])
        
        mock_trainer_class.return_value = mock_trainer
        
        # Create mock trial
        trial = Mock()
        trial.number = 1
        trial.suggest_uniform = Mock(return_value=0.3)
        trial.suggest_loguniform = Mock(return_value=1e-3)
        trial.suggest_int = Mock(return_value=2)
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
        
        with pytest.raises(optuna.TrialPruned):
            optimizer.objective_function(trial)
    
    def test_find_pareto_front(self, sample_data_loaders, temp_save_dir):
        """Test Pareto front identification"""
        config = OptimizationConfig(
            n_trials=10,
            objectives=['accuracy', 'training_time'],
            save_dir=temp_save_dir
        )
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=sample_data_loaders,
            device='cpu'
        )
        
        # Create mock trials
        trials = []
        
        # Trial 1: High accuracy, high time (dominated)
        trial1 = Mock()
        trial1.values = [0.8, 100.0]
        trials.append(trial1)
        
        # Trial 2: Medium accuracy, low time (Pareto optimal)
        trial2 = Mock()
        trial2.values = [0.7, 50.0]
        trials.append(trial2)
        
        # Trial 3: Low accuracy, medium time (dominated)
        trial3 = Mock()
        trial3.values = [0.6, 75.0]
        trials.append(trial3)
        
        # Trial 4: High accuracy, low time (Pareto optimal)
        trial4 = Mock()
        trial4.values = [0.9, 60.0]
        trials.append(trial4)
        
        pareto_front = optimizer._find_pareto_front(trials)
        
        # Should contain trials 2 and 4
        assert len(pareto_front) == 2
        assert trial2 in pareto_front
        assert trial4 in pareto_front


class TestIntegrationWithRealData:
    """Integration tests with real data pipeline"""
    
    @pytest.fixture
    def temp_save_dir(self):
        """Create temporary directory for saving results"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.slow
    @patch('src.ml.train_integrated_cnn_lstm.YahooFinanceIngestor')
    def test_end_to_end_optimization_small(self, mock_ingestor, temp_save_dir):
        """Test end-to-end optimization with mocked data"""
        # Mock data ingestor
        mock_data = {
            'Open': np.random.randn(100) * 10 + 100,
            'High': np.random.randn(100) * 10 + 105,
            'Low': np.random.randn(100) * 10 + 95,
            'Close': np.random.randn(100) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }
        
        import pandas as pd
        mock_df = pd.DataFrame(mock_data)
        mock_df.index = pd.date_range('2023-01-01', periods=100, freq='5min')
        
        mock_ingestor_instance = Mock()
        mock_ingestor_instance.fetch_symbol_data = Mock(return_value=mock_df)
        mock_ingestor.return_value = mock_ingestor_instance
        
        # Create small optimization configuration
        config = OptimizationConfig(
            n_trials=3,  # Very small for testing
            max_epochs_per_trial=2,
            timeout=300,  # 5 minutes
            objectives=['accuracy'],
            save_dir=temp_save_dir
        )
        
        # Create trainer to prepare data
        temp_config = create_hybrid_config(
            input_dim=11,
            sequence_length=20,
            device='cpu'
        )
        
        trainer = IntegratedCNNLSTMTrainer(
            config=temp_config,
            save_dir=temp_save_dir,
            device='cpu'
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-02-01',
            timeframes=['5min'],
            sequence_length=20,
            batch_size=8
        )
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(train_loader, val_loader, test_loader),
            device='cpu'
        )
        
        # Run optimization
        study = optimizer.optimize()
        
        # Verify results
        assert study is not None
        assert len(study.trials) <= config.n_trials
        
        # Check if results were saved
        assert (Path(temp_save_dir) / "optimization_analysis.json").exists()
        assert (Path(temp_save_dir) / "best_configurations.json").exists()


class TestConfigurationLoading:
    """Test configuration loading and validation"""
    
    def test_create_optimization_config_function(self):
        """Test create_optimization_config function"""
        config = create_optimization_config(
            n_trials=500,
            max_epochs_per_trial=30,
            objectives=['accuracy', 'model_size'],
            save_dir='test_results'
        )
        
        assert config.n_trials == 500
        assert config.max_epochs_per_trial == 30
        assert config.objectives == ['accuracy', 'model_size']
        assert config.save_dir == 'test_results'
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid objectives
        with pytest.raises(ValueError):
            config = OptimizationConfig(objectives=[])
        
        # Test invalid trial count
        with pytest.raises(ValueError):
            config = OptimizationConfig(n_trials=0)
        
        # Test invalid timeout
        with pytest.raises(ValueError):
            config = OptimizationConfig(timeout=-1)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_model_size_calculation(self):
        """Test model size calculation utility"""
        from src.ml.hyperparameter_optimizer import MultiObjectiveOptimizer
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Create dummy optimizer to access method
        config = OptimizationConfig(n_trials=1)
        dummy_data = torch.randn(10, 5, 20)
        dummy_targets_class = torch.randint(0, 4, (10,))
        dummy_targets_reg = torch.randn(10, 2)
        dummy_dataset = TensorDataset(dummy_data, dummy_targets_class, dummy_targets_reg)
        dummy_loader = DataLoader(dummy_dataset, batch_size=2)
        
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(dummy_loader, dummy_loader, dummy_loader),
            device='cpu'
        )
        
        size_mb = optimizer._calculate_model_size(model)
        
        assert size_mb > 0
        assert size_mb < 1.0  # Should be small for this simple model


@pytest.mark.integration
class TestFullOptimizationWorkflow:
    """Integration tests for full optimization workflow"""
    
    @pytest.fixture
    def temp_save_dir(self):
        """Create temporary directory for saving results"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_optimization_workflow_components(self, temp_save_dir):
        """Test that all optimization workflow components work together"""
        # This test verifies that all components can be instantiated and configured
        # without running the full optimization (which would be too slow for unit tests)
        
        # 1. Create configuration
        config = create_optimization_config(
            n_trials=5,
            max_epochs_per_trial=2,
            objectives=['accuracy'],
            save_dir=temp_save_dir
        )
        
        # 2. Create sample data
        sample_data = torch.randn(50, 11, 30)
        sample_targets_class = torch.randint(0, 4, (50,))
        sample_targets_reg = torch.randn(50, 2)
        
        dataset = TensorDataset(sample_data, sample_targets_class, sample_targets_reg)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # 3. Create optimizer
        optimizer = MultiObjectiveOptimizer(
            config=config,
            data_loaders=(train_loader, val_loader, test_loader),
            device='cpu'
        )
        
        # 4. Test study creation
        study = optimizer.create_study()
        assert study is not None
        
        # 5. Test hyperparameter suggestion
        trial = study.ask()
        params = optimizer.suggest_hyperparameters(trial)
        assert len(params) > 0
        
        # 6. Test model config creation
        model_config = optimizer.create_model_config(params)
        assert isinstance(model_config, HybridModelConfig)
        
        # 7. Test trainer creation
        trainer = IntegratedCNNLSTMTrainer(
            config=model_config,
            save_dir=temp_save_dir,
            device='cpu'
        )
        
        # 8. Test model building
        trainer.build_models(model_config.input_dim)
        assert trainer.hybrid_model is not None
        
        # All components work together successfully


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
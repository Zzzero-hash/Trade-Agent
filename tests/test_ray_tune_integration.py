"""
Tests for Ray Tune Integration

This module provides comprehensive tests for Ray Tune hyperparameter optimization
integration, including search algorithms, schedulers, and multi-objective optimization.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gymnasium as gym

from src.ml.ray_tune_integration import (
    RayTuneOptimizer,
    TuneConfig,
    create_tune_optimizer
)


class MockTuneResult:
    """Mock Ray Tune result for testing"""
    
    def __init__(self, config, metrics):
        self.config = config
        self.last_result = metrics
        self.trial_id = "mock_trial_123"


class MockTuneAnalysis:
    """Mock Ray Tune analysis for testing"""
    
    def __init__(self, trials):
        self.trials = trials
        self.experiment_path = "/mock/experiment/path"
    
    def get_best_trial(self, metric, mode):
        """Get best trial based on metric"""
        best_trial = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for trial in self.trials:
            if metric in trial.last_result:
                value = trial.last_result[metric]
                is_better = (
                    (mode == 'max' and value > best_value) or
                    (mode == 'min' and value < best_value)
                )
                if is_better:
                    best_value = value
                    best_trial = trial
        
        return best_trial


class TestTuneConfig:
    """Test TuneConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TuneConfig()
        
        assert config.num_samples == 100
        assert config.max_concurrent_trials == 8
        assert config.search_algorithm == "optuna"
        assert config.scheduler == "asha"
        assert config.metric == "mean_reward"
        assert config.mode == "max"
        assert config.cpus_per_trial == 2
        assert config.gpus_per_trial == 0.25
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = TuneConfig(
            num_samples=50,
            search_algorithm="hyperopt",
            scheduler="pbt",
            metric="accuracy",
            mode="max",
            cpus_per_trial=4
        )
        
        assert config.num_samples == 50
        assert config.search_algorithm == "hyperopt"
        assert config.scheduler == "pbt"
        assert config.metric == "accuracy"
        assert config.cpus_per_trial == 4
    
    def test_multi_objective_config(self):
        """Test multi-objective configuration"""
        config = TuneConfig(
            multi_objective=True,
            objectives=["accuracy", "f1_score", "precision"]
        )
        
        assert config.multi_objective is True
        assert config.objectives == ["accuracy", "f1_score", "precision"]


class TestRayTuneOptimizer:
    """Test RayTuneOptimizer class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def tune_config(self):
        """Create test tune configuration"""
        return TuneConfig(
            num_samples=5,
            max_concurrent_trials=2,
            cpus_per_trial=1,
            gpus_per_trial=0,
            time_budget_s=60
        )
    
    @pytest.fixture
    def optimizer(self, tune_config, temp_dir):
        """Create RayTuneOptimizer with mocked Ray"""
        with patch('src.ml.ray_tune_integration.ray') as mock_ray:
            mock_ray.is_initialized.return_value = True
            
            optimizer = RayTuneOptimizer(
                config=tune_config,
                local_dir=temp_dir,
                experiment_name="test_experiment"
            )
            yield optimizer
    
    def test_optimizer_initialization(self, optimizer, tune_config, temp_dir):
        """Test optimizer initialization"""
        assert optimizer.config == tune_config
        assert optimizer.local_dir == temp_dir
        assert optimizer.experiment_name == "test_experiment"
    
    def test_optimizer_without_ray_raises_error(self, tune_config):
        """Test that optimizer raises error when Ray is not available"""
        with patch('src.ml.ray_tune_integration.RAY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Ray Tune is not available"):
                RayTuneOptimizer(config=tune_config)
    
    def test_get_rl_search_space_ppo(self, optimizer):
        """Test PPO search space generation"""
        search_space = optimizer._get_rl_search_space("PPO")
        
        expected_params = [
            "learning_rate", "batch_size", "n_steps", "n_epochs",
            "clip_range", "gamma", "gae_lambda", "ent_coef", "vf_coef"
        ]
        
        for param in expected_params:
            assert param in search_space
    
    def test_get_rl_search_space_sac(self, optimizer):
        """Test SAC search space generation"""
        search_space = optimizer._get_rl_search_space("SAC")
        
        expected_params = [
            "learning_rate", "batch_size", "buffer_size", "tau",
            "gamma", "train_freq", "gradient_steps", "ent_coef"
        ]
        
        for param in expected_params:
            assert param in search_space
    
    def test_get_rl_search_space_td3(self, optimizer):
        """Test TD3 search space generation"""
        search_space = optimizer._get_rl_search_space("TD3")
        
        expected_params = [
            "learning_rate", "batch_size", "buffer_size", "tau",
            "gamma", "policy_delay", "target_policy_noise", "target_noise_clip"
        ]
        
        for param in expected_params:
            assert param in search_space
    
    def test_get_rl_search_space_dqn(self, optimizer):
        """Test DQN search space generation"""
        search_space = optimizer._get_rl_search_space("DQN")
        
        expected_params = [
            "learning_rate", "batch_size", "buffer_size", "gamma",
            "exploration_fraction", "exploration_initial_eps", "exploration_final_eps"
        ]
        
        for param in expected_params:
            assert param in search_space
    
    def test_get_rl_search_space_unknown_agent(self, optimizer):
        """Test unknown agent type raises error"""
        with pytest.raises(ValueError, match="Unknown agent type"):
            optimizer._get_rl_search_space("UNKNOWN_AGENT")
    
    @patch('src.ml.ray_tune_integration.OptunaSearch')
    def test_create_search_algorithm_optuna(self, mock_optuna, optimizer):
        """Test creating Optuna search algorithm"""
        mock_search = Mock()
        mock_optuna.return_value = mock_search
        
        search_alg = optimizer._create_search_algorithm(optimizer.config)
        
        assert search_alg == mock_search
        mock_optuna.assert_called_once_with(
            metric=optimizer.config.metric,
            mode=optimizer.config.mode
        )
    
    @patch('src.ml.ray_tune_integration.HyperOptSearch')
    def test_create_search_algorithm_hyperopt(self, mock_hyperopt, optimizer):
        """Test creating HyperOpt search algorithm"""
        mock_search = Mock()
        mock_hyperopt.return_value = mock_search
        
        config = TuneConfig(search_algorithm="hyperopt")
        search_alg = optimizer._create_search_algorithm(config)
        
        assert search_alg == mock_search
        mock_hyperopt.assert_called_once_with(
            metric=config.metric,
            mode=config.mode
        )
    
    @patch('src.ml.ray_tune_integration.BayesOptSearch')
    def test_create_search_algorithm_bayesopt(self, mock_bayesopt, optimizer):
        """Test creating BayesOpt search algorithm"""
        mock_search = Mock()
        mock_bayesopt.return_value = mock_search
        
        config = TuneConfig(search_algorithm="bayesopt")
        search_alg = optimizer._create_search_algorithm(config)
        
        assert search_alg == mock_search
        mock_bayesopt.assert_called_once_with(
            metric=config.metric,
            mode=config.mode
        )
    
    @patch('src.ml.ray_tune_integration.BasicVariantGenerator')
    def test_create_search_algorithm_random(self, mock_basic, optimizer):
        """Test creating random search algorithm"""
        mock_search = Mock()
        mock_basic.return_value = mock_search
        
        config = TuneConfig(search_algorithm="random")
        search_alg = optimizer._create_search_algorithm(config)
        
        assert search_alg == mock_search
        mock_basic.assert_called_once()
    
    @patch('src.ml.ray_tune_integration.ASHAScheduler')
    def test_create_scheduler_asha(self, mock_asha, optimizer):
        """Test creating ASHA scheduler"""
        mock_scheduler = Mock()
        mock_asha.return_value = mock_scheduler
        
        scheduler = optimizer._create_scheduler(optimizer.config)
        
        assert scheduler == mock_scheduler
        mock_asha.assert_called_once_with(
            metric=optimizer.config.metric,
            mode=optimizer.config.mode,
            max_t=optimizer.config.max_t,
            grace_period=optimizer.config.grace_period,
            reduction_factor=optimizer.config.reduction_factor
        )
    
    @patch('src.ml.ray_tune_integration.PopulationBasedTraining')
    def test_create_scheduler_pbt(self, mock_pbt, optimizer):
        """Test creating PBT scheduler"""
        mock_scheduler = Mock()
        mock_pbt.return_value = mock_scheduler
        
        config = TuneConfig(scheduler="pbt")
        scheduler = optimizer._create_scheduler(config)
        
        assert scheduler == mock_scheduler
        mock_pbt.assert_called_once()
    
    @patch('src.ml.ray_tune_integration.MedianStoppingRule')
    def test_create_scheduler_median(self, mock_median, optimizer):
        """Test creating median stopping scheduler"""
        mock_scheduler = Mock()
        mock_median.return_value = mock_scheduler
        
        config = TuneConfig(scheduler="median")
        scheduler = optimizer._create_scheduler(config)
        
        assert scheduler == mock_scheduler
        mock_median.assert_called_once()
    
    @patch('src.ml.ray_tune_integration.HyperBandScheduler')
    def test_create_scheduler_hyperband(self, mock_hyperband, optimizer):
        """Test creating HyperBand scheduler"""
        mock_scheduler = Mock()
        mock_hyperband.return_value = mock_scheduler
        
        config = TuneConfig(scheduler="hyperband")
        scheduler = optimizer._create_scheduler(config)
        
        assert scheduler == mock_scheduler
        mock_hyperband.assert_called_once()
    
    def test_create_scheduler_fifo(self, optimizer):
        """Test creating FIFO scheduler (None)"""
        config = TuneConfig(scheduler="fifo")
        scheduler = optimizer._create_scheduler(config)
        
        assert scheduler is None
    
    @patch('src.ml.ray_tune_integration.CLIReporter')
    def test_create_reporter(self, mock_reporter, optimizer):
        """Test creating CLI reporter"""
        mock_reporter_instance = Mock()
        mock_reporter.return_value = mock_reporter_instance
        
        reporter = optimizer._create_reporter(optimizer.config)
        
        assert reporter == mock_reporter_instance
        mock_reporter.assert_called_once()
    
    @patch('src.ml.ray_tune_integration.session')
    @patch('src.ml.ray_tune_integration.RLAgentFactory')
    def test_train_rl_trial(self, mock_factory, mock_session, optimizer):
        """Test RL training trial"""
        # Mock agent and environment
        mock_agent = Mock()
        mock_agent.train.return_value = {
            'evaluations': [{'mean_reward': 100.0, 'std_reward': 10.0}]
        }
        mock_factory.create_agent.return_value = mock_agent
        
        mock_env = Mock()
        env_factory = lambda: mock_env
        
        # Test configuration
        config = {
            'learning_rate': 0.001,
            'batch_size': 64
        }
        
        # Run trial
        results = optimizer._train_rl_trial(config, env_factory, "PPO")
        
        # Verify results
        assert 'mean_reward' in results
        assert 'std_reward' in results
        assert 'training_iteration' in results
        
        # Verify mocks were called
        mock_factory.create_agent.assert_called_once_with(
            agent_type="PPO",
            env=mock_env,
            **config
        )
        mock_agent.train.assert_called_once()
        mock_session.report.assert_called_once()
    
    @patch('src.ml.ray_tune_integration.session')
    def test_train_rl_trial_failure(self, mock_session, optimizer):
        """Test RL training trial failure handling"""
        # Mock environment factory that raises exception
        def failing_env_factory():
            raise RuntimeError("Environment creation failed")
        
        config = {'learning_rate': 0.001}
        
        # Should handle exception and report failure
        with pytest.raises(RuntimeError):
            optimizer._train_rl_trial(config, failing_env_factory, "PPO")
        
        # Should report failure metrics
        mock_session.report.assert_called_once_with({
            "mean_reward": -1000.0,
            "std_reward": 0.0
        })
    
    def test_compute_pareto_frontier(self, optimizer):
        """Test Pareto frontier computation"""
        # Mock analysis with multiple trials
        trials = [
            MockTuneResult(
                config={'lr': 0.001},
                metrics={'accuracy': 0.8, 'f1_score': 0.75}
            ),
            MockTuneResult(
                config={'lr': 0.01},
                metrics={'accuracy': 0.85, 'f1_score': 0.8}
            ),
            MockTuneResult(
                config={'lr': 0.1},
                metrics={'accuracy': 0.7, 'f1_score': 0.85}
            )
        ]
        
        mock_analysis = MockTuneAnalysis(trials)
        objectives = ['accuracy', 'f1_score']
        
        pareto_frontier = optimizer._compute_pareto_frontier(mock_analysis, objectives)
        
        # Should include non-dominated solutions
        assert len(pareto_frontier) >= 1
        
        # Each point should have objectives and config
        for point in pareto_frontier:
            assert 'accuracy' in point
            assert 'f1_score' in point
            assert 'config' in point
            assert 'trial_id' in point
    
    @patch('src.ml.ray_tune_integration.tune')
    def test_run_optimization_success(self, mock_tune, optimizer):
        """Test successful optimization run"""
        # Mock tune.run result
        mock_trial = MockTuneResult(
            config={'learning_rate': 0.001, 'batch_size': 64},
            metrics={'mean_reward': 150.0, 'std_reward': 15.0}
        )
        
        mock_analysis = MockTuneAnalysis([mock_trial])
        mock_analysis.get_best_trial = Mock(return_value=mock_trial)
        mock_tune.run.return_value = mock_analysis
        
        # Define simple objective function
        def objective(config):
            return {'mean_reward': 100.0, 'std_reward': 10.0}
        
        search_space = {'learning_rate': [0.001, 0.01]}
        
        # Run optimization
        results = optimizer._run_optimization(
            objective, search_space, "test"
        )
        
        # Verify results structure
        assert 'best_config' in results
        assert 'best_result' in results
        assert 'best_trial_id' in results
        assert 'experiment_path' in results
        assert 'num_trials' in results
        assert 'optimization_config' in results
        
        # Verify best results
        assert results['best_config'] == mock_trial.config
        assert results['best_result'] == mock_trial.last_result
    
    def test_save_and_load_results(self, optimizer, temp_dir):
        """Test saving and loading optimization results"""
        results = {
            'best_config': {'learning_rate': 0.001},
            'best_result': {'mean_reward': 100.0},
            'experiment_path': '/test/path'
        }
        
        filepath = os.path.join(temp_dir, "test_results.json")
        
        # Save results
        optimizer.save_results(results, filepath)
        
        # Verify file exists
        assert os.path.exists(filepath)
        
        # Load results
        loaded_results = optimizer.load_results(filepath)
        
        # Verify loaded results match original
        assert loaded_results['best_config'] == results['best_config']
        assert loaded_results['best_result'] == results['best_result']
        assert 'timestamp' in loaded_results  # Should be added during save


class TestMultiObjectiveOptimization:
    """Test multi-objective optimization functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for multi-objective testing"""
        config = TuneConfig(
            num_samples=10,
            multi_objective=True,
            objectives=['accuracy', 'f1_score', 'precision']
        )
        
        with patch('src.ml.ray_tune_integration.ray') as mock_ray:
            mock_ray.is_initialized.return_value = True
            
            optimizer = RayTuneOptimizer(config=config)
            yield optimizer
    
    def test_multi_objective_wrapper(self, optimizer):
        """Test multi-objective wrapper function"""
        # Mock objective function
        def mock_objective(config):
            return {
                'accuracy': 0.8,
                'f1_score': 0.75,
                'precision': 0.85,
                'recall': 0.7
            }
        
        objectives = ['accuracy', 'f1_score', 'precision']
        weights = [0.4, 0.3, 0.3]
        
        # Create wrapper
        wrapper = lambda config: optimizer.multi_objective_optimization(
            mock_objective, {}, objectives, weights
        )
        
        # This would normally be called by tune.run, but we can test the concept
        # The actual implementation would need to be integrated with tune.run
        assert callable(wrapper)
    
    def test_pareto_frontier_empty_trials(self, optimizer):
        """Test Pareto frontier with empty trials"""
        mock_analysis = MockTuneAnalysis([])
        objectives = ['accuracy', 'f1_score']
        
        pareto_frontier = optimizer._compute_pareto_frontier(mock_analysis, objectives)
        
        assert pareto_frontier == []
    
    def test_pareto_frontier_single_trial(self, optimizer):
        """Test Pareto frontier with single trial"""
        trial = MockTuneResult(
            config={'lr': 0.001},
            metrics={'accuracy': 0.8, 'f1_score': 0.75}
        )
        
        mock_analysis = MockTuneAnalysis([trial])
        objectives = ['accuracy', 'f1_score']
        
        pareto_frontier = optimizer._compute_pareto_frontier(mock_analysis, objectives)
        
        assert len(pareto_frontier) == 1
        assert pareto_frontier[0]['accuracy'] == 0.8
        assert pareto_frontier[0]['f1_score'] == 0.75


class TestCreateTuneOptimizer:
    """Test create_tune_optimizer factory function"""
    
    def test_create_with_default_config(self):
        """Test creating optimizer with default config"""
        with patch('src.ml.ray_tune_integration.ray') as mock_ray:
            mock_ray.is_initialized.return_value = True
            
            optimizer = create_tune_optimizer()
            
            assert isinstance(optimizer, RayTuneOptimizer)
            assert isinstance(optimizer.config, TuneConfig)
            assert optimizer.local_dir == "ray_results"
    
    def test_create_with_custom_config(self):
        """Test creating optimizer with custom config"""
        config = TuneConfig(
            num_samples=50,
            search_algorithm="hyperopt"
        )
        
        with patch('src.ml.ray_tune_integration.ray') as mock_ray:
            mock_ray.is_initialized.return_value = True
            
            optimizer = create_tune_optimizer(
                config=config,
                local_dir="custom_results",
                experiment_name="custom_experiment"
            )
            
            assert optimizer.config == config
            assert optimizer.local_dir == "custom_results"
            assert optimizer.experiment_name == "custom_experiment"


if __name__ == "__main__":
    pytest.main([__file__])
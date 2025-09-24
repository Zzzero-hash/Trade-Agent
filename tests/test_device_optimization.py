"""Unit tests for device optimization functionality."""

import pytest
import torch
import warnings
from unittest.mock import patch, MagicMock

from src.utils.device_optimizer import DeviceOptimizer, suppress_sb3_device_warnings


class TestDeviceOptimizer:
    """Test device optimization logic."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimizer = DeviceOptimizer()
    
    def test_cnn_model_uses_gpu_when_available(self):
        """Test that CNN models are assigned GPU when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            optimizer = DeviceOptimizer()
            device = optimizer.get_optimal_device('cnn')
            assert device == 'cuda', "CNN models should use GPU when available"
    
    def test_mlp_model_uses_cpu(self):
        """Test that MLP models use CPU for efficiency."""
        device = self.optimizer.get_optimal_device('mlp')
        assert device == 'cpu', "MLP models should use CPU for efficiency"
    
    def test_rl_mlp_policy_uses_cpu(self):
        """Test that RL MLP policies use CPU."""
        device = self.optimizer.get_optimal_device('rl', 'MlpPolicy')
        assert device == 'cpu', "MLP policies should use CPU"
    
    def test_rl_cnn_policy_uses_gpu_when_available(self):
        """Test that RL CNN policies use GPU when available."""
        with patch('torch.cuda.is_available', return_value=True):
            optimizer = DeviceOptimizer()
            device = optimizer.get_optimal_device('rl', 'CnnPolicy')
            assert device == 'cuda', "CNN policies should use GPU when available"
    
    def test_hybrid_model_uses_gpu_when_available(self):
        """Test that hybrid CNN+LSTM models use GPU."""
        with patch('torch.cuda.is_available', return_value=True):
            optimizer = DeviceOptimizer()
            device = optimizer.get_optimal_device('hybrid')
            assert device == 'cuda', "Hybrid models should use GPU when available"
    
    def test_fallback_to_cpu_when_cuda_unavailable(self):
        """Test fallback to CPU when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = DeviceOptimizer()
            device = optimizer.get_optimal_device('cnn')
            assert device == 'cpu', "Should fallback to CPU when CUDA unavailable"
    
    def test_force_device_override(self):
        """Test that force_device parameter overrides automatic selection."""
        device = self.optimizer.get_optimal_device('cnn', force_device='cpu')
        assert device == 'cpu', "Force device should override automatic selection"
    
    def test_invalid_force_device_fallback(self):
        """Test fallback when invalid device is forced."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = DeviceOptimizer()
            device = optimizer.get_optimal_device('cnn', force_device='cuda')
            assert device == 'cpu', "Should fallback to CPU when CUDA forced but unavailable"
    
    def test_device_info_collection(self):
        """Test that device information is collected correctly."""
        info = self.optimizer._get_device_info()
        assert 'cuda_available' in info
        assert 'cpu_count' in info
        assert isinstance(info['cuda_available'], bool)
        assert isinstance(info['cpu_count'], int)
    
    def test_suppress_warnings_function(self):
        """Test that warning suppression works."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            suppress_sb3_device_warnings()
            
            # Trigger the warnings that should be suppressed
            warnings.warn(
                "You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy",
                UserWarning
            )
            warnings.warn(
                "As shared layers in the mlp_extractor are removed since SB3 v1.8.0",
                UserWarning
            )
            
            # These warnings should be filtered out
            relevant_warnings = [warning for warning in w if 'primarily intended' in str(warning.message) or 'mlp_extractor' in str(warning.message)]
            assert len(relevant_warnings) == 0, "Warnings should be suppressed"


class TestDeviceOptimizationIntegration:
    """Integration tests for device optimization."""
    
    def test_model_config_optimization(self):
        """Test that model configurations are optimized correctly."""
        optimizer = DeviceOptimizer()
        base_config = {'learning_rate': 0.001, 'device': 'auto'}
        
        optimized_config = optimizer.get_model_config_with_device(base_config, 'cnn')
        
        assert 'device' in optimized_config
        assert optimized_config['device'] in ['cpu', 'cuda']
        assert optimized_config['learning_rate'] == 0.001  # Other config preserved
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.backends.cudnn')
    def test_cuda_optimizations_applied(self, mock_cudnn, mock_empty_cache, mock_cuda_available):
        """Test that CUDA optimizations are applied when using GPU."""
        optimizer = DeviceOptimizer()
        optimizer.optimize_torch_settings('cuda')
        
        mock_empty_cache.assert_called_once()
        assert mock_cudnn.benchmark == True
        assert mock_cudnn.deterministic == False
    
    @patch('torch.set_num_threads')
    @patch('torch.set_num_interop_threads')
    def test_cpu_optimizations_applied(self, mock_interop_threads, mock_num_threads):
        """Test that CPU optimizations are applied when using CPU."""
        optimizer = DeviceOptimizer()
        optimizer.optimize_torch_settings('cpu')
        
        mock_num_threads.assert_called()
        mock_interop_threads.assert_called_with(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Comprehensive tests for the refactored feature extraction module

This test suite covers all aspects of the new modular feature extraction
architecture with proper mocking and edge case coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch

from src.ml.feature_extraction import (
    FeatureExtractor,
    FeatureExtractionError,
    DataValidationError,
    FeatureComputationError,
    ModelLoadError,
    FeatureExtractionConfig,
    CNNLSTMExtractor,
    CachedFeatureExtractor,
    FallbackFeatureExtractor,
    FeatureExtractorFactory,
    PerformanceTracker,
    PerformanceMetrics
)


class TestFeatureExtractionConfig:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = FeatureExtractionConfig()
        assert config.fused_feature_dim == 256
        assert config.cache_size == 1000
        assert config.enable_caching is True
    
    def test_invalid_config_validation(self):
        """Test configuration validation"""
        with pytest.raises(ValueError, match="fused_feature_dim must be positive"):
            FeatureExtractionConfig(fused_feature_dim=-1)
        
        with pytest.raises(ValueError, match="cache_size must be positive"):
            FeatureExtractionConfig(cache_size=0)


class TestPerformanceMetrics:
    """Test performance tracking"""
    
    def test_metrics_update(self):
        """Test metrics update functionality"""
        metrics = PerformanceMetrics()
        
        # Update with successful execution
        metrics.update(0.1, cache_hit=False, error=False)
        assert metrics.total_calls == 1
        assert metrics.avg_time == 0.1
        assert metrics.cache_misses == 1
        
        # Update with cache hit
        metrics.update(0.05, cache_hit=True, error=False)
        assert metrics.cache_hits == 1
        assert metrics.cache_hit_rate == 0.5
    
    def test_performance_tracker(self):
        """Test performance tracker"""
        tracker = PerformanceTracker(log_interval=2)
        
        tracker.track_execution(0.1, cache_hit=False)
        tracker.track_execution(0.2, cache_hit=True)
        
        summary = tracker.get_summary()
        assert summary['total_calls'] == 2
        assert summary['cache_hit_rate'] == 0.5


class TestCNNLSTMExtractor:
    """Test CNN+LSTM feature extractor"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock CNN+LSTM model"""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        # Mock forward method
        mock_outputs = {
            'fused_features': torch.randn(1, 10, 256),
            'cnn_features': torch.randn(1, 10, 128),
            'lstm_features': torch.randn(1, 10, 128),
            'classification_probs': torch.softmax(torch.randn(1, 3), dim=1),
            'regression_uncertainty': torch.randn(1, 1),
            'ensemble_weights': torch.randn(1, 4)
        }
        model.forward.return_value = mock_outputs
        return model
    
    def test_initialization(self, mock_model):
        """Test extractor initialization"""
        extractor = CNNLSTMExtractor(mock_model, device='cpu')
        assert extractor.device == 'cpu'
        mock_model.to.assert_called_with('cpu')
        mock_model.eval.assert_called_once()
    
    def test_feature_extraction(self, mock_model):
        """Test successful feature extraction"""
        extractor = CNNLSTMExtractor(mock_model, device='cpu')
        
        # Create test data
        test_data = np.random.randn(20, 5)  # 20 timesteps, 5 features
        
        features = extractor.extract_features(test_data)
        
        assert 'fused_features' in features
        assert 'cnn_features' in features
        assert 'lstm_features' in features
        assert features['fused_features'].shape[1] == 256
    
    def test_input_validation(self, mock_model):
        """Test input validation"""
        extractor = CNNLSTMExtractor(mock_model, device='cpu')
        
        # Test None input
        with pytest.raises(DataValidationError, match="cannot be None"):
            extractor.extract_features(None)
        
        # Test insufficient timesteps
        with pytest.raises(DataValidationError, match="Insufficient time steps"):
            extractor.extract_features(np.random.randn(5, 5))
    
    def test_model_error_handling(self, mock_model):
        """Test model error handling"""
        mock_model.forward.side_effect = RuntimeError("Model error")
        extractor = CNNLSTMExtractor(mock_model, device='cpu')
        
        test_data = np.random.randn(20, 5)
        
        with pytest.raises(FeatureComputationError, match="Model inference failed"):
            extractor.extract_features(test_data)


class TestCachedFeatureExtractor:
    """Test cached feature extractor"""
    
    @pytest.fixture
    def mock_base_extractor(self):
        """Create mock base extractor"""
        extractor = Mock(spec=FeatureExtractor)
        extractor.extract_features.return_value = {
            'fused_features': np.random.randn(1, 256)
        }
        extractor.get_feature_dimensions.return_value = {'fused_features': 256}
        return extractor
    
    def test_caching_functionality(self, mock_base_extractor):
        """Test caching behavior"""
        cached_extractor = CachedFeatureExtractor(
            mock_base_extractor, cache_size=10, ttl_seconds=60
        )
        
        test_data = np.random.randn(20, 5)
        
        # First call should hit base extractor
        result1 = cached_extractor.extract_features(test_data)
        assert mock_base_extractor.extract_features.call_count == 1
        
        # Second call with same data should use cache
        result2 = cached_extractor.extract_features(test_data)
        assert mock_base_extractor.extract_features.call_count == 1
        
        # Results should be identical
        np.testing.assert_array_equal(
            result1['fused_features'], result2['fused_features']
        )
    
    def test_cache_info(self, mock_base_extractor):
        """Test cache information"""
        cached_extractor = CachedFeatureExtractor(mock_base_extractor)
        
        info = cached_extractor.get_cache_info()
        assert 'cache_size' in info
        assert 'max_size' in info
        assert 'performance_summary' in info


class TestFallbackFeatureExtractor:
    """Test fallback feature extractor"""
    
    @pytest.fixture
    def mock_primary_extractor(self):
        """Create mock primary extractor"""
        extractor = Mock(spec=FeatureExtractor)
        return extractor
    
    def test_successful_primary_extraction(self, mock_primary_extractor):
        """Test successful primary extraction"""
        mock_primary_extractor.extract_features.return_value = {
            'fused_features': np.random.randn(1, 256)
        }
        
        fallback_extractor = FallbackFeatureExtractor(mock_primary_extractor)
        test_data = np.random.randn(20, 5)
        
        result = fallback_extractor.extract_features(test_data)
        
        assert 'fused_features' in result
        assert not fallback_extractor.is_fallback_active
        mock_primary_extractor.extract_features.assert_called_once()
    
    def test_fallback_activation(self, mock_primary_extractor):
        """Test fallback activation on primary failure"""
        mock_primary_extractor.extract_features.side_effect = Exception("Primary failed")
        
        fallback_extractor = FallbackFeatureExtractor(mock_primary_extractor)
        
        # Create OHLCV test data
        test_data = np.random.randn(20, 5)
        test_data[:, 1] = np.maximum(test_data[:, 1], test_data[:, 0])  # High >= Open
        test_data[:, 2] = np.minimum(test_data[:, 2], test_data[:, 0])  # Low <= Open
        
        result = fallback_extractor.extract_features(test_data)
        
        assert 'fused_features' in result
        assert fallback_extractor.is_fallback_active
        assert result['fused_features'].shape[1] == 15  # Basic indicators


class TestFeatureExtractorFactory:
    """Test feature extractor factory"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for factory tests"""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        return model
    
    def test_create_basic_extractor(self, mock_model):
        """Test basic extractor creation"""
        extractor = FeatureExtractorFactory.create_basic_extractor(mock_model)
        assert isinstance(extractor, CNNLSTMExtractor)
    
    def test_create_cached_extractor(self, mock_model):
        """Test cached extractor creation"""
        extractor = FeatureExtractorFactory.create_cached_extractor(mock_model)
        assert isinstance(extractor, CachedFeatureExtractor)
    
    def test_create_full_extractor(self, mock_model):
        """Test full extractor creation with config"""
        config = FeatureExtractionConfig(
            enable_caching=True,
            enable_fallback=True
        )
        
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Should be wrapped in fallback (outermost)
        assert isinstance(extractor, FallbackFeatureExtractor)
        
        # Primary should be cached extractor
        assert isinstance(extractor.primary_extractor, CachedFeatureExtractor)
        
        # Base should be CNN+LSTM extractor
        assert isinstance(
            extractor.primary_extractor.extractor, CNNLSTMExtractor
        )


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_feature_extraction(self):
        """Test complete feature extraction pipeline"""
        # Create mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        mock_outputs = {
            'fused_features': torch.randn(1, 10, 256),
            'classification_probs': torch.softmax(torch.randn(1, 3), dim=1),
            'regression_uncertainty': torch.randn(1, 1)
        }
        mock_model.forward.return_value = mock_outputs
        
        # Create full extractor
        config = FeatureExtractionConfig(
            enable_caching=True,
            enable_fallback=True,
            cache_size=100,
            ttl_seconds=30
        )
        
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # Test feature extraction
        test_data = np.random.randn(20, 5)
        features = extractor.extract_features(test_data)
        
        assert 'fused_features' in features
        assert features['fused_features'].shape[1] == 256
    
    def test_error_recovery_flow(self):
        """Test error recovery and fallback flow"""
        # Create failing mock model
        mock_model = Mock()
        mock_model.to.side_effect = Exception("Model load failed")
        
        # This should trigger fallback during factory creation
        config = FeatureExtractionConfig(enable_fallback=True)
        
        # Factory should handle model load failure gracefully
        with pytest.raises(ModelLoadError):
            FeatureExtractorFactory.create_extractor(mock_model, config)
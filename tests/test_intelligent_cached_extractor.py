"""Comprehensive tests for the intelligent cached feature extractor

This test suite covers all aspects of the intelligent caching system
with proper mocking and edge case coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock
import time

from src.ml.feature_extraction import (
    FeatureExtractor,
    FeatureExtractionError,
    FeatureExtractionConfig,
    IntelligentCachedFeatureExtractor,
    FeatureExtractorFactory,
    CacheTTLConfig
)


class TestIntelligentCachedFeatureExtractor:
    """Test intelligent cached feature extractor"""
    
    @pytest.fixture
    def mock_base_extractor(self):
        """Create mock base extractor"""
        extractor = Mock(spec=FeatureExtractor)
        extractor.extract_features.return_value = {
            'fused_features': np.random.randn(1, 256)
        }
        extractor.get_feature_dimensions.return_value = {'fused_features': 256}
        return extractor
    
    @pytest.fixture
    def mock_model(self):
        """Create mock CNN+LSTM model"""
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = None
        
        # Mock forward method
        mock_outputs = {
            'fused_features': np.random.randn(1, 10, 256),
            'cnn_features': np.random.randn(1, 10, 128),
            'lstm_features': np.random.randn(1, 10, 128),
            'classification_probs': np.random.rand(1, 3),
            'regression_uncertainty': np.random.randn(1, 1),
            'ensemble_weights': np.random.randn(1, 4)
        }
        model.forward.return_value = mock_outputs
        return model
    
    def test_initialization(self, mock_base_extractor):
        """Test extractor initialization"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        assert extractor.extractor == mock_base_extractor
        assert extractor.cache.maxsize == 1000
        assert extractor.cache.ttl == 60
        assert extractor.performance_tracker is not None
    
    def test_initialization_with_config(self, mock_base_extractor):
        """Test extractor initialization with custom config"""
        config = FeatureExtractionConfig(
            cache_size=500,
            cache_ttl_seconds=30
        )
        
        extractor = IntelligentCachedFeatureExtractor(
            mock_base_extractor,
            config=config
        )
        
        assert extractor.cache.maxsize == 500
        assert extractor.cache.ttl == 30
    
    def test_caching_functionality(self, mock_base_extractor):
        """Test caching behavior"""
        extractor = IntelligentCachedFeatureExtractor(
            mock_base_extractor,
            config=FeatureExtractionConfig(cache_size=10, cache_ttl_seconds=60)
        )
        
        test_data = np.random.randn(20, 5)
        
        # First call should hit base extractor
        result1 = extractor.extract_features(test_data)
        assert mock_base_extractor.extract_features.call_count == 1
        assert 'fused_features' in result1
        
        # Second call with same data should use cache
        result2 = extractor.extract_features(test_data)
        assert mock_base_extractor.extract_features.call_count == 1
        assert 'fused_features' in result2
        # Use both results in assertion
        np.testing.assert_array_equal(
            result1['fused_features'], result2['fused_features']
        )
        
        # Results should be identical
        np.testing.assert_array_equal(
            result1['fused_features'], result2['fused_features']
        )
    
    def test_cache_expiration(self, mock_base_extractor):
        """Test cache expiration"""
        extractor = IntelligentCachedFeatureExtractor(
            mock_base_extractor,
            config=FeatureExtractionConfig(cache_size=10, cache_ttl_seconds=1)
        )
        
        test_data = np.random.randn(20, 5)
        
        # First call
        result1 = extractor.extract_features(test_data)
        assert mock_base_extractor.extract_features.call_count == 1
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Second call should hit base extractor again
        result2 = extractor.extract_features(test_data)
        assert mock_base_extractor.extract_features.call_count == 2
    
    def test_adaptive_ttl_management(self, mock_base_extractor):
        """Test adaptive TTL management"""
        ttl_config = CacheTTLConfig(
            market_features_ttl=60,
            frequent_access_factor=0.5,
            infrequent_access_factor=2.0,
            warming_threshold=5
        )
        
        extractor = IntelligentCachedFeatureExtractor(
            mock_base_extractor,
            ttl_config=ttl_config
        )
        
        test_data = np.random.randn(20, 5)
        cache_key = extractor._generate_cache_key(test_data)
        
        # Record multiple accesses
        for i in range(10):
            extractor.adaptive_ttl_manager.record_access(cache_key)
        
        # Get adaptive TTL - should be reduced due to frequent access
        ttl = extractor.adaptive_ttl_manager.get_adaptive_ttl(cache_key)
        assert ttl < 60  # Should be less than base TTL
        
        # Check if cache should be warmed
        warmed = extractor.adaptive_ttl_manager.should_warm_cache(cache_key)
        assert warmed is True
    
    def test_cache_invalidation_pattern(self, mock_base_extractor):
        """Test cache invalidation by pattern"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        # Add some items to cache
        test_data1 = np.random.randn(20, 5)
        test_data2 = np.random.randn(20, 5)
        
        extractor.extract_features(test_data1)
        extractor.extract_features(test_data2)
        
        # Check cache size
        assert len(extractor.cache) == 2
        
        # Invalidate by pattern (this is a simple test since all keys
        # are random)
        # In a real scenario, we'd have more predictable keys
        count = extractor.invalidate_cache_pattern("nonexistent")
        assert count == 0
    
    def test_cache_invalidation_all(self, mock_base_extractor):
        """Test cache invalidation of all entries"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        # Add some items to cache
        test_data1 = np.random.randn(20, 5)
        test_data2 = np.random.randn(20, 5)
        
        extractor.extract_features(test_data1)
        extractor.extract_features(test_data2)
        
        # Check cache size
        assert len(extractor.cache) == 2
        
        # Invalidate all
        extractor.invalidate_cache_all()
        
        # Check cache is empty
        assert len(extractor.cache) == 0
    
    def test_cache_info(self, mock_base_extractor):
        """Test cache information"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        info = extractor.get_cache_info()
        assert 'cache_size' in info
        assert 'max_size' in info
        assert 'ttl' in info
        assert 'adaptive_ttl_stats' in info
        assert 'performance_summary' in info
    
    def test_clear_cache(self, mock_base_extractor):
        """Test cache clearing"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        # Add some items to cache
        test_data = np.random.randn(20, 5)
        extractor.extract_features(test_data)
        
        # Check cache is not empty
        assert len(extractor.cache) == 1
        
        # Clear cache
        extractor.clear_cache()
        
        # Check cache is empty
        assert len(extractor.cache) == 0
    
    def test_performance_tracking(self, mock_base_extractor):
        """Test performance tracking"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        test_data = np.random.randn(20, 5)
        
        # Extract features to generate performance data
        extractor.extract_features(test_data)
        extractor.extract_features(test_data)  # This should be a cache hit
        
        info = extractor.get_cache_info()
        assert info['performance_summary'] is not None
        assert 'total_extractions' in info['performance_summary']
        assert 'cache_hit_rate' in info['performance_summary']
    
    def test_error_handling(self, mock_base_extractor):
        """Test error handling"""
        mock_base_extractor.extract_features.side_effect = Exception(
            "Test error"
        )
        
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        test_data = np.random.randn(20, 5)
        
        with pytest.raises(
            FeatureExtractionError,
            match="Cached feature extraction failed"
        ):
            extractor.extract_features(test_data)
    
    def test_update_ttl_config(self, mock_base_extractor):
        """Test updating TTL configuration"""
        extractor = IntelligentCachedFeatureExtractor(mock_base_extractor)
        
        new_ttl_config = CacheTTLConfig(market_features_ttl=120)
        extractor.update_ttl_config(new_ttl_config)
        
        # The cache TTL doesn't change dynamically, but the config
        # should be updated
        assert extractor.ttl_config.market_features_ttl == 120


class TestIntegration:
    """Integration tests for the intelligent caching system"""
    
    def test_end_to_end_with_cnn_lstm(self):
        """Test complete feature extraction pipeline with CNN+LSTM"""
        # Create mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        mock_outputs = {
            'fused_features': np.random.randn(1, 10, 256),
            'classification_probs': np.random.rand(1, 3),
            'regression_uncertainty': np.random.randn(1, 1)
        }
        mock_model.forward.return_value = mock_outputs
        
        # Create intelligent cached extractor
        config = FeatureExtractionConfig(
            enable_caching=True,
            cache_size=100,
            cache_ttl_seconds=30
        )
        
        extractor = (
            FeatureExtractorFactory.create_intelligent_cached_extractor(
                mock_model,
                cache_size=config.cache_size,
                ttl_seconds=config.cache_ttl_seconds
            )
        )
        
        # Test feature extraction
        test_data = np.random.randn(20, 5)
        features = extractor.extract_features(test_data)
        
        assert 'fused_features' in features
        assert features['fused_features'].shape[1] == 256
    
    def test_factory_integration(self):
        """Test factory integration with intelligent caching"""
        # Create mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        mock_outputs = {
            'fused_features': np.random.randn(1, 10, 256),
            'classification_probs': np.random.rand(1, 3),
            'regression_uncertainty': np.random.randn(1, 1)
        }
        mock_model.forward.return_value = mock_outputs
        
        # Create full extractor with intelligent caching
        config = FeatureExtractionConfig(
            enable_caching=True,
            enable_fallback=True,
            cache_size=100,
            cache_ttl_seconds=30
        )
        
        extractor = FeatureExtractorFactory.create_extractor(
            mock_model, config
        )
        
        # Depending on the configuration, the outermost extractor might
        # be fallback but somewhere in the chain should be our
        # intelligent cached extractor
        
        # Test feature extraction
        test_data = np.random.randn(20, 5)
        features = extractor.extract_features(test_data)
        
        assert 'fused_features' in features
        assert features['fused_features'].shape[1] == 256


if __name__ == "__main__":
    pytest.main([__file__])
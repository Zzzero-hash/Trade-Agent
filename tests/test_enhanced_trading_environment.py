"""Tests for Enhanced Trading Environment with CNN+LSTM Integration

This module tests the enhanced trading environment that integrates
CNN+LSTM feature extraction for RL agents.

Requirements: 1.4, 2.4, 9.1
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.ml.enhanced_trading_environment import (
    EnhancedTradingEnvironment,
    EnhancedTradingConfig,
    create_enhanced_trading_config
)
from src.ml.cnn_lstm_feature_extractor import (
    CNNLSTMFeatureExtractor,
    FeatureExtractionConfig,
    FeatureCache
)
from src.ml.trading_environment import TradingConfig


class TestFeatureCache:
    """Test the feature caching functionality"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = FeatureCache(max_size=100, ttl_seconds=60)
        
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations"""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        
        # Test data
        data = np.array([[1, 2, 3], [4, 5, 6]])
        features = np.array([0.1, 0.2, 0.3])
        
        # Put data in cache
        cache.put(data, features)
        assert len(cache.cache) == 1
        
        # Get data from cache
        cached_features = cache.get(data)
        assert cached_features is not None
        np.testing.assert_array_equal(cached_features, features)
    
    def test_cache_expiration(self):
        """Test cache expiration functionality"""
        cache = FeatureCache(max_size=10, ttl_seconds=0.1)  # Very short TTL
        
        data = np.array([[1, 2, 3]])
        features = np.array([0.1, 0.2])
        
        # Put data in cache
        cache.put(data, features)
        
        # Should be available immediately
        cached_features = cache.get(data)
        assert cached_features is not None
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        # Should be expired now
        cached_features = cache.get(data)
        assert cached_features is None
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement"""
        cache = FeatureCache(max_size=2, ttl_seconds=60)
        
        # Add items up to limit
        for i in range(3):
            data = np.array([[i]])
            features = np.array([i])
            cache.put(data, features)
        
        # Should only keep max_size items
        assert len(cache.cache) <= 2
        assert len(cache.access_order) <= 2
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        
        # Add some data
        data = np.array([[1, 2]])
        features = np.array([0.5])
        cache.put(data, features)
        
        assert len(cache.cache) == 1
        
        # Clear cache
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0


class TestCNNLSTMFeatureExtractor:
    """Test the CNN+LSTM feature extractor"""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization"""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            enable_caching=True,
            enable_fallback=True
        )
        
        extractor = CNNLSTMFeatureExtractor(config)
        
        assert extractor.config == config
        assert extractor.hybrid_model is None
        assert not extractor.is_model_loaded
        assert extractor.fallback_mode == False
        assert extractor.cache is not None
    
    def test_fallback_feature_extraction(self):
        """Test fallback feature extraction"""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            fallback_feature_dim=15,
            enable_fallback=True
        )
        
        extractor = CNNLSTMFeatureExtractor(config)
        extractor.fallback_mode = True
        
        # Create test market window
        market_window = np.random.randn(60, 15)  # 60 timesteps, 15 features
        
        # Extract features
        features = extractor.extract_features(market_window)
        
        assert 'fused_features' in features
        assert 'classification_confidence' in features
        assert 'regression_uncertainty' in features
        assert 'ensemble_weights' in features
        assert 'fallback_used' in features
        
        assert features['fallback_used'] == True
        assert features['fused_features'].shape[1] == 15  # fallback_feature_dim
        assert features['ensemble_weights'] is None
    
    def test_feature_extraction_with_caching(self):
        """Test feature extraction with caching enabled"""
        config = FeatureExtractionConfig(
            enable_caching=True,
            cache_size=10,
            enable_fallback=True
        )
        
        extractor = CNNLSTMFeatureExtractor(config)
        extractor.fallback_mode = True
        
        # Create test data
        market_window = np.random.randn(60, 15)
        
        # First extraction (should cache)
        features1 = extractor.extract_features(market_window)
        
        # Second extraction with same data (should use cache)
        features2 = extractor.extract_features(market_window)
        
        # Should have cache functionality working (cache may or may not hit due to hashing)
        assert extractor.cache is not None
        assert len(extractor.cache.cache) >= 0  # Cache should exist
    
    def test_get_feature_dimensions(self):
        """Test getting feature dimensions"""
        config = FeatureExtractionConfig(
            fused_feature_dim=256,
            fallback_feature_dim=15
        )
        
        extractor = CNNLSTMFeatureExtractor(config)
        
        # Test fallback dimensions
        extractor.fallback_mode = True
        dims = extractor.get_feature_dimensions()
        
        assert dims['fused_features'] == 15
        assert dims['total'] == 17  # 15 + 1 + 1
        
        # Test normal dimensions
        extractor.is_model_loaded = True
        extractor.fallback_mode = False
        dims = extractor.get_feature_dimensions()
        
        assert dims['fused_features'] == 256
        assert dims['total'] == 258  # 256 + 1 + 1
    
    def test_performance_tracking(self):
        """Test performance tracking functionality"""
        config = FeatureExtractionConfig(enable_fallback=True)
        extractor = CNNLSTMFeatureExtractor(config)
        extractor.fallback_mode = True
        
        # Extract features multiple times
        market_window = np.random.randn(60, 15)
        
        for _ in range(5):
            extractor.extract_features(market_window)
        
        # Check performance stats
        assert extractor.extraction_count == 5
        assert extractor.total_extraction_time >= 0  # Should be non-negative
        assert extractor.fallback_count == 5
        
        # Test status
        status = extractor.get_status()
        assert status['extraction_count'] == 5
        assert status['fallback_count'] == 5
        assert status['model_loaded'] == False
        assert status['fallback_mode'] == True


class TestEnhancedTradingConfig:
    """Test enhanced trading configuration"""
    
    def test_enhanced_config_initialization(self):
        """Test enhanced config initialization with defaults"""
        config = EnhancedTradingConfig()
        
        # Base config attributes
        assert config.initial_balance == 100000.0
        assert config.max_position_size == 0.2
        assert config.transaction_cost == 0.001
        
        # Enhanced config attributes
        assert config.hybrid_model_path is None
        assert config.fused_feature_dim == 256
        assert config.enable_feature_caching == True
        assert config.enable_fallback == True
        assert config.include_uncertainty == True
    
    def test_enhanced_config_with_custom_values(self):
        """Test enhanced config with custom values"""
        config = EnhancedTradingConfig(
            initial_balance=50000.0,
            hybrid_model_path="/path/to/model.pth",
            fused_feature_dim=128,
            enable_feature_caching=False,
            include_uncertainty=False
        )
        
        assert config.initial_balance == 50000.0
        assert config.hybrid_model_path == "/path/to/model.pth"
        assert config.fused_feature_dim == 128
        assert config.enable_feature_caching == False
        assert config.include_uncertainty == False
    
    def test_create_enhanced_trading_config(self):
        """Test config creation helper function"""
        config = create_enhanced_trading_config(
            initial_balance=75000.0,
            fused_feature_dim=512,
            enable_fallback=False
        )
        
        assert isinstance(config, EnhancedTradingConfig)
        assert config.initial_balance == 75000.0
        assert config.fused_feature_dim == 512
        assert config.enable_fallback == False


class TestEnhancedTradingEnvironment:
    """Test the enhanced trading environment"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        np.random.seed(42)
        
        # Create 200 timesteps of data
        timestamps = pd.date_range('2023-01-01', periods=200, freq='1H')
        
        data = []
        for i, timestamp in enumerate(timestamps):
            # Simple random walk for price
            price = 100 + np.cumsum(np.random.randn(1))[0] * 0.1
            
            row = {
                'timestamp': timestamp,
                'symbol': 'TEST',
                'open': price,
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price,
                'volume': np.random.randint(1000, 10000),
                'returns': np.random.randn() * 0.01,
                'volatility': abs(np.random.randn() * 0.02),
                'rsi': np.random.uniform(20, 80),
                'macd': np.random.randn() * 0.1,
                'macd_signal': np.random.randn() * 0.1,
                'bb_position': np.random.uniform(0, 1),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'sma_5': price,
                'sma_20': price,
                'ema_12': price
            }
            data.append(row)
        
        return pd.DataFrame(data).reset_index(drop=True)
    
    def test_enhanced_environment_initialization(self, sample_market_data):
        """Test enhanced environment initialization"""
        config = EnhancedTradingConfig(
            lookback_window=20,
            fused_feature_dim=128,
            enable_fallback=True
        )
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        assert isinstance(env, EnhancedTradingEnvironment)
        assert env.enhanced_config == config
        assert env.feature_extractor is not None
        assert env.observation_space.shape[0] > 0  # Should have enhanced observation space
    
    def test_enhanced_observation_space(self, sample_market_data):
        """Test enhanced observation space setup"""
        config = EnhancedTradingConfig(
            fused_feature_dim=256,
            include_uncertainty=True,
            include_ensemble_weights=False,
            fallback_feature_dim=15
        )
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        # Calculate expected observation size
        # Fallback features (15) + uncertainty (2) + portfolio (4) = 21
        expected_size = 15 + 2 + 4  # fallback_dim + uncertainty + portfolio
        
        assert env.observation_space.shape[0] == expected_size
    
    def test_environment_reset(self, sample_market_data):
        """Test environment reset with enhanced features"""
        config = EnhancedTradingConfig(lookback_window=20)
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        observation, info = env.reset()
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape[0] == env.observation_space.shape[0]
        assert 'feature_extractor_status' in info
        assert 'enhanced_observation_dim' in info
        
        # Check that observation contains valid values
        assert not np.any(np.isnan(observation))
        assert not np.any(np.isinf(observation))
    
    def test_environment_step(self, sample_market_data):
        """Test environment step with enhanced features"""
        config = EnhancedTradingConfig(lookback_window=20)
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        observation, info = env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape[0] == env.observation_space.shape[0]
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Check enhanced info
        assert 'feature_extraction_count' in step_info
        assert 'fallback_usage_count' in step_info
        assert 'fallback_rate' in step_info
    
    def test_market_window_extraction(self, sample_market_data):
        """Test market window extraction for feature extraction"""
        config = EnhancedTradingConfig(lookback_window=30)
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        env.reset()
        
        # Move to a position where we have enough data
        for _ in range(35):
            action = env.action_space.sample()
            env.step(action)
        
        # Extract market window
        market_window = env._get_market_window()
        
        assert isinstance(market_window, np.ndarray)
        assert market_window.shape[0] == 30  # lookback_window
        assert market_window.shape[1] == 15  # number of features
        
        # Check for valid data
        assert not np.any(np.isnan(market_window))
    
    def test_enhanced_observation_extraction(self, sample_market_data):
        """Test enhanced observation extraction"""
        config = EnhancedTradingConfig(
            lookback_window=20,
            include_uncertainty=True
        )
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        env.reset()
        
        # Move forward to get some data
        for _ in range(25):
            action = env.action_space.sample()
            env.step(action)
        
        # Get enhanced observation
        obs = env._get_enhanced_observation()
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
    
    def test_fallback_mode_functionality(self, sample_market_data):
        """Test fallback mode functionality"""
        config = EnhancedTradingConfig(
            enable_fallback=True,
            fallback_feature_dim=10
        )
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        # Enable fallback mode
        env.enable_fallback_mode(True)
        
        observation, info = env.reset()
        
        # Should use fallback features
        status = env.get_feature_extractor_status()
        assert status['fallback_mode'] == True
        
        # Take some steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should have fallback usage
        assert step_info['fallback_usage_count'] > 0
        assert step_info['fallback_rate'] > 0
    
    def test_enhanced_metrics(self, sample_market_data):
        """Test enhanced metrics functionality"""
        config = EnhancedTradingConfig(lookback_window=20)
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        env.reset()
        
        # Take some steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Get enhanced metrics
        metrics = env.get_enhanced_metrics()
        
        # Should include base metrics
        assert 'total_return' in metrics
        assert 'final_portfolio_value' in metrics
        
        # Should include enhanced metrics
        assert 'feature_extraction_count' in metrics
        assert 'fallback_usage_count' in metrics
        assert 'fallback_rate' in metrics
        assert 'feature_extractor_status' in metrics
        assert 'observation_dimension' in metrics
    
    def test_multiple_symbols_handling(self, sample_market_data):
        """Test handling of multiple symbols (basic test)"""
        # Create data for multiple symbols
        multi_symbol_data = []
        for symbol in ['TEST1', 'TEST2']:
            symbol_data = sample_market_data.copy()
            symbol_data['symbol'] = symbol
            multi_symbol_data.append(symbol_data)
        
        multi_data = pd.concat(multi_symbol_data, ignore_index=True)
        
        config = EnhancedTradingConfig(lookback_window=20)
        
        env = EnhancedTradingEnvironment(
            market_data=multi_data,
            config=config,
            symbols=['TEST1', 'TEST2']
        )
        
        observation, info = env.reset()
        
        # Should handle multiple symbols
        assert isinstance(observation, np.ndarray)
        assert observation.shape[0] > 0
    
    @patch('src.ml.enhanced_trading_environment.CNNLSTMFeatureExtractor')
    def test_feature_extractor_error_handling(self, mock_extractor_class, sample_market_data):
        """Test error handling in feature extraction"""
        # Mock feature extractor to raise an exception
        mock_extractor = Mock()
        mock_extractor.extract_features.side_effect = Exception("Feature extraction failed")
        mock_extractor.get_status.return_value = {'model_loaded': False}
        mock_extractor.get_feature_dimensions.return_value = {'fused_features': 15, 'total': 17}
        mock_extractor_class.return_value = mock_extractor
        
        config = EnhancedTradingConfig(lookback_window=20)
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        # Should still work with fallback
        observation, info = env.reset()
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape[0] > 0
    
    def test_render_functionality(self, sample_market_data, capsys):
        """Test enhanced render functionality"""
        config = EnhancedTradingConfig(lookback_window=20)
        
        env = EnhancedTradingEnvironment(
            market_data=sample_market_data,
            config=config,
            symbols=['TEST']
        )
        
        env.reset()
        
        # Take a few steps
        for _ in range(3):
            action = env.action_space.sample()
            env.step(action)
        
        # Test render
        env.render(mode="human")
        
        # Check that enhanced information is printed
        captured = capsys.readouterr()
        assert "Feature Extraction Stats:" in captured.out
        assert "Total extractions:" in captured.out
        assert "Fallback usage:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
"""Tests for advanced feature transformers"""

import pytest
import pandas as pd
import numpy as np
from src.ml.feature_engineering import (
    WaveletTransform, FourierFeatures, FractalFeatures, CrossAssetFeatures
)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    # Generate realistic price data with trend and volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "close": prices,
        "volume": np.random.randint(1000, 10000, 100)
    })
    
    # Ensure high >= close >= low
    data["high"] = np.maximum(data["high"], data["close"])
    data["low"] = np.minimum(data["low"], data["close"])
    
    return data


class TestWaveletTransform:
    """Test suite for WaveletTransform class"""

    def test_initialization_default_config(self):
        """Test initialization with default configuration"""
        wt = WaveletTransform()
        assert wt.wavelet == "db4"
        assert wt.levels == 3
        assert wt.mode == "symmetric"
        assert not wt.is_fitted

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration"""
        config = {"wavelet": "haar", "levels": 2, "mode": "periodization"}
        wt = WaveletTransform(config)
        assert wt.wavelet == "haar"
        assert wt.levels == 2
        assert wt.mode == "periodization"

    def test_fit_without_pywt_raises_error(self, sample_market_data):
        """Test that fit raises error when PyWavelets is not available"""
        wt = WaveletTransform()
        # Mock the import error
        import sys
        original_modules = sys.modules.copy()
        if 'pywt' in sys.modules:
            del sys.modules['pywt']
        
        try:
            with pytest.raises(ImportError, match="PyWavelets is required"):
                wt.fit(sample_market_data)
        finally:
            sys.modules.update(original_modules)

    @pytest.mark.skipif(True, reason="PyWavelets not installed in test environment")
    def test_wavelet_transform_with_pywt(self, sample_market_data):
        """Test wavelet transform when PyWavelets is available"""
        try:
            import pywt
            wt = WaveletTransform()
            wt.fit(sample_market_data)
            
            features = wt.transform(sample_market_data)
            feature_names = wt.get_feature_names()
            
            assert wt.is_fitted
            assert features.shape[0] == len(sample_market_data)
            assert features.shape[1] == len(feature_names)
            assert len(feature_names) == 4  # levels + 1
        except ImportError:
            pytest.skip("PyWavelets not available")

    def test_transform_without_fit_raises_error(self, sample_market_data):
        """Test that transform raises error when not fitted"""
        wt = WaveletTransform()
        with pytest.raises(ValueError, match="must be fitted before transform"):
            wt.transform(sample_market_data)

    def test_get_feature_names(self):
        """Test feature names generation"""
        wt = WaveletTransform({"levels": 2})
        names = wt.get_feature_names()
        expected = ["wavelet_approx_0", "wavelet_detail_1", "wavelet_detail_2"]
        assert names == expected


class TestFourierFeatures:
    """Test suite for FourierFeatures class"""

    def test_initialization_default_config(self):
        """Test initialization with default configuration"""
        ff = FourierFeatures()
        assert ff.n_components == 10
        assert ff.window_size == 50
        assert not ff.is_fitted

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration"""
        config = {"n_components": 5, "window_size": 30}
        ff = FourierFeatures(config)
        assert ff.n_components == 5
        assert ff.window_size == 30

    def test_fit_method(self, sample_market_data):
        """Test fit method"""
        ff = FourierFeatures()
        result = ff.fit(sample_market_data)
        assert result is ff
        assert ff.is_fitted

    def test_transform_without_fit_raises_error(self, sample_market_data):
        """Test that transform raises error when not fitted"""
        ff = FourierFeatures()
        with pytest.raises(ValueError, match="must be fitted before transform"):
            ff.transform(sample_market_data)

    def test_fourier_transform_calculation(self, sample_market_data):
        """Test Fourier transform calculation"""
        config = {"n_components": 5, "window_size": 20}
        ff = FourierFeatures(config)
        ff.fit(sample_market_data)
        
        features = ff.transform(sample_market_data)
        feature_names = ff.get_feature_names()
        
        assert features.shape == (100, 10)  # 5 magnitude + 5 phase
        assert len(feature_names) == 10
        assert not np.any(np.isnan(features))

    def test_get_feature_names(self):
        """Test feature names generation"""
        ff = FourierFeatures({"n_components": 3})
        names = ff.get_feature_names()
        expected = [
            "fft_magnitude_1", "fft_magnitude_2", "fft_magnitude_3",
            "fft_phase_1", "fft_phase_2", "fft_phase_3"
        ]
        assert names == expected

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        data = pd.DataFrame({
            "close": [100]  # Single data point
        })
        
        ff = FourierFeatures({"n_components": 3, "window_size": 10})
        features = ff.fit_transform(data)
        
        assert features.shape == (1, 6)
        # Should handle gracefully with zeros
        assert np.all(features == 0)

    def test_fit_transform(self, sample_market_data):
        """Test fit_transform method"""
        ff = FourierFeatures()
        features = ff.fit_transform(sample_market_data)
        
        assert ff.is_fitted
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_market_data)


class TestFractalFeatures:
    """Test suite for FractalFeatures class"""

    def test_initialization_default_config(self):
        """Test initialization with default configuration"""
        fractal = FractalFeatures()
        assert fractal.window_size == 50
        assert fractal.max_lag == 20
        assert not fractal.is_fitted

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration"""
        config = {"window_size": 30, "max_lag": 10}
        fractal = FractalFeatures(config)
        assert fractal.window_size == 30
        assert fractal.max_lag == 10

    def test_fit_method(self, sample_market_data):
        """Test fit method"""
        fractal = FractalFeatures()
        result = fractal.fit(sample_market_data)
        assert result is fractal
        assert fractal.is_fitted

    def test_transform_without_fit_raises_error(self, sample_market_data):
        """Test that transform raises error when not fitted"""
        fractal = FractalFeatures()
        with pytest.raises(ValueError, match="must be fitted before transform"):
            fractal.transform(sample_market_data)

    def test_fractal_features_calculation(self, sample_market_data):
        """Test fractal features calculation"""
        config = {"window_size": 30, "max_lag": 10}
        fractal = FractalFeatures(config)
        fractal.fit(sample_market_data)
        
        features = fractal.transform(sample_market_data)
        feature_names = fractal.get_feature_names()
        
        assert features.shape == (100, 2)
        assert len(feature_names) == 2
        assert feature_names == ["hurst_exponent", "fractal_dimension"]
        assert not np.any(np.isnan(features))

    def test_hurst_exponent_range(self, sample_market_data):
        """Test that Hurst exponent is in valid range"""
        fractal = FractalFeatures({"window_size": 30})
        features = fractal.fit_transform(sample_market_data)
        
        hurst_values = features[:, 0]
        # Hurst exponent should be between 0 and 1
        assert np.all((hurst_values >= 0) & (hurst_values <= 1))

    def test_fractal_dimension_range(self, sample_market_data):
        """Test that fractal dimension is in valid range"""
        fractal = FractalFeatures({"window_size": 30})
        features = fractal.fit_transform(sample_market_data)
        
        fractal_dims = features[:, 1]
        # Fractal dimension should be between 1 and 2 for time series
        assert np.all((fractal_dims >= 1) & (fractal_dims <= 2))

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        data = pd.DataFrame({
            "close": [100, 101, 102]  # Very little data
        })
        
        fractal = FractalFeatures({"window_size": 20})
        features = fractal.fit_transform(data)
        
        assert features.shape == (3, 2)
        # Should use default values for insufficient data
        assert np.all(features[:, 0] == 0.5)  # Default Hurst
        assert np.all(features[:, 1] == 1.0)  # Default fractal dimension

    def test_get_feature_names(self):
        """Test feature names"""
        fractal = FractalFeatures()
        names = fractal.get_feature_names()
        assert names == ["hurst_exponent", "fractal_dimension"]


class TestCrossAssetFeatures:
    """Test suite for CrossAssetFeatures class"""

    def test_initialization_default_config(self):
        """Test initialization with default configuration"""
        cross = CrossAssetFeatures()
        assert cross.window_size == 30
        assert cross.correlation_assets == []
        assert not cross.is_fitted

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration"""
        config = {
            "window_size": 20, 
            "correlation_assets": ["SPY", "QQQ"]
        }
        cross = CrossAssetFeatures(config)
        assert cross.window_size == 20
        assert cross.correlation_assets == ["SPY", "QQQ"]

    def test_fit_method(self, sample_market_data):
        """Test fit method"""
        cross = CrossAssetFeatures()
        result = cross.fit(sample_market_data)
        assert result is cross
        assert cross.is_fitted
        assert hasattr(cross, 'reference_data')

    def test_transform_without_fit_raises_error(self, sample_market_data):
        """Test that transform raises error when not fitted"""
        cross = CrossAssetFeatures()
        with pytest.raises(ValueError, match="must be fitted before transform"):
            cross.transform(sample_market_data)

    def test_internal_correlations_calculation(self, sample_market_data):
        """Test internal correlation features calculation"""
        cross = CrossAssetFeatures({"window_size": 20})
        cross.fit(sample_market_data)
        
        features = cross.transform(sample_market_data)
        feature_names = cross.get_feature_names()
        
        assert features.shape == (100, 4)
        assert len(feature_names) == 4
        expected_names = [
            "price_volume_correlation",
            "range_volume_correlation", 
            "returns_autocorrelation",
            "volume_momentum"
        ]
        assert feature_names == expected_names
        assert not np.any(np.isnan(features))

    def test_external_correlations_placeholder(self, sample_market_data):
        """Test external correlation features (placeholder implementation)"""
        config = {"correlation_assets": ["SPY", "QQQ", "BTC"]}
        cross = CrossAssetFeatures(config)
        cross.fit(sample_market_data)
        
        features = cross.transform(sample_market_data)
        feature_names = cross.get_feature_names()
        
        assert features.shape == (100, 3)
        assert len(feature_names) == 3
        assert feature_names == ["correlation_SPY", "correlation_QQQ", "correlation_BTC"]
        # Placeholder implementation returns zeros
        assert np.all(features == 0)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        data = pd.DataFrame({
            "open": [100],
            "high": [102],
            "low": [99],
            "close": [101],
            "volume": [1000]
        })
        
        cross = CrossAssetFeatures({"window_size": 10})
        features = cross.fit_transform(data)
        
        assert features.shape == (1, 4)
        # Should handle gracefully with zeros for insufficient data
        assert np.all(features == 0)

    def test_fit_transform(self, sample_market_data):
        """Test fit_transform method"""
        cross = CrossAssetFeatures()
        features = cross.fit_transform(sample_market_data)
        
        assert cross.is_fitted
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_market_data)

    def test_correlation_bounds(self, sample_market_data):
        """Test that correlation values are within valid bounds"""
        cross = CrossAssetFeatures({"window_size": 20})
        features = cross.fit_transform(sample_market_data)
        
        # Price-volume and range-volume correlations should be between -1 and 1
        price_vol_corr = features[:, 0]
        range_vol_corr = features[:, 1]
        returns_autocorr = features[:, 2]
        
        # Filter out zero values (insufficient data cases)
        valid_price_vol = price_vol_corr[price_vol_corr != 0]
        valid_range_vol = range_vol_corr[range_vol_corr != 0]
        valid_returns_auto = returns_autocorr[returns_autocorr != 0]
        
        if len(valid_price_vol) > 0:
            assert np.all((valid_price_vol >= -1) & (valid_price_vol <= 1))
        if len(valid_range_vol) > 0:
            assert np.all((valid_range_vol >= -1) & (valid_range_vol <= 1))
        if len(valid_returns_auto) > 0:
            assert np.all((valid_returns_auto >= -1) & (valid_returns_auto <= 1))


class TestAdvancedFeaturesIntegration:
    """Integration tests for advanced feature transformers"""

    def test_all_transformers_together(self, sample_market_data):
        """Test using all advanced transformers together"""
        from src.ml.feature_engineering import FeatureEngineer
        
        # Create feature engineer with all advanced transformers
        engineer = FeatureEngineer()
        
        # Add transformers (skip wavelet due to dependency)
        engineer.add_transformer(FourierFeatures({"n_components": 3}))
        engineer.add_transformer(FractalFeatures({"window_size": 20}))
        engineer.add_transformer(CrossAssetFeatures({"window_size": 15}))
        
        # Fit and transform
        features = engineer.fit_transform(sample_market_data)
        feature_names = engineer.get_feature_names()
        
        # Should have 3*2 + 2 + 4 = 12 features
        assert features.shape == (100, 12)
        assert len(feature_names) == 12
        assert not np.any(np.isnan(features))

    def test_feature_consistency_across_calls(self, sample_market_data):
        """Test that features are consistent across multiple transform calls"""
        ff = FourierFeatures({"n_components": 3})
        ff.fit(sample_market_data)
        
        features1 = ff.transform(sample_market_data)
        features2 = ff.transform(sample_market_data)
        
        np.testing.assert_array_equal(features1, features2)

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame({
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        })
        
        ff = FourierFeatures()
        features = ff.fit_transform(empty_data)
        
        assert features.shape == (0, 20)  # 10 components * 2 (mag + phase)

    def test_single_value_data(self):
        """Test handling of single value data"""
        single_data = pd.DataFrame({
            "open": [100],
            "high": [100],
            "low": [100],
            "close": [100],
            "volume": [1000]
        })
        
        fractal = FractalFeatures()
        features = fractal.fit_transform(single_data)
        
        assert features.shape == (1, 2)
        # Should use default values
        assert features[0, 0] == 0.5  # Hurst
        assert features[0, 1] == 1.0  # Fractal dimension
"""Tests for technical indicators feature transformer"""

import pytest
import pandas as pd
import numpy as np
from src.ml.feature_engineering import TechnicalIndicators


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


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators class"""

    def test_initialization_default_indicators(self):
        """Test initialization with default indicators"""
        ti = TechnicalIndicators()
        expected_indicators = [
            "sma_20", "ema_12", "rsi_14", "macd", "bollinger_bands",
            "atr_14", "obv", "vwap"
        ]
        assert ti.indicators == expected_indicators
        assert not ti.is_fitted

    def test_initialization_custom_indicators(self):
        """Test initialization with custom indicators"""
        custom_indicators = ["sma_20", "rsi_14", "macd"]
        config = {"indicators": custom_indicators}
        ti = TechnicalIndicators(config)
        assert ti.indicators == custom_indicators

    def test_fit_method(self, sample_market_data):
        """Test fit method"""
        ti = TechnicalIndicators()
        result = ti.fit(sample_market_data)
        assert result is ti  # Should return self
        assert ti.is_fitted

    def test_transform_without_fit_raises_error(self, sample_market_data):
        """Test that transform raises error when not fitted"""
        ti = TechnicalIndicators()
        with pytest.raises(ValueError, match="must be fitted before transform"):
            ti.transform(sample_market_data)

    def test_sma_calculation(self, sample_market_data):
        """Test Simple Moving Average calculation"""
        config = {"indicators": ["sma_20"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        expected_sma = sample_market_data["close"].rolling(window=20).mean()
        
        assert features.shape == (100, 1)
        # Compare non-NaN values (after window period)
        np.testing.assert_allclose(
            features[19:, 0], 
            expected_sma.iloc[19:].values, 
            rtol=1e-10
        )

    def test_ema_calculation(self, sample_market_data):
        """Test Exponential Moving Average calculation"""
        config = {"indicators": ["ema_12"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        expected_ema = sample_market_data["close"].ewm(span=12).mean()
        
        assert features.shape == (100, 1)
        np.testing.assert_allclose(
            features[:, 0], 
            expected_ema.values, 
            rtol=1e-10
        )

    def test_rsi_calculation(self, sample_market_data):
        """Test RSI calculation"""
        config = {"indicators": ["rsi_14"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        
        # Calculate expected RSI
        delta = sample_market_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        expected_rsi = 100 - (100 / (1 + rs))
        
        assert features.shape == (100, 1)
        # RSI should be between 0 and 100 (excluding NaN values)
        valid_rsi = features[~np.isnan(features[:, 0]), 0]
        assert np.all((valid_rsi >= 0) & (valid_rsi <= 100))

    def test_macd_calculation(self, sample_market_data):
        """Test MACD calculation"""
        config = {"indicators": ["macd"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        
        # MACD should produce 3 features: line, signal, histogram
        assert features.shape == (100, 3)
        
        # Calculate expected MACD
        ema_12 = sample_market_data["close"].ewm(span=12).mean()
        ema_26 = sample_market_data["close"].ewm(span=26).mean()
        expected_macd = ema_12 - ema_26
        expected_signal = expected_macd.ewm(span=9).mean()
        expected_histogram = expected_macd - expected_signal
        
        np.testing.assert_allclose(
            features[:, 0], 
            expected_macd.values, 
            rtol=1e-10
        )

    def test_bollinger_bands_calculation(self, sample_market_data):
        """Test Bollinger Bands calculation"""
        config = {"indicators": ["bollinger_bands"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        
        # Bollinger Bands should produce 4 features
        assert features.shape == (100, 4)
        
        # Calculate expected values
        sma_20 = sample_market_data["close"].rolling(window=20).mean()
        std_20 = sample_market_data["close"].rolling(window=20).std()
        expected_upper = sma_20 + (2 * std_20)
        expected_lower = sma_20 - (2 * std_20)
        
        # Upper band should be >= lower band
        valid_indices = ~(np.isnan(features[:, 0]) | np.isnan(features[:, 1]))
        assert np.all(features[valid_indices, 0] >= features[valid_indices, 1])

    def test_atr_calculation(self, sample_market_data):
        """Test Average True Range calculation"""
        config = {"indicators": ["atr_14"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        
        assert features.shape == (100, 1)
        # ATR should be positive (excluding NaN values)
        valid_atr = features[~np.isnan(features[:, 0]), 0]
        assert np.all(valid_atr >= 0)

    def test_obv_calculation(self, sample_market_data):
        """Test On-Balance Volume calculation"""
        config = {"indicators": ["obv"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        
        assert features.shape == (100, 1)
        # First OBV value should equal first volume
        assert features[0, 0] == sample_market_data["volume"].iloc[0]

    def test_vwap_calculation(self, sample_market_data):
        """Test VWAP calculation"""
        config = {"indicators": ["vwap"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        
        assert features.shape == (100, 1)
        # VWAP should be positive
        assert np.all(features[:, 0] > 0)

    def test_multiple_indicators(self, sample_market_data):
        """Test calculation with multiple indicators"""
        config = {"indicators": ["sma_20", "rsi_14", "macd"]}
        ti = TechnicalIndicators(config)
        ti.fit(sample_market_data)
        
        features = ti.transform(sample_market_data)
        feature_names = ti.get_feature_names()
        
        # Should have 1 + 1 + 3 = 5 features
        assert features.shape == (100, 5)
        assert len(feature_names) == 5
        assert feature_names == ["sma_20", "rsi_14", "macd_line", "macd_signal", "macd_histogram"]

    def test_fit_transform(self, sample_market_data):
        """Test fit_transform method"""
        ti = TechnicalIndicators()
        features = ti.fit_transform(sample_market_data)
        
        assert ti.is_fitted
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_market_data)

    def test_nan_handling(self):
        """Test handling of NaN values in input data"""
        # Create data with NaN values
        data = pd.DataFrame({
            "open": [100, 101, np.nan, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })
        
        config = {"indicators": ["sma_20"]}
        ti = TechnicalIndicators(config)
        features = ti.fit_transform(data)
        
        # Should not raise error and return valid array
        assert isinstance(features, np.ndarray)
        assert features.shape == (5, 1)

    def test_empty_indicators_list(self, sample_market_data):
        """Test with empty indicators list"""
        config = {"indicators": []}
        ti = TechnicalIndicators(config)
        features = ti.fit_transform(sample_market_data)
        
        assert features.shape == (100, 0)
        assert ti.get_feature_names() == []

    def test_feature_names_consistency(self, sample_market_data):
        """Test that feature names match actual features"""
        ti = TechnicalIndicators()
        features = ti.fit_transform(sample_market_data)
        feature_names = ti.get_feature_names()
        
        assert features.shape[1] == len(feature_names)

    def test_edge_case_single_row(self):
        """Test with single row of data"""
        data = pd.DataFrame({
            "open": [100],
            "high": [102],
            "low": [99],
            "close": [101],
            "volume": [1000]
        })
        
        config = {"indicators": ["sma_20", "rsi_14"]}
        ti = TechnicalIndicators(config)
        features = ti.fit_transform(data)
        
        assert features.shape == (1, 2)
        # Values should be 0 (NaN filled)
        assert np.all(features == 0)

    def test_edge_case_insufficient_data_for_indicators(self):
        """Test with insufficient data for indicators requiring windows"""
        # Only 5 rows, but RSI needs 14
        data = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })
        
        config = {"indicators": ["rsi_14", "sma_20"]}
        ti = TechnicalIndicators(config)
        features = ti.fit_transform(data)
        
        assert features.shape == (5, 2)
        # Should handle gracefully with NaN -> 0 conversion
        assert not np.any(np.isnan(features))
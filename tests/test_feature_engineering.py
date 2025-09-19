"""
Tests for the comprehensive feature engineering framework.

This module tests all components of the feature engineering system:
- Technical indicators
- Microstructure features  
- Regime detection
- Alternative data integration
- Comprehensive feature engine
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

from data.features import (
    TechnicalIndicatorEngine,
    MicrostructureEngine,
    RegimeDetectionEngine,
    AlternativeDataEngine,
    ComprehensiveFeatureEngine,
    FeatureEngineConfig,
    generate_comprehensive_features
)

warnings.filterwarnings('ignore')


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate realistic price data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, n_days))
    data['open'].iloc[0] = initial_price
    
    # Generate high/low with realistic spreads
    daily_range = np.abs(np.random.normal(0, 0.01, n_days))
    data['high'] = data['close'] * (1 + daily_range * 0.6)
    data['low'] = data['close'] * (1 - daily_range * 0.6)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Generate volume
    data['volume'] = np.random.lognormal(13, 0.5, n_days)  # Realistic volume distribution
    
    return data


@pytest.fixture
def sample_microstructure_data(sample_market_data):
    """Add microstructure data to market data."""
    data = sample_market_data.copy()
    
    # Add bid-ask spread data
    spread_bps = np.random.uniform(2, 15, len(data))
    data['bid_price'] = data['close'] * (1 - spread_bps / 20000)
    data['ask_price'] = data['close'] * (1 + spread_bps / 20000)
    data['bid_size'] = np.random.uniform(100, 2000, len(data))
    data['ask_size'] = np.random.uniform(100, 2000, len(data))
    
    return data


@pytest.fixture
def sample_alternative_data(sample_market_data):
    """Add alternative data to market data."""
    data = sample_market_data.copy()
    
    # Add sentiment and news data
    data['news_sentiment'] = np.random.normal(0, 0.3, len(data))
    data['social_sentiment'] = np.random.normal(0, 0.4, len(data))
    data['news_count'] = np.random.poisson(3, len(data))
    
    # Add macro indicators
    data['vix'] = 15 + 10 * np.abs(np.random.normal(0, 1, len(data)))
    data['dxy'] = 100 + np.random.normal(0, 2, len(data))
    data['yield_10y'] = 3.0 + np.random.normal(0, 0.5, len(data))
    
    return data


class TestTechnicalIndicators:
    """Test technical indicator generation."""
    
    def test_technical_indicator_engine_initialization(self):
        """Test that technical indicator engine initializes correctly."""
        engine = TechnicalIndicatorEngine()
        assert engine is not None
        assert hasattr(engine, 'indicators')
        assert len(engine.indicators) > 0
    
    def test_momentum_indicators(self, sample_market_data):
        """Test momentum indicator calculation."""
        engine = TechnicalIndicatorEngine()
        features = engine.calculate_subset(sample_market_data, ['momentum'])
        
        assert len(features.columns) > 0
        assert not features.empty
        
        # Check for specific momentum indicators
        momentum_cols = [col for col in features.columns if 'momentum' in col]
        assert len(momentum_cols) > 0
        
        # Check for RSI
        rsi_cols = [col for col in features.columns if 'rsi' in col]
        assert len(rsi_cols) > 0
        
        # Validate RSI values are in reasonable range (0-100)
        if rsi_cols:
            rsi_values = features[rsi_cols[0]].dropna()
            assert rsi_values.min() >= 0
            assert rsi_values.max() <= 100
    
    def test_volatility_indicators(self, sample_market_data):
        """Test volatility indicator calculation."""
        engine = TechnicalIndicatorEngine()
        features = engine.calculate_subset(sample_market_data, ['volatility'])
        
        assert len(features.columns) > 0
        assert not features.empty
        
        # Check for Bollinger Bands
        bb_cols = [col for col in features.columns if 'bb_' in col]
        assert len(bb_cols) > 0
        
        # Check for ATR
        atr_cols = [col for col in features.columns if 'atr' in col]
        assert len(atr_cols) > 0
    
    def test_volume_indicators(self, sample_market_data):
        """Test volume indicator calculation."""
        engine = TechnicalIndicatorEngine()
        features = engine.calculate_subset(sample_market_data, ['volume'])
        
        assert len(features.columns) > 0
        assert not features.empty
        
        # Check for OBV
        obv_cols = [col for col in features.columns if 'obv' in col]
        assert len(obv_cols) > 0
    
    def test_all_technical_indicators(self, sample_market_data):
        """Test generation of all technical indicators."""
        engine = TechnicalIndicatorEngine()
        features = engine.calculate_all_indicators(sample_market_data)
        
        assert len(features.columns) >= 50  # Should have at least 50 indicators
        assert len(features) == len(sample_market_data)
        
        # Check that we have features from all categories
        momentum_features = [col for col in features.columns if 'momentum' in col]
        volatility_features = [col for col in features.columns if 'volatility' in col]
        volume_features = [col for col in features.columns if 'volume' in col]
        
        assert len(momentum_features) > 0
        assert len(volatility_features) > 0
        assert len(volume_features) > 0


class TestMicrostructureFeatures:
    """Test microstructure feature generation."""
    
    def test_microstructure_engine_initialization(self):
        """Test microstructure engine initialization."""
        engine = MicrostructureEngine()
        assert engine is not None
        assert hasattr(engine, 'feature_calculators')
    
    def test_bid_ask_spread_features(self, sample_microstructure_data):
        """Test bid-ask spread feature calculation."""
        engine = MicrostructureEngine()
        features = engine.calculate_subset(sample_microstructure_data, ['spread'])
        
        assert len(features.columns) > 0
        assert not features.empty
        
        # Check for spread-related features
        spread_cols = [col for col in features.columns if 'spread' in col]
        assert len(spread_cols) > 0
    
    def test_order_flow_features(self, sample_microstructure_data):
        """Test order flow feature calculation."""
        engine = MicrostructureEngine()
        features = engine.calculate_subset(sample_microstructure_data, ['order_flow'])
        
        assert len(features.columns) > 0
        assert not features.empty
    
    def test_all_microstructure_features(self, sample_microstructure_data):
        """Test generation of all microstructure features."""
        engine = MicrostructureEngine()
        features = engine.calculate_all_features(sample_microstructure_data)
        
        assert len(features.columns) > 0
        assert len(features) == len(sample_microstructure_data)


class TestRegimeDetection:
    """Test regime detection features."""
    
    def test_regime_engine_initialization(self):
        """Test regime detection engine initialization."""
        engine = RegimeDetectionEngine()
        assert engine is not None
        assert hasattr(engine, 'feature_calculators')
    
    def test_volatility_regime_features(self, sample_market_data):
        """Test volatility regime feature calculation."""
        engine = RegimeDetectionEngine()
        features = engine.calculate_subset(sample_market_data, ['volatility_regime'])
        
        assert len(features.columns) > 0
        assert not features.empty
    
    def test_trend_regime_features(self, sample_market_data):
        """Test trend regime feature calculation."""
        engine = RegimeDetectionEngine()
        features = engine.calculate_subset(sample_market_data, ['trend_regime'])
        
        assert len(features.columns) > 0
        assert not features.empty
    
    def test_regime_change_detection(self, sample_market_data):
        """Test regime change detection."""
        engine = RegimeDetectionEngine()
        regime_changes = engine.detect_regime_changes(sample_market_data)
        
        assert isinstance(regime_changes, pd.Series)
        assert len(regime_changes) == len(sample_market_data)
    
    def test_all_regime_features(self, sample_market_data):
        """Test generation of all regime detection features."""
        engine = RegimeDetectionEngine()
        features = engine.calculate_all_features(sample_market_data)
        
        assert len(features.columns) > 0
        assert len(features) == len(sample_market_data)


class TestAlternativeData:
    """Test alternative data integration."""
    
    def test_alternative_engine_initialization(self):
        """Test alternative data engine initialization."""
        engine = AlternativeDataEngine()
        assert engine is not None
        assert hasattr(engine, 'feature_calculators')
    
    def test_sentiment_features(self, sample_alternative_data):
        """Test sentiment feature calculation."""
        engine = AlternativeDataEngine()
        features = engine.calculate_subset(sample_alternative_data, ['sentiment'])
        
        assert len(features.columns) > 0
        assert not features.empty
    
    def test_macro_features(self, sample_alternative_data):
        """Test macroeconomic feature calculation."""
        engine = AlternativeDataEngine()
        features = engine.calculate_subset(sample_alternative_data, ['macro'])
        
        assert len(features.columns) > 0
        assert not features.empty
    
    def test_all_alternative_features(self, sample_alternative_data):
        """Test generation of all alternative data features."""
        engine = AlternativeDataEngine()
        features = engine.calculate_all_features(sample_alternative_data)
        
        assert len(features.columns) > 0
        assert len(features) == len(sample_alternative_data)


class TestComprehensiveEngine:
    """Test the comprehensive feature engineering engine."""
    
    def test_comprehensive_engine_initialization(self):
        """Test comprehensive engine initialization."""
        engine = ComprehensiveFeatureEngine()
        assert engine is not None
        assert hasattr(engine, 'technical_engine')
        assert hasattr(engine, 'microstructure_engine')
        assert hasattr(engine, 'regime_engine')
        assert hasattr(engine, 'alternative_engine')
    
    def test_custom_configuration(self):
        """Test engine with custom configuration."""
        config = FeatureEngineConfig(
            enable_technical=True,
            enable_microstructure=False,
            enable_regime=True,
            enable_alternative=False,
            max_features=50
        )
        
        engine = ComprehensiveFeatureEngine(config)
        assert engine.config.enable_technical is True
        assert engine.config.enable_microstructure is False
        assert engine.config.max_features == 50
    
    def test_generate_all_features(self, sample_alternative_data):
        """Test generation of all features using comprehensive engine."""
        engine = ComprehensiveFeatureEngine()
        features = engine.generate_all_features(sample_alternative_data)
        
        assert len(features.columns) > 50  # Should have many features
        assert len(features) == len(sample_alternative_data)
        assert not features.empty
    
    def test_selective_feature_generation(self, sample_market_data):
        """Test selective feature generation."""
        engine = ComprehensiveFeatureEngine()
        features = engine.generate_features(sample_market_data, ['technical', 'regime'])
        
        assert len(features.columns) > 0
        assert len(features) == len(sample_market_data)
        
        # Should only have technical and regime features
        tech_features = [col for col in features.columns if col.startswith('momentum_') or col.startswith('volatility_')]
        regime_features = [col for col in features.columns if col.startswith('regime_')]
        
        assert len(tech_features) > 0
        assert len(regime_features) > 0
    
    def test_feature_importance(self, sample_market_data):
        """Test feature importance calculation."""
        engine = ComprehensiveFeatureEngine()
        features = engine.generate_features(sample_market_data, ['technical'])
        
        # Create mock target
        target = sample_market_data['close'].pct_change().shift(-1).dropna()
        common_index = features.index.intersection(target.index)
        
        if len(common_index) > 50:
            importance = engine.get_feature_importance(
                features.loc[common_index], 
                target.loc[common_index]
            )
            
            assert isinstance(importance, pd.Series)
            assert len(importance) == len(features.columns)
            assert not importance.empty
    
    def test_feature_summary(self, sample_market_data):
        """Test feature summary generation."""
        engine = ComprehensiveFeatureEngine()
        features = engine.generate_features(sample_market_data, ['technical'])
        
        summary = engine.get_feature_summary(features)
        
        assert isinstance(summary, dict)
        assert 'total_features' in summary
        assert 'feature_types' in summary
        assert 'data_quality' in summary
        assert 'statistics' in summary
        
        assert summary['total_features'] == len(features.columns)


class TestConvenienceFunction:
    """Test the convenience function for quick feature generation."""
    
    def test_generate_comprehensive_features_default(self, sample_market_data):
        """Test quick feature generation with defaults."""
        features = generate_comprehensive_features(sample_market_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) > 0
        assert len(features) == len(sample_market_data)
        assert not features.empty
    
    def test_generate_comprehensive_features_with_config(self, sample_market_data):
        """Test feature generation with custom config."""
        config = FeatureEngineConfig(
            enable_technical=True,
            enable_microstructure=False,
            enable_regime=False,
            enable_alternative=False,
            max_features=20
        )
        
        features = generate_comprehensive_features(sample_market_data, config)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) <= 20
        assert len(features) == len(sample_market_data)


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid input data."""
        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'price': [100, 101, 102],
            'vol': [1000, 1100, 900]
        })
        
        engine = ComprehensiveFeatureEngine()
        
        with pytest.raises(ValueError):
            engine.generate_all_features(invalid_data)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create data with too few rows
        insufficient_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000, 1100]
        })
        
        engine = ComprehensiveFeatureEngine()
        
        with pytest.raises(ValueError):
            engine.generate_all_features(insufficient_data)
    
    def test_missing_value_handling(self, sample_market_data):
        """Test handling of missing values in data."""
        # Introduce some missing values
        data_with_na = sample_market_data.copy()
        data_with_na.loc[data_with_na.index[10:15], 'close'] = np.nan
        data_with_na.loc[data_with_na.index[20:25], 'volume'] = np.nan
        
        engine = ComprehensiveFeatureEngine()
        features = engine.generate_features(data_with_na, ['technical'])
        
        # Should handle missing values gracefully
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data_with_na)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
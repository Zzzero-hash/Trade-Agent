"""
Comprehensive Feature Engineering Framework

This package provides a complete feature engineering solution for financial ML models:

- 100+ Technical Indicators (momentum, volatility, volume, price patterns, trend)
- Market Microstructure Features (bid-ask spread, order flow, market impact, liquidity)
- Regime Detection Features (volatility clustering, trend identification, market states)
- Alternative Data Integration (sentiment, news embeddings, macro indicators)

Main Components:
- TechnicalIndicatorEngine: 100+ technical indicators
- MicrostructureEngine: Market microstructure analysis
- RegimeDetectionEngine: Market regime and state detection
- AlternativeDataEngine: Alternative data integration
- ComprehensiveFeatureEngine: Unified interface for all features

Usage:
    from data.features import generate_comprehensive_features
    
    features = generate_comprehensive_features(market_data)
"""

from .technical_indicators import (
    TechnicalIndicatorEngine,
    IndicatorConfig,
    MomentumIndicators,
    VolatilityIndicators,
    VolumeIndicators,
    PricePatternIndicators,
    TrendIndicators
)

from .microstructure import (
    MicrostructureEngine,
    MicrostructureConfig,
    BidAskSpreadFeatures,
    OrderFlowFeatures,
    MarketImpactFeatures,
    LiquidityFeatures
)

from .regime_detection import (
    RegimeDetectionEngine,
    RegimeConfig,
    VolatilityRegimeFeatures,
    TrendRegimeFeatures,
    MarketStateFeatures,
    StructuralBreakFeatures
)

from .alternative_data import (
    AlternativeDataEngine,
    AlternativeDataConfig,
    SentimentFeatures,
    NewsEmbeddingFeatures,
    MacroeconomicFeatures,
    SocialMediaFeatures,
    FundamentalDataFeatures
)

from .feature_engine import (
    ComprehensiveFeatureEngine,
    FeatureEngineConfig,
    FeatureValidator,
    FeatureSelector,
    FeatureCache,
    generate_comprehensive_features
)

__all__ = [
    # Main engines
    'TechnicalIndicatorEngine',
    'MicrostructureEngine', 
    'RegimeDetectionEngine',
    'AlternativeDataEngine',
    'ComprehensiveFeatureEngine',
    
    # Configurations
    'IndicatorConfig',
    'MicrostructureConfig',
    'RegimeConfig',
    'AlternativeDataConfig',
    'FeatureEngineConfig',
    
    # Technical indicator components
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'PricePatternIndicators',
    'TrendIndicators',
    
    # Microstructure components
    'BidAskSpreadFeatures',
    'OrderFlowFeatures',
    'MarketImpactFeatures',
    'LiquidityFeatures',
    
    # Regime detection components
    'VolatilityRegimeFeatures',
    'TrendRegimeFeatures',
    'MarketStateFeatures',
    'StructuralBreakFeatures',
    
    # Alternative data components
    'SentimentFeatures',
    'NewsEmbeddingFeatures',
    'MacroeconomicFeatures',
    'SocialMediaFeatures',
    'FundamentalDataFeatures',
    
    # Utilities
    'FeatureValidator',
    'FeatureSelector',
    'FeatureCache',
    
    # Convenience functions
    'generate_comprehensive_features'
]
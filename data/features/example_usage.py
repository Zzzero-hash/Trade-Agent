"""
Example Usage of Comprehensive Feature Engineering Framework

This script demonstrates how to use the feature engineering framework
to generate 100+ technical indicators, microstructure features,
regime detection features, and alternative data features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

from data.features import (
    generate_comprehensive_features,
    ComprehensiveFeatureEngine,
    FeatureEngineConfig,
    TechnicalIndicatorEngine,
    MicrostructureEngine,
    RegimeDetectionEngine,
    AlternativeDataEngine
)

warnings.filterwarnings('ignore')


def create_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic price data using geometric Brownian motion
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Parameters for price simulation
    initial_price = 100.0
    drift = 0.0005  # Daily drift
    volatility = 0.02  # Daily volatility
    
    # Generate returns
    returns = np.random.normal(drift, volatility, n_days)
    
    # Add some regime changes and trends
    regime_changes = [200, 400, 600, 800]
    for change_point in regime_changes:
        if change_point < n_days:
            # Change volatility regime
            returns[change_point:change_point+100] *= np.random.uniform(1.5, 3.0)
    
    # Calculate prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = pd.DataFrame(index=dates[:len(prices)])
    data['close'] = prices
    
    # Generate realistic OHLC from close prices
    daily_range = np.abs(np.random.normal(0, volatility * 0.5, len(prices)))
    
    data['high'] = data['close'] * (1 + daily_range * np.random.uniform(0.3, 1.0, len(prices)))
    data['low'] = data['close'] * (1 - daily_range * np.random.uniform(0.3, 1.0, len(prices)))
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, volatility * 0.3, len(prices)))
    data['open'].iloc[0] = initial_price
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Generate volume data
    base_volume = 1000000
    volume_volatility = 0.3
    data['volume'] = base_volume * np.exp(np.random.normal(0, volume_volatility, len(prices)))
    
    # Add some microstructure data (optional)
    spread_bps = np.random.uniform(1, 10, len(prices))  # 1-10 bps spread
    data['bid_price'] = data['close'] * (1 - spread_bps / 20000)
    data['ask_price'] = data['close'] * (1 + spread_bps / 20000)
    data['bid_size'] = np.random.uniform(100, 1000, len(prices))
    data['ask_size'] = np.random.uniform(100, 1000, len(prices))
    
    # Add some alternative data (mock)
    data['news_sentiment'] = np.random.normal(0, 0.3, len(prices))
    data['social_sentiment'] = np.random.normal(0, 0.4, len(prices))
    data['vix'] = 15 + 10 * np.abs(np.random.normal(0, 1, len(prices)))
    data['news_count'] = np.random.poisson(5, len(prices))
    
    return data


def demonstrate_technical_indicators():
    """Demonstrate technical indicator generation."""
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(500)
    
    # Initialize technical indicator engine
    engine = TechnicalIndicatorEngine()
    
    # Generate all technical indicators
    print("Generating 100+ technical indicators...")
    technical_features = engine.calculate_all_indicators(data)
    
    print(f"Generated {len(technical_features.columns)} technical indicators")
    print(f"Feature shape: {technical_features.shape}")
    
    # Show sample of features
    print("\nSample technical indicators:")
    sample_cols = technical_features.columns[:10]
    print(technical_features[sample_cols].tail())
    
    # Show feature categories
    momentum_features = [col for col in technical_features.columns if 'momentum' in col]
    volatility_features = [col for col in technical_features.columns if 'volatility' in col]
    volume_features = [col for col in technical_features.columns if 'volume' in col]
    
    print(f"\nFeature breakdown:")
    print(f"- Momentum indicators: {len(momentum_features)}")
    print(f"- Volatility indicators: {len(volatility_features)}")
    print(f"- Volume indicators: {len(volume_features)}")
    
    return technical_features


def demonstrate_microstructure_features():
    """Demonstrate microstructure feature generation."""
    print("\n" + "="*60)
    print("MICROSTRUCTURE FEATURES DEMONSTRATION")
    print("="*60)
    
    # Create sample data with microstructure information
    data = create_sample_data(500)
    
    # Initialize microstructure engine
    engine = MicrostructureEngine()
    
    # Generate microstructure features
    print("Generating market microstructure features...")
    micro_features = engine.calculate_all_features(data)
    
    print(f"Generated {len(micro_features.columns)} microstructure features")
    print(f"Feature shape: {micro_features.shape}")
    
    # Show sample of features
    print("\nSample microstructure features:")
    sample_cols = micro_features.columns[:8]
    print(micro_features[sample_cols].tail())
    
    return micro_features


def demonstrate_regime_detection():
    """Demonstrate regime detection features."""
    print("\n" + "="*60)
    print("REGIME DETECTION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(500)
    
    # Initialize regime detection engine
    engine = RegimeDetectionEngine()
    
    # Generate regime detection features
    print("Generating regime detection features...")
    regime_features = engine.calculate_all_features(data)
    
    print(f"Generated {len(regime_features.columns)} regime detection features")
    print(f"Feature shape: {regime_features.shape}")
    
    # Show sample of features
    print("\nSample regime detection features:")
    sample_cols = regime_features.columns[:8]
    print(regime_features[sample_cols].tail())
    
    # Detect regime changes
    regime_changes = engine.detect_regime_changes(data, method='volatility')
    change_dates = regime_changes[regime_changes].index
    print(f"\nDetected {len(change_dates)} potential regime changes")
    if len(change_dates) > 0:
        print(f"Recent regime changes: {change_dates[-3:].tolist()}")
    
    return regime_features


def demonstrate_alternative_data():
    """Demonstrate alternative data integration."""
    print("\n" + "="*60)
    print("ALTERNATIVE DATA DEMONSTRATION")
    print("="*60)
    
    # Create sample data with alternative data
    data = create_sample_data(500)
    
    # Initialize alternative data engine
    engine = AlternativeDataEngine()
    
    # Generate alternative data features
    print("Generating alternative data features...")
    alt_features = engine.calculate_all_features(data)
    
    print(f"Generated {len(alt_features.columns)} alternative data features")
    print(f"Feature shape: {alt_features.shape}")
    
    # Show sample of features
    print("\nSample alternative data features:")
    sample_cols = alt_features.columns[:8]
    print(alt_features[sample_cols].tail())
    
    return alt_features


def demonstrate_comprehensive_framework():
    """Demonstrate the comprehensive feature engineering framework."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE ENGINEERING FRAMEWORK DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(1000)
    print(f"Created sample market data: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Method 1: Quick generation with defaults
    print("\n1. Quick feature generation with defaults:")
    features_quick = generate_comprehensive_features(data)
    print(f"Generated {len(features_quick.columns)} features using default configuration")
    
    # Method 2: Custom configuration
    print("\n2. Custom configuration:")
    config = FeatureEngineConfig(
        enable_technical=True,
        enable_microstructure=True,
        enable_regime=True,
        enable_alternative=True,
        normalize_features=True,
        remove_outliers=True,
        max_features=200  # Limit to top 200 features
    )
    
    engine = ComprehensiveFeatureEngine(config)
    features_custom = engine.generate_all_features(data)
    print(f"Generated {len(features_custom.columns)} features with custom configuration")
    
    # Get feature summary
    summary = engine.get_feature_summary(features_custom)
    print(f"\nFeature Summary:")
    print(f"- Total features: {summary['total_features']}")
    print(f"- Feature types: {summary['feature_types']}")
    print(f"- Missing values: {summary['data_quality']['missing_values']}")
    print(f"- Mean correlation: {summary['statistics']['mean_correlation']:.3f}")
    
    # Method 3: Selective feature generation
    print("\n3. Selective feature generation:")
    selected_features = engine.generate_features(data, ['technical', 'regime'])
    print(f"Generated {len(selected_features.columns)} features (technical + regime only)")
    
    # Feature importance (if we had a target)
    print("\n4. Feature importance analysis:")
    # Create a mock target (next day return)
    target = data['close'].pct_change().shift(-1).dropna()
    common_index = features_custom.index.intersection(target.index)
    
    if len(common_index) > 100:
        importance = engine.get_feature_importance(
            features_custom.loc[common_index], 
            target.loc[common_index]
        )
        print("Top 10 most important features:")
        print(importance.head(10))
    
    # Save features
    print("\n5. Saving features:")
    engine.save_features(features_custom, "data/processed/comprehensive_features.csv")
    print("Features saved to data/processed/comprehensive_features.csv")
    
    return features_custom


def create_feature_visualization(features: pd.DataFrame, data: pd.DataFrame):
    """Create visualizations of generated features."""
    print("\n" + "="*60)
    print("FEATURE VISUALIZATION")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Price and some technical indicators
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['close'], label='Close Price', alpha=0.7)
        
        # Find moving average columns
        ma_cols = [col for col in features.columns if 'sma_20' in col or 'ema_20' in col]
        if ma_cols:
            ax1.plot(features.index, features[ma_cols[0]], label='MA(20)', alpha=0.7)
        
        ax1.set_title('Price and Moving Average')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volatility indicators
        ax2 = axes[0, 1]
        vol_cols = [col for col in features.columns if 'realized_vol' in col]
        if vol_cols:
            ax2.plot(features.index, features[vol_cols[0]], label='Realized Volatility')
            ax2.set_title('Volatility Indicators')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature correlation heatmap (sample)
        ax3 = axes[1, 0]
        sample_features = features.iloc[:, :20]  # First 20 features
        corr_matrix = sample_features.corr()
        im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_title('Feature Correlation Matrix (Sample)')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Feature distribution
        ax4 = axes[1, 1]
        if len(features.columns) > 0:
            sample_feature = features.iloc[:, 0].dropna()
            ax4.hist(sample_feature, bins=50, alpha=0.7)
            ax4.set_title(f'Feature Distribution: {features.columns[0]}')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/features/feature_analysis.png', dpi=300, bbox_inches='tight')
        print("Feature visualization saved to data/features/feature_analysis.png")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def main():
    """Main demonstration function."""
    print("COMPREHENSIVE FEATURE ENGINEERING FRAMEWORK")
    print("=" * 80)
    print("This demonstration showcases the generation of 100+ features including:")
    print("- Technical indicators (momentum, volatility, volume, price patterns)")
    print("- Market microstructure features (spreads, order flow, market impact)")
    print("- Regime detection features (volatility clustering, trend identification)")
    print("- Alternative data integration (sentiment, news, macro indicators)")
    
    try:
        # Individual component demonstrations
        tech_features = demonstrate_technical_indicators()
        micro_features = demonstrate_microstructure_features()
        regime_features = demonstrate_regime_detection()
        alt_features = demonstrate_alternative_data()
        
        # Comprehensive framework demonstration
        all_features = demonstrate_comprehensive_framework()
        
        # Create visualizations
        sample_data = create_sample_data(500)
        create_feature_visualization(all_features, sample_data)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print(f"Successfully generated comprehensive feature set with {len(all_features.columns)} features")
        print("The framework is ready for production use in ML trading systems.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
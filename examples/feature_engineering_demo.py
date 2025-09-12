"""
Demo script showing the complete feature engineering pipeline
with technical indicators and advanced transformers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.ml.feature_engineering import (
    FeatureEngineer, TechnicalIndicators, FourierFeatures, 
    FractalFeatures, CrossAssetFeatures
)


def create_sample_data(n_points=200):
    """Create sample market data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")
    
    # Generate realistic price data with trend and volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_points)
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
        "volume": np.random.randint(1000, 10000, n_points)
    })
    
    # Ensure high >= close >= low
    data["high"] = np.maximum(data["high"], data["close"])
    data["low"] = np.minimum(data["low"], data["close"])
    
    return data


def main():
    """Main demonstration function"""
    print("=== Feature Engineering Pipeline Demo ===\n")
    
    # Create sample data
    print("1. Creating sample market data...")
    data = create_sample_data(200)
    print(f"   Generated {len(data)} data points")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}\n")
    
    # Initialize feature engineer
    print("2. Setting up feature engineering pipeline...")
    engineer = FeatureEngineer()
    
    # Add technical indicators
    tech_config = {
        "indicators": ["sma_20", "ema_12", "rsi_14", "macd", "bollinger_bands", "atr_14", "obv", "vwap"]
    }
    engineer.add_transformer(TechnicalIndicators(tech_config))
    print("   ✓ Added technical indicators")
    
    # Add Fourier features
    fourier_config = {"n_components": 5, "window_size": 30}
    engineer.add_transformer(FourierFeatures(fourier_config))
    print("   ✓ Added Fourier features")
    
    # Add fractal features
    fractal_config = {"window_size": 40, "max_lag": 15}
    engineer.add_transformer(FractalFeatures(fractal_config))
    print("   ✓ Added fractal features")
    
    # Add cross-asset features
    cross_config = {"window_size": 25}
    engineer.add_transformer(CrossAssetFeatures(cross_config))
    print("   ✓ Added cross-asset features\n")
    
    # Fit and transform
    print("3. Processing features...")
    features = engineer.fit_transform(data)
    feature_names = engineer.get_feature_names()
    
    print(f"   Generated {features.shape[1]} features from {len(data)} data points")
    print(f"   Feature matrix shape: {features.shape}")
    print(f"   No NaN values: {not np.any(np.isnan(features))}\n")
    
    # Show feature breakdown
    print("4. Feature breakdown:")
    
    # Technical indicators
    tech_features = [name for name in feature_names if any(
        indicator in name for indicator in ["sma", "ema", "rsi", "macd", "bb", "atr", "obv", "vwap"]
    )]
    print(f"   Technical indicators: {len(tech_features)} features")
    for feat in tech_features[:5]:  # Show first 5
        print(f"     - {feat}")
    if len(tech_features) > 5:
        print(f"     ... and {len(tech_features) - 5} more")
    
    # Fourier features
    fourier_features = [name for name in feature_names if "fft" in name]
    print(f"   Fourier features: {len(fourier_features)} features")
    print(f"     - {len(fourier_features)//2} magnitude + {len(fourier_features)//2} phase components")
    
    # Fractal features
    fractal_features = [name for name in feature_names if name in ["hurst_exponent", "fractal_dimension"]]
    print(f"   Fractal features: {len(fractal_features)} features")
    for feat in fractal_features:
        print(f"     - {feat}")
    
    # Cross-asset features
    cross_features = [name for name in feature_names if "correlation" in name or "momentum" in name]
    print(f"   Cross-asset features: {len(cross_features)} features")
    for feat in cross_features:
        print(f"     - {feat}")
    
    print("\n5. Feature statistics:")
    print(f"   Mean feature value: {np.mean(features):.4f}")
    print(f"   Std feature value: {np.std(features):.4f}")
    print(f"   Min feature value: {np.min(features):.4f}")
    print(f"   Max feature value: {np.max(features):.4f}")
    
    # Show some sample values
    print("\n6. Sample feature values (last 5 time points):")
    sample_features = features[-5:, :10]  # Last 5 rows, first 10 features
    sample_names = feature_names[:10]
    
    for i, name in enumerate(sample_names):
        values = sample_features[:, i]
        print(f"   {name:25s}: [{', '.join([f'{v:8.4f}' for v in values])}]")
    
    print("\n=== Demo completed successfully! ===")
    print("\nThe feature engineering pipeline successfully processed market data")
    print("and generated a comprehensive set of features for ML model training.")


if __name__ == "__main__":
    main()
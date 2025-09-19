"""
Regime Detection Features

This module implements advanced regime detection features including:
- Volatility clustering detection
- Trend identification and classification
- Market state classification
- Structural break detection
- Regime switching models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RegimeConfig:
    """Configuration for regime detection features."""
    volatility_windows: List[int] = None
    trend_windows: List[int] = None
    regime_lookback: int = 252
    n_regimes: int = 3
    min_regime_length: int = 10
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 50, 100]
        if self.trend_windows is None:
            self.trend_windows = [10, 20, 50, 100, 200]


class VolatilityRegimeFeatures:
    """Volatility clustering and regime detection features."""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility regime features."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate returns
        returns = data['close'].pct_change().fillna(0)
        
        # Realized volatility measures
        for window in self.config.volatility_windows:
            # Standard volatility
            features[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            # Parkinson volatility (using high-low)
            if 'high' in data.columns and 'low' in data.columns:
                hl_ratio = np.log(data['high'] / data['low'])
                features[f'parkinson_vol_{window}'] = np.sqrt(
                    hl_ratio.rolling(window).mean() / (4 * np.log(2))
                ) * np.sqrt(252)
            
            # Garman-Klass volatility
            if all(col in data.columns for col in ['high', 'low', 'open']):
                hl = np.log(data['high'] / data['low']) ** 2
                cc = np.log(data['close'] / data['open']) ** 2
                features[f'gk_vol_{window}'] = np.sqrt(
                    (0.5 * hl - (2*np.log(2)-1) * cc).rolling(window).mean()
                ) * np.sqrt(252)
        
        # Volatility clustering measures
        abs_returns = returns.abs()
        
        # ARCH effects (volatility clustering)
        for lag in [1, 2, 5]:
            features[f'arch_effect_lag_{lag}'] = abs_returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else 0
            )
        
        # Volatility persistence
        for window in [20, 50]:
            vol = returns.rolling(window).std()
            features[f'vol_persistence_{window}'] = vol.rolling(window).apply(
                lambda x: x.autocorr(1) if len(x) > 1 else 0
            )
        
        # Volatility regime classification using Gaussian Mixture Models
        if len(data) > 100:
            vol_features = []
            for window in [10, 20, 50]:
                vol_features.append(returns.rolling(window).std().fillna(0))
            
            vol_matrix = np.column_stack(vol_features)
            
            # Remove NaN rows
            valid_idx = ~np.isnan(vol_matrix).any(axis=1)
            if valid_idx.sum() > 50:
                vol_clean = vol_matrix[valid_idx]
                
                # Fit Gaussian Mixture Model
                gmm = GaussianMixture(n_components=self.config.n_regimes, random_state=42)
                try:
                    regime_labels = gmm.fit_predict(vol_clean)
                    
                    # Map back to full index
                    full_regimes = np.full(len(data), -1)
                    full_regimes[valid_idx] = regime_labels
                    features['volatility_regime'] = full_regimes
                    
                    # Regime probabilities
                    regime_probs = gmm.predict_proba(vol_clean)
                    for i in range(self.config.n_regimes):
                        prob_series = np.full(len(data), 0.0)
                        prob_series[valid_idx] = regime_probs[:, i]
                        features[f'vol_regime_prob_{i}'] = prob_series
                        
                except:
                    features['volatility_regime'] = 0
                    for i in range(self.config.n_regimes):
                        features[f'vol_regime_prob_{i}'] = 1.0 / self.config.n_regimes
        
        # Volatility breakout detection
        for window in [20, 50]:
            vol_ma = features[f'realized_vol_{window}'].rolling(window).mean()
            vol_std = features[f'realized_vol_{window}'].rolling(window).std()
            features[f'vol_breakout_{window}'] = (
                (features[f'realized_vol_{window}'] - vol_ma) / vol_std
            ).fillna(0)
        
        # Volatility mean reversion
        for window in [20, 50]:
            vol_mean = features[f'realized_vol_{window}'].rolling(window*2).mean()
            features[f'vol_mean_reversion_{window}'] = (
                vol_mean - features[f'realized_vol_{window}']
            ) / vol_mean
        
        return features


class TrendRegimeFeatures:
    """Trend identification and classification features."""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend regime features."""
        features = pd.DataFrame(index=data.index)
        
        # Price and returns
        price = data['close']
        returns = price.pct_change().fillna(0)
        
        # Trend strength measures
        for window in self.config.trend_windows:
            # Linear regression slope
            slope = price.rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
            )
            features[f'trend_slope_{window}'] = slope
            
            # Normalized trend strength
            price_std = price.rolling(window).std()
            features[f'trend_strength_{window}'] = (slope * window) / price_std
            
            # R-squared of linear fit
            features[f'trend_r2_{window}'] = price.rolling(window).apply(
                lambda x: self._calculate_r_squared(x) if len(x) == window else 0
            )
        
        # Moving average trends
        for short, long in [(10, 50), (20, 100), (50, 200)]:
            if len(data) > long:
                ma_short = price.rolling(short).mean()
                ma_long = price.rolling(long).mean()
                
                # MA crossover signals
                features[f'ma_cross_{short}_{long}'] = (ma_short > ma_long).astype(int)
                features[f'ma_divergence_{short}_{long}'] = (ma_short - ma_long) / ma_long
                
                # Trend consistency
                features[f'trend_consistency_{short}_{long}'] = (
                    (ma_short > ma_short.shift(1)).rolling(20).mean()
                )
        
        # Trend regime classification
        trend_features = []
        for window in [20, 50, 100]:
            if len(data) > window:
                # Cumulative returns
                cum_returns = returns.rolling(window).sum()
                trend_features.append(cum_returns.fillna(0))
                
                # Trend momentum
                momentum = price.pct_change(window)
                trend_features.append(momentum.fillna(0))
        
        if trend_features and len(data) > 100:
            trend_matrix = np.column_stack(trend_features)
            valid_idx = ~np.isnan(trend_matrix).any(axis=1)
            
            if valid_idx.sum() > 50:
                trend_clean = trend_matrix[valid_idx]
                
                # Standardize features
                scaler = StandardScaler()
                trend_scaled = scaler.fit_transform(trend_clean)
                
                # K-means clustering for trend regimes
                kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
                try:
                    trend_labels = kmeans.fit_predict(trend_scaled)
                    
                    # Map back to full index
                    full_trends = np.full(len(data), -1)
                    full_trends[valid_idx] = trend_labels
                    features['trend_regime'] = full_trends
                    
                    # Distance to cluster centers
                    distances = kmeans.transform(trend_scaled)
                    for i in range(self.config.n_regimes):
                        dist_series = np.full(len(data), np.inf)
                        dist_series[valid_idx] = distances[:, i]
                        features[f'trend_regime_dist_{i}'] = dist_series
                        
                except:
                    features['trend_regime'] = 0
                    for i in range(self.config.n_regimes):
                        features[f'trend_regime_dist_{i}'] = 1.0
        
        # Trend reversal signals
        for window in [10, 20]:
            # Price momentum reversal
            momentum = returns.rolling(window).sum()
            features[f'momentum_reversal_{window}'] = -momentum.shift(1) * returns
            
            # Overbought/oversold conditions
            rsi_proxy = returns.rolling(window).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            features[f'rsi_proxy_{window}'] = rsi_proxy
            features[f'overbought_{window}'] = (rsi_proxy > 0.7).astype(int)
            features[f'oversold_{window}'] = (rsi_proxy < 0.3).astype(int)
        
        return features
    
    def _calculate_r_squared(self, y: pd.Series) -> float:
        """Calculate R-squared for linear trend."""
        try:
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        except:
            return 0


class MarketStateFeatures:
    """Market state and regime classification features."""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market state features."""
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change().fillna(0)
        
        # Bull/Bear market indicators
        for window in [50, 100, 200]:
            if len(data) > window:
                # Price relative to moving average
                ma = data['close'].rolling(window).mean()
                features[f'price_ma_ratio_{window}'] = data['close'] / ma
                
                # Bull market indicator (price > MA for extended period)
                above_ma = (data['close'] > ma).rolling(20).sum()
                features[f'bull_strength_{window}'] = above_ma / 20
                
                # Drawdown from peak
                peak = data['close'].rolling(window).max()
                features[f'drawdown_{window}'] = (data['close'] - peak) / peak
        
        # Market stress indicators
        # VIX-like measure (rolling volatility)
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(50).std()
        features['stress_indicator'] = vol_short / vol_long
        
        # Skewness and kurtosis (tail risk measures)
        for window in [20, 50]:
            features[f'skewness_{window}'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}'] = returns.rolling(window).kurt()
            
            # Tail risk measures
            features[f'left_tail_risk_{window}'] = returns.rolling(window).quantile(0.05)
            features[f'right_tail_risk_{window}'] = returns.rolling(window).quantile(0.95)
        
        # Market efficiency measures
        # Hurst exponent estimation
        for window in [50, 100]:
            features[f'hurst_exponent_{window}'] = returns.rolling(window).apply(
                self._estimate_hurst, raw=False
            )
        
        # Autocorrelation structure
        for lag in [1, 2, 5, 10]:
            features[f'return_autocorr_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else 0
            )
        
        # Regime persistence
        if 'volatility_regime' in features.columns:
            regime_changes = features['volatility_regime'].diff() != 0
            features['regime_persistence'] = (~regime_changes).rolling(20).sum()
        
        # Crisis detection features
        # Extreme return frequency
        for threshold in [0.02, 0.05]:  # 2% and 5% daily moves
            extreme_moves = (returns.abs() > threshold).rolling(20).sum()
            features[f'extreme_freq_{int(threshold*100)}pct'] = extreme_moves / 20
        
        # Correlation breakdown (flight to quality)
        if len(data) > 100:
            # Proxy for correlation using rolling correlation with lagged returns
            features['correlation_breakdown'] = returns.rolling(50).apply(
                lambda x: abs(x.autocorr(1)) if len(x) > 1 else 0
            )
        
        return features
    
    def _estimate_hurst(self, returns: pd.Series) -> float:
        """Estimate Hurst exponent using R/S analysis."""
        try:
            if len(returns) < 10:
                return 0.5
            
            returns_array = returns.values
            n = len(returns_array)
            
            # Calculate mean
            mean_return = np.mean(returns_array)
            
            # Calculate deviations from mean
            deviations = returns_array - mean_return
            
            # Calculate cumulative deviations
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(returns_array)
            
            # Avoid division by zero
            if S == 0:
                return 0.5
            
            # Calculate R/S ratio
            rs_ratio = R / S
            
            # Hurst exponent
            if rs_ratio > 0:
                hurst = np.log(rs_ratio) / np.log(n)
                return max(0, min(1, hurst))  # Clamp between 0 and 1
            else:
                return 0.5
                
        except:
            return 0.5


class StructuralBreakFeatures:
    """Structural break and change point detection features."""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate structural break features."""
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change().fillna(0)
        
        # CUSUM-based break detection
        for window in [50, 100]:
            if len(data) > window:
                # Cumulative sum of standardized returns
                mean_return = returns.rolling(window).mean()
                std_return = returns.rolling(window).std()
                standardized_returns = (returns - mean_return) / (std_return + 1e-8)
                
                cusum_pos = np.maximum(0, standardized_returns).cumsum()
                cusum_neg = np.minimum(0, standardized_returns).cumsum()
                
                features[f'cusum_pos_{window}'] = cusum_pos
                features[f'cusum_neg_{window}'] = cusum_neg
                
                # Break detection signals
                cusum_threshold = 3.0  # Standard threshold
                features[f'break_signal_pos_{window}'] = (cusum_pos > cusum_threshold).astype(int)
                features[f'break_signal_neg_{window}'] = (cusum_neg < -cusum_threshold).astype(int)
        
        # Variance change detection
        for window in [20, 50]:
            if len(data) > window * 2:
                # Rolling variance
                rolling_var = returns.rolling(window).var()
                
                # Variance ratio test
                var_ratio = rolling_var / rolling_var.shift(window)
                features[f'variance_ratio_{window}'] = var_ratio
                
                # Significant variance changes
                features[f'variance_break_{window}'] = (
                    (var_ratio > 2) | (var_ratio < 0.5)
                ).astype(int)
        
        # Mean reversion tests
        for window in [50, 100]:
            if len(data) > window:
                # ADF test statistic proxy
                price_window = data['close'].rolling(window)
                lagged_price = data['close'].shift(1)
                
                # Simple mean reversion test
                price_deviation = data['close'] - price_window.mean()
                features[f'mean_reversion_{window}'] = price_deviation.rolling(10).apply(
                    lambda x: x.autocorr(1) if len(x) > 1 else 0
                )
        
        # Regime duration features
        if len(data) > 100:
            # Estimate regime durations using simple threshold method
            vol = returns.rolling(20).std()
            vol_threshold = vol.quantile(0.7)
            
            high_vol_regime = (vol > vol_threshold).astype(int)
            regime_changes = high_vol_regime.diff() != 0
            
            # Time since last regime change
            features['time_since_regime_change'] = 0
            change_points = regime_changes[regime_changes].index
            
            for i, idx in enumerate(data.index):
                if len(change_points) > 0:
                    last_change = change_points[change_points <= idx]
                    if len(last_change) > 0:
                        features.loc[idx, 'time_since_regime_change'] = (
                            data.index.get_loc(idx) - data.index.get_loc(last_change[-1])
                        )
        
        return features


class RegimeDetectionEngine:
    """Main engine for calculating all regime detection features."""
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.feature_calculators = {
            'volatility_regime': VolatilityRegimeFeatures(self.config),
            'trend_regime': TrendRegimeFeatures(self.config),
            'market_state': MarketStateFeatures(self.config),
            'structural_breaks': StructuralBreakFeatures(self.config)
        }
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all regime detection features."""
        all_features = pd.DataFrame(index=data.index)
        
        for feature_type, calculator in self.feature_calculators.items():
            try:
                features = calculator.calculate(data)
                # Add prefix to avoid column name conflicts
                features.columns = [f"regime_{feature_type}_{col}" for col in features.columns]
                all_features = pd.concat([all_features, features], axis=1)
            except Exception as e:
                print(f"Error calculating {feature_type} features: {e}")
                continue
        
        return all_features
    
    def calculate_subset(self, data: pd.DataFrame, 
                        feature_types: List[str]) -> pd.DataFrame:
        """Calculate only specified feature types."""
        features = pd.DataFrame(index=data.index)
        
        for feature_type in feature_types:
            if feature_type in self.feature_calculators:
                calc_features = self.feature_calculators[feature_type].calculate(data)
                calc_features.columns = [f"regime_{feature_type}_{col}" 
                                       for col in calc_features.columns]
                features = pd.concat([features, calc_features], axis=1)
        
        return features
    
    def detect_regime_changes(self, data: pd.DataFrame, 
                            method: str = 'volatility') -> pd.Series:
        """Detect regime change points."""
        if method == 'volatility':
            vol_features = self.feature_calculators['volatility_regime'].calculate(data)
            if 'volatility_regime' in vol_features.columns:
                return vol_features['volatility_regime'].diff() != 0
        elif method == 'trend':
            trend_features = self.feature_calculators['trend_regime'].calculate(data)
            if 'trend_regime' in trend_features.columns:
                return trend_features['trend_regime'].diff() != 0
        
        # Fallback: simple volatility-based detection
        returns = data['close'].pct_change()
        vol = returns.rolling(20).std()
        vol_changes = vol.pct_change().abs() > vol.pct_change().abs().quantile(0.95)
        return vol_changes
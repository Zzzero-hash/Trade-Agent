"""
Market Microstructure Features

This module implements advanced market microstructure features including:
- Bid-ask spread analysis
- Order flow indicators
- Market impact measures
- Liquidity metrics
- Price discovery features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure features."""
    tick_size: float = 0.01
    lot_size: int = 100
    impact_periods: List[int] = None
    flow_periods: List[int] = None
    
    def __post_init__(self):
        if self.impact_periods is None:
            self.impact_periods = [1, 5, 10, 30, 60]
        if self.flow_periods is None:
            self.flow_periods = [5, 10, 20, 50, 100]


class BidAskSpreadFeatures:
    """Bid-ask spread and related liquidity features."""
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bid-ask spread features.
        
        Expected columns in data:
        - bid_price, ask_price, bid_size, ask_size
        - high, low, close, volume
        """
        features = pd.DataFrame(index=data.index)
        
        if 'bid_price' in data.columns and 'ask_price' in data.columns:
            # Direct spread measures
            features['bid_ask_spread'] = data['ask_price'] - data['bid_price']
            features['relative_spread'] = features['bid_ask_spread'] / ((data['bid_price'] + data['ask_price']) / 2)
            features['percentage_spread'] = features['relative_spread'] * 100
            
            # Mid-price
            features['mid_price'] = (data['bid_price'] + data['ask_price']) / 2
            features['price_mid_deviation'] = (data['close'] - features['mid_price']) / features['mid_price']
            
            # Spread volatility
            for period in [10, 20, 50]:
                features[f'spread_volatility_{period}'] = features['relative_spread'].rolling(period).std()
                features[f'spread_mean_{period}'] = features['relative_spread'].rolling(period).mean()
            
            if 'bid_size' in data.columns and 'ask_size' in data.columns:
                # Size-weighted spread
                total_size = data['bid_size'] + data['ask_size']
                features['size_weighted_spread'] = (features['bid_ask_spread'] * total_size) / total_size.rolling(20).mean()
                
                # Bid-ask imbalance
                features['bid_ask_imbalance'] = (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size'])
                features['size_ratio'] = data['bid_size'] / data['ask_size']
                
                # Depth features
                features['total_depth'] = data['bid_size'] + data['ask_size']
                for period in [10, 20]:
                    features[f'depth_ratio_{period}'] = features['total_depth'] / features['total_depth'].rolling(period).mean()
        
        else:
            # Estimate spread from high-low data (Roll's measure)
            returns = data['close'].pct_change()
            for period in [20, 50]:
                # Roll's spread estimator
                cov_returns = returns.rolling(period).apply(lambda x: np.cov(x[:-1], x[1:])[0,1] if len(x) > 1 else 0)
                features[f'roll_spread_{period}'] = 2 * np.sqrt(-cov_returns).fillna(0)
            
            # High-Low spread proxy
            features['hl_spread_proxy'] = (data['high'] - data['low']) / data['close']
            
            # Effective spread proxy using volume
            features['volume_spread_proxy'] = (data['high'] - data['low']) / np.log(data['volume'] + 1)
        
        return features


class OrderFlowFeatures:
    """Order flow and trade direction features."""
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow features.
        
        Expected columns: close, volume, and optionally trade_direction, trade_size
        """
        features = pd.DataFrame(index=data.index)
        
        # Price change and volume relationship
        price_change = data['close'].diff()
        features['price_volume_correlation'] = price_change.rolling(20).corr(data['volume'])
        
        # Volume-weighted price change
        features['vwpc'] = (price_change * data['volume']).rolling(10).sum() / data['volume'].rolling(10).sum()
        
        if 'trade_direction' in data.columns:
            # Direct order flow measures
            buy_volume = data['volume'] * (data['trade_direction'] > 0)
            sell_volume = data['volume'] * (data['trade_direction'] < 0)
            
            features['buy_volume'] = buy_volume
            features['sell_volume'] = sell_volume
            features['net_flow'] = buy_volume - sell_volume
            features['flow_imbalance'] = features['net_flow'] / data['volume']
            
            # Cumulative order flow
            features['cumulative_flow'] = features['net_flow'].cumsum()
            
            # Order flow momentum
            for period in self.config.flow_periods:
                features[f'flow_momentum_{period}'] = features['net_flow'].rolling(period).sum()
                features[f'flow_ratio_{period}'] = buy_volume.rolling(period).sum() / data['volume'].rolling(period).sum()
        
        else:
            # Infer trade direction using tick rule and other heuristics
            features['tick_direction'] = np.sign(price_change).fillna(0)
            
            # Lee-Ready algorithm approximation
            mid_point_change = data['close'] - (data['high'] + data['low']) / 2
            features['lr_direction'] = np.where(mid_point_change > 0, 1, 
                                              np.where(mid_point_change < 0, -1, 
                                                     features['tick_direction']))
            
            # Inferred order flow
            inferred_buy_volume = data['volume'] * (features['lr_direction'] > 0)
            inferred_sell_volume = data['volume'] * (features['lr_direction'] < 0)
            
            features['inferred_net_flow'] = inferred_buy_volume - inferred_sell_volume
            features['inferred_flow_imbalance'] = features['inferred_net_flow'] / data['volume']
            
            # Flow persistence
            for period in [5, 10, 20]:
                features[f'flow_persistence_{period}'] = features['lr_direction'].rolling(period).mean()
        
        # Volume clustering
        for period in [10, 20]:
            vol_ma = data['volume'].rolling(period).mean()
            features[f'volume_surprise_{period}'] = (data['volume'] - vol_ma) / vol_ma
        
        # Trade size analysis (if available)
        if 'trade_size' in data.columns:
            features['avg_trade_size'] = data['trade_size']
            for period in [10, 20]:
                features[f'trade_size_ratio_{period}'] = (data['trade_size'] / 
                                                        data['trade_size'].rolling(period).mean())
        
        return features


class MarketImpactFeatures:
    """Market impact and price discovery features."""
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market impact features."""
        features = pd.DataFrame(index=data.index)
        
        # Price impact measures
        returns = data['close'].pct_change()
        
        # Temporary impact (mean reversion)
        for period in self.config.impact_periods:
            if period < len(data):
                # Forward-looking impact (for backtesting, use with caution)
                features[f'temp_impact_{period}'] = returns.rolling(period).mean().shift(-period)
                
                # Realized impact
                features[f'realized_impact_{period}'] = (data['close'].shift(-period) - data['close']) / data['close']
        
        # Volume impact relationship
        log_volume = np.log(data['volume'] + 1)
        features['volume_impact'] = returns.rolling(20).corr(log_volume)
        
        # Kyle's lambda (price impact coefficient)
        for period in [20, 50]:
            if len(data) > period:
                # Simplified Kyle's lambda estimation
                vol_window = data['volume'].rolling(period)
                ret_window = returns.rolling(period)
                
                # Estimate as correlation between |returns| and volume
                features[f'kyle_lambda_{period}'] = ret_window.abs().corr(vol_window)
        
        # Amihud illiquidity measure
        for period in [20, 50]:
            illiq = (returns.abs() / (data['volume'] * data['close'])).rolling(period).mean()
            features[f'amihud_illiq_{period}'] = illiq
        
        # Price efficiency measures
        # Variance ratio test statistic
        for k in [2, 4, 8]:
            if len(data) > k * 20:
                ret_k = returns.rolling(k).sum()
                var_1 = returns.var()
                var_k = ret_k.var()
                features[f'variance_ratio_{k}'] = (var_k / (k * var_1)) - 1
        
        # Autocorrelation of returns (efficiency measure)
        for lag in [1, 2, 5]:
            features[f'return_autocorr_lag_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else 0
            )
        
        # Hasbrouck's information share (simplified)
        if len(data) > 100:
            # Price discovery contribution
            price_changes = data['close'].diff()
            for period in [50, 100]:
                # Simplified information share based on price change variance
                total_var = price_changes.rolling(period).var()
                features[f'info_share_{period}'] = (price_changes.rolling(period).var() / 
                                                  total_var.rolling(period).mean())
        
        return features


class LiquidityFeatures:
    """Liquidity and market quality features."""
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity features."""
        features = pd.DataFrame(index=data.index)
        
        # Turnover measures
        if 'market_cap' in data.columns:
            features['turnover_ratio'] = (data['volume'] * data['close']) / data['market_cap']
        else:
            # Proxy using volume and price
            features['dollar_volume'] = data['volume'] * data['close']
            for period in [5, 20]:
                features[f'turnover_proxy_{period}'] = (features['dollar_volume'] / 
                                                       features['dollar_volume'].rolling(period).mean())
        
        # Liquidity ratios
        returns = data['close'].pct_change()
        
        # Martin's liquidity measure
        for period in [20, 50]:
            abs_returns = returns.abs()
            features[f'martin_liquidity_{period}'] = (abs_returns.rolling(period).mean() / 
                                                     (data['volume'].rolling(period).mean() * data['close']))
        
        # Hui-Heubel liquidity ratio
        for period in [20, 50]:
            high_low_ratio = (data['high'] - data['low']) / data['close']
            features[f'hui_heubel_{period}'] = (high_low_ratio.rolling(period).mean() / 
                                              (data['volume'].rolling(period).mean() / 1e6))
        
        # Zero return frequency (Liu's measure)
        for period in [20, 50]:
            zero_returns = (returns == 0).rolling(period).sum()
            features[f'zero_return_freq_{period}'] = zero_returns / period
        
        # Price impact measures
        # Coefficient of Elasticity of Trading (CET)
        for period in [20, 50]:
            vol_change = data['volume'].pct_change()
            price_elasticity = returns.rolling(period).corr(vol_change)
            features[f'price_elasticity_{period}'] = price_elasticity
        
        # Liquidity-adjusted momentum
        for period in [10, 20]:
            momentum = returns.rolling(period).sum()
            avg_volume = data['volume'].rolling(period).mean()
            features[f'liquidity_adj_momentum_{period}'] = momentum / np.log(avg_volume + 1)
        
        # Market depth proxy
        features['depth_proxy'] = data['volume'] / (data['high'] - data['low'])
        for period in [10, 20]:
            features[f'relative_depth_{period}'] = (features['depth_proxy'] / 
                                                   features['depth_proxy'].rolling(period).mean())
        
        return features


class MicrostructureEngine:
    """Main engine for calculating all microstructure features."""
    
    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()
        self.feature_calculators = {
            'spread': BidAskSpreadFeatures(self.config),
            'order_flow': OrderFlowFeatures(self.config),
            'market_impact': MarketImpactFeatures(self.config),
            'liquidity': LiquidityFeatures(self.config)
        }
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all microstructure features."""
        all_features = pd.DataFrame(index=data.index)
        
        for feature_type, calculator in self.feature_calculators.items():
            try:
                features = calculator.calculate(data)
                # Add prefix to avoid column name conflicts
                features.columns = [f"micro_{feature_type}_{col}" for col in features.columns]
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
                calc_features.columns = [f"micro_{feature_type}_{col}" 
                                       for col in calc_features.columns]
                features = pd.concat([features, calc_features], axis=1)
        
        return features
    
    def get_required_columns(self) -> Dict[str, List[str]]:
        """Get required columns for each feature type."""
        return {
            'spread': ['bid_price', 'ask_price', 'bid_size', 'ask_size', 'high', 'low', 'close'],
            'order_flow': ['close', 'volume', 'trade_direction', 'trade_size'],
            'market_impact': ['close', 'volume', 'high', 'low'],
            'liquidity': ['close', 'volume', 'high', 'low', 'market_cap']
        }
"""Enhanced Trading Environment with CNN+LSTM Integration

This module extends the base trading environment to integrate CNN+LSTM
feature extraction, providing rich learned representations for RL agents
instead of basic technical indicators.

Requirements: 1.4, 2.4, 9.1
"""

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import gymnasium as gym
from datetime import datetime

from .trading_environment import TradingEnvironment, TradingConfig, ActionType, MarketState
from .cnn_lstm_feature_extractor import CNNLSTMFeatureExtractor, FeatureExtractionConfig


class EnhancedTradingConfig(TradingConfig):
    """Enhanced configuration for CNN+LSTM integrated trading environment"""
    
    def __init__(self, **kwargs):
        # Extract enhanced-specific kwargs before passing to parent
        enhanced_kwargs = {}
        base_kwargs = {}
        
        enhanced_keys = {
            'hybrid_model_path', 'fused_feature_dim', 'enable_feature_caching',
            'feature_cache_size', 'enable_fallback', 'fallback_feature_dim',
            'include_uncertainty', 'include_ensemble_weights',
            'log_feature_performance', 'performance_log_interval'
        }
        
        for key, value in kwargs.items():
            if key in enhanced_keys:
                enhanced_kwargs[key] = value
            else:
                base_kwargs[key] = value
        
        # Initialize base config
        super().__init__(**base_kwargs)
        
        # CNN+LSTM feature extraction configuration
        self.hybrid_model_path: Optional[str] = enhanced_kwargs.get('hybrid_model_path', None)
        self.fused_feature_dim: int = enhanced_kwargs.get('fused_feature_dim', 256)
        self.enable_feature_caching: bool = enhanced_kwargs.get('enable_feature_caching', True)
        self.feature_cache_size: int = enhanced_kwargs.get('feature_cache_size', 1000)
        self.enable_fallback: bool = enhanced_kwargs.get('enable_fallback', True)
        self.fallback_feature_dim: int = enhanced_kwargs.get('fallback_feature_dim', 15)
        
        # Enhanced observation space configuration
        self.include_uncertainty: bool = enhanced_kwargs.get('include_uncertainty', True)
        self.include_ensemble_weights: bool = enhanced_kwargs.get('include_ensemble_weights', False)
        
        # Performance monitoring
        self.log_feature_performance: bool = enhanced_kwargs.get('log_feature_performance', True)
        self.performance_log_interval: int = enhanced_kwargs.get('performance_log_interval', 100)


class EnhancedTradingEnvironment(TradingEnvironment):
    """Enhanced Trading Environment with CNN+LSTM Feature Integration
    
    This environment extends the base trading environment to leverage
    pre-trained CNN+LSTM hybrid models for rich feature extraction,
    providing RL agents with sophisticated learned representations
    instead of basic technical indicators.
    """
    
    def __init__(
        self,
        market_data,
        config: Optional[EnhancedTradingConfig] = None,
        symbols: Optional[List[str]] = None,
        render_mode: Optional[str] = None
    ):
        # Initialize base environment first
        base_config = TradingConfig() if config is None else config
        super().__init__(market_data, base_config, symbols, render_mode)
        
        # Store enhanced config
        self.enhanced_config = config or EnhancedTradingConfig()
        
        # Initialize CNN+LSTM feature extractor
        self.feature_extractor = self._initialize_feature_extractor()
        
        # Update observation space for enhanced features
        self._setup_enhanced_observation_space()
        
        # Performance tracking
        self.feature_extraction_count = 0
        self.fallback_usage_count = 0
        
    def _initialize_feature_extractor(self) -> CNNLSTMFeatureExtractor:
        """Initialize the CNN+LSTM feature extractor"""
        
        # Create feature extraction configuration
        feature_config = FeatureExtractionConfig(
            hybrid_model_path=self.enhanced_config.hybrid_model_path,
            fused_feature_dim=self.enhanced_config.fused_feature_dim,
            enable_caching=self.enhanced_config.enable_feature_caching,
            cache_size=self.enhanced_config.feature_cache_size,
            enable_fallback=self.enhanced_config.enable_fallback,
            fallback_feature_dim=self.enhanced_config.fallback_feature_dim,
            log_performance=self.enhanced_config.log_feature_performance,
            performance_log_interval=self.enhanced_config.performance_log_interval
        )
        
        # Initialize feature extractor
        extractor = CNNLSTMFeatureExtractor(feature_config)
        
        print(f"Initialized CNN+LSTM feature extractor:")
        print(f"  Model loaded: {extractor.is_model_loaded}")
        print(f"  Fallback enabled: {feature_config.enable_fallback}")
        print(f"  Caching enabled: {feature_config.enable_caching}")
        
        return extractor
    
    def _setup_enhanced_observation_space(self):
        """Setup enhanced observation space with CNN+LSTM features"""
        
        # Get feature dimensions
        feature_dims = self.feature_extractor.get_feature_dimensions()
        
        # Calculate enhanced observation size
        # CNN+LSTM fused features + uncertainty estimates + portfolio state
        cnn_lstm_feature_size = feature_dims['fused_features']  # 256-dim or fallback
        uncertainty_size = 2 if self.enhanced_config.include_uncertainty else 0  # confidence + uncertainty
        
        # Ensemble weights (optional)
        ensemble_size = 0
        if self.enhanced_config.include_ensemble_weights and self.feature_extractor.is_model_loaded:
            ensemble_size = 5  # Assuming 5 ensemble models
        
        # Portfolio features (same as base environment)
        portfolio_obs_size = 3 + self.n_symbols  # cash_ratio, total_value_ratio, drawdown + positions
        
        # Total enhanced observation size
        total_obs_size = cnn_lstm_feature_size + uncertainty_size + ensemble_size + portfolio_obs_size
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        print(f"Enhanced observation space: {total_obs_size} dimensions")
        print(f"  CNN+LSTM features: {cnn_lstm_feature_size}")
        print(f"  Uncertainty features: {uncertainty_size}")
        print(f"  Ensemble features: {ensemble_size}")
        print(f"  Portfolio features: {portfolio_obs_size}")
    
    def _get_market_window(self) -> np.ndarray:
        """Get market data window for feature extraction"""
        
        # Get lookback window data
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step
        
        window_data = self.market_data[
            (self.market_data.index >= start_idx) & 
            (self.market_data.index < end_idx)
        ]
        
        # Prepare market window for all symbols
        market_windows = []
        
        for symbol in self.symbols:
            symbol_data = window_data[window_data['symbol'] == symbol]
            
            # Handle missing data by padding with zeros or last known values
            if len(symbol_data) < self.config.lookback_window:
                padding_length = self.config.lookback_window - len(symbol_data)
                
                if not symbol_data.empty:
                    # Use available data and pad with last known values
                    features = symbol_data[[
                        'open', 'high', 'low', 'close', 'volume',
                        'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
                        'bb_position', 'volume_ratio', 'sma_5', 'sma_20', 'ema_12'
                    ]].fillna(0).values
                    
                    # Pad with last row repeated
                    if len(features) > 0:
                        last_row = features[-1:].repeat(padding_length, axis=0)
                        features = np.vstack([last_row, features])
                    else:
                        features = np.zeros((self.config.lookback_window, 15))
                else:
                    # No data available, use zeros
                    features = np.zeros((self.config.lookback_window, 15))
            else:
                # Sufficient data available
                features = symbol_data[[
                    'open', 'high', 'low', 'close', 'volume',
                    'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
                    'bb_position', 'volume_ratio', 'sma_5', 'sma_20', 'ema_12'
                ]].fillna(0).values
            
            market_windows.append(features)
        
        # Stack all symbols: (num_symbols, sequence_length, features)
        market_window = np.stack(market_windows, axis=0)
        
        # For single symbol, remove the symbol dimension
        if len(self.symbols) == 1:
            market_window = market_window[0]  # (sequence_length, features)
        else:
            # For multiple symbols, flatten or use first symbol for now
            # In practice, you might want to handle multi-symbol differently
            market_window = market_window[0]  # Use first symbol
        
        return market_window
    
    def _get_enhanced_observation(self) -> np.ndarray:
        """Get enhanced observation with CNN+LSTM features"""
        
        try:
            # Get market data window
            market_window = self._get_market_window()
            
            # Extract CNN+LSTM features
            cnn_lstm_features = self.feature_extractor.extract_features(market_window)
            self.feature_extraction_count += 1
            
            # Track fallback usage
            if cnn_lstm_features.get('fallback_used', False):
                self.fallback_usage_count += 1
            
            # Build enhanced observation
            obs_components = []
            
            # 1. CNN+LSTM fused features
            fused_features = cnn_lstm_features['fused_features']
            if fused_features.ndim > 1:
                fused_features = fused_features.flatten()
            obs_components.append(fused_features)
            
            # 2. Uncertainty estimates (optional)
            if self.enhanced_config.include_uncertainty:
                confidence = cnn_lstm_features['classification_confidence']
                uncertainty = cnn_lstm_features['regression_uncertainty']
                
                if confidence.ndim > 0:
                    confidence = confidence.flatten()
                if uncertainty.ndim > 0:
                    uncertainty = uncertainty.flatten()
                
                obs_components.extend([confidence, uncertainty])
            
            # 3. Ensemble weights (optional)
            if (self.enhanced_config.include_ensemble_weights and 
                cnn_lstm_features['ensemble_weights'] is not None):
                ensemble_weights = cnn_lstm_features['ensemble_weights']
                if ensemble_weights.ndim > 1:
                    ensemble_weights = ensemble_weights.flatten()
                obs_components.append(ensemble_weights)
            
            # 4. Portfolio state (same as base environment)
            portfolio_features = self._get_portfolio_features()
            obs_components.append(portfolio_features)
            
            # Combine all components
            enhanced_obs = np.concatenate(obs_components)
            
            # Handle any NaN or inf values
            enhanced_obs = np.nan_to_num(enhanced_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return enhanced_obs.astype(np.float32)
            
        except Exception as e:
            print(f"Enhanced observation failed, using fallback: {e}")
            # Fall back to basic observation
            return self._get_basic_observation()
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Get portfolio state features (same as base environment)"""
        
        # Portfolio ratios
        cash_ratio = self.cash_balance / self.portfolio_value if self.portfolio_value > 0 else 0
        value_ratio = self.portfolio_value / self.initial_balance
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        portfolio_features = [cash_ratio, value_ratio, current_drawdown]
        
        # Position features (normalized by portfolio value)
        current_prices = self._get_current_prices()
        for symbol in self.symbols:
            position_value = self.positions[symbol] * current_prices[symbol]
            position_ratio = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
            portfolio_features.append(position_ratio)
        
        return np.array(portfolio_features, dtype=np.float32)
    
    def _get_basic_observation(self) -> np.ndarray:
        """Get basic observation (fallback to base environment method)"""
        return super()._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Override base method to use enhanced observation"""
        return self._get_enhanced_observation()
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return enhanced observation"""
        
        # Reset base environment
        _, info = super().reset(seed, options)
        
        # Reset feature extractor performance stats if needed
        if hasattr(self.feature_extractor, 'reset_performance_stats'):
            if self.feature_extraction_count > 1000:  # Reset periodically
                self.feature_extractor.reset_performance_stats()
        
        # Reset tracking
        self.feature_extraction_count = 0
        self.fallback_usage_count = 0
        
        # Get enhanced observation
        enhanced_observation = self._get_enhanced_observation()
        
        # Update info with feature extraction status
        info.update({
            'feature_extractor_status': self.feature_extractor.get_status(),
            'enhanced_observation_dim': len(enhanced_observation)
        })
        
        return enhanced_observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step and return enhanced observation"""
        
        # Execute base environment step
        _, reward, terminated, truncated, info = super().step(action)
        
        # Get enhanced observation
        enhanced_observation = self._get_enhanced_observation()
        
        # Update info with feature extraction metrics
        info.update({
            'feature_extraction_count': self.feature_extraction_count,
            'fallback_usage_count': self.fallback_usage_count,
            'fallback_rate': self.fallback_usage_count / max(self.feature_extraction_count, 1)
        })
        
        return enhanced_observation, reward, terminated, truncated, info
    
    def get_feature_extractor_status(self) -> Dict[str, Any]:
        """Get detailed status of the feature extractor"""
        return self.feature_extractor.get_status()
    
    def enable_fallback_mode(self, enable: bool = True) -> None:
        """Enable or disable fallback mode for feature extraction"""
        self.feature_extractor.enable_fallback_mode(enable)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced environment metrics including feature extraction stats"""
        
        base_metrics = self.get_portfolio_metrics()
        
        enhanced_metrics = {
            **base_metrics,
            'feature_extraction_count': self.feature_extraction_count,
            'fallback_usage_count': self.fallback_usage_count,
            'fallback_rate': self.fallback_usage_count / max(self.feature_extraction_count, 1),
            'feature_extractor_status': self.feature_extractor.get_status(),
            'observation_dimension': self.observation_space.shape[0]
        }
        
        return enhanced_metrics
    
    def render(self, mode: str = "human"):
        """Render environment with enhanced information"""
        super().render(mode)
        
        if mode == "human":
            print(f"Feature Extraction Stats:")
            print(f"  Total extractions: {self.feature_extraction_count}")
            print(f"  Fallback usage: {self.fallback_usage_count}")
            print(f"  Fallback rate: {self.fallback_usage_count / max(self.feature_extraction_count, 1):.2%}")
            
            status = self.feature_extractor.get_status()
            print(f"  Model loaded: {status['model_loaded']}")
            print(f"  Cache hits: {status['cache_hits']}")
            print("-" * 50)


def create_enhanced_trading_config(
    # Base trading config
    initial_balance: float = 100000.0,
    max_position_size: float = 0.2,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    lookback_window: int = 60,
    
    # Enhanced features config
    hybrid_model_path: Optional[str] = None,
    fused_feature_dim: int = 256,
    enable_feature_caching: bool = True,
    feature_cache_size: int = 1000,
    enable_fallback: bool = True,
    fallback_feature_dim: int = 15,
    include_uncertainty: bool = True,
    include_ensemble_weights: bool = False,
    
    **kwargs
) -> EnhancedTradingConfig:
    """Create configuration for enhanced trading environment
    
    Args:
        Base trading environment parameters and enhanced feature parameters
        
    Returns:
        EnhancedTradingConfig object
    """
    
    config_dict = {
        # Base config
        'initial_balance': initial_balance,
        'max_position_size': max_position_size,
        'transaction_cost': transaction_cost,
        'slippage': slippage,
        'lookback_window': lookback_window,
        
        # Enhanced config
        'hybrid_model_path': hybrid_model_path,
        'fused_feature_dim': fused_feature_dim,
        'enable_feature_caching': enable_feature_caching,
        'feature_cache_size': feature_cache_size,
        'enable_fallback': enable_fallback,
        'fallback_feature_dim': fallback_feature_dim,
        'include_uncertainty': include_uncertainty,
        'include_ensemble_weights': include_ensemble_weights,
        
        **kwargs
    }
    
    return EnhancedTradingConfig(**config_dict)
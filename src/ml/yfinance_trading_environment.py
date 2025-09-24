"""
YFinance Trading Environment for Reinforcement Learning.

This module implements a specialized trading environment using real yfinance data
with advanced features for RL training including market regime detection,
realistic transaction costs, and comprehensive state representation.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, List, Union
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler

from data.ingestion.yahoo_finance import YahooFinanceIngestor

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class MarketRegime(Enum):
    """Market regime types for environment adaptation."""
    BULL_MARKET = 0
    BEAR_MARKET = 1
    SIDEWAYS_MARKET = 2
    HIGH_VOLATILITY = 3


@dataclass
class YFinanceConfig:
    """Configuration for YFinance trading environment."""
    # Portfolio settings
    initial_balance: float = 100000.0
    max_position_size: float = 0.2  # Max 20% of portfolio per position
    max_total_exposure: float = 0.8  # Max 80% total exposure
    
    # Transaction costs and slippage (realistic for retail trading)
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage_base: float = 0.0005  # 0.05% base slippage
    slippage_impact: float = 0.0001  # Additional slippage based on position size
    
    # Risk management
    max_drawdown_limit: float = 0.05  # Max 5% drawdown before episode ends
    stop_loss_threshold: float = 0.03  # 3% stop loss per position
    position_timeout: int = 100  # Max steps to hold a position
    
    # Environment settings
    lookback_window: int = 60  # Historical periods in observation
    prediction_horizon: int = 5  # Steps ahead to predict
    min_episode_length: int = 200  # Minimum episode length
    
    # Reward function parameters
    risk_free_rate: float = 0.02  # Annual risk-free rate
    reward_scaling: float = 10.0  # Scale rewards for better learning
    sharpe_weight: float = 0.3  # Weight for Sharpe ratio in reward
    return_weight: float = 0.4  # Weight for returns in reward
    drawdown_penalty: float = 2.0  # Penalty multiplier for drawdowns
    
    # Market regime detection
    volatility_window: int = 20  # Window for volatility calculation
    trend_window: int = 50  # Window for trend detection
    regime_threshold: float = 0.02  # Threshold for regime classification
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        if not 0 < self.max_total_exposure <= 1:
            raise ValueError("Max total exposure must be between 0 and 1")
        if self.transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")


@dataclass
class MarketState:
    """Enhanced market state with regime information."""
    timestamp: datetime
    prices: Dict[str, float]
    volumes: Dict[str, float]
    returns: Dict[str, float]
    volatilities: Dict[str, float]
    regime: MarketRegime
    regime_confidence: float
    technical_indicators: Dict[str, Dict[str, float]]


class YFinanceTradingEnvironment(gym.Env):
    """
    Advanced trading environment using real yfinance data.
    
    Features:
    - Real market data from yfinance with multiple symbols
    - Market regime detection and adaptation
    - Realistic transaction costs and slippage modeling
    - Advanced risk management and position constraints
    - Multi-objective reward function with risk adjustment
    - Comprehensive state representation with technical indicators
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        config: Optional[YFinanceConfig] = None,
        data_source: str = "yfinance",  # "yfinance" or "cached"
        cache_dir: str = "data/cache",
        render_mode: Optional[str] = None
    ):
        """
        Initialize YFinance trading environment.
        
        Args:
            symbols: List of stock symbols to trade
            start_date: Start date for data in YYYY-MM-DD format
            end_date: End date for data in YYYY-MM-DD format
            config: Environment configuration
            data_source: Data source ("yfinance" or "cached")
            cache_dir: Directory for cached data
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        self.config = config or YFinanceConfig()
        self.render_mode = render_mode
        
        # Default symbols - liquid stocks for reliable data
        if symbols is None:
            self.symbols = [
                "SPY",   # S&P 500 ETF
                "QQQ",   # Nasdaq 100 ETF
                "AAPL",  # Apple
                "MSFT",  # Microsoft
                "GOOGL", # Google
                "AMZN",  # Amazon
                "TSLA",  # Tesla
                "NVDA",  # NVIDIA
                "META",  # Meta
                "NFLX"   # Netflix
            ]
        else:
            self.symbols = symbols
        
        self.n_symbols = len(self.symbols)
        self.start_date = start_date
        self.end_date = end_date
        
        # Load market data
        logger.info(f"Loading market data for {self.n_symbols} symbols from {start_date} to {end_date}")
        self.market_data = self._load_market_data(data_source, cache_dir)
        
        if self.market_data.empty:
            raise ValueError("No market data loaded. Check symbols and date range.")
        
        # Prepare data for training
        self.processed_data = self._prepare_data()
        self.scalers = self._fit_scalers()
        
        # Environment state
        self.current_step = 0
        self.episode_start_step = 0
        self.max_steps = len(self.processed_data) - self.config.lookback_window - self.config.prediction_horizon
        
        # Portfolio state
        self.reset_portfolio()
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[float] = []
        self.returns_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.position_ages: Dict[str, int] = {symbol: 0 for symbol in self.symbols}
        
        # Market regime tracking
        self.regime_history: List[MarketRegime] = []
        self.regime_detector = MarketRegimeDetector(self.config)
        
        # Define action and observation spaces
        self._setup_spaces()
        
        logger.info(f"YFinance environment initialized with {self.max_steps} steps")
    
    def _load_market_data(self, data_source: str, cache_dir: str) -> pd.DataFrame:
        """Load market data from yfinance or cache."""
        if data_source == "yfinance":
            ingestor = YahooFinanceIngestor(cache_dir)
            data_dict = ingestor.fetch_multiple_symbols(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                interval="1d",  # Daily data for stability
                cache=True
            )
            
            # Combine all symbol data
            all_data = []
            for symbol, data in data_dict.items():
                if data is not None and not data.empty:
                    # Ensure consistent column names
                    data = data.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'volume'
                    })
                    data['symbol'] = symbol
                    data['timestamp'] = data.index
                    all_data.append(data[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
                logger.info(f"Loaded {len(combined_data)} rows of market data")
                return combined_data
            else:
                logger.error("No data loaded from yfinance")
                return pd.DataFrame()
        
        else:
            # Load from cached files (implementation would depend on cache structure)
            logger.warning("Cached data loading not implemented, using yfinance")
            return self._load_market_data("yfinance", cache_dir)
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare and enhance market data with technical indicators."""
        logger.info("Preparing market data with technical indicators...")
        
        enhanced_data = []
        
        for symbol in self.symbols:
            symbol_data = self.market_data[self.market_data['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                logger.warning(f"No data for symbol {symbol}")
                continue
            
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
            
            # Basic price features
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
            symbol_data['price_change'] = symbol_data['close'] - symbol_data['open']
            symbol_data['price_range'] = symbol_data['high'] - symbol_data['low']
            symbol_data['body_size'] = abs(symbol_data['close'] - symbol_data['open'])
            
            # Moving averages
            for period in [5, 10, 12, 20, 26, 50]:
                symbol_data[f'sma_{period}'] = symbol_data['close'].rolling(period).mean()
                symbol_data[f'ema_{period}'] = symbol_data['close'].ewm(span=period).mean()
            
            # Price relative to moving averages
            symbol_data['price_vs_sma20'] = symbol_data['close'] / symbol_data['sma_20'] - 1
            symbol_data['price_vs_sma50'] = symbol_data['close'] / symbol_data['sma_50'] - 1
            
            # Volatility measures
            symbol_data['volatility'] = symbol_data['returns'].rolling(self.config.volatility_window).std()
            symbol_data['volatility_ratio'] = (
                symbol_data['volatility'] / symbol_data['volatility'].rolling(50).mean()
            )
            
            # RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            symbol_data['bb_middle'] = symbol_data['close'].rolling(bb_period).mean()
            bb_std_dev = symbol_data['close'].rolling(bb_period).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (bb_std_dev * bb_std)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (bb_std_dev * bb_std)
            symbol_data['bb_position'] = (
                (symbol_data['close'] - symbol_data['bb_lower']) / 
                (symbol_data['bb_upper'] - symbol_data['bb_lower'])
            )
            symbol_data['bb_width'] = (symbol_data['bb_upper'] - symbol_data['bb_lower']) / symbol_data['bb_middle']
            
            # Volume indicators
            symbol_data['volume_sma'] = symbol_data['volume'].rolling(20).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma']
            symbol_data['volume_price_trend'] = (
                symbol_data['volume'] * symbol_data['returns']
            ).rolling(10).sum()
            
            # Momentum indicators
            symbol_data['momentum_5'] = symbol_data['close'] / symbol_data['close'].shift(5) - 1
            symbol_data['momentum_10'] = symbol_data['close'] / symbol_data['close'].shift(10) - 1
            symbol_data['momentum_20'] = symbol_data['close'] / symbol_data['close'].shift(20) - 1
            
            # Support and resistance levels
            symbol_data['high_20'] = symbol_data['high'].rolling(20).max()
            symbol_data['low_20'] = symbol_data['low'].rolling(20).min()
            symbol_data['price_position'] = (
                (symbol_data['close'] - symbol_data['low_20']) / 
                (symbol_data['high_20'] - symbol_data['low_20'])
            )
            
            enhanced_data.append(symbol_data)
        
        if enhanced_data:
            combined_data = pd.concat(enhanced_data, ignore_index=True)
            combined_data = combined_data.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
            
            # Fill NaN values
            combined_data = combined_data.fillna(method='ffill').fillna(0)
            
            logger.info(f"Enhanced data prepared with {len(combined_data)} rows and {len(combined_data.columns)} features")
            return combined_data
        else:
            return pd.DataFrame()
    
    def _fit_scalers(self) -> Dict[str, StandardScaler]:
        """Fit scalers for feature normalization."""
        scalers = {}
        
        # Features to scale
        price_features = ['open', 'high', 'low', 'close']
        volume_features = ['volume', 'volume_sma']
        technical_features = [
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width', 'volume_ratio', 'volume_price_trend',
            'momentum_5', 'momentum_10', 'momentum_20', 'price_position',
            'price_vs_sma20', 'price_vs_sma50', 'volatility_ratio'
        ]
        
        for feature_group, features in [
            ('price', price_features),
            ('volume', volume_features), 
            ('technical', technical_features)
        ]:
            available_features = [f for f in features if f in self.processed_data.columns]
            if available_features:
                scaler = RobustScaler()  # More robust to outliers
                scaler.fit(self.processed_data[available_features])
                scalers[feature_group] = scaler
        
        return scalers
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [action_type, position_size] for each symbol
        # action_type: 0=HOLD, 1=BUY, 2=SELL
        # position_size: 0.0 to max_position_size
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0] * self.n_symbols),
            high=np.array([2.0, self.config.max_position_size] * self.n_symbols),
            dtype=np.float32
        )
        
        # Observation space calculation
        # Market features per symbol per timestep (count actual features)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'bb_width', 'volume_ratio', 'momentum_5', 'momentum_10',
            'price_vs_sma20', 'price_vs_sma50', 'volatility_ratio', 'price_position',
            'volume_price_trend', 'macd_histogram', 'price_change', 'price_range',
            'body_size', 'momentum_20'
        ]
        features_per_symbol = len(feature_columns)  # Actual count: 24
        market_obs_size = features_per_symbol * self.n_symbols * self.config.lookback_window
        
        # Portfolio state features
        portfolio_features = 4 + self.n_symbols  # cash_ratio, value_ratio, drawdown, exposure + positions
        
        # Market regime features
        regime_features = 4  # One-hot encoded regime
        
        total_obs_size = market_obs_size + portfolio_features + regime_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        logger.info(f"Action space: {self.action_space.shape}, Observation space: {self.observation_space.shape}")
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.cash_balance = self.config.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.config.initial_balance
        self.max_portfolio_value = self.config.initial_balance
        self.position_ages = {symbol: 0 for symbol in self.symbols}
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to random starting point (but ensure minimum episode length)
        max_start = self.max_steps - self.config.min_episode_length
        self.episode_start_step = np.random.randint(
            self.config.lookback_window, 
            max(self.config.lookback_window + 1, max_start)
        )
        self.current_step = self.episode_start_step
        
        # Reset portfolio
        self.reset_portfolio()
        
        # Reset tracking
        self.trade_history = []
        self.portfolio_history = []
        self.returns_history = []
        self.drawdown_history = []
        self.regime_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset at step {self.current_step}")
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Parse and validate actions
        actions = self._parse_action(action)
        
        # Execute trades
        trade_results = self._execute_trades(actions)
        
        # Update portfolio value and positions
        self._update_portfolio_value()
        self._update_position_ages()
        
        # Detect market regime
        current_regime = self._detect_market_regime()
        self.regime_history.append(current_regime)
        
        # Calculate reward
        reward = self._calculate_reward(trade_results, current_regime)
        
        # Update step
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Record step data
        self._record_step(actions, reward, trade_results, current_regime)
        
        return observation, reward, terminated, truncated, info
    
    def _parse_action(self, action: np.ndarray) -> List[Dict]:
        """Parse and validate action array."""
        actions = []
        
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 2
            action_type = int(np.clip(action[action_idx], 0, 2))
            position_size = np.clip(action[action_idx + 1], 0, self.config.max_position_size)
            
            actions.append({
                'symbol': symbol,
                'action_type': action_type,  # 0=HOLD, 1=BUY, 2=SELL
                'position_size': position_size
            })
        
        return actions
    
    def _execute_trades(self, actions: List[Dict]) -> List[Dict]:
        """Execute trades with realistic costs and constraints."""
        trade_results = []
        current_prices = self._get_current_prices()
        current_exposure = self._calculate_current_exposure(current_prices)
        
        for action in actions:
            symbol = action['symbol']
            action_type = action['action_type']
            position_size = action['position_size']
            current_price = current_prices[symbol]
            
            result = {
                'symbol': symbol,
                'action_type': action_type,
                'executed': False,
                'quantity': 0.0,
                'price': current_price,
                'cost': 0.0,
                'slippage': 0.0,
                'reason': 'hold'
            }
            
            if action_type == 0:  # HOLD
                trade_results.append(result)
                continue
            
            # Check position timeout (force close old positions)
            if (action_type == 1 and self.position_ages[symbol] > self.config.position_timeout):
                action_type = 2  # Force sell
                result['reason'] = 'timeout_forced_sell'
            
            # Execute BUY orders
            if action_type == 1:
                # Check exposure limits
                position_value = self.cash_balance * position_size
                if current_exposure + (position_value / self.portfolio_value) > self.config.max_total_exposure:
                    result['reason'] = 'exposure_limit'
                    trade_results.append(result)
                    continue
                
                max_shares = position_value / current_price
                shares_to_buy = max_shares
                
                if shares_to_buy > 0 and position_value > 100:  # Minimum trade size
                    # Calculate slippage based on position size
                    size_impact = (position_value / self.portfolio_value) * self.config.slippage_impact
                    total_slippage = self.config.slippage_base + size_impact
                    execution_price = current_price * (1 + total_slippage)
                    
                    # Calculate total cost
                    gross_cost = shares_to_buy * execution_price
                    transaction_cost = gross_cost * self.config.transaction_cost
                    total_cost = gross_cost + transaction_cost
                    
                    if total_cost <= self.cash_balance:
                        # Execute trade
                        self.cash_balance -= total_cost
                        self.positions[symbol] += shares_to_buy
                        self.position_ages[symbol] = 0  # Reset age
                        
                        result.update({
                            'executed': True,
                            'quantity': shares_to_buy,
                            'price': execution_price,
                            'cost': total_cost,
                            'slippage': total_slippage,
                            'reason': 'buy_executed'
                        })
                    else:
                        result['reason'] = 'insufficient_cash'
                else:
                    result['reason'] = 'min_trade_size'
            
            # Execute SELL orders
            elif action_type == 2:
                current_position = self.positions[symbol]
                shares_to_sell = current_position * position_size
                
                if shares_to_sell > 0:
                    # Calculate slippage
                    position_value = shares_to_sell * current_price
                    size_impact = (position_value / self.portfolio_value) * self.config.slippage_impact
                    total_slippage = self.config.slippage_base + size_impact
                    execution_price = current_price * (1 - total_slippage)
                    
                    # Calculate proceeds
                    gross_proceeds = shares_to_sell * execution_price
                    transaction_cost = gross_proceeds * self.config.transaction_cost
                    net_proceeds = gross_proceeds - transaction_cost
                    
                    # Execute trade
                    self.cash_balance += net_proceeds
                    self.positions[symbol] -= shares_to_sell
                    
                    # Reset age if position fully closed
                    if self.positions[symbol] < 1e-6:
                        self.positions[symbol] = 0.0
                        self.position_ages[symbol] = 0
                    
                    result.update({
                        'executed': True,
                        'quantity': -shares_to_sell,
                        'price': execution_price,
                        'cost': -net_proceeds,
                        'slippage': total_slippage,
                        'reason': 'sell_executed'
                    })
                else:
                    result['reason'] = 'no_position'
            
            trade_results.append(result)
        
        return trade_results
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        current_data = self.processed_data[
            self.processed_data.index.isin(
                range(self.current_step * self.n_symbols, 
                     (self.current_step + 1) * self.n_symbols)
            )
        ]
        
        prices = {}
        for symbol in self.symbols:
            symbol_data = current_data[current_data['symbol'] == symbol]
            if not symbol_data.empty:
                prices[symbol] = symbol_data['close'].iloc[0]
            else:
                # Fallback to last known price
                try:
                    mask = (self.processed_data.index < self.current_step * self.n_symbols) & (self.processed_data['symbol'] == symbol)
                    last_data = self.processed_data.loc[mask]
                except Exception:
                    # If indexing fails, use iloc for safety
                    symbol_mask = self.processed_data['symbol'] == symbol
                    symbol_data_all = self.processed_data.loc[symbol_mask]
                    if len(symbol_data_all) > 0:
                        last_data = symbol_data_all.iloc[:min(self.current_step, len(symbol_data_all))]
                    else:
                        last_data = pd.DataFrame()
                if not last_data.empty:
                    prices[symbol] = last_data['close'].iloc[-1]
                else:
                    prices[symbol] = 100.0  # Default fallback
        
        return prices
    
    def _calculate_current_exposure(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio exposure."""
        total_position_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )
        return total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        current_prices = self._get_current_prices()
        
        positions_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )
        
        self.portfolio_value = self.cash_balance + positions_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
    
    def _update_position_ages(self):
        """Update position ages for timeout management."""
        for symbol in self.symbols:
            if self.positions[symbol] > 0:
                self.position_ages[symbol] += 1
    
    def _detect_market_regime(self) -> MarketRegime:
        """Detect current market regime."""
        return self.regime_detector.detect_regime(
            self.processed_data, 
            self.current_step, 
            self.symbols
        )
    
    def _calculate_reward(self, trade_results: List[Dict], current_regime: MarketRegime) -> float:
        """Calculate multi-objective reward with risk adjustment."""
        # Portfolio return component (step-wise return)
        if len(self.portfolio_history) > 0:
            portfolio_return = (self.portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
        else:
            portfolio_return = (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance
        
        # Sharpe ratio component (rolling)
        sharpe_component = 0.0
        if len(self.returns_history) >= 10:
            recent_returns = np.array(self.returns_history[-20:])
            excess_returns = recent_returns - (self.config.risk_free_rate / 252)
            if np.std(excess_returns) > 0:
                sharpe_component = np.mean(excess_returns) / np.std(excess_returns)
        
        # Drawdown penalty (much smaller)
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -current_drawdown * 0.1  # Reduced from 2.0 to 0.1
        
        # Transaction cost penalty (simplified)
        transaction_penalty = -len([r for r in trade_results if r['executed']]) * 0.01
        
        # Risk management bonus/penalty
        risk_penalty = 0.0
        current_exposure = self._calculate_current_exposure(self._get_current_prices())
        if current_exposure > self.config.max_total_exposure:
            risk_penalty = -0.01  # Small penalty for over-exposure
        
        # Market regime adaptation bonus
        regime_bonus = 0.0
        if current_regime == MarketRegime.HIGH_VOLATILITY and current_exposure < 0.5:
            regime_bonus = 0.01  # Small bonus for reducing exposure in volatile markets
        elif current_regime == MarketRegime.BULL_MARKET and current_exposure > 0.6:
            regime_bonus = 0.005  # Small bonus for higher exposure in bull markets
        
        # Very simple reward: just portfolio return with minimal penalties
        reward = (
            portfolio_return * 10.0 +   # Reduced scaling from 100 to 10
            sharpe_component * 0.01 +   # Minimal Sharpe component
            drawdown_penalty * 0.1 +    # Reduced drawdown penalty
            transaction_penalty * 0.1   # Reduced transaction penalty
            # Removed risk and regime bonuses for simplicity
        )
        
        # Optional debug logging (disabled for performance)
        # if self.current_step < 5:
        #     print(f"Step {self.current_step}: portfolio_return={portfolio_return:.6f}, "
        #           f"reward_component={portfolio_return * 100.0:.2f}, "
        #           f"total_reward={reward:.2f}")
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check termination conditions."""
        # Maximum drawdown exceeded (tighter limit)
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        if current_drawdown > self.config.max_drawdown_limit:
            return True
        
        # Portfolio value too low (less strict)
        if self.portfolio_value < self.config.initial_balance * 0.8:  # Stop at 20% loss
            return True
        
        # Episode too long (prevent infinite episodes)
        if self.current_step > 500:  # Limit episodes to 500 steps max
            return True
        
        # Check individual position stop losses
        current_prices = self._get_current_prices()
        for symbol in self.symbols:
            if self.positions[symbol] > 0:
                position_value = self.positions[symbol] * current_prices[symbol]
                # This is a simplified stop loss check - in practice you'd track entry prices
                if position_value < self.positions[symbol] * current_prices[symbol] * (1 - self.config.stop_loss_threshold):
                    return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get comprehensive observation state."""
        # Market features for lookback window
        market_features = []
        
        start_idx = max(0, (self.current_step - self.config.lookback_window) * self.n_symbols)
        end_idx = self.current_step * self.n_symbols
        
        window_data = self.processed_data.iloc[start_idx:end_idx]
        
        # Features to include in observation
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'bb_width', 'volume_ratio', 'momentum_5', 'momentum_10',
            'price_vs_sma20', 'price_vs_sma50', 'volatility_ratio', 'price_position',
            'volume_price_trend', 'macd_histogram', 'price_change', 'price_range',
            'body_size', 'momentum_20'
        ]
        
        for symbol in self.symbols:
            symbol_data = window_data[window_data['symbol'] == symbol]
            
            # Pad with zeros if insufficient data
            if len(symbol_data) < self.config.lookback_window:
                padding_length = self.config.lookback_window - len(symbol_data)
                padding = np.zeros((padding_length, len(feature_columns)))
                
                if not symbol_data.empty:
                    available_features = [f for f in feature_columns if f in symbol_data.columns]
                    features = symbol_data[available_features].fillna(0).values
                    # Pad missing features
                    if len(available_features) < len(feature_columns):
                        feature_padding = np.zeros((len(features), len(feature_columns) - len(available_features)))
                        features = np.hstack([features, feature_padding])
                    features = np.vstack([padding, features])
                else:
                    features = padding
            else:
                available_features = [f for f in feature_columns if f in symbol_data.columns]
                features = symbol_data[available_features].fillna(0).values
                # Pad missing features
                if len(available_features) < len(feature_columns):
                    feature_padding = np.zeros((len(features), len(feature_columns) - len(available_features)))
                    features = np.hstack([features, feature_padding])
            
            market_features.extend(features.flatten())
        
        # Portfolio state features
        current_prices = self._get_current_prices()
        cash_ratio = self.cash_balance / self.portfolio_value if self.portfolio_value > 0 else 0
        value_ratio = self.portfolio_value / self.config.initial_balance
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        current_exposure = self._calculate_current_exposure(current_prices)
        
        portfolio_features = [cash_ratio, value_ratio, current_drawdown, current_exposure]
        
        # Position features (normalized)
        for symbol in self.symbols:
            position_value = self.positions[symbol] * current_prices[symbol]
            position_ratio = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
            portfolio_features.append(position_ratio)
        
        # Market regime features (one-hot encoded)
        current_regime = self._detect_market_regime()
        regime_features = [0.0] * 4
        regime_features[current_regime.value] = 1.0
        
        # Combine all features
        observation = np.array(
            market_features + portfolio_features + regime_features, 
            dtype=np.float32
        )
        
        # Handle NaN/inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get comprehensive environment info."""
        current_prices = self._get_current_prices()
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        current_exposure = self._calculate_current_exposure(current_prices)
        current_regime = self._detect_market_regime()
        
        return {
            'step': self.current_step,
            'episode_step': self.current_step - self.episode_start_step,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'positions': self.positions.copy(),
            'position_ages': self.position_ages.copy(),
            'current_prices': current_prices,
            'drawdown': current_drawdown,
            'max_portfolio_value': self.max_portfolio_value,
            'total_return': (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance,
            'exposure': current_exposure,
            'market_regime': current_regime.name,
            'num_trades': len(self.trade_history),
            'symbols': self.symbols
        }
    
    def _record_step(self, actions: List[Dict], reward: float, trade_results: List[Dict], regime: MarketRegime):
        """Record step information for analysis."""
        portfolio_return = (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance
        self.returns_history.append(portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        self.drawdown_history.append(current_drawdown)
        
        # Record executed trades
        for result in trade_results:
            if result['executed']:
                self.trade_history.append({
                    'step': self.current_step,
                    'timestamp': datetime.now(),
                    'symbol': result['symbol'],
                    'action': result['action_type'],
                    'quantity': result['quantity'],
                    'price': result['price'],
                    'cost': result['cost'],
                    'slippage': result['slippage'],
                    'portfolio_value': self.portfolio_value,
                    'regime': regime.name,
                    'reason': result['reason']
                })
    
    def render(self, mode: str = "human"):
        """Render environment state."""
        if mode == "human":
            current_regime = self._detect_market_regime()
            current_exposure = self._calculate_current_exposure(self._get_current_prices())
            
            print(f"Step: {self.current_step} (Episode: {self.current_step - self.episode_start_step})")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash Balance: ${self.cash_balance:,.2f}")
            print(f"Total Return: {((self.portfolio_value - self.config.initial_balance) / self.config.initial_balance) * 100:.2f}%")
            print(f"Current Exposure: {current_exposure:.1%}")
            print(f"Market Regime: {current_regime.name}")
            print(f"Active Positions: {sum(1 for pos in self.positions.values() if pos > 0)}")
            print(f"Total Trades: {len(self.trade_history)}")
            print("-" * 60)
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        if len(self.returns_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_history)
        
        # Basic metrics
        total_return = returns[-1]
        days_elapsed = len(returns)
        annualized_return = (1 + total_return) ** (252 / days_elapsed) - 1
        
        # Risk metrics
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Drawdown analysis
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0
        
        # Risk-adjusted returns
        excess_returns = daily_returns - (self.config.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = (
            np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        )
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trading metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['cost'] < 0)  # Sells with profit
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_portfolio_value': self.portfolio_value,
            'days_elapsed': days_elapsed
        }


class MarketRegimeDetector:
    """Detect market regimes based on price and volatility patterns."""
    
    def __init__(self, config: YFinanceConfig):
        self.config = config
    
    def detect_regime(
        self, 
        data: pd.DataFrame, 
        current_step: int, 
        symbols: List[str]
    ) -> MarketRegime:
        """Detect current market regime."""
        # Get recent data for analysis
        lookback_steps = self.config.trend_window
        start_idx = max(0, (current_step - lookback_steps) * len(symbols))
        end_idx = current_step * len(symbols)
        
        recent_data = data.iloc[start_idx:end_idx]
        
        if recent_data.empty:
            return MarketRegime.SIDEWAYS_MARKET
        
        # Calculate aggregate market metrics
        market_returns = []
        market_volatilities = []
        
        for symbol in symbols:
            symbol_data = recent_data[recent_data['symbol'] == symbol]
            if not symbol_data.empty and 'returns' in symbol_data.columns:
                returns = symbol_data['returns'].dropna()
                if len(returns) > 0:
                    market_returns.extend(returns.tolist())
                    if 'volatility' in symbol_data.columns:
                        volatilities = symbol_data['volatility'].dropna()
                        if len(volatilities) > 0:
                            market_volatilities.extend(volatilities.tolist())
        
        if not market_returns:
            return MarketRegime.SIDEWAYS_MARKET
        
        # Calculate regime indicators
        avg_return = np.mean(market_returns)
        avg_volatility = np.mean(market_volatilities) if market_volatilities else np.std(market_returns)
        
        # Regime classification logic
        high_vol_threshold = self.config.regime_threshold * 2
        trend_threshold = self.config.regime_threshold
        
        if avg_volatility > high_vol_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif avg_return > trend_threshold:
            return MarketRegime.BULL_MARKET
        elif avg_return < -trend_threshold:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET


# Factory function for easy environment creation
def create_yfinance_environment(
    symbols: List[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    config: Optional[YFinanceConfig] = None,
    **kwargs
) -> YFinanceTradingEnvironment:
    """
    Create a YFinance trading environment with sensible defaults.
    
    Args:
        symbols: List of stock symbols to trade
        start_date: Start date for data
        end_date: End date for data
        config: Environment configuration
        **kwargs: Additional arguments passed to environment
    
    Returns:
        Configured YFinanceTradingEnvironment
    """
    if config is None:
        config = YFinanceConfig()
    
    return YFinanceTradingEnvironment(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        config=config,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create environment with default settings
    env = create_yfinance_environment(
        symbols=["SPY", "QQQ", "AAPL"],
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    # Test environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}")
        
        if terminated or truncated:
            break
    
    # Get final metrics
    metrics = env.get_portfolio_metrics()
    print(f"Final metrics: {metrics}")
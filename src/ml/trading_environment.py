"""
Trading Environment for Reinforcement Learning.

This module implements a Gymnasium-compatible trading environment that simulates
realistic market conditions for training RL agents.
"""

from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, List

import gymnasium as gym
import numpy as np
import pandas as pd


class ActionType(Enum):
    """Trading action types for the environment."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingConfig:
    """Configuration for the trading environment."""
    initial_balance: float = 100000.0
    max_position_size: float = 0.2  # Max 20% of portfolio per position
    transaction_cost: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.0005  # 0.05% slippage
    lookback_window: int = 60  # Historical periods in observation
    max_drawdown_limit: float = 0.2  # Max 20% drawdown before episode ends
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe calc
    reward_scaling: float = 1000.0  # Scale rewards for better learning
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        if self.transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if self.slippage < 0:
            raise ValueError("Slippage cannot be negative")
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")
        if not 0 < self.max_drawdown_limit <= 1:
            raise ValueError("Max drawdown limit must be between 0 and 1")


@dataclass
class MarketState:
    """Current market state information."""
    current_price: float
    volume: float
    volatility: float
    returns: float
    technical_indicators: Dict[str, float]
    timestamp: datetime


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment for RL training.
    
    This environment simulates realistic trading conditions including:
    - Transaction costs and slippage
    - Risk-adjusted reward functions
    - Portfolio state representation
    - Dynamic market conditions
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        market_data: pd.DataFrame,
        config: Optional[TradingConfig] = None,
        symbols: Optional[List[str]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            market_data: Historical market data with OHLCV columns
            config: Trading configuration parameters
            symbols: List of symbols to trade (if None, uses all symbols in data)
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        self.config = config or TradingConfig()
        self.render_mode = render_mode
        
        # Prepare market data
        self.market_data = self._prepare_market_data(market_data)
        self.symbols = symbols or self.market_data['symbol'].unique().tolist()
        self.n_symbols = len(self.symbols)
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(self.market_data) - self.config.lookback_window - 1
        
        # Portfolio state
        self.initial_balance = self.config.initial_balance
        self.cash_balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[float] = []
        self.returns_history: List[float] = []
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Initialize market state
        self.market_state = None
        
    def _prepare_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate market data."""
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Market data must contain columns: {required_columns}")
        
        # Sort by timestamp and symbol
        data = data.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        # Calculate returns and technical indicators
        data = self._calculate_features(data)
        
        return data
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features."""
        features_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Price-based features
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
            
            # Moving averages
            symbol_data['sma_5'] = symbol_data['close'].rolling(5).mean()
            symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
            symbol_data['ema_12'] = symbol_data['close'].ewm(span=12).mean()
            symbol_data['ema_26'] = symbol_data['close'].ewm(span=26).mean()
            
            # Volatility
            symbol_data['volatility'] = symbol_data['returns'].rolling(20).std()
            
            # RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            symbol_data['bb_middle'] = symbol_data['close'].rolling(bb_period).mean()
            bb_std_dev = symbol_data['close'].rolling(bb_period).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (bb_std_dev * bb_std)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (bb_std_dev * bb_std)
            symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / (symbol_data['bb_upper'] - symbol_data['bb_lower'])
            
            # Volume indicators
            symbol_data['volume_sma'] = symbol_data['volume'].rolling(20).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma']
            
            features_data.append(symbol_data)
        
        return pd.concat(features_data, ignore_index=True)
    
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
        
        # Observation space includes:
        # - Market features for each symbol (price, volume, technical indicators)
        # - Portfolio state (cash, positions, total value)
        # - Historical lookback window
        
        # Market features per symbol
        market_features_per_symbol = 15  # OHLCV + technical indicators
        market_obs_size = market_features_per_symbol * self.n_symbols * self.config.lookback_window
        
        # Portfolio features
        portfolio_obs_size = 3 + self.n_symbols  # cash_ratio, total_value_ratio, drawdown + positions
        
        total_obs_size = market_obs_size + portfolio_obs_size
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset environment state
        self.current_step = self.config.lookback_window
        
        # Reset portfolio
        self.cash_balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        
        # Reset tracking
        self.trade_history = []
        self.portfolio_history = []
        self.returns_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Parse actions
        actions = self._parse_action(action)
        
        # Execute trades
        trade_results = self._execute_trades(actions)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Calculate reward
        reward = self._calculate_reward(trade_results)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Record history
        self._record_step(actions, reward, trade_results)
        
        return observation, reward, terminated, truncated, info
    
    def _parse_action(self, action: np.ndarray) -> List[Dict]:
        """Parse the action array into trading actions for each symbol."""
        actions = []
        
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 2
            action_type = int(np.clip(action[action_idx], 0, 2))
            position_size = np.clip(action[action_idx + 1], 0, self.config.max_position_size)
            
            actions.append({
                'symbol': symbol,
                'action_type': ActionType(action_type),
                'position_size': position_size
            })
        
        return actions
    
    def _execute_trades(self, actions: List[Dict]) -> List[Dict]:
        """Execute trading actions and return results."""
        trade_results = []
        current_prices = self._get_current_prices()
        
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
                'slippage': 0.0
            }
            
            if action_type == ActionType.HOLD:
                trade_results.append(result)
                continue
            
            # Calculate trade details
            if action_type == ActionType.BUY:
                max_shares = (self.cash_balance * position_size) / current_price
                shares_to_buy = max_shares
                
                if shares_to_buy > 0:
                    # Apply slippage (price moves against us)
                    slippage = current_price * self.config.slippage
                    execution_price = current_price + slippage
                    
                    # Calculate total cost including transaction costs
                    gross_cost = shares_to_buy * execution_price
                    transaction_cost = gross_cost * self.config.transaction_cost
                    total_cost = gross_cost + transaction_cost
                    
                    if total_cost <= self.cash_balance:
                        # Execute trade
                        self.cash_balance -= total_cost
                        self.positions[symbol] += shares_to_buy
                        
                        result.update({
                            'executed': True,
                            'quantity': shares_to_buy,
                            'price': execution_price,
                            'cost': total_cost,
                            'slippage': slippage
                        })
            
            elif action_type == ActionType.SELL:
                current_position = self.positions[symbol]
                shares_to_sell = min(current_position, current_position * position_size)
                
                if shares_to_sell > 0:
                    # Apply slippage (price moves against us)
                    slippage = current_price * self.config.slippage
                    execution_price = current_price - slippage
                    
                    # Calculate proceeds after transaction costs
                    gross_proceeds = shares_to_sell * execution_price
                    transaction_cost = gross_proceeds * self.config.transaction_cost
                    net_proceeds = gross_proceeds - transaction_cost
                    
                    # Execute trade
                    self.cash_balance += net_proceeds
                    self.positions[symbol] -= shares_to_sell
                    
                    result.update({
                        'executed': True,
                        'quantity': -shares_to_sell,  # Negative for sell
                        'price': execution_price,
                        'cost': -net_proceeds,  # Negative cost (we receive money)
                        'slippage': slippage
                    })
            
            trade_results.append(result)
        
        return trade_results
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        current_data = self.market_data[self.market_data.index == self.current_step]
        prices = {}
        
        for symbol in self.symbols:
            symbol_data = current_data[current_data['symbol'] == symbol]
            if not symbol_data.empty:
                prices[symbol] = symbol_data['close'].iloc[0]
            else:
                # Use last known price if current data is missing
                last_data = self.market_data[
                    (self.market_data.index < self.current_step) & 
                    (self.market_data['symbol'] == symbol)
                ]
                if not last_data.empty:
                    prices[symbol] = last_data['close'].iloc[-1]
                else:
                    prices[symbol] = 100.0  # Default price
        
        return prices
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        current_prices = self._get_current_prices()
        
        positions_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )
        
        self.portfolio_value = self.cash_balance + positions_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
    
    def _calculate_reward(self, trade_results: List[Dict]) -> float:
        """Calculate risk-adjusted reward for the current step."""
        # Portfolio return
        if len(self.portfolio_history) > 0:
            portfolio_return = (self.portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
        else:
            portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Calculate Sharpe ratio component
        if len(self.returns_history) >= 2:
            returns_array = np.array(self.returns_history[-20:])  # Last 20 returns
            excess_returns = returns_array - (self.config.risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate drawdown penalty
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -current_drawdown * 2.0  # Penalize drawdowns
        
        # Transaction cost penalty
        transaction_penalty = -sum(
            abs(result['cost']) * self.config.transaction_cost 
            for result in trade_results if result['executed']
        ) / self.portfolio_value
        
        # Combine reward components
        reward = (
            portfolio_return * self.config.reward_scaling +
            sharpe_ratio * 0.1 +
            drawdown_penalty +
            transaction_penalty
        )
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if maximum drawdown exceeded
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        if current_drawdown > self.config.max_drawdown_limit:
            return True
        
        # Terminate if portfolio value drops too low
        if self.portfolio_value < self.initial_balance * 0.1:  # 90% loss
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Market features for lookback window
        market_features = []
        
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step
        
        window_data = self.market_data[
            (self.market_data.index >= start_idx) & 
            (self.market_data.index < end_idx)
        ]
        
        for symbol in self.symbols:
            symbol_data = window_data[window_data['symbol'] == symbol]
            
            # Pad with zeros if not enough data
            if len(symbol_data) < self.config.lookback_window:
                padding_length = self.config.lookback_window - len(symbol_data)
                padding = np.zeros((padding_length, 15))  # 15 features per timestep
                
                if not symbol_data.empty:
                    features = symbol_data[[
                        'open', 'high', 'low', 'close', 'volume',
                        'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
                        'bb_position', 'volume_ratio', 'sma_5', 'sma_20', 'ema_12'
                    ]].fillna(0).values
                    features = np.vstack([padding, features])
                else:
                    features = padding
            else:
                features = symbol_data[[
                    'open', 'high', 'low', 'close', 'volume',
                    'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
                    'bb_position', 'volume_ratio', 'sma_5', 'sma_20', 'ema_12'
                ]].fillna(0).values
            
            market_features.extend(features.flatten())
        
        # Portfolio features
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
        
        # Combine all features
        observation = np.array(market_features + portfolio_features, dtype=np.float32)
        
        # Handle any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        current_prices = self._get_current_prices()
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'positions': self.positions.copy(),
            'current_prices': current_prices,
            'drawdown': current_drawdown,
            'max_portfolio_value': self.max_portfolio_value,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance
        }
    
    def _record_step(self, actions: List[Dict], reward: float, trade_results: List[Dict]):
        """Record step information for analysis."""
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        self.returns_history.append(portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Record executed trades
        for result in trade_results:
            if result['executed']:
                self.trade_history.append({
                    'step': self.current_step,
                    'timestamp': datetime.now(),
                    'symbol': result['symbol'],
                    'action': result['action_type'].name,
                    'quantity': result['quantity'],
                    'price': result['price'],
                    'cost': result['cost'],
                    'portfolio_value': self.portfolio_value
                })
    
    def render(self, mode: str = "human"):
        """Render the environment state."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash Balance: ${self.cash_balance:,.2f}")
            print(f"Total Return: {((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100:.2f}%")
            print(f"Positions: {self.positions}")
            print("-" * 50)
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        if len(self.returns_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        
        # Basic metrics
        total_return = returns[-1]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        max_drawdown = max(
            (max(self.portfolio_history[:i+1]) - self.portfolio_history[i]) / max(self.portfolio_history[:i+1])
            for i in range(len(self.portfolio_history))
        )
        
        # Sharpe ratio
        excess_returns = returns - (self.config.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = (
            np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        )
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trade_history),
            'final_portfolio_value': self.portfolio_value
        }
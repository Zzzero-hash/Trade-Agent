"""
Comprehensive risk management service for real-time monitoring and control.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
import logging
import uuid
from scipy.stats import norm
from collections import defaultdict, deque

from src.models.portfolio import Portfolio, Position
from src.models.risk_management import (
    RiskLimit, RiskMetrics, RiskAlert, StressTestScenario, StressTestResult,
    PositionSizingRule, RiskLimitType, RiskLimitStatus
)


class RiskManager:
    """
    Comprehensive risk management system with real-time monitoring,
    automated limit enforcement, and stress testing capabilities.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk manager.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
        # Risk limits and rules
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.position_sizing_rules: Dict[str, PositionSizingRule] = {}
        
        # Historical data for tracking
        self.portfolio_history: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=252)  # 1 year of daily data
        self.drawdown_history: deque = deque(maxlen=252)
        
        # Active alerts
        self.active_alerts: Dict[str, RiskAlert] = {}
        
        # Performance tracking
        self.high_water_mark = 0.0
        self.max_drawdown = 0.0
    
    def add_risk_limit(self, limit: RiskLimit) -> None:
        """Add or update a risk limit."""
        limit_key = f"{limit.limit_type.value}_{limit.symbol or 'portfolio'}"
        self.risk_limits[limit_key] = limit
        self.logger.info(f"Added risk limit: {limit_key}")
    
    def add_position_sizing_rule(self, rule: PositionSizingRule) -> None:
        """Add or update a position sizing rule."""
        self.position_sizing_rules[rule.rule_name] = rule
        self.logger.info(f"Added position sizing rule: {rule.rule_name}")
    
    def calculate_risk_metrics(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        confidence_level: float = 0.05
    ) -> RiskMetrics:
        """
        Calculate comprehensive real-time risk metrics.
        
        Args:
            portfolio: Current portfolio
            market_data: Recent market data
            confidence_level: Confidence level for VaR calculation
        
        Returns:
            RiskMetrics object with current risk measures
        """
        try:
            # Calculate returns and covariance
            returns_data = self._calculate_returns(market_data)
            covariance_matrix = self._calculate_covariance_matrix(returns_data)
            
            # Current portfolio weights
            current_weights = self._calculate_portfolio_weights(
                portfolio, market_data
            )
            
            # Portfolio volatility
            portfolio_vol = self._calculate_portfolio_volatility(
                current_weights, covariance_matrix
            )
            
            # Value at Risk
            portfolio_var = self._calculate_var(
                portfolio.total_value, portfolio_vol, confidence_level
            )
            
            # Drawdown metrics
            current_drawdown, max_dd = self._calculate_drawdown_metrics(portfolio)
            
            # Daily P&L
            daily_pnl = self._calculate_daily_pnl(portfolio)
            
            # Concentration risk
            concentration_risk = self._calculate_concentration_risk(current_weights)
            
            # Leverage
            leverage = self._calculate_leverage(portfolio)
            
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio.total_value,
                daily_pnl=daily_pnl,
                unrealized_pnl=portfolio.unrealized_pnl,
                realized_pnl=portfolio.realized_pnl,
                max_drawdown=max_dd,
                current_drawdown=current_drawdown,
                portfolio_var=portfolio_var,
                portfolio_volatility=portfolio_vol,
                concentration_risk=concentration_risk,
                leverage=leverage,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update historical tracking
            self._update_historical_data(portfolio, risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            raise
    
    def check_risk_limits(
        self,
        portfolio: Portfolio,
        risk_metrics: RiskMetrics
    ) -> List[RiskAlert]:
        """
        Check all risk limits and generate alerts for breaches.
        
        Args:
            portfolio: Current portfolio
            risk_metrics: Current risk metrics
        
        Returns:
            List of risk alerts for any limit breaches
        """
        alerts = []
        
        for limit_key, limit in self.risk_limits.items():
            if not limit.enabled:
                continue
            
            try:
                current_value = self._get_limit_current_value(
                    limit, portfolio, risk_metrics
                )
                
                if current_value is None:
                    continue
                
                # Check for breach or warning
                status = self._determine_limit_status(
                    current_value, limit.warning_threshold, limit.threshold
                )
                
                if status != RiskLimitStatus.NORMAL:
                    alert = self._create_risk_alert(
                        limit, status, current_value, portfolio
                    )
                    alerts.append(alert)
                    self.active_alerts[alert.alert_id] = alert
                
            except Exception as e:
                self.logger.error(f"Error checking limit {limit_key}: {e}")
        
        return alerts
    
    def enforce_position_sizing(
        self,
        portfolio: Portfolio,
        symbol: str,
        intended_size: float,
        expected_return: float,
        volatility: float,
        market_data: pd.DataFrame
    ) -> float:
        """
        Enforce position sizing rules and return adjusted position size.
        
        Args:
            portfolio: Current portfolio
            symbol: Symbol to size
            intended_size: Intended position size
            expected_return: Expected return for the asset
            volatility: Asset volatility
            market_data: Market data for correlation calculation
        
        Returns:
            Risk-adjusted position size
        """
        try:
            # Start with intended size
            adjusted_size = intended_size
            
            # Apply each active position sizing rule
            for rule_name, rule in self.position_sizing_rules.items():
                if not rule.enabled:
                    continue
                
                # Calculate optimal size based on rule
                rule_size = self._calculate_rule_based_size(
                    portfolio, symbol, expected_return, volatility,
                    market_data, rule
                )
                
                # Take the minimum (most conservative)
                adjusted_size = min(adjusted_size, rule_size)
            
            # Apply hard position limits
            max_position = self._get_max_position_limit(symbol)
            adjusted_size = min(adjusted_size, max_position)
            
            self.logger.info(
                f"Position sizing for {symbol}: "
                f"intended={intended_size:.4f}, "
                f"adjusted={adjusted_size:.4f}"
            )
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Position sizing enforcement failed: {e}")
            return min(intended_size, 0.05)  # Conservative fallback
    
    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenario: StressTestScenario,
        market_data: pd.DataFrame
    ) -> StressTestResult:
        """
        Run stress test scenario on portfolio.
        
        Args:
            portfolio: Current portfolio
            scenario: Stress test scenario
            market_data: Market data for calculations
        
        Returns:
            StressTestResult with scenario impact
        """
        try:
            # Calculate current portfolio value
            portfolio_value_before = portfolio.total_value
            
            # Apply market shocks to positions
            stressed_positions = {}
            total_impact = 0.0
            position_impacts = {}
            
            for symbol, position in portfolio.positions.items():
                shock = scenario.market_shocks.get(symbol, 0.0)
                
                # Calculate position impact
                position_value = position.market_value
                impact = position_value * shock
                total_impact += impact
                
                # Create stressed position
                new_price = position.current_price * (1 + shock)
                stressed_positions[symbol] = Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_cost=position.avg_cost,
                    current_price=new_price,
                    unrealized_pnl=position.quantity * (new_price - position.avg_cost),
                    realized_pnl=position.realized_pnl
                )
                
                position_impacts[symbol] = impact
            
            # Create stressed portfolio
            portfolio_value_after = portfolio_value_before + total_impact
            
            # Calculate stressed risk metrics
            stressed_portfolio = Portfolio(
                user_id=portfolio.user_id,
                positions=stressed_positions,
                cash_balance=portfolio.cash_balance,
                total_value=portfolio_value_after,
                last_updated=datetime.now(timezone.utc)
            )
            
            # Adjust market data for stress scenario
            stressed_market_data = self._apply_stress_to_market_data(
                market_data, scenario
            )
            
            stressed_risk_metrics = self.calculate_risk_metrics(
                stressed_portfolio, stressed_market_data
            )
            
            return StressTestResult(
                scenario_name=scenario.scenario_name,
                portfolio_value_before=portfolio_value_before,
                portfolio_value_after=portfolio_value_after,
                total_loss=total_impact,
                loss_percentage=total_impact / portfolio_value_before,
                position_impacts=position_impacts,
                risk_metrics_after=stressed_risk_metrics,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            raise
    
    def get_predefined_stress_scenarios(self) -> List[StressTestScenario]:
        """Get predefined stress test scenarios."""
        scenarios = [
            StressTestScenario(
                scenario_name="market_crash_2008",
                market_shocks={
                    "SPY": -0.20, "QQQ": -0.25, "IWM": -0.30,
                    "AAPL": -0.15, "GOOGL": -0.18, "MSFT": -0.12
                },
                correlation_adjustment=1.5,
                volatility_multiplier=2.0,
                description="2008-style market crash with increased correlations"
            ),
            StressTestScenario(
                scenario_name="covid_crash_2020",
                market_shocks={
                    "SPY": -0.35, "QQQ": -0.30, "IWM": -0.40,
                    "AAPL": -0.20, "GOOGL": -0.15, "MSFT": -0.10
                },
                correlation_adjustment=1.8,
                volatility_multiplier=3.0,
                description="COVID-19 market crash scenario"
            ),
            StressTestScenario(
                scenario_name="interest_rate_shock",
                market_shocks={
                    "SPY": -0.10, "QQQ": -0.15, "IWM": -0.12,
                    "TLT": -0.25, "IEF": -0.15
                },
                correlation_adjustment=1.2,
                volatility_multiplier=1.5,
                description="Sudden interest rate increase scenario"
            ),
            StressTestScenario(
                scenario_name="sector_rotation",
                market_shocks={
                    "QQQ": -0.15, "XLK": -0.20,  # Tech down
                    "XLE": 0.10, "XLF": 0.08     # Energy/Finance up
                },
                correlation_adjustment=0.8,
                volatility_multiplier=1.2,
                description="Major sector rotation scenario"
            )
        ]
        
        return scenarios
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active (unacknowledged) alerts."""
        return [
            alert for alert in self.active_alerts.values()
            if not alert.acknowledged
        ]
    
    # Private helper methods
    
    def _calculate_returns(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from market data."""
        returns = market_data.pct_change().dropna()
        return returns.tail(252)  # Last year of data
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized covariance matrix."""
        return returns_data.cov() * 252
    
    def _calculate_portfolio_weights(
        self, portfolio: Portfolio, market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        if portfolio.total_value <= 0:
            return {}
        
        weights = {}
        for symbol, position in portfolio.positions.items():
            if symbol in market_data.columns:
                weight = position.market_value / portfolio.total_value
                weights[symbol] = weight
        
        return weights
    
    def _calculate_portfolio_volatility(
        self, weights: Dict[str, float], covariance_matrix: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility."""
        if not weights:
            return 0.0
        
        symbols = list(weights.keys())
        weight_array = np.array([weights[symbol] for symbol in symbols])
        
        # Filter covariance matrix to available symbols
        available_symbols = [s for s in symbols if s in covariance_matrix.index]
        if not available_symbols:
            return 0.0
        
        cov_subset = covariance_matrix.loc[available_symbols, available_symbols]
        weight_subset = np.array([weights[s] for s in available_symbols])
        
        portfolio_variance = np.dot(weight_subset.T, np.dot(cov_subset, weight_subset))
        return np.sqrt(max(0, portfolio_variance))
    
    def _calculate_var(
        self, portfolio_value: float, volatility: float, confidence_level: float
    ) -> float:
        """Calculate Value at Risk."""
        if volatility <= 0:
            return 0.0
        
        # Daily VaR
        daily_vol = volatility / np.sqrt(252)
        var_multiplier = norm.ppf(confidence_level)
        daily_var = portfolio_value * daily_vol * abs(var_multiplier)
        
        return daily_var
    
    def _calculate_drawdown_metrics(self, portfolio: Portfolio) -> Tuple[float, float]:
        """Calculate current and maximum drawdown."""
        current_value = portfolio.total_value
        
        # Update high water mark
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
        
        # Calculate current drawdown
        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        else:
            current_drawdown = 0.0
        
        # Update maximum drawdown
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return current_drawdown, self.max_drawdown
    
    def _calculate_daily_pnl(self, portfolio: Portfolio) -> float:
        """Calculate daily P&L."""
        if len(self.portfolio_history) == 0:
            return 0.0
        
        previous_value = self.portfolio_history[-1].get('total_value', portfolio.total_value)
        return portfolio.total_value - previous_value
    
    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> float:
        """Calculate concentration risk using Herfindahl index."""
        if not weights:
            return 0.0
        
        weight_values = list(weights.values())
        return sum(w ** 2 for w in weight_values)
    
    def _calculate_leverage(self, portfolio: Portfolio) -> float:
        """Calculate portfolio leverage."""
        if portfolio.total_value <= 0:
            return 0.0
        
        gross_exposure = sum(abs(pos.market_value) for pos in portfolio.positions.values())
        return gross_exposure / portfolio.total_value
    
    def _update_historical_data(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> None:
        """Update historical tracking data."""
        portfolio_snapshot = {
            'timestamp': risk_metrics.timestamp,
            'total_value': portfolio.total_value,
            'daily_pnl': risk_metrics.daily_pnl,
            'drawdown': risk_metrics.current_drawdown
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        self.pnl_history.append(risk_metrics.daily_pnl)
        self.drawdown_history.append(risk_metrics.current_drawdown)
    
    def _get_limit_current_value(
        self, limit: RiskLimit, portfolio: Portfolio, risk_metrics: RiskMetrics
    ) -> Optional[float]:
        """Get current value for a specific risk limit."""
        if limit.limit_type == RiskLimitType.MAX_DRAWDOWN:
            return risk_metrics.current_drawdown
        elif limit.limit_type == RiskLimitType.PORTFOLIO_VAR:
            return risk_metrics.portfolio_var
        elif limit.limit_type == RiskLimitType.CONCENTRATION:
            return risk_metrics.concentration_risk
        elif limit.limit_type == RiskLimitType.LEVERAGE:
            return risk_metrics.leverage
        elif limit.limit_type == RiskLimitType.DAILY_LOSS:
            return abs(min(0, risk_metrics.daily_pnl))
        elif limit.limit_type == RiskLimitType.POSITION_SIZE and limit.symbol:
            position = portfolio.get_position(limit.symbol)
            if position and portfolio.total_value > 0:
                return position.market_value / portfolio.total_value
        
        return None
    
    def _determine_limit_status(
        self, current_value: float, warning_threshold: float, breach_threshold: float
    ) -> RiskLimitStatus:
        """Determine risk limit status."""
        if current_value >= breach_threshold:
            return RiskLimitStatus.BREACH
        elif current_value >= warning_threshold:
            return RiskLimitStatus.WARNING
        else:
            return RiskLimitStatus.NORMAL
    
    def _create_risk_alert(
        self, limit: RiskLimit, status: RiskLimitStatus,
        current_value: float, portfolio: Portfolio
    ) -> RiskAlert:
        """Create a risk alert."""
        alert_id = str(uuid.uuid4())
        
        if status == RiskLimitStatus.BREACH:
            message = f"{limit.limit_type.value} limit breached: {current_value:.4f} >= {limit.threshold:.4f}"
        else:
            message = f"{limit.limit_type.value} warning: {current_value:.4f} >= {limit.warning_threshold:.4f}"
        
        return RiskAlert(
            alert_id=alert_id,
            limit_type=limit.limit_type,
            status=status,
            current_value=current_value,
            threshold=limit.threshold,
            symbol=limit.symbol,
            message=message,
            timestamp=datetime.now(timezone.utc),
            acknowledged=False
        )
    
    def _calculate_rule_based_size(
        self, portfolio: Portfolio, symbol: str, expected_return: float,
        volatility: float, market_data: pd.DataFrame, rule: PositionSizingRule
    ) -> float:
        """Calculate position size based on a specific rule."""
        if volatility <= 0:
            return 0.0
        
        # Kelly criterion with fraction
        kelly_size = (expected_return / (volatility ** 2)) * rule.kelly_fraction
        
        # Volatility targeting
        vol_target_size = rule.volatility_target / volatility
        
        # Correlation penalty
        correlation = self._calculate_symbol_correlation(symbol, portfolio, market_data)
        correlation_adjusted_size = kelly_size * (1 - abs(correlation) * rule.correlation_penalty)
        
        # Take minimum of all constraints
        rule_size = min(kelly_size, vol_target_size, correlation_adjusted_size, rule.max_position_size)
        
        return max(0.0, rule_size)
    
    def _calculate_symbol_correlation(
        self, symbol: str, portfolio: Portfolio, market_data: pd.DataFrame
    ) -> float:
        """Calculate correlation between symbol and existing portfolio."""
        if symbol not in market_data.columns or len(portfolio.positions) == 0:
            return 0.0
        
        # Calculate portfolio returns (simplified)
        portfolio_symbols = [s for s in portfolio.positions.keys() if s in market_data.columns]
        if not portfolio_symbols:
            return 0.0
        
        returns = market_data[portfolio_symbols + [symbol]].pct_change().dropna()
        
        if len(returns) < 30:  # Need sufficient data
            return 0.0
        
        # Weight portfolio returns by position sizes
        weights = {}
        total_value = sum(pos.market_value for pos in portfolio.positions.values())
        
        for pos_symbol in portfolio_symbols:
            position = portfolio.positions[pos_symbol]
            weights[pos_symbol] = position.market_value / total_value
        
        # Calculate weighted portfolio returns
        portfolio_returns = sum(
            returns[pos_symbol] * weights[pos_symbol]
            for pos_symbol in portfolio_symbols
        )
        
        # Calculate correlation
        correlation_matrix = returns[[symbol]].corrwith(portfolio_returns)
        return correlation_matrix.iloc[0] if not correlation_matrix.empty else 0.0
    
    def _get_max_position_limit(self, symbol: str) -> float:
        """Get maximum position limit for a symbol."""
        # Check for symbol-specific limits
        limit_key = f"{RiskLimitType.POSITION_SIZE.value}_{symbol}"
        if limit_key in self.risk_limits:
            return self.risk_limits[limit_key].threshold
        
        # Default maximum position size
        return 0.20  # 20% maximum
    
    def _apply_stress_to_market_data(
        self, market_data: pd.DataFrame, scenario: StressTestScenario
    ) -> pd.DataFrame:
        """Apply stress scenario to market data."""
        stressed_data = market_data.copy()
        
        # Apply shocks to final prices
        for symbol, shock in scenario.market_shocks.items():
            if symbol in stressed_data.columns:
                stressed_data.loc[stressed_data.index[-1], symbol] *= (1 + shock)
        
        # Adjust volatility if needed
        if scenario.volatility_multiplier != 1.0:
            returns = stressed_data.pct_change().dropna()
            adjusted_returns = returns * scenario.volatility_multiplier
            
            # Reconstruct prices (simplified)
            for col in stressed_data.columns:
                if col in adjusted_returns.columns:
                    base_price = stressed_data[col].iloc[0]
                    cumulative_returns = (1 + adjusted_returns[col]).cumprod()
                    stressed_data[col] = base_price * cumulative_returns
        
        return stressed_data
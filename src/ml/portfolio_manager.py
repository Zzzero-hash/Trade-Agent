"""
Portfolio management for trading environments.
"""
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TradeResult:
    """Result of a trade execution."""
    symbol: str
    action_type: str
    executed: bool
    quantity: float
    price: float
    cost: float
    slippage: float


class PortfolioManager:
    """Manages portfolio state and trade execution."""
    
    def __init__(self, initial_balance: float, symbols: List[str]):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.portfolio_value = initial_balance
        self.max_portfolio_value = initial_balance
        self.symbols = symbols
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash_balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
    
    def execute_trade(
        self,
        symbol: str,
        action_type: str,
        position_size: float,
        current_price: float,
        transaction_cost: float,
        slippage: float
    ) -> TradeResult:
        """Execute a single trade."""
        result = TradeResult(
            symbol=symbol,
            action_type=action_type,
            executed=False,
            quantity=0.0,
            price=current_price,
            cost=0.0,
            slippage=0.0
        )
        
        if action_type == "HOLD":
            return result
        
        if action_type == "BUY":
            return self._execute_buy(
                symbol, position_size, current_price, 
                transaction_cost, slippage, result
            )
        elif action_type == "SELL":
            return self._execute_sell(
                symbol, position_size, current_price, 
                transaction_cost, slippage, result
            )
        
        return result
    
    def _execute_buy(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        transaction_cost: float,
        slippage: float,
        result: TradeResult
    ) -> TradeResult:
        """Execute buy order."""
        max_shares = (self.cash_balance * position_size) / current_price
        
        if max_shares <= 0:
            return result
        
        # Apply slippage
        slippage_amount = current_price * slippage
        execution_price = current_price + slippage_amount
        
        # Calculate costs
        gross_cost = max_shares * execution_price
        transaction_cost_amount = gross_cost * transaction_cost
        total_cost = gross_cost + transaction_cost_amount
        
        if total_cost <= self.cash_balance:
            self.cash_balance -= total_cost
            self.positions[symbol] += max_shares
            
            result.executed = True
            result.quantity = max_shares
            result.price = execution_price
            result.cost = total_cost
            result.slippage = slippage_amount
        
        return result
    
    def _execute_sell(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        transaction_cost: float,
        slippage: float,
        result: TradeResult
    ) -> TradeResult:
        """Execute sell order."""
        current_position = self.positions[symbol]
        shares_to_sell = min(current_position, current_position * position_size)
        
        if shares_to_sell <= 0:
            return result
        
        # Apply slippage
        slippage_amount = current_price * slippage
        execution_price = current_price - slippage_amount
        
        # Calculate proceeds
        gross_proceeds = shares_to_sell * execution_price
        transaction_cost_amount = gross_proceeds * transaction_cost
        net_proceeds = gross_proceeds - transaction_cost_amount
        
        self.cash_balance += net_proceeds
        self.positions[symbol] -= shares_to_sell
        
        result.executed = True
        result.quantity = -shares_to_sell
        result.price = execution_price
        result.cost = -net_proceeds
        result.slippage = slippage_amount
        
        return result
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> None:
        """Update total portfolio value."""
        positions_value = sum(
            self.positions[symbol] * current_prices.get(symbol, 0.0)
            for symbol in self.symbols
        )
        
        self.portfolio_value = self.cash_balance + positions_value
        self.max_portfolio_value = max(
            self.max_portfolio_value, 
            self.portfolio_value
        )
    
    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state."""
        return {
            'cash_balance': self.cash_balance,
            'positions': self.positions.copy(),
            'portfolio_value': self.portfolio_value,
            'max_portfolio_value': self.max_portfolio_value
        }
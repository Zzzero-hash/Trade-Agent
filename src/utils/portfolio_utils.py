"""
Portfolio Utilities

Centralized utilities for portfolio operations to eliminate DRY violations.
"""

from typing import Dict, Any, Optional


def extract_portfolio_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract portfolio information from environment info dict"""
    return {
        'portfolio_value': info.get('portfolio_value', 0.0),
        'cash_balance': info.get('cash_balance', 0.0),
        'positions': info.get('positions', {}),
        'portfolio_return': info.get('portfolio_return', 0.0),
        'total_return': info.get('total_return', 0.0),
        'drawdown': info.get('drawdown', 0.0)
    }


def format_portfolio_status(info: Dict[str, Any], step: Optional[int] = None) -> str:
    """Format portfolio status for consistent logging"""
    portfolio_info = extract_portfolio_info(info)
    
    lines = []
    if step is not None:
        lines.append(f"Step {step}:")
    
    lines.extend([
        f"Portfolio Value: ${portfolio_info['portfolio_value']:,.2f}",
        f"Cash Balance: ${portfolio_info['cash_balance']:,.2f}",
        f"Portfolio Return: {portfolio_info['portfolio_return']:.6f}",
        f"Positions: {portfolio_info['positions']}"
    ])
    
    return "\n".join(lines)


def calculate_portfolio_metrics(portfolio_history: list, initial_balance: float) -> Dict[str, float]:
    """Calculate portfolio performance metrics"""
    if not portfolio_history:
        return {}
    
    current_value = portfolio_history[-1]
    total_return = (current_value - initial_balance) / initial_balance
    
    # Calculate drawdown
    peak_value = max(portfolio_history)
    current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
    
    # Calculate max drawdown
    max_drawdown = 0
    peak = initial_balance
    for value in portfolio_history:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'total_return': total_return,
        'current_drawdown': current_drawdown,
        'max_drawdown': max_drawdown,
        'current_value': current_value,
        'peak_value': peak_value
    }


def log_action_result(action: Any, reward: float, info: Dict[str, Any], step: int) -> str:
    """Format action result for consistent logging"""
    portfolio_info = extract_portfolio_info(info)
    
    return (
        f"Step {step}: Action executed\n"
        f"  Reward: {reward:.6f}\n"
        f"  Portfolio Value: ${portfolio_info['portfolio_value']:,.2f}\n"
        f"  Cash Balance: ${portfolio_info['cash_balance']:,.2f}\n"
        f"  Return: {portfolio_info['portfolio_return']:.6f}"
    )
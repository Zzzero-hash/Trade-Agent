"""
AI Trading Platform Python SDK

A comprehensive Python SDK for interacting with the AI Trading Platform API.
Provides async support for all endpoints including trading signals, portfolio
management, risk monitoring, and real-time data streaming.

Requirements: 11.3, 11.4, 11.5
"""

from .client import TradingPlatformClient
from .models import *
from .exceptions import *
from .auth import AuthManager
from .websocket import WebSocketClient
from .plugins import PluginManager

__version__ = "1.0.0"
__author__ = "AI Trading Platform Team"

__all__ = [
    "TradingPlatformClient",
    "AuthManager", 
    "WebSocketClient",
    "PluginManager",
    # Exceptions
    "TradingPlatformError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "APIError",
    # Models
    "TradingSignal",
    "Portfolio",
    "Position",
    "RiskMetrics",
    "MarketData"
]
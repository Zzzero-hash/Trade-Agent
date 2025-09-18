"""
Exception classes for the AI Trading Platform SDK
"""

from typing import Optional, Dict, Any


class TradingPlatformError(Exception):
    """Base exception for all SDK errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class AuthenticationError(TradingPlatformError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(TradingPlatformError):
    """Raised when user lacks permission for requested action"""
    pass


class RateLimitError(TradingPlatformError):
    """Raised when API rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ValidationError(TradingPlatformError):
    """Raised when request validation fails"""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []


class APIError(TradingPlatformError):
    """Raised for general API errors"""
    pass


class NetworkError(TradingPlatformError):
    """Raised for network-related errors"""
    pass


class TimeoutError(TradingPlatformError):
    """Raised when request times out"""
    pass


class WebSocketError(TradingPlatformError):
    """Raised for WebSocket-related errors"""
    pass


class PluginError(TradingPlatformError):
    """Raised for plugin-related errors"""
    pass
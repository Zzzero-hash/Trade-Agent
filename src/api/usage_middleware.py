"""
Middleware for automatic usage tracking.

This middleware automatically tracks API usage for all requests,
enabling freemium functionality and billing calculations.

Requirements: 7.1, 7.2, 7.5
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

from src.models.usage_tracking import UsageType
from src.services.usage_tracking_service import get_usage_tracking_service
from src.api.auth import verify_token, TokenData
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

# JWT security for extracting user info
security = HTTPBearer(auto_error=False)


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically track API usage for billing and limits.
    """
    
    def __init__(self, app, track_all_requests: bool = True):
        super().__init__(app)
        self.track_all_requests = track_all_requests
        self.usage_service = get_usage_tracking_service()
        
        # Define which endpoints should be tracked as AI signals
        self.ai_signal_endpoints = {
            "/api/v1/predict",
            "/api/v1/predict/batch",
            "/api/v1/trading/signals",
            "/api/v1/trading/recommendations"
        }
        
        # Endpoints that should not be tracked
        self.excluded_endpoints = {
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/api/v1/health",
            "/api/v1/usage/plans",  # Public endpoint
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track usage."""
        start_time = time.time()
        
        # Skip tracking for excluded endpoints
        if request.url.path in self.excluded_endpoints:
            return await call_next(request)
        
        # Extract user information from JWT token
        user_id = await self._extract_user_id(request)
        
        # Skip tracking if no user (unauthenticated requests)
        if not user_id:
            return await call_next(request)
        
        # Determine usage type based on endpoint
        usage_type = self._determine_usage_type(request.url.path)
        
        # Check usage limits before processing request
        if usage_type == UsageType.AI_SIGNAL_REQUEST:
            can_proceed, error_message = await self.usage_service.check_usage_limits(
                user_id, usage_type
            )
            
            if not can_proceed:
                # Return 429 Too Many Requests with upgrade information
                upgrade_info = await self.usage_service.get_upgrade_recommendations(user_id)
                
                return Response(
                    content=f'{{"error": "{error_message}", "upgrade_options": {upgrade_info}}}',
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    media_type="application/json"
                )
        
        # Get request size
        request_size = await self._get_request_size(request)
        
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get response size
        response_size = self._get_response_size(response)
        
        # Track the usage asynchronously (don't block response)
        try:
            await self.usage_service.track_usage(
                user_id=user_id,
                usage_type=usage_type,
                endpoint=request.url.path,
                request_size=request_size,
                response_size=response_size,
                processing_time_ms=processing_time_ms,
                metadata={
                    "method": request.method,
                    "status_code": response.status_code,
                    "user_agent": request.headers.get("user-agent", ""),
                    "ip_address": request.client.host if request.client else None
                }
            )
        except Exception as e:
            # Log error but don't fail the request
            logger.error(f"Failed to track usage for user {user_id}: {e}")
        
        return response
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from JWT token in request headers."""
        try:
            # Get Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            # Extract token
            token = auth_header.split(" ")[1]
            
            # Verify and decode token
            token_data = verify_token(token)
            return token_data.user_id
            
        except Exception as e:
            # Token verification failed, but don't block request
            logger.debug(f"Failed to extract user ID from token: {e}")
            return None
    
    def _determine_usage_type(self, endpoint: str) -> UsageType:
        """Determine usage type based on endpoint."""
        if endpoint in self.ai_signal_endpoints:
            return UsageType.AI_SIGNAL_REQUEST
        elif endpoint.startswith("/api/v1/predict"):
            return UsageType.MODEL_PREDICTION
        elif "batch" in endpoint:
            return UsageType.BATCH_PREDICTION
        elif endpoint.startswith("/api/v1/data"):
            return UsageType.DATA_REQUEST
        else:
            return UsageType.API_REQUEST
    
    async def _get_request_size(self, request: Request) -> Optional[int]:
        """Get request body size in bytes."""
        try:
            # For requests with body content
            if hasattr(request, '_body'):
                return len(request._body) if request._body else 0
            
            # Estimate from content-length header
            content_length = request.headers.get("content-length")
            if content_length:
                return int(content_length)
            
            return None
        except Exception:
            return None
    
    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response size in bytes."""
        try:
            # Check content-length header
            content_length = response.headers.get("content-length")
            if content_length:
                return int(content_length)
            
            # Estimate from response body if available
            if hasattr(response, 'body') and response.body:
                return len(response.body)
            
            return None
        except Exception:
            return None


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware specifically for enforcing usage limits on AI signal endpoints.
    
    This provides an additional layer of protection for high-value endpoints.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.usage_service = get_usage_tracking_service()
        
        # High-value endpoints that need strict limit enforcement
        self.protected_endpoints = {
            "/api/v1/predict",
            "/api/v1/predict/batch",
            "/api/v1/trading/signals",
            "/api/v1/trading/recommendations"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce usage limits on protected endpoints."""
        
        # Only check limits for protected endpoints
        if request.url.path not in self.protected_endpoints:
            return await call_next(request)
        
        # Extract user ID
        user_id = await self._extract_user_id(request)
        if not user_id:
            # Let auth middleware handle unauthenticated requests
            return await call_next(request)
        
        # Check limits
        can_proceed, error_message = await self.usage_service.check_usage_limits(
            user_id, UsageType.AI_SIGNAL_REQUEST
        )
        
        if not can_proceed:
            # Get upgrade recommendations
            try:
                upgrade_info = await self.usage_service.get_upgrade_recommendations(user_id)
                
                # Return detailed error with upgrade options
                error_response = {
                    "error": error_message,
                    "error_code": "USAGE_LIMIT_EXCEEDED",
                    "upgrade_recommendations": upgrade_info.get("recommendations", []),
                    "available_plans": upgrade_info.get("available_plans", []),
                    "current_usage": upgrade_info.get("usage_summary", {}).get("current_limits", {})
                }
                
                return Response(
                    content=str(error_response).replace("'", '"'),  # Simple JSON conversion
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    media_type="application/json",
                    headers={
                        "Retry-After": "86400",  # Retry after 24 hours (daily limit reset)
                        "X-RateLimit-Limit": str(upgrade_info.get("usage_summary", {}).get("current_limits", {}).get("daily_ai_signals_limit", "unknown")),
                        "X-RateLimit-Remaining": str(upgrade_info.get("usage_summary", {}).get("current_limits", {}).get("daily_remaining", 0)),
                        "X-RateLimit-Reset": str(int((datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + 
                                                     timedelta(days=1)).timestamp()))
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to get upgrade recommendations: {e}")
                
                # Fallback error response
                return Response(
                    content=f'{{"error": "{error_message}", "error_code": "USAGE_LIMIT_EXCEEDED"}}',
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    media_type="application/json"
                )
        
        # Proceed with request
        return await call_next(request)
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from JWT token."""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.split(" ")[1]
            token_data = verify_token(token)
            return token_data.user_id
            
        except Exception:
            return None


# Utility functions for manual usage tracking

async def track_ai_signal_usage(user_id: str, endpoint: str, metadata: dict = None):
    """
    Manually track AI signal usage.
    
    Use this function in endpoints that generate AI trading signals.
    """
    try:
        usage_service = get_usage_tracking_service()
        
        # Check limits first
        can_proceed, error_message = await usage_service.check_usage_limits(
            user_id, UsageType.AI_SIGNAL_REQUEST
        )
        
        if not can_proceed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error_message
            )
        
        # Track usage
        await usage_service.track_usage(
            user_id=user_id,
            usage_type=UsageType.AI_SIGNAL_REQUEST,
            endpoint=endpoint,
            metadata=metadata or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track AI signal usage: {e}")
        # Don't fail the request for tracking errors


async def track_model_prediction_usage(user_id: str, endpoint: str, processing_time_ms: float = None):
    """
    Manually track model prediction usage.
    """
    try:
        usage_service = get_usage_tracking_service()
        
        await usage_service.track_usage(
            user_id=user_id,
            usage_type=UsageType.MODEL_PREDICTION,
            endpoint=endpoint,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Failed to track model prediction usage: {e}")
        # Don't fail the request for tracking errors
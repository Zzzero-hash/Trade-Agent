"""Main FastAPI application for model serving

This module creates and configures the FastAPI application with all
endpoints, middleware, and error handling for model serving.

Requirements: 6.2, 11.1
"""

from typing import Dict, Any
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from .endpoints import router
from .trading_endpoints import router as trading_router
from .auth import router as auth_router
from .usage_endpoints import router as usage_router
from .monitoring_endpoints import router as monitoring_router
from .ab_testing_endpoints import router as ab_testing_router
from .model_serving import serving_service
from .usage_middleware import UsageTrackingMiddleware, UsageLimitMiddleware
from src.config.settings import get_settings
from src.utils.logging import get_logger, setup_logging
from src.utils.monitoring import get_metrics_collector, setup_monitoring

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    settings = get_settings()
    
    # Startup
    logger.info("Starting AI Trading Platform Model Serving API")
    
    # Initialize serving service
    await serving_service.initialize()
    
    # Setup monitoring
    metrics = setup_monitoring(settings.monitoring)
    
    logger.info(f"Model serving API started on {settings.api.host}:{settings.api.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Trading Platform Model Serving API")
    await serving_service.shutdown()
    logger.info("Model serving API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="AI Trading Platform - Model Serving API",
        description="""
        Model serving infrastructure for the AI Trading Platform.
        
        Features:
        - CNN+LSTM hybrid model serving
        - RL ensemble model serving  
        - Model caching with TTL and LRU eviction
        - Batch inference optimization
        - A/B testing framework for model comparison
        - Real-time metrics and monitoring
        
        This API provides endpoints for making predictions, managing models,
        and running A/B tests to compare model performance.
        """,
        version="1.0.0",
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(router)
    app.include_router(trading_router)
    app.include_router(auth_router)
    app.include_router(usage_router)
    app.include_router(monitoring_router)
    app.include_router(ab_testing_router)
    
    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """Setup middleware for the FastAPI app"""
    
    # Usage tracking middleware (add first to track all requests)
    app.add_middleware(UsageTrackingMiddleware, track_all_requests=True)
    
    # Usage limit enforcement middleware
    app.add_middleware(UsageLimitMiddleware)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (security)
    if not settings.api.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", settings.api.host]
        )
    
    # Request ID and timing middleware
    @app.middleware("http")
    async def add_request_id_and_timing(request: Request, call_next):
        """Add request ID and measure request timing"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Record metrics
            metrics = get_metrics_collector()
            metrics.record_histogram(
                "http_request_duration_seconds",
                process_time,
                {
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "status_code": str(response.status_code)
                }
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            process_time = time.time() - start_time
            metrics = get_metrics_collector()
            metrics.increment_counter(
                "http_requests_errors_total",
                {
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "error_type": type(e).__name__
                }
            )
            raise
    
    # Rate limiting middleware (simple implementation)
    @app.middleware("http")
    async def rate_limiting(request: Request, call_next):
        """Simple rate limiting middleware"""
        # This is a basic implementation - in production, use Redis-based rate limiting
        client_ip = request.client.host
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # TODO: Implement proper rate limiting with Redis
        # For now, just pass through
        return await call_next(request)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup custom exception handlers"""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error for {request.url}: {exc}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP error {exc.status_code} for {request.url}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "detail": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled error for {request.url}: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time()
            }
        )


# Create the app instance
app = create_app()


# Additional root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Trading Platform - Model Serving API",
        "version": "1.0.0",
        "description": "Model serving infrastructure with caching and A/B testing",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "metrics_url": "/api/v1/metrics"
    }


@app.get("/version")
async def version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "api_version": "v1",
        "build_time": "2024-01-01T00:00:00Z",  # Would be set during build
        "git_commit": "unknown"  # Would be set during build
    }


def run_server():
    """Run the FastAPI server"""
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.logging)
    
    # Run with uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level="info" if not settings.api.debug else "debug",
        access_log=True
    )


if __name__ == "__main__":
    run_server()
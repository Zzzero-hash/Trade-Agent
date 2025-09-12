"""Main application entry point"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings, set_settings, Settings
from utils.logging import setup_logging, get_logger
from utils.monitoring import setup_monitoring, get_metrics_collector


async def initialize_application():
    """Initialize the AI Trading Platform application"""
    
    # Load configuration
    settings = get_settings()
    
    # Setup logging
    logger = setup_logging(settings.logging)
    logger.info(f"Starting AI Trading Platform in {settings.environment.value} mode")
    
    # Setup monitoring
    metrics_collector = setup_monitoring(settings.monitoring)
    
    # Log configuration summary
    logger.info(f"Database: {settings.database.host}:{settings.database.port}")
    logger.info(f"Redis: {settings.redis.host}:{settings.redis.port}")
    logger.info(f"ML Device: {settings.ml.device}")
    logger.info(f"Ray Address: {settings.ray.address or 'Local'}")
    logger.info(f"API Server: {settings.api.host}:{settings.api.port}")
    
    # Log exchange configurations
    for name, exchange in settings.exchanges.items():
        logger.info(f"Exchange {name}: {'Sandbox' if exchange.sandbox else 'Live'}")
    
    # Register basic health checks
    def database_health_check():
        # Placeholder - implement actual database connectivity check
        return True
    
    def redis_health_check():
        # Placeholder - implement actual Redis connectivity check
        return True
    
    metrics_collector.register_health_check("database", database_health_check)
    metrics_collector.register_health_check("redis", redis_health_check)
    
    logger.info("Application initialization completed successfully")
    
    return {
        "settings": settings,
        "logger": logger,
        "metrics_collector": metrics_collector
    }


async def main():
    """Main application function"""
    try:
        # Initialize application
        app_context = await initialize_application()
        logger = app_context["logger"]
        
        logger.info("AI Trading Platform is ready")
        
        # Keep the application running
        # In a real application, this would start the API server, ML training, etc.
        while True:
            await asyncio.sleep(60)  # Sleep for 1 minute
            logger.debug("Application heartbeat")
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("AI Trading Platform shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
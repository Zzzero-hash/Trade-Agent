"""Logging infrastructure setup"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from config.settings import LoggingConfig, get_settings


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Setup application logging"""
    
    if config is None:
        settings = get_settings()
        config = settings.logging
    
    # Create root logger
    logger = logging.getLogger("ai_trading_platform")
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper()))
    
    console_formatter = ColoredFormatter(config.format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        
        file_formatter = logging.Formatter(config.format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(f"ai_trading_platform.{name}")


# Module-level logger instances
def get_exchange_logger() -> logging.Logger:
    """Get logger for exchange operations"""
    return get_logger("exchanges")


def get_ml_logger() -> logging.Logger:
    """Get logger for ML operations"""
    return get_logger("ml")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations"""
    return get_logger("api")


def get_service_logger() -> logging.Logger:
    """Get logger for service operations"""
    return get_logger("services")


def get_repository_logger() -> logging.Logger:
    """Get logger for repository operations"""
    return get_logger("repositories")
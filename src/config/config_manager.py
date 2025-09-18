"""
Configuration Management System.

This module provides a robust configuration management system with
validation, environment variable support, and hierarchical configuration.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_platform"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    
    def __post_init__(self):
        """Validate database configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid database port: {self.port}")
        if self.pool_size <= 0:
            raise ValueError(f"Pool size must be positive: {self.pool_size}")


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    
    def __post_init__(self):
        """Validate Redis configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid Redis port: {self.port}")
        if not 0 <= self.db <= 15:
            raise ValueError(f"Invalid Redis database: {self.db}")


@dataclass
class MLConfig:
    """Machine learning configuration."""
    model_registry_path: str = "models/"
    feature_store_path: str = "features/"
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    random_seed: int = 42
    device: str = "cpu"
    
    def __post_init__(self):
        """Validate ML configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {self.batch_size}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"Learning rate must be in (0, 1]: {self.learning_rate}")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Validation split must be in (0, 1): {self.validation_split}")
        if self.device not in ["cpu", "cuda", "auto"]:
            raise ValueError(f"Invalid device: {self.device}")


@dataclass
class RayConfig:
    """Ray distributed computing configuration."""
    address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    object_store_memory: Optional[int] = None
    
    def __post_init__(self):
        """Validate Ray configuration."""
        if self.num_cpus is not None and self.num_cpus <= 0:
            raise ValueError(f"num_cpus must be positive: {self.num_cpus}")
        if self.num_gpus is not None and self.num_gpus < 0:
            raise ValueError(f"num_gpus cannot be negative: {self.num_gpus}")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}")
        if self.max_file_size <= 0:
            raise ValueError(f"Max file size must be positive: {self.max_file_size}")


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    metrics_port: int = 8000
    health_check_interval: int = 30
    alert_webhook_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        if not 1 <= self.metrics_port <= 65535:
            raise ValueError(f"Invalid metrics port: {self.metrics_port}")
        if self.health_check_interval <= 0:
            raise ValueError(f"Health check interval must be positive: {self.health_check_interval}")


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: str = "100/minute"
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24
    
    def __post_init__(self):
        """Validate API configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid API port: {self.port}")
        if self.jwt_expiry_hours <= 0:
            raise ValueError(f"JWT expiry must be positive: {self.jwt_expiry_hours}")
        if self.debug and self.jwt_secret == "change-me-in-production":
            logger.warning("Using default JWT secret in debug mode")


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    sandbox: bool = True
    rate_limit: int = 100
    timeout: int = 30
    
    def __post_init__(self):
        """Validate exchange configuration."""
        if self.rate_limit <= 0:
            raise ValueError(f"Rate limit must be positive: {self.rate_limit}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")


@dataclass
class ExchangesConfig:
    """All exchanges configuration."""
    robinhood: ExchangeConfig = field(default_factory=ExchangeConfig)
    oanda: ExchangeConfig = field(default_factory=ExchangeConfig)
    coinbase: ExchangeConfig = field(default_factory=ExchangeConfig)


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    exchanges: ExchangesConfig = field(default_factory=ExchangesConfig)
    
    def __post_init__(self):
        """Validate application configuration."""
        # Production-specific validations
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                logger.warning("Debug mode enabled in production")
            if self.api.jwt_secret == "change-me-in-production":
                raise ValueError("Must set JWT secret in production")
            if not self.database.password:
                logger.warning("Empty database password in production")


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


class ConfigManager:
    """Configuration manager with environment variable support."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._config: Optional[AppConfig] = None
        self._environment = self._detect_environment()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def load_config(self, config_file: Optional[str] = None) -> AppConfig:
        """
        Load configuration from files and environment variables.
        
        Args:
            config_file: Specific config file to load (optional)
            
        Returns:
            Loaded application configuration
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            # Load base configuration
            base_config = self._load_config_file("settings.yaml")
            
            # Load environment-specific configuration
            env_config_file = f"{self._environment.value}.yaml"
            env_config = self._load_config_file(env_config_file, required=False)
            
            # Merge configurations
            merged_config = self._merge_configs(base_config, env_config)
            
            # Apply environment variable overrides
            final_config = self._apply_env_overrides(merged_config)
            
            # Create and validate configuration object
            self._config = self._create_config_object(final_config)
            
            logger.info(f"Configuration loaded for environment: {self._environment.value}")
            return self._config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_config_file(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            if required:
                raise ConfigurationError(f"Required config file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                elif filename.endswith('.json'):
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported config file format: {filename}")
        
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config file {filename}: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        if not override:
            return base.copy()
        
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Define environment variable mappings
        env_mappings = {
            'DB_HOST': ['database', 'host'],
            'DB_PORT': ['database', 'port'],
            'DB_NAME': ['database', 'database'],
            'DB_USER': ['database', 'username'],
            'DB_PASSWORD': ['database', 'password'],
            'REDIS_HOST': ['redis', 'host'],
            'REDIS_PORT': ['redis', 'port'],
            'REDIS_PASSWORD': ['redis', 'password'],
            'RAY_ADDRESS': ['ray', 'address'],
            'JWT_SECRET': ['api', 'jwt_secret'],
            'ALERT_WEBHOOK_URL': ['monitoring', 'alert_webhook_url'],
        }
        
        result = config.copy()
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested configuration
                current = result
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value with appropriate type conversion
                final_key = config_path[-1]
                current[final_key] = self._convert_env_value(env_value, final_key)
        
        return result
    
    def _convert_env_value(self, value: str, key: str) -> Union[str, int, bool, None]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # None conversion
        if value.lower() in ('null', 'none', ''):
            return None
        
        # Integer conversion for known integer fields
        integer_fields = ['port', 'db', 'pool_size', 'max_overflow', 'max_connections', 
                         'batch_size', 'epochs', 'num_cpus', 'num_gpus', 'metrics_port',
                         'health_check_interval', 'jwt_expiry_hours', 'rate_limit', 'timeout']
        
        if any(field in key.lower() for field in integer_fields):
            try:
                return int(value)
            except ValueError:
                pass
        
        # Float conversion for known float fields
        float_fields = ['learning_rate', 'validation_split']
        
        if any(field in key.lower() for field in float_fields):
            try:
                return float(value)
            except ValueError:
                pass
        
        # Default to string
        return value
    
    def _create_config_object(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Create AppConfig object from configuration dictionary."""
        try:
            # Convert environment string to enum
            if 'environment' in config_dict:
                config_dict['environment'] = Environment(config_dict['environment'])
            
            # Create nested configuration objects
            nested_configs = {}
            
            for field_name, field_type in AppConfig.__annotations__.items():
                if field_name in config_dict and hasattr(field_type, '__annotations__'):
                    # This is a nested configuration object
                    nested_configs[field_name] = field_type(**config_dict[field_name])
            
            # Update config_dict with nested objects
            config_dict.update(nested_configs)
            
            return AppConfig(**config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration object: {e}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration (load if not already loaded)."""
        if self._config is None:
            self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from files."""
        self._config = None
        return self.load_config()
    
    def validate_config(self, config: Optional[AppConfig] = None) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            List of validation issues (empty if valid)
        """
        if config is None:
            config = self.get_config()
        
        issues = []
        
        # Production-specific validations
        if config.environment == Environment.PRODUCTION:
            if config.debug:
                issues.append("Debug mode should be disabled in production")
            
            if config.api.jwt_secret == "change-me-in-production":
                issues.append("JWT secret must be changed in production")
            
            if not config.database.password:
                issues.append("Database password should be set in production")
            
            if "*" in config.api.cors_origins:
                issues.append("CORS origins should be restricted in production")
        
        # Security validations
        if len(config.api.jwt_secret) < 32:
            issues.append("JWT secret should be at least 32 characters long")
        
        # Resource validations
        if config.database.pool_size > 100:
            issues.append("Database pool size seems excessive (>100)")
        
        if config.ml.batch_size > 10000:
            issues.append("ML batch size seems excessive (>10000)")
        
        return issues


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get current application configuration."""
    return config_manager.get_config()


def reload_config() -> AppConfig:
    """Reload application configuration."""
    return config_manager.reload_config()
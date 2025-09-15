"""Configuration management with environment-specific settings"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from enum import Enum


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_platform"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10


@dataclass
class TimescaleDBConfig:
    """TimescaleDB configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_platform_timeseries"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10


@dataclass
class InfluxDBConfig:
    """InfluxDB configuration"""
    url: str = "http://localhost:8086"
    token: str = "dev-token"
    org: str = "trading-platform"
    bucket: str = "market-data"
    timeout: int = 30000


@dataclass
class DataStorageConfig:
    """Data storage configuration"""
    primary_db: str = "timescaledb"
    enable_influxdb: bool = True
    enable_redis: bool = True
    backup_enabled: bool = True
    retention_days: int = 30
    backup_path: str = "/tmp/trading_platform_backups"
    compression_enabled: bool = True


@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    name: str
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds


@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_registry_path: str = "models/"
    feature_store_path: str = "features/"
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    random_seed: int = 42
    device: str = "cpu"  # or "cuda"


@dataclass
class RayConfig:
    """Ray distributed computing configuration"""
    address: Optional[str] = None  # None for local, "ray://head-node:10001" for cluster
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    object_store_memory: Optional[int] = None


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enabled: bool = True
    metrics_port: int = 8000
    health_check_interval: int = 30  # seconds
    alert_webhook_url: Optional[str] = None
    drift_detection_enabled: bool = True
    drift_detection_window: int = 100
    performance_monitoring_enabled: bool = True
    automated_retraining_enabled: bool = True
    alert_cooldown_minutes: int = 30
    data_retention_days: int = 30


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit: str = "100/minute"
    jwt_secret: str = "your-secret-key"
    jwt_expiry_hours: int = 24


@dataclass
class Settings:
    """Main application settings"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    timescaledb: TimescaleDBConfig = field(default_factory=TimescaleDBConfig)
    influxdb: InfluxDBConfig = field(default_factory=InfluxDBConfig)
    data_storage: DataStorageConfig = field(default_factory=DataStorageConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Exchange configurations
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables"""
        env_name = os.getenv("ENVIRONMENT", "development")
        environment = Environment(env_name)
        
        settings = cls(environment=environment)
        
        # Override with environment variables
        if os.getenv("DEBUG"):
            settings.debug = os.getenv("DEBUG").lower() == "true"
        
        # Database settings
        if os.getenv("DB_HOST"):
            settings.database.host = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            settings.database.port = int(os.getenv("DB_PORT"))
        if os.getenv("DB_NAME"):
            settings.database.database = os.getenv("DB_NAME")
        if os.getenv("DB_USER"):
            settings.database.username = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"):
            settings.database.password = os.getenv("DB_PASSWORD")
        
        # Redis settings
        if os.getenv("REDIS_HOST"):
            settings.redis.host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            settings.redis.port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            settings.redis.password = os.getenv("REDIS_PASSWORD")
        
        # API settings
        if os.getenv("API_HOST"):
            settings.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            settings.api.port = int(os.getenv("API_PORT"))
        if os.getenv("JWT_SECRET"):
            settings.api.jwt_secret = os.getenv("JWT_SECRET")
        
        # Exchange API keys
        exchanges = {}
        
        # Robinhood
        if os.getenv("ROBINHOOD_API_KEY"):
            exchanges["robinhood"] = ExchangeConfig(
                name="robinhood",
                api_key=os.getenv("ROBINHOOD_API_KEY"),
                api_secret=os.getenv("ROBINHOOD_API_SECRET", ""),
                sandbox=os.getenv("ROBINHOOD_SANDBOX", "true").lower() == "true"
            )
        
        # OANDA
        if os.getenv("OANDA_API_KEY"):
            exchanges["oanda"] = ExchangeConfig(
                name="oanda",
                api_key=os.getenv("OANDA_API_KEY"),
                api_secret=os.getenv("OANDA_API_SECRET", ""),
                sandbox=os.getenv("OANDA_SANDBOX", "true").lower() == "true"
            )
        
        # Coinbase
        if os.getenv("COINBASE_API_KEY"):
            exchanges["coinbase"] = ExchangeConfig(
                name="coinbase",
                api_key=os.getenv("COINBASE_API_KEY"),
                api_secret=os.getenv("COINBASE_API_SECRET", ""),
                sandbox=os.getenv("COINBASE_SANDBOX", "true").lower() == "true"
            )
        
        settings.exchanges = exchanges
        
        return settings
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Settings':
        """Load settings from configuration file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                config_data = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        # Convert dict to Settings object
        return cls._from_dict(config_data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create Settings from dictionary"""
        # This is a simplified implementation
        # In a real application, you might want to use a library like pydantic
        settings = cls()
        
        if 'environment' in data:
            settings.environment = Environment(data['environment'])
        
        if 'debug' in data:
            settings.debug = data['debug']
        
        # Update nested configurations
        if 'database' in data:
            for key, value in data['database'].items():
                if hasattr(settings.database, key):
                    setattr(settings.database, key, value)
        
        if 'redis' in data:
            for key, value in data['redis'].items():
                if hasattr(settings.redis, key):
                    setattr(settings.redis, key, value)
        
        if 'exchanges' in data:
            exchanges = {}
            for name, config in data['exchanges'].items():
                exchanges[name] = ExchangeConfig(name=name, **config)
            settings.exchanges = exchanges
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'environment': self.environment.value,
            'debug': self.debug,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'username': self.database.username,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'max_connections': self.redis.max_connections
            },
            'ml': {
                'model_registry_path': self.ml.model_registry_path,
                'feature_store_path': self.ml.feature_store_path,
                'batch_size': self.ml.batch_size,
                'learning_rate': self.ml.learning_rate,
                'epochs': self.ml.epochs,
                'validation_split': self.ml.validation_split,
                'random_seed': self.ml.random_seed,
                'device': self.ml.device
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'cors_origins': self.api.cors_origins,
                'rate_limit': self.api.rate_limit,
                'jwt_expiry_hours': self.api.jwt_expiry_hours
            },
            'exchanges': {
                name: {
                    'sandbox': config.sandbox,
                    'rate_limit': config.rate_limit,
                    'timeout': config.timeout
                }
                for name, config in self.exchanges.items()
            }
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    
    if _settings is None:
        # Try to load from file first, then fall back to environment
        config_file = os.getenv("CONFIG_FILE", "config/settings.yaml")
        
        if os.path.exists(config_file):
            _settings = Settings.from_file(config_file)
        else:
            _settings = Settings.from_env()
    
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance"""
    global _settings
    _settings = settings
"""
Configuration specifications for connection pooling.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DatabasePoolConfig:
    """Configuration for database connection pools."""
    
    # Connection parameters
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_platform"
    username: str = "postgres"
    password: str = ""
    
    # Pool sizing
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    
    # Timeout settings
    command_timeout: float = 60.0
    connect_timeout: float = 10.0
    
    # Additional settings
    server_settings: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid database port: {self.port}")
        if self.min_size <= 0:
            raise ValueError(f"Min pool size must be positive: {self.min_size}")
        if self.max_size < self.min_size:
            raise ValueError(f"Max pool size ({self.max_size}) cannot be less than min size ({self.min_size})")
        if self.command_timeout <= 0:
            raise ValueError(f"Command timeout must be positive: {self.command_timeout}")
        if self.connect_timeout <= 0:
            raise ValueError(f"Connect timeout must be positive: {self.connect_timeout}")


@dataclass
class RedisPoolConfig:
    """Configuration for Redis connection pools."""
    
    # Connection parameters
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Pool sizing
    max_connections: int = 10
    
    # Timeout settings
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # Behavior settings
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid Redis port: {self.port}")
        if not 0 <= self.db <= 15:
            raise ValueError(f"Invalid Redis database: {self.db}")
        if self.max_connections <= 0:
            raise ValueError(f"Max connections must be positive: {self.max_connections}")
        if self.socket_timeout <= 0:
            raise ValueError(f"Socket timeout must be positive: {self.socket_timeout}")
        if self.socket_connect_timeout <= 0:
            raise ValueError(f"Socket connect timeout must be positive: {self.socket_connect_timeout}")


# Default configurations
DEFAULT_DATABASE_POOL_CONFIG = DatabasePoolConfig()
DEFAULT_REDIS_POOL_CONFIG = RedisPoolConfig()


def create_database_pool_config_from_dict(config_dict: Dict[str, Any]) -> DatabasePoolConfig:
    """
    Create DatabasePoolConfig from dictionary.
    
    Args:
        config_dict: Dictionary containing configuration parameters
        
    Returns:
        DatabasePoolConfig instance
    """
    # Filter out any keys that are not in the dataclass
    valid_keys = {f.name for f in DatabasePoolConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    return DatabasePoolConfig(**filtered_config)


def create_redis_pool_config_from_dict(config_dict: Dict[str, Any]) -> RedisPoolConfig:
    """
    Create RedisPoolConfig from dictionary.
    
    Args:
        config_dict: Dictionary containing configuration parameters
        
    Returns:
        RedisPoolConfig instance
    """
    # Filter out any keys that are not in the dataclass
    valid_keys = {f.name for f in RedisPoolConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    return RedisPoolConfig(**filtered_config)
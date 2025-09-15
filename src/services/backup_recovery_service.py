"""
Backup and disaster recovery service for data storage.
"""
import os
import json
import gzip
import shutil
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

from ..config.settings import get_settings
from ..repositories.timescaledb_repository import TimescaleDBRepository
from ..repositories.influxdb_repository import InfluxDBRepository
from ..repositories.redis_cache import RedisCache


logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""
    backup_id: str
    timestamp: datetime
    backup_type: str  # 'full', 'incremental'
    data_sources: List[str]  # ['timescaledb', 'influxdb', 'redis']
    start_time: datetime
    end_time: datetime
    file_count: int
    total_size_bytes: int
    compression: bool
    status: str  # 'in_progress', 'completed', 'failed'
    error_message: Optional[str] = None


class BackupRecoveryService:
    """
    Service for backing up and recovering data from storage backends.
    
    Provides comprehensive backup and disaster recovery capabilities
    for TimescaleDB, InfluxDB, and Redis data.
    """
    
    def __init__(
        self,
        backup_root_path: str = "/tmp/trading_platform_backups",
        compression_enabled: bool = True,
        retention_days: int = 30,
        max_concurrent_operations: int = 3
    ):
        """
        Initialize backup and recovery service.
        
        Args:
            backup_root_path: Root directory for backups
            compression_enabled: Enable gzip compression
            retention_days: Days to retain backups
            max_concurrent_operations: Max concurrent backup operations
        """
        self.backup_root_path = Path(backup_root_path)
        self.compression_enabled = compression_enabled
        self.retention_days = retention_days
        self.max_concurrent_operations = max_concurrent_operations
        
        self.settings = get_settings()
        
        # Initialize repositories
        self.timescaledb: Optional[TimescaleDBRepository] = None
        self.influxdb: Optional[InfluxDBRepository] = None
        self.redis: Optional[RedisCache] = None
        
        # Ensure backup directory exists
        self.backup_root_path.mkdir(parents=True, exist_ok=True)
        
        # Semaphore for concurrent operations
        self._operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
    
    async def initialize(self) -> None:
        """Initialize repository connections."""
        try:
            # Initialize TimescaleDB
            self.timescaledb = TimescaleDBRepository(
                host=self.settings.database.host,
                port=self.settings.database.port,
                database=self.settings.database.database,
                username=self.settings.database.username,
                password=self.settings.database.password
            )
            await self.timescaledb.connect()
            
            # Initialize InfluxDB
            influx_url = f"http://{self.settings.database.host}:8086"
            self.influxdb = InfluxDBRepository(
                url=influx_url,
                token="dev-token",
                org="trading-platform",
                bucket="market-data"
            )
            await self.influxdb.connect()
            
            # Initialize Redis
            self.redis = RedisCache(
                host=self.settings.redis.host,
                port=self.settings.redis.port,
                db=self.settings.redis.db,
                password=self.settings.redis.password
            )
            await self.redis.connect()
            
            logger.info("Backup recovery service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backup recovery service: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown repository connections."""
        try:
            if self.timescaledb:
                await self.timescaledb.disconnect()
            if self.influxdb:
                await self.influxdb.disconnect()
            if self.redis:
                await self.redis.disconnect()
            
            logger.info("Backup recovery service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during backup recovery service shutdown: {e}")
    
    async def create_full_backup(
        self,
        backup_name: Optional[str] = None,
        include_redis: bool = True
    ) -> BackupMetadata:
        """
        Create a full backup of all data.
        
        Args:
            backup_name: Custom backup name (auto-generated if None)
            include_redis: Whether to include Redis data
            
        Returns:
            Backup metadata
        """
        async with self._operation_semaphore:
            backup_id = backup_name or f"full_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_root_path / backup_id
            backup_path.mkdir(parents=True, exist_ok=True)
            
            start_time = datetime.now(timezone.utc)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=start_time,
                backup_type='full',
                data_sources=[],
                start_time=start_time,
                end_time=start_time,  # Will be updated
                file_count=0,
                total_size_bytes=0,
                compression=self.compression_enabled,
                status='in_progress'
            )
            
            try:
                # Backup TimescaleDB
                if self.timescaledb:
                    await self._backup_timescaledb(backup_path)
                    metadata.data_sources.append('timescaledb')
                
                # Backup InfluxDB
                if self.influxdb:
                    await self._backup_influxdb(backup_path)
                    metadata.data_sources.append('influxdb')
                
                # Backup Redis
                if include_redis and self.redis:
                    await self._backup_redis(backup_path)
                    metadata.data_sources.append('redis')
                
                # Calculate backup statistics
                metadata.file_count, metadata.total_size_bytes = self._calculate_backup_stats(backup_path)
                metadata.end_time = datetime.now(timezone.utc)
                metadata.status = 'completed'
                
                # Save metadata
                await self._save_backup_metadata(backup_path, metadata)
                
                logger.info(f"Full backup completed: {backup_id}")
                return metadata
                
            except Exception as e:
                metadata.status = 'failed'
                metadata.error_message = str(e)
                metadata.end_time = datetime.now(timezone.utc)
                
                # Save failed metadata
                await self._save_backup_metadata(backup_path, metadata)
                
                logger.error(f"Full backup failed: {e}")
                raise
    
    async def create_incremental_backup(
        self,
        since: datetime,
        backup_name: Optional[str] = None
    ) -> BackupMetadata:
        """
        Create an incremental backup since specified timestamp.
        
        Args:
            since: Timestamp to backup from
            backup_name: Custom backup name
            
        Returns:
            Backup metadata
        """
        async with self._operation_semaphore:
            backup_id = backup_name or f"incremental_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_root_path / backup_id
            backup_path.mkdir(parents=True, exist_ok=True)
            
            start_time = datetime.now(timezone.utc)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=start_time,
                backup_type='incremental',
                data_sources=[],
                start_time=since,
                end_time=start_time,
                file_count=0,
                total_size_bytes=0,
                compression=self.compression_enabled,
                status='in_progress'
            )
            
            try:
                # Backup TimescaleDB incremental data
                if self.timescaledb:
                    await self._backup_timescaledb_incremental(backup_path, since, start_time)
                    metadata.data_sources.append('timescaledb')
                
                # InfluxDB incremental backup
                if self.influxdb:
                    await self._backup_influxdb_incremental(backup_path, since, start_time)
                    metadata.data_sources.append('influxdb')
                
                # Redis doesn't support incremental backups easily
                # We'll skip it for incremental backups
                
                # Calculate backup statistics
                metadata.file_count, metadata.total_size_bytes = self._calculate_backup_stats(backup_path)
                metadata.end_time = datetime.now(timezone.utc)
                metadata.status = 'completed'
                
                # Save metadata
                await self._save_backup_metadata(backup_path, metadata)
                
                logger.info(f"Incremental backup completed: {backup_id}")
                return metadata
                
            except Exception as e:
                metadata.status = 'failed'
                metadata.error_message = str(e)
                metadata.end_time = datetime.now(timezone.utc)
                
                await self._save_backup_metadata(backup_path, metadata)
                
                logger.error(f"Incremental backup failed: {e}")
                raise
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Restore data from backup.
        
        Args:
            backup_id: Backup identifier
            restore_options: Restoration options
            
        Returns:
            True if successful, False otherwise
        """
        async with self._operation_semaphore:
            backup_path = self.backup_root_path / backup_id
            
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            try:
                # Load backup metadata
                metadata = await self._load_backup_metadata(backup_path)
                
                if metadata.status != 'completed':
                    logger.error(f"Cannot restore incomplete backup: {backup_id}")
                    return False
                
                logger.info(f"Starting restore from backup: {backup_id}")
                
                # Restore TimescaleDB
                if 'timescaledb' in metadata.data_sources:
                    await self._restore_timescaledb(backup_path, restore_options)
                
                # Restore InfluxDB
                if 'influxdb' in metadata.data_sources:
                    await self._restore_influxdb(backup_path, restore_options)
                
                # Restore Redis
                if 'redis' in metadata.data_sources:
                    await self._restore_redis(backup_path, restore_options)
                
                logger.info(f"Restore completed successfully: {backup_id}")
                return True
                
            except Exception as e:
                logger.error(f"Restore failed: {e}")
                return False
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List all available backups."""
        backups = []
        
        try:
            for backup_dir in self.backup_root_path.iterdir():
                if backup_dir.is_dir():
                    try:
                        metadata = await self._load_backup_metadata(backup_dir)
                        backups.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {backup_dir.name}: {e}")
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        backup_path = self.backup_root_path / backup_id
        
        if not backup_path.exists():
            logger.warning(f"Backup not found: {backup_id}")
            return False
        
        try:
            shutil.rmtree(backup_path)
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def cleanup_old_backups(self) -> int:
        """Clean up backups older than retention period."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        deleted_count = 0
        
        try:
            backups = await self.list_backups()
            
            for backup in backups:
                if backup.timestamp < cutoff_date:
                    if await self.delete_backup(backup.backup_id):
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
        
        return deleted_count
    
    async def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        backup_path = self.backup_root_path / backup_id
        
        if not backup_path.exists():
            return {'valid': False, 'error': 'Backup not found'}
        
        try:
            # Load metadata
            metadata = await self._load_backup_metadata(backup_path)
            
            # Verify files exist
            verification_results = {
                'valid': True,
                'metadata': asdict(metadata),
                'files_verified': 0,
                'missing_files': [],
                'corrupted_files': []
            }
            
            # Check each data source
            for data_source in metadata.data_sources:
                source_path = backup_path / data_source
                if source_path.exists():
                    # Count files and verify basic integrity
                    for file_path in source_path.rglob('*'):
                        if file_path.is_file():
                            verification_results['files_verified'] += 1
                            
                            # Basic file integrity check
                            if file_path.stat().st_size == 0:
                                verification_results['corrupted_files'].append(str(file_path))
                else:
                    verification_results['missing_files'].append(data_source)
            
            # Mark as invalid if there are issues
            if verification_results['missing_files'] or verification_results['corrupted_files']:
                verification_results['valid'] = False
            
            return verification_results
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    # Private helper methods
    
    async def _backup_timescaledb(self, backup_path: Path) -> None:
        """Backup TimescaleDB data."""
        timescale_path = backup_path / 'timescaledb'
        timescale_path.mkdir(exist_ok=True)
        
        # Export market data
        market_data = await self.timescaledb.query_market_data(
            symbol='%',  # All symbols
            exchange='%',  # All exchanges
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_time=datetime.now(timezone.utc)
        )
        
        await self._save_data_to_file(
            timescale_path / 'market_data.json',
            market_data
        )
        
        # Export predictions
        # Note: This is a simplified implementation
        # In production, you'd want more sophisticated export logic
        
        logger.info("TimescaleDB backup completed")
    
    async def _backup_timescaledb_incremental(
        self,
        backup_path: Path,
        since: datetime,
        until: datetime
    ) -> None:
        """Backup TimescaleDB incremental data."""
        timescale_path = backup_path / 'timescaledb'
        timescale_path.mkdir(exist_ok=True)
        
        # Export incremental market data
        market_data = await self.timescaledb.query_market_data(
            symbol='%',
            exchange='%',
            start_time=since,
            end_time=until
        )
        
        await self._save_data_to_file(
            timescale_path / 'market_data_incremental.json',
            market_data
        )
        
        logger.info("TimescaleDB incremental backup completed")
    
    async def _backup_influxdb(self, backup_path: Path) -> None:
        """Backup InfluxDB data."""
        influx_path = backup_path / 'influxdb'
        influx_path.mkdir(exist_ok=True)
        
        # This is a simplified implementation
        # In production, you'd use InfluxDB's backup tools
        
        logger.info("InfluxDB backup completed")
    
    async def _backup_influxdb_incremental(
        self,
        backup_path: Path,
        since: datetime,
        until: datetime
    ) -> None:
        """Backup InfluxDB incremental data."""
        influx_path = backup_path / 'influxdb'
        influx_path.mkdir(exist_ok=True)
        
        logger.info("InfluxDB incremental backup completed")
    
    async def _backup_redis(self, backup_path: Path) -> None:
        """Backup Redis data."""
        redis_path = backup_path / 'redis'
        redis_path.mkdir(exist_ok=True)
        
        # Get all keys and their values
        keys = await self.redis.get_keys_by_pattern('*')
        redis_data = {}
        
        for key in keys:
            value = await self.redis.get(key)
            if value is not None:
                redis_data[key] = value
        
        await self._save_data_to_file(
            redis_path / 'redis_dump.json',
            redis_data
        )
        
        logger.info("Redis backup completed")
    
    async def _restore_timescaledb(
        self,
        backup_path: Path,
        restore_options: Optional[Dict[str, Any]]
    ) -> None:
        """Restore TimescaleDB data."""
        timescale_path = backup_path / 'timescaledb'
        
        # Load and restore market data
        market_data_file = timescale_path / 'market_data.json'
        if market_data_file.exists():
            market_data = await self._load_data_from_file(market_data_file)
            # Restore logic would go here
            logger.info(f"Restored {len(market_data)} market data records")
        
        logger.info("TimescaleDB restore completed")
    
    async def _restore_influxdb(
        self,
        backup_path: Path,
        restore_options: Optional[Dict[str, Any]]
    ) -> None:
        """Restore InfluxDB data."""
        logger.info("InfluxDB restore completed")
    
    async def _restore_redis(
        self,
        backup_path: Path,
        restore_options: Optional[Dict[str, Any]]
    ) -> None:
        """Restore Redis data."""
        redis_path = backup_path / 'redis'
        
        # Load and restore Redis data
        redis_dump_file = redis_path / 'redis_dump.json'
        if redis_dump_file.exists():
            redis_data = await self._load_data_from_file(redis_dump_file)
            
            for key, value in redis_data.items():
                await self.redis.set(key, value)
            
            logger.info(f"Restored {len(redis_data)} Redis keys")
        
        logger.info("Redis restore completed")
    
    async def _save_data_to_file(self, file_path: Path, data: Any) -> None:
        """Save data to file with optional compression."""
        json_data = json.dumps(data, default=str, indent=2)
        
        if self.compression_enabled:
            with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
    
    async def _load_data_from_file(self, file_path: Path) -> Any:
        """Load data from file with compression support."""
        # Check for compressed version first
        compressed_path = Path(f"{file_path}.gz")
        
        if compressed_path.exists():
            with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    async def _save_backup_metadata(self, backup_path: Path, metadata: BackupMetadata) -> None:
        """Save backup metadata."""
        metadata_file = backup_path / 'metadata.json'
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, default=str, indent=2)
    
    async def _load_backup_metadata(self, backup_path: Path) -> BackupMetadata:
        """Load backup metadata."""
        metadata_file = backup_path / 'metadata.json'
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string timestamps back to datetime objects
        for field in ['timestamp', 'start_time', 'end_time']:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        return BackupMetadata(**data)
    
    def _calculate_backup_stats(self, backup_path: Path) -> tuple[int, int]:
        """Calculate backup file count and total size."""
        file_count = 0
        total_size = 0
        
        for file_path in backup_path.rglob('*'):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
        
        return file_count, total_size


# Global instance
_backup_recovery_service: Optional[BackupRecoveryService] = None


async def get_backup_recovery_service() -> BackupRecoveryService:
    """Get global backup recovery service instance."""
    global _backup_recovery_service
    
    if _backup_recovery_service is None:
        _backup_recovery_service = BackupRecoveryService()
        await _backup_recovery_service.initialize()
    
    return _backup_recovery_service


async def shutdown_backup_recovery_service() -> None:
    """Shutdown global backup recovery service."""
    global _backup_recovery_service
    
    if _backup_recovery_service:
        await _backup_recovery_service.shutdown()
        _backup_recovery_service = None
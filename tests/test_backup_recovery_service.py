"""
Tests for backup and recovery service.
"""
import pytest
import asyncio
import tempfile
import shutil
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.services.backup_recovery_service import (
    BackupRecoveryService,
    BackupMetadata,
    get_backup_recovery_service
)


class TestBackupRecoveryService:
    """Test cases for BackupRecoveryService."""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        with patch('src.services.backup_recovery_service.TimescaleDBRepository') as mock_timescale, \
             patch('src.services.backup_recovery_service.InfluxDBRepository') as mock_influx, \
             patch('src.services.backup_recovery_service.RedisCache') as mock_redis:
            
            # Setup mock instances
            mock_timescale_instance = AsyncMock()
            mock_influx_instance = AsyncMock()
            mock_redis_instance = AsyncMock()
            
            mock_timescale.return_value = mock_timescale_instance
            mock_influx.return_value = mock_influx_instance
            mock_redis.return_value = mock_redis_instance
            
            # Mock query methods
            mock_timescale_instance.query_market_data.return_value = [
                {'symbol': 'AAPL', 'close': 150.0, 'timestamp': datetime.now(timezone.utc).isoformat()}
            ]
            
            mock_redis_instance.get_keys_by_pattern.return_value = ['key1', 'key2']
            mock_redis_instance.get.return_value = {'test': 'data'}
            
            yield {
                'timescale': mock_timescale_instance,
                'influx': mock_influx_instance,
                'redis': mock_redis_instance
            }
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_repositories, temp_backup_dir):
        """Test service initialization."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        assert service.timescaledb is not None
        assert service.influxdb is not None
        assert service.redis is not None
        
        # Verify connections were established
        mock_repositories['timescale'].connect.assert_called_once()
        mock_repositories['influx'].connect.assert_called_once()
        mock_repositories['redis'].connect.assert_called_once()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_full_backup(self, mock_repositories, temp_backup_dir):
        """Test creating a full backup."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create full backup
        metadata = await service.create_full_backup(backup_name="test_backup")
        
        assert metadata.backup_id == "test_backup"
        assert metadata.backup_type == "full"
        assert metadata.status == "completed"
        assert "timescaledb" in metadata.data_sources
        assert "influxdb" in metadata.data_sources
        assert "redis" in metadata.data_sources
        
        # Verify backup directory was created
        backup_path = Path(temp_backup_dir) / "test_backup"
        assert backup_path.exists()
        assert (backup_path / "metadata.json").exists()
        assert (backup_path / "timescaledb").exists()
        assert (backup_path / "influxdb").exists()
        assert (backup_path / "redis").exists()
        
        # Verify data was queried
        mock_repositories['timescale'].query_market_data.assert_called_once()
        mock_repositories['redis'].get_keys_by_pattern.assert_called_once()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_incremental_backup(self, mock_repositories, temp_backup_dir):
        """Test creating an incremental backup."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Create incremental backup
        metadata = await service.create_incremental_backup(
            since=since,
            backup_name="test_incremental"
        )
        
        assert metadata.backup_id == "test_incremental"
        assert metadata.backup_type == "incremental"
        assert metadata.status == "completed"
        assert metadata.start_time == since
        
        # Verify backup directory was created
        backup_path = Path(temp_backup_dir) / "test_incremental"
        assert backup_path.exists()
        assert (backup_path / "metadata.json").exists()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_list_backups(self, mock_repositories, temp_backup_dir):
        """Test listing backups."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create multiple backups
        await service.create_full_backup(backup_name="backup1")
        await service.create_full_backup(backup_name="backup2")
        
        # List backups
        backups = await service.list_backups()
        
        assert len(backups) == 2
        backup_ids = [b.backup_id for b in backups]
        assert "backup1" in backup_ids
        assert "backup2" in backup_ids
        
        # Should be sorted by timestamp (newest first)
        assert backups[0].timestamp >= backups[1].timestamp
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_delete_backup(self, mock_repositories, temp_backup_dir):
        """Test deleting a backup."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create backup
        await service.create_full_backup(backup_name="test_delete")
        
        # Verify backup exists
        backup_path = Path(temp_backup_dir) / "test_delete"
        assert backup_path.exists()
        
        # Delete backup
        result = await service.delete_backup("test_delete")
        assert result is True
        
        # Verify backup was deleted
        assert not backup_path.exists()
        
        # Try to delete non-existent backup
        result = await service.delete_backup("non_existent")
        assert result is False
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_verify_backup(self, mock_repositories, temp_backup_dir):
        """Test backup verification."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create backup
        metadata = await service.create_full_backup(backup_name="test_verify")
        
        # Verify backup
        verification = await service.verify_backup("test_verify")
        
        assert verification['valid'] is True
        assert verification['files_verified'] > 0
        assert len(verification['missing_files']) == 0
        assert len(verification['corrupted_files']) == 0
        assert verification['metadata']['backup_id'] == "test_verify"
        
        # Test verification of non-existent backup
        verification = await service.verify_backup("non_existent")
        assert verification['valid'] is False
        assert 'error' in verification
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_restore_backup(self, mock_repositories, temp_backup_dir):
        """Test restoring from backup."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create backup
        await service.create_full_backup(backup_name="test_restore")
        
        # Restore backup
        result = await service.restore_backup("test_restore")
        assert result is True
        
        # Verify Redis set was called (for Redis restore)
        mock_repositories['redis'].set.assert_called()
        
        # Test restore of non-existent backup
        result = await service.restore_backup("non_existent")
        assert result is False
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, mock_repositories, temp_backup_dir):
        """Test cleaning up old backups."""
        service = BackupRecoveryService(
            backup_root_path=temp_backup_dir,
            retention_days=1  # Short retention for testing
        )
        await service.initialize()
        
        # Create backup
        await service.create_full_backup(backup_name="old_backup")
        
        # Manually modify backup timestamp to make it old
        backup_path = Path(temp_backup_dir) / "old_backup"
        metadata_file = backup_path / "metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Set timestamp to 2 days ago
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=2)
        metadata['timestamp'] = old_timestamp.isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Cleanup old backups
        deleted_count = await service.cleanup_old_backups()
        
        assert deleted_count == 1
        assert not backup_path.exists()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_backup_with_compression(self, mock_repositories, temp_backup_dir):
        """Test backup with compression enabled."""
        service = BackupRecoveryService(
            backup_root_path=temp_backup_dir,
            compression_enabled=True
        )
        await service.initialize()
        
        # Create backup
        metadata = await service.create_full_backup(backup_name="compressed_backup")
        
        assert metadata.compression is True
        
        # Check for compressed files
        backup_path = Path(temp_backup_dir) / "compressed_backup"
        compressed_files = list(backup_path.rglob("*.gz"))
        assert len(compressed_files) > 0
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_backup_without_redis(self, mock_repositories, temp_backup_dir):
        """Test backup without Redis data."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create backup without Redis
        metadata = await service.create_full_backup(
            backup_name="no_redis_backup",
            include_redis=False
        )
        
        assert "redis" not in metadata.data_sources
        
        # Verify Redis directory was not created
        backup_path = Path(temp_backup_dir) / "no_redis_backup"
        assert not (backup_path / "redis").exists()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_backup_failure_handling(self, mock_repositories, temp_backup_dir):
        """Test handling of backup failures."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Mock TimescaleDB failure
        mock_repositories['timescale'].query_market_data.side_effect = Exception("Database error")
        
        # Attempt backup (should fail)
        with pytest.raises(Exception):
            await service.create_full_backup(backup_name="failed_backup")
        
        # Verify failed backup metadata was saved
        backup_path = Path(temp_backup_dir) / "failed_backup"
        metadata_file = backup_path / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            assert metadata['status'] == 'failed'
            assert 'error_message' in metadata
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_limit(self, mock_repositories, temp_backup_dir):
        """Test concurrent operations limit."""
        service = BackupRecoveryService(
            backup_root_path=temp_backup_dir,
            max_concurrent_operations=1  # Limit to 1 concurrent operation
        )
        await service.initialize()
        
        # Start multiple backup operations concurrently
        tasks = [
            service.create_full_backup(backup_name=f"concurrent_backup_{i}")
            for i in range(3)
        ]
        
        # All should complete successfully (queued by semaphore)
        results = await asyncio.gather(*tasks)
        
        assert all(result.status == "completed" for result in results)
        assert len(set(result.backup_id for result in results)) == 3
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_backup_metadata_persistence(self, mock_repositories, temp_backup_dir):
        """Test that backup metadata is properly persisted and loaded."""
        service = BackupRecoveryService(backup_root_path=temp_backup_dir)
        await service.initialize()
        
        # Create backup
        original_metadata = await service.create_full_backup(backup_name="metadata_test")
        
        # Load metadata directly
        backup_path = Path(temp_backup_dir) / "metadata_test"
        loaded_metadata = await service._load_backup_metadata(backup_path)
        
        # Compare key fields
        assert loaded_metadata.backup_id == original_metadata.backup_id
        assert loaded_metadata.backup_type == original_metadata.backup_type
        assert loaded_metadata.status == original_metadata.status
        assert loaded_metadata.data_sources == original_metadata.data_sources
        
        await service.shutdown()


class TestBackupRecoveryServiceIntegration:
    """Integration tests for BackupRecoveryService."""
    
    @pytest.mark.asyncio
    async def test_global_service_instance(self):
        """Test global service instance management."""
        with patch('src.services.backup_recovery_service.TimescaleDBRepository'), \
             patch('src.services.backup_recovery_service.InfluxDBRepository'), \
             patch('src.services.backup_recovery_service.RedisCache'):
            
            # Get service instance
            service1 = await get_backup_recovery_service()
            service2 = await get_backup_recovery_service()
            
            # Should be the same instance
            assert service1 is service2
            
            # Cleanup
            from src.services.backup_recovery_service import shutdown_backup_recovery_service
            await shutdown_backup_recovery_service()
    
    @pytest.mark.asyncio
    async def test_end_to_end_backup_restore_cycle(self, temp_backup_dir):
        """Test complete backup and restore cycle."""
        with patch('src.services.backup_recovery_service.TimescaleDBRepository') as mock_timescale, \
             patch('src.services.backup_recovery_service.InfluxDBRepository') as mock_influx, \
             patch('src.services.backup_recovery_service.RedisCache') as mock_redis:
            
            # Setup mocks
            mock_timescale_instance = AsyncMock()
            mock_influx_instance = AsyncMock()
            mock_redis_instance = AsyncMock()
            
            mock_timescale.return_value = mock_timescale_instance
            mock_influx.return_value = mock_influx_instance
            mock_redis.return_value = mock_redis_instance
            
            # Mock data
            test_market_data = [
                {'symbol': 'AAPL', 'close': 150.0},
                {'symbol': 'GOOGL', 'close': 2500.0}
            ]
            test_redis_data = {'key1': 'value1', 'key2': 'value2'}
            
            mock_timescale_instance.query_market_data.return_value = test_market_data
            mock_redis_instance.get_keys_by_pattern.return_value = list(test_redis_data.keys())
            mock_redis_instance.get.side_effect = lambda key: test_redis_data.get(key)
            
            service = BackupRecoveryService(backup_root_path=temp_backup_dir)
            await service.initialize()
            
            # Create backup
            backup_metadata = await service.create_full_backup(backup_name="e2e_test")
            assert backup_metadata.status == "completed"
            
            # Verify backup
            verification = await service.verify_backup("e2e_test")
            assert verification['valid'] is True
            
            # Restore backup
            restore_result = await service.restore_backup("e2e_test")
            assert restore_result is True
            
            # Verify restore operations were called
            assert mock_redis_instance.set.call_count == len(test_redis_data)
            
            await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
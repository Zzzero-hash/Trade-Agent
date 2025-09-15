"""
Demo of the data storage and caching layer.

This example demonstrates how to use the unified data storage service
to store and retrieve market data, predictions, and performance metrics
across TimescaleDB, InfluxDB, and Redis.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List

from src.services.data_storage_service import DataStorageService
from src.services.backup_recovery_service import BackupRecoveryService
from src.models.market_data import MarketData, ExchangeType
from src.models.trading_signal import TradingSignal
from src.models.time_series import PredictionPoint, PerformanceMetric


async def demo_market_data_storage():
    """Demonstrate market data storage and retrieval."""
    print("=== Market Data Storage Demo ===")
    
    # Initialize data storage service
    storage_service = DataStorageService(
        enable_influxdb=False,  # Disable for demo (requires InfluxDB server)
        enable_redis=False      # Disable for demo (requires Redis server)
    )
    
    try:
        await storage_service.initialize()
        print("✓ Data storage service initialized")
        
        # Create sample market data
        market_data_list = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(10):
            market_data = MarketData(
                symbol="AAPL",
                timestamp=base_time + timedelta(minutes=i),
                open=150.0 + i * 0.5,
                high=152.0 + i * 0.5,
                low=149.0 + i * 0.5,
                close=151.0 + i * 0.5,
                volume=1000000.0 + i * 1000,
                exchange=ExchangeType.ROBINHOOD
            )
            market_data_list.append(market_data)
        
        # Store market data in batch
        print(f"Storing {len(market_data_list)} market data points...")
        success = await storage_service.store_market_data_batch(market_data_list)
        
        if success:
            print("✓ Market data stored successfully")
        else:
            print("✗ Failed to store market data")
        
        # Retrieve market data
        print("Retrieving market data...")
        start_time = base_time - timedelta(minutes=5)
        end_time = base_time + timedelta(minutes=15)
        
        retrieved_data = await storage_service.get_market_data(
            symbol="AAPL",
            exchange="robinhood",
            start_time=start_time,
            end_time=end_time,
            use_cache=False  # Query from database
        )
        
        print(f"✓ Retrieved {len(retrieved_data)} market data records")
        
        # Get latest market data
        latest_data = await storage_service.get_latest_market_data("AAPL", "robinhood")
        if latest_data:
            print(f"✓ Latest market data: {latest_data.get('close', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Error in market data demo: {e}")
    
    finally:
        await storage_service.shutdown()
        print("✓ Data storage service shutdown")


async def demo_prediction_storage():
    """Demonstrate prediction storage and retrieval."""
    print("\n=== Prediction Storage Demo ===")
    
    storage_service = DataStorageService(
        enable_influxdb=False,
        enable_redis=False
    )
    
    try:
        await storage_service.initialize()
        print("✓ Data storage service initialized")
        
        # Create sample predictions
        predictions = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(5):
            prediction = PredictionPoint(
                symbol="AAPL",
                timestamp=base_time + timedelta(minutes=i * 5),
                model_name="cnn_lstm_hybrid",
                model_version="v1.0",
                predicted_price=155.0 + i * 0.2,
                predicted_direction="BUY" if i % 2 == 0 else "SELL",
                confidence_score=0.8 + i * 0.02,
                uncertainty=0.1 - i * 0.01,
                feature_importance={
                    "rsi": 0.3 + i * 0.01,
                    "macd": 0.2 + i * 0.01,
                    "volume": 0.15 + i * 0.005
                }
            )
            predictions.append(prediction)
        
        # Store predictions
        print(f"Storing {len(predictions)} predictions...")
        for prediction in predictions:
            success = await storage_service.store_prediction(prediction)
            if not success:
                print(f"✗ Failed to store prediction for {prediction.timestamp}")
        
        print("✓ Predictions stored successfully")
        
        # Retrieve predictions
        print("Retrieving predictions...")
        start_time = base_time - timedelta(minutes=5)
        end_time = base_time + timedelta(minutes=30)
        
        retrieved_predictions = await storage_service.get_predictions(
            symbol="AAPL",
            model_name="cnn_lstm_hybrid",
            start_time=start_time,
            end_time=end_time,
            use_cache=False
        )
        
        print(f"✓ Retrieved {len(retrieved_predictions)} predictions")
        
    except Exception as e:
        print(f"✗ Error in prediction demo: {e}")
    
    finally:
        await storage_service.shutdown()
        print("✓ Data storage service shutdown")


async def demo_trading_signals():
    """Demonstrate trading signal storage and caching."""
    print("\n=== Trading Signal Demo ===")
    
    storage_service = DataStorageService(
        enable_influxdb=False,
        enable_redis=False  # Would normally be True for caching
    )
    
    try:
        await storage_service.initialize()
        print("✓ Data storage service initialized")
        
        # Create sample trading signals
        signals = []
        base_time = datetime.now(timezone.utc)
        
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            signal = TradingSignal(
                symbol=symbol,
                action="BUY" if i % 2 == 0 else "SELL",
                confidence=0.85 + i * 0.02,
                position_size=0.1 + i * 0.05,
                target_price=150.0 + i * 50,
                stop_loss=140.0 + i * 45,
                timestamp=base_time + timedelta(minutes=i),
                model_version="v1.0"
            )
            signals.append(signal)
        
        # Store trading signals
        print(f"Storing {len(signals)} trading signals...")
        for signal in signals:
            success = await storage_service.store_trading_signal(signal)
            if success:
                print(f"✓ Stored signal for {signal.symbol}: {signal.action}")
            else:
                print(f"✗ Failed to store signal for {signal.symbol}")
        
        # Retrieve trading signals (would come from cache if Redis enabled)
        print("Retrieving trading signals...")
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            cached_signal = await storage_service.get_trading_signal(symbol)
            if cached_signal:
                print(f"✓ Retrieved signal for {symbol}: {cached_signal.get('action', 'N/A')}")
            else:
                print(f"- No cached signal for {symbol}")
        
    except Exception as e:
        print(f"✗ Error in trading signal demo: {e}")
    
    finally:
        await storage_service.shutdown()
        print("✓ Data storage service shutdown")


async def demo_performance_metrics():
    """Demonstrate performance metrics storage."""
    print("\n=== Performance Metrics Demo ===")
    
    storage_service = DataStorageService(
        enable_influxdb=False,
        enable_redis=False
    )
    
    try:
        await storage_service.initialize()
        print("✓ Data storage service initialized")
        
        # Create sample performance metrics
        metrics = []
        base_time = datetime.now(timezone.utc)
        
        metric_names = ["sharpe_ratio", "max_drawdown", "total_return", "win_rate"]
        metric_values = [1.5, -0.15, 0.25, 0.65]
        
        for i, (name, value) in enumerate(zip(metric_names, metric_values)):
            metric = PerformanceMetric(
                timestamp=base_time + timedelta(minutes=i),
                metric_name=name,
                metric_value=value,
                symbol="AAPL",
                strategy="momentum",
                model_name="cnn_lstm_hybrid",
                timeframe="1d"
            )
            metrics.append(metric)
        
        # Store performance metrics
        print(f"Storing {len(metrics)} performance metrics...")
        for metric in metrics:
            success = await storage_service.store_performance_metric(metric)
            if success:
                print(f"✓ Stored {metric.metric_name}: {metric.metric_value}")
            else:
                print(f"✗ Failed to store {metric.metric_name}")
        
    except Exception as e:
        print(f"✗ Error in performance metrics demo: {e}")
    
    finally:
        await storage_service.shutdown()
        print("✓ Data storage service shutdown")


async def demo_data_quality_report():
    """Demonstrate data quality reporting."""
    print("\n=== Data Quality Report Demo ===")
    
    storage_service = DataStorageService(
        enable_influxdb=False,
        enable_redis=False
    )
    
    try:
        await storage_service.initialize()
        print("✓ Data storage service initialized")
        
        # First, store some test data with gaps and anomalies
        market_data_list = []
        base_time = datetime.now(timezone.utc)
        
        # Normal data
        for i in range(5):
            market_data = MarketData(
                symbol="TEST",
                timestamp=base_time + timedelta(minutes=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000000.0,
                exchange=ExchangeType.ROBINHOOD
            )
            market_data_list.append(market_data)
        
        # Data with gap (10 minutes later)
        gap_time = base_time + timedelta(minutes=15)
        market_data = MarketData(
            symbol="TEST",
            timestamp=gap_time,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        market_data_list.append(market_data)
        
        # Data with price anomaly (large jump)
        anomaly_time = base_time + timedelta(minutes=16)
        market_data = MarketData(
            symbol="TEST",
            timestamp=anomaly_time,
            open=100.0,
            high=120.0,  # 20% jump
            low=99.0,
            close=115.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        market_data_list.append(market_data)
        
        # Store test data
        await storage_service.store_market_data_batch(market_data_list)
        print("✓ Test data stored")
        
        # Generate data quality report
        print("Generating data quality report...")
        start_time = base_time - timedelta(minutes=5)
        end_time = base_time + timedelta(minutes=20)
        
        quality_report = await storage_service.get_data_quality_report(
            symbol="TEST",
            exchange="robinhood",
            start_time=start_time,
            end_time=end_time
        )
        
        if 'error' not in quality_report:
            print("✓ Data Quality Report:")
            print(f"  - Total records: {quality_report['total_records']}")
            print(f"  - Data gaps: {quality_report['data_gaps']['count']}")
            print(f"  - Large price movements: {quality_report['price_anomalies']['large_movements_count']}")
            print(f"  - Quality score: {quality_report['quality_score']:.1f}/100")
        else:
            print(f"✗ Failed to generate quality report: {quality_report['error']}")
        
    except Exception as e:
        print(f"✗ Error in data quality demo: {e}")
    
    finally:
        await storage_service.shutdown()
        print("✓ Data storage service shutdown")


async def demo_backup_recovery():
    """Demonstrate backup and recovery functionality."""
    print("\n=== Backup and Recovery Demo ===")
    
    backup_service = BackupRecoveryService(
        backup_root_path="/tmp/demo_backups",
        compression_enabled=True,
        retention_days=7
    )
    
    try:
        await backup_service.initialize()
        print("✓ Backup recovery service initialized")
        
        # Create a full backup
        print("Creating full backup...")
        backup_metadata = await backup_service.create_full_backup(
            backup_name="demo_backup",
            include_redis=False  # Skip Redis for demo
        )
        
        print(f"✓ Backup created: {backup_metadata.backup_id}")
        print(f"  - Status: {backup_metadata.status}")
        print(f"  - Data sources: {backup_metadata.data_sources}")
        print(f"  - File count: {backup_metadata.file_count}")
        print(f"  - Size: {backup_metadata.total_size_bytes} bytes")
        
        # List all backups
        print("\nListing all backups...")
        backups = await backup_service.list_backups()
        for backup in backups:
            print(f"  - {backup.backup_id} ({backup.backup_type}) - {backup.status}")
        
        # Verify backup
        print("\nVerifying backup...")
        verification = await backup_service.verify_backup("demo_backup")
        if verification['valid']:
            print("✓ Backup verification passed")
            print(f"  - Files verified: {verification['files_verified']}")
        else:
            print("✗ Backup verification failed")
            print(f"  - Error: {verification.get('error', 'Unknown error')}")
        
        # Cleanup demo backup
        print("\nCleaning up demo backup...")
        deleted = await backup_service.delete_backup("demo_backup")
        if deleted:
            print("✓ Demo backup deleted")
        
    except Exception as e:
        print(f"✗ Error in backup demo: {e}")
    
    finally:
        await backup_service.shutdown()
        print("✓ Backup recovery service shutdown")


async def demo_storage_statistics():
    """Demonstrate storage statistics and monitoring."""
    print("\n=== Storage Statistics Demo ===")
    
    storage_service = DataStorageService(
        enable_influxdb=False,
        enable_redis=False
    )
    
    try:
        await storage_service.initialize()
        print("✓ Data storage service initialized")
        
        # Get health status
        print("Checking storage health...")
        health_status = await storage_service.health_check()
        for backend, status in health_status.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {backend}: {'Healthy' if status else 'Unhealthy'}")
        
        # Get storage statistics
        print("\nGetting storage statistics...")
        stats = await storage_service.get_storage_stats()
        
        if 'timescaledb' in stats:
            db_stats = stats['timescaledb']
            print(f"✓ TimescaleDB statistics:")
            print(f"  - Database size: {db_stats.get('database_size', 'N/A')}")
            if 'row_counts' in db_stats:
                for table, count in db_stats['row_counts'].items():
                    print(f"  - {table}: {count} rows")
        
        if 'redis' in stats:
            cache_stats = stats['redis']
            print(f"✓ Redis statistics:")
            print(f"  - Used memory: {cache_stats.get('used_memory_human', 'N/A')}")
            print(f"  - Keyspace hits: {cache_stats.get('keyspace_hits', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Error in statistics demo: {e}")
    
    finally:
        await storage_service.shutdown()
        print("✓ Data storage service shutdown")


async def main():
    """Run all demos."""
    print("🚀 Data Storage and Caching Layer Demo")
    print("=" * 50)
    
    # Note: These demos use mocked repositories since we don't have
    # actual database servers running. In a real environment, you would
    # need TimescaleDB, InfluxDB, and Redis servers running.
    
    print("\n⚠️  Note: This demo uses simplified storage (no actual databases)")
    print("   In production, you would need TimescaleDB, InfluxDB, and Redis servers")
    
    try:
        await demo_market_data_storage()
        await demo_prediction_storage()
        await demo_trading_signals()
        await demo_performance_metrics()
        await demo_data_quality_report()
        await demo_backup_recovery()
        await demo_storage_statistics()
        
        print("\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
"""Integration between feature extraction monitoring and caching/connection pooling.

This module provides integration between the feature extraction performance
monitoring system and the caching/connection pooling monitoring systems
to provide unified visibility into all components affecting feature extraction
performance.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from src.ml.feature_extraction.monitoring import FeatureExtractionMetrics
from src.ml.feature_extraction.enhanced_metrics import EnhancedMetricsCollector
from src.connection_pool.redis_pool import get_redis_pool_manager
from src.connection_pool.database_pool import get_database_pool_manager
from src.repositories.redis_cache import RedisCache
from src.ml.feature_extraction.cached_extractor import CachedFeatureExtractor

logger = logging.getLogger(__name__)


class CacheConnectionIntegration:
    """Integration between feature extraction monitoring and caching/connection pooling"""
    
    def __init__(self):
        """Initialize cache and connection pooling integration."""
        self.enhanced_metrics_collector = EnhancedMetricsCollector()
        self.redis_pool_manager = get_redis_pool_manager()
        self.database_pool_manager = get_database_pool_manager()
        
        # Integration state
        self.is_initialized = False
        self.last_sync_time = 0
        self.sync_interval = 30  # Sync every 30 seconds for pool stats
        
        logger.info("Cache and connection pooling integration for feature extraction monitoring initialized")
    
    def initialize_integration(self) -> None:
        """Initialize the integration between monitoring systems."""
        if self.is_initialized:
            logger.warning("Cache/Connection integration already initialized")
            return
        
        # Verify pool managers are available
        if self.redis_pool_manager is None:
            logger.warning("Redis pool manager not available")
        
        if self.database_pool_manager is None:
            logger.warning("Database pool manager not available")
        
        self.is_initialized = True
        logger.info("Cache/Connection integration initialized successfully")
    
    def sync_cache_metrics(self, cached_extractor: Optional[CachedFeatureExtractor] = None) -> None:
        """Sync cache metrics with monitoring system.
        
        Args:
            cached_extractor: Optional cached extractor to get cache stats from
        """
        if not self.is_initialized:
            self.initialize_integration()
        
        # Get cache information if available
        cache_info = {}
        if cached_extractor and hasattr(cached_extractor, 'get_cache_info'):
            try:
                cache_info = cached_extractor.get_cache_info()
                if cache_info:
                    # Add cache metrics to enhanced collector
                    if 'cache_size' in cache_info:
                        self.enhanced_metrics_collector.metrics_collector.set_gauge(
                            "feature_extraction_cache_size",
                            cache_info['cache_size']
                        )
                    
                    if 'performance_summary' in cache_info and cache_info['performance_summary']:
                        perf_summary = cache_info['performance_summary']
                        if 'cache_hit_rate' in perf_summary:
                            self.enhanced_metrics_collector.metrics_collector.set_gauge(
                                "feature_extraction_cache_hit_rate",
                                perf_summary['cache_hit_rate']
                            )
            except Exception as e:
                logger.warning(f"Failed to sync cache metrics: {e}")
        
        # Sync with Redis pool stats if available
        self._sync_redis_pool_stats()
        
        # Sync with database pool stats if available
        self._sync_database_pool_stats()
    
    def _sync_redis_pool_stats(self) -> None:
        """Sync Redis pool statistics."""
        if self.redis_pool_manager is None:
            return
        
        try:
            # Get all Redis pools
            pool_ids = list(self.redis_pool_manager._pools.keys()) if hasattr(self.redis_pool_manager, '_pools') else []
            
            for pool_id in pool_ids:
                stats = self.redis_pool_manager.get_pool_stats(pool_id)
                if stats:
                    # Record Redis pool metrics
                    self.enhanced_metrics_collector.metrics_collector.set_gauge(
                        "redis_pool_connections_used",
                        stats.get('connected_clients', 0),
                        {"pool_id": pool_id}
                    )
                    
                    self.enhanced_metrics_collector.metrics_collector.set_gauge(
                        "redis_pool_memory_used_mb",
                        stats.get('used_memory', 0) / (1024 * 1024),
                        {"pool_id": pool_id}
                    )
                    
                    # Record key metrics
                    hits = stats.get('keyspace_hits', 0)
                    misses = stats.get('keyspace_misses', 0)
                    total_commands = stats.get('total_commands_processed', 0)
                    
                    if hits + misses > 0:
                        hit_rate = hits / (hits + misses)
                        self.enhanced_metrics_collector.metrics_collector.set_gauge(
                            "redis_keyspace_hit_rate",
                            hit_rate,
                            {"pool_id": pool_id}
                        )
                    
                    self.enhanced_metrics_collector.metrics_collector.set_gauge(
                        "redis_total_commands_processed",
                        total_commands,
                        {"pool_id": pool_id}
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to sync Redis pool stats: {e}")
    
    def _sync_database_pool_stats(self) -> None:
        """Sync database pool statistics."""
        if self.database_pool_manager is None:
            return
        
        try:
            # Get all database pools
            pool_ids = list(self.database_pool_manager._pools.keys()) if hasattr(self.database_pool_manager, '_pools') else []
            
            for pool_id in pool_ids:
                stats = self.database_pool_manager.get_pool_stats(pool_id)
                if stats:
                    # Record database pool metrics
                    self.enhanced_metrics_collector.metrics_collector.set_gauge(
                        "database_pool_current_size",
                        stats.get('current_size', 0),
                        {"pool_id": pool_id}
                    )
                    
                    self.enhanced_metrics_collector.metrics_collector.set_gauge(
                        "database_pool_idle_size",
                        stats.get('idle_size', 0),
                        {"pool_id": pool_id}
                    )
                    
                    self.enhanced_metrics_collector.metrics_collector.set_gauge(
                        "database_pool_queue_length",
                        stats.get('queue_length', 0),
                        {"pool_id": pool_id}
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to sync database pool stats: {e}")
    
    def get_unified_resource_stats(self) -> Dict[str, Any]:
        """Get unified resource statistics from all integrated systems.
        
        Returns:
            Dictionary with unified resource statistics
        """
        # Get feature extraction resource utilization
        feature_resources = self.enhanced_metrics_collector.get_resource_utilization()
        
        # Get cache statistics
        cache_stats = self._get_cache_statistics()
        
        # Get connection pool statistics
        connection_stats = self._get_connection_pool_statistics()
        
        # Combine all stats
        unified_stats = {
            'timestamp': datetime.now().isoformat(),
            'feature_extraction': feature_resources,
            'cache': cache_stats,
            'connection_pools': connection_stats
        }
        
        return unified_stats
    
    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = {
            'redis_pools': {},
            'cache_hit_rates': {}
        }
        
        # Get Redis pool stats
        if self.redis_pool_manager:
            try:
                pool_ids = list(self.redis_pool_manager._pools.keys()) if hasattr(self.redis_pool_manager, '_pools') else []
                for pool_id in pool_ids:
                    stats = self.redis_pool_manager.get_pool_stats(pool_id)
                    if stats:
                        cache_stats['redis_pools'][pool_id] = {
                            'connected_clients': stats.get('connected_clients', 0),
                            'memory_used_mb': stats.get('used_memory', 0) / (1024 * 1024),
                            'keyspace_hits': stats.get('keyspace_hits', 0),
                            'keyspace_misses': stats.get('keyspace_misses', 0)
                        }
                        
                        # Calculate hit rate
                        hits = stats.get('keyspace_hits', 0)
                        misses = stats.get('keyspace_misses', 0)
                        if hits + misses > 0:
                            hit_rate = hits / (hits + misses)
                            cache_stats['cache_hit_rates'][pool_id] = hit_rate
            except Exception as e:
                logger.warning(f"Failed to get Redis cache statistics: {e}")
        
        return cache_stats
    
    def _get_connection_pool_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary with connection pool statistics
        """
        pool_stats = {
            'redis_pools': {},
            'database_pools': {}
        }
        
        # Get Redis pool stats
        if self.redis_pool_manager:
            try:
                pool_ids = list(self.redis_pool_manager._pools.keys()) if hasattr(self.redis_pool_manager, '_pools') else []
                for pool_id in pool_ids:
                    stats = self.redis_pool_manager.get_pool_stats(pool_id)
                    if stats:
                        pool_stats['redis_pools'][pool_id] = stats
            except Exception as e:
                logger.warning(f"Failed to get Redis pool statistics: {e}")
        
        # Get database pool stats
        if self.database_pool_manager:
            try:
                pool_ids = list(self.database_pool_manager._pools.keys()) if hasattr(self.database_pool_manager, '_pools') else []
                for pool_id in pool_ids:
                    stats = self.database_pool_manager.get_pool_stats(pool_id)
                    if stats:
                        pool_stats['database_pools'][pool_id] = {
                            'current_size': stats.get('current_size', 0),
                            'idle_size': stats.get('idle_size', 0),
                            'min_size': stats.get('min_size', 0),
                            'max_size': stats.get('max_size', 0),
                            'queue_length': stats.get('queue_length', 0)
                        }
            except Exception as e:
                logger.warning(f"Failed to get database pool statistics: {e}")
        
        return pool_stats
    
    def get_performance_impact_analysis(self) -> Dict[str, Any]:
        """Analyze performance impact of caching and connection pooling.
        
        Returns:
            Dictionary with performance impact analysis
        """
        # Get current performance stats
        performance_summary = self.enhanced_metrics_collector.get_performance_summary()
        resource_utilization = self.enhanced_metrics_collector.get_resource_utilization()
        
        # Get cache stats
        cache_stats = self._get_cache_statistics()
        
        # Get connection pool stats
        connection_stats = self._get_connection_pool_statistics()
        
        # Analyze cache impact
        cache_analysis = self._analyze_cache_impact(cache_stats, performance_summary)
        
        # Analyze connection pool impact
        pool_analysis = self._analyze_pool_impact(connection_stats, resource_utilization)
        
        # Combine analysis
        impact_analysis = {
            'timestamp': datetime.now().isoformat(),
            'cache_impact': cache_analysis,
            'pool_impact': pool_analysis,
            'recommendations': self._generate_recommendations(cache_analysis, pool_analysis)
        }
        
        return impact_analysis
    
    def _analyze_cache_impact(self, cache_stats: Dict[str, Any], performance_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of caching on performance.
        
        Args:
            cache_stats: Cache statistics
            performance_summary: Performance summary
            
        Returns:
            Dictionary with cache impact analysis
        """
        analysis = {
            'cache_hit_rate': performance_summary.get('cache_hit_rate', 0),
            'avg_latency_ms': performance_summary.get('avg_latency_ms', 0),
            'p95_latency_ms': performance_summary.get('p95_latency_ms', 0),
            'pool_hit_rates': cache_stats.get('cache_hit_rates', {}),
            'issues': []
        }
        
        # Check for cache-related issues
        cache_hit_rate = analysis['cache_hit_rate']
        if cache_hit_rate < 0.7:  # Less than 70% cache hit rate
            analysis['issues'].append({
                'type': 'low_cache_hit_rate',
                'severity': 'high',
                'message': f"Cache hit rate is low ({cache_hit_rate:.1%}), consider increasing cache size or TTL"
            })
        
        # Check individual pool hit rates
        pool_hit_rates = analysis['pool_hit_rates']
        for pool_id, hit_rate in pool_hit_rates.items():
            if hit_rate < 0.6:
                analysis['issues'].append({
                    'type': 'low_pool_hit_rate',
                    'severity': 'medium',
                    'message': f"Redis pool '{pool_id}' has low hit rate ({hit_rate:.1%})"
                })
        
        return analysis
    
    def _analyze_pool_impact(self, connection_stats: Dict[str, Any], resource_utilization: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of connection pooling on performance.
        
        Args:
            connection_stats: Connection pool statistics
            resource_utilization: Resource utilization metrics
            
        Returns:
            Dictionary with pool impact analysis
        """
        analysis = {
            'database_pools': {},
            'redis_pools': {},
            'issues': []
        }
        
        # Analyze database pools
        db_pools = connection_stats.get('database_pools', {})
        for pool_id, stats in db_pools.items():
            queue_length = stats.get('queue_length', 0)
            current_size = stats.get('current_size', 0)
            max_size = stats.get('max_size', 1)
            
            utilization = current_size / max_size if max_size > 0 else 0
            
            analysis['database_pools'][pool_id] = {
                'utilization': utilization,
                'queue_length': queue_length
            }
            
            # Check for issues
            if queue_length > 5:
                analysis['issues'].append({
                    'type': 'high_db_queue_length',
                    'severity': 'high',
                    'message': f"Database pool '{pool_id}' has high queue length ({queue_length}), consider increasing pool size"
                })
            
            if utilization > 0.9:
                analysis['issues'].append({
                    'type': 'high_db_utilization',
                    'severity': 'medium',
                    'message': f"Database pool '{pool_id}' is highly utilized ({utilization:.1%})"
                })
        
        # Analyze Redis pools
        redis_pools = connection_stats.get('redis_pools', {})
        for pool_id, stats in redis_pools.items():
            analysis['redis_pools'][pool_id] = {
                'connected_clients': stats.get('connected_clients', 0),
                'memory_used_mb': stats.get('used_memory', 0) / (1024 * 1024)
            }
        
        return analysis
    
    def _generate_recommendations(self, cache_analysis: Dict[str, Any], pool_analysis: Dict[str, Any]) -> list:
        """Generate recommendations based on analysis.
        
        Args:
            cache_analysis: Cache impact analysis
            pool_analysis: Pool impact analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Add cache-related recommendations
        for issue in cache_analysis.get('issues', []):
            recommendations.append({
                'category': 'cache',
                'severity': issue['severity'],
                'message': issue['message']
            })
        
        # Add pool-related recommendations
        for issue in pool_analysis.get('issues', []):
            recommendations.append({
                'category': 'connection_pool',
                'severity': issue['severity'],
                'message': issue['message']
            })
        
        # General performance recommendations
        cache_hit_rate = cache_analysis.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.8:
            recommendations.append({
                'category': 'general',
                'severity': 'medium',
                'message': "Consider optimizing cache strategy to improve hit rate"
            })
        
        return recommendations
    
    def reset_monitoring(self) -> None:
        """Reset all monitoring systems."""
        # Reset enhanced metrics collector
        self.enhanced_metrics_collector.reset_metrics()
        
        logger.info("Cache/Connection monitoring systems reset")
    
    async def start_background_monitoring(self) -> None:
        """Start background monitoring task."""
        async def monitoring_loop():
            while True:
                try:
                    current_time = time.time()
                    if current_time - self.last_sync_time > self.sync_interval:
                        # Sync pool statistics
                        self._sync_redis_pool_stats()
                        self._sync_database_pool_stats()
                        self.last_sync_time = current_time
                    
                    # Wait for next check
                    await asyncio.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error in background monitoring loop: {e}")
                    await asyncio.sleep(5)  # Short sleep on error
        
        # Start monitoring loop in background
        asyncio.create_task(monitoring_loop())
        logger.info("Background cache/connection monitoring started")
    
    def get_monitoring_health(self) -> Dict[str, Any]:
        """Get health status of the cache/connection monitoring integration.
        
        Returns:
            Dictionary with monitoring health status
        """
        return {
            'is_initialized': self.is_initialized,
            'redis_pool_manager_available': self.redis_pool_manager is not None,
            'database_pool_manager_available': self.database_pool_manager is not None,
            'enhanced_metrics_collector_size': len(self.enhanced_metrics_collector.metrics_history),
            'last_sync_time': self.last_sync_time,
            'sync_interval': self.sync_interval
        }
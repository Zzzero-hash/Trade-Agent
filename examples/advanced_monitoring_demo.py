"""
Demo script for the advanced monitoring and alerting system.

This script demonstrates how to use the advanced monitoring system with:
- Real-time anomaly detection
- Model drift detection with statistical significance testing
- Performance degradation monitoring with automated retraining triggers
- System health monitoring with predictive maintenance
- Alert routing and escalation procedures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.services.monitoring.advanced_monitoring_system import (
    AdvancedMonitoringSystem,
    AnomalyDetectionConfig,
    ModelDriftConfig,
    PerformanceDegradationConfig,
    SystemHealthConfig,
    AlertRoutingConfig
)
from src.models.monitoring import ModelPerformanceMetrics, Alert
from src.utils.logging import get_logger

logger = get_logger("monitoring_demo")


class DemoAlertObserver:
    """Demo alert observer that prints alerts to console."""
    
    def __init__(self, name: str):
        self.name = name
        self.alerts_received: List[Alert] = []
    
    async def notify(self, alert: Alert) -> None:
        """Handle alert notification."""
        self.alerts_received.append(alert)
        print(f"\n🚨 [{self.name}] ALERT RECEIVED:")
        print(f"   ID: {alert.id}")
        print(f"   Severity: {alert.severity.value.upper()}")
        print(f"   Title: {alert.title}")
        print(f"   Message: {alert.message}")
        print(f"   Model: {alert.model_name}")
        print(f"   Timestamp: {alert.timestamp}")
        if alert.metadata:
            print(f"   Metadata: {alert.metadata}")
        print("-" * 60)


async def demo_real_time_anomaly_detection():
    """Demonstrate real-time anomaly detection."""
    print("\n" + "="*60)
    print("🔍 DEMO: Real-time Anomaly Detection")
    print("="*60)
    
    # Configure monitoring system
    anomaly_config = AnomalyDetectionConfig(
        z_score_threshold=2.5,
        min_samples_for_detection=10,
        enable_real_time_alerts=True
    )
    
    drift_config = ModelDriftConfig()
    performance_config = PerformanceDegradationConfig()
    health_config = SystemHealthConfig()
    routing_config = AlertRoutingConfig()
    
    monitoring_system = AdvancedMonitoringSystem(
        anomaly_config, drift_config, performance_config,
        health_config, routing_config
    )
    
    # Add alert observer
    alert_observer = DemoAlertObserver("Anomaly Detection")
    monitoring_system.add_alert_observer(alert_observer)
    
    model_name = "demo_trading_model"
    
    print(f"📊 Feeding normal data to model '{model_name}'...")
    
    # Feed normal data
    for i in range(15):
        data_point = {
            "price_change": np.random.normal(0, 1),
            "volume_ratio": np.random.normal(1, 0.2),
            "volatility": np.random.normal(0.15, 0.05),
            "rsi": np.random.normal(50, 10)
        }
        await monitoring_system.process_model_data(model_name, data_point)
        print(f"   Sample {i+1}: price_change={data_point['price_change']:.3f}")
    
    print(f"\n⚠️  Injecting anomalous data...")
    
    # Inject anomalous data
    anomalous_data = {
        "price_change": 8.0,  # 8 standard deviations away
        "volume_ratio": 1.0,
        "volatility": 0.15,
        "rsi": 50.0
    }
    await monitoring_system.process_model_data(model_name, anomalous_data)
    print(f"   Anomaly: price_change={anomalous_data['price_change']:.3f}")
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    print(f"\n📈 Results: {len(alert_observer.alerts_received)} alerts generated")
    
    return len(alert_observer.alerts_received) > 0


async def demo_model_drift_detection():
    """Demonstrate model drift detection with statistical tests."""
    print("\n" + "="*60)
    print("📊 DEMO: Model Drift Detection")
    print("="*60)
    
    # Configure monitoring system
    anomaly_config = AnomalyDetectionConfig()
    drift_config = ModelDriftConfig(
        data_drift_threshold=0.05,
        statistical_significance_level=0.01,
        enable_statistical_tests=True
    )
    performance_config = PerformanceDegradationConfig()
    health_config = SystemHealthConfig()
    routing_config = AlertRoutingConfig()
    
    monitoring_system = AdvancedMonitoringSystem(
        anomaly_config, drift_config, performance_config,
        health_config, routing_config
    )
    
    # Add alert observer
    alert_observer = DemoAlertObserver("Drift Detection")
    monitoring_system.add_alert_observer(alert_observer)
    
    model_name = "drift_demo_model"
    
    print("📊 Generating reference and current data with clear distribution shift...")
    
    # Generate reference data (training distribution)
    reference_data = np.random.normal(0, 1, (100, 5))
    print(f"   Reference data: mean={np.mean(reference_data):.3f}, std={np.std(reference_data):.3f}")
    
    # Generate current data with drift (production distribution)
    current_data = np.random.normal(2, 1.5, (100, 5))  # Clear shift
    print(f"   Current data: mean={np.mean(current_data):.3f}, std={np.std(current_data):.3f}")
    
    # Process data for drift detection
    await monitoring_system.process_model_data(
        model_name,
        {"dummy": 1.0},
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    print(f"\n📈 Results: {len(alert_observer.alerts_received)} drift alerts generated")
    
    return len(alert_observer.alerts_received) > 0


async def demo_performance_degradation_monitoring():
    """Demonstrate performance degradation monitoring."""
    print("\n" + "="*60)
    print("📉 DEMO: Performance Degradation Monitoring")
    print("="*60)
    
    # Configure monitoring system
    anomaly_config = AnomalyDetectionConfig()
    drift_config = ModelDriftConfig()
    performance_config = PerformanceDegradationConfig(
        accuracy_threshold=0.05,
        f1_threshold=0.05,
        consecutive_failures_threshold=2,
        enable_automated_retraining=True
    )
    health_config = SystemHealthConfig()
    routing_config = AlertRoutingConfig()
    
    monitoring_system = AdvancedMonitoringSystem(
        anomaly_config, drift_config, performance_config,
        health_config, routing_config
    )
    
    # Add alert observer
    alert_observer = DemoAlertObserver("Performance Monitoring")
    monitoring_system.add_alert_observer(alert_observer)
    
    model_name = "performance_demo_model"
    
    print("📊 Simulating model performance over time...")
    
    # Simulate good performance initially
    good_metrics = ModelPerformanceMetrics(
        timestamp=datetime.now(),
        model_name=model_name,
        model_version="1.0",
        accuracy=0.92,
        precision=0.89,
        recall=0.91,
        f1_score=0.90,
        sharpe_ratio=1.8,
        max_drawdown=0.08
    )
    
    print(f"   Initial performance: accuracy={good_metrics.accuracy:.3f}, f1={good_metrics.f1_score:.3f}")
    
    await monitoring_system.process_model_data(
        model_name, {"dummy": 1.0}, good_metrics
    )
    
    # Simulate performance degradation
    degraded_metrics = ModelPerformanceMetrics(
        timestamp=datetime.now(),
        model_name=model_name,
        model_version="1.0",
        accuracy=0.78,  # 15% degradation
        precision=0.75,  # 16% degradation
        recall=0.76,     # 16% degradation
        f1_score=0.75,   # 17% degradation
        sharpe_ratio=1.2,  # 33% degradation
        max_drawdown=0.18  # 125% increase (worse)
    )
    
    print(f"   Degraded performance: accuracy={degraded_metrics.accuracy:.3f}, f1={degraded_metrics.f1_score:.3f}")
    
    await monitoring_system.process_model_data(
        model_name, {"dummy": 1.0}, degraded_metrics
    )
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    print(f"\n📈 Results: {len(alert_observer.alerts_received)} performance alerts generated")
    
    return len(alert_observer.alerts_received) > 0


async def demo_system_health_monitoring():
    """Demonstrate system health monitoring."""
    print("\n" + "="*60)
    print("🖥️  DEMO: System Health Monitoring")
    print("="*60)
    
    # Configure monitoring system
    anomaly_config = AnomalyDetectionConfig()
    drift_config = ModelDriftConfig()
    performance_config = PerformanceDegradationConfig()
    health_config = SystemHealthConfig(
        cpu_threshold=75.0,
        memory_threshold=80.0,
        disk_threshold=85.0,
        enable_predictive_maintenance=True
    )
    routing_config = AlertRoutingConfig()
    
    monitoring_system = AdvancedMonitoringSystem(
        anomaly_config, drift_config, performance_config,
        health_config, routing_config
    )
    
    # Add alert observer
    alert_observer = DemoAlertObserver("System Health")
    monitoring_system.add_alert_observer(alert_observer)
    
    print("📊 Starting system health monitoring...")
    
    # Start monitoring
    await monitoring_system.start_monitoring()
    
    # Let it run for a short time
    await asyncio.sleep(2)
    
    # Stop monitoring
    await monitoring_system.stop_monitoring()
    
    print(f"\n📈 Results: {len(alert_observer.alerts_received)} health alerts generated")
    print("   (Note: Health alerts depend on actual system resource usage)")
    
    return True


async def demo_alert_routing_and_escalation():
    """Demonstrate alert routing and escalation."""
    print("\n" + "="*60)
    print("🚨 DEMO: Alert Routing and Escalation")
    print("="*60)
    
    # Configure monitoring system with escalation
    anomaly_config = AnomalyDetectionConfig()
    drift_config = ModelDriftConfig()
    performance_config = PerformanceDegradationConfig()
    health_config = SystemHealthConfig()
    routing_config = AlertRoutingConfig(
        escalation_levels=["team", "manager", "executive"],
        escalation_timeouts=[1, 2, 3],  # Short timeouts for demo (minutes)
        severity_routing={
            "low": ["team"],
            "medium": ["team", "manager"],
            "high": ["team", "manager"],
            "critical": ["team", "manager", "executive"]
        }
    )
    
    monitoring_system = AdvancedMonitoringSystem(
        anomaly_config, drift_config, performance_config,
        health_config, routing_config
    )
    
    # Add multiple alert observers for different levels
    team_observer = DemoAlertObserver("Team")
    manager_observer = DemoAlertObserver("Manager")
    executive_observer = DemoAlertObserver("Executive")
    
    monitoring_system.add_alert_observer(team_observer)
    monitoring_system.add_alert_observer(manager_observer)
    monitoring_system.add_alert_observer(executive_observer)
    
    print("📊 Generating critical alert to test routing...")
    
    # Generate critical performance issue
    critical_metrics = ModelPerformanceMetrics(
        timestamp=datetime.now(),
        model_name="critical_demo_model",
        model_version="1.0",
        accuracy=0.45,  # Severely degraded
        precision=0.40,
        recall=0.42,
        f1_score=0.41
    )
    
    await monitoring_system.process_model_data(
        "critical_demo_model", {"dummy": 1.0}, critical_metrics
    )
    
    # Wait for initial alerts
    await asyncio.sleep(0.1)
    
    total_alerts = (len(team_observer.alerts_received) + 
                   len(manager_observer.alerts_received) + 
                   len(executive_observer.alerts_received))
    
    print(f"\n📈 Results: {total_alerts} total alerts routed")
    print(f"   Team alerts: {len(team_observer.alerts_received)}")
    print(f"   Manager alerts: {len(manager_observer.alerts_received)}")
    print(f"   Executive alerts: {len(executive_observer.alerts_received)}")
    
    return total_alerts > 0


async def main():
    """Run all monitoring system demos."""
    print("🚀 Advanced Monitoring and Alerting System Demo")
    print("=" * 60)
    
    demos = [
        ("Real-time Anomaly Detection", demo_real_time_anomaly_detection),
        ("Model Drift Detection", demo_model_drift_detection),
        ("Performance Degradation Monitoring", demo_performance_degradation_monitoring),
        ("System Health Monitoring", demo_system_health_monitoring),
        ("Alert Routing and Escalation", demo_alert_routing_and_escalation)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🎯 Running: {demo_name}")
            success = await demo_func()
            results[demo_name] = "✅ PASSED" if success else "⚠️  NO ALERTS"
        except Exception as e:
            results[demo_name] = f"❌ FAILED: {e}"
            logger.error(f"Demo '{demo_name}' failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 DEMO RESULTS SUMMARY")
    print("="*60)
    
    for demo_name, result in results.items():
        print(f"   {demo_name}: {result}")
    
    print("\n🎉 Demo completed!")
    print("\nKey Features Demonstrated:")
    print("   • Real-time anomaly detection with Z-score and change point detection")
    print("   • Statistical drift detection with KS test and PSI")
    print("   • Performance degradation monitoring with automated retraining triggers")
    print("   • System health monitoring with predictive maintenance")
    print("   • Alert routing and escalation with multiple severity levels")
    print("   • Comprehensive error handling and reliability features")


if __name__ == "__main__":
    asyncio.run(main())
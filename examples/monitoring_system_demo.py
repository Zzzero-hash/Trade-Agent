"""
Monitoring System Demo

This example demonstrates the comprehensive monitoring and alerting system
for the AI trading platform, including drift detection, performance monitoring,
and automated retraining capabilities.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.services.model_monitoring_service import (
    ModelMonitoringService,
    EmailAlertChannel,
    SlackAlertChannel,
)
from src.models.monitoring import Alert, AlertSeverity, ModelPerformanceMetrics
from src.services.monitoring_dashboard_service import MonitoringDashboardService
from src.services.automated_retraining_service import (
    AutomatedRetrainingService,
    RetrainingConfig
)
from src.utils.monitoring import MetricsCollector, setup_monitoring
from src.config.settings import MonitoringConfig


async def simulate_model_predictions(monitoring_service: ModelMonitoringService,
                                   model_name: str, num_predictions: int = 100,
                                   introduce_drift: bool = False) -> None:
    """Simulate model predictions with optional drift"""
    
    print(f"Simulating {num_predictions} predictions for model {model_name}")
    
    for i in range(num_predictions):
        # Generate features
        if introduce_drift and i > num_predictions // 2:
            # Introduce data drift in second half
            features = np.random.normal(2, 1.5, 10)  # Shifted distribution
            # Introduce performance drift
            prediction = np.random.choice([0, 1], p=[0.8, 0.2])  # Biased predictions
            actual = np.random.choice([0, 1], p=[0.5, 0.5])     # Balanced actuals
        else:
            # Normal distribution
            features = np.random.normal(0, 1, 10)
            prediction = np.random.choice([0, 1])
            actual = prediction if np.random.random() > 0.2 else 1 - prediction  # 80% accuracy
        
        confidence = np.random.uniform(0.6, 0.95)
        
        await monitoring_service.track_prediction(
            model_name=model_name,
            model_version="v1.0",
            features=features,
            prediction=prediction,
            actual=actual,
            confidence=confidence
        )
        
        # Small delay to simulate real-time predictions
        if i % 10 == 0:
            await asyncio.sleep(0.01)


async def setup_alert_channels(monitoring_service: ModelMonitoringService) -> None:
    """Setup alert notification channels"""
    
    # Email alerts (mock configuration)
    email_config = {
        'host': 'smtp.gmail.com',
        'port': 587,
        'username': 'alerts@tradingplatform.com',
        'password': 'app_password'
    }
    email_channel = EmailAlertChannel(email_config)
    monitoring_service.register_alert_channel(email_channel)
    
    # Slack alerts
    slack_webhook = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    slack_channel = SlackAlertChannel(slack_webhook)
    monitoring_service.register_alert_channel(slack_channel)
    
    print("Alert channels configured: Email, Slack")


async def setup_retraining_callbacks(retraining_service: AutomatedRetrainingService) -> None:
    """Setup automated retraining callbacks"""
    
    def cnn_lstm_retraining_callback(job):
        """Custom retraining logic for CNN+LSTM models"""
        print(f"Executing CNN+LSTM retraining for job {job.job_id}")
        # Custom retraining logic would go here
        job.progress = 100.0
        print(f"CNN+LSTM retraining completed for {job.model_name}")
    
    def rl_ensemble_retraining_callback(job):
        """Custom retraining logic for RL ensemble models"""
        print(f"Executing RL ensemble retraining for job {job.job_id}")
        # Custom retraining logic would go here
        job.progress = 100.0
        print(f"RL ensemble retraining completed for {job.model_name}")
    
    # Register callbacks for different model types
    retraining_service.register_retraining_callback("cnn_lstm_model", cnn_lstm_retraining_callback)
    retraining_service.register_retraining_callback("rl_ensemble_model", rl_ensemble_retraining_callback)
    
    print("Retraining callbacks registered for CNN+LSTM and RL ensemble models")


async def demonstrate_monitoring_workflow():
    """Demonstrate complete monitoring workflow"""
    
    print("=== AI Trading Platform Monitoring System Demo ===\n")
    
    # 1. Initialize monitoring infrastructure
    print("1. Initializing monitoring infrastructure...")
    
    monitoring_config = MonitoringConfig(
        enabled=True,
        drift_detection_enabled=True,
        performance_monitoring_enabled=True,
        automated_retraining_enabled=True
    )
    
    metrics_collector = setup_monitoring(monitoring_config)
    monitoring_service = ModelMonitoringService(metrics_collector)
    dashboard_service = MonitoringDashboardService(monitoring_service)
    retraining_service = AutomatedRetrainingService(monitoring_service)
    
    # Configure retraining
    retraining_config = RetrainingConfig(
        enabled=True,
        max_concurrent_jobs=2,
        cooldown_period_hours=1,  # Short for demo
        min_samples_for_retraining=50,  # Low for demo
        performance_threshold=0.05
    )
    retraining_service.configure(retraining_config)
    
    print("✓ Monitoring infrastructure initialized\n")
    
    # 2. Setup alert channels and retraining callbacks
    print("2. Setting up alert channels and retraining callbacks...")
    await setup_alert_channels(monitoring_service)
    await setup_retraining_callbacks(retraining_service)
    print("✓ Alert channels and callbacks configured\n")
    
    # 3. Simulate normal model operation
    print("3. Simulating normal model operation...")
    model_name = "cnn_lstm_model"
    
    # Generate baseline performance
    await simulate_model_predictions(monitoring_service, model_name, 50, introduce_drift=False)
    
    # Calculate and set baseline metrics
    baseline_metrics = await monitoring_service.calculate_performance_metrics(model_name)
    if baseline_metrics:
        monitoring_service.set_baseline_metrics(model_name, baseline_metrics)
        print(f"✓ Baseline metrics set - Accuracy: {baseline_metrics.accuracy:.3f}")
    
    # Get initial dashboard
    system_dashboard = await dashboard_service.get_system_dashboard()
    print(f"✓ System health: {system_dashboard.system_health}")
    print(f"✓ Active models: {system_dashboard.active_models}")
    print(f"✓ Total predictions: {system_dashboard.total_predictions}\n")
    
    # 4. Simulate model drift and performance degradation
    print("4. Simulating model drift and performance degradation...")
    await simulate_model_predictions(monitoring_service, model_name, 100, introduce_drift=True)
    print("✓ Drift simulation completed\n")
    
    # 5. Run monitoring cycle to detect issues
    print("5. Running monitoring cycle to detect drift...")
    monitoring_results = await monitoring_service.run_monitoring_cycle(model_name)
    
    print(f"✓ Monitoring cycle completed for {model_name}")
    if monitoring_results['performance_metrics']:
        current_accuracy = monitoring_results['performance_metrics'].accuracy
        print(f"✓ Current accuracy: {current_accuracy:.3f}")
    
    # Check for drift detection
    data_drift = monitoring_results['drift_detection']['data_drift']
    performance_drift = monitoring_results['drift_detection']['performance_drift']
    
    if data_drift and data_drift.detected:
        print(f"⚠️  Data drift detected - Score: {data_drift.drift_score:.4f}")
    
    if performance_drift and performance_drift.detected:
        print(f"⚠️  Performance drift detected - Score: {performance_drift.drift_score:.4f}")
    
    print()
    
    # 6. Demonstrate automated retraining
    print("6. Demonstrating automated retraining...")
    
    # Trigger manual retraining
    job_id = await retraining_service.schedule_manual_retraining(
        model_name, 
        {"learning_rate": 0.001, "epochs": 10}
    )
    
    print(f"✓ Manual retraining scheduled - Job ID: {job_id}")
    
    # Wait a moment for job to process
    await asyncio.sleep(0.1)
    
    # Check job status
    job_status = retraining_service.get_job_status(job_id)
    if job_status:
        print(f"✓ Job status: {job_status['status']}")
        print(f"✓ Progress: {job_status['progress']:.1f}%")
    
    # Get retraining summary
    retraining_summary = retraining_service.get_retraining_summary()
    print(f"✓ Active retraining jobs: {retraining_summary['active_jobs']}")
    print(f"✓ Success rate: {retraining_summary['success_rate']:.1%}\n")
    
    # 7. Generate comprehensive reports
    print("7. Generating comprehensive monitoring reports...")
    
    # System health report
    health_report = await dashboard_service.get_system_health_report()
    print(f"✓ Overall system health: {health_report['overall_status']}")
    print(f"✓ Health score: {health_report['overall_health_score']:.1f}/100")
    
    # Model dashboard
    model_dashboard = await dashboard_service.get_model_dashboard(model_name)
    if model_dashboard:
        print(f"✓ Model health score: {model_dashboard.health_score:.1f}/100")
        print(f"✓ Model status: {model_dashboard.status}")
        print(f"✓ Predictions today: {model_dashboard.predictions_today}")
    
    # Alert summary
    alert_summary = await dashboard_service.get_alert_summary(hours=1)
    print(f"✓ Alerts in last hour: {alert_summary['total_alerts']}")
    
    # Performance trends
    trends = await dashboard_service.get_performance_trends(model_name, days=1)
    if trends['accuracy']:
        latest_trend = trends['accuracy'][-1] if trends['accuracy'] else None
        if latest_trend:
            print(f"✓ Latest accuracy trend: {latest_trend['value']:.3f}")
    
    print()
    
    # 8. Demonstrate export functionality
    print("8. Demonstrating data export...")
    
    export_data = await dashboard_service.export_metrics("json", timedelta(days=1))
    export_size = len(export_data)
    print(f"✓ Exported monitoring data: {export_size} characters")
    
    # Cleanup old data
    await monitoring_service.cleanup_old_data(days_to_keep=1)
    print("✓ Old monitoring data cleaned up\n")
    
    print("=== Monitoring System Demo Completed Successfully ===")
    
    # Summary of capabilities demonstrated
    print("\n📊 Capabilities Demonstrated:")
    print("• Real-time prediction tracking and metrics collection")
    print("• Statistical drift detection (data and performance)")
    print("• Automated alerting with multiple notification channels")
    print("• Performance monitoring with trend analysis")
    print("• Automated retraining triggers based on drift detection")
    print("• Comprehensive dashboards and health reporting")
    print("• Data export and cleanup functionality")
    print("• Integration with existing ML pipeline components")


async def demonstrate_alert_scenarios():
    """Demonstrate different alert scenarios"""
    
    print("\n=== Alert Scenarios Demo ===\n")
    
    # Initialize minimal monitoring setup
    metrics_collector = MetricsCollector()
    monitoring_service = ModelMonitoringService(metrics_collector)
    
    # Setup mock alert channel
    class MockAlertChannel:
        def __init__(self):
            self.alerts_received = []
        
        def __call__(self, alert):
            self.alerts_received.append(alert)
            print(f"🚨 ALERT: [{alert.severity.value.upper()}] {alert.title}")
            print(f"   Message: {alert.message}")
            if alert.model_name:
                print(f"   Model: {alert.model_name}")
            print()
    
    mock_channel = MockAlertChannel()
    monitoring_service.register_alert_channel(mock_channel)
    
    # Scenario 1: Performance threshold breach
    print("Scenario 1: Performance Threshold Breach")
    model_name = "test_model"
    
    # Simulate poor performance
    for i in range(20):
        await monitoring_service.track_prediction(
            model_name, "v1.0", 
            np.random.normal(0, 1, 5),
            0,  # Always predict 0
            1,  # Always actual 1 (0% accuracy)
            0.5
        )
    
    # Run monitoring to trigger alerts
    await monitoring_service.run_monitoring_cycle(model_name)
    
    # Scenario 2: Manual alert creation
    print("Scenario 2: Manual System Alert")
    manual_alert = Alert(
        id="manual_system_alert",
        severity=AlertSeverity.CRITICAL,
        title="System Resource Critical",
        message="CPU usage exceeded 95% for 10 minutes",
        timestamp=datetime.now(),
        metadata={"cpu_usage": 97.5, "duration_minutes": 12}
    )
    
    await monitoring_service._send_alert(manual_alert)
    
    print(f"Total alerts generated: {len(mock_channel.alerts_received)}")


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(demonstrate_monitoring_workflow())
    
    # Run alert scenarios demo
    asyncio.run(demonstrate_alert_scenarios())

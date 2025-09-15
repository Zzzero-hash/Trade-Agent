"""
API endpoints for monitoring and alerting system
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json

from src.services.model_monitoring_service import (
    ModelMonitoringService,
    AlertSeverity,
    DriftType
)
from src.services.monitoring_dashboard_service import MonitoringDashboardService
from src.services.automated_retraining_service import (
    AutomatedRetrainingService,
    RetrainingTrigger,
    RetrainingConfig
)
from src.utils.logging import get_logger
from src.api.auth import get_current_user

logger = get_logger("monitoring_api")

# Create router
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Pydantic models for API
class AlertRequest(BaseModel):
    """Request model for creating alerts"""
    severity: str = Field(..., description="Alert severity (low, medium, high, critical)")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    model_name: Optional[str] = Field(None, description="Associated model name")
    metric_name: Optional[str] = Field(None, description="Associated metric name")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RetrainingRequest(BaseModel):
    """Request model for manual retraining"""
    model_name: str = Field(..., description="Model name to retrain")
    config_overrides: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Training configuration overrides"
    )


class RetrainingConfigRequest(BaseModel):
    """Request model for updating retraining configuration"""
    enabled: bool = Field(True, description="Enable automated retraining")
    max_concurrent_jobs: int = Field(2, description="Maximum concurrent retraining jobs")
    cooldown_period_hours: int = Field(6, description="Cooldown period between retrainings")
    min_samples_for_retraining: int = Field(1000, description="Minimum samples required")
    performance_threshold: float = Field(0.1, description="Minimum improvement threshold")
    auto_deploy: bool = Field(False, description="Auto-deploy improved models")


# Dependency injection
async def get_monitoring_service() -> ModelMonitoringService:
    """Get monitoring service instance"""
    # In practice, this would be injected from a dependency container
    from src.utils.monitoring import get_metrics_collector
    return ModelMonitoringService(get_metrics_collector())


async def get_dashboard_service(
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service)
) -> MonitoringDashboardService:
    """Get dashboard service instance"""
    return MonitoringDashboardService(monitoring_service)


async def get_retraining_service(
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service)
) -> AutomatedRetrainingService:
    """Get retraining service instance"""
    return AutomatedRetrainingService(monitoring_service)


# Dashboard endpoints
@router.get("/dashboard/system")
async def get_system_dashboard(
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get system-wide monitoring dashboard"""
    try:
        dashboard = await dashboard_service.get_system_dashboard()
        return {
            "timestamp": dashboard.timestamp.isoformat(),
            "system_health": dashboard.system_health,
            "active_models": dashboard.active_models,
            "total_predictions": dashboard.total_predictions,
            "alerts_last_24h": dashboard.alerts_last_24h,
            "avg_model_accuracy": dashboard.avg_model_accuracy,
            "avg_prediction_confidence": dashboard.avg_prediction_confidence,
            "system_uptime": dashboard.system_uptime,
            "cpu_usage": dashboard.cpu_usage,
            "memory_usage": dashboard.memory_usage,
            "active_alerts": dashboard.active_alerts
        }
    except Exception as e:
        logger.error(f"Error getting system dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system dashboard")


@router.get("/dashboard/model/{model_name}")
async def get_model_dashboard(
    model_name: str,
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get model-specific monitoring dashboard"""
    try:
        dashboard = await dashboard_service.get_model_dashboard(model_name)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return {
            "model_name": dashboard.model_name,
            "status": dashboard.status,
            "health_score": dashboard.health_score,
            "last_prediction": dashboard.last_prediction.isoformat() if dashboard.last_prediction else None,
            "predictions_today": dashboard.predictions_today,
            "accuracy_trend": dashboard.accuracy_trend,
            "confidence_trend": dashboard.confidence_trend,
            "recent_alerts": dashboard.recent_alerts,
            "performance_metrics": dashboard.performance_metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model dashboard for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model dashboard")


@router.get("/dashboard/trends/{model_name}")
async def get_performance_trends(
    model_name: str,
    days: int = Query(7, ge=1, le=90, description="Number of days to include"),
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get performance trends for a model"""
    try:
        trends = await dashboard_service.get_performance_trends(model_name, days)
        return trends
    except Exception as e:
        logger.error(f"Error getting performance trends for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance trends")


# Health and status endpoints
@router.get("/health")
async def get_system_health(
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get comprehensive system health report"""
    try:
        health_report = await dashboard_service.get_system_health_report()
        return health_report
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")


@router.get("/models/{model_name}/status")
async def get_model_status(
    model_name: str,
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed status for a specific model"""
    try:
        status = monitoring_service.get_model_status(model_name)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Convert datetime objects to ISO strings
        if status.get('timestamp'):
            status['timestamp'] = status['timestamp'].isoformat()
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")


# Alert endpoints
@router.get("/alerts")
async def get_alerts(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get alert summary"""
    try:
        alert_summary = await dashboard_service.get_alert_summary(hours)
        
        # Apply filters if provided
        if severity or model_name:
            filtered_alerts = []
            for alert in alert_summary.get('most_recent', []):
                if severity and alert.get('severity') != severity:
                    continue
                if model_name and alert.get('model_name') != model_name:
                    continue
                filtered_alerts.append(alert)
            alert_summary['most_recent'] = filtered_alerts
        
        return alert_summary
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@router.post("/alerts")
async def create_alert(
    alert_request: AlertRequest,
    background_tasks: BackgroundTasks,
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service),
    current_user: Dict = Depends(get_current_user)
):
    """Create a manual alert"""
    try:
        # Validate severity
        try:
            severity = AlertSeverity(alert_request.severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid severity: {alert_request.severity}"
            )
        
        # Create alert
        from src.services.model_monitoring_service import Alert
        alert = Alert(
            id=f"manual_{datetime.now().timestamp()}",
            severity=severity,
            title=alert_request.title,
            message=alert_request.message,
            timestamp=datetime.now(),
            model_name=alert_request.model_name,
            metric_name=alert_request.metric_name,
            metadata=alert_request.metadata
        )
        
        # Send alert in background
        background_tasks.add_task(monitoring_service._send_alert, alert)
        
        return {"message": "Alert created successfully", "alert_id": alert.id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create alert")


# Monitoring control endpoints
@router.post("/models/{model_name}/monitor")
async def start_monitoring(
    model_name: str,
    background_tasks: BackgroundTasks,
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service),
    current_user: Dict = Depends(get_current_user)
):
    """Start monitoring cycle for a model"""
    try:
        # Run monitoring cycle in background
        background_tasks.add_task(monitoring_service.run_monitoring_cycle, model_name)
        
        return {"message": f"Monitoring started for model {model_name}"}
    
    except Exception as e:
        logger.error(f"Error starting monitoring for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")


@router.post("/models/{model_name}/baseline")
async def set_baseline_metrics(
    model_name: str,
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service),
    current_user: Dict = Depends(get_current_user)
):
    """Set baseline metrics for a model using current performance"""
    try:
        # Calculate current metrics as baseline
        current_metrics = await monitoring_service.calculate_performance_metrics(model_name)
        
        if not current_metrics:
            raise HTTPException(
                status_code=400, 
                detail=f"No performance data available for model {model_name}"
            )
        
        monitoring_service.set_baseline_metrics(model_name, current_metrics)
        
        return {
            "message": f"Baseline metrics set for model {model_name}",
            "baseline_metrics": {
                "accuracy": current_metrics.accuracy,
                "precision": current_metrics.precision,
                "recall": current_metrics.recall,
                "f1_score": current_metrics.f1_score,
                "timestamp": current_metrics.timestamp.isoformat()
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting baseline for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to set baseline metrics")


# Retraining endpoints
@router.get("/retraining/summary")
async def get_retraining_summary(
    retraining_service: AutomatedRetrainingService = Depends(get_retraining_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get retraining activity summary"""
    try:
        summary = retraining_service.get_retraining_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting retraining summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get retraining summary")


@router.post("/retraining/manual")
async def trigger_manual_retraining(
    request: RetrainingRequest,
    retraining_service: AutomatedRetrainingService = Depends(get_retraining_service),
    current_user: Dict = Depends(get_current_user)
):
    """Trigger manual retraining for a model"""
    try:
        job_id = await retraining_service.schedule_manual_retraining(
            request.model_name, 
            request.config_overrides
        )
        
        return {
            "message": f"Retraining scheduled for model {request.model_name}",
            "job_id": job_id
        }
    
    except Exception as e:
        logger.error(f"Error triggering retraining for {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")


@router.get("/retraining/jobs/{job_id}")
async def get_retraining_job_status(
    job_id: str,
    retraining_service: AutomatedRetrainingService = Depends(get_retraining_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get status of a retraining job"""
    try:
        status = retraining_service.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")


@router.delete("/retraining/jobs/{job_id}")
async def cancel_retraining_job(
    job_id: str,
    retraining_service: AutomatedRetrainingService = Depends(get_retraining_service),
    current_user: Dict = Depends(get_current_user)
):
    """Cancel a retraining job"""
    try:
        success = await retraining_service.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
        
        return {"message": f"Job {job_id} cancelled successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@router.get("/retraining/config")
async def get_retraining_config(
    retraining_service: AutomatedRetrainingService = Depends(get_retraining_service),
    current_user: Dict = Depends(get_current_user)
):
    """Get current retraining configuration"""
    try:
        config = retraining_service.config
        return {
            "enabled": config.enabled,
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "cooldown_period_hours": config.cooldown_period_hours,
            "min_samples_for_retraining": config.min_samples_for_retraining,
            "performance_threshold": config.performance_threshold,
            "auto_deploy": config.auto_deploy
        }
    except Exception as e:
        logger.error(f"Error getting retraining config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get retraining config")


@router.put("/retraining/config")
async def update_retraining_config(
    config_request: RetrainingConfigRequest,
    retraining_service: AutomatedRetrainingService = Depends(get_retraining_service),
    current_user: Dict = Depends(get_current_user)
):
    """Update retraining configuration"""
    try:
        new_config = RetrainingConfig(
            enabled=config_request.enabled,
            max_concurrent_jobs=config_request.max_concurrent_jobs,
            cooldown_period_hours=config_request.cooldown_period_hours,
            min_samples_for_retraining=config_request.min_samples_for_retraining,
            performance_threshold=config_request.performance_threshold,
            auto_deploy=config_request.auto_deploy
        )
        
        retraining_service.configure(new_config)
        
        return {"message": "Retraining configuration updated successfully"}
    
    except Exception as e:
        logger.error(f"Error updating retraining config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update retraining config")


# Export and utility endpoints
@router.get("/export")
async def export_monitoring_data(
    format_type: str = Query("json", description="Export format (json)"),
    days: int = Query(7, ge=1, le=90, description="Days of data to export"),
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Export monitoring data"""
    try:
        time_range = timedelta(days=days)
        export_data = await dashboard_service.export_metrics(format_type, time_range)
        
        if format_type.lower() == "json":
            return JSONResponse(
                content=json.loads(export_data),
                headers={"Content-Disposition": f"attachment; filename=monitoring_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting monitoring data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export monitoring data")


@router.post("/cache/clear")
async def clear_dashboard_cache(
    dashboard_service: MonitoringDashboardService = Depends(get_dashboard_service),
    current_user: Dict = Depends(get_current_user)
):
    """Clear dashboard cache to force refresh"""
    try:
        dashboard_service.clear_cache()
        return {"message": "Dashboard cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.post("/cleanup")
async def cleanup_old_data(
    days_to_keep: int = Query(30, ge=1, le=365, description="Days of data to keep"),
    monitoring_service: ModelMonitoringService = Depends(get_monitoring_service),
    current_user: Dict = Depends(get_current_user)
):
    """Clean up old monitoring data"""
    try:
        await monitoring_service.cleanup_old_data(days_to_keep)
        return {"message": f"Cleaned up data older than {days_to_keep} days"}
    except Exception as e:
        logger.error(f"Error cleaning up data: {e}")
        raise HTTPException(status_code=500, detail="Failed to clean up data")
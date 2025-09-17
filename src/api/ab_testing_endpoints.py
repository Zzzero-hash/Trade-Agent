"""A/B Testing API endpoints for model comparison and experimentation.

This module provides FastAPI endpoints for managing A/B testing experiments,
including experiment creation, monitoring, statistical analysis, and gradual rollouts.

Requirements: 6.2, 11.1
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import uuid

from src.ml.ray_serve.ab_testing import (
    RayServeABTestManager, 
    VariantConfig, 
    VariantStatus,
    ab_test_manager
)
from src.api.auth import get_current_user, require_role
from src.models.user import User, UserRole
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

router = APIRouter(prefix="/api/v1/ab-testing", tags=["A/B Testing"])


class ExperimentStatus(str, Enum):
    """Experiment status options"""
    ACTIVE = "active"
    COMPLETED = "completed"
    STOPPED = "stopped"
    PLANNED = "planned"


class CreateExperimentRequest(BaseModel):
    """Request model for creating A/B test experiments"""
    
    experiment_name: str = Field(..., description="Human-readable experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    variants: List[Dict[str, Any]] = Field(..., description="List of variant configurations")
    duration_hours: int = Field(24, ge=1, le=168, description="Experiment duration in hours")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Statistical confidence level")
    
    @validator("variants")
    def validate_variants(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 variants required for A/B testing")
        
        # Check that weights sum to 1.0
        total_weight = sum(variant.get("weight", 0) for variant in v)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError("Variant weights must sum to 1.0")
        
        # Validate required fields
        for variant in v:
            if "name" not in variant or "model_path" not in variant or "weight" not in variant:
                raise ValueError("Each variant must have 'name', 'model_path', and 'weight' fields")
        
        return v


class UpdateExperimentRequest(BaseModel):
    """Request model for updating experiment configuration"""
    
    description: Optional[str] = None
    duration_hours: Optional[int] = Field(None, ge=1, le=168)
    confidence_level: Optional[float] = Field(None, ge=0.8, le=0.99)


class GradualRolloutRequest(BaseModel):
    """Request model for gradual rollout configuration"""
    
    winning_variant: str = Field(..., description="Name of the winning variant")
    rollout_steps: Optional[List[float]] = Field(
        None, 
        description="List of traffic percentages for gradual rollout"
    )
    step_duration_hours: int = Field(2, ge=1, le=24, description="Duration of each rollout step")
    
    @validator("rollout_steps")
    def validate_rollout_steps(cls, v):
        if v is not None:
            if not all(0 < step <= 1.0 for step in v):
                raise ValueError("All rollout steps must be between 0 and 1.0")
            if v != sorted(v):
                raise ValueError("Rollout steps must be in ascending order")
        return v


class ExperimentResponse(BaseModel):
    """Response model for experiment information"""
    
    experiment_id: str
    experiment_name: str
    description: Optional[str]
    status: ExperimentStatus
    start_time: datetime
    end_time: datetime
    confidence_level: float
    variants: Dict[str, Any]
    created_by: str
    created_at: datetime


class StatisticalTestResponse(BaseModel):
    """Response model for statistical test results"""
    
    experiment_id: str
    total_tests: int
    significant_tests: int
    confidence_level: float
    tests: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SafetyCheckResponse(BaseModel):
    """Response model for safety check results"""
    
    experiment_id: str
    safety_violations: List[Dict[str, Any]]
    recommendations: List[str]
    should_stop: bool
    checked_at: datetime


class WinnerRecommendationResponse(BaseModel):
    """Response model for winner recommendation"""
    
    experiment_id: str
    winner: Optional[str]
    statistically_significant: bool
    confidence_level: float
    variant_scores: Dict[str, Any]
    recommendation: str


# Experiment Management Endpoints

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: CreateExperimentRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> ExperimentResponse:
    """Create a new A/B testing experiment."""
    
    try:
        # Generate unique experiment ID
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # Convert request variants to VariantConfig objects
        variants = []
        for variant_data in request.variants:
            variant = VariantConfig(
                name=variant_data["name"],
                model_path=variant_data["model_path"],
                weight=variant_data["weight"],
                status=VariantStatus.ACTIVE,
                metadata=variant_data.get("metadata", {})
            )
            variants.append(variant)
        
        # Create experiment
        experiment = ab_test_manager.create_experiment(
            experiment_id=experiment_id,
            variants=variants,
            duration_hours=request.duration_hours,
            confidence_level=request.confidence_level
        )
        
        # Add metadata
        experiment.experiment_name = request.experiment_name
        experiment.description = request.description
        experiment.created_by = current_user.username
        
        # Record metrics
        metrics.increment_counter("ab_experiments_created_total")
        
        logger.info(f"Created A/B test experiment {experiment_id} by user {current_user.username}")
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            experiment_name=request.experiment_name,
            description=request.description,
            status=ExperimentStatus.ACTIVE,
            start_time=experiment.start_time,
            end_time=experiment.end_time,
            confidence_level=experiment.confidence_level,
            variants={name: {
                "name": variant.name,
                "model_path": variant.model_path,
                "weight": variant.weight,
                "status": variant.status.value,
                "metadata": variant.metadata
            } for name, variant in experiment.variants.items()},
            created_by=current_user.username,
            created_at=experiment.created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[ExperimentStatus] = Query(None, description="Filter by experiment status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of experiments to return"),
    current_user: User = Depends(get_current_user)
) -> List[ExperimentResponse]:
    """List A/B testing experiments."""
    
    try:
        experiments = ab_test_manager.list_experiments()
        
        # Filter by status if specified
        if status:
            experiments = [exp for exp in experiments if exp.get("status") == status.value]
        
        # Limit results
        experiments = experiments[:limit]
        
        # Convert to response format
        response_experiments = []
        for exp_data in experiments:
            experiment = ab_test_manager.get_experiment(exp_data["experiment_id"])
            if experiment:
                response_experiments.append(ExperimentResponse(
                    experiment_id=exp_data["experiment_id"],
                    experiment_name=getattr(experiment, "experiment_name", exp_data["experiment_id"]),
                    description=getattr(experiment, "description", None),
                    status=ExperimentStatus(experiment.status),
                    start_time=experiment.start_time,
                    end_time=experiment.end_time,
                    confidence_level=experiment.confidence_level,
                    variants={name: {
                        "name": variant.name,
                        "model_path": variant.model_path,
                        "weight": variant.weight,
                        "status": variant.status.value,
                        "metadata": variant.metadata
                    } for name, variant in experiment.variants.items()},
                    created_by=getattr(experiment, "created_by", "unknown"),
                    created_at=experiment.created_at
                ))
        
        return response_experiments
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
) -> ExperimentResponse:
    """Get details of a specific experiment."""
    
    experiment = ab_test_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return ExperimentResponse(
        experiment_id=experiment_id,
        experiment_name=getattr(experiment, "experiment_name", experiment_id),
        description=getattr(experiment, "description", None),
        status=ExperimentStatus(experiment.status),
        start_time=experiment.start_time,
        end_time=experiment.end_time,
        confidence_level=experiment.confidence_level,
        variants={name: {
            "name": variant.name,
            "model_path": variant.model_path,
            "weight": variant.weight,
            "status": variant.status.value,
            "metadata": variant.metadata
        } for name, variant in experiment.variants.items()},
        created_by=getattr(experiment, "created_by", "unknown"),
        created_at=experiment.created_at
    )


@router.put("/experiments/{experiment_id}")
async def update_experiment(
    experiment_id: str,
    request: UpdateExperimentRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """Update experiment configuration."""
    
    experiment = ab_test_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != "active":
        raise HTTPException(status_code=400, detail="Can only update active experiments")
    
    # Update fields
    if request.description is not None:
        experiment.description = request.description
    
    if request.duration_hours is not None:
        experiment.end_time = experiment.start_time + timedelta(hours=request.duration_hours)
    
    if request.confidence_level is not None:
        experiment.confidence_level = request.confidence_level
    
    logger.info(f"Updated experiment {experiment_id} by user {current_user.username}")
    
    return {"message": "Experiment updated successfully", "experiment_id": experiment_id}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """Stop a running experiment."""
    
    success = ab_test_manager.stop_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    metrics.increment_counter("ab_experiments_stopped_total")
    logger.info(f"Stopped experiment {experiment_id} by user {current_user.username}")
    
    return {"message": "Experiment stopped successfully", "experiment_id": experiment_id}


# Results and Analysis Endpoints

@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get experiment results and metrics."""
    
    results = ab_test_manager.get_experiment_results(experiment_id)
    if not results:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return results


@router.get("/experiments/{experiment_id}/statistical-analysis", response_model=StatisticalTestResponse)
async def get_statistical_analysis(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
) -> StatisticalTestResponse:
    """Get statistical analysis of experiment results."""
    
    summary = ab_test_manager.get_statistical_summary(experiment_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Add additional summary information
    experiment = ab_test_manager.get_experiment(experiment_id)
    summary["summary"] = {
        "experiment_duration_hours": (experiment.end_time - experiment.start_time).total_seconds() / 3600,
        "total_requests": sum(metrics.requests for metrics in experiment.variant_metrics.values()),
        "total_errors": sum(metrics.errors for metrics in experiment.variant_metrics.values()),
        "variants_with_sufficient_data": len([
            name for name, metrics in experiment.variant_metrics.items()
            if metrics.requests >= ab_test_manager.safety_controls["min_sample_size"]
        ])
    }
    
    return StatisticalTestResponse(**summary)


@router.get("/experiments/{experiment_id}/safety-check", response_model=SafetyCheckResponse)
async def check_experiment_safety(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
) -> SafetyCheckResponse:
    """Check safety controls for an experiment."""
    
    safety_results = ab_test_manager.check_safety_controls(experiment_id)
    if "error" in safety_results:
        raise HTTPException(status_code=404, detail=safety_results["error"])
    
    return SafetyCheckResponse(
        experiment_id=experiment_id,
        safety_violations=safety_results["safety_violations"],
        recommendations=safety_results["recommendations"],
        should_stop=safety_results["should_stop"],
        checked_at=datetime.now()
    )


@router.get("/experiments/{experiment_id}/winner-recommendation", response_model=WinnerRecommendationResponse)
async def get_winner_recommendation(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
) -> WinnerRecommendationResponse:
    """Get winner recommendation based on risk-adjusted metrics."""
    
    recommendation = ab_test_manager.get_winner_recommendation(experiment_id)
    if "error" in recommendation:
        raise HTTPException(status_code=404, detail=recommendation["error"])
    
    return WinnerRecommendationResponse(**recommendation)


# Gradual Rollout Endpoints

@router.post("/experiments/{experiment_id}/gradual-rollout")
async def create_gradual_rollout(
    experiment_id: str,
    request: GradualRolloutRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """Create a gradual rollout plan for the winning variant."""
    
    rollout_plan = ab_test_manager.implement_gradual_rollout(
        experiment_id=experiment_id,
        winning_variant=request.winning_variant,
        rollout_steps=request.rollout_steps
    )
    
    if "error" in rollout_plan:
        raise HTTPException(status_code=400, detail=rollout_plan["error"])
    
    # Update step durations
    for step in rollout_plan["rollout_steps"]:
        step["duration_hours"] = request.step_duration_hours
    
    logger.info(f"Created gradual rollout plan for experiment {experiment_id} by user {current_user.username}")
    
    return rollout_plan


@router.post("/experiments/{experiment_id}/execute-rollout-step")
async def execute_rollout_step(
    experiment_id: str,
    rollout_plan: Dict[str, Any],
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """Execute the next step in a gradual rollout."""
    
    result = ab_test_manager.execute_rollout_step(rollout_plan)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    logger.info(f"Executed rollout step for experiment {experiment_id} by user {current_user.username}")
    
    return result


# Monitoring and Metrics Endpoints

@router.get("/experiments/{experiment_id}/metrics/real-time")
async def get_real_time_metrics(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get real-time metrics for an experiment."""
    
    experiment = ab_test_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Calculate real-time metrics
    real_time_metrics = {
        "experiment_id": experiment_id,
        "status": experiment.status,
        "elapsed_hours": (datetime.now() - experiment.start_time).total_seconds() / 3600,
        "remaining_hours": max(0, (experiment.end_time - datetime.now()).total_seconds() / 3600),
        "variants": {}
    }
    
    for variant_name, metrics in experiment.variant_metrics.items():
        real_time_metrics["variants"][variant_name] = {
            "requests": metrics.requests,
            "errors": metrics.errors,
            "error_rate": metrics.error_rate,
            "avg_latency_ms": metrics.avg_latency_ms,
            "avg_confidence": metrics.avg_confidence,
            "traffic_weight": experiment.variants[variant_name].weight,
            "last_updated": datetime.now().isoformat()
        }
    
    return real_time_metrics


@router.get("/experiments/dashboard")
async def get_experiments_dashboard(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get dashboard overview of all experiments."""
    
    experiments = ab_test_manager.list_experiments()
    
    dashboard = {
        "total_experiments": len(experiments),
        "active_experiments": len([exp for exp in experiments if exp.get("status") == "active"]),
        "completed_experiments": len([exp for exp in experiments if exp.get("status") == "completed"]),
        "stopped_experiments": len([exp for exp in experiments if exp.get("status") == "stopped"]),
        "experiments_with_winners": 0,
        "total_requests_processed": 0,
        "recent_experiments": []
    }
    
    # Calculate additional metrics
    for exp_data in experiments:
        experiment = ab_test_manager.get_experiment(exp_data["experiment_id"])
        if experiment:
            # Count total requests
            total_requests = sum(metrics.requests for metrics in experiment.variant_metrics.values())
            dashboard["total_requests_processed"] += total_requests
            
            # Check if experiment has a clear winner
            if experiment.status == "completed":
                winner_rec = ab_test_manager.get_winner_recommendation(exp_data["experiment_id"])
                if winner_rec.get("statistically_significant"):
                    dashboard["experiments_with_winners"] += 1
            
            # Add to recent experiments (last 10)
            if len(dashboard["recent_experiments"]) < 10:
                dashboard["recent_experiments"].append({
                    "experiment_id": exp_data["experiment_id"],
                    "experiment_name": getattr(experiment, "experiment_name", exp_data["experiment_id"]),
                    "status": experiment.status,
                    "start_time": experiment.start_time.isoformat(),
                    "total_requests": total_requests
                })
    
    # Sort recent experiments by start time
    dashboard["recent_experiments"].sort(key=lambda x: x["start_time"], reverse=True)
    
    return dashboard


# Visualization and Reporting Endpoints

@router.get("/experiments/{experiment_id}/visualizations")
async def get_experiment_visualizations(
    experiment_id: str,
    chart_type: Optional[str] = Query(None, description="Specific chart type to generate"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get visualizations for an experiment."""
    
    experiment = ab_test_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    from src.utils.ab_testing_visualization import visualizer
    
    visualizations = {}
    
    try:
        if not chart_type or chart_type == "performance_comparison":
            viz = visualizer.create_performance_comparison_chart(experiment)
            if viz:
                visualizations["performance_comparison"] = viz
        
        if not chart_type or chart_type == "metrics_heatmap":
            viz = visualizer.create_variant_metrics_heatmap(experiment)
            if viz:
                visualizations["metrics_heatmap"] = viz
        
        if not chart_type or chart_type == "statistical_significance":
            statistical_tests = experiment.perform_statistical_tests()
            viz = visualizer.create_statistical_significance_chart(statistical_tests)
            if viz:
                visualizations["statistical_significance"] = viz
        
        return {
            "experiment_id": experiment_id,
            "visualizations": visualizations,
            "available_charts": ["performance_comparison", "metrics_heatmap", "statistical_significance"]
        }
        
    except Exception as e:
        logger.error(f"Error generating visualizations for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")


@router.get("/experiments/{experiment_id}/report")
async def generate_experiment_report(
    experiment_id: str,
    format: str = Query("json", regex="^(json|html)$", description="Report format"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate a comprehensive experiment report."""
    
    experiment = ab_test_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    try:
        from src.utils.ab_testing_visualization import visualizer, create_html_report
        
        # Get all necessary data
        statistical_tests = experiment.perform_statistical_tests()
        winner_recommendation = ab_test_manager.get_winner_recommendation(experiment_id)
        
        # Generate report
        report_data = visualizer.generate_experiment_report(
            experiment=experiment,
            statistical_tests=statistical_tests,
            winner_recommendation=winner_recommendation
        )
        
        if format == "html":
            html_content = create_html_report(report_data)
            return {
                "experiment_id": experiment_id,
                "format": "html",
                "content": html_content,
                "generated_at": report_data["generated_at"]
            }
        else:
            return {
                "experiment_id": experiment_id,
                "format": "json",
                "report": report_data
            }
            
    except Exception as e:
        logger.error(f"Error generating report for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.get("/experiments/timeline-visualization")
async def get_experiments_timeline(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get timeline visualization of all experiments."""
    
    try:
        from src.utils.ab_testing_visualization import visualizer
        
        experiments = ab_test_manager.list_experiments()
        
        # Convert to format expected by visualizer
        experiment_data = []
        for exp_dict in experiments:
            experiment = ab_test_manager.get_experiment(exp_dict["experiment_id"])
            if experiment:
                experiment_data.append({
                    "experiment_id": exp_dict["experiment_id"],
                    "start_time": experiment.start_time.isoformat(),
                    "end_time": experiment.end_time.isoformat(),
                    "status": experiment.status
                })
        
        timeline_chart = visualizer.create_experiment_timeline_chart(experiment_data)
        
        return {
            "timeline_visualization": timeline_chart,
            "total_experiments": len(experiment_data),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating timeline visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating timeline: {str(e)}")


# Background Tasks

@router.post("/experiments/cleanup")
async def cleanup_completed_experiments(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """Clean up completed experiments (background task)."""
    
    async def cleanup_task():
        count = await ab_test_manager.cleanup_completed_experiments()
        logger.info(f"Cleaned up {count} completed experiments")
        metrics.increment_counter("ab_experiments_cleaned_up_total", {"count": count})
    
    background_tasks.add_task(cleanup_task)
    
    return {"message": "Cleanup task scheduled"}


@router.post("/experiments/{experiment_id}/auto-safety-monitor")
async def enable_auto_safety_monitoring(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    check_interval_minutes: int = Query(15, ge=5, le=60, description="Safety check interval in minutes"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """Enable automatic safety monitoring for an experiment."""
    
    experiment = ab_test_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    async def safety_monitor_task():
        """Background task to monitor experiment safety"""
        while experiment.status == "active":
            try:
                safety_results = ab_test_manager.check_safety_controls(experiment_id)
                
                if safety_results["should_stop"]:
                    # Auto-stop experiment due to safety violations
                    ab_test_manager.stop_experiment(experiment_id)
                    logger.warning(f"Auto-stopped experiment {experiment_id} due to safety violations: {safety_results['safety_violations']}")
                    metrics.increment_counter("ab_experiments_auto_stopped_total")
                    break
                
                # Wait for next check
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in safety monitoring for experiment {experiment_id}: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    background_tasks.add_task(safety_monitor_task)
    
    logger.info(f"Enabled auto safety monitoring for experiment {experiment_id} with {check_interval_minutes}min intervals")
    
    return {
        "message": "Auto safety monitoring enabled",
        "experiment_id": experiment_id,
        "check_interval_minutes": check_interval_minutes
    }
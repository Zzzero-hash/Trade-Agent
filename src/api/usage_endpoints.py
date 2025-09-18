"""
API endpoints for usage tracking and billing management.

This module provides REST API endpoints for freemium usage tracking,
subscription management, and billing operations.

Requirements: 7.1, 7.2, 7.5
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.auth import get_current_user, User, UserRole, require_role
from src.services.usage_tracking_service import get_usage_tracking_service, UsageTrackingService
from src.models.usage_tracking import UsageType, SubscriptionTier
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

# Create router
router = APIRouter(prefix="/api/v1/usage", tags=["usage-tracking"])


class UsageTrackingRequest(BaseModel):
    """Request model for manual usage tracking."""
    usage_type: UsageType
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubscriptionUpgradeRequest(BaseModel):
    """Request model for subscription upgrades."""
    target_tier: SubscriptionTier
    billing_period: Optional[str] = "monthly"


@router.post("/track")
async def track_usage_endpoint(
    request: UsageTrackingRequest,
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> JSONResponse:
    """
    Manually track a usage event.
    
    This endpoint is primarily for internal use or testing.
    Most usage tracking happens automatically via middleware.
    """
    try:
        # Check if user can make this request
        can_proceed, error_message = await usage_service.check_usage_limits(
            current_user.id, request.usage_type
        )
        
        if not can_proceed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error_message
            )
        
        # Track the usage
        usage_record = await usage_service.track_usage(
            user_id=current_user.id,
            usage_type=request.usage_type,
            endpoint=request.endpoint,
            metadata=request.metadata
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Usage tracked successfully",
                "usage_id": usage_record.id,
                "cost_cents": usage_record.cost_cents,
                "timestamp": usage_record.timestamp.isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track usage for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track usage"
        )


@router.get("/summary")
async def get_usage_summary(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get usage summary for the current user.
    
    Returns comprehensive usage statistics including current limits,
    subscription details, and usage history.
    """
    try:
        # Parse dates if provided
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        summary = await usage_service.get_usage_summary(
            current_user.id, start_dt, end_dt
        )
        
        return summary
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get usage summary for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage summary"
        )


@router.get("/limits")
async def get_current_limits(
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get current usage limits for the user.
    
    Returns daily and monthly limits along with current usage counts.
    """
    try:
        limits = await usage_service.usage_repository.get_usage_limits(current_user.id)
        subscription = await usage_service.usage_repository.get_user_subscription(current_user.id)
        
        if not limits or not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription or limits found"
            )
        
        return {
            "user_id": current_user.id,
            "subscription_tier": subscription.subscription_tier,
            "is_trial": subscription.is_trial_active,
            "trial_expires_at": subscription.trial_end_date.isoformat() if subscription.trial_end_date else None,
            "limits": {
                "daily_ai_signals": {
                    "limit": limits.daily_ai_signals_limit,
                    "used": limits.daily_ai_signals_used,
                    "remaining": limits.get_daily_remaining(),
                    "reset_time": limits.daily_reset_time.isoformat()
                },
                "monthly_api_requests": {
                    "limit": limits.monthly_api_requests_limit,
                    "used": limits.monthly_api_requests_used,
                    "remaining": limits.get_monthly_remaining(),
                    "reset_time": limits.monthly_reset_time.isoformat()
                }
            },
            "status": {
                "daily_limit_exceeded": limits.is_daily_limit_exceeded(),
                "monthly_limit_exceeded": limits.is_monthly_limit_exceeded()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get limits for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage limits"
        )


@router.get("/upgrade-recommendations")
async def get_upgrade_recommendations(
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get personalized upgrade recommendations based on usage patterns.
    
    Requirement 7.2: Clear upgrade options when free tier limits are reached
    """
    try:
        recommendations = await usage_service.get_upgrade_recommendations(current_user.id)
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get upgrade recommendations for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve upgrade recommendations"
        )


@router.post("/upgrade")
async def upgrade_subscription(
    request: SubscriptionUpgradeRequest,
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> JSONResponse:
    """
    Upgrade user's subscription to a higher tier.
    
    In production, this would integrate with payment processing.
    """
    try:
        # Validate target tier
        if request.target_tier in [SubscriptionTier.TRIAL]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot upgrade to trial tier"
            )
        
        # Perform upgrade
        updated_subscription = await usage_service.upgrade_subscription(
            current_user.id, request.target_tier
        )
        
        # Track metrics
        metrics.increment_counter("subscription_upgrades", 1)
        metrics.increment_counter(f"upgrades_to_{request.target_tier}", 1)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully upgraded to {request.target_tier}",
                "subscription": {
                    "tier": updated_subscription.subscription_tier,
                    "next_billing_date": updated_subscription.next_billing_date.isoformat() if updated_subscription.next_billing_date else None,
                    "auto_renew": updated_subscription.auto_renew
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upgrade subscription for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upgrade subscription"
        )


@router.get("/subscription")
async def get_subscription_details(
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get detailed subscription information for the current user.
    """
    try:
        subscription = await usage_service.usage_repository.get_user_subscription(current_user.id)
        
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found"
            )
        
        # Get plan details
        plan = await usage_service.usage_repository.get_subscription_plan(subscription.subscription_tier)
        
        return {
            "subscription": {
                "user_id": subscription.user_id,
                "tier": subscription.subscription_tier,
                "plan_id": subscription.plan_id,
                "is_active": subscription.is_active,
                "auto_renew": subscription.auto_renew,
                "subscription_start_date": subscription.subscription_start_date.isoformat() if subscription.subscription_start_date else None,
                "next_billing_date": subscription.next_billing_date.isoformat() if subscription.next_billing_date else None
            },
            "trial": {
                "is_trial_active": subscription.is_trial_active,
                "trial_start_date": subscription.trial_start_date.isoformat() if subscription.trial_start_date else None,
                "trial_end_date": subscription.trial_end_date.isoformat() if subscription.trial_end_date else None,
                "days_remaining": (subscription.trial_end_date - datetime.now(timezone.utc)).days if subscription.trial_end_date and subscription.is_trial_active else None
            },
            "plan": {
                "name": plan.name if plan else None,
                "description": plan.description if plan else None,
                "price_cents": plan.price_cents if plan else None,
                "billing_period": plan.billing_period if plan else None,
                "features": plan.features if plan else []
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription details for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription details"
        )


@router.get("/billing-history")
async def get_billing_history(
    current_user: User = Depends(get_current_user),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get billing history for the current user.
    """
    try:
        billing_records = await usage_service.usage_repository.get_user_billing_records(current_user.id)
        
        formatted_records = []
        for record in billing_records:
            formatted_records.append({
                "id": record.id,
                "amount_cents": record.amount_cents,
                "subscription_tier": record.subscription_tier,
                "billing_period_start": record.billing_period_start.isoformat(),
                "billing_period_end": record.billing_period_end.isoformat(),
                "status": record.status,
                "created_at": record.created_at.isoformat(),
                "paid_at": record.paid_at.isoformat() if record.paid_at else None
            })
        
        return {
            "user_id": current_user.id,
            "billing_records": formatted_records,
            "total_records": len(formatted_records),
            "total_amount_cents": sum(record.amount_cents for record in billing_records if record.status == "completed")
        }
        
    except Exception as e:
        logger.error(f"Failed to get billing history for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve billing history"
        )


@router.get("/plans")
async def get_subscription_plans(
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get all available subscription plans.
    
    Public endpoint that doesn't require authentication.
    """
    try:
        plans = await usage_service.usage_repository.get_all_subscription_plans()
        
        formatted_plans = []
        for plan in plans:
            formatted_plans.append({
                "tier": plan.tier,
                "name": plan.name,
                "description": plan.description,
                "price_cents": plan.price_cents,
                "billing_period": plan.billing_period,
                "limits": {
                    "daily_ai_signals": plan.daily_ai_signals,
                    "monthly_api_requests": plan.monthly_api_requests
                },
                "features": plan.features
            })
        
        return {
            "plans": formatted_plans,
            "currency": "USD",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get subscription plans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription plans"
        )


# Admin endpoints for analytics and monitoring

@router.get("/analytics/unit-economics")
async def get_unit_economics(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get unit economics analysis for cost optimization.
    
    Requirement 7.5: Monitor cost-per-signal to maintain positive unit economics
    
    Admin only endpoint.
    """
    try:
        unit_economics = await usage_service.calculate_cost_per_signal()
        return unit_economics
        
    except Exception as e:
        logger.error(f"Failed to get unit economics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve unit economics"
        )


@router.get("/analytics/usage-stats")
async def get_usage_analytics(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> Dict[str, Any]:
    """
    Get comprehensive usage analytics.
    
    Admin only endpoint for monitoring platform usage.
    """
    try:
        # Default to last 30 days if no dates provided
        if not end_date:
            end_dt = datetime.now(timezone.utc)
        else:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if not start_date:
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        
        analytics = await usage_service.usage_repository.get_usage_analytics(start_dt, end_dt)
        
        return {
            "period": {
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat()
            },
            "analytics": analytics
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage analytics"
        )


@router.post("/admin/create-trial")
async def create_trial_subscription_admin(
    user_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> JSONResponse:
    """
    Create a trial subscription for a user (admin only).
    
    Useful for customer support or testing purposes.
    """
    try:
        subscription = await usage_service.create_trial_subscription(user_id)
        
        return JSONResponse(
            status_code=201,
            content={
                "message": f"Trial subscription created for user {user_id}",
                "subscription": {
                    "tier": subscription.subscription_tier,
                    "trial_end_date": subscription.trial_end_date.isoformat() if subscription.trial_end_date else None
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create trial subscription for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create trial subscription"
        )
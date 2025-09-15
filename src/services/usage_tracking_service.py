"""
Usage tracking service for freemium functionality.

This service handles usage tracking, billing calculations,
and subscription management for the freemium model.

Requirements: 7.1, 7.2, 7.5
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
import uuid
import asyncio
from decimal import Decimal

from src.models.usage_tracking import (
    UsageRecord, UsageSummary, UserSubscription, UsageLimits,
    BillingRecord, ConversionMetrics, SubscriptionPlan,
    UsageType, SubscriptionTier, BillingPeriod
)
from src.repositories.usage_repository import UsageRepository
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class UsageTrackingService:
    """
    Service for tracking usage, managing subscriptions, and handling billing.
    """
    
    def __init__(self, usage_repository: Optional[UsageRepository] = None):
        self.usage_repository = usage_repository or UsageRepository()
        
        # Cost configuration (in cents)
        self.cost_config = {
            UsageType.AI_SIGNAL_REQUEST: 10,  # 10 cents per AI signal
            UsageType.MODEL_PREDICTION: 5,   # 5 cents per prediction
            UsageType.BATCH_PREDICTION: 20,  # 20 cents per batch
            UsageType.API_REQUEST: 1,        # 1 cent per API request
            UsageType.DATA_REQUEST: 2        # 2 cents per data request
        }
    
    async def track_usage(
        self,
        user_id: str,
        usage_type: UsageType,
        endpoint: Optional[str] = None,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """
        Track a usage event for a user.
        
        Args:
            user_id: User who made the request
            usage_type: Type of usage being tracked
            endpoint: API endpoint called
            request_size: Size of request in bytes
            response_size: Size of response in bytes
            processing_time_ms: Processing time in milliseconds
            metadata: Additional metadata
        
        Returns:
            Created usage record
        """
        try:
            # Calculate cost
            cost_cents = self.cost_config.get(usage_type, 0)
            
            # Create usage record
            usage_record = UsageRecord(
                id=f"usage_{uuid.uuid4().hex}",
                user_id=user_id,
                usage_type=usage_type,
                timestamp=datetime.now(timezone.utc),
                endpoint=endpoint,
                request_size=request_size,
                response_size=response_size,
                processing_time_ms=processing_time_ms,
                cost_cents=cost_cents,
                metadata=metadata or {}
            )
            
            # Save to repository
            await self.usage_repository.create_usage_record(usage_record)
            
            # Update metrics
            metrics.increment_counter(f"usage_{usage_type}", 1)
            if cost_cents > 0:
                metrics.increment_counter("total_cost_cents", cost_cents)
            
            logger.debug(f"Tracked usage for user {user_id}: {usage_type}")
            return usage_record
            
        except Exception as e:
            logger.error(f"Failed to track usage for user {user_id}: {e}")
            raise
    
    async def check_usage_limits(self, user_id: str, usage_type: UsageType) -> Tuple[bool, Optional[str]]:
        """
        Check if user can make a request based on their usage limits.
        
        Args:
            user_id: User ID to check
            usage_type: Type of usage being requested
        
        Returns:
            Tuple of (can_proceed, error_message)
        """
        try:
            # Get user's current limits
            limits = await self.usage_repository.get_usage_limits(user_id)
            if not limits:
                # No limits found, check subscription
                subscription = await self.usage_repository.get_user_subscription(user_id)
                if not subscription:
                    return False, "No active subscription found"
                
                # Initialize limits
                await self.usage_repository._initialize_usage_limits(subscription)
                limits = await self.usage_repository.get_usage_limits(user_id)
            
            # Check trial expiration
            subscription = await self.usage_repository.get_user_subscription(user_id)
            if subscription and subscription.is_trial_active:
                if subscription.trial_end_date and datetime.now(timezone.utc) > subscription.trial_end_date:
                    return False, "Trial period has expired. Please upgrade to continue."
            
            # Check specific limits based on usage type
            if usage_type == UsageType.AI_SIGNAL_REQUEST:
                if limits.is_daily_limit_exceeded():
                    remaining = limits.get_daily_remaining() or 0
                    return False, f"Daily AI signal limit exceeded. {remaining} signals remaining today. Upgrade for more signals."
            
            # Check monthly API limits for all request types
            if limits.is_monthly_limit_exceeded():
                remaining = limits.get_monthly_remaining() or 0
                return False, f"Monthly API request limit exceeded. {remaining} requests remaining this month. Upgrade for more requests."
            
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to check usage limits for user {user_id}: {e}")
            return False, "Unable to verify usage limits"
    
    async def create_trial_subscription(self, user_id: str) -> UserSubscription:
        """
        Create a 7-day trial subscription for a new user.
        
        Requirement 7.1: New users get 5 free AI signal requests per day for 7 consecutive days
        """
        try:
            now = datetime.now(timezone.utc)
            trial_end = now + timedelta(days=7)
            
            subscription = UserSubscription(
                user_id=user_id,
                subscription_tier=SubscriptionTier.TRIAL,
                plan_id="trial_7_day",
                trial_start_date=now,
                trial_end_date=trial_end,
                is_trial_active=True,
                subscription_start_date=now,
                is_active=True,
                auto_renew=False,
                current_period_start=now,
                daily_usage_reset_time=now.replace(hour=0, minute=0, second=0, microsecond=0)
            )
            
            await self.usage_repository.create_user_subscription(subscription)
            
            # Track conversion metrics
            await self._update_conversion_metrics("trial_started")
            
            logger.info(f"Created trial subscription for user {user_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to create trial subscription for user {user_id}: {e}")
            raise
    
    async def upgrade_subscription(
        self, 
        user_id: str, 
        target_tier: SubscriptionTier
    ) -> UserSubscription:
        """
        Upgrade user's subscription to a higher tier.
        
        Requirement 7.2: Clear upgrade options when free tier limits are reached
        """
        try:
            # Get current subscription
            current_subscription = await self.usage_repository.get_user_subscription(user_id)
            if not current_subscription:
                raise ValueError("No current subscription found")
            
            # Get target plan
            target_plan = await self.usage_repository.get_subscription_plan(target_tier)
            if not target_plan:
                raise ValueError(f"No plan found for tier: {target_tier}")
            
            now = datetime.now(timezone.utc)
            
            # Update subscription
            current_subscription.subscription_tier = target_tier
            current_subscription.plan_id = f"{target_tier}_plan"
            
            # Handle trial to paid conversion
            if current_subscription.is_trial_active:
                current_subscription.is_trial_active = False
                current_subscription.trial_end_date = now
                await self._update_conversion_metrics("trial_to_paid")
            elif current_subscription.subscription_tier == SubscriptionTier.FREE:
                await self._update_conversion_metrics("free_to_paid")
            
            # Set billing dates
            if target_plan.billing_period == BillingPeriod.MONTHLY:
                current_subscription.next_billing_date = now + timedelta(days=30)
            elif target_plan.billing_period == BillingPeriod.YEARLY:
                current_subscription.next_billing_date = now + timedelta(days=365)
            
            current_subscription.auto_renew = True
            
            # Save updated subscription
            await self.usage_repository.create_user_subscription(current_subscription)
            
            # Create billing record
            billing_record = BillingRecord(
                id=f"bill_{uuid.uuid4().hex}",
                user_id=user_id,
                subscription_tier=target_tier,
                amount_cents=target_plan.price_cents,
                billing_period_start=now,
                billing_period_end=current_subscription.next_billing_date or now + timedelta(days=30),
                status="completed",
                paid_at=now
            )
            
            await self.usage_repository.create_billing_record(billing_record)
            
            logger.info(f"Upgraded user {user_id} to {target_tier}")
            return current_subscription
            
        except Exception as e:
            logger.error(f"Failed to upgrade subscription for user {user_id}: {e}")
            raise
    
    async def get_usage_summary(
        self, 
        user_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive usage summary for a user.
        """
        try:
            if not start_date:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if not end_date:
                end_date = datetime.now(timezone.utc)
            
            # Get usage records
            usage_records = await self.usage_repository.get_user_usage_records(
                user_id, start_date, end_date
            )
            
            # Get current limits
            limits = await self.usage_repository.get_usage_limits(user_id)
            
            # Get subscription
            subscription = await self.usage_repository.get_user_subscription(user_id)
            
            # Calculate summary statistics
            total_requests = len(usage_records)
            total_cost_cents = sum(record.cost_cents or 0 for record in usage_records)
            
            usage_by_type = {}
            for record in usage_records:
                if record.usage_type not in usage_by_type:
                    usage_by_type[record.usage_type] = 0
                usage_by_type[record.usage_type] += 1
            
            # Get today's usage
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            today_summary = await self.usage_repository.get_daily_usage_summary(user_id, today)
            
            return {
                "user_id": user_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "subscription": {
                    "tier": subscription.subscription_tier if subscription else None,
                    "is_trial": subscription.is_trial_active if subscription else False,
                    "trial_expires_at": subscription.trial_end_date.isoformat() if subscription and subscription.trial_end_date else None
                },
                "usage_summary": {
                    "total_requests": total_requests,
                    "total_cost_cents": total_cost_cents,
                    "usage_by_type": usage_by_type
                },
                "current_limits": {
                    "daily_ai_signals_limit": limits.daily_ai_signals_limit if limits else None,
                    "daily_ai_signals_used": limits.daily_ai_signals_used if limits else 0,
                    "daily_remaining": limits.get_daily_remaining() if limits else None,
                    "monthly_api_requests_limit": limits.monthly_api_requests_limit if limits else None,
                    "monthly_api_requests_used": limits.monthly_api_requests_used if limits else 0,
                    "monthly_remaining": limits.get_monthly_remaining() if limits else None
                },
                "today_usage": {
                    "total_requests": today_summary.total_requests if today_summary else 0,
                    "ai_signals": today_summary.usage_counts.get(UsageType.AI_SIGNAL_REQUEST, 0) if today_summary else 0,
                    "cost_cents": today_summary.total_cost_cents if today_summary else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage summary for user {user_id}: {e}")
            raise
    
    async def get_upgrade_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Get upgrade recommendations based on user's usage patterns.
        
        Requirement 7.2: Clear upgrade options when free tier limits are reached
        """
        try:
            # Get current subscription and limits
            subscription = await self.usage_repository.get_user_subscription(user_id)
            limits = await self.usage_repository.get_usage_limits(user_id)
            
            if not subscription or not limits:
                return {"recommendations": [], "message": "Unable to generate recommendations"}
            
            recommendations = []
            
            # Check if user is hitting limits
            if limits.is_daily_limit_exceeded():
                recommendations.append({
                    "reason": "daily_limit_exceeded",
                    "message": "You've reached your daily AI signal limit",
                    "suggested_tiers": [SubscriptionTier.BASIC, SubscriptionTier.PREMIUM],
                    "benefits": ["More daily signals", "Advanced analytics"]
                })
            
            if limits.is_monthly_limit_exceeded():
                recommendations.append({
                    "reason": "monthly_limit_exceeded",
                    "message": "You've reached your monthly API request limit",
                    "suggested_tiers": [SubscriptionTier.PREMIUM],
                    "benefits": ["Unlimited API requests", "Priority support"]
                })
            
            # Check trial expiration
            if subscription.is_trial_active and subscription.trial_end_date:
                days_remaining = (subscription.trial_end_date - datetime.now(timezone.utc)).days
                if days_remaining <= 2:
                    recommendations.append({
                        "reason": "trial_expiring",
                        "message": f"Your trial expires in {days_remaining} days",
                        "suggested_tiers": [SubscriptionTier.FREE, SubscriptionTier.BASIC, SubscriptionTier.PREMIUM],
                        "benefits": ["Continue using AI signals", "Keep your trading history"]
                    })
            
            # Get available plans
            plans = await self.usage_repository.get_all_subscription_plans()
            available_plans = [
                {
                    "tier": plan.tier,
                    "name": plan.name,
                    "description": plan.description,
                    "price_cents": plan.price_cents,
                    "billing_period": plan.billing_period,
                    "daily_ai_signals": plan.daily_ai_signals,
                    "monthly_api_requests": plan.monthly_api_requests,
                    "features": plan.features
                }
                for plan in plans
                if plan.tier != subscription.subscription_tier
            ]
            
            return {
                "current_tier": subscription.subscription_tier,
                "recommendations": recommendations,
                "available_plans": available_plans,
                "usage_summary": await self.get_usage_summary(user_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get upgrade recommendations for user {user_id}: {e}")
            raise
    
    async def calculate_cost_per_signal(self) -> Dict[str, Any]:
        """
        Calculate cost per signal to maintain positive unit economics.
        
        Requirement 7.5: Monitor cost-per-signal to maintain positive unit economics
        """
        try:
            # Get usage analytics for the last 30 days
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            analytics = await self.usage_repository.get_usage_analytics(start_date, end_date)
            
            # Calculate metrics
            total_signals = analytics["usage_by_type"].get(UsageType.AI_SIGNAL_REQUEST, 0)
            total_cost_cents = analytics["total_cost_cents"]
            total_revenue_cents = 0
            
            # Estimate revenue from paid users
            for tier, count in analytics["usage_by_tier"].items():
                if tier in [SubscriptionTier.BASIC, SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE]:
                    plan = await self.usage_repository.get_subscription_plan(tier)
                    if plan:
                        # Approximate monthly revenue
                        monthly_revenue = (plan.price_cents * count) if plan.billing_period == BillingPeriod.MONTHLY else 0
                        total_revenue_cents += monthly_revenue
            
            # Calculate unit economics
            cost_per_signal = total_cost_cents / total_signals if total_signals > 0 else 0
            revenue_per_signal = total_revenue_cents / total_signals if total_signals > 0 else 0
            profit_per_signal = revenue_per_signal - cost_per_signal
            
            # Calculate conversion rates
            total_users = analytics["total_users"]
            paid_users = sum(
                analytics["usage_by_tier"].get(tier, 0)
                for tier in [SubscriptionTier.BASIC, SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE]
            )
            conversion_rate = (paid_users / total_users) if total_users > 0 else 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "metrics": {
                    "total_signals": total_signals,
                    "total_users": total_users,
                    "paid_users": paid_users,
                    "conversion_rate": round(conversion_rate * 100, 2),
                    "cost_per_signal_cents": round(cost_per_signal, 2),
                    "revenue_per_signal_cents": round(revenue_per_signal, 2),
                    "profit_per_signal_cents": round(profit_per_signal, 2),
                    "total_cost_cents": total_cost_cents,
                    "total_revenue_cents": total_revenue_cents,
                    "profit_margin": round((profit_per_signal / revenue_per_signal * 100), 2) if revenue_per_signal > 0 else 0
                },
                "recommendations": self._generate_unit_economics_recommendations(
                    cost_per_signal, revenue_per_signal, conversion_rate
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate cost per signal: {e}")
            raise
    
    def _generate_unit_economics_recommendations(
        self, 
        cost_per_signal: float, 
        revenue_per_signal: float, 
        conversion_rate: float
    ) -> List[str]:
        """Generate recommendations based on unit economics."""
        recommendations = []
        
        if revenue_per_signal <= cost_per_signal:
            recommendations.append("Revenue per signal is below cost. Consider increasing prices or reducing costs.")
        
        if conversion_rate < 0.05:  # Less than 5%
            recommendations.append("Low conversion rate. Consider improving free tier value or onboarding experience.")
        
        if cost_per_signal > 15:  # More than 15 cents
            recommendations.append("High cost per signal. Consider optimizing model inference or infrastructure costs.")
        
        if not recommendations:
            recommendations.append("Unit economics look healthy. Continue monitoring and optimizing.")
        
        return recommendations
    
    async def _update_conversion_metrics(self, event_type: str):
        """Update conversion metrics for tracking."""
        try:
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            metrics = await self.usage_repository.get_conversion_metrics(today)
            
            if not metrics:
                metrics = ConversionMetrics(date=today)
            
            if event_type == "trial_started":
                metrics.total_trial_users += 1
            elif event_type == "trial_to_paid":
                metrics.trial_to_paid_conversions += 1
            elif event_type == "free_to_paid":
                metrics.free_to_paid_conversions += 1
            elif event_type == "trial_expired":
                metrics.trial_expirations += 1
            
            # Recalculate rates
            if metrics.total_trial_users > 0:
                metrics.trial_conversion_rate = metrics.trial_to_paid_conversions / metrics.total_trial_users
            
            await self.usage_repository.update_conversion_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to update conversion metrics: {e}")


# Dependency injection
_usage_tracking_service: Optional[UsageTrackingService] = None


def get_usage_tracking_service() -> UsageTrackingService:
    """Get usage tracking service instance."""
    global _usage_tracking_service
    if _usage_tracking_service is None:
        _usage_tracking_service = UsageTrackingService()
    return _usage_tracking_service
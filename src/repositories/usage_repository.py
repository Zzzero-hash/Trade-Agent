"""
Repository for usage tracking data access.

This module provides data access methods for usage tracking,
billing, and subscription management.

Requirements: 7.1, 7.2, 7.5
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import asyncio
from collections import defaultdict
import json

from src.models.usage_tracking import (
    UsageRecord, UsageSummary, UserSubscription, UsageLimits,
    BillingRecord, ConversionMetrics, SubscriptionPlan,
    UsageType, SubscriptionTier, BillingPeriod
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class UsageRepository:
    """
    Repository for usage tracking data operations.
    
    In production, this would use a proper database like PostgreSQL or MongoDB.
    For now, using in-memory storage with persistence simulation.
    """
    
    def __init__(self):
        # In-memory storage (replace with actual database in production)
        self._usage_records: Dict[str, UsageRecord] = {}
        self._usage_summaries: Dict[str, UsageSummary] = {}  # Key: f"{user_id}_{date}"
        self._user_subscriptions: Dict[str, UserSubscription] = {}
        self._usage_limits: Dict[str, UsageLimits] = {}
        self._billing_records: Dict[str, BillingRecord] = {}
        self._conversion_metrics: Dict[str, ConversionMetrics] = {}  # Key: date
        self._subscription_plans: Dict[str, SubscriptionPlan] = {}
        
        # Initialize default subscription plans
        self._initialize_default_plans()
    
    def _initialize_default_plans(self):
        """Initialize default subscription plans."""
        plans = [
            SubscriptionPlan(
                tier=SubscriptionTier.TRIAL,
                name="7-Day Trial",
                description="Try our AI trading signals for free",
                price_cents=0,
                billing_period=BillingPeriod.DAILY,
                daily_ai_signals=5,
                monthly_api_requests=1000,
                features=["5_daily_signals", "basic_analytics"]
            ),
            SubscriptionPlan(
                tier=SubscriptionTier.FREE,
                name="Free Plan",
                description="Limited access to AI trading signals",
                price_cents=0,
                billing_period=BillingPeriod.MONTHLY,
                daily_ai_signals=5,
                monthly_api_requests=1000,
                features=["5_daily_signals", "basic_analytics"]
            ),
            SubscriptionPlan(
                tier=SubscriptionTier.BASIC,
                name="Basic Plan",
                description="More AI signals and features",
                price_cents=1999,  # $19.99
                billing_period=BillingPeriod.MONTHLY,
                daily_ai_signals=50,
                monthly_api_requests=10000,
                features=["50_daily_signals", "advanced_analytics", "email_support"]
            ),
            SubscriptionPlan(
                tier=SubscriptionTier.PREMIUM,
                name="Premium Plan",
                description="Unlimited AI signals and premium features",
                price_cents=4999,  # $49.99
                billing_period=BillingPeriod.MONTHLY,
                daily_ai_signals=None,  # Unlimited
                monthly_api_requests=None,  # Unlimited
                features=["unlimited_signals", "premium_analytics", "priority_support", "custom_models"]
            ),
            SubscriptionPlan(
                tier=SubscriptionTier.ENTERPRISE,
                name="Enterprise Plan",
                description="Custom solutions for institutional traders",
                price_cents=19999,  # $199.99
                billing_period=BillingPeriod.MONTHLY,
                daily_ai_signals=None,  # Unlimited
                monthly_api_requests=None,  # Unlimited
                features=["unlimited_everything", "dedicated_support", "custom_integration", "sla_guarantee"]
            )
        ]
        
        for plan in plans:
            self._subscription_plans[plan.tier] = plan
    
    async def create_usage_record(self, usage_record: UsageRecord) -> UsageRecord:
        """Create a new usage record."""
        self._usage_records[usage_record.id] = usage_record
        
        # Update daily summary
        await self._update_daily_summary(usage_record)
        
        # Update usage limits
        await self._update_usage_limits(usage_record)
        
        logger.debug(f"Created usage record: {usage_record.id}")
        return usage_record
    
    async def get_usage_record(self, record_id: str) -> Optional[UsageRecord]:
        """Get a usage record by ID."""
        return self._usage_records.get(record_id)
    
    async def get_user_usage_records(
        self, 
        user_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        usage_type: Optional[UsageType] = None
    ) -> List[UsageRecord]:
        """Get usage records for a user with optional filtering."""
        records = []
        
        for record in self._usage_records.values():
            if record.user_id != user_id:
                continue
            
            if start_date and record.timestamp < start_date:
                continue
            
            if end_date and record.timestamp > end_date:
                continue
            
            if usage_type and record.usage_type != usage_type:
                continue
            
            records.append(record)
        
        # Sort by timestamp descending
        records.sort(key=lambda x: x.timestamp, reverse=True)
        return records
    
    async def get_daily_usage_summary(self, user_id: str, date: datetime) -> Optional[UsageSummary]:
        """Get daily usage summary for a user."""
        # Normalize date to start of day
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        key = f"{user_id}_{date.date()}"
        return self._usage_summaries.get(key)
    
    async def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Get user's current subscription."""
        return self._user_subscriptions.get(user_id)
    
    async def create_user_subscription(self, subscription: UserSubscription) -> UserSubscription:
        """Create or update user subscription."""
        self._user_subscriptions[subscription.user_id] = subscription
        
        # Initialize usage limits
        await self._initialize_usage_limits(subscription)
        
        logger.info(f"Created subscription for user {subscription.user_id}: {subscription.subscription_tier}")
        return subscription
    
    async def get_usage_limits(self, user_id: str) -> Optional[UsageLimits]:
        """Get current usage limits for a user."""
        limits = self._usage_limits.get(user_id)
        
        if limits:
            # Check if we need to reset daily/monthly counters
            now = datetime.now(timezone.utc)
            
            # Reset daily counter if needed
            if now.date() > limits.daily_reset_time.date():
                limits.daily_ai_signals_used = 0
                limits.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Reset monthly counter if needed
            if now.month != limits.monthly_reset_time.month or now.year != limits.monthly_reset_time.year:
                limits.monthly_api_requests_used = 0
                limits.monthly_reset_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            self._usage_limits[user_id] = limits
        
        return limits
    
    async def update_usage_limits(self, limits: UsageLimits) -> UsageLimits:
        """Update usage limits for a user."""
        self._usage_limits[limits.user_id] = limits
        return limits
    
    async def create_billing_record(self, billing_record: BillingRecord) -> BillingRecord:
        """Create a new billing record."""
        self._billing_records[billing_record.id] = billing_record
        logger.info(f"Created billing record: {billing_record.id}")
        return billing_record
    
    async def get_user_billing_records(self, user_id: str) -> List[BillingRecord]:
        """Get billing records for a user."""
        records = [
            record for record in self._billing_records.values()
            if record.user_id == user_id
        ]
        records.sort(key=lambda x: x.created_at, reverse=True)
        return records
    
    async def get_subscription_plan(self, tier: SubscriptionTier) -> Optional[SubscriptionPlan]:
        """Get subscription plan by tier."""
        return self._subscription_plans.get(tier)
    
    async def get_all_subscription_plans(self) -> List[SubscriptionPlan]:
        """Get all available subscription plans."""
        return list(self._subscription_plans.values())
    
    async def get_conversion_metrics(self, date: datetime) -> Optional[ConversionMetrics]:
        """Get conversion metrics for a specific date."""
        date_key = date.date().isoformat()
        return self._conversion_metrics.get(date_key)
    
    async def update_conversion_metrics(self, metrics: ConversionMetrics) -> ConversionMetrics:
        """Update conversion metrics for a date."""
        date_key = metrics.date.date().isoformat()
        self._conversion_metrics[date_key] = metrics
        return metrics
    
    async def get_usage_analytics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage analytics for a date range."""
        analytics = {
            "total_requests": 0,
            "total_users": set(),
            "usage_by_type": defaultdict(int),
            "usage_by_tier": defaultdict(int),
            "total_cost_cents": 0,
            "average_processing_time_ms": 0,
            "daily_breakdown": defaultdict(lambda: {
                "requests": 0,
                "users": set(),
                "cost_cents": 0
            })
        }
        
        processing_times = []
        
        for record in self._usage_records.values():
            if start_date <= record.timestamp <= end_date:
                analytics["total_requests"] += 1
                analytics["total_users"].add(record.user_id)
                analytics["usage_by_type"][record.usage_type] += 1
                
                if record.cost_cents:
                    analytics["total_cost_cents"] += record.cost_cents
                
                if record.processing_time_ms:
                    processing_times.append(record.processing_time_ms)
                
                # Get user subscription tier
                subscription = await self.get_user_subscription(record.user_id)
                if subscription:
                    analytics["usage_by_tier"][subscription.subscription_tier] += 1
                
                # Daily breakdown
                date_key = record.timestamp.date().isoformat()
                analytics["daily_breakdown"][date_key]["requests"] += 1
                analytics["daily_breakdown"][date_key]["users"].add(record.user_id)
                if record.cost_cents:
                    analytics["daily_breakdown"][date_key]["cost_cents"] += record.cost_cents
        
        # Calculate averages
        analytics["total_users"] = len(analytics["total_users"])
        if processing_times:
            analytics["average_processing_time_ms"] = sum(processing_times) / len(processing_times)
        
        # Convert daily breakdown sets to counts
        for date_data in analytics["daily_breakdown"].values():
            date_data["users"] = len(date_data["users"])
        
        return dict(analytics)
    
    async def _update_daily_summary(self, usage_record: UsageRecord):
        """Update daily usage summary for a user."""
        date = usage_record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        key = f"{usage_record.user_id}_{date.date()}"
        
        summary = self._usage_summaries.get(key)
        if not summary:
            summary = UsageSummary(
                user_id=usage_record.user_id,
                date=date,
                usage_counts={},
                total_requests=0,
                total_cost_cents=0
            )
        
        # Update counts
        if usage_record.usage_type not in summary.usage_counts:
            summary.usage_counts[usage_record.usage_type] = 0
        summary.usage_counts[usage_record.usage_type] += 1
        summary.total_requests += 1
        
        if usage_record.cost_cents:
            summary.total_cost_cents += usage_record.cost_cents
        
        # Update average processing time
        if usage_record.processing_time_ms:
            if summary.average_processing_time_ms is None:
                summary.average_processing_time_ms = usage_record.processing_time_ms
            else:
                # Simple moving average approximation
                summary.average_processing_time_ms = (
                    summary.average_processing_time_ms + usage_record.processing_time_ms
                ) / 2
        
        self._usage_summaries[key] = summary
    
    async def _update_usage_limits(self, usage_record: UsageRecord):
        """Update usage limits based on a new usage record."""
        limits = await self.get_usage_limits(usage_record.user_id)
        if not limits:
            return
        
        # Update counters based on usage type
        if usage_record.usage_type == UsageType.AI_SIGNAL_REQUEST:
            limits.daily_ai_signals_used += 1
        
        # All requests count towards monthly API limit
        limits.monthly_api_requests_used += 1
        
        await self.update_usage_limits(limits)
    
    async def _initialize_usage_limits(self, subscription: UserSubscription):
        """Initialize usage limits for a new subscription."""
        plan = await self.get_subscription_plan(subscription.subscription_tier)
        if not plan:
            logger.warning(f"No plan found for tier: {subscription.subscription_tier}")
            return
        
        now = datetime.now(timezone.utc)
        limits = UsageLimits(
            user_id=subscription.user_id,
            subscription_tier=subscription.subscription_tier,
            daily_ai_signals_limit=plan.daily_ai_signals,
            daily_ai_signals_used=0,
            monthly_api_requests_limit=plan.monthly_api_requests,
            monthly_api_requests_used=0,
            daily_reset_time=now.replace(hour=0, minute=0, second=0, microsecond=0),
            monthly_reset_time=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        )
        
        await self.update_usage_limits(limits)
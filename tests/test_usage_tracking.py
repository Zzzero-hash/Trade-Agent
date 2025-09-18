"""
Tests for usage tracking and billing functionality.

This module tests the freemium usage tracking system including
usage limits, billing calculations, and subscription management.

Requirements: 7.1, 7.2, 7.5
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid

from src.models.usage_tracking import (
    UsageRecord, UsageSummary, UserSubscription, UsageLimits,
    BillingRecord, ConversionMetrics, SubscriptionPlan,
    UsageType, SubscriptionTier, BillingPeriod
)
from src.repositories.usage_repository import UsageRepository
from src.services.usage_tracking_service import UsageTrackingService


class TestUsageTrackingModels:
    """Test usage tracking data models."""
    
    def test_usage_record_creation(self):
        """Test creating a usage record with validation."""
        record = UsageRecord(
            id="usage_123",
            user_id="user_1",
            usage_type=UsageType.AI_SIGNAL_REQUEST,
            timestamp=datetime.now(timezone.utc),
            endpoint="/api/v1/predict",
            cost_cents=10,
            metadata={"model": "cnn_lstm"}
        )
        
        assert record.id == "usage_123"
        assert record.user_id == "user_1"
        assert record.usage_type == UsageType.AI_SIGNAL_REQUEST
        assert record.cost_cents == 10
        assert record.metadata["model"] == "cnn_lstm"
    
    def test_usage_limits_calculations(self):
        """Test usage limits calculations."""
        limits = UsageLimits(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            daily_ai_signals_limit=5,
            daily_ai_signals_used=3,
            monthly_api_requests_limit=1000,
            monthly_api_requests_used=500
        )
        
        assert not limits.is_daily_limit_exceeded()
        assert limits.get_daily_remaining() == 2
        assert not limits.is_monthly_limit_exceeded()
        assert limits.get_monthly_remaining() == 500
        
        # Test limit exceeded
        limits.daily_ai_signals_used = 5
        assert limits.is_daily_limit_exceeded()
        assert limits.get_daily_remaining() == 0
    
    def test_subscription_plan_validation(self):
        """Test subscription plan model validation."""
        plan = SubscriptionPlan(
            tier=SubscriptionTier.PREMIUM,
            name="Premium Plan",
            description="Unlimited signals",
            price_cents=4999,
            billing_period=BillingPeriod.MONTHLY,
            daily_ai_signals=None,  # Unlimited
            monthly_api_requests=None,  # Unlimited
            features=["unlimited_signals", "priority_support"]
        )
        
        assert plan.tier == SubscriptionTier.PREMIUM
        assert plan.price_cents == 4999
        assert plan.daily_ai_signals is None
        assert "unlimited_signals" in plan.features


class TestUsageRepository:
    """Test usage repository functionality."""
    
    @pytest.fixture
    def repository(self):
        """Create a fresh repository for each test."""
        return UsageRepository()
    
    @pytest.mark.asyncio
    async def test_create_usage_record(self, repository):
        """Test creating and retrieving usage records."""
        record = UsageRecord(
            id="usage_test_1",
            user_id="user_1",
            usage_type=UsageType.AI_SIGNAL_REQUEST,
            timestamp=datetime.now(timezone.utc),
            cost_cents=10
        )
        
        created_record = await repository.create_usage_record(record)
        assert created_record.id == "usage_test_1"
        
        retrieved_record = await repository.get_usage_record("usage_test_1")
        assert retrieved_record is not None
        assert retrieved_record.user_id == "user_1"
    
    @pytest.mark.asyncio
    async def test_user_subscription_management(self, repository):
        """Test subscription creation and retrieval."""
        now = datetime.now(timezone.utc)
        subscription = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.TRIAL,
            plan_id="trial_7_day",
            trial_start_date=now,
            trial_end_date=now + timedelta(days=7),
            is_trial_active=True,
            is_active=True
        )
        
        created_sub = await repository.create_user_subscription(subscription)
        assert created_sub.user_id == "user_1"
        assert created_sub.is_trial_active
        
        retrieved_sub = await repository.get_user_subscription("user_1")
        assert retrieved_sub is not None
        assert retrieved_sub.subscription_tier == SubscriptionTier.TRIAL
    
    @pytest.mark.asyncio
    async def test_usage_limits_reset(self, repository):
        """Test automatic usage limits reset."""
        # Create subscription first
        subscription = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            plan_id="free_plan",
            is_active=True
        )
        await repository.create_user_subscription(subscription)
        
        # Create usage limits with yesterday's reset time
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        limits = UsageLimits(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            daily_ai_signals_limit=5,
            daily_ai_signals_used=5,  # At limit
            daily_reset_time=yesterday
        )
        
        await repository.update_usage_limits(limits)
        
        # Retrieve limits - should auto-reset daily counter
        retrieved_limits = await repository.get_usage_limits("user_1")
        assert retrieved_limits.daily_ai_signals_used == 0  # Reset
        assert retrieved_limits.daily_reset_time.date() == datetime.now(timezone.utc).date()
    
    @pytest.mark.asyncio
    async def test_usage_analytics(self, repository):
        """Test usage analytics calculation."""
        # Create test usage records
        now = datetime.now(timezone.utc)
        records = [
            UsageRecord(
                id=f"usage_{i}",
                user_id=f"user_{i % 3}",  # 3 different users
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                timestamp=now - timedelta(hours=i),
                cost_cents=10,
                processing_time_ms=100.0 + i
            )
            for i in range(10)
        ]
        
        for record in records:
            await repository.create_usage_record(record)
        
        # Get analytics
        start_date = now - timedelta(days=1)
        end_date = now + timedelta(hours=1)
        analytics = await repository.get_usage_analytics(start_date, end_date)
        
        assert analytics["total_requests"] == 10
        assert analytics["total_users"] == 3
        assert analytics["usage_by_type"][UsageType.AI_SIGNAL_REQUEST] == 10
        assert analytics["total_cost_cents"] == 100
        assert analytics["average_processing_time_ms"] > 100


class TestUsageTrackingService:
    """Test usage tracking service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service with mock repository."""
        mock_repo = Mock(spec=UsageRepository)
        return UsageTrackingService(mock_repo)
    
    @pytest.mark.asyncio
    async def test_track_usage(self, service):
        """Test usage tracking functionality."""
        service.usage_repository.create_usage_record = AsyncMock()
        service.usage_repository.get_usage_limits = AsyncMock(return_value=None)
        service.usage_repository.get_user_subscription = AsyncMock(return_value=None)
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "test123"
            
            record = await service.track_usage(
                user_id="user_1",
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                endpoint="/api/v1/predict"
            )
            
            assert record.id == "usage_test123"
            assert record.user_id == "user_1"
            assert record.usage_type == UsageType.AI_SIGNAL_REQUEST
            assert record.cost_cents == 10  # From cost config
            
            service.usage_repository.create_usage_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_usage_limits_exceeded(self, service):
        """Test usage limit checking when limits are exceeded."""
        # Mock limits that are exceeded
        limits = UsageLimits(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            daily_ai_signals_limit=5,
            daily_ai_signals_used=5  # At limit
        )
        
        # Mock subscription (needed for trial expiration check)
        subscription = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            plan_id="free_plan",
            is_trial_active=False,
            is_active=True
        )
        
        service.usage_repository.get_usage_limits = AsyncMock(return_value=limits)
        service.usage_repository.get_user_subscription = AsyncMock(return_value=subscription)
        
        can_proceed, error_message = await service.check_usage_limits(
            "user_1", UsageType.AI_SIGNAL_REQUEST
        )
        
        assert not can_proceed
        assert "Daily AI signal limit exceeded" in error_message
    
    @pytest.mark.asyncio
    async def test_check_usage_limits_trial_expired(self, service):
        """Test usage limit checking when trial has expired."""
        # Mock expired trial subscription
        expired_trial = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.TRIAL,
            plan_id="trial_7_day",
            is_trial_active=True,
            trial_end_date=datetime.now(timezone.utc) - timedelta(days=1)  # Expired
        )
        
        limits = UsageLimits(
            user_id="user_1",
            subscription_tier=SubscriptionTier.TRIAL,
            daily_ai_signals_limit=5,
            daily_ai_signals_used=0
        )
        
        service.usage_repository.get_usage_limits = AsyncMock(return_value=limits)
        service.usage_repository.get_user_subscription = AsyncMock(return_value=expired_trial)
        
        can_proceed, error_message = await service.check_usage_limits(
            "user_1", UsageType.AI_SIGNAL_REQUEST
        )
        
        assert not can_proceed
        assert "Trial period has expired" in error_message
    
    @pytest.mark.asyncio
    async def test_create_trial_subscription(self, service):
        """Test creating a 7-day trial subscription."""
        service.usage_repository.create_user_subscription = AsyncMock()
        service._update_conversion_metrics = AsyncMock()
        
        subscription = await service.create_trial_subscription("user_1")
        
        assert subscription.user_id == "user_1"
        assert subscription.subscription_tier == SubscriptionTier.TRIAL
        assert subscription.is_trial_active
        
        # Check trial duration (7 days)
        trial_duration = subscription.trial_end_date - subscription.trial_start_date
        assert trial_duration.days == 7
        
        service.usage_repository.create_user_subscription.assert_called_once()
        service._update_conversion_metrics.assert_called_once_with("trial_started")
    
    @pytest.mark.asyncio
    async def test_upgrade_subscription(self, service):
        """Test subscription upgrade functionality."""
        # Mock current trial subscription
        current_sub = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.TRIAL,
            plan_id="trial_7_day",
            is_trial_active=True,
            trial_end_date=datetime.now(timezone.utc) + timedelta(days=3)
        )
        
        # Mock target plan
        target_plan = SubscriptionPlan(
            tier=SubscriptionTier.PREMIUM,
            name="Premium Plan",
            description="Unlimited signals",
            price_cents=4999,
            billing_period=BillingPeriod.MONTHLY
        )
        
        service.usage_repository.get_user_subscription = AsyncMock(return_value=current_sub)
        service.usage_repository.get_subscription_plan = AsyncMock(return_value=target_plan)
        service.usage_repository.create_user_subscription = AsyncMock(return_value=current_sub)
        service.usage_repository.create_billing_record = AsyncMock()
        service._update_conversion_metrics = AsyncMock()
        
        upgraded_sub = await service.upgrade_subscription("user_1", SubscriptionTier.PREMIUM)
        
        assert upgraded_sub.subscription_tier == SubscriptionTier.PREMIUM
        assert not upgraded_sub.is_trial_active  # Trial should be ended
        assert upgraded_sub.auto_renew
        
        service._update_conversion_metrics.assert_called_once_with("trial_to_paid")
        service.usage_repository.create_billing_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_cost_per_signal(self, service):
        """Test unit economics calculation."""
        # Mock analytics data
        mock_analytics = {
            "total_requests": 1000,
            "total_users": 100,
            "usage_by_type": {UsageType.AI_SIGNAL_REQUEST: 500},
            "usage_by_tier": {
                SubscriptionTier.PREMIUM: 20,
                SubscriptionTier.FREE: 80
            },
            "total_cost_cents": 5000
        }
        
        # Mock premium plan
        premium_plan = SubscriptionPlan(
            tier=SubscriptionTier.PREMIUM,
            name="Premium Plan",
            description="Unlimited signals",
            price_cents=4999,
            billing_period=BillingPeriod.MONTHLY
        )
        
        service.usage_repository.get_usage_analytics = AsyncMock(return_value=mock_analytics)
        service.usage_repository.get_subscription_plan = AsyncMock(return_value=premium_plan)
        
        unit_economics = await service.calculate_cost_per_signal()
        
        assert "metrics" in unit_economics
        assert unit_economics["metrics"]["total_signals"] == 500
        assert unit_economics["metrics"]["cost_per_signal_cents"] == 10.0  # 5000/500
        assert unit_economics["metrics"]["conversion_rate"] == 20.0  # 20/100 * 100
        assert "recommendations" in unit_economics
    
    @pytest.mark.asyncio
    async def test_get_usage_summary(self, service):
        """Test comprehensive usage summary generation."""
        # Mock data
        usage_records = [
            UsageRecord(
                id="usage_1",
                user_id="user_1",
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                timestamp=datetime.now(timezone.utc),
                cost_cents=10
            )
        ]
        
        limits = UsageLimits(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            daily_ai_signals_limit=5,
            daily_ai_signals_used=1
        )
        
        subscription = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            plan_id="free_plan",
            is_trial_active=False
        )
        
        today_summary = UsageSummary(
            user_id="user_1",
            date=datetime.now(timezone.utc),
            usage_counts={UsageType.AI_SIGNAL_REQUEST: 1},
            total_requests=1,
            total_cost_cents=10
        )
        
        service.usage_repository.get_user_usage_records = AsyncMock(return_value=usage_records)
        service.usage_repository.get_usage_limits = AsyncMock(return_value=limits)
        service.usage_repository.get_user_subscription = AsyncMock(return_value=subscription)
        service.usage_repository.get_daily_usage_summary = AsyncMock(return_value=today_summary)
        
        summary = await service.get_usage_summary("user_1")
        
        assert summary["user_id"] == "user_1"
        assert summary["subscription"]["tier"] == SubscriptionTier.FREE
        assert summary["usage_summary"]["total_requests"] == 1
        assert summary["current_limits"]["daily_ai_signals_used"] == 1
        assert summary["today_usage"]["ai_signals"] == 1
    
    @pytest.mark.asyncio
    async def test_get_upgrade_recommendations(self, service):
        """Test upgrade recommendations based on usage patterns."""
        # Mock user hitting daily limits
        subscription = UserSubscription(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            plan_id="free_plan",
            is_trial_active=False
        )
        
        limits = UsageLimits(
            user_id="user_1",
            subscription_tier=SubscriptionTier.FREE,
            daily_ai_signals_limit=5,
            daily_ai_signals_used=5  # At limit
        )
        
        plans = [
            SubscriptionPlan(
                tier=SubscriptionTier.PREMIUM,
                name="Premium Plan",
                description="Unlimited signals",
                price_cents=4999,
                billing_period=BillingPeriod.MONTHLY,
                features=["unlimited_signals"]
            )
        ]
        
        service.usage_repository.get_user_subscription = AsyncMock(return_value=subscription)
        service.usage_repository.get_usage_limits = AsyncMock(return_value=limits)
        service.usage_repository.get_all_subscription_plans = AsyncMock(return_value=plans)
        service.get_usage_summary = AsyncMock(return_value={"current_limits": {}})
        
        recommendations = await service.get_upgrade_recommendations("user_1")
        
        assert recommendations["current_tier"] == SubscriptionTier.FREE
        assert len(recommendations["recommendations"]) > 0
        assert recommendations["recommendations"][0]["reason"] == "daily_limit_exceeded"
        assert len(recommendations["available_plans"]) == 1


class TestUsageLimitsAndBilling:
    """Test usage limits enforcement and billing calculations."""
    
    @pytest.mark.asyncio
    async def test_trial_user_limits(self):
        """Test that trial users get 5 free AI signals per day for 7 days."""
        repository = UsageRepository()
        service = UsageTrackingService(repository)
        
        # Create trial subscription
        subscription = await service.create_trial_subscription("trial_user_1")
        
        # Verify trial limits
        limits = await repository.get_usage_limits("trial_user_1")
        assert limits.daily_ai_signals_limit == 5
        assert limits.daily_ai_signals_used == 0
        
        # Verify trial duration
        trial_duration = subscription.trial_end_date - subscription.trial_start_date
        assert trial_duration.days == 7
    
    @pytest.mark.asyncio
    async def test_free_tier_limits_enforcement(self):
        """Test that free tier limits are properly enforced."""
        repository = UsageRepository()
        service = UsageTrackingService(repository)
        
        # Create free subscription
        subscription = UserSubscription(
            user_id="free_user_1",
            subscription_tier=SubscriptionTier.FREE,
            plan_id="free_plan",
            is_active=True
        )
        await repository.create_user_subscription(subscription)
        
        # Use up daily limit
        for i in range(5):
            can_proceed, _ = await service.check_usage_limits("free_user_1", UsageType.AI_SIGNAL_REQUEST)
            assert can_proceed
            
            await service.track_usage(
                user_id="free_user_1",
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                endpoint="/api/v1/predict"
            )
        
        # 6th request should be blocked
        can_proceed, error_message = await service.check_usage_limits("free_user_1", UsageType.AI_SIGNAL_REQUEST)
        assert not can_proceed
        assert "Daily AI signal limit exceeded" in error_message
    
    @pytest.mark.asyncio
    async def test_billing_calculation_accuracy(self):
        """Test that billing calculations are accurate."""
        repository = UsageRepository()
        service = UsageTrackingService(repository)
        
        # Create premium subscription
        subscription = UserSubscription(
            user_id="premium_user_1",
            subscription_tier=SubscriptionTier.PREMIUM,
            plan_id="premium_plan",
            is_active=True
        )
        await repository.create_user_subscription(subscription)
        
        # Track various usage types
        usage_costs = []
        for usage_type in [UsageType.AI_SIGNAL_REQUEST, UsageType.MODEL_PREDICTION, UsageType.API_REQUEST]:
            record = await service.track_usage(
                user_id="premium_user_1",
                usage_type=usage_type,
                endpoint="/api/v1/test"
            )
            usage_costs.append(record.cost_cents)
        
        # Verify costs match configuration
        expected_costs = [10, 5, 1]  # From service cost_config
        assert usage_costs == expected_costs
        
        # Get usage summary and verify total cost
        summary = await service.get_usage_summary("premium_user_1")
        assert summary["usage_summary"]["total_cost_cents"] == sum(expected_costs)
    
    @pytest.mark.asyncio
    async def test_conversion_metrics_tracking(self):
        """Test that conversion metrics are properly tracked."""
        repository = UsageRepository()
        service = UsageTrackingService(repository)
        
        # Create trial and upgrade to paid
        await service.create_trial_subscription("convert_user_1")
        await service.upgrade_subscription("convert_user_1", SubscriptionTier.PREMIUM)
        
        # Check conversion metrics
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        metrics = await repository.get_conversion_metrics(today)
        
        assert metrics is not None
        assert metrics.total_trial_users >= 1
        assert metrics.trial_to_paid_conversions >= 1
        assert metrics.trial_conversion_rate > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
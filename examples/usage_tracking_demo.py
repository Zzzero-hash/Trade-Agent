"""
Usage tracking and freemium functionality demonstration.

This example shows how the usage tracking system works for the freemium model,
including trial subscriptions, usage limits, and billing calculations.

Requirements: 7.1, 7.2, 7.5
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.usage_tracking_service import UsageTrackingService
from src.models.usage_tracking import UsageType, SubscriptionTier

# Simple logging for demo
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def demo_trial_user_journey():
    """
    Demonstrate a typical trial user journey.
    
    Requirement 7.1: New users get 5 free AI signal requests per day for 7 consecutive days
    """
    print("\n=== Trial User Journey Demo ===")
    
    service = UsageTrackingService()
    user_id = "demo_trial_user"
    
    # 1. Create trial subscription
    print(f"1. Creating 7-day trial subscription for user {user_id}")
    subscription = await service.create_trial_subscription(user_id)
    
    print(f"   ✓ Trial created: {subscription.subscription_tier}")
    print(f"   ✓ Trial expires: {subscription.trial_end_date}")
    print(f"   ✓ Trial duration: {(subscription.trial_end_date - subscription.trial_start_date).days} days")
    
    # 2. Check initial limits
    limits = await service.usage_repository.get_usage_limits(user_id)
    print(f"\n2. Initial usage limits:")
    print(f"   ✓ Daily AI signals: {limits.daily_ai_signals_used}/{limits.daily_ai_signals_limit}")
    print(f"   ✓ Monthly API requests: {limits.monthly_api_requests_used}/{limits.monthly_api_requests_limit}")
    
    # 3. Use AI signals (within limit)
    print(f"\n3. Using AI signals (within daily limit of {limits.daily_ai_signals_limit})")
    for i in range(3):
        can_proceed, error = await service.check_usage_limits(user_id, UsageType.AI_SIGNAL_REQUEST)
        if can_proceed:
            await service.track_usage(
                user_id=user_id,
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                endpoint="/api/v1/predict",
                metadata={"signal_type": "buy_recommendation"}
            )
            print(f"   ✓ AI signal {i+1} tracked successfully")
        else:
            print(f"   ✗ AI signal {i+1} blocked: {error}")
    
    # 4. Check updated limits
    updated_limits = await service.usage_repository.get_usage_limits(user_id)
    print(f"\n4. Updated usage limits:")
    print(f"   ✓ Daily AI signals: {updated_limits.daily_ai_signals_used}/{updated_limits.daily_ai_signals_limit}")
    print(f"   ✓ Remaining today: {updated_limits.get_daily_remaining()}")
    
    # 5. Try to exceed daily limit
    print(f"\n5. Attempting to exceed daily limit...")
    for i in range(3):  # Try to use 3 more (total would be 6, limit is 5)
        can_proceed, error = await service.check_usage_limits(user_id, UsageType.AI_SIGNAL_REQUEST)
        if can_proceed:
            await service.track_usage(
                user_id=user_id,
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                endpoint="/api/v1/predict"
            )
            print(f"   ✓ AI signal {i+4} tracked successfully")
        else:
            print(f"   ✗ AI signal {i+4} blocked: {error}")
            break
    
    # 6. Get usage summary
    summary = await service.get_usage_summary(user_id)
    print(f"\n6. Usage summary:")
    print(f"   ✓ Total requests: {summary['usage_summary']['total_requests']}")
    print(f"   ✓ Total cost: ${summary['usage_summary']['total_cost_cents']/100:.2f}")
    print(f"   ✓ AI signals today: {summary['today_usage']['ai_signals']}")
    
    return user_id


async def demo_upgrade_recommendations():
    """
    Demonstrate upgrade recommendations when limits are reached.
    
    Requirement 7.2: Clear upgrade options when free tier limits are reached
    """
    print("\n=== Upgrade Recommendations Demo ===")
    
    service = UsageTrackingService()
    user_id = "demo_upgrade_user"
    
    # Create a user who has hit their limits
    subscription = await service.create_trial_subscription(user_id)
    
    # Max out their daily usage
    for i in range(5):  # Use all 5 daily signals
        await service.track_usage(
            user_id=user_id,
            usage_type=UsageType.AI_SIGNAL_REQUEST,
            endpoint="/api/v1/predict"
        )
    
    print(f"1. User {user_id} has used all daily AI signals")
    
    # Get upgrade recommendations
    recommendations = await service.get_upgrade_recommendations(user_id)
    
    print(f"\n2. Upgrade recommendations:")
    print(f"   Current tier: {recommendations['current_tier']}")
    
    for rec in recommendations['recommendations']:
        print(f"   ✓ Reason: {rec['reason']}")
        print(f"     Message: {rec['message']}")
        print(f"     Suggested tiers: {rec['suggested_tiers']}")
        print(f"     Benefits: {rec['benefits']}")
    
    print(f"\n3. Available plans:")
    for plan in recommendations['available_plans']:
        price = f"${plan['price_cents']/100:.2f}" if plan['price_cents'] > 0 else "Free"
        signals = plan['daily_ai_signals'] or "Unlimited"
        print(f"   ✓ {plan['name']}: {price}/{plan['billing_period']}")
        print(f"     - {signals} daily AI signals")
        print(f"     - Features: {', '.join(plan['features'])}")
    
    return user_id


async def demo_subscription_upgrade():
    """Demonstrate subscription upgrade process."""
    print("\n=== Subscription Upgrade Demo ===")
    
    service = UsageTrackingService()
    user_id = "demo_premium_user"
    
    # 1. Start with trial
    print(f"1. Creating trial subscription for user {user_id}")
    trial_sub = await service.create_trial_subscription(user_id)
    print(f"   ✓ Trial tier: {trial_sub.subscription_tier}")
    
    # 2. Upgrade to premium
    print(f"\n2. Upgrading to Premium tier")
    premium_sub = await service.upgrade_subscription(user_id, SubscriptionTier.PREMIUM)
    print(f"   ✓ New tier: {premium_sub.subscription_tier}")
    print(f"   ✓ Auto-renew: {premium_sub.auto_renew}")
    print(f"   ✓ Next billing: {premium_sub.next_billing_date}")
    
    # 3. Check new limits
    limits = await service.usage_repository.get_usage_limits(user_id)
    print(f"\n3. Premium user limits:")
    print(f"   ✓ Daily AI signals: {limits.daily_ai_signals_limit or 'Unlimited'}")
    print(f"   ✓ Monthly API requests: {limits.monthly_api_requests_limit or 'Unlimited'}")
    
    # 4. Test unlimited usage
    print(f"\n4. Testing unlimited usage (10 AI signals)")
    for i in range(10):
        can_proceed, error = await service.check_usage_limits(user_id, UsageType.AI_SIGNAL_REQUEST)
        if can_proceed:
            await service.track_usage(
                user_id=user_id,
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                endpoint="/api/v1/predict"
            )
            print(f"   ✓ AI signal {i+1} processed")
        else:
            print(f"   ✗ AI signal {i+1} blocked: {error}")
            break
    
    # 5. Get billing history
    billing_records = await service.usage_repository.get_user_billing_records(user_id)
    print(f"\n5. Billing history:")
    for record in billing_records:
        print(f"   ✓ {record.subscription_tier}: ${record.amount_cents/100:.2f}")
        print(f"     Status: {record.status}, Created: {record.created_at}")
    
    return user_id


async def demo_unit_economics():
    """
    Demonstrate unit economics calculation for cost optimization.
    
    Requirement 7.5: Monitor cost-per-signal to maintain positive unit economics
    """
    print("\n=== Unit Economics Demo ===")
    
    service = UsageTrackingService()
    
    # Create some sample users and usage
    users = []
    for i in range(10):
        user_id = f"demo_user_{i}"
        
        # Mix of subscription tiers
        if i < 3:
            tier = SubscriptionTier.PREMIUM
        elif i < 6:
            tier = SubscriptionTier.FREE
        else:
            # Start with trial, some upgrade
            await service.create_trial_subscription(user_id)
            if i % 2 == 0:
                await service.upgrade_subscription(user_id, SubscriptionTier.BASIC)
            continue
        
        # Create subscription
        from src.models.usage_tracking import UserSubscription
        subscription = UserSubscription(
            user_id=user_id,
            subscription_tier=tier,
            plan_id=f"{tier}_plan",
            is_active=True
        )
        await service.usage_repository.create_user_subscription(subscription)
        users.append(user_id)
    
    # Generate usage data
    print("1. Generating sample usage data...")
    for user_id in users:
        # Each user makes some AI signal requests
        num_requests = 5 + (hash(user_id) % 10)  # 5-15 requests per user
        for _ in range(num_requests):
            await service.track_usage(
                user_id=user_id,
                usage_type=UsageType.AI_SIGNAL_REQUEST,
                endpoint="/api/v1/predict",
                processing_time_ms=150.0
            )
    
    # Calculate unit economics
    print("\n2. Calculating unit economics...")
    unit_economics = await service.calculate_cost_per_signal()
    
    print(f"\n3. Unit Economics Analysis:")
    metrics = unit_economics['metrics']
    print(f"   ✓ Total AI signals: {metrics['total_signals']}")
    print(f"   ✓ Total users: {metrics['total_users']}")
    print(f"   ✓ Paid users: {metrics['paid_users']}")
    print(f"   ✓ Conversion rate: {metrics['conversion_rate']:.1f}%")
    print(f"   ✓ Cost per signal: ${metrics['cost_per_signal_cents']/100:.3f}")
    print(f"   ✓ Revenue per signal: ${metrics['revenue_per_signal_cents']/100:.3f}")
    print(f"   ✓ Profit per signal: ${metrics['profit_per_signal_cents']/100:.3f}")
    print(f"   ✓ Profit margin: {metrics['profit_margin']:.1f}%")
    
    print(f"\n4. Recommendations:")
    for rec in unit_economics['recommendations']:
        print(f"   • {rec}")
    
    return unit_economics


async def demo_comprehensive_analytics():
    """Demonstrate comprehensive usage analytics."""
    print("\n=== Comprehensive Analytics Demo ===")
    
    service = UsageTrackingService()
    
    # Get analytics for the demo period
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1)
    
    analytics = await service.usage_repository.get_usage_analytics(start_date, end_date)
    
    print(f"1. Usage Analytics (last 24 hours):")
    print(f"   ✓ Total requests: {analytics['total_requests']}")
    print(f"   ✓ Unique users: {analytics['total_users']}")
    print(f"   ✓ Total cost: ${analytics['total_cost_cents']/100:.2f}")
    print(f"   ✓ Avg processing time: {analytics['average_processing_time_ms']:.1f}ms")
    
    print(f"\n2. Usage by type:")
    for usage_type, count in analytics['usage_by_type'].items():
        print(f"   ✓ {usage_type}: {count}")
    
    print(f"\n3. Usage by subscription tier:")
    for tier, count in analytics['usage_by_tier'].items():
        print(f"   ✓ {tier}: {count}")
    
    return analytics


async def main():
    """Run all usage tracking demos."""
    print("🚀 AI Trading Platform - Usage Tracking Demo")
    print("=" * 50)
    
    try:
        # Demo 1: Trial user journey
        trial_user = await demo_trial_user_journey()
        
        # Demo 2: Upgrade recommendations
        upgrade_user = await demo_upgrade_recommendations()
        
        # Demo 3: Subscription upgrade
        premium_user = await demo_subscription_upgrade()
        
        # Demo 4: Unit economics
        unit_economics = await demo_unit_economics()
        
        # Demo 5: Comprehensive analytics
        analytics = await demo_comprehensive_analytics()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• 7-day trial with 5 daily AI signals (Requirement 7.1)")
        print("• Clear upgrade recommendations when limits reached (Requirement 7.2)")
        print("• Cost-per-signal tracking for unit economics (Requirement 7.5)")
        print("• Comprehensive usage analytics and billing")
        print("• Automatic limit enforcement and reset")
        print("• Conversion metrics tracking")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
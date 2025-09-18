"""Usage tracking and billing data models.

These dataclasses back the freemium usage tracking implementation by
providing structured containers for usage events, limits, subscriptions,
billing, and conversion analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class UsageType(str, Enum):
    """Supported usage event categories."""

    AI_SIGNAL_REQUEST = "ai_signal_request"
    MODEL_PREDICTION = "model_prediction"
    BATCH_PREDICTION = "batch_prediction"
    API_REQUEST = "api_request"
    DATA_REQUEST = "data_request"


class SubscriptionTier(str, Enum):
    """Subscription tiers offered by the platform."""

    TRIAL = "trial"
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class BillingPeriod(str, Enum):
    """Supported billing cadences."""

    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass(slots=True)
class UsageRecord:
    """Represents a single usage event captured by the system."""

    id: str
    user_id: str
    usage_type: UsageType
    timestamp: datetime
    endpoint: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    processing_time_ms: Optional[float] = None
    cost_cents: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UsageSummary:
    """Aggregated usage statistics for a user and day."""

    user_id: str
    date: datetime
    usage_counts: Dict[UsageType, int] = field(default_factory=dict)
    total_requests: int = 0
    total_cost_cents: int = 0
    average_processing_time_ms: Optional[float] = None

    def add_record(self, record: UsageRecord) -> None:
        """Accumulate a usage record into the summary."""
        self.total_requests += 1
        if record.cost_cents:
            self.total_cost_cents += record.cost_cents

        self.usage_counts[record.usage_type] = self.usage_counts.get(record.usage_type, 0) + 1

        if record.processing_time_ms is not None:
            if self.average_processing_time_ms is None:
                self.average_processing_time_ms = record.processing_time_ms
            else:
                self.average_processing_time_ms = (
                    self.average_processing_time_ms + record.processing_time_ms
                ) / 2


@dataclass(slots=True)
class UsageLimits:
    """Tracks per-user daily and monthly allowance usage."""

    user_id: str
    subscription_tier: SubscriptionTier
    daily_ai_signals_limit: Optional[int] = None
    daily_ai_signals_used: int = 0
    monthly_api_requests_limit: Optional[int] = None
    monthly_api_requests_used: int = 0
    daily_reset_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    monthly_reset_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_daily_limit_exceeded(self) -> bool:
        """Return True if the daily AI signal quota has been consumed."""
        return (
            self.daily_ai_signals_limit is not None
            and self.daily_ai_signals_used >= self.daily_ai_signals_limit
        )

    def get_daily_remaining(self) -> Optional[int]:
        """Return remaining daily AI signals, or None for unlimited."""
        if self.daily_ai_signals_limit is None:
            return None
        return max(self.daily_ai_signals_limit - self.daily_ai_signals_used, 0)

    def is_monthly_limit_exceeded(self) -> bool:
        """Return True if the monthly API quota has been consumed."""
        return (
            self.monthly_api_requests_limit is not None
            and self.monthly_api_requests_used >= self.monthly_api_requests_limit
        )

    def get_monthly_remaining(self) -> Optional[int]:
        """Return remaining monthly API requests, or None for unlimited."""
        if self.monthly_api_requests_limit is None:
            return None
        return max(self.monthly_api_requests_limit - self.monthly_api_requests_used, 0)

    def reset_daily_limits(self) -> None:
        """Reset daily counters and timestamp."""
        self.daily_ai_signals_used = 0
        self.daily_reset_time = datetime.now(timezone.utc)

    def reset_monthly_limits(self) -> None:
        """Reset monthly counters and timestamp."""
        self.monthly_api_requests_used = 0
        now = datetime.now(timezone.utc)
        self.monthly_reset_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@dataclass(slots=True)
class SubscriptionPlan:
    """Defines capabilities and pricing for a subscription tier."""

    tier: SubscriptionTier
    name: str
    description: str
    price_cents: int
    billing_period: BillingPeriod
    daily_ai_signals: Optional[int] = None
    monthly_api_requests: Optional[int] = None
    features: List[str] = field(default_factory=list)


@dataclass(slots=True)
class UserSubscription:
    """Represents an individual user's subscription state."""

    user_id: str
    subscription_tier: SubscriptionTier
    plan_id: Optional[str] = None
    is_active: bool = True
    subscription_start_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscription_end_date: Optional[datetime] = None
    trial_start_date: Optional[datetime] = None
    trial_end_date: Optional[datetime] = None
    is_trial_active: bool = False
    auto_renew: bool = False
    billing_period: BillingPeriod = BillingPeriod.MONTHLY
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    next_billing_date: Optional[datetime] = None
    last_billed_at: Optional[datetime] = None
    payment_method_id: Optional[str] = None
    daily_usage_reset_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_period_start is None:
            self.current_period_start = self.subscription_start_date
        if self.daily_usage_reset_time is None:
            start = self.subscription_start_date
            self.daily_usage_reset_time = start.replace(hour=0, minute=0, second=0, microsecond=0)

    def end_trial(self) -> None:
        """Mark the trial as finished."""
        self.is_trial_active = False
        if self.trial_end_date is None:
            self.trial_end_date = datetime.now(timezone.utc)


@dataclass(slots=True)
class BillingRecord:
    """Records a billing transaction for a user subscription."""

    id: str
    user_id: str
    subscription_tier: SubscriptionTier
    amount_cents: int
    billing_period_start: datetime
    billing_period_end: datetime
    currency: str = "USD"
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    paid_at: Optional[datetime] = None
    usage_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_paid(self, paid_at: Optional[datetime] = None) -> None:
        """Update the record to reflect a completed payment."""
        self.status = "completed"
        self.paid_at = paid_at or datetime.now(timezone.utc)


@dataclass(slots=True)
class ConversionMetrics:
    """Captures daily conversion funnel statistics."""

    date: datetime
    total_trial_users: int = 0
    trial_to_paid_conversions: int = 0
    free_to_paid_conversions: int = 0
    trial_expirations: int = 0
    trial_conversion_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_conversion_rate(self) -> None:
        """Recalculate the trial conversion rate."""
        if self.total_trial_users > 0:
            self.trial_conversion_rate = (
                self.trial_to_paid_conversions / self.total_trial_users
            )
        else:
            self.trial_conversion_rate = 0.0


__all__ = [
    "UsageRecord",
    "UsageSummary",
    "UserSubscription",
    "UsageLimits",
    "BillingRecord",
    "ConversionMetrics",
    "SubscriptionPlan",
    "UsageType",
    "SubscriptionTier",
    "BillingPeriod",
]

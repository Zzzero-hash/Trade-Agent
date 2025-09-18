"""User domain models used across the trading platform."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserRole(str, Enum):
    """Supported user roles for access control."""

    ADMIN = "admin"
    PREMIUM = "premium"
    FREE = "free"
    TRIAL = "trial"


class User(BaseModel):
    """Primary user model shared by API modules."""

    id: str
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    trial_expires_at: Optional[datetime] = None
    daily_signal_count: int = 0
    daily_signal_limit: int = 5

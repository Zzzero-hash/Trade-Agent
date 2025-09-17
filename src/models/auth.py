"""
Authentication and authorization data models.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr


class UserRole(str, Enum):
    CUSTOMER = "customer"
    ADMIN = "admin"
    SUPPORT = "support"
    COMPLIANCE = "compliance"
    DEVELOPER = "developer"


class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING_VERIFICATION = "pending_verification"


class MFAType(str, Enum):
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    RESET_PASSWORD = "reset_password"
    EMAIL_VERIFICATION = "email_verification"


class User(BaseModel):
    """User account model."""
    user_id: UUID
    email: EmailStr
    password_hash: str
    full_name: str
    phone_number: Optional[str] = None
    role: UserRole = UserRole.CUSTOMER
    status: UserStatus = UserStatus.PENDING_VERIFICATION
    is_active: bool = True
    is_verified: bool = False
    email_verified: bool = False
    phone_verified: bool = False
    mfa_enabled: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    password_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class UserProfile(BaseModel):
    """Extended user profile information."""
    profile_id: UUID
    user_id: UUID
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    nationality: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    occupation: Optional[str] = None
    employer: Optional[str] = None
    annual_income: Optional[float] = None
    net_worth: Optional[float] = None
    investment_experience: Optional[str] = None
    risk_tolerance: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class MFADevice(BaseModel):
    """Multi-factor authentication device."""
    device_id: UUID
    user_id: UUID
    mfa_type: MFAType
    device_name: Optional[str] = None
    secret: Optional[str] = None  # Encrypted TOTP secret
    phone_number: Optional[str] = None  # For SMS MFA
    email: Optional[str] = None  # For email MFA
    backup_codes: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    last_used: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class Session(BaseModel):
    """User session model."""
    session_id: str
    user_id: UUID
    ip_address: str
    user_agent: str
    device_fingerprint: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class AuthToken(BaseModel):
    """Authentication token model."""
    token_id: UUID
    user_id: UUID
    session_id: Optional[str] = None
    token_type: TokenType
    token_hash: str  # Hashed token value
    scopes: List[str] = Field(default_factory=list)
    is_active: bool = True
    expires_at: datetime
    last_used: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class APIKey(BaseModel):
    """API key for programmatic access."""
    key_id: UUID
    user_id: UUID
    name: str
    key_hash: str  # Hashed API key
    scopes: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = None  # Requests per minute
    is_active: bool = True
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class Permission(BaseModel):
    """System permission model."""
    permission_id: UUID
    name: str
    description: str
    resource: str
    action: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class Role(BaseModel):
    """User role model."""
    role_id: UUID
    name: str
    description: str
    permissions: List[UUID] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class UserRole(BaseModel):
    """User-role assignment model."""
    assignment_id: UUID
    user_id: UUID
    role_id: UUID
    granted_by: UUID
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class LoginAttempt(BaseModel):
    """Login attempt tracking."""
    attempt_id: UUID
    email: Optional[str] = None
    user_id: Optional[UUID] = None
    ip_address: str
    user_agent: str
    success: bool
    failure_reason: Optional[str] = None
    mfa_required: bool = False
    mfa_success: Optional[bool] = None
    attempted_at: datetime = Field(default_factory=datetime.utcnow)
    location: Optional[Dict[str, str]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class PasswordReset(BaseModel):
    """Password reset request model."""
    reset_id: UUID
    user_id: UUID
    token_hash: str
    expires_at: datetime
    used: bool = False
    used_at: Optional[datetime] = None
    ip_address: str
    user_agent: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class EmailVerification(BaseModel):
    """Email verification model."""
    verification_id: UUID
    user_id: UUID
    email: EmailStr
    token_hash: str
    expires_at: datetime
    verified: bool = False
    verified_at: Optional[datetime] = None
    attempts: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SecurityEvent(BaseModel):
    """Security event model."""
    event_id: UUID
    user_id: Optional[UUID] = None
    event_type: str
    severity: str  # low, medium, high, critical
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class DeviceFingerprint(BaseModel):
    """Device fingerprint for fraud detection."""
    fingerprint_id: UUID
    user_id: UUID
    fingerprint_hash: str
    device_info: Dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = 1
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
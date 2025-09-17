"""Security and compliance data models."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


class KYCStatus(str, Enum):
    """KYC verification status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DocumentType(str, Enum):
    """Identity document types."""
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"


class AMLStatus(str, Enum):
    """AML screening status."""
    CLEAR = "clear"
    FLAGGED = "flagged"
    BLOCKED = "blocked"
    UNDER_REVIEW = "under_review"


class RiskTolerance(str, Enum):
    """Customer risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AccountTier(str, Enum):
    """Customer account tiers."""
    BASIC = "basic"
    PREMIUM = "premium"
    INSTITUTIONAL = "institutional"


class VerificationProvider(str, Enum):
    """Identity verification providers."""
    JUMIO = "jumio"
    ONFIDO = "onfido"


class ActionResult(str, Enum):
    """Audit action results."""
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"


class KYCDocument(BaseModel):
    """KYC document model."""
    document_id: UUID
    customer_id: UUID
    document_type: DocumentType
    file_path: str
    verification_provider: VerificationProvider
    verification_status: KYCStatus
    verification_result: Optional[Dict[str, Any]] = None
    uploaded_at: datetime
    verified_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class KYCData(BaseModel):
    """Customer KYC data model."""
    customer_id: UUID
    kyc_status: KYCStatus
    identity_verified: bool = False
    address_verified: bool = False
    document_type: Optional[DocumentType] = None
    verification_provider: Optional[VerificationProvider] = None
    aml_status: AMLStatus = AMLStatus.CLEAR
    accredited_investor: bool = False
    risk_score: Optional[float] = None
    last_updated: datetime
    documents: List[KYCDocument] = []


class OFACScreeningResult(BaseModel):
    """OFAC screening result."""
    customer_id: UUID
    screening_id: UUID
    is_match: bool
    match_score: Optional[float] = None
    matched_entries: List[Dict[str, Any]] = []
    screened_at: datetime
    list_version: str


class PEPScreeningResult(BaseModel):
    """PEP (Politically Exposed Person) screening result."""
    customer_id: UUID
    screening_id: UUID
    is_pep: bool
    pep_category: Optional[str] = None
    risk_level: Optional[str] = None
    matched_entries: List[Dict[str, Any]] = []
    screened_at: datetime
    list_version: str


class AuditLog(BaseModel):
    """Audit log entry model."""
    log_id: UUID
    customer_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    action: str
    resource: str
    resource_id: Optional[str] = None
    timestamp: datetime
    ip_address: str
    user_agent: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    result: ActionResult
    error_message: Optional[str] = None
    correlation_id: Optional[str] = None


class EncryptionKey(BaseModel):
    """Encryption key metadata."""
    key_id: str
    key_arn: str
    algorithm: str
    created_at: datetime
    status: str
    description: Optional[str] = None


class AuthToken(BaseModel):
    """Authentication token model."""
    token_id: UUID
    user_id: UUID
    token_type: str  # access, refresh, mfa
    expires_at: datetime
    created_at: datetime
    last_used: Optional[datetime] = None
    ip_address: str
    user_agent: Optional[str] = None
    is_revoked: bool = False


class MFADevice(BaseModel):
    """Multi-factor authentication device."""
    device_id: UUID
    user_id: UUID
    device_type: str  # totp, sms, email
    device_name: str
    secret_key: Optional[str] = None  # Encrypted
    phone_number: Optional[str] = None
    email: Optional[EmailStr] = None
    is_verified: bool = False
    is_primary: bool = False
    created_at: datetime
    last_used: Optional[datetime] = None


class SecurityEvent(BaseModel):
    """Security event model."""
    event_id: UUID
    event_type: str
    severity: str  # low, medium, high, critical
    user_id: Optional[UUID] = None
    ip_address: str
    user_agent: Optional[str] = None
    event_data: Dict[str, Any]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    status: str = "open"  # open, investigating, resolved, false_positive


class ComplianceReport(BaseModel):
    """Compliance report model."""
    report_id: UUID
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    generated_by: UUID
    file_path: str
    status: str = "generated"  # generated, submitted, approved
    metadata: Optional[Dict[str, Any]] = None
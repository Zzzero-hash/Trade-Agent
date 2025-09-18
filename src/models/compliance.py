"""
Compliance data models for KYC, AML, and audit logging.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr


class KYCStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    EXPIRED = "expired"


class AMLStatus(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    MANUAL_REVIEW = "manual_review"
    PENDING = "pending"


class DocumentType(str, Enum):
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"


class ActionResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


class KYCData(BaseModel):
    """KYC verification data model."""
    kyc_id: UUID
    customer_id: UUID
    status: KYCStatus
    provider: str  # jumio, onfido, etc.
    verification_id: str
    document_type: DocumentType
    identity_verified: bool = False
    address_verified: bool = False
    document_verified: bool = False
    selfie_verified: bool = False
    confidence_score: float = 0.0
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    verification_timestamp: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class AMLScreeningResult(BaseModel):
    """AML screening result model."""
    screening_id: UUID
    customer_id: UUID
    status: AMLStatus
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    screening_timestamp: datetime = Field(default_factory=datetime.utcnow)
    lists_version: str
    risk_score: float = 0.0
    manual_review_required: bool = False
    reviewer_id: Optional[UUID] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class AuditEvent(BaseModel):
    """Audit event model."""
    event_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str
    category: str
    severity: str
    source: str
    user_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: str
    action: str
    result: ActionResult
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class AuditLog(BaseModel):
    """Audit log entry model."""
    log_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    customer_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    action: str
    resource: str
    result: ActionResult
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ComplianceReport(BaseModel):
    """Compliance report model."""
    report_id: UUID
    report_type: str
    title: str
    description: Optional[str] = None
    period_start: datetime
    period_end: datetime
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: UUID
    customer_id: Optional[UUID] = None  # For customer-specific reports
    data: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    status: str = "generated"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class RegulatoryRequirement(BaseModel):
    """Regulatory requirement model."""
    requirement_id: UUID
    regulation: str  # SEC, FINRA, CFTC, etc.
    requirement_code: str
    title: str
    description: str
    compliance_deadline: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ComplianceCheck(BaseModel):
    """Compliance check model."""
    check_id: UUID
    requirement_id: UUID
    customer_id: Optional[UUID] = None
    check_type: str
    status: str  # compliant, non_compliant, pending
    last_checked: datetime = Field(default_factory=datetime.utcnow)
    next_check_due: Optional[datetime] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    remediation_actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SanctionsList(BaseModel):
    """Sanctions list model."""
    list_id: UUID
    list_name: str
    list_type: str  # OFAC_SDN, OFAC_CONSOLIDATED, PEP, etc.
    source_url: str
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: str
    record_count: int = 0
    checksum: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SanctionsEntry(BaseModel):
    """Individual sanctions list entry."""
    entry_id: UUID
    list_id: UUID
    external_id: str  # ID from the source list
    names: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    date_of_birth: Optional[str] = None
    place_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    programs: List[str] = Field(default_factory=list)
    entity_type: str  # individual, entity, vessel, etc.
    remarks: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class DataRetentionPolicy(BaseModel):
    """Data retention policy model."""
    policy_id: UUID
    data_type: str
    retention_period_days: int
    deletion_method: str  # secure_delete, anonymize, archive
    legal_basis: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class DataProcessingRecord(BaseModel):
    """GDPR data processing record."""
    record_id: UUID
    customer_id: UUID
    processing_purpose: str
    legal_basis: str
    data_categories: List[str] = Field(default_factory=list)
    recipients: List[str] = Field(default_factory=list)
    retention_period: str
    security_measures: List[str] = Field(default_factory=list)
    cross_border_transfers: bool = False
    transfer_safeguards: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ConsentRecord(BaseModel):
    """Customer consent record."""
    consent_id: UUID
    customer_id: UUID
    consent_type: str  # marketing, data_processing, cookies, etc.
    consent_given: bool
    consent_timestamp: datetime = Field(default_factory=datetime.utcnow)
    consent_method: str  # web_form, email, phone, etc.
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    withdrawn_at: Optional[datetime] = None
    withdrawal_method: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class IncidentReport(BaseModel):
    """Security/compliance incident report."""
    incident_id: UUID
    incident_type: str  # security_breach, compliance_violation, etc.
    severity: str  # low, medium, high, critical
    status: str  # open, investigating, resolved, closed
    title: str
    description: str
    affected_customers: List[UUID] = Field(default_factory=list)
    affected_systems: List[str] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    reported_by: UUID
    assigned_to: Optional[UUID] = None
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    regulatory_notification_required: bool = False
    regulatory_notification_sent: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
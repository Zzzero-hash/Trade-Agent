"""
Security and compliance API endpoints.

This module provides REST API endpoints for:
- Security monitoring and management
- Audit log access and querying
- Compliance reporting and data export
- Regulatory compliance checks

Requirements: 6.5, 9.5
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel, Field
from enum import Enum

from src.api.auth import get_current_user, require_role, UserRole, User
from src.services.security_service import get_security_service, DataClassification
from src.services.audit_service import (
    get_audit_service, AuditEventType, AuditSeverity, AuditQuery
)
from src.services.compliance_service import (
    get_compliance_service, RegulationType, ReportFormat, DataRetentionPolicy
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/security", tags=["security", "compliance"])


# Request/Response Models

class EncryptDataRequest(BaseModel):
    """Request model for data encryption."""
    data: Dict[str, Any]
    classification: DataClassification = DataClassification.CONFIDENTIAL


class EncryptDataResponse(BaseModel):
    """Response model for data encryption."""
    encrypted_data: str
    key_id: str
    algorithm: str
    timestamp: str


class AuditQueryRequest(BaseModel):
    """Request model for audit log queries."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    user_id: Optional[str] = None
    severity: Optional[AuditSeverity] = None
    system_component: Optional[str] = None
    correlation_id: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checks."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RegulatoryReportRequest(BaseModel):
    """Request model for regulatory reports."""
    regulation_type: RegulationType
    report_type: str
    start_time: datetime
    end_time: datetime
    format: ReportFormat = ReportFormat.JSON
    include_sensitive: bool = False


class DataRetentionRequest(BaseModel):
    """Request model for data retention policy application."""
    policy: DataRetentionPolicy


# Security Endpoints

@router.get("/status")
async def get_security_status(
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get current security status and key information.
    
    Requires admin role.
    """
    try:
        security_service = get_security_service()
        status = security_service.get_security_status()
        
        logger.info(f"Security status requested by admin: {current_user.id}")
        
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security status"
        )


@router.post("/encrypt", response_model=EncryptDataResponse)
async def encrypt_data(
    request: EncryptDataRequest,
    current_user: User = Depends(require_role(UserRole.PREMIUM))
) -> EncryptDataResponse:
    """
    Encrypt sensitive data.
    
    Requires premium role or higher.
    """
    try:
        security_service = get_security_service()
        
        # Encrypt the data
        encrypted_data = security_service.encrypt_data(
            request.data,
            classification=request.classification
        )
        
        # Log encryption action
        audit_service = get_audit_service()
        audit_service.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            description="Data encryption requested",
            details={
                "classification": request.classification.value,
                "data_size": len(str(request.data))
            },
            user_id=current_user.id,
            severity=AuditSeverity.MEDIUM,
            system_component="security_api"
        )
        
        return EncryptDataResponse(
            encrypted_data=encrypted_data.data.hex(),
            key_id=encrypted_data.key_id,
            algorithm=encrypted_data.algorithm,
            timestamp=encrypted_data.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Data encryption failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data encryption failed"
        )


@router.post("/rotate-keys")
async def rotate_encryption_keys(
    classification: Optional[DataClassification] = Body(None),
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Rotate encryption keys for security.
    
    Requires admin role.
    """
    try:
        security_service = get_security_service()
        
        # Rotate keys
        rotated_keys = security_service.rotate_keys(classification)
        
        # Log key rotation
        audit_service = get_audit_service()
        audit_service.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            description="Encryption keys rotated",
            details={
                "classification": classification.value if classification else "all",
                "rotated_keys": len(rotated_keys)
            },
            user_id=current_user.id,
            severity=AuditSeverity.HIGH,
            system_component="security_api"
        )
        
        logger.info(f"Keys rotated by admin {current_user.id}: {len(rotated_keys)} keys")
        
        return {
            "status": "success",
            "message": f"Rotated {len(rotated_keys)} encryption keys",
            "rotated_keys": rotated_keys,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Key rotation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Key rotation failed"
        )


# Audit Endpoints

@router.post("/audit/query")
async def query_audit_logs(
    request: AuditQueryRequest,
    current_user: User = Depends(require_role(UserRole.PREMIUM))
) -> Dict[str, Any]:
    """
    Query audit logs with filtering.
    
    Requires premium role or higher.
    """
    try:
        audit_service = get_audit_service()
        
        # Create audit query
        query = AuditQuery(
            start_time=request.start_time,
            end_time=request.end_time,
            event_types=request.event_types,
            user_id=request.user_id,
            severity=request.severity,
            system_component=request.system_component,
            correlation_id=request.correlation_id,
            limit=request.limit,
            offset=request.offset
        )
        
        # Non-admin users can only see their own events
        if current_user.role != UserRole.ADMIN:
            query.user_id = current_user.id
        
        # Execute query
        events = audit_service.query_events(query)
        
        # Convert events to dict format
        event_data = []
        for event in events:
            event_dict = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "severity": event.severity.value,
                "description": event.description,
                "system_component": event.system_component,
                "correlation_id": event.correlation_id
            }
            
            # Include details for admin users only
            if current_user.role == UserRole.ADMIN:
                event_dict["details"] = event.details
                event_dict["ip_address"] = event.ip_address
                event_dict["user_agent"] = event.user_agent
            
            event_data.append(event_dict)
        
        # Log audit query
        audit_service.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            description="Audit logs queried",
            details={
                "query_params": request.dict(exclude_none=True),
                "results_count": len(events)
            },
            user_id=current_user.id,
            severity=AuditSeverity.LOW,
            system_component="audit_api"
        )
        
        return {
            "status": "success",
            "data": {
                "events": event_data,
                "total_results": len(events),
                "query_params": request.dict(exclude_none=True)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit query failed"
        )


@router.get("/audit/user-activity/{user_id}")
async def get_user_activity(
    user_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    current_user: User = Depends(require_role(UserRole.PREMIUM))
) -> Dict[str, Any]:
    """
    Get activity for a specific user.
    
    Users can only see their own activity unless they are admin.
    """
    try:
        # Check permissions
        if current_user.role != UserRole.ADMIN and current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only view your own activity"
            )
        
        audit_service = get_audit_service()
        
        # Get user activity
        events = audit_service.get_user_activity(user_id, start_time, end_time)
        
        # Convert to response format
        activity_data = []
        for event in events:
            activity_data.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "description": event.description,
                "severity": event.severity.value
            })
        
        return {
            "status": "success",
            "data": {
                "user_id": user_id,
                "activity": activity_data,
                "total_events": len(events),
                "period": {
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user activity"
        )


@router.get("/audit/trading-activity")
async def get_trading_activity(
    user_id: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    current_user: User = Depends(require_role(UserRole.PREMIUM))
) -> Dict[str, Any]:
    """
    Get trading activity with optional filters.
    
    Non-admin users can only see their own trading activity.
    """
    try:
        # Check permissions for user_id filter
        if user_id and current_user.role != UserRole.ADMIN and current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only view your own trading activity"
            )
        
        # Set user_id for non-admin users
        if current_user.role != UserRole.ADMIN:
            user_id = current_user.id
        
        audit_service = get_audit_service()
        
        # Get trading activity
        events = audit_service.get_trading_activity(user_id, symbol, start_time, end_time)
        
        # Convert to response format
        trading_data = []
        for event in events:
            trading_data.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "symbol": event.details.get("symbol"),
                "description": event.description,
                "details": event.details if current_user.role == UserRole.ADMIN else {}
            })
        
        return {
            "status": "success",
            "data": {
                "trading_activity": trading_data,
                "total_events": len(events),
                "filters": {
                    "user_id": user_id,
                    "symbol": symbol,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trading activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trading activity"
        )


@router.post("/audit/verify-integrity")
async def verify_audit_integrity(
    event_ids: Optional[List[str]] = Body(None),
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Verify integrity of audit events.
    
    Requires admin role.
    """
    try:
        audit_service = get_audit_service()
        
        # Verify integrity
        results = audit_service.verify_audit_integrity(event_ids)
        
        # Log integrity check
        audit_service.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            description="Audit integrity verification performed",
            details={
                "events_checked": results["total_events"],
                "integrity_score": results["integrity_score"],
                "corrupted_events": len(results["corrupted_events"])
            },
            user_id=current_user.id,
            severity=AuditSeverity.HIGH,
            system_component="audit_api"
        )
        
        return {
            "status": "success",
            "data": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audit integrity verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit integrity verification failed"
        )


# Compliance Endpoints

@router.get("/compliance/dashboard")
async def get_compliance_dashboard(
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get compliance dashboard data.
    
    Requires admin role.
    """
    try:
        compliance_service = get_compliance_service()
        dashboard = compliance_service.get_compliance_dashboard()
        
        return {
            "status": "success",
            "data": dashboard,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get compliance dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance dashboard"
        )


@router.post("/compliance/check")
async def check_compliance(
    request: ComplianceCheckRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Perform compliance check against all active rules.
    
    Requires admin role.
    """
    try:
        compliance_service = get_compliance_service()
        
        # Perform compliance check
        results = compliance_service.check_compliance(
            request.start_time,
            request.end_time
        )
        
        return {
            "status": "success",
            "data": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed"
        )


@router.post("/compliance/report")
async def generate_regulatory_report(
    request: RegulatoryReportRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Generate regulatory compliance report.
    
    Requires admin role.
    """
    try:
        compliance_service = get_compliance_service()
        
        # Generate report
        report = compliance_service.generate_regulatory_report(
            regulation_type=request.regulation_type,
            report_type=request.report_type,
            start_time=request.start_time,
            end_time=request.end_time,
            format=request.format,
            include_sensitive=request.include_sensitive
        )
        
        return {
            "status": "success",
            "data": {
                "report_id": report.report_id,
                "regulation_type": report.regulation_type.value,
                "report_type": report.report_type,
                "file_path": report.file_path,
                "total_records": report.total_records,
                "generated_at": report.generated_at.isoformat(),
                "checksum": report.checksum
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Regulatory report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Regulatory report generation failed"
        )


@router.post("/compliance/data-retention")
async def apply_data_retention_policy(
    request: DataRetentionRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Apply data retention policy to audit logs.
    
    Requires admin role.
    """
    try:
        compliance_service = get_compliance_service()
        
        # Apply retention policy
        results = compliance_service.apply_data_retention_policy(request.policy)
        
        return {
            "status": "success",
            "data": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data retention policy application failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data retention policy application failed"
        )


@router.post("/audit/export")
async def export_audit_data(
    query_request: AuditQueryRequest,
    format: ReportFormat = ReportFormat.JSON,
    include_sensitive: bool = False,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Export audit data for regulatory compliance.
    
    Requires admin role.
    """
    try:
        audit_service = get_audit_service()
        
        # Create audit query
        query = AuditQuery(
            start_time=query_request.start_time,
            end_time=query_request.end_time,
            event_types=query_request.event_types,
            user_id=query_request.user_id,
            severity=query_request.severity,
            system_component=query_request.system_component,
            correlation_id=query_request.correlation_id,
            limit=query_request.limit,
            offset=query_request.offset
        )
        
        # Export data
        export_info = audit_service.export_audit_data(
            query=query,
            format=format.value,
            include_sensitive=include_sensitive
        )
        
        return {
            "status": "success",
            "data": export_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audit data export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit data export failed"
        )
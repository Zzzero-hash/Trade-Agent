"""Audit logging service with immutable storage and compliance reporting."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from uuid import UUID, uuid4
import hashlib
import hmac

from pydantic import BaseModel

from src.models.security import AuditLog, ActionResult, ComplianceReport
from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class AuditConfig(BaseModel):
    """Audit service configuration."""
    enable_immutable_storage: bool = True
    retention_days: int = 2555  # 7 years for financial compliance
    hash_algorithm: str = "sha256"
    signing_key: str
    s3_bucket: Optional[str] = None
    encryption_enabled: bool = True


class ImmutableLogEntry(BaseModel):
    """Immutable audit log entry with cryptographic integrity."""
    log_id: UUID
    timestamp: datetime
    sequence_number: int
    previous_hash: Optional[str]
    log_data: Dict[str, Any]
    content_hash: str
    signature: str


class AuditService:
    """Audit logging service with immutable storage and compliance features."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize audit service."""
        self.config = config or AuditConfig(signing_key="default-key")
        self.settings = get_settings()
        self._sequence_counter = 0
        self._last_hash = None
        
    async def log_action(self, 
                        action: str,
                        resource: str,
                        result: ActionResult,
                        customer_id: Optional[UUID] = None,
                        user_id: Optional[UUID] = None,
                        session_id: Optional[str] = None,
                        ip_address: str = "unknown",
                        user_agent: Optional[str] = None,
                        request_data: Optional[Dict[str, Any]] = None,
                        response_data: Optional[Dict[str, Any]] = None,
                        resource_id: Optional[str] = None,
                        error_message: Optional[str] = None,
                        correlation_id: Optional[str] = None) -> UUID:
        """Log an audit action with immutable storage."""
        
        log_id = uuid4()
        timestamp = datetime.utcnow()
        
        # Create audit log entry
        audit_log = AuditLog(
            log_id=log_id,
            customer_id=customer_id,
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            timestamp=timestamp,
            ip_address=ip_address,
            user_agent=user_agent,
            request_data=self._sanitize_data(request_data),
            response_data=self._sanitize_data(response_data),
            result=result,
            error_message=error_message,
            correlation_id=correlation_id or str(uuid4())
        )
        
        # Store in immutable format if enabled
        if self.config.enable_immutable_storage:
            await self._store_immutable_log(audit_log)
        else:
            await self._store_regular_log(audit_log)
        
        # Log to application logger
        log_level = logging.ERROR if result == ActionResult.FAILURE else logging.INFO
        logger.log(log_level, f"Audit: {action} on {resource} - {result}")
        
        return log_id
    
    async def _store_immutable_log(self, audit_log: AuditLog) -> None:
        """Store audit log in immutable format with cryptographic integrity."""
        
        # Increment sequence number
        self._sequence_counter += 1
        
        # Prepare log data
        log_data = audit_log.dict()
        
        # Calculate content hash
        content_json = json.dumps(log_data, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()
        
        # Create immutable entry
        immutable_entry = ImmutableLogEntry(
            log_id=audit_log.log_id,
            timestamp=audit_log.timestamp,
            sequence_number=self._sequence_counter,
            previous_hash=self._last_hash,
            log_data=log_data,
            content_hash=content_hash,
            signature=""  # Will be set below
        )
        
        # Calculate signature
        signature_data = f"{immutable_entry.sequence_number}:{immutable_entry.previous_hash}:{content_hash}"
        signature = hmac.new(
            self.config.signing_key.encode(),
            signature_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        immutable_entry.signature = signature
        
        # Calculate chain hash for next entry
        chain_data = f"{immutable_entry.sequence_number}:{signature}:{content_hash}"
        self._last_hash = hashlib.sha256(chain_data.encode()).hexdigest()
        
        # Store the immutable entry
        await self._persist_immutable_entry(immutable_entry)
        
        logger.debug(f"Stored immutable audit log {audit_log.log_id} with sequence {self._sequence_counter}")
    
    async def _store_regular_log(self, audit_log: AuditLog) -> None:
        """Store audit log in regular database format."""
        # This would store in the main database
        logger.info(f"Storing regular audit log {audit_log.log_id}")
        # Implement database storage
    
    async def _persist_immutable_entry(self, entry: ImmutableLogEntry) -> None:
        """Persist immutable entry to secure storage."""
        # In production, this would write to:
        # 1. Append-only database table
        # 2. S3 with object lock
        # 3. Blockchain or distributed ledger
        
        if self.config.s3_bucket:
            await self._store_to_s3(entry)
        
        # Also store in local append-only file for backup
        await self._store_to_file(entry)
    
    async def _store_to_s3(self, entry: ImmutableLogEntry) -> None:
        """Store immutable entry to S3 with object lock."""
        # Implementation would use boto3 to store in S3
        # with object lock enabled for immutability
        logger.debug(f"Would store to S3: {entry.log_id}")
    
    async def _store_to_file(self, entry: ImmutableLogEntry) -> None:
        """Store immutable entry to append-only file."""
        # Implementation would append to a local file
        logger.debug(f"Would store to file: {entry.log_id}")
    
    def _sanitize_data(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize sensitive data from audit logs."""
        if not data:
            return data
        
        # List of sensitive fields to redact
        sensitive_fields = {
            'password', 'token', 'secret', 'key', 'authorization',
            'ssn', 'social_security_number', 'credit_card', 'cvv',
            'account_number', 'routing_number', 'api_key'
        }
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def verify_log_integrity(self, start_sequence: int = 1, 
                                 end_sequence: Optional[int] = None) -> Dict[str, Any]:
        """Verify the integrity of the audit log chain."""
        
        if not self.config.enable_immutable_storage:
            return {"status": "skipped", "reason": "immutable storage disabled"}
        
        # Retrieve log entries in sequence
        entries = await self._retrieve_log_entries(start_sequence, end_sequence)
        
        if not entries:
            return {"status": "error", "reason": "no entries found"}
        
        verification_results = {
            "status": "success",
            "verified_entries": 0,
            "failed_entries": 0,
            "errors": []
        }
        
        previous_hash = None
        
        for entry in entries:
            try:
                # Verify content hash
                content_json = json.dumps(entry.log_data, sort_keys=True, default=str)
                calculated_hash = hashlib.sha256(content_json.encode()).hexdigest()
                
                if calculated_hash != entry.content_hash:
                    verification_results["errors"].append({
                        "sequence": entry.sequence_number,
                        "error": "content hash mismatch"
                    })
                    verification_results["failed_entries"] += 1
                    continue
                
                # Verify signature
                signature_data = f"{entry.sequence_number}:{entry.previous_hash}:{entry.content_hash}"
                expected_signature = hmac.new(
                    self.config.signing_key.encode(),
                    signature_data.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                if expected_signature != entry.signature:
                    verification_results["errors"].append({
                        "sequence": entry.sequence_number,
                        "error": "signature verification failed"
                    })
                    verification_results["failed_entries"] += 1
                    continue
                
                # Verify chain integrity
                if previous_hash is not None and entry.previous_hash != previous_hash:
                    verification_results["errors"].append({
                        "sequence": entry.sequence_number,
                        "error": "chain integrity broken"
                    })
                    verification_results["failed_entries"] += 1
                    continue
                
                # Calculate next hash in chain
                chain_data = f"{entry.sequence_number}:{entry.signature}:{entry.content_hash}"
                previous_hash = hashlib.sha256(chain_data.encode()).hexdigest()
                
                verification_results["verified_entries"] += 1
                
            except Exception as e:
                verification_results["errors"].append({
                    "sequence": entry.sequence_number,
                    "error": f"verification exception: {str(e)}"
                })
                verification_results["failed_entries"] += 1
        
        if verification_results["failed_entries"] > 0:
            verification_results["status"] = "failed"
        
        return verification_results
    
    async def _retrieve_log_entries(self, start_sequence: int, 
                                  end_sequence: Optional[int] = None) -> List[ImmutableLogEntry]:
        """Retrieve immutable log entries by sequence range."""
        # This would query the immutable storage
        # For now, returning empty list
        return []
    
    async def search_audit_logs(self, 
                              customer_id: Optional[UUID] = None,
                              user_id: Optional[UUID] = None,
                              action: Optional[str] = None,
                              resource: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              result: Optional[ActionResult] = None,
                              limit: int = 100) -> List[AuditLog]:
        """Search audit logs with filters."""
        
        # Build search criteria
        criteria = {}
        if customer_id:
            criteria["customer_id"] = customer_id
        if user_id:
            criteria["user_id"] = user_id
        if action:
            criteria["action"] = action
        if resource:
            criteria["resource"] = resource
        if result:
            criteria["result"] = result
        
        # This would query the database with filters
        logger.info(f"Searching audit logs with criteria: {criteria}")
        
        # Return empty list for now - implement with actual database
        return []
    
    async def generate_compliance_report(self, 
                                       report_type: str,
                                       start_date: datetime,
                                       end_date: datetime,
                                       generated_by: UUID) -> ComplianceReport:
        """Generate compliance report from audit logs."""
        
        report_id = uuid4()
        
        # Retrieve relevant audit logs
        audit_logs = await self.search_audit_logs(
            start_time=start_date,
            end_time=end_date
        )
        
        # Generate report based on type
        if report_type == "SEC_COMPLIANCE":
            report_data = await self._generate_sec_report(audit_logs, start_date, end_date)
        elif report_type == "FINRA_COMPLIANCE":
            report_data = await self._generate_finra_report(audit_logs, start_date, end_date)
        elif report_type == "AML_COMPLIANCE":
            report_data = await self._generate_aml_report(audit_logs, start_date, end_date)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # Store report
        file_path = f"compliance_reports/{report_type}_{report_id}.json"
        await self._store_compliance_report(file_path, report_data)
        
        report = ComplianceReport(
            report_id=report_id,
            report_type=report_type,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.utcnow(),
            generated_by=generated_by,
            file_path=file_path,
            metadata={"total_entries": len(audit_logs)}
        )
        
        logger.info(f"Generated compliance report {report_id} for {report_type}")
        
        return report
    
    async def _generate_sec_report(self, audit_logs: List[AuditLog], 
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate SEC compliance report."""
        return {
            "report_type": "SEC_COMPLIANCE",
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "total_transactions": len([log for log in audit_logs if log.action == "TRADE"]),
                "failed_transactions": len([log for log in audit_logs if log.result == ActionResult.FAILURE]),
                "customer_actions": len([log for log in audit_logs if log.customer_id is not None])
            },
            "details": [log.dict() for log in audit_logs]
        }
    
    async def _generate_finra_report(self, audit_logs: List[AuditLog],
                                   start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate FINRA compliance report."""
        return {
            "report_type": "FINRA_COMPLIANCE", 
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "kyc_verifications": len([log for log in audit_logs if log.action == "KYC_VERIFICATION"]),
                "aml_screenings": len([log for log in audit_logs if log.action == "AML_SCREENING"]),
                "suspicious_activities": len([log for log in audit_logs if log.result == ActionResult.BLOCKED])
            },
            "details": [log.dict() for log in audit_logs]
        }
    
    async def _generate_aml_report(self, audit_logs: List[AuditLog],
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate AML compliance report."""
        return {
            "report_type": "AML_COMPLIANCE",
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "ofac_screenings": len([log for log in audit_logs if log.action == "OFAC_SCREENING"]),
                "pep_screenings": len([log for log in audit_logs if log.action == "PEP_SCREENING"]),
                "flagged_customers": len([log for log in audit_logs if "flagged" in str(log.response_data)])
            },
            "details": [log.dict() for log in audit_logs]
        }
    
    async def _store_compliance_report(self, file_path: str, report_data: Dict[str, Any]) -> None:
        """Store compliance report to secure storage."""
        # This would store the report to S3 or secure file system
        logger.info(f"Storing compliance report to {file_path}")
        # Implement actual storage
    
    async def cleanup_old_logs(self) -> Dict[str, int]:
        """Clean up audit logs older than retention period."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        
        # This would delete logs older than cutoff_date
        # For immutable logs, this might involve archiving to cold storage
        
        logger.info(f"Would clean up audit logs older than {cutoff_date}")
        
        return {
            "deleted_logs": 0,
            "archived_logs": 0,
            "cutoff_date": cutoff_date.isoformat()
        }
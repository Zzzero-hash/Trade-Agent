"""
Audit logging system with immutable storage and compliance reporting.
"""
import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError

from src.models.compliance import AuditLog, ActionResult, AuditEvent
from src.config.settings import get_settings
from src.repositories.audit_repository import AuditRepository
from src.services.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    TRADING = "trading"
    COMPLIANCE = "compliance"
    SYSTEM = "system"
    SECURITY = "security"


class AuditLogEntry(BaseModel):
    log_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    customer_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    action: str
    resource: str
    category: AuditCategory
    level: AuditLevel
    result: ActionResult
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_immutable_record(self) -> Dict[str, Any]:
        """Convert to immutable record format."""
        record = self.dict()
        record["timestamp"] = self.timestamp.isoformat()
        record["log_id"] = str(self.log_id)
        if self.customer_id:
            record["customer_id"] = str(self.customer_id)
        if self.user_id:
            record["user_id"] = str(self.user_id)
        
        # Calculate hash for integrity
        record_str = json.dumps(record, sort_keys=True)
        record["integrity_hash"] = hashlib.sha256(record_str.encode()).hexdigest()
        
        return record


class ImmutableStorage:
    """Immutable storage backend using AWS S3 with object lock."""
    
    def __init__(self):
        self.settings = get_settings()
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            region_name=self.settings.aws_region
        )
        self.bucket_name = self.settings.audit_s3_bucket
        self.encryption_service = EncryptionService()
    
    async def store_audit_log(self, log_entry: AuditLogEntry) -> str:
        """Store audit log in immutable S3 storage."""
        try:
            # Convert to immutable record
            record = log_entry.to_immutable_record()
            
            # Encrypt sensitive data
            encrypted_record = await self.encryption_service.encrypt_audit_data(record)
            
            # Generate S3 key with timestamp partitioning
            timestamp = log_entry.timestamp
            s3_key = (
                f"audit-logs/"
                f"year={timestamp.year}/"
                f"month={timestamp.month:02d}/"
                f"day={timestamp.day:02d}/"
                f"hour={timestamp.hour:02d}/"
                f"{log_entry.log_id}.json"
            )
            
            # Store in S3 with object lock
            response = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(encrypted_record),
                ContentType='application/json',
                Metadata={
                    'category': log_entry.category.value,
                    'level': log_entry.level.value,
                    'customer_id': str(log_entry.customer_id) if log_entry.customer_id else '',
                    'action': log_entry.action
                },
                ObjectLockMode='COMPLIANCE',
                ObjectLockRetainUntilDate=datetime.utcnow() + timedelta(days=2555)  # 7 years
            )
            
            logger.debug(f"Audit log stored in S3: {s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to store audit log in S3: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error storing audit log: {str(e)}")
            raise
    
    async def retrieve_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        customer_id: Optional[UUID] = None,
        category: Optional[AuditCategory] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve audit logs from immutable storage."""
        try:
            logs = []
            
            # Generate S3 prefixes for date range
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            while current_date <= end_date_only:
                prefix = (
                    f"audit-logs/"
                    f"year={current_date.year}/"
                    f"month={current_date.month:02d}/"
                    f"day={current_date.day:02d}/"
                )
                
                # List objects with prefix
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    for obj in page.get('Contents', []):
                        # Apply filters
                        metadata = obj.get('Metadata', {})
                        
                        if customer_id and metadata.get('customer_id') != str(customer_id):
                            continue
                        
                        if category and metadata.get('category') != category.value:
                            continue
                        
                        # Retrieve and decrypt log
                        response = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        
                        encrypted_data = json.loads(response['Body'].read())
                        decrypted_data = await self.encryption_service.decrypt_audit_data(encrypted_data)
                        logs.append(decrypted_data)
                
                current_date = current_date.replace(day=current_date.day + 1)
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit logs: {str(e)}")
            raise


class ComplianceReporter:
    """Generate compliance reports from audit logs."""
    
    def __init__(self, immutable_storage: ImmutableStorage):
        self.immutable_storage = immutable_storage
    
    async def generate_trading_activity_report(
        self,
        start_date: datetime,
        end_date: datetime,
        customer_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Generate trading activity report for compliance."""
        try:
            # Retrieve trading-related audit logs
            logs = await self.immutable_storage.retrieve_audit_logs(
                start_date=start_date,
                end_date=end_date,
                customer_id=customer_id,
                category=AuditCategory.TRADING
            )
            
            # Analyze trading patterns
            report = {
                "report_id": str(uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "customer_id": str(customer_id) if customer_id else "all",
                "summary": {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "failed_trades": 0,
                    "total_volume": 0.0,
                    "unique_symbols": set(),
                    "trading_days": set()
                },
                "details": []
            }
            
            for log in logs:
                if log.get("action") in ["place_order", "execute_trade", "cancel_order"]:
                    report["summary"]["total_trades"] += 1
                    
                    if log.get("result") == "success":
                        report["summary"]["successful_trades"] += 1
                    else:
                        report["summary"]["failed_trades"] += 1
                    
                    # Extract trading details
                    details = log.get("details", {})
                    if "symbol" in details:
                        report["summary"]["unique_symbols"].add(details["symbol"])
                    
                    if "volume" in details:
                        report["summary"]["total_volume"] += float(details.get("volume", 0))
                    
                    # Track trading days
                    trade_date = datetime.fromisoformat(log["timestamp"]).date()
                    report["summary"]["trading_days"].add(trade_date.isoformat())
                    
                    report["details"].append({
                        "timestamp": log["timestamp"],
                        "action": log["action"],
                        "result": log["result"],
                        "details": details
                    })
            
            # Convert sets to lists for JSON serialization
            report["summary"]["unique_symbols"] = list(report["summary"]["unique_symbols"])
            report["summary"]["trading_days"] = list(report["summary"]["trading_days"])
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate trading activity report: {str(e)}")
            raise
    
    async def generate_access_report(
        self,
        start_date: datetime,
        end_date: datetime,
        customer_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Generate data access report for compliance."""
        try:
            # Retrieve access-related audit logs
            logs = await self.immutable_storage.retrieve_audit_logs(
                start_date=start_date,
                end_date=end_date,
                customer_id=customer_id,
                category=AuditCategory.DATA_ACCESS
            )
            
            report = {
                "report_id": str(uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "customer_id": str(customer_id) if customer_id else "all",
                "summary": {
                    "total_accesses": len(logs),
                    "successful_accesses": 0,
                    "failed_accesses": 0,
                    "unique_resources": set(),
                    "unique_ip_addresses": set()
                },
                "access_patterns": [],
                "security_events": []
            }
            
            for log in logs:
                if log.get("result") == "success":
                    report["summary"]["successful_accesses"] += 1
                else:
                    report["summary"]["failed_accesses"] += 1
                    
                    # Track security events
                    if log.get("level") in ["warning", "error", "critical"]:
                        report["security_events"].append({
                            "timestamp": log["timestamp"],
                            "action": log["action"],
                            "resource": log["resource"],
                            "ip_address": log.get("ip_address"),
                            "details": log.get("details", {})
                        })
                
                report["summary"]["unique_resources"].add(log.get("resource", ""))
                if log.get("ip_address"):
                    report["summary"]["unique_ip_addresses"].add(log["ip_address"])
            
            # Convert sets to lists
            report["summary"]["unique_resources"] = list(report["summary"]["unique_resources"])
            report["summary"]["unique_ip_addresses"] = list(report["summary"]["unique_ip_addresses"])
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate access report: {str(e)}")
            raise


class AuditService:
    """Main audit service for logging and compliance reporting."""
    
    def __init__(self, audit_repo: AuditRepository):
        self.audit_repo = audit_repo
        self.immutable_storage = ImmutableStorage()
        self.compliance_reporter = ComplianceReporter(self.immutable_storage)
        
        # Buffer for batch processing
        self.log_buffer: List[AuditLogEntry] = []
        self.buffer_size = 100
        self.last_flush = datetime.utcnow()
        self.flush_interval = timedelta(minutes=5)
    
    async def log_event(
        self,
        action: str,
        resource: str,
        category: AuditCategory,
        level: AuditLevel = AuditLevel.INFO,
        result: ActionResult = ActionResult.SUCCESS,
        customer_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Log an audit event."""
        try:
            log_entry = AuditLogEntry(
                action=action,
                resource=resource,
                category=category,
                level=level,
                result=result,
                customer_id=customer_id,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                details=details or {},
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.log_buffer.append(log_entry)
            
            # Store in database for fast access
            await self.audit_repo.store_audit_log(log_entry)
            
            # Store in immutable storage (async)
            asyncio.create_task(self.immutable_storage.store_audit_log(log_entry))
            
            # Flush buffer if needed
            if len(self.log_buffer) >= self.buffer_size:
                await self._flush_buffer()
            
            logger.debug(f"Audit event logged: {action} on {resource}")
            return log_entry.log_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
            raise
    
    async def log_authentication_event(
        self,
        action: str,
        result: ActionResult,
        customer_id: Optional[UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Log authentication-related event."""
        level = AuditLevel.WARNING if result != ActionResult.SUCCESS else AuditLevel.INFO
        
        return await self.log_event(
            action=action,
            resource="authentication",
            category=AuditCategory.AUTHENTICATION,
            level=level,
            result=result,
            customer_id=customer_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
    
    async def log_trading_event(
        self,
        action: str,
        result: ActionResult,
        customer_id: UUID,
        symbol: str,
        details: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Log trading-related event."""
        level = AuditLevel.ERROR if result != ActionResult.SUCCESS else AuditLevel.INFO
        
        trading_details = {"symbol": symbol}
        if details:
            trading_details.update(details)
        
        return await self.log_event(
            action=action,
            resource=f"trading/{symbol}",
            category=AuditCategory.TRADING,
            level=level,
            result=result,
            customer_id=customer_id,
            details=trading_details
        )
    
    async def log_compliance_event(
        self,
        action: str,
        result: ActionResult,
        customer_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Log compliance-related event."""
        level = AuditLevel.CRITICAL if result != ActionResult.SUCCESS else AuditLevel.INFO
        
        return await self.log_event(
            action=action,
            resource="compliance",
            category=AuditCategory.COMPLIANCE,
            level=level,
            result=result,
            customer_id=customer_id,
            details=details
        )
    
    async def generate_compliance_report(
        self,
        report_type: str,
        start_date: datetime,
        end_date: datetime,
        customer_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Generate compliance report."""
        try:
            if report_type == "trading_activity":
                return await self.compliance_reporter.generate_trading_activity_report(
                    start_date, end_date, customer_id
                )
            elif report_type == "data_access":
                return await self.compliance_reporter.generate_access_report(
                    start_date, end_date, customer_id
                )
            else:
                raise ValueError(f"Unknown report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {str(e)}")
            raise
    
    async def _flush_buffer(self) -> None:
        """Flush log buffer to storage."""
        if not self.log_buffer:
            return
        
        try:
            # Process buffer
            buffer_copy = self.log_buffer.copy()
            self.log_buffer.clear()
            self.last_flush = datetime.utcnow()
            
            # Store in immutable storage (batch)
            tasks = [
                self.immutable_storage.store_audit_log(log_entry)
                for log_entry in buffer_copy
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.debug(f"Flushed {len(buffer_copy)} audit logs to immutable storage")
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {str(e)}")
    
    async def periodic_flush(self) -> None:
        """Periodic flush of audit buffer."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.utcnow()
                if (now - self.last_flush) >= self.flush_interval:
                    await self._flush_buffer()
                    
            except Exception as e:
                logger.error(f"Error in periodic flush: {str(e)}")
                await asyncio.sleep(60)
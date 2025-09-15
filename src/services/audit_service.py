"""
Audit logging service for comprehensive tracking of trading decisions and user actions.

This module provides:
- Comprehensive audit logging for all trading activities
- User action tracking and compliance monitoring
- Tamper-proof audit trails with cryptographic integrity
- Regulatory reporting and data export capabilities

Requirements: 6.5, 9.5
"""

import json
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.services.security_service import get_security_service, DataClassification

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # User actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_REGISTRATION = "user_registration"
    USER_PROFILE_UPDATE = "user_profile_update"
    PASSWORD_CHANGE = "password_change"

    # Trading actions
    TRADING_SIGNAL_GENERATED = "trading_signal_generated"
    TRADING_SIGNAL_EXECUTED = "trading_signal_executed"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Portfolio actions
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    RISK_LIMIT_TRIGGERED = "risk_limit_triggered"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"

    # Model actions
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_PREDICTION = "model_prediction"
    FEATURE_EXTRACTION = "feature_extraction"

    # System actions
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    ERROR_OCCURRED = "error_occurred"

    # Compliance actions
    REGULATORY_REPORT_GENERATED = "regulatory_report_generated"
    AUDIT_EXPORT = "audit_export"
    DATA_RETENTION_POLICY_APPLIED = "data_retention_policy_applied"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    severity: AuditSeverity
    description: str
    details: Dict[str, Any]
    system_component: str
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum for tamper detection."""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data."""
        # Create deterministic string representation
        data_dict = asdict(self)
        data_dict.pop('checksum', None)  # Remove checksum from calculation

        # Sort keys for consistent ordering
        sorted_data = json.dumps(data_dict, sort_keys=True, default=str)

        return hashlib.sha256(sorted_data.encode('utf-8')).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        expected_checksum = self._calculate_checksum()
        return hmac.compare_digest(self.checksum or "", expected_checksum)


@dataclass
class AuditQuery:
    """Query parameters for audit log searches."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    user_id: Optional[str] = None
    severity: Optional[AuditSeverity] = None
    system_component: Optional[str] = None
    correlation_id: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class AuditService:
    """Comprehensive audit logging service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.security_service = get_security_service()
        self._audit_log: List[AuditEvent] = []
        self._session_events: Dict[str, List[str]] = {}
        self._initialize_audit_storage()
    
    def _initialize_audit_storage(self) -> None:
        """Initialize audit log storage."""
        # In production, this would connect to a secure audit database
        # with write-only access and tamper detection

        audit_dir = Path("logs/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Audit service initialized")
    
    def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        system_component: str = "trading_platform",
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            description: Human-readable description
            details: Additional event details
            user_id: User ID if applicable
            session_id: Session ID if applicable
            ip_address: Client IP address
            user_agent: Client user agent
            severity: Event severity level
            system_component: Component that generated the event
            correlation_id: Correlation ID for related events
            parent_event_id: Parent event ID for hierarchical events

        Returns:
            Event ID of the logged event
        """
        try:
            event_id = str(uuid.uuid4())

            # Mask sensitive data in details
            safe_details = self.security_service.mask_sensitive_data(
                details or {})

            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                severity=severity,
                description=description,
                details=safe_details,
                system_component=system_component,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id
            )

            # Store event
            self._audit_log.append(event)

            # Track session events
            if session_id:
                if session_id not in self._session_events:
                    self._session_events[session_id] = []
                self._session_events[session_id].append(event_id)

            # Persist to storage
            self._persist_event(event)

            # Log high-severity events to system log
            if severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                logger.warning("High-severity audit event: %s - %s",
                               event_type.value, description)

            return event_id

        except Exception as e:
            logger.error("Failed to log audit event: %s", e)
            raise
    
    def _persist_event(self, event: AuditEvent) -> None:
        """Persist audit event to secure storage."""
        try:
            # Encrypt sensitive audit data
            event_data = asdict(event)
            encrypted_data = self.security_service.encrypt_data(
                event_data,
                classification=DataClassification.CONFIDENTIAL
            )

            # Write to daily audit log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            audit_file = Path(f"logs/audit/audit_{date_str}.log")

            # Append encrypted event to file
            with open(audit_file, "ab") as f:
                # Write encrypted data with metadata
                log_entry = {
                    'encrypted_data': encrypted_data.data.hex(),
                    'key_id': encrypted_data.key_id,
                    'algorithm': encrypted_data.algorithm,
                    'timestamp': encrypted_data.timestamp.isoformat()
                }
                f.write(json.dumps(log_entry).encode('utf-8') + b'\n')

        except Exception as e:
            logger.error("Failed to persist audit event: %s", e)
            # Don't raise - audit logging should not break main functionality
    
    def log_user_action(
        self,
        action: str,
        user_id: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log user action with appropriate event type."""
        # Map common actions to event types
        action_mapping = {
            'login': AuditEventType.USER_LOGIN,
            'logout': AuditEventType.USER_LOGOUT,
            'register': AuditEventType.USER_REGISTRATION,
            'profile_update': AuditEventType.USER_PROFILE_UPDATE,
            'password_change': AuditEventType.PASSWORD_CHANGE
        }

        event_type = action_mapping.get(
            action.lower(), AuditEventType.USER_PROFILE_UPDATE)

        return self.log_event(
            event_type=event_type,
            description=f"User {action}: {user_id}",
            details=details,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            severity=AuditSeverity.LOW,
            system_component="user_management"
        )
    
    def log_trading_action(
        self,
        action: str,
        user_id: str,
        symbol: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Log trading action with high severity."""
        # Map trading actions to event types
        action_mapping = {
            'signal_generated': AuditEventType.TRADING_SIGNAL_GENERATED,
            'signal_executed': AuditEventType.TRADING_SIGNAL_EXECUTED,
            'order_placed': AuditEventType.ORDER_PLACED,
            'order_cancelled': AuditEventType.ORDER_CANCELLED,
            'order_filled': AuditEventType.ORDER_FILLED,
            'position_opened': AuditEventType.POSITION_OPENED,
            'position_closed': AuditEventType.POSITION_CLOSED
        }

        event_type = action_mapping.get(
            action.lower(), AuditEventType.TRADING_SIGNAL_EXECUTED)

        # Add symbol to details
        trading_details = details.copy()
        trading_details['symbol'] = symbol

        return self.log_event(
            event_type=event_type,
            description=f"Trading {action}: {symbol} for user {user_id}",
            details=trading_details,
            user_id=user_id,
            session_id=session_id,
            severity=AuditSeverity.HIGH,
            system_component="trading_engine",
            correlation_id=correlation_id
        )
    
    def log_model_action(
        self,
        action: str,
        model_name: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Log ML model action."""
        # Map model actions to event types
        action_mapping = {
            'trained': AuditEventType.MODEL_TRAINED,
            'deployed': AuditEventType.MODEL_DEPLOYED,
            'prediction': AuditEventType.MODEL_PREDICTION,
            'feature_extraction': AuditEventType.FEATURE_EXTRACTION
        }
        
        event_type = action_mapping.get(action.lower(), AuditEventType.MODEL_PREDICTION)
        
        # Add model name to details
        model_details = details.copy()
        model_details['model_name'] = model_name
        
        return self.log_event(
            event_type=event_type,
            description=f"Model {action}: {model_name}",
            details=model_details,
            user_id=user_id,
            severity=AuditSeverity.MEDIUM,
            system_component="ml_engine",
            correlation_id=correlation_id
        )
    
    def log_security_event(
        self,
        event_description: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.HIGH
    ) -> str:
        """Log security-related event."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            description=event_description,
            details=details,
            user_id=user_id,
            ip_address=ip_address,
            severity=severity,
            system_component="security_service"
        )
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """
        Query audit events based on criteria.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching audit events
        """
        try:
            filtered_events = []
            
            for event in self._audit_log:
                # Apply filters
                if query.start_time and event.timestamp < query.start_time:
                    continue
                
                if query.end_time and event.timestamp > query.end_time:
                    continue
                
                if query.event_types and event.event_type not in query.event_types:
                    continue
                
                if query.user_id and event.user_id != query.user_id:
                    continue
                
                if query.severity and event.severity != query.severity:
                    continue
                
                if query.system_component and event.system_component != query.system_component:
                    continue
                
                if query.correlation_id and event.correlation_id != query.correlation_id:
                    continue
                
                filtered_events.append(event)
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            
            return filtered_events[start_idx:end_idx]
            
        except Exception as e:
            logger.error("Failed to query audit events: %s", e)
            raise
    
    def get_user_activity(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get all activity for a specific user."""
        query = AuditQuery(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit for comprehensive view
        )
        
        return self.query_events(query)
    
    def get_trading_activity(
        self,
        user_id: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get trading activity with optional filters."""
        trading_event_types = [
            AuditEventType.TRADING_SIGNAL_GENERATED,
            AuditEventType.TRADING_SIGNAL_EXECUTED,
            AuditEventType.ORDER_PLACED,
            AuditEventType.ORDER_CANCELLED,
            AuditEventType.ORDER_FILLED,
            AuditEventType.POSITION_OPENED,
            AuditEventType.POSITION_CLOSED
        ]
        
        query = AuditQuery(
            event_types=trading_event_types,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        events = self.query_events(query)
        
        # Filter by symbol if specified
        if symbol:
            events = [
                event for event in events
                if event.details.get('symbol') == symbol
            ]
        
        return events
    
    def verify_audit_integrity(
            self, event_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Verify integrity of audit events.

        Args:
            event_ids: Specific event IDs to verify (all if None)

        Returns:
            Integrity verification results
        """
        results: Dict[str, Any] = {
            'total_events': 0,
            'verified_events': 0,
            'corrupted_events': [],
            'integrity_score': 0.0
        }

        events_to_check = (
            [e for e in self._audit_log if e.event_id in event_ids]
            if event_ids
            else self._audit_log
        )

        for event in events_to_check:
            results['total_events'] += 1

            if event.verify_integrity():
                results['verified_events'] += 1
            else:
                results['corrupted_events'].append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value
                })

        # Calculate integrity score
        if results['total_events'] > 0:
            results['integrity_score'] = (
                results['verified_events'] / results['total_events'])

        return results
    
    def export_audit_data(
        self,
        query: AuditQuery,
        export_format: str = "json",
        include_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Export audit data for regulatory compliance.

        Args:
            query: Query parameters for data selection
            export_format: Export format (json, csv)
            include_sensitive: Whether to include sensitive data

        Returns:
            Export metadata and data location
        """
        try:
            events = self.query_events(query)

            # Create export metadata
            export_id = str(uuid.uuid4())
            export_timestamp = datetime.now(timezone.utc)

            export_data = {
                'export_id': export_id,
                'export_timestamp': export_timestamp.isoformat(),
                'query_parameters': asdict(query),
                'total_events': len(events),
                'format': export_format,
                'events': []
            }

            # Process events for export
            for event in events:
                event_dict = asdict(event)

                # Remove sensitive data if not requested
                if not include_sensitive:
                    event_dict['details'] = (
                        self.security_service.mask_sensitive_data(
                            event_dict['details']
                        ))

                export_data['events'].append(event_dict)

            # Save export file
            export_filename = (
                f"audit_export_{export_id}_"
                f"{export_timestamp.strftime('%Y%m%d_%H%M%S')}.{export_format}")
            export_path = Path(f"logs/exports/{export_filename}")
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, 'w', encoding='utf-8') as f:
                if export_format == 'json':
                    json.dump(export_data, f, indent=2, default=str)
                else:
                    # CSV format would require additional processing
                    raise ValueError(f"Unsupported export format: {export_format}")

            # Log export action
            self.log_event(
                event_type=AuditEventType.AUDIT_EXPORT,
                description=f"Audit data exported: {export_filename}",
                details={
                    'export_id': export_id,
                    'total_events': len(events),
                    'format': export_format,
                    'include_sensitive': include_sensitive
                },
                severity=AuditSeverity.HIGH,
                system_component="audit_service"
            )

            return {
                'export_id': export_id,
                'filename': export_filename,
                'path': str(export_path),
                'total_events': len(events),
                'timestamp': export_timestamp.isoformat()
            }

        except Exception as e:
            logger.error("Failed to export audit data: %s", e)
            raise
    
    def get_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for regulatory requirements.

        Args:
            start_time: Report start time
            end_time: Report end time
            user_id: Specific user ID (all users if None)

        Returns:
            Comprehensive compliance report
        """
        try:
            query = AuditQuery(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                limit=100000  # Large limit for comprehensive report
            )

            events = self.query_events(query)

            # Generate report statistics
            unique_users = set()
            events_by_type: Dict[str, int] = {}
            events_by_severity: Dict[str, int] = {}
            trading_activity = []
            security_incidents = []
            user_activity: Dict[str, Dict[str, Any]] = {}

            # Process events for report
            for event in events:
                # Count by type
                event_type = event.event_type.value
                events_by_type[event_type] = (
                    events_by_type.get(event_type, 0) + 1
                )

                # Count by severity
                severity = event.severity.value
                events_by_severity[severity] = (
                    events_by_severity.get(severity, 0) + 1
                )

                # Track unique users
                if event.user_id:
                    unique_users.add(event.user_id)

                # Collect trading activity
                if event.event_type in [
                    AuditEventType.ORDER_PLACED,
                    AuditEventType.ORDER_FILLED,
                    AuditEventType.POSITION_OPENED,
                    AuditEventType.POSITION_CLOSED
                ]:
                    trading_activity.append({
                        'timestamp': event.timestamp.isoformat(),
                        'user_id': event.user_id,
                        'action': event.event_type.value,
                        'symbol': event.details.get('symbol'),
                        'amount': event.details.get('amount'),
                        'price': event.details.get('price')
                    })

                # Collect security incidents
                if event.event_type == AuditEventType.SECURITY_EVENT:
                    security_incidents.append({
                        'timestamp': event.timestamp.isoformat(),
                        'description': event.description,
                        'severity': event.severity.value,
                        'user_id': event.user_id,
                        'ip_address': event.ip_address
                    })

                # Track user activity
                if event.user_id:
                    if event.user_id not in user_activity:
                        user_activity[event.user_id] = {
                            'total_events': 0,
                            'login_count': 0,
                            'trading_actions': 0,
                            'last_activity': None
                        }

                    user_stats = user_activity[event.user_id]
                    user_stats['total_events'] += 1

                    if event.event_type == AuditEventType.USER_LOGIN:
                        user_stats['login_count'] += 1

                    if (event.event_type.value.startswith('trading_') or
                            event.event_type.value.startswith('order_')):
                        user_stats['trading_actions'] += 1

                    if (not user_stats['last_activity'] or
                            event.timestamp.isoformat() > user_stats['last_activity']):
                        user_stats['last_activity'] = event.timestamp.isoformat()

            report = {
                'report_id': str(uuid.uuid4()),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'period': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                },
                'user_id': user_id,
                'summary': {
                    'total_events': len(events),
                    'events_by_type': events_by_type,
                    'events_by_severity': events_by_severity,
                    'unique_users': len(unique_users),
                    'trading_volume': 0,
                    'security_events': len(security_incidents)
                },
                'trading_activity': trading_activity,
                'security_incidents': security_incidents,
                'user_activity': user_activity
            }

            return report

        except Exception as e:
            logger.error("Failed to generate compliance report: %s", e)
            raise


# Global audit service instance
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get global audit service instance."""
    global _audit_service

    if _audit_service is None:
        _audit_service = AuditService()

    return _audit_service
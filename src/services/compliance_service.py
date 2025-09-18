"""
Compliance service for regulatory reporting and data export.

This module provides:
- Regulatory compliance reporting (MiFID II, GDPR, SOX, etc.)
- Data retention and archival policies
- Automated compliance monitoring and alerting
- Regulatory data export in required formats

Requirements: 6.5, 9.5
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import logging
from pathlib import Path
import zipfile
import hashlib

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.services.audit_service import get_audit_service, AuditQuery, AuditEventType
from src.services.security_service import get_security_service, DataClassification

logger = get_logger(__name__)


class RegulationType(str, Enum):
    """Types of regulatory frameworks."""
    MIFID_II = "mifid_ii"          # Markets in Financial Instruments Directive
    GDPR = "gdpr"                  # General Data Protection Regulation
    SOX = "sox"                    # Sarbanes-Oxley Act
    FINRA = "finra"                # Financial Industry Regulatory Authority
    SEC = "sec"                    # Securities and Exchange Commission
    CFTC = "cftc"                  # Commodity Futures Trading Commission
    PCI_DSS = "pci_dss"            # Payment Card Industry Data Security Standard
    CCPA = "ccpa"                  # California Consumer Privacy Act


class ReportFormat(str, Enum):
    """Supported report formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PDF = "pdf"
    EXCEL = "xlsx"


class DataRetentionPolicy(str, Enum):
    """Data retention policies."""
    SHORT_TERM = "short_term"      # 1 year
    MEDIUM_TERM = "medium_term"    # 3 years
    LONG_TERM = "long_term"        # 7 years
    PERMANENT = "permanent"        # Indefinite


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    regulation_type: RegulationType
    name: str
    description: str
    event_types: List[AuditEventType]
    retention_policy: DataRetentionPolicy
    monitoring_enabled: bool = True
    alert_threshold: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_id: str
    regulation_type: RegulationType
    description: str
    severity: str
    detected_at: datetime
    user_id: Optional[str]
    event_ids: List[str]
    resolved: bool = False
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class RegulatoryReport:
    """Regulatory report metadata."""
    report_id: str
    regulation_type: RegulationType
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    format: ReportFormat
    file_path: str
    checksum: str
    total_records: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceService:
    """Comprehensive compliance and regulatory reporting service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.audit_service = get_audit_service()
        self.security_service = get_security_service()
        self._compliance_rules: Dict[str, ComplianceRule] = {}
        self._violations: List[ComplianceViolation] = []
        self._reports: List[RegulatoryReport] = []
        self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize default compliance rules."""
        # MiFID II rules
        self.add_compliance_rule(ComplianceRule(
            rule_id="mifid_ii_transaction_reporting",
            regulation_type=RegulationType.MIFID_II,
            name="Transaction Reporting",
            description="All trading transactions must be reported within T+1",
            event_types=[
                AuditEventType.ORDER_PLACED,
                AuditEventType.ORDER_FILLED,
                AuditEventType.POSITION_OPENED,
                AuditEventType.POSITION_CLOSED
            ],
            retention_policy=DataRetentionPolicy.LONG_TERM
        ))
        
        # GDPR rules
        self.add_compliance_rule(ComplianceRule(
            rule_id="gdpr_data_processing",
            regulation_type=RegulationType.GDPR,
            name="Data Processing Consent",
            description="User consent required for personal data processing",
            event_types=[
                AuditEventType.USER_REGISTRATION,
                AuditEventType.USER_PROFILE_UPDATE
            ],
            retention_policy=DataRetentionPolicy.MEDIUM_TERM
        ))
        
        # SOX rules
        self.add_compliance_rule(ComplianceRule(
            rule_id="sox_financial_controls",
            regulation_type=RegulationType.SOX,
            name="Financial Controls",
            description="Internal controls over financial reporting",
            event_types=[
                AuditEventType.TRADING_SIGNAL_EXECUTED,
                AuditEventType.PORTFOLIO_REBALANCED,
                AuditEventType.RISK_LIMIT_TRIGGERED
            ],
            retention_policy=DataRetentionPolicy.LONG_TERM
        ))
        
        # Security compliance
        self.add_compliance_rule(ComplianceRule(
            rule_id="security_incident_reporting",
            regulation_type=RegulationType.SEC,
            name="Security Incident Reporting",
            description="Security incidents must be logged and reported",
            event_types=[AuditEventType.SECURITY_EVENT],
            retention_policy=DataRetentionPolicy.LONG_TERM,
            alert_threshold=1  # Alert on any security event
        ))
        
        logger.info(f"Initialized {len(self._compliance_rules)} compliance rules")
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a new compliance rule."""
        self._compliance_rules[rule.rule_id] = rule
        logger.info(f"Added compliance rule: {rule.rule_id}")
    
    def remove_compliance_rule(self, rule_id: str) -> bool:
        """Remove a compliance rule."""
        if rule_id in self._compliance_rules:
            del self._compliance_rules[rule_id]
            logger.info(f"Removed compliance rule: {rule_id}")
            return True
        return False
    
    def check_compliance(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check compliance against all active rules.
        
        Args:
            start_time: Check period start (default: last 24 hours)
            end_time: Check period end (default: now)
            
        Returns:
            Compliance check results
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        results = {
            'check_id': str(uuid.uuid4()),
            'period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'total_rules': len(self._compliance_rules),
            'rules_checked': 0,
            'violations_found': 0,
            'compliance_score': 0.0,
            'rule_results': {},
            'new_violations': []
        }
        
        for rule_id, rule in self._compliance_rules.items():
            if not rule.monitoring_enabled:
                continue
            
            rule_result = self._check_rule_compliance(rule, start_time, end_time)
            results['rule_results'][rule_id] = rule_result
            results['rules_checked'] += 1
            
            if rule_result['violations']:
                results['violations_found'] += len(rule_result['violations'])
                results['new_violations'].extend(rule_result['violations'])
        
        # Calculate compliance score
        if results['rules_checked'] > 0:
            compliant_rules = sum(
                1 for result in results['rule_results'].values()
                if not result['violations']
            )
            results['compliance_score'] = compliant_rules / results['rules_checked']
        
        logger.info(f"Compliance check completed: {results['violations_found']} violations found")
        
        return results
    
    def _check_rule_compliance(
        self,
        rule: ComplianceRule,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Check compliance for a specific rule."""
        # Query relevant audit events
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            event_types=rule.event_types,
            limit=10000
        )
        
        events = self.audit_service.query_events(query)
        
        rule_result = {
            'rule_id': rule.rule_id,
            'regulation_type': rule.regulation_type.value,
            'events_checked': len(events),
            'violations': [],
            'compliant': True
        }
        
        # Apply rule-specific compliance checks
        if rule.regulation_type == RegulationType.MIFID_II:
            violations = self._check_mifid_ii_compliance(rule, events)
        elif rule.regulation_type == RegulationType.GDPR:
            violations = self._check_gdpr_compliance(rule, events)
        elif rule.regulation_type == RegulationType.SOX:
            violations = self._check_sox_compliance(rule, events)
        else:
            violations = self._check_generic_compliance(rule, events)
        
        if violations:
            rule_result['violations'] = violations
            rule_result['compliant'] = False
            
            # Store violations
            for violation in violations:
                self._violations.append(violation)
        
        return rule_result
    
    def _check_mifid_ii_compliance(
        self,
        rule: ComplianceRule,
        events: List[Any]
    ) -> List[ComplianceViolation]:
        """Check MiFID II specific compliance requirements."""
        violations = []
        
        # Check transaction reporting timeliness (T+1 rule)
        for event in events:
            if event.event_type in [AuditEventType.ORDER_FILLED, AuditEventType.POSITION_CLOSED]:
                # Check if reported within T+1
                reporting_deadline = event.timestamp + timedelta(days=1)
                current_time = datetime.now(timezone.utc)
                
                if current_time > reporting_deadline:
                    # Check if already reported
                    # This is a simplified check - in practice, would verify against regulatory database
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        regulation_type=rule.regulation_type,
                        description=f"Transaction not reported within T+1: {event.event_id}",
                        severity="HIGH",
                        detected_at=current_time,
                        user_id=event.user_id,
                        event_ids=[event.event_id]
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_gdpr_compliance(
        self,
        rule: ComplianceRule,
        events: List[Any]
    ) -> List[ComplianceViolation]:
        """Check GDPR specific compliance requirements."""
        violations = []
        
        # Check for data processing without consent
        for event in events:
            if event.event_type == AuditEventType.USER_REGISTRATION:
                # Verify consent was obtained
                consent_given = event.details.get('consent_given', False)
                
                if not consent_given:
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        regulation_type=rule.regulation_type,
                        description=f"User registration without GDPR consent: {event.user_id}",
                        severity="HIGH",
                        detected_at=datetime.now(timezone.utc),
                        user_id=event.user_id,
                        event_ids=[event.event_id]
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_sox_compliance(
        self,
        rule: ComplianceRule,
        events: List[Any]
    ) -> List[ComplianceViolation]:
        """Check SOX specific compliance requirements."""
        violations = []
        
        # Check for proper authorization of financial transactions
        for event in events:
            if event.event_type in [
                AuditEventType.TRADING_SIGNAL_EXECUTED,
                AuditEventType.PORTFOLIO_REBALANCED
            ]:
                # Verify proper authorization
                authorized = event.details.get('authorized', False)
                authorization_level = event.details.get('authorization_level')
                
                if not authorized or not authorization_level:
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        regulation_type=rule.regulation_type,
                        description=f"Financial transaction without proper authorization: {event.event_id}",
                        severity="CRITICAL",
                        detected_at=datetime.now(timezone.utc),
                        user_id=event.user_id,
                        event_ids=[event.event_id]
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_generic_compliance(
        self,
        rule: ComplianceRule,
        events: List[Any]
    ) -> List[ComplianceViolation]:
        """Check generic compliance requirements."""
        violations = []
        
        # Check alert thresholds
        if rule.alert_threshold and len(events) >= rule.alert_threshold:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                regulation_type=rule.regulation_type,
                description=f"Alert threshold exceeded: {len(events)} events >= {rule.alert_threshold}",
                severity="MEDIUM",
                detected_at=datetime.now(timezone.utc),
                user_id=None,
                event_ids=[event.event_id for event in events]
            )
            violations.append(violation)
        
        return violations
    
    def generate_regulatory_report(
        self,
        regulation_type: RegulationType,
        report_type: str,
        start_time: datetime,
        end_time: datetime,
        format: ReportFormat = ReportFormat.JSON,
        include_sensitive: bool = False
    ) -> RegulatoryReport:
        """
        Generate regulatory report for specific regulation.
        
        Args:
            regulation_type: Type of regulation
            report_type: Specific report type
            start_time: Report period start
            end_time: Report period end
            format: Output format
            include_sensitive: Include sensitive data
            
        Returns:
            Generated report metadata
        """
        try:
            report_id = str(uuid.uuid4())
            generated_at = datetime.now(timezone.utc)
            
            # Get relevant audit events
            relevant_rules = [
                rule for rule in self._compliance_rules.values()
                if rule.regulation_type == regulation_type
            ]
            
            all_event_types = []
            for rule in relevant_rules:
                all_event_types.extend(rule.event_types)
            
            query = AuditQuery(
                start_time=start_time,
                end_time=end_time,
                event_types=list(set(all_event_types)),
                limit=100000
            )
            
            events = self.audit_service.query_events(query)
            
            # Generate report data based on regulation type
            if regulation_type == RegulationType.MIFID_II:
                report_data = self._generate_mifid_ii_report(events, report_type)
            elif regulation_type == RegulationType.GDPR:
                report_data = self._generate_gdpr_report(events, report_type)
            elif regulation_type == RegulationType.SOX:
                report_data = self._generate_sox_report(events, report_type)
            else:
                report_data = self._generate_generic_report(events, report_type)
            
            # Add report metadata
            report_data.update({
                'report_id': report_id,
                'regulation_type': regulation_type.value,
                'report_type': report_type,
                'period': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                },
                'generated_at': generated_at.isoformat(),
                'total_events': len(events),
                'format': format.value
            })
            
            # Save report file
            filename = f"{regulation_type.value}_{report_type}_{report_id}_{generated_at.strftime('%Y%m%d_%H%M%S')}.{format.value}"
            file_path = Path(f"logs/compliance/{filename}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write report in specified format
            if format == ReportFormat.JSON:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, default=str)
            elif format == ReportFormat.CSV:
                self._write_csv_report(file_path, report_data)
            elif format == ReportFormat.XML:
                self._write_xml_report(file_path, report_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate checksum
            checksum = self._calculate_file_checksum(file_path)
            
            # Create report record
            report = RegulatoryReport(
                report_id=report_id,
                regulation_type=regulation_type,
                report_type=report_type,
                period_start=start_time,
                period_end=end_time,
                generated_at=generated_at,
                format=format,
                file_path=str(file_path),
                checksum=checksum,
                total_records=len(events),
                metadata={
                    'rules_applied': [rule.rule_id for rule in relevant_rules],
                    'include_sensitive': include_sensitive
                }
            )
            
            self._reports.append(report)
            
            # Log report generation
            self.audit_service.log_event(
                event_type=AuditEventType.REGULATORY_REPORT_GENERATED,
                description=f"Generated {regulation_type.value} report: {report_type}",
                details={
                    'report_id': report_id,
                    'regulation_type': regulation_type.value,
                    'report_type': report_type,
                    'total_records': len(events),
                    'format': format.value
                },
                severity="HIGH",
                system_component="compliance_service"
            )
            
            logger.info(f"Generated regulatory report: {filename}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate regulatory report: {e}")
            raise
    
    def _generate_mifid_ii_report(self, events: List[Any], report_type: str) -> Dict[str, Any]:
        """Generate MiFID II specific report."""
        if report_type == "transaction_report":
            transactions = []
            
            for event in events:
                if event.event_type in [
                    AuditEventType.ORDER_PLACED,
                    AuditEventType.ORDER_FILLED,
                    AuditEventType.POSITION_OPENED,
                    AuditEventType.POSITION_CLOSED
                ]:
                    transactions.append({
                        'transaction_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'user_id': event.user_id,
                        'instrument': event.details.get('symbol'),
                        'action': event.event_type.value,
                        'quantity': event.details.get('quantity'),
                        'price': event.details.get('price'),
                        'venue': event.details.get('exchange', 'INTERNAL')
                    })
            
            return {
                'report_name': 'MiFID II Transaction Report',
                'transactions': transactions
            }
        
        return {'report_name': f'MiFID II {report_type}', 'data': []}
    
    def _generate_gdpr_report(self, events: List[Any], report_type: str) -> Dict[str, Any]:
        """Generate GDPR specific report."""
        if report_type == "data_processing_report":
            processing_activities = []
            
            for event in events:
                if event.event_type in [
                    AuditEventType.USER_REGISTRATION,
                    AuditEventType.USER_PROFILE_UPDATE
                ]:
                    processing_activities.append({
                        'activity_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'user_id': event.user_id,
                        'activity_type': event.event_type.value,
                        'legal_basis': event.details.get('legal_basis', 'consent'),
                        'data_categories': event.details.get('data_categories', []),
                        'consent_given': event.details.get('consent_given', False)
                    })
            
            return {
                'report_name': 'GDPR Data Processing Report',
                'processing_activities': processing_activities
            }
        
        return {'report_name': f'GDPR {report_type}', 'data': []}
    
    def _generate_sox_report(self, events: List[Any], report_type: str) -> Dict[str, Any]:
        """Generate SOX specific report."""
        if report_type == "financial_controls_report":
            control_activities = []
            
            for event in events:
                if event.event_type in [
                    AuditEventType.TRADING_SIGNAL_EXECUTED,
                    AuditEventType.PORTFOLIO_REBALANCED,
                    AuditEventType.RISK_LIMIT_TRIGGERED
                ]:
                    control_activities.append({
                        'control_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'user_id': event.user_id,
                        'control_type': event.event_type.value,
                        'authorized': event.details.get('authorized', False),
                        'authorization_level': event.details.get('authorization_level'),
                        'financial_impact': event.details.get('financial_impact', 0)
                    })
            
            return {
                'report_name': 'SOX Financial Controls Report',
                'control_activities': control_activities
            }
        
        return {'report_name': f'SOX {report_type}', 'data': []}
    
    def _generate_generic_report(self, events: List[Any], report_type: str) -> Dict[str, Any]:
        """Generate generic compliance report."""
        return {
            'report_name': f'Generic {report_type}',
            'events': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'user_id': event.user_id,
                    'description': event.description
                }
                for event in events
            ]
        }
    
    def _write_csv_report(self, file_path: Path, report_data: Dict[str, Any]) -> None:
        """Write report data to CSV format."""
        # This is a simplified CSV writer - would need more sophisticated handling
        # for complex nested data structures
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            if 'transactions' in report_data:
                fieldnames = ['transaction_id', 'timestamp', 'user_id', 'instrument', 'action', 'quantity', 'price', 'venue']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_data['transactions'])
            elif 'events' in report_data:
                fieldnames = ['event_id', 'timestamp', 'event_type', 'user_id', 'description']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_data['events'])
    
    def _write_xml_report(self, file_path: Path, report_data: Dict[str, Any]) -> None:
        """Write report data to XML format."""
        root = ET.Element("ComplianceReport")
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        for key, value in report_data.items():
            if not isinstance(value, (list, dict)):
                elem = ET.SubElement(metadata, key)
                elem.text = str(value)
        
        # Add data
        if 'transactions' in report_data:
            transactions_elem = ET.SubElement(root, "Transactions")
            for transaction in report_data['transactions']:
                trans_elem = ET.SubElement(transactions_elem, "Transaction")
                for key, value in transaction.items():
                    elem = ET.SubElement(trans_elem, key)
                    elem.text = str(value) if value is not None else ""
        
        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def apply_data_retention_policy(self, policy: DataRetentionPolicy) -> Dict[str, Any]:
        """
        Apply data retention policy to audit logs.
        
        Args:
            policy: Retention policy to apply
            
        Returns:
            Results of retention policy application
        """
        try:
            # Define retention periods
            retention_periods = {
                DataRetentionPolicy.SHORT_TERM: timedelta(days=365),
                DataRetentionPolicy.MEDIUM_TERM: timedelta(days=365 * 3),
                DataRetentionPolicy.LONG_TERM: timedelta(days=365 * 7),
                DataRetentionPolicy.PERMANENT: None
            }
            
            retention_period = retention_periods.get(policy)
            if retention_period is None:
                return {'message': 'Permanent retention - no data removed'}
            
            cutoff_date = datetime.now(timezone.utc) - retention_period
            
            # Get events older than retention period
            query = AuditQuery(
                end_time=cutoff_date,
                limit=100000
            )
            
            old_events = self.audit_service.query_events(query)
            
            results = {
                'policy': policy.value,
                'cutoff_date': cutoff_date.isoformat(),
                'events_identified': len(old_events),
                'events_archived': 0,
                'events_deleted': 0,
                'archive_files': []
            }
            
            if old_events:
                # Archive old events before deletion
                archive_filename = f"archived_events_{policy.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
                archive_path = Path(f"logs/archives/{archive_filename}")
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Create JSON file with archived events
                    archive_data = {
                        'archived_at': datetime.now(timezone.utc).isoformat(),
                        'retention_policy': policy.value,
                        'cutoff_date': cutoff_date.isoformat(),
                        'events': [asdict(event) for event in old_events]
                    }
                    
                    json_data = json.dumps(archive_data, indent=2, default=str)
                    zipf.writestr(f"events_{policy.value}.json", json_data)
                
                results['events_archived'] = len(old_events)
                results['archive_files'].append(str(archive_path))
                
                # In a real implementation, would delete events from database here
                # For now, just log the action
                logger.info(f"Archived {len(old_events)} events under {policy.value} retention policy")
            
            # Log retention policy application
            self.audit_service.log_event(
                event_type=AuditEventType.DATA_RETENTION_POLICY_APPLIED,
                description=f"Applied {policy.value} retention policy",
                details=results,
                severity="MEDIUM",
                system_component="compliance_service"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to apply data retention policy: {e}")
            raise
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        try:
            # Get recent compliance check
            recent_check = self.check_compliance()
            
            # Get violation statistics
            total_violations = len(self._violations)
            open_violations = len([v for v in self._violations if not v.resolved])
            
            # Get report statistics
            total_reports = len(self._reports)
            recent_reports = [
                r for r in self._reports
                if r.generated_at > datetime.now(timezone.utc) - timedelta(days=30)
            ]
            
            dashboard = {
                'compliance_score': recent_check['compliance_score'],
                'total_rules': len(self._compliance_rules),
                'active_rules': len([r for r in self._compliance_rules.values() if r.monitoring_enabled]),
                'violations': {
                    'total': total_violations,
                    'open': open_violations,
                    'resolved': total_violations - open_violations
                },
                'reports': {
                    'total': total_reports,
                    'recent': len(recent_reports)
                },
                'recent_violations': [
                    {
                        'violation_id': v.violation_id,
                        'regulation_type': v.regulation_type.value,
                        'description': v.description,
                        'severity': v.severity,
                        'detected_at': v.detected_at.isoformat()
                    }
                    for v in sorted(self._violations, key=lambda x: x.detected_at, reverse=True)[:10]
                ],
                'regulation_coverage': {
                    reg_type.value: len([
                        r for r in self._compliance_rules.values()
                        if r.regulation_type == reg_type
                    ])
                    for reg_type in RegulationType
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to generate compliance dashboard: {e}")
            raise


# Global compliance service instance
_compliance_service: Optional[ComplianceService] = None


def get_compliance_service() -> ComplianceService:
    """Get global compliance service instance."""
    global _compliance_service
    
    if _compliance_service is None:
        _compliance_service = ComplianceService()
    
    return _compliance_service
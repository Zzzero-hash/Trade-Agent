"""Compliance service for coordinating security and regulatory requirements."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from pydantic import BaseModel

from src.models.security import (
    KYCStatus, AMLStatus, ComplianceReport, SecurityEvent
)
from src.security.kyc_service import KYCService
from src.security.audit_service import AuditService
from src.security.encryption_service import EncryptionService
from src.security.auth_service import AuthService


logger = logging.getLogger(__name__)


class ComplianceConfig(BaseModel):
    """Compliance service configuration."""
    enable_kyc_monitoring: bool = True
    enable_aml_monitoring: bool = True
    enable_audit_monitoring: bool = True
    enable_security_monitoring: bool = True
    alert_threshold_high: int = 5
    alert_threshold_critical: int = 10
    report_generation_schedule: str = "daily"  # daily, weekly, monthly


class ComplianceAlert(BaseModel):
    """Compliance alert model."""
    alert_id: UUID
    alert_type: str
    severity: str  # low, medium, high, critical
    title: str
    description: str
    customer_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    status: str = "open"  # open, investigating, resolved, false_positive
    metadata: Optional[Dict[str, Any]] = None


class ComplianceMetrics(BaseModel):
    """Compliance metrics summary."""
    period_start: datetime
    period_end: datetime
    kyc_verifications: int
    kyc_rejections: int
    aml_screenings: int
    aml_flags: int
    security_events: int
    high_risk_events: int
    compliance_violations: int
    audit_log_entries: int


class ComplianceService:
    """Main compliance service coordinating all security and regulatory functions."""
    
    def __init__(self, 
                 kyc_service: KYCService,
                 audit_service: AuditService,
                 encryption_service: EncryptionService,
                 auth_service: AuthService,
                 config: Optional[ComplianceConfig] = None):
        """Initialize compliance service with dependent services."""
        self.kyc_service = kyc_service
        self.audit_service = audit_service
        self.encryption_service = encryption_service
        self.auth_service = auth_service
        self.config = config or ComplianceConfig()
        
        # In-memory stores (use database in production)
        self._compliance_alerts: Dict[UUID, ComplianceAlert] = {}
        self._compliance_metrics: List[ComplianceMetrics] = []
    
    async def perform_customer_onboarding_compliance(self, customer_id: UUID,
                                                   customer_data: Dict[str, Any],
                                                   ip_address: str) -> Dict[str, Any]:
        """Perform complete compliance check for customer onboarding."""
        
        try:
            # Log onboarding start
            await self.audit_service.log_action(
                action="CUSTOMER_ONBOARDING_START",
                resource="customer",
                result="success",
                customer_id=customer_id,
                ip_address=ip_address,
                request_data={"email": customer_data.get("email")}
            )
            
            # Initiate KYC verification
            kyc_result = await self.kyc_service.initiate_kyc_verification(
                customer_id=customer_id,
                provider="jumio",  # or "onfido"
                customer_data=customer_data
            )
            
            # Encrypt sensitive customer data
            encrypted_data = await self.encryption_service.encrypt_database_record(
                record=customer_data,
                sensitive_fields=["ssn", "date_of_birth", "address", "phone"],
                customer_id=customer_id
            )
            
            # Check for immediate compliance issues
            compliance_issues = await self._check_immediate_compliance_issues(
                customer_id, customer_data
            )
            
            if compliance_issues:
                await self._create_compliance_alert(
                    alert_type="ONBOARDING_COMPLIANCE_ISSUE",
                    severity="high",
                    title="Compliance Issues Detected During Onboarding",
                    description=f"Issues found: {', '.join(compliance_issues)}",
                    customer_id=customer_id
                )
            
            # Log successful onboarding initiation
            await self.audit_service.log_action(
                action="CUSTOMER_ONBOARDING_INITIATED",
                resource="customer",
                result="success",
                customer_id=customer_id,
                ip_address=ip_address,
                response_data={
                    "kyc_initiated": True,
                    "verification_url": kyc_result.get("verification_url"),
                    "compliance_issues": len(compliance_issues)
                }
            )
            
            return {
                "status": "initiated",
                "kyc_verification": kyc_result,
                "encrypted_data": encrypted_data,
                "compliance_issues": compliance_issues
            }
            
        except Exception as e:
            # Log onboarding failure
            await self.audit_service.log_action(
                action="CUSTOMER_ONBOARDING_FAILED",
                resource="customer",
                result="failure",
                customer_id=customer_id,
                ip_address=ip_address,
                error_message=str(e)
            )
            
            await self._create_compliance_alert(
                alert_type="ONBOARDING_FAILURE",
                severity="medium",
                title="Customer Onboarding Failed",
                description=f"Onboarding failed for customer {customer_id}: {str(e)}",
                customer_id=customer_id
            )
            
            raise
    
    async def process_kyc_completion(self, customer_id: UUID, 
                                   kyc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process KYC completion and update compliance status."""
        
        try:
            kyc_status = kyc_result.get("kyc_status")
            aml_status = kyc_result.get("aml_status", AMLStatus.CLEAR)
            
            # Log KYC completion
            await self.audit_service.log_action(
                action="KYC_VERIFICATION_COMPLETED",
                resource="customer",
                result="success",
                customer_id=customer_id,
                response_data={
                    "kyc_status": kyc_status,
                    "aml_status": aml_status,
                    "verification_provider": kyc_result.get("provider")
                }
            )
            
            # Check for compliance alerts based on results
            if kyc_status == KYCStatus.REJECTED:
                await self._create_compliance_alert(
                    alert_type="KYC_REJECTION",
                    severity="high",
                    title="KYC Verification Rejected",
                    description=f"KYC verification rejected for customer {customer_id}",
                    customer_id=customer_id,
                    metadata=kyc_result
                )
            
            if aml_status == AMLStatus.FLAGGED:
                await self._create_compliance_alert(
                    alert_type="AML_FLAG",
                    severity="critical",
                    title="AML Screening Flagged Customer",
                    description=f"Customer {customer_id} flagged in AML screening",
                    customer_id=customer_id,
                    metadata=kyc_result
                )
            
            # Update customer compliance status
            compliance_status = await self._calculate_customer_compliance_status(
                customer_id, kyc_status, aml_status
            )
            
            return {
                "compliance_status": compliance_status,
                "kyc_status": kyc_status,
                "aml_status": aml_status,
                "alerts_created": len([
                    alert for alert in self._compliance_alerts.values()
                    if alert.customer_id == customer_id and 
                    alert.triggered_at > datetime.utcnow() - timedelta(minutes=5)
                ])
            }
            
        except Exception as e:
            logger.error(f"Failed to process KYC completion: {e}")
            raise
    
    async def monitor_trading_compliance(self, customer_id: UUID,
                                       trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor trading activity for compliance violations."""
        
        try:
            # Log trade for audit
            await self.audit_service.log_action(
                action="TRADE_EXECUTED",
                resource="trade",
                result="success",
                customer_id=customer_id,
                request_data={
                    "symbol": trade_data.get("symbol"),
                    "side": trade_data.get("side"),
                    "quantity": trade_data.get("quantity"),
                    "price": trade_data.get("price")
                }
            )
            
            # Check for suspicious trading patterns
            suspicious_patterns = await self._detect_suspicious_trading_patterns(
                customer_id, trade_data
            )
            
            if suspicious_patterns:
                await self._create_compliance_alert(
                    alert_type="SUSPICIOUS_TRADING",
                    severity="high",
                    title="Suspicious Trading Pattern Detected",
                    description=f"Patterns detected: {', '.join(suspicious_patterns)}",
                    customer_id=customer_id,
                    metadata={"trade_data": trade_data, "patterns": suspicious_patterns}
                )
            
            # Check position limits and risk controls
            risk_violations = await self._check_risk_compliance(customer_id, trade_data)
            
            if risk_violations:
                await self._create_compliance_alert(
                    alert_type="RISK_VIOLATION",
                    severity="critical",
                    title="Risk Compliance Violation",
                    description=f"Violations: {', '.join(risk_violations)}",
                    customer_id=customer_id,
                    metadata={"trade_data": trade_data, "violations": risk_violations}
                )
            
            return {
                "compliance_status": "monitored",
                "suspicious_patterns": suspicious_patterns,
                "risk_violations": risk_violations,
                "alerts_created": len(suspicious_patterns) + len(risk_violations)
            }
            
        except Exception as e:
            logger.error(f"Trading compliance monitoring failed: {e}")
            raise
    
    async def generate_regulatory_reports(self, report_types: List[str],
                                        start_date: datetime,
                                        end_date: datetime,
                                        generated_by: UUID) -> List[ComplianceReport]:
        """Generate multiple regulatory reports."""
        
        reports = []
        
        for report_type in report_types:
            try:
                report = await self.audit_service.generate_compliance_report(
                    report_type=report_type,
                    start_date=start_date,
                    end_date=end_date,
                    generated_by=generated_by
                )
                reports.append(report)
                
                # Log report generation
                await self.audit_service.log_action(
                    action="COMPLIANCE_REPORT_GENERATED",
                    resource="compliance_report",
                    result="success",
                    user_id=generated_by,
                    response_data={
                        "report_type": report_type,
                        "report_id": str(report.report_id),
                        "period_start": start_date.isoformat(),
                        "period_end": end_date.isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to generate {report_type} report: {e}")
                
                await self._create_compliance_alert(
                    alert_type="REPORT_GENERATION_FAILED",
                    severity="medium",
                    title="Compliance Report Generation Failed",
                    description=f"Failed to generate {report_type} report: {str(e)}",
                    metadata={"report_type": report_type, "error": str(e)}
                )
        
        return reports
    
    async def perform_security_audit(self, audit_scope: List[str]) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        
        audit_results = {
            "audit_id": uuid4(),
            "started_at": datetime.utcnow(),
            "scope": audit_scope,
            "findings": [],
            "recommendations": []
        }
        
        try:
            # Audit log integrity
            if "audit_logs" in audit_scope:
                log_integrity = await self.audit_service.verify_log_integrity()
                audit_results["findings"].append({
                    "category": "audit_logs",
                    "status": log_integrity["status"],
                    "details": log_integrity
                })
            
            # Audit encryption status
            if "encryption" in audit_scope:
                encryption_status = await self.encryption_service.get_encryption_status()
                audit_results["findings"].append({
                    "category": "encryption",
                    "status": "compliant" if encryption_status["kms_available"] else "non_compliant",
                    "details": encryption_status
                })
            
            # Audit authentication security
            if "authentication" in audit_scope:
                auth_findings = await self._audit_authentication_security()
                audit_results["findings"].append({
                    "category": "authentication",
                    "status": auth_findings["status"],
                    "details": auth_findings
                })
            
            # Generate recommendations
            audit_results["recommendations"] = await self._generate_security_recommendations(
                audit_results["findings"]
            )
            
            audit_results["completed_at"] = datetime.utcnow()
            audit_results["status"] = "completed"
            
            # Log audit completion
            await self.audit_service.log_action(
                action="SECURITY_AUDIT_COMPLETED",
                resource="security_audit",
                result="success",
                response_data={
                    "audit_id": str(audit_results["audit_id"]),
                    "scope": audit_scope,
                    "findings_count": len(audit_results["findings"]),
                    "recommendations_count": len(audit_results["recommendations"])
                }
            )
            
            return audit_results
            
        except Exception as e:
            audit_results["status"] = "failed"
            audit_results["error"] = str(e)
            logger.error(f"Security audit failed: {e}")
            raise
    
    async def get_compliance_dashboard(self, period_days: int = 30) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Calculate metrics
        metrics = await self._calculate_compliance_metrics(start_date, end_date)
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self._compliance_alerts.values()
            if alert.triggered_at >= start_date
        ]
        
        # Get alert summary by severity
        alert_summary = {
            "critical": len([a for a in recent_alerts if a.severity == "critical"]),
            "high": len([a for a in recent_alerts if a.severity == "high"]),
            "medium": len([a for a in recent_alerts if a.severity == "medium"]),
            "low": len([a for a in recent_alerts if a.severity == "low"])
        }
        
        # Get compliance status summary
        compliance_status = await self._get_overall_compliance_status()
        
        return {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "metrics": metrics.dict(),
            "alert_summary": alert_summary,
            "recent_alerts": [alert.dict() for alert in recent_alerts[-10:]],
            "compliance_status": compliance_status,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _check_immediate_compliance_issues(self, customer_id: UUID,
                                               customer_data: Dict[str, Any]) -> List[str]:
        """Check for immediate compliance issues during onboarding."""
        
        issues = []
        
        # Check for required fields
        required_fields = ["first_name", "last_name", "email", "date_of_birth", "address"]
        for field in required_fields:
            if not customer_data.get(field):
                issues.append(f"Missing required field: {field}")
        
        # Check for sanctioned countries (mock check)
        country = customer_data.get("country", "").upper()
        sanctioned_countries = ["IR", "KP", "SY", "CU"]  # Example sanctioned countries
        if country in sanctioned_countries:
            issues.append(f"Customer from sanctioned country: {country}")
        
        # Check age requirement
        if customer_data.get("date_of_birth"):
            try:
                birth_date = datetime.strptime(customer_data["date_of_birth"], "%Y-%m-%d")
                age = (datetime.utcnow() - birth_date).days / 365.25
                if age < 18:
                    issues.append("Customer under minimum age requirement")
            except ValueError:
                issues.append("Invalid date of birth format")
        
        return issues
    
    async def _detect_suspicious_trading_patterns(self, customer_id: UUID,
                                                trade_data: Dict[str, Any]) -> List[str]:
        """Detect suspicious trading patterns."""
        
        patterns = []
        
        # Mock pattern detection - in production, implement sophisticated algorithms
        
        # Check for unusual trade size
        quantity = trade_data.get("quantity", 0)
        if quantity > 10000:  # Example threshold
            patterns.append("Unusually large trade size")
        
        # Check for rapid trading (would need historical data)
        # This is a simplified example
        patterns.append("Rapid trading pattern") if quantity > 5000 else None
        
        return [p for p in patterns if p]
    
    async def _check_risk_compliance(self, customer_id: UUID,
                                   trade_data: Dict[str, Any]) -> List[str]:
        """Check for risk compliance violations."""
        
        violations = []
        
        # Mock risk checks - implement actual risk calculations
        
        # Check position limits
        quantity = trade_data.get("quantity", 0)
        if quantity > 50000:  # Example limit
            violations.append("Position limit exceeded")
        
        # Check concentration limits
        symbol = trade_data.get("symbol", "")
        if symbol and quantity > 25000:  # Example concentration limit
            violations.append(f"Concentration limit exceeded for {symbol}")
        
        return violations
    
    async def _calculate_customer_compliance_status(self, customer_id: UUID,
                                                  kyc_status: KYCStatus,
                                                  aml_status: AMLStatus) -> str:
        """Calculate overall compliance status for customer."""
        
        if kyc_status == KYCStatus.VERIFIED and aml_status == AMLStatus.CLEAR:
            return "compliant"
        elif kyc_status == KYCStatus.REJECTED or aml_status == AMLStatus.BLOCKED:
            return "non_compliant"
        elif aml_status == AMLStatus.FLAGGED:
            return "under_review"
        else:
            return "pending"
    
    async def _create_compliance_alert(self, alert_type: str, severity: str,
                                     title: str, description: str,
                                     customer_id: Optional[UUID] = None,
                                     user_id: Optional[UUID] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> ComplianceAlert:
        """Create compliance alert."""
        
        alert = ComplianceAlert(
            alert_id=uuid4(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            customer_id=customer_id,
            user_id=user_id,
            triggered_at=datetime.utcnow(),
            metadata=metadata
        )
        
        self._compliance_alerts[alert.alert_id] = alert
        
        # Log alert creation
        await self.audit_service.log_action(
            action="COMPLIANCE_ALERT_CREATED",
            resource="compliance_alert",
            result="success",
            customer_id=customer_id,
            user_id=user_id,
            response_data={
                "alert_id": str(alert.alert_id),
                "alert_type": alert_type,
                "severity": severity
            }
        )
        
        logger.warning(f"Compliance alert created: {alert_type} - {severity}")
        
        return alert
    
    async def _calculate_compliance_metrics(self, start_date: datetime,
                                          end_date: datetime) -> ComplianceMetrics:
        """Calculate compliance metrics for period."""
        
        # Mock metrics calculation - implement with actual data queries
        return ComplianceMetrics(
            period_start=start_date,
            period_end=end_date,
            kyc_verifications=100,
            kyc_rejections=5,
            aml_screenings=100,
            aml_flags=2,
            security_events=50,
            high_risk_events=3,
            compliance_violations=1,
            audit_log_entries=1000
        )
    
    async def _audit_authentication_security(self) -> Dict[str, Any]:
        """Audit authentication security configuration."""
        
        # Mock audit - implement actual security checks
        return {
            "status": "compliant",
            "mfa_enabled": True,
            "password_policy_enforced": True,
            "session_timeout_configured": True,
            "token_expiration_appropriate": True
        }
    
    async def _generate_security_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on audit findings."""
        
        recommendations = []
        
        for finding in findings:
            if finding["status"] == "non_compliant":
                if finding["category"] == "encryption":
                    recommendations.append("Enable AWS KMS encryption for all sensitive data")
                elif finding["category"] == "authentication":
                    recommendations.append("Implement multi-factor authentication for all users")
                elif finding["category"] == "audit_logs":
                    recommendations.append("Fix audit log integrity issues immediately")
        
        return recommendations
    
    async def _get_overall_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status summary."""
        
        # Mock status - implement actual compliance calculations
        return {
            "overall_status": "compliant",
            "kyc_compliance": "compliant",
            "aml_compliance": "compliant", 
            "security_compliance": "compliant",
            "audit_compliance": "compliant",
            "last_assessment": datetime.utcnow().isoformat()
        }
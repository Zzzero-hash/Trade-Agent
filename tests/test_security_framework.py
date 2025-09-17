"""Comprehensive security tests for authentication and data protection."""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from src.security.kyc_service import KYCService, JumioConfig, OnfidoConfig
from src.security.audit_service import AuditService, AuditConfig
from src.security.encryption_service import EncryptionService, EncryptionConfig
from src.security.auth_service import AuthService, AuthConfig
from src.security.compliance_service import ComplianceService, ComplianceConfig
from src.models.security import (
    KYCStatus, AMLStatus, ActionResult, VerificationProvider,
    DocumentType, MFADevice, AuthToken, SecurityEvent
)


class TestKYCService:
    """Test KYC/AML service functionality."""
    
    @pytest.fixture
    def kyc_service(self):
        """Create KYC service for testing."""
        jumio_config = JumioConfig(
            api_token="test_token",
            api_secret="test_secret",
            callback_url="https://test.com/callback"
        )
        onfido_config = OnfidoConfig(
            api_token="test_token",
            webhook_token="test_webhook"
        )
        return KYCService(jumio_config=jumio_config, onfido_config=onfido_config)
    
    @pytest.mark.asyncio
    async def test_initiate_jumio_verification(self, kyc_service):
        """Test Jumio verification initiation."""
        customer_id = uuid4()
        customer_data = {
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        # Mock HTTP client response
        mock_response = Mock()
        mock_response.json.return_value = {
            "redirectUrl": "https://jumio.com/verify/123",
            "jumioIdScanReference": "scan_123"
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(kyc_service.jumio_client, 'post', return_value=mock_response):
            result = await kyc_service.initiate_kyc_verification(
                customer_id, VerificationProvider.JUMIO, customer_data
            )
        
        assert result["provider"] == VerificationProvider.JUMIO
        assert "verification_url" in result
        assert "session_id" in result
    
    @pytest.mark.asyncio
    async def test_initiate_onfido_verification(self, kyc_service):
        """Test Onfido verification initiation."""
        customer_id = uuid4()
        customer_data = {
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1990-01-01"
        }
        
        # Mock HTTP client responses
        mock_applicant_response = Mock()
        mock_applicant_response.json.return_value = {"id": "applicant_123"}
        mock_applicant_response.raise_for_status = Mock()
        
        mock_token_response = Mock()
        mock_token_response.json.return_value = {"token": "sdk_token_123"}
        mock_token_response.raise_for_status = Mock()
        
        with patch.object(kyc_service.onfido_client, 'post') as mock_post:
            mock_post.side_effect = [mock_applicant_response, mock_token_response]
            
            result = await kyc_service.initiate_kyc_verification(
                customer_id, VerificationProvider.ONFIDO, customer_data
            )
        
        assert result["provider"] == VerificationProvider.ONFIDO
        assert "sdk_token" in result
        assert "applicant_id" in result
    
    @pytest.mark.asyncio
    async def test_aml_screening(self, kyc_service):
        """Test AML screening functionality."""
        customer_id = uuid4()
        identity_data = {
            "firstName": "John",
            "lastName": "Doe",
            "dob": "1990-01-01"
        }
        
        result = await kyc_service.perform_aml_screening(customer_id, identity_data)
        
        assert "aml_status" in result
        assert "ofac_result" in result
        assert "pep_result" in result
        assert result["aml_status"] in [AMLStatus.CLEAR, AMLStatus.FLAGGED]
    
    @pytest.mark.asyncio
    async def test_ofac_screening(self, kyc_service):
        """Test OFAC screening."""
        customer_id = uuid4()
        
        # Test clean name
        result = await kyc_service._screen_ofac(customer_id, "Jane", "Smith", "1990-01-01")
        assert not result.is_match
        
        # Test flagged name
        result = await kyc_service._screen_ofac(customer_id, "John", "Doe", "1990-01-01")
        assert result.is_match
        assert result.match_score > 0
    
    @pytest.mark.asyncio
    async def test_pep_screening(self, kyc_service):
        """Test PEP screening."""
        customer_id = uuid4()
        
        # Test non-PEP name
        result = await kyc_service._screen_pep(customer_id, "Jane", "Smith", "1990-01-01")
        assert not result.is_pep
        
        # Test PEP name
        result = await kyc_service._screen_pep(customer_id, "Political", "Figure", "1950-01-01")
        assert result.is_pep
        assert result.pep_category == "government"


class TestAuditService:
    """Test audit logging service."""
    
    @pytest.fixture
    def audit_service(self):
        """Create audit service for testing."""
        config = AuditConfig(
            signing_key="test_signing_key",
            enable_immutable_storage=True
        )
        return AuditService(config)
    
    @pytest.mark.asyncio
    async def test_log_action(self, audit_service):
        """Test basic audit logging."""
        customer_id = uuid4()
        
        log_id = await audit_service.log_action(
            action="TEST_ACTION",
            resource="test_resource",
            result=ActionResult.SUCCESS,
            customer_id=customer_id,
            ip_address="192.168.1.1",
            request_data={"test": "data"}
        )
        
        assert log_id is not None
        assert isinstance(log_id, type(uuid4()))
    
    @pytest.mark.asyncio
    async def test_immutable_logging(self, audit_service):
        """Test immutable audit logging."""
        # Log multiple entries
        log_ids = []
        for i in range(3):
            log_id = await audit_service.log_action(
                action=f"TEST_ACTION_{i}",
                resource="test_resource",
                result=ActionResult.SUCCESS,
                ip_address="192.168.1.1"
            )
            log_ids.append(log_id)
        
        # Verify sequence numbers are incremented
        assert audit_service._sequence_counter == 3
        assert audit_service._last_hash is not None
    
    @pytest.mark.asyncio
    async def test_data_sanitization(self, audit_service):
        """Test sensitive data sanitization."""
        sensitive_data = {
            "password": "secret123",
            "api_key": "key123",
            "normal_field": "normal_value",
            "nested": {
                "token": "token123",
                "safe_field": "safe_value"
            }
        }
        
        sanitized = audit_service._sanitize_data(sensitive_data)
        
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["normal_field"] == "normal_value"
        assert sanitized["nested"]["token"] == "[REDACTED]"
        assert sanitized["nested"]["safe_field"] == "safe_value"
    
    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, audit_service):
        """Test compliance report generation."""
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        generated_by = uuid4()
        
        report = await audit_service.generate_compliance_report(
            report_type="SEC_COMPLIANCE",
            start_date=start_date,
            end_date=end_date,
            generated_by=generated_by
        )
        
        assert report.report_type == "SEC_COMPLIANCE"
        assert report.period_start == start_date
        assert report.period_end == end_date
        assert report.generated_by == generated_by
        assert report.file_path is not None


class TestEncryptionService:
    """Test encryption service."""
    
    @pytest.fixture
    def encryption_service(self):
        """Create encryption service for testing."""
        config = EncryptionConfig(
            local_encryption_enabled=True,
            enable_field_encryption=True
        )
        return EncryptionService(config)
    
    @pytest.mark.asyncio
    async def test_field_encryption_decryption(self, encryption_service):
        """Test field-level encryption and decryption."""
        customer_id = uuid4()
        original_value = "sensitive_data_123"
        field_name = "ssn"
        
        # Encrypt field
        encrypted_data = await encryption_service.encrypt_field(
            original_value, field_name, customer_id
        )
        
        assert encrypted_data.encryption_method in ["AWS_KMS", "LOCAL_FERNET"]
        assert encrypted_data.encrypted_data != original_value
        
        # Decrypt field
        decrypted_value = await encryption_service.decrypt_field(
            encrypted_data, field_name, customer_id
        )
        
        assert decrypted_value == original_value
    
    @pytest.mark.asyncio
    async def test_database_record_encryption(self, encryption_service):
        """Test database record encryption."""
        customer_id = uuid4()
        record = {
            "id": 123,
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john@example.com",
            "account_number": "ACC123456"
        }
        sensitive_fields = ["ssn", "account_number"]
        
        # Encrypt record
        encrypted_record = await encryption_service.encrypt_database_record(
            record, sensitive_fields, customer_id
        )
        
        assert encrypted_record["id"] == 123  # Non-sensitive field unchanged
        assert encrypted_record["name"] == "John Doe"  # Non-sensitive field unchanged
        assert isinstance(encrypted_record["ssn"], dict)  # Sensitive field encrypted
        assert isinstance(encrypted_record["account_number"], dict)  # Sensitive field encrypted
        
        # Decrypt record
        decrypted_record = await encryption_service.decrypt_database_record(
            encrypted_record, sensitive_fields, customer_id
        )
        
        assert decrypted_record == record
    
    @pytest.mark.asyncio
    async def test_large_data_encryption(self, encryption_service):
        """Test large data encryption with envelope encryption."""
        large_data = b"This is a large piece of data that needs envelope encryption" * 1000
        customer_id = uuid4()
        
        with patch.object(encryption_service, 'generate_data_key') as mock_generate_key:
            mock_generate_key.return_value = {
                'plaintext_key': b'0' * 32,  # 32-byte key
                'encrypted_key': 'encrypted_key_data',
                'key_id': 'test_key_id'
            }
            
            # Encrypt large data
            encrypted_info = await encryption_service.encrypt_large_data(large_data, customer_id)
            
            assert "encrypted_data" in encrypted_info
            assert "encrypted_key" in encrypted_info
            assert "iv" in encrypted_info
            assert "tag" in encrypted_info
    
    @pytest.mark.asyncio
    async def test_encryption_status(self, encryption_service):
        """Test encryption service status."""
        status = await encryption_service.get_encryption_status()
        
        assert "kms_available" in status
        assert "local_encryption_available" in status
        assert "field_encryption_enabled" in status
        assert "database_encryption_enabled" in status


class TestAuthService:
    """Test authentication service."""
    
    @pytest.fixture
    def auth_service(self):
        """Create auth service for testing."""
        config = AuthConfig(
            jwt_secret_key="test_secret_key",
            require_mfa=True,
            max_login_attempts=3
        )
        return AuthService(config)
    
    @pytest.mark.asyncio
    async def test_user_registration(self, auth_service):
        """Test user registration."""
        result = await auth_service.register_user(
            email="test@example.com",
            password="SecurePassword123!",
            first_name="John",
            last_name="Doe",
            ip_address="192.168.1.1"
        )
        
        assert "user_id" in result
        assert "password_hash" in result
        assert result["email"] == "test@example.com"
        assert result["mfa_required"] == True
    
    def test_password_strength_validation(self, auth_service):
        """Test password strength validation."""
        # Valid password
        assert auth_service._validate_password_strength("SecurePassword123!")
        
        # Too short
        assert not auth_service._validate_password_strength("Short1!")
        
        # Missing uppercase
        assert not auth_service._validate_password_strength("securepassword123!")
        
        # Missing lowercase
        assert not auth_service._validate_password_strength("SECUREPASSWORD123!")
        
        # Missing digit
        assert not auth_service._validate_password_strength("SecurePassword!")
        
        # Missing special character
        assert not auth_service._validate_password_strength("SecurePassword123")
    
    @pytest.mark.asyncio
    async def test_totp_mfa_setup(self, auth_service):
        """Test TOTP MFA setup."""
        user_id = uuid4()
        
        with patch.object(auth_service, '_get_user_by_id') as mock_get_user:
            mock_get_user.return_value = {"email": "test@example.com"}
            
            result = await auth_service.setup_totp_mfa(user_id, "My Phone")
            
            assert "device_id" in result
            assert "secret_key" in result
            assert "qr_code" in result
            assert "provisioning_uri" in result
    
    @pytest.mark.asyncio
    async def test_token_validation(self, auth_service):
        """Test JWT token validation."""
        user_id = uuid4()
        session_id = "test_session"
        
        # Generate token
        access_token = await auth_service._generate_access_token(
            user_id, session_id, mfa_verified=True
        )
        
        # Create mock session
        from src.security.auth_service import UserSession
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address="192.168.1.1",
            mfa_verified=True
        )
        auth_service._active_sessions[session_id] = session
        
        # Validate token
        payload = await auth_service.validate_token(access_token, "access")
        
        assert payload.sub == str(user_id)
        assert payload.type == "access"
        assert payload.session_id == session_id
        assert payload.mfa_verified == True
    
    @pytest.mark.asyncio
    async def test_account_lockout(self, auth_service):
        """Test account lockout after failed attempts."""
        email = "test@example.com"
        ip_address = "192.168.1.1"
        
        # Record multiple failed attempts
        for i in range(auth_service.config.max_login_attempts):
            await auth_service._record_login_attempt(uuid4(), ip_address, False)
        
        # Check if account is locked
        is_locked = await auth_service._is_account_locked(email, ip_address)
        assert is_locked


class TestComplianceService:
    """Test compliance service integration."""
    
    @pytest.fixture
    def compliance_service(self):
        """Create compliance service for testing."""
        kyc_service = Mock(spec=KYCService)
        audit_service = Mock(spec=AuditService)
        encryption_service = Mock(spec=EncryptionService)
        auth_service = Mock(spec=AuthService)
        
        # Configure mocks
        kyc_service.initiate_kyc_verification = AsyncMock()
        audit_service.log_action = AsyncMock()
        encryption_service.encrypt_database_record = AsyncMock()
        
        config = ComplianceConfig()
        return ComplianceService(
            kyc_service, audit_service, encryption_service, auth_service, config
        )
    
    @pytest.mark.asyncio
    async def test_customer_onboarding_compliance(self, compliance_service):
        """Test complete customer onboarding compliance."""
        customer_id = uuid4()
        customer_data = {
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1990-01-01",
            "country": "US"
        }
        
        # Configure mocks
        compliance_service.kyc_service.initiate_kyc_verification.return_value = {
            "verification_url": "https://test.com/verify",
            "session_id": "session_123"
        }
        compliance_service.encryption_service.encrypt_database_record.return_value = {
            "encrypted_data": "encrypted"
        }
        
        result = await compliance_service.perform_customer_onboarding_compliance(
            customer_id, customer_data, "192.168.1.1"
        )
        
        assert result["status"] == "initiated"
        assert "kyc_verification" in result
        assert "encrypted_data" in result
        assert "compliance_issues" in result
    
    @pytest.mark.asyncio
    async def test_trading_compliance_monitoring(self, compliance_service):
        """Test trading compliance monitoring."""
        customer_id = uuid4()
        trade_data = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "price": 150.00
        }
        
        result = await compliance_service.monitor_trading_compliance(
            customer_id, trade_data
        )
        
        assert result["compliance_status"] == "monitored"
        assert "suspicious_patterns" in result
        assert "risk_violations" in result
        assert "alerts_created" in result
    
    @pytest.mark.asyncio
    async def test_compliance_dashboard(self, compliance_service):
        """Test compliance dashboard data."""
        dashboard = await compliance_service.get_compliance_dashboard(period_days=30)
        
        assert "period" in dashboard
        assert "metrics" in dashboard
        assert "alert_summary" in dashboard
        assert "compliance_status" in dashboard
        assert "last_updated" in dashboard


class TestSecurityIntegration:
    """Test security framework integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_flow(self):
        """Test complete security flow from registration to trading."""
        # This would test the complete flow:
        # 1. User registration with password validation
        # 2. KYC verification initiation
        # 3. MFA setup
        # 4. Authentication with MFA
        # 5. Trading with compliance monitoring
        # 6. Audit logging throughout
        
        # Create services
        auth_config = AuthConfig(jwt_secret_key="test_key", require_mfa=True)
        auth_service = AuthService(auth_config)
        
        audit_config = AuditConfig(signing_key="test_key")
        audit_service = AuditService(audit_config)
        
        encryption_config = EncryptionConfig(local_encryption_enabled=True)
        encryption_service = EncryptionService(encryption_config)
        
        # Test user registration
        registration_result = await auth_service.register_user(
            email="test@example.com",
            password="SecurePassword123!",
            first_name="John",
            last_name="Doe",
            ip_address="192.168.1.1"
        )
        
        assert "user_id" in registration_result
        user_id = registration_result["user_id"]
        
        # Test field encryption
        encrypted_ssn = await encryption_service.encrypt_field(
            "123-45-6789", "ssn", user_id
        )
        
        decrypted_ssn = await encryption_service.decrypt_field(
            encrypted_ssn, "ssn", user_id
        )
        
        assert decrypted_ssn == "123-45-6789"
        
        # Test audit logging
        log_id = await audit_service.log_action(
            action="USER_REGISTRATION",
            resource="user",
            result=ActionResult.SUCCESS,
            user_id=user_id,
            ip_address="192.168.1.1"
        )
        
        assert log_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
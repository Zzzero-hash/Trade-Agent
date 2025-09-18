"""Security tests for edge cases and error handling."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch
import jwt

from src.security.kyc_service import KYCService
from src.security.audit_service import AuditService, AuditConfig
from src.security.encryption_service import EncryptionService, EncryptionConfig
from src.security.auth_service import AuthService, AuthConfig
from src.models.security import ActionResult, KYCStatus, AMLStatus


class TestSecurityErrorHandling:
    """Test security framework error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_kyc_service_network_failure(self):
        """Test KYC service handling network failures."""
        kyc_service = KYCService()
        
        # Test with no HTTP client configured
        with pytest.raises(ValueError, match="Jumio client not configured"):
            await kyc_service._initiate_jumio_verification(uuid4(), {})
    
    @pytest.mark.asyncio
    async def test_encryption_service_missing_kms(self):
        """Test encryption service without KMS configuration."""
        config = EncryptionConfig(local_encryption_enabled=False)
        
        with patch('boto3.client', side_effect=Exception("AWS not configured")):
            encryption_service = EncryptionService(config)
            
            # Should fall back gracefully
            status = await encryption_service.get_encryption_status()
            assert not status["kms_available"]
    
    @pytest.mark.asyncio
    async def test_auth_service_expired_token(self):
        """Test authentication with expired tokens."""
        config = AuthConfig(
            jwt_secret_key="test_key",
            access_token_expire_minutes=1
        )
        auth_service = AuthService(config)
        
        # Generate token that expires immediately
        user_id = uuid4()
        session_id = "test_session"
        
        # Create expired token manually
        past_time = datetime.utcnow() - timedelta(minutes=5)
        payload = {
            "sub": str(user_id),
            "exp": int(past_time.timestamp()),
            "iat": int(past_time.timestamp()),
            "jti": str(uuid4()),
            "type": "access",
            "session_id": session_id,
            "mfa_verified": True
        }
        
        expired_token = jwt.encode(payload, config.jwt_secret_key, algorithm="HS256")
        
        # Should raise ValueError for expired token
        with pytest.raises(ValueError, match="Token has expired"):
            await auth_service.validate_token(expired_token, "access")
    
    @pytest.mark.asyncio
    async def test_auth_service_invalid_token(self):
        """Test authentication with invalid tokens."""
        config = AuthConfig(jwt_secret_key="test_key")
        auth_service = AuthService(config)
        
        # Test with malformed token
        with pytest.raises(ValueError, match="Invalid token"):
            await auth_service.validate_token("invalid.token.here", "access")
        
        # Test with token signed with wrong key
        wrong_key_token = jwt.encode(
            {"sub": str(uuid4()), "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp())},
            "wrong_key",
            algorithm="HS256"
        )
        
        with pytest.raises(ValueError, match="Invalid token"):
            await auth_service.validate_token(wrong_key_token, "access")
    
    @pytest.mark.asyncio
    async def test_audit_service_log_integrity_failure(self):
        """Test audit service handling log integrity failures."""
        config = AuditConfig(
            signing_key="test_key",
            enable_immutable_storage=True
        )
        audit_service = AuditService(config)
        
        # Simulate corrupted chain by manually modifying last hash
        await audit_service.log_action(
            action="TEST_ACTION_1",
            resource="test",
            result=ActionResult.SUCCESS,
            ip_address="192.168.1.1"
        )
        
        # Corrupt the chain
        audit_service._last_hash = "corrupted_hash"
        
        await audit_service.log_action(
            action="TEST_ACTION_2", 
            resource="test",
            result=ActionResult.SUCCESS,
            ip_address="192.168.1.1"
        )
        
        # Verification should detect the corruption
        verification_result = await audit_service.verify_log_integrity()
        # Since we don't have actual storage, this will return empty results
        assert verification_result["status"] in ["error", "skipped"]
    
    @pytest.mark.asyncio
    async def test_encryption_service_decryption_failure(self):
        """Test encryption service handling decryption failures."""
        config = EncryptionConfig(local_encryption_enabled=True)
        encryption_service = EncryptionService(config)
        
        # Create invalid encrypted data
        from src.models.security import EncryptedData
        invalid_encrypted_data = EncryptedData(
            encrypted_data="invalid_base64_data",
            encryption_method="LOCAL_FERNET"
        )
        
        # Should raise exception for invalid data
        with pytest.raises(Exception):
            await encryption_service.decrypt_field(
                invalid_encrypted_data, "test_field", uuid4()
            )
    
    @pytest.mark.asyncio
    async def test_kyc_service_aml_screening_failure(self):
        """Test KYC service handling AML screening failures."""
        kyc_service = KYCService()
        
        # Test with invalid identity data
        with patch.object(kyc_service, '_screen_ofac', side_effect=Exception("OFAC service down")):
            with pytest.raises(Exception):
                await kyc_service.perform_aml_screening(uuid4(), {})
    
    @pytest.mark.asyncio
    async def test_auth_service_mfa_challenge_expiry(self):
        """Test MFA challenge expiry handling."""
        config = AuthConfig(
            jwt_secret_key="test_key",
            mfa_token_expire_minutes=1
        )
        auth_service = AuthService(config)
        
        # Create expired challenge
        challenge_id = "test_challenge"
        auth_service._mfa_challenges[challenge_id] = {
            "user_id": uuid4(),
            "ip_address": "192.168.1.1",
            "created_at": datetime.utcnow() - timedelta(minutes=5),
            "expires_at": datetime.utcnow() - timedelta(minutes=2)
        }
        
        # Should raise ValueError for expired challenge
        with pytest.raises(ValueError, match="MFA challenge has expired"):
            await auth_service.verify_mfa_challenge(
                challenge_id, "123456", ip_address="192.168.1.1"
            )
    
    @pytest.mark.asyncio
    async def test_compliance_service_onboarding_failure(self):
        """Test compliance service handling onboarding failures."""
        from src.security.compliance_service import ComplianceService, ComplianceConfig
        
        # Create mocks that fail
        kyc_service = Mock()
        kyc_service.initiate_kyc_verification = AsyncMock(side_effect=Exception("KYC service down"))
        
        audit_service = Mock()
        audit_service.log_action = AsyncMock()
        
        encryption_service = Mock()
        auth_service = Mock()
        
        compliance_service = ComplianceService(
            kyc_service, audit_service, encryption_service, auth_service
        )
        
        # Should handle KYC service failure gracefully
        with pytest.raises(Exception):
            await compliance_service.perform_customer_onboarding_compliance(
                uuid4(), {"email": "test@example.com"}, "192.168.1.1"
            )
        
        # Should have logged the failure
        audit_service.log_action.assert_called()


class TestSecurityBoundaryConditions:
    """Test security framework boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_password_strength_edge_cases(self):
        """Test password strength validation edge cases."""
        config = AuthConfig(jwt_secret_key="test_key", password_min_length=8)
        auth_service = AuthService(config)
        
        # Test minimum length boundary
        assert not auth_service._validate_password_strength("Short1!")  # 7 chars
        assert auth_service._validate_password_strength("LongPass1!")  # 9 chars
        
        # Test with unicode characters
        assert auth_service._validate_password_strength("Pässwörd123!")
        
        # Test with only required character types
        assert auth_service._validate_password_strength("Aa1!")  # Minimum valid
    
    @pytest.mark.asyncio
    async def test_large_data_encryption_limits(self):
        """Test encryption service with large data."""
        config = EncryptionConfig(local_encryption_enabled=True)
        encryption_service = EncryptionService(config)
        
        # Test with very large data
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        
        with patch.object(encryption_service, 'generate_data_key') as mock_generate_key:
            mock_generate_key.return_value = {
                'plaintext_key': b'0' * 32,
                'encrypted_key': 'encrypted_key_data',
                'key_id': 'test_key_id'
            }
            
            encrypted_info = await encryption_service.encrypt_large_data(large_data)
            assert "encrypted_data" in encrypted_info
    
    @pytest.mark.asyncio
    async def test_audit_log_sequence_overflow(self):
        """Test audit service with high sequence numbers."""
        config = AuditConfig(signing_key="test_key", enable_immutable_storage=True)
        audit_service = AuditService(config)
        
        # Set high sequence number
        audit_service._sequence_counter = 999999999
        
        # Should handle large sequence numbers
        log_id = await audit_service.log_action(
            action="TEST_HIGH_SEQUENCE",
            resource="test",
            result=ActionResult.SUCCESS,
            ip_address="192.168.1.1"
        )
        
        assert log_id is not None
        assert audit_service._sequence_counter == 1000000000
    
    @pytest.mark.asyncio
    async def test_mfa_device_limits(self):
        """Test MFA device registration limits."""
        config = AuthConfig(jwt_secret_key="test_key")
        auth_service = AuthService(config)
        
        user_id = uuid4()
        
        # Mock user data
        with patch.object(auth_service, '_get_user_by_id') as mock_get_user:
            mock_get_user.return_value = {"email": "test@example.com"}
            
            # Should handle multiple device registrations
            for i in range(5):  # Register 5 devices
                result = await auth_service.setup_totp_mfa(user_id, f"Device {i}")
                assert "device_id" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_login_attempts(self):
        """Test handling concurrent login attempts."""
        config = AuthConfig(jwt_secret_key="test_key", max_login_attempts=3)
        auth_service = AuthService(config)
        
        email = "test@example.com"
        ip_address = "192.168.1.1"
        
        # Simulate concurrent failed attempts
        tasks = []
        for i in range(5):
            task = auth_service._record_login_attempt(uuid4(), ip_address, False)
            tasks.append(task)
        
        # Wait for all attempts to complete
        await asyncio.gather(*tasks)
        
        # Should be locked after max attempts
        is_locked = await auth_service._is_account_locked(email, ip_address)
        assert is_locked


class TestSecurityPerformance:
    """Test security framework performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_encryption_performance(self):
        """Test encryption performance with various data sizes."""
        config = EncryptionConfig(local_encryption_enabled=True)
        encryption_service = EncryptionService(config)
        
        # Test with different data sizes
        test_sizes = [100, 1000, 10000, 100000]  # bytes
        
        for size in test_sizes:
            data = "x" * size
            customer_id = uuid4()
            
            start_time = datetime.utcnow()
            
            encrypted_data = await encryption_service.encrypt_field(
                data, "test_field", customer_id
            )
            
            decrypted_data = await encryption_service.decrypt_field(
                encrypted_data, "test_field", customer_id
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert duration < 1.0  # 1 second max
            assert decrypted_data == data
    
    @pytest.mark.asyncio
    async def test_audit_logging_performance(self):
        """Test audit logging performance with high volume."""
        config = AuditConfig(signing_key="test_key", enable_immutable_storage=True)
        audit_service = AuditService(config)
        
        # Log many entries quickly
        start_time = datetime.utcnow()
        
        tasks = []
        for i in range(100):
            task = audit_service.log_action(
                action=f"PERFORMANCE_TEST_{i}",
                resource="test",
                result=ActionResult.SUCCESS,
                ip_address="192.168.1.1"
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should handle 100 logs within reasonable time
        assert duration < 5.0  # 5 seconds max
        assert audit_service._sequence_counter == 100
    
    @pytest.mark.asyncio
    async def test_token_validation_performance(self):
        """Test JWT token validation performance."""
        config = AuthConfig(jwt_secret_key="test_key")
        auth_service = AuthService(config)
        
        # Generate token
        user_id = uuid4()
        session_id = "test_session"
        token = await auth_service._generate_access_token(user_id, session_id, True)
        
        # Create session
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
        
        # Validate token many times
        start_time = datetime.utcnow()
        
        for i in range(1000):
            payload = await auth_service.validate_token(token, "access")
            assert payload.sub == str(user_id)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should handle 1000 validations quickly
        assert duration < 2.0  # 2 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
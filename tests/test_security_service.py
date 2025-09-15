"""
Tests for security service functionality.

This module tests:
- End-to-end encryption and decryption
- Key management and rotation
- Data masking and anonymization
- Security status monitoring

Requirements: 6.5, 9.5
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from src.services.security_service import (
    SecurityService,
    DataClassification,
    EncryptedData,
    EncryptionKey,
    get_security_service
)


class TestSecurityService:
    """Test cases for SecurityService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.security_service = SecurityService()

    def test_initialization(self):
        """Test security service initialization."""
        assert self.security_service is not None
        assert len(self.security_service._key_manager._key_store) > 0
        assert len(self.security_service._key_manager._active_keys) == 4

    def test_encrypt_decrypt_string_data(self):
        """Test encryption and decryption of string data."""
        test_data = "sensitive trading signal data"

        # Encrypt data
        encrypted = self.security_service.encrypt_data(
            test_data,
            classification=DataClassification.CONFIDENTIAL
        )

        assert isinstance(encrypted, EncryptedData)
        assert encrypted.data != test_data.encode('utf-8')
        assert encrypted.key_id is not None
        assert encrypted.algorithm.startswith("Fernet")

        # Decrypt data
        decrypted = self.security_service.decrypt_data(encrypted)
        assert decrypted.decode('utf-8') == test_data

    def test_encrypt_decrypt_dict_data(self):
        """Test encryption and decryption of dictionary data."""
        test_data = {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "price": 150.25,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Encrypt data
        encrypted = self.security_service.encrypt_data(
            test_data,
            classification=DataClassification.RESTRICTED
        )
        
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.key_id is not None
        
        # Decrypt data
        decrypted = self.security_service.decrypt_data(encrypted)
        decrypted_dict = json.loads(decrypted.decode('utf-8'))
        
        assert decrypted_dict == test_data

    def test_encrypt_trading_signal(self):
        """Test encryption of trading signal data."""
        signal_data = {
            "signal_id": "sig_123",
            "symbol": "TSLA",
            "action": "SELL",
            "confidence": 0.85,
            "model_version": "v2.1"
        }
        
        encrypted = self.security_service.encrypt_trading_signal(signal_data)
        
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.key_id is not None
        
        # Verify it uses restricted classification
        key_info = self.security_service._key_store[encrypted.key_id]
        assert key_info.classification == DataClassification.RESTRICTED

    def test_encrypt_user_credentials(self):
        """Test encryption of user credentials."""
        credentials = {
            "username": "trader123",
            "api_key": "ak_test_12345",
            "api_secret": "secret_key_67890",
            "exchange": "robinhood"
        }
        
        encrypted = self.security_service.encrypt_user_credentials(credentials)
        
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.key_id is not None
        
        # Verify it uses restricted classification
        key_info = self.security_service._key_store[encrypted.key_id]
        assert key_info.classification == DataClassification.RESTRICTED

    def test_encrypt_portfolio_data(self):
        """Test encryption of portfolio data."""
        portfolio_data = {
            "user_id": "user_456",
            "total_value": 50000.00,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "value": 15000},
                {"symbol": "GOOGL", "quantity": 50, "value": 35000}
            ],
            "cash_balance": 5000.00
        }
        
        encrypted = self.security_service.encrypt_portfolio_data(portfolio_data)
        
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.key_id is not None
        
        # Verify it uses confidential classification
        key_info = self.security_service._key_store[encrypted.key_id]
        assert key_info.classification == DataClassification.CONFIDENTIAL

    def test_mask_sensitive_data(self):
        """Test data masking functionality."""
        sensitive_data = {
            "username": "trader123",
            "password": "secret123",
            "api_key": "ak_test_12345",
            "api_secret": "very_secret_key",
            "email": "trader@example.com",
            "phone": "555-123-4567",
            "symbol": "AAPL",  # Not sensitive
            "quantity": 100    # Not sensitive
        }
        
        masked = self.security_service.mask_sensitive_data(sensitive_data)
        
        # Check sensitive fields are masked
        assert masked["password"] == "se***23"
        assert masked["api_key"] == "ak***45"
        assert masked["api_secret"] == "ve***ey"
        assert masked["email"] == "tr***om"
        assert masked["phone"] == "55***67"
        
        # Check non-sensitive fields are preserved
        assert masked["symbol"] == "AAPL"
        assert masked["quantity"] == 100

    def test_mask_short_sensitive_data(self):
        """Test masking of short sensitive data."""
        data = {
            "password": "123",  # Short password
            "token": "ab"       # Very short token
        }
        
        masked = self.security_service.mask_sensitive_data(data)
        
        assert masked["password"] == "***"
        assert masked["token"] == "***"

    def test_generate_secure_token(self):
        """Test secure token generation."""
        token1 = self.security_service.generate_secure_token()
        token2 = self.security_service.generate_secure_token()
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Should be unique

    def test_generate_secure_token_custom_length(self):
        """Test secure token generation with custom length."""
        token = self.security_service.generate_secure_token(length=16)
        
        # URL-safe base64 encoding adds padding, so length may vary
        assert len(token) > 0

    def test_hash_data_string(self):
        """Test data hashing with string input."""
        test_data = "password123"
        
        hash_info = self.security_service.hash_data(test_data)
        
        assert 'hash' in hash_info
        assert 'salt' in hash_info
        assert 'algorithm' in hash_info
        assert 'iterations' in hash_info
        assert hash_info['algorithm'] == 'PBKDF2-SHA256'
        assert hash_info['iterations'] == 100000

    def test_hash_data_bytes(self):
        """Test data hashing with bytes input."""
        test_data = b"password123"
        
        hash_info = self.security_service.hash_data(test_data)
        
        assert 'hash' in hash_info
        assert 'salt' in hash_info
        assert hash_info['algorithm'] == 'PBKDF2-SHA256'

    def test_hash_data_with_salt(self):
        """Test data hashing with provided salt."""
        test_data = "password123"
        salt = b"fixed_salt_for_testing_12345678901"
        
        hash_info1 = self.security_service.hash_data(test_data, salt)
        hash_info2 = self.security_service.hash_data(test_data, salt)
        
        # Same data and salt should produce same hash
        assert hash_info1['hash'] == hash_info2['hash']
        assert hash_info1['salt'] == hash_info2['salt']

    def test_verify_hash_correct(self):
        """Test hash verification with correct data."""
        test_data = "password123"
        
        hash_info = self.security_service.hash_data(test_data)
        is_valid = self.security_service.verify_hash(test_data, hash_info)
        
        assert is_valid is True

    def test_verify_hash_incorrect(self):
        """Test hash verification with incorrect data."""
        test_data = "password123"
        wrong_data = "password456"
        
        hash_info = self.security_service.hash_data(test_data)
        is_valid = self.security_service.verify_hash(wrong_data, hash_info)
        
        assert is_valid is False

    def test_key_rotation_single_classification(self):
        """Test key rotation for single classification."""
        classification = DataClassification.CONFIDENTIAL
        old_key_id = self.security_service._active_keys[classification]
        
        rotated = self.security_service.rotate_keys(classification)
        
        assert old_key_id in rotated
        new_key_id = rotated[old_key_id]
        assert new_key_id != old_key_id
        assert self.security_service._active_keys[classification] == new_key_id
        
        # Old key should be marked inactive
        old_key = self.security_service._key_store[old_key_id]
        assert old_key.is_active is False

    def test_key_rotation_all_classifications(self):
        """Test key rotation for all classifications."""
        old_keys = self.security_service._active_keys.copy()
        
        rotated = self.security_service.rotate_keys()
        
        assert len(rotated) == len(DataClassification)
        
        for classification in DataClassification:
            old_key_id = old_keys[classification]
            assert old_key_id in rotated
            new_key_id = rotated[old_key_id]
            assert new_key_id != old_key_id
            assert self.security_service._active_keys[classification] == new_key_id

    def test_get_security_status(self):
        """Test security status retrieval."""
        status = self.security_service.get_security_status()
        
        assert 'total_keys' in status
        assert 'active_keys' in status
        assert 'keys_by_classification' in status
        assert 'keys_expiring_soon' in status
        assert 'security_level' in status
        
        assert status['total_keys'] > 0
        assert status['active_keys'] > 0
        assert status['security_level'] == 'HIGH'

    def test_encryption_with_invalid_key_id(self):
        """Test encryption with invalid key ID."""
        test_data = "test data"
        
        with pytest.raises(ValueError, match="Key not found"):
            self.security_service.encrypt_data(
                test_data,
                key_id="invalid_key_id"
            )

    def test_decryption_with_invalid_key_id(self):
        """Test decryption with invalid key ID."""
        encrypted_data = EncryptedData(
            data=b"fake_encrypted_data",
            key_id="invalid_key_id",
            algorithm="Fernet-AES256"
        )
        
        with pytest.raises(ValueError, match="Key not found"):
            self.security_service.decrypt_data(encrypted_data)

    def test_encryption_no_active_key(self):
        """Test encryption when no active key exists for classification."""
        # Clear active keys
        self.security_service._active_keys.clear()
        
        with pytest.raises(ValueError, match="No active key for classification"):
            self.security_service.encrypt_data(
                "test data",
                classification=DataClassification.CONFIDENTIAL
            )

    @patch('src.services.security_service.datetime')
    def test_keys_expiring_soon(self, mock_datetime):
        """Test detection of keys expiring soon."""
        # Mock current time
        current_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = current_time
        
        # Create a key that expires in 3 days
        expiring_key = EncryptionKey(
            key_id="expiring_key",
            algorithm="Fernet-AES256",
            created_at=current_time - timedelta(days=87),
            expires_at=current_time + timedelta(days=3),
            classification=DataClassification.CONFIDENTIAL,
            is_active=True
        )
        
        self.security_service._key_store["expiring_key"] = expiring_key
        
        status = self.security_service.get_security_status()
        
        assert len(status['keys_expiring_soon']) > 0
        expiring_key_info = next(
            (k for k in status['keys_expiring_soon'] if k['key_id'] == "expiring_key"),
            None
        )
        assert expiring_key_info is not None


class TestGlobalSecurityService:
    """Test global security service instance."""

    def test_get_security_service_singleton(self):
        """Test that get_security_service returns singleton instance."""
        service1 = get_security_service()
        service2 = get_security_service()
        
        assert service1 is service2
        assert isinstance(service1, SecurityService)

    def test_get_security_service_initialization(self):
        """Test that global service is properly initialized."""
        service = get_security_service()
        
        assert len(service._key_store) > 0
        assert len(service._active_keys) == 4


class TestEncryptedData:
    """Test EncryptedData dataclass."""

    def test_encrypted_data_creation(self):
        """Test EncryptedData creation."""
        encrypted = EncryptedData(
            data=b"encrypted_bytes",
            key_id="test_key",
            algorithm="Fernet-AES256"
        )
        
        assert encrypted.data == b"encrypted_bytes"
        assert encrypted.key_id == "test_key"
        assert encrypted.algorithm == "Fernet-AES256"
        assert isinstance(encrypted.timestamp, datetime)

    def test_encrypted_data_with_optional_fields(self):
        """Test EncryptedData with optional fields."""
        encrypted = EncryptedData(
            data=b"encrypted_bytes",
            key_id="test_key",
            algorithm="Fernet-AES256",
            iv=b"initialization_vector",
            tag=b"auth_tag"
        )
        
        assert encrypted.iv == b"initialization_vector"
        assert encrypted.tag == b"auth_tag"


class TestEncryptionKey:
    """Test EncryptionKey dataclass."""

    def test_encryption_key_creation(self):
        """Test EncryptionKey creation."""
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(days=90)
        
        key = EncryptionKey(
            key_id="test_key_123",
            algorithm="Fernet-AES256",
            created_at=created_at,
            expires_at=expires_at,
            classification=DataClassification.CONFIDENTIAL
        )
        
        assert key.key_id == "test_key_123"
        assert key.algorithm == "Fernet-AES256"
        assert key.created_at == created_at
        assert key.expires_at == expires_at
        assert key.classification == DataClassification.CONFIDENTIAL
        assert key.is_active is True

    def test_encryption_key_defaults(self):
        """Test EncryptionKey default values."""
        key = EncryptionKey(
            key_id="test_key",
            algorithm="Fernet-AES256",
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            classification=DataClassification.PUBLIC
        )
        
        assert key.is_active is True
        assert key.expires_at is None


if __name__ == "__main__":
    pytest.main([__file__])
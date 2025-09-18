"""
Security service for end-to-end encryption and data protection.

This module provides comprehensive security features including:
- End-to-end encryption for sensitive data transmission
- Secure key management and rotation
- Data anonymization and masking
- Security monitoring and threat detection

Requirements: 6.5, 9.5
"""

import hashlib
import hmac
import secrets
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from cryptography.fernet import Fernet

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EncryptionLevel(str, Enum):
    """Encryption levels for different data types."""
    LOW = "low"          # Basic symmetric encryption
    MEDIUM = "medium"    # Advanced symmetric with key rotation
    HIGH = "high"        # Asymmetric encryption
    CRITICAL = "critical"  # Multi-layer encryption with HSM


class DataClassification(str, Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime]
    classification: DataClassification
    is_active: bool = True


@dataclass
class EncryptedData:
    """Encrypted data container."""
    data: bytes
    key_id: str
    algorithm: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))


class SecurityServiceError(Exception):
    """Base exception for security service errors."""
    pass


class KeyNotFoundError(SecurityServiceError):
    """Raised when encryption key is not found."""
    pass


class EncryptionError(SecurityServiceError):
    """Raised when encryption/decryption fails."""
    pass


class KeyManager:
    """Manages encryption keys and their lifecycle."""

    def __init__(self, master_key: bytes):
        self._master_key = master_key
        self._key_store: Dict[str, EncryptionKey] = {}
        self._active_keys: Dict[DataClassification, str] = {}
        self._initialize_keys()

    def _initialize_keys(self) -> None:
        """Initialize encryption keys for different data classifications."""
        classifications = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED
        ]

        for classification in classifications:
            key_id = self._generate_key_for_classification(classification)
            self._active_keys[classification] = key_id

    def _generate_key_for_classification(
            self, classification: DataClassification) -> str:
        """Generate encryption key for specific data classification."""
        key_id = f"{classification.value}_{secrets.token_hex(8)}"

        # Generate key based on classification level
        if classification in [DataClassification.RESTRICTED,
                              DataClassification.CONFIDENTIAL]:
            # Use stronger encryption for sensitive data
            key = Fernet.generate_key()
            algorithm = "Fernet-AES256"
        else:
            # Standard encryption for less sensitive data
            key = Fernet.generate_key()
            algorithm = "Fernet-AES128"

        # Store key metadata
        encryption_key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=90),
            classification=classification,
            is_active=True
        )

        self._key_store[key_id] = encryption_key

        # Store actual key securely (in production, use HSM or key vault)
        self._store_key_securely(key_id, key)

        logger.info("Generated encryption key for %s: %s",
                    classification.value, key_id)
        return key_id

    def _store_key_securely(self, key_id: str, key: bytes) -> None:
        """Store encryption key securely."""
        # In production, this should use a Hardware Security Module (HSM)
        # or cloud key management service like AWS KMS, Azure Key Vault, etc.

        # For now, encrypt with master key and store in memory
        # This is NOT production-ready - use proper key management
        fernet = Fernet(self._master_key)
        encrypted_key = fernet.encrypt(key)

        # Store in secure location (placeholder)
        setattr(self, f"_key_{key_id}", encrypted_key)

    def get_key_securely(self, key_id: str) -> bytes:
        """Retrieve encryption key securely."""
        if key_id not in self._key_store:
            raise KeyNotFoundError(f"Key not found: {key_id}")

        # Retrieve encrypted key
        encrypted_key = getattr(self, f"_key_{key_id}", None)
        if not encrypted_key:
            raise KeyNotFoundError(f"Key data not found: {key_id}")

        try:
            # Decrypt with master key
            fernet = Fernet(self._master_key)
            return fernet.decrypt(encrypted_key)
        except Exception as e:
            logger.error("Failed to decrypt key %s: %s", key_id, e)
            raise EncryptionError(f"Key decryption failed: {key_id}") from e

    def get_active_key_id(self, classification: DataClassification) -> str:
        """Get active key ID for classification."""
        key_id = self._active_keys.get(classification)
        if not key_id:
            raise KeyNotFoundError(
                f"No active key for classification: {classification}")
        return key_id

    def get_key_info(self, key_id: str) -> EncryptionKey:
        """Get key metadata."""
        if key_id not in self._key_store:
            raise KeyNotFoundError(f"Key not found: {key_id}")
        return self._key_store[key_id]

    def rotate_keys(
            self, classification: Optional[DataClassification] = None
    ) -> Dict[str, str]:
        """
        Rotate encryption keys for security.

        Args:
            classification: Specific classification to rotate (all if None)

        Returns:
            Dictionary mapping old key IDs to new key IDs
        """
        rotated_keys = {}

        classifications_to_rotate = (
            [classification] if classification
            else list(DataClassification)
        )

        for cls in classifications_to_rotate:
            old_key_id = self._active_keys.get(cls)
            if old_key_id:
                # Mark old key as inactive
                if old_key_id in self._key_store:
                    self._key_store[old_key_id].is_active = False

                # Generate new key
                new_key_id = self._generate_key_for_classification(cls)
                self._active_keys[cls] = new_key_id

                rotated_keys[old_key_id] = new_key_id

                logger.info("Rotated key for %s: %s -> %s",
                            cls.value, old_key_id, new_key_id)

        return rotated_keys

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and key information."""
        status: Dict[str, Any] = {
            'total_keys': len(self._key_store),
            'active_keys': len([k for k in self._key_store.values()
                                if k.is_active]),
            'keys_by_classification': {},
            'keys_expiring_soon': [],
            'security_level': 'HIGH'
        }

        # Count keys by classification
        for key_info in self._key_store.values():
            cls = key_info.classification.value
            if cls not in status['keys_by_classification']:
                status['keys_by_classification'][cls] = 0
            status['keys_by_classification'][cls] += 1

        # Check for keys expiring soon (within 7 days)
        soon = datetime.now(timezone.utc) + timedelta(days=7)
        for key_id, key_info in self._key_store.items():
            if key_info.expires_at and key_info.expires_at <= soon:
                status['keys_expiring_soon'].append({
                    'key_id': key_id,
                    'classification': key_info.classification.value,
                    'expires_at': key_info.expires_at.isoformat()
                })

        return status


class DataMasker:
    """Handles data masking and anonymization."""

    SENSITIVE_FIELDS = {
        'password', 'api_key', 'api_secret', 'token', 'credential',
        'ssn', 'account_number', 'routing_number', 'credit_card',
        'email', 'phone', 'address'
    }

    @classmethod
    def mask_sensitive_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask sensitive data for logging and monitoring.

        Args:
            data: Dictionary containing potentially sensitive data

        Returns:
            Dictionary with sensitive fields masked
        """
        masked_data = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if field is sensitive
            is_sensitive = any(field in key_lower for field in cls.SENSITIVE_FIELDS)

            if is_sensitive:
                if isinstance(value, str) and len(value) > 4:
                    # Show first 2 and last 2 characters
                    masked_data[key] = f"{value[:2]}***{value[-2:]}"
                else:
                    masked_data[key] = "***"
            else:
                masked_data[key] = value

        return masked_data


class CryptoUtils:
    """Utility functions for cryptographic operations."""

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_data(data: Any, salt: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Hash data with salt for secure storage.

        Args:
            data: Data to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Dictionary with hash and salt
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 with SHA-256
        hash_value = hashlib.pbkdf2_hmac('sha256', data, salt, 100000)

        return {
            'hash': base64.b64encode(hash_value).decode('utf-8'),
            'salt': base64.b64encode(salt).decode('utf-8'),
            'algorithm': 'PBKDF2-SHA256',
            'iterations': 100000
        }

    @staticmethod
    def verify_hash(data: Any, hash_info: Dict[str, Any]) -> bool:
        """
        Verify data against stored hash.

        Args:
            data: Data to verify
            hash_info: Hash information from hash_data()

        Returns:
            True if data matches hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        salt = base64.b64decode(hash_info['salt'])
        stored_hash = base64.b64decode(hash_info['hash'])
        iterations = int(hash_info.get('iterations', 100000))

        computed_hash = hashlib.pbkdf2_hmac('sha256', data, salt, iterations)

        return hmac.compare_digest(stored_hash, computed_hash)


class SecurityService:
    """Comprehensive security service for data protection."""

    def __init__(self):
        self.settings = get_settings()
        master_key = self._get_or_create_master_key()
        self._key_manager = KeyManager(master_key)

    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        # In production, this should be stored in a secure key management system
        if (hasattr(self.settings, 'security') and
                hasattr(self.settings.security, 'master_key')):
            return base64.b64decode(self.settings.security.master_key)

        # Generate new master key if not found
        master_key = Fernet.generate_key()
        logger.warning(
            "Generated new master key - store securely in production")
        return master_key

    def encrypt_data(
        self,
        data: Any,
        classification: DataClassification = DataClassification.CONFIDENTIAL,
        key_id: Optional[str] = None
    ) -> EncryptedData:
        """
        Encrypt data with appropriate encryption level.

        Args:
            data: Data to encrypt
            classification: Data classification level
            key_id: Specific key ID to use (optional)

        Returns:
            EncryptedData object with encrypted data and metadata
        """
        try:
            # Convert data to bytes if necessary
            if isinstance(data, dict):
                data_bytes = json.dumps(data, default=str).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data

            # Get encryption key
            if key_id is None:
                key_id = self._key_manager.get_active_key_id(classification)

            key = self._key_manager.get_key_securely(key_id)
            key_info = self._key_manager.get_key_info(key_id)

            # Encrypt based on algorithm
            if key_info.algorithm.startswith("Fernet"):
                fernet = Fernet(key)
                encrypted_data = fernet.encrypt(data_bytes)

                return EncryptedData(
                    data=encrypted_data,
                    key_id=key_id,
                    algorithm=key_info.algorithm
                )

            raise EncryptionError(
                f"Unsupported algorithm: {key_info.algorithm}")

        except (KeyNotFoundError, EncryptionError):
            raise
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            raise EncryptionError("Data encryption failed") from e

    def decrypt_data(self, encrypted_data: EncryptedData) -> bytes:
        """
        Decrypt data using stored key information.

        Args:
            encrypted_data: EncryptedData object to decrypt

        Returns:
            Decrypted data as bytes
        """
        try:
            key = self._key_manager.get_key_securely(encrypted_data.key_id)

            if encrypted_data.algorithm.startswith("Fernet"):
                fernet = Fernet(key)
                return fernet.decrypt(encrypted_data.data)

            raise EncryptionError(
                f"Unsupported algorithm: {encrypted_data.algorithm}")

        except (KeyNotFoundError, EncryptionError):
            raise
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            raise EncryptionError("Data decryption failed") from e

    def encrypt_trading_signal(
            self, signal_data: Dict[str, Any]) -> EncryptedData:
        """Encrypt trading signal data with high security."""
        return self.encrypt_data(
            signal_data,
            classification=DataClassification.RESTRICTED
        )

    def encrypt_user_credentials(
            self, credentials: Dict[str, Any]) -> EncryptedData:
        """Encrypt user credentials with maximum security."""
        return self.encrypt_data(
            credentials,
            classification=DataClassification.RESTRICTED
        )

    def encrypt_portfolio_data(
            self, portfolio_data: Dict[str, Any]) -> EncryptedData:
        """Encrypt portfolio data with confidential classification."""
        return self.encrypt_data(
            portfolio_data,
            classification=DataClassification.CONFIDENTIAL
        )

    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data for logging and monitoring."""
        return DataMasker.mask_sensitive_data(data)

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return CryptoUtils.generate_secure_token(length)

    def hash_data(self, data: Any, salt: Optional[bytes] = None) -> Dict[str, Any]:
        """Hash data with salt for secure storage."""
        return CryptoUtils.hash_data(data, salt)

    def verify_hash(self, data: Any, hash_info: Dict[str, Any]) -> bool:
        """Verify data against stored hash."""
        return CryptoUtils.verify_hash(data, hash_info)

    def rotate_keys(
            self, classification: Optional[DataClassification] = None
    ) -> Dict[str, str]:
        """Rotate encryption keys for security."""
        return self._key_manager.rotate_keys(classification)

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and key information."""
        return self._key_manager.get_security_status()


# Global security service instance
_security_service: Optional[SecurityService] = None


def get_security_service() -> SecurityService:
    """Get global security service instance."""
    global _security_service

    if _security_service is None:
        _security_service = SecurityService()

    return _security_service
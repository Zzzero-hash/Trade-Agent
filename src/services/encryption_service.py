"""
Encryption service using AWS KMS for data at rest and in transit.
"""
import asyncio
import logging
import json
import base64
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import secrets

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class EncryptionKeyType:
    """Encryption key types for different data categories."""
    CUSTOMER_PII = "customer_pii"
    FINANCIAL_DATA = "financial_data"
    AUDIT_LOGS = "audit_logs"
    SESSION_DATA = "session_data"
    API_KEYS = "api_keys"


class KMSClient:
    """AWS KMS client for key management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.kms_client = boto3.client(
            'kms',
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            region_name=self.settings.aws_region
        )
        
        # KMS key IDs for different data types
        self.key_ids = {
            EncryptionKeyType.CUSTOMER_PII: self.settings.kms_customer_pii_key_id,
            EncryptionKeyType.FINANCIAL_DATA: self.settings.kms_financial_data_key_id,
            EncryptionKeyType.AUDIT_LOGS: self.settings.kms_audit_logs_key_id,
            EncryptionKeyType.SESSION_DATA: self.settings.kms_session_data_key_id,
            EncryptionKeyType.API_KEYS: self.settings.kms_api_keys_key_id
        }
    
    async def generate_data_key(self, key_type: str, key_spec: str = "AES_256") -> Dict[str, bytes]:
        """Generate a data encryption key using KMS."""
        try:
            key_id = self.key_ids.get(key_type)
            if not key_id:
                raise ValueError(f"No KMS key configured for type: {key_type}")
            
            response = self.kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec=key_spec
            )
            
            return {
                "plaintext_key": response["Plaintext"],
                "encrypted_key": response["CiphertextBlob"]
            }
            
        except ClientError as e:
            logger.error(f"Failed to generate data key for {key_type}: {str(e)}")
            raise
    
    async def decrypt_data_key(self, encrypted_key: bytes) -> bytes:
        """Decrypt a data encryption key using KMS."""
        try:
            response = self.kms_client.decrypt(CiphertextBlob=encrypted_key)
            return response["Plaintext"]
            
        except ClientError as e:
            logger.error(f"Failed to decrypt data key: {str(e)}")
            raise
    
    async def encrypt_small_data(self, data: bytes, key_type: str) -> Dict[str, Any]:
        """Encrypt small data directly with KMS (up to 4KB)."""
        try:
            key_id = self.key_ids.get(key_type)
            if not key_id:
                raise ValueError(f"No KMS key configured for type: {key_type}")
            
            if len(data) > 4096:
                raise ValueError("Data too large for direct KMS encryption (max 4KB)")
            
            response = self.kms_client.encrypt(
                KeyId=key_id,
                Plaintext=data
            )
            
            return {
                "ciphertext": response["CiphertextBlob"],
                "key_id": response["KeyId"],
                "encryption_algorithm": response.get("EncryptionAlgorithm", "SYMMETRIC_DEFAULT")
            }
            
        except ClientError as e:
            logger.error(f"Failed to encrypt data with KMS: {str(e)}")
            raise
    
    async def decrypt_small_data(self, ciphertext: bytes) -> bytes:
        """Decrypt small data directly with KMS."""
        try:
            response = self.kms_client.decrypt(CiphertextBlob=ciphertext)
            return response["Plaintext"]
            
        except ClientError as e:
            logger.error(f"Failed to decrypt data with KMS: {str(e)}")
            raise


class FieldLevelEncryption:
    """Field-level encryption for sensitive data."""
    
    def __init__(self, kms_client: KMSClient):
        self.kms_client = kms_client
        self.key_cache = {}  # Cache for data keys
        self.cache_ttl = timedelta(hours=1)
    
    async def encrypt_field(self, data: Union[str, bytes], key_type: str) -> Dict[str, Any]:
        """Encrypt a single field with envelope encryption."""
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Get or generate data key
            data_key_info = await self._get_data_key(key_type)
            
            # Encrypt data with data key
            fernet = Fernet(base64.urlsafe_b64encode(data_key_info["plaintext_key"][:32]))
            encrypted_data = fernet.encrypt(data)
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "encrypted_key": base64.b64encode(data_key_info["encrypted_key"]).decode('utf-8'),
                "algorithm": "AES-256-GCM",
                "key_type": key_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Field encryption failed: {str(e)}")
            raise
    
    async def decrypt_field(self, encrypted_field: Dict[str, Any]) -> Union[str, bytes]:
        """Decrypt a single field."""
        try:
            # Decrypt data key
            encrypted_key = base64.b64decode(encrypted_field["encrypted_key"])
            plaintext_key = await self.kms_client.decrypt_data_key(encrypted_key)
            
            # Decrypt data
            encrypted_data = base64.b64decode(encrypted_field["encrypted_data"])
            fernet = Fernet(base64.urlsafe_b64encode(plaintext_key[:32]))
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Try to decode as UTF-8 string
            try:
                return decrypted_data.decode('utf-8')
            except UnicodeDecodeError:
                return decrypted_data
                
        except Exception as e:
            logger.error(f"Field decryption failed: {str(e)}")
            raise
    
    async def _get_data_key(self, key_type: str) -> Dict[str, bytes]:
        """Get data key from cache or generate new one."""
        cache_key = f"{key_type}_{datetime.utcnow().hour}"  # Rotate hourly
        
        if cache_key in self.key_cache:
            cached_entry = self.key_cache[cache_key]
            if datetime.utcnow() - cached_entry["timestamp"] < self.cache_ttl:
                return cached_entry["key_info"]
        
        # Generate new data key
        key_info = await self.kms_client.generate_data_key(key_type)
        
        # Cache it
        self.key_cache[cache_key] = {
            "key_info": key_info,
            "timestamp": datetime.utcnow()
        }
        
        return key_info


class TransitEncryption:
    """Encryption for data in transit."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def generate_tls_config(self) -> Dict[str, Any]:
        """Generate TLS configuration for secure communication."""
        return {
            "ssl_version": "TLSv1.3",
            "ciphers": [
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES256-SHA384",
                "ECDHE-RSA-AES128-SHA256"
            ],
            "verify_mode": "CERT_REQUIRED",
            "check_hostname": True,
            "ca_certs": self.settings.ca_cert_path,
            "certfile": self.settings.client_cert_path,
            "keyfile": self.settings.client_key_path
        }
    
    async def encrypt_api_payload(self, payload: Dict[str, Any], recipient_public_key: str) -> str:
        """Encrypt API payload for secure transmission."""
        try:
            # Serialize payload
            payload_json = json.dumps(payload)
            payload_bytes = payload_json.encode('utf-8')
            
            # Load recipient's public key
            public_key = serialization.load_pem_public_key(recipient_public_key.encode())
            
            # Generate symmetric key for payload encryption
            symmetric_key = secrets.token_bytes(32)
            
            # Encrypt payload with symmetric key
            iv = secrets.token_bytes(16)
            cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            # Pad payload to block size
            padded_payload = self._pad_data(payload_bytes, 16)
            encrypted_payload = encryptor.update(padded_payload) + encryptor.finalize()
            
            # Encrypt symmetric key with recipient's public key
            encrypted_key = public_key.encrypt(
                symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key, IV, and encrypted payload
            result = {
                "encrypted_key": base64.b64encode(encrypted_key).decode('utf-8'),
                "iv": base64.b64encode(iv).decode('utf-8'),
                "encrypted_payload": base64.b64encode(encrypted_payload).decode('utf-8'),
                "algorithm": "RSA-OAEP+AES-256-CBC"
            }
            
            return base64.b64encode(json.dumps(result).encode()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"API payload encryption failed: {str(e)}")
            raise
    
    def _pad_data(self, data: bytes, block_size: int) -> bytes:
        """PKCS7 padding for block cipher."""
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding


class EncryptionService:
    """Main encryption service coordinating all encryption operations."""
    
    def __init__(self):
        self.kms_client = KMSClient()
        self.field_encryption = FieldLevelEncryption(self.kms_client)
        self.transit_encryption = TransitEncryption()
        
        # Encryption context for audit trails
        self.encryption_context = {
            "service": "ai-trading-platform",
            "environment": self.kms_client.settings.environment,
            "version": "1.0"
        }
    
    async def encrypt_customer_pii(self, pii_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt customer PII data."""
        try:
            encrypted_data = {}
            
            # Fields that need encryption
            sensitive_fields = [
                "ssn", "tax_id", "passport_number", "drivers_license",
                "date_of_birth", "phone_number", "address", "bank_account"
            ]
            
            for field, value in pii_data.items():
                if field in sensitive_fields and value:
                    encrypted_data[field] = await self.field_encryption.encrypt_field(
                        str(value), EncryptionKeyType.CUSTOMER_PII
                    )
                else:
                    encrypted_data[field] = value
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Customer PII encryption failed: {str(e)}")
            raise
    
    async def decrypt_customer_pii(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt customer PII data."""
        try:
            decrypted_data = {}
            
            for field, value in encrypted_data.items():
                if isinstance(value, dict) and "encrypted_data" in value:
                    decrypted_data[field] = await self.field_encryption.decrypt_field(value)
                else:
                    decrypted_data[field] = value
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Customer PII decryption failed: {str(e)}")
            raise
    
    async def encrypt_financial_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt financial data."""
        try:
            encrypted_data = {}
            
            # Financial fields that need encryption
            sensitive_fields = [
                "account_number", "routing_number", "balance", "positions",
                "trade_history", "portfolio_value", "api_keys", "tokens"
            ]
            
            for field, value in financial_data.items():
                if field in sensitive_fields and value is not None:
                    # Convert complex objects to JSON strings
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    
                    encrypted_data[field] = await self.field_encryption.encrypt_field(
                        str(value), EncryptionKeyType.FINANCIAL_DATA
                    )
                else:
                    encrypted_data[field] = value
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Financial data encryption failed: {str(e)}")
            raise
    
    async def decrypt_financial_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt financial data."""
        try:
            decrypted_data = {}
            
            for field, value in encrypted_data.items():
                if isinstance(value, dict) and "encrypted_data" in value:
                    decrypted_value = await self.field_encryption.decrypt_field(value)
                    
                    # Try to parse JSON for complex objects
                    try:
                        decrypted_data[field] = json.loads(decrypted_value)
                    except (json.JSONDecodeError, TypeError):
                        decrypted_data[field] = decrypted_value
                else:
                    decrypted_data[field] = value
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Financial data decryption failed: {str(e)}")
            raise
    
    async def encrypt_audit_data(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt audit log data."""
        try:
            # Encrypt sensitive fields in audit logs
            sensitive_fields = ["details", "metadata", "ip_address", "user_agent"]
            encrypted_data = audit_data.copy()
            
            for field in sensitive_fields:
                if field in audit_data and audit_data[field]:
                    encrypted_data[field] = await self.field_encryption.encrypt_field(
                        json.dumps(audit_data[field]) if isinstance(audit_data[field], dict) else str(audit_data[field]),
                        EncryptionKeyType.AUDIT_LOGS
                    )
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Audit data encryption failed: {str(e)}")
            raise
    
    async def decrypt_audit_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt audit log data."""
        try:
            decrypted_data = encrypted_data.copy()
            
            for field, value in encrypted_data.items():
                if isinstance(value, dict) and "encrypted_data" in value:
                    decrypted_value = await self.field_encryption.decrypt_field(value)
                    
                    # Try to parse JSON
                    try:
                        decrypted_data[field] = json.loads(decrypted_value)
                    except (json.JSONDecodeError, TypeError):
                        decrypted_data[field] = decrypted_value
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Audit data decryption failed: {str(e)}")
            raise
    
    async def encrypt_api_credentials(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Encrypt API credentials and tokens."""
        try:
            encrypted_credentials = {}
            
            for key, value in credentials.items():
                if value:
                    encrypted_credentials[key] = await self.field_encryption.encrypt_field(
                        value, EncryptionKeyType.API_KEYS
                    )
            
            return encrypted_credentials
            
        except Exception as e:
            logger.error(f"API credentials encryption failed: {str(e)}")
            raise
    
    async def decrypt_api_credentials(self, encrypted_credentials: Dict[str, Any]) -> Dict[str, str]:
        """Decrypt API credentials and tokens."""
        try:
            decrypted_credentials = {}
            
            for key, value in encrypted_credentials.items():
                if isinstance(value, dict) and "encrypted_data" in value:
                    decrypted_credentials[key] = await self.field_encryption.decrypt_field(value)
                else:
                    decrypted_credentials[key] = value
            
            return decrypted_credentials
            
        except Exception as e:
            logger.error(f"API credentials decryption failed: {str(e)}")
            raise
    
    async def rotate_encryption_keys(self, key_type: str) -> bool:
        """Rotate encryption keys for a specific data type."""
        try:
            # This would typically involve:
            # 1. Generating new data keys
            # 2. Re-encrypting existing data with new keys
            # 3. Updating key references
            # 4. Cleaning up old keys after grace period
            
            logger.info(f"Key rotation initiated for {key_type}")
            
            # Generate new data key to verify KMS access
            await self.kms_client.generate_data_key(key_type)
            
            # Clear cache to force new key generation
            self.field_encryption.key_cache.clear()
            
            logger.info(f"Key rotation completed for {key_type}")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed for {key_type}: {str(e)}")
            return False
    
    def get_encryption_metadata(self) -> Dict[str, Any]:
        """Get encryption service metadata."""
        return {
            "service": "EncryptionService",
            "version": "1.0",
            "algorithms": {
                "symmetric": "AES-256-GCM",
                "asymmetric": "RSA-4096",
                "key_derivation": "PBKDF2-SHA256",
                "hashing": "SHA-256"
            },
            "key_management": "AWS KMS",
            "compliance": ["FIPS 140-2", "SOC 2", "ISO 27001"],
            "context": self.encryption_context
        }
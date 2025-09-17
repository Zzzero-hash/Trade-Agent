"""Encryption service using AWS KMS for data at rest and in transit."""

import base64
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from uuid import UUID, uuid4

import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import secrets

from pydantic import BaseModel

from src.models.security import EncryptionKey
from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class EncryptionConfig(BaseModel):
    """Encryption service configuration."""
    aws_region: str = "us-east-1"
    kms_key_id: Optional[str] = None
    enable_field_encryption: bool = True
    enable_database_encryption: bool = True
    key_rotation_days: int = 90
    local_encryption_enabled: bool = True


class EncryptedData(BaseModel):
    """Encrypted data container."""
    encrypted_data: str
    encryption_method: str
    key_id: Optional[str] = None
    iv: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EncryptionService:
    """Encryption service for data at rest and in transit using AWS KMS."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """Initialize encryption service."""
        self.config = config or EncryptionConfig()
        self.settings = get_settings()
        
        # Initialize AWS KMS client
        try:
            self.kms_client = boto3.client('kms', region_name=self.config.aws_region)
        except Exception as e:
            logger.warning(f"Failed to initialize KMS client: {e}")
            self.kms_client = None
        
        # Local encryption key for fallback
        self._local_key = None
        if self.config.local_encryption_enabled:
            self._initialize_local_key()
    
    def _initialize_local_key(self) -> None:
        """Initialize local encryption key for fallback."""
        try:
            # In production, this would be loaded from secure storage
            key_material = os.environ.get('LOCAL_ENCRYPTION_KEY')
            if key_material:
                self._local_key = Fernet(key_material.encode())
            else:
                # Generate new key (for development only)
                self._local_key = Fernet(Fernet.generate_key())
                logger.warning("Using generated local encryption key - not suitable for production")
        except Exception as e:
            logger.error(f"Failed to initialize local encryption key: {e}")
    
    async def create_kms_key(self, description: str, 
                           key_usage: str = "ENCRYPT_DECRYPT") -> EncryptionKey:
        """Create a new KMS key for encryption."""
        
        if not self.kms_client:
            raise ValueError("KMS client not available")
        
        try:
            # Create KMS key
            response = self.kms_client.create_key(
                Description=description,
                KeyUsage=key_usage,
                KeySpec='SYMMETRIC_DEFAULT',
                Origin='AWS_KMS',
                MultiRegion=False
            )
            
            key_metadata = response['KeyMetadata']
            
            # Create alias for the key
            alias_name = f"alias/trading-platform-{uuid4().hex[:8]}"
            self.kms_client.create_alias(
                AliasName=alias_name,
                TargetKeyId=key_metadata['KeyId']
            )
            
            encryption_key = EncryptionKey(
                key_id=key_metadata['KeyId'],
                key_arn=key_metadata['Arn'],
                algorithm='AES_256',
                created_at=datetime.utcnow(),
                status='Enabled',
                description=description
            )
            
            logger.info(f"Created KMS key: {encryption_key.key_id}")
            
            return encryption_key
            
        except ClientError as e:
            logger.error(f"Failed to create KMS key: {e}")
            raise
    
    async def encrypt_with_kms(self, plaintext: Union[str, bytes], 
                             key_id: Optional[str] = None,
                             encryption_context: Optional[Dict[str, str]] = None) -> EncryptedData:
        """Encrypt data using AWS KMS."""
        
        if not self.kms_client:
            raise ValueError("KMS client not available")
        
        # Use default key if none specified
        if not key_id:
            key_id = self.config.kms_key_id
            if not key_id:
                raise ValueError("No KMS key ID specified")
        
        # Convert string to bytes if necessary
        if isinstance(plaintext, str):
            plaintext_bytes = plaintext.encode('utf-8')
        else:
            plaintext_bytes = plaintext
        
        try:
            # Encrypt with KMS
            response = self.kms_client.encrypt(
                KeyId=key_id,
                Plaintext=plaintext_bytes,
                EncryptionContext=encryption_context or {}
            )
            
            # Encode ciphertext as base64
            encrypted_data = base64.b64encode(response['CiphertextBlob']).decode('utf-8')
            
            return EncryptedData(
                encrypted_data=encrypted_data,
                encryption_method='AWS_KMS',
                key_id=response['KeyId'],
                metadata={
                    'encryption_context': encryption_context,
                    'encrypted_at': datetime.utcnow().isoformat()
                }
            )
            
        except ClientError as e:
            logger.error(f"KMS encryption failed: {e}")
            raise
    
    async def decrypt_with_kms(self, encrypted_data: EncryptedData,
                             encryption_context: Optional[Dict[str, str]] = None) -> bytes:
        """Decrypt data using AWS KMS."""
        
        if not self.kms_client:
            raise ValueError("KMS client not available")
        
        if encrypted_data.encryption_method != 'AWS_KMS':
            raise ValueError(f"Invalid encryption method: {encrypted_data.encryption_method}")
        
        try:
            # Decode base64 ciphertext
            ciphertext_blob = base64.b64decode(encrypted_data.encrypted_data)
            
            # Decrypt with KMS
            response = self.kms_client.decrypt(
                CiphertextBlob=ciphertext_blob,
                EncryptionContext=encryption_context or {}
            )
            
            return response['Plaintext']
            
        except ClientError as e:
            logger.error(f"KMS decryption failed: {e}")
            raise
    
    async def encrypt_field(self, value: Union[str, int, float], 
                          field_name: str,
                          customer_id: Optional[UUID] = None) -> EncryptedData:
        """Encrypt a single field value."""
        
        if not self.config.enable_field_encryption:
            raise ValueError("Field encryption is disabled")
        
        # Convert value to string
        plaintext = str(value)
        
        # Create encryption context for additional security
        encryption_context = {
            'field_name': field_name,
            'service': 'trading-platform'
        }
        
        if customer_id:
            encryption_context['customer_id'] = str(customer_id)
        
        # Try KMS encryption first, fallback to local encryption
        try:
            if self.kms_client and self.config.kms_key_id:
                return await self.encrypt_with_kms(plaintext, 
                                                 encryption_context=encryption_context)
            else:
                return await self._encrypt_local(plaintext, encryption_context)
                
        except Exception as e:
            logger.error(f"Field encryption failed: {e}")
            # Fallback to local encryption
            return await self._encrypt_local(plaintext, encryption_context)
    
    async def decrypt_field(self, encrypted_data: EncryptedData,
                          expected_field: str,
                          customer_id: Optional[UUID] = None) -> str:
        """Decrypt a single field value."""
        
        # Verify encryption context
        if encrypted_data.metadata:
            context = encrypted_data.metadata.get('encryption_context', {})
            if context.get('field_name') != expected_field:
                raise ValueError("Field name mismatch in encryption context")
            
            if customer_id and context.get('customer_id') != str(customer_id):
                raise ValueError("Customer ID mismatch in encryption context")
        
        # Decrypt based on method
        if encrypted_data.encryption_method == 'AWS_KMS':
            decrypted_bytes = await self.decrypt_with_kms(encrypted_data)
            return decrypted_bytes.decode('utf-8')
        elif encrypted_data.encryption_method == 'LOCAL_FERNET':
            return await self._decrypt_local(encrypted_data)
        else:
            raise ValueError(f"Unsupported encryption method: {encrypted_data.encryption_method}")
    
    async def _encrypt_local(self, plaintext: str, 
                           encryption_context: Dict[str, str]) -> EncryptedData:
        """Encrypt data using local Fernet encryption."""
        
        if not self._local_key:
            raise ValueError("Local encryption key not available")
        
        try:
            encrypted_bytes = self._local_key.encrypt(plaintext.encode('utf-8'))
            encrypted_data = base64.b64encode(encrypted_bytes).decode('utf-8')
            
            return EncryptedData(
                encrypted_data=encrypted_data,
                encryption_method='LOCAL_FERNET',
                metadata={
                    'encryption_context': encryption_context,
                    'encrypted_at': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Local encryption failed: {e}")
            raise
    
    async def _decrypt_local(self, encrypted_data: EncryptedData) -> str:
        """Decrypt data using local Fernet encryption."""
        
        if not self._local_key:
            raise ValueError("Local encryption key not available")
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encrypted_data)
            decrypted_bytes = self._local_key.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Local decryption failed: {e}")
            raise
    
    async def encrypt_database_record(self, record: Dict[str, Any],
                                    sensitive_fields: List[str],
                                    customer_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Encrypt sensitive fields in a database record."""
        
        if not self.config.enable_database_encryption:
            return record
        
        encrypted_record = record.copy()
        
        for field_name in sensitive_fields:
            if field_name in record and record[field_name] is not None:
                try:
                    encrypted_data = await self.encrypt_field(
                        record[field_name], 
                        field_name, 
                        customer_id
                    )
                    encrypted_record[field_name] = encrypted_data.dict()
                except Exception as e:
                    logger.error(f"Failed to encrypt field {field_name}: {e}")
                    # Keep original value if encryption fails
                    pass
        
        return encrypted_record
    
    async def decrypt_database_record(self, record: Dict[str, Any],
                                    sensitive_fields: List[str],
                                    customer_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Decrypt sensitive fields in a database record."""
        
        decrypted_record = record.copy()
        
        for field_name in sensitive_fields:
            if field_name in record and isinstance(record[field_name], dict):
                try:
                    encrypted_data = EncryptedData(**record[field_name])
                    decrypted_value = await self.decrypt_field(
                        encrypted_data,
                        field_name,
                        customer_id
                    )
                    decrypted_record[field_name] = decrypted_value
                except Exception as e:
                    logger.error(f"Failed to decrypt field {field_name}: {e}")
                    # Keep encrypted value if decryption fails
                    pass
        
        return decrypted_record
    
    async def generate_data_key(self, key_spec: str = 'AES_256') -> Dict[str, Any]:
        """Generate a data encryption key using KMS."""
        
        if not self.kms_client or not self.config.kms_key_id:
            raise ValueError("KMS not configured for data key generation")
        
        try:
            response = self.kms_client.generate_data_key(
                KeyId=self.config.kms_key_id,
                KeySpec=key_spec
            )
            
            return {
                'plaintext_key': response['Plaintext'],
                'encrypted_key': base64.b64encode(response['CiphertextBlob']).decode('utf-8'),
                'key_id': response['KeyId']
            }
            
        except ClientError as e:
            logger.error(f"Data key generation failed: {e}")
            raise
    
    async def encrypt_large_data(self, data: bytes, 
                               customer_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Encrypt large data using envelope encryption."""
        
        # Generate data encryption key
        data_key_info = await self.generate_data_key()
        
        # Encrypt data with the plaintext data key
        cipher = Cipher(
            algorithms.AES(data_key_info['plaintext_key'][:32]),
            modes.GCM(os.urandom(12))
        )
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'encrypted_data': base64.b64encode(ciphertext).decode('utf-8'),
            'encrypted_key': data_key_info['encrypted_key'],
            'iv': base64.b64encode(encryptor.initialization_vector).decode('utf-8'),
            'tag': base64.b64encode(encryptor.tag).decode('utf-8'),
            'key_id': data_key_info['key_id'],
            'customer_id': str(customer_id) if customer_id else None
        }
    
    async def decrypt_large_data(self, encrypted_info: Dict[str, Any]) -> bytes:
        """Decrypt large data using envelope encryption."""
        
        if not self.kms_client:
            raise ValueError("KMS client not available")
        
        try:
            # Decrypt the data encryption key
            encrypted_key = base64.b64decode(encrypted_info['encrypted_key'])
            response = self.kms_client.decrypt(CiphertextBlob=encrypted_key)
            plaintext_key = response['Plaintext']
            
            # Decrypt the data
            ciphertext = base64.b64decode(encrypted_info['encrypted_data'])
            iv = base64.b64decode(encrypted_info['iv'])
            tag = base64.b64decode(encrypted_info['tag'])
            
            cipher = Cipher(
                algorithms.AES(plaintext_key[:32]),
                modes.GCM(iv, tag)
            )
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Large data decryption failed: {e}")
            raise
    
    async def rotate_encryption_keys(self) -> Dict[str, Any]:
        """Rotate encryption keys according to policy."""
        
        if not self.kms_client:
            return {"status": "skipped", "reason": "KMS not available"}
        
        try:
            # Enable automatic key rotation for KMS keys
            if self.config.kms_key_id:
                self.kms_client.enable_key_rotation(KeyId=self.config.kms_key_id)
            
            # Generate new local key
            if self.config.local_encryption_enabled:
                self._local_key = Fernet(Fernet.generate_key())
            
            logger.info("Encryption key rotation completed")
            
            return {
                "status": "success",
                "rotated_at": datetime.utcnow().isoformat(),
                "kms_rotation_enabled": bool(self.config.kms_key_id),
                "local_key_rotated": self.config.local_encryption_enabled
            }
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def get_encryption_status(self) -> Dict[str, Any]:
        """Get current encryption service status."""
        
        return {
            "kms_available": self.kms_client is not None,
            "kms_key_configured": bool(self.config.kms_key_id),
            "local_encryption_available": self._local_key is not None,
            "field_encryption_enabled": self.config.enable_field_encryption,
            "database_encryption_enabled": self.config.enable_database_encryption,
            "key_rotation_days": self.config.key_rotation_days
        }
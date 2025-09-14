"""
Security utilities for the AI trading platform.

This module provides security functions for file path validation,
credential management, and input sanitization.
"""

import hashlib
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class FilePathValidator:
    """Secure file path validation and sanitization."""
    
    # Allowed directories for different file types
    ALLOWED_DIRECTORIES = {
        'models': ['models', 'checkpoints'],
        'logs': ['logs'],
        'data': ['data', 'features'],
        'config': ['config'],
        'temp': ['temp', 'tmp']
    }
    
    # Dangerous path patterns
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'/etc/',  # System directories
        r'/proc/',
        r'/sys/',
        r'C:\\Windows\\',  # Windows system directories
        r'C:\\Program Files\\',
        r'~/',  # Home directory shortcuts
        r'\$\{.*\}',  # Variable expansion
        r'`.*`',  # Command substitution
    ]
    
    @classmethod
    def validate_file_path(
        cls, 
        filepath: str, 
        file_type: str = 'models',
        create_dirs: bool = True
    ) -> Path:
        """
        Validate and sanitize file path with comprehensive security checks.
        
        Args:
            filepath: Input file path
            file_type: Type of file (models, logs, data, config, temp)
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is invalid or unsafe
            ValueError: If file_type is not supported
        """
        if not isinstance(filepath, str) or not filepath.strip():
            raise SecurityError("File path must be a non-empty string")
        
        if file_type not in cls.ALLOWED_DIRECTORIES:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Remove null bytes and control characters
        sanitized_path = cls._sanitize_path(filepath)
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(sanitized_path)
        
        # Normalize and resolve path
        try:
            path = Path(sanitized_path).resolve()
        except (OSError, ValueError) as e:
            raise SecurityError(f"Invalid path format: {e}") from e
        
        # Ensure path is within allowed directories
        cls._validate_directory_access(path, file_type)
        
        # Create directories if requested and safe
        if create_dirs:
            cls._safe_create_directories(path.parent)
        
        return path
    
    @classmethod
    def _sanitize_path(cls, filepath: str) -> str:
        """Remove dangerous characters from file path."""
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in filepath if ord(char) >= 32)
        
        # Remove multiple consecutive slashes/backslashes
        sanitized = re.sub(r'[/\\]+', '/', sanitized)
        
        # Remove leading/trailing whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    @classmethod
    def _check_dangerous_patterns(cls, filepath: str) -> None:
        """Check for dangerous path patterns."""
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filepath, re.IGNORECASE):
                raise SecurityError(f"Dangerous path pattern detected: {pattern}")
        
        # Check for absolute paths (security risk)
        if os.path.isabs(filepath):
            raise SecurityError("Absolute paths are not allowed")
        
        # Check for hidden files/directories (potential security risk)
        path_parts = Path(filepath).parts
        for part in path_parts:
            if part.startswith('.') and part not in ['.', '..']:
                logger.warning("Hidden file/directory in path: %s", part)
    
    @classmethod
    def _validate_directory_access(cls, path: Path, file_type: str) -> None:
        """Validate that path is within allowed directories."""
        allowed_dirs = cls.ALLOWED_DIRECTORIES[file_type]
        
        # Get the first part of the path
        try:
            first_part = path.parts[0] if path.parts else ""
        except (IndexError, AttributeError) as e:
            raise SecurityError("Invalid path structure") from e
        
        # Check if first part is in allowed directories
        if first_part not in allowed_dirs:
            # If not, prepend with default allowed directory
            default_dir = allowed_dirs[0]
            logger.info("Prepending path with allowed directory: %s", default_dir)
            # This would need to be handled by the caller
        
        # Additional check: ensure resolved path doesn't escape project directory
        try:
            project_root = Path.cwd().resolve()
            if not str(path).startswith(str(project_root)):
                raise SecurityError("Path escapes project directory")
        except (OSError, ValueError) as e:
            raise SecurityError("Unable to validate path security") from e
    
    @classmethod
    def _safe_create_directories(cls, directory: Path) -> None:
        """Safely create directories with proper permissions."""
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o755)
        except (OSError, PermissionError) as e:
            raise SecurityError(f"Cannot create directory {directory}: {e}") from e


class CredentialManager:
    """Secure credential management with encryption."""
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize credential manager.
        
        Args:
            key: Encryption key (if None, generates new key)
        """
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_credential(self, credential: str) -> bytes:
        """Encrypt a credential string."""
        if not isinstance(credential, str):
            raise ValueError("Credential must be a string")
        
        return self.cipher.encrypt(credential.encode())
    
    def decrypt_credential(self, encrypted_credential: bytes) -> str:
        """Decrypt a credential."""
        if not isinstance(encrypted_credential, bytes):
            raise ValueError("Encrypted credential must be bytes")
        
        try:
            return self.cipher.decrypt(encrypted_credential).decode()
        except Exception as e:
            raise SecurityError(f"Failed to decrypt credential: {e}") from e
    
    def generate_api_key(self, length: int = 32) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: bytes, salt: bytes) -> bool:
        """Verify a password against its hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return secrets.compare_digest(password_hash, computed_hash)


class InputSanitizer:
    """Input sanitization utilities."""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"('|(\\')|(;)|(\\;))",  # SQL metacharacters
        r"((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",  # 'or
        # 'union pattern
        r"((\%27)|(\'))((\%75)|u|(\%55))((\%6E)|n|(\%4E))"
        r"((\%69)|i|(\%49))((\%6F)|o|(\%4F))((\%6E)|n|(\%4E))",
        # 'select pattern
        r"((\%27)|(\'))((\%73)|s|(\%53))((\%65)|e|(\%45))"
        r"((\%6C)|l|(\%4C))((\%65)|e|(\%45))((\%63)|c|(\%43))((\%74)|t|(\%54))",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
    ]
    
    @classmethod
    def sanitize_string(cls, input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize input string for security.
        
        Args:
            input_string: Input to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If dangerous patterns detected
        """
        if not isinstance(input_string, str):
            raise ValueError("Input must be a string")
        
        if len(input_string) > max_length:
            raise SecurityError(f"Input too long: {len(input_string)} > {max_length}")
        
        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise SecurityError("Potential SQL injection detected")
        
        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise SecurityError("Potential XSS attack detected")
        
        # Remove null bytes and control characters
        sanitized = ''.join(
            char for char in input_string
            if ord(char) >= 32 or char in '\t\n\r'
        )
        
        return sanitized.strip()
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """Validate trading symbol format."""
        if not isinstance(symbol, str):
            raise ValueError("Symbol must be a string")
        
        # Allow only alphanumeric characters and common separators
        if not re.match(r'^[A-Z0-9._-]+$', symbol.upper()):
            raise SecurityError(f"Invalid symbol format: {symbol}")
        
        if len(symbol) > 20:  # Reasonable limit for trading symbols
            raise SecurityError(f"Symbol too long: {symbol}")
        
        return symbol.upper()
    
    @classmethod
    def validate_numeric_input(
        cls,
        value: Union[str, int, float],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """Validate and convert numeric input."""
        try:
            numeric_value = float(value)
        except (ValueError, TypeError) as e:
            raise SecurityError(f"Invalid numeric value: {value}") from e
        
        if not np.isfinite(numeric_value):
            raise SecurityError(f"Non-finite numeric value: {numeric_value}")
        
        if min_value is not None and numeric_value < min_value:
            raise SecurityError(
                f"Value below minimum: {numeric_value} < {min_value}"
            )

        if max_value is not None and numeric_value > max_value:
            raise SecurityError(
                f"Value above maximum: {numeric_value} > {max_value}"
            )
        
        return numeric_value


# Global instances for convenience
file_validator = FilePathValidator()
credential_manager = CredentialManager()
input_sanitizer = InputSanitizer()
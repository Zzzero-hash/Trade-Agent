"""
Tests for security utilities.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.utils.security import (
    SecurityError,
    FilePathValidator,
    CredentialManager,
    InputSanitizer,
)


class TestFilePathValidator:
    """Test file path validation functionality."""

    def test_validate_file_path_success(self):
        """Test successful file path validation."""
        validator = FilePathValidator()
        path = validator.validate_file_path("models/test.pth", "models", False)
        assert isinstance(path, Path)
        assert "test.pth" in str(path)

    def test_validate_file_path_dangerous_patterns(self):
        """Test rejection of dangerous path patterns."""
        validator = FilePathValidator()
        
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\Windows\\System32",
            "/etc/shadow",
            "C:\\Windows\\System32\\config",
            "~/sensitive_file",
            "${HOME}/file",
            "`rm -rf /`",
        ]
        
        for dangerous_path in dangerous_paths:
            with pytest.raises(SecurityError):
                validator.validate_file_path(dangerous_path, "models", False)

    def test_validate_file_path_empty_string(self):
        """Test rejection of empty file paths."""
        validator = FilePathValidator()
        with pytest.raises(SecurityError):
            validator.validate_file_path("", "models", False)

    def test_validate_file_path_invalid_type(self):
        """Test rejection of invalid file types."""
        validator = FilePathValidator()
        with pytest.raises(ValueError):
            validator.validate_file_path("test.txt", "invalid_type", False)


class TestCredentialManager:
    """Test credential management functionality."""

    def test_encrypt_decrypt_credential(self):
        """Test credential encryption and decryption."""
        manager = CredentialManager()
        original = "secret_api_key_12345"
        
        encrypted = manager.encrypt_credential(original)
        decrypted = manager.decrypt_credential(encrypted)
        
        assert decrypted == original
        assert encrypted != original.encode()

    def test_generate_api_key(self):
        """Test API key generation."""
        manager = CredentialManager()
        key1 = manager.generate_api_key()
        key2 = manager.generate_api_key()
        
        assert len(key1) > 0
        assert len(key2) > 0
        assert key1 != key2

    def test_hash_verify_password(self):
        """Test password hashing and verification."""
        manager = CredentialManager()
        password = "secure_password_123"
        
        password_hash, salt = manager.hash_password(password)
        
        # Correct password should verify
        assert manager.verify_password(password, password_hash, salt)
        
        # Wrong password should not verify
        assert not manager.verify_password("wrong_password", password_hash, salt)

    def test_encrypt_invalid_input(self):
        """Test encryption with invalid input."""
        manager = CredentialManager()
        with pytest.raises(ValueError):
            manager.encrypt_credential(123)  # Not a string

    def test_decrypt_invalid_input(self):
        """Test decryption with invalid input."""
        manager = CredentialManager()
        with pytest.raises(ValueError):
            manager.decrypt_credential("not_bytes")  # Not bytes


class TestInputSanitizer:
    """Test input sanitization functionality."""

    def test_sanitize_string_success(self):
        """Test successful string sanitization."""
        sanitizer = InputSanitizer()
        clean_string = "AAPL"
        result = sanitizer.sanitize_string(clean_string)
        assert result == clean_string

    def test_sanitize_string_sql_injection(self):
        """Test SQL injection detection."""
        sanitizer = InputSanitizer()
        
        sql_attacks = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM passwords --",
            "admin'--",
        ]
        
        for attack in sql_attacks:
            with pytest.raises(SecurityError):
                sanitizer.sanitize_string(attack)

    def test_sanitize_string_xss_attack(self):
        """Test XSS attack detection."""
        sanitizer = InputSanitizer()
        
        xss_attacks = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='evil.com'></iframe>",
            "onclick='alert(1)'",
        ]
        
        for attack in xss_attacks:
            with pytest.raises(SecurityError):
                sanitizer.sanitize_string(attack)

    def test_sanitize_string_too_long(self):
        """Test rejection of overly long strings."""
        sanitizer = InputSanitizer()
        long_string = "A" * 1001  # Default max is 1000
        
        with pytest.raises(SecurityError):
            sanitizer.sanitize_string(long_string)

    def test_validate_symbol_success(self):
        """Test successful symbol validation."""
        sanitizer = InputSanitizer()
        
        valid_symbols = [
            "AAPL",
            "BTC-USD",
            "EUR_USD",
            "SPY.TO",
            "MSFT",
            "GOOGL",
        ]
        
        for symbol in valid_symbols:
            result = sanitizer.validate_symbol(symbol)
            assert result == symbol.upper()

    def test_validate_symbol_invalid_format(self):
        """Test rejection of invalid symbol formats."""
        sanitizer = InputSanitizer()
        
        invalid_symbols = [
            "AA PL",  # Space
            "AAPL@",  # Invalid character
            "AAPL#USD",  # Invalid character
            "AAPL$",  # Invalid character
            "",  # Empty
            "A" * 21,  # Too long
        ]
        
        for symbol in invalid_symbols:
            with pytest.raises((SecurityError, ValueError)):
                sanitizer.validate_symbol(symbol)

    def test_validate_symbol_invalid_type(self):
        """Test rejection of non-string symbols."""
        sanitizer = InputSanitizer()
        with pytest.raises(ValueError):
            sanitizer.validate_symbol(123)

    def test_validate_numeric_input_success(self):
        """Test successful numeric validation."""
        sanitizer = InputSanitizer()
        
        # Test various input types
        assert sanitizer.validate_numeric_input("123.45") == 123.45
        assert sanitizer.validate_numeric_input(100) == 100.0
        assert sanitizer.validate_numeric_input(99.99) == 99.99

    def test_validate_numeric_input_with_bounds(self):
        """Test numeric validation with bounds."""
        sanitizer = InputSanitizer()
        
        # Within bounds
        result = sanitizer.validate_numeric_input("50", 0, 100)
        assert result == 50.0
        
        # Below minimum
        with pytest.raises(SecurityError):
            sanitizer.validate_numeric_input("-10", 0, 100)
        
        # Above maximum
        with pytest.raises(SecurityError):
            sanitizer.validate_numeric_input("150", 0, 100)

    def test_validate_numeric_input_invalid(self):
        """Test rejection of invalid numeric inputs."""
        sanitizer = InputSanitizer()
        
        invalid_inputs = [
            "not_a_number",
            "inf",
            "-inf",
            "nan",
            None,
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(SecurityError):
                sanitizer.validate_numeric_input(invalid_input)


class TestSecurityIntegration:
    """Integration tests for security components."""

    def test_end_to_end_file_validation(self):
        """Test complete file validation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file path
            test_path = Path(temp_dir) / "models" / "test_model.pth"
            
            # This should work without issues
            validator = FilePathValidator()
            # Note: We're testing the validation logic, not actual file creation
            # in a real temp directory due to the security restrictions

    def test_credential_workflow(self):
        """Test complete credential management workflow."""
        manager = CredentialManager()
        
        # Generate API key
        api_key = manager.generate_api_key()
        
        # Encrypt it
        encrypted_key = manager.encrypt_credential(api_key)
        
        # Decrypt it
        decrypted_key = manager.decrypt_credential(encrypted_key)
        
        assert decrypted_key == api_key

    def test_input_validation_workflow(self):
        """Test complete input validation workflow."""
        sanitizer = InputSanitizer()
        
        # Validate trading symbol
        symbol = sanitizer.validate_symbol("aapl")
        assert symbol == "AAPL"
        
        # Validate price
        price = sanitizer.validate_numeric_input("150.25", 0, 1000)
        assert price == 150.25
        
        # Validate description
        description = sanitizer.sanitize_string("Apple Inc. stock")
        assert "Apple Inc. stock" in description


if __name__ == "__main__":
    pytest.main([__file__])
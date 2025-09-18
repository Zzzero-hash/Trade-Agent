"""Security service factory for initializing and configuring security components."""

import logging
import os
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel

from src.security.kyc_service import KYCService, JumioConfig, OnfidoConfig
from src.security.audit_service import AuditService, AuditConfig
from src.security.encryption_service import EncryptionService, EncryptionConfig
from src.security.auth_service import AuthService, AuthConfig
from src.security.compliance_service import ComplianceService, ComplianceConfig


logger = logging.getLogger(__name__)


class SecurityConfiguration(BaseModel):
    """Complete security configuration."""
    kyc: Dict[str, Any]
    audit: Dict[str, Any]
    encryption: Dict[str, Any]
    authentication: Dict[str, Any]
    compliance: Dict[str, Any]
    monitoring: Dict[str, Any]
    data_protection: Dict[str, Any]
    integrations: Dict[str, Any]
    development: Optional[Dict[str, Any]] = None


class SecurityServiceFactory:
    """Factory for creating and configuring security services."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize security factory with configuration."""
        self.config_path = config_path or "config/security.yaml"
        self.config = self._load_configuration()
        self._services = {}
    
    def _load_configuration(self) -> SecurityConfiguration:
        """Load security configuration from file."""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)
            
            return SecurityConfiguration(**config_data)
            
        except FileNotFoundError:
            logger.warning(f"Security config file not found: {self.config_path}")
            return self._get_default_configuration()
        except Exception as e:
            logger.error(f"Failed to load security configuration: {e}")
            return self._get_default_configuration()
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            # Extract environment variable name and default value
            env_expr = data[2:-1]  # Remove ${ and }
            if ':-' in env_expr:
                env_name, default_value = env_expr.split(':-', 1)
            else:
                env_name, default_value = env_expr, None
            
            return os.getenv(env_name, default_value)
        else:
            return data
    
    def _get_default_configuration(self) -> SecurityConfiguration:
        """Get default security configuration."""
        return SecurityConfiguration(
            kyc={
                "providers": {
                    "jumio": {
                        "api_token": "default_token",
                        "api_secret": "default_secret",
                        "base_url": "https://netverify.com/api/netverify/v2"
                    }
                }
            },
            audit={
                "enable_immutable_storage": True,
                "retention_days": 2555,
                "signing_key": "default_signing_key"
            },
            encryption={
                "enable_field_encryption": True,
                "enable_database_encryption": True,
                "local_encryption_enabled": True
            },
            authentication={
                "jwt": {
                    "secret_key": "default_jwt_secret",
                    "algorithm": "HS256"
                },
                "security": {
                    "require_mfa": True,
                    "max_login_attempts": 5
                }
            },
            compliance={
                "monitoring": {
                    "enable_kyc_monitoring": True,
                    "enable_aml_monitoring": True
                }
            },
            monitoring={
                "security_events": {
                    "retention_days": 365
                }
            },
            data_protection={
                "retention": {
                    "customer_data": 2555
                }
            },
            integrations={
                "external_services": {}
            }
        )
    
    def create_kyc_service(self) -> KYCService:
        """Create and configure KYC service."""
        if 'kyc' in self._services:
            return self._services['kyc']
        
        kyc_config = self.config.kyc
        
        # Configure Jumio
        jumio_config = None
        if 'jumio' in kyc_config.get('providers', {}):
            jumio_data = kyc_config['providers']['jumio']
            jumio_config = JumioConfig(
                api_token=jumio_data.get('api_token', ''),
                api_secret=jumio_data.get('api_secret', ''),
                base_url=jumio_data.get('base_url', 'https://netverify.com/api/netverify/v2'),
                callback_url=jumio_data.get('callback_url')
            )
        
        # Configure Onfido
        onfido_config = None
        if 'onfido' in kyc_config.get('providers', {}):
            onfido_data = kyc_config['providers']['onfido']
            onfido_config = OnfidoConfig(
                api_token=onfido_data.get('api_token', ''),
                base_url=onfido_data.get('base_url', 'https://api.onfido.com/v3.6'),
                webhook_token=onfido_data.get('webhook_token')
            )
        
        service = KYCService(jumio_config=jumio_config, onfido_config=onfido_config)
        self._services['kyc'] = service
        
        logger.info("KYC service created and configured")
        return service
    
    def create_audit_service(self) -> AuditService:
        """Create and configure audit service."""
        if 'audit' in self._services:
            return self._services['audit']
        
        audit_config_data = self.config.audit
        
        audit_config = AuditConfig(
            enable_immutable_storage=audit_config_data.get('enable_immutable_storage', True),
            retention_days=audit_config_data.get('retention_days', 2555),
            hash_algorithm=audit_config_data.get('hash_algorithm', 'sha256'),
            signing_key=audit_config_data.get('signing_key', 'default_key'),
            s3_bucket=audit_config_data.get('s3_bucket'),
            encryption_enabled=audit_config_data.get('encryption_enabled', True)
        )
        
        service = AuditService(audit_config)
        self._services['audit'] = service
        
        logger.info("Audit service created and configured")
        return service
    
    def create_encryption_service(self) -> EncryptionService:
        """Create and configure encryption service."""
        if 'encryption' in self._services:
            return self._services['encryption']
        
        encryption_config_data = self.config.encryption
        
        encryption_config = EncryptionConfig(
            aws_region=encryption_config_data.get('aws_region', 'us-east-1'),
            kms_key_id=encryption_config_data.get('kms_key_id'),
            enable_field_encryption=encryption_config_data.get('enable_field_encryption', True),
            enable_database_encryption=encryption_config_data.get('enable_database_encryption', True),
            key_rotation_days=encryption_config_data.get('key_rotation_days', 90),
            local_encryption_enabled=encryption_config_data.get('local_encryption_enabled', True)
        )
        
        service = EncryptionService(encryption_config)
        self._services['encryption'] = service
        
        logger.info("Encryption service created and configured")
        return service
    
    def create_auth_service(self) -> AuthService:
        """Create and configure authentication service."""
        if 'auth' in self._services:
            return self._services['auth']
        
        auth_config_data = self.config.authentication
        
        auth_config = AuthConfig(
            jwt_secret_key=auth_config_data['jwt']['secret_key'],
            jwt_algorithm=auth_config_data['jwt'].get('algorithm', 'HS256'),
            access_token_expire_minutes=auth_config_data['jwt'].get('access_token_expire_minutes', 30),
            refresh_token_expire_days=auth_config_data['jwt'].get('refresh_token_expire_days', 30),
            mfa_token_expire_minutes=auth_config_data['jwt'].get('mfa_token_expire_minutes', 5),
            max_login_attempts=auth_config_data['security'].get('max_login_attempts', 5),
            lockout_duration_minutes=auth_config_data['security'].get('lockout_duration_minutes', 15),
            session_timeout_minutes=auth_config_data['security'].get('session_timeout_minutes', 60),
            require_mfa=auth_config_data['security'].get('require_mfa', True),
            password_min_length=auth_config_data.get('password_policy', {}).get('min_length', 12)
        )
        
        service = AuthService(auth_config)
        self._services['auth'] = service
        
        logger.info("Authentication service created and configured")
        return service
    
    def create_compliance_service(self) -> ComplianceService:
        """Create and configure compliance service."""
        if 'compliance' in self._services:
            return self._services['compliance']
        
        # Create dependent services
        kyc_service = self.create_kyc_service()
        audit_service = self.create_audit_service()
        encryption_service = self.create_encryption_service()
        auth_service = self.create_auth_service()
        
        compliance_config_data = self.config.compliance
        
        compliance_config = ComplianceConfig(
            enable_kyc_monitoring=compliance_config_data['monitoring'].get('enable_kyc_monitoring', True),
            enable_aml_monitoring=compliance_config_data['monitoring'].get('enable_aml_monitoring', True),
            enable_audit_monitoring=compliance_config_data['monitoring'].get('enable_audit_monitoring', True),
            enable_security_monitoring=compliance_config_data['monitoring'].get('enable_security_monitoring', True),
            alert_threshold_high=compliance_config_data.get('alerts', {}).get('high_threshold', 5),
            alert_threshold_critical=compliance_config_data.get('alerts', {}).get('critical_threshold', 10)
        )
        
        service = ComplianceService(
            kyc_service=kyc_service,
            audit_service=audit_service,
            encryption_service=encryption_service,
            auth_service=auth_service,
            config=compliance_config
        )
        
        self._services['compliance'] = service
        
        logger.info("Compliance service created and configured")
        return service
    
    def create_all_services(self) -> Dict[str, Any]:
        """Create all security services."""
        services = {
            'kyc': self.create_kyc_service(),
            'audit': self.create_audit_service(),
            'encryption': self.create_encryption_service(),
            'auth': self.create_auth_service(),
            'compliance': self.create_compliance_service()
        }
        
        logger.info("All security services created and configured")
        return services
    
    def get_service(self, service_name: str) -> Any:
        """Get a specific service by name."""
        if service_name not in self._services:
            if service_name == 'kyc':
                return self.create_kyc_service()
            elif service_name == 'audit':
                return self.create_audit_service()
            elif service_name == 'encryption':
                return self.create_encryption_service()
            elif service_name == 'auth':
                return self.create_auth_service()
            elif service_name == 'compliance':
                return self.create_compliance_service()
            else:
                raise ValueError(f"Unknown service: {service_name}")
        
        return self._services[service_name]
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate security configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate JWT secret key
        jwt_secret = self.config.authentication['jwt']['secret_key']
        if not jwt_secret or jwt_secret == 'default_jwt_secret':
            validation_results["errors"].append("JWT secret key not configured or using default")
            validation_results["valid"] = False
        
        # Validate audit signing key
        audit_key = self.config.audit.get('signing_key')
        if not audit_key or audit_key == 'default_signing_key':
            validation_results["errors"].append("Audit signing key not configured or using default")
            validation_results["valid"] = False
        
        # Validate KYC provider configuration
        kyc_providers = self.config.kyc.get('providers', {})
        if not kyc_providers:
            validation_results["warnings"].append("No KYC providers configured")
        
        # Validate encryption configuration
        if not self.config.encryption.get('kms_key_id') and not self.config.encryption.get('local_encryption_enabled'):
            validation_results["errors"].append("No encryption method configured")
            validation_results["valid"] = False
        
        # Validate MFA requirement
        if not self.config.authentication['security'].get('require_mfa'):
            validation_results["warnings"].append("MFA is not required - consider enabling for production")
        
        return validation_results
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all security services."""
        health_status = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
        try:
            # Check KYC service
            kyc_service = self.get_service('kyc')
            health_status["services"]["kyc"] = {
                "status": "healthy",
                "jumio_configured": kyc_service.jumio_client is not None,
                "onfido_configured": kyc_service.onfido_client is not None
            }
        except Exception as e:
            health_status["services"]["kyc"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        try:
            # Check encryption service
            encryption_service = self.get_service('encryption')
            encryption_status = await encryption_service.get_encryption_status()
            health_status["services"]["encryption"] = {
                "status": "healthy",
                **encryption_status
            }
        except Exception as e:
            health_status["services"]["encryption"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        try:
            # Check audit service
            audit_service = self.get_service('audit')
            health_status["services"]["audit"] = {
                "status": "healthy",
                "immutable_storage": audit_service.config.enable_immutable_storage
            }
        except Exception as e:
            health_status["services"]["audit"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        try:
            # Check auth service
            auth_service = self.get_service('auth')
            health_status["services"]["auth"] = {
                "status": "healthy",
                "mfa_required": auth_service.config.require_mfa,
                "active_sessions": len(auth_service._active_sessions)
            }
        except Exception as e:
            health_status["services"]["auth"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        return health_status
    
    async def cleanup(self) -> None:
        """Cleanup all services and resources."""
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                logger.info(f"Cleaned up {service_name} service")
            except Exception as e:
                logger.error(f"Failed to cleanup {service_name} service: {e}")
        
        self._services.clear()


# Global factory instance
_security_factory = None


def get_security_factory(config_path: Optional[str] = None) -> SecurityServiceFactory:
    """Get global security factory instance."""
    global _security_factory
    if _security_factory is None:
        _security_factory = SecurityServiceFactory(config_path)
    return _security_factory


def get_security_service(service_name: str) -> Any:
    """Get a security service by name."""
    factory = get_security_factory()
    return factory.get_service(service_name)
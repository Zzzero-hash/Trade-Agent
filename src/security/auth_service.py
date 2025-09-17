"""JWT authentication service with MFA support and session management."""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID, uuid4

import jwt
import pyotp
import qrcode
from io import BytesIO
import base64
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from src.models.security import AuthToken, MFADevice, SecurityEvent
from src.config.settings import get_settings


logger = logging.getLogger(__name__)


class AuthConfig(BaseModel):
    """Authentication service configuration."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    mfa_token_expire_minutes: int = 5
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 60
    require_mfa: bool = True
    password_min_length: int = 12


class LoginAttempt(BaseModel):
    """Login attempt tracking."""
    user_id: UUID
    ip_address: str
    timestamp: datetime
    success: bool
    mfa_required: bool = False
    mfa_success: Optional[bool] = None


class UserSession(BaseModel):
    """User session information."""
    session_id: str
    user_id: UUID
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: Optional[str] = None
    is_active: bool = True
    mfa_verified: bool = False


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    jti: str  # JWT ID
    type: str  # token type: access, refresh, mfa
    session_id: Optional[str] = None
    mfa_verified: bool = False


class AuthService:
    """Authentication service with JWT tokens, MFA, and session management."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        """Initialize authentication service."""
        self.config = config or AuthConfig(jwt_secret_key="default-secret")
        self.settings = get_settings()
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # In-memory stores (in production, use Redis or database)
        self._login_attempts: Dict[str, List[LoginAttempt]] = {}
        self._active_sessions: Dict[str, UserSession] = {}
        self._revoked_tokens: set = set()
        self._mfa_challenges: Dict[str, Dict[str, Any]] = {}
    
    async def register_user(self, email: EmailStr, password: str,
                          first_name: str, last_name: str,
                          ip_address: str) -> Dict[str, Any]:
        """Register a new user with password validation."""
        
        # Validate password strength
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        # Hash password
        password_hash = self.pwd_context.hash(password)
        
        # Create user (this would integrate with user service)
        user_id = uuid4()
        
        # Log registration event
        await self._log_security_event(
            event_type="USER_REGISTRATION",
            severity="low",
            user_id=user_id,
            ip_address=ip_address,
            event_data={
                "email": email,
                "first_name": first_name,
                "last_name": last_name
            }
        )
        
        logger.info(f"User registered: {email}")
        
        return {
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "mfa_required": self.config.require_mfa
        }
    
    async def authenticate_user(self, email: str, password: str,
                              ip_address: str, user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate user with email and password."""
        
        # Check for account lockout
        if await self._is_account_locked(email, ip_address):
            await self._log_security_event(
                event_type="LOGIN_BLOCKED_LOCKOUT",
                severity="medium",
                ip_address=ip_address,
                event_data={"email": email, "reason": "account_locked"}
            )
            raise ValueError("Account is temporarily locked due to multiple failed attempts")
        
        # Retrieve user (this would query the database)
        user_data = await self._get_user_by_email(email)
        if not user_data:
            await self._record_login_attempt(None, ip_address, False)
            raise ValueError("Invalid credentials")
        
        # Verify password
        if not self.pwd_context.verify(password, user_data["password_hash"]):
            await self._record_login_attempt(user_data["user_id"], ip_address, False)
            await self._log_security_event(
                event_type="LOGIN_FAILED",
                severity="medium",
                user_id=user_data["user_id"],
                ip_address=ip_address,
                event_data={"email": email, "reason": "invalid_password"}
            )
            raise ValueError("Invalid credentials")
        
        # Check if MFA is required
        mfa_devices = await self._get_user_mfa_devices(user_data["user_id"])
        mfa_required = self.config.require_mfa and len(mfa_devices) > 0
        
        if mfa_required:
            # Generate MFA challenge
            challenge_id = await self._create_mfa_challenge(user_data["user_id"], ip_address)
            
            await self._record_login_attempt(
                user_data["user_id"], ip_address, True, mfa_required=True
            )
            
            return {
                "status": "mfa_required",
                "challenge_id": challenge_id,
                "mfa_methods": [device.device_type for device in mfa_devices]
            }
        else:
            # Complete authentication without MFA
            session = await self._create_user_session(
                user_data["user_id"], ip_address, user_agent, mfa_verified=False
            )
            
            tokens = await self._generate_tokens(
                user_data["user_id"], session.session_id, mfa_verified=False
            )
            
            await self._record_login_attempt(
                user_data["user_id"], ip_address, True, mfa_required=False
            )
            
            await self._log_security_event(
                event_type="LOGIN_SUCCESS",
                severity="low",
                user_id=user_data["user_id"],
                ip_address=ip_address,
                event_data={"email": email, "mfa_used": False}
            )
            
            return {
                "status": "authenticated",
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "session_id": session.session_id,
                "user_id": user_data["user_id"]
            }
    
    async def verify_mfa_challenge(self, challenge_id: str, mfa_code: str,
                                 device_id: Optional[UUID] = None,
                                 ip_address: str = "unknown",
                                 user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Verify MFA challenge with provided code."""
        
        # Retrieve challenge
        challenge = self._mfa_challenges.get(challenge_id)
        if not challenge:
            raise ValueError("Invalid or expired MFA challenge")
        
        # Check challenge expiration
        if datetime.utcnow() > challenge["expires_at"]:
            del self._mfa_challenges[challenge_id]
            raise ValueError("MFA challenge has expired")
        
        user_id = challenge["user_id"]
        
        # Get user's MFA devices
        mfa_devices = await self._get_user_mfa_devices(user_id)
        
        # Verify MFA code
        verified = False
        used_device = None
        
        for device in mfa_devices:
            if device_id and device.device_id != device_id:
                continue
                
            if device.device_type == "totp":
                totp = pyotp.TOTP(device.secret_key)
                if totp.verify(mfa_code, valid_window=1):
                    verified = True
                    used_device = device
                    break
            elif device.device_type == "sms":
                # In production, this would verify SMS code
                if mfa_code == challenge.get("sms_code"):
                    verified = True
                    used_device = device
                    break
        
        # Clean up challenge
        del self._mfa_challenges[challenge_id]
        
        if not verified:
            await self._record_login_attempt(user_id, ip_address, False, mfa_success=False)
            await self._log_security_event(
                event_type="MFA_FAILED",
                severity="high",
                user_id=user_id,
                ip_address=ip_address,
                event_data={"device_type": device.device_type if device_id else "unknown"}
            )
            raise ValueError("Invalid MFA code")
        
        # Create authenticated session
        session = await self._create_user_session(
            user_id, ip_address, user_agent, mfa_verified=True
        )
        
        tokens = await self._generate_tokens(
            user_id, session.session_id, mfa_verified=True
        )
        
        # Update device last used
        if used_device:
            used_device.last_used = datetime.utcnow()
            await self._update_mfa_device(used_device)
        
        await self._record_login_attempt(user_id, ip_address, True, mfa_success=True)
        
        await self._log_security_event(
            event_type="MFA_SUCCESS",
            severity="low",
            user_id=user_id,
            ip_address=ip_address,
            event_data={"device_type": used_device.device_type if used_device else "unknown"}
        )
        
        return {
            "status": "authenticated",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "session_id": session.session_id,
            "user_id": user_id
        }
    
    async def setup_totp_mfa(self, user_id: UUID, device_name: str) -> Dict[str, Any]:
        """Set up TOTP MFA for a user."""
        
        # Generate secret key
        secret_key = pyotp.random_base32()
        
        # Create TOTP instance
        totp = pyotp.TOTP(secret_key)
        
        # Generate QR code
        user_data = await self._get_user_by_id(user_id)
        if not user_data:
            raise ValueError("User not found")
        
        provisioning_uri = totp.provisioning_uri(
            name=user_data["email"],
            issuer_name="AI Trading Platform"
        )
        
        # Generate QR code image
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        qr_code_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Create MFA device (not yet verified)
        device = MFADevice(
            device_id=uuid4(),
            user_id=user_id,
            device_type="totp",
            device_name=device_name,
            secret_key=secret_key,
            is_verified=False,
            created_at=datetime.utcnow()
        )
        
        # Store device (temporarily)
        await self._store_mfa_device(device)
        
        return {
            "device_id": device.device_id,
            "secret_key": secret_key,
            "qr_code": qr_code_data,
            "manual_entry_key": secret_key,
            "provisioning_uri": provisioning_uri
        }
    
    async def verify_totp_setup(self, device_id: UUID, verification_code: str) -> bool:
        """Verify TOTP setup with verification code."""
        
        device = await self._get_mfa_device(device_id)
        if not device or device.device_type != "totp":
            raise ValueError("Invalid device")
        
        # Verify code
        totp = pyotp.TOTP(device.secret_key)
        if totp.verify(verification_code, valid_window=1):
            device.is_verified = True
            await self._update_mfa_device(device)
            
            await self._log_security_event(
                event_type="MFA_DEVICE_VERIFIED",
                severity="low",
                user_id=device.user_id,
                event_data={"device_type": "totp", "device_name": device.device_name}
            )
            
            return True
        
        return False
    
    async def validate_token(self, token: str, token_type: str = "access") -> TokenPayload:
        """Validate JWT token and return payload."""
        
        try:
            # Check if token is revoked
            if token in self._revoked_tokens:
                raise ValueError("Token has been revoked")
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            token_payload = TokenPayload(**payload)
            
            # Verify token type
            if token_payload.type != token_type:
                raise ValueError(f"Invalid token type: expected {token_type}, got {token_payload.type}")
            
            # Check if session is still active
            if token_payload.session_id:
                session = self._active_sessions.get(token_payload.session_id)
                if not session or not session.is_active:
                    raise ValueError("Session is no longer active")
                
                # Update last activity
                session.last_activity = datetime.utcnow()
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token."""
        
        # Validate refresh token
        payload = await self.validate_token(refresh_token, "refresh")
        
        # Generate new access token
        new_access_token = await self._generate_access_token(
            UUID(payload.sub), payload.session_id, payload.mfa_verified
        )
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
    
    async def logout(self, session_id: str, revoke_all_sessions: bool = False) -> None:
        """Logout user and invalidate session."""
        
        if revoke_all_sessions:
            # Revoke all sessions for the user
            session = self._active_sessions.get(session_id)
            if session:
                user_sessions = [
                    s for s in self._active_sessions.values() 
                    if s.user_id == session.user_id
                ]
                for user_session in user_sessions:
                    user_session.is_active = False
                    
                await self._log_security_event(
                    event_type="LOGOUT_ALL_SESSIONS",
                    severity="low",
                    user_id=session.user_id,
                    event_data={"session_count": len(user_sessions)}
                )
        else:
            # Revoke single session
            if session_id in self._active_sessions:
                self._active_sessions[session_id].is_active = False
                
                await self._log_security_event(
                    event_type="LOGOUT",
                    severity="low",
                    user_id=self._active_sessions[session_id].user_id,
                    event_data={"session_id": session_id}
                )
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        
        if len(password) < self.config.password_min_length:
            return False
        
        # Check for required character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    async def _is_account_locked(self, email: str, ip_address: str) -> bool:
        """Check if account is locked due to failed attempts."""
        
        key = f"{email}:{ip_address}"
        attempts = self._login_attempts.get(key, [])
        
        # Remove old attempts
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.config.lockout_duration_minutes)
        recent_attempts = [a for a in attempts if a.timestamp > cutoff_time]
        
        # Count failed attempts
        failed_attempts = [a for a in recent_attempts if not a.success]
        
        return len(failed_attempts) >= self.config.max_login_attempts
    
    async def _record_login_attempt(self, user_id: Optional[UUID], ip_address: str,
                                  success: bool, mfa_required: bool = False,
                                  mfa_success: Optional[bool] = None) -> None:
        """Record login attempt for rate limiting."""
        
        key = f"{user_id}:{ip_address}" if user_id else f"unknown:{ip_address}"
        
        attempt = LoginAttempt(
            user_id=user_id or uuid4(),
            ip_address=ip_address,
            timestamp=datetime.utcnow(),
            success=success,
            mfa_required=mfa_required,
            mfa_success=mfa_success
        )
        
        if key not in self._login_attempts:
            self._login_attempts[key] = []
        
        self._login_attempts[key].append(attempt)
        
        # Keep only recent attempts
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self._login_attempts[key] = [
            a for a in self._login_attempts[key] if a.timestamp > cutoff_time
        ]
    
    async def _create_user_session(self, user_id: UUID, ip_address: str,
                                 user_agent: Optional[str], mfa_verified: bool) -> UserSession:
        """Create new user session."""
        
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified
        )
        
        self._active_sessions[session_id] = session
        
        return session
    
    async def _generate_tokens(self, user_id: UUID, session_id: str,
                             mfa_verified: bool) -> Dict[str, str]:
        """Generate access and refresh tokens."""
        
        access_token = await self._generate_access_token(user_id, session_id, mfa_verified)
        refresh_token = await self._generate_refresh_token(user_id, session_id, mfa_verified)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    
    async def _generate_access_token(self, user_id: UUID, session_id: str,
                                   mfa_verified: bool) -> str:
        """Generate access token."""
        
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.config.access_token_expire_minutes)
        
        payload = TokenPayload(
            sub=str(user_id),
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            jti=str(uuid4()),
            type="access",
            session_id=session_id,
            mfa_verified=mfa_verified
        )
        
        return jwt.encode(
            payload.dict(),
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
    
    async def _generate_refresh_token(self, user_id: UUID, session_id: str,
                                    mfa_verified: bool) -> str:
        """Generate refresh token."""
        
        now = datetime.utcnow()
        expire = now + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = TokenPayload(
            sub=str(user_id),
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            jti=str(uuid4()),
            type="refresh",
            session_id=session_id,
            mfa_verified=mfa_verified
        )
        
        return jwt.encode(
            payload.dict(),
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
    
    async def _create_mfa_challenge(self, user_id: UUID, ip_address: str) -> str:
        """Create MFA challenge for user."""
        
        challenge_id = str(uuid4())
        expires_at = datetime.utcnow() + timedelta(minutes=self.config.mfa_token_expire_minutes)
        
        self._mfa_challenges[challenge_id] = {
            "user_id": user_id,
            "ip_address": ip_address,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at
        }
        
        return challenge_id
    
    async def _log_security_event(self, event_type: str, severity: str,
                                user_id: Optional[UUID] = None,
                                ip_address: str = "unknown",
                                event_data: Optional[Dict[str, Any]] = None) -> None:
        """Log security event."""
        
        event = SecurityEvent(
            event_id=uuid4(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            event_data=event_data or {},
            detected_at=datetime.utcnow()
        )
        
        # In production, store in database and alert if high severity
        logger.info(f"Security event: {event_type} - {severity}")
    
    # Mock database methods (implement with actual database)
    async def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email from database."""
        return None
    
    async def _get_user_by_id(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user by ID from database."""
        return None
    
    async def _get_user_mfa_devices(self, user_id: UUID) -> List[MFADevice]:
        """Get user's MFA devices from database."""
        return []
    
    async def _get_mfa_device(self, device_id: UUID) -> Optional[MFADevice]:
        """Get MFA device by ID."""
        return None
    
    async def _store_mfa_device(self, device: MFADevice) -> None:
        """Store MFA device in database."""
        pass
    
    async def _update_mfa_device(self, device: MFADevice) -> None:
        """Update MFA device in database."""
        pass
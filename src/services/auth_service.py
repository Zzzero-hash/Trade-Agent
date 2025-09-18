"""
JWT Authentication service with MFA support and session management.
"""
import asyncio
import logging
import secrets
import hashlib
import hmac
import base64
import qrcode
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import jwt
import pyotp
import bcrypt
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID, uuid4
import redis.asyncio as redis

from src.models.auth import User, Session, MFADevice, AuthToken
from src.config.settings import get_settings
from src.services.encryption_service import EncryptionService
from src.services.audit_service import AuditService, AuditCategory, AuditLevel, ActionResult
from src.repositories.auth_repository import AuthRepository

logger = logging.getLogger(__name__)


class AuthenticationMethod(str, Enum):
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    API_KEY = "api_key"
    OAUTH = "oauth"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"


class MFAType(str, Enum):
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"    # SMS verification
    EMAIL = "email"  # Email verification


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    mfa_code: Optional[str] = None
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    requires_mfa: bool = False
    mfa_methods: List[MFAType] = []
    session_id: str


class MFASetupRequest(BaseModel):
    user_id: UUID
    mfa_type: MFAType
    phone_number: Optional[str] = None


class MFASetupResponse(BaseModel):
    secret: Optional[str] = None  # For TOTP
    qr_code: Optional[str] = None  # Base64 encoded QR code
    backup_codes: List[str] = []


class PasswordHasher:
    """Secure password hashing using bcrypt."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


class JWTManager:
    """JWT token management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=15)
        self.refresh_token_expire = timedelta(days=30)
    
    def create_access_token(self, user_id: UUID, session_id: str, scopes: List[str] = None) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        payload = {
            "sub": str(user_id),
            "session_id": session_id,
            "iat": now,
            "exp": now + self.access_token_expire,
            "type": "access",
            "scopes": scopes or []
        }
        
        return jwt.encode(payload, self.settings.jwt_secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: UUID, session_id: str) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        payload = {
            "sub": str(user_id),
            "session_id": session_id,
            "iat": now,
            "exp": now + self.refresh_token_expire,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.settings.jwt_secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")


class MFAManager:
    """Multi-Factor Authentication manager."""
    
    def __init__(self, encryption_service: EncryptionService):
        self.encryption_service = encryption_service
        self.settings = get_settings()
    
    async def setup_totp(self, user_id: UUID, user_email: str) -> MFASetupResponse:
        """Set up TOTP (Time-based One-Time Password) for user."""
        try:
            # Generate secret
            secret = pyotp.random_base32()
            
            # Create TOTP URI for QR code
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user_email,
                issuer_name="AI Trading Platform"
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            qr_image = qr.make_image(fill_color="black", back_color="white")
            qr_buffer = io.BytesIO()
            qr_image.save(qr_buffer, format='PNG')
            qr_code_b64 = base64.b64encode(qr_buffer.getvalue()).decode()
            
            # Generate backup codes
            backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
            
            return MFASetupResponse(
                secret=secret,
                qr_code=qr_code_b64,
                backup_codes=backup_codes
            )
            
        except Exception as e:
            logger.error(f"TOTP setup failed for user {user_id}: {str(e)}")
            raise
    
    def verify_totp(self, secret: str, code: str, window: int = 1) -> bool:
        """Verify TOTP code."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=window)
        except Exception as e:
            logger.error(f"TOTP verification failed: {str(e)}")
            return False
    
    async def send_sms_code(self, phone_number: str) -> str:
        """Send SMS verification code."""
        try:
            # Generate 6-digit code
            code = f"{secrets.randbelow(1000000):06d}"
            
            # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
            # For now, just log the code
            logger.info(f"SMS code for {phone_number}: {code}")
            
            # Store code in Redis with expiration
            redis_client = redis.from_url(self.settings.redis_url)
            await redis_client.setex(f"sms_code:{phone_number}", 300, code)  # 5 minutes
            
            return code
            
        except Exception as e:
            logger.error(f"SMS code generation failed for {phone_number}: {str(e)}")
            raise
    
    async def verify_sms_code(self, phone_number: str, code: str) -> bool:
        """Verify SMS code."""
        try:
            redis_client = redis.from_url(self.settings.redis_url)
            stored_code = await redis_client.get(f"sms_code:{phone_number}")
            
            if not stored_code:
                return False
            
            if stored_code.decode() == code:
                # Delete code after successful verification
                await redis_client.delete(f"sms_code:{phone_number}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"SMS code verification failed: {str(e)}")
            return False


class SessionManager:
    """Session management with Redis backend."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = redis.from_url(self.settings.redis_url)
        self.session_expire = timedelta(hours=24)
    
    async def create_session(
        self,
        user_id: UUID,
        ip_address: str,
        user_agent: str,
        device_fingerprint: Optional[str] = None
    ) -> Session:
        """Create new user session."""
        try:
            session = Session(
                session_id=str(uuid4()),
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                expires_at=datetime.utcnow() + self.session_expire,
                status=SessionStatus.ACTIVE
            )
            
            # Store in Redis
            session_key = f"session:{session.session_id}"
            session_data = session.dict()
            session_data["created_at"] = session.created_at.isoformat()
            session_data["last_activity"] = session.last_activity.isoformat()
            session_data["expires_at"] = session.expires_at.isoformat()
            session_data["user_id"] = str(session.user_id)
            
            await self.redis_client.setex(
                session_key,
                int(self.session_expire.total_seconds()),
                json.dumps(session_data)
            )
            
            # Track user sessions
            user_sessions_key = f"user_sessions:{user_id}"
            await self.redis_client.sadd(user_sessions_key, session.session_id)
            await self.redis_client.expire(user_sessions_key, int(self.session_expire.total_seconds()))
            
            return session
            
        except Exception as e:
            logger.error(f"Session creation failed for user {user_id}: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        try:
            session_key = f"session:{session_id}"
            session_data = await self.redis_client.get(session_key)
            
            if not session_data:
                return None
            
            data = json.loads(session_data)
            data["user_id"] = UUID(data["user_id"])
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["last_activity"] = datetime.fromisoformat(data["last_activity"])
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
            
            return Session(**data)
            
        except Exception as e:
            logger.error(f"Session retrieval failed for {session_id}: {str(e)}")
            return None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.last_activity = datetime.utcnow()
            
            # Update in Redis
            session_key = f"session:{session_id}"
            session_data = session.dict()
            session_data["created_at"] = session.created_at.isoformat()
            session_data["last_activity"] = session.last_activity.isoformat()
            session_data["expires_at"] = session.expires_at.isoformat()
            session_data["user_id"] = str(session.user_id)
            
            await self.redis_client.setex(
                session_key,
                int(self.session_expire.total_seconds()),
                json.dumps(session_data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Session activity update failed for {session_id}: {str(e)}")
            return False
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Remove from Redis
            session_key = f"session:{session_id}"
            await self.redis_client.delete(session_key)
            
            # Remove from user sessions set
            user_sessions_key = f"user_sessions:{session.user_id}"
            await self.redis_client.srem(user_sessions_key, session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Session revocation failed for {session_id}: {str(e)}")
            return False
    
    async def revoke_all_user_sessions(self, user_id: UUID) -> int:
        """Revoke all sessions for a user."""
        try:
            user_sessions_key = f"user_sessions:{user_id}"
            session_ids = await self.redis_client.smembers(user_sessions_key)
            
            revoked_count = 0
            for session_id in session_ids:
                if await self.revoke_session(session_id.decode()):
                    revoked_count += 1
            
            return revoked_count
            
        except Exception as e:
            logger.error(f"Bulk session revocation failed for user {user_id}: {str(e)}")
            return 0


class AuthService:
    """Main authentication service."""
    
    def __init__(
        self,
        auth_repo: AuthRepository,
        encryption_service: EncryptionService,
        audit_service: AuditService
    ):
        self.auth_repo = auth_repo
        self.encryption_service = encryption_service
        self.audit_service = audit_service
        self.password_hasher = PasswordHasher()
        self.jwt_manager = JWTManager()
        self.mfa_manager = MFAManager(encryption_service)
        self.session_manager = SessionManager()
        
        # Rate limiting
        self.redis_client = redis.from_url(get_settings().redis_url)
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    async def register_user(
        self,
        email: str,
        password: str,
        full_name: str,
        phone_number: Optional[str] = None
    ) -> User:
        """Register a new user."""
        try:
            # Check if user already exists
            existing_user = await self.auth_repo.get_user_by_email(email)
            if existing_user:
                raise ValueError("User already exists")
            
            # Hash password
            password_hash = self.password_hasher.hash_password(password)
            
            # Create user
            user = User(
                user_id=uuid4(),
                email=email,
                password_hash=password_hash,
                full_name=full_name,
                phone_number=phone_number,
                is_active=True,
                is_verified=False,
                created_at=datetime.utcnow(),
                mfa_enabled=False
            )
            
            # Store user
            await self.auth_repo.create_user(user)
            
            # Log registration
            await self.audit_service.log_authentication_event(
                action="user_registration",
                result=ActionResult.SUCCESS,
                customer_id=user.user_id,
                details={"email": email, "full_name": full_name}
            )
            
            logger.info(f"User registered successfully: {email}")
            return user
            
        except Exception as e:
            await self.audit_service.log_authentication_event(
                action="user_registration",
                result=ActionResult.FAILURE,
                details={"email": email, "error": str(e)}
            )
            logger.error(f"User registration failed for {email}: {str(e)}")
            raise
    
    async def authenticate_user(self, request: LoginRequest) -> LoginResponse:
        """Authenticate user with email/password and optional MFA."""
        try:
            # Check rate limiting
            if await self._is_rate_limited(request.email, request.ip_address):
                raise ValueError("Too many login attempts. Please try again later.")
            
            # Get user
            user = await self.auth_repo.get_user_by_email(request.email)
            if not user or not user.is_active:
                await self._record_failed_attempt(request.email, request.ip_address)
                raise ValueError("Invalid credentials")
            
            # Verify password
            if not self.password_hasher.verify_password(request.password, user.password_hash):
                await self._record_failed_attempt(request.email, request.ip_address)
                await self.audit_service.log_authentication_event(
                    action="login_attempt",
                    result=ActionResult.FAILURE,
                    customer_id=user.user_id,
                    ip_address=request.ip_address,
                    user_agent=request.user_agent,
                    details={"reason": "invalid_password"}
                )
                raise ValueError("Invalid credentials")
            
            # Check if MFA is required
            if user.mfa_enabled and not request.mfa_code:
                mfa_devices = await self.auth_repo.get_user_mfa_devices(user.user_id)
                mfa_methods = [device.mfa_type for device in mfa_devices]
                
                return LoginResponse(
                    access_token="",
                    refresh_token="",
                    expires_in=0,
                    requires_mfa=True,
                    mfa_methods=mfa_methods,
                    session_id=""
                )
            
            # Verify MFA if provided
            if user.mfa_enabled and request.mfa_code:
                if not await self._verify_mfa(user.user_id, request.mfa_code):
                    await self._record_failed_attempt(request.email, request.ip_address)
                    await self.audit_service.log_authentication_event(
                        action="mfa_verification",
                        result=ActionResult.FAILURE,
                        customer_id=user.user_id,
                        ip_address=request.ip_address,
                        details={"reason": "invalid_mfa_code"}
                    )
                    raise ValueError("Invalid MFA code")
            
            # Create session
            session = await self.session_manager.create_session(
                user_id=user.user_id,
                ip_address=request.ip_address or "",
                user_agent=request.user_agent or "",
                device_fingerprint=request.device_fingerprint
            )
            
            # Generate tokens
            access_token = self.jwt_manager.create_access_token(
                user_id=user.user_id,
                session_id=session.session_id,
                scopes=["read", "write", "trade"]
            )
            
            refresh_token = self.jwt_manager.create_refresh_token(
                user_id=user.user_id,
                session_id=session.session_id
            )
            
            # Clear failed attempts
            await self._clear_failed_attempts(request.email, request.ip_address)
            
            # Update last login
            await self.auth_repo.update_user_last_login(user.user_id)
            
            # Log successful login
            await self.audit_service.log_authentication_event(
                action="login_success",
                result=ActionResult.SUCCESS,
                customer_id=user.user_id,
                ip_address=request.ip_address,
                user_agent=request.user_agent,
                details={"session_id": session.session_id}
            )
            
            return LoginResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=int(self.jwt_manager.access_token_expire.total_seconds()),
                session_id=session.session_id
            )
            
        except Exception as e:
            logger.error(f"Authentication failed for {request.email}: {str(e)}")
            raise
    
    async def setup_mfa(self, user_id: UUID, mfa_type: MFAType, phone_number: Optional[str] = None) -> MFASetupResponse:
        """Set up MFA for user."""
        try:
            user = await self.auth_repo.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")
            
            if mfa_type == MFAType.TOTP:
                setup_response = await self.mfa_manager.setup_totp(user_id, user.email)
                
                # Store MFA device
                mfa_device = MFADevice(
                    device_id=uuid4(),
                    user_id=user_id,
                    mfa_type=mfa_type,
                    secret=setup_response.secret,
                    backup_codes=setup_response.backup_codes,
                    is_active=False,  # Activated after verification
                    created_at=datetime.utcnow()
                )
                
                # Encrypt secret before storage
                encrypted_secret = await self.encryption_service.encrypt_api_credentials({
                    "secret": setup_response.secret
                })
                mfa_device.secret = encrypted_secret["secret"]
                
                await self.auth_repo.create_mfa_device(mfa_device)
                
                return setup_response
                
            elif mfa_type == MFAType.SMS:
                if not phone_number:
                    raise ValueError("Phone number required for SMS MFA")
                
                # Store MFA device
                mfa_device = MFADevice(
                    device_id=uuid4(),
                    user_id=user_id,
                    mfa_type=mfa_type,
                    phone_number=phone_number,
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                
                await self.auth_repo.create_mfa_device(mfa_device)
                
                return MFASetupResponse()
            
            else:
                raise ValueError(f"Unsupported MFA type: {mfa_type}")
                
        except Exception as e:
            logger.error(f"MFA setup failed for user {user_id}: {str(e)}")
            raise
    
    async def verify_token(self, token: str) -> Tuple[UUID, str]:
        """Verify JWT token and return user ID and session ID."""
        try:
            payload = self.jwt_manager.verify_token(token)
            user_id = UUID(payload["sub"])
            session_id = payload["session_id"]
            
            # Verify session is still active
            session = await self.session_manager.get_session(session_id)
            if not session or session.status != SessionStatus.ACTIVE:
                raise ValueError("Session not active")
            
            # Update session activity
            await self.session_manager.update_session_activity(session_id)
            
            return user_id, session_id
            
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise
    
    async def refresh_token(self, refresh_token: str) -> LoginResponse:
        """Refresh access token using refresh token."""
        try:
            payload = self.jwt_manager.verify_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")
            
            user_id = UUID(payload["sub"])
            session_id = payload["session_id"]
            
            # Verify session
            session = await self.session_manager.get_session(session_id)
            if not session or session.status != SessionStatus.ACTIVE:
                raise ValueError("Session not active")
            
            # Generate new access token
            access_token = self.jwt_manager.create_access_token(
                user_id=user_id,
                session_id=session_id,
                scopes=["read", "write", "trade"]
            )
            
            return LoginResponse(
                access_token=access_token,
                refresh_token=refresh_token,  # Keep same refresh token
                expires_in=int(self.jwt_manager.access_token_expire.total_seconds()),
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise
    
    async def logout(self, session_id: str) -> bool:
        """Logout user by revoking session."""
        try:
            success = await self.session_manager.revoke_session(session_id)
            
            if success:
                await self.audit_service.log_authentication_event(
                    action="logout",
                    result=ActionResult.SUCCESS,
                    details={"session_id": session_id}
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Logout failed for session {session_id}: {str(e)}")
            return False
    
    async def _verify_mfa(self, user_id: UUID, mfa_code: str) -> bool:
        """Verify MFA code for user."""
        try:
            mfa_devices = await self.auth_repo.get_user_mfa_devices(user_id)
            
            for device in mfa_devices:
                if not device.is_active:
                    continue
                
                if device.mfa_type == MFAType.TOTP:
                    # Decrypt secret
                    decrypted_creds = await self.encryption_service.decrypt_api_credentials({
                        "secret": device.secret
                    })
                    secret = decrypted_creds["secret"]
                    
                    if self.mfa_manager.verify_totp(secret, mfa_code):
                        return True
                
                elif device.mfa_type == MFAType.SMS:
                    if await self.mfa_manager.verify_sms_code(device.phone_number, mfa_code):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"MFA verification failed for user {user_id}: {str(e)}")
            return False
    
    async def _is_rate_limited(self, email: str, ip_address: Optional[str]) -> bool:
        """Check if login attempts are rate limited."""
        try:
            # Check email-based rate limiting
            email_key = f"login_attempts:email:{email}"
            email_attempts = await self.redis_client.get(email_key)
            
            if email_attempts and int(email_attempts) >= self.max_login_attempts:
                return True
            
            # Check IP-based rate limiting
            if ip_address:
                ip_key = f"login_attempts:ip:{ip_address}"
                ip_attempts = await self.redis_client.get(ip_key)
                
                if ip_attempts and int(ip_attempts) >= self.max_login_attempts * 2:  # Higher limit for IP
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return False
    
    async def _record_failed_attempt(self, email: str, ip_address: Optional[str]) -> None:
        """Record failed login attempt."""
        try:
            lockout_seconds = int(self.lockout_duration.total_seconds())
            
            # Record email-based attempt
            email_key = f"login_attempts:email:{email}"
            await self.redis_client.incr(email_key)
            await self.redis_client.expire(email_key, lockout_seconds)
            
            # Record IP-based attempt
            if ip_address:
                ip_key = f"login_attempts:ip:{ip_address}"
                await self.redis_client.incr(ip_key)
                await self.redis_client.expire(ip_key, lockout_seconds)
                
        except Exception as e:
            logger.error(f"Failed to record login attempt: {str(e)}")
    
    async def _clear_failed_attempts(self, email: str, ip_address: Optional[str]) -> None:
        """Clear failed login attempts after successful login."""
        try:
            email_key = f"login_attempts:email:{email}"
            await self.redis_client.delete(email_key)
            
            if ip_address:
                ip_key = f"login_attempts:ip:{ip_address}"
                await self.redis_client.delete(ip_key)
                
        except Exception as e:
            logger.error(f"Failed to clear login attempts: {str(e)}")
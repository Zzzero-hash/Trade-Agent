"""
Authentication and authorization system for the trading platform.

This module provides JWT-based authentication, user management,
and role-based access control.

Requirements: 11.2, 11.6
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
import jwt
from enum import Enum

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


class UserRole(str, Enum):
    """User roles for access control."""
    ADMIN = "admin"
    PREMIUM = "premium"
    FREE = "free"
    TRIAL = "trial"


class User(BaseModel):
    """User model for authentication."""
    id: str
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    trial_expires_at: Optional[datetime] = None
    daily_signal_count: int = 0
    daily_signal_limit: int = 5


class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    email: str
    role: UserRole
    exp: datetime


# Mock user database (in production, use actual database)
fake_users_db: Dict[str, Dict[str, Any]] = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=settings.api.jwt_expiry_hours)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.api.jwt_secret, 
        algorithm="HS256"
    )
    
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token."""
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token, 
            settings.api.jwt_secret, 
            algorithms=["HS256"]
        )
        
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        exp: float = payload.get("exp")
        
        if user_id is None or email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        return TokenData(
            user_id=user_id,
            email=email,
            role=UserRole(role),
            exp=datetime.fromtimestamp(exp, tz=timezone.utc)
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email from database."""
    user_data = fake_users_db.get(email)
    if user_data:
        return User(**user_data)
    return None


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID from database."""
    for user_data in fake_users_db.values():
        if user_data["id"] == user_id:
            return User(**user_data)
    return None


def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user = get_user_by_email(email)
    if not user:
        return None
    
    user_data = fake_users_db[email]
    if not verify_password(password, user_data["password_hash"]):
        return None
    
    return user


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    user = get_user_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user


def require_role(required_role: UserRole):
    """Dependency to require specific user role."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        role_hierarchy = {
            UserRole.ADMIN: 4,
            UserRole.PREMIUM: 3,
            UserRole.FREE: 2,
            UserRole.TRIAL: 1
        }
        
        if role_hierarchy.get(current_user.role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker


def check_usage_limits(user: User) -> None:
    """Check if user has exceeded usage limits."""
    if user.role == UserRole.TRIAL:
        # Check trial expiration
        if user.trial_expires_at and datetime.now(timezone.utc) > user.trial_expires_at:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Trial period has expired. Please upgrade to continue."
            )
    
    if user.role in [UserRole.FREE, UserRole.TRIAL]:
        # Check daily signal limits
        if user.daily_signal_count >= user.daily_signal_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily signal limit ({user.daily_signal_limit}) exceeded. Upgrade for unlimited signals."
            )


# Authentication endpoints

@router.post("/register", response_model=Token)
async def register_user(user_data: UserCreate) -> Token:
    """
    Register a new user account.
    
    Creates a new user with trial access for 7 days.
    """
    try:
        # Check if user already exists
        if get_user_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user_id = f"user_{len(fake_users_db) + 1}"
        password_hash = get_password_hash(user_data.password)
        
        now = datetime.now(timezone.utc)
        trial_expires = now + timedelta(days=7)
        
        new_user_data = {
            "id": user_id,
            "email": user_data.email,
            "username": user_data.username,
            "password_hash": password_hash,
            "role": UserRole.TRIAL,
            "is_active": True,
            "created_at": now,
            "last_login": None,
            "trial_expires_at": trial_expires,
            "daily_signal_count": 0,
            "daily_signal_limit": 5
        }
        
        fake_users_db[user_data.email] = new_user_data
        user = User(**new_user_data)
        
        # Create access token
        access_token = create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "role": user.role
            }
        )
        
        # Record metrics
        metrics.increment_counter("user_registrations", 1)
        
        logger.info(f"New user registered: {user.email}")
        
        return Token(
            access_token=access_token,
            expires_in=get_settings().api.jwt_expiry_hours * 3600,
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login_user(login_data: UserLogin) -> Token:
    """
    Authenticate user and return access token.
    """
    try:
        # Authenticate user
        user = authenticate_user(login_data.email, login_data.password)
        if not user:
            metrics.increment_counter("login_failures", 1)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Update last login
        fake_users_db[login_data.email]["last_login"] = datetime.now(timezone.utc)
        
        # Create access token
        access_token = create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "role": user.role
            }
        )
        
        # Record metrics
        metrics.increment_counter("successful_logins", 1)
        
        logger.info(f"User logged in: {user.email}")
        
        return Token(
            access_token=access_token,
            expires_in=get_settings().api.jwt_expiry_hours * 3600,
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current user information.
    """
    return current_user


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_user)) -> Token:
    """
    Refresh access token for authenticated user.
    """
    try:
        # Create new access token
        access_token = create_access_token(
            data={
                "sub": current_user.id,
                "email": current_user.email,
                "role": current_user.role
            }
        )
        
        return Token(
            access_token=access_token,
            expires_in=get_settings().api.jwt_expiry_hours * 3600,
            user=current_user
        )
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    """
    Logout user (client should discard token).
    """
    logger.info(f"User logged out: {current_user.email}")
    
    return {"message": "Successfully logged out"}


@router.post("/upgrade")
async def upgrade_user(
    target_role: UserRole,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upgrade user to premium role.
    
    In production, this would integrate with payment processing.
    """
    try:
        if target_role not in [UserRole.PREMIUM, UserRole.FREE]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid target role"
            )
        
        if current_user.role == UserRole.PREMIUM:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already premium"
            )
        
        # Update user role
        fake_users_db[current_user.email]["role"] = target_role
        
        # Update limits based on role
        if target_role == UserRole.PREMIUM:
            fake_users_db[current_user.email]["daily_signal_limit"] = 999999  # Unlimited
        elif target_role == UserRole.FREE:
            fake_users_db[current_user.email]["daily_signal_limit"] = 5
        
        # Reset daily count
        fake_users_db[current_user.email]["daily_signal_count"] = 0
        
        logger.info(f"User upgraded to {target_role}: {current_user.email}")
        
        return {
            "message": f"Successfully upgraded to {target_role}",
            "new_role": target_role,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User upgrade failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upgrade failed"
        )


@router.get("/usage")
async def get_usage_stats(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get current usage statistics for the user.
    """
    return {
        "user_id": current_user.id,
        "role": current_user.role,
        "daily_signal_count": current_user.daily_signal_count,
        "daily_signal_limit": current_user.daily_signal_limit,
        "trial_expires_at": current_user.trial_expires_at.isoformat() if current_user.trial_expires_at else None,
        "is_trial_expired": (
            current_user.trial_expires_at and 
            datetime.now(timezone.utc) > current_user.trial_expires_at
        ) if current_user.trial_expires_at else False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
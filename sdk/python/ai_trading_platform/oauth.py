"""
OAuth2 integration for third-party applications
"""

import asyncio
import secrets
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import jwt
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .exceptions import AuthenticationError, AuthorizationError, ValidationError


class GrantType(str, Enum):
    """OAuth2 grant types"""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


class TokenType(str, Enum):
    """OAuth2 token types"""
    BEARER = "Bearer"


@dataclass
class OAuthClient:
    """OAuth2 client configuration"""
    client_id: str
    client_secret: str
    name: str
    redirect_uris: List[str]
    scopes: List[str]
    grant_types: List[GrantType]
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AuthorizationCode:
    """OAuth2 authorization code"""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scopes: List[str]
    expires_at: datetime
    used: bool = False


@dataclass
class AccessToken:
    """OAuth2 access token"""
    token: str
    client_id: str
    user_id: str
    scopes: List[str]
    expires_at: datetime
    token_type: TokenType = TokenType.BEARER


@dataclass
class RefreshToken:
    """OAuth2 refresh token"""
    token: str
    client_id: str
    user_id: str
    scopes: List[str]
    expires_at: datetime


class OAuthScope:
    """OAuth2 scopes"""
    READ_SIGNALS = "read:signals"
    WRITE_SIGNALS = "write:signals"
    READ_PORTFOLIO = "read:portfolio"
    WRITE_PORTFOLIO = "write:portfolio"
    READ_MARKET_DATA = "read:market_data"
    READ_RISK = "read:risk"
    WRITE_RISK = "write:risk"
    READ_MONITORING = "read:monitoring"
    ADMIN = "admin"
    
    @classmethod
    def get_all_scopes(cls) -> List[str]:
        """Get all available scopes"""
        return [
            cls.READ_SIGNALS,
            cls.WRITE_SIGNALS,
            cls.READ_PORTFOLIO,
            cls.WRITE_PORTFOLIO,
            cls.READ_MARKET_DATA,
            cls.READ_RISK,
            cls.WRITE_RISK,
            cls.READ_MONITORING,
            cls.ADMIN
        ]
    
    @classmethod
    def get_scope_description(cls, scope: str) -> str:
        """Get human-readable scope description"""
        descriptions = {
            cls.READ_SIGNALS: "Read trading signals",
            cls.WRITE_SIGNALS: "Generate trading signals",
            cls.READ_PORTFOLIO: "Read portfolio information",
            cls.WRITE_PORTFOLIO: "Modify portfolio",
            cls.READ_MARKET_DATA: "Access market data",
            cls.READ_RISK: "Read risk metrics",
            cls.WRITE_RISK: "Modify risk settings",
            cls.READ_MONITORING: "Access monitoring data",
            cls.ADMIN: "Full administrative access"
        }
        return descriptions.get(scope, scope)


class OAuthManager:
    """Manages OAuth2 clients, tokens, and authorization flow"""
    
    def __init__(self, jwt_secret: str, token_expiry_minutes: int = 60):
        self.jwt_secret = jwt_secret
        self.token_expiry_minutes = token_expiry_minutes
        self.clients: Dict[str, OAuthClient] = {}
        self.authorization_codes: Dict[str, AuthorizationCode] = {}
        self.access_tokens: Dict[str, AccessToken] = {}
        self.refresh_tokens: Dict[str, RefreshToken] = {}
    
    def register_client(self, client: OAuthClient) -> str:
        """Register OAuth2 client"""
        self.clients[client.client_id] = client
        return client.client_id
    
    def get_client(self, client_id: str) -> Optional[OAuthClient]:
        """Get OAuth2 client"""
        return self.clients.get(client_id)
    
    def validate_client(self, client_id: str, client_secret: Optional[str] = None) -> bool:
        """Validate OAuth2 client"""
        client = self.get_client(client_id)
        if not client or not client.active:
            return False
        
        if client_secret is not None:
            return client.client_secret == client_secret
        
        return True
    
    def generate_authorization_code(self, client_id: str, user_id: str, 
                                  redirect_uri: str, scopes: List[str]) -> str:
        """Generate authorization code"""
        code = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=10)  # 10 minutes
        
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            expires_at=expires_at
        )
        
        self.authorization_codes[code] = auth_code
        return code
    
    def exchange_authorization_code(self, code: str, client_id: str, 
                                  redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        auth_code = self.authorization_codes.get(code)
        
        if not auth_code:
            raise AuthenticationError("Invalid authorization code")
        
        if auth_code.used:
            raise AuthenticationError("Authorization code already used")
        
        if auth_code.expires_at < datetime.now():
            raise AuthenticationError("Authorization code expired")
        
        if auth_code.client_id != client_id:
            raise AuthenticationError("Client ID mismatch")
        
        if auth_code.redirect_uri != redirect_uri:
            raise AuthenticationError("Redirect URI mismatch")
        
        # Mark code as used
        auth_code.used = True
        
        # Generate tokens
        access_token = self.generate_access_token(
            client_id, auth_code.user_id, auth_code.scopes
        )
        refresh_token = self.generate_refresh_token(
            client_id, auth_code.user_id, auth_code.scopes
        )
        
        return {
            "access_token": access_token.token,
            "token_type": access_token.token_type.value,
            "expires_in": int((access_token.expires_at - datetime.now()).total_seconds()),
            "refresh_token": refresh_token.token,
            "scope": " ".join(access_token.scopes)
        }
    
    def generate_access_token(self, client_id: str, user_id: str, 
                            scopes: List[str]) -> AccessToken:
        """Generate access token"""
        expires_at = datetime.now() + timedelta(minutes=self.token_expiry_minutes)
        
        payload = {
            "sub": user_id,
            "client_id": client_id,
            "scopes": scopes,
            "exp": int(expires_at.timestamp()),
            "iat": int(datetime.now().timestamp()),
            "token_type": "access"
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        access_token = AccessToken(
            token=token,
            client_id=client_id,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at
        )
        
        self.access_tokens[token] = access_token
        return access_token
    
    def generate_refresh_token(self, client_id: str, user_id: str, 
                             scopes: List[str]) -> RefreshToken:
        """Generate refresh token"""
        expires_at = datetime.now() + timedelta(days=30)  # 30 days
        
        token = secrets.token_urlsafe(64)
        
        refresh_token = RefreshToken(
            token=token,
            client_id=client_id,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at
        )
        
        self.refresh_tokens[token] = refresh_token
        return refresh_token
    
    def validate_access_token(self, token: str) -> Optional[AccessToken]:
        """Validate access token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            if payload.get("token_type") != "access":
                return None
            
            access_token = self.access_tokens.get(token)
            if not access_token:
                return None
            
            if access_token.expires_at < datetime.now():
                # Remove expired token
                del self.access_tokens[token]
                return None
            
            return access_token
            
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str, client_id: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        refresh_token_obj = self.refresh_tokens.get(refresh_token)
        
        if not refresh_token_obj:
            raise AuthenticationError("Invalid refresh token")
        
        if refresh_token_obj.expires_at < datetime.now():
            # Remove expired refresh token
            del self.refresh_tokens[refresh_token]
            raise AuthenticationError("Refresh token expired")
        
        if refresh_token_obj.client_id != client_id:
            raise AuthenticationError("Client ID mismatch")
        
        # Generate new access token
        access_token = self.generate_access_token(
            client_id, refresh_token_obj.user_id, refresh_token_obj.scopes
        )
        
        return {
            "access_token": access_token.token,
            "token_type": access_token.token_type.value,
            "expires_in": int((access_token.expires_at - datetime.now()).total_seconds()),
            "scope": " ".join(access_token.scopes)
        }
    
    def revoke_token(self, token: str) -> bool:
        """Revoke access or refresh token"""
        # Try access token first
        if token in self.access_tokens:
            del self.access_tokens[token]
            return True
        
        # Try refresh token
        if token in self.refresh_tokens:
            del self.refresh_tokens[token]
            return True
        
        return False
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens"""
        now = datetime.now()
        
        # Clean up authorization codes
        expired_codes = [
            code for code, auth_code in self.authorization_codes.items()
            if auth_code.expires_at < now
        ]
        for code in expired_codes:
            del self.authorization_codes[code]
        
        # Clean up access tokens
        expired_access_tokens = [
            token for token, access_token in self.access_tokens.items()
            if access_token.expires_at < now
        ]
        for token in expired_access_tokens:
            del self.access_tokens[token]
        
        # Clean up refresh tokens
        expired_refresh_tokens = [
            token for token, refresh_token in self.refresh_tokens.items()
            if refresh_token.expires_at < now
        ]
        for token in expired_refresh_tokens:
            del self.refresh_tokens[token]


# FastAPI OAuth2 integration
security = HTTPBearer()

def create_oauth_router(oauth_manager: OAuthManager) -> FastAPI:
    """Create FastAPI router for OAuth2 endpoints"""
    app = FastAPI(title="OAuth2 Authorization Server")
    
    @app.get("/oauth/authorize")
    async def authorize(
        request: Request,
        response_type: str,
        client_id: str,
        redirect_uri: str,
        scope: str = "",
        state: str = ""
    ):
        """OAuth2 authorization endpoint"""
        # Validate response type
        if response_type != "code":
            raise HTTPException(status_code=400, detail="Unsupported response type")
        
        # Validate client
        client = oauth_manager.get_client(client_id)
        if not client:
            raise HTTPException(status_code=400, detail="Invalid client")
        
        # Validate redirect URI
        if redirect_uri not in client.redirect_uris:
            raise HTTPException(status_code=400, detail="Invalid redirect URI")
        
        # Parse scopes
        requested_scopes = scope.split() if scope else []
        
        # Validate scopes
        valid_scopes = OAuthScope.get_all_scopes()
        invalid_scopes = [s for s in requested_scopes if s not in valid_scopes]
        if invalid_scopes:
            raise HTTPException(status_code=400, detail=f"Invalid scopes: {invalid_scopes}")
        
        # In a real implementation, you would:
        # 1. Check if user is authenticated
        # 2. Show consent screen
        # 3. Get user approval
        # For this example, we'll assume user is authenticated and approved
        
        # Mock user ID (in real implementation, get from session)
        user_id = "mock_user_123"
        
        # Generate authorization code
        code = oauth_manager.generate_authorization_code(
            client_id, user_id, redirect_uri, requested_scopes
        )
        
        # Redirect back to client
        redirect_url = f"{redirect_uri}?code={code}"
        if state:
            redirect_url += f"&state={state}"
        
        return RedirectResponse(url=redirect_url)
    
    @app.post("/oauth/token")
    async def token(
        grant_type: str = Form(...),
        code: str = Form(None),
        redirect_uri: str = Form(None),
        client_id: str = Form(...),
        client_secret: str = Form(...),
        refresh_token: str = Form(None)
    ):
        """OAuth2 token endpoint"""
        # Validate client
        if not oauth_manager.validate_client(client_id, client_secret):
            raise HTTPException(status_code=401, detail="Invalid client credentials")
        
        if grant_type == GrantType.AUTHORIZATION_CODE.value:
            if not code or not redirect_uri:
                raise HTTPException(status_code=400, detail="Missing code or redirect_uri")
            
            try:
                token_data = oauth_manager.exchange_authorization_code(
                    code, client_id, redirect_uri
                )
                return token_data
            except AuthenticationError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        elif grant_type == GrantType.REFRESH_TOKEN.value:
            if not refresh_token:
                raise HTTPException(status_code=400, detail="Missing refresh_token")
            
            try:
                token_data = oauth_manager.refresh_access_token(refresh_token, client_id)
                return token_data
            except AuthenticationError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        elif grant_type == GrantType.CLIENT_CREDENTIALS.value:
            # Client credentials flow (for machine-to-machine)
            client = oauth_manager.get_client(client_id)
            if not client:
                raise HTTPException(status_code=400, detail="Invalid client")
            
            # Generate access token for client
            access_token = oauth_manager.generate_access_token(
                client_id, client_id, client.scopes  # Use client_id as user_id for M2M
            )
            
            return {
                "access_token": access_token.token,
                "token_type": access_token.token_type.value,
                "expires_in": int((access_token.expires_at - datetime.now()).total_seconds()),
                "scope": " ".join(access_token.scopes)
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported grant type")
    
    @app.post("/oauth/revoke")
    async def revoke(
        token: str = Form(...),
        client_id: str = Form(...),
        client_secret: str = Form(...)
    ):
        """OAuth2 token revocation endpoint"""
        # Validate client
        if not oauth_manager.validate_client(client_id, client_secret):
            raise HTTPException(status_code=401, detail="Invalid client credentials")
        
        # Revoke token
        revoked = oauth_manager.revoke_token(token)
        
        return {"revoked": revoked}
    
    @app.get("/oauth/userinfo")
    async def userinfo(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """OAuth2 user info endpoint"""
        token = credentials.credentials
        access_token = oauth_manager.validate_access_token(token)
        
        if not access_token:
            raise HTTPException(status_code=401, detail="Invalid access token")
        
        # Return user info (mock data)
        return {
            "sub": access_token.user_id,
            "client_id": access_token.client_id,
            "scopes": access_token.scopes,
            "expires_at": access_token.expires_at.isoformat()
        }
    
    @app.get("/oauth/clients")
    async def list_clients():
        """List registered OAuth2 clients (admin endpoint)"""
        return {
            client_id: {
                "name": client.name,
                "redirect_uris": client.redirect_uris,
                "scopes": client.scopes,
                "grant_types": [gt.value for gt in client.grant_types],
                "active": client.active,
                "created_at": client.created_at.isoformat()
            }
            for client_id, client in oauth_manager.clients.items()
        }
    
    @app.post("/oauth/clients")
    async def register_client(
        name: str,
        redirect_uris: List[str],
        scopes: List[str] = None,
        grant_types: List[str] = None
    ):
        """Register new OAuth2 client (admin endpoint)"""
        client_id = f"client_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)
        
        # Default values
        if scopes is None:
            scopes = [OAuthScope.READ_SIGNALS, OAuthScope.READ_PORTFOLIO]
        if grant_types is None:
            grant_types = [GrantType.AUTHORIZATION_CODE.value]
        
        # Convert grant types
        grant_type_enums = []
        for gt in grant_types:
            try:
                grant_type_enums.append(GrantType(gt))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid grant type: {gt}")
        
        client = OAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            name=name,
            redirect_uris=redirect_uris,
            scopes=scopes,
            grant_types=grant_type_enums
        )
        
        oauth_manager.register_client(client)
        
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "name": name,
            "redirect_uris": redirect_uris,
            "scopes": scopes,
            "grant_types": grant_types
        }
    
    return app


# Dependency for validating OAuth2 tokens in protected endpoints
async def get_oauth_token(oauth_manager: OAuthManager, 
                         credentials: HTTPAuthorizationCredentials = Depends(security)) -> AccessToken:
    """Dependency to validate OAuth2 access token"""
    token = credentials.credentials
    access_token = oauth_manager.validate_access_token(token)
    
    if not access_token:
        raise HTTPException(status_code=401, detail="Invalid access token")
    
    return access_token


def require_scope(required_scope: str):
    """Decorator to require specific OAuth2 scope"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get access token from kwargs (injected by dependency)
            access_token = None
            for value in kwargs.values():
                if isinstance(value, AccessToken):
                    access_token = value
                    break
            
            if not access_token:
                raise HTTPException(status_code=401, detail="Access token required")
            
            if required_scope not in access_token.scopes:
                raise HTTPException(status_code=403, detail=f"Scope '{required_scope}' required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
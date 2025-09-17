"""
Authentication manager for the AI Trading Platform SDK
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import httpx

from .exceptions import AuthenticationError, AuthorizationError, NetworkError


class AuthManager:
    """Manages authentication and token lifecycle"""
    
    def __init__(self, base_url: str, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self._refresh_lock = asyncio.Lock()
    
    async def authenticate_with_credentials(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate using username and password"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={
                        "username": username,
                        "password": password,
                        "client_id": self.client_id
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    await self._store_tokens(token_data)
                    return token_data
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid credentials")
                else:
                    raise AuthenticationError(f"Authentication failed: {response.text}")
                    
            except httpx.RequestError as e:
                raise NetworkError(f"Network error during authentication: {e}")
    
    async def authenticate_with_oauth(self, authorization_code: str, 
                                    redirect_uri: str) -> Dict[str, Any]:
        """Authenticate using OAuth2 authorization code"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/oauth/token",
                    json={
                        "grant_type": "authorization_code",
                        "code": authorization_code,
                        "redirect_uri": redirect_uri,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    await self._store_tokens(token_data)
                    return token_data
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid authorization code")
                else:
                    raise AuthenticationError(f"OAuth authentication failed: {response.text}")
                    
            except httpx.RequestError as e:
                raise NetworkError(f"Network error during OAuth authentication: {e}")
    
    async def authenticate_with_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate using API key"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/api-key",
                    headers={"X-API-Key": api_key}
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    await self._store_tokens(token_data)
                    return token_data
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                else:
                    raise AuthenticationError(f"API key authentication failed: {response.text}")
                    
            except httpx.RequestError as e:
                raise NetworkError(f"Network error during API key authentication: {e}")
    
    async def _store_tokens(self, token_data: Dict[str, Any]) -> None:
        """Store authentication tokens"""
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        
        # Calculate expiration time
        expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
    
    async def get_valid_token(self) -> str:
        """Get a valid access token, refreshing if necessary"""
        if not self.access_token:
            raise AuthenticationError("No access token available. Please authenticate first.")
        
        # Check if token is expired or will expire soon (5 minutes buffer)
        if (self.token_expires_at and 
            datetime.now() + timedelta(minutes=5) >= self.token_expires_at):
            await self._refresh_access_token()
        
        return self.access_token
    
    async def _refresh_access_token(self) -> None:
        """Refresh the access token using refresh token"""
        async with self._refresh_lock:
            # Double-check if token still needs refreshing
            if (self.token_expires_at and 
                datetime.now() + timedelta(minutes=5) < self.token_expires_at):
                return
            
            if not self.refresh_token:
                raise AuthenticationError("No refresh token available. Please re-authenticate.")
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/v1/auth/refresh",
                        json={
                            "refresh_token": self.refresh_token,
                            "client_id": self.client_id
                        }
                    )
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        await self._store_tokens(token_data)
                    elif response.status_code == 401:
                        # Refresh token is invalid, clear all tokens
                        self.access_token = None
                        self.refresh_token = None
                        self.token_expires_at = None
                        raise AuthenticationError("Refresh token expired. Please re-authenticate.")
                    else:
                        raise AuthenticationError(f"Token refresh failed: {response.text}")
                        
                except httpx.RequestError as e:
                    raise NetworkError(f"Network error during token refresh: {e}")
    
    async def logout(self) -> None:
        """Logout and invalidate tokens"""
        if self.access_token:
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(
                        f"{self.base_url}/api/v1/auth/logout",
                        headers={"Authorization": f"Bearer {self.access_token}"}
                    )
                except httpx.RequestError:
                    pass  # Ignore network errors during logout
        
        # Clear stored tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return (self.access_token is not None and 
                self.token_expires_at is not None and
                datetime.now() < self.token_expires_at)
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Extract user information from access token"""
        if not self.access_token:
            return None
        
        try:
            # Decode JWT token (without verification for user info)
            payload = jwt.decode(self.access_token, options={"verify_signature": False})
            return {
                "user_id": payload.get("sub"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "roles": payload.get("roles", []),
                "tier": payload.get("tier", "free"),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0))
            }
        except jwt.InvalidTokenError:
            return None
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        token = await self.get_valid_token()
        return {"Authorization": f"Bearer {token}"}
    
    def get_oauth_authorization_url(self, redirect_uri: str, 
                                  state: Optional[str] = None,
                                  scopes: Optional[List[str]] = None) -> str:
        """Generate OAuth2 authorization URL"""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri
        }
        
        if state:
            params["state"] = state
        
        if scopes:
            params["scope"] = " ".join(scopes)
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}/api/v1/auth/oauth/authorize?{query_string}"
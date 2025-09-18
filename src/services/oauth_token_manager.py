"""OAuth token management service with automatic refresh and secure storage"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
import json
import hashlib
import hmac
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import base64
import os

from ..exchanges.broker_base import BrokerCredentials, BrokerType
from ..services.encryption_service import EncryptionService


@dataclass
class TokenInfo:
    """Token information with metadata"""
    access_token: str
    refresh_token: Optional[str]
    expires_at: datetime
    token_type: str = "Bearer"
    scope: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def is_expired(self, buffer_minutes: int = 5) -> bool:
        """Check if token is expired with buffer"""
        buffer_time = datetime.now() + timedelta(minutes=buffer_minutes)
        return buffer_time >= self.expires_at
    
    def time_until_expiry(self) -> timedelta:
        """Get time until token expires"""
        return self.expires_at - datetime.now()


class OAuthTokenManager:
    """Manages OAuth tokens with automatic refresh and secure storage"""
    
    def __init__(self, encryption_service: EncryptionService, storage_path: str = "tokens"):
        self.encryption_service = encryption_service
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        
        # In-memory token cache
        self._token_cache: Dict[str, TokenInfo] = {}
        
        # Token refresh callbacks
        self._refresh_callbacks: Dict[BrokerType, Callable] = {}
        
        # Background refresh tasks
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
    
    def register_refresh_callback(self, broker_type: BrokerType, callback: Callable):
        """Register callback function for token refresh"""
        self._refresh_callbacks[broker_type] = callback
    
    async def store_token(self, account_id: str, broker_type: BrokerType, token_info: TokenInfo) -> bool:
        """Store token securely"""
        try:
            token_key = self._generate_token_key(account_id, broker_type)
            
            # Serialize token info
            token_data = {
                "access_token": token_info.access_token,
                "refresh_token": token_info.refresh_token,
                "expires_at": token_info.expires_at.isoformat(),
                "token_type": token_info.token_type,
                "scope": token_info.scope,
                "created_at": token_info.created_at.isoformat()
            }
            
            # Encrypt and store
            encrypted_data = await self.encryption_service.encrypt_data(json.dumps(token_data))
            
            file_path = os.path.join(self.storage_path, f"{token_key}.token")
            with open(file_path, "wb") as f:
                f.write(encrypted_data)
            
            # Update cache
            self._token_cache[token_key] = token_info
            
            # Start background refresh task
            await self._start_refresh_task(account_id, broker_type, token_info)
            
            self.logger.info(f"Token stored for {broker_type.value} account {account_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing token: {str(e)}")
            return False
    
    async def get_token(self, account_id: str, broker_type: BrokerType) -> Optional[TokenInfo]:
        """Retrieve token from cache or storage"""
        try:
            token_key = self._generate_token_key(account_id, broker_type)
            
            # Check cache first
            if token_key in self._token_cache:
                token_info = self._token_cache[token_key]
                
                # Check if token needs refresh
                if token_info.is_expired():
                    refreshed_token = await self._refresh_token(account_id, broker_type, token_info)
                    if refreshed_token:
                        return refreshed_token
                    else:
                        # Remove expired token
                        await self.remove_token(account_id, broker_type)
                        return None
                
                return token_info
            
            # Load from storage
            file_path = os.path.join(self.storage_path, f"{token_key}.token")
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt and deserialize
            decrypted_data = await self.encryption_service.decrypt_data(encrypted_data)
            token_data = json.loads(decrypted_data)
            
            token_info = TokenInfo(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                expires_at=datetime.fromisoformat(token_data["expires_at"]),
                token_type=token_data.get("token_type", "Bearer"),
                scope=token_data.get("scope"),
                created_at=datetime.fromisoformat(token_data["created_at"])
            )
            
            # Update cache
            self._token_cache[token_key] = token_info
            
            # Check if token needs refresh
            if token_info.is_expired():
                refreshed_token = await self._refresh_token(account_id, broker_type, token_info)
                if refreshed_token:
                    return refreshed_token
                else:
                    await self.remove_token(account_id, broker_type)
                    return None
            
            # Start background refresh task
            await self._start_refresh_task(account_id, broker_type, token_info)
            
            return token_info
            
        except Exception as e:
            self.logger.error(f"Error retrieving token: {str(e)}")
            return None
    
    async def remove_token(self, account_id: str, broker_type: BrokerType) -> bool:
        """Remove token from cache and storage"""
        try:
            token_key = self._generate_token_key(account_id, broker_type)
            
            # Remove from cache
            if token_key in self._token_cache:
                del self._token_cache[token_key]
            
            # Cancel refresh task
            if token_key in self._refresh_tasks:
                self._refresh_tasks[token_key].cancel()
                del self._refresh_tasks[token_key]
            
            # Remove from storage
            file_path = os.path.join(self.storage_path, f"{token_key}.token")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            self.logger.info(f"Token removed for {broker_type.value} account {account_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing token: {str(e)}")
            return False
    
    async def refresh_token_if_needed(self, account_id: str, broker_type: BrokerType) -> Optional[TokenInfo]:
        """Refresh token if it's close to expiry"""
        token_info = await self.get_token(account_id, broker_type)
        if not token_info:
            return None
        
        if token_info.is_expired(buffer_minutes=10):  # Refresh 10 minutes before expiry
            return await self._refresh_token(account_id, broker_type, token_info)
        
        return token_info
    
    async def _refresh_token(self, account_id: str, broker_type: BrokerType, current_token: TokenInfo) -> Optional[TokenInfo]:
        """Refresh token using broker-specific callback"""
        try:
            if broker_type not in self._refresh_callbacks:
                self.logger.error(f"No refresh callback registered for {broker_type.value}")
                return None
            
            if not current_token.refresh_token:
                self.logger.error(f"No refresh token available for {broker_type.value}")
                return None
            
            # Call broker-specific refresh function
            refresh_callback = self._refresh_callbacks[broker_type]
            new_token_data = await refresh_callback(current_token.refresh_token)
            
            if not new_token_data:
                self.logger.error(f"Token refresh failed for {broker_type.value}")
                return None
            
            # Create new token info
            new_token = TokenInfo(
                access_token=new_token_data["access_token"],
                refresh_token=new_token_data.get("refresh_token", current_token.refresh_token),
                expires_at=datetime.now() + timedelta(seconds=new_token_data.get("expires_in", 3600)),
                token_type=new_token_data.get("token_type", "Bearer"),
                scope=new_token_data.get("scope")
            )
            
            # Store refreshed token
            await self.store_token(account_id, broker_type, new_token)
            
            self.logger.info(f"Token refreshed for {broker_type.value} account {account_id}")
            return new_token
            
        except Exception as e:
            self.logger.error(f"Error refreshing token: {str(e)}")
            return None
    
    async def _start_refresh_task(self, account_id: str, broker_type: BrokerType, token_info: TokenInfo):
        """Start background task to refresh token before expiry"""
        token_key = self._generate_token_key(account_id, broker_type)
        
        # Cancel existing task
        if token_key in self._refresh_tasks:
            self._refresh_tasks[token_key].cancel()
        
        # Calculate refresh time (refresh 10 minutes before expiry)
        refresh_time = token_info.expires_at - timedelta(minutes=10)
        delay = (refresh_time - datetime.now()).total_seconds()
        
        if delay > 0:
            task = asyncio.create_task(self._background_refresh(account_id, broker_type, delay))
            self._refresh_tasks[token_key] = task
    
    async def _background_refresh(self, account_id: str, broker_type: BrokerType, delay: float):
        """Background task to refresh token"""
        try:
            await asyncio.sleep(delay)
            
            current_token = await self.get_token(account_id, broker_type)
            if current_token:
                await self._refresh_token(account_id, broker_type, current_token)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Background refresh error: {str(e)}")
    
    def _generate_token_key(self, account_id: str, broker_type: BrokerType) -> str:
        """Generate unique key for token storage"""
        key_data = f"{account_id}:{broker_type.value}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get_all_tokens(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored tokens (for admin/debugging)"""
        tokens = {}
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".token"):
                    token_key = filename[:-6]  # Remove .token extension
                    
                    # Try to find corresponding account info
                    for cache_key, token_info in self._token_cache.items():
                        if cache_key == token_key:
                            tokens[token_key] = {
                                "expires_at": token_info.expires_at.isoformat(),
                                "is_expired": token_info.is_expired(),
                                "time_until_expiry": str(token_info.time_until_expiry()),
                                "token_type": token_info.token_type,
                                "scope": token_info.scope
                            }
                            break
            
            return tokens
            
        except Exception as e:
            self.logger.error(f"Error getting all tokens: {str(e)}")
            return {}
    
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens from storage"""
        try:
            cleaned_count = 0
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".token"):
                    file_path = os.path.join(self.storage_path, filename)
                    
                    try:
                        with open(file_path, "rb") as f:
                            encrypted_data = f.read()
                        
                        decrypted_data = await self.encryption_service.decrypt_data(encrypted_data)
                        token_data = json.loads(decrypted_data)
                        
                        expires_at = datetime.fromisoformat(token_data["expires_at"])
                        
                        # Remove if expired for more than 24 hours
                        if datetime.now() - expires_at > timedelta(hours=24):
                            os.remove(file_path)
                            cleaned_count += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing token file {filename}: {str(e)}")
                        # Remove corrupted files
                        os.remove(file_path)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired/corrupted token files")
                
        except Exception as e:
            self.logger.error(f"Error during token cleanup: {str(e)}")
    
    async def shutdown(self):
        """Shutdown token manager and cancel all tasks"""
        try:
            # Cancel all refresh tasks
            for task in self._refresh_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self._refresh_tasks:
                await asyncio.gather(*self._refresh_tasks.values(), return_exceptions=True)
            
            self._refresh_tasks.clear()
            self._token_cache.clear()
            
            self.logger.info("OAuth token manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
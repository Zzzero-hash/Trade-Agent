"""
WebSocket client for real-time data streaming
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .auth import AuthManager
from .exceptions import WebSocketError, AuthenticationError


class WebSocketClient:
    """WebSocket client for real-time data streaming"""
    
    def __init__(self, base_url: str, user_id: str, auth_manager: AuthManager):
        self.base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.user_id = user_id
        self.auth_manager = auth_manager
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self._message_queue = asyncio.Queue()
        self._listen_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1.0
    
    async def connect(self) -> None:
        """Connect to WebSocket server"""
        try:
            # Get authentication token
            token = await self.auth_manager.get_valid_token()
            
            # Build WebSocket URL
            ws_url = f"{self.base_url}/api/v1/trading/ws/{self.user_id}"
            
            # Connect with authentication header
            extra_headers = {"Authorization": f"Bearer {token}"}
            
            self.websocket = await websockets.connect(
                ws_url,
                extra_headers=extra_headers,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.is_connected = True
            self._reconnect_attempts = 0
            
            # Start listening for messages
            self._listen_task = asyncio.create_task(self._listen_loop())
            
        except Exception as e:
            raise WebSocketError(f"Failed to connect to WebSocket: {e}")
    
    async def close(self) -> None:
        """Close WebSocket connection"""
        self.is_connected = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def _listen_loop(self) -> None:
        """Main listening loop for WebSocket messages"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    await self._message_queue.put(data)
                    
                except ConnectionClosed:
                    if self.is_connected:
                        await self._handle_disconnect()
                    break
                    
                except json.JSONDecodeError as e:
                    # Log invalid JSON but continue listening
                    print(f"Received invalid JSON: {e}")
                    continue
                    
        except Exception as e:
            if self.is_connected:
                await self._handle_disconnect()
    
    async def _handle_disconnect(self) -> None:
        """Handle WebSocket disconnection and attempt reconnection"""
        if self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))  # Exponential backoff
            
            print(f"WebSocket disconnected. Attempting reconnection {self._reconnect_attempts}/{self._max_reconnect_attempts} in {delay}s")
            
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                print("WebSocket reconnected successfully")
            except Exception as e:
                print(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    self.is_connected = False
                    raise WebSocketError(f"Failed to reconnect after {self._max_reconnect_attempts} attempts")
        else:
            self.is_connected = False
            raise WebSocketError("Maximum reconnection attempts exceeded")
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to WebSocket server"""
        if not self.is_connected or not self.websocket:
            raise WebSocketError("WebSocket is not connected")
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            raise WebSocketError(f"Failed to send message: {e}")
    
    async def listen(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Listen for incoming messages"""
        while self.is_connected:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                yield message
            except asyncio.TimeoutError:
                # Continue listening if no message received
                continue
            except Exception as e:
                if self.is_connected:
                    raise WebSocketError(f"Error listening for messages: {e}")
                break
    
    async def subscribe_to_signals(self) -> None:
        """Subscribe to trading signals"""
        await self.send_message({
            "type": "subscribe_signals",
            "timestamp": datetime.now().isoformat()
        })
    
    async def subscribe_to_market_data(self, symbols: List[str]) -> None:
        """Subscribe to market data for specific symbols"""
        await self.send_message({
            "type": "subscribe_market_data",
            "symbols": symbols,
            "timestamp": datetime.now().isoformat()
        })
    
    async def unsubscribe_from_signals(self) -> None:
        """Unsubscribe from trading signals"""
        await self.send_message({
            "type": "unsubscribe_signals",
            "timestamp": datetime.now().isoformat()
        })
    
    async def unsubscribe_from_market_data(self, symbols: Optional[List[str]] = None) -> None:
        """Unsubscribe from market data"""
        message = {
            "type": "unsubscribe_market_data",
            "timestamp": datetime.now().isoformat()
        }
        
        if symbols:
            message["symbols"] = symbols
        
        await self.send_message(message)
    
    async def ping(self) -> None:
        """Send ping to keep connection alive"""
        await self.send_message({
            "type": "ping",
            "timestamp": datetime.now().isoformat()
        })
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "connected": self.is_connected,
            "reconnect_attempts": self._reconnect_attempts,
            "max_reconnect_attempts": self._max_reconnect_attempts,
            "user_id": self.user_id
        }
"""
Webhook system for event-driven integrations with external systems
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import httpx
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .exceptions import WebSocketError, ValidationError
from .models import WebSocketMessage
from .auth import AuthManager


class WebhookEventType(str, Enum):
    """Webhook event types"""
    SIGNAL_GENERATED = "signal_generated"
    PORTFOLIO_UPDATED = "portfolio_updated"
    RISK_ALERT = "risk_alert"
    MODEL_RETRAINED = "model_retrained"
    TRADE_EXECUTED = "trade_executed"
    MARKET_DATA_UPDATE = "market_data_update"
    USER_REGISTERED = "user_registered"
    SUBSCRIPTION_CHANGED = "subscription_changed"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    url: str
    secret: Optional[str] = None
    events: List[WebhookEventType] = None
    active: bool = True
    retry_attempts: int = 3
    timeout: int = 30
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.headers is None:
            self.headers = {}


@dataclass
class WebhookEvent:
    """Webhook event data"""
    id: str
    type: WebhookEventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "ai-trading-platform"
    version: str = "1.0"


class WebhookDelivery:
    """Tracks webhook delivery attempts"""
    
    def __init__(self, event: WebhookEvent, endpoint: WebhookEndpoint):
        self.event = event
        self.endpoint = endpoint
        self.attempts = 0
        self.last_attempt: Optional[datetime] = None
        self.last_response_status: Optional[int] = None
        self.last_error: Optional[str] = None
        self.delivered = False


class WebhookManager:
    """Manages webhook endpoints and event delivery"""
    
    def __init__(self):
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.event_handlers: Dict[WebhookEventType, List[Callable]] = {}
        self.delivery_queue = asyncio.Queue()
        self.delivery_workers: List[asyncio.Task] = []
        self.running = False
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def start(self, num_workers: int = 3):
        """Start webhook delivery workers"""
        if self.running:
            return
        
        self.running = True
        self._http_client = httpx.AsyncClient()
        
        # Start delivery workers
        for i in range(num_workers):
            worker = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            self.delivery_workers.append(worker)
    
    async def stop(self):
        """Stop webhook delivery workers"""
        self.running = False
        
        # Cancel workers
        for worker in self.delivery_workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.delivery_workers, return_exceptions=True)
        self.delivery_workers.clear()
        
        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    def add_endpoint(self, endpoint_id: str, endpoint: WebhookEndpoint):
        """Add webhook endpoint"""
        self.endpoints[endpoint_id] = endpoint
    
    def remove_endpoint(self, endpoint_id: str):
        """Remove webhook endpoint"""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
    
    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint"""
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(self) -> Dict[str, WebhookEndpoint]:
        """List all webhook endpoints"""
        return self.endpoints.copy()
    
    def add_event_handler(self, event_type: WebhookEventType, 
                         handler: Callable[[WebhookEvent], Awaitable[None]]):
        """Add event handler for processing before webhook delivery"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: WebhookEvent):
        """Emit webhook event"""
        # Run event handlers
        if event.type in self.event_handlers:
            for handler in self.event_handlers[event.type]:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"Event handler error: {e}")
        
        # Queue for delivery to endpoints
        for endpoint_id, endpoint in self.endpoints.items():
            if not endpoint.active:
                continue
            
            if endpoint.events and event.type not in endpoint.events:
                continue
            
            delivery = WebhookDelivery(event, endpoint)
            await self.delivery_queue.put((endpoint_id, delivery))
    
    async def _delivery_worker(self, worker_name: str):
        """Worker for delivering webhooks"""
        while self.running:
            try:
                # Get delivery from queue
                endpoint_id, delivery = await asyncio.wait_for(
                    self.delivery_queue.get(), timeout=1.0
                )
                
                await self._deliver_webhook(endpoint_id, delivery)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Webhook delivery worker {worker_name} error: {e}")
    
    async def _deliver_webhook(self, endpoint_id: str, delivery: WebhookDelivery):
        """Deliver webhook to endpoint"""
        endpoint = delivery.endpoint
        event = delivery.event
        
        for attempt in range(endpoint.retry_attempts):
            delivery.attempts += 1
            delivery.last_attempt = datetime.now()
            
            try:
                # Prepare payload
                payload = {
                    "id": event.id,
                    "type": event.type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data,
                    "source": event.source,
                    "version": event.version
                }
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "AI-Trading-Platform-Webhook/1.0",
                    **endpoint.headers
                }
                
                # Add signature if secret is provided
                if endpoint.secret:
                    signature = self._generate_signature(
                        json.dumps(payload, sort_keys=True), 
                        endpoint.secret
                    )
                    headers["X-Webhook-Signature"] = signature
                
                # Send webhook
                response = await self._http_client.post(
                    endpoint.url,
                    json=payload,
                    headers=headers,
                    timeout=endpoint.timeout
                )
                
                delivery.last_response_status = response.status_code
                
                if 200 <= response.status_code < 300:
                    delivery.delivered = True
                    print(f"Webhook delivered to {endpoint_id}: {event.type}")
                    break
                else:
                    delivery.last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except Exception as e:
                delivery.last_error = str(e)
                print(f"Webhook delivery attempt {attempt + 1} failed for {endpoint_id}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < endpoint.retry_attempts - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        if not delivery.delivered:
            print(f"Failed to deliver webhook to {endpoint_id} after {endpoint.retry_attempts} attempts")
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self._generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)


class WebhookReceiver:
    """Receives and processes incoming webhooks"""
    
    def __init__(self, webhook_manager: WebhookManager):
        self.webhook_manager = webhook_manager
        self.handlers: Dict[str, Callable] = {}
    
    def add_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """Add handler for incoming webhook events"""
        self.handlers[event_type] = handler
    
    async def process_webhook(self, request: Request) -> JSONResponse:
        """Process incoming webhook request"""
        try:
            # Get request body
            body = await request.body()
            payload = body.decode('utf-8')
            
            # Parse JSON
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON payload")
            
            # Validate required fields
            required_fields = ['type', 'data', 'timestamp']
            for field in required_fields:
                if field not in data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
            
            # Verify signature if provided
            signature = request.headers.get('X-Webhook-Signature')
            if signature:
                # This would need to be configured per sender
                # For now, we'll skip signature verification
                pass
            
            # Process event
            event_type = data['type']
            event_data = data['data']
            
            if event_type in self.handlers:
                try:
                    result = await self.handlers[event_type](event_data)
                    return JSONResponse(content={"status": "success", "result": result})
                except Exception as e:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": str(e)}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Unsupported event type: {event_type}"}
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Webhook processing error: {e}")


# FastAPI integration
def create_webhook_router(webhook_manager: WebhookManager) -> FastAPI:
    """Create FastAPI router for webhook management"""
    app = FastAPI(title="Webhook Management API")
    receiver = WebhookReceiver(webhook_manager)
    
    @app.post("/webhooks/endpoints")
    async def create_webhook_endpoint(
        endpoint_id: str,
        url: str,
        secret: Optional[str] = None,
        events: List[str] = None,
        headers: Dict[str, str] = None
    ):
        """Create webhook endpoint"""
        try:
            # Convert string events to enum
            event_types = []
            if events:
                for event in events:
                    try:
                        event_types.append(WebhookEventType(event))
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid event type: {event}")
            
            endpoint = WebhookEndpoint(
                url=url,
                secret=secret,
                events=event_types,
                headers=headers or {}
            )
            
            webhook_manager.add_endpoint(endpoint_id, endpoint)
            
            return {"message": "Webhook endpoint created", "endpoint_id": endpoint_id}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/webhooks/endpoints")
    async def list_webhook_endpoints():
        """List webhook endpoints"""
        endpoints = webhook_manager.list_endpoints()
        return {
            endpoint_id: {
                "url": endpoint.url,
                "events": [event.value for event in endpoint.events],
                "active": endpoint.active,
                "retry_attempts": endpoint.retry_attempts,
                "timeout": endpoint.timeout
            }
            for endpoint_id, endpoint in endpoints.items()
        }
    
    @app.delete("/webhooks/endpoints/{endpoint_id}")
    async def delete_webhook_endpoint(endpoint_id: str):
        """Delete webhook endpoint"""
        webhook_manager.remove_endpoint(endpoint_id)
        return {"message": "Webhook endpoint deleted"}
    
    @app.post("/webhooks/test")
    async def test_webhook(endpoint_id: str, event_type: str):
        """Test webhook endpoint"""
        endpoint = webhook_manager.get_endpoint(endpoint_id)
        if not endpoint:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        
        # Create test event
        test_event = WebhookEvent(
            id=f"test_{int(time.time())}",
            type=WebhookEventType(event_type),
            timestamp=datetime.now(),
            data={"test": True, "message": "This is a test webhook"}
        )
        
        # Emit event
        await webhook_manager.emit_event(test_event)
        
        return {"message": "Test webhook sent"}
    
    @app.post("/webhooks/receive")
    async def receive_webhook(request: Request):
        """Receive incoming webhook"""
        return await receiver.process_webhook(request)
    
    return app


# Utility functions for common webhook events
async def emit_signal_generated(webhook_manager: WebhookManager, signal_data: Dict[str, Any]):
    """Emit signal generated event"""
    event = WebhookEvent(
        id=f"signal_{signal_data.get('id', int(time.time()))}",
        type=WebhookEventType.SIGNAL_GENERATED,
        timestamp=datetime.now(),
        data=signal_data
    )
    await webhook_manager.emit_event(event)


async def emit_portfolio_updated(webhook_manager: WebhookManager, portfolio_data: Dict[str, Any]):
    """Emit portfolio updated event"""
    event = WebhookEvent(
        id=f"portfolio_{portfolio_data.get('user_id', 'unknown')}_{int(time.time())}",
        type=WebhookEventType.PORTFOLIO_UPDATED,
        timestamp=datetime.now(),
        data=portfolio_data
    )
    await webhook_manager.emit_event(event)


async def emit_risk_alert(webhook_manager: WebhookManager, alert_data: Dict[str, Any]):
    """Emit risk alert event"""
    event = WebhookEvent(
        id=f"alert_{alert_data.get('id', int(time.time()))}",
        type=WebhookEventType.RISK_ALERT,
        timestamp=datetime.now(),
        data=alert_data
    )
    await webhook_manager.emit_event(event)
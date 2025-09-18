"""
Tests for trading API endpoints and WebSocket connections.

This module tests the REST API endpoints for trading signals,
portfolio management, and real-time WebSocket functionality.

Requirements: 3.1, 3.2, 11.1, 11.2, 11.6
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import status
import websockets
from unittest.mock import Mock, patch, AsyncMock

from src.api.app import app
from src.api.auth import create_access_token, UserRole
from src.models.trading_signal import TradingSignal, TradingAction
from src.models.portfolio import Portfolio, Position


class TestTradingEndpoints:
    """Test trading-specific API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for testing."""
        token = create_access_token(
            data={
                "sub": "test_user_123",
                "email": "test@example.com",
                "role": UserRole.PREMIUM
            }
        )
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def mock_user(self):
        """Mock user for testing."""
        return {
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "role": UserRole.PREMIUM,
            "is_active": True,
            "created_at": datetime.now(),
            "daily_signal_count": 0,
            "daily_signal_limit": 999999
        }
    
    def test_generate_trading_signal_success(self, client, auth_headers, mock_user):
        """Test successful trading signal generation."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            with patch('src.services.trading_decision_engine.TradingDecisionEngine.generate_signal') as mock_generate:
                # Mock signal response
                mock_signal = TradingSignal(
                    symbol="AAPL",
                    action=TradingAction.BUY,
                    confidence=0.85,
                    position_size=0.1,
                    target_price=155.0,
                    stop_loss=145.0,
                    timestamp=datetime.now(),
                    model_version="cnn-lstm-v1.0"
                )
                mock_generate.return_value = mock_signal
                
                response = client.post(
                    "/api/v1/trading/signals/generate?symbol=AAPL",
                    headers=auth_headers
                )
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["symbol"] == "AAPL"
                assert data["action"] == "BUY"
                assert data["confidence"] == 0.85
                assert data["position_size"] == 0.1
    
    def test_generate_trading_signal_unauthorized(self, client):
        """Test trading signal generation without authentication."""
        response = client.post("/api/v1/trading/signals/generate?symbol=AAPL")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_signal_history(self, client, auth_headers, mock_user):
        """Test retrieving signal history."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            response = client.get(
                "/api/v1/trading/signals/history?symbol=AAPL&limit=10",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
    
    def test_get_signal_performance(self, client, auth_headers, mock_user):
        """Test retrieving signal performance metrics."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            response = client.get(
                "/api/v1/trading/signals/performance?symbol=AAPL&days=30",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "symbol" in data
            assert "accuracy" in data
            assert "total_return" in data
    
    def test_get_portfolio(self, client, auth_headers, mock_user):
        """Test retrieving user portfolio."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            with patch('src.services.portfolio_management_service.PortfolioManagementService.get_portfolio') as mock_get:
                mock_portfolio = Portfolio(
                    user_id="test_user_123",
                    positions={
                        "AAPL": Position(
                            symbol="AAPL",
                            quantity=100.0,
                            avg_cost=150.0,
                            current_price=155.0,
                            unrealized_pnl=500.0
                        )
                    },
                    cash_balance=10000.0,
                    total_value=25500.0,
                    last_updated=datetime.now()
                )
                mock_get.return_value = mock_portfolio
                
                response = client.get("/api/v1/trading/portfolio", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["user_id"] == "test_user_123"
                assert data["cash_balance"] == 10000.0
                assert data["total_value"] == 25500.0
                assert "AAPL" in data["positions"]
    
    def test_rebalance_portfolio(self, client, auth_headers, mock_user):
        """Test portfolio rebalancing."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            with patch('src.services.portfolio_management_service.PortfolioManagementService.rebalance_portfolio') as mock_rebalance:
                mock_rebalance.return_value = {
                    "rebalance_id": "rebal_123",
                    "trades": [
                        {"symbol": "AAPL", "action": "BUY", "quantity": 10}
                    ]
                }
                
                target_allocation = {
                    "AAPL": 0.3,
                    "GOOGL": 0.2,
                    "MSFT": 0.2,
                    "TSLA": 0.1,
                    "CASH": 0.2
                }
                
                response = client.post(
                    "/api/v1/trading/portfolio/rebalance",
                    json={"target_allocation": target_allocation},
                    headers=auth_headers
                )
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert "rebalance_id" in data
                assert "estimated_trades" in data
    
    def test_rebalance_portfolio_invalid_allocation(self, client, auth_headers, mock_user):
        """Test portfolio rebalancing with invalid allocation."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            # Allocation that doesn't sum to 1.0
            target_allocation = {
                "AAPL": 0.3,
                "GOOGL": 0.2,
                "MSFT": 0.2,
                "TSLA": 0.1
                # Missing allocation, sums to 0.8
            }
            
            response = client.post(
                "/api/v1/trading/portfolio/rebalance",
                json={"target_allocation": target_allocation},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_market_data(self, client, auth_headers, mock_user):
        """Test retrieving market data."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            with patch('src.services.data_aggregator.DataAggregator.get_historical_data') as mock_get_data:
                mock_data = [
                    {
                        "timestamp": "2023-12-01T10:00:00",
                        "open": 150.0,
                        "high": 155.0,
                        "low": 149.0,
                        "close": 154.0,
                        "volume": 1000000
                    }
                ]
                mock_get_data.return_value = mock_data
                
                response = client.get(
                    "/api/v1/trading/market-data/AAPL?timeframe=1h&limit=100",
                    headers=auth_headers
                )
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["symbol"] == "AAPL"
                assert data["timeframe"] == "1h"
                assert len(data["data"]) == 1
    
    def test_trading_health_check(self, client):
        """Test trading endpoints health check."""
        response = client.get("/api/v1/trading/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_websocket_connections" in data
        assert "connected_users" in data


class TestWebSocketConnections:
    """Test WebSocket functionality for real-time data streaming."""
    
    @pytest.fixture
    def mock_user(self):
        """Mock user for WebSocket testing."""
        return {
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "role": UserRole.PREMIUM
        }
    
    @pytest.mark.asyncio
    async def test_websocket_connection_success(self, mock_user):
        """Test successful WebSocket connection."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            # Mock WebSocket connection
            mock_websocket = Mock()
            mock_websocket.accept = AsyncMock()
            mock_websocket.send_text = AsyncMock()
            mock_websocket.receive_text = AsyncMock(side_effect=asyncio.CancelledError())
            
            from src.api.trading_endpoints import manager
            
            # Test connection
            await manager.connect(mock_websocket, "test_user_123")
            
            assert mock_websocket in manager.active_connections
            assert "test_user_123" in manager.user_connections
            assert mock_websocket in manager.user_connections["test_user_123"]
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, mock_user):
        """Test WebSocket message handling."""
        with patch('src.api.trading_endpoints.get_current_user', return_value=mock_user):
            mock_websocket = Mock()
            mock_websocket.accept = AsyncMock()
            mock_websocket.send_text = AsyncMock()
            
            # Mock message sequence
            messages = [
                json.dumps({"type": "subscribe_market_data", "symbols": ["AAPL", "GOOGL"]}),
                json.dumps({"type": "subscribe_signals"}),
                json.dumps({"type": "ping"})
            ]
            mock_websocket.receive_text = AsyncMock(side_effect=messages + [asyncio.CancelledError()])
            
            from src.api.trading_endpoints import manager
            
            await manager.connect(mock_websocket, "test_user_123")
            
            # Verify connection was established
            assert mock_websocket.accept.called
            assert mock_websocket in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_message(self):
        """Test broadcasting messages to all connections."""
        from src.api.trading_endpoints import manager
        
        # Mock multiple connections
        mock_ws1 = Mock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_text = AsyncMock()
        
        manager.active_connections = [mock_ws1, mock_ws2]
        
        test_message = json.dumps({"type": "market_update", "data": {"symbol": "AAPL", "price": 155.0}})
        
        await manager.broadcast(test_message)
        
        mock_ws1.send_text.assert_called_once_with(test_message)
        mock_ws2.send_text.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_websocket_personal_message(self):
        """Test sending personal messages to specific user."""
        from src.api.trading_endpoints import manager
        
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        
        manager.user_connections = {
            "test_user_123": [mock_websocket]
        }
        
        test_message = json.dumps({"type": "signal_generated", "data": {"symbol": "AAPL"}})
        
        await manager.send_personal_message(test_message, "test_user_123")
        
        mock_websocket.send_text.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_websocket_disconnect(self):
        """Test WebSocket disconnection cleanup."""
        from src.api.trading_endpoints import manager
        
        mock_websocket = Mock()
        
        # Setup connection
        manager.active_connections = [mock_websocket]
        manager.user_connections = {
            "test_user_123": [mock_websocket]
        }
        
        # Test disconnection
        manager.disconnect(mock_websocket, "test_user_123")
        
        assert mock_websocket not in manager.active_connections
        assert "test_user_123" not in manager.user_connections


class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_user_registration_success(self, client):
        """Test successful user registration."""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "securepassword123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == user_data["email"]
        assert data["user"]["role"] == "trial"
    
    def test_user_registration_duplicate_email(self, client):
        """Test registration with duplicate email."""
        user_data = {
            "email": "duplicate@example.com",
            "username": "user1",
            "password": "password123"
        }
        
        # First registration
        response1 = client.post("/api/v1/auth/register", json=user_data)
        assert response1.status_code == status.HTTP_200_OK
        
        # Second registration with same email
        user_data["username"] = "user2"
        response2 = client.post("/api/v1/auth/register", json=user_data)
        assert response2.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_user_login_success(self, client):
        """Test successful user login."""
        # First register a user
        user_data = {
            "email": "logintest@example.com",
            "username": "logintest",
            "password": "password123"
        }
        client.post("/api/v1/auth/register", json=user_data)
        
        # Then login
        login_data = {
            "email": "logintest@example.com",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == login_data["email"]
    
    def test_user_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user(self, client):
        """Test getting current user information."""
        # Register and login
        user_data = {
            "email": "currentuser@example.com",
            "username": "currentuser",
            "password": "password123"
        }
        register_response = client.post("/api/v1/auth/register", json=user_data)
        token = register_response.json()["access_token"]
        
        # Get current user
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
    
    def test_token_refresh(self, client):
        """Test token refresh functionality."""
        # Register and login
        user_data = {
            "email": "refreshtest@example.com",
            "username": "refreshtest",
            "password": "password123"
        }
        register_response = client.post("/api/v1/auth/register", json=user_data)
        token = register_response.json()["access_token"]
        
        # Refresh token
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/api/v1/auth/refresh", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != token  # Should be a new token
    
    def test_usage_limits_trial_user(self, client):
        """Test usage limits for trial users."""
        # Register trial user
        user_data = {
            "email": "trialuser@example.com",
            "username": "trialuser",
            "password": "password123"
        }
        register_response = client.post("/api/v1/auth/register", json=user_data)
        token = register_response.json()["access_token"]
        
        # Check usage stats
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/usage", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["role"] == "trial"
        assert data["daily_signal_limit"] == 5
        assert data["daily_signal_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
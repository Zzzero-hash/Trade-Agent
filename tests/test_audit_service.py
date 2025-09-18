"""
Tests for audit service functionality.

This module tests:
- Audit event logging and retrieval
- Event integrity verification
- Compliance reporting
- Data export capabilities

Requirements: 6.5, 9.5
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, mock_open

import pytest

from src.services.audit_service import (
    AuditService,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditQuery,
    get_audit_service,
)


class TestAuditService:
    """Test cases for AuditService."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock security service to return data unchanged
        mock_security_service = Mock()
        mock_security_service.mask_sensitive_data.side_effect = lambda x: x
        mock_security_service.encrypt_data.return_value = Mock(
            data=b"encrypted_data",
            key_id="test_key",
            algorithm="test_algo",
            timestamp=datetime.now(timezone.utc),
        )

        with patch(
            "src.services.audit_service.get_security_service",
            return_value=mock_security_service,
        ):
            self.audit_service = AuditService()

    def test_initialization(self):
        """Test audit service initialization."""
        assert self.audit_service is not None
        assert len(self.audit_service._audit_log) == 0
        assert len(self.audit_service._session_events) == 0

    def test_log_basic_event(self):
        """Test logging a basic audit event."""
        event_id = self.audit_service.log_event(
            event_type=AuditEventType.USER_LOGIN,
            description="User logged in successfully",
            user_id="user123",
            session_id="session456",
            ip_address="192.168.1.100",
            severity=AuditSeverity.LOW,
        )

        assert event_id is not None
        assert len(self.audit_service._audit_log) == 1

        event = self.audit_service._audit_log[0]
        assert event.event_id == event_id
        assert event.event_type == AuditEventType.USER_LOGIN
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.ip_address == "192.168.1.100"
        assert event.severity == AuditSeverity.LOW

    def test_log_event_with_details(self):
        """Test logging event with additional details."""
        details = {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.25,
            "exchange": "NASDAQ",
        }

        event_id = self.audit_service.log_event(
            event_type=AuditEventType.ORDER_PLACED,
            description="Order placed for AAPL",
            details=details,
            user_id="trader123",
            severity=AuditSeverity.HIGH,
        )

        assert event_id is not None
        event = self.audit_service._audit_log[0]
        assert event.details["symbol"] == "AAPL"
        assert event.details["quantity"] == 100

    def test_log_user_action(self):
        """Test logging user actions."""
        event_id = self.audit_service.log_user_action(
            action="login",
            user_id="user123",
            session_id="session456",
            ip_address="192.168.1.100",
        )

        assert event_id is not None
        event = self.audit_service._audit_log[0]
        assert event.event_type == AuditEventType.USER_LOGIN
        assert event.severity == AuditSeverity.LOW
        assert event.system_component == "user_management"

    def test_log_trading_action(self):
        """Test logging trading actions."""
        details = {"quantity": 50, "price": 2500.00, "order_type": "market"}

        event_id = self.audit_service.log_trading_action(
            action="order_placed",
            user_id="trader123",
            symbol="GOOGL",
            details=details,
            correlation_id="trade_corr_456",
        )

        assert event_id is not None
        event = self.audit_service._audit_log[0]
        assert event.event_type == AuditEventType.ORDER_PLACED
        assert event.severity == AuditSeverity.HIGH
        assert event.system_component == "trading_engine"
        assert event.details["symbol"] == "GOOGL"
        assert event.correlation_id == "trade_corr_456"

    def test_log_model_action(self):
        """Test logging ML model actions."""
        details = {
            "accuracy": 0.95, 
            "training_time": 3600, 
            "dataset_size": 10000
        }

        event_id = self.audit_service.log_model_action(
            action="trained",
            model_name="cnn_lstm_v2",
            details=details,
            user_id="ml_engineer",
        )

        assert event_id is not None
        event = self.audit_service._audit_log[0]
        assert event.event_type == AuditEventType.MODEL_TRAINED
        assert event.severity == AuditSeverity.MEDIUM
        assert event.system_component == "ml_engine"
        assert event.details["model_name"] == "cnn_lstm_v2"

    def test_log_security_event(self):
        """Test logging security events."""
        details = {
            "threat_type": "brute_force", 
            "attempts": 5, 
            "blocked": True
        }

        event_id = self.audit_service.log_security_event(
            event_description="Multiple failed login attempts detected",
            details=details,
            user_id="attacker123",
            ip_address="10.0.0.1",
            severity=AuditSeverity.CRITICAL,
        )

        assert event_id is not None
        event = self.audit_service._audit_log[0]
        assert event.event_type == AuditEventType.SECURITY_EVENT
        assert event.severity == AuditSeverity.CRITICAL
        assert event.system_component == "security_service"

    def test_query_events_no_filters(self):
        """Test querying events without filters."""
        # Add some test events
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN, "User login", user_id="user1"
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order placed", user_id="user2"
        )

        query = AuditQuery()
        events = self.audit_service.query_events(query)

        assert len(events) == 2

    def test_query_events_by_user(self):
        """Test querying events by user ID."""
        # Add events for different users
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN, "User login", user_id="user1"
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order placed", user_id="user2"
        )
        self.audit_service.log_event(
            AuditEventType.USER_LOGOUT, "User logout", user_id="user1"
        )

        query = AuditQuery(user_id="user1")
        events = self.audit_service.query_events(query)

        assert len(events) == 2
        assert all(event.user_id == "user1" for event in events)

    def test_query_events_by_time_range(self):
        """Test querying events by time range."""
        base_time = datetime.now(timezone.utc)

        # Mock timestamps for events
        with patch("src.services.audit_service.datetime") as mock_dt:
            # First event - older
            mock_dt.now.return_value = base_time - timedelta(hours=2)
            self.audit_service.log_event(AuditEventType.USER_LOGIN, "Old login")

            # Second event - newer
            mock_dt.now.return_value = base_time
            self.audit_service.log_event(AuditEventType.USER_LOGOUT, "Recent logout")

        # Query for events in the last hour
        query = AuditQuery(
            start_time=base_time - timedelta(hours=1),
            end_time=base_time + timedelta(minutes=1),
        )
        events = self.audit_service.query_events(query)

        assert len(events) == 1
        assert events[0].description == "Recent logout"

    def test_query_events_by_event_type(self):
        """Test querying events by event type."""
        self.audit_service.log_event(AuditEventType.USER_LOGIN, "User login")
        self.audit_service.log_event(AuditEventType.ORDER_PLACED, "Order placed")
        self.audit_service.log_event(AuditEventType.USER_LOGOUT, "User logout")

        query = AuditQuery(
            event_types=[
                AuditEventType.USER_LOGIN, 
                AuditEventType.USER_LOGOUT
            ]
        )
        events = self.audit_service.query_events(query)

        assert len(events) == 2
        event_types = {event.event_type for event in events}
        assert AuditEventType.USER_LOGIN in event_types
        assert AuditEventType.USER_LOGOUT in event_types
        assert AuditEventType.ORDER_PLACED not in event_types

    def test_query_events_pagination(self):
        """Test event query pagination."""
        # Add multiple events
        for i in range(10):
            self.audit_service.log_event(AuditEventType.USER_LOGIN, f"Login {i}")

        # Query first page
        query = AuditQuery(limit=5, offset=0)
        events_page1 = self.audit_service.query_events(query)

        # Query second page
        query = AuditQuery(limit=5, offset=5)
        events_page2 = self.audit_service.query_events(query)

        assert len(events_page1) == 5
        assert len(events_page2) == 5

        # Ensure no overlap
        page1_ids = {event.event_id for event in events_page1}
        page2_ids = {event.event_id for event in events_page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_get_user_activity(self):
        """Test getting user activity."""
        # Add events for specific user
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN, "Login", user_id="user123"
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order", user_id="user123"
        )
        self.audit_service.log_event(
            AuditEventType.USER_LOGOUT, "Logout", user_id="user456"
        )

        activity = self.audit_service.get_user_activity("user123")

        assert len(activity) == 2
        assert all(event.user_id == "user123" for event in activity)

    def test_get_trading_activity(self):
        """Test getting trading activity."""
        # Add trading events
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order placed", details={"symbol": "AAPL"}
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_FILLED, "Order filled", details={"symbol": "GOOGL"}
        )
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN,
            "Login",  # Non-trading event
        )

        activity = self.audit_service.get_trading_activity()

        assert len(activity) == 2
        trading_types = {event.event_type for event in activity}
        assert AuditEventType.ORDER_PLACED in trading_types
        assert AuditEventType.ORDER_FILLED in trading_types
        assert AuditEventType.USER_LOGIN not in trading_types

    def test_get_trading_activity_by_symbol(self):
        """Test getting trading activity filtered by symbol."""
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "AAPL order", details={"symbol": "AAPL"}
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "GOOGL order", details={"symbol": "GOOGL"}
        )

        activity = self.audit_service.get_trading_activity(symbol="AAPL")

        assert len(activity) == 1
        assert activity[0].details["symbol"] == "AAPL"

    def test_event_integrity_verification(self):
        """Test audit event integrity verification."""
        event_id = self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Test order"
        )

        # Verify integrity of all events
        results = self.audit_service.verify_audit_integrity()

        assert results["total_events"] == 1
        assert results["verified_events"] == 1
        assert len(results["corrupted_events"]) == 0
        assert results["integrity_score"] == 1.0

    def test_event_integrity_verification_specific_events(self):
        """Test integrity verification for specific events."""
        self.audit_service.log_event(AuditEventType.USER_LOGIN, "Login")
        self.audit_service.log_event(AuditEventType.USER_LOGOUT, "Logout")

        # Get first event ID for verification
        events = self.audit_service.query_events(AuditQuery())
        event_id1 = events[0].event_id

        # Verify specific event
        results = self.audit_service.verify_audit_integrity([event_id1])

        assert results["total_events"] == 1
        assert results["verified_events"] == 1

    def test_export_audit_data_json(self):
        """Test exporting audit data in JSON format."""
        # Add test events
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN, "Login", user_id="user123"
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order", user_id="user123"
        )

        query = AuditQuery(user_id="user123")

        with patch("builtins.open", mock_open()):
            with patch("pathlib.Path.mkdir"):
                export_result = self.audit_service.export_audit_data(
                    query, export_format="json"
                )

        assert "export_id" in export_result
        assert "filename" in export_result
        assert "total_events" in export_result
        assert export_result["total_events"] == 2

    def test_export_audit_data_unsupported_format(self):
        """Test exporting audit data with unsupported format."""
        query = AuditQuery()

        with pytest.raises(ValueError, match="Unsupported export format"):
            self.audit_service.export_audit_data(query, export_format="xml")

    def test_get_compliance_report(self):
        """Test generating compliance report."""
        # Use a wider time range to ensure all events are captured
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc) + timedelta(days=1)

        # Add test events
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN, "Login", user_id="user123"
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order", user_id="user123"
        )
        self.audit_service.log_event(AuditEventType.SECURITY_EVENT, "Security incident")

        report = self.audit_service.get_compliance_report(start_time, end_time)

        assert "report_id" in report
        assert "summary" in report
        assert "trading_activity" in report
        assert "security_incidents" in report
        assert "user_activity" in report

        assert report["summary"]["total_events"] == 3
        assert report["summary"]["unique_users"] == 1
        assert report["summary"]["security_events"] == 1

    def test_session_event_tracking(self):
        """Test session event tracking."""
        session_id = "session123"

        # Log events for session
        self.audit_service.log_event(
            AuditEventType.USER_LOGIN, "Login", session_id=session_id
        )
        self.audit_service.log_event(
            AuditEventType.ORDER_PLACED, "Order", session_id=session_id
        )

        # Get event IDs for verification
        events = self.audit_service.query_events(AuditQuery())
        event_id1 = events[0].event_id
        event_id2 = events[1].event_id

        # Check session tracking
        assert session_id in self.audit_service._session_events
        session_events = self.audit_service._session_events[session_id]
        assert event_id1 in session_events
        assert event_id2 in session_events

    @patch("src.services.audit_service.logger")
    def test_high_severity_event_logging(self, mock_logger):
        """Test that high-severity events are logged to system log."""
        self.audit_service.log_event(
            AuditEventType.SECURITY_EVENT,
            "Critical security incident",
            severity=AuditSeverity.CRITICAL,
        )

        mock_logger.warning.assert_called_once()

    def test_audit_event_checksum_calculation(self):
        """Test audit event checksum calculation."""
        event = AuditEvent(
            event_id="test_123",
            event_type=AuditEventType.USER_LOGIN,
            timestamp=datetime.now(timezone.utc),
            user_id="user123",
            session_id=None,
            ip_address="192.168.1.1",
            user_agent=None,
            severity=AuditSeverity.LOW,
            description="Test event",
            details={},
            system_component="test",
        )

        # Checksum should be calculated automatically
        assert event.checksum is not None
        assert len(event.checksum) == 64  # SHA-256 hex digest length

    def test_audit_event_integrity_verification(self):
        """Test individual audit event integrity verification."""
        event = AuditEvent(
            event_id="test_123",
            event_type=AuditEventType.USER_LOGIN,
            timestamp=datetime.now(timezone.utc),
            user_id="user123",
            session_id=None,
            ip_address="192.168.1.1",
            user_agent=None,
            severity=AuditSeverity.LOW,
            description="Test event",
            details={},
            system_component="test",
        )

        # Event should verify as intact
        assert event.verify_integrity() is True

        # Tamper with event
        event.description = "Modified description"

        # Event should now fail verification
        assert event.verify_integrity() is False


class TestGlobalAuditService:
    """Test global audit service instance."""

    def test_get_audit_service_singleton(self):
        """Test that get_audit_service returns singleton instance."""
        # Mock security service
        mock_security_service = Mock()
        mock_security_service.mask_sensitive_data.side_effect = lambda x: x
        mock_security_service.encrypt_data.return_value = Mock()

        with patch(
            "src.services.audit_service.get_security_service",
            return_value=mock_security_service,
        ):
            service1 = get_audit_service()
            service2 = get_audit_service()

        assert service1 is service2
        assert isinstance(service1, AuditService)


class TestAuditQuery:
    """Test AuditQuery dataclass."""

    def test_audit_query_defaults(self):
        """Test AuditQuery default values."""
        query = AuditQuery()

        assert query.start_time is None
        assert query.end_time is None
        assert query.event_types is None
        assert query.user_id is None
        assert query.severity is None
        assert query.system_component is None
        assert query.correlation_id is None
        assert query.limit == 1000
        assert query.offset == 0

    def test_audit_query_custom_values(self):
        """Test AuditQuery with custom values."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            event_types=[AuditEventType.USER_LOGIN],
            user_id="user123",
            severity=AuditSeverity.HIGH,
            limit=500,
            offset=100,
        )

        assert query.start_time == start_time
        assert query.end_time == end_time
        assert query.event_types == [AuditEventType.USER_LOGIN]
        assert query.user_id == "user123"
        assert query.severity == AuditSeverity.HIGH
        assert query.limit == 500
        assert query.offset == 100


if __name__ == "__main__":
    pytest.main([__file__])

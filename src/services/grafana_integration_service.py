"""
Grafana Integration Service

Provides integration with Grafana for dashboard creation, management,
and automated dashboard provisioning for production monitoring.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import aiohttp
import yaml
from pydantic import BaseModel


@dataclass
class GrafanaDashboard:
    title: str
    uid: str
    tags: List[str]
    panels: List[Dict[str, Any]]
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"


class GrafanaConfig(BaseModel):
    url: str
    api_key: str
    datasource: str = "prometheus"
    timeout: int = 30


class GrafanaIntegrationService:
    """Service for managing Grafana dashboards and integration."""
    
    def __init__(self, config: GrafanaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def create_trading_platform_dashboards(self) -> List[str]:
        """Create all trading platform dashboards."""
        dashboards = [
            self._create_system_overview_dashboard(),
            self._create_trading_performance_dashboard(),
            self._create_risk_management_dashboard(),
            self._create_ml_model_dashboard(),
            self._create_business_metrics_dashboard()
        ]
        
        created_uids = []
        for dashboard in dashboards:
            uid = await self.create_dashboard(dashboard)
            if uid:
                created_uids.append(uid)
        
        return created_uids
    
    async def create_dashboard(self, dashboard: GrafanaDashboard) -> Optional[str]:
        """Create a Grafana dashboard."""
        try:
            dashboard_json = {
                "dashboard": {
                    "uid": dashboard.uid,
                    "title": dashboard.title,
                    "tags": dashboard.tags,
                    "refresh": dashboard.refresh,
                    "time": {
                        "from": dashboard.time_from,
                        "to": dashboard.time_to
                    },
                    "panels": dashboard.panels,
                    "schemaVersion": 30,
                    "version": 1
                },
                "overwrite": True
            }
            
            async with self.session.post(
                f"{self.config.url}/api/dashboards/db",
                json=dashboard_json
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info(f"Created dashboard: {dashboard.title}")
                    return result.get("uid")
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to create dashboard {dashboard.title}: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error creating dashboard {dashboard.title}: {e}")
            return None
    
    def _create_system_overview_dashboard(self) -> GrafanaDashboard:
        """Create system overview dashboard."""
        panels = [
            {
                "id": 1,
                "title": "System Health Score",
                "type": "stat",
                "targets": [
                    {
                        "expr": "system_health_score",
                        "legendFormat": "Health Score"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 1,
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.9}
                            ]
                        }
                    }
                },
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Request Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(trading_platform_requests_total[5m])",
                        "legendFormat": "{{method}} {{endpoint}}"
                    }
                ],
                "yAxes": [
                    {"label": "Requests/sec", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 6, "y": 0}
            },
            {
                "id": 3,
                "title": "Response Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(trading_platform_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "95th percentile"
                    },
                    {
                        "expr": "histogram_quantile(0.50, rate(trading_platform_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "50th percentile"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 15, "y": 0}
            },
            {
                "id": 4,
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(trading_platform_requests_total{status!~\"2..\"}[5m]) / rate(trading_platform_requests_total[5m])",
                        "legendFormat": "Error Rate"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 1}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 5,
                "title": "System Resources",
                "type": "graph",
                "targets": [
                    {
                        "expr": "system_cpu_usage_percent",
                        "legendFormat": "CPU Usage %"
                    },
                    {
                        "expr": "system_memory_usage_percent",
                        "legendFormat": "Memory Usage %"
                    },
                    {
                        "expr": "system_disk_usage_percent",
                        "legendFormat": "Disk Usage %"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 100}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            }
        ]
        
        return GrafanaDashboard(
            title="Trading Platform - System Overview",
            uid="trading-system-overview",
            tags=["trading", "system", "overview"],
            panels=panels
        )
    
    def _create_trading_performance_dashboard(self) -> GrafanaDashboard:
        """Create trading performance dashboard."""
        panels = [
            {
                "id": 1,
                "title": "Trades Executed",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(trading_platform_trades_executed_total[1h])",
                        "legendFormat": "Trades/Hour"
                    }
                ],
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Trade Execution Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(trading_platform_trade_latency_seconds_bucket[5m]))",
                        "legendFormat": "95th percentile"
                    },
                    {
                        "expr": "histogram_quantile(0.50, rate(trading_platform_trade_latency_seconds_bucket[5m]))",
                        "legendFormat": "50th percentile"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 6, "y": 0}
            },
            {
                "id": 3,
                "title": "Portfolio Values by Customer",
                "type": "graph",
                "targets": [
                    {
                        "expr": "trading_platform_portfolio_value",
                        "legendFormat": "Customer {{customer_id}}"
                    }
                ],
                "yAxes": [
                    {"label": "USD", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 15, "y": 0}
            },
            {
                "id": 4,
                "title": "Portfolio P&L",
                "type": "graph",
                "targets": [
                    {
                        "expr": "trading_platform_portfolio_pnl",
                        "legendFormat": "Customer {{customer_id}}"
                    }
                ],
                "yAxes": [
                    {"label": "USD"}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 5,
                "title": "Active Positions",
                "type": "stat",
                "targets": [
                    {
                        "expr": "sum(trading_platform_active_positions)",
                        "legendFormat": "Total Positions"
                    }
                ],
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8}
            },
            {
                "id": 6,
                "title": "External API Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(trading_platform_external_api_latency_seconds_bucket[5m]))",
                        "legendFormat": "{{service}} - 95th percentile"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8}
            }
        ]
        
        return GrafanaDashboard(
            title="Trading Platform - Trading Performance",
            uid="trading-performance",
            tags=["trading", "performance", "execution"],
            panels=panels
        )
    
    def _create_risk_management_dashboard(self) -> GrafanaDashboard:
        """Create risk management dashboard."""
        panels = [
            {
                "id": 1,
                "title": "Risk Violations",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(trading_platform_risk_violations_total[1h])",
                        "legendFormat": "Violations/Hour"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 5},
                                {"color": "red", "value": 10}
                            ]
                        }
                    }
                },
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Stop-Loss Executions",
                "type": "graph",
                "targets": [
                    {
                        "expr": "increase(trading_platform_stop_loss_executions_total[5m])",
                        "legendFormat": "{{symbol}}"
                    }
                ],
                "yAxes": [
                    {"label": "Count", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 6, "y": 0}
            },
            {
                "id": 3,
                "title": "Circuit Breaker Status",
                "type": "stat",
                "targets": [
                    {
                        "expr": "trading_platform_circuit_breaker_activations_total",
                        "legendFormat": "Active Circuit Breakers"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "red", "value": 1}
                            ]
                        }
                    }
                },
                "gridPos": {"h": 8, "w": 9, "x": 15, "y": 0}
            },
            {
                "id": 4,
                "title": "Portfolio Risk Metrics",
                "type": "graph",
                "targets": [
                    {
                        "expr": "portfolio_risk",
                        "legendFormat": "Customer {{customer_id}}"
                    }
                ],
                "yAxes": [
                    {"label": "Risk Score", "min": 0, "max": 1}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 5,
                "title": "Risk Monitoring Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "risk_monitoring_latency_seconds",
                        "legendFormat": "Monitoring Latency"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            }
        ]
        
        return GrafanaDashboard(
            title="Trading Platform - Risk Management",
            uid="risk-management",
            tags=["trading", "risk", "monitoring"],
            panels=panels
        )
    
    def _create_ml_model_dashboard(self) -> GrafanaDashboard:
        """Create ML model performance dashboard."""
        panels = [
            {
                "id": 1,
                "title": "Model Predictions",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(trading_platform_model_predictions_total[1h])",
                        "legendFormat": "Predictions/Hour"
                    }
                ],
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Model Accuracy",
                "type": "graph",
                "targets": [
                    {
                        "expr": "trading_platform_model_accuracy",
                        "legendFormat": "{{model_name}}"
                    }
                ],
                "yAxes": [
                    {"label": "Accuracy", "min": 0, "max": 1}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 6, "y": 0}
            },
            {
                "id": 3,
                "title": "Feature Extraction Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(trading_platform_feature_extraction_latency_seconds_bucket[5m]))",
                        "legendFormat": "95th percentile"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 15, "y": 0}
            },
            {
                "id": 4,
                "title": "Model Performance Trends",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(trading_platform_model_predictions_total[5m])",
                        "legendFormat": "{{model_name}} - Predictions/sec"
                    }
                ],
                "yAxes": [
                    {"label": "Predictions/sec", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
            }
        ]
        
        return GrafanaDashboard(
            title="Trading Platform - ML Models",
            uid="ml-models",
            tags=["trading", "ml", "models", "performance"],
            panels=panels
        )
    
    def _create_business_metrics_dashboard(self) -> GrafanaDashboard:
        """Create business metrics dashboard."""
        panels = [
            {
                "id": 1,
                "title": "Customer Signups",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(trading_platform_customer_signups_total[24h])",
                        "legendFormat": "Signups/Day"
                    }
                ],
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Revenue",
                "type": "graph",
                "targets": [
                    {
                        "expr": "trading_platform_revenue_usd",
                        "legendFormat": "{{revenue_type}}"
                    }
                ],
                "yAxes": [
                    {"label": "USD", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 6, "y": 0}
            },
            {
                "id": 3,
                "title": "Customer Churn",
                "type": "graph",
                "targets": [
                    {
                        "expr": "increase(trading_platform_customer_churn_total[24h])",
                        "legendFormat": "{{reason}}"
                    }
                ],
                "yAxes": [
                    {"label": "Customers", "min": 0}
                ],
                "gridPos": {"h": 8, "w": 9, "x": 15, "y": 0}
            },
            {
                "id": 4,
                "title": "Active Customers",
                "type": "stat",
                "targets": [
                    {
                        "expr": "count(count by (customer_id) (trading_platform_portfolio_value > 0))",
                        "legendFormat": "Active Customers"
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 5,
                "title": "Total Assets Under Management",
                "type": "stat",
                "targets": [
                    {
                        "expr": "sum(trading_platform_portfolio_value)",
                        "legendFormat": "Total AUM"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "currencyUSD"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            }
        ]
        
        return GrafanaDashboard(
            title="Trading Platform - Business Metrics",
            uid="business-metrics",
            tags=["trading", "business", "revenue", "customers"],
            panels=panels
        )
    
    async def update_dashboard(self, uid: str, dashboard: GrafanaDashboard) -> bool:
        """Update an existing dashboard."""
        try:
            # Get existing dashboard first
            async with self.session.get(f"{self.config.url}/api/dashboards/uid/{uid}") as response:
                if response.status != 200:
                    self.logger.error(f"Dashboard {uid} not found")
                    return False
                
                existing = await response.json()
                version = existing["dashboard"]["version"]
            
            # Update with new version
            dashboard_json = {
                "dashboard": {
                    "uid": dashboard.uid,
                    "title": dashboard.title,
                    "tags": dashboard.tags,
                    "refresh": dashboard.refresh,
                    "time": {
                        "from": dashboard.time_from,
                        "to": dashboard.time_to
                    },
                    "panels": dashboard.panels,
                    "schemaVersion": 30,
                    "version": version + 1
                },
                "overwrite": True
            }
            
            async with self.session.post(
                f"{self.config.url}/api/dashboards/db",
                json=dashboard_json
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Updated dashboard: {dashboard.title}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to update dashboard {dashboard.title}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error updating dashboard {dashboard.title}: {e}")
            return False
    
    async def delete_dashboard(self, uid: str) -> bool:
        """Delete a dashboard."""
        try:
            async with self.session.delete(f"{self.config.url}/api/dashboards/uid/{uid}") as response:
                if response.status == 200:
                    self.logger.info(f"Deleted dashboard: {uid}")
                    return True
                else:
                    self.logger.error(f"Failed to delete dashboard {uid}: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error deleting dashboard {uid}: {e}")
            return False
    
    async def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards."""
        try:
            async with self.session.get(f"{self.config.url}/api/search?type=dash-db") as response:
                if response.status == 200:
                    dashboards = await response.json()
                    return dashboards
                else:
                    self.logger.error(f"Failed to list dashboards: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error listing dashboards: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test connection to Grafana."""
        try:
            async with self.session.get(f"{self.config.url}/api/health") as response:
                if response.status == 200:
                    self.logger.info("Grafana connection successful")
                    return True
                else:
                    self.logger.error(f"Grafana connection failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Grafana connection error: {e}")
            return False


async def setup_production_dashboards(config: GrafanaConfig) -> List[str]:
    """Setup all production dashboards."""
    async with GrafanaIntegrationService(config) as grafana:
        # Test connection first
        if not await grafana.test_connection():
            raise Exception("Failed to connect to Grafana")
        
        # Create dashboards
        dashboard_uids = await grafana.create_trading_platform_dashboards()
        
        return dashboard_uids


if __name__ == "__main__":
    # Example usage
    import os
    
    config = GrafanaConfig(
        url=os.getenv("GRAFANA_URL", "http://localhost:3000"),
        api_key=os.getenv("GRAFANA_API_KEY", ""),
        datasource="prometheus"
    )
    
    async def main():
        try:
            uids = await setup_production_dashboards(config)
            print(f"Created dashboards: {uids}")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())
"""
FastAPI endpoints for risk management functionality.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime, timezone

from src.models.portfolio import Portfolio
from src.models.risk_management import (
    RiskLimit, RiskMetrics, RiskAlert, StressTestScenario, StressTestResult,
    PositionSizingRule, RiskLimitType, RiskLimitStatus
)
from src.services.risk_manager import RiskManager
from src.services.risk_monitoring_service import RiskMonitoringService
from src.services.data_aggregator import DataAggregator


router = APIRouter(prefix="/api/risk", tags=["risk_management"])

# Global instances (in production, these would be dependency injected)
risk_manager = RiskManager()
risk_monitoring_service = RiskMonitoringService(risk_manager)

# Data aggregator will be initialized when needed
data_aggregator = None


@router.post("/limits", response_model=Dict[str, str])
async def add_risk_limit(limit: RiskLimit):
    """Add or update a risk limit."""
    try:
        risk_manager.add_risk_limit(limit)
        return {"status": "success", "message": "Risk limit added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/limits", response_model=List[RiskLimit])
async def get_risk_limits():
    """Get all configured risk limits."""
    try:
        return list(risk_manager.risk_limits.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/limits/{limit_type}")
async def remove_risk_limit(limit_type: str, symbol: Optional[str] = None):
    """Remove a risk limit."""
    try:
        limit_key = f"{limit_type}_{symbol or 'portfolio'}"
        if limit_key in risk_manager.risk_limits:
            del risk_manager.risk_limits[limit_key]
            return {"status": "success", "message": "Risk limit removed"}
        else:
            raise HTTPException(status_code=404, detail="Risk limit not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/position-sizing-rules", response_model=Dict[str, str])
async def add_position_sizing_rule(rule: PositionSizingRule):
    """Add or update a position sizing rule."""
    try:
        risk_manager.add_position_sizing_rule(rule)
        return {"status": "success", "message": "Position sizing rule added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/position-sizing-rules", response_model=List[PositionSizingRule])
async def get_position_sizing_rules():
    """Get all position sizing rules."""
    try:
        return list(risk_manager.position_sizing_rules.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics", response_model=RiskMetrics)
async def calculate_risk_metrics(
    portfolio: Portfolio,
    confidence_level: float = 0.05
):
    """Calculate current risk metrics for a portfolio."""
    try:
        # Get recent market data (simplified for demo)
        symbols = list(portfolio.positions.keys())
        # In production, this would use the actual data aggregator
        # For now, create dummy market data
        import numpy as np
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        market_data = pd.DataFrame({
            symbol: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
            for symbol in symbols
        }, index=dates)
        
        metrics = risk_manager.calculate_risk_metrics(
            portfolio, market_data, confidence_level
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-limits", response_model=List[RiskAlert])
async def check_risk_limits(portfolio: Portfolio):
    """Check risk limits and return any alerts."""
    try:
        # Get recent market data (simplified for demo)
        symbols = list(portfolio.positions.keys())
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        market_data = pd.DataFrame({
            symbol: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
            for symbol in symbols
        }, index=dates)
        
        # Calculate metrics and check limits
        metrics = risk_manager.calculate_risk_metrics(portfolio, market_data)
        alerts = risk_manager.check_risk_limits(portfolio, metrics)
        
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/position-sizing", response_model=Dict[str, float])
async def calculate_position_sizing(
    portfolio: Portfolio,
    symbol: str,
    intended_size: float,
    expected_return: float,
    volatility: float
):
    """Calculate risk-adjusted position size."""
    try:
        # Get market data for correlation calculation (simplified for demo)
        symbols = list(portfolio.positions.keys()) + [symbol]
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        market_data = pd.DataFrame({
            sym: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
            for sym in symbols
        }, index=dates)
        
        adjusted_size = risk_manager.enforce_position_sizing(
            portfolio=portfolio,
            symbol=symbol,
            intended_size=intended_size,
            expected_return=expected_return,
            volatility=volatility,
            market_data=market_data
        )
        
        return {
            "symbol": symbol,
            "intended_size": intended_size,
            "adjusted_size": adjusted_size,
            "adjustment_ratio": adjusted_size / intended_size if intended_size > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stress-test", response_model=StressTestResult)
async def run_stress_test(
    portfolio: Portfolio,
    scenario: StressTestScenario
):
    """Run stress test on portfolio."""
    try:
        # Get market data (simplified for demo)
        symbols = list(portfolio.positions.keys())
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        market_data = pd.DataFrame({
            symbol: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
            for symbol in symbols
        }, index=dates)
        
        result = risk_manager.run_stress_test(portfolio, scenario, market_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stress-scenarios", response_model=List[StressTestScenario])
async def get_predefined_stress_scenarios():
    """Get predefined stress test scenarios."""
    try:
        scenarios = risk_manager.get_predefined_stress_scenarios()
        return scenarios
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stress-test/batch", response_model=List[StressTestResult])
async def run_batch_stress_tests(
    portfolio: Portfolio,
    scenario_names: Optional[List[str]] = None
):
    """Run multiple stress tests on portfolio."""
    try:
        # Get market data (simplified for demo)
        symbols = list(portfolio.positions.keys())
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        market_data = pd.DataFrame({
            symbol: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
            for symbol in symbols
        }, index=dates)
        
        # Get scenarios to run
        all_scenarios = risk_manager.get_predefined_stress_scenarios()
        
        if scenario_names:
            scenarios_to_run = [
                s for s in all_scenarios 
                if s.scenario_name in scenario_names
            ]
        else:
            scenarios_to_run = all_scenarios
        
        # Run stress tests
        results = []
        for scenario in scenarios_to_run:
            try:
                result = risk_manager.run_stress_test(portfolio, scenario, market_data)
                results.append(result)
            except Exception as e:
                # Log error but continue with other scenarios
                print(f"Stress test failed for {scenario.scenario_name}: {e}")
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[RiskAlert])
async def get_active_alerts():
    """Get all active (unacknowledged) risk alerts."""
    try:
        alerts = risk_manager.get_active_alerts()
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge", response_model=Dict[str, str])
async def acknowledge_alert(alert_id: str):
    """Acknowledge a risk alert."""
    try:
        success = risk_manager.acknowledge_alert(alert_id)
        if success:
            return {"status": "success", "message": "Alert acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/status", response_model=Dict[str, Any])
async def get_monitoring_status():
    """Get current risk monitoring status."""
    try:
        status = risk_monitoring_service.get_current_risk_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/start", response_model=Dict[str, str])
async def start_risk_monitoring(background_tasks: BackgroundTasks):
    """Start risk monitoring service."""
    try:
        # In a real implementation, you'd need to provide portfolio and market data providers
        # This is a simplified version
        
        async def dummy_portfolio_provider():
            # This would connect to your portfolio service
            return Portfolio(
                user_id="system",
                positions={},
                cash_balance=0.0,
                total_value=0.0,
                last_updated=datetime.now(timezone.utc)
            )
        
        async def dummy_market_data_provider():
            # This would connect to your market data service
            return pd.DataFrame()
        
        background_tasks.add_task(
            risk_monitoring_service.start_monitoring,
            dummy_portfolio_provider,
            dummy_market_data_provider
        )
        
        return {"status": "success", "message": "Risk monitoring started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop", response_model=Dict[str, str])
async def stop_risk_monitoring():
    """Stop risk monitoring service."""
    try:
        await risk_monitoring_service.stop_monitoring()
        return {"status": "success", "message": "Risk monitoring stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/history", response_model=List[RiskMetrics])
async def get_risk_metrics_history(hours: int = 24):
    """Get risk metrics history."""
    try:
        history = risk_monitoring_service.get_risk_metrics_history(hours)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/alert-history", response_model=List[RiskAlert])
async def get_alert_history(
    hours: int = 24,
    status_filter: Optional[RiskLimitStatus] = None
):
    """Get alert history."""
    try:
        history = risk_monitoring_service.get_alert_history(hours, status_filter)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report", response_model=Dict[str, Any])
async def generate_risk_report(
    portfolio: Portfolio,
    include_stress_tests: bool = True
):
    """Generate comprehensive risk report."""
    try:
        # Get market data (simplified for demo)
        symbols = list(portfolio.positions.keys())
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        market_data = pd.DataFrame({
            symbol: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
            for symbol in symbols
        }, index=dates)
        
        report = risk_monitoring_service.generate_risk_report(
            portfolio, market_data, include_stress_tests
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=Dict[str, str])
async def export_risk_data(
    filepath: str,
    hours: int = 24,
    format: str = "json"
):
    """Export risk monitoring data."""
    try:
        success = risk_monitoring_service.export_risk_data(filepath, hours, format)
        if success:
            return {"status": "success", "message": f"Data exported to {filepath}"}
        else:
            raise HTTPException(status_code=500, detail="Export failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/cleanup", response_model=Dict[str, str])
async def cleanup_old_monitoring_data(days: int = 7):
    """Clean up old monitoring data."""
    try:
        risk_monitoring_service.cleanup_old_data(days)
        return {"status": "success", "message": f"Cleaned up data older than {days} days"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check for risk management system."""
    try:
        # Basic health checks
        checks = {
            "risk_manager": "ok" if risk_manager else "error",
            "monitoring_service": "ok" if risk_monitoring_service else "error",
            "data_aggregator": "ok" if data_aggregator else "error"
        }
        
        overall_status = "ok" if all(status == "ok" for status in checks.values()) else "error"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": checks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
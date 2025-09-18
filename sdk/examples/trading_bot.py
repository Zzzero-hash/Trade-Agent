"""
Complete Trading Bot Example

This example demonstrates how to build a comprehensive trading bot using
the AI Trading Platform SDK with custom strategies, risk management,
and real-time monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from ai_trading_platform import TradingPlatformClient
from ai_trading_platform.plugins import PluginManager, TradingStrategy
from ai_trading_platform.models import TradingSignal, Portfolio, TradingAction
from ai_trading_platform.exceptions import RateLimitError, NetworkError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumStrategy(TradingStrategy):
    """Simple momentum-based trading strategy"""
    
    async def initialize(self):
        self.lookback_period = self.config.get("lookback_period", 20)
        self.momentum_threshold = self.config.get("momentum_threshold", 0.02)
        self.max_position_size = self.config.get("max_position_size", 0.1)
    
    async def cleanup(self):
        pass
    
    async def generate_signal(self, market_data: List[Dict], 
                            current_portfolio: Dict) -> Optional[TradingSignal]:
        """Generate trading signal based on momentum"""
        if len(market_data) < self.lookback_period:
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(market_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate momentum
        current_price = df['close'].iloc[-1]
        past_price = df['close'].iloc[-self.lookback_period]
        momentum = (current_price - past_price) / past_price
        
        # Calculate volatility for confidence scoring
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Generate signal based on momentum
        if momentum > self.momentum_threshold:
            action = TradingAction.BUY
            confidence = min(0.9, abs(momentum) / (volatility + 0.01))
        elif momentum < -self.momentum_threshold:
            action = TradingAction.SELL
            confidence = min(0.9, abs(momentum) / (volatility + 0.01))
        else:
            return None
        
        # Check position sizing
        symbol = df.attrs.get('symbol', 'UNKNOWN')
        position_size = min(self.max_position_size, confidence * 0.1)
        
        return TradingSignal(
            id=f"momentum_{symbol}_{datetime.now().timestamp()}",
            symbol=symbol,
            action=action,
            confidence=confidence,
            price_target=current_price,
            position_size=position_size,
            reasoning=f"Momentum: {momentum:.3f}, Volatility: {volatility:.3f}",
            timestamp=datetime.now()
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate generated signal"""
        return (
            0.5 <= signal.confidence <= 1.0 and
            signal.position_size <= self.max_position_size
        )


class TradingBot:
    """Comprehensive trading bot with risk management"""
    
    def __init__(self, api_key: str, config: Dict = None):
        self.client = TradingPlatformClient("https://api.tradingplatform.com")
        self.api_key = api_key
        self.config = config or {}
        self.plugin_manager = PluginManager()
        self.running = False
        
        # Bot configuration
        self.symbols = self.config.get("symbols", ["AAPL", "GOOGL", "MSFT", "TSLA"])
        self.check_interval = self.config.get("check_interval", 300)  # 5 minutes
        self.max_daily_trades = self.config.get("max_daily_trades", 10)
        self.max_portfolio_risk = self.config.get("max_portfolio_risk", 0.02)  # 2% VaR
        
        # Trading state
        self.daily_trades = 0
        self.last_trade_date = None
        self.active_signals: Dict[str, TradingSignal] = {}
    
    async def initialize(self):
        """Initialize the trading bot"""
        logger.info("Initializing trading bot...")
        
        # Authenticate with API
        await self.client.login_with_api_key(self.api_key)
        logger.info("Authenticated with trading platform")
        
        # Load trading strategies
        await self.plugin_manager.load_plugin(
            MomentumStrategy,
            {
                "lookback_period": 20,
                "momentum_threshold": 0.02,
                "max_position_size": 0.1
            }
        )
        logger.info("Loaded trading strategies")
        
        # Verify initial portfolio state
        portfolio = await self.client.get_portfolio()
        logger.info(f"Initial portfolio value: ${portfolio.total_value:,.2f}")
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting trading bot...")
        
        while self.running:
            try:
                await self.trading_cycle()
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        logger.info("Trading bot stopped")
    
    async def trading_cycle(self):
        """Execute one trading cycle"""
        # Reset daily trade counter if new day
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        # Check if we've hit daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            logger.info("Daily trade limit reached")
            return
        
        # Get current portfolio
        portfolio = await self.client.get_portfolio()
        
        # Check portfolio risk
        if await self.check_portfolio_risk(portfolio):
            logger.warning("Portfolio risk too high, skipping trades")
            return
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                await self.process_symbol(symbol, portfolio)
            except RateLimitError as e:
                logger.warning(f"Rate limited for {symbol}, waiting...")
                await asyncio.sleep(e.retry_after or 60)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    async def process_symbol(self, symbol: str, portfolio: Portfolio):
        """Process trading signals for a specific symbol"""
        # Get market data
        market_data = await self.client.get_market_data(
            symbol=symbol,
            timeframe="1h",
            limit=50
        )
        
        # Add symbol to data for strategy
        for data_point in market_data['data']:
            data_point['symbol'] = symbol
        
        # Execute strategy
        signal = await self.plugin_manager.execute_strategy(
            "MomentumStrategy",
            market_data['data'],
            portfolio.model_dump()
        )
        
        if signal:
            logger.info(f"Generated signal: {signal.action} {signal.symbol} "
                       f"(confidence: {signal.confidence:.2f})")
            
            # Validate and execute signal
            if await self.should_execute_signal(signal, portfolio):
                await self.execute_signal(signal)
            else:
                logger.info(f"Signal validation failed for {signal.symbol}")
    
    async def should_execute_signal(self, signal: TradingSignal, 
                                  portfolio: Portfolio) -> bool:
        """Determine if signal should be executed"""
        # Check confidence threshold
        if signal.confidence < 0.6:
            return False
        
        # Check if we already have a position
        current_position = portfolio.positions.get(signal.symbol)
        if current_position:
            # Don't add to existing position in same direction
            if (signal.action == TradingAction.BUY and current_position.quantity > 0) or \
               (signal.action == TradingAction.SELL and current_position.quantity < 0):
                return False
        
        # Check risk limits
        risk_alerts = await self.client.check_risk_limits(portfolio)
        if risk_alerts:
            logger.warning(f"Risk alerts prevent trading: {[alert.title for alert in risk_alerts]}")
            return False
        
        return True
    
    async def execute_signal(self, signal: TradingSignal):
        """Execute trading signal"""
        try:
            # In a real implementation, this would place orders through your broker
            # For this example, we'll just log the intended trade
            logger.info(f"EXECUTING TRADE: {signal.action} {signal.symbol} "
                       f"size: {signal.position_size:.3f} confidence: {signal.confidence:.2f}")
            
            # Store active signal
            self.active_signals[signal.symbol] = signal
            self.daily_trades += 1
            
            # Here you would integrate with your broker's API to place the actual order
            # Example:
            # if signal.action == TradingAction.BUY:
            #     await broker.place_buy_order(signal.symbol, signal.position_size)
            # elif signal.action == TradingAction.SELL:
            #     await broker.place_sell_order(signal.symbol, signal.position_size)
            
        except Exception as e:
            logger.error(f"Failed to execute signal for {signal.symbol}: {e}")
    
    async def check_portfolio_risk(self, portfolio: Portfolio) -> bool:
        """Check if portfolio risk is within acceptable limits"""
        try:
            risk_metrics = await self.client.get_risk_metrics(portfolio)
            
            # Check Value at Risk
            if abs(risk_metrics.portfolio_var) > self.max_portfolio_risk:
                return True
            
            # Check maximum drawdown
            if risk_metrics.max_drawdown > 0.1:  # 10% max drawdown
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return True  # Err on the side of caution
    
    async def monitor_positions(self):
        """Monitor existing positions and manage exits"""
        portfolio = await self.client.get_portfolio()
        
        for symbol, position in portfolio.positions.items():
            if symbol in self.active_signals:
                signal = self.active_signals[symbol]
                
                # Check if we should exit the position
                if await self.should_exit_position(signal, position):
                    await self.exit_position(symbol, position)
    
    async def should_exit_position(self, signal: TradingSignal, position) -> bool:
        """Determine if we should exit a position"""
        # Check if signal has expired
        if signal.expires_at and datetime.now() > signal.expires_at:
            return True
        
        # Check stop loss
        if signal.stop_loss:
            if (signal.action == TradingAction.BUY and position.current_price <= signal.stop_loss) or \
               (signal.action == TradingAction.SELL and position.current_price >= signal.stop_loss):
                return True
        
        # Check profit target
        if signal.price_target:
            profit_threshold = 0.05  # 5% profit target
            if signal.action == TradingAction.BUY:
                if position.current_price >= signal.price_target * (1 + profit_threshold):
                    return True
            elif signal.action == TradingAction.SELL:
                if position.current_price <= signal.price_target * (1 - profit_threshold):
                    return True
        
        return False
    
    async def exit_position(self, symbol: str, position):
        """Exit a position"""
        logger.info(f"EXITING POSITION: {symbol} quantity: {position.quantity}")
        
        # Remove from active signals
        if symbol in self.active_signals:
            del self.active_signals[symbol]
        
        # Here you would place the exit order through your broker
        # Example:
        # if position.quantity > 0:
        #     await broker.place_sell_order(symbol, abs(position.quantity))
        # else:
        #     await broker.place_buy_order(symbol, abs(position.quantity))
    
    async def get_performance_report(self) -> Dict:
        """Generate performance report"""
        portfolio = await self.client.get_portfolio()
        
        # Get signal performance
        signal_performance = {}
        for symbol in self.symbols:
            try:
                perf = await self.client.get_signal_performance(symbol=symbol, days=30)
                signal_performance[symbol] = perf
            except Exception as e:
                logger.error(f"Error getting performance for {symbol}: {e}")
        
        return {
            "portfolio_value": portfolio.total_value,
            "total_pnl": portfolio.total_pnl,
            "total_pnl_percent": portfolio.total_pnl_percent,
            "daily_trades": self.daily_trades,
            "active_positions": len(portfolio.positions),
            "signal_performance": signal_performance,
            "timestamp": datetime.now().isoformat()
        }
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False


async def main():
    """Main function to run the trading bot"""
    # Configuration
    config = {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "check_interval": 300,  # 5 minutes
        "max_daily_trades": 20,
        "max_portfolio_risk": 0.02
    }
    
    # Initialize bot
    bot = TradingBot("your-api-key", config)
    
    try:
        await bot.initialize()
        
        # Start monitoring task
        monitor_task = asyncio.create_task(periodic_monitoring(bot))
        
        # Run main trading loop
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down trading bot...")
        bot.stop()
        monitor_task.cancel()
    
    finally:
        await bot.client.close()


async def periodic_monitoring(bot: TradingBot):
    """Periodic monitoring and reporting"""
    while bot.running:
        try:
            # Monitor positions
            await bot.monitor_positions()
            
            # Generate performance report every hour
            if datetime.now().minute == 0:
                report = await bot.get_performance_report()
                logger.info(f"Performance Report: {report}")
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
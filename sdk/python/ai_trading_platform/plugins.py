"""
Plugin architecture for custom trading strategies and indicators
"""

import asyncio
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Callable
from datetime import datetime
import pandas as pd

from .models import TradingSignal, MarketData, PluginConfig, StrategyConfig
from .exceptions import PluginError


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.enabled = True
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "config": self.config
        }


class TradingStrategy(BasePlugin):
    """Base class for trading strategies"""
    
    @abstractmethod
    async def generate_signal(self, market_data: pd.DataFrame, 
                            current_portfolio: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal based on market data and portfolio"""
        pass
    
    @abstractmethod
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate generated signal"""
        pass
    
    def get_required_data_period(self) -> int:
        """Get required historical data period in days"""
        return 30  # Default 30 days


class TechnicalIndicator(BasePlugin):
    """Base class for technical indicators"""
    
    @abstractmethod
    async def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values"""
        pass
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ["close"]  # Default requires close price


class RiskManager(BasePlugin):
    """Base class for risk management plugins"""
    
    @abstractmethod
    async def assess_risk(self, signal: TradingSignal, 
                         portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a trading signal"""
        pass
    
    @abstractmethod
    async def adjust_position_size(self, signal: TradingSignal,
                                 risk_assessment: Dict[str, Any]) -> float:
        """Adjust position size based on risk assessment"""
        pass


class WebhookHandler(BasePlugin):
    """Base class for webhook handlers"""
    
    @abstractmethod
    async def handle_webhook(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook"""
        pass
    
    def get_supported_events(self) -> List[str]:
        """Get list of supported event types"""
        return []


class PluginManager:
    """Manages plugin lifecycle and execution"""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.strategies: Dict[str, TradingStrategy] = {}
        self.indicators: Dict[str, TechnicalIndicator] = {}
        self.risk_managers: Dict[str, RiskManager] = {}
        self.webhook_handlers: Dict[str, WebhookHandler] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
    
    async def load_plugin(self, plugin_class: Type[BasePlugin], 
                         config: Dict[str, Any]) -> None:
        """Load and initialize a plugin"""
        try:
            # Create plugin instance
            plugin = plugin_class(config)
            
            # Initialize plugin
            await plugin.initialize()
            
            # Register plugin
            self.plugins[plugin.name] = plugin
            
            # Register in specific category
            if isinstance(plugin, TradingStrategy):
                self.strategies[plugin.name] = plugin
            elif isinstance(plugin, TechnicalIndicator):
                self.indicators[plugin.name] = plugin
            elif isinstance(plugin, RiskManager):
                self.risk_managers[plugin.name] = plugin
            elif isinstance(plugin, WebhookHandler):
                self.webhook_handlers[plugin.name] = plugin
            
            # Store config
            self.plugin_configs[plugin.name] = PluginConfig(
                name=plugin.name,
                version=plugin.version,
                enabled=plugin.enabled,
                config=config
            )
            
        except Exception as e:
            raise PluginError(f"Failed to load plugin {plugin_class.__name__}: {e}")
    
    async def load_plugin_from_module(self, module_path: str, 
                                    config: Dict[str, Any]) -> None:
        """Load plugin from Python module"""
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                raise PluginError(f"No plugin class found in module {module_path}")
            
            await self.load_plugin(plugin_class, config)
            
        except ImportError as e:
            raise PluginError(f"Failed to import plugin module {module_path}: {e}")
    
    async def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin {plugin_name} not found")
        
        plugin = self.plugins[plugin_name]
        
        try:
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from registries
            del self.plugins[plugin_name]
            
            if plugin_name in self.strategies:
                del self.strategies[plugin_name]
            if plugin_name in self.indicators:
                del self.indicators[plugin_name]
            if plugin_name in self.risk_managers:
                del self.risk_managers[plugin_name]
            if plugin_name in self.webhook_handlers:
                del self.webhook_handlers[plugin_name]
            if plugin_name in self.plugin_configs:
                del self.plugin_configs[plugin_name]
                
        except Exception as e:
            raise PluginError(f"Failed to unload plugin {plugin_name}: {e}")
    
    async def execute_strategy(self, strategy_name: str, market_data: pd.DataFrame,
                             portfolio: Dict[str, Any]) -> Optional[TradingSignal]:
        """Execute a trading strategy"""
        if strategy_name not in self.strategies:
            raise PluginError(f"Strategy {strategy_name} not found")
        
        strategy = self.strategies[strategy_name]
        
        if not strategy.enabled:
            return None
        
        try:
            signal = await strategy.generate_signal(market_data, portfolio)
            
            if signal:
                # Validate signal
                is_valid = await strategy.validate_signal(signal)
                if not is_valid:
                    return None
            
            return signal
            
        except Exception as e:
            raise PluginError(f"Strategy {strategy_name} execution failed: {e}")
    
    async def calculate_indicator(self, indicator_name: str, 
                                data: pd.DataFrame) -> pd.Series:
        """Calculate technical indicator"""
        if indicator_name not in self.indicators:
            raise PluginError(f"Indicator {indicator_name} not found")
        
        indicator = self.indicators[indicator_name]
        
        if not indicator.enabled:
            raise PluginError(f"Indicator {indicator_name} is disabled")
        
        try:
            return await indicator.calculate(data)
        except Exception as e:
            raise PluginError(f"Indicator {indicator_name} calculation failed: {e}")
    
    async def assess_risk(self, risk_manager_name: str, signal: TradingSignal,
                         portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk using risk manager"""
        if risk_manager_name not in self.risk_managers:
            raise PluginError(f"Risk manager {risk_manager_name} not found")
        
        risk_manager = self.risk_managers[risk_manager_name]
        
        if not risk_manager.enabled:
            raise PluginError(f"Risk manager {risk_manager_name} is disabled")
        
        try:
            return await risk_manager.assess_risk(signal, portfolio)
        except Exception as e:
            raise PluginError(f"Risk assessment failed: {e}")
    
    async def handle_webhook(self, event_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle webhook event with all registered handlers"""
        results = []
        
        for handler_name, handler in self.webhook_handlers.items():
            if not handler.enabled:
                continue
            
            if event_type not in handler.get_supported_events():
                continue
            
            try:
                result = await handler.handle_webhook(event_type, data)
                results.append({
                    "handler": handler_name,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "handler": handler_name,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin"""
        if plugin_name not in self.plugins:
            return None
        
        return self.plugins[plugin_name].get_info()
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all plugins or plugins of specific type"""
        if plugin_type == "strategy":
            plugins = self.strategies
        elif plugin_type == "indicator":
            plugins = self.indicators
        elif plugin_type == "risk_manager":
            plugins = self.risk_managers
        elif plugin_type == "webhook_handler":
            plugins = self.webhook_handlers
        else:
            plugins = self.plugins
        
        return [plugin.get_info() for plugin in plugins.values()]
    
    async def enable_plugin(self, plugin_name: str) -> None:
        """Enable a plugin"""
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin {plugin_name} not found")
        
        self.plugins[plugin_name].enabled = True
        self.plugin_configs[plugin_name].enabled = True
    
    async def disable_plugin(self, plugin_name: str) -> None:
        """Disable a plugin"""
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin {plugin_name} not found")
        
        self.plugins[plugin_name].enabled = False
        self.plugin_configs[plugin_name].enabled = False
    
    async def update_plugin_config(self, plugin_name: str, 
                                 new_config: Dict[str, Any]) -> None:
        """Update plugin configuration"""
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin {plugin_name} not found")
        
        plugin = self.plugins[plugin_name]
        plugin.config.update(new_config)
        self.plugin_configs[plugin_name].config = plugin.config
    
    async def reload_plugin(self, plugin_name: str) -> None:
        """Reload a plugin"""
        if plugin_name not in self.plugins:
            raise PluginError(f"Plugin {plugin_name} not found")
        
        # Get current config
        config = self.plugin_configs[plugin_name].config
        plugin_class = type(self.plugins[plugin_name])
        
        # Unload and reload
        await self.unload_plugin(plugin_name)
        await self.load_plugin(plugin_class, config)
    
    async def shutdown(self) -> None:
        """Shutdown all plugins"""
        for plugin_name in list(self.plugins.keys()):
            try:
                await self.unload_plugin(plugin_name)
            except Exception as e:
                print(f"Error unloading plugin {plugin_name}: {e}")


# Example plugin implementations
class MovingAverageStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    async def initialize(self) -> None:
        self.short_window = self.config.get("short_window", 10)
        self.long_window = self.config.get("long_window", 30)
    
    async def cleanup(self) -> None:
        pass
    
    async def generate_signal(self, market_data: pd.DataFrame,
                            current_portfolio: Dict[str, Any]) -> Optional[TradingSignal]:
        if len(market_data) < self.long_window:
            return None
        
        # Calculate moving averages
        short_ma = market_data['close'].rolling(window=self.short_window).mean()
        long_ma = market_data['close'].rolling(window=self.long_window).mean()
        
        # Get latest values
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        # Determine signal
        if prev_short <= prev_long and current_short > current_long:
            action = "BUY"
            confidence = 0.7
        elif prev_short >= prev_long and current_short < current_long:
            action = "SELL"
            confidence = 0.7
        else:
            return None
        
        return TradingSignal(
            id=f"ma_strategy_{datetime.now().timestamp()}",
            symbol=market_data.attrs.get('symbol', 'UNKNOWN'),
            action=action,
            confidence=confidence,
            price_target=market_data['close'].iloc[-1],
            reasoning=f"MA crossover: {self.short_window}MA crossed {'above' if action == 'BUY' else 'below'} {self.long_window}MA",
            timestamp=datetime.now()
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return 0.5 <= signal.confidence <= 1.0


class RSIIndicator(TechnicalIndicator):
    """Relative Strength Index indicator"""
    
    async def initialize(self) -> None:
        self.period = self.config.get("period", 14)
    
    async def cleanup(self) -> None:
        pass
    
    async def calculate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
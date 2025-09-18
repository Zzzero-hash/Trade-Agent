"""Asynchronous data processing orchestrator for exchange feeds.

This module coordinates the real-time aggregation pipeline that normalizes
market data from the supported exchanges before persisting or handing it off
to downstream ML services. The implementation focuses on resiliency so the
Kubernetes liveness/readiness probes can rely on it even when credentials are
missing in local development environments.
"""

from __future__ import annotations

import asyncio
import os
from typing import Iterable, List, Optional, Sequence

from ..config.settings import ExchangeConfig, Settings, get_settings
from ..exchanges import OANDAConnector, RobinhoodConnector
from ..exchanges.base import ExchangeConnector
from ..exchanges.coinbase import CoinbaseConnector
from ..services.data_aggregator import AggregatedData, DataAggregator
from ..utils.logging import get_logger, setup_logging
from ..utils.monitoring import get_metrics_collector, setup_monitoring


class DataProcessor:
    """Coordinates exchange connectors and data aggregation loops."""

    def __init__(self, symbols: Optional[Sequence[str]] = None):
        self.settings: Settings = get_settings()

        # Ensure logging/monitoring are configured for standalone execution.
        setup_logging(self.settings.logging)
        setup_monitoring(self.settings.monitoring)

        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()

        self._symbols: List[str] = list(symbols) if symbols else []
        self._connectors: List[ExchangeConnector] = self._build_exchange_connectors(
            self.settings.exchanges.values()
        )
        if not self._symbols:
            self._symbols = self._resolve_default_symbols(self._connectors)

        self.aggregator = DataAggregator(self._connectors)
        self._stop_event = asyncio.Event()
        self._runner_task: Optional[asyncio.Task[None]] = None

        self.logger.info(
            "Initialized data processor with %d connector(s) and %d symbol(s)",
            len(self._connectors),
            len(self._symbols),
        )

    async def start(self) -> None:
        """Start the asynchronous processing loop."""
        if self._runner_task is not None:
            return

        self._runner_task = asyncio.create_task(self._run_forever(), name="data-processor-loop")

    async def stop(self) -> None:
        """Signal the processor to stop and wait for completion."""
        self._stop_event.set()
        if self._runner_task:
            await self._runner_task
            self._runner_task = None

        await self._disconnect_all()

    async def _run_forever(self) -> None:
        """Continuously execute aggregation cycles until asked to stop."""
        retry_delay_seconds = 30

        while not self._stop_event.is_set():
            if not self._connectors:
                self.logger.warning(
                    "No exchange connectors configured; sleeping for %s seconds",
                    retry_delay_seconds,
                )
                await self._sleep_with_stop(retry_delay_seconds)
                # Rebuild connectors in case secrets became available later.
                self._connectors = self._build_exchange_connectors(
                    self.settings.exchanges.values()
                )
                self.aggregator = DataAggregator(self._connectors)
                self._symbols = self._resolve_default_symbols(self._connectors)
                continue

            try:
                processed = False
                async for aggregated in self.aggregator.start_aggregation(self._symbols):
                    processed = True
                    await self._handle_aggregated_data(aggregated)
                    if self._stop_event.is_set():
                        break

                if not processed:
                    self.logger.debug(
                        "Aggregation yielded no data; retrying in %s seconds",
                        retry_delay_seconds,
                    )
                    await self._sleep_with_stop(retry_delay_seconds)

            except Exception as exc:  # noqa: BLE001 - log unexpected failures
                self.logger.error("Data processing loop error: %s", exc, exc_info=True)
                self.metrics.increment_counter(
                    "data_processor.errors_total",
                    tags={"component": "data_processor"},
                )
                await self._sleep_with_stop(retry_delay_seconds)

    async def _handle_aggregated_data(self, aggregated: AggregatedData) -> None:
        """Handle aggregated data emitted by the aggregator."""
        self.metrics.increment_counter(
            "data_processor.records_processed_total", tags={"symbol": aggregated.symbol}
        )
        self.metrics.set_gauge(
            "data_processor.last_volume",
            aggregated.volume,
            tags={"symbol": aggregated.symbol},
        )

        self.logger.debug(
            "Processed %s from %s exchange(s) at %s",
            aggregated.symbol,
            aggregated.source_count,
            aggregated.timestamp.isoformat(),
        )

    async def _disconnect_all(self) -> None:
        """Disconnect all connectors gracefully."""
        for connector in self._connectors:
            try:
                await connector.disconnect()
            except Exception as exc:  # noqa: BLE001 - best effort cleanup
                self.logger.debug("Connector disconnect failed: %s", exc, exc_info=True)

    async def _sleep_with_stop(self, delay: int) -> None:
        """Sleep for ``delay`` seconds unless a stop signal is received."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return

    def _build_exchange_connectors(
        self, exchange_configs: Iterable[ExchangeConfig]
    ) -> List[ExchangeConnector]:
        """Instantiate connectors for exchanges with available credentials."""
        connectors: List[ExchangeConnector] = []

        for config in exchange_configs:
            name = config.name.lower()
            if name == "robinhood":
                username = os.getenv("ROBINHOOD_USERNAME")
                password = os.getenv("ROBINHOOD_PASSWORD")
                if username and password:
                    connectors.append(
                        RobinhoodConnector(
                            username=username,
                            password=password,
                            sandbox=config.sandbox,
                        )
                    )
                else:
                    self.logger.warning(
                        "Skipping Robinhood connector; missing username/password environment variables",
                    )

            elif name == "oanda":
                api_key = os.getenv("OANDA_API_KEY")
                account_id = os.getenv("OANDA_ACCOUNT_ID")
                if api_key and account_id:
                    connectors.append(
                        OANDAConnector(
                            api_key=api_key,
                            account_id=account_id,
                            sandbox=config.sandbox,
                        )
                    )
                else:
                    self.logger.warning(
                        "Skipping OANDA connector; missing OANDA_API_KEY or OANDA_ACCOUNT_ID",
                    )

            elif name == "coinbase":
                api_key = os.getenv("COINBASE_API_KEY")
                api_secret = os.getenv("COINBASE_API_SECRET")
                passphrase = os.getenv("COINBASE_PASSPHRASE")
                if api_key and api_secret and passphrase:
                    connectors.append(
                        CoinbaseConnector(
                            api_key=api_key,
                            api_secret=api_secret,
                            passphrase=passphrase,
                            sandbox=config.sandbox,
                        )
                    )
                else:
                    self.logger.warning(
                        "Skipping Coinbase connector; missing API credentials or passphrase",
                    )

            else:
                self.logger.debug("No connector implementation mapped for exchange '%s'", name)

        self.metrics.set_gauge(
            "data_processor.active_connectors",
            float(len(connectors)),
            tags={"environment": self.settings.environment.value},
        )

        return connectors

    def _resolve_default_symbols(
        self, connectors: Sequence[ExchangeConnector]
    ) -> List[str]:
        """Determine a reasonable default universe of symbols to track."""
        symbols = set()
        for connector in connectors:
            try:
                for symbol in connector.get_supported_symbols():
                    symbols.add(symbol)
            except Exception as exc:  # noqa: BLE001 - skip misbehaving connectors
                self.logger.debug(
                    "Failed to load supported symbols from %s: %s",
                    connector.__class__.__name__,
                    exc,
                    exc_info=True,
                )

        if not symbols:
            # Fallback to a minimal, diverse basket for local development.
            symbols.update({"AAPL", "MSFT", "SPY", "BTC-USD", "ETH-USD"})

        return sorted(symbols)


async def main() -> None:
    """Entry point for running the processor as a standalone service."""
    processor = DataProcessor()
    await processor.start()

    try:
        while True:
            await asyncio.sleep(60)
    except (asyncio.CancelledError, KeyboardInterrupt):  # noqa: PERF203 - deliberate shutdown handling
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())

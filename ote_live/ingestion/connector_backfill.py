from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import AbstractBackfillConnector, BackfillWindow, ensure_utc
from ote_live.ingestion.normalizer import (
    canonical_asset_to_fmp_symbol,
    canonical_timeframe_to_fmp_interval,
    normalize_fmp_historical_chart_response,
)


class FMPAPIError(RuntimeError):
    pass


class FMPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(
        default_factory=lambda: os.getenv("FMP_API_KEY") or os.getenv("TWELVEDATA_API_KEY", "")
    )
    base_url: str = "https://financialmodelingprep.com"
    default_timezone: str = "UTC"
    timeout_seconds: float = 15.0
    max_retries: int = 3

    @model_validator(mode="after")
    def _validate_api_key(self) -> "FMPConfig":
        if not self.api_key:
            raise ValueError(
                "Missing Financial Modeling Prep API key. Set FMP_API_KEY or pass api_key explicitly. "
                "The legacy TWELVEDATA_API_KEY env var is still accepted as a fallback."
            )
        return self


class FMPRESTClient:
    def __init__(
        self,
        config: FMPConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owned_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def __aenter__(self) -> "FMPRESTClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def fetch_historical_chart(
        self,
        *,
        symbol: str,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        outputsize: int | None = None,
    ) -> list[MarketBar]:
        payload = await self._get_json(
            f"/stable/historical-chart/{interval}",
            params=self.build_historical_chart_params(symbol=symbol),
        )
        bars = normalize_fmp_historical_chart_response(
            payload,
            symbol=symbol,
            interval=interval,
            source_timezone=self.config.default_timezone,
        )
        if start_date is not None:
            start_utc = ensure_utc(start_date)
            bars = [bar for bar in bars if bar.timestamp >= start_utc]
        if end_date is not None:
            end_utc = ensure_utc(end_date)
            bars = [bar for bar in bars if bar.timestamp <= end_utc]
        if outputsize is not None:
            outputsize = max(0, int(outputsize))
            if outputsize == 0:
                return []
            bars = bars[-outputsize:]
        return bars

    async def fetch_time_series(
        self,
        *,
        symbol: str,
        interval: str,
        timezone: str = "UTC",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        outputsize: int | None = None,
        order: str = "asc",
    ) -> list[MarketBar]:
        # Legacy compatibility shim for older tests and call sites.
        del timezone, order
        return await self.fetch_historical_chart(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            outputsize=outputsize,
        )

    def build_historical_chart_params(
        self,
        *,
        symbol: str,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "apikey": self.config.api_key,
        }

    async def _get_json(self, path: str, *, params: dict[str, Any]) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, FMPAPIError)),
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    message = payload.get("Error Message") or payload.get("message") or payload.get("error")
                    if message:
                        raise FMPAPIError(str(message))
                return payload
        raise FMPAPIError("Financial Modeling Prep request failed after retries.")


class FMPBackfillConnector(AbstractBackfillConnector):
    def __init__(self, client: FMPRESTClient) -> None:
        self.client = client

    async def backfill_bars(self, window: BackfillWindow) -> list[MarketBar]:
        bars = await self.client.fetch_historical_chart(
            symbol=canonical_asset_to_fmp_symbol(window.asset),
            interval=canonical_timeframe_to_fmp_interval(window.timeframe),
            start_date=window.start,
            end_date=window.end,
        )
        return [
            bar
            for bar in bars
            if ensure_utc(window.start) <= bar.timestamp <= ensure_utc(window.end)
        ]


# Legacy aliases kept so older imports continue to work during the provider migration.
TwelveDataAPIError = FMPAPIError
TwelveDataConfig = FMPConfig
TwelveDataRESTClient = FMPRESTClient
TwelveDataBackfillConnector = FMPBackfillConnector

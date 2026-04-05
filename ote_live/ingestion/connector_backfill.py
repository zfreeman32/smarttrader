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
    canonical_asset_to_twelvedata_symbol,
    canonical_timeframe_to_twelvedata_interval,
    normalize_twelvedata_time_series_response,
)


class TwelveDataAPIError(RuntimeError):
    pass


class TwelveDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(default_factory=lambda: os.getenv("TWELVEDATA_API_KEY", ""))
    base_url: str = "https://api.twelvedata.com"
    default_timezone: str = "UTC"
    timeout_seconds: float = 15.0
    max_retries: int = 3

    @model_validator(mode="after")
    def _validate_api_key(self) -> "TwelveDataConfig":
        if not self.api_key:
            raise ValueError(
                "Missing Twelve Data API key. Set TWELVEDATA_API_KEY or pass api_key explicitly."
            )
        return self


class TwelveDataRESTClient:
    def __init__(
        self,
        config: TwelveDataConfig,
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

    async def __aenter__(self) -> "TwelveDataRESTClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

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
        payload = await self._get_json(
            "/time_series",
            params=self.build_time_series_params(
                symbol=symbol,
                interval=interval,
                timezone=timezone,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
                order=order,
            ),
        )
        return normalize_twelvedata_time_series_response(payload)

    def build_time_series_params(
        self,
        *,
        symbol: str,
        interval: str,
        timezone: str,
        start_date: datetime | None,
        end_date: datetime | None,
        outputsize: int | None,
        order: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "apikey": self.config.api_key,
            "symbol": symbol,
            "interval": interval,
            "timezone": timezone,
            "order": order,
        }
        if start_date is not None:
            params["start_date"] = _format_twelvedata_datetime(start_date)
        if end_date is not None:
            params["end_date"] = _format_twelvedata_datetime(end_date)
        if outputsize is not None:
            params["outputsize"] = int(outputsize)
        return params

    async def _get_json(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                status = str(payload.get("status", "")).lower()
                if status == "error":
                    message = payload.get("message") or payload.get("code") or "Unknown Twelve Data API error."
                    raise TwelveDataAPIError(str(message))
                return payload
        raise TwelveDataAPIError("Twelve Data request failed after retries.")


class TwelveDataBackfillConnector(AbstractBackfillConnector):
    def __init__(self, client: TwelveDataRESTClient) -> None:
        self.client = client

    async def backfill_bars(self, window: BackfillWindow) -> list[MarketBar]:
        bars = await self.client.fetch_time_series(
            symbol=canonical_asset_to_twelvedata_symbol(window.asset),
            interval=canonical_timeframe_to_twelvedata_interval(window.timeframe),
            timezone=self.client.config.default_timezone,
            start_date=window.start,
            end_date=window.end,
            order="asc",
        )
        return [
            bar
            for bar in bars
            if ensure_utc(window.start) <= bar.timestamp <= ensure_utc(window.end)
        ]


def _format_twelvedata_datetime(value: datetime) -> str:
    normalized = ensure_utc(value)
    return normalized.strftime("%Y-%m-%dT%H:%M:%S")

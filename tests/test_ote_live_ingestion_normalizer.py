from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.ingestion.base import BackfillWindow
from ote_live.ingestion.connector_backfill import (
    FMPBackfillConnector,
    FMPConfig,
    FMPRESTClient,
)
from ote_live.ingestion.normalizer import (
    load_local_csv_bars,
    normalize_fmp_historical_chart_response,
)


def test_normalize_fmp_historical_chart_response_maps_to_canonical_market_bars() -> None:
    payload = [
        {
            "date": "2024-01-02 10:00:00",
            "open": 1.1020,
            "high": 1.1024,
            "low": 1.1018,
            "close": 1.1022,
            "volume": 17,
        }
    ]

    bars = normalize_fmp_historical_chart_response(
        payload,
        symbol="EURUSD",
        interval="5min",
    )

    assert len(bars) == 1
    assert bars[0].asset == "EURUSD"
    assert bars[0].timeframe == "5m"
    assert bars[0].timestamp == datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    assert bars[0].open == 1.102
    assert bars[0].close == 1.1022
    assert bars[0].volume == 17.0


def test_load_local_csv_bars_handles_headerless_one_minute_fixture() -> None:
    bars = load_local_csv_bars(
        ROOT / "data" / "currency_data" / "eurusd-1m.csv",
        timeframe="1m",
        source_timezone="UTC",
        canonical_timezone="UTC",
        nrows=5,
    )

    assert len(bars) == 5
    assert bars[0].timestamp == datetime(2002, 10, 21, 1, 0, tzinfo=timezone.utc)
    assert bars[0].open == 0.9738
    assert bars[-1].timestamp == datetime(2002, 10, 21, 1, 4, tzinfo=timezone.utc)


def test_fmp_backfill_connector_uses_five_minute_forex_endpoint_and_filters_locally() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["symbol"] = request.url.params["symbol"]
        captured["apikey"] = request.url.params["apikey"]
        return httpx.Response(
            200,
            json=[
                {
                    "date": "2024-01-02 09:55:00",
                    "open": 1.1000,
                    "high": 1.1005,
                    "low": 1.0998,
                    "close": 1.1002,
                    "volume": 10,
                },
                {
                    "date": "2024-01-02 10:00:00",
                    "open": 1.1010,
                    "high": 1.1020,
                    "low": 1.1005,
                    "close": 1.1015,
                    "volume": 11,
                },
            ],
        )

    async def run_case() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://financialmodelingprep.com",
        ) as http_client:
            client = FMPRESTClient(
                FMPConfig(api_key="test-key"),
                http_client=http_client,
            )
            connector = FMPBackfillConnector(client)
            bars = await connector.backfill_bars(
                BackfillWindow(
                    asset="EURUSD",
                    timeframe="5m",
                    start=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                )
            )
            assert len(bars) == 1
            assert bars[0].close == 1.1015

    asyncio.run(run_case())

    assert captured["path"] == "/stable/historical-chart/5min"
    assert captured["symbol"] == "EURUSD"
    assert captured["apikey"] == "test-key"

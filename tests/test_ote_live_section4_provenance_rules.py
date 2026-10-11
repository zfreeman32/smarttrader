"""Section 4 protocol v2: D6 FMP completion rule and D8 provenance carry (before/after)."""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.ingestion.connector_backfill import FMPConfig, FMPRESTClient
from ote_live.ingestion.provenance import (
    FMP_COMPLETION_GRACE_SECONDS,
    apply_fmp_completion_rule,
    observe_bar,
    source_bar_version,
)
from ote_live.models.loaders import load_live_runtime_manifest

ES_BASE = ROOT / "ote_live/runtime_manifests/ict_es_paper_signal_20260813/ict_long_meta_xgb_v1/live_runtime_manifest.json"
HTF_FEATURES = ["htf_30m_ema_alignment", "htf_1h_ema_alignment", "htf_alignment_score", "htf_30m_close_vs_ema_21_atr"]
PLAIN_FEATURES = ["atr_14", "body_to_range"]


def _fx_bar(observed_after_close: float) -> MarketBar:
    start = datetime(2024, 1, 2, 10, 0, tzinfo=UTC)
    return observe_bar(
        MarketBar(asset="EURUSD", timeframe="5m", timestamp=start, open=1.1, high=1.1005, low=1.0995, close=1.1002),
        observed_at=start + timedelta(minutes=5, seconds=observed_after_close),
    )


def test_d6_rule_marks_complete_only_after_close_plus_grace_and_revises_version() -> None:
    early, late = _fx_bar(FMP_COMPLETION_GRACE_SECONDS - 1), _fx_bar(FMP_COMPLETION_GRACE_SECONDS)
    assert apply_fmp_completion_rule(early) == early and early.is_complete is None
    done = apply_fmp_completion_rule(late)
    assert done.is_complete is True and done.bar_version == source_bar_version(done) != late.bar_version
    assert done.model_dump(exclude={"is_complete", "bar_version"}) == late.model_dump(exclude={"is_complete", "bar_version"})
    provisional = late.model_copy(update={"is_complete": False})
    assert apply_fmp_completion_rule(provisional) == provisional  # never overrides a provider flag


def test_d6_fmp_client_rule_is_opt_in() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[
            {"date": "2024-01-02 09:55:00", "open": 1.1, "high": 1.1005, "low": 1.0998, "close": 1.1002, "volume": 10},
        ])

    async def fetch(enabled: bool) -> list[MarketBar]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                     base_url="https://financialmodelingprep.com") as http_client:
            client = FMPRESTClient(FMPConfig(api_key="test-key", apply_completion_rule=enabled), http_client=http_client)
            return await client.fetch_historical_chart(symbol="EURUSD", interval="5min")

    off, on = asyncio.run(fetch(False)), asyncio.run(fetch(True))
    assert off[0].is_complete is None and on[0].is_complete is True  # a 2024 bar is long past its grace
    assert off[0].model_dump(include={"timestamp", "open", "high", "low", "close", "volume"}) == on[0].model_dump(
        include={"timestamp", "open", "high", "low", "close", "volume"})


def _es_engine() -> IncrementalFeatureEngine:
    base = load_live_runtime_manifest(ES_BASE)
    selected = [*HTF_FEATURES, *PLAIN_FEATURES]
    manifest = base.model_copy(update={"feature_manifest": base.feature_manifest.model_copy(update={
        "selected_feature_names": selected, "selected_feature_count": len(selected), "strategy_feature_count": 0})})
    return IncrementalFeatureEngine([manifest], rolling_window_bars=1500)


def _es_frame(completion: bool | None) -> pd.DataFrame:
    start = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)
    bars = []
    for index in range(1500):
        price = 5000.0 + 8.0 * np.sin(index / 40.0) + 0.05 * index
        bars.append(MarketBar(asset="ES", timeframe="5m", timestamp=start + timedelta(minutes=5 * index), open=price,
                              high=price + 1.5, low=price - 1.5, close=price + 0.25, volume=1000.0 + index,
                              contract_symbol="ESM4", is_complete=completion))
    return IncrementalFeatureEngine.market_frame_from_bars(bars)


def test_d8_before_after_changes_only_completion_gated_features() -> None:
    engine = _es_engine()
    carried = engine.build_feature_frame(_es_frame(True))  # D8 on: canonical completion reaches the engine
    no_gate = engine.build_feature_frame(_es_frame(True).drop(columns=["is_complete"]))  # offline semantics
    dropped = engine.build_feature_frame(_es_frame(None))  # D8 off: completion lost on the store-fetch path
    pd.testing.assert_frame_equal(carried, no_gate, check_exact=True)
    pd.testing.assert_frame_equal(carried[PLAIN_FEATURES], dropped[PLAIN_FEATURES], check_exact=True)
    assert dropped[HTF_FEATURES].isna().all().all()
    assert carried[HTF_FEATURES].iloc[-1].notna().all()

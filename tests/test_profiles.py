from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frvp.continuity.continuous_contract import RawProfileBars  # noqa: E402
from frvp.continuity.types import ProfileSlice  # noqa: E402
from frvp.profiles.anchors import FRVP_ANCHOR_DEFINITIONS, FRVPAnchorEngine, NakedVPOCLevel, NakedVPOCTracker  # noqa: E402
from frvp.profiles.builder import VolumeProfileBuilder  # noqa: E402


def _ts(local_value: str) -> pd.Timestamp:
    return pd.Timestamp(local_value, tz="America/New_York").tz_convert("UTC")


def _bar(
    local_value: str,
    *,
    price: float,
    volume: float,
    contract_id: str = "ESH24",
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
) -> dict[str, object]:
    return {
        "timestamp": _ts(local_value),
        "contract_id": contract_id,
        "open": float(price if open_price is None else open_price),
        "high": float(price if high is None else high),
        "low": float(price if low is None else low),
        "close": float(price if close is None else close),
        "volume": float(volume),
    }


def test_volume_profile_extracts_known_poc_and_value_area() -> None:
    bars = pd.DataFrame(
        [
            _bar("2024-01-03 09:30:00", price=100.00, volume=100),
            _bar("2024-01-03 09:35:00", price=100.25, volume=50),
            _bar("2024-01-03 09:40:00", price=100.50, volume=30),
            _bar("2024-01-03 09:45:00", price=100.75, volume=20),
        ]
    )
    profile = VolumeProfileBuilder(instrument="es").build(
        ProfileSlice(
            contract_id="ESH24",
            start=pd.Timestamp(bars["timestamp"].min()),
            end=pd.Timestamp(bars["timestamp"].max()),
            bars=bars,
        )
    )

    assert profile.poc == pytest.approx(100.00)
    assert profile.val == pytest.approx(100.00)
    assert profile.vah == pytest.approx(100.25)
    assert profile.histogram.loc[100.00] == pytest.approx(100.0)
    assert profile.histogram.loc[100.25] == pytest.approx(50.0)


def test_initial_balance_anchor_excludes_current_bar_from_profile_window() -> None:
    first_hour_times = [
        "2024-01-03 09:30:00",
        "2024-01-03 09:35:00",
        "2024-01-03 09:40:00",
        "2024-01-03 09:45:00",
        "2024-01-03 09:50:00",
        "2024-01-03 09:55:00",
        "2024-01-03 10:00:00",
        "2024-01-03 10:05:00",
        "2024-01-03 10:10:00",
        "2024-01-03 10:15:00",
        "2024-01-03 10:20:00",
        "2024-01-03 10:25:00",
    ]
    bars = pd.DataFrame(
        [_bar(value, price=100.00, volume=100 if index < 3 else 10) for index, value in enumerate(first_hour_times)]
        + [_bar("2024-01-03 10:30:00", price=101.00, volume=500)]
    )
    raw = RawProfileBars(bars)
    anchors = FRVPAnchorEngine(raw, instrument="es")
    builder = VolumeProfileBuilder(instrument="es")

    ib_anchor = anchors.initial_balance(_ts("2024-01-03 10:30:00"))
    assert ib_anchor is not None
    assert pd.Timestamp(ib_anchor.bars["timestamp"].max()) == _ts("2024-01-03 10:25:00")

    anchor_profile = builder.build(ib_anchor.profile_slice)
    inclusive_profile = builder.build(bars.loc[bars["timestamp"] <= _ts("2024-01-03 10:30:00")].copy())

    assert anchor_profile.poc == pytest.approx(100.00)
    assert inclusive_profile.poc == pytest.approx(101.00)


def test_naked_vpoc_resets_on_first_overlap() -> None:
    tracker = NakedVPOCTracker()
    level = NakedVPOCLevel(
        price=100.25,
        contract_id="ESH24",
        formed_at=_ts("2024-01-03 16:00:00"),
        anchor_name="prior_rth",
    )
    tracker.register_level(level)

    untouched = tracker.process_bar(pd.Series(_bar("2024-01-03 16:05:00", price=100.75, volume=10, low=100.50, high=101.00)))
    assert untouched == ()
    assert tracker.active_levels(contract_id="ESH24") == (level,)

    touched = tracker.process_bar(pd.Series(_bar("2024-01-03 16:10:00", price=100.00, volume=10, low=100.00, high=100.50)))
    assert touched == (level,)
    assert tracker.active_levels(contract_id="ESH24") == ()


def test_initial_balance_unavailable_before_1030_and_available_after() -> None:
    bars = pd.DataFrame(
        [
            _bar("2024-01-03 09:30:00", price=100.00, volume=10),
            _bar("2024-01-03 09:35:00", price=100.25, volume=10),
            _bar("2024-01-03 09:40:00", price=100.25, volume=10),
            _bar("2024-01-03 09:45:00", price=100.50, volume=10),
            _bar("2024-01-03 09:50:00", price=100.50, volume=10),
            _bar("2024-01-03 09:55:00", price=100.50, volume=10),
            _bar("2024-01-03 10:00:00", price=100.75, volume=10),
            _bar("2024-01-03 10:05:00", price=100.75, volume=10),
            _bar("2024-01-03 10:10:00", price=100.75, volume=10),
            _bar("2024-01-03 10:15:00", price=100.25, volume=10),
            _bar("2024-01-03 10:20:00", price=100.25, volume=10),
            _bar("2024-01-03 10:25:00", price=100.00, volume=10),
            _bar("2024-01-03 10:30:00", price=99.75, volume=10),
        ]
    )
    anchors = FRVPAnchorEngine(RawProfileBars(bars), instrument="es")
    builder = VolumeProfileBuilder(instrument="es")

    assert anchors.initial_balance(_ts("2024-01-03 10:25:00")) is None

    ib_anchor = anchors.initial_balance(_ts("2024-01-03 10:30:00"))
    assert ib_anchor is not None
    profile = builder.build(ib_anchor.profile_slice)

    assert profile.start == _ts("2024-01-03 09:30:00")
    assert profile.end == _ts("2024-01-03 10:30:00")
    assert profile.poc == pytest.approx(100.25)


def test_anchor_definitions_codify_operational_range_roles() -> None:
    assert [definition.anchor_name for definition in FRVP_ANCHOR_DEFINITIONS] == [
        "prior_rth",
        "overnight_eth",
        "initial_balance",
        "swing_to_swing",
        "rolling_composite",
    ]
    assert FRVP_ANCHOR_DEFINITIONS[0].range_role == "primary_decision_range"
    assert FRVP_ANCHOR_DEFINITIONS[0].level_fields == ("poc", "vah", "val", "hvn", "lvn")
    assert FRVP_ANCHOR_DEFINITIONS[-1].context_only is True
    assert FRVPAnchorEngine.definitions() == FRVP_ANCHOR_DEFINITIONS


def test_anchor_profiles_never_mix_contracts() -> None:
    bars = pd.DataFrame(
        [
            _bar("2024-03-13 09:30:00", price=100.00, volume=10, contract_id="ESH24"),
            _bar("2024-03-13 09:35:00", price=100.25, volume=10, contract_id="ESH24"),
            _bar("2024-03-13 16:00:00", price=100.50, volume=10, contract_id="ESH24"),
            _bar("2024-03-13 16:05:00", price=100.75, volume=10, contract_id="ESH24"),
            _bar("2024-03-13 18:00:00", price=101.00, volume=10, contract_id="ESM24"),
            _bar("2024-03-13 18:05:00", price=101.25, volume=10, contract_id="ESM24"),
            _bar("2024-03-14 09:25:00", price=101.50, volume=10, contract_id="ESM24"),
            _bar("2024-03-14 09:30:00", price=101.75, volume=10, contract_id="ESM24"),
        ]
    )
    anchors = FRVPAnchorEngine(RawProfileBars(bars), instrument="es")
    resolved = anchors.resolve_all(_ts("2024-03-14 09:30:00"))

    assert resolved["prior_rth"] is not None
    assert resolved["prior_rth"].bars["contract_id"].astype(str).nunique() == 1
    assert resolved["overnight_eth"] is None
    for anchor in resolved.values():
        if anchor is None:
            continue
        assert anchor.bars["contract_id"].astype(str).nunique() == 1

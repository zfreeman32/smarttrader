from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.labeling.frvp_labeling_engine as frvp_engine  # noqa: E402
from data.labeling.frvp_labeling_engine import (  # noqa: E402
    FRVPLabelingParams,
    _session_context,
    build_frvp_labels,
    frvp_events_to_frame,
)
from data.labeling.reversal_labeling_engine import infer_bar_minutes  # noqa: E402
from frvp.continuity.continuous_contract import BackAdjustedPathBars, RawProfileBars  # noqa: E402


def _market_fixture(
    *,
    session_count: int,
    bars_per_session: int,
    start_date: str = "2024-01-02",
    base_price: float = 5000.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_close = float(base_price)
    for session_idx, session_date in enumerate(pd.bdate_range(start_date, periods=session_count)):
        session_date = pd.Timestamp(session_date).normalize()
        start_ts = pd.Timestamp(session_date, tz="UTC") + pd.Timedelta(hours=14, minutes=30)
        for bar_idx in range(bars_per_session):
            timestamp = start_ts + pd.Timedelta(minutes=5 * bar_idx)
            open_price = previous_close
            close_price = open_price + 0.02
            high_price = close_price + 0.08
            low_price = open_price - 0.08
            rows.append(
                {
                    "timestamp": timestamp,
                    "session_date": session_date,
                    "is_rth": True,
                    "open": float(open_price),
                    "high": float(high_price),
                    "low": float(low_price),
                    "close": float(close_price),
                    "volume": 100.0,
                    "contract_id": "ESH24",
                    "in_roll_bracket": False,
                }
            )
            previous_close = close_price
    return pd.DataFrame(rows)


def _input_frame(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    frame = frame.set_index("timestamp").sort_index()
    return frame


def _features_for_market(market: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frvp_setup_type": pd.Series(0, index=market.index, dtype="Int64"),
            "frvp_setup_side": pd.Series(0, index=market.index, dtype="Int64"),
            "frvp_setup_confidence_rule": 0.0,
        }
    )


def _mark_setup(
    features: pd.DataFrame,
    index: int,
    *,
    setup_type: int,
    side: int,
    confidence: float = 0.9,
) -> None:
    features.loc[index, "frvp_setup_type"] = int(setup_type)
    features.loc[index, "frvp_setup_side"] = int(side)
    features.loc[index, "frvp_setup_confidence_rule"] = float(confidence)


def _force_long_tp(market: pd.DataFrame, event_idx: int, *, tp_atr: float, hit_offset: int = 1) -> None:
    entry_idx = event_idx + 1
    hit_idx = entry_idx + hit_offset - 1
    entry_price = float(market.at[entry_idx, "open"])
    market.at[hit_idx, "high"] = max(float(market.at[hit_idx, "high"]), entry_price + tp_atr + 0.05)
    market.at[hit_idx, "low"] = min(float(market.at[hit_idx, "low"]), entry_price - 0.05)
    market.at[hit_idx, "close"] = max(float(market.at[hit_idx, "close"]), entry_price + tp_atr * 0.6)


def _force_short_tp(market: pd.DataFrame, event_idx: int, *, tp_atr: float, hit_offset: int = 1) -> None:
    entry_idx = event_idx + 1
    hit_idx = entry_idx + hit_offset - 1
    entry_price = float(market.at[entry_idx, "open"])
    market.at[hit_idx, "low"] = min(float(market.at[hit_idx, "low"]), entry_price - tp_atr - 0.05)
    market.at[hit_idx, "high"] = max(float(market.at[hit_idx, "high"]), entry_price + 0.05)
    market.at[hit_idx, "close"] = min(float(market.at[hit_idx, "close"]), entry_price - tp_atr * 0.6)


def _mock_prepare_context(market: pd.DataFrame) -> SimpleNamespace:
    raw = market.loc[:, ["timestamp", "contract_id", "open", "high", "low", "close", "volume"]].copy()
    path = raw.rename(columns={"contract_id": "source_contract_id"})
    return SimpleNamespace(
        market=market.reset_index(drop=True).copy(),
        continuous=SimpleNamespace(
            raw_profile_bars=RawProfileBars(raw),
            path_bars=BackAdjustedPathBars(path),
        ),
    )


def _patch_frvp_context(monkeypatch, market: pd.DataFrame, features: pd.DataFrame) -> None:
    prepared = _mock_prepare_context(market)

    def fake_prepare_context(df, config, instrument=None):
        return prepared

    def fake_build_features(df, config, instrument=None):
        return features.reset_index(drop=True).copy()

    def fake_detect_htf_swings(path_5m, params):
        atr_5m = pd.Series(1.0, index=path_5m.index, dtype=float)
        atr_30m = pd.Series(1.0, index=path_5m.iloc[::6].index, dtype=float)
        structural_atr_5m = pd.Series(1.0, index=path_5m.index, dtype=float)
        path_30m = path_5m.iloc[::6].copy()
        path_1hr = path_5m.iloc[::12].copy()
        return atr_5m, atr_30m, structural_atr_5m, [], [], path_30m, path_1hr

    monkeypatch.setattr(frvp_engine, "_prepare_frvp_context", fake_prepare_context)
    monkeypatch.setattr(frvp_engine, "build_frvp_context_features", fake_build_features)
    monkeypatch.setattr(frvp_engine, "_detect_htf_swings", fake_detect_htf_swings)


def _params(**overrides) -> FRVPLabelingParams:
    return FRVPLabelingParams(
        instrument="es",
        warmup_bars=0,
        auto_scale_bar_counts=False,
        **overrides,
    )


def test_setup_fire_sampling_is_causal_without_future_bar_lookahead(monkeypatch) -> None:
    market = _market_fixture(session_count=1, bars_per_session=60)
    features = _features_for_market(market)
    _mark_setup(features, 10, setup_type=1, side=1, confidence=0.85)
    _force_long_tp(market, 10, tp_atr=0.8)

    _patch_frvp_context(monkeypatch, market, features)
    base_labels, _, base_events = build_frvp_labels(_input_frame(market), params=_params(), verbose=False)

    future_row = market.iloc[[-1]].copy()
    future_row["timestamp"] = future_row["timestamp"] + pd.Timedelta(minutes=5)
    future_row["open"] = future_row["open"] + 25.0
    future_row["high"] = future_row["high"] + 40.0
    future_row["low"] = future_row["low"] - 15.0
    future_row["close"] = future_row["close"] + 30.0
    future_row["volume"] = future_row["volume"] * 10.0
    extended_market = pd.concat([market, future_row], ignore_index=True)
    extended_features = _features_for_market(extended_market)
    _mark_setup(extended_features, 10, setup_type=1, side=1, confidence=0.85)

    _patch_frvp_context(monkeypatch, extended_market, extended_features)
    extended_labels, _, extended_events = build_frvp_labels(_input_frame(extended_market), params=_params(), verbose=False)

    compare_columns = [
        "label_long_frvp_reversal",
        "exclude_long_frvp_reversal",
        "neg_ok_long_frvp_reversal",
        "sample_weight_long_frvp_reversal",
    ]
    pd.testing.assert_frame_equal(
        base_labels.loc[:, compare_columns],
        extended_labels.loc[base_labels.index, compare_columns],
    )
    base_event_frame = frvp_events_to_frame(base_events)
    extended_event_frame = frvp_events_to_frame(extended_events)
    pd.testing.assert_frame_equal(
        base_event_frame.loc[:, ["setup_type", "setup_side", "swing_index", "entry_index", "tb_outcome"]],
        extended_event_frame.loc[: len(base_event_frame) - 1, ["setup_type", "setup_side", "swing_index", "entry_index", "tb_outcome"]].reset_index(drop=True),
    )


def test_infer_bar_minutes_uses_timedelta_not_raw_timestamp_units() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2024-01-02T14:30:00Z",
                "2024-01-02T14:35:00Z",
                "2024-01-02T14:40:00Z",
            ],
            utc=True,
        )
    )

    assert infer_bar_minutes(index) == 5.0


def test_each_family_uses_the_expected_tp_sl_and_holding(monkeypatch) -> None:
    market = _market_fixture(session_count=2, bars_per_session=70)
    features = _features_for_market(market)
    _mark_setup(features, 10, setup_type=1, side=1)
    _mark_setup(features, 30, setup_type=2, side=1)
    _mark_setup(features, 50, setup_type=4, side=-1)

    _patch_frvp_context(monkeypatch, market, features)
    _, _, events = build_frvp_labels(_input_frame(market), params=_params(), verbose=False)
    event_frame = frvp_events_to_frame(events).set_index("setup_type")

    assert float(event_frame.at[1, "tp_atr"]) == 0.8
    assert float(event_frame.at[1, "sl_atr"]) == 1.0
    assert int(event_frame.at[1, "max_holding_bars"]) == 60

    assert float(event_frame.at[2, "tp_atr"]) == 1.2
    assert float(event_frame.at[2, "sl_atr"]) == 1.0
    assert int(event_frame.at[2, "max_holding_bars"]) == 96

    assert float(event_frame.at[4, "tp_atr"]) == 1.0
    assert float(event_frame.at[4, "sl_atr"]) == 0.9
    assert int(event_frame.at[4, "max_holding_bars"]) == 48


def test_continuation_barrier_overrides_apply_per_setup_and_direction(monkeypatch) -> None:
    market = _market_fixture(session_count=2, bars_per_session=80)
    features = _features_for_market(market)
    _mark_setup(features, 10, setup_type=2, side=1)
    _mark_setup(features, 30, setup_type=3, side=1)
    _mark_setup(features, 50, setup_type=5, side=1)
    _mark_setup(features, 70, setup_type=5, side=-1)

    _patch_frvp_context(monkeypatch, market, features)
    _, _, events = build_frvp_labels(
        _input_frame(market),
        params=_params(
            continuation_profit_atr=1.0,
            continuation_setup3_long_profit_atr=0.95,
            continuation_setup5_long_profit_atr=0.90,
            continuation_setup5_short_profit_atr=0.85,
        ),
        verbose=False,
    )
    event_frame = frvp_events_to_frame(events).set_index(["setup_type", "setup_side"])

    assert float(event_frame.at[(2, 1), "tp_atr"]) == 1.0
    assert float(event_frame.at[(3, 1), "tp_atr"]) == 0.95
    assert float(event_frame.at[(5, 1), "tp_atr"]) == 0.90
    assert float(event_frame.at[(5, -1), "tp_atr"]) == 0.85
    assert float(event_frame.at[(5, 1), "sl_atr"]) == 1.0
    assert int(event_frame.at[(5, -1), "max_holding_bars"]) == 96


def test_roll_spanning_barrier_events_are_excluded(monkeypatch) -> None:
    market = _market_fixture(session_count=2, bars_per_session=75)
    features = _features_for_market(market)
    _mark_setup(features, 20, setup_type=2, side=1)

    market.loc[40:, "contract_id"] = "ESM24"
    _patch_frvp_context(monkeypatch, market, features)
    labeled, _, events = build_frvp_labels(_input_frame(market), params=_params(), verbose=False)
    event_frame = frvp_events_to_frame(events)

    assert bool(event_frame.iloc[0]["flag_roll_span"]) is True
    assert bool(event_frame.iloc[0]["excluded"]) is True
    assert int(labeled["label_long_frvp_continuation"].sum()) == 0
    assert int((~labeled["exclude_long_frvp_continuation"]).sum()) == 0


def test_all_exclusion_masks_and_flags_fire_on_constructed_fixture(monkeypatch) -> None:
    bars_per_session = 78
    market = _market_fixture(session_count=25, bars_per_session=bars_per_session)
    features = _features_for_market(market)

    event_positions = {
        1: 2,
        6: 10 * bars_per_session + 6,
        2: 11 * bars_per_session + 6,
        3: 12 * bars_per_session + 6,
        5: 13 * bars_per_session + 6,
        4: 24 * bars_per_session + 6,
    }
    _mark_setup(features, event_positions[1], setup_type=1, side=1)
    _mark_setup(features, event_positions[6], setup_type=6, side=1)
    _mark_setup(features, event_positions[2], setup_type=2, side=1)
    _mark_setup(features, event_positions[3], setup_type=3, side=1)
    _mark_setup(features, event_positions[5], setup_type=5, side=1)
    _mark_setup(features, event_positions[4], setup_type=4, side=-1)

    market.loc[event_positions[6], "in_roll_bracket"] = True
    market.loc[event_positions[2], "macro_event_flag"] = True
    market.loc[event_positions[3], "early_close_flag"] = True
    market.loc[event_positions[5], "quad_witching_flag"] = True
    market.loc[24 * bars_per_session : 25 * bars_per_session - 1, "volume"] = 5.0

    _patch_frvp_context(monkeypatch, market, features)
    _, _, events = build_frvp_labels(_input_frame(market), params=_params(), verbose=False)
    event_frame = frvp_events_to_frame(events).set_index("setup_type")

    assert bool(event_frame.at[1, "flag_first_rth"]) is True
    assert bool(event_frame.at[1, "excluded"]) is True

    assert bool(event_frame.at[6, "flag_roll_bracket"]) is True
    assert bool(event_frame.at[6, "excluded"]) is False

    assert bool(event_frame.at[2, "flag_macro"]) is True
    assert bool(event_frame.at[2, "excluded"]) is True

    assert bool(event_frame.at[3, "flag_half_day"]) is True
    assert bool(event_frame.at[3, "excluded"]) is True

    assert bool(event_frame.at[5, "flag_quad_or_rebalance"]) is True

    assert bool(event_frame.at[4, "flag_thin_session"]) is True
    assert bool(event_frame.at[4, "excluded"]) is True


def test_failed_auction_labels_can_be_disabled_for_phase3(monkeypatch) -> None:
    market = _market_fixture(session_count=1, bars_per_session=70)
    features = _features_for_market(market)
    _mark_setup(features, 20, setup_type=4, side=-1)

    _patch_frvp_context(monkeypatch, market, features)
    labeled, diagnostics, events = build_frvp_labels(
        _input_frame(market),
        params=_params(enable_failed_auction_labels=False),
        verbose=False,
    )
    event_frame = frvp_events_to_frame(events).set_index("setup_type")

    assert bool(event_frame.at[4, "flag_failed_auction_disabled"]) is True
    assert bool(event_frame.at[4, "excluded"]) is True
    assert "failed_auction_setup_disabled" in str(event_frame.at[4, "exclude_reasons"])
    assert int(diagnostics["events_excluded_disabled_setup"]) == 1
    assert int(labeled["label_short_frvp_reversal"].sum()) == 0
    assert bool(labeled.iloc[20]["exclude_short_frvp_reversal"]) is True


def test_setup6_reversal_cooldown_excludes_nearby_duplicates(monkeypatch) -> None:
    market = _market_fixture(session_count=2, bars_per_session=80)
    features = _features_for_market(market)
    _mark_setup(features, 10, setup_type=1, side=1)
    _mark_setup(features, 12, setup_type=6, side=1)
    _mark_setup(features, 20, setup_type=6, side=1)

    _patch_frvp_context(monkeypatch, market, features)
    labeled, diagnostics, events = build_frvp_labels(
        _input_frame(market),
        params=_params(setup6_reversal_cooldown_bars=6),
        verbose=False,
    )
    event_frame = frvp_events_to_frame(events).set_index("swing_index")

    assert bool(event_frame.at[10, "excluded"]) is False
    assert bool(event_frame.at[12, "flag_reversal_cooldown"]) is True
    assert bool(event_frame.at[12, "excluded"]) is True
    assert "setup6_reversal_cooldown" in str(event_frame.at[12, "exclude_reasons"])
    assert bool(event_frame.at[20, "excluded"]) is False
    assert int(diagnostics["events_excluded_reversal_cooldown"]) == 1
    assert bool(labeled.iloc[12]["exclude_long_frvp_reversal"]) is True


def test_reversal_past_htf_confluence_gate_uses_setup_specific_windows(monkeypatch) -> None:
    market = _market_fixture(session_count=2, bars_per_session=80)
    features = _features_for_market(market)
    _mark_setup(features, 10, setup_type=1, side=-1)
    _mark_setup(features, 20, setup_type=1, side=1)
    _mark_setup(features, 30, setup_type=6, side=-1)
    _mark_setup(features, 40, setup_type=6, side=1)

    _patch_frvp_context(monkeypatch, market, features)

    def fake_annotate(events, *_args, **_kwargs):
        spacing = {
            10: (40, 80),
            20: (40, 80),
            30: (40, 80),
            40: (60, 120),
        }
        for event in events:
            bars_30m, bars_1h = spacing.get(int(event.swing_index), (None, None))
            event.bars_since_prev_30m_same = bars_30m
            event.bars_since_prev_1h_same = bars_1h
            event.htf_match_30m = False
            event.htf_match_1h = False
        return events

    monkeypatch.setattr(frvp_engine, "annotate_swing_context", fake_annotate)
    labeled, diagnostics, events = build_frvp_labels(
        _input_frame(market),
        params=_params(enable_reversal_past_htf_confluence_gate=True),
        verbose=False,
    )
    event_frame = frvp_events_to_frame(events).set_index("swing_index")

    assert bool(event_frame.at[10, "flag_past_htf_confluence"]) is True
    assert bool(event_frame.at[10, "excluded"]) is True
    assert "past_htf_confluence_gate" in str(event_frame.at[10, "exclude_reasons"])

    assert bool(event_frame.at[20, "excluded"]) is False
    assert bool(event_frame.at[30, "excluded"]) is False

    assert bool(event_frame.at[40, "flag_past_htf_confluence"]) is True
    assert bool(event_frame.at[40, "excluded"]) is True

    assert int(diagnostics["events_excluded_past_htf_confluence"]) == 2
    assert bool(labeled.iloc[10]["exclude_short_frvp_reversal"]) is True
    assert bool(labeled.iloc[20]["exclude_long_frvp_reversal"]) is False
    assert bool(labeled.iloc[30]["exclude_short_frvp_reversal"]) is False
    assert bool(labeled.iloc[40]["exclude_long_frvp_reversal"]) is True


def test_frvp_concurrency_downweights_overlapping_events(monkeypatch) -> None:
    market = _market_fixture(session_count=2, bars_per_session=80)
    features = _features_for_market(market)
    _mark_setup(features, 10, setup_type=1, side=1)
    _mark_setup(features, 12, setup_type=6, side=1)
    _mark_setup(features, 90, setup_type=1, side=1)

    _patch_frvp_context(monkeypatch, market, features)
    labeled, _, events = build_frvp_labels(_input_frame(market), params=_params(), verbose=False)
    event_frame = frvp_events_to_frame(events).set_index("swing_index")

    assert int(labeled.iloc[12]["concurrency_long_frvp_reversal"]) == 2
    assert int(event_frame.at[12, "frvp_concurrency"]) == 2
    assert float(labeled.iloc[12]["sample_weight_long_frvp_reversal"]) < float(
        labeled.iloc[90]["sample_weight_long_frvp_reversal"]
    )


def test_session_context_prefers_explicit_calendar_flags_over_empirical_fallback() -> None:
    market = _market_fixture(session_count=1, bars_per_session=78)
    market["equity_early_close_flag"] = True
    market["macro_event_flag"] = False

    session_summary, diagnostics = _session_context(market, 5.0, _params())

    assert bool(session_summary.iloc[0]["half_day_or_early_close"]) is True
    assert diagnostics["todo_halfday_calendar_overrides_pending"] is False
    assert diagnostics["halfday_detection_uses_empirical_fallback"] is False


def test_session_context_disables_empirical_half_day_when_explicit_calendar_columns_exist() -> None:
    market = _market_fixture(session_count=1, bars_per_session=40)
    market["equity_early_close_flag"] = False
    market["equity_half_day_flag"] = False
    market["macro_event_flag"] = False

    session_summary, diagnostics = _session_context(market, 5.0, _params())

    assert bool(session_summary.iloc[0]["empirical_half_day"]) is False
    assert bool(session_summary.iloc[0]["half_day_or_early_close"]) is False
    assert diagnostics["todo_halfday_calendar_overrides_pending"] is False
    assert diagnostics["halfday_detection_uses_empirical_fallback"] is False

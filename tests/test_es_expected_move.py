"""Causality, raw-app parity, units and missing-value gates for ES movement."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ote_live.features.manifest import LivePolicy
from ote_live.ingestion.signals import LiveSignalProcessor
from ote_live.policies.abstain import LiveAbstainState, evaluate_live_abstain
from ote_live.policies.expected_move import CONTRACT, SPREAD_TICKS, build_expected_move_frame, expected_move_spread_check
from scripts.backtest_b2_s3_corrected import full_policy_decision


def bars(n=70):
    close = 5000 + np.arange(n) * .25
    return pd.DataFrame(dict(timestamp=pd.date_range("2026-01-05T14:00Z", periods=n, freq="5min"),
        open=close, high=close+.25, low=close-.25, close=close, instrument_id=42,
        is_complete=True, asset="ES", timeframe="5m"))


@pytest.fixture
def policy():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "configs/b2_s3_corrected_backtest_20260927.json").read_text())["full_policy"]
    p["expected_move_contract"] = CONTRACT
    p["cost_assumptions"]["session_spread_ticks"] = SPREAD_TICKS
    return LivePolicy.model_validate(p)


def row_for(market):
    row = build_expected_move_frame(market).iloc[-1].to_dict()
    row.update(datetime=market.timestamp.iloc[-1], session_regime="new_york", stress_regime="normal")
    return row


def test_known_movement_zero_missing_and_future_prefix():
    market = bars()
    result = build_expected_move_frame(market)
    assert result.expected_move_ticks.iloc[:20].isna().all()
    assert result.expected_move_ticks.iloc[20:].eq(1.).all()
    assert result.expected_move_sample_count.iloc[20:].eq(20).all()
    prefix = build_expected_move_frame(market.iloc[:40])
    changed = market.copy()
    changed.loc[40:, ["open", "high", "low", "close"]] *= 2
    pd.testing.assert_frame_equal(prefix, build_expected_move_frame(changed).iloc[:40])
    flat = market.copy()
    flat[["open", "high", "low", "close"]] = 5000.
    assert build_expected_move_frame(flat).iloc[-1].expected_move_ticks == 0
    assert build_expected_move_frame(flat).iloc[-1].expected_move_status == "ready"


@pytest.mark.parametrize("kind", ["gap", "roll", "incomplete", "missing_identity", "nan", "invalid_ohlc"])
def test_bad_context_resets_warmup(kind):
    market = bars()
    if kind == "gap": market.loc[30:, "timestamp"] += pd.Timedelta(minutes=5)
    if kind == "roll": market.loc[30:, "instrument_id"] = 43
    if kind == "incomplete": market.loc[30, "is_complete"] = False
    if kind == "missing_identity": market.loc[30, "instrument_id"] = np.nan
    if kind == "nan": market.loc[30, "close"] = np.nan
    if kind == "invalid_ohlc": market.loc[30, "high"] = 1.
    result = build_expected_move_frame(market)
    assert result.expected_move_ticks.iloc[30:50].isna().all()
    assert result.expected_move_ticks.iloc[-1] == 1.


def test_raw_app_uses_full_history_before_feature_trimming_and_never_imputed_close():
    market = bars()
    features = pd.DataFrame({"close": [99999.] * 8, "expected_move_ticks": [99999.] * 8})
    processor = SimpleNamespace(asset="ES", timeframe="5m")
    app = LiveSignalProcessor._build_policy_frame_from_market_frame(processor, features, market_frame=market)
    offline = build_expected_move_frame(market).tail(8).reset_index(drop=True)
    pd.testing.assert_frame_equal(app[list(offline)], offline)
    assert app.expected_move_ticks.eq(1).all()


def test_units_b1_schedule_and_legacy_serialization(policy):
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "configs/outcome_contracts/b1_v3.json").read_text())
    contract = next(c for c in catalog["contracts"] if c["contract_id"] == "es_frvp_setup3_long_eval_v2")
    assert dict(contract["costs"]["session_spreads"]) == SPREAD_TICKS
    assert contract["costs"]["unit"] == "ticks" and contract["costs"]["price_increment"] == .25
    raw = json.loads((root / "configs/b2_s3_corrected_backtest_20260927.json").read_text())["full_policy"]
    assert LivePolicy.model_validate(raw).model_dump(mode="json") == raw
    row = row_for(bars())
    assert expected_move_spread_check(policy, row)["status"] == "rejected"  # 1 tick < 2*1
    row["expected_move_ticks"] = 2.
    assert expected_move_spread_check(policy, row)["status"] == "passed"
    row["expected_move_ticks"] = 0.
    assert expected_move_spread_check(policy, row)["status"] == "rejected"  # valid zero
    row["expected_move_ticks"] = np.nan
    row["expected_move_pips"] = 100000.
    assert expected_move_spread_check(policy, row)["status"] == "unverified"
    assert evaluate_live_abstain(policy, probability=.7, source_row_idx=70, policy_context=row).abstain


def test_shadow_full_inputs_and_all_isolated_gates(policy):
    row = row_for(bars())
    row["expected_move_ticks"] = 4.
    for changed, reason in [({}, None), ({"stress_regime": "high"}, "high_stress"),
                            ({"expected_move_status": "warmup"}, "expected_move_not_ready"),
                            ({"expected_move_source_close_ts": "2025-01-01T00:00Z"}, "expected_move_source_timestamp_mismatch")]:
        result = full_policy_decision(policy, {**row, **changed}, probability=.7, threshold_passed=True,
                                      source_row_idx=70, ready=True, state=LiveAbstainState())
        assert result["policy_candidate_eligible"] == (reason is None)
        if reason: assert reason in result["rejection_reasons"]
        check = result["policy_checks"]["expected_move_spread"]
        assert check["unit"] == "ticks"


def test_wrong_units_spread_missing_and_out_of_order_rejected(policy):
    row = row_for(bars())
    for change in [{"expected_move_unit": "pips"}, {"expected_move_ticks": -1.},
                   {"expected_move_ticks": np.inf}, {"session_regime": "unknown"}]:
        assert expected_move_spread_check(policy, {**row, **change})["status"] == "unverified"
    wrong = policy.model_copy(update={"cost_assumptions": policy.cost_assumptions.model_copy(update={"session_spread_ticks": None})})
    assert expected_move_spread_check(wrong, row)["status"] == "unverified"
    market = bars()
    market.loc[30, "timestamp"] = market.timestamp.iloc[29]
    assert pd.isna(build_expected_move_frame(market).expected_move_ticks.iloc[30])

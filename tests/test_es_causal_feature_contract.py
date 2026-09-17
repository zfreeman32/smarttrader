from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.config import FeatureBuilderConfig
from features.feature_sets.htf_context import _resample_ohlcv, build_htf_context
from ict.structure.liquidity import build_reference_level_features
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.input_contract import INPUT_CONTRACT_VERSION, evaluate_model_input_contract
from ote_live.features.manifest import LiveRuntimeManifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimpleNamespace(instrument="es", source_timezone="UTC", canonical_timezone="UTC")


def _bars(times) -> pd.DataFrame:
    timestamps = pd.to_datetime(times, utc=True)
    close = 5000 + np.arange(len(timestamps), dtype=float)
    return pd.DataFrame({"datetime": timestamps, "open": close - 0.5, "high": close + 2,
                         "low": close - 2, "close": close, "volume": 100., "atr_14": 4.})


def _manifest(features: list[str], *, verified: bool = True, backend: str = "xgboost") -> LiveRuntimeManifest:
    path = ROOT / "ote_live/runtime_manifests/frvp_es_shadow_20260715/frvp_short_meta_xgb_v1/live_runtime_manifest.json"
    manifest = LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return manifest.model_copy(update={
        "backend": backend,
        "feature_manifest": manifest.feature_manifest.model_copy(update={
            "selected_feature_names": features, "selected_feature_count": len(features),
            "lag_steps": [0],
            "input_contract_version": INPUT_CONTRACT_VERSION if verified else None,
        }),
        "context_requirements": manifest.context_requirements.model_copy(update={"window_size": 1}),
    })


def test_reference_levels_are_prefix_causal_across_missing_sessions_and_weeks() -> None:
    frame = _bars([
        "2026-08-28 13:30Z", "2026-08-28 19:55Z", "2026-08-28 20:00Z",
        "2026-08-31 12:00Z", "2026-08-31 12:30Z", "2026-08-31 13:25Z",
        "2026-08-31 13:30Z", "2026-08-31 19:55Z", "2026-08-31 20:00Z",
        "2026-09-11 12:00Z", "2026-09-11 13:25Z", "2026-09-11 13:35Z", "2026-09-11 19:55Z",
    ])
    full = build_reference_level_features(frame, CONFIG)
    for count in range(1, len(frame) + 1):
        prefix = build_reference_level_features(frame.iloc[:count], CONFIG)
        pd.testing.assert_frame_equal(prefix, full.iloc[:count], check_dtype=False)
    assert full.loc[2, "ict_prior_rth_high"] == 5003.
    assert full.loc[3, "ict_prior_week_high"] == 5003.
    assert full.loc[9, "ict_prior_rth_high"] == 5009.
    assert np.isnan(full.loc[3, "ict_rth_open"])
    assert np.isnan(full.loc[3, "ict_open_0830"])
    assert full.loc[4, "ict_open_0830"] == frame.loc[4, "open"]
    # Missing the opening bar must not substitute a later RTH open.
    assert np.isnan(full.loc[11, "ict_rth_open"])


def test_reference_levels_with_no_rth_history_and_dst_boundary() -> None:
    frame = _bars(["2026-03-06 14:30Z", "2026-03-06 20:55Z", "2026-03-08 22:00Z",
                   "2026-03-09 12:30Z", "2026-03-09 13:30Z", "2026-03-09 14:30Z"])
    full = build_reference_level_features(frame, CONFIG)
    for count in range(1, len(frame) + 1):
        pd.testing.assert_frame_equal(build_reference_level_features(frame.iloc[:count], CONFIG), full.iloc[:count], check_dtype=False)
    assert full.loc[3, "ict_prior_rth_high"] == 5003.
    no_history = build_reference_level_features(frame.iloc[2:4].reset_index(drop=True), CONFIG)
    assert no_history[["ict_prior_rth_high", "ict_prior_week_high", "ict_rth_open"]].isna().all().all()


def test_live_engine_reference_features_match_causal_producer_and_cached_prefixes() -> None:
    frame = _bars([
        "2026-08-28 13:30Z", "2026-08-28 19:55Z", "2026-08-28 20:00Z",
        "2026-08-30 22:00Z", "2026-09-14 12:25Z", "2026-09-14 12:30Z",
        "2026-09-14 13:25Z", "2026-09-14 13:30Z", "2026-09-14 14:30Z",
    ])
    expected = build_reference_level_features(frame, CONFIG)
    # Exercise the real registry, ICT detector stack, builder, and engine cache.
    # No model artifact needs to be loaded to prove producer/runtime parity.
    names = list(expected.columns)
    engine = IncrementalFeatureEngine([_manifest(names, verified=False)])
    full = engine.build_feature_frame(frame)
    pd.testing.assert_frame_equal(full, expected, check_dtype=False)
    for count in range(1, len(frame) + 1):
        prefix = engine.build_feature_frame(frame.iloc[:count])
        pd.testing.assert_frame_equal(prefix, full.iloc[:count], check_dtype=False)
        pd.testing.assert_frame_equal(engine.build_feature_frame(frame.iloc[:count]), prefix)
        assert engine.last_build_from_cache
    # Correct producer output does not verify the deployed training lineage.
    assert "corrected_feature_lineage_unverified" in engine.input_contract_status(
        engine.manifests[0].model_id,
    )["reasons"]


def test_htf_bar_open_buckets_match_constituents_and_publish_only_at_completion() -> None:
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=25, freq="5min"))
    indexed = frame.set_index("datetime")
    result = _resample_ohlcv(indexed, "30min", bar_minutes=5)
    first = result.iloc[0]
    assert result.index[0] == pd.Timestamp("2026-09-10 12:30Z")
    assert first["open"] == indexed.iloc[0]["open"]
    assert first["high"] == indexed.iloc[:6]["high"].max()
    assert first["low"] == indexed.iloc[:6]["low"].min()
    assert first["close"] == indexed.iloc[5]["close"]
    assert first["volume"] == 600.
    full = build_htf_context(frame, FeatureBuilderConfig(instrument="es"))
    assert full.loc[:4, "htf_30m_ema_alignment"].isna().all()
    assert full.loc[5, "htf_30m_source_age_minutes"] == 0.
    assert full.loc[10, "htf_30m_source_age_minutes"] == 25.
    assert full.loc[11, "htf_1h_source_age_minutes"] == 0.
    for count in (1, 5, 6, 7, 11, 12, 13, 24):
        prefix = build_htf_context(frame.iloc[:count], FeatureBuilderConfig(instrument="es"))
        pd.testing.assert_frame_equal(prefix, full.iloc[:count], check_dtype=False)


@pytest.mark.parametrize("defect", ["gap", "provisional", "unknown_completion"])
def test_htf_rejects_partial_buckets_and_marks_stale_required_inputs(defect: str) -> None:
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=12, freq="5min"))
    frame["is_complete"] = True
    if defect == "gap":
        frame = frame.drop(index=8).reset_index(drop=True)
    elif defect == "provisional":
        frame.loc[8, "is_complete"] = False
    else:
        frame["is_complete"] = frame["is_complete"].astype(object)
        frame.loc[8, "is_complete"] = None
    result = build_htf_context(frame, FeatureBuilderConfig(instrument="es"))
    result["datetime"] = frame["datetime"]
    status = evaluate_model_input_contract(_manifest(["htf_30m_ema_alignment"]), result)
    assert status["diagnostic_only"]
    assert status["feature_states"]["stale_input"] == ["htf_30m_ema_alignment"]


def test_htf_daily_weekly_levels_publish_the_just_completed_period_across_weekend() -> None:
    frame = _bars(["2026-09-04 20:50Z", "2026-09-04 20:55Z", "2026-09-06 22:00Z",
                   "2026-09-08 13:30Z", "2026-09-08 20:55Z"])
    config = FeatureBuilderConfig(instrument="es")
    full = build_htf_context(frame, config)
    assert np.isnan(full.loc[0, "htf_prev_day_high"])
    assert full.loc[1, "htf_prev_day_high"] == 5003.
    assert full.loc[2, "htf_prev_day_high"] == 5003.
    assert full.loc[2, "htf_rolling_weekly_high"] == 5003.
    for count in range(1, len(frame) + 1):
        pd.testing.assert_frame_equal(build_htf_context(frame.iloc[:count], config), full.iloc[:count], check_dtype=False)


def test_input_contract_separates_event_missingness_fallback_missing_producers_and_lineage() -> None:
    names = ["dist_to_bear_order_block_atr", "ict_prior_rth_high", "htf_confluence_short_ict_reversal", "unbuilt"]
    frame = pd.DataFrame({"datetime": [pd.Timestamp("2026-09-11 12:00Z")],
                          "dist_to_bear_order_block_atr": [np.nan], "ict_prior_rth_high": [np.nan],
                          "htf_confluence_short_ict_reversal": [0.]})
    status = evaluate_model_input_contract(_manifest(names, verified=False), frame,
        fallback_features=["htf_confluence_short_ict_reversal"])
    assert status["feature_states"]["event_dependent_missing"] == ["dist_to_bear_order_block_atr"]
    assert status["feature_states"]["missing_history_or_input"] == ["ict_prior_rth_high"]
    assert status["feature_states"]["missing_producer"] == ["htf_confluence_short_ict_reversal", "unbuilt"]
    assert "corrected_feature_lineage_unverified" in status["reasons"]


def test_missing_current_open_is_valid_only_before_open_is_observable() -> None:
    manifest = _manifest(["ict_rth_open", "dist_to_bear_order_block_atr"])
    frame = pd.DataFrame({"datetime": [pd.Timestamp("2026-09-11 12:00Z")], "ict_rth_open": [np.nan], "dist_to_bear_order_block_atr": [np.nan]})
    assert evaluate_model_input_contract(manifest, frame)["status"] == "passed"
    frame["datetime"] = pd.Timestamp("2026-09-11 14:00Z")
    assert evaluate_model_input_contract(manifest, frame)["feature_states"]["missing_history_or_input"] == ["ict_rth_open"]


def test_external_htf_helper_requires_producer_parity_and_current_source() -> None:
    name = "htf_confluence_short_frvp_continuation"
    frame = pd.DataFrame({"datetime": [pd.Timestamp("2026-09-11 12:00Z")], name: [0.]})
    manifest = _manifest([name])
    assert evaluate_model_input_contract(manifest, frame)["feature_states"]["unverified_producer"] == [name]
    producer = {name: {"producer_id": "event-confluence", "version": "v1", "parity_verified": True,
                       "source_timestamp": "2026-09-11 11:55Z"}}
    assert evaluate_model_input_contract(manifest, frame, producer_contracts=producer)["feature_states"]["stale_input"] == [name]
    producer[name]["source_timestamp"] = "2026-09-11 12:00Z"
    assert evaluate_model_input_contract(manifest, frame, producer_contracts=producer)["status"] == "passed"


def test_external_helper_attests_every_consumed_source_row() -> None:
    name = "htf_confluence_short_frvp_continuation"
    manifest = _manifest([name], backend="tcn")
    manifest = manifest.model_copy(update={
        "context_requirements": manifest.context_requirements.model_copy(update={"window_size": 2}),
    })
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=2, freq="5min"))
    frame[name] = 0.
    producer = {name: {"producer_id": "test", "version": "v1", "parity_verified": True,
                       "source_timestamps": [frame.iloc[-1]["datetime"].isoformat()]}}
    assert evaluate_model_input_contract(manifest, frame, producer_contracts=producer)["diagnostic_only"]
    producer[name]["source_timestamps"] = [value.isoformat() for value in frame["datetime"]]
    assert evaluate_model_input_contract(manifest, frame, producer_contracts=producer)["status"] == "passed"


def test_engine_keeps_diagnostic_fallback_scores_and_publishes_cached_contract(monkeypatch) -> None:
    feature = "htf_confluence_short_frvp_continuation"
    engine = IncrementalFeatureEngine([_manifest([feature])])
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=2, freq="5min"))
    def build(source):
        output = source.copy()
        for name in engine.plan.built_feature_names:
            if name != feature:
                output[name] = 0.
        return output, {}
    monkeypatch.setattr(engine.builder, "build", build)
    result = engine.build_feature_frame(frame)
    assert result[feature].tolist() == [0, 0]
    status = engine.input_contract_status(engine.manifests[0].model_id, feature_frame=result)
    assert status["diagnostic_only"]
    assert status["fallback_features"] == [feature]
    engine.build_feature_frame(frame)
    assert engine.last_build_from_cache
    assert engine.last_build_metadata["model_input_contracts"][engine.manifests[0].model_id] == status


def test_engine_checks_raw_missingness_before_sequence_model_zero_imputation(monkeypatch) -> None:
    engine = IncrementalFeatureEngine([_manifest(["ict_prior_rth_high"], backend="tcn")])
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=2, freq="5min"))
    def build(source):
        output = source.copy()
        for name in engine.plan.built_feature_names:
            output[name] = np.nan if name == "ict_prior_rth_high" else 0.
        return output, {}
    monkeypatch.setattr(engine.builder, "build", build)
    result = engine.build_feature_frame(frame)
    assert result["ict_prior_rth_high"].tolist() == [0., 0.]
    assert engine.input_contract_status(engine.manifests[0].model_id)["feature_states"]["missing_history_or_input"] == ["ict_prior_rth_high"]


@pytest.mark.parametrize("defect", ["off_grid", "duplicate", "missing_high", "infinite_low"])
def test_htf_rejects_invalid_interior_constituents(defect) -> None:
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=12, freq="5min"))
    if defect == "off_grid":
        frame.loc[8, "datetime"] += pd.Timedelta(minutes=1)
    elif defect == "duplicate":
        frame.loc[8, "datetime"] = frame.loc[7, "datetime"]
    elif defect == "missing_high":
        frame.loc[8, "high"] = np.nan
    else:
        frame.loc[8, "low"] = -np.inf
    result = _resample_ohlcv(frame.set_index("datetime"), "30min", bar_minutes=5)
    assert list(result.index) == [pd.Timestamp("2026-09-10 12:30Z")]


def test_htf_preserves_original_rows_with_unknown_timestamps() -> None:
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=13, freq="5min"))
    frame.loc[12, "datetime"] = pd.NaT
    frame.index = pd.Index(range(100, 113))
    result = build_htf_context(frame, FeatureBuilderConfig(instrument="es"))
    assert result.index.equals(frame.index)
    assert result.loc[111, "htf_1h_source_age_minutes"] == 0
    assert pd.isna(result.loc[112, "htf_1h_source_complete"])


@pytest.mark.parametrize("complete,age", [(np.nan, 0), (None, 0), (False, 0), (True, -1), (True, np.inf)])
def test_htf_contract_rejects_unknown_completion_or_invalid_age(complete, age) -> None:
    name = "htf_30m_latest_swing_high"
    frame = pd.DataFrame({name: [np.nan], "htf_30m_source_complete": [complete],
                          "htf_30m_source_age_minutes": [age]})
    status = evaluate_model_input_contract(_manifest([name]), frame)
    assert status["diagnostic_only"]
    assert status["feature_states"]["event_dependent_missing"] == []


def test_event_transform_warmup_and_infinity_are_not_valid_event_absence() -> None:
    names = ["ict_nearest_bull_fvg_lower_zscore_20", "dist_to_bear_order_block_atr"]
    frame = pd.DataFrame({names[0]: [np.nan], "ict_nearest_bull_fvg_lower": [1.], names[1]: [np.inf]})
    status = evaluate_model_input_contract(_manifest(names), frame)
    assert status["feature_states"]["missing_history_or_input"] == names
    name = "frvp_s3_pullback_channel_slope_zscore_20"
    frame = pd.DataFrame({name: [np.nan] * 20, "frvp_s3_pullback_channel_slope": [0.] * 20})
    assert evaluate_model_input_contract(_manifest([name]), frame)["status"] == "passed"
    assert evaluate_model_input_contract(_manifest([name]), frame.iloc[:19])["diagnostic_only"]
    name = "ict_nearest_bull_fvg_lower_lag_1"
    frame = pd.DataFrame({name: [np.nan] * 2, "ict_nearest_bull_fvg_lower": [np.nan] * 2})
    assert evaluate_model_input_contract(_manifest([name]), frame)["status"] == "passed"
    assert evaluate_model_input_contract(_manifest([name]), frame.iloc[:1])["diagnostic_only"]


def test_transformed_alignment_and_fallback_dependencies_cannot_bypass_contract() -> None:
    name = "htf_alignment_score_roll_mean_20"
    frame = pd.DataFrame({name: [1.], "htf_30m_source_complete": [1], "htf_1h_source_complete": [1],
                          "htf_30m_source_age_minutes": [30.], "htf_1h_source_age_minutes": [0.]})
    assert evaluate_model_input_contract(_manifest([name]), frame)["feature_states"]["stale_input"] == [name]
    name = "htf_confluence_short_ict_reversal_lag_1"
    frame[name] = 0.
    status = evaluate_model_input_contract(_manifest([name]), frame,
        missing_producer_features=["htf_confluence_short_ict_reversal"])
    assert status["feature_states"]["missing_producer"] == [name]


@pytest.mark.parametrize("backend", ["xgboost", "tcn", "lstm"])
def test_contract_checks_earlier_rows_consumed_by_inference(backend) -> None:
    name = "htf_30m_ema_alignment"
    manifest = _manifest([name], backend=backend)
    manifest = manifest.model_copy(update={
        "feature_manifest": manifest.feature_manifest.model_copy(update={"lag_steps": [0, 2]}),
        "context_requirements": manifest.context_requirements.model_copy(update={"window_size": 3}),
    })
    frame = pd.DataFrame({name: [1.] * 3, "htf_30m_source_complete": [1] * 3,
                          "htf_30m_source_age_minutes": [30., 5., 0.]})
    status = evaluate_model_input_contract(manifest, frame)
    assert status["feature_states"]["available"] == [name]
    assert "historical_required_inputs_failed" in status["reasons"]
    assert status["history_failures"][0]["row_offset"] == 2
    frame.loc[0, "htf_30m_source_age_minutes"] = 25.
    assert evaluate_model_input_contract(manifest, frame)["status"] == "passed"
    assert evaluate_model_input_contract(manifest, frame.iloc[-1:])["diagnostic_only"]


def test_external_producer_attestation_changes_invalidate_cached_contract(monkeypatch) -> None:
    name = "htf_confluence_short_frvp_continuation"
    engine = IncrementalFeatureEngine([_manifest([name])])
    source = _bars(pd.date_range("2026-09-10 12:00Z", periods=2, freq="5min"))
    source[name] = 0.
    def build(frame):
        return frame.assign(**{feature: 0. for feature in engine.plan.built_feature_names if feature not in frame}), {}
    monkeypatch.setattr(engine.builder, "build", build)
    engine.build_feature_frame(source)
    assert engine.input_contract_status(engine.manifests[0].model_id)["diagnostic_only"]
    source.attrs["feature_producer_contracts"] = {name: {
        "producer_id": "verified-test-producer", "version": "v1", "parity_verified": True,
        "source_timestamp": source.iloc[-1]["datetime"].isoformat(),
    }}
    engine.build_feature_frame(source)
    assert not engine.last_build_from_cache
    assert engine.input_contract_status(engine.manifests[0].model_id)["status"] == "passed"
    source.attrs["feature_producer_contracts"][name]["source_timestamp"] = source.iloc[0]["datetime"].isoformat()
    engine.build_feature_frame(source)
    assert not engine.last_build_from_cache
    assert engine.input_contract_status(engine.manifests[0].model_id)["feature_states"]["stale_input"] == [name]


def test_real_live_htf_builder_matches_producer_and_prefix_cache() -> None:
    names = ["htf_30m_ema_alignment", "htf_1h_ema_alignment", "htf_alignment_score"]
    engine = IncrementalFeatureEngine([_manifest(names)])
    frame = _bars(pd.date_range("2026-09-10 12:00Z", periods=25, freq="5min"))
    expected = build_htf_context(frame, FeatureBuilderConfig(instrument="es"))[names]
    full = engine.build_feature_frame(frame)
    pd.testing.assert_frame_equal(full, expected, check_dtype=False)
    for count in (5, 6, 11, 12, 13, 25):
        prefix = engine.build_feature_frame(frame.iloc[:count])
        pd.testing.assert_frame_equal(prefix, full.iloc[:count], check_dtype=False)
        assert engine.input_contract_status(engine.manifests[0].model_id)["diagnostic_only"] == (count < 12)
        pd.testing.assert_frame_equal(engine.build_feature_frame(frame.iloc[:count]), prefix)
        assert engine.last_build_from_cache


def test_engine_does_not_qualify_an_older_row_when_builder_drops_latest(monkeypatch) -> None:
    engine = IncrementalFeatureEngine([_manifest(["close"])])
    source = _bars(pd.date_range("2026-09-10 12:00Z", periods=2, freq="5min"))
    def build(frame):
        return frame.iloc[:-1].assign(**{name: 0. for name in engine.plan.built_feature_names if name not in frame}), {}
    monkeypatch.setattr(engine.builder, "build", build)
    engine.build_feature_frame(source)
    status = engine.input_contract_status(engine.manifests[0].model_id)
    assert status["diagnostic_only"]
    assert "feature_source_timestamp_mismatch" in status["reasons"]


@pytest.mark.parametrize("ict_review_block", [False, True])
def test_runtime_persists_fallback_contract_and_forces_diagnostic_score_to_shadow(tmp_path, monkeypatch, ict_review_block) -> None:
    from ote_live.contracts.market_data import MarketBar
    from ote_live.contracts.prediction import ModelPrediction
    from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
    from ote_live.models.loaders import LoadedRuntimeModel
    from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
    from ote_live.dashboard.queries import fetch_recent_health_events

    name = "close" if ict_review_block else "htf_confluence_short_frvp_continuation"
    manifest = _manifest([name]).model_copy(update={
        "model_id": "ict_short_meta_xgb_v1" if ict_review_block else "a3_contract_test"})
    manifest = manifest.model_copy(update={
        "context_requirements": manifest.context_requirements.model_copy(update={"minimum_runtime_history_bars": 1}),
        "live_policy": manifest.live_policy.model_copy(update={
            "thresholds": manifest.live_policy.thresholds.model_copy(update={"global_threshold": 0.5, "regime_thresholds": {}}),
            "abstain_policy": manifest.live_policy.abstain_policy.model_copy(update={"enabled": False}),
        }),
    })
    engine = IncrementalFeatureEngine([manifest])
    def build(frame):
        return frame.assign(**{feature: 0. for feature in engine.plan.built_feature_names if feature != name}), {}
    monkeypatch.setattr(engine.builder, "build", build)
    loaded = LoadedRuntimeModel(manifest=manifest, model=None, scaler=None, calibrator=None,
                               training_summary={}, scale_clip=0., batch_size=1, use_amp=False)
    timestamp = pd.Timestamp("2026-09-10 14:00Z").to_pydatetime()
    observed = timestamp + pd.Timedelta(minutes=5, seconds=1)
    bar = MarketBar(asset="ES", timeframe="5m", timestamp=timestamp, open=5000, high=5002,
                    low=4999, close=5001, volume=100, is_complete=True, feed_type="live",
                    observation_kind="live", first_observed_at=observed, last_observed_at=observed,
                    source_timestamp=timestamp, bar_version="a3-test-v1",
                    feature_context={"ibkr_trading_hours": "20260909:1700-20260910:1600", "ibkr_timezone": "America/Chicago"})
    monkeypatch.setattr("ote_live.ingestion.signals.utc_now", lambda: observed)
    with SQLiteLiveDataStore(tmp_path / "a3.sqlite") as store:
        audit = LiveAuditRepository(store)
        processor = LiveSignalProcessor(bindings=[SignalRuntimeModelBinding(loaded_model=loaded, shadow_mode=False)],
                                        audit_repository=audit, feature_engine=engine, enable_ict_paper_signal_ledger=False)
        monkeypatch.setattr(processor._runner_by_model_id[manifest.model_id], "predict_latest",
            lambda frame, **kwargs: ModelPrediction(model_id=manifest.model_id, direction=manifest.direction,
                backend="xgboost", raw_score=0.9, calibrated_probability=0.9, **kwargs))
        results = processor.process_bars([bar], emit_operator_artifacts=False)
        assert len(results) == 1
        trail = audit.reconstruct_signal(results[0].signal_decision_id, include_bars=False)
        assert trail.signal.decision == "shadow"
        metadata = trail.feature_snapshot.observation_metadata
        assert metadata["bar_eligibility"]["eligible"], metadata["bar_eligibility"]
        assert metadata["diagnostic_only"]
        if ict_review_block:
            from ote_live.models.ict_research import ICT_LINEAGE_BLOCK_REASON
            assert metadata["model_input_contract"]["status"] == "passed"
            assert metadata["diagnostic_reasons"] == [ICT_LINEAGE_BLOCK_REASON]
            assert ICT_LINEAGE_BLOCK_REASON in trail.signal.shadow_evaluation["rejection_reasons"]
            assert not trail.signal.shadow_evaluation["policy_candidate_eligible"]
        else:
            assert metadata["model_input_contract"]["fallback_features"] == [name]
        assert not metadata["qualified_shadow_entry"]
        assert trail.prediction.raw_score == 0.9
        if not ict_review_block:
            health = fetch_recent_health_events(audit)
            contract = health.loc[health.event_type == "model_input_contract"].iloc[-1]
            assert contract["severity"] == "warning"
            assert contract["payload"]["model_id"] == manifest.model_id
            assert contract["payload"]["diagnostic_only"]

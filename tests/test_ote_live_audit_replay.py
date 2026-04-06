from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.io import standardize_market_frame
from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore, replay_audited_signal

SHORT_XGB_V1_MANIFEST_PATH = (
    ROOT / "ote_live" / "runtime_manifests" / "short_ote_xgb_v1_candidate" / "live_runtime_manifest.json"
)
EURUSD_5M_PATH = ROOT / "data" / "currency_data" / "eurusd-5m.csv"


def test_audit_replay_service_rebuilds_prediction_and_signal_from_stored_rows() -> None:
    base_manifest = _subset_manifest(
        load_live_runtime_manifest(SHORT_XGB_V1_MANIFEST_PATH),
        [
            "strategy__smi_signals__smi",
            "ema_200",
            "close_vs_ema_8_atr_lag_1",
        ],
    )
    live_policy = base_manifest.live_policy.model_copy(
        update={
            "thresholds": base_manifest.live_policy.thresholds.model_copy(
                update={
                    "global_threshold": 0.0,
                    "regime_thresholds": {},
                }
            ),
            "abstain_policy": base_manifest.live_policy.abstain_policy.model_copy(
                update={"enabled": False}
            ),
        }
    )
    manifest = base_manifest.model_copy(update={"live_policy": live_policy})

    market_frame = _load_standardized_market_frame().tail(420).reset_index(drop=True)
    bars = [
        _market_bar_from_row(row, asset=manifest.asset, timeframe=manifest.timeframe)
        for row in market_frame.itertuples(index=False)
    ]

    tmp_root = ROOT / "tmp" / "ote_live_audit_replay_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        store.upsert_bars(bars)
        audit = LiveAuditRepository(store)
        runner = RuntimeModelRunner(_build_fake_loaded_model(manifest))
        feature_engine = IncrementalFeatureEngine([manifest], rolling_window_bars=420)
        feature_engine.extend(bars)
        feature_frame = feature_engine.build_feature_frame()
        feature_frame_for_runner = feature_frame.copy()
        feature_frame_for_runner["timestamp"] = [bar.timestamp for bar in bars[-len(feature_frame_for_runner) :]]
        feature_frame_for_runner["source_row_idx"] = list(
            range(len(bars) - len(feature_frame_for_runner), len(bars))
        )

        decision_engine = LiveDecisionEngine(audit_repository=audit, persist_decisions=("emit",))
        decision_path = decision_engine.run_runner(
            runner,
            feature_frame_for_runner,
            policy_context={
                "session_regime": "new_york",
                "stress_regime": "normal",
                "composite_regime": "flat_normal",
            },
        )
        assert decision_path.audit_record is not None
        assert decision_path.signal.decision == "emit"

        replay_report = replay_audited_signal(
            audit,
            decision_path.audit_record.signal_decision_id,
            model_loader=_build_fake_loaded_model,
        )

    assert replay_report.matches
    assert replay_report.feature_snapshot_matches
    assert replay_report.prediction_matches
    assert replay_report.signal_matches
    assert replay_report.feature_mismatches == ()
    assert replay_report.raw_probability_abs_diff <= replay_report.raw_probability_atol
    assert replay_report.calibrated_probability_abs_diff <= replay_report.calibrated_probability_atol
    assert replay_report.replayed_signal == decision_path.signal
    assert replay_report.replayed_prediction == decision_path.prediction


def _subset_manifest(
    manifest: LiveRuntimeManifest,
    selected_feature_names: list[str],
) -> LiveRuntimeManifest:
    feature_manifest = manifest.feature_manifest.model_copy(
        update={
            "selected_feature_names": list(selected_feature_names),
            "selected_feature_count": len(selected_feature_names),
            "strategy_feature_count": sum(name.startswith("strategy__") for name in selected_feature_names),
        }
    )
    return manifest.model_copy(update={"feature_manifest": feature_manifest})


def _build_fake_loaded_model(manifest: LiveRuntimeManifest) -> LoadedRuntimeModel:
    return LoadedRuntimeModel(
        manifest=manifest,
        model=_ConstantBooster(probability=0.6),
        scaler=None,
        calibrator=None,
        training_summary={},
        scale_clip=0.0,
        batch_size=1,
        use_amp=False,
    )


def _load_standardized_market_frame() -> pd.DataFrame:
    raw = pd.read_csv(EURUSD_5M_PATH)
    standardized = standardize_market_frame(raw, source_timezone="GMT-6", canonical_timezone="UTC")
    standardized["datetime"] = pd.to_datetime(standardized["datetime"], errors="coerce", utc=True)
    return standardized.dropna(subset=["datetime"]).reset_index(drop=True)


def _market_bar_from_row(row, *, asset: str, timeframe: str) -> MarketBar:
    return MarketBar(
        asset=asset,
        timeframe=timeframe,
        timestamp=pd.Timestamp(getattr(row, "datetime")).to_pydatetime(),
        open=float(getattr(row, "open")),
        high=float(getattr(row, "high")),
        low=float(getattr(row, "low")),
        close=float(getattr(row, "close")),
        volume=float(getattr(row, "volume", 0.0) or 0.0),
        bid=None,
        ask=None,
        spread=None,
        source=getattr(row, "source", None),
    )


class _ConstantBooster:
    def __init__(self, *, probability: float) -> None:
        self.probability = float(probability)

    def predict(self, _dmatrix) -> np.ndarray:
        clipped_probability = min(max(self.probability, 1e-6), 1.0 - 1e-6)
        logit = np.log(clipped_probability / (1.0 - clipped_probability))
        return np.asarray([logit], dtype=np.float32)

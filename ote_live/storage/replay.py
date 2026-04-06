from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Callable

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.models.loaders import LoadedRuntimeModel, load_runtime_model
from ote_live.models.replay import (
    DEFAULT_TORCH_CALIBRATED_PROBABILITY_ATOL,
    DEFAULT_TORCH_RAW_PROBABILITY_ATOL,
    DEFAULT_XGBOOST_CALIBRATED_PROBABILITY_ATOL,
    DEFAULT_XGBOOST_RAW_PROBABILITY_ATOL,
)
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.decision_engine import LiveDecisionEngine, ModelDecisionPath
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.repositories import LiveAuditRepository, SignalAuditTrail


@dataclass(frozen=True)
class FeatureReplayMismatch:
    feature_name: str
    expected_value: Any
    actual_value: Any


@dataclass(frozen=True)
class AuditReplayReport:
    signal_decision_id: int
    model_id: str
    backend: str
    direction: str
    bars_replayed: int
    feature_values_compared: int
    feature_mismatches: tuple[FeatureReplayMismatch, ...]
    raw_probability_abs_diff: float
    calibrated_probability_abs_diff: float
    raw_probability_atol: float
    calibrated_probability_atol: float
    prediction_matches: bool
    signal_matches: bool
    replayed_feature_snapshot: FeatureSnapshot
    replayed_prediction: ModelPrediction
    replayed_signal: SignalDecision
    stored_audit_trail: SignalAuditTrail

    @property
    def feature_snapshot_matches(self) -> bool:
        return not self.feature_mismatches

    @property
    def matches(self) -> bool:
        return self.feature_snapshot_matches and self.prediction_matches and self.signal_matches

    def assertion_message(self) -> str:
        issues: list[str] = []
        if not self.feature_snapshot_matches:
            sample = ", ".join(
                f"{item.feature_name}: expected={item.expected_value!r} actual={item.actual_value!r}"
                for item in self.feature_mismatches[:5]
            )
            issues.append(f"feature snapshot mismatches={len(self.feature_mismatches)} ({sample})")
        if not self.prediction_matches:
            issues.append(
                "prediction mismatch "
                f"(raw_diff={self.raw_probability_abs_diff:.6g}, calibrated_diff={self.calibrated_probability_abs_diff:.6g})"
            )
        if not self.signal_matches:
            issues.append("signal decision mismatch")
        if not issues:
            return (
                f"Audit replay for signal_decision_id={self.signal_decision_id} unexpectedly reported a mismatch-free "
                "state without any mismatch details."
            )
        return f"Audit replay mismatch for signal_decision_id={self.signal_decision_id}: " + "; ".join(issues)


def replay_audited_signal(
    audit_repository_or_store: LiveAuditRepository | SQLiteLiveDataStore | str | Path,
    signal_decision_id: int,
    *,
    feature_value_atol: float = 1e-9,
    raw_probability_atol: float | None = None,
    calibrated_probability_atol: float | None = None,
    assert_matches: bool = True,
    model_loader: Callable[[LiveRuntimeManifest], LoadedRuntimeModel] | None = None,
) -> AuditReplayReport:
    audit_repository = _coerce_audit_repository(audit_repository_or_store)
    audit_trail = audit_repository.reconstruct_signal(signal_decision_id, include_bars=True)
    if audit_trail.runtime_manifest is None:
        raise ValueError(
            f"Signal decision {signal_decision_id} does not have a stored runtime manifest for audit replay."
        )
    if not audit_trail.bars:
        raise ValueError(f"Signal decision {signal_decision_id} does not have stored bar history for audit replay.")

    manifest = audit_trail.runtime_manifest.manifest
    loaded_model = (
        model_loader(manifest)
        if model_loader is not None
        else load_runtime_model(manifest)
    )
    raw_probability_atol = float(
        raw_probability_atol
        if raw_probability_atol is not None
        else (
            DEFAULT_XGBOOST_RAW_PROBABILITY_ATOL
            if loaded_model.backend == "xgboost"
            else DEFAULT_TORCH_RAW_PROBABILITY_ATOL
        )
    )
    calibrated_probability_atol = float(
        calibrated_probability_atol
        if calibrated_probability_atol is not None
        else (
            DEFAULT_XGBOOST_CALIBRATED_PROBABILITY_ATOL
            if loaded_model.backend == "xgboost"
            else DEFAULT_TORCH_CALIBRATED_PROBABILITY_ATOL
        )
    )

    replayed_feature_snapshot, replayed_path = _rebuild_prediction_and_signal(
        audit_trail=audit_trail,
        loaded_model=loaded_model,
    )

    feature_mismatches = _compare_feature_snapshots(
        expected=audit_trail.feature_snapshot,
        actual=replayed_feature_snapshot,
        atol=feature_value_atol,
    )
    raw_probability_abs_diff = abs(
        float(replayed_path.prediction.raw_score or 0.0) - float(audit_trail.prediction.raw_score or 0.0)
    )
    calibrated_probability_abs_diff = abs(
        float(replayed_path.prediction.calibrated_probability) - float(audit_trail.prediction.calibrated_probability)
    )
    prediction_matches = _predictions_match(
        expected=audit_trail.prediction,
        actual=replayed_path.prediction,
        raw_probability_atol=raw_probability_atol,
        calibrated_probability_atol=calibrated_probability_atol,
    )
    signal_matches = _signals_match(
        expected=audit_trail.signal,
        actual=replayed_path.signal,
        probability_atol=calibrated_probability_atol,
    )

    report = AuditReplayReport(
        signal_decision_id=int(signal_decision_id),
        model_id=manifest.model_id,
        backend=manifest.backend,
        direction=manifest.direction,
        bars_replayed=len(audit_trail.bars),
        feature_values_compared=len(audit_trail.feature_snapshot.feature_values),
        feature_mismatches=tuple(feature_mismatches),
        raw_probability_abs_diff=float(raw_probability_abs_diff),
        calibrated_probability_abs_diff=float(calibrated_probability_abs_diff),
        raw_probability_atol=raw_probability_atol,
        calibrated_probability_atol=calibrated_probability_atol,
        prediction_matches=prediction_matches,
        signal_matches=signal_matches,
        replayed_feature_snapshot=replayed_feature_snapshot,
        replayed_prediction=replayed_path.prediction,
        replayed_signal=replayed_path.signal,
        stored_audit_trail=audit_trail,
    )
    if assert_matches and not report.matches:
        raise AssertionError(report.assertion_message())
    return report


def _rebuild_prediction_and_signal(
    *,
    audit_trail: SignalAuditTrail,
    loaded_model: LoadedRuntimeModel,
) -> tuple[FeatureSnapshot, ModelDecisionPath]:
    manifest = loaded_model.manifest
    feature_engine = IncrementalFeatureEngine([manifest])
    feature_engine.extend(audit_trail.bars)
    feature_frame = feature_engine.build_feature_frame()
    snapshots = feature_engine.compute_latest_snapshots(require_ready=True)
    replayed_feature_snapshot = snapshots.get(manifest.model_id)
    if replayed_feature_snapshot is None:
        raise ValueError(
            f"Feature engine could not rebuild a ready feature snapshot for model {manifest.model_id} "
            f"while replaying signal decision {audit_trail.signal_decision_id}."
        )

    runner = RuntimeModelRunner(loaded_model)
    replayed_prediction = runner.predict_latest(
        feature_frame,
        timestamp=audit_trail.feature_snapshot.timestamp,
        source_row_idx=audit_trail.feature_snapshot.source_row_idx,
        regime=None,
    )

    policy_context = (
        audit_trail.prediction_metadata.get("policy_context")
        or audit_trail.signal_metadata.get("policy_context")
    )
    if policy_context is None and audit_trail.prediction.regime is not None:
        replayed_prediction = replayed_prediction.model_copy(update={"regime": audit_trail.prediction.regime})

    shadow_mode = bool(
        audit_trail.prediction_metadata.get("shadow_mode")
        or audit_trail.signal_metadata.get("shadow_mode")
        or False
    )
    replayed_path = LiveDecisionEngine(persist_decisions=()).evaluate_prediction(
        replayed_prediction,
        live_policy=manifest.live_policy,
        policy_context=policy_context,
        timezone_contract=manifest.timezone_contract,
        shadow_mode=shadow_mode,
        feature_snapshot=replayed_feature_snapshot,
        runtime_manifest=manifest,
    )
    return replayed_feature_snapshot, replayed_path


def _compare_feature_snapshots(
    *,
    expected: FeatureSnapshot,
    actual: FeatureSnapshot,
    atol: float,
) -> list[FeatureReplayMismatch]:
    mismatches: list[FeatureReplayMismatch] = []
    if expected.asset != actual.asset:
        mismatches.append(FeatureReplayMismatch("asset", expected.asset, actual.asset))
    if expected.timeframe != actual.timeframe:
        mismatches.append(FeatureReplayMismatch("timeframe", expected.timeframe, actual.timeframe))
    if expected.direction != actual.direction:
        mismatches.append(FeatureReplayMismatch("direction", expected.direction, actual.direction))
    if expected.timestamp != actual.timestamp:
        mismatches.append(FeatureReplayMismatch("timestamp", expected.timestamp.isoformat(), actual.timestamp.isoformat()))

    expected_feature_names = set(expected.feature_values)
    actual_feature_names = set(actual.feature_values)
    for missing_feature_name in sorted(expected_feature_names - actual_feature_names):
        mismatches.append(
            FeatureReplayMismatch(missing_feature_name, expected.feature_values[missing_feature_name], "<missing>")
        )
    for extra_feature_name in sorted(actual_feature_names - expected_feature_names):
        mismatches.append(
            FeatureReplayMismatch(extra_feature_name, "<missing>", actual.feature_values[extra_feature_name])
        )

    for feature_name in sorted(expected_feature_names & actual_feature_names):
        expected_value = expected.feature_values[feature_name]
        actual_value = actual.feature_values[feature_name]
        if not _values_match(expected_value, actual_value, atol=atol):
            mismatches.append(FeatureReplayMismatch(feature_name, expected_value, actual_value))
    return mismatches


def _predictions_match(
    *,
    expected: ModelPrediction,
    actual: ModelPrediction,
    raw_probability_atol: float,
    calibrated_probability_atol: float,
) -> bool:
    if expected.model_id != actual.model_id:
        return False
    if expected.direction != actual.direction:
        return False
    if expected.backend != actual.backend:
        return False
    if expected.timestamp != actual.timestamp:
        return False
    if expected.source_row_idx != actual.source_row_idx:
        return False
    if expected.regime != actual.regime:
        return False
    if expected.threshold_source != actual.threshold_source:
        return False
    if not _values_match(expected.threshold_applied, actual.threshold_applied, atol=1e-12):
        return False
    if not _values_match(expected.raw_score, actual.raw_score, atol=raw_probability_atol):
        return False
    if not _values_match(
        expected.calibrated_probability,
        actual.calibrated_probability,
        atol=calibrated_probability_atol,
    ):
        return False
    return True


def _signals_match(
    *,
    expected: SignalDecision,
    actual: SignalDecision,
    probability_atol: float,
) -> bool:
    if expected.model_id != actual.model_id:
        return False
    if expected.direction != actual.direction:
        return False
    if expected.timestamp != actual.timestamp:
        return False
    if expected.source_row_idx != actual.source_row_idx:
        return False
    if expected.decision != actual.decision:
        return False
    if expected.regime != actual.regime:
        return False
    if expected.reasons != actual.reasons:
        return False
    if expected.cooldown_bars_remaining != actual.cooldown_bars_remaining:
        return False
    if not _values_match(expected.threshold, actual.threshold, atol=1e-12):
        return False
    if not _values_match(expected.probability, actual.probability, atol=probability_atol):
        return False
    return True


def _values_match(expected: Any, actual: Any, *, atol: float) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=float(atol))
    return expected == actual


def _coerce_audit_repository(
    audit_repository_or_store: LiveAuditRepository | SQLiteLiveDataStore | str | Path,
) -> LiveAuditRepository:
    if isinstance(audit_repository_or_store, LiveAuditRepository):
        return audit_repository_or_store
    if isinstance(audit_repository_or_store, SQLiteLiveDataStore):
        return LiveAuditRepository(audit_repository_or_store)
    return LiveAuditRepository(SQLiteLiveDataStore(audit_repository_or_store))

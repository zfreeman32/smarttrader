from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.features.manifest import LivePolicy, LiveRuntimeManifest
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.abstain import LiveAbstainState, evaluate_live_abstain
from ote_live.policies.regime import resolve_latest_regime, resolve_policy_context
from ote_live.policies.threshold import clears_threshold, resolve_live_threshold
from ote_live.storage.repositories import LiveAuditRepository


@dataclass(frozen=True)
class ModelDecisionPath:
    prediction: ModelPrediction
    signal: SignalDecision
    audit_record: PersistedAuditRecord | None = None


@dataclass(frozen=True)
class PersistedAuditRecord:
    runtime_manifest_id: int
    feature_snapshot_id: int
    prediction_id: int
    signal_decision_id: int


class LiveDecisionEngine:
    def __init__(
        self,
        *,
        audit_repository: LiveAuditRepository | None = None,
        persist_decisions: set[str] | frozenset[str] | tuple[str, ...] = ("emit",),
    ) -> None:
        self.audit_repository = audit_repository
        self.persist_decisions = frozenset(str(item) for item in persist_decisions)
        self._abstain_state_by_model: dict[str, LiveAbstainState] = {}
        self._runtime_manifest_id_by_key: dict[tuple[str, str, str], int] = {}

    def evaluate_prediction(
        self,
        prediction: ModelPrediction,
        *,
        live_policy: LivePolicy,
        policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None = None,
        timezone_contract: Any | None = None,
        shadow_mode: bool = False,
        feature_snapshot: FeatureSnapshot | None = None,
        runtime_manifest: LiveRuntimeManifest | None = None,
    ) -> ModelDecisionPath:
        resolved_policy_context = resolve_policy_context(
            policy_context,
            timezone_contract=timezone_contract,
            strict=False,
        )
        latest_regime = resolve_latest_regime(
            resolved_policy_context,
            timezone_contract=timezone_contract,
            strict=False,
        )
        composite_regime = prediction.regime or (latest_regime.composite_regime if latest_regime else None)
        threshold_resolution = resolve_live_threshold(
            live_policy,
            regime=composite_regime,
        )

        enriched_prediction = prediction.model_copy(
            update={
                "regime": composite_regime,
                "threshold_applied": float(threshold_resolution.threshold),
                "threshold_source": threshold_resolution.source,
            }
        )
        candidate_passed = clears_threshold(
            enriched_prediction.calibrated_probability,
            threshold_resolution,
        )

        if shadow_mode:
            shadow_reasons = ["shadow_mode"]
            if candidate_passed:
                shadow_reasons.append("candidate_passed_threshold")
            else:
                shadow_reasons.append("probability_below_threshold")
            if threshold_resolution.note:
                shadow_reasons.append(threshold_resolution.note)

            return self._build_decision_path(
                prediction=enriched_prediction,
                signal=SignalDecision(
                    model_id=enriched_prediction.model_id,
                    direction=enriched_prediction.direction,
                    timestamp=enriched_prediction.timestamp,
                    source_row_idx=enriched_prediction.source_row_idx,
                    decision="shadow",
                    probability=enriched_prediction.calibrated_probability,
                    threshold=enriched_prediction.threshold_applied,
                    regime=enriched_prediction.regime,
                    reasons=shadow_reasons,
                    cooldown_bars_remaining=None,
                ),
                feature_snapshot=feature_snapshot,
                runtime_manifest=runtime_manifest,
                policy_context=policy_context,
                shadow_mode=shadow_mode,
            )

        if not candidate_passed:
            reasons = ["probability_below_threshold"]
            if threshold_resolution.note:
                reasons.append(threshold_resolution.note)
            return self._build_decision_path(
                prediction=enriched_prediction,
                signal=SignalDecision(
                    model_id=enriched_prediction.model_id,
                    direction=enriched_prediction.direction,
                    timestamp=enriched_prediction.timestamp,
                    source_row_idx=enriched_prediction.source_row_idx,
                    decision="hold",
                    probability=enriched_prediction.calibrated_probability,
                    threshold=enriched_prediction.threshold_applied,
                    regime=enriched_prediction.regime,
                    reasons=reasons,
                    cooldown_bars_remaining=None,
                ),
                feature_snapshot=feature_snapshot,
                runtime_manifest=runtime_manifest,
                policy_context=policy_context,
                shadow_mode=shadow_mode,
            )

        state = self._abstain_state_by_model.setdefault(
            enriched_prediction.model_id,
            LiveAbstainState(),
        )
        abstain_decision = evaluate_live_abstain(
            live_policy,
            probability=enriched_prediction.calibrated_probability,
            source_row_idx=enriched_prediction.source_row_idx,
            policy_context=resolved_policy_context,
            state=state,
        )
        state.remember_candidate(enriched_prediction.calibrated_probability)

        if abstain_decision.abstain:
            reasons = [abstain_decision.reason] if abstain_decision.reason else ["abstain"]
            if threshold_resolution.note:
                reasons.append(threshold_resolution.note)
            return self._build_decision_path(
                prediction=enriched_prediction,
                signal=SignalDecision(
                    model_id=enriched_prediction.model_id,
                    direction=enriched_prediction.direction,
                    timestamp=enriched_prediction.timestamp,
                    source_row_idx=enriched_prediction.source_row_idx,
                    decision="abstain",
                    probability=enriched_prediction.calibrated_probability,
                    threshold=enriched_prediction.threshold_applied,
                    regime=enriched_prediction.regime,
                    reasons=reasons,
                    cooldown_bars_remaining=abstain_decision.cooldown_bars_remaining,
                ),
                feature_snapshot=feature_snapshot,
                runtime_manifest=runtime_manifest,
                policy_context=policy_context,
                shadow_mode=shadow_mode,
            )

        state.record_emit(enriched_prediction.source_row_idx)
        emit_reasons = ["threshold_passed"]
        if threshold_resolution.note:
            emit_reasons.append(threshold_resolution.note)
        return self._build_decision_path(
            prediction=enriched_prediction,
            signal=SignalDecision(
                model_id=enriched_prediction.model_id,
                direction=enriched_prediction.direction,
                timestamp=enriched_prediction.timestamp,
                source_row_idx=enriched_prediction.source_row_idx,
                decision="emit",
                probability=enriched_prediction.calibrated_probability,
                threshold=enriched_prediction.threshold_applied,
                regime=enriched_prediction.regime,
                reasons=emit_reasons,
                cooldown_bars_remaining=None,
            ),
            feature_snapshot=feature_snapshot,
            runtime_manifest=runtime_manifest,
            policy_context=policy_context,
            shadow_mode=shadow_mode,
        )

    def run_runner(
        self,
        runner: RuntimeModelRunner,
        feature_frame: pd.DataFrame,
        *,
        policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None = None,
        shadow_mode: bool = False,
        feature_snapshot: FeatureSnapshot | None = None,
    ) -> ModelDecisionPath:
        prediction = runner.predict_latest(feature_frame)
        resolved_snapshot = feature_snapshot
        if resolved_snapshot is None and self.audit_repository is not None:
            resolved_snapshot = runner.build_latest_snapshot(feature_frame)
        return self.evaluate_prediction(
            prediction,
            live_policy=runner.loaded_model.manifest.live_policy,
            policy_context=policy_context,
            timezone_contract=runner.loaded_model.manifest.timezone_contract,
            shadow_mode=shadow_mode,
            feature_snapshot=resolved_snapshot,
            runtime_manifest=runner.loaded_model.manifest,
        )

    def _build_decision_path(
        self,
        *,
        prediction: ModelPrediction,
        signal: SignalDecision,
        feature_snapshot: FeatureSnapshot | None,
        runtime_manifest: LiveRuntimeManifest | None,
        policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
        shadow_mode: bool,
    ) -> ModelDecisionPath:
        audit_record = self._persist_decision_path(
            prediction=prediction,
            signal=signal,
            feature_snapshot=feature_snapshot,
            runtime_manifest=runtime_manifest,
            policy_context=policy_context,
            shadow_mode=shadow_mode,
        )
        return ModelDecisionPath(
            prediction=prediction,
            signal=signal,
            audit_record=audit_record,
        )

    def _persist_decision_path(
        self,
        *,
        prediction: ModelPrediction,
        signal: SignalDecision,
        feature_snapshot: FeatureSnapshot | None,
        runtime_manifest: LiveRuntimeManifest | None,
        policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
        shadow_mode: bool,
    ) -> PersistedAuditRecord | None:
        if self.audit_repository is None:
            return None
        if signal.decision not in self.persist_decisions:
            return None
        if feature_snapshot is None or runtime_manifest is None:
            return None

        runtime_manifest_id = self._ensure_runtime_manifest_record(runtime_manifest)
        audit_metadata = {
            "policy_context": _serialize_policy_context(policy_context),
            "shadow_mode": bool(shadow_mode),
        }
        feature_snapshot_id = self.audit_repository.record_feature_snapshot(
            feature_snapshot,
            runtime_manifest_id=runtime_manifest_id,
            metadata={
                "model_id": runtime_manifest.model_id,
                "selected_feature_count": len(runtime_manifest.feature_manifest.selected_feature_names),
            },
        )
        prediction_id = self.audit_repository.record_prediction(
            prediction,
            feature_snapshot_id=feature_snapshot_id,
            runtime_manifest_id=runtime_manifest_id,
            metadata=audit_metadata,
        )
        signal_decision_id = self.audit_repository.record_signal_decision(
            signal,
            prediction_id=prediction_id,
            runtime_manifest_id=runtime_manifest_id,
            metadata=audit_metadata,
        )
        return PersistedAuditRecord(
            runtime_manifest_id=runtime_manifest_id,
            feature_snapshot_id=feature_snapshot_id,
            prediction_id=prediction_id,
            signal_decision_id=signal_decision_id,
        )

    def _ensure_runtime_manifest_record(self, runtime_manifest: LiveRuntimeManifest) -> int:
        cache_key = (
            runtime_manifest.model_id,
            runtime_manifest.generated_at_utc.isoformat(),
            runtime_manifest.manifest_version,
        )
        cached_id = self._runtime_manifest_id_by_key.get(cache_key)
        if cached_id is not None:
            return cached_id

        persisted = self.audit_repository.record_runtime_manifest(runtime_manifest)
        self._runtime_manifest_id_by_key[cache_key] = persisted.runtime_manifest_id
        return persisted.runtime_manifest_id


def _serialize_policy_context(
    policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if policy_context is None:
        return None
    if isinstance(policy_context, pd.DataFrame):
        if policy_context.empty:
            return {}
        return {
            str(key): _to_json_safe(value)
            for key, value in policy_context.iloc[-1].to_dict().items()
        }
    if isinstance(policy_context, pd.Series):
        return {
            str(key): _to_json_safe(value)
            for key, value in policy_context.to_dict().items()
        }
    return {
        str(key): _to_json_safe(value)
        for key, value in dict(policy_context).items()
    }


def _to_json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.features.manifest import LivePolicy, LiveRuntimeManifest
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.abstain import LiveAbstainState, evaluate_live_abstain
from ote_live.policies.regime import resolve_latest_regime, resolve_policy_context
from ote_live.policies.threshold import clears_threshold, resolve_live_threshold
from ote_live.policies.shadow import SHADOW_POLICY_CONTRACT, evaluate_shadow_policy, policy_identity, restore_state, state_payload
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
        self._shadow_state_by_key: dict[tuple, LiveAbstainState] = {}

    def restore_last_emitted_source_row_idx(
        self,
        *,
        model_id: str,
        source_row_idx: int,
    ) -> None:
        """Restore monotonic cooldown state from a canonical persisted emit."""

        restored_idx = int(source_row_idx)
        if restored_idx < 0:
            raise ValueError("Persisted emitted source_row_idx must be non-negative.")
        state = self._abstain_state_by_model.setdefault(str(model_id), LiveAbstainState())
        if (
            state.last_emitted_source_row_idx is None
            or restored_idx > state.last_emitted_source_row_idx
        ):
            state.record_emit(restored_idx)

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
        paper_signal_event_eligible: bool = False,
        force_hold_reasons: Sequence[str] = (),
        shadow_setup_match: dict | None = None,
        shadow_state_before: dict | None = None,
        evaluate_shadow_contract: bool = True,
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
            shadow_evaluation = None
            shadow_key = None
            if (evaluate_shadow_contract and feature_snapshot is not None and feature_snapshot.asset == "ES"
                    and prediction.model_id.lower().startswith(("frvp_", "ict_"))):
                shadow_key = (feature_snapshot.collection_version, prediction.model_id,
                              feature_snapshot.asset, feature_snapshot.timeframe,
                              SHADOW_POLICY_CONTRACT, policy_identity(live_policy))
                previous = shadow_state_before
                if previous is None and shadow_key in self._shadow_state_by_key:
                    previous = state_payload(self._shadow_state_by_key[shadow_key])
                if previous is None and self.audit_repository is not None:
                    previous = self.audit_repository.latest_shadow_policy_state(
                        collection_version=shadow_key[0], model_id=shadow_key[1],
                        contract_version=shadow_key[4], policy_sha256=shadow_key[5],
                        asset=shadow_key[2], timeframe=shadow_key[3],
                    )
                shadow_state = restore_state(previous or {})
                shadow_evaluation = evaluate_shadow_policy(
                    prediction=enriched_prediction, live_policy=live_policy,
                    policy_context=resolved_policy_context, snapshot=feature_snapshot,
                    setup_match=shadow_setup_match, threshold_passed=candidate_passed,
                    state=shadow_state, force_hold_reasons=force_hold_reasons,
                )
            shadow_reasons = ["shadow_mode"]
            if candidate_passed:
                shadow_reasons.append("candidate_passed_threshold")
            else:
                shadow_reasons.append("probability_below_threshold")
            if threshold_resolution.note:
                shadow_reasons.append(threshold_resolution.note)
            if shadow_evaluation is not None:
                shadow_reasons.extend(shadow_evaluation["rejection_reasons"])
                shadow_reasons.extend(shadow_evaluation["prerequisite_reasons"])

            path = self._build_decision_path(
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
                    shadow_evaluation=shadow_evaluation,
                ),
                feature_snapshot=feature_snapshot,
                runtime_manifest=runtime_manifest,
                policy_context=resolved_policy_context,
                shadow_mode=shadow_mode,
                paper_signal_event_eligible=paper_signal_event_eligible,
            )
            if shadow_key is not None:
                self._shadow_state_by_key[shadow_key] = shadow_state
            return path

        forced_reasons = tuple(str(reason) for reason in force_hold_reasons if str(reason))
        if forced_reasons:
            reasons = list(forced_reasons)
            if candidate_passed:
                reasons.append("candidate_passed_threshold")
            else:
                reasons.append("probability_below_threshold")
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
                policy_context=resolved_policy_context,
                shadow_mode=shadow_mode,
                paper_signal_event_eligible=paper_signal_event_eligible,
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
                policy_context=resolved_policy_context,
                shadow_mode=shadow_mode,
                paper_signal_event_eligible=paper_signal_event_eligible,
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
                policy_context=resolved_policy_context,
                shadow_mode=shadow_mode,
                paper_signal_event_eligible=paper_signal_event_eligible,
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
            policy_context=resolved_policy_context,
            shadow_mode=shadow_mode,
            paper_signal_event_eligible=paper_signal_event_eligible,
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
        paper_signal_event_eligible: bool,
    ) -> ModelDecisionPath:
        if feature_snapshot is not None:
            if signal.shadow_evaluation is not None:
                feature_snapshot = feature_snapshot.model_copy(update={"observation_metadata": {
                    **feature_snapshot.observation_metadata, "shadow_evaluation": signal.shadow_evaluation,
                }})
            prediction = prediction.model_copy(update={"collection_version": feature_snapshot.collection_version})
            signal = signal.model_copy(update={"collection_version": feature_snapshot.collection_version})
            diagnostic_reasons = feature_snapshot.observation_metadata.get("diagnostic_reasons", [])
            if diagnostic_reasons:
                signal = signal.model_copy(update={"reasons": list(dict.fromkeys([*signal.reasons, *diagnostic_reasons]))})
        audit_record = self._persist_decision_path(
            prediction=prediction,
            signal=signal,
            feature_snapshot=feature_snapshot,
            runtime_manifest=runtime_manifest,
            policy_context=policy_context,
            shadow_mode=shadow_mode,
            paper_signal_event_eligible=paper_signal_event_eligible,
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
        paper_signal_event_eligible: bool,
    ) -> PersistedAuditRecord | None:
        if self.audit_repository is None:
            return None
        if signal.decision not in self.persist_decisions and signal.shadow_evaluation is None:
            return None
        if feature_snapshot is None or runtime_manifest is None:
            return None

        runtime_manifest_id = self._ensure_runtime_manifest_record(runtime_manifest)
        audit_metadata = {
            **feature_snapshot.observation_metadata,
            "collection_version": feature_snapshot.collection_version,
            "prediction_recorded_at_utc": (
                prediction.prediction_recorded_at_utc.isoformat()
                if prediction.prediction_recorded_at_utc is not None else None
            ),
            "policy_context": _serialize_policy_context(policy_context),
            "shadow_mode": bool(shadow_mode),
            "paper_signal_event_eligible": bool(paper_signal_event_eligible),
        }
        feature_snapshot_id = self.audit_repository.record_feature_snapshot(
            feature_snapshot,
            runtime_manifest_id=runtime_manifest_id,
            metadata={
                **audit_metadata,
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
            runtime_manifest.model_dump_json(),
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

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import pandas as pd

from ote_live.alerts.emailer import EmailDispatchResult, LiveSignalEmailer
from ote_live.alerts.sms import LiveSignalSmsSender, SmsDispatchResult
from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.signal import SignalDecision
from ote_live.ingestion.base import utc_now
from ote_live.ingestion.provenance import assess_shadow_bar_eligibility
from ote_live.storage.collection import collection_identity, register_collection
from ote_live.storage.setup_events import SetupEventRepository
from ote_live.ingestion.setup_revisions import (
    initialize_setup_revision_cursor, mark_setup_evaluated, reconcile_setup_revisions,
)
from ote_live.dashboard.view_state import persist_frvp_dashboard_state, persist_ict_dashboard_state
from ote_live.models.ict_research import ict_research_priority
from ote_live.features.incremental_engine import (
    FRVP_DASHBOARD_EXTRA_FEATURE_NAMES,
    ICT_DASHBOARD_EXTRA_FEATURE_NAMES,
    IncrementalFeatureEngine,
)
from ote_live.media.chart_capture import CapturedChartArtifact, SignalChartCaptureService
from ote_live.models.ensemble import load_direction_models
from ote_live.models.loaders import LoadedRuntimeModel
from ote_live.models.setup_family import resolve_setup_family_gate
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.policies.shadow import match_shadow_setups
from ote_live.storage.repositories import LiveAuditRepository
from ote_live.storage.ict_paper_signal import (
    IctPaperSignalLedgerRepository,
    supports_ict_paper_signal_manifest,
)
from ote_live.storage.frvp_paper_signal import (
    FRVP_PAPER_SIGNAL_MODEL_IDS,
    FrvpPaperSignalLedgerRepository,
    is_frvp_paper_signal_manifest_identity,
    supports_frvp_paper_signal_manifest,
    validate_frvp_paper_signal_manifest,
)

LOGGER = logging.getLogger(__name__)

_FRVP_PAPER_SIGNAL_RUNTIME_AUTHORIZATION = object()


class FrvpPaperSignalContractError(RuntimeError, ValueError):
    """Controlled FRVP construction failed its authorization/integrity contract."""


def _issue_frvp_paper_signal_runtime_authorization() -> object:
    """Issue the in-process capability used by the guarded shared collector."""

    return _FRVP_PAPER_SIGNAL_RUNTIME_AUTHORIZATION


@dataclass(frozen=True)
class SignalRuntimeModelBinding:
    loaded_model: LoadedRuntimeModel
    shadow_mode: bool = False


@dataclass(frozen=True)
class RuntimeSignalResult:
    model_id: str
    direction: str
    decision: str
    shadow_mode: bool
    timestamp: datetime
    signal_decision_id: int | None = None
    notification_status: str | None = None
    sms_notification_status: str | None = None
    media_artifact_id: int | None = None
    snapshot_id: str | None = None
    generation_id: int | None = None
    timings_seconds: dict[str, float] | None = None


@dataclass(frozen=True)
class MarketDataSnapshot:
    snapshot_id: str
    generation_id: int
    asset: str
    timeframe: str
    bar_timestamp: datetime
    source_row_idx: int
    bar: MarketBar
    market_frame: pd.DataFrame
    previous_bar_timestamp: datetime | None = None


@dataclass(frozen=True)
class PreparedSignalSnapshot:
    group_name: str
    snapshot: MarketDataSnapshot
    feature_frame: pd.DataFrame
    policy_frame: pd.DataFrame
    feature_metadata: dict[str, Any]
    timings_seconds: dict[str, float]
    feature_cache_hit: bool = False


class LiveSignalProcessor:
    def __init__(
        self,
        *,
        bindings: Sequence[SignalRuntimeModelBinding],
        audit_repository: LiveAuditRepository,
        feature_engine: IncrementalFeatureEngine | None = None,
        decision_engine: LiveDecisionEngine | None = None,
        emailer: LiveSignalEmailer | None = None,
        sms_sender: LiveSignalSmsSender | None = None,
        chart_capture_service: SignalChartCaptureService | None = None,
        dashboard_url: str | None = None,
        rolling_window_bars: int | None = None,
        group_name: str = "OTE",
        data_supplier: str = "FMP",
        frvp_paper_signal_authorization: object | None = None,
        enable_ict_paper_signal_ledger: bool = True,
        force_shadow_mode: bool = False,
    ) -> None:
        self.bindings = tuple(bindings)
        if not self.bindings:
            raise ValueError("At least one loaded runtime model binding is required.")

        manifests = [binding.loaded_model.manifest for binding in self.bindings]
        controlled_frvp_bindings = tuple(
            binding
            for binding in self.bindings
            if (
                binding.loaded_model.model_id in FRVP_PAPER_SIGNAL_MODEL_IDS
                and (
                    not binding.shadow_mode
                    or is_frvp_paper_signal_manifest_identity(
                        binding.loaded_model.manifest
                    )
                )
            )
        )
        if controlled_frvp_bindings:
            if (
                frvp_paper_signal_authorization
                is not _FRVP_PAPER_SIGNAL_RUNTIME_AUTHORIZATION
            ):
                raise FrvpPaperSignalContractError(
                    "The controlled FRVP paper-signal runtime requires authorization "
                    "from the validated shared ES collector launch guard."
                )
            if len(controlled_frvp_bindings) != 1:
                raise FrvpPaperSignalContractError(
                    "The controlled FRVP paper-signal runtime requires exactly one "
                    "active reversal binding."
                )
            controlled_binding = controlled_frvp_bindings[0]
            if controlled_binding.shadow_mode:
                raise FrvpPaperSignalContractError(
                    "The controlled FRVP reversal binding must load as non-shadow."
                )
            try:
                validate_frvp_paper_signal_manifest(
                    controlled_binding.loaded_model.manifest
                )
            except Exception as exc:
                raise FrvpPaperSignalContractError(
                    "The controlled FRVP paper-signal contract failed its frozen "
                    "manifest hash/economics validation."
                ) from exc
        self.audit_repository = audit_repository
        self.group_name = str(group_name).strip().upper() or "OTE"
        self.data_supplier = str(data_supplier).strip().upper() or "FMP"
        self.force_shadow_mode = bool(force_shadow_mode)
        self._frvp_dashboard_state_enabled = _contains_frvp_models(manifests)
        self._ict_dashboard_state_enabled = _contains_ict_models(manifests)
        self.feature_engine = feature_engine or IncrementalFeatureEngine(
            manifests,
            rolling_window_bars=rolling_window_bars,
            extra_feature_names=_resolve_dashboard_extra_feature_names(
                enable_frvp_dashboard_state=self._frvp_dashboard_state_enabled,
                enable_ict_dashboard_state=self._ict_dashboard_state_enabled,
            ),
        )
        self.decision_engine = decision_engine or LiveDecisionEngine(
            audit_repository=audit_repository,
            persist_decisions=("emit", "shadow", "hold", "abstain"),
        )
        self.emailer = emailer
        self.sms_sender = sms_sender
        self.chart_capture_service = chart_capture_service
        self.dashboard_url = dashboard_url
        self._runner_by_model_id = {
            binding.loaded_model.model_id: RuntimeModelRunner(binding.loaded_model)
            for binding in self.bindings
        }
        first_manifest = manifests[0]
        self.asset = first_manifest.asset
        self.timeframe = first_manifest.timeframe
        self.collection_version, collection_contract = collection_identity(manifests)
        register_collection(self.audit_repository.store, self.collection_version, collection_contract)
        self.setup_event_repository = SetupEventRepository(self.audit_repository.store)
        self._input_contract_health: dict[str, dict] = {}
        self.last_processed_timestamp: datetime | None = None
        if self.asset == "ES" and (self._frvp_dashboard_state_enabled or self._ict_dashboard_state_enabled):
            initialize_setup_revision_cursor(self)
        self._generation_id = 0
        self.ict_paper_signal_ledger = (
            IctPaperSignalLedgerRepository(audit_repository)
            if (
                enable_ict_paper_signal_ledger
                and any(
                    supports_ict_paper_signal_manifest(manifest)
                    for manifest in manifests
                )
            )
            else None
        )
        self.frvp_paper_signal_ledger = (
            FrvpPaperSignalLedgerRepository(audit_repository)
            if any(supports_frvp_paper_signal_manifest(manifest) for manifest in manifests)
            else None
        )
        if self.frvp_paper_signal_ledger is not None:
            for manifest in manifests:
                if not supports_frvp_paper_signal_manifest(manifest):
                    continue
                latest_emit_idx = self.frvp_paper_signal_ledger.latest_emitted_source_row_idx(
                    manifest
                )
                if latest_emit_idx is not None:
                    self.decision_engine.restore_last_emitted_source_row_idx(
                        model_id=manifest.model_id,
                        source_row_idx=latest_emit_idx,
                    )

    @classmethod
    def from_direction_manifest_paths(
        cls,
        *,
        audit_repository: LiveAuditRepository,
        long_manifest_path: str | Path | None,
        short_manifest_path: str | Path | None,
        emailer: LiveSignalEmailer | None = None,
        sms_sender: LiveSignalSmsSender | None = None,
        chart_capture_service: SignalChartCaptureService | None = None,
        dashboard_url: str | None = None,
        rolling_window_bars: int | None = None,
        skip_unavailable_backends: bool = True,
        require_complete_policy: bool = False,
        all_models_active: bool = False,
        force_shadow_mode: bool = False,
        group_name: str = "OTE",
        data_supplier: str = "FMP",
        frvp_paper_signal_authorization: object | None = None,
        enable_ict_paper_signal_ledger: bool = True,
    ) -> LiveSignalProcessor | None:
        bindings: list[SignalRuntimeModelBinding] = []
        for manifest_path in (long_manifest_path, short_manifest_path):
            if manifest_path is None:
                continue
            resolved_manifest_path = Path(manifest_path)
            try:
                bundle = load_direction_models(
                    resolved_manifest_path,
                    skip_unavailable_backends=skip_unavailable_backends,
                    require_complete_policy=require_complete_policy,
                )
            except FileNotFoundError as exc:
                _record_runtime_load_health_event(
                    audit_repository,
                    component="signal_runtime.manifests",
                    event_type="runtime_manifest_missing",
                    severity="warning",
                    message="Runtime direction manifest was not found; skipping that direction.",
                    payload={
                        "manifest_path": str(resolved_manifest_path),
                        "group_name": group_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                continue
            if bundle.unavailable_models:
                _record_runtime_load_health_event(
                    audit_repository,
                    component="signal_runtime.models",
                    event_type="runtime_model_unavailable",
                    severity="warning",
                    message="One or more runtime models were unavailable and skipped.",
                    payload={
                        "manifest_path": str(resolved_manifest_path),
                        "group_name": group_name,
                        "unavailable_models": dict(bundle.unavailable_models),
                    },
                )
            controlled_manifests = tuple(
                manifest
                for manifest in bundle.direction_manifest.models
                if (
                    manifest.model_id in FRVP_PAPER_SIGNAL_MODEL_IDS
                    and (
                        all_models_active
                        or manifest.status == "active"
                        or is_frvp_paper_signal_manifest_identity(manifest)
                    )
                )
            )
            if (
                controlled_manifests
                and frvp_paper_signal_authorization
                is not _FRVP_PAPER_SIGNAL_RUNTIME_AUTHORIZATION
            ):
                raise FrvpPaperSignalContractError(
                    "The controlled FRVP paper-signal runtime requires authorization "
                    "from the validated shared ES collector launch guard."
                )
            configured_primary_id = bundle.direction_manifest.recommendations.recommended_primary_model_id
            primary = bundle.primary_model
            if configured_primary_id is not None and primary is None and not all_models_active:
                loaded_model_ids = ", ".join(sorted(bundle.loaded_models)) or "none"
                unavailable_reason = bundle.unavailable_models.get(configured_primary_id)
                detail = unavailable_reason or "configured primary model was not loaded from the direction bundle"
                raise RuntimeError(
                    "Configured primary model "
                    f"{configured_primary_id!r} could not be loaded from {manifest_path!s}. "
                    f"Loaded models: {loaded_model_ids}. Reason: {detail}"
                )
            if all_models_active:
                for model_manifest in bundle.direction_manifest.models:
                    if getattr(model_manifest, "status", None) == "deprecated":
                        continue
                    loaded_model = bundle.loaded_models.get(model_manifest.model_id)
                    if loaded_model is None:
                        continue
                    bindings.append(
                        SignalRuntimeModelBinding(
                            loaded_model=loaded_model,
                            shadow_mode=bool(force_shadow_mode),
                        )
                    )
                continue
            for model_manifest in bundle.direction_manifest.models:
                if getattr(model_manifest, "status", None) == "deprecated":
                    continue
                loaded_model = bundle.loaded_models.get(model_manifest.model_id)
                if loaded_model is None:
                    continue
                shadow_mode = (
                    bool(force_shadow_mode)
                    or getattr(model_manifest, "status", None) != "active"
                )
                bindings.append(
                    SignalRuntimeModelBinding(
                        loaded_model=loaded_model,
                        shadow_mode=shadow_mode,
                    )
                )
        if not bindings:
            return None
        return cls(
            bindings=bindings,
            audit_repository=audit_repository,
            emailer=emailer,
            sms_sender=sms_sender,
            chart_capture_service=chart_capture_service,
            dashboard_url=dashboard_url,
            rolling_window_bars=rolling_window_bars,
            group_name=group_name,
            data_supplier=data_supplier,
            frvp_paper_signal_authorization=frvp_paper_signal_authorization,
            enable_ict_paper_signal_ledger=enable_ict_paper_signal_ledger,
            force_shadow_mode=force_shadow_mode,
        )

    def warm_from_store(self) -> int:
        if self.feature_engine.state.latest_timestamp is not None:
            return 0
        bars = self._fetch_recent_signal_bars(
            limit=self.feature_engine.plan.runtime_history_bars,
        )
        if not bars:
            return 0
        self.feature_engine.extend(bars)
        self.last_processed_timestamp = bars[-1].timestamp
        return len(bars)

    def process_new_bars_from_store(
        self,
        *,
        emit_operator_artifacts: bool,
        max_timestamp: datetime | None = None,
    ) -> tuple[RuntimeSignalResult, ...]:
        self._reconcile_frvp_paper_signal_events()
        self._reconcile_setup_history()
        if self.last_processed_timestamp is None:
            self.warm_from_store()
            return ()

        new_bars = self._fetch_signal_bars_after(
            self.last_processed_timestamp,
            max_timestamp=max_timestamp,
        )
        if not new_bars:
            return ()
        return self.process_bars(
            new_bars,
            emit_operator_artifacts=emit_operator_artifacts,
        )

    def seed_latest_predictions_from_store(self) -> tuple[RuntimeSignalResult, ...]:
        """Evaluate the latest warmed bar once for models without a persisted prediction."""

        self._reconcile_frvp_paper_signal_events()
        if self.feature_engine.state.latest_timestamp is None:
            self.warm_from_store()
        self._reconcile_setup_history()
        latest_timestamp = self.feature_engine.state.latest_timestamp
        if latest_timestamp is None:
            return ()

        latest_bars = self._fetch_recent_signal_bars(limit=1)
        if not latest_bars or latest_bars[-1].timestamp != latest_timestamp:
            return ()

        existing_model_ids = self._existing_prediction_model_ids(latest_timestamp)
        missing_bindings = tuple(
            binding
            for binding in self.bindings
            if binding.loaded_model.model_id not in existing_model_ids
        )
        if not missing_bindings:
            return ()

        self.last_processed_timestamp = latest_timestamp
        return self._evaluate_ingested_bar(
            latest_bars[-1],
            emit_operator_artifacts=False,
            emit_notifications=False,
            record_paper_signal_events=False,
            bindings=missing_bindings,
        )

    def process_bars(
        self,
        bars: Iterable[MarketBar],
        *,
        emit_operator_artifacts: bool,
    ) -> tuple[RuntimeSignalResult, ...]:
        self._reconcile_frvp_paper_signal_events()
        self._reconcile_setup_history()
        results: list[RuntimeSignalResult] = []
        for bar in sorted(bars, key=lambda item: item.timestamp):
            if not self.ingest_bar_for_evaluation(bar):
                continue
            results.extend(
                self._evaluate_ingested_bar(
                    bar,
                    emit_operator_artifacts=emit_operator_artifacts,
                    emit_notifications=True,
                    record_paper_signal_events=True,
                )
            )

        return tuple(results)

    def ingest_bar_for_evaluation(self, bar: MarketBar) -> bool:
        if bar.asset != self.asset or bar.timeframe != self.timeframe:
            return False

        previous_timestamp = self.feature_engine.state.latest_timestamp
        if previous_timestamp != bar.timestamp:
            self._previous_bar_timestamp = previous_timestamp
        self.feature_engine.ingest_bar(bar)
        self.last_processed_timestamp = bar.timestamp
        ict_paper_signal_ledger = getattr(self, "ict_paper_signal_ledger", None)
        if ict_paper_signal_ledger is not None:
            try:
                ict_paper_signal_ledger.settle_completed_events(bar)
            except Exception as exc:
                self._record_health_event(
                    component="signal_runtime.ict_paper_signal_ledger",
                    event_type="paper_signal_settlement_failed",
                    severity="error",
                    message="Settling completed ICT paper-signal markouts failed.",
                    payload={
                        "timestamp_utc": bar.timestamp.isoformat(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
        frvp_paper_signal_ledger = getattr(self, "frvp_paper_signal_ledger", None)
        if frvp_paper_signal_ledger is not None:
            try:
                frvp_paper_signal_ledger.settle_completed_events(bar)
            except Exception as exc:
                self._record_health_event(
                    component="signal_runtime.frvp_paper_signal_ledger",
                    event_type="paper_signal_settlement_failed",
                    severity="error",
                    message="Settling completed FRVP paper-signal markouts failed.",
                    payload={
                        "timestamp_utc": bar.timestamp.isoformat(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                raise RuntimeError(
                    "Settling FRVP paper-signal markouts failed; signal "
                    "processing is stopped until ledger health is restored."
                ) from exc
        return True

    def prepare_ingested_bar_snapshot(
        self,
        bar: MarketBar,
        *,
        snapshot_id: str | None = None,
        generation_id: int | None = None,
        source_row_idx: int | None = None,
    ) -> PreparedSignalSnapshot | None:
        prepare_started = perf_counter()
        freeze_started = perf_counter()
        snapshot = self._freeze_market_snapshot(
            bar,
            snapshot_id=snapshot_id,
            generation_id=generation_id,
            source_row_idx=source_row_idx,
        )
        freeze_seconds = perf_counter() - freeze_started

        feature_started = perf_counter()
        feature_frame = self.feature_engine.build_feature_frame(
            snapshot.market_frame,
            include_policy_features=True,
        )
        feature_seconds = perf_counter() - feature_started
        if feature_frame.empty:
            return None

        policy_started = perf_counter()
        policy_frame = self._build_policy_frame_from_market_frame(
            feature_frame,
            market_frame=snapshot.market_frame,
        )
        policy_seconds = perf_counter() - policy_started
        metadata = dict(getattr(self.feature_engine, "last_build_metadata", {}) or {})
        builder_timings = {
            f"builder.{key}": float(value)
            for key, value in dict(getattr(self.feature_engine, "last_build_timings_seconds", {}) or {}).items()
        }
        timings = {
            "snapshot_freeze": freeze_seconds,
            "feature_generation": feature_seconds,
            "policy_frame": policy_seconds,
            "prepare_total": perf_counter() - prepare_started,
            **builder_timings,
        }
        return PreparedSignalSnapshot(
            group_name=self.group_name,
            snapshot=snapshot,
            feature_frame=feature_frame,
            policy_frame=policy_frame,
            feature_metadata=metadata,
            timings_seconds={key: round(float(value), 6) for key, value in timings.items()},
            feature_cache_hit=bool(getattr(self.feature_engine, "last_build_from_cache", False)),
        )

    def publish_prepared_snapshot(
        self,
        prepared: PreparedSignalSnapshot,
        *,
        emit_operator_artifacts: bool,
        emit_notifications: bool,
        record_paper_signal_events: bool = False,
        bindings: Sequence[SignalRuntimeModelBinding] | None = None,
    ) -> tuple[RuntimeSignalResult, ...]:
        return self._evaluate_prepared_snapshot(
            prepared,
            emit_operator_artifacts=emit_operator_artifacts,
            emit_notifications=emit_notifications,
            record_paper_signal_events=record_paper_signal_events,
            bindings=bindings,
        )

    def _evaluate_ingested_bar(
        self,
        bar: MarketBar,
        *,
        emit_operator_artifacts: bool,
        emit_notifications: bool,
        record_paper_signal_events: bool = False,
        bindings: Sequence[SignalRuntimeModelBinding] | None = None,
    ) -> tuple[RuntimeSignalResult, ...]:
        try:
            prepared = self.prepare_ingested_bar_snapshot(bar)
        except Exception as exc:
            self._record_health_event(
                component="signal_runtime.features",
                event_type="feature_build_failed",
                severity="error",
                message="Live signal feature build failed.",
                payload={
                    "asset": bar.asset,
                    "timeframe": bar.timeframe,
                    "timestamp_utc": bar.timestamp.isoformat(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return ()

        if prepared is None:
            return ()

        return self._evaluate_prepared_snapshot(
            prepared,
            emit_operator_artifacts=emit_operator_artifacts,
            emit_notifications=emit_notifications,
            record_paper_signal_events=record_paper_signal_events,
            bindings=bindings,
        )

    def _evaluate_prepared_snapshot(
        self,
        prepared: PreparedSignalSnapshot,
        *,
        emit_operator_artifacts: bool,
        emit_notifications: bool,
        record_paper_signal_events: bool = False,
        bindings: Sequence[SignalRuntimeModelBinding] | None = None,
    ) -> tuple[RuntimeSignalResult, ...]:
        snapshot_context = prepared.snapshot
        bar = snapshot_context.bar
        if snapshot_context.bar_timestamp != bar.timestamp:
            self._record_health_event(
                component="signal_runtime.snapshot",
                event_type="snapshot_timestamp_mismatch",
                severity="error",
                message="Prepared signal snapshot did not match its market bar timestamp.",
                payload={
                    "snapshot_id": snapshot_context.snapshot_id,
                    "generation_id": snapshot_context.generation_id,
                    "bar_timestamp_utc": bar.timestamp.isoformat(),
                    "snapshot_timestamp_utc": snapshot_context.bar_timestamp.isoformat(),
                },
            )
            return ()

        feature_frame = prepared.feature_frame
        policy_frame = prepared.policy_frame
        source_row_idx = snapshot_context.source_row_idx
        setup_events = []
        setup_history_errors = list(getattr(self, "_setup_revision_diagnostic_reasons", ()))
        observed_at = utc_now()
        strategies = tuple(name for name, enabled in (
            ("FRVP", self._frvp_dashboard_state_enabled), ("ICT", self._ict_dashboard_state_enabled)
        ) if enabled)
        if bar.asset == "ES" and strategies:
            self.setup_event_repository.record_followup_bar(
                bar, collection_version=self.collection_version, observed_at=observed_at,
                source_bar_version=bar.bar_version,
            )
            for strategy in strategies:
                try:
                    setup_events.extend(self.setup_event_repository.collect_from_frame(
                        policy_frame, bar=bar, collection_version=self.collection_version,
                        observed_at=observed_at, source_bar_version=bar.bar_version, strategies=(strategy,),
                    ))
                except Exception as exc:
                    setup_history_errors.append(f"{strategy.lower()}_setup_history_unavailable")
                    self._record_health_event(
                        component="signal_runtime.setup_history", event_type="setup_history_failed",
                        severity="error", message="Setup history collection failed; evaluation remains diagnostic.",
                        payload={"strategy": strategy, "timestamp": bar.timestamp.isoformat(), "error": str(exc)},
                    )
            if not setup_history_errors:
                mark_setup_evaluated(self, bar)
        dashboard_started = perf_counter()
        self._persist_dashboard_state(bar=bar, policy_frame=policy_frame)
        dashboard_seconds = perf_counter() - dashboard_started
        results: list[RuntimeSignalResult] = []

        for binding in bindings or self.bindings:
            model_started = perf_counter()
            loaded_model = binding.loaded_model
            contract_status = prepared.feature_metadata.get("model_input_contracts", {}).get(loaded_model.model_id)
            if contract_status is not None and self._input_contract_health.get(loaded_model.model_id) != contract_status:
                self._input_contract_health[loaded_model.model_id] = dict(contract_status)
                self._record_health_event(
                    component="signal_runtime.input_contract", event_type="model_input_contract",
                    severity="warning" if contract_status.get("diagnostic_only") else "info",
                    message="Model required-input qualification status changed.",
                    payload={"collection_version": self.collection_version, **contract_status},
                )
            warmup = self.feature_engine.warmup_status(
                model_id=loaded_model.model_id,
                feature_frame=feature_frame,
            )
            if not warmup.ready:
                continue

            ordered_features = feature_frame.loc[:, list(loaded_model.selected_feature_names)].copy()
            latest_values = ordered_features.iloc[-1]
            snapshot = FeatureSnapshot(
                asset=loaded_model.manifest.asset,
                timeframe=loaded_model.manifest.timeframe,
                direction=loaded_model.manifest.direction,
                timestamp=bar.timestamp,
                collection_version=self.collection_version,
                snapshot_id=snapshot_context.snapshot_id,
                generation_id=snapshot_context.generation_id,
                source_row_idx=source_row_idx,
                feature_values={
                    feature_name: _to_python_scalar(value)
                    for feature_name, value in latest_values.items()
                },
                valid_feature_count=warmup.valid_feature_count,
            )

            try:
                runner = self._runner_by_model_id[loaded_model.model_id]
                prediction = runner.predict_latest(
                    ordered_features,
                    timestamp=bar.timestamp,
                    source_row_idx=source_row_idx,
                )
                prediction_recorded_at = utc_now()
                prediction = prediction.model_copy(update={
                    "collection_version": self.collection_version,
                    "prediction_recorded_at_utc": prediction_recorded_at,
                })
                bar_contract = assess_shadow_bar_eligibility(
                    bar, recorded_at=prediction_recorded_at,
                    previous_bar_timestamp=snapshot_context.previous_bar_timestamp,
                )
                input_contract = dict(prepared.feature_metadata.get("model_input_contracts", {}).get(
                    loaded_model.model_id, {"diagnostic_only": True, "reasons": ["input_contract_unverified"]}
                ))
                input_diagnostic = bool(input_contract.get("diagnostic_only", True))
                diagnostic_reasons = [*bar_contract.reasons, *setup_history_errors]
                if input_diagnostic:
                    diagnostic_reasons.extend(input_contract.get("reasons", ["input_contract_unverified"]))
                ict_priority = ict_research_priority(loaded_model.model_id) if bar.asset == "ES" else None
                if ict_priority is not None and ict_priority.candidate_block_reason:
                    diagnostic_reasons.append(ict_priority.candidate_block_reason)
                # B1/B2/B3 and A7 remain prerequisites. Policy acceptance alone is never a trade.
                observation_metadata = {
                    "bar_eligibility": bar_contract.to_payload(),
                    "model_input_contract": input_contract,
                    "diagnostic_only": bool(diagnostic_reasons),
                    "diagnostic_reasons": list(dict.fromkeys(diagnostic_reasons)),
                    "qualified_shadow_entry": False,
                    "qualification_status": "prerequisites_pending",
                    "executable_entry_at": None,
                    "executable_entry_price": None,
                    "source_bar": bar.model_dump(mode="json"),
                }
                snapshot = snapshot.model_copy(update={"observation_metadata": observation_metadata})
                effective_shadow_mode = bool(binding.shadow_mode or self.force_shadow_mode)
                if bar.asset == "ES" and diagnostic_reasons:
                    effective_shadow_mode = True
                setup_gate = resolve_setup_family_gate(
                    loaded_model.model_id,
                    policy_frame=policy_frame,
                )
                if setup_gate.error_type is not None:
                    self._record_health_event(
                        component="signal_runtime.setup_family",
                        event_type="setup_family_gate_failed",
                        severity="error",
                        message="Setup-family model gate failed; model decision was held closed.",
                        payload={
                            "model_id": loaded_model.model_id,
                            "direction": loaded_model.manifest.direction,
                            "shadow_mode": bool(binding.shadow_mode),
                            "timestamp_utc": bar.timestamp.isoformat(),
                            "error_type": setup_gate.error_type,
                            "error_message": setup_gate.error_message,
                        },
                    )
                decision_path = self.decision_engine.evaluate_prediction(
                    prediction,
                    live_policy=loaded_model.manifest.live_policy,
                    policy_context=policy_frame,
                    timezone_contract=loaded_model.manifest.timezone_contract,
                    shadow_mode=effective_shadow_mode,
                    feature_snapshot=snapshot,
                    runtime_manifest=loaded_model.manifest,
                    paper_signal_event_eligible=(
                        record_paper_signal_events
                        and not effective_shadow_mode
                        and getattr(self, "frvp_paper_signal_ledger", None) is not None
                        and supports_frvp_paper_signal_manifest(loaded_model.manifest)
                    ),
                    force_hold_reasons=setup_gate.forced_hold_reasons,
                    shadow_setup_match=(match_shadow_setups(prediction, snapshot, setup_events)
                                        if effective_shadow_mode and bar.asset == "ES" else None),
                )
                model_seconds = perf_counter() - model_started
            except Exception as exc:
                self._record_health_event(
                    component="signal_runtime.models",
                    event_type="signal_decision_failed",
                    severity="error",
                    message="Live signal evaluation failed.",
                    payload={
                        "model_id": loaded_model.model_id,
                        "direction": loaded_model.manifest.direction,
                        "shadow_mode": bool(binding.shadow_mode),
                        "timestamp_utc": bar.timestamp.isoformat(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                continue

            signal_decision_id = (
                decision_path.audit_record.signal_decision_id
                if decision_path.audit_record is not None
                else None
            )
            if decision_path.audit_record is not None:
                for event in setup_events:
                    if not loaded_model.model_id.lower().startswith(event.strategy.lower() + "_"):
                        continue
                    self.setup_event_repository.link_prediction(
                        event.event_id, decision_path.audit_record.prediction_id,
                        model_id=loaded_model.model_id, decision=decision_path.signal.decision,
                        rejection_reasons=decision_path.signal.reasons, observed_at=prediction_recorded_at,
                        payload={
                            "probability": decision_path.signal.probability,
                            "threshold": decision_path.signal.threshold,
                            "selected_setup": event.selected,
                            "model_direction_matches": (event.setup_side == (1 if loaded_model.manifest.direction == "long" else -1)),
                            "setup_gate_matched": setup_gate.matched,
                            "setup_gate_reasons": list(setup_gate.forced_hold_reasons),
                            "shadow_setup_matched": event.event_id in (
                                (decision_path.signal.shadow_evaluation or {}).get("setup_match", {}).get("matched_event_ids", [])
                            ),
                            "shadow_evaluation": decision_path.signal.shadow_evaluation,
                            "collection_version": self.collection_version,
                            "qualified_shadow_entry": False,
                        },
                    )
            if (
                record_paper_signal_events
                and getattr(self, "ict_paper_signal_ledger", None) is not None
                and decision_path.audit_record is not None
                and decision_path.signal.decision == "emit"
                and supports_ict_paper_signal_manifest(loaded_model.manifest)
            ):
                try:
                    session_regime = _to_python_scalar(policy_frame.iloc[-1].get("session_regime"))
                    self.ict_paper_signal_ledger.open_event(
                        manifest=loaded_model.manifest,
                        audit_record=decision_path.audit_record,
                        signal=decision_path.signal,
                        bar=bar,
                        session_regime=(str(session_regime) if session_regime is not None else None),
                    )
                except Exception as exc:
                    self._record_health_event(
                        component="signal_runtime.ict_paper_signal_ledger",
                        event_type="paper_signal_open_failed",
                        severity="error",
                        message="Persisting an emitted ICT paper-signal markout failed.",
                        payload={
                            "model_id": loaded_model.model_id,
                            "signal_decision_id": signal_decision_id,
                            "timestamp_utc": bar.timestamp.isoformat(),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
            if (
                record_paper_signal_events
                and getattr(self, "frvp_paper_signal_ledger", None) is not None
                and decision_path.audit_record is not None
                and decision_path.signal.decision == "emit"
                and supports_frvp_paper_signal_manifest(loaded_model.manifest)
            ):
                try:
                    session_regime = _to_python_scalar(policy_frame.iloc[-1].get("session_regime"))
                    self.frvp_paper_signal_ledger.open_event(
                        manifest=loaded_model.manifest,
                        audit_record=decision_path.audit_record,
                        signal=decision_path.signal,
                        bar=bar,
                        session_regime=(str(session_regime) if session_regime is not None else None),
                    )
                except Exception as exc:
                    self._record_health_event(
                        component="signal_runtime.frvp_paper_signal_ledger",
                        event_type="paper_signal_open_failed",
                        severity="error",
                        message="Persisting an emitted FRVP paper-signal markout failed.",
                        payload={
                            "model_id": loaded_model.model_id,
                            "signal_decision_id": signal_decision_id,
                            "timestamp_utc": bar.timestamp.isoformat(),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    raise RuntimeError(
                        "Persisting the emitted FRVP paper-signal markout failed; "
                        "signal processing is stopped until deterministic reconciliation succeeds."
                    ) from exc
            media_artifact = None
            notification = None
            sms_notification = None
            if (
                emit_operator_artifacts
                and _should_capture_operator_artifacts(decision_path.signal)
                and signal_decision_id is not None
            ):
                media_artifact = self._capture_chart(signal_decision_id)
            if (
                emit_notifications
                and _should_send_notifications(decision_path.signal)
                and signal_decision_id is not None
            ):
                notification = self._send_email_alert(
                    signal_decision_id,
                    screenshot_path=media_artifact.file_path if media_artifact is not None else None,
                )
                sms_notification = self._send_sms_alert(signal_decision_id)

            results.append(
                RuntimeSignalResult(
                    model_id=loaded_model.model_id,
                    direction=loaded_model.manifest.direction,
                    decision=decision_path.signal.decision,
                    shadow_mode=effective_shadow_mode,
                    timestamp=bar.timestamp,
                    signal_decision_id=signal_decision_id,
                    notification_status=notification.status if notification is not None else None,
                    sms_notification_status=(
                        sms_notification.status if sms_notification is not None else None
                    ),
                    media_artifact_id=media_artifact.artifact_id if media_artifact is not None else None,
                    snapshot_id=snapshot_context.snapshot_id,
                    generation_id=snapshot_context.generation_id,
                    timings_seconds={
                        **prepared.timings_seconds,
                        "dashboard_publish": round(float(dashboard_seconds), 6),
                        "model_decision": round(float(model_seconds), 6),
                    },
                )
            )

        if results:
            LOGGER.debug(
                "Published signal snapshot %s generation=%s group=%s results=%s timings=%s",
                snapshot_context.snapshot_id,
                snapshot_context.generation_id,
                self.group_name,
                len(results),
                prepared.timings_seconds,
            )
        return tuple(results)

    def _reconcile_setup_history(self) -> None:
        if self.asset != "ES" or not (self._frvp_dashboard_state_enabled or self._ict_dashboard_state_enabled):
            return
        try:
            reconcile_setup_revisions(self)
            state = initialize_setup_revision_cursor(self)
            self._setup_revision_diagnostic_reasons = (
                ("setup_revision_pending",) if state.get("pending") else ()
            )
        except Exception as exc:
            self._setup_revision_diagnostic_reasons = ("setup_revision_failed",)
            self._record_health_event(
                component="signal_runtime.setup_history", event_type="setup_revision_failed",
                severity="error", message="Setup revision reconciliation failed; its cursor remains pending for retry.",
                payload={"collection_version": self.collection_version, "error": str(exc)},
            )

    def _reconcile_frvp_paper_signal_events(self) -> None:
        ledger = getattr(self, "frvp_paper_signal_ledger", None)
        if ledger is None:
            return
        try:
            ledger.reconcile_missing_events()
        except Exception as exc:
            self._record_health_event(
                component="signal_runtime.frvp_paper_signal_ledger",
                event_type="paper_signal_reconciliation_failed",
                severity="error",
                message="Reconciling eligible emitted FRVP paper-signal markouts failed.",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise RuntimeError(
                "FRVP paper-signal reconciliation failed; signal processing remains stopped."
            ) from exc

    def _existing_prediction_model_ids(self, timestamp: datetime) -> set[str]:
        rows = self.audit_repository.store.connection.execute(
            """
            SELECT DISTINCT model_id
            FROM model_predictions
            WHERE timestamp_utc = ? AND collection_version = ?
            """,
            (timestamp.isoformat(), self.collection_version),
        ).fetchall()
        return {str(row["model_id"]) for row in rows}

    def _build_policy_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        return self._build_policy_frame_from_market_frame(
            feature_frame,
            market_frame=self.feature_engine.state.to_frame(),
        )

    def _build_policy_frame_from_market_frame(
        self,
        feature_frame: pd.DataFrame,
        *,
        market_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        portable_attrs = _portable_feature_attrs(feature_frame)
        aligned_market = market_frame.iloc[-len(feature_frame) :].reset_index(drop=True)
        aligned_features = feature_frame.reset_index(drop=True)
        overlapping_columns = aligned_market.columns.intersection(aligned_features.columns)
        if not overlapping_columns.empty:
            aligned_market = aligned_market.drop(columns=list(overlapping_columns))

        policy_frame = pd.concat([aligned_market, aligned_features], axis=1)
        if "timestamp" in policy_frame.columns and "datetime" not in policy_frame.columns:
            insert_at = policy_frame.columns.get_loc("timestamp") + 1
            policy_frame.insert(
                insert_at,
                "datetime",
                pd.to_datetime(policy_frame["timestamp"], errors="coerce", utc=True),
            )
        ordered_market_columns = list(market_frame.columns)
        if "datetime" in policy_frame.columns:
            if "timestamp" in ordered_market_columns:
                insert_at = ordered_market_columns.index("timestamp") + 1
                ordered_market_columns.insert(insert_at, "datetime")
            else:
                ordered_market_columns.append("datetime")
        ordered_columns = [
            *ordered_market_columns,
            *[
                column
                for column in aligned_features.columns
                if column not in market_frame.columns and column != "datetime"
            ],
        ]
        prepared = policy_frame.loc[:, ordered_columns]
        _restore_feature_attrs(prepared, portable_attrs)
        return prepared

    def _freeze_market_snapshot(
        self,
        bar: MarketBar,
        *,
        snapshot_id: str | None,
        generation_id: int | None,
        source_row_idx: int | None,
    ) -> MarketDataSnapshot:
        market_frame = self.feature_engine.state.to_frame()
        if market_frame.empty:
            raise ValueError("Cannot freeze a market snapshot without warmed market state.")
        latest_timestamp = IncrementalFeatureEngine._extract_latest_timestamp(market_frame)
        if latest_timestamp != bar.timestamp:
            raise ValueError(
                "Market snapshot latest timestamp does not match the evaluated bar. "
                f"latest={latest_timestamp!r}, bar={bar.timestamp!r}"
            )
        resolved_source_row_idx = (
            int(source_row_idx)
            if source_row_idx is not None
            else self._resolve_source_row_idx(bar)
        )
        resolved_generation_id = (
            int(generation_id)
            if generation_id is not None
            else self._next_generation_id()
        )
        resolved_snapshot_id = snapshot_id or build_market_snapshot_id(
            asset=bar.asset,
            timeframe=bar.timeframe,
            timestamp=bar.timestamp,
            source_row_idx=resolved_source_row_idx,
        )
        return MarketDataSnapshot(
            snapshot_id=resolved_snapshot_id,
            generation_id=resolved_generation_id,
            asset=bar.asset,
            timeframe=bar.timeframe,
            bar_timestamp=bar.timestamp,
            source_row_idx=resolved_source_row_idx,
            bar=bar,
            market_frame=market_frame.copy(deep=True),
            previous_bar_timestamp=(
                pd.Timestamp(market_frame.iloc[-2]["timestamp"]).to_pydatetime()
                if len(market_frame) > 1 else getattr(self, "_previous_bar_timestamp", None)
            ),
        )

    def _next_generation_id(self) -> int:
        self._generation_id += 1
        return int(self._generation_id)

    def _resolve_source_row_idx(self, bar: MarketBar) -> int:
        row = self.audit_repository.store.connection.execute(
            """
            SELECT COUNT(*) AS bar_count
            FROM canonical_bars
            WHERE asset = ? AND timeframe = ? AND timestamp_utc <= ?
            """,
            (bar.asset, bar.timeframe, bar.timestamp.isoformat()),
        ).fetchone()
        count = int(row["bar_count"]) if row is not None else 1
        return max(0, count - 1)

    def _fetch_recent_signal_bars(self, *, limit: int) -> list[MarketBar]:
        rows = self.audit_repository.store.connection.execute(
            """
            SELECT
                asset,
                timeframe,
                timestamp_utc,
                open,
                high,
                low,
                close,
                volume,
                bid,
                ask,
                spread,
                source,
                symbol,
                contract_symbol,
                instrument_id,
                feature_context_json
            FROM canonical_bars
            WHERE asset = ? AND timeframe = ?
            ORDER BY timestamp_utc DESC
            LIMIT ?
            """,
            (self.asset, self.timeframe, int(limit)),
        ).fetchall()
        return [
            _bar_from_row(row)
            for row in reversed(rows)
        ]

    def _fetch_signal_bars_after(
        self,
        timestamp: datetime,
        *,
        max_timestamp: datetime | None = None,
    ) -> list[MarketBar]:
        query = """
            SELECT
                asset,
                timeframe,
                timestamp_utc,
                open,
                high,
                low,
                close,
                volume,
                bid,
                ask,
                spread,
                source,
                symbol,
                contract_symbol,
                instrument_id,
                feature_context_json
            FROM canonical_bars
            WHERE asset = ? AND timeframe = ? AND timestamp_utc > ?
        """
        params: list[object] = [self.asset, self.timeframe, timestamp.isoformat()]
        if max_timestamp is not None:
            query += " AND timestamp_utc <= ?"
            params.append(max_timestamp.isoformat())
        query += " ORDER BY timestamp_utc ASC"

        rows = self.audit_repository.store.connection.execute(query, params).fetchall()
        return [_bar_from_row(row) for row in rows]

    def _capture_chart(self, signal_decision_id: int) -> CapturedChartArtifact | None:
        if self.chart_capture_service is None:
            return None
        try:
            return self.chart_capture_service.capture_signal_chart(
                signal_decision_id=signal_decision_id,
            )
        except Exception as exc:
            self._record_health_event(
                component="signal_runtime.media",
                event_type="chart_capture_failed",
                severity="error",
                message="Capturing a signal chart failed.",
                payload={
                    "signal_decision_id": signal_decision_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return None

    def _send_sms_alert(
        self,
        signal_decision_id: int,
    ) -> SmsDispatchResult | None:
        if self.sms_sender is None:
            return None
        try:
            return self.sms_sender.send_signal_alert(
                signal_decision_id=signal_decision_id,
            )
        except Exception as exc:
            self._record_health_event(
                component="signal_runtime.notifications",
                event_type="sms_alert_failed",
                severity="error",
                message="Sending a live signal SMS alert failed.",
                payload={
                    "signal_decision_id": signal_decision_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return None

    def _send_email_alert(
        self,
        signal_decision_id: int,
        *,
        screenshot_path: Path | None,
    ) -> EmailDispatchResult | None:
        if self.emailer is None:
            return None
        try:
            return self.emailer.send_signal_alert(
                signal_decision_id=signal_decision_id,
                dashboard_url=self.dashboard_url,
                screenshot_path=screenshot_path,
            )
        except Exception as exc:
            self._record_health_event(
                component="signal_runtime.notifications",
                event_type="email_alert_failed",
                severity="error",
                message="Sending a live signal email alert failed.",
                payload={
                    "signal_decision_id": signal_decision_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return None

    def _record_health_event(
        self,
        *,
        component: str,
        event_type: str,
        severity: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        self.audit_repository.record_health_event(
            component=component,
            event_type=event_type,
            severity=severity,
            message=message,
            payload=payload,
        )

    def _persist_dashboard_state(self, *, bar: MarketBar, policy_frame: pd.DataFrame) -> None:
        if not self._frvp_dashboard_state_enabled and not self._ict_dashboard_state_enabled:
            return
        if self._frvp_dashboard_state_enabled:
            self._persist_dashboard_state_for_family(
                persist_fn=persist_frvp_dashboard_state,
                family_name="FRVP",
                bar=bar,
                policy_frame=policy_frame,
            )
        if self._ict_dashboard_state_enabled:
            self._persist_dashboard_state_for_family(
                persist_fn=persist_ict_dashboard_state,
                family_name="ICT",
                bar=bar,
                policy_frame=policy_frame,
            )

    def _persist_dashboard_state_for_family(
        self,
        *,
        persist_fn,
        family_name: str,
        bar: MarketBar,
        policy_frame: pd.DataFrame,
    ) -> None:
        try:
            persist_fn(
                self.audit_repository.store,
                group_name=self.group_name,
                asset=self.asset,
                timeframe=self.timeframe,
                data_supplier=self.data_supplier,
                policy_frame=policy_frame,
                bar=bar,
            )
        except Exception as exc:
            self._record_health_event(
                component="signal_runtime.dashboard_state",
                event_type="dashboard_state_persist_failed",
                severity="error",
                message=f"Persisting {family_name} dashboard state failed.",
                payload={
                    "group_name": self.group_name,
                    "family": family_name,
                    "asset": bar.asset,
                    "timeframe": bar.timeframe,
                    "timestamp_utc": bar.timestamp.isoformat(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )


class MultiGroupLiveSignalProcessor:
    """Coordinate multiple strategy processors against one canonical bar snapshot."""

    def __init__(
        self,
        processors: tuple[LiveSignalProcessor, ...],
        *,
        max_workers: int | None = None,
    ) -> None:
        if not processors:
            raise ValueError("At least one signal processor is required.")
        self.processors = tuple(processors)
        self.bindings = tuple(
            binding
            for processor in self.processors
            for binding in processor.bindings
        )
        runtime_history_bars = max(
            int(
                getattr(
                    getattr(getattr(processor, "feature_engine", None), "plan", None),
                    "runtime_history_bars",
                    0,
                )
                or 0
            )
            for processor in self.processors
        )
        self.feature_engine = SimpleNamespace(
            plan=SimpleNamespace(runtime_history_bars=runtime_history_bars)
        )
        self.max_workers = min(
            len(self.processors),
            max(1, int(max_workers or len(self.processors))),
        )
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="ote-live-feature",
            )
            if self.max_workers > 1
            else None
        )
        self._generation_id = 0
        self._closed = False

    def warm_from_store(self) -> int:
        if not self._can_share_market_timeline():
            return sum(processor.warm_from_store() for processor in self.processors)
        for processor in self.processors:
            if (
                processor.last_processed_timestamp is None
                and processor.feature_engine.state.latest_timestamp is not None
            ):
                processor.last_processed_timestamp = processor.feature_engine.state.latest_timestamp
        if all(processor.feature_engine.state.latest_timestamp is not None for processor in self.processors):
            return 0

        max_limit = max(
            int(processor.feature_engine.plan.runtime_history_bars)
            for processor in self.processors
        )
        bars = self.processors[0]._fetch_recent_signal_bars(limit=max_limit)
        if not bars:
            return 0

        warmed = 0
        for processor in self.processors:
            if processor.feature_engine.state.latest_timestamp is not None:
                continue
            limit = int(processor.feature_engine.plan.runtime_history_bars)
            selected = bars[-limit:] if limit > 0 else []
            if not selected:
                continue
            processor.feature_engine.extend(selected)
            processor.last_processed_timestamp = selected[-1].timestamp
            warmed += len(selected)
        return warmed

    def seed_latest_predictions_from_store(self):
        self._reconcile_processors()
        if any(processor.feature_engine.state.latest_timestamp is None for processor in self.processors):
            self.warm_from_store()
        if not self._can_share_market_timeline() or not self._has_common_processed_timestamp():
            return tuple(
                result
                for processor in self.processors
                for result in processor.seed_latest_predictions_from_store()
            )

        latest_timestamp = self.processors[0].feature_engine.state.latest_timestamp
        if latest_timestamp is None:
            return ()
        latest_bars = self.processors[0]._fetch_recent_signal_bars(limit=1)
        if not latest_bars or latest_bars[-1].timestamp != latest_timestamp:
            return ()

        missing_bindings_by_processor: dict[LiveSignalProcessor, tuple[SignalRuntimeModelBinding, ...]] = {}
        for processor in self.processors:
            existing_model_ids = processor._existing_prediction_model_ids(latest_timestamp)
            missing_bindings = tuple(
                binding
                for binding in processor.bindings
                if binding.loaded_model.model_id not in existing_model_ids
            )
            if missing_bindings:
                missing_bindings_by_processor[processor] = missing_bindings
        if not missing_bindings_by_processor:
            return ()

        bar = latest_bars[-1]
        source_row_idx = self.processors[0]._resolve_source_row_idx(bar)
        generation_id = self._next_generation_id()
        snapshot_id = build_market_snapshot_id(
            asset=bar.asset,
            timeframe=bar.timeframe,
            timestamp=bar.timestamp,
            source_row_idx=source_row_idx,
        )
        prepared = self._prepare_processors(
            tuple(missing_bindings_by_processor),
            bar=bar,
            snapshot_id=snapshot_id,
            generation_id=generation_id,
            source_row_idx=source_row_idx,
        )
        if len(prepared) != len(missing_bindings_by_processor):
            self._record_incomplete_snapshot(
                snapshot_id=snapshot_id,
                generation_id=generation_id,
                bar=bar,
                expected=len(missing_bindings_by_processor),
                prepared=len(prepared),
            )
            return ()
        if not self._prepared_snapshots_are_consistent(
            prepared.values(),
            snapshot_id=snapshot_id,
            generation_id=generation_id,
            bar=bar,
        ):
            return ()

        results: list[RuntimeSignalResult] = []
        for processor in self.processors:
            if processor not in missing_bindings_by_processor:
                continue
            results.extend(
                processor.publish_prepared_snapshot(
                    prepared[processor],
                    emit_operator_artifacts=False,
                    emit_notifications=False,
                    record_paper_signal_events=False,
                    bindings=missing_bindings_by_processor[processor],
                )
            )
        return tuple(results)

    def process_new_bars_from_store(
        self,
        *,
        emit_operator_artifacts: bool,
        max_timestamp,
    ):
        self._reconcile_processors()
        if any(processor.last_processed_timestamp is None for processor in self.processors):
            self.warm_from_store()
            return ()
        if not self._can_share_market_timeline() or not self._has_common_processed_timestamp():
            return tuple(
                result
                for processor in self.processors
                for result in processor.process_new_bars_from_store(
                    emit_operator_artifacts=emit_operator_artifacts,
                    max_timestamp=max_timestamp,
                )
            )

        last_processed_timestamp = self.processors[0].last_processed_timestamp
        if last_processed_timestamp is None:
            return ()
        new_bars = self.processors[0]._fetch_signal_bars_after(
            last_processed_timestamp,
            max_timestamp=max_timestamp,
        )
        if not new_bars:
            return ()
        return self.process_bars(
            new_bars,
            emit_operator_artifacts=emit_operator_artifacts,
        )

    def process_bars(
        self,
        bars: Iterable[MarketBar],
        *,
        emit_operator_artifacts: bool,
    ) -> tuple[RuntimeSignalResult, ...]:
        self._reconcile_processors()
        ordered_bars = tuple(sorted(bars, key=lambda item: item.timestamp))
        if not self._can_share_market_timeline():
            return tuple(
                result
                for processor in self.processors
                for result in processor.process_bars(
                    ordered_bars,
                    emit_operator_artifacts=emit_operator_artifacts,
                )
            )

        results: list[RuntimeSignalResult] = []
        for bar in ordered_bars:
            if not self._bar_matches_shared_timeline(bar):
                continue
            cycle_started = perf_counter()
            previous_processed_timestamps = {
                processor: processor.last_processed_timestamp
                for processor in self.processors
            }
            source_row_idx = self.processors[0]._resolve_source_row_idx(bar)
            generation_id = self._next_generation_id()
            snapshot_id = build_market_snapshot_id(
                asset=bar.asset,
                timeframe=bar.timeframe,
                timestamp=bar.timestamp,
                source_row_idx=source_row_idx,
            )

            for processor in self.processors:
                processor.ingest_bar_for_evaluation(bar)

            prepared = self._prepare_processors(
                self.processors,
                bar=bar,
                snapshot_id=snapshot_id,
                generation_id=generation_id,
                source_row_idx=source_row_idx,
            )
            if len(prepared) != len(self.processors):
                self._record_incomplete_snapshot(
                    snapshot_id=snapshot_id,
                    generation_id=generation_id,
                    bar=bar,
                    expected=len(self.processors),
                    prepared=len(prepared),
                )
                self._restore_last_processed_timestamps(previous_processed_timestamps)
                break
            if not self._prepared_snapshots_are_consistent(
                prepared.values(),
                snapshot_id=snapshot_id,
                generation_id=generation_id,
                bar=bar,
            ):
                self._restore_last_processed_timestamps(previous_processed_timestamps)
                break

            for processor in self.processors:
                results.extend(
                    processor.publish_prepared_snapshot(
                        prepared[processor],
                        emit_operator_artifacts=emit_operator_artifacts,
                        emit_notifications=True,
                        record_paper_signal_events=True,
                    )
                )
            LOGGER.debug(
                "Published multi-group signal snapshot %s generation=%s processors=%s total_seconds=%.6f",
                snapshot_id,
                generation_id,
                len(self.processors),
                perf_counter() - cycle_started,
            )
        return tuple(results)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        for processor in self.processors:
            close = getattr(processor, "close", None)
            if callable(close):
                close()

    def _prepare_processors(
        self,
        processors: tuple[LiveSignalProcessor, ...],
        *,
        bar: MarketBar,
        snapshot_id: str,
        generation_id: int,
        source_row_idx: int,
    ) -> dict[LiveSignalProcessor, PreparedSignalSnapshot]:
        if self._executor is None or len(processors) <= 1:
            prepared: dict[LiveSignalProcessor, PreparedSignalSnapshot] = {}
            for processor in processors:
                item = self._prepare_one_processor(
                    processor,
                    bar=bar,
                    snapshot_id=snapshot_id,
                    generation_id=generation_id,
                    source_row_idx=source_row_idx,
                )
                if item is not None:
                    prepared[processor] = item
            return prepared

        futures = {
            self._executor.submit(
                self._prepare_one_processor,
                processor,
                bar=bar,
                snapshot_id=snapshot_id,
                generation_id=generation_id,
                source_row_idx=source_row_idx,
            ): processor
            for processor in processors
        }
        prepared: dict[LiveSignalProcessor, PreparedSignalSnapshot] = {}
        for future in as_completed(futures):
            processor = futures[future]
            item = future.result()
            if item is not None:
                prepared[processor] = item
        return prepared

    @staticmethod
    def _restore_last_processed_timestamps(
        timestamps: dict[LiveSignalProcessor, datetime | None],
    ) -> None:
        for processor, timestamp in timestamps.items():
            processor.last_processed_timestamp = timestamp

    def _prepare_one_processor(
        self,
        processor: LiveSignalProcessor,
        *,
        bar: MarketBar,
        snapshot_id: str,
        generation_id: int,
        source_row_idx: int,
    ) -> PreparedSignalSnapshot | None:
        try:
            return processor.prepare_ingested_bar_snapshot(
                bar,
                snapshot_id=snapshot_id,
                generation_id=generation_id,
                source_row_idx=source_row_idx,
            )
        except Exception as exc:
            processor._record_health_event(
                component="signal_runtime.features",
                event_type="feature_branch_failed",
                severity="error",
                message="Parallel live signal feature branch failed.",
                payload={
                    "group_name": processor.group_name,
                    "snapshot_id": snapshot_id,
                    "generation_id": generation_id,
                    "asset": bar.asset,
                    "timeframe": bar.timeframe,
                    "timestamp_utc": bar.timestamp.isoformat(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return None

    def _prepared_snapshots_are_consistent(
        self,
        prepared: Iterable[PreparedSignalSnapshot],
        *,
        snapshot_id: str,
        generation_id: int,
        bar: MarketBar,
    ) -> bool:
        for item in prepared:
            snapshot = item.snapshot
            if (
                snapshot.snapshot_id == snapshot_id
                and snapshot.generation_id == generation_id
                and snapshot.bar_timestamp == bar.timestamp
                and snapshot.asset == bar.asset
                and snapshot.timeframe == bar.timeframe
            ):
                continue
            self._record_snapshot_consistency_failure(
                expected_snapshot_id=snapshot_id,
                expected_generation_id=generation_id,
                expected_bar=bar,
                actual=snapshot,
            )
            return False
        return True

    def _record_incomplete_snapshot(
        self,
        *,
        snapshot_id: str,
        generation_id: int,
        bar: MarketBar,
        expected: int,
        prepared: int,
    ) -> None:
        self.processors[0]._record_health_event(
            component="signal_runtime.snapshot",
            event_type="snapshot_fan_in_incomplete",
            severity="error",
            message="A live signal snapshot was not published because not every required branch completed.",
            payload={
                "snapshot_id": snapshot_id,
                "generation_id": generation_id,
                "asset": bar.asset,
                "timeframe": bar.timeframe,
                "timestamp_utc": bar.timestamp.isoformat(),
                "expected_branches": int(expected),
                "prepared_branches": int(prepared),
            },
        )

    def _record_snapshot_consistency_failure(
        self,
        *,
        expected_snapshot_id: str,
        expected_generation_id: int,
        expected_bar: MarketBar,
        actual: MarketDataSnapshot,
    ) -> None:
        self.processors[0]._record_health_event(
            component="signal_runtime.snapshot",
            event_type="snapshot_consistency_failed",
            severity="error",
            message="A live signal snapshot was not published because branch snapshot identity did not match.",
            payload={
                "expected_snapshot_id": expected_snapshot_id,
                "actual_snapshot_id": actual.snapshot_id,
                "expected_generation_id": expected_generation_id,
                "actual_generation_id": actual.generation_id,
                "expected_timestamp_utc": expected_bar.timestamp.isoformat(),
                "actual_timestamp_utc": actual.bar_timestamp.isoformat(),
                "expected_asset": expected_bar.asset,
                "actual_asset": actual.asset,
                "expected_timeframe": expected_bar.timeframe,
                "actual_timeframe": actual.timeframe,
            },
        )

    def _reconcile_processors(self) -> None:
        for processor in self.processors:
            processor._reconcile_frvp_paper_signal_events()
            reconcile = getattr(processor, "_reconcile_setup_history", None)
            if callable(reconcile):
                reconcile()

    def _can_share_market_timeline(self) -> bool:
        first = self.processors[0]
        return all(
            processor.asset == first.asset
            and processor.timeframe == first.timeframe
            and processor.audit_repository is first.audit_repository
            for processor in self.processors
        )

    def _has_common_processed_timestamp(self) -> bool:
        timestamps = {processor.last_processed_timestamp for processor in self.processors}
        return len(timestamps) == 1

    def _bar_matches_shared_timeline(self, bar: MarketBar) -> bool:
        first = self.processors[0]
        return bar.asset == first.asset and bar.timeframe == first.timeframe

    def _next_generation_id(self) -> int:
        self._generation_id += 1
        return int(self._generation_id)


def build_market_snapshot_id(
    *,
    asset: str,
    timeframe: str,
    timestamp: datetime,
    source_row_idx: int,
) -> str:
    resolved_timestamp = pd.Timestamp(timestamp)
    if resolved_timestamp.tzinfo is None:
        resolved_timestamp = resolved_timestamp.tz_localize("UTC")
    else:
        resolved_timestamp = resolved_timestamp.tz_convert("UTC")
    return f"{asset}:{timeframe}:{resolved_timestamp.isoformat()}:{int(source_row_idx)}"


def _bar_from_row(row) -> MarketBar:
    return MarketBar(
        asset=str(row["asset"]),
        timeframe=str(row["timeframe"]),
        timestamp=datetime.fromisoformat(str(row["timestamp_utc"])),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        bid=float(row["bid"]) if row["bid"] is not None else None,
        ask=float(row["ask"]) if row["ask"] is not None else None,
        spread=float(row["spread"]) if row["spread"] is not None else None,
        source=row["source"],
        symbol=row["symbol"],
        contract_symbol=row["contract_symbol"],
        instrument_id=int(row["instrument_id"]) if row["instrument_id"] is not None else None,
        feature_context=_json_loads(row["feature_context_json"]),
    )


def _to_python_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _json_loads(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload)
    return loaded if isinstance(loaded, dict) else {}


def _should_send_notifications(signal: SignalDecision) -> bool:
    # Operator notifications represent actionable signals, not model scores.
    # Shadow/candidate decisions remain available in the audit trail and may
    # capture artifacts, but must not alert merely for clearing a threshold.
    return str(signal.decision) == "emit"


def _should_capture_operator_artifacts(signal: SignalDecision) -> bool:
    # Automatic chart-image persistence is temporarily disabled. Keep the
    # capture plumbing in place so it can be re-enabled without changing the
    # signal, notification, or audit paths.
    return False


def _contains_frvp_models(manifests: Sequence[object]) -> bool:
    for manifest in manifests:
        model_id = str(getattr(manifest, "model_id", "") or "").lower()
        if model_id.startswith("frvp_"):
            return True
    return False


def _contains_ict_models(manifests: Sequence[object]) -> bool:
    for manifest in manifests:
        model_id = str(getattr(manifest, "model_id", "") or "").lower()
        if model_id.startswith("ict_"):
            return True
    return False


def _resolve_dashboard_extra_feature_names(
    *,
    enable_frvp_dashboard_state: bool,
    enable_ict_dashboard_state: bool,
) -> tuple[str, ...]:
    extra_feature_names: list[str] = []
    if enable_frvp_dashboard_state:
        extra_feature_names.extend(FRVP_DASHBOARD_EXTRA_FEATURE_NAMES)
    if enable_ict_dashboard_state:
        extra_feature_names.extend(ICT_DASHBOARD_EXTRA_FEATURE_NAMES)
    return tuple(dict.fromkeys(extra_feature_names))


def _portable_feature_attrs(frame: pd.DataFrame) -> dict[str, object]:
    return {
        name: frame.attrs[name]
        for name in ("fvg_zones",)
        if name in frame.attrs
    }


def _restore_feature_attrs(frame: pd.DataFrame, attrs: dict[str, object]) -> None:
    for name, value in attrs.items():
        frame.attrs[name] = value


def _record_runtime_load_health_event(
    audit_repository: object,
    *,
    component: str,
    event_type: str,
    severity: str,
    message: str,
    payload: dict[str, Any],
) -> None:
    record_health_event = getattr(audit_repository, "record_health_event", None)
    if not callable(record_health_event):
        return
    record_health_event(
        component=component,
        event_type=event_type,
        severity=severity,
        message=message,
        payload=payload,
    )

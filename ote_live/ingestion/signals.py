from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ote_live.alerts.emailer import EmailDispatchResult, LiveSignalEmailer
from ote_live.alerts.sms import LiveSignalSmsSender, SmsDispatchResult
from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.view_state import persist_frvp_dashboard_state
from ote_live.features.incremental_engine import (
    FRVP_DASHBOARD_EXTRA_FEATURE_NAMES,
    IncrementalFeatureEngine,
)
from ote_live.media.chart_capture import CapturedChartArtifact, SignalChartCaptureService
from ote_live.models.ensemble import load_direction_models
from ote_live.models.loaders import LoadedRuntimeModel
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.storage.repositories import LiveAuditRepository


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
    ) -> None:
        self.bindings = tuple(bindings)
        if not self.bindings:
            raise ValueError("At least one loaded runtime model binding is required.")

        manifests = [binding.loaded_model.manifest for binding in self.bindings]
        self.audit_repository = audit_repository
        self.group_name = str(group_name).strip().upper() or "OTE"
        self.data_supplier = str(data_supplier).strip().upper() or "FMP"
        self._frvp_dashboard_state_enabled = _contains_frvp_models(manifests)
        self.feature_engine = feature_engine or IncrementalFeatureEngine(
            manifests,
            rolling_window_bars=rolling_window_bars,
            extra_feature_names=(
                FRVP_DASHBOARD_EXTRA_FEATURE_NAMES
                if self._frvp_dashboard_state_enabled
                else ()
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
        self.last_processed_timestamp: datetime | None = None

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
        group_name: str = "OTE",
        data_supplier: str = "FMP",
    ) -> LiveSignalProcessor | None:
        bindings: list[SignalRuntimeModelBinding] = []
        for manifest_path in (long_manifest_path, short_manifest_path):
            if manifest_path is None:
                continue
            bundle = load_direction_models(
                manifest_path,
                skip_unavailable_backends=skip_unavailable_backends,
                require_complete_policy=require_complete_policy,
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
                            shadow_mode=False,
                        )
                    )
                continue
            for model_manifest in bundle.direction_manifest.models:
                if getattr(model_manifest, "status", None) == "deprecated":
                    continue
                loaded_model = bundle.loaded_models.get(model_manifest.model_id)
                if loaded_model is None:
                    continue
                shadow_mode = getattr(model_manifest, "status", None) != "active"
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

    def process_bars(
        self,
        bars: Iterable[MarketBar],
        *,
        emit_operator_artifacts: bool,
    ) -> tuple[RuntimeSignalResult, ...]:
        results: list[RuntimeSignalResult] = []
        for bar in sorted(bars, key=lambda item: item.timestamp):
            if bar.asset != self.asset or bar.timeframe != self.timeframe:
                continue

            self.feature_engine.ingest_bar(bar)
            self.last_processed_timestamp = bar.timestamp

            try:
                feature_frame = self.feature_engine.build_feature_frame(include_policy_features=True)
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
                continue

            if feature_frame.empty:
                continue

            policy_frame = self._build_policy_frame(feature_frame)
            source_row_idx = self._resolve_source_row_idx(bar)
            self._persist_dashboard_state(bar=bar, policy_frame=policy_frame)

            for binding in self.bindings:
                loaded_model = binding.loaded_model
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
                    decision_path = self.decision_engine.evaluate_prediction(
                        prediction,
                        live_policy=loaded_model.manifest.live_policy,
                        policy_context=policy_frame,
                        timezone_contract=loaded_model.manifest.timezone_contract,
                        shadow_mode=binding.shadow_mode,
                        feature_snapshot=snapshot,
                        runtime_manifest=loaded_model.manifest,
                    )
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
                    _should_send_notifications(decision_path.signal)
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
                        shadow_mode=binding.shadow_mode,
                        timestamp=bar.timestamp,
                        signal_decision_id=signal_decision_id,
                        notification_status=notification.status if notification is not None else None,
                        sms_notification_status=(
                            sms_notification.status if sms_notification is not None else None
                        ),
                        media_artifact_id=media_artifact.artifact_id if media_artifact is not None else None,
                    )
                )

        return tuple(results)

    def _build_policy_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        market_frame = self.feature_engine.state.to_frame()
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
        return policy_frame.loc[:, ordered_columns]

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
                instrument_id
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
                instrument_id
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
        if not self._frvp_dashboard_state_enabled:
            return
        try:
            persist_frvp_dashboard_state(
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
                message="Persisting FRVP dashboard state failed.",
                payload={
                    "group_name": self.group_name,
                    "asset": bar.asset,
                    "timeframe": bar.timeframe,
                    "timestamp_utc": bar.timestamp.isoformat(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )


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
    )


def _to_python_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _should_send_notifications(signal: SignalDecision) -> bool:
    return str(signal.decision) == "emit"


def _should_capture_operator_artifacts(signal: SignalDecision) -> bool:
    decision = str(signal.decision)
    if decision == "emit":
        return True
    if decision != "shadow":
        return False
    if signal.threshold is None:
        return False
    return float(signal.probability) >= float(signal.threshold)


def _contains_frvp_models(manifests: Sequence[object]) -> bool:
    for manifest in manifests:
        model_id = str(getattr(manifest, "model_id", "") or "").lower()
        if model_id.startswith("frvp_"):
            return True
    return False

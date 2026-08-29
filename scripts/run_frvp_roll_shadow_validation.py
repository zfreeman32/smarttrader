from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.io import standardize_market_frame
from ote_live.contracts.market_data import MarketBar
from ote_live.dashboard.view_state import DASHBOARD_VIEW_STATE_SCOPE
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.parity_replay import replay_feature_parity
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.models.ensemble import load_direction_models
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.replay import replay_audited_signal
from ote_live.storage.repositories import LiveAuditRepository

DEFAULT_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260715" / "live_runtime_manifest_long.json"
)
DEFAULT_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260715" / "live_runtime_manifest_short.json"
)
DEFAULT_SELECTION_SUMMARY_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260715" / "shadow_selection_summary.json"
)
DEFAULT_MARKET_CSV_PATH = REPO_ROOT / "data" / "futures_data" / "ES-5m-tagged.csv"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "model_testing" / "reports" / "frvp_roll_shadow_validation" / "frvp_es_shadow_roll_20260717"
)
DEFAULT_DB_PATH = DEFAULT_OUTPUT_ROOT / "roll_shadow_validation.sqlite3"
DEFAULT_AUDIT_REPLAY_FEATURE_ATOL = 1e-2
_BASE_MARKET_FRAME_COLUMNS = frozenset(
    {
        "asset",
        "timeframe",
        "datetime",
        "date",
        "time",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid",
        "ask",
        "spread",
        "source",
        "symbol",
        "contract_symbol",
        "instrument_id",
    }
)


@dataclass(frozen=True)
class RollBoundary:
    boundary_index: int
    boundary_timestamp: pd.Timestamp
    from_contract: str
    to_contract: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a historical ES contract roll through the FRVP shadow runtime, "
            "audit the live decisions, and compare selected live features to the canonical offline frame."
        )
    )
    parser.add_argument("--long-runtime-manifest-path", type=Path, default=DEFAULT_LONG_RUNTIME_MANIFEST_PATH)
    parser.add_argument("--short-runtime-manifest-path", type=Path, default=DEFAULT_SHORT_RUNTIME_MANIFEST_PATH)
    parser.add_argument("--selection-summary-path", type=Path, default=DEFAULT_SELECTION_SUMMARY_PATH)
    parser.add_argument("--market-csv-path", type=Path, default=DEFAULT_MARKET_CSV_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--group-name",
        type=str,
        default="FRVP_ROLL_VALIDATION",
        help="Runtime-state key used for dashboard persistence during the replay.",
    )
    parser.add_argument(
        "--roll-timestamp",
        type=str,
        default="",
        help="Optional exact roll boundary timestamp in ISO format. Defaults to the latest boundary with enough context.",
    )
    parser.add_argument("--pre-roll-bars", type=int, default=288)
    parser.add_argument("--post-roll-bars", type=int, default=288)
    parser.add_argument("--sample-audit-replays", type=int, default=12)
    parser.add_argument("--skip-feature-parity", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    db_path = args.db_path.resolve()
    if db_path.exists():
        db_path.unlink()

    selected_models = _load_selected_model_ids(args.selection_summary_path)
    bindings = _load_runtime_bindings(
        long_runtime_manifest_path=args.long_runtime_manifest_path,
        short_runtime_manifest_path=args.short_runtime_manifest_path,
        selected_model_ids=selected_models,
    )
    if not bindings:
        raise ValueError("No FRVP runtime models were loaded for roll validation.")

    manifests = [binding.loaded_model.manifest for binding in bindings]
    processor = _build_processor(bindings=bindings, db_path=db_path, group_name=args.group_name)
    feature_context_columns = _resolve_runtime_feature_context_columns(manifests)
    market = _load_market_frame(
        args.market_csv_path,
        manifests=manifests,
        feature_context_columns=feature_context_columns,
    )
    required_prefill_bars = (
        processor.feature_engine.plan.runtime_history_bars + processor.feature_engine.plan.additive_seed_prefix_bars
    )
    roll_boundary = _select_roll_boundary(
        market=market,
        requested_timestamp=args.roll_timestamp,
        required_prefill_bars=required_prefill_bars,
        pre_roll_bars=args.pre_roll_bars,
        post_roll_bars=args.post_roll_bars,
    )

    warmup_start = roll_boundary.boundary_index - required_prefill_bars - args.pre_roll_bars
    replay_start = roll_boundary.boundary_index - args.pre_roll_bars
    replay_end = roll_boundary.boundary_index + args.post_roll_bars

    warmup_frame = market.iloc[warmup_start:replay_start].reset_index(drop=True)
    pre_roll_frame = market.iloc[replay_start:roll_boundary.boundary_index].reset_index(drop=True)
    post_roll_frame = market.iloc[roll_boundary.boundary_index : replay_end + 1].reset_index(drop=True)
    replay_frame = market.iloc[replay_start : replay_end + 1].reset_index(drop=True)
    parity_market_frame = market.iloc[warmup_start : replay_end + 1].reset_index(drop=True)

    parity_reports: list[dict[str, Any]] = []
    parity_error: dict[str, Any] | None = None
    store = processor.audit_repository.store
    try:
        _bulk_upsert_bars(
            store,
            _frame_to_bars(warmup_frame, feature_context_columns=feature_context_columns),
        )
        warmed_bars = processor.warm_from_store()

        _bulk_upsert_bars(
            store,
            _frame_to_bars(pre_roll_frame, feature_context_columns=feature_context_columns),
        )
        pre_roll_results = processor.process_new_bars_from_store(
            emit_operator_artifacts=False,
            max_timestamp=_frame_last_timestamp(pre_roll_frame),
        )
        pre_roll_dashboard_state = store.get_runtime_state(
            scope=DASHBOARD_VIEW_STATE_SCOPE,
            state_key=args.group_name,
        ) or {}

        _bulk_upsert_bars(
            store,
            _frame_to_bars(post_roll_frame, feature_context_columns=feature_context_columns),
        )
        post_roll_results = processor.process_new_bars_from_store(
            emit_operator_artifacts=False,
            max_timestamp=_frame_last_timestamp(post_roll_frame),
        )
        post_roll_dashboard_state = store.get_runtime_state(
            scope=DASHBOARD_VIEW_STATE_SCOPE,
            state_key=args.group_name,
        ) or {}

        health_events = processor.audit_repository.fetch_health_events(severity="error")
        signal_decision_ids = _fetch_signal_decision_ids(store)
        audit_replay_reports = _run_sample_audit_replays(
            audit_repository=processor.audit_repository,
            signal_decision_ids=signal_decision_ids,
            sample_limit=args.sample_audit_replays,
        )
        if args.skip_feature_parity:
            parity_error = {
                "status": "skipped",
                "error_type": "FeatureParitySkipped",
                "message": "Feature parity was skipped for this validation run.",
            }
        else:
            try:
                parity_reports = _run_feature_parity_reports(
                    long_runtime_manifest_path=args.long_runtime_manifest_path,
                    short_runtime_manifest_path=args.short_runtime_manifest_path,
                    selected_model_ids=selected_models,
                    market_frame=parity_market_frame,
                    evaluation_bars=len(replay_frame),
                )
            except Exception as exc:
                parity_error = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
    finally:
        store.close()

    summary_payload = _build_summary_payload(
        output_root=output_root,
        db_path=db_path,
        selected_model_ids=selected_models,
        required_prefill_bars=required_prefill_bars,
        roll_boundary=roll_boundary,
        warmed_bars=warmed_bars,
        warmup_frame=warmup_frame,
        pre_roll_frame=pre_roll_frame,
        post_roll_frame=post_roll_frame,
        replay_frame=replay_frame,
        pre_roll_results=pre_roll_results,
        post_roll_results=post_roll_results,
        pre_roll_dashboard_state=pre_roll_dashboard_state,
        post_roll_dashboard_state=post_roll_dashboard_state,
        health_events=health_events,
        signal_decision_ids=signal_decision_ids,
        audit_replay_reports=audit_replay_reports,
        parity_reports=parity_reports,
        parity_error=parity_error,
    )

    summary_path = output_root / "roll_shadow_validation_summary.json"
    markdown_path = output_root / "roll_shadow_validation_summary.md"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(summary_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": _repo_relative(summary_path),
                "markdown_path": _repo_relative(markdown_path),
                "selected_model_ids": selected_models,
                "roll_boundary_timestamp": roll_boundary.boundary_timestamp.isoformat(),
                "from_contract": roll_boundary.from_contract,
                "to_contract": roll_boundary.to_contract,
                "signal_decision_count": len(signal_decision_ids),
                "health_error_count": len(health_events),
                "feature_parity_report_count": len(parity_reports),
                "feature_parity_status": None if parity_error is None else parity_error["status"],
            },
            indent=2,
        )
    )
    return 0


def _load_selected_model_ids(selection_summary_path: Path) -> list[str]:
    payload = _read_json(selection_summary_path)
    active_model_ids = payload.get("active_model_ids") or []
    if active_model_ids:
        return [str(model_id) for model_id in active_model_ids if model_id]
    recommended = payload.get("recommended_shadow_dashboard_models") or []
    model_ids = [str(item["model_id"]) for item in recommended if item.get("model_id")]
    if model_ids:
        return model_ids
    family_leaders = payload.get("family_leaders") or []
    return [str(item["model_id"]) for item in family_leaders if item.get("model_id")]


def _load_runtime_bindings(
    *,
    long_runtime_manifest_path: Path,
    short_runtime_manifest_path: Path,
    selected_model_ids: Sequence[str],
) -> list[SignalRuntimeModelBinding]:
    requested_ids = set(selected_model_ids)
    bindings: list[SignalRuntimeModelBinding] = []
    for path in (long_runtime_manifest_path, short_runtime_manifest_path):
        bundle = load_direction_models(
            path,
            model_ids=selected_model_ids,
            skip_unavailable_backends=True,
        )
        for manifest in bundle.direction_manifest.models:
            if manifest.model_id not in requested_ids:
                continue
            loaded_model = bundle.loaded_models.get(manifest.model_id)
            if loaded_model is None:
                continue
            bindings.append(
                SignalRuntimeModelBinding(
                    loaded_model=loaded_model,
                    shadow_mode=getattr(manifest, "status", None) != "active",
                )
            )
    return bindings


def _build_processor(
    *,
    bindings: Sequence[SignalRuntimeModelBinding],
    db_path: Path,
    group_name: str,
) -> LiveSignalProcessor:
    store = SQLiteLiveDataStore(db_path)
    audit_repository = LiveAuditRepository(store)
    return LiveSignalProcessor(
        bindings=bindings,
        audit_repository=audit_repository,
        emailer=None,
        sms_sender=None,
        chart_capture_service=None,
        group_name=group_name,
        data_supplier="REPLAY",
    )


def _load_market_frame(
    path: Path,
    *,
    manifests: Sequence[LiveRuntimeManifest],
    feature_context_columns: Sequence[str],
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    market = standardize_market_frame(
        raw,
        source_timezone="UTC",
        canonical_timezone="UTC",
    )
    market["datetime"] = pd.to_datetime(market["datetime"], errors="coerce", utc=True)
    market = market.dropna(subset=["datetime"]).reset_index(drop=True)
    if "contract_symbol" not in market.columns:
        raise ValueError("Roll validation market frame is missing contract_symbol.")
    market["contract_symbol"] = market["contract_symbol"].astype(str)
    if feature_context_columns:
        market = _attach_feature_context_columns(
            market,
            manifests=manifests,
            feature_context_columns=feature_context_columns,
        )
    return market


def _select_roll_boundary(
    *,
    market: pd.DataFrame,
    requested_timestamp: str,
    required_prefill_bars: int,
    pre_roll_bars: int,
    post_roll_bars: int,
) -> RollBoundary:
    boundaries: list[RollBoundary] = []
    contract_series = market["contract_symbol"].astype(str)
    changed = contract_series.ne(contract_series.shift()).fillna(False)
    for boundary_index in market.index[changed]:
        if boundary_index <= 0:
            continue
        boundaries.append(
            RollBoundary(
                boundary_index=int(boundary_index),
                boundary_timestamp=pd.Timestamp(market.at[boundary_index, "datetime"]),
                from_contract=str(contract_series.iloc[boundary_index - 1]),
                to_contract=str(contract_series.iloc[boundary_index]),
            )
        )

    required_lead = required_prefill_bars + pre_roll_bars
    eligible = [
        boundary
        for boundary in boundaries
        if boundary.boundary_index >= required_lead
        and (len(market) - boundary.boundary_index - 1) >= post_roll_bars
    ]
    if not eligible:
        raise ValueError("No eligible roll boundary had enough prefill and post-roll bars for validation.")

    if requested_timestamp:
        target = pd.Timestamp(requested_timestamp)
        for boundary in eligible:
            if boundary.boundary_timestamp == target:
                return boundary
        raise ValueError(f"Could not find an eligible roll boundary at {target.isoformat()}.")

    return eligible[-1]


def _frame_to_bars(frame: pd.DataFrame, *, feature_context_columns: Sequence[str]) -> list[MarketBar]:
    return [
        MarketBar(
            asset="ES",
            timeframe="5m",
            timestamp=pd.Timestamp(row.datetime).to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(getattr(row, "volume", 0.0) or 0.0),
            bid=_optional_float(getattr(row, "bid", None)),
            ask=_optional_float(getattr(row, "ask", None)),
            spread=_optional_float(getattr(row, "spread", None)),
            source=str(getattr(row, "source", "historical_replay") or "historical_replay"),
            symbol=str(getattr(row, "symbol", "ES.v.0") or "ES.v.0"),
            contract_symbol=str(getattr(row, "contract_symbol", "") or "") or None,
            instrument_id=_optional_int(getattr(row, "instrument_id", None)),
            feature_context=_extract_feature_context_from_row(
                row,
                feature_context_columns=feature_context_columns,
            ),
        )
        for row in frame.itertuples(index=False)
    ]


def _bulk_upsert_bars(store: SQLiteLiveDataStore, bars: Iterable[MarketBar]) -> None:
    payloads = []
    now = datetime.now(timezone.utc).isoformat()
    for bar in bars:
        payloads.append(
            (
                bar.asset,
                bar.timeframe,
                bar.timestamp.astimezone(timezone.utc).isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.bid,
                bar.ask,
                bar.spread,
                bar.source,
                bar.symbol,
                bar.contract_symbol,
                bar.instrument_id,
                json.dumps(bar.feature_context or {}, sort_keys=True, default=str),
                now,
                now,
            )
        )
    if not payloads:
        return
    store.connection.executemany(
        """
        INSERT INTO canonical_bars (
            asset, timeframe, timestamp_utc, open, high, low, close, volume,
            bid, ask, spread, source, symbol, contract_symbol, instrument_id, feature_context_json,
            inserted_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset, timeframe, timestamp_utc) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            bid = excluded.bid,
            ask = excluded.ask,
            spread = excluded.spread,
            source = excluded.source,
            symbol = excluded.symbol,
            contract_symbol = excluded.contract_symbol,
            instrument_id = excluded.instrument_id,
            feature_context_json = excluded.feature_context_json,
            updated_at_utc = excluded.updated_at_utc
        """,
        payloads,
    )
    store.connection.commit()


def _fetch_signal_decision_ids(store: SQLiteLiveDataStore) -> list[int]:
    rows = store.connection.execute(
        """
        SELECT id
        FROM signal_decisions
        ORDER BY timestamp_utc ASC, id ASC
        """
    ).fetchall()
    return [int(row["id"]) for row in rows]


def _run_sample_audit_replays(
    *,
    audit_repository: LiveAuditRepository,
    signal_decision_ids: Sequence[int],
    sample_limit: int,
) -> list[dict[str, Any]]:
    if sample_limit <= 0 or not signal_decision_ids:
        return []
    indices = sorted(
        {
            0,
            max(0, len(signal_decision_ids) // 2),
            len(signal_decision_ids) - 1,
            *range(min(sample_limit, len(signal_decision_ids))),
        }
    )
    sampled_ids = [signal_decision_ids[index] for index in indices[:sample_limit]]
    reports: list[dict[str, Any]] = []
    for signal_decision_id in sampled_ids:
        report = replay_audited_signal(
            audit_repository,
            signal_decision_id=signal_decision_id,
            feature_value_atol=DEFAULT_AUDIT_REPLAY_FEATURE_ATOL,
            assert_matches=True,
        )
        reports.append(
            {
                "signal_decision_id": signal_decision_id,
                "model_id": report.model_id,
                "backend": report.backend,
                "direction": report.direction,
                "bars_replayed": report.bars_replayed,
                "feature_snapshot_matches": report.feature_snapshot_matches,
                "prediction_matches": report.prediction_matches,
                "signal_matches": report.signal_matches,
                "raw_probability_abs_diff": report.raw_probability_abs_diff,
                "calibrated_probability_abs_diff": report.calibrated_probability_abs_diff,
                "decision": report.replayed_signal.decision,
                "timestamp_utc": report.replayed_signal.timestamp.isoformat(),
            }
        )
    return reports


def _run_feature_parity_reports(
    *,
    long_runtime_manifest_path: Path,
    short_runtime_manifest_path: Path,
    selected_model_ids: Sequence[str],
    market_frame: pd.DataFrame,
    evaluation_bars: int,
) -> list[dict[str, Any]]:
    reports = []
    for bundle_path, label in (
        (long_runtime_manifest_path, "long"),
        (short_runtime_manifest_path, "short"),
    ):
        direction_manifest = load_direction_runtime_manifest(bundle_path)
        manifests = [
            manifest
            for manifest in direction_manifest.models
            if manifest.model_id in set(selected_model_ids)
        ]
        if not manifests:
            continue
        report = replay_feature_parity(
            manifests,
            market_frame=market_frame,
            evaluation_bars=evaluation_bars,
        )
        reports.append(
            {
                "direction_bundle": label,
                "model_ids": list(report.model_ids),
                "rows_evaluated": report.rows_evaluated,
                "rows_ready_for_comparison": report.rows_ready_for_comparison,
                "missing_reference_rows": report.missing_reference_rows,
                "compared_cells": report.compared_cells,
                "matched_cells": report.matched_cells,
                "mismatched_cells": report.mismatched_cells,
                "matched_cell_ratio": report.matched_cell_ratio,
                "max_abs_diff": report.max_abs_diff,
                "sample_mismatches": [
                    {
                        "timestamp_utc": mismatch.timestamp.isoformat(),
                        "feature_name": mismatch.feature_name,
                        "expected_value": mismatch.expected_value,
                        "actual_value": mismatch.actual_value,
                        "abs_diff": mismatch.abs_diff,
                    }
                    for mismatch in report.sample_mismatches
                ],
            }
        )
    return reports


def _resolve_runtime_feature_context_columns(manifests: Sequence[LiveRuntimeManifest]) -> tuple[str, ...]:
    selected: list[str] = []
    for manifest in manifests:
        for feature_name in manifest.feature_manifest.selected_feature_names:
            if not str(feature_name).startswith("htf_confluence_"):
                continue
            if feature_name not in selected:
                selected.append(str(feature_name))
    return tuple(selected)


def _attach_feature_context_columns(
    market: pd.DataFrame,
    *,
    manifests: Sequence[LiveRuntimeManifest],
    feature_context_columns: Sequence[str],
) -> pd.DataFrame:
    enriched = market.copy()
    for column_name in feature_context_columns:
        if column_name in enriched.columns:
            continue
        context_frame = _load_feature_context_column_from_manifests(
            manifests=manifests,
            column_name=column_name,
        )
        if context_frame is None:
            continue
        enriched = enriched.merge(context_frame, on="datetime", how="left")
    return enriched


def _load_feature_context_column_from_manifests(
    *,
    manifests: Sequence[LiveRuntimeManifest],
    column_name: str,
) -> pd.DataFrame | None:
    for manifest in manifests:
        feature_path = _resolve_repo_path(manifest.source_lineage.feature_csv)
        frame = pd.read_csv(
            feature_path,
            usecols=lambda candidate: candidate in {"datetime", column_name},
        )
        if column_name not in frame.columns:
            continue
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last")
        return frame.loc[:, ["datetime", column_name]].reset_index(drop=True)
    return None


def _build_summary_payload(
    *,
    output_root: Path,
    db_path: Path,
    selected_model_ids: Sequence[str],
    required_prefill_bars: int,
    roll_boundary: RollBoundary,
    warmed_bars: int,
    warmup_frame: pd.DataFrame,
    pre_roll_frame: pd.DataFrame,
    post_roll_frame: pd.DataFrame,
    replay_frame: pd.DataFrame,
    pre_roll_results: Sequence[Any],
    post_roll_results: Sequence[Any],
    pre_roll_dashboard_state: dict[str, Any],
    post_roll_dashboard_state: dict[str, Any],
    health_events: Sequence[Any],
    signal_decision_ids: Sequence[int],
    audit_replay_reports: Sequence[dict[str, Any]],
    parity_reports: Sequence[dict[str, Any]],
    parity_error: dict[str, Any] | None,
) -> dict[str, Any]:
    pre_counts = _count_results(pre_roll_results)
    post_counts = _count_results(post_roll_results)
    signal_rows = int(len(signal_decision_ids))
    parity_attempted = parity_error is None or parity_error.get("status") != "skipped"
    parity_passed = (
        parity_error is None
        and all(report["mismatched_cells"] == 0 and report["missing_reference_rows"] == 0 for report in parity_reports)
    )
    health_event_messages = [
        str(item.message)
        for item in health_events
    ]
    missing_live_features = sorted(
        {
            fragment.strip()
            for item in health_events
            if item.component == "signal_runtime.features"
            for fragment in str(item.payload.get("error_message") or "").split(":", maxsplit=1)[-1].split(",")
            if "selected features" not in fragment
            and fragment.strip()
        }
    )
    blocking_findings: list[str] = []
    if missing_live_features:
        blocking_findings.append(
            "Live FRVP runtime could not rebuild required carry-through features at the roll boundary: "
            + ", ".join(missing_live_features)
        )
    if parity_error is not None and parity_error.get("status") == "failed":
        blocking_findings.append(
            "Feature parity could not complete because the live feature engine does not currently recreate every "
            f"selected feature from market bars alone ({parity_error.get('error_type')}: {parity_error.get('message')})."
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": _repo_relative(output_root),
        "db_path": _repo_relative(db_path),
        "selected_model_ids": list(selected_model_ids),
        "required_prefill_bars": int(required_prefill_bars),
        "roll_boundary": {
            "boundary_timestamp_utc": roll_boundary.boundary_timestamp.isoformat(),
            "from_contract": roll_boundary.from_contract,
            "to_contract": roll_boundary.to_contract,
            "boundary_index": roll_boundary.boundary_index,
        },
        "window_sizes": {
            "warmup_bars": int(len(warmup_frame)),
            "pre_roll_replay_bars": int(len(pre_roll_frame)),
            "post_roll_replay_bars": int(len(post_roll_frame)),
            "total_replay_bars": int(len(replay_frame)),
            "warmed_bars_used_by_processor": int(warmed_bars),
        },
        "runtime_transition": {
            "pre_roll_last_bar_timestamp_utc": _frame_last_timestamp(pre_roll_frame).isoformat(),
            "pre_roll_last_contract_symbol": str(pre_roll_frame.iloc[-1]["contract_symbol"]),
            "post_roll_first_bar_timestamp_utc": pd.Timestamp(post_roll_frame.iloc[0]["datetime"]).isoformat(),
            "post_roll_first_contract_symbol": str(post_roll_frame.iloc[0]["contract_symbol"]),
            "dashboard_state_pre_roll_contract_symbol": pre_roll_dashboard_state.get("contract_symbol"),
            "dashboard_state_post_roll_contract_symbol": post_roll_dashboard_state.get("contract_symbol"),
            "dashboard_state_post_roll_latest_bar_timestamp_utc": post_roll_dashboard_state.get("latest_bar_timestamp_utc"),
            "dashboard_state_contract_switch_recorded": (
                pre_roll_dashboard_state.get("contract_symbol") == str(pre_roll_frame.iloc[-1]["contract_symbol"])
                and post_roll_dashboard_state.get("contract_symbol") == str(post_roll_frame.iloc[-1]["contract_symbol"])
            ),
        },
        "shadow_decision_counts": {
            "pre_roll": pre_counts,
            "post_roll": post_counts,
            "total_signal_decisions_persisted": signal_rows,
        },
        "health_events": [
            {
                "component": item.component,
                "event_type": item.event_type,
                "severity": item.severity,
                "message": item.message,
                "event_timestamp_utc": item.event_timestamp_utc.isoformat(),
            }
            for item in health_events
        ],
        "health_error_count": len(health_events),
        "missing_live_feature_names": missing_live_features,
        "sample_audit_replays": list(audit_replay_reports),
        "feature_parity_attempted": parity_attempted,
        "feature_parity_passed": parity_passed,
        "feature_parity_reports": list(parity_reports),
        "feature_parity_error": parity_error,
        "blocking_findings": blocking_findings,
        "validation_passed": (
            len(health_events) == 0
            and all(report["feature_snapshot_matches"] and report["prediction_matches"] and report["signal_matches"] for report in audit_replay_reports)
            and (not parity_attempted or parity_passed)
            and pre_roll_dashboard_state.get("contract_symbol") == str(pre_roll_frame.iloc[-1]["contract_symbol"])
            and post_roll_dashboard_state.get("contract_symbol") == str(post_roll_frame.iloc[-1]["contract_symbol"])
        ),
        "advisories": [
            "This replay validates the FRVP shadow runtime against a historical contract switch using tagged contract_symbol data.",
            "It does not remove the current live-ops requirement to switch the active IBKR front-month contract explicitly at roll time.",
            *health_event_messages[:1],
        ],
    }


def _count_results(results: Sequence[Any]) -> dict[str, Any]:
    decision_counter = Counter()
    by_model: dict[str, Counter[str]] = {}
    for item in results:
        decision_counter[str(item.decision)] += 1
        model_counter = by_model.setdefault(str(item.model_id), Counter())
        model_counter[str(item.decision)] += 1
    return {
        "decision_counts": dict(sorted(decision_counter.items())),
        "model_decision_counts": {
            model_id: dict(sorted(counter.items()))
            for model_id, counter in sorted(by_model.items())
        },
        "result_count": len(results),
    }


def _frame_last_timestamp(frame: pd.DataFrame) -> datetime:
    return pd.Timestamp(frame.iloc[-1]["datetime"]).to_pydatetime()


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _extract_feature_context_from_row(
    row,
    *,
    feature_context_columns: Sequence[str],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for column_name in feature_context_columns:
        if column_name in _BASE_MARKET_FRAME_COLUMNS:
            continue
        value = getattr(row, column_name, None)
        if value is None or pd.isna(value):
            continue
        payload[column_name] = value.item() if hasattr(value, "item") else value
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    roll = payload["roll_boundary"]
    runtime = payload["runtime_transition"]
    return "\n".join(
        [
            "# FRVP Roll Shadow Validation",
            "",
            f"- Generated: `{payload['generated_at_utc']}`",
            f"- Selected models: `{', '.join(payload['selected_model_ids'])}`",
            f"- Roll boundary: `{roll['boundary_timestamp_utc']}` from `{roll['from_contract']}` to `{roll['to_contract']}`",
            f"- Validation passed: `{payload['validation_passed']}`",
            f"- Health error count: `{payload['health_error_count']}`",
            f"- Feature parity attempted: `{payload['feature_parity_attempted']}`",
            f"- Feature parity passed: `{payload['feature_parity_passed']}`",
            "",
            "## Runtime Transition",
            "",
            f"- Pre-roll dashboard contract: `{runtime['dashboard_state_pre_roll_contract_symbol']}`",
            f"- Post-roll dashboard contract: `{runtime['dashboard_state_post_roll_contract_symbol']}`",
            f"- Dashboard contract switch recorded: `{runtime['dashboard_state_contract_switch_recorded']}`",
            "",
            "## Decision Counts",
            "",
            f"- Pre-roll: `{payload['shadow_decision_counts']['pre_roll']['decision_counts']}`",
            f"- Post-roll: `{payload['shadow_decision_counts']['post_roll']['decision_counts']}`",
            f"- Total persisted signal decisions: `{payload['shadow_decision_counts']['total_signal_decisions_persisted']}`",
            "",
            "## Blocking Findings",
            "",
            *([f"- {line}" for line in payload["blocking_findings"]] or ["- None"]),
            "",
            "## Advisories",
            "",
            *[f"- {line}" for line in payload["advisories"]],
        ]
    ) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _resolve_repo_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

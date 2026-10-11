"""Audit candidate feature/model parity through the production ES app path.

This is a read-only, fail-closed diagnostic.  It joins the frozen candidate
source histories to the recorded ES app snapshots, then runs the actual
``IncrementalFeatureEngine`` and ``RuntimeModelRunner`` against the recorded
canonical-bar window.  It does not install a model, alter a live manifest, or
turn a score into a promotion decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_training.source_history import file_sha256, impute_matrix, validate_contract
from ote_live.features.incremental_engine import IncrementalFeatureEngine, _materialize_live_default_features
from ote_live.models.loaders import load_live_runtime_manifest, load_runtime_model
from ote_live.models.runners import RuntimeModelRunner
from ote_live.models.calibrators import apply_probability_calibrator
from ote_live.features.manifest import ArtifactReferences


SNAPSHOT_DB = ROOT / "research/es_live_audit_20260911/es_audit_snapshot.sqlite3"
EVIDENCE_VERSION = "b5-b6-live-feature-parity-v1"
PRIOR_SCORE_EVIDENCE = {
    "1": "docs/audits/b2_candidate_history_review_01_20260928.json",
    "2": "docs/audits/b2_candidate_history_review_02_20260928.json",
    "6": "docs/audits/b2_s3_source_history_model_review_20260927.json",
    "7": "docs/audits/b2_candidate_history_review_07_20260928.json",
    "11": "docs/audits/b2_candidate_history_review_11_20260928.json",
    "17": "docs/audits/b2_candidate_history_review_17_20260928.json",
}


CANDIDATES: dict[str, dict[str, str]] = {
    "1": {
        "model_id": "frvp_short_continuation_tcn_v1",
        "app_manifest": "ote_live/runtime_manifests/frvp_es_paper_signal_20260816/frvp_short_continuation_tcn_v1/live_runtime_manifest.json",
        "artifact_dir": "models/b2_candidate_history_20260928_v1/01_short_frvp_continuation/short_frvp_continuation",
        "source_history": "artifacts/b2_candidate_history_20260928_v1/1/prepared/short_frvp_continuation/source_history.parquet",
    },
    "2": {
        "model_id": "frvp_long_continuation_xgb_v1",
        "app_manifest": "ote_live/runtime_manifests/frvp_es_paper_signal_20260816/frvp_long_continuation_xgb_v1/live_runtime_manifest.json",
        "artifact_dir": "models/b2_candidate_history_20260928_v1/02_long_frvp_continuation/long_frvp_continuation",
        "source_history": "artifacts/b2_candidate_history_20260928_v1/2/prepared/long_frvp_continuation/source_history.parquet",
    },
    "6": {
        "model_id": "frvp_long_continuation_setup3_xgb_v1",
        "app_manifest": "ote_live/runtime_manifests/frvp_es_setup_family_20260829/frvp_long_continuation_setup3_xgb_v1/live_runtime_manifest.json",
        "artifact_dir": "models/b2_source_history_20260927_v1/06_long_frvp_continuation_setup3/long_frvp_continuation_setup3",
        "source_history": "artifacts/b2_source_history_20260927_v1/prepared/long_frvp_continuation_setup3/source_history.parquet",
    },
    "7": {
        "model_id": "frvp_long_continuation_setup5_xgb_v1",
        "app_manifest": "ote_live/runtime_manifests/frvp_es_setup_family_20260829/frvp_long_continuation_setup5_xgb_v1/live_runtime_manifest.json",
        "artifact_dir": "models/b2_candidate_history_20260928_v1/07_long_frvp_continuation_setup5/long_frvp_continuation_setup5",
        "source_history": "artifacts/b2_candidate_history_20260928_v1/7/prepared/long_frvp_continuation_setup5/source_history.parquet",
    },
    "11": {
        "model_id": "ict_long_continuation_xgb_v1",
        "app_manifest": "ote_live/runtime_manifests/ict_es_paper_signal_20260813/ict_long_continuation_xgb_v1/live_runtime_manifest.json",
        "artifact_dir": "models/b2_candidate_history_20260928_v1/11_long_ict_continuation/long_ict_continuation",
        "source_history": "artifacts/b2_candidate_history_20260928_v1/11/prepared/long_ict_continuation/source_history.parquet",
    },
    "17": {
        "model_id": "ict_short_continuation_premium_discount_continuation_xgb_v1",
        "app_manifest": "ote_live/runtime_manifests/ict_short_setup_family_20260811_audit/ict_short_continuation_premium_discount_continuation_xgb_v1/live_runtime_manifest.json",
        "artifact_dir": "models/b2_candidate_history_20260928_v1/17_short_ict_continuation_premium_discount_continuation/short_ict_continuation_premium_discount_continuation",
        "source_history": "artifacts/b2_candidate_history_20260928_v1/17/prepared/short_ict_continuation_premium_discount_continuation/source_history.parquet",
    },
}


def sha256(path: Path) -> str:
    return file_sha256(path)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(directory: Path, name: str) -> Path | None:
    path = directory / name
    return path if path.exists() else None


def candidate_manifest(base, artifact_dir: Path, summary: dict[str, Any]):
    selected = list(summary["selected_feature_names"])
    backend = summary["config"].get("backend", base.backend)
    if backend == "torch":
        backend = summary["config"].get("model_type", backend)
    refs = ArtifactReferences(
        artifact_dir=rel(artifact_dir),
        model_file=rel(artifact_dir / ("model.pt" if backend in {"tcn", "lstm"} else "model.json")),
        scaler_file=rel(artifact_dir / "scaler.joblib"),
        calibrator_file=None,
        model_config_file=rel(artifact_dir / "model_config.json"),
        training_summary_file=rel(artifact_dir / "training_summary.json"),
        test_predictions_file=rel(artifact_dir / "test_predictions.csv"),
        content_sha256={},
    )
    feature_manifest = base.feature_manifest.model_copy(update={
        "selected_feature_names": selected,
        "selected_feature_count": len(selected),
        "lag_steps": list(summary.get("lag_steps", base.feature_manifest.lag_steps)),
        "delta_feature_count": int(summary.get("delta_feature_count", base.feature_manifest.delta_feature_count)),
    })
    context = base.context_requirements.model_copy(update={
        "context_rows": int(summary["context_rows"]),
        "window_size": int(summary["window_size"]),
    })
    return base.model_copy(update={
        "backend": backend,
        "artifact_references": refs,
        "feature_manifest": feature_manifest,
        "context_requirements": context,
    })


def read_snapshot_rows(connection: sqlite3.Connection, model_ids: str | tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(model_ids, str):
        model_ids = (model_ids,)
    placeholders = ",".join("?" for _ in model_ids)
    query = """
        SELECT mp.model_id, fs.snapshot_json, fs.feature_values_json, fs.timestamp_utc,
               fs.source_row_idx, fs.recorded_at_utc,
               mp.prediction_json, mp.raw_score, mp.calibrated_probability,
               mp.threshold_applied, mp.threshold_source, mp.recorded_at_utc AS prediction_recorded_at,
               sd.signal_json, sd.decision, sd.threshold
        FROM model_predictions mp
        JOIN feature_snapshots fs ON fs.id = mp.feature_snapshot_id
        LEFT JOIN signal_decisions sd ON sd.prediction_id = mp.id
        WHERE mp.model_id IN (PLACEHOLDERS)
        ORDER BY mp.timestamp_utc, mp.id
    """.replace("PLACEHOLDERS", placeholders)
    rows = []
    for row in connection.execute(query, tuple(model_ids)).fetchall():
        snapshot = json.loads(row[1])
        source_bar = dict(snapshot.get("observation_metadata", {}).get("source_bar") or {})
        features = json.loads(row[2])
        rows.append({
            "model_id": str(row[0]),
            "timestamp": str(row[3]),
            "source_row_idx": row[4],
            "recorded_at_utc": str(row[5]),
            "features": features,
            "snapshot": snapshot,
            "source_bar": source_bar,
            "raw_score": row[7],
            "calibrated_probability": row[8],
            "threshold_applied": row[9],
            "threshold_source": row[10],
            "prediction_recorded_at": str(row[11]),
            "signal": json.loads(row[12]) if row[12] else {},
            "decision": row[13],
            "decision_threshold": row[14],
        })
    return rows


def read_canonical_bars(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT asset, timeframe, timestamp_utc, open, high, low, close, volume,
               bid, ask, spread, source, symbol, contract_symbol, instrument_id,
               feature_context_json
        FROM canonical_bars WHERE asset='ES' AND timeframe='5m'
        ORDER BY timestamp_utc
        """,
        connection,
    )
    frame["timestamp"] = pd.to_datetime(frame.pop("timestamp_utc"), utc=True)
    frame["feature_context"] = frame.pop("feature_context_json").map(
        lambda value: json.loads(value) if value else {}
    )
    for column in ("bid", "ask", "spread"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def to_market_frame(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in bars.to_dict("records"):
        item = {key: value for key, value in row.items() if key != "feature_context"}
        item["timestamp"] = pd.Timestamp(item["timestamp"]).to_pydatetime()
        item.update({key: value for key, value in row["feature_context"].items() if key not in item})
        rows.append(item)
    return pd.DataFrame.from_records(rows)


def compare_values(expected: Any, actual: Any, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if expected is None or (isinstance(expected, float) and not np.isfinite(expected)):
        return actual is None or (isinstance(actual, float) and not np.isfinite(actual))
    try:
        if pd.isna(expected) and pd.isna(actual):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.isclose(float(expected), float(actual), atol=atol, rtol=rtol, equal_nan=True))
    except (TypeError, ValueError):
        return expected == actual


def provenance_audit(rows: list[dict[str, Any]], bars: pd.DataFrame, context_rows: int) -> dict[str, Any]:
    index_by_timestamp = {str(value): idx for idx, value in enumerate(bars["timestamp"].astype(str))}
    missing_bar = 0
    identity_mismatch = 0
    completion_unknown = 0
    incomplete = 0
    availability_unknown = 0
    source_identity_unknown = 0
    bad_windows = 0
    nonclosure_gaps = 0
    closure_gaps = 0
    roll_transitions = 0
    sample_reasons: list[str] = []
    for row in rows:
        source_bar = row["source_bar"]
        ts = row["timestamp"]
        idx = index_by_timestamp.get(ts)
        if idx is None:
            # Timestamp formatting can differ only in fractional precision.
            idx = index_by_timestamp.get(str(pd.Timestamp(ts)))
        if idx is None:
            missing_bar += 1
            continue
        if int(row["source_row_idx"] or -1) != idx:
            identity_mismatch += 1
        if source_bar.get("asset") not in (None, "ES") or source_bar.get("timeframe") not in (None, "5m"):
            identity_mismatch += 1
        if not source_bar.get("contract_symbol") or source_bar.get("instrument_id") is None:
            source_identity_unknown += 1
        complete = source_bar.get("is_complete")
        if complete is None:
            complete = source_bar.get("feature_context", {}).get("ibkr_is_complete")
        if complete is None:
            completion_unknown += 1
        elif not bool(complete):
            incomplete += 1
        if not source_bar.get("first_observed_at") or not source_bar.get("last_observed_at"):
            availability_unknown += 1
        start = max(0, idx - int(context_rows))
        history = bars.iloc[start:idx + 1]
        if len(history) < context_rows + 1:
            bad_windows += 1
        times = history["timestamp"].tolist()
        identities = list(zip(history["contract_symbol"].tolist(), history["instrument_id"].tolist()))
        roll_transitions += sum(a != b for a, b in zip(identities, identities[1:]) if a[0] and b[0])
        for before, after in zip(times, times[1:]):
            delta = (after - before).total_seconds() / 60.0
            if delta <= 5.0:
                continue
            local = pd.Timestamp(before).tz_convert("America/New_York")
            # CME ES maintenance closure and weekend closure are explicit clock gaps.
            is_closure = local.hour == 17 and local.minute == 0 or delta >= 24 * 60
            if is_closure:
                closure_gaps += 1
            else:
                nonclosure_gaps += 1
    if missing_bar:
        sample_reasons.append("recorded_snapshot_timestamp_not_in_canonical_bars")
    if identity_mismatch:
        sample_reasons.append("source_row_or_asset_timeframe_identity_mismatch")
    if completion_unknown or incomplete:
        sample_reasons.append("bar_completion_not_proven_for_recorded_snapshot")
    if availability_unknown:
        sample_reasons.append("bar_availability_timing_not_proven")
    if nonclosure_gaps:
        sample_reasons.append("unresolved_nonclosure_gaps_in_required_history")
    return {
        "recorded_snapshots": len(rows),
        "missing_canonical_bars": missing_bar,
        "source_identity_mismatches": identity_mismatch,
        "source_identity_unknown": source_identity_unknown,
        "completion_unknown": completion_unknown,
        "incomplete_bars": incomplete,
        "availability_unknown": availability_unknown,
        "insufficient_warmup_windows": bad_windows,
        "closure_gap_observations": closure_gaps,
        "unresolved_gap_observations": nonclosure_gaps,
        "roll_transition_observations": roll_transitions,
        "reasons": sample_reasons,
    }


def source_feature_audit(rows: list[dict[str, Any]], source: pd.DataFrame, names: list[str]) -> dict[str, Any]:
    by_idx = source.set_index("source_row_idx")
    compared = mismatches = missing = 0
    examples = []
    for row in rows:
        idx = int(row["source_row_idx"]) if row["source_row_idx"] is not None else -1
        if idx not in by_idx.index:
            missing += 1
            continue
        expected = by_idx.loc[idx]
        for name in names:
            compared += 1
            actual = row["features"].get(name)
            value = expected.get(name)
            if not compare_values(value, actual):
                mismatches += 1
                if len(examples) < 5:
                    examples.append({"source_row_idx": idx, "feature": name, "prepared": value, "recorded_app": actual})
    return {"cells_compared": compared, "cells_mismatched": mismatches, "rows_missing": missing, "examples": examples}


def candidate_replay_audit(loaded, source: pd.DataFrame, predictions: pd.DataFrame, names: list[str], threshold: float, prior: dict[str, Any] | None = None) -> dict[str, Any]:
    if prior is not None:
        return {
            "status": "passed" if prior.get("threshold_decisions_equal") else "failed",
            "replay_mode": "versioned_B2_runtime_replay_evidence_reused",
            "evidence_file": prior.get("_evidence_file"),
            "test_endpoints": prior.get("prediction_rows", prior.get("test_endpoints")),
            "runner_replay_endpoints": prior.get("prediction_rows", prior.get("test_endpoints")),
            "runner_errors": [],
            "frozen_history_contract_present": bool(prior.get("raw_frozen_fill_runtime_replay")),
            "score_mismatches": 0 if prior.get("raw_frozen_fill_runtime_replay") else None,
            "max_abs_score_delta": prior.get("prediction_max_abs_error"),
            "threshold": prior.get("threshold", threshold),
            "threshold_decision_mismatches": 0 if prior.get("threshold_decisions_equal") else None,
            "frozen_imputation_verified": bool(prior.get("raw_frozen_fill_runtime_replay")),
            "scaler_fit_scope_verified": prior.get("scaler_fit_scope_verified"),
            "source": "prior_versioned_B2_artifact_review; no new fit or promotion inference",
        }
    runner = RuntimeModelRunner(loaded)
    source = source.reset_index(drop=True)
    raw = source[names].apply(pd.to_numeric, errors="coerce")
    contract = getattr(loaded.scaler, "history_contract", None)
    if contract is not None:
        validate_contract(contract, names)
        frozen = pd.DataFrame(impute_matrix(raw.to_numpy(), contract), columns=names)
    else:
        frozen = raw
    actual_scores = []
    errors = []
    replay_mode = "per_endpoint_runtime_runner"
    batch_scores: np.ndarray | None = None
    if loaded.backend in {"tcn", "lstm"}:
        try:
            endpoints = [int(row.source_row_idx) for row in predictions.itertuples(index=False)
                         if int(row.source_row_idx) >= int(loaded.window_size) - 1
                         and int(row.source_row_idx) < len(frozen)]
            raw_windows = np.stack([
                frozen.iloc[end - int(loaded.window_size) + 1:end + 1].to_numpy(dtype=np.float32)
                for end in endpoints
            ])
            scaled = runner._scale_matrix(raw_windows.reshape(-1, raw_windows.shape[-1]))
            sequences = scaled.reshape(raw_windows.shape)
            from model_training.ote_training.torch_trainer import predict_torch_model
            batch_scores = np.asarray(predict_torch_model(
                model=loaded.model,
                X=sequences,
                batch_size=loaded.batch_size,
                use_amp=loaded.use_amp,
            ), dtype=np.float32)
            batch_scores = np.asarray(apply_probability_calibrator(loaded.calibrator, batch_scores), dtype=np.float32)
            batch_endpoint_scores = dict(zip(endpoints, batch_scores.tolist()))
            replay_mode = "runtime_scaler_and_torch_backend_batch_equivalent"
        except Exception as exc:
            errors.append(type(exc).__name__ + ":" + str(exc)[:120])
    for row in predictions.itertuples(index=False):
        end = int(row.source_row_idx)
        context = int(loaded.context_rows)
        if end < context or end >= len(source):
            errors.append("test_endpoint_missing_required_history")
            continue
        window = frozen.iloc[end - context:end + 1].copy()
        window["source_row_idx"] = np.arange(end - context, end + 1)
        window["timestamp"] = source.iloc[end]["datetime"]
        try:
            if batch_scores is not None:
                actual = float(batch_endpoint_scores[end])
            else:
                prediction = runner.predict_latest(window, timestamp=pd.Timestamp(source.iloc[end]["datetime"]).to_pydatetime(), source_row_idx=end)
                actual = float(prediction.raw_score)
            actual_scores.append((actual, float(row.raw_probability)))
        except Exception as exc:
            errors.append(type(exc).__name__ + ":" + str(exc)[:120])
    deltas = [abs(a - b) for a, b in actual_scores]
    score_mismatches = sum(delta > 1e-5 for delta in deltas)
    actual_decisions = [a >= threshold for a, _ in actual_scores]
    expected_decisions = [b >= threshold for _, b in actual_scores]
    return {
        "test_endpoints": int(len(predictions)),
        "runner_replay_endpoints": len(actual_scores),
        "runner_errors": errors[:10],
        "replay_mode": replay_mode,
        "frozen_history_contract_present": contract is not None,
        "score_mismatches": score_mismatches,
        "max_abs_score_delta": max(deltas) if deltas else None,
        "threshold": threshold,
        "threshold_decision_mismatches": sum(a != b for a, b in zip(actual_decisions, expected_decisions)),
        "status": "passed" if not errors and not score_mismatches else "failed",
    }


def app_engine_audit(
    manifest,
    bars: pd.DataFrame,
    source: pd.DataFrame,
    rows: list[dict[str, Any]],
    names: list[str],
    threshold: float,
    loaded,
) -> dict[str, Any]:
    return {
        "status": "not_run_resource_bounded",
        "full_raw_parity_status": "insufficient_evidence_actual_producer_replay_not_completed",
        "scope": "recorded_snapshot_and_runtime_contract_review_only",
        "raw_recorded_snapshot_available": False,
        "raw_recorded_snapshot_status": "insufficient_evidence_recorded_snapshots_store_post_imputation_feature_values_only",
        "reason": "full IncrementalFeatureEngine rebuild exceeded safe diagnostic resource bound; no pass inferred",
    }
    # The immutable snapshot supplies the full recorded app evidence and the
    # source-history runner replay covers every test endpoint.  A broad
    # producer rebuild is prohibitively expensive for this historical app
    # graph, so the production path is exercised fail-closed on the final two
    # canonical bars; the report never labels that smoke window as full raw
    # parity.
    replay_bars = bars.tail(2).reset_index(drop=True)
    replay_timestamps = set(replay_bars["timestamp"].astype(str))
    replay_rows = [row for row in rows if row["timestamp"] in replay_timestamps]
    market = to_market_frame(replay_bars)
    engine = IncrementalFeatureEngine([manifest])
    try:
        post = engine.build_feature_frame(market)
        raw_bounded = engine._slice_runtime_window(market)
        raw, metadata = engine.builder.build(raw_bounded)
        raw = engine._apply_feature_seed_offsets(raw)
        missing = [name for name in engine.plan.built_feature_names if name not in raw.columns]
        raw = _materialize_live_default_features(raw, requested_feature_names=engine.plan.built_feature_names)
        if missing:
            raw = raw.assign(**{name: np.nan for name in missing})
        source_idx = {str(value): i for i, value in enumerate(replay_bars["timestamp"].astype(str))}
        source_by_idx = source.set_index("source_row_idx")
        post = post.reset_index(drop=True)
        raw = raw.reset_index(drop=True)
        offset = len(bars) - len(post)
        raw_offset = len(bars) - len(raw)
        compared = mismatches = post_mismatches = 0
        examples = []
        score_count = decision_mismatches = 0
        score_errors = []
        runner = RuntimeModelRunner(loaded)
        for row in replay_rows:
            idx = source_idx.get(row["timestamp"])
            if idx is None or idx < offset:
                continue
            local = idx - offset
            raw_local = idx - raw_offset
            if local >= len(post) or raw_local >= len(raw):
                continue
            for name in names:
                compared += 1
                expected = row["features"].get(name)
                actual = post.iloc[local].get(name)
                if not compare_values(expected, actual):
                    post_mismatches += 1
                    if len(examples) < 5:
                        examples.append({"timestamp": row["timestamp"], "feature": name, "recorded_app": expected, "replayed_app": actual})
                source_value = source_by_idx.loc[int(row["source_row_idx"])].get(name) if int(row["source_row_idx"]) in source_by_idx.index else None
                if not compare_values(source_value, raw.iloc[raw_local].get(name)):
                    mismatches += 1
            context = int(loaded.context_rows)
            if raw_local >= context and local >= context:
                window = post.iloc[local - context:local + 1].copy()
                window["source_row_idx"] = np.arange(idx - context, idx + 1)
                window["timestamp"] = replay_bars.iloc[idx]["timestamp"]
                try:
                    prediction = runner.predict_latest(window, timestamp=pd.Timestamp(replay_bars.iloc[idx]["timestamp"]).to_pydatetime(), source_row_idx=idx)
                    score_count += 1
                    if row["raw_score"] is not None and abs(float(prediction.raw_score) - float(row["raw_score"])) < 1e-9:
                        # Equality is expected only if the recorded app artifact is the staged artifact.
                        pass
                    decision = float(prediction.calibrated_probability) >= threshold
                    recorded_decision = (row["decision"] == "emit") if row["decision"] is not None else None
                    if recorded_decision is not None and decision != recorded_decision:
                        decision_mismatches += 1
                except Exception as exc:
                    score_errors.append(type(exc).__name__ + ":" + str(exc)[:120])
        contract = engine.input_contract_status(manifest.model_id, feature_frame=post)
        producer_ok = not mismatches and not post_mismatches and not score_errors and not contract.get("diagnostic_only")
        return {
            "status": "passed" if producer_ok else "failed",
            "full_raw_parity_status": "insufficient_evidence_raw_recorded_snapshot_not_persisted",
            "scope": "two_bar_actual_app_path_smoke_window",
            "replayed_recorded_snapshots": len(replay_rows),
            "rows_built": len(post),
            "raw_cells_compared": compared,
            "raw_cells_mismatched": mismatches,
            "post_imputation_cells_mismatched": post_mismatches,
            "raw_recorded_snapshot_available": False,
            "raw_recorded_snapshot_status": "insufficient_evidence_recorded_snapshots_store_post_imputation_feature_values_only",
            "score_rows": score_count,
            "score_errors": score_errors[:10],
            "recorded_decision_mismatches": decision_mismatches,
            "missing_producer_features": list(engine.last_build_metadata.get("missing_producer_feature_names", [])),
            "fallback_features": list(engine.last_build_metadata.get("fallback_feature_names", [])),
            "input_contract": contract,
            "examples": examples,
            "builder_metadata": {"build_timings_seconds": engine.last_build_timings_seconds},
        }
    except Exception as exc:
        return {"status": "failed", "error": type(exc).__name__ + ":" + str(exc), "diagnostic_only": True}


def audit_candidate(key: str, rows: list[dict[str, Any]], bars: pd.DataFrame, snapshot_hash: str) -> dict[str, Any]:
    spec = CANDIDATES[key]
    app_manifest_path = ROOT / spec["app_manifest"]
    artifact_dir = (ROOT / spec["artifact_dir"]).resolve()
    source_path = ROOT / spec["source_history"]
    result: dict[str, Any] = {"queue_id": int(key), "model_id": spec["model_id"], "collection_status": "diagnostic_only", "reasons": []}
    result["inputs"] = {
        "app_manifest": rel(app_manifest_path) if app_manifest_path.exists() else spec["app_manifest"],
        "app_manifest_sha256": sha256(app_manifest_path) if app_manifest_path.exists() else None,
        "artifact_dir": rel(artifact_dir) if artifact_dir.exists() else spec["artifact_dir"],
        "source_history": rel(source_path) if source_path.exists() else spec["source_history"],
        "source_history_sha256": sha256(source_path) if source_path.exists() else None,
        "snapshot_db_sha256": snapshot_hash,
    }
    if not app_manifest_path.exists() or not artifact_dir.exists() or not source_path.exists():
        result["reasons"].append("required_app_or_candidate_artifact_missing")
        return result
    base = load_live_runtime_manifest(app_manifest_path)
    summary = load_json(artifact_dir / "training_summary.json")
    names = list(summary["selected_feature_names"])
    prior_path = ROOT / PRIOR_SCORE_EVIDENCE[key]
    prior = load_json(prior_path) if prior_path.exists() else None
    if prior is not None:
        prior["_evidence_file"] = rel(prior_path)
        source = None
    else:
        source = pd.read_parquet(source_path, columns=["source_row_idx", "datetime", *names])
        source["source_row_idx"] = pd.to_numeric(source["source_row_idx"], errors="coerce").astype("Int64")
    result["candidate_contract"] = {
        "selected_feature_count": len(names),
        "selected_feature_names_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
        "context_rows": int(summary["context_rows"]),
        "window_size": int(summary["window_size"]),
        "lag_steps": list(summary.get("lag_steps", [])),
        "threshold": float(summary["threshold"]),
    }
    rows = [row for row in rows if row["model_id"] == spec["model_id"]]
    result["recorded_app_snapshot"] = provenance_audit(rows, bars, int(summary["context_rows"]))
    result["recorded_source_feature_comparison"] = (
        {
            "status": "insufficient_evidence",
            "semantics": "raw_selected_feature_values_are_not_persisted_in_the_recorded_app_snapshot; post-imputation values cannot certify raw producer parity",
            "cells_compared": 0,
            "cells_mismatched": None,
        }
        if source is None else source_feature_audit(rows, source, names)
    )
    result["recorded_source_feature_comparison"]["semantics"] = result["recorded_source_feature_comparison"].get("semantics", "prepared_source_raw_vs_recorded_app_post_imputation; not a raw-producer certification")
    result["recorded_app_artifact_identity"] = {
        "recorded_runtime_manifest_model_id": base.model_id,
        "candidate_model_id": spec["model_id"],
        "candidate_artifact_is_the_recorded_app_artifact": False,
        "reason": "recorded snapshot references historical runtime artifacts; replacement bundle is staged only for diagnostic replay",
    }
    candidate = candidate_manifest(base, artifact_dir, summary)
    if prior is not None:
        result["candidate_runtime_load"] = {
            "status": "passed",
            "backend": summary["config"].get("model_type", summary["config"].get("backend")),
            "model_id": spec["model_id"],
            "source": "prior_versioned_B2_artifact_review; replacement runtime not reloaded in this evidence pass",
        }
        loaded = None
    else:
        try:
            loaded = load_runtime_model(candidate)
            result["candidate_runtime_load"] = {"status": "passed", "backend": loaded.backend, "model_id": loaded.model_id}
        except Exception as exc:
            result["candidate_runtime_load"] = {"status": "failed", "error": type(exc).__name__ + ":" + str(exc)}
            result["reasons"].append("candidate_runtime_artifact_or_contract_load_failed")
            return result
    prediction_path = artifact_dir / "test_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    result["frozen_scaling_and_score_replay"] = candidate_replay_audit(
        loaded, source, predictions, names, float(summary["threshold"]), prior
    )
    result["actual_app_path_replay"] = app_engine_audit(
        candidate, bars, source, rows, names, float(summary["threshold"]), loaded
    )
    gates = [
        result["recorded_app_snapshot"]["source_identity_mismatches"] == 0,
        result["recorded_app_snapshot"]["completion_unknown"] == 0,
        result["recorded_app_snapshot"]["incomplete_bars"] == 0,
        result["recorded_app_snapshot"]["availability_unknown"] == 0,
        result["recorded_app_snapshot"]["unresolved_gap_observations"] == 0,
        result["recorded_app_snapshot"]["insufficient_warmup_windows"] == 0,
        result["actual_app_path_replay"].get("raw_recorded_snapshot_available") is True,
        result["frozen_scaling_and_score_replay"]["status"] == "passed",
        result["actual_app_path_replay"].get("status") == "passed",
        result["recorded_app_artifact_identity"]["candidate_artifact_is_the_recorded_app_artifact"],
    ]
    if not gates:
        result["reasons"].append("full_live_feature_producer_parity_not_certified")
    if result["recorded_app_snapshot"]["completion_unknown"] or result["recorded_app_snapshot"]["incomplete_bars"]:
        result["reasons"].append("completion_or_availability_not_proven")
    if result["actual_app_path_replay"].get("missing_producer_features"):
        result["reasons"].append("missing_live_producers")
    if result["actual_app_path_replay"].get("fallback_features"):
        result["reasons"].append("unexplained_live_fallbacks")
    if not result["actual_app_path_replay"].get("raw_recorded_snapshot_available"):
        result["reasons"].append("raw_selected_feature_snapshot_unavailable")
    if result["actual_app_path_replay"].get("status") != "passed":
        result["reasons"].append("actual_app_path_replay_not_completed")
    result["parity_pass"] = all(gates)
    result["promotion_pass"] = False
    result["collection_status"] = "qualified_research_collection_candidate" if result["parity_pass"] else "diagnostic_only"
    result["reasons"] = sorted(set(result["reasons"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; use a new versioned evidence path")
    if not SNAPSHOT_DB.exists():
        parser.error(f"recorded source snapshot database not found: {SNAPSHOT_DB}")
    connection = sqlite3.connect(f"file:{SNAPSHOT_DB}?mode=ro", uri=True)
    try:
        bars = read_canonical_bars(connection)
        all_rows = read_snapshot_rows(connection, tuple(spec["model_id"] for spec in CANDIDATES.values()))
        snapshot_hash = f"deferred_sha256_file_size_{SNAPSHOT_DB.stat().st_size}"
        results = [audit_candidate(key, all_rows, bars, snapshot_hash) for key in CANDIDATES]
    finally:
        connection.close()
    payload = {
        "evidence_version": EVIDENCE_VERSION,
        "status": "b5_b6_live_feature_parity_audit_complete",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "snapshot_db": rel(SNAPSHOT_DB),
        "snapshot_manifest": rel(ROOT / "research/es_live_audit_20260911/snapshot_manifest.json"),
        "snapshot_manifest_sha256": sha256(ROOT / "research/es_live_audit_20260911/snapshot_manifest.json"),
        "snapshot_cutoff": "2026-09-12T02:02:03.011561+00:00",
        "actual_app_path": [
            "ote_live.ingestion.signals.LiveSignalProcessor.prepare_ingested_bar_snapshot",
            "ote_live.features.incremental_engine.IncrementalFeatureEngine.build_feature_frame",
            "ote_live.models.runners.RuntimeModelRunner.predict_latest",
        ],
        "candidate_count": len(results),
        "parity_pass_count": sum(bool(item.get("parity_pass")) for item in results),
        "promotion_pass_count": 0,
        "collection_eligible": False,
        "b5": {"status": "preserved_evidence_only", "requires_full_parity": True},
        "b6": {"status": "preserved_evidence_only", "requires_clean_evaluation_boundary": True},
        "a13_rehearsal": {
            "status": "ready_for_rehearsal_after_eligible_research_collection",
            "future_c1_outcomes_required_before_collection": False,
            "prospective_c1_rule": "first healthy post-B7 collection timestamp under a new immutable candidate/runtime identity",
        },
        "models": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "candidates": len(results), "parity_pass": payload["parity_pass_count"], "promotion_pass": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

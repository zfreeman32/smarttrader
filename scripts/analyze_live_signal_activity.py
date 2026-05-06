from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "live_market_data.sqlite3"
DEFAULT_LONG_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"
DEFAULT_SHORT_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_short.json"
REGIME_REQUIRED_COLUMNS = ("ema_alignment", "atr_14", "range_shock_20")


@dataclass(frozen=True)
class RuntimeModelSpec:
    model_id: str
    direction: str
    backend: str
    threshold: float | None
    role: str
    status: str
    registry_metrics: dict[str, Any]
    test_predictions_path: Path | None


@dataclass(frozen=True)
class DirectionRuntimeSpec:
    direction: str
    primary_model_id: str | None
    models_by_id: dict[str, RuntimeModelSpec]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze live signal activity, confidence distributions, and threshold behavior.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to the live SQLite database.")
    parser.add_argument(
        "--long-manifest",
        type=Path,
        default=DEFAULT_LONG_MANIFEST_PATH,
        help="Path to the long-direction runtime manifest bundle.",
    )
    parser.add_argument(
        "--short-manifest",
        type=Path,
        default=DEFAULT_SHORT_MANIFEST_PATH,
        help="Path to the short-direction runtime manifest bundle.",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_direction_manifest(path: Path) -> DirectionRuntimeSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models_by_id: dict[str, RuntimeModelSpec] = {}
    for model_payload in payload.get("models", []):
        policy = model_payload.get("live_policy", {}) or {}
        thresholds = policy.get("thresholds", {}) or {}
        test_predictions_path = resolve_repo_path(
            (model_payload.get("artifact_references", {}) or {}).get("test_predictions_file")
        )
        spec = RuntimeModelSpec(
            model_id=str(model_payload["model_id"]),
            direction=str(model_payload["direction"]),
            backend=str(model_payload["backend"]),
            threshold=(
                float(thresholds["global_threshold"])
                if thresholds.get("global_threshold") is not None
                else None
            ),
            role=str(model_payload.get("role", "unknown")),
            status=str(model_payload.get("status", "unknown")),
            registry_metrics=dict(model_payload.get("registry_metrics", {}) or {}),
            test_predictions_path=test_predictions_path,
        )
        models_by_id[spec.model_id] = spec
    recommendations = payload.get("recommendations", {}) or {}
    return DirectionRuntimeSpec(
        direction=str(payload["direction"]),
        primary_model_id=recommendations.get("recommended_primary_model_id"),
        models_by_id=models_by_id,
    )


def connect_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def fetch_rows(connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return connection.execute(query, params).fetchall()


def fetch_scalar(connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> Any:
    row = connection.execute(query, params).fetchone()
    return row[0] if row is not None else None


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def distinct_prediction_dates(connection: sqlite3.Connection, model_id: str) -> list[str]:
    rows = fetch_rows(
        connection,
        """
        SELECT DISTINCT substr(timestamp_utc, 1, 10) AS day_utc
        FROM model_predictions
        WHERE model_id = ?
        ORDER BY day_utc
        """,
        (model_id,),
    )
    return [str(row["day_utc"]) for row in rows]


def count_clusters(timestamps: list[datetime], gap: timedelta) -> int:
    if not timestamps:
        return 0
    clusters = 1
    previous = timestamps[0]
    for current in timestamps[1:]:
        if current - previous > gap:
            clusters += 1
        previous = current
    return clusters


def summarize_live_window(connection: sqlite3.Connection) -> None:
    print("=== Live Coverage ===")
    rows = fetch_rows(
        connection,
        """
        SELECT asset, timeframe, MIN(timestamp_utc) AS min_ts, MAX(timestamp_utc) AS max_ts, COUNT(*) AS bar_count
        FROM canonical_bars
        GROUP BY asset, timeframe
        ORDER BY asset, timeframe
        """,
    )
    for row in rows:
        print(
            f"{row['asset']} {row['timeframe']}: {row['bar_count']} bars "
            f"from {row['min_ts']} to {row['max_ts']}"
        )

    latest_bar = fetch_scalar(connection, "SELECT MAX(timestamp_utc) FROM canonical_bars")
    latest_prediction = fetch_scalar(connection, "SELECT MAX(timestamp_utc) FROM model_predictions")
    latest_signal = fetch_scalar(connection, "SELECT MAX(timestamp_utc) FROM signal_decisions")
    unresolved_gaps = fetch_scalar(
        connection,
        "SELECT COUNT(*) FROM ingestion_gaps WHERE resolved_at_utc IS NULL",
    )
    print(f"Latest bar: {latest_bar}")
    print(f"Latest prediction: {latest_prediction}")
    print(f"Latest signal: {latest_signal}")
    print(f"Unresolved gaps: {unresolved_gaps}")
    print()


def summarize_decision_mix(connection: sqlite3.Connection) -> None:
    print("=== Decision Mix ===")
    rows = fetch_rows(
        connection,
        """
        SELECT model_id, decision, COUNT(*) AS decision_count
        FROM signal_decisions
        GROUP BY model_id, decision
        ORDER BY model_id, decision_count DESC
        """,
    )
    for row in rows:
        print(f"{row['model_id']}: {row['decision']}={row['decision_count']}")
    print()


def summarize_feature_coverage(connection: sqlite3.Connection) -> None:
    print("=== Feature Coverage ===")
    rows = fetch_rows(
        connection,
        """
        SELECT
            json_extract(metadata_json, '$.model_id') AS model_id,
            COUNT(*) AS snapshot_count,
            MIN(valid_feature_count) AS min_valid,
            AVG(valid_feature_count) AS avg_valid,
            MAX(valid_feature_count) AS max_valid,
            MAX(json_extract(metadata_json, '$.selected_feature_count')) AS selected_count
        FROM feature_snapshots
        GROUP BY json_extract(metadata_json, '$.model_id')
        ORDER BY model_id
        """,
    )
    for row in rows:
        print(
            f"{row['model_id']}: {row['snapshot_count']} snapshots, "
            f"valid_features min/avg/max = {row['min_valid']}/{format_float(row['avg_valid'], 2)}/{row['max_valid']}, "
            f"selected_features = {row['selected_count']}"
        )
    print()


def summarize_shadow_models(
    connection: sqlite3.Connection,
    direction_specs: list[DirectionRuntimeSpec],
) -> None:
    primary_ids = {spec.primary_model_id for spec in direction_specs if spec.primary_model_id}
    print("=== Shadow Models With Live Threshold Passes ===")
    rows = fetch_rows(
        connection,
        """
        SELECT
            mp.model_id,
            COUNT(*) AS total_predictions,
            SUM(CASE WHEN mp.calibrated_probability >= mp.threshold_applied THEN 1 ELSE 0 END) AS pass_count,
            MAX(mp.calibrated_probability) AS max_probability,
            AVG(mp.calibrated_probability) AS avg_probability
        FROM model_predictions AS mp
        GROUP BY mp.model_id
        ORDER BY pass_count DESC, max_probability DESC
        """,
    )
    for row in rows:
        model_id = str(row["model_id"])
        if model_id in primary_ids:
            continue
        print(
            f"{model_id}: pass_count={row['pass_count']}, total={row['total_predictions']}, "
            f"avg_prob={format_float(row['avg_probability'])}, max_prob={format_float(row['max_probability'])}"
        )
    print()


def load_test_predictions(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, usecols=["calibrated_probability"])


def print_primary_model_report(connection: sqlite3.Connection, spec: RuntimeModelSpec) -> None:
    rows = fetch_rows(
        connection,
        """
        SELECT timestamp_utc, raw_score, calibrated_probability, threshold_applied, regime, metadata_json
        FROM model_predictions
        WHERE model_id = ?
        ORDER BY timestamp_utc ASC
        """,
        (spec.model_id,),
    )
    if not rows:
        print(f"=== {spec.model_id} ===")
        print("No live predictions found.")
        print()
        return

    probabilities = pd.Series([float(row["calibrated_probability"]) for row in rows], dtype="float64")
    raw_scores = pd.Series(
        [float(row["raw_score"]) for row in rows if row["raw_score"] is not None],
        dtype="float64",
    )
    thresholds = pd.Series(
        [float(row["threshold_applied"]) for row in rows if row["threshold_applied"] is not None],
        dtype="float64",
    )
    pass_mask = probabilities >= thresholds.iloc[0]
    active_dates = distinct_prediction_dates(connection, spec.model_id)

    print(f"=== Primary: {spec.model_id} ({spec.direction}, {spec.backend}) ===")
    print(
        f"threshold={format_float(spec.threshold, 2)} | "
        f"test_ap={format_float(_coerce_metric(spec.registry_metrics.get('test_ap')), 4)} | "
        f"test_event_f0.5={format_float(_coerce_metric(spec.registry_metrics.get('test_event_f05')), 4)}"
    )
    print(
        f"live_predictions={len(rows)} across {len(active_dates)} active UTC dates | "
        f"live_pass_rate={format_pct(float(pass_mask.mean()))} | "
        f"avg_prob={format_float(float(probabilities.mean()))} | "
        f"max_prob={format_float(float(probabilities.max()))}"
    )
    print(
        "probability quantiles: "
        f"p50={format_float(float(probabilities.quantile(0.50)))} "
        f"p90={format_float(float(probabilities.quantile(0.90)))} "
        f"p95={format_float(float(probabilities.quantile(0.95)))} "
        f"p99={format_float(float(probabilities.quantile(0.99)))}"
    )
    if not raw_scores.empty:
        print(
            "raw-score quantiles: "
            f"p50={format_float(float(raw_scores.quantile(0.50)))} "
            f"p90={format_float(float(raw_scores.quantile(0.90)))} "
            f"p95={format_float(float(raw_scores.quantile(0.95)))} "
            f"p99={format_float(float(raw_scores.quantile(0.99)))}"
        )

    live_timestamps = [
        datetime.fromisoformat(str(row["timestamp_utc"]))
        for row in rows
    ]
    threshold_grid = _build_threshold_grid(spec.threshold)
    print("threshold sweep:")
    for threshold in threshold_grid:
        hit_timestamps = [
            timestamp
            for timestamp, probability in zip(live_timestamps, probabilities.tolist())
            if probability >= threshold
        ]
        cluster_count = count_clusters(hit_timestamps, timedelta(minutes=5))
        active_date_count = max(len(active_dates), 1)
        print(
            f"  th={threshold:.2f}: rows={len(hit_timestamps)} "
            f"clusters={cluster_count} "
            f"rows/day={len(hit_timestamps) / active_date_count:.2f} "
            f"clusters/day={cluster_count / active_date_count:.2f}"
        )

    test_predictions = load_test_predictions(spec.test_predictions_path)
    if test_predictions is not None and spec.threshold is not None:
        test_probs = test_predictions["calibrated_probability"]
        print("live vs offline confidence buckets:")
        for label, lower, upper in (
            ("<0.05", None, 0.05),
            ("0.05-0.10", 0.05, 0.10),
            ("0.10-threshold", 0.10, spec.threshold),
            (">=threshold", spec.threshold, None),
        ):
            live_rate = _bucket_rate(probabilities, lower, upper)
            test_rate = _bucket_rate(test_probs, lower, upper)
            print(
                f"  {label}: live={format_pct(live_rate)} | test={format_pct(test_rate)}"
            )

    latest_metadata = json.loads(str(rows[-1]["metadata_json"])) if rows[-1]["metadata_json"] else {}
    policy_context = latest_metadata.get("policy_context") or {}
    missing_regime_columns = [
        column
        for column in REGIME_REQUIRED_COLUMNS
        if column not in policy_context
    ]
    if missing_regime_columns:
        print(
            "regime diagnostic: missing policy_context columns required by the regime labeler -> "
            + ", ".join(missing_regime_columns)
        )
    else:
        print("regime diagnostic: required regime columns are present in the latest policy_context.")

    top_rows = fetch_rows(
        connection,
        """
        SELECT timestamp_utc, calibrated_probability, threshold_applied, raw_score
        FROM model_predictions
        WHERE model_id = ?
        ORDER BY calibrated_probability DESC
        LIMIT 5
        """,
        (spec.model_id,),
    )
    print("top live confidence timestamps:")
    for row in top_rows:
        print(
            f"  {row['timestamp_utc']} | prob={format_float(float(row['calibrated_probability']))} "
            f"| threshold={format_float(float(row['threshold_applied']))} "
            f"| raw={format_float(float(row['raw_score'])) if row['raw_score'] is not None else 'n/a'}"
        )
    print()


def summarize_health(connection: sqlite3.Connection) -> None:
    print("=== Recent Health Events ===")
    rows = fetch_rows(
        connection,
        """
        SELECT event_timestamp_utc, component, severity, event_type, payload_json
        FROM health_events
        ORDER BY event_timestamp_utc DESC, id DESC
        LIMIT 8
        """,
    )
    for row in rows:
        print(
            f"{row['event_timestamp_utc']} | {row['severity']} | {row['component']} | {row['event_type']}"
        )
    print()


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_threshold_grid(primary_threshold: float | None) -> list[float]:
    raw_grid = [0.75, 0.71, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05]
    if primary_threshold is not None:
        raw_grid.append(float(primary_threshold))
    unique = sorted({round(value, 2) for value in raw_grid}, reverse=True)
    return unique


def _bucket_rate(series: pd.Series, lower: float | None, upper: float | None) -> float:
    mask = pd.Series(True, index=series.index)
    if lower is not None:
        mask &= series >= lower
    if upper is not None:
        mask &= series < upper
    return float(mask.mean())


def main() -> None:
    args = parse_args()
    connection = connect_db(args.db)
    try:
        long_spec = load_direction_manifest(args.long_manifest)
        short_spec = load_direction_manifest(args.short_manifest)

        summarize_live_window(connection)
        summarize_decision_mix(connection)
        summarize_feature_coverage(connection)
        summarize_shadow_models(connection, [long_spec, short_spec])

        for direction_spec in (long_spec, short_spec):
            if direction_spec.primary_model_id is None:
                continue
            primary = direction_spec.models_by_id.get(direction_spec.primary_model_id)
            if primary is None:
                continue
            print_primary_model_report(connection, primary)

        summarize_health(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a recency-weighted FRVP prepared root for a single target by "
            "reweighting only the train/val sample weights from a clean base prepared root."
        )
    )
    parser.add_argument(
        "--base-prepared-root",
        type=Path,
        required=True,
        help="Prepared root with the clean split CSVs and metadata to copy from.",
    )
    parser.add_argument(
        "--output-prepared-root",
        type=Path,
        required=True,
        help="Destination prepared root to create.",
    )
    parser.add_argument(
        "--target",
        default="long_frvp_reversal",
        help="Prepared target folder name to materialize.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=None,
        help="Optional override for the phase-02 source dataset. Defaults to the path recorded in summary/report lineage.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        required=True,
        help="Exponential half-life in days for the recency decay.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.20,
        help="Lower floor applied to the recency factor.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output root.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_source_dataset(
    *,
    base_prepared_root: Path,
    target: str,
    explicit_source_dataset: Path | None,
) -> Path:
    if explicit_source_dataset is not None:
        return explicit_source_dataset.resolve()

    summary_path = base_prepared_root / "summary.json"
    report_path = base_prepared_root / target / "report.json"
    summary_payload = load_json(summary_path) if summary_path.exists() else {}
    report_payload = load_json(report_path) if report_path.exists() else {}

    candidates = [
        summary_payload.get("source_lineage", {}).get("feature_csv"),
        summary_payload.get("input_file"),
        report_payload.get("source_lineage", {}).get("feature_csv"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = (REPO_ROOT / str(candidate)).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Could not resolve the source dataset from the prepared-root metadata. "
        "Pass --source-dataset explicitly."
    )


def load_source_timestamps(source_dataset: Path) -> pd.Series:
    frame = pd.read_csv(source_dataset, usecols=["datetime"])
    timestamps = pd.to_datetime(frame["datetime"], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError(f"Found null or invalid datetimes in {source_dataset}.")
    return timestamps


def copy_base_layout(
    *,
    base_prepared_root: Path,
    output_prepared_root: Path,
    target: str,
) -> Path:
    target_root = base_prepared_root / target
    if not target_root.exists():
        raise FileNotFoundError(f"Base target folder does not exist: {target_root}")

    output_prepared_root.mkdir(parents=True, exist_ok=True)

    summary_path = base_prepared_root / "summary.json"
    if summary_path.exists():
        shutil.copy2(summary_path, output_prepared_root / "summary.json")

    output_target_root = output_prepared_root / target
    output_target_root.mkdir(parents=True, exist_ok=True)
    for child in target_root.iterdir():
        if child.is_file():
            shutil.copy2(child, output_target_root / child.name)
    return output_target_root


def recency_factor(age_days: np.ndarray, *, half_life_days: float, floor: float) -> np.ndarray:
    raw_factor = np.power(0.5, age_days / half_life_days)
    return np.maximum(floor, raw_factor).astype(np.float64, copy=False)


def reweight_split(
    *,
    split_path: Path,
    timestamps: pd.Series,
    latest_ts: pd.Timestamp,
    half_life_days: float,
    floor: float,
) -> dict[str, Any]:
    frame = pd.read_csv(split_path)
    if "source_row_idx" not in frame.columns:
        raise ValueError(f"{split_path} is missing source_row_idx.")
    if "sample_weight" not in frame.columns:
        raise ValueError(f"{split_path} is missing sample_weight.")

    source_row_idx = pd.to_numeric(frame["source_row_idx"], errors="coerce").astype("Int64")
    if source_row_idx.isna().any():
        raise ValueError(f"{split_path} contains invalid source_row_idx values.")

    source_rows = source_row_idx.to_numpy(dtype=np.int64, copy=False)
    if source_rows.size == 0:
        raise ValueError(f"{split_path} contains no rows.")
    if source_rows.min() < 0 or source_rows.max() >= len(timestamps):
        raise IndexError(
            f"{split_path} contains out-of-range source_row_idx values: "
            f"min={source_rows.min()} max={source_rows.max()} source_rows={len(timestamps)}"
        )

    split_ts = timestamps.iloc[source_rows].reset_index(drop=True)
    age_days = ((latest_ts - split_ts).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float64, copy=False)
    factors = recency_factor(age_days, half_life_days=half_life_days, floor=floor)

    weight_before = pd.to_numeric(frame["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    frame["sample_weight"] = weight_before * factors
    frame.to_csv(split_path, index=False)

    return {
        "rows": int(len(frame)),
        "source_min_ts_utc": split_ts.min().isoformat(),
        "source_max_ts_utc": split_ts.max().isoformat(),
        "factor_min": float(np.min(factors)),
        "factor_p25": float(np.percentile(factors, 25)),
        "factor_median": float(np.percentile(factors, 50)),
        "factor_p75": float(np.percentile(factors, 75)),
        "factor_max": float(np.max(factors)),
        "sample_weight_mean_before": float(np.mean(weight_before)),
        "sample_weight_mean_after": float(np.mean(frame["sample_weight"].to_numpy(dtype=np.float64))),
    }


def main() -> None:
    args = parse_args()
    base_prepared_root = args.base_prepared_root.resolve()
    output_prepared_root = args.output_prepared_root.resolve()
    target = str(args.target).strip()

    if args.half_life_days <= 0:
        raise ValueError("--half-life-days must be positive.")
    if not 0.0 <= args.floor <= 1.0:
        raise ValueError("--floor must be between 0 and 1.")

    if output_prepared_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output prepared root already exists: {output_prepared_root}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_prepared_root)

    output_target_root = copy_base_layout(
        base_prepared_root=base_prepared_root,
        output_prepared_root=output_prepared_root,
        target=target,
    )
    source_dataset = resolve_source_dataset(
        base_prepared_root=base_prepared_root,
        target=target,
        explicit_source_dataset=args.source_dataset,
    )
    timestamps = load_source_timestamps(source_dataset)
    latest_ts = timestamps.max()

    split_summaries = {}
    for split_name in ("train", "val"):
        split_summaries[split_name] = reweight_split(
            split_path=output_target_root / f"{split_name}.csv",
            timestamps=timestamps,
            latest_ts=latest_ts,
            half_life_days=float(args.half_life_days),
            floor=float(args.floor),
        )

    # Keep the test split untouched so full-span and recent-regime evaluations stay comparable.
    weighting_summary = {
        "target": target,
        "half_life_days": float(args.half_life_days),
        "floor": float(args.floor),
        "latest_ts_utc": latest_ts.isoformat(),
        "base_prepared_root": str(base_prepared_root),
        "output_prepared_root": str(output_prepared_root),
        "source_dataset": str(source_dataset),
        "splits": split_summaries,
    }
    write_json(output_target_root / "recency_weighting_summary.json", weighting_summary)
    print(json.dumps(weighting_summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_training.ote_training.feature_ranking import (
    load_feature_allowlist,
    merge_feature_rankings,
)


DEFAULT_PREPARED_ROOT = Path("data/prepared/eurusd_5min_ote_full")
DEFAULT_TARGETS = ("long_ote", "short_ote")


def print_step(message: str) -> None:
    print(f"[merge-ote-rankings] {message}")


def validate_targets(targets: Iterable[str]) -> list[str]:
    allowed = set(DEFAULT_TARGETS)
    deduped = list(dict.fromkeys(targets))
    invalid = [target for target in deduped if target not in allowed]
    if invalid:
        supported = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported target(s): {invalid}. Supported values: {supported}")
    return deduped


def load_base_ranking(prepared_dir: Path) -> pd.DataFrame:
    ranking_path = prepared_dir / "feature_importance.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing base feature importance file: {ranking_path}")
    return pd.read_csv(ranking_path)


def load_shap_ranking(shap_dir: Path) -> pd.DataFrame:
    shap_path = shap_dir / "shap_feature_stats.csv"
    if not shap_path.exists():
        raise FileNotFoundError(f"Missing SHAP feature stats file: {shap_path}")
    return pd.read_csv(shap_path)


def summarize_rank_changes(merged: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    if "feature" not in merged.columns or "merged_rank" not in merged.columns:
        return []
    if "composite_score" not in merged.columns:
        return []

    base_rank = merged["composite_score"].rank(method="average", ascending=False)
    summary = merged.loc[:, ["feature", "merged_rank"]].copy()
    summary["base_rank"] = base_rank
    summary["rank_change"] = summary["base_rank"] - summary["merged_rank"]
    summary = summary.sort_values(["rank_change", "merged_rank"], ascending=[False, True], kind="stable")
    payload: list[dict[str, object]] = []
    for _, row in summary.head(limit).iterrows():
        payload.append(
            {
                "feature": str(row["feature"]),
                "base_rank": int(round(float(row["base_rank"]))),
                "merged_rank": int(row["merged_rank"]),
                "rank_change": float(row["rank_change"]),
            }
        )
    return payload


def write_target_ranking(
    prepared_root: Path,
    target: str,
    *,
    shap_root: Path,
    output_filename: str,
    metadata_filename: str,
    base_weight: float,
    shap_weight: float,
    shap_positive_weight: float,
) -> Path:
    prepared_dir = prepared_root / target
    shap_dir = shap_root / target

    allowed_features = load_feature_allowlist(prepared_dir)
    base_ranking = load_base_ranking(prepared_dir)
    shap_ranking = load_shap_ranking(shap_dir)

    merged = merge_feature_rankings(
        base_ranking=base_ranking,
        shap_ranking=shap_ranking,
        allowed_features=allowed_features,
        base_weight=base_weight,
        shap_weight=shap_weight,
        shap_positive_weight=shap_positive_weight,
    )

    output_path = prepared_dir / output_filename
    merged.to_csv(output_path, index=False)

    metadata_path = prepared_dir / metadata_filename
    metadata = {
        "target": target,
        "prepared_root": str(prepared_root),
        "prepared_dir": str(prepared_dir),
        "base_ranking_path": str(prepared_dir / "feature_importance.csv"),
        "shap_ranking_path": str(shap_dir / "shap_feature_stats.csv"),
        "output_path": str(output_path),
        "weights": {
            "base_weight": base_weight,
            "shap_weight": shap_weight,
            "shap_positive_weight": shap_positive_weight,
        },
        "feature_counts": {
            "allowed_features": len(allowed_features),
            "base_features": int(base_ranking["feature"].nunique()) if "feature" in base_ranking.columns else 0,
            "shap_features": int(shap_ranking["feature"].nunique()) if "feature" in shap_ranking.columns else 0,
            "merged_features": int(len(merged)),
        },
        "top_rank_gains": summarize_rank_changes(merged, limit=10),
        "top_features_after_merge": merged["feature"].head(20).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend prepared-data feature importance with recent-window SHAP rankings for OTE targets. "
            "The OTE trainer will automatically prefer feature_importance_merged.csv when it exists."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=DEFAULT_PREPARED_ROOT,
        help="Prepared dataset root containing long_ote/short_ote directories.",
    )
    parser.add_argument(
        "--shap-root",
        type=Path,
        default=None,
        help="Root directory containing SHAP outputs by target. Defaults to <prepared-root>/shap_outputs.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Targets to merge. Supported values: long_ote short_ote",
    )
    parser.add_argument(
        "--output-filename",
        default="feature_importance_merged.csv",
        help="Filename written into each prepared target directory.",
    )
    parser.add_argument(
        "--metadata-filename",
        default="feature_importance_merged.json",
        help="Summary metadata filename written into each prepared target directory.",
    )
    parser.add_argument(
        "--base-weight",
        type=float,
        default=0.20,
        help="Weight for the original prepared-data composite ranking.",
    )
    parser.add_argument(
        "--shap-weight",
        type=float,
        default=0.55,
        help="Weight for mean_abs_shap_all.",
    )
    parser.add_argument(
        "--shap-positive-weight",
        type=float,
        default=0.25,
        help="Weight for mean_abs_shap_positive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = validate_targets(args.targets)
    prepared_root = args.prepared_root
    shap_root = args.shap_root or (prepared_root / "shap_outputs")

    print_step(f"prepared_root={prepared_root}")
    print_step(f"shap_root={shap_root}")

    for target in targets:
        output_path = write_target_ranking(
            prepared_root=prepared_root,
            target=target,
            shap_root=shap_root,
            output_filename=args.output_filename,
            metadata_filename=args.metadata_filename,
            base_weight=float(args.base_weight),
            shap_weight=float(args.shap_weight),
            shap_positive_weight=float(args.shap_positive_weight),
        )
        print_step(f"{target}: wrote {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_primary_model_registry_current.json"
DEFAULT_OUTPUT_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_primary_model_registry_generated.json"
DEFAULT_MODEL_ROOTS = (
    REPO_ROOT / "models" / "frvp_es_primary_xgb_v1",
    REPO_ROOT / "models" / "frvp_es_primary_tcn_reversal_v1",
    REPO_ROOT / "models" / "frvp_es_primary_tcn_continuation_v1",
)
DEFAULT_PROMOTION_RULES = {
    "min_cv_splits": 3,
    "min_test_event_f05": 0.65,
    "require_regime_robustness": True,
    "require_post_cost_profitability": True,
    "require_paper_trading_confirmation": True,
}


def build_registry_payload(
    *,
    model_roots: Iterable[Path],
    source_registry_path: Path | None,
) -> Dict[str, Any]:
    source_payload = _load_json(source_registry_path) if source_registry_path and source_registry_path.exists() else {}
    source_models = {
        str(record["model_id"]): record
        for record in source_payload.get("models", [])
        if isinstance(record, dict) and "model_id" in record
    }
    promotion_rules = dict(source_payload.get("promotion_rules") or DEFAULT_PROMOTION_RULES)

    generated_models: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()
    for training_summary_path in _discover_training_summaries(model_roots):
        training_summary = _load_json(training_summary_path)
        artifact_dir = training_summary_path.parent
        model_record = _build_model_record(
            training_summary=training_summary,
            artifact_dir=artifact_dir,
            existing_record=source_models.get(
                _infer_model_id(
                    str(training_summary.get("target") or ""),
                    _resolve_backend_alias(training_summary),
                )
            ),
        )
        if model_record is None:
            continue
        if model_record["model_id"] in seen_model_ids:
            raise ValueError(f"Duplicate generated FRVP model_id: {model_record['model_id']}")
        seen_model_ids.add(model_record["model_id"])
        generated_models.append(model_record)

    generated_models.sort(key=lambda record: str(record["model_id"]))
    return {
        "promotion_rules": promotion_rules,
        "models": generated_models,
    }


def _discover_training_summaries(model_roots: Iterable[Path]) -> List[Path]:
    discovered: list[Path] = []
    for model_root in model_roots:
        resolved_model_root = Path(model_root).resolve()
        if not resolved_model_root.exists():
            continue
        discovered.extend(sorted(resolved_model_root.rglob("training_summary.json")))
    return discovered


def _build_model_record(
    *,
    training_summary: Dict[str, Any],
    artifact_dir: Path,
    existing_record: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    target = str(training_summary.get("target") or "").strip()
    if not target:
        return None

    direction = _infer_direction(target)
    backend = _resolve_backend_alias(training_summary)
    model_id = _infer_model_id(target, backend)
    artifact_path = str(artifact_dir.resolve().relative_to(REPO_ROOT))

    cv_summary = training_summary.get("cv_summary", {})
    test_metrics = training_summary.get("test_metrics", {})
    calibration = training_summary.get("calibration", {})

    existing_record = dict(existing_record or {})
    return {
        "model_id": model_id,
        "direction": direction,
        "role": str(existing_record.get("role") or "candidate"),
        "backend": backend,
        "artifact_path": artifact_path,
        "cv_mean_ap": float(cv_summary.get("mean_average_precision") or 0.0),
        "cv_mean_event_f05": float(cv_summary.get("mean_event_fbeta_0_5") or 0.0),
        "test_ap": float(test_metrics.get("average_precision") or 0.0),
        "test_event_f05": float(test_metrics.get("event_fbeta_0_5") or 0.0),
        "global_threshold": _resolve_threshold(training_summary, existing_record),
        "regime_thresholds": existing_record.get("regime_thresholds"),
        "abstain_policy": existing_record.get("abstain_policy"),
        "calibration_method": str(
            calibration.get("resolved_method")
            or existing_record.get("calibration_method")
            or "none"
        ),
        "promotion_date": str(existing_record.get("promotion_date") or date.today().isoformat()),
        "promotion_reason": str(
            existing_record.get("promotion_reason")
            or _default_promotion_reason(target)
        ),
        "status": str(existing_record.get("status") or "candidate"),
    }


def _resolve_threshold(
    training_summary: Dict[str, Any],
    existing_record: Dict[str, Any],
) -> float | None:
    threshold = training_summary.get("threshold")
    if threshold is not None:
        return float(threshold)
    if existing_record.get("global_threshold") is not None:
        return float(existing_record["global_threshold"])
    return None


def _infer_direction(target: str) -> str:
    normalized = str(target).strip().lower()
    if normalized.startswith("long_"):
        return "long"
    if normalized.startswith("short_"):
        return "short"
    raise ValueError(f"Could not infer FRVP direction from target {target!r}")


def _resolve_backend_alias(training_summary: Dict[str, Any]) -> str:
    model_type = str(
        training_summary.get("model_config", {}).get("model_type")
        or training_summary.get("config", {}).get("model_type")
        or ""
    ).strip().lower()
    backend = str(training_summary.get("backend") or "").strip().lower()

    if backend == "xgboost":
        return "xgboost"
    if model_type == "tcn":
        return "tcn"
    if backend in {"tcn", "lstm"}:
        return backend
    raise ValueError(
        f"Could not resolve a registry backend from training summary target={training_summary.get('target')!r} "
        f"backend={backend!r} model_type={model_type!r}"
    )


def _infer_model_id(target: str, backend: str) -> str:
    normalized_target = str(target).strip().lower().replace("_frvp_", "_")
    backend_suffix = "xgb" if backend == "xgboost" else backend
    return f"frvp_{normalized_target}_{backend_suffix}_v1"


def _default_promotion_reason(target: str) -> str:
    normalized = str(target).strip().lower()
    if normalized.endswith("_meta"):
        return "Auto-generated FRVP pooled meta-model candidate from training summaries."
    if any(normalized.endswith(f"_setup{setup_type}") for setup_type in range(1, 7)):
        return "Auto-generated FRVP setup-specific candidate from training summaries."
    return "Auto-generated FRVP candidate from training summaries."


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fresh FRVP candidate registry from on-disk training summaries."
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        action="append",
        dest="model_roots",
        default=None,
        help="Repeat to scan additional FRVP model roots. Defaults to the existing XGBoost and TCN roots.",
    )
    parser.add_argument(
        "--source-registry-path",
        type=Path,
        default=DEFAULT_SOURCE_REGISTRY_PATH,
        help="Existing FRVP registry used only to preserve promotion rules and matching model metadata when available.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_REGISTRY_PATH,
        help="Path for the generated registry JSON.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_roots = [Path(model_root).resolve() for model_root in (args.model_roots or list(DEFAULT_MODEL_ROOTS))]
    payload = build_registry_payload(
        model_roots=model_roots,
        source_registry_path=Path(args.source_registry_path).resolve(),
    )

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "model_count": len(payload["models"])}, indent=2))


if __name__ == "__main__":
    main()

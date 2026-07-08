from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from model_testing.ote_abstain_policy import DEFAULT_SESSION_SPREAD_PIPS
from models.ote_registry_loader import load_ote_model_registry
from ote_live.features.manifest import AbstainPolicy, LivePolicy, PolicyCostAssumptions, PolicyLineage, ThresholdConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_POLICY_ROOT = REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / "multifamily_live_v1"
DEFAULT_POLICY_ARTIFACT_DIR = REPO_ROOT / "ote_live" / "policy_artifacts"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "models" / "ote_model_registry_live_multifamily.json"


def package_candidate_policy_artifacts(
    *,
    threshold_policy_root: str | Path = DEFAULT_THRESHOLD_POLICY_ROOT,
    output_dir: str | Path = DEFAULT_POLICY_ARTIFACT_DIR,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    model_ids: Sequence[str] | None = None,
) -> list[Path]:
    threshold_policy_root = _resolve_path(threshold_policy_root)
    output_dir = _resolve_path(output_dir)
    registry_path = _resolve_path(registry_path)

    registry = load_ote_model_registry(registry_path)
    resolved_model_ids = tuple(model_ids) if model_ids is not None else _default_target_model_ids(registry_path)
    run_summary_path = threshold_policy_root / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    model_outputs = {item["model_id"]: item for item in run_summary.get("model_outputs", [])}

    written_paths: list[Path] = []
    for model_id in resolved_model_ids:
        model_record = registry.get_model(model_id)
        model_output = model_outputs.get(model_id)
        if model_output is None:
            existing_paths = _reuse_existing_packaged_policy_artifacts(
                model_id=model_id,
                output_dir=output_dir,
            )
            if existing_paths is not None:
                written_paths.extend(existing_paths)
                continue
            raise KeyError(f"Threshold policy run {run_summary_path} is missing model {model_id}.")

        policy_table_path = _resolve_path(model_output["policy_table_path"])
        policy_evaluation_path = _resolve_path(model_output["policy_evaluation_path"])
        if not policy_table_path.exists() or not policy_evaluation_path.exists():
            existing_paths = _reuse_existing_packaged_policy_artifacts(
                model_id=model_id,
                output_dir=output_dir,
            )
            if existing_paths is not None:
                written_paths.extend(existing_paths)
                continue
            missing_paths = [
                str(path)
                for path in (policy_table_path, policy_evaluation_path)
                if not path.exists()
            ]
            raise FileNotFoundError(
                f"Threshold policy artifacts are missing for {model_id}: {', '.join(missing_paths)}"
            )
        policy_table = pd.read_csv(policy_table_path)
        policy_evaluation = pd.read_csv(policy_evaluation_path)

        selected_policy_name = str(model_output["selected_policy_name"] or "global_threshold")
        qualified_policy_names = [str(value) for value in model_output.get("qualified_policy_names", [])]
        global_threshold = float(model_output["global_threshold"])
        regime_thresholds = _build_regime_threshold_map(policy_table) if selected_policy_name.startswith("regime_threshold") else None
        use_abstain = selected_policy_name.endswith("_plus_abstain")
        abstain_metadata = dict(model_record.abstain_policy or {})
        targeted_filter_preset = str(
            abstain_metadata.get("targeted_filter_preset") or threshold_policy_root.name
        )
        policy_backtest_summary_path = abstain_metadata.get("policy_backtest_summary_path")

        live_policy = LivePolicy(
            model_id=model_record.model_id,
            direction=model_record.direction,
            backend=model_record.backend,
            calibration_method=model_record.calibration_method,
            policy_status="complete",
            thresholds=ThresholdConfig(
                global_threshold=global_threshold,
                regime_thresholds=regime_thresholds,
            ),
            abstain_policy=_build_abstain_policy(
                use_abstain=use_abstain,
                metadata=abstain_metadata,
            ),
            cost_assumptions=PolicyCostAssumptions(
                fixed_slippage_pips_per_trade=0.0,
                commission_pips_per_trade=0.0,
                session_spread_pips=dict(DEFAULT_SESSION_SPREAD_PIPS),
                targeted_filter_preset=targeted_filter_preset,
            ),
            lineage=PolicyLineage(
                threshold_registry_path=_repo_relative_str(registry_path),
                policy_source_type="threshold_policy_search_package",
                policy_backtest_summary_path=(
                    _repo_relative_str(policy_backtest_summary_path)
                    if policy_backtest_summary_path
                    else None
                ),
                policy_search_summary_path=_repo_relative_str(run_summary_path),
                policy_table_path=_repo_relative_str(policy_table_path),
                policy_evaluation_path=_repo_relative_str(policy_evaluation_path),
                source_model_id=model_record.model_id,
                selected_policy_name=selected_policy_name,
                qualified_policy_names=qualified_policy_names,
                source_match_type="artifact_path",
                notes=_build_lineage_notes(
                    model_id=model_record.model_id,
                    selected_policy_name=selected_policy_name,
                    qualified_policy_names=qualified_policy_names,
                    use_abstain=use_abstain,
                    abstain_metadata=abstain_metadata,
                    targeted_filter_preset=targeted_filter_preset,
                ),
            ),
        )

        policy_output_dir = output_dir / model_record.model_id
        policy_output_dir.mkdir(parents=True, exist_ok=True)
        live_policy_path = policy_output_dir / "live_policy.json"
        selection_summary_path = policy_output_dir / "policy_selection.json"

        live_policy_path.write_text(live_policy.model_dump_json(indent=2), encoding="utf-8")
        selection_summary_path.write_text(
            json.dumps(
                _build_selection_summary_payload(
                    model_output=model_output,
                    policy_table=policy_table,
                    policy_evaluation=policy_evaluation,
                    selected_policy_name=selected_policy_name,
                    regime_thresholds=regime_thresholds,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        written_paths.extend([live_policy_path, selection_summary_path])

    return written_paths


def _reuse_existing_packaged_policy_artifacts(
    *,
    model_id: str,
    output_dir: Path,
) -> list[Path] | None:
    source_model_dir = DEFAULT_POLICY_ARTIFACT_DIR / model_id
    live_policy_path = source_model_dir / "live_policy.json"
    if not live_policy_path.exists():
        return None

    policy = LivePolicy.model_validate_json(live_policy_path.read_text(encoding="utf-8"))
    if policy.model_id != model_id:
        raise ValueError(
            f"Existing packaged live policy {live_policy_path} has model_id={policy.model_id!r}, expected {model_id!r}."
        )

    target_model_dir = output_dir / model_id
    target_model_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for filename in ("live_policy.json", "policy_selection.json"):
        source_path = source_model_dir / filename
        if not source_path.exists():
            continue
        target_path = target_model_dir / filename
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        written_paths.append(target_path)
    return written_paths or None


def _default_target_model_ids(registry_path: str | Path) -> tuple[str, ...]:
    registry = load_ote_model_registry(registry_path)
    return tuple(model.model_id for model in registry.models if model.status != "deprecated")


def _build_regime_threshold_map(policy_table: pd.DataFrame) -> dict[str, float]:
    return {
        str(row["composite_regime"]): float(row["threshold"])
        for _, row in policy_table.iterrows()
        if pd.notna(row.get("threshold"))
    }


def _build_abstain_policy(
    *,
    use_abstain: bool,
    metadata: dict[str, Any] | None = None,
) -> AbstainPolicy:
    metadata = dict(metadata or {})
    if use_abstain:
        return AbstainPolicy(
            enabled=bool(metadata.get("enabled", True)),
            abstain_high_stress=bool(metadata.get("abstain_high_stress", True)),
            abstain_off_hours=bool(metadata.get("abstain_off_hours", True)),
            cooldown_bars=int(metadata.get("cooldown_bars", 4)),
            minimum_expected_move_to_spread=float(
                metadata.get("minimum_expected_move_to_spread", 2.0)
            ),
            abstain_session_regimes=[
                str(value) for value in metadata.get("abstain_session_regimes", [])
            ],
            abstain_composite_regimes=[
                str(value) for value in metadata.get("abstain_composite_regimes", [])
            ],
            abstain_composite_session_pairs=[
                (str(value[0]), str(value[1]))
                for value in metadata.get("abstain_composite_session_pairs", [])
                if isinstance(value, (list, tuple)) and len(value) == 2
            ],
            abstain_composite_stress_pairs=[
                (str(value[0]), str(value[1]))
                for value in metadata.get("abstain_composite_stress_pairs", [])
                if isinstance(value, (list, tuple)) and len(value) == 2
            ],
            minimum_probability_quantile=(
                None
                if metadata.get("minimum_probability_quantile") is None
                else float(metadata["minimum_probability_quantile"])
            ),
        )

    return AbstainPolicy(
        enabled=False,
        abstain_high_stress=False,
        abstain_off_hours=False,
        cooldown_bars=0,
        minimum_expected_move_to_spread=0.0,
        abstain_session_regimes=[],
        abstain_composite_regimes=[],
        abstain_composite_session_pairs=[],
        abstain_composite_stress_pairs=[],
        minimum_probability_quantile=None,
    )


def _build_lineage_notes(
    *,
    model_id: str,
    selected_policy_name: str,
    qualified_policy_names: Sequence[str],
    use_abstain: bool,
    abstain_metadata: dict[str, Any] | None,
    targeted_filter_preset: str | None,
) -> list[str]:
    notes: list[str] = []
    abstain_metadata = dict(abstain_metadata or {})
    if selected_policy_name == "global_threshold" and not qualified_policy_names:
        notes.append(
            "No non-baseline threshold policy qualified in the search run; this package fixes the model-specific baseline global threshold."
        )
    if selected_policy_name == "regime_threshold":
        notes.append("This package uses the candidate-specific regime-threshold table selected by the threshold search run.")
    if not use_abstain:
        notes.append("Hard abstain filters are explicitly disabled in this package because no abstain-aware policy qualified.")
    if use_abstain:
        blocked_composites = [str(value) for value in abstain_metadata.get("abstain_composite_regimes", [])]
        composite_session_pairs = [
            (str(value[0]), str(value[1]))
            for value in abstain_metadata.get("abstain_composite_session_pairs", [])
            if isinstance(value, (list, tuple)) and len(value) == 2
        ]
        composite_stress_pairs = [
            (str(value[0]), str(value[1]))
            for value in abstain_metadata.get("abstain_composite_stress_pairs", [])
            if isinstance(value, (list, tuple)) and len(value) == 2
        ]
        probability_quantile = abstain_metadata.get("minimum_probability_quantile")
        if targeted_filter_preset and (
            blocked_composites
            or composite_session_pairs
            or composite_stress_pairs
            or probability_quantile is not None
        ):
            notes.append(
                f"Registry abstain overrides from {targeted_filter_preset} are applied to the packaged live policy."
            )
        if blocked_composites:
            notes.append(
                "Composite abstain regimes: " + ", ".join(sorted(blocked_composites)) + "."
            )
        if composite_session_pairs:
            notes.append(
                "Composite/session abstains: "
                + ", ".join(
                    f"{composite_regime} x {session_regime}"
                    for composite_regime, session_regime in sorted(composite_session_pairs)
                )
                + "."
            )
        if composite_stress_pairs:
            notes.append(
                "Composite/stress abstains: "
                + ", ".join(
                    f"{composite_regime} x {stress_regime}"
                    for composite_regime, stress_regime in sorted(composite_stress_pairs)
                )
                + "."
            )
        if probability_quantile is not None:
            notes.append(
                f"Signals below probability quantile {float(probability_quantile):.2f} are filtered out."
            )
    notes.append("Costs in this threshold-search package reflect spread-aware search defaults with fixed slippage and commission set to 0.0.")
    notes.append(f"Packaged from threshold search output for {model_id}.")
    return notes


def _build_selection_summary_payload(
    *,
    model_output: dict[str, Any],
    policy_table: pd.DataFrame,
    policy_evaluation: pd.DataFrame,
    selected_policy_name: str,
    regime_thresholds: dict[str, float] | None,
) -> dict[str, Any]:
    test_rows = policy_evaluation.loc[policy_evaluation["dataset_split"] == "test"].copy()
    selected_policy_row = test_rows.loc[test_rows["policy_name"] == selected_policy_name]
    baseline_row = test_rows.loc[test_rows["policy_name"] == "global_threshold"]
    return {
        "model_id": model_output["model_id"],
        "selected_policy_name": selected_policy_name,
        "qualified_policy_names": list(model_output.get("qualified_policy_names", [])),
        "selected_policy_metrics": dict(model_output.get("selected_policy_metrics", {})),
        "global_threshold": float(model_output["global_threshold"]),
        "regime_thresholds": regime_thresholds,
        "test_selected_policy_row": _frame_row_to_payload(selected_policy_row),
        "test_baseline_policy_row": _frame_row_to_payload(baseline_row),
        "policy_table_rows": json.loads(policy_table.to_json(orient="records")),
    }


def _frame_row_to_payload(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    return json.loads(frame.iloc[0].to_json())


def _repo_relative_str(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    try:
        return path_obj.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path_obj.resolve().as_posix()


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (REPO_ROOT / path_obj).resolve()

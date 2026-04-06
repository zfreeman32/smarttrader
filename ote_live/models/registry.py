from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry
from ote_live.features.manifest import (
    AbstainPolicy,
    ArtifactReferences,
    ArtifactTestMetrics,
    DirectionRecommendations,
    DirectionRuntimeManifest,
    FeatureManifest,
    FeatureValidationSummary,
    LabelHorizonAssumptions,
    LivePolicy,
    LiveRuntimeManifest,
    PolicyCostAssumptions,
    PolicyLineage,
    RegistryMetrics,
    RuntimeContextRequirements,
    SourceLineage,
    ThresholdConfig,
    TimezoneContract,
    UpstreamTimezoneContract,
)
from ote_live.policies.packager import DEFAULT_POLICY_ARTIFACT_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_REGISTRY_PATH = REPO_ROOT / "models" / "ote_model_registry_v1_v2_candidates.json"
DEFAULT_ACTIVE_REGISTRY_PATH = REPO_ROOT / "models" / "ote_model_registry.json"
DEFAULT_PREPARED_SUMMARY_PATH = REPO_ROOT / "data" / "prepared" / "eurusd_5min_ote_full" / "summary.json"
DEFAULT_FEATURE_METADATA_PATH = REPO_ROOT / "data" / "features" / "eurusd_5min_ote_full.metadata.json"
DEFAULT_LONG_FEATURE_PATH = REPO_ROOT / "data" / "prepared" / "eurusd_5min_ote_full" / "long_ote" / "features.json"
DEFAULT_SHORT_FEATURE_PATH = REPO_ROOT / "data" / "prepared" / "eurusd_5min_ote_full" / "short_ote" / "features.json"
DEFAULT_POLICY_BACKTEST_SUMMARY_PATH = (
    REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / "v1_v2_tcn_focus" / "run_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ote_live" / "runtime_manifests"
NON_LIVE_SAFE_SELECTED_FEATURE_NAMES = frozenset(
    {
        "strategy__price_action_signals__signal__short",
    }
)

_VERSION_PATTERN = re.compile(r"_v(?P<version>\d+)(?:_|$)")


@dataclass(frozen=True)
class FeatureCatalogs:
    metadata_path: Path
    canonical_features: tuple[str, ...]
    long_feature_path: Path
    long_features: tuple[str, ...]
    short_feature_path: Path
    short_features: tuple[str, ...]

    def for_direction(self, direction: str) -> tuple[Path, tuple[str, ...]]:
        if direction == "long":
            return self.long_feature_path, self.long_features
        if direction == "short":
            return self.short_feature_path, self.short_features
        raise ValueError(f"Unsupported direction: {direction!r}")


@dataclass(frozen=True)
class PolicySourceRecord:
    direction: str
    model_id: str
    artifact_path: str
    targeted_filters: dict[str, Any]
    targeted_filter_preset: str | None
    fixed_slippage_pips_per_trade: float | None
    commission_pips_per_trade: float | None
    session_spread_pips: dict[str, float]
    run_summary_path: str
    active_registry_path: str
    active_registry_record: OTEModelRecord | None
    monthly_sharpe: float
    profit_factor: float
    expectancy_pips: float
    profitable_quarter_share: float
    positive_composite_expectancy_share: float
    accepted_for_paper_trading_gate: bool
    overall_wfe: float
    max_drawdown_pips: float


def build_direction_runtime_manifests(
    *,
    registry_path: str | Path = DEFAULT_CANDIDATE_REGISTRY_PATH,
    prepared_summary_path: str | Path = DEFAULT_PREPARED_SUMMARY_PATH,
    feature_metadata_path: str | Path = DEFAULT_FEATURE_METADATA_PATH,
    long_feature_path: str | Path = DEFAULT_LONG_FEATURE_PATH,
    short_feature_path: str | Path = DEFAULT_SHORT_FEATURE_PATH,
    policy_backtest_summary_path: str | Path = DEFAULT_POLICY_BACKTEST_SUMMARY_PATH,
    packaged_policy_dir: str | Path = DEFAULT_POLICY_ARTIFACT_DIR,
) -> dict[str, DirectionRuntimeManifest]:
    registry_path = _resolve_path(registry_path)
    prepared_summary_path = _resolve_path(prepared_summary_path)
    feature_metadata_path = _resolve_path(feature_metadata_path)
    long_feature_path = _resolve_path(long_feature_path)
    short_feature_path = _resolve_path(short_feature_path)
    policy_backtest_summary_path = _resolve_path(policy_backtest_summary_path)
    packaged_policy_dir = _resolve_path(packaged_policy_dir)

    registry = load_ote_model_registry(registry_path)
    prepared_summary = _read_json(prepared_summary_path)
    feature_catalogs = _load_feature_catalogs(
        feature_metadata_path=feature_metadata_path,
        long_feature_path=long_feature_path,
        short_feature_path=short_feature_path,
    )
    policy_sources = _load_policy_sources(policy_backtest_summary_path)

    generated_at_utc = datetime.now(timezone.utc)
    manifests_by_direction: dict[str, list[LiveRuntimeManifest]] = {"long": [], "short": []}

    for model_record in registry.models:
        manifest = _build_live_runtime_manifest(
            model_record=model_record,
            registry_path=registry_path,
            prepared_summary=prepared_summary,
            feature_catalogs=feature_catalogs,
            policy_sources=policy_sources,
            policy_backtest_summary_path=policy_backtest_summary_path,
            packaged_policy_dir=packaged_policy_dir,
            generated_at_utc=generated_at_utc,
        )
        if not _is_live_eligible_manifest(manifest):
            continue
        manifests_by_direction[model_record.direction].append(manifest)

    direction_manifests: dict[str, DirectionRuntimeManifest] = {}
    _, _, policy_sources_by_model = policy_sources
    for direction, manifests in manifests_by_direction.items():
        if not manifests:
            continue
        manifests.sort(key=lambda manifest: _manifest_sort_key(manifest, policy_sources_by_model), reverse=True)
        recommendations = DirectionRecommendations(
            recommended_primary_model_id=_recommended_primary_model_id(manifests, policy_sources_by_model),
            recommended_strongest_overall_model_id=manifests[0].model_id,
            recommended_strongest_v2_model_id=_recommended_v2_model_id(manifests, policy_sources_by_model),
        )
        direction_manifests[direction] = DirectionRuntimeManifest(
            generated_at_utc=generated_at_utc,
            registry_path=_repo_relative_str(registry_path),
            policy_backtest_summary_path=_repo_relative_str(policy_backtest_summary_path),
            direction=direction,
            asset=manifests[0].asset,
            timeframe=manifests[0].timeframe,
            recommendations=recommendations,
            models=manifests,
        )
    return direction_manifests


def write_direction_runtime_manifests(
    direction_manifests: dict[str, DirectionRuntimeManifest],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for direction, manifest in direction_manifests.items():
        bundle_path = output_dir / f"live_runtime_manifest_{direction}.json"
        bundle_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        for model_manifest in manifest.models:
            model_output_dir = output_dir / model_manifest.model_id
            model_output_dir.mkdir(parents=True, exist_ok=True)
            (model_output_dir / "live_runtime_manifest.json").write_text(
                model_manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (model_output_dir / "live_policy.json").write_text(
                model_manifest.live_policy.model_dump_json(indent=2),
                encoding="utf-8",
            )
    return output_dir


def validate_manifest_for_live_decisions(
    manifest: LiveRuntimeManifest,
    *,
    require_complete_policy: bool = True,
) -> None:
    if manifest.context_requirements.context_rows <= 0:
        raise ValueError(f"Manifest {manifest.model_id} is missing a valid context_rows requirement.")
    if manifest.context_requirements.window_size <= 0:
        raise ValueError(f"Manifest {manifest.model_id} is missing a valid window_size requirement.")
    if require_complete_policy and manifest.live_policy.policy_status != "complete":
        raise ValueError(
            f"Manifest {manifest.model_id} does not have a complete policy package for live promotion."
        )


def _build_live_runtime_manifest(
    *,
    model_record: OTEModelRecord,
    registry_path: Path,
    prepared_summary: dict[str, Any],
    feature_catalogs: FeatureCatalogs,
    policy_sources: tuple[
        dict[str, PolicySourceRecord],
        dict[str, PolicySourceRecord],
        dict[str, PolicySourceRecord],
    ],
    policy_backtest_summary_path: Path,
    packaged_policy_dir: Path,
    generated_at_utc: datetime,
) -> LiveRuntimeManifest:
    artifact_dir = model_record.resolve_artifact_path(REPO_ROOT)
    model_config_path = artifact_dir / "model_config.json"
    training_summary_path = artifact_dir / "training_summary.json"
    model_config = _read_json(model_config_path)
    training_summary = _read_json(training_summary_path)

    selected_feature_names = list(
        _first_non_none(
            model_config.get("selected_feature_names"),
            training_summary.get("selected_feature_names"),
            _nested_get(training_summary, "model_config", "selected_feature_names"),
            [],
        )
    )
    if not selected_feature_names:
        raise ValueError(f"Model {model_record.model_id} is missing selected_feature_names.")

    context_rows = _as_int(
        _first_non_none(
            model_config.get("context_rows"),
            training_summary.get("context_rows"),
            _nested_get(training_summary, "model_config", "context_rows"),
        )
    )
    if context_rows is None:
        raise ValueError(f"Model {model_record.model_id} is missing context_rows.")

    window_size = _as_int(
        _first_non_none(
            model_config.get("window_size"),
            training_summary.get("window_size"),
            _nested_get(training_summary, "model_config", "window_size"),
        )
    )
    if window_size is None:
        raise ValueError(f"Model {model_record.model_id} is missing window_size.")

    lag_steps = list(
        _first_non_none(
            model_config.get("lag_steps"),
            training_summary.get("lag_steps"),
            _nested_get(training_summary, "model_config", "lag_steps"),
            [],
        )
    )
    delta_feature_count = int(
        _first_non_none(
            model_config.get("delta_feature_count"),
            training_summary.get("delta_feature_count"),
            _nested_get(training_summary, "model_config", "delta_feature_count"),
            0,
        )
    )

    feature_path, direction_features = feature_catalogs.for_direction(model_record.direction)
    canonical_feature_set = set(feature_catalogs.canonical_features)
    direction_feature_set = set(direction_features)
    missing_from_canonical = sorted(set(selected_feature_names) - canonical_feature_set)
    missing_from_direction = sorted(set(selected_feature_names) - direction_feature_set)
    if missing_from_canonical or missing_from_direction:
        raise ValueError(
            f"Model {model_record.model_id} failed feature validation: "
            f"missing_from_canonical={missing_from_canonical}, "
            f"missing_from_direction={missing_from_direction}"
        )

    timezone_payload = _first_non_none(
        model_config.get("timezone_contract"),
        training_summary.get("prepared_summary", {}).get("timezone_contract"),
        training_summary.get("report", {}).get("timezone_contract"),
        prepared_summary.get("timezone_contract"),
    )
    source_lineage_payload = _first_non_none(
        model_config.get("source_lineage"),
        training_summary.get("prepared_summary", {}).get("source_lineage"),
        training_summary.get("report", {}).get("source_lineage"),
        prepared_summary.get("source_lineage"),
    )
    if timezone_payload is None or source_lineage_payload is None:
        raise ValueError(f"Model {model_record.model_id} is missing timezone or lineage metadata.")

    live_policy = _build_live_policy(
        model_record=model_record,
        registry_path=registry_path,
        policy_sources=policy_sources,
        policy_backtest_summary_path=policy_backtest_summary_path,
        packaged_policy_dir=packaged_policy_dir,
    )

    upstream_preprocessing = prepared_summary.get("upstream_preprocessing", {})
    resolved_purge_bars = _as_int(training_summary.get("resolved_purge_bars"))
    feature_builder_warmup_rows = _as_int(upstream_preprocessing.get("warmup_rows_configured"))
    minimum_runtime_history_bars = max(
        value
        for value in (
            window_size,
            context_rows + 1,
            resolved_purge_bars or 0,
            feature_builder_warmup_rows or 0,
        )
    )

    artifacts = ArtifactReferences(
        artifact_dir=_repo_relative_str(artifact_dir),
        model_file=_repo_relative_str(_resolve_model_file(artifact_dir)),
        scaler_file=_optional_repo_relative_str(artifact_dir / "scaler.joblib"),
        calibrator_file=_optional_repo_relative_str(artifact_dir / "calibrator.joblib"),
        model_config_file=_repo_relative_str(model_config_path),
        training_summary_file=_repo_relative_str(training_summary_path),
        test_predictions_file=_optional_repo_relative_str(artifact_dir / "test_predictions.csv"),
    )

    asset, timeframe = _infer_asset_and_timeframe(source_lineage_payload["upstream_source_path"])
    notes = _build_manifest_notes(
        model_record=model_record,
        training_summary=training_summary,
        live_policy=live_policy,
    )

    return LiveRuntimeManifest(
        generated_at_utc=generated_at_utc,
        registry_path=_repo_relative_str(registry_path),
        policy_backtest_summary_path=_repo_relative_str(policy_backtest_summary_path),
        asset=asset,
        timeframe=timeframe,
        model_id=model_record.model_id,
        direction=model_record.direction,
        role=model_record.role,
        backend=model_record.backend,
        status=model_record.status,
        promotion_date=model_record.promotion_date,
        promotion_reason=model_record.promotion_reason,
        calibration_method=model_record.calibration_method,
        artifact_references=artifacts,
        registry_metrics=RegistryMetrics(
            cv_mean_ap=model_record.cv_mean_ap,
            cv_mean_event_f05=model_record.cv_mean_event_f05,
            test_ap=model_record.test_ap,
            test_event_f05=model_record.test_event_f05,
        ),
        artifact_test_metrics=ArtifactTestMetrics(
            **{
                key: training_summary.get("test_metrics", {}).get(key)
                for key in ArtifactTestMetrics.model_fields
            }
        ),
        context_requirements=RuntimeContextRequirements(
            window_size=window_size,
            context_rows=context_rows,
            feature_builder_warmup_rows=feature_builder_warmup_rows,
            resolved_purge_bars=resolved_purge_bars,
            minimum_runtime_history_bars=minimum_runtime_history_bars,
            label_horizon_assumptions=LabelHorizonAssumptions(
                **training_summary.get("label_horizon_assumptions", {})
            ),
        ),
        feature_manifest=FeatureManifest(
            canonical_feature_metadata_file=_repo_relative_str(feature_catalogs.metadata_path),
            canonical_feature_count=len(feature_catalogs.canonical_features),
            direction_feature_file=_repo_relative_str(feature_path),
            direction_feature_count=len(direction_features),
            selected_feature_names=selected_feature_names,
            selected_feature_count=len(selected_feature_names),
            strategy_feature_count=sum(1 for name in selected_feature_names if name.startswith("strategy__")),
            lag_steps=lag_steps,
            delta_feature_count=delta_feature_count,
            validation=FeatureValidationSummary(
                selected_features_in_canonical_catalog=not missing_from_canonical,
                selected_features_in_direction_feature_list=not missing_from_direction,
                missing_from_canonical_catalog=missing_from_canonical,
                missing_from_direction_feature_list=missing_from_direction,
            ),
        ),
        timezone_contract=TimezoneContract(**timezone_payload),
        source_lineage=SourceLineage(
            **{
                **source_lineage_payload,
                "upstream_timezone_contract": UpstreamTimezoneContract(
                    **source_lineage_payload["upstream_timezone_contract"]
                ),
            }
        ),
        live_policy=live_policy,
        notes=notes,
    )


def _build_live_policy(
    *,
    model_record: OTEModelRecord,
    registry_path: Path,
    policy_sources: tuple[
        dict[str, PolicySourceRecord],
        dict[str, PolicySourceRecord],
        dict[str, PolicySourceRecord],
    ],
    policy_backtest_summary_path: Path,
    packaged_policy_dir: Path,
) -> LivePolicy:
    packaged_policy = _load_packaged_live_policy(
        packaged_policy_dir=packaged_policy_dir,
        model_record=model_record,
    )
    if packaged_policy is not None:
        return packaged_policy

    by_artifact, by_direction, _ = policy_sources
    artifact_key = _repo_relative_str(model_record.artifact_path)
    exact_match = by_artifact.get(artifact_key)
    source_record = exact_match or by_direction.get(model_record.direction)
    if source_record is None:
        raise ValueError(f"No policy source available for {model_record.model_id}.")

    policy_notes: list[str] = []
    match_type = "artifact_path" if exact_match is not None else "direction_fallback"
    policy_status = "complete" if exact_match is not None else "provisional"

    if exact_match is None:
        policy_notes.append(
            "Candidate-specific abstain filters were not packaged on disk; "
            "using the strongest same-direction backtest filter package as a provisional fallback."
        )

    active_regime_thresholds = (
        source_record.active_registry_record.regime_thresholds if source_record.active_registry_record else None
    )
    if model_record.regime_thresholds is None and active_regime_thresholds:
        policy_notes.append(
            "An active registry entry for this direction has regime thresholds, "
            "but the candidate registry export intentionally uses the candidate registry threshold package."
        )

    return LivePolicy(
        model_id=model_record.model_id,
        direction=model_record.direction,
        backend=model_record.backend,
        calibration_method=model_record.calibration_method,
        policy_status=policy_status,
        thresholds=ThresholdConfig(
            global_threshold=model_record.global_threshold,
            regime_thresholds=model_record.regime_thresholds,
        ),
        abstain_policy=AbstainPolicy(**source_record.targeted_filters),
        cost_assumptions=PolicyCostAssumptions(
            fixed_slippage_pips_per_trade=source_record.fixed_slippage_pips_per_trade,
            commission_pips_per_trade=source_record.commission_pips_per_trade,
            session_spread_pips=source_record.session_spread_pips,
            targeted_filter_preset=source_record.targeted_filter_preset,
        ),
        lineage=PolicyLineage(
            threshold_registry_path=_repo_relative_str(registry_path),
            policy_source_type="active_backtest_fallback",
            policy_backtest_summary_path=_repo_relative_str(policy_backtest_summary_path),
            active_registry_path=source_record.active_registry_path,
            source_model_id=source_record.model_id,
            source_match_type=match_type,
            notes=policy_notes,
        ),
    )


def _build_manifest_notes(
    *,
    model_record: OTEModelRecord,
    training_summary: dict[str, Any],
    live_policy: LivePolicy,
) -> list[str]:
    notes = list(live_policy.lineage.notes)
    artifact_threshold = training_summary.get("threshold") or training_summary.get("test_metrics", {}).get("threshold")
    if artifact_threshold is not None and model_record.global_threshold is not None:
        if abs(float(artifact_threshold) - float(model_record.global_threshold)) > 1e-9:
            notes.append(
                f"Training artifact threshold ({artifact_threshold}) differs from registry threshold "
                f"({model_record.global_threshold}); runtime uses the registry value."
            )
    live_threshold = live_policy.thresholds.global_threshold
    if live_threshold is not None and model_record.global_threshold is not None:
        if abs(float(live_threshold) - float(model_record.global_threshold)) > 1e-9:
            notes.append(
                f"Packaged live policy threshold ({live_threshold}) overrides the candidate registry threshold "
                f"({model_record.global_threshold})."
            )
    return notes


def _is_live_eligible_manifest(manifest: LiveRuntimeManifest) -> bool:
    return not any(
        feature_name in NON_LIVE_SAFE_SELECTED_FEATURE_NAMES
        for feature_name in manifest.feature_manifest.selected_feature_names
    )


def _load_feature_catalogs(
    *,
    feature_metadata_path: Path,
    long_feature_path: Path,
    short_feature_path: Path,
) -> FeatureCatalogs:
    metadata_payload = _read_json(feature_metadata_path)
    long_payload = _read_json(long_feature_path)
    short_payload = _read_json(short_feature_path)
    return FeatureCatalogs(
        metadata_path=feature_metadata_path,
        canonical_features=tuple(metadata_payload.get("feature_columns", [])),
        long_feature_path=long_feature_path,
        long_features=tuple(long_payload.get("features", [])),
        short_feature_path=short_feature_path,
        short_features=tuple(short_payload.get("features", [])),
    )


def _load_policy_sources(
    policy_backtest_summary_path: Path,
) -> tuple[dict[str, PolicySourceRecord], dict[str, PolicySourceRecord], dict[str, PolicySourceRecord]]:
    payload = _read_json(policy_backtest_summary_path)
    source_registry_path = _resolve_path(payload.get("registry_path", DEFAULT_ACTIVE_REGISTRY_PATH))
    source_registry = load_ote_model_registry(source_registry_path)
    active_registry_path = _resolve_path(DEFAULT_ACTIVE_REGISTRY_PATH)
    active_registry = load_ote_model_registry(active_registry_path)

    by_artifact: dict[str, PolicySourceRecord] = {}
    by_direction: dict[str, PolicySourceRecord] = {}
    by_model: dict[str, PolicySourceRecord] = {}
    for model_output in payload.get("model_outputs", []):
        source_record = source_registry.get_model(model_output["model_id"])
        try:
            active_record = active_registry.get_model(model_output["model_id"])
        except KeyError:
            active_record = None

        overall_metrics = model_output.get("overall_test_metrics", {})
        walk_forward = model_output.get("walk_forward_efficiency", {})
        artifact_path = _repo_relative_str(source_record.artifact_path)
        source_record = PolicySourceRecord(
            direction=model_output["direction"],
            model_id=model_output["model_id"],
            artifact_path=artifact_path,
            targeted_filters=model_output["targeted_filters"],
            targeted_filter_preset=payload.get("targeted_filter_preset"),
            fixed_slippage_pips_per_trade=payload.get("fixed_slippage_pips_per_trade"),
            commission_pips_per_trade=payload.get("commission_pips_per_trade"),
            session_spread_pips=payload.get("session_spread_pips", {}),
            run_summary_path=_repo_relative_str(policy_backtest_summary_path),
            active_registry_path=_repo_relative_str(active_registry_path),
            active_registry_record=active_record,
            monthly_sharpe=float(overall_metrics.get("monthly_sharpe", 0.0)),
            profit_factor=float(overall_metrics.get("profit_factor", 0.0)),
            expectancy_pips=float(overall_metrics.get("expectancy_pips", 0.0)),
            profitable_quarter_share=float(overall_metrics.get("profitable_quarter_share", 0.0)),
            positive_composite_expectancy_share=float(model_output.get("positive_composite_expectancy_share", 0.0)),
            accepted_for_paper_trading_gate=bool(
                model_output.get("acceptance", {}).get("policy_profitable_after_costs", False)
                and model_output.get("acceptance", {}).get("wfe_above_threshold", False)
                and model_output.get("acceptance", {}).get("profitable_quarter_share_above_threshold", False)
                and model_output.get("acceptance", {}).get("positive_composite_expectancy_share_above_threshold", False)
                and model_output.get("acceptance", {}).get("largest_single_trade_share_below_limit", False)
                and model_output.get("acceptance", {}).get("max_drawdown_less_than_two_times_average_monthly_profit", False)
            ),
            overall_wfe=float(walk_forward.get("overall_wfe", 0.0)),
            max_drawdown_pips=float(overall_metrics.get("max_drawdown_pips", float("inf"))),
        )
        by_artifact[artifact_path] = source_record
        by_model[source_record.model_id] = source_record

        incumbent = by_direction.get(source_record.direction)
        if incumbent is None or _policy_source_sort_key(source_record) > _policy_source_sort_key(incumbent):
            by_direction[source_record.direction] = source_record
    return by_artifact, by_direction, by_model


def _load_packaged_live_policy(
    *,
    packaged_policy_dir: Path,
    model_record: OTEModelRecord,
) -> LivePolicy | None:
    policy_path = packaged_policy_dir / model_record.model_id / "live_policy.json"
    if not policy_path.exists():
        return None

    policy = LivePolicy.model_validate_json(policy_path.read_text(encoding="utf-8"))
    if policy.model_id != model_record.model_id:
        raise ValueError(
            f"Packaged policy {policy_path} has model_id={policy.model_id!r}, expected {model_record.model_id!r}."
        )
    if policy.direction != model_record.direction:
        raise ValueError(
            f"Packaged policy {policy_path} has direction={policy.direction!r}, expected {model_record.direction!r}."
        )
    if policy.backend != model_record.backend:
        raise ValueError(
            f"Packaged policy {policy_path} has backend={policy.backend!r}, expected {model_record.backend!r}."
        )
    return policy


def _recommended_primary_model_id(
    manifests: list[LiveRuntimeManifest],
    policy_sources_by_model: dict[str, PolicySourceRecord],
) -> str | None:
    complete_manifests = [manifest for manifest in manifests if manifest.live_policy.policy_status == "complete"]
    if complete_manifests:
        complete_manifests.sort(
            key=lambda manifest: _manifest_sort_key(manifest, policy_sources_by_model),
            reverse=True,
        )
        return complete_manifests[0].model_id
    return manifests[0].model_id if manifests else None


def _recommended_v2_model_id(
    manifests: list[LiveRuntimeManifest],
    policy_sources_by_model: dict[str, PolicySourceRecord],
) -> str | None:
    v2_manifests = [manifest for manifest in manifests if _extract_version(manifest.model_id) == 2]
    if not v2_manifests:
        return None
    v2_manifests.sort(
        key=lambda manifest: _manifest_sort_key(manifest, policy_sources_by_model),
        reverse=True,
    )
    return v2_manifests[0].model_id


def _manifest_sort_key(
    manifest: LiveRuntimeManifest,
    policy_sources_by_model: dict[str, PolicySourceRecord],
) -> tuple[float, ...]:
    policy_source = policy_sources_by_model.get(manifest.model_id)
    if policy_source is not None:
        return (
            1.0 if manifest.live_policy.policy_status == "complete" else 0.0,
            *_policy_source_sort_key(policy_source),
            manifest.registry_metrics.test_ap,
            manifest.registry_metrics.test_event_f05,
        )
    return (
        1.0 if manifest.live_policy.policy_status == "complete" else 0.0,
        manifest.registry_metrics.test_ap,
        manifest.registry_metrics.cv_mean_ap,
        manifest.registry_metrics.test_event_f05,
        manifest.registry_metrics.cv_mean_event_f05,
    )


def _policy_source_sort_key(source_record: PolicySourceRecord) -> tuple[float, ...]:
    max_drawdown_score = -source_record.max_drawdown_pips
    return (
        1.0 if source_record.accepted_for_paper_trading_gate else 0.0,
        source_record.monthly_sharpe,
        source_record.profit_factor,
        source_record.expectancy_pips,
        source_record.profitable_quarter_share,
        source_record.positive_composite_expectancy_share,
        max_drawdown_score,
        source_record.overall_wfe,
    )


def _extract_version(model_id: str) -> int | None:
    match = _VERSION_PATTERN.search(model_id)
    if not match:
        return None
    return int(match.group("version"))


def _resolve_model_file(artifact_dir: Path) -> Path:
    for filename in ("model.pt", "model.json", "model.joblib"):
        candidate = artifact_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported model file found under {artifact_dir}")


def _infer_asset_and_timeframe(upstream_source_path: str) -> tuple[str, str]:
    stem = Path(upstream_source_path).stem.lower()
    if "-" not in stem:
        return "EURUSD", "5m"
    asset, timeframe = stem.split("-", 1)
    return asset.upper(), timeframe


def _repo_relative_str(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    try:
        return path_obj.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path_obj.resolve().as_posix()


def _optional_repo_relative_str(path: Path) -> str | None:
    if not path.exists():
        return None
    return _repo_relative_str(path)


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (REPO_ROOT / path_obj).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)

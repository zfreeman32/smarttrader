from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpstreamTimezoneContract(ManifestModel):
    source_timezone: str
    canonical_timezone: str
    csv_timezone: str
    csv_timestamps_are_timezone_aware: bool
    timestamp_column: str


class TimezoneContract(ManifestModel):
    source_timezone: str
    canonical_timezone: str
    feature_clock_timezone: str
    london_timezone: str
    new_york_timezone: str
    market_close_timezone: str
    market_close_time: str


class SourceLineage(ManifestModel):
    feature_csv: str
    feature_metadata_file: str
    feature_builder_source_path: str
    feature_builder_source_metadata_file: str
    upstream_source_path: str
    upstream_metadata_file: str
    upstream_bar_timestamp_semantics: str
    upstream_timezone_contract: UpstreamTimezoneContract


class ArtifactReferences(ManifestModel):
    artifact_dir: str
    model_file: str
    scaler_file: str | None = None
    calibrator_file: str | None = None
    model_config_file: str
    training_summary_file: str
    test_predictions_file: str | None = None


class FeatureValidationSummary(ManifestModel):
    selected_features_in_canonical_catalog: bool
    selected_features_in_direction_feature_list: bool
    missing_from_canonical_catalog: list[str] = Field(default_factory=list)
    missing_from_direction_feature_list: list[str] = Field(default_factory=list)


class FeatureManifest(ManifestModel):
    canonical_feature_metadata_file: str
    canonical_feature_count: int
    direction_feature_file: str
    direction_feature_count: int
    selected_feature_names: list[str]
    selected_feature_count: int
    strategy_feature_count: int
    lag_steps: list[int] = Field(default_factory=list)
    delta_feature_count: int = 0
    validation: FeatureValidationSummary


class LabelHorizonAssumptions(ManifestModel):
    label_max_holding_bars: int | None = None
    label_exclusion_pre_bars: int | None = None
    label_zone_pre_bars: int | None = None
    purge_buffer_bars: int | None = None
    event_tolerance_bars: int | None = None
    event_cooldown_bars: int | None = None


class RuntimeContextRequirements(ManifestModel):
    window_size: int
    context_rows: int
    feature_builder_warmup_rows: int | None = None
    resolved_purge_bars: int | None = None
    minimum_runtime_history_bars: int
    label_horizon_assumptions: LabelHorizonAssumptions


class RegistryMetrics(ManifestModel):
    cv_mean_ap: float
    cv_mean_event_f05: float
    test_ap: float
    test_event_f05: float


class ArtifactTestMetrics(ManifestModel):
    average_precision: float | None = None
    roc_auc: float | None = None
    brier: float | None = None
    threshold: float | None = None
    event_precision: float | None = None
    event_recall: float | None = None
    event_f1: float | None = None
    event_fbeta_0_5: float | None = None
    predicted_events: float | None = None
    true_events: float | None = None
    matched_events: float | None = None


class ThresholdConfig(ManifestModel):
    global_threshold: float | None = None
    regime_thresholds: dict[str, float] | None = None


class AbstainPolicy(ManifestModel):
    enabled: bool = True
    abstain_high_stress: bool
    abstain_off_hours: bool
    cooldown_bars: int
    minimum_expected_move_to_spread: float
    abstain_session_regimes: list[str] = Field(default_factory=list)
    abstain_composite_regimes: list[str] = Field(default_factory=list)
    minimum_probability_quantile: float | None = None


class PolicyCostAssumptions(ManifestModel):
    fixed_slippage_pips_per_trade: float | None = None
    commission_pips_per_trade: float | None = None
    session_spread_pips: dict[str, float] = Field(default_factory=dict)
    targeted_filter_preset: str | None = None


class PolicyLineage(ManifestModel):
    threshold_registry_path: str
    policy_source_type: str = "active_backtest_fallback"
    policy_backtest_summary_path: str | None = None
    policy_search_summary_path: str | None = None
    policy_table_path: str | None = None
    policy_evaluation_path: str | None = None
    active_registry_path: str | None = None
    source_model_id: str | None = None
    selected_policy_name: str | None = None
    qualified_policy_names: list[str] = Field(default_factory=list)
    source_match_type: Literal["artifact_path", "direction_fallback"] = "direction_fallback"
    notes: list[str] = Field(default_factory=list)


class LivePolicy(ManifestModel):
    schema_version: str = "1.0.0"
    model_id: str
    direction: Literal["long", "short"]
    backend: Literal["tcn", "lstm", "xgboost"]
    calibration_method: str
    policy_status: Literal["complete", "provisional"]
    thresholds: ThresholdConfig
    abstain_policy: AbstainPolicy
    cost_assumptions: PolicyCostAssumptions
    lineage: PolicyLineage


class LiveRuntimeManifest(ManifestModel):
    manifest_version: str = "1.0.0"
    generated_at_utc: datetime
    registry_path: str
    policy_backtest_summary_path: str | None = None
    asset: str
    timeframe: str
    model_id: str
    direction: Literal["long", "short"]
    role: str
    backend: Literal["tcn", "lstm", "xgboost"]
    status: str
    promotion_date: str
    promotion_reason: str
    calibration_method: str
    artifact_references: ArtifactReferences
    registry_metrics: RegistryMetrics
    artifact_test_metrics: ArtifactTestMetrics
    context_requirements: RuntimeContextRequirements
    feature_manifest: FeatureManifest
    timezone_contract: TimezoneContract
    source_lineage: SourceLineage
    live_policy: LivePolicy
    notes: list[str] = Field(default_factory=list)


class DirectionRecommendations(ManifestModel):
    recommended_primary_model_id: str | None = None
    recommended_strongest_overall_model_id: str | None = None
    recommended_strongest_v2_model_id: str | None = None


class DirectionRuntimeManifest(ManifestModel):
    manifest_version: str = "1.0.0"
    generated_at_utc: datetime
    registry_path: str
    policy_backtest_summary_path: str | None = None
    direction: Literal["long", "short"]
    asset: str
    timeframe: str
    recommendations: DirectionRecommendations
    models: list[LiveRuntimeManifest]

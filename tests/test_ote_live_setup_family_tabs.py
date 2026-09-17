from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ote_live.contracts.prediction import ModelPrediction
from ote_live.dashboard.view_registry import build_default_dashboard_views
from ote_live.features.manifest import DirectionRuntimeManifest
from ote_live.models.ensemble import load_direction_models
from ote_live.models.loaders import load_live_runtime_manifest
from ote_live.models.setup_family import (
    infer_setup_model_route,
    resolve_setup_family_gate,
)
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.ingestion.signals import LiveSignalProcessor
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONG_MODEL_MANIFEST_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "long_reversal_tcn_v2_20260525_narrow48"
    / "live_runtime_manifest.json"
)


def test_default_dashboard_views_include_setup_family_tabs() -> None:
    view_by_id = {view.view_id: view for view in build_default_dashboard_views()}

    assert list(view_by_id) == ["OTE", "FRVP", "FRVP_SETUP", "ICT", "ICT_SETUP"]
    frvp_setup = view_by_id["FRVP_SETUP"]
    assert frvp_setup.label == "FRVP Setup Models"
    assert frvp_setup.enable_frvp_overlays is True
    assert frvp_setup.resolved_runtime_state_key == "FRVP"
    assert frvp_setup.long_runtime_manifest_path.parent.name == "frvp_es_setup_family_20260829"
    assert frvp_setup.registry_path is not None
    assert frvp_setup.registry_path.name == "frvp_es_primary_model_registry_long_setup_fullspan_20260829.json"
    assert "frvp_long_reversal_setup4_xgb_v1" in frvp_setup.preferred_model_order
    assert frvp_setup.active_weight_model_ids == ()

    ict_setup = view_by_id["ICT_SETUP"]
    assert ict_setup.label == "ICT Setup Models"
    assert ict_setup.enable_ict_overlays is True
    assert ict_setup.resolved_runtime_state_key == "ICT"
    assert ict_setup.short_runtime_manifest_path.parent.name == "ict_short_setup_family_20260811_audit"
    assert ict_setup.registry_path is not None
    assert ict_setup.registry_path.name == "ict_short_setup_family_model_registry_20260811_audit.json"
    assert "ict_short_reversal_sweep_reclaim_xgb_v1" in ict_setup.preferred_model_order
    assert ict_setup.active_weight_model_ids == ()


def test_setup_model_route_parses_real_frvp_and_ict_model_ids() -> None:
    frvp_route = infer_setup_model_route("frvp_long_continuation_setup2_xgb_v1")
    assert frvp_route is not None
    assert frvp_route.strategy == "FRVP"
    assert frvp_route.setup_id == 2
    assert frvp_route.setup_label == "S2"
    assert frvp_route.expected_side == 1

    ict_route = infer_setup_model_route("ict_short_reversal_sweep_reclaim_xgb_v1")
    assert ict_route is not None
    assert ict_route.strategy == "ICT"
    assert ict_route.setup_type == "sweep_reclaim"
    assert ict_route.setup_label == "Sweep Reclaim"
    assert ict_route.expected_side == -1

    assert infer_setup_model_route("ict_short_reversal_xgb_v1") is None


def test_frvp_setup_family_gate_matches_only_the_current_setup_and_side() -> None:
    policy_frame = pd.DataFrame(
        [
            {
                "frvp_setup_type": 2,
                "frvp_setup_side": 1,
                "frvp_setup_confidence_rule": 0.78,
            }
        ]
    )

    matched = resolve_setup_family_gate(
        "frvp_long_continuation_setup2_xgb_v1",
        policy_frame=policy_frame,
    )
    assert matched.matched is True
    assert matched.forced_hold_reasons == ()

    mismatch = resolve_setup_family_gate(
        "frvp_long_reversal_setup1_xgb_v1",
        policy_frame=policy_frame,
    )
    assert mismatch.matched is False
    assert mismatch.forced_hold_reasons == ("setup_family_gate_mismatch",)

    no_fire = resolve_setup_family_gate(
        "frvp_long_continuation_setup2_xgb_v1",
        policy_frame=pd.DataFrame([{"frvp_setup_type": 0, "frvp_setup_side": 0}]),
    )
    assert no_fire.matched is False
    assert no_fire.forced_hold_reasons == ("setup_family_gate_no_setup_fire",)


def test_ict_setup_family_gate_uses_detector_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ote_live.models.setup_family.detect_ict_setups",
        lambda frame: pd.DataFrame(
            [
                {
                    "fired": True,
                    "setup_type": "sweep_reclaim",
                    "setup_family": "reversal",
                    "setup_side": -1,
                    "confidence": 0.72,
                }
            ],
            index=frame.index,
        ),
    )
    policy_frame = pd.DataFrame([{"close": 6280.0}])

    matched = resolve_setup_family_gate(
        "ict_short_reversal_sweep_reclaim_xgb_v1",
        policy_frame=policy_frame,
    )
    assert matched.matched is True

    mismatch = resolve_setup_family_gate(
        "ict_short_reversal_ifvg_reversal_xgb_v1",
        policy_frame=policy_frame,
    )
    assert mismatch.matched is False
    assert mismatch.forced_hold_reasons == ("setup_family_gate_mismatch",)


def test_decision_engine_force_hold_preserves_probability_and_threshold() -> None:
    manifest = load_live_runtime_manifest(DEFAULT_LONG_MODEL_MANIFEST_PATH)
    live_policy = manifest.live_policy.model_copy(
        update={
            "thresholds": manifest.live_policy.thresholds.model_copy(
                update={"global_threshold": 0.50, "regime_thresholds": None}
            )
        }
    )
    prediction = ModelPrediction(
        model_id=manifest.model_id,
        direction=manifest.direction,
        backend=manifest.backend,
        timestamp=datetime(2026, 9, 2, 15, 0, tzinfo=UTC),
        source_row_idx=10,
        raw_score=0.84,
        calibrated_probability=0.84,
    )

    path = LiveDecisionEngine().evaluate_prediction(
        prediction,
        live_policy=live_policy,
        force_hold_reasons=("setup_family_gate_mismatch",),
    )

    assert path.signal.decision == "hold"
    assert path.signal.probability == pytest.approx(0.84)
    assert path.signal.threshold == pytest.approx(0.50)
    assert path.signal.reasons[:2] == [
        "setup_family_gate_mismatch",
        "candidate_passed_threshold",
    ]


def test_direction_loader_isolates_non_import_model_load_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_live_runtime_manifest(DEFAULT_LONG_MODEL_MANIFEST_PATH)
    direction_manifest = DirectionRuntimeManifest(
        generated_at_utc=manifest.generated_at_utc,
        registry_path=manifest.registry_path,
        policy_backtest_summary_path=manifest.policy_backtest_summary_path,
        direction=manifest.direction,
        asset=manifest.asset,
        timeframe=manifest.timeframe,
        recommendations={"recommended_primary_model_id": None},
        models=[manifest],
    )

    def fail_load(*_args: object, **_kwargs: object) -> object:
        raise ValueError("simulated bad artifact")

    monkeypatch.setattr("ote_live.models.ensemble.load_runtime_model", fail_load)

    skipped = load_direction_models(
        direction_manifest,
        skip_unavailable_backends=True,
    )
    assert skipped.loaded_models == {}
    assert skipped.unavailable_models[manifest.model_id] == "ValueError: simulated bad artifact"

    with pytest.raises(ValueError, match="simulated bad artifact"):
        load_direction_models(
            direction_manifest,
            skip_unavailable_backends=False,
        )


def test_signal_processor_records_unavailable_model_health_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = SimpleNamespace(
        direction_manifest=SimpleNamespace(
            recommendations=SimpleNamespace(recommended_primary_model_id=None),
            models=(),
        ),
        primary_model=None,
        loaded_models={},
        unavailable_models={"frvp_long_reversal_setup4_xgb_v1": "ValueError: missing artifact"},
    )
    monkeypatch.setattr(
        "ote_live.ingestion.signals.load_direction_models",
        lambda *_args, **_kwargs: bundle,
    )

    with SQLiteLiveDataStore(tmp_path / "setup-family-health.sqlite") as store:
        audit = LiveAuditRepository(store)
        processor = LiveSignalProcessor.from_direction_manifest_paths(
            audit_repository=audit,
            long_manifest_path="fake-setup-family-long.json",
            short_manifest_path=None,
            group_name="FRVP",
            data_supplier="IBKR",
        )
        events = audit.fetch_health_events(
            component="signal_runtime.models",
            event_type="runtime_model_unavailable",
        )

    assert processor is None
    assert len(events) == 1
    assert events[0].severity == "warning"
    assert events[0].payload["unavailable_models"] == {
        "frvp_long_reversal_setup4_xgb_v1": "ValueError: missing artifact"
    }

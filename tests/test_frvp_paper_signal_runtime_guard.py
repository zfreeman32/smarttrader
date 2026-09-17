from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.ingestion.runtime import DEFAULT_DB_PATH
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.scripts.run_es_live_collector import (
    ES_SHARED_DEFAULT_HEARTBEAT_PATH,
    ES_SHARED_DEFAULT_SERVICE_NAME,
    _validate_frvp_paper_signal_runtime,
)
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from scripts.audit_frvp_paper_signal_readiness import (
    BUNDLE_ID,
    DEFAULT_BUNDLE_DIR,
    EXPECTED_MODEL_IDS,
)


def _args(**overrides):
    values = {
        "include_frvp": True,
        "frvp_long_runtime_manifest_path": str(
            DEFAULT_BUNDLE_DIR / "live_runtime_manifest_long.json"
        ),
        "frvp_short_runtime_manifest_path": str(
            DEFAULT_BUNDLE_DIR / "live_runtime_manifest_short.json"
        ),
        "all_models_active": False,
        "frvp_paper_signal_trial_enabled": True,
        "ibkr_enabled": True,
        "ibkr_account_mode": "paper",
        "ibkr_port": 4002,
        "ibkr_allow_delayed_fallback": False,
        "enable_signal_runtime": True,
        "asset": "ES",
        "data_supplier": "IBKR",
        "source_timeframe": "5m",
        "ibkr_market_data_type": "live",
        "ibkr_what_to_show": "TRADES",
        "ibkr_use_rth": False,
        "ibkr_bar_size": "5 mins",
        "ibkr_keep_up_to_date": True,
        "ibkr_symbol": "ES",
        "ibkr_security_type": "FUT",
        "ibkr_exchange": "CME",
        "ibkr_currency": "USD",
        "ibkr_multiplier": "50",
        "ibkr_trading_class": "ES",
        "group_name": "ES_SHARED",
        "service_name": ES_SHARED_DEFAULT_SERVICE_NAME,
        "heartbeat_file": str(ES_SHARED_DEFAULT_HEARTBEAT_PATH),
        "db_path": str(DEFAULT_DB_PATH),
        "max_cycles": None,
        "allow_frvp_clean_handoff": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ready_audit(**_kwargs):
    return {
        "bundle_id": BUNDLE_ID,
        "status": "ready_to_start",
        "ready_to_start": True,
        "blocking_reasons": [],
        "facts": {
            "bundle_dir": str(DEFAULT_BUNDLE_DIR.resolve()),
            "model_ids": sorted(EXPECTED_MODEL_IDS),
            "heartbeat": {"acceptance": "clean_stopped_handoff"},
        },
    }


def test_frvp_paper_signal_runtime_accepts_exact_ready_paper_contract() -> None:
    authorization = _validate_frvp_paper_signal_runtime(
        _args(), readiness_audit=_ready_audit
    )
    assert authorization is not None


def test_frvp_shadow_bundle_does_not_require_trial_switch() -> None:
    authorization = _validate_frvp_paper_signal_runtime(
        _args(
            frvp_long_runtime_manifest_path=(
                "ote_live/runtime_manifests/frvp_es_shadow_20260721/"
                "live_runtime_manifest_long.json"
            ),
            frvp_short_runtime_manifest_path=(
                "ote_live/runtime_manifests/frvp_es_shadow_20260721/"
                "live_runtime_manifest_short.json"
            ),
            frvp_paper_signal_trial_enabled=False,
        ),
        readiness_audit=lambda **_kwargs: pytest.fail("shadow must not run readiness"),
    )
    assert authorization is None


def test_frvp_candidate_only_bundle_still_rejects_all_models_active() -> None:
    with pytest.raises(ValueError, match="ALL_MODELS_ACTIVE=false"):
        _validate_frvp_paper_signal_runtime(
            _args(
                frvp_long_runtime_manifest_path=(
                    "ote_live/runtime_manifests/frvp_es_shadow_20260721/"
                    "live_runtime_manifest_long.json"
                ),
                frvp_short_runtime_manifest_path=(
                    "ote_live/runtime_manifests/frvp_es_shadow_20260721/"
                    "live_runtime_manifest_short.json"
                ),
                frvp_paper_signal_trial_enabled=False,
                all_models_active=True,
            ),
            readiness_audit=lambda **_kwargs: pytest.fail(
                "candidate-only all-models override must fail before readiness"
            ),
        )


def test_frvp_paper_signal_runtime_rejects_disabled_switch() -> None:
    with pytest.raises(ValueError, match="launch switch is disabled"):
        _validate_frvp_paper_signal_runtime(
            _args(frvp_paper_signal_trial_enabled=False)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("all_models_active", True, "ALL_MODELS_ACTIVE=false"),
        ("ibkr_enabled", False, "IBKR_ENABLED=true"),
        ("ibkr_account_mode", "live", "IBKR_ACCOUNT_MODE=paper"),
        ("ibkr_port", 4001, "paper port"),
        ("ibkr_allow_delayed_fallback", True, "ALLOW_DELAYED_FALLBACK=false"),
        ("enable_signal_runtime", False, "ENABLE_SIGNAL_RUNTIME=true"),
        ("asset", "NQ", "asset ES"),
        ("data_supplier", "FMP", "data supplier IBKR"),
        ("source_timeframe", "1m", "source timeframe 5m"),
        ("ibkr_market_data_type", "delayed", "live IBKR market data"),
        ("ibkr_what_to_show", "MIDPOINT", "WHAT_TO_SHOW=TRADES"),
        ("ibkr_use_rth", True, "USE_RTH=false"),
        ("ibkr_bar_size", "1 min", "BAR_SIZE='5 mins'"),
        ("ibkr_keep_up_to_date", False, "KEEP_UP_TO_DATE=true"),
        ("ibkr_symbol", "NQ", "ibkr_symbol=ES"),
        ("ibkr_security_type", "STK", "ibkr_security_type=FUT"),
        ("ibkr_exchange", "SMART", "ibkr_exchange=CME"),
        ("ibkr_currency", "EUR", "ibkr_currency=USD"),
        ("ibkr_multiplier", "20", "ibkr_multiplier=50"),
        ("ibkr_trading_class", "MES", "ibkr_trading_class=ES"),
        ("service_name", "custom-es-service", "service name"),
        (
            "heartbeat_file",
            "ote_live/runtime_data/health/custom-heartbeat.json",
            "heartbeat path",
        ),
        ("db_path", "tmp/custom-frvp.sqlite3", "database path"),
        ("max_cycles", 1, "max_cycles"),
        ("allow_frvp_clean_handoff", False, "clean-stopped collector handoff"),
    ),
)
def test_frvp_paper_signal_runtime_rejects_unsafe_launch_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_frvp_paper_signal_runtime(_args(**{field: value}))


def test_frvp_active_bundle_copy_cannot_bypass_exact_path_guard(tmp_path: Path) -> None:
    copied = tmp_path / "copied_active_bundle"
    copied.mkdir()
    for direction in ("long", "short"):
        source = DEFAULT_BUNDLE_DIR / f"live_runtime_manifest_{direction}.json"
        shutil.copyfile(source, copied / source.name)

    with pytest.raises(ValueError, match="locked to the exact controlled bundle"):
        _validate_frvp_paper_signal_runtime(
            _args(
                frvp_long_runtime_manifest_path=copied / "live_runtime_manifest_long.json",
                frvp_short_runtime_manifest_path=copied / "live_runtime_manifest_short.json",
            )
        )


def test_frvp_runtime_requires_full_ready_contract() -> None:
    def blocked_audit(**_kwargs):
        result = _ready_audit()
        result.update(
            {
                "status": "blocked",
                "ready_to_start": False,
                "blocking_reasons": [
                    {"code": "historical_feature_parity_not_ready", "message": "blocked"}
                ],
            }
        )
        return result

    with pytest.raises(ValueError, match="historical_feature_parity_not_ready"):
        _validate_frvp_paper_signal_runtime(_args(), readiness_audit=blocked_audit)


def test_frvp_runtime_passes_clean_handoff_only_when_explicit() -> None:
    observed: list[bool] = []

    def capture(**kwargs):
        observed.append(bool(kwargs["allow_clean_stopped_handoff"]))
        return _ready_audit()

    _validate_frvp_paper_signal_runtime(
        _args(allow_frvp_clean_handoff=True), readiness_audit=capture
    )
    with pytest.raises(ValueError, match="clean-stopped collector handoff"):
        _validate_frvp_paper_signal_runtime(
            _args(allow_frvp_clean_handoff=False), readiness_audit=capture
        )
    assert observed == [True]


def test_frvp_exclusion_skips_guard() -> None:
    authorization = _validate_frvp_paper_signal_runtime(
        _args(include_frvp=False),
        readiness_audit=lambda **_kwargs: pytest.fail("excluded FRVP must not audit"),
    )
    assert authorization is None


def test_active_frvp_processor_construction_requires_guard_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _reversal_manifest()
    loaded_model = _loaded_model(manifest)
    bundle = SimpleNamespace(
        direction_manifest=SimpleNamespace(
            recommendations=SimpleNamespace(
                recommended_primary_model_id=manifest.model_id
            ),
            models=(manifest,),
        ),
        primary_model=loaded_model,
        loaded_models={manifest.model_id: loaded_model},
        unavailable_models={},
    )
    monkeypatch.setattr(
        "ote_live.ingestion.signals.load_direction_models",
        lambda *_args, **_kwargs: bundle,
    )
    db_path = tmp_path / "authorized-construction.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        with pytest.raises(RuntimeError, match="authorization"):
            LiveSignalProcessor.from_direction_manifest_paths(
                audit_repository=audit,
                long_manifest_path="controlled-long.json",
                short_manifest_path=None,
                group_name="FRVP",
                data_supplier="IBKR",
            )

        authorization = _validate_frvp_paper_signal_runtime(
            _args(), readiness_audit=_ready_audit
        )
        processor = LiveSignalProcessor.from_direction_manifest_paths(
            audit_repository=audit,
            long_manifest_path="controlled-long.json",
            short_manifest_path=None,
            group_name="FRVP",
            data_supplier="IBKR",
            frvp_paper_signal_authorization=authorization,
        )
        assert processor is not None
        assert processor.frvp_paper_signal_ledger is not None
        assert [
            binding.loaded_model.model_id
            for binding in processor.bindings
            if not binding.shadow_mode
        ] == ["frvp_long_reversal_xgb_v1"]


def test_active_frvp_processor_rejects_hash_mismatch_even_with_authorization(
    tmp_path: Path,
) -> None:
    manifest = _reversal_manifest()
    payload = manifest.model_dump(mode="json")
    payload["live_policy"]["thresholds"]["global_threshold"] = 0.99
    altered_manifest = LiveRuntimeManifest.model_validate(payload)
    authorization = _validate_frvp_paper_signal_runtime(
        _args(), readiness_audit=_ready_audit
    )
    db_path = tmp_path / "hash-mismatch.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        with pytest.raises(ValueError, match="FRVP paper-signal contract"):
            LiveSignalProcessor(
                bindings=(
                    SignalRuntimeModelBinding(
                        loaded_model=_loaded_model(altered_manifest),
                        shadow_mode=False,
                    ),
                ),
                audit_repository=LiveAuditRepository(store),
                group_name="FRVP",
                data_supplier="IBKR",
                frvp_paper_signal_authorization=authorization,
            )


def test_frvp_cooldown_restores_across_processor_restarts_by_exact_manifest(
    tmp_path: Path,
) -> None:
    manifest = _reversal_manifest()
    authorization = _validate_frvp_paper_signal_runtime(
        _args(), readiness_audit=_ready_audit
    )
    db_path = tmp_path / "restart-cooldown.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        first = _active_processor(
            audit=audit,
            manifest=manifest,
            authorization=authorization,
        )
        assert _evaluate_at(first.decision_engine, manifest, 100).signal.decision == "emit"

        # A later emit with the same model_id but a different manifest hash must
        # not become cooldown state for the frozen active contract.
        altered_policy = manifest.live_policy.model_copy(
            update={
                "abstain_policy": manifest.live_policy.abstain_policy.model_copy(
                    update={"enabled": False}
                )
            }
        )
        altered_manifest = manifest.model_copy(update={"live_policy": altered_policy})
        unrelated_engine = LiveDecisionEngine(
            audit_repository=audit,
            persist_decisions=("emit",),
        )
        assert _evaluate_at(unrelated_engine, altered_manifest, 102).signal.decision == "emit"

        restarted_inside = _active_processor(
            audit=audit,
            manifest=manifest,
            authorization=authorization,
        )
        inside = _evaluate_at(restarted_inside.decision_engine, manifest, 103)
        assert inside.signal.decision == "abstain"
        assert inside.signal.reasons == ["cooldown"]
        assert inside.signal.cooldown_bars_remaining == 2

        restarted_beyond = _active_processor(
            audit=audit,
            manifest=manifest,
            authorization=authorization,
        )
        beyond = _evaluate_at(restarted_beyond.decision_engine, manifest, 105)
        assert beyond.signal.decision == "emit"


def _reversal_manifest() -> LiveRuntimeManifest:
    return load_live_runtime_manifest(
        DEFAULT_BUNDLE_DIR
        / "frvp_long_reversal_xgb_v1"
        / "live_runtime_manifest.json"
    )


def _loaded_model(manifest: LiveRuntimeManifest) -> LoadedRuntimeModel:
    return LoadedRuntimeModel(
        manifest=manifest,
        model=object(),
        scaler=None,
        calibrator=None,
        training_summary={},
        scale_clip=8.0,
        batch_size=256,
        use_amp=False,
    )


def _active_processor(
    *,
    audit: LiveAuditRepository,
    manifest: LiveRuntimeManifest,
    authorization: object,
) -> LiveSignalProcessor:
    return LiveSignalProcessor(
        bindings=(
            SignalRuntimeModelBinding(
                loaded_model=_loaded_model(manifest),
                shadow_mode=False,
            ),
        ),
        audit_repository=audit,
        group_name="FRVP",
        data_supplier="IBKR",
        frvp_paper_signal_authorization=authorization,
    )


def _evaluate_at(
    engine: LiveDecisionEngine,
    manifest: LiveRuntimeManifest,
    source_row_idx: int,
):
    timestamp = datetime(2026, 8, 17, tzinfo=UTC) + timedelta(
        minutes=5 * source_row_idx
    )
    prediction = ModelPrediction(
        model_id=manifest.model_id,
        direction=manifest.direction,
        backend=manifest.backend,
        timestamp=timestamp,
        source_row_idx=source_row_idx,
        regime="strong_down_medium",
        raw_score=0.95,
        calibrated_probability=0.95,
    )
    snapshot = FeatureSnapshot(
        asset=manifest.asset,
        timeframe=manifest.timeframe,
        direction=manifest.direction,
        timestamp=timestamp,
        source_row_idx=source_row_idx,
        feature_values={},
        valid_feature_count=0,
    )
    return engine.evaluate_prediction(
        prediction,
        live_policy=manifest.live_policy,
        policy_context={
            "trend_regime": "strong_down",
            "vol_regime": "medium",
            "session_regime": "overlap",
            "stress_regime": "normal",
            "composite_regime": "strong_down_medium",
            "expected_move_pips": 100.0,
        },
        timezone_contract=manifest.timezone_contract,
        shadow_mode=False,
        feature_snapshot=snapshot,
        runtime_manifest=manifest,
    )

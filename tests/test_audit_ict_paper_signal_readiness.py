from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.audit_ict_paper_signal_readiness import (
    EXPECTED_POLICY_BACKTEST_SUMMARY_PATH,
    EXPECTED_REGISTRY_PATH,
    META_MODEL_ID,
    REVERSAL_MODEL_ID,
    audit_readiness,
    main,
)


NOW = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)


def test_ready_when_bundle_modes_and_collector_health_match_contract(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "ready_to_start"
    assert result["ready_to_start"] is True
    assert result["preflight_ready"] is True
    assert result["blocking_reasons"] == []
    assert result["facts"]["active_model_ids"] == [META_MODEL_ID]
    assert result["facts"]["reversal_status"] == "candidate"
    assert result["facts"]["meta_global_threshold"] == 0.4
    assert result["facts"]["meta_abstain_enabled"] is False
    assert result["safety"]["connects_to_ibkr"] is False


def test_preflight_reports_only_disabled_switch_as_activation_step(
    tmp_path: Path,
    capsys,
) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED=true",
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED=false",
        ),
        encoding="utf-8",
    )

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
        preflight=True,
    )

    assert result["status"] == "ready_except_enable_switch"
    assert result["ready_to_start"] is False
    assert result["preflight_ready"] is True
    assert result["blocking_reasons"] == []
    assert {item["code"] for item in result["activation_requirements"]} == {
        "trial_enable_switch_disabled"
    }

    exit_code = main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--env-file",
            str(env_path),
            "--heartbeat-file",
            str(heartbeat_path),
            "--max-heartbeat-age-seconds",
            "999999999",
            "--preflight",
        ]
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_payload["status"] == "ready_except_enable_switch"


def test_blocks_extra_active_model_and_non_candidate_reversal(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    long_path = bundle_dir / "live_runtime_manifest_long.json"
    payload = json.loads(long_path.read_text(encoding="utf-8"))
    reversal = next(model for model in payload["models"] if model["model_id"] == REVERSAL_MODEL_ID)
    reversal["status"] = "active"
    long_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert _reason_codes(result) >= {
        "active_model_roster_mismatch",
        "reversal_status_mismatch",
    }


def test_blocks_policy_drift_from_global_point_four_no_abstain(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    long_path = bundle_dir / "live_runtime_manifest_long.json"
    payload = json.loads(long_path.read_text(encoding="utf-8"))
    meta = next(model for model in payload["models"] if model["model_id"] == META_MODEL_ID)
    meta["live_policy"]["thresholds"] = {
        "global_threshold": 0.39,
        "regime_thresholds": {"ranging_low": 0.4},
    }
    meta["live_policy"]["abstain_policy"]["enabled"] = True
    meta["live_policy"]["lineage"]["selected_policy_name"] = "regime_threshold_plus_abstain"
    long_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert _reason_codes(result) >= {
        "meta_threshold_mismatch",
        "meta_regime_thresholds_enabled",
        "meta_abstain_enabled",
        "meta_no_abstain_contract_mismatch",
        "meta_policy_name_mismatch",
    }


def test_blocks_any_drift_from_exact_six_model_roster(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    short_path = bundle_dir / "live_runtime_manifest_short.json"
    payload = json.loads(short_path.read_text(encoding="utf-8"))
    payload["models"].append(
        _model("ict_short_uncontrolled_xgb_v1", "candidate", direction="short")
    )
    short_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert "bundle_model_roster_mismatch" in _reason_codes(result)


def test_blocks_status_drift_for_any_of_the_six_models(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    short_path = bundle_dir / "live_runtime_manifest_short.json"
    payload = json.loads(short_path.read_text(encoding="utf-8"))
    payload["models"][0]["status"] = "deprecated"
    short_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert "bundle_model_status_contract_mismatch" in _reason_codes(result)


def test_blocks_manifest_bundle_identity_drift(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    long_path = bundle_dir / "live_runtime_manifest_long.json"
    payload = json.loads(long_path.read_text(encoding="utf-8"))
    payload["registry_path"] = "models/renamed_registry.json"
    long_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert "bundle_manifest_identity_mismatch" in _reason_codes(result)


def test_blocks_all_models_override_and_live_ibkr_mode_without_reporting_secrets(
    tmp_path: Path,
) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    env_path.write_text(
        "\n".join(
            (
                "ES_LIVE_ALL_MODELS_ACTIVE=true",
                "IBKR_ENABLED=true",
                "IBKR_ACCOUNT_MODE=live",
                "IBKR_PORT=4001",
                "IBKR_ALLOW_DELAYED_FALLBACK=false",
                "ICT_PAPER_SIGNAL_TRIAL_ENABLED=true",
                "IBKR_ACCOUNT=do-not-report",
                "IBKR_PASSWORD=do-not-report",
            )
        ),
        encoding="utf-8",
    )

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )
    serialized = json.dumps(result)

    assert result["status"] == "blocked"
    assert _reason_codes(result) >= {
        "all_models_active_enabled",
        "ibkr_not_paper_mode",
        "ibkr_not_paper_endpoint",
    }
    assert "do-not-report" not in serialized
    assert "IBKR_PASSWORD" not in serialized


def test_blocks_disabled_ibkr_and_delayed_fallback(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    env_path.write_text(
        "ES_LIVE_ALL_MODELS_ACTIVE=false\n"
        "IBKR_ENABLED=false\n"
        "IBKR_ACCOUNT_MODE=paper\n"
        "IBKR_PORT=4002\n"
        "IBKR_ALLOW_DELAYED_FALLBACK=true\n"
        "ICT_PAPER_SIGNAL_TRIAL_ENABLED=true\n",
        encoding="utf-8",
    )

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert _reason_codes(result) >= {
        "ibkr_disabled",
        "ibkr_delayed_fallback_enabled",
    }


def test_blocks_old_or_unhealthy_heartbeat(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    heartbeat_path.write_text(
        json.dumps(
            {
                "generated_at_utc": (NOW - timedelta(hours=1)).isoformat(),
                "health_state": "unhealthy",
                "service_status": "failed",
                "heartbeat_is_stale": True,
            }
        ),
        encoding="utf-8",
    )

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
        max_heartbeat_age_seconds=900,
    )

    assert result["status"] == "blocked"
    assert _reason_codes(result) >= {
        "heartbeat_too_old",
        "heartbeat_unhealthy",
        "collector_not_running",
        "collector_source_stale",
    }


def test_default_audit_rejects_fresh_clean_stopped_heartbeat(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    payload["service_status"] = "stopped"
    heartbeat_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert result["facts"]["heartbeat"]["clean_stopped_handoff_accepted"] is False
    assert "collector_not_running" in _reason_codes(result)


def test_explicit_handoff_accepts_only_fresh_clean_stopped_heartbeat(
    tmp_path: Path,
) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    payload["service_status"] = "stopped"
    heartbeat_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
        max_heartbeat_age_seconds=999999999,
        allow_clean_stopped_handoff=True,
    )

    assert result["status"] == "ready_to_start"
    assert result["ready_to_start"] is True
    assert result["facts"]["heartbeat"]["acceptance"] == "clean_stopped_handoff"
    assert result["facts"]["heartbeat"]["clean_stopped_handoff_accepted"] is True

    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED=true",
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED=false",
        ),
        encoding="utf-8",
    )
    preflight = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
        preflight=True,
        allow_clean_stopped_handoff=True,
    )
    assert preflight["status"] == "ready_except_enable_switch"


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    (
        ({"service_status": "cancelled"}, "collector_not_running"),
        ({"service_status": "failed"}, "collector_not_running"),
        (
            {"service_status": "stopped", "health_state": "unhealthy"},
            "heartbeat_unhealthy",
        ),
        (
            {"service_status": "stopped", "heartbeat_is_stale": True},
            "collector_source_stale",
        ),
        (
            {"service_status": "stopped", "service_name": "wrong-service"},
            "heartbeat_identity_mismatch",
        ),
        (
            {
                "service_status": "stopped",
                "generated_at_utc": (NOW - timedelta(seconds=121)).isoformat(),
            },
            "clean_handoff_too_old",
        ),
    ),
)
def test_explicit_handoff_rejects_nonclean_or_wrong_heartbeat(
    tmp_path: Path,
    overrides: dict,
    expected_reason: str,
) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    payload.update(overrides)
    heartbeat_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
        max_heartbeat_age_seconds=999999999,
        allow_clean_stopped_handoff=True,
    )

    assert result["status"] == "blocked"
    assert result["ready_to_start"] is False
    assert result["facts"]["heartbeat"]["clean_stopped_handoff_accepted"] is False
    assert expected_reason in _reason_codes(result)


def test_blocks_failed_feature_parity_and_missing_confirmation_ledger(
    tmp_path: Path,
) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    validation_path = bundle_dir / "paper_signal_validation_summary.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["historical_feature_parity"]["status"] = "fail"
    payload["paper_signal_ledger"]["status"] = "not_ready"
    payload["trial_start_authorized"] = False
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert _reason_codes(result) >= {
        "historical_feature_parity_not_passed",
        "paper_signal_ledger_not_ready",
        "trial_start_not_authorized",
    }


def test_accepts_documented_dead_inputs_for_active_meta_parity(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    validation_path = bundle_dir / "paper_signal_validation_summary.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["historical_feature_parity"]["status"] = "pass_with_documented_dead_inputs"
    payload["active_meta_dead_input_audit"] = {"status": "pass"}
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "ready_to_start"
    assert result["facts"]["validation"]["historical_feature_parity"] == (
        "pass_with_documented_dead_inputs"
    )


def test_blocks_documented_dead_inputs_without_active_meta_audit(tmp_path: Path) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    validation_path = bundle_dir / "paper_signal_validation_summary.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["historical_feature_parity"]["status"] = "pass_with_documented_dead_inputs"
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        bundle_dir=bundle_dir,
        env_path=env_path,
        heartbeat_path=heartbeat_path,
        environ={},
        now=NOW,
    )

    assert result["status"] == "blocked"
    assert "active_meta_dead_input_audit_not_passed" in _reason_codes(result)


def test_cli_emits_blocked_json_and_nonzero_status(tmp_path: Path, capsys) -> None:
    bundle_dir, env_path, heartbeat_path = _write_ready_inputs(tmp_path)
    env_path.write_text("IBKR_ACCOUNT_MODE=live\n", encoding="utf-8")

    exit_code = main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--env-file",
            str(env_path),
            "--heartbeat-file",
            str(heartbeat_path),
            "--max-heartbeat-age-seconds",
            "999999999",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert "ibkr_not_paper_mode" in _reason_codes(payload)


def _write_ready_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle_dir = tmp_path / "ict_es_paper_signal_20260813"
    bundle_dir.mkdir()
    (bundle_dir / "live_runtime_manifest_long.json").write_text(
        json.dumps(
            {
                "direction": "long",
                "asset": "ES",
                "timeframe": "5m",
                "registry_path": EXPECTED_REGISTRY_PATH,
                "policy_backtest_summary_path": EXPECTED_POLICY_BACKTEST_SUMMARY_PATH,
                "models": [
                    _model(META_MODEL_ID, "active", direction="long", policy=_meta_policy()),
                    _model(REVERSAL_MODEL_ID, "candidate", direction="long"),
                    _model("ict_long_continuation_xgb_v1", "candidate", direction="long"),
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "live_runtime_manifest_short.json").write_text(
        json.dumps(
            {
                "direction": "short",
                "asset": "ES",
                "timeframe": "5m",
                "registry_path": EXPECTED_REGISTRY_PATH,
                "policy_backtest_summary_path": EXPECTED_POLICY_BACKTEST_SUMMARY_PATH,
                "models": [
                    _model("ict_short_meta_xgb_v1", "candidate", direction="short"),
                    _model("ict_short_reversal_xgb_v1", "candidate", direction="short"),
                    _model("ict_short_continuation_xgb_v1", "candidate", direction="short"),
                ],
            }
        ),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ES_LIVE_ALL_MODELS_ACTIVE=false\n"
        "IBKR_ENABLED=true\n"
        "IBKR_ACCOUNT_MODE=paper\n"
        "IBKR_PORT=4002\n"
        "IBKR_ALLOW_DELAYED_FALLBACK=false\n"
        "ICT_PAPER_SIGNAL_TRIAL_ENABLED=true\n",
        encoding="utf-8",
    )
    (bundle_dir / "paper_signal_validation_summary.json").write_text(
        json.dumps(
            {
                "bundle_id": "ict_es_paper_signal_20260813",
                "artifact_prediction_parity": {"status": "pass"},
                "historical_feature_parity": {"status": "pass"},
                "runtime_policy_regressions": {"status": "pass"},
                "paper_signal_ledger": {"status": "ready"},
                "trial_start_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "generated_at_utc": (NOW - timedelta(minutes=2)).isoformat(),
                "service_name": "es-shared-live-signal-service",
                "health_state": "healthy",
                "service_status": "running",
                "heartbeat_is_stale": False,
                "asset": "ES",
                "source_timeframe": "5m",
                "signal_timeframe": "5m",
            }
        ),
        encoding="utf-8",
    )
    return bundle_dir, env_path, heartbeat_path


def _model(
    model_id: str,
    status: str,
    *,
    direction: str,
    policy: dict | None = None,
) -> dict:
    return {
        "model_id": model_id,
        "direction": direction,
        "asset": "ES",
        "timeframe": "5m",
        "registry_path": EXPECTED_REGISTRY_PATH,
        "policy_backtest_summary_path": EXPECTED_POLICY_BACKTEST_SUMMARY_PATH,
        "status": status,
        "live_policy": policy or {
            "policy_status": "complete",
            "thresholds": {"global_threshold": 0.5, "regime_thresholds": None},
            "abstain_policy": _disabled_abstain_policy(),
            "lineage": {"selected_policy_name": "global_threshold"},
        },
    }


def _meta_policy() -> dict:
    return {
        "policy_status": "complete",
        "thresholds": {"global_threshold": 0.4, "regime_thresholds": None},
        "abstain_policy": _disabled_abstain_policy(),
        "lineage": {"selected_policy_name": "global_threshold"},
    }


def _disabled_abstain_policy() -> dict:
    return {
        "enabled": False,
        "abstain_high_stress": False,
        "abstain_off_hours": False,
        "cooldown_bars": 0,
        "minimum_expected_move_to_spread": 0.0,
        "abstain_session_regimes": [],
        "abstain_composite_regimes": [],
        "abstain_composite_session_pairs": [],
        "abstain_composite_stress_pairs": [],
        "minimum_probability_quantile": None,
    }


def _reason_codes(result: dict) -> set[str]:
    return {reason["code"] for reason in result["blocking_reasons"]}

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_frvp_paper_signal_readiness import (
    DEFAULT_BUNDLE_DIR,
    audit_readiness,
)
from ote_live.ingestion.connector_ibkr import IBKRRuntimeConfig
from ote_live.ingestion.runtime import LiveCollectorConfig
from ote_live.scripts.run_es_live_collector import _build_es_collector_contract


NOW = datetime(2026, 8, 16, 14, 0, tzinfo=UTC)


def _environment(*, enabled: bool) -> dict[str, str]:
    return {
        "ES_LIVE_ALL_MODELS_ACTIVE": "false",
        "ES_LIVE_ENABLE_SIGNAL_RUNTIME": "true",
        "FRVP_LIVE_DATA_SUPPLIER": "IBKR",
        "FRVP_LIVE_ASSET": "ES",
        "FRVP_LIVE_SOURCE_TIMEFRAME": "5m",
        "FRVP_PAPER_SIGNAL_TRIAL_ENABLED": str(enabled).lower(),
        "IBKR_ENABLED": "true",
        "IBKR_ACCOUNT_MODE": "paper",
        "IBKR_PORT": "4002",
        "IBKR_ALLOW_DELAYED_FALLBACK": "false",
        "IBKR_MARKET_DATA_TYPE": "live",
        "IBKR_WHAT_TO_SHOW": "TRADES",
        "IBKR_ES_USE_RTH": "false",
        "IBKR_ES_BAR_SIZE": "5 mins",
        "IBKR_KEEP_UP_TO_DATE": "true",
        "IBKR_ES_SYMBOL": "ES",
        "IBKR_ES_SECURITY_TYPE": "FUT",
        "IBKR_ES_EXCHANGE": "CME",
        "IBKR_ES_CURRENCY": "USD",
        "IBKR_ES_MULTIPLIER": "50",
        "IBKR_ES_TRADING_CLASS": "ES",
    }


def _write_heartbeat(
    path: Path,
    *,
    status: str = "running",
    health: str = "healthy",
    stale: bool = False,
    age_seconds: float = 30.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": (NOW - timedelta(seconds=age_seconds)).isoformat(),
                "service_name": "es-shared-live-signal-service",
                "service_status": status,
                "health_state": health,
                "heartbeat_is_stale": stale,
                "asset": "ES",
                "source_timeframe": "5m",
                "signal_timeframe": "5m",
                "latest_heartbeat_source": "ibkr.historical.polling",
                "runtime": {
                    "terminal_status": (
                        "completed" if status == "stopped" else "running"
                    ),
                    "collector_contract": {
                        "schema_version": 1,
                        "data_supplier": "IBKR",
                        "heartbeat_source": "ibkr.historical.polling",
                        "signal_runtime_enabled": True,
                        "signal_runtime_loaded": True,
                        "all_models_active": False,
                        "asset": "ES",
                        "source_timeframe": "5m",
                        "ibkr_enabled": True,
                        "ibkr_account_mode": "paper",
                        "ibkr_market_data_type_requested": "live",
                        "ibkr_market_data_type_received": "live",
                        "ibkr_connection_state": "connected",
                        "ibkr_allow_delayed_fallback": False,
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
                        "active_model_ids": [],
                        "feed": {
                            "data_supplier": "IBKR",
                            "heartbeat_source": "ibkr.historical.polling",
                            "asset": "ES",
                            "source_timeframe": "5m",
                            "ibkr_enabled": True,
                            "account_mode": "paper",
                            "market_data_type_requested": "live",
                            "market_data_type_received": "live",
                            "connection_state": "connected",
                            "allow_delayed_fallback": False,
                            "what_to_show": "TRADES",
                            "use_rth": False,
                            "bar_size": "5 mins",
                            "keep_up_to_date": True,
                            "contract": {
                                "symbol": "ES",
                                "security_type": "FUT",
                                "exchange": "CME",
                                "currency": "USD",
                                "multiplier": "50",
                                "trading_class": "ES",
                            },
                        },
                        "signal_runtime": {
                            "enabled": True,
                            "loaded": True,
                            "all_models_active": False,
                            "loaded_model_ids": [
                                "frvp_long_continuation_xgb_v1"
                            ],
                            "active_model_ids": [],
                            "shadow_model_ids": [
                                "frvp_long_continuation_xgb_v1"
                            ],
                            "registry_paths": [
                                "models/frvp_es_shadow_live_registry_20260721.json"
                            ],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_frvp_readiness_reaches_preflight_then_final_ready(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat)

    preflight = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=False),
        now=NOW,
        preflight=True,
    )
    assert preflight["status"] == "ready_except_enable_switch"
    assert preflight["blocking_reasons"] == []
    assert [item["code"] for item in preflight["activation_requirements"]] == [
        "enable_trial_switch"
    ]

    final = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
    )
    assert final["status"] == "ready_to_start"
    assert final["ready_to_start"] is True
    assert final["facts"]["expected_active_model_ids"] == [
        "frvp_long_reversal_xgb_v1"
    ]


def test_frvp_preflight_blocks_when_active_artifact_bytes_change(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(DEFAULT_BUNDLE_DIR, bundle_dir)
    aggregate_path = bundle_dir / "live_runtime_manifest_long.json"
    nested_path = (
        bundle_dir
        / "frvp_long_reversal_xgb_v1"
        / "live_runtime_manifest.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    active = next(
        model
        for model in aggregate["models"]
        if model["model_id"] == "frvp_long_reversal_xgb_v1"
    )
    source_model_path = ROOT / active["artifact_references"]["model_file"]
    mutated_model_path = tmp_path / "mutated-model.json"
    shutil.copyfile(source_model_path, mutated_model_path)
    with mutated_model_path.open("ab") as handle:
        handle.write(b"frvp-integrity-mutation")
    active["artifact_references"]["model_file"] = str(mutated_model_path)
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    nested_path.write_text(json.dumps(active), encoding="utf-8")

    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat)
    result = audit_readiness(
        bundle_dir=bundle_dir,
        heartbeat_path=heartbeat,
        environ=_environment(enabled=False),
        now=NOW,
        preflight=True,
    )

    assert result["status"] == "blocked"
    assert result["facts"]["active_artifact_integrity"]["all_match"] is False
    assert "active_artifact_hash_mismatch" in {
        item["code"] for item in result["blocking_reasons"]
    }


def test_frvp_readiness_accepts_clean_stop_only_in_explicit_handoff(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat, status="stopped", age_seconds=60.0)

    normal = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
    )
    assert "collector_not_running" in {
        item["code"] for item in normal["blocking_reasons"]
    }

    handoff = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
        allow_clean_stopped_handoff=True,
    )
    assert handoff["status"] == "ready_to_start"
    assert handoff["facts"]["heartbeat"]["acceptance"] == "clean_stopped_handoff"
    assert handoff["facts"]["heartbeat"]["feed_contract_matches"] is True


def test_frvp_clean_handoff_rejects_wrong_feed_contract(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat, status="stopped", age_seconds=60.0)
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    payload["runtime"]["collector_contract"]["ibkr_what_to_show"] = "MIDPOINT"
    payload["runtime"]["collector_contract"]["feed"]["what_to_show"] = "MIDPOINT"
    heartbeat.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
        allow_clean_stopped_handoff=True,
    )
    assert result["status"] == "blocked"
    assert result["ready_to_start"] is False
    assert result["facts"]["heartbeat"]["feed_contract_matches"] is False
    assert "heartbeat_feed_contract_mismatch" in {
        item["code"] for item in result["blocking_reasons"]
    }


def test_frvp_clean_handoff_rejects_missing_active_model_identity(
    tmp_path: Path,
) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat, status="stopped", age_seconds=60.0)
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    del payload["runtime"]["collector_contract"]["signal_runtime"][
        "active_model_ids"
    ]
    heartbeat.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
        allow_clean_stopped_handoff=True,
    )
    assert result["status"] == "blocked"
    assert "heartbeat_feed_contract_mismatch" in {
        item["code"] for item in result["blocking_reasons"]
    }


def test_frvp_readiness_accepts_real_collector_contract_builder_output(
    tmp_path: Path,
) -> None:
    manifest = SimpleNamespace(
        registry_path="models/frvp_es_shadow_live_registry_20260721.json"
    )
    binding = SimpleNamespace(
        loaded_model=SimpleNamespace(
            model_id="frvp_long_continuation_xgb_v1",
            manifest=manifest,
        ),
        shadow_mode=True,
    )
    processor = SimpleNamespace(
        bindings=(binding,),
        frvp_paper_signal_ledger=None,
    )
    config = LiveCollectorConfig(
        group_name="FRVP",
        data_supplier="IBKR",
        asset="ES",
        source_timeframe="5m",
        enable_signal_runtime=True,
        all_models_active=False,
        ibkr=IBKRRuntimeConfig(
            enabled=True,
            account_mode="paper",
            market_data_type="live",
            allow_delayed_fallback=False,
            symbol="ES",
            security_type="FUT",
            exchange="CME",
            currency="USD",
            multiplier="50",
            trading_class="ES",
            bar_size="5 mins",
            what_to_show="TRADES",
            use_rth=False,
            keep_up_to_date=True,
        ),
    )
    runtime = SimpleNamespace(
        config=config,
        signal_processor=processor,
        client=SimpleNamespace(
            service=SimpleNamespace(
                get_status=lambda: {
                    "market_data_type_received": "live",
                    "connection_state": "connected",
                }
            )
        ),
    )
    heartbeat = tmp_path / "builder-heartbeat.json"
    _write_heartbeat(heartbeat, status="stopped", age_seconds=60.0)
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    payload["runtime"]["collector_contract"] = _build_es_collector_contract(runtime)
    heartbeat.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
        allow_clean_stopped_handoff=True,
    )
    assert result["status"] == "ready_to_start"
    assert result["facts"]["heartbeat"]["feed_contract_matches"] is True


def test_frvp_clean_stop_older_than_120_seconds_remains_blocked(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat, status="stopped", age_seconds=121.0)
    result = audit_readiness(
        heartbeat_path=heartbeat,
        environ=_environment(enabled=True),
        now=NOW,
        allow_clean_stopped_handoff=True,
    )
    assert "clean_handoff_too_old" in {
        item["code"] for item in result["blocking_reasons"]
    }


def test_frvp_readiness_rejects_delayed_fallback(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat)
    environ = _environment(enabled=True)
    environ["IBKR_ALLOW_DELAYED_FALLBACK"] = "true"
    result = audit_readiness(
        heartbeat_path=heartbeat,
        environ=environ,
        now=NOW,
    )
    assert "delayed_fallback_enabled" in {
        item["code"] for item in result["blocking_reasons"]
    }


def test_materialized_reversal_policy_contains_restored_overlap_pair() -> None:
    payload = json.loads(
        (
            DEFAULT_BUNDLE_DIR
            / "frvp_long_reversal_xgb_v1"
            / "live_policy.json"
        ).read_text(encoding="utf-8")
    )
    pairs = {
        tuple(pair)
        for pair in payload["abstain_policy"]["abstain_composite_session_pairs"]
    }
    assert ("strong_down_high", "overlap") in pairs
    assert len(pairs) == 11

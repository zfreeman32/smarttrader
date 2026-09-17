from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from ote_live.dashboard.view_registry import (
    DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_FRVP_SETUP_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_FRVP_SETUP_SHORT_RUNTIME_MANIFEST_PATH,
    DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_SETUP_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_SETUP_SHORT_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_SHORT_RUNTIME_MANIFEST_PATH,
)
from ote_live.env import env_bool, env_float, env_int, env_path, env_str, load_repo_env
from ote_live.ingestion.connector_ibkr import IBKRRuntimeConfig
from ote_live.ingestion.ibkr.errors import IBKRError
from ote_live.ingestion.runtime import (
    DEFAULT_DB_PATH,
    DEFAULT_SIGNAL_CHART_OUTPUT_ROOT,
    LiveCollectorConfig,
    LiveCollectorRuntime,
)
from ote_live.ingestion.signals import (
    LiveSignalProcessor,
    MultiGroupLiveSignalProcessor,
    _issue_frvp_paper_signal_runtime_authorization,
)
from ote_live.ops import (
    DiskSpaceMonitor,
    ServiceHealthSnapshot,
    ServiceHeartbeatWriter,
    capture_startup_recovery_state,
    collect_service_health_snapshot,
    configure_live_logging,
    record_service_shutdown,
    record_service_started,
    record_service_starting,
)
from ote_live.scripts.run_live_collector import (
    _build_bootstrap_payload,
    _build_runtime_payload,
    _resolve_service_name,
    _service_status_for_summary,
    build_parser as build_base_parser,
)
from ote_live.storage.frvp_paper_signal import (
    FRVP_PAPER_SIGNAL_MODEL_IDS as FRVP_LEDGER_MODEL_IDS,
    supports_frvp_paper_signal_manifest,
)
from ote_live.storage.frvp_confirmation_lifecycle import (
    record_frvp_confirmation_start,
)
from scripts.audit_ict_paper_signal_readiness import (
    BUNDLE_ID as ICT_PAPER_SIGNAL_BUNDLE_ID,
    DEFAULT_BUNDLE_DIR as ICT_PAPER_SIGNAL_BUNDLE_DIR,
    DEFAULT_ENV_PATH as ICT_PAPER_SIGNAL_ENV_PATH,
    EXPECTED_MODEL_IDS as ICT_PAPER_SIGNAL_MODEL_IDS,
    audit_readiness as audit_ict_paper_signal_readiness,
)
from scripts.audit_frvp_paper_signal_readiness import (
    BUNDLE_ID as FRVP_PAPER_SIGNAL_BUNDLE_ID,
    DEFAULT_BUNDLE_DIR as FRVP_PAPER_SIGNAL_BUNDLE_DIR,
    DEFAULT_ENV_PATH as FRVP_PAPER_SIGNAL_ENV_PATH,
    EXPECTED_MODEL_IDS as FRVP_PAPER_SIGNAL_MODEL_IDS,
    audit_readiness as audit_frvp_paper_signal_readiness,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
ES_SHARED_DEFAULT_SERVICE_NAME = "es-shared-live-signal-service"
ES_SHARED_DEFAULT_LOG_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "logs" / "es_shared_live_signal_service.log"
ES_SHARED_DEFAULT_HEARTBEAT_PATH = (
    REPO_ROOT / "ote_live" / "runtime_data" / "health" / "es_shared_live_signal_service_heartbeat.json"
)
ICT_PAPER_SIGNAL_LONG_MANIFEST_PATH = (
    ICT_PAPER_SIGNAL_BUNDLE_DIR / "live_runtime_manifest_long.json"
)
ICT_PAPER_SIGNAL_SHORT_MANIFEST_PATH = (
    ICT_PAPER_SIGNAL_BUNDLE_DIR / "live_runtime_manifest_short.json"
)
FRVP_PAPER_SIGNAL_LONG_MANIFEST_PATH = (
    FRVP_PAPER_SIGNAL_BUNDLE_DIR / "live_runtime_manifest_long.json"
)
FRVP_PAPER_SIGNAL_SHORT_MANIFEST_PATH = (
    FRVP_PAPER_SIGNAL_BUNDLE_DIR / "live_runtime_manifest_short.json"
)


def build_parser():
    load_repo_env()
    parser = build_base_parser()
    parser.description = (
        "Run one IBKR ES 5m collector loop and evaluate FRVP and/or ICT runtime manifests "
        "against the shared completed-bar stream."
    )
    parser.set_defaults(
        group_name=env_str("ES_LIVE_GROUP_NAME", "ES_SHARED"),
        data_supplier=str(env_str("FRVP_LIVE_DATA_SUPPLIER", env_str("ICT_LIVE_DATA_SUPPLIER", "IBKR")) or "IBKR").upper(),
        asset=env_str("FRVP_LIVE_ASSET", env_str("ICT_LIVE_ASSET", "ES")),
        source_timeframe=env_str(
            "FRVP_LIVE_SOURCE_TIMEFRAME",
            env_str("ICT_LIVE_SOURCE_TIMEFRAME", env_str("FRVP_LIVE_TIMEFRAME", env_str("ICT_LIVE_TIMEFRAME", "5m"))),
        ),
        db_path=str(env_path("FRVP_LIVE_DB_PATH", env_path("ICT_LIVE_DB_PATH", env_path("OTE_LIVE_DB_PATH", DEFAULT_DB_PATH) or DEFAULT_DB_PATH))),
        service_name=env_str("ES_LIVE_SERVICE_NAME", ES_SHARED_DEFAULT_SERVICE_NAME),
        heartbeat_file=str(env_path("ES_LIVE_HEARTBEAT_FILE", ES_SHARED_DEFAULT_HEARTBEAT_PATH)),
        log_file=str(env_path("ES_LIVE_LOG_FILE", ES_SHARED_DEFAULT_LOG_PATH)),
        log_max_bytes=env_int("ES_LIVE_LOG_MAX_BYTES", env_int("FRVP_LIVE_LOG_MAX_BYTES", 5_000_000)),
        log_backup_count=env_int("ES_LIVE_LOG_BACKUP_COUNT", env_int("FRVP_LIVE_LOG_BACKUP_COUNT", 10)),
        min_disk_free_gb=env_float("ES_LIVE_MIN_DISK_FREE_GB", env_float("FRVP_LIVE_MIN_DISK_FREE_GB", 2.0)),
        poll_interval_seconds=env_float("ES_LIVE_COLLECTOR_POLL_INTERVAL_SECONDS", env_float("FRVP_LIVE_COLLECTOR_POLL_INTERVAL_SECONDS", 300.0)),
        stream_outputsize=env_int("ES_LIVE_STREAM_OUTPUTSIZE", env_int("FRVP_LIVE_STREAM_OUTPUTSIZE", 2)),
        finalized_bar_grace_seconds=env_float("ES_LIVE_FINALIZED_BAR_GRACE_SECONDS", env_float("FRVP_LIVE_FINALIZED_BAR_GRACE_SECONDS", 90.0)),
        signal_processing_delay_seconds=env_float("ES_LIVE_SIGNAL_PROCESSING_DELAY_SECONDS", env_float("FRVP_LIVE_SIGNAL_PROCESSING_DELAY_SECONDS")),
        cycle_finalized_refresh_lookback_bars=env_int("ES_LIVE_CYCLE_FINALIZED_REFRESH_LOOKBACK_BARS", env_int("FRVP_LIVE_CYCLE_FINALIZED_REFRESH_LOOKBACK_BARS", 6)),
        startup_history_bars=env_int("ES_LIVE_STARTUP_HISTORY_BARS", env_int("FRVP_LIVE_STARTUP_HISTORY_BARS", 240)),
        startup_warmup_lookback_bars=env_int("ES_LIVE_STARTUP_WARMUP_LOOKBACK_BARS", env_int("FRVP_LIVE_STARTUP_WARMUP_LOOKBACK_BARS", 90)),
        startup_backfill_chunk_bars=env_int("ES_LIVE_STARTUP_BACKFILL_CHUNK_BARS", env_int("FRVP_LIVE_STARTUP_BACKFILL_CHUNK_BARS", 1000)),
        heartbeat_stale_after_seconds=env_float("ES_LIVE_HEARTBEAT_STALE_AFTER_SECONDS", env_float("FRVP_LIVE_HEARTBEAT_STALE_AFTER_SECONDS")),
        timezone=env_str("ES_LIVE_TIMEZONE", env_str("FRVP_LIVE_TIMEZONE", "UTC")),
        max_cycles=env_int("ES_LIVE_MAX_CYCLES", env_int("FRVP_LIVE_MAX_CYCLES", env_int("OTE_LIVE_MAX_CYCLES"))),
        log_level=env_str("ES_LIVE_LOG_LEVEL", env_str("FRVP_LIVE_LOG_LEVEL", "INFO")),
        enable_signal_runtime=env_bool("ES_LIVE_ENABLE_SIGNAL_RUNTIME", True),
        all_models_active=env_bool("ES_LIVE_ALL_MODELS_ACTIVE", env_bool("FRVP_LIVE_ALL_MODELS_ACTIVE", False)),
        ict_paper_signal_trial_enabled=env_bool(
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED",
            False,
        ),
        frvp_paper_signal_trial_enabled=env_bool(
            "FRVP_PAPER_SIGNAL_TRIAL_ENABLED",
            False,
        ),
        enable_signal_chart_capture=env_bool("ES_LIVE_ENABLE_SIGNAL_CHART_CAPTURE", env_bool("FRVP_LIVE_ENABLE_SIGNAL_CHART_CAPTURE", True)),
        signal_chart_output_root=(
            str(env_path("ES_LIVE_SIGNAL_CHART_OUTPUT_ROOT"))
            if env_path("ES_LIVE_SIGNAL_CHART_OUTPUT_ROOT") is not None
            else (
                str(env_path("FRVP_LIVE_SIGNAL_CHART_OUTPUT_ROOT"))
                if env_path("FRVP_LIVE_SIGNAL_CHART_OUTPUT_ROOT") is not None
                else None
            )
        ),
        signal_chart_lookback_bars=env_int("ES_LIVE_SIGNAL_CHART_LOOKBACK_BARS", env_int("FRVP_LIVE_SIGNAL_CHART_LOOKBACK_BARS", 120)),
        dashboard_url=env_str("ES_LIVE_DASHBOARD_URL", env_str("FRVP_LIVE_DASHBOARD_URL", env_str("OTE_LIVE_DASHBOARD_URL"))),
        alert_email_recipients=env_str(
            "ES_LIVE_ALERT_EMAIL_RECIPIENTS",
            env_str(
                "FRVP_LIVE_ALERT_EMAIL_RECIPIENTS",
                env_str("OTE_LIVE_ALERT_EMAIL_RECIPIENTS", ""),
            ),
        ),
        alert_sms_recipients=env_str(
            "ES_LIVE_ALERT_SMS_RECIPIENTS",
            env_str(
                "FRVP_LIVE_ALERT_SMS_RECIPIENTS",
                env_str("OTE_LIVE_ALERT_SMS_RECIPIENTS", ""),
            ),
        ),
        ibkr_enabled=env_bool("IBKR_ENABLED", False),
        ibkr_host=env_str("IBKR_HOST", "127.0.0.1"),
        ibkr_port=env_int("IBKR_PORT", 4002),
        ibkr_client_id=env_int("IBKR_CLIENT_ID", 21),
        ibkr_account_mode=env_str("IBKR_ACCOUNT_MODE", "paper"),
        ibkr_market_data_type=env_str("IBKR_MARKET_DATA_TYPE", "live"),
        ibkr_allow_delayed_fallback=env_bool("IBKR_ALLOW_DELAYED_FALLBACK", False),
        ibkr_symbol=env_str("IBKR_ES_SYMBOL", "ES"),
        ibkr_security_type=env_str("IBKR_ES_SECURITY_TYPE", "FUT"),
        ibkr_exchange=env_str("IBKR_ES_EXCHANGE", "CME"),
        ibkr_currency=env_str("IBKR_ES_CURRENCY", "USD"),
        ibkr_multiplier=env_str("IBKR_ES_MULTIPLIER", "50"),
        ibkr_trading_class=env_str("IBKR_ES_TRADING_CLASS", "ES"),
        ibkr_local_symbol=None,
        ibkr_contract_month=env_str("IBKR_ES_MANUAL_CONTRACT_MONTH"),
        ibkr_con_id=env_int("IBKR_ES_MANUAL_CONID"),
        ibkr_bar_size=env_str("IBKR_ES_BAR_SIZE", "5 mins"),
        ibkr_what_to_show=env_str("IBKR_WHAT_TO_SHOW", "TRADES"),
        ibkr_backfill_duration=env_str("IBKR_ES_HISTORY_DURATION", "2 D"),
        ibkr_use_rth=env_bool("IBKR_ES_USE_RTH", False),
        ibkr_keep_up_to_date=env_bool("IBKR_KEEP_UP_TO_DATE", True),
        ibkr_roll_policy=env_str("IBKR_ES_ROLL_POLICY", "cme_calendar"),
        ibkr_roll_time_et=env_str("IBKR_ES_ROLL_TIME_ET", "09:30"),
        ibkr_stale_quote_seconds=env_float("IBKR_STALE_QUOTE_SECONDS", 15.0),
        ibkr_stale_bar_seconds=env_float("IBKR_STALE_BAR_SECONDS", 420.0),
        ibkr_delayed_history_refresh_seconds=env_float(
            "IBKR_DELAYED_HISTORY_REFRESH_SECONDS",
            240.0,
        ),
        ibkr_reconnect_initial_seconds=env_float("IBKR_RECONNECT_INITIAL_SECONDS", 2.0),
        ibkr_reconnect_max_seconds=env_float("IBKR_RECONNECT_MAX_SECONDS", 60.0),
        debug=env_bool("ES_LIVE_DEBUG", env_bool("FRVP_LIVE_DEBUG", False)),
    )
    include_frvp_group = parser.add_mutually_exclusive_group()
    include_frvp_group.add_argument("--include-frvp", dest="include_frvp", action="store_true")
    include_frvp_group.add_argument("--exclude-frvp", dest="include_frvp", action="store_false")
    include_ict_group = parser.add_mutually_exclusive_group()
    include_ict_group.add_argument("--include-ict", dest="include_ict", action="store_true")
    include_ict_group.add_argument("--exclude-ict", dest="include_ict", action="store_false")
    include_frvp_setup_group = parser.add_mutually_exclusive_group()
    include_frvp_setup_group.add_argument(
        "--include-frvp-setup",
        dest="include_frvp_setup",
        action="store_true",
    )
    include_frvp_setup_group.add_argument(
        "--exclude-frvp-setup",
        dest="include_frvp_setup",
        action="store_false",
    )
    include_ict_setup_group = parser.add_mutually_exclusive_group()
    include_ict_setup_group.add_argument(
        "--include-ict-setup",
        dest="include_ict_setup",
        action="store_true",
    )
    include_ict_setup_group.add_argument(
        "--exclude-ict-setup",
        dest="include_ict_setup",
        action="store_false",
    )
    parser.add_argument(
        "--allow-ict-clean-handoff",
        action="store_true",
        default=False,
        help=(
            "Allow the final ICT launch guard to use an exact, healthy, non-stale "
            "clean stopped ES heartbeat no more than 120 seconds old. This is an "
            "explicit one-launch handoff mode, not a persistent environment setting."
        ),
    )
    parser.add_argument(
        "--allow-frvp-clean-handoff",
        action="store_true",
        default=False,
        help=(
            "Allow the final FRVP launch guard to use an exact, healthy, non-stale "
            "clean stopped ES heartbeat no more than 120 seconds old. This is an "
            "explicit one-launch handoff mode, not a persistent environment setting."
        ),
    )
    ict_delayed_test_group = parser.add_mutually_exclusive_group()
    ict_delayed_test_group.add_argument(
        "--allow-ict-delayed-test",
        dest="allow_ict_delayed_test",
        action="store_true",
        help=(
            "Run ICT on delayed IBKR data for dashboard/operator testing. ICT "
            "models are forced to shadow mode and ICT paper-signal ledger writes "
            "are disabled; this does not authorize the controlled paper trial."
        ),
    )
    ict_delayed_test_group.add_argument(
        "--no-ict-delayed-test",
        dest="allow_ict_delayed_test",
        action="store_false",
    )
    parser.set_defaults(
        include_frvp=env_bool("ES_LIVE_INCLUDE_FRVP", True),
        include_ict=env_bool("ES_LIVE_INCLUDE_ICT", True),
        include_frvp_setup=env_bool("ES_LIVE_INCLUDE_FRVP_SETUP", False),
        include_ict_setup=env_bool("ES_LIVE_INCLUDE_ICT_SETUP", False),
        allow_ict_delayed_test=env_bool("ICT_DELAYED_DATA_TEST_ENABLED", False),
    )
    parser.add_argument(
        "--frvp-long-runtime-manifest-path",
        default=str(env_path("FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH", DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument(
        "--frvp-short-runtime-manifest-path",
        default=str(env_path("FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH", DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument(
        "--ict-long-runtime-manifest-path",
        default=str(env_path("ICT_LIVE_LONG_RUNTIME_MANIFEST_PATH", DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument(
        "--ict-short-runtime-manifest-path",
        default=str(env_path("ICT_LIVE_SHORT_RUNTIME_MANIFEST_PATH", DEFAULT_ICT_SHORT_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument(
        "--frvp-setup-long-runtime-manifest-path",
        default=str(
            env_path(
                "FRVP_SETUP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
                DEFAULT_FRVP_SETUP_LONG_RUNTIME_MANIFEST_PATH,
            )
        ),
    )
    parser.add_argument(
        "--frvp-setup-short-runtime-manifest-path",
        default=str(
            env_path(
                "FRVP_SETUP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
                DEFAULT_FRVP_SETUP_SHORT_RUNTIME_MANIFEST_PATH,
            )
        ),
    )
    parser.add_argument(
        "--ict-setup-long-runtime-manifest-path",
        default=str(
            env_path(
                "ICT_SETUP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
                DEFAULT_ICT_SETUP_LONG_RUNTIME_MANIFEST_PATH,
            )
        ),
    )
    parser.add_argument(
        "--ict-setup-short-runtime-manifest-path",
        default=str(
            env_path(
                "ICT_SETUP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
                DEFAULT_ICT_SETUP_SHORT_RUNTIME_MANIFEST_PATH,
            )
        ),
    )
    return parser


def _build_group_specs(args) -> list[dict[str, object]]:
    group_specs: list[dict[str, object]] = []
    if bool(args.include_frvp):
        group_specs.append(
            {
                "name": "FRVP",
                "long_path": Path(args.frvp_long_runtime_manifest_path),
                "short_path": Path(args.frvp_short_runtime_manifest_path),
            }
        )
    if bool(args.include_ict):
        group_specs.append(
            {
                "name": "ICT",
                "long_path": Path(args.ict_long_runtime_manifest_path),
                "short_path": Path(args.ict_short_runtime_manifest_path),
            }
        )
    if bool(args.include_frvp_setup):
        group_specs.append(
            {
                "name": "FRVP",
                "long_path": Path(args.frvp_setup_long_runtime_manifest_path),
                "short_path": Path(args.frvp_setup_short_runtime_manifest_path),
            }
        )
    if bool(args.include_ict_setup):
        group_specs.append(
            {
                "name": "ICT",
                "long_path": Path(args.ict_setup_long_runtime_manifest_path),
                "short_path": Path(args.ict_setup_short_runtime_manifest_path),
            }
        )
    return group_specs


def _ict_delayed_test_enabled(args) -> bool:
    return bool(getattr(args, "allow_ict_delayed_test", False))


def _validate_ict_delayed_test_runtime(args) -> None:
    """Allow delayed-data ICT chart testing without authorizing paper signals."""

    manifest_paths = (
        _resolve_repo_runtime_path(args.ict_long_runtime_manifest_path),
        _resolve_repo_runtime_path(args.ict_short_runtime_manifest_path),
    )
    expected_manifest_paths = (
        ICT_PAPER_SIGNAL_LONG_MANIFEST_PATH.resolve(),
        ICT_PAPER_SIGNAL_SHORT_MANIFEST_PATH.resolve(),
    )
    if manifest_paths != expected_manifest_paths:
        raise ValueError(
            "ICT delayed test mode is locked to the exact controlled bundle "
            f"{ICT_PAPER_SIGNAL_BUNDLE_ID}; renamed, copied, or custom manifest "
            "paths are not permitted."
        )
    if bool(args.all_models_active):
        raise ValueError(
            "ICT delayed test mode requires ES_LIVE_ALL_MODELS_ACTIVE=false "
            "so ICT models remain shadow-only."
        )
    if not bool(args.ibkr_enabled):
        raise ValueError("ICT delayed test mode requires IBKR_ENABLED=true.")
    if str(args.ibkr_account_mode or "").strip().lower() != "paper":
        raise ValueError("ICT delayed test mode requires IBKR_ACCOUNT_MODE=paper.")
    requested_type = str(getattr(args, "ibkr_market_data_type", "") or "").strip().lower()
    delayed_requested = requested_type in {"delayed", "3", "delayed_frozen", "4"}
    if not delayed_requested and not bool(args.ibkr_allow_delayed_fallback):
        raise ValueError(
            "ICT delayed test mode requires IBKR_MARKET_DATA_TYPE=delayed "
            "or IBKR_ALLOW_DELAYED_FALLBACK=true."
        )
    LOGGER.warning(
        "ICT delayed test mode enabled; ICT models will run in shadow mode and "
        "ICT paper-signal ledger writes are disabled."
    )


def _validate_ict_paper_signal_runtime(
    args,
    *,
    readiness_audit: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Fail closed on overrides that would invalidate the controlled ICT trial."""

    if not bool(args.include_ict):
        return
    if _ict_delayed_test_enabled(args):
        _validate_ict_delayed_test_runtime(args)
        return
    manifest_paths = (
        _resolve_repo_runtime_path(args.ict_long_runtime_manifest_path),
        _resolve_repo_runtime_path(args.ict_short_runtime_manifest_path),
    )
    expected_manifest_paths = (
        ICT_PAPER_SIGNAL_LONG_MANIFEST_PATH.resolve(),
        ICT_PAPER_SIGNAL_SHORT_MANIFEST_PATH.resolve(),
    )
    if manifest_paths != expected_manifest_paths:
        raise ValueError(
            "ICT launch is locked to the exact controlled bundle "
            f"{ICT_PAPER_SIGNAL_BUNDLE_ID}; renamed, copied, or custom manifest "
            "paths are not permitted. Use --exclude-ict for a non-ICT feed bootstrap."
        )
    if not bool(args.ict_paper_signal_trial_enabled):
        raise ValueError(
            "The ICT controlled paper-signal bundle is configured but its "
            "fail-closed launch switch is disabled. Run the readiness audit, "
            "resolve every blocker, then set ICT_PAPER_SIGNAL_TRIAL_ENABLED=true."
        )
    if bool(args.all_models_active):
        raise ValueError(
            "The ICT controlled paper-signal bundle requires "
            "ES_LIVE_ALL_MODELS_ACTIVE=false; the override would activate every "
            "non-deprecated ICT model."
        )
    if str(args.ibkr_account_mode or "").strip().lower() != "paper":
        raise ValueError(
            "The ICT controlled paper-signal bundle requires "
            "IBKR_ACCOUNT_MODE=paper. This guard does not replace verification "
            "that TWS/IB Gateway is authenticated to the paper account."
        )
    if int(args.ibkr_port) not in {4002, 7497}:
        raise ValueError(
            "The ICT controlled paper-signal bundle requires a standard IBKR "
            "paper endpoint: port 4002 for IB Gateway or 7497 for TWS."
        )
    if not bool(args.ibkr_enabled):
        raise ValueError(
            "The ICT controlled paper-signal bundle requires IBKR_ENABLED=true."
        )
    if bool(args.ibkr_allow_delayed_fallback):
        raise ValueError(
            "The ICT controlled paper-signal bundle requires "
            "IBKR_ALLOW_DELAYED_FALLBACK=false."
        )

    audit = readiness_audit or audit_ict_paper_signal_readiness
    result = audit(
        bundle_dir=ICT_PAPER_SIGNAL_BUNDLE_DIR,
        env_path=ICT_PAPER_SIGNAL_ENV_PATH,
        heartbeat_path=Path(args.heartbeat_file),
        environ={
            "ES_LIVE_ALL_MODELS_ACTIVE": str(bool(args.all_models_active)).lower(),
            "IBKR_ENABLED": str(bool(args.ibkr_enabled)).lower(),
            "IBKR_ACCOUNT_MODE": str(args.ibkr_account_mode or ""),
            "IBKR_PORT": str(int(args.ibkr_port)),
            "IBKR_ALLOW_DELAYED_FALLBACK": str(
                bool(args.ibkr_allow_delayed_fallback)
            ).lower(),
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED": str(
                bool(args.ict_paper_signal_trial_enabled)
            ).lower(),
        },
        preflight=False,
        allow_clean_stopped_handoff=bool(args.allow_ict_clean_handoff),
    )
    _require_exact_ready_contract(result)


def _validate_frvp_paper_signal_runtime(
    args,
    *,
    readiness_audit: Callable[..., dict[str, Any]] | None = None,
) -> object | None:
    """Fail closed whenever an active FRVP manifest is requested."""

    if not bool(args.include_frvp):
        return None
    if bool(args.all_models_active):
        raise ValueError(
            "Every FRVP runtime, including candidate-only bootstrap bundles, "
            "requires ES_LIVE_ALL_MODELS_ACTIVE=false."
        )
    manifest_paths = (
        _resolve_repo_runtime_path(args.frvp_long_runtime_manifest_path),
        _resolve_repo_runtime_path(args.frvp_short_runtime_manifest_path),
    )
    has_active_models = _frvp_manifest_set_has_active_models(manifest_paths)
    trial_enabled = bool(
        getattr(args, "frvp_paper_signal_trial_enabled", False)
    )
    if not has_active_models:
        if trial_enabled:
            raise ValueError(
                "FRVP_PAPER_SIGNAL_TRIAL_ENABLED=true is only valid for the exact "
                "controlled active bundle."
            )
        return None
    expected_manifest_paths = (
        FRVP_PAPER_SIGNAL_LONG_MANIFEST_PATH.resolve(),
        FRVP_PAPER_SIGNAL_SHORT_MANIFEST_PATH.resolve(),
    )
    if manifest_paths != expected_manifest_paths:
        raise ValueError(
            "Active FRVP launch is locked to the exact controlled bundle "
            f"{FRVP_PAPER_SIGNAL_BUNDLE_ID}; renamed, copied, or custom active "
            "manifest paths are not permitted."
        )
    if not trial_enabled:
        raise ValueError(
            "The FRVP controlled paper-signal bundle is configured but its "
            "fail-closed launch switch is disabled. Run the readiness audit, "
            "resolve every blocker, then set FRVP_PAPER_SIGNAL_TRIAL_ENABLED=true."
        )
    if not bool(getattr(args, "enable_signal_runtime", False)):
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires "
            "ES_LIVE_ENABLE_SIGNAL_RUNTIME=true."
        )
    if str(getattr(args, "asset", "") or "").strip().upper() != "ES":
        raise ValueError("The FRVP controlled paper-signal bundle requires asset ES.")
    if str(getattr(args, "source_timeframe", "") or "").strip() != "5m":
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires source timeframe 5m."
        )
    if str(getattr(args, "data_supplier", "") or "").strip().upper() != "IBKR":
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires data supplier IBKR."
        )
    if not bool(args.ibkr_enabled):
        raise ValueError("The FRVP controlled paper-signal bundle requires IBKR_ENABLED=true.")
    if str(args.ibkr_account_mode or "").strip().lower() != "paper":
        raise ValueError("The FRVP controlled paper-signal bundle requires IBKR_ACCOUNT_MODE=paper.")
    if int(args.ibkr_port) not in {4002, 7497}:
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires paper port 4002 or 7497."
        )
    if bool(args.ibkr_allow_delayed_fallback):
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires "
            "IBKR_ALLOW_DELAYED_FALLBACK=false."
        )
    if str(getattr(args, "ibkr_market_data_type", "") or "").strip().lower() not in {
        "live",
        "1",
    }:
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires live IBKR market data."
        )
    if str(getattr(args, "ibkr_what_to_show", "") or "").strip().upper() != "TRADES":
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires IBKR_WHAT_TO_SHOW=TRADES."
        )
    if bool(getattr(args, "ibkr_use_rth", True)):
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires IBKR_ES_USE_RTH=false."
        )
    if str(getattr(args, "ibkr_bar_size", "") or "").strip().lower() != "5 mins":
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires IBKR_ES_BAR_SIZE='5 mins'."
        )
    if not bool(getattr(args, "ibkr_keep_up_to_date", False)):
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires IBKR_KEEP_UP_TO_DATE=true."
        )
    contract_fields = {
        "ibkr_symbol": "ES",
        "ibkr_security_type": "FUT",
        "ibkr_exchange": "CME",
        "ibkr_currency": "USD",
        "ibkr_multiplier": "50",
        "ibkr_trading_class": "ES",
    }
    for field, expected in contract_fields.items():
        actual = str(getattr(args, field, "") or "").strip().upper()
        if actual != expected:
            raise ValueError(
                "The FRVP controlled paper-signal bundle requires "
                f"{field}={expected}."
            )
    if getattr(args, "max_cycles", None) is not None:
        raise ValueError(
            "The 28-day FRVP paper-signal launch does not permit finite max_cycles."
        )
    if (
        _resolve_service_name(
            configured_service_name=str(getattr(args, "service_name", "")),
            group_name=str(getattr(args, "group_name", "")),
        )
        != ES_SHARED_DEFAULT_SERVICE_NAME
    ):
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires the shared ES service name."
        )
    if _resolve_repo_runtime_path(args.heartbeat_file) != ES_SHARED_DEFAULT_HEARTBEAT_PATH.resolve():
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires the default shared heartbeat path."
        )
    if _resolve_repo_runtime_path(args.db_path) != DEFAULT_DB_PATH.resolve():
        raise ValueError(
            "The FRVP controlled paper-signal bundle requires the default live database path."
        )
    if not bool(getattr(args, "allow_frvp_clean_handoff", False)):
        raise ValueError(
            "The final FRVP launch requires an explicit clean-stopped collector handoff."
        )

    audit = readiness_audit or audit_frvp_paper_signal_readiness
    result = audit(
        bundle_dir=FRVP_PAPER_SIGNAL_BUNDLE_DIR,
        env_path=FRVP_PAPER_SIGNAL_ENV_PATH,
        heartbeat_path=Path(args.heartbeat_file),
        environ={
            "ES_LIVE_ALL_MODELS_ACTIVE": str(bool(args.all_models_active)).lower(),
            "IBKR_ENABLED": str(bool(args.ibkr_enabled)).lower(),
            "IBKR_ACCOUNT_MODE": str(args.ibkr_account_mode or ""),
            "IBKR_PORT": str(int(args.ibkr_port)),
            "IBKR_ALLOW_DELAYED_FALLBACK": str(
                bool(args.ibkr_allow_delayed_fallback)
            ).lower(),
            "FRVP_PAPER_SIGNAL_TRIAL_ENABLED": str(
                trial_enabled
            ).lower(),
            "ES_LIVE_ENABLE_SIGNAL_RUNTIME": str(
                bool(args.enable_signal_runtime)
            ).lower(),
            "FRVP_LIVE_DATA_SUPPLIER": str(args.data_supplier),
            "FRVP_LIVE_ASSET": str(args.asset),
            "FRVP_LIVE_SOURCE_TIMEFRAME": str(args.source_timeframe),
            "IBKR_MARKET_DATA_TYPE": str(args.ibkr_market_data_type),
            "IBKR_WHAT_TO_SHOW": str(args.ibkr_what_to_show),
            "IBKR_ES_USE_RTH": str(bool(args.ibkr_use_rth)).lower(),
            "IBKR_ES_BAR_SIZE": str(args.ibkr_bar_size),
            "IBKR_KEEP_UP_TO_DATE": str(bool(args.ibkr_keep_up_to_date)).lower(),
            "IBKR_ES_SYMBOL": str(args.ibkr_symbol),
            "IBKR_ES_SECURITY_TYPE": str(args.ibkr_security_type),
            "IBKR_ES_EXCHANGE": str(args.ibkr_exchange),
            "IBKR_ES_CURRENCY": str(args.ibkr_currency),
            "IBKR_ES_MULTIPLIER": str(args.ibkr_multiplier),
            "IBKR_ES_TRADING_CLASS": str(args.ibkr_trading_class),
        },
        preflight=False,
        allow_clean_stopped_handoff=True,
    )
    _require_exact_frvp_ready_contract(result)
    return _issue_frvp_paper_signal_runtime_authorization()


def _frvp_manifest_set_has_active_models(paths: tuple[Path, Path]) -> bool:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(
                f"FRVP runtime manifest {path} is missing or invalid."
            ) from exc
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise ValueError(f"FRVP runtime manifest {path} has no valid model roster.")
        for model in models:
            if not isinstance(model, dict):
                raise ValueError(f"FRVP runtime manifest {path} has a malformed model entry.")
            if not str(model.get("model_id") or "").strip():
                raise ValueError(f"FRVP runtime manifest {path} has a model without an ID.")
            if str(model.get("status") or "") not in {"active", "candidate", "deprecated"}:
                raise ValueError(
                    f"FRVP runtime manifest {path} has an invalid model status."
                )
        if any(str(model["status"]) == "active" for model in models):
            return True
    return False


def _resolve_repo_runtime_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_exact_ready_contract(result: dict[str, Any]) -> None:
    facts = result.get("facts")
    facts = facts if isinstance(facts, dict) else {}
    bundle_dir = facts.get("bundle_dir")
    try:
        resolved_result_bundle = (
            _resolve_repo_runtime_path(bundle_dir) if bundle_dir else None
        )
    except (OSError, TypeError, ValueError):
        resolved_result_bundle = None
    identity_matches = (
        result.get("bundle_id") == ICT_PAPER_SIGNAL_BUNDLE_ID
        and resolved_result_bundle == ICT_PAPER_SIGNAL_BUNDLE_DIR.resolve()
        and frozenset(facts.get("model_ids") or ()) == ICT_PAPER_SIGNAL_MODEL_IDS
    )
    if not identity_matches:
        raise ValueError(
            "ICT readiness audit returned a bundle identity/content mismatch; "
            "launch remains blocked."
        )
    if result.get("status") != "ready_to_start" or result.get("ready_to_start") is not True:
        blocking_reasons = result.get("blocking_reasons")
        reason_codes = sorted(
            {
                str(reason.get("code") or "unknown")
                for reason in blocking_reasons or ()
                if isinstance(reason, dict)
            }
        )
        reason_summary = ", ".join(reason_codes) if reason_codes else "readiness_not_confirmed"
        raise ValueError(
            "ICT readiness contract blocked launch: "
            f"{reason_summary}. Resolve the default audit before enabling ICT."
        )


def _require_exact_frvp_ready_contract(result: dict[str, Any]) -> None:
    facts = result.get("facts")
    facts = facts if isinstance(facts, dict) else {}
    bundle_dir = facts.get("bundle_dir")
    try:
        resolved_result_bundle = (
            _resolve_repo_runtime_path(bundle_dir) if bundle_dir else None
        )
    except (OSError, TypeError, ValueError):
        resolved_result_bundle = None
    identity_matches = (
        result.get("bundle_id") == FRVP_PAPER_SIGNAL_BUNDLE_ID
        and resolved_result_bundle == FRVP_PAPER_SIGNAL_BUNDLE_DIR.resolve()
        and frozenset(facts.get("model_ids") or ()) == FRVP_PAPER_SIGNAL_MODEL_IDS
    )
    if not identity_matches:
        raise ValueError(
            "FRVP readiness audit returned a bundle identity/content mismatch; "
            "launch remains blocked."
        )
    if result.get("status") != "ready_to_start" or result.get("ready_to_start") is not True:
        blocking_reasons = result.get("blocking_reasons")
        reason_codes = sorted(
            {
                str(reason.get("code") or "unknown")
                for reason in blocking_reasons or ()
                if isinstance(reason, dict)
            }
        )
        reason_summary = ", ".join(reason_codes) if reason_codes else "readiness_not_confirmed"
        raise ValueError(
            "FRVP readiness contract blocked launch: "
            f"{reason_summary}. Resolve the default audit before enabling FRVP."
        )
    heartbeat = facts.get("heartbeat")
    heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
    if heartbeat.get("acceptance") != "clean_stopped_handoff":
        raise ValueError(
            "FRVP final launch requires readiness acceptance from the explicit "
            "clean-stopped collector handoff."
        )


def _require_authorized_frvp_signal_processor(
    processor: LiveSignalProcessor | MultiGroupLiveSignalProcessor | None,
) -> None:
    """Verify the exact non-shadow reversal and its ledger before heartbeat start."""

    if processor is None:
        raise RuntimeError(
            "The authorized FRVP paper-signal runtime has no signal processor."
        )
    processors = (
        processor.processors
        if isinstance(processor, MultiGroupLiveSignalProcessor)
        else (processor,)
    )
    controlled: list[tuple[LiveSignalProcessor, object]] = []
    for item in processors:
        for binding in item.bindings:
            if binding.loaded_model.model_id in FRVP_LEDGER_MODEL_IDS:
                controlled.append((item, binding))
    if len(controlled) != 1:
        raise RuntimeError(
            "The authorized FRVP runtime must load exactly one controlled reversal binding."
        )
    owner, binding = controlled[0]
    if binding.shadow_mode:
        raise RuntimeError("The controlled FRVP reversal binding loaded in shadow mode.")
    if not supports_frvp_paper_signal_manifest(binding.loaded_model.manifest):
        raise RuntimeError(
            "The loaded FRVP reversal binding does not match the frozen manifest hash."
        )
    if getattr(owner, "frvp_paper_signal_ledger", None) is None:
        raise RuntimeError(
            "The controlled FRVP reversal binding loaded without its confirmation ledger."
        )


async def _run(args) -> int:
    frvp_paper_signal_authorization = _validate_frvp_paper_signal_runtime(args)
    _validate_ict_paper_signal_runtime(args)
    ict_delayed_test_enabled = _ict_delayed_test_enabled(args)
    group_specs = _build_group_specs(args)
    if not group_specs:
        raise ValueError("At least one ES signal family must be enabled.")

    primary_group = group_specs[0]
    primary_is_ict_delayed_test = (
        ict_delayed_test_enabled and str(primary_group["name"]).upper() == "ICT"
    )
    config = LiveCollectorConfig(
        group_name=str(primary_group["name"]),
        data_supplier=str(args.data_supplier).upper(),
        asset=args.asset,
        source_timeframe=args.source_timeframe,
        db_path=Path(args.db_path),
        poll_interval_seconds=args.poll_interval_seconds,
        stream_outputsize=args.stream_outputsize,
        finalized_bar_grace_seconds=args.finalized_bar_grace_seconds,
        signal_processing_delay_seconds=args.signal_processing_delay_seconds,
        cycle_finalized_refresh_lookback_bars=args.cycle_finalized_refresh_lookback_bars,
        startup_history_bars=args.startup_history_bars,
        startup_warmup_lookback_bars=args.startup_warmup_lookback_bars,
        startup_backfill_chunk_bars=args.startup_backfill_chunk_bars,
        heartbeat_stale_after_seconds=args.heartbeat_stale_after_seconds,
        default_timezone=args.timezone,
        log_level=args.log_level,
        max_cycles=args.max_cycles,
        enable_signal_runtime=bool(args.enable_signal_runtime),
        long_runtime_manifest_path=Path(primary_group["long_path"]),
        short_runtime_manifest_path=Path(primary_group["short_path"]),
        all_models_active=bool(args.all_models_active),
        enable_signal_chart_capture=bool(args.enable_signal_chart_capture),
        signal_chart_output_root=(
            Path(args.signal_chart_output_root)
            if args.signal_chart_output_root
            else DEFAULT_SIGNAL_CHART_OUTPUT_ROOT
        ),
        signal_chart_lookback_bars=args.signal_chart_lookback_bars,
        dashboard_url=args.dashboard_url,
        alert_email_recipients=tuple(
            part.strip() for part in str(args.alert_email_recipients).split(",") if part.strip()
        ),
        alert_sms_recipients=tuple(
            part.strip() for part in str(args.alert_sms_recipients).split(",") if part.strip()
        ),
        frvp_paper_signal_authorization=frvp_paper_signal_authorization,
        force_signal_shadow_mode=primary_is_ict_delayed_test,
        enable_ict_paper_signal_ledger=not primary_is_ict_delayed_test,
        ibkr=IBKRRuntimeConfig(
            enabled=bool(args.ibkr_enabled),
            host=args.ibkr_host,
            port=int(args.ibkr_port),
            client_id=int(args.ibkr_client_id),
            account_mode=args.ibkr_account_mode,
            market_data_type=args.ibkr_market_data_type,
            allow_delayed_fallback=bool(args.ibkr_allow_delayed_fallback),
            symbol=args.ibkr_symbol,
            security_type=args.ibkr_security_type,
            exchange=args.ibkr_exchange,
            currency=args.ibkr_currency,
            multiplier=args.ibkr_multiplier,
            trading_class=args.ibkr_trading_class,
            local_symbol=args.ibkr_local_symbol,
            manual_contract_month=args.ibkr_contract_month,
            manual_conid=args.ibkr_con_id,
            use_rth=bool(args.ibkr_use_rth),
            bar_size=args.ibkr_bar_size,
            what_to_show=args.ibkr_what_to_show,
            keep_up_to_date=bool(args.ibkr_keep_up_to_date),
            history_duration=args.ibkr_backfill_duration,
            roll_policy=args.ibkr_roll_policy,
            roll_time_et=args.ibkr_roll_time_et,
            stale_quote_seconds=args.ibkr_stale_quote_seconds,
            stale_bar_seconds=args.ibkr_stale_bar_seconds,
            delayed_history_refresh_seconds=args.ibkr_delayed_history_refresh_seconds,
            reconnect_initial_seconds=args.ibkr_reconnect_initial_seconds,
            reconnect_max_seconds=args.ibkr_reconnect_max_seconds,
        ),
    )
    runtime = LiveCollectorRuntime.from_config(config, api_key=args.api_key)
    if not bool(args.enable_signal_runtime):
        return await _run_runtime_with_ops(runtime, args)

    processors: list[LiveSignalProcessor] = []
    if runtime.signal_processor is not None:
        processors.append(runtime.signal_processor)

    shared_emailer = getattr(runtime.signal_processor, "emailer", None)
    shared_sms_sender = getattr(runtime.signal_processor, "sms_sender", None)
    shared_chart_capture = getattr(runtime.signal_processor, "chart_capture_service", None)

    for group_spec in group_specs[1:]:
        processor = LiveSignalProcessor.from_direction_manifest_paths(
            audit_repository=runtime.audit_repository,
            long_manifest_path=Path(group_spec["long_path"]),
            short_manifest_path=Path(group_spec["short_path"]),
            emailer=shared_emailer,
            sms_sender=shared_sms_sender,
            chart_capture_service=shared_chart_capture,
            dashboard_url=config.dashboard_url,
            skip_unavailable_backends=config.skip_unavailable_model_backends,
            all_models_active=config.all_models_active,
            group_name=str(group_spec["name"]),
            data_supplier=config.data_supplier,
            frvp_paper_signal_authorization=frvp_paper_signal_authorization,
            force_shadow_mode=(
                ict_delayed_test_enabled and str(group_spec["name"]).upper() == "ICT"
            ),
            enable_ict_paper_signal_ledger=not (
                ict_delayed_test_enabled and str(group_spec["name"]).upper() == "ICT"
            ),
        )
        if processor is None:
            LOGGER.warning("No eligible runtime models were loaded for %s.", group_spec["name"])
            continue
        processors.append(processor)

    if not processors:
        LOGGER.warning("No ES runtime model groups were loaded; collector will run without signal evaluation.")
        runtime.signal_processor = None
    elif len(processors) == 1:
        runtime.signal_processor = processors[0]
    else:
        runtime.signal_processor = MultiGroupLiveSignalProcessor(tuple(processors))

    if frvp_paper_signal_authorization is not None:
        _require_authorized_frvp_signal_processor(runtime.signal_processor)

    return await _run_runtime_with_ops(
        runtime,
        args,
        frvp_confirmation_clock_enabled=(
            frvp_paper_signal_authorization is not None
        ),
    )


async def _run_runtime_with_ops(
    runtime: LiveCollectorRuntime,
    args,
    *,
    frvp_confirmation_clock_enabled: bool = False,
) -> int:
    config = runtime.config
    service_name = _resolve_service_name(
        configured_service_name=str(args.service_name),
        group_name=str(args.group_name),
    )
    heartbeat_writer = ServiceHeartbeatWriter(Path(args.heartbeat_file))
    disk_monitor = DiskSpaceMonitor(
        minimum_free_bytes=int(
            max(0.0, float(args.min_disk_free_gb)) * (1024 ** 3)
        ),
    )
    ops_paths = (
        config.db_path,
        Path(args.log_file),
        Path(args.heartbeat_file),
    )
    recovery_state = capture_startup_recovery_state(
        runtime.store,
        service_name=service_name,
        asset=config.asset,
        source_timeframe=config.source_timeframe,
        signal_timeframe=config.signal_timeframe,
    )
    record_service_starting(
        runtime.store,
        runtime.audit_repository,
        service_name=service_name,
        db_path=config.db_path,
        asset=config.asset,
        source_timeframe=config.source_timeframe,
        signal_timeframe=config.signal_timeframe,
        recovery_state=recovery_state,
    )
    latest_cycle_result: dict[str, object | None] = {"value": None}

    def _write_service_snapshot(service_status: str, summary=None) -> None:
        disk_statuses = disk_monitor.evaluate(
            runtime.store,
            runtime.audit_repository,
            paths=ops_paths,
        )
        runtime_payload = _build_runtime_payload(
            summary,
            latest_cycle_result["value"],
        )
        runtime_payload["collector_contract"] = _build_es_collector_contract(runtime)
        snapshot = collect_service_health_snapshot(
            runtime.store,
            service_name=service_name,
            service_status=service_status,
            db_path=config.db_path,
            asset=config.asset,
            source_timeframe=config.source_timeframe,
            signal_timeframe=config.signal_timeframe,
            disk_statuses=disk_statuses,
            bootstrap_payload=_build_bootstrap_payload(summary),
            runtime_payload=runtime_payload,
        )
        heartbeat_writer.write_snapshot(snapshot)
        _record_frvp_confirmation_clock_after_heartbeat(
            runtime,
            snapshot,
            enabled=frvp_confirmation_clock_enabled,
        )

    _write_service_snapshot("starting")

    async def _after_bootstrap(summary) -> None:
        record_service_started(
            runtime.store,
            runtime.audit_repository,
            service_name=service_name,
            bootstrap_summary=summary.bootstrap,
        )
        _write_service_snapshot("running", summary)

    async def _after_cycle(summary, cycle_result, cycle_index: int) -> None:
        del cycle_index
        latest_cycle_result["value"] = cycle_result
        _write_service_snapshot("running", summary)

    async def _before_close(summary) -> None:
        record_service_shutdown(
            runtime.store,
            runtime.audit_repository,
            service_name=service_name,
            summary=summary,
        )
        _write_service_snapshot(_service_status_for_summary(summary), summary)

    summary = await runtime.run(
        after_bootstrap=_after_bootstrap,
        after_cycle=_after_cycle,
        before_close=_before_close,
    )
    return 0 if summary.terminal_status == "completed" else 1


def _record_frvp_confirmation_clock_after_heartbeat(
    runtime: LiveCollectorRuntime,
    snapshot: ServiceHealthSnapshot,
    *,
    enabled: bool,
) -> None:
    """Start the no-order confirmation clock after the first exact healthy heartbeat."""

    if (
        not enabled
        or snapshot.service_status != "running"
        or snapshot.health_state != "healthy"
    ):
        return
    lifecycle = record_frvp_confirmation_start(
        runtime.store,
        heartbeat=snapshot.to_dict(),
        bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
        first_healthy_final_heartbeat_confirmed=True,
        recorded_at_utc=snapshot.generated_at_utc,
    )
    if not lifecycle.started:
        refusal = ", ".join(lifecycle.refusal_reasons) or "unknown_contract_mismatch"
        raise RuntimeError(
            "The healthy FRVP final-process heartbeat could not start the "
            f"confirmation clock: {refusal}."
        )
    if lifecycle.newly_started:
        LOGGER.info(
            "FRVP controlled paper-signal confirmation clock started at %s; "
            "minimum completion is %s and broker orders remain unauthorized.",
            lifecycle.confirmation_start_utc.isoformat()
            if lifecycle.confirmation_start_utc is not None
            else None,
            lifecycle.earliest_minimum_completion_utc.isoformat()
            if lifecycle.earliest_minimum_completion_utc is not None
            else None,
        )


def _build_es_collector_contract(runtime: LiveCollectorRuntime) -> dict[str, Any]:
    """Describe configured and observed feed/runtime identity in each heartbeat."""

    config = runtime.config
    ibkr = config.ibkr
    service_status: dict[str, Any] = {}
    service = getattr(getattr(runtime, "client", None), "service", None)
    get_status = getattr(service, "get_status", None)
    if callable(get_status):
        try:
            raw_status = get_status()
            if isinstance(raw_status, dict):
                service_status = raw_status
        except Exception:
            service_status = {}

    signal_processor = runtime.signal_processor
    processors = (
        signal_processor.processors
        if isinstance(signal_processor, MultiGroupLiveSignalProcessor)
        else (signal_processor,)
        if signal_processor is not None
        else ()
    )
    bindings = [binding for processor in processors for binding in processor.bindings]
    loaded_model_ids = sorted({binding.loaded_model.model_id for binding in bindings})
    active_model_ids = sorted(
        {
            binding.loaded_model.model_id
            for binding in bindings
            if not binding.shadow_mode
        }
    )
    shadow_model_ids = sorted(
        {
            binding.loaded_model.model_id
            for binding in bindings
            if binding.shadow_mode
        }
    )
    registry_paths = sorted(
        {
            binding.loaded_model.manifest.registry_path.replace("\\", "/")
            for binding in bindings
        }
    )
    frvp_ledger_ready = any(
        getattr(processor, "frvp_paper_signal_ledger", None) is not None
        for processor in processors
    )
    ict_ledger_ready = any(
        getattr(processor, "ict_paper_signal_ledger", None) is not None
        for processor in processors
    )
    ict_delayed_test_mode = any(
        bool(getattr(processor, "force_shadow_mode", False))
        and any(
            str(binding.loaded_model.model_id).startswith("ict_")
            for binding in getattr(processor, "bindings", ())
        )
        for processor in processors
    )
    return {
        "schema_version": 1,
        "feed": {
            "data_supplier": config.data_supplier,
            "heartbeat_source": "ibkr.historical.polling"
            if config.data_supplier == "IBKR"
            else "fmp.polling",
            "asset": config.asset,
            "source_timeframe": config.source_timeframe,
            "ibkr_enabled": bool(ibkr.enabled) if ibkr is not None else False,
            "account_mode": str(ibkr.account_mode) if ibkr is not None else None,
            "market_data_type_requested": (
                str(ibkr.market_data_type).lower() if ibkr is not None else None
            ),
            "market_data_type_received": service_status.get(
                "market_data_type_received"
            ),
            "connection_state": service_status.get("connection_state"),
            "allow_delayed_fallback": (
                bool(ibkr.allow_delayed_fallback) if ibkr is not None else None
            ),
            "what_to_show": str(ibkr.what_to_show) if ibkr is not None else None,
            "use_rth": bool(ibkr.use_rth) if ibkr is not None else None,
            "bar_size": str(ibkr.bar_size) if ibkr is not None else None,
            "keep_up_to_date": (
                bool(ibkr.keep_up_to_date) if ibkr is not None else None
            ),
            "contract": {
                "symbol": str(ibkr.symbol) if ibkr is not None else None,
                "security_type": (
                    str(ibkr.security_type) if ibkr is not None else None
                ),
                "exchange": str(ibkr.exchange) if ibkr is not None else None,
                "currency": str(ibkr.currency) if ibkr is not None else None,
                "multiplier": str(ibkr.multiplier) if ibkr is not None else None,
                "trading_class": (
                    str(ibkr.trading_class) if ibkr is not None else None
                ),
            },
        },
        "signal_runtime": {
            "enabled": bool(config.enable_signal_runtime),
            "loaded": bool(bindings),
            "all_models_active": bool(config.all_models_active),
            "loaded_model_ids": loaded_model_ids,
            "active_model_ids": active_model_ids,
            "shadow_model_ids": shadow_model_ids,
            "registry_paths": registry_paths,
            "frvp_paper_signal_ledger_ready": frvp_ledger_ready,
            "ict_paper_signal_ledger_ready": ict_ledger_ready,
            "ict_delayed_test_mode": ict_delayed_test_mode,
        },
    }


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    if not (
        args.include_frvp
        or args.include_ict
        or args.include_frvp_setup
        or args.include_ict_setup
    ):
        parser.error(
            "At least one of --include-frvp, --include-ict, "
            "--include-frvp-setup, or --include-ict-setup must be enabled."
        )
    configure_live_logging(
        args.log_level,
        log_path=Path(args.log_file),
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except IBKRError as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

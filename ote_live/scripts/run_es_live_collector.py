from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ote_live.dashboard.view_registry import (
    DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH,
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
from ote_live.ingestion.signals import LiveSignalProcessor
from ote_live.ops import (
    DiskSpaceMonitor,
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
from scripts.audit_ict_paper_signal_readiness import (
    BUNDLE_ID as ICT_PAPER_SIGNAL_BUNDLE_ID,
    DEFAULT_BUNDLE_DIR as ICT_PAPER_SIGNAL_BUNDLE_DIR,
    DEFAULT_ENV_PATH as ICT_PAPER_SIGNAL_ENV_PATH,
    EXPECTED_MODEL_IDS as ICT_PAPER_SIGNAL_MODEL_IDS,
    audit_readiness as audit_ict_paper_signal_readiness,
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


class MultiGroupLiveSignalProcessor:
    def __init__(self, processors: tuple[LiveSignalProcessor, ...]) -> None:
        if not processors:
            raise ValueError("At least one signal processor is required.")
        self.processors = tuple(processors)
        self.bindings = tuple(
            binding
            for processor in self.processors
            for binding in processor.bindings
        )
        runtime_history_bars = max(
            int(
                getattr(
                    getattr(getattr(processor, "feature_engine", None), "plan", None),
                    "runtime_history_bars",
                    0,
                )
                or 0
            )
            for processor in self.processors
        )
        self.feature_engine = SimpleNamespace(
            plan=SimpleNamespace(runtime_history_bars=runtime_history_bars)
        )

    def warm_from_store(self) -> int:
        return sum(processor.warm_from_store() for processor in self.processors)

    def seed_latest_predictions_from_store(self):
        results = []
        for processor in self.processors:
            results.extend(processor.seed_latest_predictions_from_store())
        return tuple(results)

    def process_new_bars_from_store(
        self,
        *,
        emit_operator_artifacts: bool,
        max_timestamp,
    ):
        results = []
        for processor in self.processors:
            results.extend(
                processor.process_new_bars_from_store(
                    emit_operator_artifacts=emit_operator_artifacts,
                    max_timestamp=max_timestamp,
                )
            )
        return tuple(results)


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
    parser.set_defaults(
        include_frvp=env_bool("ES_LIVE_INCLUDE_FRVP", True),
        include_ict=env_bool("ES_LIVE_INCLUDE_ICT", True),
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
    return group_specs


def _validate_ict_paper_signal_runtime(
    args,
    *,
    readiness_audit: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Fail closed on overrides that would invalidate the controlled ICT trial."""

    if not bool(args.include_ict):
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


async def _run(args) -> int:
    _validate_ict_paper_signal_runtime(args)
    group_specs = _build_group_specs(args)
    if not group_specs:
        raise ValueError("At least one ES signal family must be enabled.")

    primary_group = group_specs[0]
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

    return await _run_runtime_with_ops(runtime, args)


async def _run_runtime_with_ops(runtime: LiveCollectorRuntime, args) -> int:
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
        heartbeat_writer.write_snapshot(
            collect_service_health_snapshot(
                runtime.store,
                service_name=service_name,
                service_status=service_status,
                db_path=config.db_path,
                asset=config.asset,
                source_timeframe=config.source_timeframe,
                signal_timeframe=config.signal_timeframe,
                disk_statuses=disk_statuses,
                bootstrap_payload=_build_bootstrap_payload(summary),
                runtime_payload=_build_runtime_payload(
                    summary,
                    latest_cycle_result["value"],
                ),
            )
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


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    if not args.include_frvp and not args.include_ict:
        parser.error("At least one of --include-frvp or --include-ict must be enabled.")
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

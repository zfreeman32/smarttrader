from __future__ import annotations

import asyncio
from pathlib import Path

from ote_live.env import env_bool, env_float, env_int, env_path, env_str, load_repo_env
from ote_live.ingestion.runtime import DEFAULT_DB_PATH
from ote_live.ops import configure_live_logging
from ote_live.scripts.run_live_collector import _run as _run_live_collector
from ote_live.scripts.run_live_collector import build_parser as build_base_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
FRVP_DEFAULT_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260705" / "live_runtime_manifest_long.json"
)
FRVP_DEFAULT_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260705" / "live_runtime_manifest_short.json"
)
FRVP_DEFAULT_SERVICE_NAME = "frvp-live-signal-service"
FRVP_DEFAULT_LOG_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "logs" / "frvp_live_signal_service.log"
FRVP_DEFAULT_HEARTBEAT_PATH = (
    REPO_ROOT / "ote_live" / "runtime_data" / "health" / "frvp_live_signal_service_heartbeat.json"
)


def build_parser():
    load_repo_env()
    parser = build_base_parser()
    parser.description = (
        "Run the FRVP ES 5m live collector with IBKR as the ES futures bar supplier "
        "and the FRVP shadow runtime manifests as the default model bundle."
    )
    parser.set_defaults(
        group_name=env_str("FRVP_LIVE_GROUP_NAME", "FRVP"),
        data_supplier=str(env_str("FRVP_LIVE_DATA_SUPPLIER", "IBKR") or "IBKR").upper(),
        asset=env_str("FRVP_LIVE_ASSET", "ES"),
        source_timeframe=env_str("FRVP_LIVE_SOURCE_TIMEFRAME", env_str("FRVP_LIVE_TIMEFRAME", "5m")),
        db_path=str(env_path("FRVP_LIVE_DB_PATH", env_path("OTE_LIVE_DB_PATH", DEFAULT_DB_PATH) or DEFAULT_DB_PATH)),
        service_name=env_str("FRVP_LIVE_SERVICE_NAME", FRVP_DEFAULT_SERVICE_NAME),
        heartbeat_file=str(env_path("FRVP_LIVE_HEARTBEAT_FILE", FRVP_DEFAULT_HEARTBEAT_PATH)),
        log_file=str(env_path("FRVP_LIVE_LOG_FILE", FRVP_DEFAULT_LOG_PATH)),
        log_max_bytes=env_int("FRVP_LIVE_LOG_MAX_BYTES", 5_000_000),
        log_backup_count=env_int("FRVP_LIVE_LOG_BACKUP_COUNT", 10),
        min_disk_free_gb=env_float("FRVP_LIVE_MIN_DISK_FREE_GB", 2.0),
        poll_interval_seconds=env_float("FRVP_LIVE_COLLECTOR_POLL_INTERVAL_SECONDS", 300.0),
        stream_outputsize=env_int("FRVP_LIVE_STREAM_OUTPUTSIZE", 2),
        finalized_bar_grace_seconds=env_float("FRVP_LIVE_FINALIZED_BAR_GRACE_SECONDS", 90.0),
        signal_processing_delay_seconds=env_float("FRVP_LIVE_SIGNAL_PROCESSING_DELAY_SECONDS"),
        cycle_finalized_refresh_lookback_bars=env_int("FRVP_LIVE_CYCLE_FINALIZED_REFRESH_LOOKBACK_BARS", 6),
        long_runtime_manifest_path=str(
            env_path("FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH", FRVP_DEFAULT_LONG_RUNTIME_MANIFEST_PATH)
        ),
        short_runtime_manifest_path=str(
            env_path("FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH", FRVP_DEFAULT_SHORT_RUNTIME_MANIFEST_PATH)
        ),
        startup_history_bars=env_int("FRVP_LIVE_STARTUP_HISTORY_BARS", 240),
        startup_warmup_lookback_bars=env_int("FRVP_LIVE_STARTUP_WARMUP_LOOKBACK_BARS", 90),
        startup_backfill_chunk_bars=env_int("FRVP_LIVE_STARTUP_BACKFILL_CHUNK_BARS", 1000),
        heartbeat_stale_after_seconds=env_float("FRVP_LIVE_HEARTBEAT_STALE_AFTER_SECONDS"),
        timezone=env_str("FRVP_LIVE_TIMEZONE", "UTC"),
        max_cycles=env_int("FRVP_LIVE_MAX_CYCLES", env_int("OTE_LIVE_MAX_CYCLES")),
        log_level=env_str("FRVP_LIVE_LOG_LEVEL", "INFO"),
        enable_signal_runtime=env_bool("FRVP_LIVE_ENABLE_SIGNAL_RUNTIME", True),
        all_models_active=env_bool("FRVP_LIVE_ALL_MODELS_ACTIVE", False),
        enable_signal_chart_capture=env_bool("FRVP_LIVE_ENABLE_SIGNAL_CHART_CAPTURE", True),
        signal_chart_output_root=(
            str(env_path("FRVP_LIVE_SIGNAL_CHART_OUTPUT_ROOT"))
            if env_path("FRVP_LIVE_SIGNAL_CHART_OUTPUT_ROOT") is not None
            else None
        ),
        signal_chart_lookback_bars=env_int("FRVP_LIVE_SIGNAL_CHART_LOOKBACK_BARS", 120),
        dashboard_url=env_str("FRVP_LIVE_DASHBOARD_URL", env_str("OTE_LIVE_DASHBOARD_URL")),
        alert_email_recipients=env_str("FRVP_LIVE_ALERT_EMAIL_RECIPIENTS", ""),
        alert_sms_recipients=env_str("FRVP_LIVE_ALERT_SMS_RECIPIENTS", ""),
        ibkr_host=env_str("IBKR_HOST", "127.0.0.1"),
        ibkr_port=env_int("IBKR_PORT", 7497),
        ibkr_client_id=env_int("IBKR_CLIENT_ID", 17),
        ibkr_market_data_type=env_int("IBKR_MARKET_DATA_TYPE", 1),
        ibkr_symbol=env_str("IBKR_ES_SYMBOL", "ES"),
        ibkr_exchange=env_str("IBKR_ES_EXCHANGE", "CME"),
        ibkr_currency=env_str("IBKR_ES_CURRENCY", "USD"),
        ibkr_local_symbol=env_str("IBKR_ES_LOCAL_SYMBOL"),
        ibkr_contract_month=env_str("IBKR_ES_CONTRACT_MONTH"),
        ibkr_con_id=env_int("IBKR_ES_CON_ID"),
        ibkr_bar_size=env_str("IBKR_BAR_SIZE", "5 mins"),
        ibkr_what_to_show=env_str("IBKR_WHAT_TO_SHOW", "TRADES"),
        ibkr_backfill_duration=env_str("IBKR_BACKFILL_DURATION", "2 D"),
        ibkr_use_rth=env_bool("IBKR_USE_RTH", False),
        ibkr_keep_up_to_date=env_bool("IBKR_KEEP_UP_TO_DATE", True),
        debug=env_bool("FRVP_LIVE_DEBUG", False),
    )
    return parser


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    configure_live_logging(
        args.log_level,
        log_path=Path(args.log_file),
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
    )
    try:
        return asyncio.run(_run_live_collector(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

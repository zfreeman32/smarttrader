from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from ote_live.env import env_bool, env_float, env_int, env_path, env_str, load_repo_env
from ote_live.ingestion.runtime import (
    DEFAULT_COLLECTOR_POLL_INTERVAL_SECONDS,
    DEFAULT_DB_PATH,
    DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_SIGNAL_CHART_OUTPUT_ROOT,
    DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
    LiveCollectorConfig,
    LiveCollectorRuntime,
)
from ote_live.ops import (
    DEFAULT_HEARTBEAT_PATH,
    DEFAULT_LOG_PATH,
    DiskSpaceMonitor,
    ServiceHeartbeatWriter,
    capture_startup_recovery_state,
    collect_service_health_snapshot,
    configure_live_logging,
    record_service_shutdown,
    record_service_started,
    record_service_starting,
)

DEFAULT_SERVICE_NAME = "ote-live-signal-service"


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description="Run the OTE live collector with runtime supervision hooks and local operator artifacts.",
    )
    parser.add_argument("--asset", default=env_str("OTE_LIVE_ASSET", "EURUSD"))
    parser.add_argument("--source-timeframe", default="1m", choices=("1m",))
    parser.add_argument("--db-path", default=str(env_path("OTE_LIVE_DB_PATH", DEFAULT_DB_PATH)))
    parser.add_argument("--service-name", default=env_str("OTE_LIVE_SERVICE_NAME", DEFAULT_SERVICE_NAME))
    parser.add_argument(
        "--heartbeat-file",
        default=str(env_path("OTE_LIVE_HEARTBEAT_FILE", DEFAULT_HEARTBEAT_PATH)),
    )
    parser.add_argument("--log-file", default=str(env_path("OTE_LIVE_LOG_FILE", DEFAULT_LOG_PATH)))
    parser.add_argument("--log-max-bytes", type=int, default=env_int("OTE_LIVE_LOG_MAX_BYTES", 5_000_000))
    parser.add_argument("--log-backup-count", type=int, default=env_int("OTE_LIVE_LOG_BACKUP_COUNT", 10))
    parser.add_argument("--min-disk-free-gb", type=float, default=env_float("OTE_LIVE_MIN_DISK_FREE_GB", 2.0))
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=env_float("OTE_LIVE_COLLECTOR_POLL_INTERVAL_SECONDS", DEFAULT_COLLECTOR_POLL_INTERVAL_SECONDS),
    )
    parser.add_argument("--stream-outputsize", type=int, default=env_int("OTE_LIVE_STREAM_OUTPUTSIZE", 2))
    parser.add_argument(
        "--long-runtime-manifest-path",
        default=str(env_path("OTE_LIVE_LONG_RUNTIME_MANIFEST_PATH", DEFAULT_LONG_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument(
        "--short-runtime-manifest-path",
        default=str(env_path("OTE_LIVE_SHORT_RUNTIME_MANIFEST_PATH", DEFAULT_SHORT_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument("--startup-history-bars", type=int, default=env_int("OTE_LIVE_STARTUP_HISTORY_BARS", 240))
    parser.add_argument(
        "--startup-warmup-lookback-bars",
        type=int,
        default=env_int("OTE_LIVE_STARTUP_WARMUP_LOOKBACK_BARS", 90),
    )
    parser.add_argument(
        "--startup-backfill-chunk-bars",
        type=int,
        default=env_int("OTE_LIVE_STARTUP_BACKFILL_CHUNK_BARS", 1000),
    )
    parser.add_argument(
        "--heartbeat-stale-after-seconds",
        type=float,
        default=env_float("OTE_LIVE_HEARTBEAT_STALE_AFTER_SECONDS"),
    )
    parser.add_argument("--timezone", default=env_str("OTE_LIVE_TIMEZONE", "UTC"))
    parser.add_argument("--api-key", default=env_str("TWELVEDATA_API_KEY"))
    parser.add_argument("--max-cycles", type=int, default=env_int("OTE_LIVE_MAX_CYCLES"))
    parser.add_argument("--log-level", default=env_str("OTE_LIVE_LOG_LEVEL", "INFO"))
    signal_runtime_group = parser.add_mutually_exclusive_group()
    signal_runtime_group.add_argument("--enable-signal-runtime", dest="enable_signal_runtime", action="store_true")
    signal_runtime_group.add_argument("--disable-signal-runtime", dest="enable_signal_runtime", action="store_false")
    parser.set_defaults(enable_signal_runtime=env_bool("OTE_LIVE_ENABLE_SIGNAL_RUNTIME", True))
    chart_capture_group = parser.add_mutually_exclusive_group()
    chart_capture_group.add_argument(
        "--enable-signal-chart-capture",
        dest="enable_signal_chart_capture",
        action="store_true",
    )
    chart_capture_group.add_argument(
        "--disable-signal-chart-capture",
        dest="enable_signal_chart_capture",
        action="store_false",
    )
    parser.set_defaults(enable_signal_chart_capture=env_bool("OTE_LIVE_ENABLE_SIGNAL_CHART_CAPTURE", True))
    parser.add_argument(
        "--signal-chart-output-root",
        default=(
            str(env_path("OTE_LIVE_SIGNAL_CHART_OUTPUT_ROOT", DEFAULT_SIGNAL_CHART_OUTPUT_ROOT))
            if env_path("OTE_LIVE_SIGNAL_CHART_OUTPUT_ROOT") is not None
            else None
        ),
    )
    parser.add_argument(
        "--signal-chart-lookback-bars",
        type=int,
        default=env_int("OTE_LIVE_SIGNAL_CHART_LOOKBACK_BARS", 120),
    )
    parser.add_argument("--dashboard-url", default=env_str("OTE_LIVE_DASHBOARD_URL"))
    parser.add_argument(
        "--alert-email-recipients",
        default=env_str("OTE_LIVE_ALERT_EMAIL_RECIPIENTS", ""),
        help="Comma-separated email recipients. SMTP config is read from OTE_ALERT_EMAIL_* env vars.",
    )
    parser.add_argument(
        "--alert-sms-recipients",
        default=env_str("OTE_LIVE_ALERT_SMS_RECIPIENTS", ""),
        help="Comma-separated SMS recipients. Twilio config is read from OTE_ALERT_SMS_* env vars.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    config = LiveCollectorConfig(
        asset=args.asset,
        source_timeframe=args.source_timeframe,
        db_path=Path(args.db_path),
        poll_interval_seconds=args.poll_interval_seconds,
        stream_outputsize=args.stream_outputsize,
        startup_history_bars=args.startup_history_bars,
        startup_warmup_lookback_bars=args.startup_warmup_lookback_bars,
        startup_backfill_chunk_bars=args.startup_backfill_chunk_bars,
        heartbeat_stale_after_seconds=args.heartbeat_stale_after_seconds,
        default_timezone=args.timezone,
        log_level=args.log_level,
        max_cycles=args.max_cycles,
        enable_signal_runtime=bool(args.enable_signal_runtime),
        long_runtime_manifest_path=Path(args.long_runtime_manifest_path),
        short_runtime_manifest_path=Path(args.short_runtime_manifest_path),
        enable_signal_chart_capture=bool(args.enable_signal_chart_capture),
        signal_chart_output_root=Path(args.signal_chart_output_root) if args.signal_chart_output_root else DEFAULT_SIGNAL_CHART_OUTPUT_ROOT,
        signal_chart_lookback_bars=args.signal_chart_lookback_bars,
        dashboard_url=args.dashboard_url,
        alert_email_recipients=_parse_recipients(args.alert_email_recipients),
        alert_sms_recipients=_parse_recipients(args.alert_sms_recipients),
    )
    runtime = LiveCollectorRuntime.from_config(config, api_key=args.api_key)
    service_name = str(args.service_name)
    heartbeat_writer = ServiceHeartbeatWriter(Path(args.heartbeat_file))
    disk_monitor = DiskSpaceMonitor(
        minimum_free_bytes=int(max(0.0, float(args.min_disk_free_gb)) * (1024 ** 3)),
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
                runtime_payload=_build_runtime_payload(summary, latest_cycle_result["value"]),
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

    logging.getLogger(__name__).info(
        "Collector stopped. status=%s cycles=%s seeded=%s catchup=%s flushed=%s db=%s",
        summary.terminal_status,
        summary.cycles_run,
        summary.bootstrap.seeded_history_bars,
        summary.bootstrap.catchup_bars,
        summary.flushed_aggregated_bars,
        config.db_path,
    )
    return 0


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    if args.api_key is None:
        parser.error("Provide --api-key or set TWELVEDATA_API_KEY in the environment.")
    configure_live_logging(
        args.log_level,
        log_path=Path(args.log_file),
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Collector interrupted by operator shutdown request.")
        return 130


def _parse_recipients(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_bootstrap_payload(summary) -> dict[str, object]:
    if summary is None:
        return {}
    bootstrap = getattr(summary, "bootstrap", None)
    if bootstrap is None:
        return {}
    return {
        "latest_stored_timestamp": (
            bootstrap.latest_stored_timestamp.isoformat()
            if bootstrap.latest_stored_timestamp is not None
            else None
        ),
        "seeded_history_bars": int(bootstrap.seeded_history_bars),
        "catchup_bars": int(bootstrap.catchup_bars),
        "stored_source_bars": int(bootstrap.stored_source_bars),
        "stored_aggregated_bars": int(bootstrap.stored_aggregated_bars),
        "startup_gap_recovery_attempts": int(getattr(bootstrap, "startup_gap_recovery_attempts", 0)),
        "startup_gap_backfilled_bars": int(getattr(bootstrap, "startup_gap_backfilled_bars", 0)),
        "resolved_startup_gaps": int(getattr(bootstrap, "resolved_startup_gaps", 0)),
        "remaining_unresolved_gaps": int(getattr(bootstrap, "remaining_unresolved_gaps", 0)),
    }


def _build_runtime_payload(summary, cycle_result) -> dict[str, object]:
    if summary is None:
        return {"terminal_status": "starting"}
    payload: dict[str, object] = {
        "cycles_run": int(getattr(summary, "cycles_run", 0)),
        "evaluated_signals": int(getattr(summary, "evaluated_signals", 0)),
        "emitted_signals": int(getattr(summary, "emitted_signals", 0)),
        "sent_notifications": int(getattr(summary, "sent_notifications", 0)),
        "captured_media_artifacts": int(getattr(summary, "captured_media_artifacts", 0)),
        "flushed_aggregated_bars": int(getattr(summary, "flushed_aggregated_bars", 0)),
        "terminal_status": str(getattr(summary, "terminal_status", "running")),
        "terminal_error": dict(getattr(summary, "terminal_error", None) or {}),
    }
    if cycle_result is not None:
        payload["latest_cycle"] = {
            "polled_bars": int(cycle_result.polled_bars),
            "stored_source_bars": int(cycle_result.stored_source_bars),
            "stored_aggregated_bars": int(cycle_result.stored_aggregated_bars),
            "backfilled_bars": int(cycle_result.backfilled_bars),
            "gaps_detected": int(cycle_result.gaps_detected),
            "duplicates": int(cycle_result.duplicates),
            "out_of_order": int(cycle_result.out_of_order),
        }
    return payload


def _service_status_for_summary(summary) -> str:
    terminal_status = str(getattr(summary, "terminal_status", "completed"))
    if terminal_status == "completed":
        return "stopped"
    if terminal_status in {"cancelled", "failed"}:
        return terminal_status
    return "running"


if __name__ == "__main__":
    raise SystemExit(main())

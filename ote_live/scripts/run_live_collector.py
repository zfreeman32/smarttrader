from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from ote_live.ingestion.runtime import DEFAULT_DB_PATH, LiveCollectorConfig, LiveCollectorRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 Twelve Data live collector for canonical EURUSD bar storage.",
    )
    parser.add_argument("--asset", default="EURUSD")
    parser.add_argument("--source-timeframe", default="1m", choices=("1m",))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--stream-outputsize", type=int, default=10)
    parser.add_argument("--startup-history-bars", type=int, default=240)
    parser.add_argument("--startup-warmup-lookback-bars", type=int, default=90)
    parser.add_argument("--startup-backfill-chunk-bars", type=int, default=1000)
    parser.add_argument("--heartbeat-stale-after-seconds", type=float, default=90.0)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


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
    )
    runtime = LiveCollectorRuntime.from_config(config, api_key=args.api_key)
    summary = await runtime.run()

    logging.getLogger(__name__).info(
        "Collector stopped. cycles=%s seeded=%s catchup=%s flushed=%s db=%s",
        summary.cycles_run,
        summary.bootstrap.seeded_history_bars,
        summary.bootstrap.catchup_bars,
        summary.flushed_aggregated_bars,
        config.db_path,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.api_key is None and not os.getenv("TWELVEDATA_API_KEY"):
        parser.error("Provide --api-key or set TWELVEDATA_API_KEY in the environment.")
    configure_logging(args.log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from ote_live.env import env_bool, env_int, env_path, env_str, load_repo_env
from ote_live.dashboard.app import create_dashboard_app
from ote_live.ingestion.runtime import (
    DEFAULT_DB_PATH,
    DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description="Run the Phase 5 live operator dashboard against the local live SQLite store.",
    )
    parser.add_argument("--db-path", default=str(env_path("OTE_LIVE_DB_PATH", DEFAULT_DB_PATH)))
    parser.add_argument(
        "--long-runtime-manifest-path",
        default=str(env_path("OTE_LIVE_LONG_RUNTIME_MANIFEST_PATH", DEFAULT_LONG_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument(
        "--short-runtime-manifest-path",
        default=str(env_path("OTE_LIVE_SHORT_RUNTIME_MANIFEST_PATH", DEFAULT_SHORT_RUNTIME_MANIFEST_PATH)),
    )
    parser.add_argument("--asset", default=env_str("OTE_LIVE_ASSET", "EURUSD"))
    parser.add_argument("--timeframe", default=env_str("OTE_LIVE_DASHBOARD_TIMEFRAME", "5m"))
    parser.add_argument("--host", default=env_str("OTE_LIVE_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env_int("OTE_LIVE_DASHBOARD_PORT", 8050))
    parser.add_argument(
        "--refresh-interval-ms",
        type=int,
        default=env_int("OTE_LIVE_DASHBOARD_REFRESH_INTERVAL_MS", 10_000),
    )
    parser.add_argument("--signal-limit", type=int, default=env_int("OTE_LIVE_DASHBOARD_SIGNAL_LIMIT", 120))
    parser.add_argument("--debug", action="store_true", default=env_bool("OTE_LIVE_DASHBOARD_DEBUG", False))
    return parser


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    app = create_dashboard_app(
        Path(args.db_path),
        asset=args.asset,
        timeframe=args.timeframe,
        refresh_interval_ms=args.refresh_interval_ms,
        signal_limit=args.signal_limit,
        long_runtime_manifest_path=Path(args.long_runtime_manifest_path),
        short_runtime_manifest_path=Path(args.short_runtime_manifest_path),
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=bool(args.debug),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

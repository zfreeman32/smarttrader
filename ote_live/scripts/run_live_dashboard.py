from __future__ import annotations

import argparse
from pathlib import Path

from ote_live.env import env_bool, env_int, env_path, env_str, load_repo_env
from ote_live.dashboard.app import create_dashboard_app
from ote_live.dashboard.view_registry import (
    DEFAULT_FRVP_ACTIVE_WEIGHT_MODEL_IDS,
    DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_FRVP_REGISTRY_PATH,
    DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_ACTIVE_WEIGHT_MODEL_IDS,
    DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_ICT_MODEL_ORDER,
    DEFAULT_ICT_REGISTRY_PATH,
    DEFAULT_ICT_SHORT_RUNTIME_MANIFEST_PATH,
    DashboardViewConfig,
)
from ote_live.ingestion.runtime import DEFAULT_DB_PATH, DEFAULT_LONG_RUNTIME_MANIFEST_PATH, DEFAULT_SHORT_RUNTIME_MANIFEST_PATH


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description="Run the Phase 5 live operator dashboard against the local live SQLite store.",
    )
    parser.add_argument(
        "--db-path",
        default=str(env_path("OTE_LIVE_DB_PATH", env_path("FRVP_LIVE_DB_PATH", DEFAULT_DB_PATH) or DEFAULT_DB_PATH)),
    )
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
    parser.add_argument("--frvp-asset", default=env_str("FRVP_LIVE_ASSET", "ES"))
    parser.add_argument(
        "--frvp-timeframe",
        default=env_str("FRVP_LIVE_DASHBOARD_TIMEFRAME", env_str("FRVP_LIVE_TIMEFRAME", "5m")),
    )
    parser.add_argument("--ict-asset", default=env_str("ICT_LIVE_ASSET", "ES"))
    parser.add_argument(
        "--ict-timeframe",
        default=env_str("ICT_LIVE_DASHBOARD_TIMEFRAME", env_str("ICT_LIVE_TIMEFRAME", "5m")),
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
        "--frvp-registry-path",
        default=str(env_path("FRVP_LIVE_REGISTRY_PATH", DEFAULT_FRVP_REGISTRY_PATH)),
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
        "--ict-registry-path",
        default=str(env_path("ICT_LIVE_REGISTRY_PATH", DEFAULT_ICT_REGISTRY_PATH)),
    )
    parser.add_argument("--frvp-data-supplier", default=env_str("FRVP_LIVE_DATA_SUPPLIER", "IBKR"))
    parser.add_argument("--ict-data-supplier", default=env_str("ICT_LIVE_DATA_SUPPLIER", "IBKR"))
    parser.add_argument("--host", default=env_str("OTE_LIVE_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env_int("OTE_LIVE_DASHBOARD_PORT", 8050))
    parser.add_argument(
        "--refresh-interval-ms",
        type=int,
        default=env_int("OTE_LIVE_DASHBOARD_REFRESH_INTERVAL_MS", 10_000),
    )
    parser.add_argument("--signal-limit", type=int, default=env_int("OTE_LIVE_DASHBOARD_SIGNAL_LIMIT", 120))
    parser.add_argument(
        "--model-chart-lookback-hours",
        type=int,
        default=env_int("OTE_LIVE_DASHBOARD_MODEL_CHART_LOOKBACK_HOURS", 24),
    )
    parser.add_argument("--debug", action="store_true", default=env_bool("OTE_LIVE_DASHBOARD_DEBUG", False))
    return parser


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()
    view_configs = (
        DashboardViewConfig(
            view_id="OTE",
            label="OTE",
            asset=args.asset,
            timeframe=args.timeframe,
            data_supplier=env_str("OTE_LIVE_DATA_SUPPLIER", "FMP") or "FMP",
            long_runtime_manifest_path=Path(args.long_runtime_manifest_path),
            short_runtime_manifest_path=Path(args.short_runtime_manifest_path),
            description=f"{args.asset} {args.timeframe} live operator view",
            recent_activity_title="Recent Signals",
        ),
        DashboardViewConfig(
            view_id="FRVP",
            label="FRVP",
            asset=args.frvp_asset,
            timeframe=args.frvp_timeframe,
            data_supplier=args.frvp_data_supplier,
            long_runtime_manifest_path=Path(args.frvp_long_runtime_manifest_path),
            short_runtime_manifest_path=Path(args.frvp_short_runtime_manifest_path),
            registry_path=Path(args.frvp_registry_path),
            preferred_model_order=(
                "frvp_long_continuation_xgb_v1",
                "frvp_long_reversal_xgb_v1",
                "frvp_short_meta_xgb_v1",
            ),
            active_weight_model_ids=DEFAULT_FRVP_ACTIVE_WEIGHT_MODEL_IDS,
            description=f"{args.frvp_asset} {args.frvp_timeframe} FRVP shadow operator view",
            recent_activity_title="Recent FRVP Setups",
            enable_frvp_overlays=True,
            runtime_state_key="FRVP",
        ),
        DashboardViewConfig(
            view_id="ICT",
            label="ICT",
            asset=args.ict_asset,
            timeframe=args.ict_timeframe,
            data_supplier=args.ict_data_supplier,
            long_runtime_manifest_path=Path(args.ict_long_runtime_manifest_path),
            short_runtime_manifest_path=Path(args.ict_short_runtime_manifest_path),
            registry_path=Path(args.ict_registry_path),
            preferred_model_order=DEFAULT_ICT_MODEL_ORDER,
            active_weight_model_ids=DEFAULT_ICT_ACTIVE_WEIGHT_MODEL_IDS,
            description=f"{args.ict_asset} {args.ict_timeframe} ICT controlled paper-signal view",
            recent_activity_title="Recent ICT Setups",
            enable_ict_overlays=True,
            runtime_state_key="ICT",
        ),
    )
    app = create_dashboard_app(
        Path(args.db_path),
        refresh_interval_ms=args.refresh_interval_ms,
        signal_limit=args.signal_limit,
        model_chart_lookback_hours=args.model_chart_lookback_hours,
        view_configs=view_configs,
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=bool(args.debug),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ote_live.env import env_path, env_str
from ote_live.ingestion.runtime import (
    DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260705" / "live_runtime_manifest_long.json"
)
DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260705" / "live_runtime_manifest_short.json"
)
DEFAULT_FRVP_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_shadow_live_registry_20260705.json"
DEFAULT_FRVP_MODEL_ORDER = (
    "frvp_long_continuation_xgb_v1",
    "frvp_long_reversal_xgb_v1",
    "frvp_short_meta_xgb_v1",
)


@dataclass(frozen=True)
class DashboardViewConfig:
    view_id: str
    label: str
    asset: str
    timeframe: str
    data_supplier: str
    long_runtime_manifest_path: Path
    short_runtime_manifest_path: Path
    registry_path: Path | None = None
    preferred_model_order: tuple[str, ...] = ()
    description: str = ""
    recent_activity_title: str = "Recent Signals"
    enable_frvp_overlays: bool = False
    runtime_state_key: str | None = None

    @property
    def resolved_runtime_state_key(self) -> str:
        return self.runtime_state_key or self.view_id


def build_default_dashboard_views() -> tuple[DashboardViewConfig, ...]:
    ote_asset = env_str("OTE_LIVE_ASSET", "EURUSD") or "EURUSD"
    ote_timeframe = (
        env_str("OTE_LIVE_DASHBOARD_TIMEFRAME")
        or env_str("OTE_LIVE_SOURCE_TIMEFRAME")
        or "5m"
    )
    frvp_asset = env_str("FRVP_LIVE_ASSET", "ES") or "ES"
    frvp_timeframe = (
        env_str("FRVP_LIVE_DASHBOARD_TIMEFRAME")
        or env_str("FRVP_LIVE_TIMEFRAME")
        or "5m"
    )

    return (
        DashboardViewConfig(
            view_id="OTE",
            label="OTE",
            asset=ote_asset,
            timeframe=ote_timeframe,
            data_supplier=env_str("OTE_LIVE_DATA_SUPPLIER", "FMP") or "FMP",
            long_runtime_manifest_path=env_path(
                "OTE_LIVE_LONG_RUNTIME_MANIFEST_PATH",
                DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
            )
            or DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
            short_runtime_manifest_path=env_path(
                "OTE_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
                DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
            )
            or DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
            description=f"{ote_asset} {ote_timeframe} live operator view",
            recent_activity_title="Recent Signals",
        ),
        DashboardViewConfig(
            view_id="FRVP",
            label="FRVP",
            asset=frvp_asset,
            timeframe=frvp_timeframe,
            data_supplier=env_str("FRVP_LIVE_DATA_SUPPLIER", "IBKR") or "IBKR",
            long_runtime_manifest_path=env_path(
                "FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
                DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH,
            )
            or DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH,
            short_runtime_manifest_path=env_path(
                "FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
                DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH,
            )
            or DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH,
            registry_path=env_path("FRVP_LIVE_REGISTRY_PATH", DEFAULT_FRVP_REGISTRY_PATH) or DEFAULT_FRVP_REGISTRY_PATH,
            preferred_model_order=DEFAULT_FRVP_MODEL_ORDER,
            description=f"{frvp_asset} {frvp_timeframe} FRVP shadow operator view",
            recent_activity_title="Recent FRVP Setups",
            enable_frvp_overlays=True,
            runtime_state_key="FRVP",
        ),
    )

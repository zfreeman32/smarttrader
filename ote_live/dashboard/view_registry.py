from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ote_live.env import env_path, env_str
from ote_live.ingestion.runtime import (
    DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
    DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260721" / "live_runtime_manifest_long.json"
)
DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260721" / "live_runtime_manifest_short.json"
)
DEFAULT_FRVP_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_shadow_live_registry_20260721.json"
DEFAULT_FRVP_MODEL_ORDER = (
    "frvp_short_continuation_tcn_v1",
    "frvp_long_continuation_xgb_v1",
    "frvp_long_reversal_xgb_v1",
    "frvp_short_meta_xgb_v1",
)
DEFAULT_FRVP_ACTIVE_WEIGHT_MODEL_IDS = ()
FRVP_PAPER_SIGNAL_BUNDLE_ID = "frvp_es_paper_signal_20260816"
FRVP_PAPER_SIGNAL_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / FRVP_PAPER_SIGNAL_BUNDLE_ID
    / "live_runtime_manifest_long.json"
)
FRVP_PAPER_SIGNAL_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / FRVP_PAPER_SIGNAL_BUNDLE_ID
    / "live_runtime_manifest_short.json"
)
FRVP_PAPER_SIGNAL_REGISTRY_PATH = (
    REPO_ROOT / "models" / "frvp_es_paper_signal_registry_20260816.json"
)
FRVP_PAPER_SIGNAL_ACTIVE_WEIGHT_MODEL_IDS = (
    "frvp_long_reversal_xgb_v1",
)
FRVP_PAPER_SIGNAL_CONTENT_SHA256 = {
    "long_runtime_manifest": "35f8aecb213d9a2bae7e641f6e87f4a569309b2f5adae7b1e0bdf4ebbcf549d8",
    "short_runtime_manifest": "d1dfc8276f40f1f6fa82090d9752030249714d5e08503697d57a495e062f0140",
    "registry": "98c0f63514c1ec93f6899929976774fc0f984f1a142001db07e1ed238b9b071e",
}
FRVP_SETUP_FAMILY_BUNDLE_ID = "frvp_es_setup_family_20260829"
DEFAULT_FRVP_SETUP_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / FRVP_SETUP_FAMILY_BUNDLE_ID
    / "live_runtime_manifest_long.json"
)
DEFAULT_FRVP_SETUP_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / FRVP_SETUP_FAMILY_BUNDLE_ID
    / "live_runtime_manifest_short.json"
)
DEFAULT_FRVP_SETUP_REGISTRY_PATH = (
    REPO_ROOT / "models" / "frvp_es_primary_model_registry_long_setup_fullspan_20260829.json"
)
DEFAULT_FRVP_SETUP_MODEL_ORDER = (
    "frvp_long_continuation_setup3_xgb_v1",
    "frvp_long_continuation_setup5_xgb_v1",
    "frvp_long_reversal_setup1_xgb_v1",
    "frvp_long_continuation_setup2_xgb_v1",
    "frvp_long_reversal_setup6_xgb_v1",
    "frvp_long_reversal_setup4_xgb_v1",
)
DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "ict_es_paper_signal_20260813" / "live_runtime_manifest_long.json"
)
DEFAULT_ICT_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "ict_es_paper_signal_20260813" / "live_runtime_manifest_short.json"
)
DEFAULT_ICT_REGISTRY_PATH = REPO_ROOT / "models" / "ict_es_paper_signal_registry_20260813.json"
DEFAULT_ICT_MODEL_ORDER = (
    "ict_long_meta_xgb_v1",
    "ict_long_reversal_xgb_v1",
    "ict_short_reversal_xgb_v1",
    "ict_short_meta_xgb_v1",
    "ict_short_continuation_xgb_v1",
    "ict_long_continuation_xgb_v1",
)
DEFAULT_ICT_ACTIVE_WEIGHT_MODEL_IDS = (
    "ict_long_meta_xgb_v1",
)
ICT_SETUP_FAMILY_BUNDLE_ID = "ict_short_setup_family_20260811_audit"
DEFAULT_ICT_SETUP_LONG_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / ICT_SETUP_FAMILY_BUNDLE_ID
    / "live_runtime_manifest_long.json"
)
DEFAULT_ICT_SETUP_SHORT_RUNTIME_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / ICT_SETUP_FAMILY_BUNDLE_ID
    / "live_runtime_manifest_short.json"
)
DEFAULT_ICT_SETUP_REGISTRY_PATH = (
    REPO_ROOT / "models" / "ict_short_setup_family_model_registry_20260811_audit.json"
)
DEFAULT_ICT_SETUP_MODEL_ORDER = (
    "ict_short_reversal_sweep_reclaim_xgb_v1",
    "ict_short_reversal_ifvg_reversal_xgb_v1",
    "ict_short_continuation_premium_discount_continuation_xgb_v1",
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
    active_weight_model_ids: tuple[str, ...] = ()
    description: str = ""
    recent_activity_title: str = "Recent Signals"
    enable_frvp_overlays: bool = False
    enable_ict_overlays: bool = False
    runtime_state_key: str | None = None

    @property
    def resolved_runtime_state_key(self) -> str:
        return self.runtime_state_key or self.view_id

    @property
    def has_runtime_state_overlays(self) -> bool:
        return self.enable_frvp_overlays or self.enable_ict_overlays


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
    frvp_long_manifest_path = env_path(
        "FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
        DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH,
    ) or DEFAULT_FRVP_LONG_RUNTIME_MANIFEST_PATH
    frvp_short_manifest_path = env_path(
        "FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
        DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH,
    ) or DEFAULT_FRVP_SHORT_RUNTIME_MANIFEST_PATH
    frvp_registry_path = (
        env_path("FRVP_LIVE_REGISTRY_PATH", DEFAULT_FRVP_REGISTRY_PATH)
        or DEFAULT_FRVP_REGISTRY_PATH
    )
    frvp_setup_long_manifest_path = env_path(
        "FRVP_SETUP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
        DEFAULT_FRVP_SETUP_LONG_RUNTIME_MANIFEST_PATH,
    ) or DEFAULT_FRVP_SETUP_LONG_RUNTIME_MANIFEST_PATH
    frvp_setup_short_manifest_path = env_path(
        "FRVP_SETUP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
        DEFAULT_FRVP_SETUP_SHORT_RUNTIME_MANIFEST_PATH,
    ) or DEFAULT_FRVP_SETUP_SHORT_RUNTIME_MANIFEST_PATH
    frvp_setup_registry_path = (
        env_path("FRVP_SETUP_LIVE_REGISTRY_PATH", DEFAULT_FRVP_SETUP_REGISTRY_PATH)
        or DEFAULT_FRVP_SETUP_REGISTRY_PATH
    )
    frvp_active_weight_model_ids, frvp_view_label = (
        resolve_frvp_dashboard_presentation(
            frvp_long_manifest_path,
            frvp_short_manifest_path,
            frvp_registry_path,
        )
    )
    ict_asset = env_str("ICT_LIVE_ASSET", "ES") or "ES"
    ict_timeframe = (
        env_str("ICT_LIVE_DASHBOARD_TIMEFRAME")
        or env_str("ICT_LIVE_TIMEFRAME")
        or "5m"
    )
    ict_setup_long_manifest_path = env_path(
        "ICT_SETUP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
        DEFAULT_ICT_SETUP_LONG_RUNTIME_MANIFEST_PATH,
    ) or DEFAULT_ICT_SETUP_LONG_RUNTIME_MANIFEST_PATH
    ict_setup_short_manifest_path = env_path(
        "ICT_SETUP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
        DEFAULT_ICT_SETUP_SHORT_RUNTIME_MANIFEST_PATH,
    ) or DEFAULT_ICT_SETUP_SHORT_RUNTIME_MANIFEST_PATH
    ict_setup_registry_path = (
        env_path("ICT_SETUP_LIVE_REGISTRY_PATH", DEFAULT_ICT_SETUP_REGISTRY_PATH)
        or DEFAULT_ICT_SETUP_REGISTRY_PATH
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
            long_runtime_manifest_path=frvp_long_manifest_path,
            short_runtime_manifest_path=frvp_short_manifest_path,
            registry_path=frvp_registry_path,
            preferred_model_order=DEFAULT_FRVP_MODEL_ORDER,
            active_weight_model_ids=frvp_active_weight_model_ids,
            description=f"{frvp_asset} {frvp_timeframe} FRVP {frvp_view_label}",
            recent_activity_title="Recent FRVP Setups",
            enable_frvp_overlays=True,
            runtime_state_key="FRVP",
        ),
        DashboardViewConfig(
            view_id="FRVP_SETUP",
            label="FRVP Setup Models",
            asset=env_str("FRVP_SETUP_LIVE_ASSET", frvp_asset) or frvp_asset,
            timeframe=(
                env_str("FRVP_SETUP_LIVE_DASHBOARD_TIMEFRAME")
                or env_str("FRVP_SETUP_LIVE_TIMEFRAME")
                or frvp_timeframe
            ),
            data_supplier=(
                env_str(
                    "FRVP_SETUP_LIVE_DATA_SUPPLIER",
                    env_str("FRVP_LIVE_DATA_SUPPLIER", "IBKR"),
                )
                or "IBKR"
            ),
            long_runtime_manifest_path=frvp_setup_long_manifest_path,
            short_runtime_manifest_path=frvp_setup_short_manifest_path,
            registry_path=frvp_setup_registry_path,
            preferred_model_order=DEFAULT_FRVP_SETUP_MODEL_ORDER,
            active_weight_model_ids=(),
            description=f"{frvp_asset} {frvp_timeframe} FRVP setup-family shadow operator view",
            recent_activity_title="Recent FRVP Setup-Model Decisions",
            enable_frvp_overlays=True,
            runtime_state_key="FRVP",
        ),
        DashboardViewConfig(
            view_id="ICT",
            label="ICT",
            asset=ict_asset,
            timeframe=ict_timeframe,
            data_supplier=env_str("ICT_LIVE_DATA_SUPPLIER", "IBKR") or "IBKR",
            long_runtime_manifest_path=env_path(
                "ICT_LIVE_LONG_RUNTIME_MANIFEST_PATH",
                DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH,
            )
            or DEFAULT_ICT_LONG_RUNTIME_MANIFEST_PATH,
            short_runtime_manifest_path=env_path(
                "ICT_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
                DEFAULT_ICT_SHORT_RUNTIME_MANIFEST_PATH,
            )
            or DEFAULT_ICT_SHORT_RUNTIME_MANIFEST_PATH,
            registry_path=env_path("ICT_LIVE_REGISTRY_PATH", DEFAULT_ICT_REGISTRY_PATH) or DEFAULT_ICT_REGISTRY_PATH,
            preferred_model_order=DEFAULT_ICT_MODEL_ORDER,
            active_weight_model_ids=DEFAULT_ICT_ACTIVE_WEIGHT_MODEL_IDS,
            description=f"{ict_asset} {ict_timeframe} ICT controlled paper-signal operator view",
            recent_activity_title="Recent ICT Setups",
            enable_ict_overlays=True,
            runtime_state_key="ICT",
        ),
        DashboardViewConfig(
            view_id="ICT_SETUP",
            label="ICT Setup Models",
            asset=env_str("ICT_SETUP_LIVE_ASSET", ict_asset) or ict_asset,
            timeframe=(
                env_str("ICT_SETUP_LIVE_DASHBOARD_TIMEFRAME")
                or env_str("ICT_SETUP_LIVE_TIMEFRAME")
                or ict_timeframe
            ),
            data_supplier=(
                env_str(
                    "ICT_SETUP_LIVE_DATA_SUPPLIER",
                    env_str("ICT_LIVE_DATA_SUPPLIER", "IBKR"),
                )
                or "IBKR"
            ),
            long_runtime_manifest_path=ict_setup_long_manifest_path,
            short_runtime_manifest_path=ict_setup_short_manifest_path,
            registry_path=ict_setup_registry_path,
            preferred_model_order=DEFAULT_ICT_SETUP_MODEL_ORDER,
            active_weight_model_ids=(),
            description=f"{ict_asset} {ict_timeframe} ICT setup-family shadow operator view",
            recent_activity_title="Recent ICT Setup-Model Decisions",
            enable_ict_overlays=True,
            runtime_state_key="ICT",
        ),
    )


def resolve_frvp_dashboard_presentation(
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
    registry_path: str | Path,
) -> tuple[tuple[str, ...], str]:
    """Return status badges/copy that match the exact FRVP bundle on screen."""

    if not is_frvp_paper_signal_bundle_path_set(
        long_runtime_manifest_path,
        short_runtime_manifest_path,
        registry_path,
    ):
        return DEFAULT_FRVP_ACTIVE_WEIGHT_MODEL_IDS, "shadow operator view"
    if validate_frvp_paper_signal_bundle_contents(
        long_runtime_manifest_path,
        short_runtime_manifest_path,
        registry_path,
    ):
        return (
            FRVP_PAPER_SIGNAL_ACTIVE_WEIGHT_MODEL_IDS,
            "controlled paper-signal view",
        )
    return (), "INVALID controlled-bundle content (controlled label refused)"


def is_frvp_paper_signal_bundle_path_set(
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
    registry_path: str | Path,
) -> bool:
    """Return whether the dashboard points at the immutable bundle locations."""

    configured_paths = tuple(
        Path(path).resolve()
        for path in (
            long_runtime_manifest_path,
            short_runtime_manifest_path,
            registry_path,
        )
    )
    controlled_paths = (
        FRVP_PAPER_SIGNAL_LONG_RUNTIME_MANIFEST_PATH.resolve(),
        FRVP_PAPER_SIGNAL_SHORT_RUNTIME_MANIFEST_PATH.resolve(),
        FRVP_PAPER_SIGNAL_REGISTRY_PATH.resolve(),
    )
    return configured_paths == controlled_paths


def validate_frvp_paper_signal_bundle_contents(
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
    registry_path: str | Path,
) -> bool:
    """Verify every configured controlled-bundle file against its frozen digest."""

    configured_files = (
        ("long_runtime_manifest", Path(long_runtime_manifest_path)),
        ("short_runtime_manifest", Path(short_runtime_manifest_path)),
        ("registry", Path(registry_path)),
    )
    try:
        return all(
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest()
            == FRVP_PAPER_SIGNAL_CONTENT_SHA256[name]
            for name, path in configured_files
        )
    except OSError:
        return False

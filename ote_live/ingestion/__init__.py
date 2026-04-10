from importlib import import_module

from .aggregator import MultiTimeframeBarAggregator, aggregate_bar_sequence
from .base import (
    AbstractBackfillConnector,
    AbstractBarStream,
    BackfillWindow,
    CanonicalTimeframe,
    GapCheckResult,
    HeartbeatStatus,
    IngestionGap,
    canonical_asset_symbol,
    timeframe_to_minutes,
    timeframe_to_timedelta,
)
from .connector_backfill import (
    FMPAPIError,
    FMPBackfillConnector,
    FMPConfig,
    FMPRESTClient,
    TwelveDataAPIError,
    TwelveDataBackfillConnector,
    TwelveDataConfig,
    TwelveDataRESTClient,
)
from .connector_stream import FMPPollingBarStream, TwelveDataPollingBarStream
from .gap_detector import GapDetector, detect_gaps
from .heartbeat import HeartbeatMonitor
from .normalizer import (
    bars_to_dataframe,
    canonical_asset_to_fmp_symbol,
    canonical_timeframe_to_fmp_interval,
    dataframe_to_market_bars,
    fmp_interval_to_canonical_timeframe,
    load_local_csv_bars,
    normalize_fmp_historical_chart_response,
)

__all__ = [
    "AbstractBackfillConnector",
    "AbstractBarStream",
    "BackfillWindow",
    "BootstrapSummary",
    "CanonicalTimeframe",
    "CollectorCycleResult",
    "CollectorRunSummary",
    "DEFAULT_DB_PATH",
    "FMPAPIError",
    "FMPBackfillConnector",
    "FMPConfig",
    "FMPPollingBarStream",
    "FMPRESTClient",
    "GapCheckResult",
    "GapDetector",
    "HeartbeatMonitor",
    "HeartbeatStatus",
    "IngestionGap",
    "LiveBarIngestionService",
    "LiveCollectorConfig",
    "LiveCollectorRuntime",
    "MultiTimeframeBarAggregator",
    "TwelveDataAPIError",
    "TwelveDataBackfillConnector",
    "TwelveDataConfig",
    "TwelveDataPollingBarStream",
    "TwelveDataRESTClient",
    "aggregate_bar_sequence",
    "bars_to_dataframe",
    "canonical_asset_symbol",
    "canonical_asset_to_fmp_symbol",
    "canonical_timeframe_to_fmp_interval",
    "dataframe_to_market_bars",
    "detect_gaps",
    "fmp_interval_to_canonical_timeframe",
    "load_local_csv_bars",
    "normalize_fmp_historical_chart_response",
    "timeframe_to_minutes",
    "timeframe_to_timedelta",
]


def __getattr__(name: str):
    if name in {
        "CollectorCycleResult",
        "LiveBarIngestionService",
    }:
        module = import_module(".service", __name__)
        return getattr(module, name)
    if name in {
        "DEFAULT_DB_PATH",
        "BootstrapSummary",
        "CollectorRunSummary",
        "LiveCollectorConfig",
        "LiveCollectorRuntime",
    }:
        module = import_module(".runtime", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

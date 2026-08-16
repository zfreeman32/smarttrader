from __future__ import annotations

from .queries import (
    DashboardHealthSummary,
    DashboardPerformanceSummary,
    build_health_summary,
    compute_signal_markouts,
    fetch_confidence_history,
    fetch_recent_bars,
    fetch_recent_health_events,
    fetch_recent_signals,
    summarize_signal_markouts,
)

__all__ = [
    "DashboardHealthSummary",
    "DashboardPerformanceSummary",
    "build_health_summary",
    "compute_signal_markouts",
    "create_dashboard_app",
    "fetch_confidence_history",
    "fetch_recent_bars",
    "fetch_recent_health_events",
    "fetch_recent_signals",
    "summarize_signal_markouts",
]


def __getattr__(name: str):
    if name == "create_dashboard_app":
        from .app import create_dashboard_app

        return create_dashboard_app
    raise AttributeError(name)

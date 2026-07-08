"""Compatibility shim for the design-paper equity-session path."""

from frvp.sessions.equity import build_equity_market_day_labels, build_equity_session_frame

__all__ = [
    "build_equity_market_day_labels",
    "build_equity_session_frame",
]

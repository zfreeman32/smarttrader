from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ICTThresholdSearchConfig:
    """Starting point for ICT threshold and policy search."""

    instrument: str = "es"
    min_positive_events: int = 50
    min_events_per_month: float = 3.0
    min_trades_per_week: float = 3.0
    max_trades_per_day: int = 3


@dataclass(frozen=True)
class ICTAcceptanceGates:
    """Initial OOS acceptance gates aligned with the design paper."""

    minimum_sharpe: float = 1.0
    preferred_sharpe: float = 1.5
    minimum_dsr: float = 0.3
    preferred_dsr: float = 0.8
    minimum_profit_factor: float = 1.2
    minimum_profitable_quarter_share: float = 0.60
    minimum_positive_composite_expectancy_share: float = 0.55

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from frvp.config.instruments import get_instrument_config
from model_testing.ote_abstain_policy import DEFAULT_SESSION_SPREAD_PIPS


DEFAULT_ES_SESSION_SPREAD_TICKS = {
    "overlap": 1.0,
    "london": 1.0,
    "new_york": 1.0,
    "asia": 1.5,
    "off_hours": 2.0,
}

DEFAULT_6E_SESSION_SPREAD_TICKS = {
    "overlap": 1.0,
    "london": 1.0,
    "new_york": 1.0,
    "asia": 1.5,
    "off_hours": 2.0,
}


@dataclass(frozen=True)
class EvaluationCostConfig:
    instrument: str
    unit_label: str
    price_increment: float
    session_spread_units: Mapping[str, float]
    fixed_slippage_units_per_trade: float
    commission_units_per_trade: float
    tick_size: float | None = None
    tick_value: float | None = None


def resolve_evaluation_cost_config(
    instrument: str | None,
    *,
    fixed_slippage_units_per_trade: float | None = None,
    commission_units_per_trade: float | None = None,
) -> EvaluationCostConfig:
    normalized = normalize_evaluation_instrument(instrument)
    if normalized in {"es", "6e"}:
        instrument_config = get_instrument_config(normalized)
        default_spread_units = (
            DEFAULT_ES_SESSION_SPREAD_TICKS
            if normalized == "es"
            else DEFAULT_6E_SESSION_SPREAD_TICKS
        )
        default_slippage_units = 0.25
        default_commission_units = 0.40
        return EvaluationCostConfig(
            instrument=normalized,
            unit_label="ticks",
            price_increment=float(instrument_config.tick_size),
            session_spread_units={
                str(key): float(value) for key, value in default_spread_units.items()
            },
            fixed_slippage_units_per_trade=float(
                default_slippage_units
                if fixed_slippage_units_per_trade is None
                else fixed_slippage_units_per_trade
            ),
            commission_units_per_trade=float(
                default_commission_units
                if commission_units_per_trade is None
                else commission_units_per_trade
            ),
            tick_size=float(instrument_config.tick_size),
            tick_value=float(instrument_config.tick_value),
        )

    default_spread_units = {
        str(key): float(value) for key, value in DEFAULT_SESSION_SPREAD_PIPS.items()
    }
    return EvaluationCostConfig(
        instrument=normalized,
        unit_label="pips",
        price_increment=1e-4,
        session_spread_units=default_spread_units,
        fixed_slippage_units_per_trade=float(
            0.3 if fixed_slippage_units_per_trade is None else fixed_slippage_units_per_trade
        ),
        commission_units_per_trade=float(
            0.35 if commission_units_per_trade is None else commission_units_per_trade
        ),
    )


def describe_evaluation_cost_config(config: EvaluationCostConfig) -> dict[str, object]:
    description: dict[str, object] = {
        "instrument": config.instrument,
        "unit_label": config.unit_label,
        "price_increment": float(config.price_increment),
        "session_spread_units": {
            str(key): float(value) for key, value in dict(config.session_spread_units).items()
        },
        "fixed_slippage_units_per_trade": float(config.fixed_slippage_units_per_trade),
        "commission_units_per_trade": float(config.commission_units_per_trade),
    }
    if config.tick_size is not None:
        description["tick_size"] = float(config.tick_size)
    if config.tick_value is not None:
        description["tick_value"] = float(config.tick_value)
    return description


def normalize_evaluation_instrument(instrument: str | None) -> str:
    if instrument is None:
        return "fx"
    normalized = str(instrument).strip().lower()
    if normalized in {"eurusd", "spot_fx"}:
        return "fx"
    return normalized

from __future__ import annotations

from typing import Any, Mapping


_DRAWDOWN_ACCEPTANCE_KEYS = frozenset(
    {
        "max_drawdown_pct_below_threshold",
        "max_drawdown_less_than_two_times_average_monthly_profit",
    }
)


def drawdown_acceptance_passed(acceptance: Mapping[str, Any] | None) -> bool:
    if not isinstance(acceptance, Mapping):
        return False
    for key in _DRAWDOWN_ACCEPTANCE_KEYS:
        if key in acceptance:
            return bool(acceptance.get(key))
    return False


def accepted_for_paper_trading(
    acceptance: Mapping[str, Any] | None,
    *,
    ignore_drawdown_gate: bool = True,
) -> bool:
    if not isinstance(acceptance, Mapping) or not acceptance:
        return False

    checked_non_drawdown_gate = False
    for key, value in acceptance.items():
        if ignore_drawdown_gate and key in _DRAWDOWN_ACCEPTANCE_KEYS:
            continue
        checked_non_drawdown_gate = True
        if not bool(value):
            return False

    if checked_non_drawdown_gate:
        return True
    return all(bool(value) for value in acceptance.values())

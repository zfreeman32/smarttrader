from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from model_testing.ote_regime_labeler import RegimeLabelConfig, label_regimes
from ote_live.features.manifest import TimezoneContract


@dataclass(frozen=True)
class LatestRegime:
    trend_regime: str | None
    vol_regime: str | None
    session_regime: str | None
    stress_regime: str | None
    composite_regime: str | None


_REGIME_COLUMNS = frozenset(
    {
        "trend_regime",
        "vol_regime",
        "session_regime",
        "stress_regime",
        "composite_regime",
    }
)


def label_policy_frame(
    policy_frame: pd.DataFrame,
    *,
    timezone_contract: TimezoneContract | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    config = build_regime_config(timezone_contract)
    return label_regimes(policy_frame, config=config)


def resolve_policy_context(
    policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
    *,
    timezone_contract: TimezoneContract | Mapping[str, Any] | None = None,
    strict: bool = False,
) -> pd.DataFrame | None:
    if policy_context is None:
        return None

    frame = _coerce_policy_frame(policy_context)
    if frame.empty or _REGIME_COLUMNS.issubset(frame.columns):
        return frame

    try:
        labeled = label_policy_frame(frame, timezone_contract=timezone_contract)
    except Exception:
        if strict:
            raise
        return frame

    for column in _REGIME_COLUMNS.intersection(frame.columns):
        labeled[column] = frame[column].where(frame[column].notna(), labeled[column])
    return labeled


def resolve_latest_regime(
    policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
    *,
    timezone_contract: TimezoneContract | Mapping[str, Any] | None = None,
    strict: bool = False,
) -> LatestRegime | None:
    working = resolve_policy_context(
        policy_context,
        timezone_contract=timezone_contract,
        strict=strict,
    )
    if working is None or working.empty:
        return None
    if "composite_regime" not in working.columns:
        return None

    latest = working.iloc[-1]
    return LatestRegime(
        trend_regime=_optional_str(latest.get("trend_regime")),
        vol_regime=_optional_str(latest.get("vol_regime")),
        session_regime=_optional_str(latest.get("session_regime")),
        stress_regime=_optional_str(latest.get("stress_regime")),
        composite_regime=_optional_str(latest.get("composite_regime")),
    )


def build_regime_config(
    timezone_contract: TimezoneContract | Mapping[str, Any] | None = None,
) -> RegimeLabelConfig:
    if timezone_contract is None:
        return RegimeLabelConfig()

    payload = (
        timezone_contract.model_dump()
        if isinstance(timezone_contract, TimezoneContract)
        else dict(timezone_contract)
    )
    canonical_timezone = str(payload.get("canonical_timezone", "UTC"))
    return RegimeLabelConfig(
        source_timezone=canonical_timezone,
        canonical_timezone=canonical_timezone,
        feature_clock_timezone=str(payload.get("feature_clock_timezone", "America/New_York")),
        london_timezone=str(payload.get("london_timezone", "Europe/London")),
        new_york_timezone=str(payload.get("new_york_timezone", "America/New_York")),
        asia_session_reference_timezone=str(payload.get("new_york_timezone", "America/New_York")),
    )


def _coerce_policy_frame(
    policy_context: pd.DataFrame | pd.Series | Mapping[str, Any],
) -> pd.DataFrame:
    if isinstance(policy_context, pd.DataFrame):
        return _normalize_policy_frame_columns(policy_context.copy())
    if isinstance(policy_context, pd.Series):
        return _normalize_policy_frame_columns(policy_context.to_frame().T)
    if isinstance(policy_context, Mapping):
        return _normalize_policy_frame_columns(pd.DataFrame([dict(policy_context)]))
    raise TypeError("policy_context must be a pandas DataFrame, Series, or mapping.")


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _normalize_policy_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in frame.columns and "timestamp" in frame.columns:
        frame["datetime"] = frame["timestamp"]
    return frame

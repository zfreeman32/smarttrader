from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from features.fx_calendar import normalize_datetime_series

from frvp.sessions.equity import build_equity_market_day_labels

from .types import RollCalendarResult


STANDARD_PRICE_COLUMNS = ("open", "high", "low", "close")


def _find_matching_column(
    columns,
    candidates,
):
    normalized_lookup = {
        str(column).strip().lower(): column
        for column in columns
    }
    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    return None


def build_volume_roll_calendar(
    contract_bars: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    contract_col: str = "contract_id",
    volume_col: str = "volume",
    metric_col: str | None = None,
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
    market_close_timezone: str = "America/New_York",
    market_close_hour: int = 17,
    market_close_minute: int = 0,
    initial_lead_contract: str | None = None,
    spread_price_col: str = "close",
    roll_bracket_sessions: int = 2,
) -> RollCalendarResult:
    """Build a causal daily lead-contract calendar from completed-session volume.

    This implements design Section 4.2 Rule 4:
    the lead contract for session N is chosen from session N-1 volume only,
    with an explicit bootstrap for the first observed session.
    """

    metric_source_col = metric_col or volume_col
    standardized = _standardize_contract_bars(
        contract_bars,
        timestamp_col=timestamp_col,
        contract_col=contract_col,
        volume_col=volume_col,
        metric_col=metric_source_col,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=market_close_timezone,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )
    if standardized.empty:
        raise ValueError("No contract bars were provided after standardization.")

    session_metrics = (
        standardized.groupby(["session_close", "contract_id"], sort=True)
        .agg(
            metric_value=(metric_source_col, "sum"),
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            bars=("timestamp", "size"),
        )
        .reset_index()
        .sort_values(["session_close", "contract_id"], kind="stable")
        .reset_index(drop=True)
    )
    session_info = (
        standardized.groupby("session_close", sort=True)
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
        )
        .reset_index()
        .sort_values("session_close", kind="stable")
        .reset_index(drop=True)
    )
    if session_info.empty:
        raise ValueError("Could not derive any market-day sessions from the provided bars.")

    winner_by_session: dict[pd.Timestamp, str] = {}
    metric_by_session: dict[pd.Timestamp, float] = {}
    previous_winner = initial_lead_contract

    for session_close in session_info["session_close"]:
        session_frame = session_metrics.loc[session_metrics["session_close"] == session_close]
        winner_contract, winner_metric = _choose_leader(session_frame, previous_winner)
        winner_by_session[pd.Timestamp(session_close)] = winner_contract
        metric_by_session[pd.Timestamp(session_close)] = winner_metric
        previous_winner = winner_contract

    first_session = pd.Timestamp(session_info["session_close"].iloc[0])
    bootstrap_contract = initial_lead_contract or winner_by_session[first_session]
    if bootstrap_contract not in set(
        session_metrics.loc[session_metrics["session_close"] == first_session, "contract_id"]
    ):
        raise ValueError(
            f"Initial lead contract '{bootstrap_contract}' is not present in the first observed session."
        )

    lead_rows: list[dict[str, object]] = []
    ordered_sessions = [pd.Timestamp(value) for value in session_info["session_close"].tolist()]
    for position, session_close in enumerate(ordered_sessions):
        session_row = session_info.loc[session_info["session_close"] == session_close].iloc[0]
        if position == 0:
            lead_contract = bootstrap_contract
            decision_session_close = pd.NaT
            decision_contract = bootstrap_contract
            decision_metric_value = np.nan
            is_bootstrap = True
        else:
            prior_session = ordered_sessions[position - 1]
            lead_contract = winner_by_session[prior_session]
            decision_session_close = prior_session
            decision_contract = lead_contract
            decision_metric_value = metric_by_session[prior_session]
            is_bootstrap = False

        session_contracts = set(
            session_metrics.loc[session_metrics["session_close"] == session_close, "contract_id"]
        )
        if lead_contract not in session_contracts:
            raise ValueError(
                f"Lead contract '{lead_contract}' has no bars in session {session_close.isoformat()}."
            )

        lead_rows.append(
            {
                "session_close": session_close,
                "session_start": pd.Timestamp(session_row["session_start"]),
                "session_end": pd.Timestamp(session_row["session_end"]),
                "lead_contract": lead_contract,
                "decision_session_close": decision_session_close,
                "decision_contract": decision_contract,
                "decision_metric_value": decision_metric_value,
                "is_bootstrap": is_bootstrap,
            }
        )

    lead_schedule = pd.DataFrame(lead_rows)
    lead_schedule["is_roll"] = lead_schedule["lead_contract"].ne(lead_schedule["lead_contract"].shift())
    if not lead_schedule.empty:
        lead_schedule.loc[0, "is_roll"] = False
    lead_schedule["is_roll_bracket"] = _compute_roll_brackets(
        len(lead_schedule),
        lead_schedule.index[lead_schedule["is_roll"]].tolist(),
        roll_bracket_sessions,
    )

    rolls = _build_roll_events(
        lead_schedule,
        standardized,
        price_col=spread_price_col,
    )

    return RollCalendarResult(
        contract_bars=standardized,
        session_metrics=session_metrics,
        lead_schedule=lead_schedule.reset_index(drop=True),
        rolls=rolls.reset_index(drop=True),
    )


def _standardize_contract_bars(
    contract_bars: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    timestamp_col: str,
    contract_col: str,
    volume_col: str,
    metric_col: str,
    source_timezone: str,
    canonical_timezone: str,
    market_close_timezone: str,
    market_close_hour: int,
    market_close_minute: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if isinstance(contract_bars, Mapping):
        for contract_id, frame in contract_bars.items():
            prepared = frame.copy()
            prepared[contract_col] = contract_id
            frames.append(prepared)
        working = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    else:
        working = contract_bars.copy()

    if working.empty:
        columns = [
            "timestamp",
            "contract_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_close",
        ]
        if metric_col != "volume":
            columns.insert(7, metric_col)
        return pd.DataFrame(
            columns=columns
        )

    rename_map: dict[str, str] = {}
    column_sources = {
        "timestamp": _find_matching_column(
            working.columns,
            [timestamp_col, "timestamp", "datetime", "ts_event"],
        ),
        "open": _find_matching_column(working.columns, ["open"]),
        "high": _find_matching_column(working.columns, ["high"]),
        "low": _find_matching_column(working.columns, ["low"]),
        "close": _find_matching_column(working.columns, ["close", "adj close"]),
        "volume": _find_matching_column(working.columns, [volume_col, "volume", "vol"]),
        "contract_id": _find_matching_column(
            working.columns,
            [
                contract_col,
                "contract_symbol",
                "raw_symbol",
                "instrument_id",
                "contract_id",
                "contract",
                "symbol",
            ],
        ),
        metric_col: _find_matching_column(working.columns, [metric_col]),
    }
    for target, source in column_sources.items():
        if source is not None:
            rename_map[source] = target

    working = working.rename(columns=rename_map)
    if "timestamp" not in working.columns:
        if isinstance(working.index, pd.DatetimeIndex):
            working = working.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise KeyError("Could not find a timestamp column in contract bars.")
    if "contract_id" not in working.columns:
        raise KeyError("Could not find or derive a contract_id column in contract bars.")

    required_columns = {"timestamp", "contract_id", *STANDARD_PRICE_COLUMNS, "volume"}
    missing = required_columns - set(working.columns)
    if missing:
        raise KeyError(f"Missing required contract-bar columns: {sorted(missing)}")
    if metric_col not in working.columns:
        raise KeyError(f"Missing metric column '{metric_col}' for lead selection.")

    selected_columns = ["timestamp", "contract_id", "open", "high", "low", "close", "volume"]
    if metric_col != "volume":
        selected_columns.append(metric_col)

    standardized = working.loc[:, selected_columns].copy()
    standardized["timestamp"] = normalize_datetime_series(
        standardized["timestamp"],
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    for column in ("open", "high", "low", "close", "volume", metric_col):
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    standardized = standardized.dropna(
        subset=["timestamp", "contract_id", "open", "high", "low", "close", metric_col]
    )
    standardized["contract_id"] = standardized["contract_id"].astype(str)
    standardized = standardized.sort_values(["contract_id", "timestamp"], kind="stable")
    standardized = standardized.drop_duplicates(subset=["contract_id", "timestamp"], keep="last")
    standardized["session_close"] = build_equity_market_day_labels(
        standardized["timestamp"],
        source_timezone=canonical_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=market_close_timezone,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )
    return standardized.reset_index(drop=True)


def _choose_leader(
    session_frame: pd.DataFrame,
    previous_winner: str | None,
) -> tuple[str, float]:
    ranked = session_frame.sort_values(
        ["metric_value", "contract_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    if ranked.empty:
        raise ValueError("Cannot choose a lead contract from an empty session frame.")

    top_metric = float(ranked.loc[0, "metric_value"])
    top_rows = ranked.loc[ranked["metric_value"] == top_metric].copy()
    if previous_winner is not None and previous_winner in set(top_rows["contract_id"]):
        row = top_rows.loc[top_rows["contract_id"] == previous_winner].iloc[0]
        return str(row["contract_id"]), float(row["metric_value"])

    row = top_rows.sort_values("contract_id", kind="stable").iloc[0]
    return str(row["contract_id"]), float(row["metric_value"])


def _compute_roll_brackets(
    session_count: int,
    roll_positions: list[int],
    bracket_sessions: int,
) -> pd.Series:
    if session_count == 0:
        return pd.Series(dtype=bool)
    if bracket_sessions < 0:
        raise ValueError("roll_bracket_sessions must be non-negative.")
    if not roll_positions:
        return pd.Series([False] * session_count, dtype=bool)

    flags = []
    for position in range(session_count):
        distance = min(abs(position - roll_position) for roll_position in roll_positions)
        flags.append(distance <= bracket_sessions)
    return pd.Series(flags, dtype=bool)


def _build_roll_events(
    lead_schedule: pd.DataFrame,
    contract_bars: pd.DataFrame,
    *,
    price_col: str,
) -> pd.DataFrame:
    roll_rows: list[dict[str, object]] = []
    cumulative_spread = 0.0

    for position in range(1, len(lead_schedule)):
        previous_row = lead_schedule.iloc[position - 1]
        current_row = lead_schedule.iloc[position]
        previous_contract = str(previous_row["lead_contract"])
        current_contract = str(current_row["lead_contract"])
        if previous_contract == current_contract:
            continue

        effective_from = pd.Timestamp(current_row["session_start"])
        spread_time = _first_common_timestamp(
            contract_bars,
            previous_contract=previous_contract,
            current_contract=current_contract,
            effective_from=effective_from,
        )
        previous_price = _price_at(contract_bars, previous_contract, spread_time, price_col)
        current_price = _price_at(contract_bars, current_contract, spread_time, price_col)
        roll_spread = float(current_price - previous_price)
        cumulative_spread += roll_spread

        roll_rows.append(
            {
                "effective_from": effective_from,
                "spread_time": spread_time,
                "from_contract": previous_contract,
                "to_contract": current_contract,
                "from_price": previous_price,
                "to_price": current_price,
                "roll_spread": roll_spread,
                "cumulative_roll_spread": cumulative_spread,
                "decision_session_close": current_row["decision_session_close"],
            }
        )

    return pd.DataFrame(roll_rows)


def _first_common_timestamp(
    contract_bars: pd.DataFrame,
    *,
    previous_contract: str,
    current_contract: str,
    effective_from: pd.Timestamp,
) -> pd.Timestamp:
    previous_times = contract_bars.loc[
        (contract_bars["contract_id"] == previous_contract) & (contract_bars["timestamp"] >= effective_from),
        "timestamp",
    ]
    current_times = contract_bars.loc[
        (contract_bars["contract_id"] == current_contract) & (contract_bars["timestamp"] >= effective_from),
        "timestamp",
    ]
    current_time_set = set(pd.Timestamp(value) for value in current_times.tolist())
    for timestamp in previous_times.tolist():
        candidate = pd.Timestamp(timestamp)
        if candidate in current_time_set:
            return candidate
    raise ValueError(
        f"Could not find a common timestamp for roll {previous_contract}->{current_contract} from {effective_from}."
    )


def _price_at(
    contract_bars: pd.DataFrame,
    contract_id: str,
    timestamp: pd.Timestamp,
    price_col: str,
) -> float:
    row = contract_bars.loc[
        (contract_bars["contract_id"] == contract_id) & (contract_bars["timestamp"] == timestamp),
        price_col,
    ]
    if row.empty:
        raise KeyError(f"Missing {price_col} for contract {contract_id} at {timestamp}.")
    return float(row.iloc[0])

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.fx_calendar import normalize_datetime_series

from frvp.sessions.equity import build_equity_market_day_labels

from .roll_calendar import _find_matching_column, build_volume_roll_calendar
from .types import (
    AbsolutePriceLevel,
    CoordinateMismatchError,
    ProfileSlice,
    RollBoundaryError,
    RollCalendarResult,
)


PRICE_COLUMNS = ("open", "high", "low", "close")


class RawProfileBars:
    """Raw lead-contract bars for profile construction in true price coordinates."""

    def __init__(self, bars: pd.DataFrame) -> None:
        working = bars.reset_index(drop=True).copy()
        if "timestamp" not in working.columns or "contract_id" not in working.columns:
            raise KeyError("RawProfileBars requires timestamp and contract_id columns.")
        self._bars = working
        self._indexed = working.set_index("timestamp", drop=False)

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars.copy()

    def profile_slice(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> ProfileSlice:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("profile_slice end must be >= start.")

        mask = (self._bars["timestamp"] >= start_ts) & (self._bars["timestamp"] <= end_ts)
        window = self._bars.loc[mask].copy()
        if window.empty:
            raise KeyError(f"No raw lead bars found between {start_ts} and {end_ts}.")

        contracts = list(dict.fromkeys(window["contract_id"].astype(str).tolist()))
        if len(contracts) != 1:
            raise RollBoundaryError(
                "Profile slices must stay inside one contract under reset-at-roll mode. "
                f"Observed contracts: {contracts}"
            )

        return ProfileSlice(
            contract_id=contracts[0],
            start=start_ts,
            end=end_ts,
            bars=window.reset_index(drop=True),
        )

    def level_from_close(self, timestamp: pd.Timestamp) -> AbsolutePriceLevel:
        row = self.bar_at(timestamp)
        return AbsolutePriceLevel(
            price=float(row["close"]),
            contract_id=str(row["contract_id"]),
            source_time=pd.Timestamp(row["timestamp"]),
        )

    def distance_to_close(
        self,
        level: AbsolutePriceLevel,
        timestamp: pd.Timestamp,
    ) -> float:
        row = self.bar_at(timestamp)
        contract_id = str(row["contract_id"])
        if contract_id != level.contract_id:
            raise CoordinateMismatchError(
                "Absolute FRVP levels cannot cross raw contract coordinates without translation. "
                f"Level contract={level.contract_id}, price contract={contract_id}."
            )
        return float(row["close"] - level.price)

    def bar_at(self, timestamp: pd.Timestamp) -> pd.Series:
        lookup = pd.Timestamp(timestamp)
        try:
            row = self._indexed.loc[lookup]
        except KeyError as exc:
            raise KeyError(f"No raw lead bar found at {lookup}.") from exc
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row


class BackAdjustedPathBars:
    """Back-adjusted bars for translation-invariant path features only."""

    def __init__(self, bars: pd.DataFrame) -> None:
        working = bars.reset_index(drop=True).copy()
        if "timestamp" not in working.columns or "source_contract_id" not in working.columns:
            raise KeyError("BackAdjustedPathBars requires timestamp and source_contract_id columns.")
        self._bars = working

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars.copy()


@dataclass(frozen=True)
class ContinuousContractResult:
    """Phase 0 output with separated raw and adjusted coordinate views."""

    raw_profile_bars: RawProfileBars
    path_bars: BackAdjustedPathBars
    session_metrics: pd.DataFrame
    lead_schedule: pd.DataFrame
    rolls: pd.DataFrame

    def window_spans_roll(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        if self.rolls.empty:
            return False
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("window end must be >= start.")
        roll_times = self.rolls["effective_from"]
        mask = (roll_times > start_ts) & (roll_times <= end_ts)
        return bool(mask.any())

    def flag_event_windows(
        self,
        start_times,
        end_times,
    ) -> pd.Series:
        starts = pd.Series(start_times)
        ends = pd.Series(end_times)
        if len(starts) != len(ends):
            raise ValueError("start_times and end_times must have the same length.")

        flags = []
        for index in range(len(starts)):
            flags.append(self.window_spans_roll(starts.iloc[index], ends.iloc[index]))
        return pd.Series(flags, index=starts.index, dtype=bool)


def build_continuous_contract(
    contract_bars,
    *,
    roll_calendar: RollCalendarResult | None = None,
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
) -> ContinuousContractResult:
    """Build a causal continuous-contract view with raw and back-adjusted layers.

    Design references:
    - Section 4.2 Rules 1-6
    - Section 8.4 look-ahead rules for lead assignment and roll-spanning windows
    """

    calendar = roll_calendar or build_volume_roll_calendar(
        contract_bars,
        timestamp_col=timestamp_col,
        contract_col=contract_col,
        volume_col=volume_col,
        metric_col=metric_col,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=market_close_timezone,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
        initial_lead_contract=initial_lead_contract,
        spread_price_col=spread_price_col,
        roll_bracket_sessions=roll_bracket_sessions,
    )
    if calendar.contract_bars.empty:
        raise ValueError("Cannot build a continuous-contract view from empty contract bars.")

    lead_lookup = calendar.lead_schedule.loc[
        :,
        [
            "session_close",
            "lead_contract",
            "decision_session_close",
            "decision_contract",
            "decision_metric_value",
            "is_bootstrap",
            "is_roll_bracket",
        ],
    ]
    raw = calendar.contract_bars.merge(
        lead_lookup,
        on="session_close",
        how="left",
        validate="many_to_one",
    )
    raw = raw.loc[raw["contract_id"] == raw["lead_contract"]].copy()
    raw = raw.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if raw.empty:
        raise ValueError("Lead assignment removed every bar. Check the input contract data.")

    raw["roll_segment_id"] = raw["contract_id"].ne(raw["contract_id"].shift()).cumsum().astype(int) - 1
    raw["cumulative_roll_spread"] = 0.0
    if not calendar.rolls.empty:
        for roll in calendar.rolls.itertuples(index=False):
            raw.loc[raw["timestamp"] >= roll.effective_from, "cumulative_roll_spread"] += float(roll.roll_spread)

    raw_profile_frame = raw.loc[
        :,
        [
            "timestamp",
            "contract_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_close",
            "decision_session_close",
            "decision_contract",
            "decision_metric_value",
            "is_bootstrap",
            "is_roll_bracket",
            "roll_segment_id",
        ],
    ].copy()

    adjusted = raw.loc[
        :,
        [
            "timestamp",
            "contract_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_close",
            "is_roll_bracket",
            "roll_segment_id",
            "cumulative_roll_spread",
        ],
    ].copy()
    adjusted = adjusted.rename(columns={"contract_id": "source_contract_id"})
    adjusted.loc[:, list(PRICE_COLUMNS)] = adjusted.loc[:, list(PRICE_COLUMNS)].sub(
        adjusted["cumulative_roll_spread"],
        axis=0,
    )

    return ContinuousContractResult(
        raw_profile_bars=RawProfileBars(raw_profile_frame),
        path_bars=BackAdjustedPathBars(adjusted),
        session_metrics=calendar.session_metrics.copy(),
        lead_schedule=calendar.lead_schedule.copy(),
        rolls=calendar.rolls.copy(),
    )


def build_continuous_contract_from_tagged_series(
    bars: pd.DataFrame,
    *,
    timestamp_col: str = "ts_event",
    contract_col: str = "contract_symbol",
    volume_col: str = "volume",
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
    market_close_timezone: str = "America/New_York",
    market_close_hour: int = 17,
    market_close_minute: int = 0,
    roll_bracket_sessions: int = 3,
) -> ContinuousContractResult:
    """Build continuity views from a pre-tagged Databento-style continuous series.

    This path is for a lead-contract continuous file that already carries per-bar
    contract tags (for example `symbol=ES.v.0` plus `contract_symbol`).
    The raw-profile layer remains authoritative; the adjusted path layer uses the
    observed seam step at each roll boundary as a practical continuity adjustment.
    """

    standardized = _standardize_tagged_continuous_bars(
        bars,
        timestamp_col=timestamp_col,
        contract_col=contract_col,
        volume_col=volume_col,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=market_close_timezone,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )
    if standardized.empty:
        raise ValueError("Cannot build a tagged continuous-contract view from empty bars.")

    derived_roll_boundary = standardized["contract_id"].ne(standardized["contract_id"].shift())
    derived_roll_boundary = derived_roll_boundary.fillna(False)
    if not standardized.empty:
        derived_roll_boundary.iloc[0] = False
    if "input_is_roll_boundary" in standardized.columns:
        declared_roll_boundary = standardized["input_is_roll_boundary"].fillna(False).astype(bool)
        if not declared_roll_boundary.equals(derived_roll_boundary.astype(bool)):
            mismatches = standardized.loc[
                declared_roll_boundary.ne(derived_roll_boundary),
                ["timestamp", "contract_id"],
            ].head(10)
            raise ValueError(
                "Tagged continuous series has inconsistent roll-boundary annotations. "
                f"Examples: {mismatches.to_dict(orient='records')}"
            )
    standardized["is_roll_boundary"] = derived_roll_boundary.astype(bool)

    if "input_in_roll_bracket" in standardized.columns:
        standardized["is_roll_bracket"] = standardized["input_in_roll_bracket"].fillna(False).astype(bool)
    else:
        roll_session_indices = sorted(
            set(
                standardized.loc[standardized["is_roll_boundary"], "market_day_index"].astype(int).tolist()
            )
        )
        standardized["is_roll_bracket"] = _derive_roll_brackets_from_boundaries(
            standardized["market_day_index"],
            roll_session_indices,
            roll_bracket_sessions,
        )

    standardized = standardized.sort_values("timestamp", kind="stable").reset_index(drop=True)
    standardized["roll_segment_id"] = (
        standardized["contract_id"].ne(standardized["contract_id"].shift()).cumsum().astype(int) - 1
    )

    session_metrics = (
        standardized.groupby(["session_close", "contract_id"], sort=True)
        .agg(
            metric_value=("volume", "sum"),
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            bars=("timestamp", "size"),
        )
        .reset_index()
        .sort_values(["session_close", "contract_id"], kind="stable")
        .reset_index(drop=True)
    )

    lead_schedule = (
        standardized.groupby("session_close", sort=True)
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            lead_contract=("contract_id", "last"),
            is_roll=("is_roll_boundary", "any"),
            is_roll_bracket=("is_roll_bracket", "any"),
        )
        .reset_index()
        .sort_values("session_close", kind="stable")
        .reset_index(drop=True)
    )
    lead_schedule["decision_session_close"] = pd.NaT
    lead_schedule["decision_contract"] = lead_schedule["lead_contract"]
    lead_schedule["decision_metric_value"] = pd.NA
    for row in lead_schedule.itertuples(index=False):
        metric_match = session_metrics.loc[
            (session_metrics["session_close"] == row.session_close)
            & (session_metrics["contract_id"] == row.lead_contract),
            "metric_value",
        ]
        if not metric_match.empty:
            lead_schedule.loc[
                lead_schedule["session_close"] == row.session_close,
                "decision_metric_value",
            ] = float(metric_match.iloc[0])
    lead_schedule["is_bootstrap"] = False
    if not lead_schedule.empty:
        lead_schedule.loc[0, "is_bootstrap"] = True

    raw = standardized.merge(
        lead_schedule.loc[
            :,
            [
                "session_close",
                "decision_session_close",
                "decision_contract",
                "decision_metric_value",
                "is_bootstrap",
            ],
        ],
        on="session_close",
        how="left",
        validate="many_to_one",
    )
    raw["cumulative_roll_spread"] = 0.0

    roll_rows: list[dict[str, object]] = []
    cumulative_roll_spread = 0.0
    roll_indices = raw.index[raw["is_roll_boundary"]].tolist()
    for index in roll_indices:
        previous_row = raw.iloc[index - 1]
        current_row = raw.iloc[index]
        seam_step = float(current_row["open"] - previous_row["close"])
        cumulative_roll_spread += seam_step
        roll_rows.append(
            {
                "effective_from": pd.Timestamp(current_row["timestamp"]),
                "spread_time": pd.Timestamp(current_row["timestamp"]),
                "from_contract": str(previous_row["contract_id"]),
                "to_contract": str(current_row["contract_id"]),
                "from_price": float(previous_row["close"]),
                "to_price": float(current_row["open"]),
                "roll_spread": seam_step,
                "cumulative_roll_spread": cumulative_roll_spread,
                "decision_session_close": pd.NaT,
            }
        )
        raw.loc[raw["timestamp"] >= pd.Timestamp(current_row["timestamp"]), "cumulative_roll_spread"] = (
            cumulative_roll_spread
        )
    rolls = pd.DataFrame(roll_rows)

    raw_profile_frame = raw.loc[
        :,
        [
            "timestamp",
            "contract_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_close",
            "decision_session_close",
            "decision_contract",
            "decision_metric_value",
            "is_bootstrap",
            "is_roll_bracket",
            "roll_segment_id",
        ],
    ].copy()

    adjusted = raw.loc[
        :,
        [
            "timestamp",
            "contract_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_close",
            "is_roll_bracket",
            "roll_segment_id",
            "cumulative_roll_spread",
        ],
    ].copy()
    adjusted = adjusted.rename(columns={"contract_id": "source_contract_id"})
    adjusted.loc[:, list(PRICE_COLUMNS)] = adjusted.loc[:, list(PRICE_COLUMNS)].sub(
        adjusted["cumulative_roll_spread"],
        axis=0,
    )

    return ContinuousContractResult(
        raw_profile_bars=RawProfileBars(raw_profile_frame),
        path_bars=BackAdjustedPathBars(adjusted),
        session_metrics=session_metrics,
        lead_schedule=lead_schedule,
        rolls=rolls,
    )


def _standardize_tagged_continuous_bars(
    bars: pd.DataFrame,
    *,
    timestamp_col: str,
    contract_col: str,
    volume_col: str,
    source_timezone: str,
    canonical_timezone: str,
    market_close_timezone: str,
    market_close_hour: int,
    market_close_minute: int,
) -> pd.DataFrame:
    working = bars.copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "contract_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "session_close",
                "market_day_index",
            ]
        )

    rename_map: dict[str, str] = {}
    column_sources = {
        "timestamp": _find_matching_column(
            working.columns,
            [timestamp_col, "timestamp", "datetime", "ts_event"],
        ),
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
        "open": _find_matching_column(working.columns, ["open"]),
        "high": _find_matching_column(working.columns, ["high"]),
        "low": _find_matching_column(working.columns, ["low"]),
        "close": _find_matching_column(working.columns, ["close", "adj close"]),
        "volume": _find_matching_column(working.columns, [volume_col, "volume", "vol"]),
        "input_is_roll_boundary": _find_matching_column(working.columns, ["is_roll_boundary"]),
        "input_in_roll_bracket": _find_matching_column(working.columns, ["in_roll_bracket", "is_roll_bracket"]),
        "session_close": _find_matching_column(working.columns, ["market_day_close", "session_close"]),
        "market_day_index": _find_matching_column(working.columns, ["market_day_index"]),
    }
    for target, source in column_sources.items():
        if source is not None:
            rename_map[source] = target

    working = working.rename(columns=rename_map)
    required_columns = {"timestamp", "contract_id", *PRICE_COLUMNS, "volume"}
    missing = required_columns - set(working.columns)
    if missing:
        raise KeyError(f"Tagged continuous series is missing required columns: {sorted(missing)}")

    selected_columns = [
        "timestamp",
        "contract_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    optional_columns = [
        column
        for column in (
            "input_is_roll_boundary",
            "input_in_roll_bracket",
            "session_close",
            "market_day_index",
        )
        if column in working.columns
    ]
    standardized = working.loc[:, selected_columns + optional_columns].copy()
    standardized["timestamp"] = normalize_datetime_series(
        standardized["timestamp"],
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    for column in PRICE_COLUMNS + ("volume",):
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
    standardized["contract_id"] = standardized["contract_id"].astype(str)
    if "session_close" in standardized.columns:
        standardized["session_close"] = normalize_datetime_series(
            standardized["session_close"],
            source_timezone=source_timezone,
            canonical_timezone=canonical_timezone,
        )
    else:
        standardized["session_close"] = build_equity_market_day_labels(
            standardized["timestamp"],
            source_timezone=canonical_timezone,
            canonical_timezone=canonical_timezone,
            market_close_timezone=market_close_timezone,
            market_close_hour=market_close_hour,
            market_close_minute=market_close_minute,
        )
    if "market_day_index" in standardized.columns:
        standardized["market_day_index"] = pd.to_numeric(standardized["market_day_index"], errors="coerce")
    else:
        session_codes, _ = pd.factorize(standardized["session_close"], sort=True)
        standardized["market_day_index"] = session_codes.astype(int)

    standardized = standardized.dropna(
        subset=["timestamp", "contract_id", "open", "high", "low", "close", "volume", "session_close"]
    )
    standardized = standardized.sort_values("timestamp", kind="stable").reset_index(drop=True)
    standardized = standardized.drop_duplicates(subset=["timestamp"], keep="last")
    standardized["market_day_index"] = standardized["market_day_index"].astype(int)
    return standardized


def _derive_roll_brackets_from_boundaries(
    market_day_index: pd.Series,
    roll_session_indices: list[int],
    bracket_sessions: int,
) -> pd.Series:
    if bracket_sessions < 0:
        raise ValueError("roll_bracket_sessions must be non-negative.")
    if not roll_session_indices:
        return pd.Series([False] * len(market_day_index), index=market_day_index.index, dtype=bool)

    distances = [
        min(abs(int(session_index) - roll_session) for roll_session in roll_session_indices)
        for session_index in market_day_index.tolist()
    ]
    return pd.Series(distances, index=market_day_index.index).le(bracket_sessions)

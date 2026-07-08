from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import pandas as pd

from features.fx_calendar import normalize_datetime_series
from features.transforms import detect_confirmed_swings

from frvp.config.instruments import get_instrument_config
from frvp.continuity.continuous_contract import RawProfileBars
from frvp.continuity.types import ProfileSlice
from frvp.sessions.equity import build_equity_session_frame


@dataclass(frozen=True)
class AnchorWindow:
    """A causal profile window for one FRVP anchor."""

    anchor_name: str
    profile_slice: ProfileSlice
    completed_at: pd.Timestamp
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def bars(self) -> pd.DataFrame:
        return self.profile_slice.bars

    @property
    def contract_id(self) -> str:
        return self.profile_slice.contract_id

    @property
    def start(self) -> pd.Timestamp:
        return self.profile_slice.start

    @property
    def end(self) -> pd.Timestamp:
        return self.profile_slice.end


@dataclass(frozen=True)
class NakedVPOCLevel:
    """One unretested VPOC level in a single raw contract coordinate system."""

    price: float
    contract_id: str
    formed_at: pd.Timestamp
    anchor_name: str
    session_date: pd.Timestamp | None = None


class NakedVPOCTracker:
    """Backward-looking VPOC tracker with reset-at-roll behavior."""

    def __init__(self) -> None:
        self._active_levels: list[NakedVPOCLevel] = []
        self._last_contract_id: str | None = None

    def register_level(self, level: NakedVPOCLevel) -> None:
        self._active_levels.append(level)
        self._active_levels.sort(key=lambda item: (item.formed_at, item.price, item.anchor_name))

    def process_bar(self, bar: pd.Series) -> tuple[NakedVPOCLevel, ...]:
        contract_id = str(bar["contract_id"])
        if self._last_contract_id is None:
            self._last_contract_id = contract_id
        elif contract_id != self._last_contract_id:
            self._active_levels = []
            self._last_contract_id = contract_id

        low = float(bar["low"])
        high = float(bar["high"])
        touched: list[NakedVPOCLevel] = []
        survivors: list[NakedVPOCLevel] = []
        for level in self._active_levels:
            if level.contract_id != contract_id:
                continue
            if low <= level.price <= high:
                touched.append(level)
            else:
                survivors.append(level)
        self._active_levels = survivors
        return tuple(touched)

    def active_levels(self, *, contract_id: str | None = None) -> tuple[NakedVPOCLevel, ...]:
        if contract_id is None:
            return tuple(self._active_levels)
        return tuple(level for level in self._active_levels if level.contract_id == contract_id)


class FRVPAnchorEngine:
    """Resolve causal FRVP anchor windows from raw single-contract bars."""

    def __init__(
        self,
        raw_profile_bars: RawProfileBars | pd.DataFrame,
        *,
        instrument: str = "es",
        source_timezone: str = "UTC",
        canonical_timezone: str = "UTC",
        swing_window: int = 3,
        composite_sessions: int | None = None,
    ) -> None:
        self.raw_profile_bars = raw_profile_bars if isinstance(raw_profile_bars, RawProfileBars) else RawProfileBars(raw_profile_bars)
        self.instrument_config = get_instrument_config(instrument)
        self.source_timezone = source_timezone
        self.canonical_timezone = canonical_timezone
        self.swing_window = int(swing_window)
        self.composite_sessions = (
            self.instrument_config.composite_sessions if composite_sessions is None else int(composite_sessions)
        )
        if self.swing_window < 1:
            raise ValueError("swing_window must be at least 1.")
        if self.composite_sessions < 1:
            raise ValueError("composite_sessions must be at least 1.")

        base_bars = self.raw_profile_bars.bars.reset_index(drop=True)
        session_frame = build_equity_session_frame(
            base_bars["timestamp"],
            instrument=self.instrument_config,
            source_timezone=canonical_timezone,
            canonical_timezone=canonical_timezone,
        )
        self._bars = pd.concat(
            [
                base_bars,
                session_frame.drop(columns=["datetime_utc"]).reset_index(drop=True),
            ],
            axis=1,
        )
        self._rth_sessions = self._summarize_segment(
            frame=self._bars.loc[self._bars["is_rth"]].copy(),
            start_column="rth_start",
            end_column="rth_end",
        )
        self._overnight_sessions = self._summarize_segment(
            frame=self._bars.loc[self._bars["is_overnight"]].copy(),
            start_column="overnight_start",
            end_column="overnight_end",
        )
        self._swing_events = self._build_swing_events()
        self._swing_confirmed_indices = [int(event["confirmed_index"]) for event in self._swing_events]
        self._timestamps = [pd.Timestamp(value) for value in self._bars["timestamp"].tolist()]
        self._timestamp_ns = pd.DatetimeIndex(self._bars["timestamp"]).asi8.copy()

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars.copy()

    @property
    def rth_sessions(self) -> pd.DataFrame:
        return self._rth_sessions.copy()

    @property
    def overnight_sessions(self) -> pd.DataFrame:
        return self._overnight_sessions.copy()

    def prior_rth(self, timestamp) -> AnchorWindow | None:
        current_ts = self._normalize_timestamp(timestamp)
        completed = self._rth_sessions.loc[self._rth_sessions["end"] < current_ts]
        if completed.empty:
            return None
        row = completed.iloc[-1]
        return self._build_time_window(
            anchor_name="prior_rth",
            start=row["start"],
            end=row["end"],
            completed_at=row["end"],
            metadata={"session_date": row["session_date"]},
        )

    def overnight_eth(self, timestamp) -> AnchorWindow | None:
        current_ts = self._normalize_timestamp(timestamp)
        completed = self._overnight_sessions.loc[self._overnight_sessions["end"] < current_ts]
        if completed.empty:
            return None
        row = completed.iloc[-1]
        return self._build_time_window(
            anchor_name="overnight_eth",
            start=row["start"],
            end=row["end"],
            completed_at=row["end"],
            metadata={"session_date": row["session_date"]},
        )

    def initial_balance(self, timestamp) -> AnchorWindow | None:
        current_ts = self._normalize_timestamp(timestamp)
        row = build_equity_session_frame(
            [current_ts],
            instrument=self.instrument_config,
            source_timezone=self.canonical_timezone,
            canonical_timezone=self.canonical_timezone,
        ).iloc[0]
        ib_end = pd.Timestamp(row["ib_end"])
        if current_ts < ib_end:
            return None
        return self._build_time_window(
            anchor_name="initial_balance",
            start=row["ib_start"],
            end=ib_end,
            completed_at=ib_end,
            metadata={"session_date": row["session_date"]},
        )

    def swing_to_swing(self, timestamp) -> AnchorWindow | None:
        current_ts = self._normalize_timestamp(timestamp)
        current_position = bisect_left(self._timestamps, current_ts)
        if current_position < (self.swing_window * 2) + 1:
            return None

        cutoff = bisect_left(self._swing_confirmed_indices, current_position)
        if cutoff < 2:
            return None

        previous_event = self._swing_events[cutoff - 2]
        current_event = self._swing_events[cutoff - 1]
        start_index = int(previous_event["swing_index"])
        end_index = int(current_event["swing_index"])
        start_kind = str(previous_event["swing_kind"])
        start_level = float(previous_event["swing_level"])
        end_kind = str(current_event["swing_kind"])
        end_level = float(current_event["swing_level"])
        if end_index <= start_index:
            return None

        window = self._bars.iloc[start_index : end_index + 1].copy()
        if window.empty or window["contract_id"].astype(str).nunique() != 1:
            return None

        profile_slice = ProfileSlice(
            contract_id=str(window["contract_id"].iloc[0]),
            start=pd.Timestamp(window["timestamp"].iloc[0]),
            end=pd.Timestamp(window["timestamp"].iloc[-1]),
            bars=window.reset_index(drop=True),
        )
        return AnchorWindow(
            anchor_name="swing_to_swing",
            profile_slice=profile_slice,
            completed_at=current_ts,
            metadata={
                "start_swing_kind": start_kind,
                "start_swing_level": start_level,
                "end_swing_kind": end_kind,
                "end_swing_level": end_level,
            },
        )

    def rolling_composite(self, timestamp) -> AnchorWindow | None:
        current_ts = self._normalize_timestamp(timestamp)
        completed = self._rth_sessions.loc[self._rth_sessions["end"] < current_ts].copy()
        if completed.empty:
            return None

        selected_dates: list[pd.Timestamp] = []
        contract_id: str | None = None
        for row in reversed(list(completed.itertuples(index=False))):
            if row.contract_count != 1 or row.contract_id is None:
                if contract_id is not None:
                    break
                continue
            if contract_id is None:
                contract_id = str(row.contract_id)
            if str(row.contract_id) != contract_id:
                break
            selected_dates.append(pd.Timestamp(row.session_date))
            if len(selected_dates) >= self.composite_sessions:
                break

        if not selected_dates:
            return None

        window = self._bars.loc[self._bars["session_date"].isin(selected_dates)].copy()
        window = window.loc[window["is_rth"]].copy()
        if window.empty or window["contract_id"].astype(str).nunique() != 1:
            return None

        profile_slice = ProfileSlice(
            contract_id=str(window["contract_id"].iloc[0]),
            start=pd.Timestamp(window["timestamp"].iloc[0]),
            end=pd.Timestamp(window["timestamp"].iloc[-1]),
            bars=window.reset_index(drop=True),
        )
        return AnchorWindow(
            anchor_name="rolling_composite",
            profile_slice=profile_slice,
            completed_at=pd.Timestamp(completed.iloc[-1]["end"]),
            metadata={"session_dates": tuple(reversed(selected_dates))},
        )

    def resolve_all(self, timestamp) -> dict[str, AnchorWindow | None]:
        return {
            "prior_rth": self.prior_rth(timestamp),
            "overnight_eth": self.overnight_eth(timestamp),
            "initial_balance": self.initial_balance(timestamp),
            "swing_to_swing": self.swing_to_swing(timestamp),
            "rolling_composite": self.rolling_composite(timestamp),
        }

    def _summarize_segment(
        self,
        *,
        frame: pd.DataFrame,
        start_column: str,
        end_column: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=["session_date", "start", "end", "bar_count", "contract_count", "contract_id"]
            )

        summary = (
            frame.groupby("session_date", sort=True)
            .agg(
                start=(start_column, "first"),
                end=(end_column, "first"),
                bar_count=("timestamp", "size"),
                contract_count=("contract_id", "nunique"),
                contract_id=("contract_id", lambda values: str(values.iloc[0]) if values.nunique() == 1 else None),
            )
            .reset_index()
            .sort_values("session_date", kind="stable")
            .reset_index(drop=True)
        )
        return summary

    def _build_swing_events(self) -> list[dict[str, object]]:
        if len(self._bars) < (self.swing_window * 2) + 1:
            return []

        confirmed_highs, confirmed_high_levels, confirmed_lows, confirmed_low_levels = detect_confirmed_swings(
            self._bars["high"],
            self._bars["low"],
            window=self.swing_window,
        )
        events: list[dict[str, object]] = []
        for confirmed_index in confirmed_highs.index[confirmed_highs.fillna(False).astype(bool)].tolist():
            swing_index = int(confirmed_index) - self.swing_window
            if swing_index >= 0 and pd.notna(confirmed_high_levels.iloc[confirmed_index]):
                events.append(
                    {
                        "confirmed_index": int(confirmed_index),
                        "swing_index": swing_index,
                        "swing_kind": "high",
                        "swing_level": float(confirmed_high_levels.iloc[confirmed_index]),
                    }
                )
        for confirmed_index in confirmed_lows.index[confirmed_lows.fillna(False).astype(bool)].tolist():
            swing_index = int(confirmed_index) - self.swing_window
            if swing_index >= 0 and pd.notna(confirmed_low_levels.iloc[confirmed_index]):
                events.append(
                    {
                        "confirmed_index": int(confirmed_index),
                        "swing_index": swing_index,
                        "swing_kind": "low",
                        "swing_level": float(confirmed_low_levels.iloc[confirmed_index]),
                    }
                )
        events.sort(key=lambda item: (int(item["confirmed_index"]), int(item["swing_index"]), str(item["swing_kind"])))
        return events

    def _build_time_window(
        self,
        *,
        anchor_name: str,
        start,
        end,
        completed_at,
        metadata: dict[str, object] | None = None,
    ) -> AnchorWindow | None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window = self._bars.loc[(self._bars["timestamp"] >= start_ts) & (self._bars["timestamp"] < end_ts)].copy()
        if window.empty or window["contract_id"].astype(str).nunique() != 1:
            return None

        profile_slice = ProfileSlice(
            contract_id=str(window["contract_id"].iloc[0]),
            start=start_ts,
            end=end_ts,
            bars=window.reset_index(drop=True),
        )
        return AnchorWindow(
            anchor_name=anchor_name,
            profile_slice=profile_slice,
            completed_at=pd.Timestamp(completed_at),
            metadata={} if metadata is None else dict(metadata),
        )

    def _normalize_timestamp(self, timestamp) -> pd.Timestamp:
        normalized = normalize_datetime_series(
            [timestamp],
            source_timezone=self.source_timezone,
            canonical_timezone=self.canonical_timezone,
        )
        return pd.Timestamp(normalized.iloc[0])

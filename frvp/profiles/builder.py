from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from features.transforms import calculate_atr, calculate_true_range

from frvp.config.instruments import get_instrument_config
from frvp.continuity.types import ProfileSlice, RollBoundaryError


@dataclass(frozen=True)
class VolumeProfile:
    """Fixed-range volume profile built in one raw contract coordinate system."""

    contract_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    bin_width: float
    histogram: pd.Series
    total_volume: float
    poc: float
    vah: float
    val: float
    hvn_levels: tuple[float, ...]
    lvn_levels: tuple[float, ...]
    shape: str
    vol_skew: float
    vol_concentration_top_pct: float
    vol_concentration_bot_pct: float
    poc_vol_pct: float
    source_bars: pd.DataFrame


class VolumeProfileBuilder:
    """Build FRVP histograms and derived levels from raw single-contract bars."""

    def __init__(
        self,
        *,
        instrument: str = "es",
        value_area_pct: float | None = None,
        atr_period: int = 14,
        atr_bin_divisor: float = 20.0,
        hvn_lvn_std_multiplier: float = 0.5,
        shape_skew_threshold: float = 0.25,
    ) -> None:
        config = get_instrument_config(instrument)
        self.instrument = config.instrument
        self.tick_size = float(config.tick_size)
        self.value_area_pct = float(config.profile_value_area_pct if value_area_pct is None else value_area_pct)
        self.atr_period = int(atr_period)
        self.atr_bin_divisor = float(atr_bin_divisor)
        self.hvn_lvn_std_multiplier = float(hvn_lvn_std_multiplier)
        self.shape_skew_threshold = float(shape_skew_threshold)

        if not 0.0 < self.value_area_pct <= 1.0:
            raise ValueError("value_area_pct must be inside (0, 1].")
        if self.atr_period < 1:
            raise ValueError("atr_period must be at least 1.")
        if self.atr_bin_divisor <= 0:
            raise ValueError("atr_bin_divisor must be positive.")

    def build(self, window: ProfileSlice | pd.DataFrame) -> VolumeProfile:
        bars, contract_id, start_ts, end_ts = self._coerce_window(window)
        bin_width = self._resolve_bin_width(bars)
        histogram = self._build_histogram(bars, bin_width)
        if histogram.empty:
            raise ValueError("Cannot build a profile from an empty histogram.")

        total_volume = float(histogram.sum())
        poc = float(histogram.idxmax())
        vah, val = self._extract_value_area(histogram)
        hvn_levels, lvn_levels = self._extract_volume_nodes(histogram)
        vol_skew, vol_concentration_top_pct, vol_concentration_bot_pct, poc_vol_pct = self._distribution_metrics(
            histogram
        )
        shape = self._classify_shape(
            vol_skew=vol_skew,
            vol_concentration_top_pct=vol_concentration_top_pct,
            vol_concentration_bot_pct=vol_concentration_bot_pct,
        )

        return VolumeProfile(
            contract_id=contract_id,
            start=start_ts,
            end=end_ts,
            bin_width=bin_width,
            histogram=histogram,
            total_volume=total_volume,
            poc=poc,
            vah=vah,
            val=val,
            hvn_levels=hvn_levels,
            lvn_levels=lvn_levels,
            shape=shape,
            vol_skew=vol_skew,
            vol_concentration_top_pct=vol_concentration_top_pct,
            vol_concentration_bot_pct=vol_concentration_bot_pct,
            poc_vol_pct=poc_vol_pct,
            source_bars=bars.reset_index(drop=True),
        )

    def _coerce_window(
        self,
        window: ProfileSlice | pd.DataFrame,
    ) -> tuple[pd.DataFrame, str, pd.Timestamp, pd.Timestamp]:
        if isinstance(window, ProfileSlice):
            bars = window.bars.copy()
            contract_id = str(window.contract_id)
            start_ts = pd.Timestamp(window.start)
            end_ts = pd.Timestamp(window.end)
        else:
            bars = window.reset_index(drop=True).copy()
            if bars.empty:
                raise ValueError("Cannot build a profile from empty bars.")
            if "contract_id" not in bars.columns:
                raise KeyError("VolumeProfileBuilder requires a contract_id column.")
            contracts = list(dict.fromkeys(bars["contract_id"].astype(str).tolist()))
            if len(contracts) != 1:
                raise RollBoundaryError(
                    "FRVP profiles must stay inside one raw contract coordinate. "
                    f"Observed contracts: {contracts}"
                )
            contract_id = contracts[0]
            start_ts = pd.Timestamp(bars["timestamp"].min())
            end_ts = pd.Timestamp(bars["timestamp"].max())

        required = {"timestamp", "contract_id", "open", "high", "low", "close", "volume"}
        missing = required - set(bars.columns)
        if missing:
            raise KeyError(f"VolumeProfileBuilder is missing required columns: {sorted(missing)}")
        if bars["contract_id"].astype(str).nunique() != 1:
            raise RollBoundaryError("FRVP profiles must stay inside one raw contract coordinate.")
        if bars.empty:
            raise ValueError("Cannot build a profile from empty bars.")
        return bars.reset_index(drop=True), contract_id, start_ts, end_ts

    def _resolve_bin_width(self, bars: pd.DataFrame) -> float:
        atr = calculate_atr(bars, period=self.atr_period).dropna()
        if not atr.empty and float(atr.iloc[-1]) > 0.0:
            atr_component = float(atr.iloc[-1]) / self.atr_bin_divisor
        else:
            true_range = calculate_true_range(bars).dropna()
            atr_component = float(true_range.mean()) / self.atr_bin_divisor if not true_range.empty else 0.0
        return max(self.tick_size, atr_component)

    def _build_histogram(self, bars: pd.DataFrame, bin_width: float) -> pd.Series:
        lows = pd.to_numeric(bars["low"], errors="coerce")
        highs = pd.to_numeric(bars["high"], errors="coerce")
        volumes = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
        if lows.isna().all() or highs.isna().all():
            return pd.Series(dtype=float)

        price_min = float(np.nanmin(lows.to_numpy(dtype=float)))
        price_max = float(np.nanmax(highs.to_numpy(dtype=float)))
        start_edge = math.floor(price_min / bin_width) * bin_width
        end_edge = math.ceil(price_max / bin_width) * bin_width
        if math.isclose(end_edge, start_edge):
            end_edge = start_edge + bin_width

        edge_count = int(round((end_edge - start_edge) / bin_width)) + 1
        edges = start_edge + (np.arange(edge_count + 1, dtype=float) * bin_width)
        levels = edges[:-1]
        node_volume = np.zeros(len(levels), dtype=float)

        for low, high, volume in zip(lows.to_numpy(dtype=float), highs.to_numpy(dtype=float), volumes.to_numpy(dtype=float), strict=False):
            if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(volume):
                continue
            bar_low = min(float(low), float(high))
            bar_high = max(float(low), float(high))
            if math.isclose(bar_low, bar_high):
                bin_index = int(np.clip(np.searchsorted(edges, bar_low, side="right") - 1, 0, len(levels) - 1))
                node_volume[bin_index] += float(volume)
                continue

            overlaps = np.minimum(bar_high, edges[1:]) - np.maximum(bar_low, edges[:-1])
            overlaps = np.clip(overlaps, 0.0, None)
            total_overlap = float(overlaps.sum())
            if total_overlap <= 0.0:
                bin_index = int(np.clip(np.searchsorted(edges, bar_low, side="right") - 1, 0, len(levels) - 1))
                node_volume[bin_index] += float(volume)
                continue
            node_volume += overlaps / total_overlap * float(volume)

        histogram = pd.Series(node_volume, index=pd.Index(levels, dtype=float, name="price"), dtype=float)
        histogram = histogram[histogram > 0.0]
        return histogram.sort_index()

    def _extract_value_area(self, histogram: pd.Series) -> tuple[float, float]:
        ranked = histogram.sort_values(ascending=False, kind="stable")
        cumulative = ranked.cumsum()
        threshold = self.value_area_pct * float(histogram.sum())
        included = ranked.loc[cumulative <= threshold].index.tolist()
        if not included:
            included = [float(ranked.index[0])]
        elif float(ranked.loc[included].sum()) < threshold and len(included) < len(ranked):
            next_index = ranked.index[len(included)]
            included.append(float(next_index))
        values = np.asarray(included, dtype=float)
        return float(values.max()), float(values.min())

    def _extract_volume_nodes(self, histogram: pd.Series) -> tuple[tuple[float, ...], tuple[float, ...]]:
        mean = float(histogram.mean())
        std = float(histogram.std(ddof=0))
        threshold = self.hvn_lvn_std_multiplier * std
        hvn_levels = tuple(float(level) for level, volume in histogram.items() if float(volume) >= mean + threshold)
        lvn_levels = tuple(float(level) for level, volume in histogram.items() if float(volume) <= mean - threshold)
        return hvn_levels, lvn_levels

    def _distribution_metrics(self, histogram: pd.Series) -> tuple[float, float, float, float]:
        if histogram.empty:
            return 0.0, 0.0, 0.0, 0.0

        levels = histogram.index.to_numpy(dtype=float)
        weights = histogram.to_numpy(dtype=float)
        total_volume = float(weights.sum())
        if total_volume <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        price_min = float(levels.min())
        price_max = float(levels.max())
        if math.isclose(price_min, price_max):
            poc_vol_pct = float(histogram.max()) / total_volume
            return 0.0, 1.0, 1.0, poc_vol_pct

        weighted_mean = float(np.average(levels, weights=weights))
        centered = levels - weighted_mean
        weighted_var = float(np.average(centered ** 2, weights=weights))
        if weighted_var <= 0.0:
            skew = 0.0
        else:
            weighted_std = math.sqrt(weighted_var)
            skew = float(np.average((centered / weighted_std) ** 3, weights=weights))

        band = price_max - price_min
        top_cutoff = price_max - (0.25 * band)
        bottom_cutoff = price_min + (0.25 * band)
        vol_concentration_top_pct = float(histogram.loc[histogram.index >= top_cutoff].sum()) / total_volume
        vol_concentration_bot_pct = float(histogram.loc[histogram.index <= bottom_cutoff].sum()) / total_volume
        poc_vol_pct = float(histogram.max()) / total_volume
        return skew, vol_concentration_top_pct, vol_concentration_bot_pct, poc_vol_pct

    def _classify_shape(
        self,
        *,
        vol_skew: float,
        vol_concentration_top_pct: float,
        vol_concentration_bot_pct: float,
    ) -> str:
        if vol_concentration_bot_pct > 0.35 and vol_skew > 0.5:
            return "P"
        if vol_concentration_top_pct > 0.35 and vol_skew < -0.5:
            return "b"
        return "D"

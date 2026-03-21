"""
OTE (Optimal Trade Entry) Labeling Engine — v2
================================================
A fully causal, backtest-safe labeling pipeline for FX swing-entry detection.

v2 CHANGES (fixing consolidation clustering + missed large moves):
  - Structural ATR: computed on 1hr, mapped to 5m — thresholds are now
    meaningful (20-30 pips for EUR/USD instead of 1 pip).
  - Minimum price distance: consecutive swings must be separated by
    min_swing_distance_atr × structural_ATR in price space.
  - Adaptive min_bars_between_swings: scales with timeframe to prevent
    rapid-fire triggers in consolidation.
  - Fixed HTF confluence: binary search, wider windows, marks LTF bars
    near any HTF swing even without a matching LTF swing.

Integrates concepts from Dr. Marcos Lopez de Prado's
"Advances in Financial Machine Learning" (Wiley, 2018):
  1. CUSUM Filter (Ch. 2.5.2) — event-driven sampling
  2. Dynamic Thresholds (Ch. 3.3) — volatility-adaptive barriers  
  3. Triple Barrier Validation (Ch. 3.4) — outcome-based label quality
  4. Exclusion Zones / Purging (Ch. 7.1) — anti-lookahead contamination
  5. Sample Uniqueness (Ch. 4.4-4.5) — concurrency-based weighting
  6. Meta-Labeling Readiness (Ch. 3.6) — structured for secondary model

Author: Quantitative ML Pipeline — Module 1 (Labeling)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
from bisect import bisect_left
import warnings

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OTEParams:
    """All configurable parameters for the OTE labeling pipeline."""
    
    # --- ATR Configuration ---
    atr_period: int = 14
    atr_smoothing: str = "ema"       # "ema" (Wilder's) or "sma"
    
    # --- Structural ATR (NEW in v2) ---
    # Instead of using 5m ATR (~1 pip) for thresholds, we compute ATR
    # on 1hr bars and map it to 5m. This gives thresholds of 20-30 pips 
    # for EUR/USD — large enough to distinguish real structure from noise.
    structural_atr_tf: str = "1hr"   # Timeframe for structural ATR: "30m" or "1hr"
    
    # --- CUSUM Filter (Lopez de Prado Ch. 2.5.2) ---
    cusum_atr_mult: float = 0.5      # Multiplier on structural ATR for CUSUM
    cusum_use_dynamic: bool = True
    cusum_fixed_threshold: float = 0.0020
    
    # --- Zigzag / Swing Detection ---
    # Confirmation retrace required = confirm_atr_mult × structural_ATR.
    # At 0.5× on 1hr ATR this means ~10-15 pips for EUR/USD.
    confirm_atr_mult: float = 0.8    # Retrace depth for confirmation
    confirm_use_close: bool = True   # Close for confirmation (vs high/low)
    
    # --- Swing Quality Filters (v2: much stricter) ---
    min_swing_atr: float = 1.35      # Min swing size in structural ATR units
    min_swing_distance_atr: float = 0.75  # Min price distance between consecutive swings
    min_bars_between_swings: int = 18     # Min bars between swings (90m at 5m)
    
    # --- Triple Barrier Validation (Lopez de Prado Ch. 3.4) ---
    tb_enable: bool = True
    tb_profit_atr: float = 1.0       # TP at 1× structural ATR
    tb_stop_atr: float = 2.0         # SL at 2× structural ATR
    tb_max_bars: int = 120           # 10 hours at 5m (was 60 = too tight)
    
    # --- Zone-Based Labels ---
    zone_pre_bars: int = 2           # Bars before swing
    zone_post_bars: int = 0          # Bars after swing

    # --- Precise Entry Labels ---
    entry_lookback_bars: int = 3     # Candidate bars before the realized swing
    entry_max_delay_after_swing: int = 4  # Allow a few bars after the extreme if reversal starts immediately
    entry_eval_bars_after_confirm: int = 12  # Keep scoring horizon short and localized
    entry_label_bars: int = 1        # Number of bars to mark starting from best entry
    entry_price_tolerance_atr: float = 0.35  # How close entry should be to the swing price
    entry_min_followthrough_atr: float = 0.35
    entry_min_rr: float = 1.25

    # --- Trend-Scanning Cross-Validation ---
    trend_scan_windows: Tuple[int, ...] = (12, 24, 48, 96)
    trend_scan_min_abs_t: float = 2.0
    trend_scan_enforce: bool = True

    # --- Label Quality / Event Metrics ---
    min_label_quality: float = 0.35
    event_match_bars: int = 3
    
    # --- Exclusion Zones / Purging (Lopez de Prado Ch. 7.1) ---
    exclusion_pre_bars: int = 10
    
    # --- Multi-Timeframe ---
    htf_confirm_atr_mult: float = 0.6
    htf_min_swing_atr: float = 0.8
    htf_min_bars_between: int = 4    # 2 hours at 30m
    htf_confluence_window_minutes: int = 120  # +/- 2 hours (was 60)
    
    # --- Sample Uniqueness (Lopez de Prado Ch. 4.4) ---
    compute_uniqueness: bool = True
    
    # --- Warmup ---
    warmup_bars: int = 50
    
    def validate(self):
        assert 0 <= self.zone_pre_bars <= 5
        assert 5 <= self.exclusion_pre_bars <= 20
        assert self.atr_period >= 5
        assert self.confirm_atr_mult > 0
        assert self.min_swing_atr > 0
        assert self.entry_lookback_bars >= 0
        assert self.entry_max_delay_after_swing >= 0
        assert self.entry_eval_bars_after_confirm >= 1
        assert self.entry_label_bars >= 1
        assert self.entry_price_tolerance_atr > 0
        assert self.entry_min_followthrough_atr >= 0
        assert self.entry_min_rr > 0
        assert len(self.trend_scan_windows) > 0
        assert self.trend_scan_min_abs_t > 0
        assert 0 <= self.min_label_quality <= 1
        assert self.event_match_bars >= 0


# ============================================================================
# ATR COMPUTATION
# ============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14, smoothing: str = "ema") -> pd.Series:
    """Compute Average True Range with Wilder's smoothing."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    if smoothing == "ema":
        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return tr.rolling(window=period, min_periods=period).mean()


def map_htf_atr_to_ltf(atr_htf: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    """Map higher-timeframe ATR to lower-timeframe index via forward-fill (causal)."""
    combined = atr_htf.reindex(atr_htf.index.union(ltf_index)).ffill()
    return combined.reindex(ltf_index)


# ============================================================================
# CUSUM FILTER (Lopez de Prado, AFML Ch. 2.5.2)
# ============================================================================

def cusum_filter(close, threshold, dynamic=True, fixed_threshold=0.002):
    """Symmetric CUSUM filter for structural break detection."""
    t_events = []
    s_pos, s_neg = 0.0, 0.0
    log_ret = np.log(close).diff().dropna()
    
    for i in range(len(log_ret)):
        idx = log_ret.index[i]
        ret = log_ret.iloc[i]
        h = threshold.get(idx, fixed_threshold) if dynamic else fixed_threshold
        if np.isnan(h) or h <= 0:
            continue
        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)
        if s_pos >= h:
            t_events.append(idx)
            s_pos = s_neg = 0.0
        elif s_neg <= -h:
            t_events.append(idx)
            s_pos = s_neg = 0.0
    return pd.DatetimeIndex(t_events)


# ============================================================================
# SWING POINT
# ============================================================================

@dataclass
class SwingPoint:
    """A detected swing high or low with full metadata."""
    swing_type: str
    swing_time: pd.Timestamp
    swing_index: int
    swing_price: float
    confirm_time: pd.Timestamp
    confirm_index: int
    confirm_lag: int
    atr_at_swing: float
    swing_size_atr: float
    source_tf: str = "5m"
    tb_outcome: Optional[str] = None
    tb_return: Optional[float] = None
    tb_bars_held: Optional[int] = None
    entry_index: Optional[int] = None
    entry_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    entry_score: Optional[float] = None
    entry_rr: Optional[float] = None
    entry_followthrough_atr: Optional[float] = None
    trend_scan_t: Optional[float] = None
    trend_scan_window: Optional[int] = None
    trend_scan_pass: bool = False
    htf_match_30m: bool = False
    htf_match_1h: bool = False
    bars_since_prev_30m_same: Optional[int] = None
    bars_since_prev_1h_same: Optional[int] = None
    label_quality: float = 0.0
    label_tier: str = "reject"


# ============================================================================
# ZIGZAG SWING DETECTOR v2 (Structural-ATR-aware)
# ============================================================================

def detect_swings_zigzag(
    df, structural_atr, confirm_atr_mult, min_swing_atr,
    min_swing_distance_atr, min_bars_between, warmup_bars,
    confirm_use_close=True, source_tf="5m"
) -> List[SwingPoint]:
    """
    Causal zigzag using structural ATR for meaningful thresholds.
    
    v2 key changes:
    - structural_atr from 1hr (20-30 pips) instead of 5m (~1 pip)
    - Enforces minimum PRICE DISTANCE between consecutive swings
    - This prevents clustering in consolidation zones
    """
    n = len(df)
    if n < warmup_bars + 20:
        return []
    
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index
    atr_vals = structural_atr.values
    
    swings: List[SwingPoint] = []
    direction = 0
    cand_high, cand_high_idx = -np.inf, 0
    cand_low, cand_low_idx = np.inf, 0
    prev_swing_price = None
    last_swing_idx = -min_bars_between - 1
    
    # Skip warmup / NaN ATR
    start_idx = warmup_bars
    while start_idx < n and (np.isnan(atr_vals[start_idx]) or atr_vals[start_idx] <= 0):
        start_idx += 1
    if start_idx >= n - 10:
        return []
    
    # Initialize direction from wider window
    init_win = min(50, n - start_idx)
    hi_idx = start_idx + int(np.argmax(highs[start_idx:start_idx + init_win]))
    lo_idx = start_idx + int(np.argmin(lows[start_idx:start_idx + init_win]))
    
    if hi_idx < lo_idx:
        direction = -1
        cand_high, cand_high_idx = highs[hi_idx], hi_idx
        cand_low, cand_low_idx = lows[lo_idx], lo_idx
        prev_swing_price = cand_high
    else:
        direction = 1
        cand_low, cand_low_idx = lows[lo_idx], lo_idx
        cand_high, cand_high_idx = highs[hi_idx], hi_idx
        prev_swing_price = cand_low
    
    scan_start = max(hi_idx, lo_idx) + 1
    
    for i in range(scan_start, n):
        a = atr_vals[i]
        if np.isnan(a) or a <= 0:
            continue
        confirm_thresh = confirm_atr_mult * a
        
        if direction == 1:
            if highs[i] > cand_high:
                cand_high, cand_high_idx = highs[i], i
            
            retrace = cand_high - (closes[i] if confirm_use_close else lows[i])
            if retrace >= confirm_thresh and i > cand_high_idx:
                # Quality filters
                if cand_high_idx - last_swing_idx < min_bars_between:
                    direction = -1
                    cand_low, cand_low_idx = lows[i], i
                    continue
                if prev_swing_price is not None:
                    if abs(cand_high - prev_swing_price) < min_swing_distance_atr * a:
                        direction = -1
                        cand_low, cand_low_idx = lows[i], i
                        continue
                
                swing_size = abs(cand_high - prev_swing_price) if prev_swing_price is not None else np.inf
                atr_at = atr_vals[cand_high_idx] if not np.isnan(atr_vals[cand_high_idx]) else a
                size_atr = swing_size / atr_at if atr_at > 0 else 0
                
                if size_atr >= min_swing_atr:
                    swings.append(SwingPoint(
                        "high", timestamps[cand_high_idx], cand_high_idx, cand_high,
                        timestamps[i], i, i - cand_high_idx, atr_at, size_atr, source_tf
                    ))
                    prev_swing_price = cand_high
                    last_swing_idx = cand_high_idx
                
                direction = -1
                cand_low, cand_low_idx = lows[i], i
        
        elif direction == -1:
            if lows[i] < cand_low:
                cand_low, cand_low_idx = lows[i], i
            
            rally = (closes[i] if confirm_use_close else highs[i]) - cand_low
            if rally >= confirm_thresh and i > cand_low_idx:
                if cand_low_idx - last_swing_idx < min_bars_between:
                    direction = 1
                    cand_high, cand_high_idx = highs[i], i
                    continue
                if prev_swing_price is not None:
                    if abs(prev_swing_price - cand_low) < min_swing_distance_atr * a:
                        direction = 1
                        cand_high, cand_high_idx = highs[i], i
                        continue
                
                swing_size = abs(prev_swing_price - cand_low) if prev_swing_price is not None else np.inf
                atr_at = atr_vals[cand_low_idx] if not np.isnan(atr_vals[cand_low_idx]) else a
                size_atr = swing_size / atr_at if atr_at > 0 else 0
                
                if size_atr >= min_swing_atr:
                    swings.append(SwingPoint(
                        "low", timestamps[cand_low_idx], cand_low_idx, cand_low,
                        timestamps[i], i, i - cand_low_idx, atr_at, size_atr, source_tf
                    ))
                    prev_swing_price = cand_low
                    last_swing_idx = cand_low_idx
                
                direction = 1
                cand_high, cand_high_idx = highs[i], i
    
    return swings


# ============================================================================
# TRIPLE BARRIER VALIDATION (Lopez de Prado, AFML Ch. 3.4)
# ============================================================================

def validate_swings_triple_barrier(df, swings, structural_atr, params):
    """Validate swings via Triple Barrier Method using structural ATR."""
    if not params.tb_enable:
        return swings
    
    n = len(df)
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    atr_vals = structural_atr.values
    validated = []
    
    for sp in swings:
        idx = sp.confirm_index
        if idx >= n - 1:
            continue
        entry = closes[idx]
        a = atr_vals[idx]
        if np.isnan(a) or a <= 0:
            a = sp.atr_at_swing
        if a <= 0:
            sp.tb_outcome = "skip"
            validated.append(sp)
            continue
        
        if sp.swing_type == "low":
            tp, sl = entry + params.tb_profit_atr * a, entry - params.tb_stop_atr * a
        else:
            tp, sl = entry - params.tb_profit_atr * a, entry + params.tb_stop_atr * a
        
        mx = min(idx + params.tb_max_bars, n)
        outcome, held = "timeout", params.tb_max_bars
        final = closes[min(mx - 1, n - 1)]
        
        for j in range(idx + 1, mx):
            if sp.swing_type == "low":
                if highs[j] >= tp:
                    outcome, held, final = "tp", j - idx, tp; break
                if lows[j] <= sl:
                    outcome, held, final = "sl", j - idx, sl; break
            else:
                if lows[j] <= tp:
                    outcome, held, final = "tp", j - idx, tp; break
                if highs[j] >= sl:
                    outcome, held, final = "sl", j - idx, sl; break
        
        sp.tb_outcome = outcome
        sp.tb_return = (final - entry) / entry * (-1 if sp.swing_type == "high" else 1)
        sp.tb_bars_held = held
        validated.append(sp)
    return validated


def _linear_trend_tstat(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.log(np.maximum(values.astype(np.float64), 1e-12))
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx <= 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    resid = y - (y_mean + slope * (x - x_mean))
    dof = len(values) - 2
    if dof <= 0:
        return 0.0
    sigma2 = np.sum(resid ** 2) / dof
    se = np.sqrt(max(sigma2 / sxx, 1e-12))
    return float(slope / se)


def trend_scan_swings(df, swings, params):
    """Cross-validate swing labels with forward trend-scanning t-stats."""
    closes = df["close"].values
    n = len(df)
    for sp in swings:
        best_t = None
        best_window = None
        for window in params.trend_scan_windows:
            end = sp.swing_index + window
            if end >= n:
                continue
            t_stat = _linear_trend_tstat(closes[sp.swing_index:end + 1])
            signed_t = t_stat if sp.swing_type == "low" else -t_stat
            if best_t is None or signed_t > best_t:
                best_t = signed_t
                best_window = window
        if best_t is None:
            sp.trend_scan_t = 0.0
            sp.trend_scan_window = None
            sp.trend_scan_pass = False
        else:
            sp.trend_scan_t = best_t
            sp.trend_scan_window = best_window
            sp.trend_scan_pass = best_t >= params.trend_scan_min_abs_t
    return swings


def annotate_swing_context(swings, htf_swings_30m=None, htf_swings_1h=None, htf_match_30m=None, htf_match_1h=None):
    """Attach HTF agreement and same-direction spacing metadata to swings."""
    htf_30m_ns = {
        "low": sorted(s.swing_time.value for s in (htf_swings_30m or []) if s.swing_type == "low"),
        "high": sorted(s.swing_time.value for s in (htf_swings_30m or []) if s.swing_type == "high"),
    }
    htf_1h_ns = {
        "low": sorted(s.swing_time.value for s in (htf_swings_1h or []) if s.swing_type == "low"),
        "high": sorted(s.swing_time.value for s in (htf_swings_1h or []) if s.swing_type == "high"),
    }
    bar_ns = int(pd.Timedelta(minutes=5).value)
    for sp in sorted(swings, key=lambda item: item.swing_index):
        if htf_match_30m:
            sp.htf_match_30m = bool(htf_match_30m.get(sp.swing_index, False))
        if htf_match_1h:
            sp.htf_match_1h = bool(htf_match_1h.get(sp.swing_index, False))
        arr30 = htf_30m_ns.get(sp.swing_type, [])
        arr1h = htf_1h_ns.get(sp.swing_type, [])
        pos30 = bisect_left(arr30, sp.swing_time.value) - 1
        pos1h = bisect_left(arr1h, sp.swing_time.value) - 1
        if pos30 >= 0:
            sp.bars_since_prev_30m_same = int((sp.swing_time.value - arr30[pos30]) / bar_ns)
        if pos1h >= 0:
            sp.bars_since_prev_1h_same = int((sp.swing_time.value - arr1h[pos1h]) / bar_ns)
    return swings


def compute_swing_quality(swings, params):
    """Score each swing so labels can be filtered or down-weighted by quality."""
    for sp in swings:
        tb_score = {
            "tp": 1.0,
            "timeout": 0.45,
            "skip": 0.35,
            "sl": 0.0,
            None: 0.35,
        }.get(sp.tb_outcome, 0.35)
        trend_score = 0.0 if sp.trend_scan_t is None else min(max(sp.trend_scan_t, 0.0) / 4.0, 1.0)
        htf_score = 0.5 * float(sp.htf_match_30m) + 0.5 * float(sp.htf_match_1h)
        entry_score = 0.0 if sp.entry_score is None else min(max(sp.entry_score, 0.0) / 5.0, 1.0)
        quality = 0.35 * tb_score + 0.30 * trend_score + 0.20 * htf_score + 0.15 * entry_score
        sp.label_quality = float(min(max(quality, 0.0), 1.0))
        if sp.label_quality >= 0.80:
            sp.label_tier = "A"
        elif sp.label_quality >= 0.65:
            sp.label_tier = "B"
        elif sp.label_quality >= params.min_label_quality:
            sp.label_tier = "C"
        else:
            sp.label_tier = "reject"
    return swings


def is_positive_swing(sp, params):
    if params.tb_enable and sp.tb_outcome == "sl":
        return False
    if params.trend_scan_enforce and not sp.trend_scan_pass:
        return False
    return sp.label_quality == 0.0 or sp.label_quality >= params.min_label_quality


# ============================================================================
# ZONE LABELS, EXCLUSION, UNIQUENESS
# ============================================================================

def create_zone_labels(n, swings, params):
    """Binary labels around confirmed swing points."""
    ll, ls = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    for sp in swings:
        if not is_positive_swing(sp, params):
            continue
        s = max(0, sp.swing_index - params.zone_pre_bars)
        e = min(n, sp.swing_index + params.zone_post_bars + 1)
        if sp.swing_type == "low":
            ll[s:e] = 1
        else:
            ls[s:e] = 1
    return ll, ls


def _safe_bar_range(opens, highs, lows, closes, idx):
    bar_range = highs[idx] - lows[idx]
    if bar_range > 0:
        return bar_range
    return max(abs(closes[idx] - opens[idx]), 1e-8)


def assign_entry_labels(df, swings, structural_atr, params):
    """
    Pick a single "best executable" bar for each successful swing.

    Labels are still allowed to use future information because they are training
    targets. The goal is to mark the bar that offered the best localized
    swing-entry quality before the move became fully confirmed.
    """
    n = len(df)
    ll, ls = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    if not swings:
        return ll, ls

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index
    atr_vals = structural_atr.values

    for sp in swings:
        if not is_positive_swing(sp, params):
            continue

        window_start = max(0, sp.swing_index - params.entry_lookback_bars)
        window_end = min(
            n - 2,
            sp.confirm_index - 1,
            sp.swing_index + params.entry_max_delay_after_swing,
        )
        if window_end < window_start:
            continue

        eval_end = min(n - 1, sp.confirm_index + params.entry_eval_bars_after_confirm)
        best_idx = None
        best_score = -np.inf
        best_rr = None
        best_followthrough_atr = None

        for idx in range(window_start, window_end + 1):
            atr_at = atr_vals[idx]
            if np.isnan(atr_at) or atr_at <= 0:
                atr_at = sp.atr_at_swing
            if atr_at <= 0 or np.isnan(atr_at):
                continue

            entry = closes[idx]
            future_high = np.max(highs[idx + 1:eval_end + 1])
            future_low = np.min(lows[idx + 1:eval_end + 1])
            bar_range = _safe_bar_range(opens, highs, lows, closes, idx)

            if sp.swing_type == "low":
                favorable = future_high - entry
                adverse = max(0.0, entry - future_low)
                wick_ratio = (min(opens[idx], closes[idx]) - lows[idx]) / bar_range
                close_location = (closes[idx] - lows[idx]) / bar_range
                proximity = 1.0 - min(abs(lows[idx] - sp.swing_price) / (params.entry_price_tolerance_atr * atr_at), 1.0)
                net_progress = closes[sp.confirm_index] - entry
            else:
                favorable = entry - future_low
                adverse = max(0.0, future_high - entry)
                wick_ratio = (highs[idx] - max(opens[idx], closes[idx])) / bar_range
                close_location = (highs[idx] - closes[idx]) / bar_range
                proximity = 1.0 - min(abs(highs[idx] - sp.swing_price) / (params.entry_price_tolerance_atr * atr_at), 1.0)
                net_progress = entry - closes[sp.confirm_index]

            rr = favorable / max(adverse, 0.10 * atr_at)
            followthrough_atr = favorable / atr_at
            timing_bonus = 1.0 - (abs(idx - sp.swing_index) / max(params.entry_lookback_bars + params.entry_max_delay_after_swing, 1))

            score = (
                0.35 * rr
                + 0.20 * followthrough_atr
                + 0.15 * wick_ratio
                + 0.10 * close_location
                + 0.10 * proximity
                + 0.10 * timing_bonus
            )

            if (
                followthrough_atr >= params.entry_min_followthrough_atr
                and rr >= params.entry_min_rr
                and net_progress > 0
                and score > best_score
            ):
                best_idx = idx
                best_score = score
                best_rr = rr
                best_followthrough_atr = followthrough_atr

        if best_idx is None:
            continue

        sp.entry_index = best_idx
        sp.entry_time = timestamps[best_idx]
        sp.entry_price = closes[best_idx]
        sp.entry_score = best_score
        sp.entry_rr = best_rr
        sp.entry_followthrough_atr = best_followthrough_atr

        label_end = min(n, best_idx + params.entry_label_bars)
        if sp.swing_type == "low":
            ll[best_idx:label_end] = 1
        else:
            ls[best_idx:label_end] = 1

    return ll, ls


def create_quality_arrays(n, swings, params):
    long_quality = np.zeros(n, dtype=np.float32)
    short_quality = np.zeros(n, dtype=np.float32)
    entry_long_quality = np.zeros(n, dtype=np.float32)
    entry_short_quality = np.zeros(n, dtype=np.float32)
    for sp in swings:
        if not is_positive_swing(sp, params):
            continue
        s = max(0, sp.swing_index - params.zone_pre_bars)
        e = min(n, sp.swing_index + params.zone_post_bars + 1)
        if sp.swing_type == "low":
            long_quality[s:e] = np.maximum(long_quality[s:e], sp.label_quality)
            if sp.entry_index is not None:
                entry_long_quality[sp.entry_index:min(n, sp.entry_index + params.entry_label_bars)] = np.maximum(
                    entry_long_quality[sp.entry_index:min(n, sp.entry_index + params.entry_label_bars)],
                    sp.label_quality,
                )
        else:
            short_quality[s:e] = np.maximum(short_quality[s:e], sp.label_quality)
            if sp.entry_index is not None:
                entry_short_quality[sp.entry_index:min(n, sp.entry_index + params.entry_label_bars)] = np.maximum(
                    entry_short_quality[sp.entry_index:min(n, sp.entry_index + params.entry_label_bars)],
                    sp.label_quality,
                )
    return long_quality, short_quality, entry_long_quality, entry_short_quality


def _bars_since_last_event(n, event_indices):
    out = np.full(n, -1, dtype=np.int32)
    if not event_indices:
        return out
    event_indices = np.asarray(sorted(set(int(i) for i in event_indices)), dtype=np.int32)
    ptr = 0
    last = -1
    for i in range(n):
        while ptr < len(event_indices) and event_indices[ptr] <= i:
            last = int(event_indices[ptr])
            ptr += 1
        if last >= 0:
            out[i] = i - last
    return out


def compute_htf_timing_features(n_ltf, ltf_index, swings, prefix):
    ts_to_idx = pd.Series(np.arange(n_ltf, dtype=np.int32), index=ltf_index)
    low_idx = []
    high_idx = []
    for sp in swings:
        idx = ts_to_idx.get(sp.swing_time)
        if pd.isna(idx):
            continue
        if sp.swing_type == "low":
            low_idx.append(int(idx))
        else:
            high_idx.append(int(idx))
    return {
        f"bars_since_{prefix}_swing_low": _bars_since_last_event(n_ltf, low_idx),
        f"bars_since_{prefix}_swing_high": _bars_since_last_event(n_ltf, high_idx),
    }


def _event_indices_from_labels(labels):
    labels = np.asarray(labels).astype(bool)
    starts = np.flatnonzero(labels & ~np.roll(labels, 1))
    if len(starts) and labels[0]:
        starts[0] = 0
    return starts.tolist()


def compute_event_metrics(swings, event_indices, swing_type, tolerance):
    eligible = [s for s in swings if s.swing_type == swing_type and s.entry_index is not None and s.label_tier != "reject"]
    if not eligible:
        return {
            "event_precision": 0.0,
            "event_recall": 0.0,
            "temporal_offset_mean": 0.0,
            "temporal_offset_med": 0.0,
            "num_events": len(event_indices),
            "num_swings": 0,
        }

    matched_swings = set()
    offsets = []
    matched_events = 0
    swing_indices = [s.entry_index for s in eligible]
    for event_idx in event_indices:
        best_j = None
        best_abs = None
        for j, swing_idx in enumerate(swing_indices):
            delta = event_idx - swing_idx
            abs_delta = abs(delta)
            if abs_delta <= tolerance and (best_abs is None or abs_delta < best_abs):
                best_j = j
                best_abs = abs_delta
        if best_j is not None and best_j not in matched_swings:
            matched_swings.add(best_j)
            matched_events += 1
            offsets.append(event_idx - swing_indices[best_j])

    return {
        "event_precision": matched_events / max(len(event_indices), 1),
        "event_recall": matched_events / len(eligible),
        "temporal_offset_mean": float(np.mean(np.abs(offsets))) if offsets else 0.0,
        "temporal_offset_med": float(np.median(np.abs(offsets))) if offsets else 0.0,
        "num_events": len(event_indices),
        "num_swings": len(eligible),
    }


def create_exclusion_masks(n, swings, label_long, label_short, params):
    """Purging zones around confirmations (Lopez de Prado Ch. 7.1)."""
    el, es = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for sp in swings:
        s = max(0, sp.confirm_index - params.exclusion_pre_bars)
        e = sp.confirm_index + 1
        if sp.swing_type == "low":
            el[s:e] = True
        else:
            es[s:e] = True
    el |= label_long.astype(bool)
    es |= label_short.astype(bool)
    return el, es, (~label_long.astype(bool)) & (~el), (~label_short.astype(bool)) & (~es)


def compute_label_concurrency(n, swings, params):
    """Count overlapping label windows at each bar."""
    cl, cs = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    for sp in swings:
        s = max(0, sp.swing_index - params.zone_pre_bars)
        e = min(n, sp.confirm_index + 1)
        if sp.swing_type == "low":
            cl[s:e] += 1
        else:
            cs[s:e] += 1
    return cl, cs


def compute_sample_weights(conc, labels):
    """Inverse-concurrency weights (AFML Ch. 4.4)."""
    w = np.ones(len(labels), dtype=np.float64)
    pos = labels.astype(bool)
    w[pos] = 1.0 / np.maximum(conc[pos], 1)
    return w


# ============================================================================
# HTF CONFLUENCE (v2: binary search, marks LTF bars near HTF swings)
# ============================================================================

def compute_htf_confluence(ltf_swings, htf_swings, n_ltf, ltf_timestamps, window_minutes=120):
    """
    v2: For each HTF swing, mark all LTF bars within +/- window.
    This ensures HTF structure is visible even without a matching LTF swing.
    """
    cl, cs = np.zeros(n_ltf, dtype=np.int8), np.zeros(n_ltf, dtype=np.int8)
    if not htf_swings:
        return cl, cs
    
    ltf_ns = ltf_timestamps.values.astype(np.int64)
    window_ns = int(pd.Timedelta(minutes=window_minutes).value)
    
    for sp in htf_swings:
        t_ns = sp.swing_time.value
        i_s = int(np.searchsorted(ltf_ns, t_ns - window_ns, side='left'))
        i_e = int(np.searchsorted(ltf_ns, t_ns + window_ns, side='right'))
        i_s, i_e = max(0, i_s), min(n_ltf, i_e)
        if sp.swing_type == "low":
            cl[i_s:i_e] = 1
        else:
            cs[i_s:i_e] = 1
    return cl, cs


def compute_htf_swing_match(ltf_swings, htf_swings, window_minutes=120):
    """Per LTF swing: does it have HTF confirmation? Returns dict {swing_index: bool}."""
    result = {}
    if not htf_swings:
        return result
    htf_low_ns = sorted([s.swing_time.value for s in htf_swings if s.swing_type == "low"])
    htf_high_ns = sorted([s.swing_time.value for s in htf_swings if s.swing_type == "high"])
    w = int(pd.Timedelta(minutes=window_minutes).value)
    
    for sp in ltf_swings:
        t = sp.swing_time.value
        arr = htf_low_ns if sp.swing_type == "low" else htf_high_ns
        idx = bisect_left(arr, t)
        found = False
        for j in [idx - 1, idx]:
            if 0 <= j < len(arr) and abs(arr[j] - t) <= w:
                found = True; break
        result[sp.swing_index] = found
    return result


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics(df, swings, ll, ls, el, es, nl, ns, satr, params,
                    hcl=None, hcs=None, htf_match=None, entry_long=None, entry_short=None,
                    anomaly_count=0):
    n = len(df)
    d = {"total_bars": n}
    d["positive_rate_long_pct"] = 100.0 * ll.sum() / n if n else 0
    d["positive_rate_short_pct"] = 100.0 * ls.sum() / n if n else 0
    
    longs = [s for s in swings if s.swing_type == "low"]
    shorts = [s for s in swings if s.swing_type == "high"]
    d["num_swings_long"] = len(longs)
    d["num_swings_short"] = len(shorts)
    d["num_entry_long"] = int(entry_long.sum()) if entry_long is not None else 0
    d["num_entry_short"] = int(entry_short.sum()) if entry_short is not None else 0
    d["anomaly_count"] = int(anomaly_count)
    
    for lab, grp in [("long", longs), ("short", shorts)]:
        if grp:
            lags = [s.confirm_lag for s in grp]
            d[f"avg_lag_{lab}"] = np.mean(lags)
            d[f"med_lag_{lab}"] = np.median(lags)
            d[f"p90_lag_{lab}"] = np.percentile(lags, 90)
        else:
            d[f"avg_lag_{lab}"] = d[f"med_lag_{lab}"] = d[f"p90_lag_{lab}"] = 0
    
    if n > 0 and isinstance(df.index, pd.DatetimeIndex):
        days = max((df.index[-1] - df.index[0]).total_seconds() / 86400, 1)
        d["swings_per_day"] = len(swings) / days
    else:
        d["swings_per_day"] = 0
    
    sizes = [s.swing_size_atr for s in swings if s.swing_size_atr > 0 and np.isfinite(s.swing_size_atr)]
    if sizes:
        d["size_mean"] = np.mean(sizes); d["size_med"] = np.median(sizes)
        d["size_p10"] = np.percentile(sizes, 10); d["size_p90"] = np.percentile(sizes, 90)
    else:
        d["size_mean"] = d["size_med"] = d["size_p10"] = d["size_p90"] = 0
    
    if params.tb_enable:
        oc = [s.tb_outcome for s in swings if s.tb_outcome]
        d["tb_tp"] = sum(1 for o in oc if o == "tp")
        d["tb_sl"] = sum(1 for o in oc if o == "sl")
        d["tb_to"] = sum(1 for o in oc if o == "timeout")
        d["tb_wr"] = 100.0 * d["tb_tp"] / len(oc) if oc else 0
        rets = [s.tb_return for s in swings if s.tb_return is not None]
        if rets:
            d["tb_ret_avg"] = np.mean(rets); d["tb_ret_med"] = np.median(rets)

    entry_scores = [s.entry_score for s in swings if s.entry_score is not None]
    entry_rr = [s.entry_rr for s in swings if s.entry_rr is not None]
    label_quality = [s.label_quality for s in swings if s.label_quality > 0]
    trend_t = [s.trend_scan_t for s in swings if s.trend_scan_t is not None]
    if entry_scores:
        d["entry_score_mean"] = float(np.mean(entry_scores))
    if entry_rr:
        d["entry_rr_mean"] = float(np.mean(entry_rr))
    if label_quality:
        d["label_quality_mean"] = float(np.mean(label_quality))
        d["label_quality_med"] = float(np.median(label_quality))
    if trend_t:
        d["trend_scan_mean"] = float(np.mean(trend_t))
        d["trend_scan_med"] = float(np.median(trend_t))
        d["trend_scan_pass_rate"] = 100.0 * sum(1 for s in swings if s.trend_scan_pass) / max(len(swings), 1)
    
    d["excl_l"] = int(el.sum()); d["excl_s"] = int(es.sum())
    d["neg_l"] = int(nl.sum()); d["neg_s"] = int(ns.sum())
    
    if hcl is not None:
        d["htf_bars_l"] = int(hcl.sum()); d["htf_bars_s"] = int(hcs.sum()) if hcs is not None else 0
    if htf_match:
        m = sum(1 for v in htf_match.values() if v)
        d["htf_match"] = m; d["htf_nomatch"] = len(htf_match) - m
        d["htf_match_pct"] = 100.0 * m / len(htf_match) if htf_match else 0
    
    va = satr.dropna()
    if len(va):
        d["satr_mean"] = va.mean(); d["satr_pips"] = va.mean() * 10000

    if entry_long is not None and entry_short is not None:
        ev_long = compute_event_metrics(swings, _event_indices_from_labels(entry_long), "low", params.event_match_bars)
        ev_short = compute_event_metrics(swings, _event_indices_from_labels(entry_short), "high", params.event_match_bars)
        for prefix, values in [("long", ev_long), ("short", ev_short)]:
            d[f"event_precision_{prefix}"] = values["event_precision"]
            d[f"event_recall_{prefix}"] = values["event_recall"]
            d[f"event_temporal_mean_{prefix}"] = values["temporal_offset_mean"]
            d[f"event_temporal_med_{prefix}"] = values["temporal_offset_med"]
            d[f"event_count_{prefix}"] = values["num_events"]
    
    chk = []
    chk.append(("no_pos_neg_overlap_long", int((ll.astype(bool) & nl).sum()) == 0, int((ll.astype(bool) & nl).sum())))
    chk.append(("no_pos_neg_overlap_short", int((ls.astype(bool) & ns).sum()) == 0, int((ls.astype(bool) & ns).sum())))
    chk.append(("pos_implies_excluded_long", int((ll.astype(bool) & ~el).sum()) == 0, int((ll.astype(bool) & ~el).sum())))
    chk.append(("pos_implies_excluded_short", int((ls.astype(bool) & ~es).sum()) == 0, int((ls.astype(bool) & ~es).sum())))
    chk.append(("no_nan_labels", not np.any(np.isnan(ll[params.warmup_bars:].astype(float))), 0))
    
    for dir, rate in [("long", d["positive_rate_long_pct"]), ("short", d["positive_rate_short_pct"])]:
        if rate > 5:
            chk.append((f"pos_rate_{dir}_not_high", False, round(rate, 3)))
        elif rate < 0.1 and swings:
            chk.append((f"pos_rate_{dir}_not_low", False, round(rate, 3)))
        else:
            chk.append((f"pos_rate_{dir}_ok", True, round(rate, 3)))
    
    chk.append(("no_zero_lag", sum(1 for s in swings if s.confirm_lag == 0) == 0,
                 sum(1 for s in swings if s.confirm_lag == 0)))
    
    if len(swings) > 5:
        ss = sorted(swings, key=lambda s: s.swing_index)
        gaps = [ss[i].swing_index - ss[i-1].swing_index for i in range(1, len(ss))]
        cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 99
        d["inter_swing_cv"] = cv
        chk.append(("distribution_ok", cv < 3.0, round(cv, 2)))
    
    d["sanity_checks"] = chk
    return d


def print_diagnostics(d):
    print("\n" + "=" * 70)
    print("OTE LABELING DIAGNOSTICS (v2 — Structural ATR)")
    print("=" * 70)
    
    if "satr_mean" in d:
        print(f"\n  Structural ATR mean: {d['satr_mean']:.6f}  ({d['satr_pips']:.1f} pips)")
    
    print(f"\n  Total bars:            {d['total_bars']:,}")
    print(f"  Pos rate long:         {d['positive_rate_long_pct']:.3f}%")
    print(f"  Pos rate short:        {d['positive_rate_short_pct']:.3f}%")
    print(f"  Swings (long/short):   {d['num_swings_long']} / {d['num_swings_short']}")
    print(f"  Swings per day:        {d['swings_per_day']:.2f}")
    
    print(f"\n  Confirm lag long:      mean={d['avg_lag_long']:.1f}  med={d['med_lag_long']:.1f}  p90={d['p90_lag_long']:.1f}")
    print(f"  Confirm lag short:     mean={d['avg_lag_short']:.1f}  med={d['med_lag_short']:.1f}  p90={d['p90_lag_short']:.1f}")
    
    print(f"\n  Swing size (ATR):      mean={d['size_mean']:.2f}  med={d['size_med']:.2f}  "
          f"p10={d['size_p10']:.2f}  p90={d['size_p90']:.2f}")
    
    if "tb_tp" in d:
        print(f"\n  Triple Barrier:        TP={d['tb_tp']}  SL={d['tb_sl']}  TO={d['tb_to']}  "
              f"WR={d['tb_wr']:.1f}%")
        if "tb_ret_avg" in d:
            print(f"  Returns:               avg={d['tb_ret_avg']:.5f}  med={d['tb_ret_med']:.5f}")
    
    print(f"\n  Excluded long/short:   {d['excl_l']:,} / {d['excl_s']:,}")
    print(f"  Trainable neg l/s:     {d['neg_l']:,} / {d['neg_s']:,}")
    print(f"  Entry bars long/short: {d['num_entry_long']:,} / {d['num_entry_short']:,}")
    print(f"  Anomaly bars removed:  {d['anomaly_count']:,}")

    if "entry_score_mean" in d:
        print(f"  Entry score mean:      {d['entry_score_mean']:.2f}")
    if "entry_rr_mean" in d:
        print(f"  Entry RR mean:         {d['entry_rr_mean']:.2f}")
    if "label_quality_mean" in d:
        print(f"  Label quality mean:    {d['label_quality_mean']:.2f}  med={d['label_quality_med']:.2f}")
    if "trend_scan_mean" in d:
        print(f"  Trend-scan t:          mean={d['trend_scan_mean']:.2f}  med={d['trend_scan_med']:.2f}  "
              f"pass={d['trend_scan_pass_rate']:.1f}%")
    
    if "htf_bars_l" in d:
        print(f"\n  HTF conf bars l/s:     {d['htf_bars_l']:,} / {d['htf_bars_s']:,}")
    if "htf_match" in d:
        print(f"  HTF matched swings:    {d['htf_match']} / {d['htf_match'] + d['htf_nomatch']}  "
              f"({d['htf_match_pct']:.1f}%)")
    if "inter_swing_cv" in d:
        print(f"\n  Inter-swing CV:        {d['inter_swing_cv']:.2f}  (< 3.0 = well-distributed)")
    if "event_precision_long" in d:
        print(f"\n  Event precision L/S:   {d['event_precision_long']:.2%} / {d['event_precision_short']:.2%}")
        print(f"  Event recall L/S:      {d['event_recall_long']:.2%} / {d['event_recall_short']:.2%}")
        print(f"  Timing offset L/S:     {d['event_temporal_mean_long']:.2f} / {d['event_temporal_mean_short']:.2f} bars")
    
    print(f"\n  {'SANITY CHECKS':=^50}")
    ok = True
    for name, passed, val in d["sanity_checks"]:
        s = "PASS" if passed else "FAIL"
        if not passed: ok = False
        print(f"  [{s}] {name}  ({val})")
    print(f"\n  {'ALL PASSED' if ok else 'SOME FAILED'}")
    print("=" * 70)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_swings(df, swings, ll, ls, start_date=None, end_date=None, save_path=None):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available."); return
    
    ts, cl = df.index, df["close"].values
    mask = np.ones(len(ts), dtype=bool)
    if start_date: mask &= ts >= pd.Timestamp(start_date)
    if end_date:   mask &= ts <= pd.Timestamp(end_date)
    idx = np.where(mask)[0]
    if not len(idx):
        print("No data in range."); return
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(ts[mask], cl[mask], color="#333", linewidth=0.7)
    
    for i in idx:
        if ll[i]: ax.axvspan(ts[i], ts[min(i+1, len(ts)-1)], alpha=0.2, color="green", lw=0)
        if ls[i]: ax.axvspan(ts[i], ts[min(i+1, len(ts)-1)], alpha=0.2, color="red", lw=0)
    
    idx_set = set(idx)
    for sp in swings:
        if sp.swing_index not in idx_set: continue
        if sp.tb_outcome == "sl": continue  # Don't plot failed swings
        m, c = ("^", "green") if sp.swing_type == "low" else ("v", "red")
        ax.plot(sp.swing_time, sp.swing_price, m, color=c, markersize=12,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        if sp.confirm_index in idx_set:
            cc = "blue" if sp.swing_type == "low" else "orange"
            ax.plot(sp.confirm_time, cl[sp.confirm_index], "x", color=cc,
                    markersize=8, markeredgewidth=2, zorder=4)
    
    ax.set_title("OTE v2: Structural Swing Detection (1hr ATR thresholds)", fontsize=14)
    ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Line2D([0],[0], color="#333", lw=1, label="Close"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="green", ms=10, label="Swing Low"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="red", ms=10, label="Swing High"),
        Line2D([0],[0], marker="x", color="blue", ms=8, mew=2, ls="None", label="Confirm (L)"),
        Line2D([0],[0], marker="x", color="orange", ms=8, mew=2, ls="None", label="Confirm (S)"),
    ], loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def build_ote_labels(df_5m, df_30m, params=None, df_1m=None, df_1hr=None, verbose=True):
    """
    Master pipeline for OTE labels.
    
    v2: Uses structural ATR from 1hr timeframe for all thresholds.
    """
    if params is None: params = OTEParams()
    params.validate()
    
    df_5m = _prep(df_5m); df_30m = _prep(df_30m)
    anomaly_count = int(df_5m.attrs.get("anomaly_count", 0))
    n = len(df_5m)
    if verbose: print(f"[OTE v2] {n:,} bars on 5m")
    
    # ATR
    atr_5m = compute_atr(df_5m, params.atr_period, params.atr_smoothing)
    atr_30m = compute_atr(df_30m, params.atr_period, params.atr_smoothing)
    
    # Structural ATR
    if df_1hr is not None and params.structural_atr_tf == "1hr":
        df_1hr = _prep(df_1hr)
        atr_1hr = compute_atr(df_1hr, params.atr_period, params.atr_smoothing)
        satr_5m = map_htf_atr_to_ltf(atr_1hr, df_5m.index)
        satr_30m = map_htf_atr_to_ltf(atr_1hr, df_30m.index)
        if verbose:
            v = satr_5m.dropna()
            if len(v): print(f"[OTE v2] Structural ATR (1hr→5m): {v.mean():.6f} ({v.mean()*10000:.1f} pips)")
    else:
        satr_5m = map_htf_atr_to_ltf(atr_30m, df_5m.index)
        satr_30m = atr_30m
        if verbose:
            v = satr_5m.dropna()
            if len(v): print(f"[OTE v2] Structural ATR (30m→5m): {v.mean():.6f} ({v.mean()*10000:.1f} pips)")
    
    df_5m["atr"] = atr_5m; df_5m["structural_atr"] = satr_5m
    
    # CUSUM
    ct = satr_5m * params.cusum_atr_mult
    cusum_ev = cusum_filter(df_5m["close"], ct, params.cusum_use_dynamic, params.cusum_fixed_threshold)
    if verbose: print(f"[OTE v2] CUSUM events: {len(cusum_ev):,}")
    
    # 5m swings with structural ATR
    sw5 = detect_swings_zigzag(
        df_5m, satr_5m, params.confirm_atr_mult, params.min_swing_atr,
        params.min_swing_distance_atr, params.min_bars_between_swings,
        params.warmup_bars, params.confirm_use_close, "5m"
    )
    if verbose: print(f"[OTE v2] 5m swings: {len(sw5)}")
    
    # 30m swings
    sw30 = detect_swings_zigzag(
        df_30m, satr_30m, params.htf_confirm_atr_mult, params.htf_min_swing_atr,
        params.min_swing_distance_atr * 0.8, params.htf_min_bars_between,
        params.warmup_bars, params.confirm_use_close, "30m"
    )
    if verbose: print(f"[OTE v2] 30m swings: {len(sw30)}")
    
    # 1hr swings
    sw1h = []
    if df_1hr is not None:
        atr1h = compute_atr(df_1hr, params.atr_period, params.atr_smoothing)
        sw1h = detect_swings_zigzag(
            df_1hr, atr1h, params.htf_confirm_atr_mult,
            params.htf_min_swing_atr * 0.7, params.min_swing_distance_atr * 0.7,
            max(2, params.htf_min_bars_between // 2), params.warmup_bars,
            params.confirm_use_close, "1hr"
        )
        if verbose: print(f"[OTE v2] 1hr swings: {len(sw1h)}")

    # HTF matching / timing metadata
    htf_match = compute_htf_swing_match(sw5, sw30, params.htf_confluence_window_minutes)
    htf_match_1h = compute_htf_swing_match(sw5, sw1h, params.htf_confluence_window_minutes * 2) if sw1h else {}
    sw5 = annotate_swing_context(sw5, sw30, sw1h, htf_match, htf_match_1h)
    
    # Triple barrier
    sw5 = validate_swings_triple_barrier(df_5m, sw5, satr_5m, params)
    if params.tb_enable:
        nv = sum(1 for s in sw5 if s.tb_outcome != "sl")
        if verbose: print(f"[OTE v2] TB validation: {nv}/{len(sw5)} passed")

    # Trend scanning validation and entry scoring
    sw5 = trend_scan_swings(df_5m, sw5, params)
    _, _ = assign_entry_labels(df_5m, sw5, satr_5m, params)
    sw5 = compute_swing_quality(sw5, params)
    entry_long, entry_short = assign_entry_labels(df_5m, sw5, satr_5m, params)
    
    # Labels
    ll, ls = create_zone_labels(n, sw5, params)
    df_5m["label_long_ote"] = ll; df_5m["label_short_ote"] = ls
    df_5m["label_long_entry"] = entry_long; df_5m["label_short_entry"] = entry_short
    ql, qs, eql, eqs = create_quality_arrays(n, sw5, params)
    df_5m["label_quality_long"] = ql; df_5m["label_quality_short"] = qs
    df_5m["entry_quality_long"] = eql; df_5m["entry_quality_short"] = eqs
    
    # Exclusion
    el, es, nl, ns = create_exclusion_masks(n, sw5, ll, ls, params)
    df_5m["exclude_long"] = el; df_5m["exclude_short"] = es
    df_5m["neg_ok_long"] = nl; df_5m["neg_ok_short"] = ns
    
    # HTF confluence
    hcl, hcs = compute_htf_confluence(sw5, sw30, n, df_5m.index, params.htf_confluence_window_minutes)
    df_5m["htf_confluence_long"] = hcl; df_5m["htf_confluence_short"] = hcs
    
    if sw1h:
        h1l, h1s = compute_htf_confluence(sw5, sw1h, n, df_5m.index, params.htf_confluence_window_minutes * 2)
        df_5m["htf_confluence_long_1hr"] = h1l; df_5m["htf_confluence_short_1hr"] = h1s
        timing_1h = compute_htf_timing_features(n, df_5m.index, sw1h, "1h")
        for col, values in timing_1h.items():
            df_5m[col] = values
    timing_30m = compute_htf_timing_features(n, df_5m.index, sw30, "30m")
    for col, values in timing_30m.items():
        df_5m[col] = values
    
    # Uniqueness
    if params.compute_uniqueness:
        cl, cs = compute_label_concurrency(n, sw5, params)
        df_5m["concurrency_long"] = cl; df_5m["concurrency_short"] = cs
        df_5m["sample_weight_long"] = compute_sample_weights(cl, ll) * np.where(ll > 0, np.maximum(ql, params.min_label_quality), 1.0)
        df_5m["sample_weight_short"] = compute_sample_weights(cs, ls) * np.where(ls > 0, np.maximum(qs, params.min_label_quality), 1.0)
        df_5m["sample_weight_entry_long"] = compute_sample_weights(np.maximum(cl, 1), entry_long) * np.where(entry_long > 0, np.maximum(eql, params.min_label_quality), 1.0)
        df_5m["sample_weight_entry_short"] = compute_sample_weights(np.maximum(cs, 1), entry_short) * np.where(entry_short > 0, np.maximum(eqs, params.min_label_quality), 1.0)
    
    wm = np.zeros(n, dtype=bool); wm[:params.warmup_bars] = True
    df_5m["warmup_mask"] = wm
    
    diag = run_diagnostics(
        df_5m, sw5, ll, ls, el, es, nl, ns, satr_5m, params,
        hcl, hcs, htf_match, entry_long, entry_short, anomaly_count
    )
    if verbose: print_diagnostics(diag)
    
    return df_5m, diag, sw5


def _prep(df):
    df = df.copy()
    required = {"open", "high", "low", "close"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            raise ValueError("Need DatetimeIndex or 'timestamp' column")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "volume" not in df.columns: df["volume"] = 0
    return df


# ============================================================================
# SYNTHETIC DATA + EXAMPLE
# ============================================================================

def generate_synthetic_fx_data(start="2023-01-02", end="2023-03-31", freq_minutes=5,
                                base_price=1.0800, volatility=0.0001, seed=42):
    np.random.seed(seed)
    ts = pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")
    mask = ~((ts.dayofweek == 5) | ((ts.dayofweek == 6) & (ts.hour < 17)))
    ts = ts[mask]; n = len(ts)
    
    vm = np.ones(n)
    for rc in np.random.choice(n, size=max(1, n//500), replace=False):
        vm[rc:min(rc + np.random.randint(50,200), n)] = np.random.uniform(0.5, 3.0)
    hours = ts.hour
    iv = np.where((hours >= 8) & (hours <= 12), 1.5,
         np.where((hours >= 13) & (hours <= 16), 1.8, np.where(hours <= 5, 0.6, 1.0)))
    ev = volatility * vm * iv
    lr = np.random.normal(0, ev)
    p = np.zeros(n); p[0] = base_price
    for i in range(1, n):
        p[i] = p[i-1] * np.exp(lr[i] - 0.01 * (p[i-1] - base_price) * volatility)
    sp = ev * 0.5
    h = p + np.abs(np.random.normal(0, sp))
    l = p - np.abs(np.random.normal(0, sp))
    o = np.roll(p, 1) + np.random.normal(0, sp * 0.3); o[0] = p[0]
    h = np.maximum(h, np.maximum(o, p)); l = np.minimum(l, np.minimum(o, p))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": p,
                          "volume": (np.random.poisson(100, n) * iv).astype(int)}, index=ts)


def run_example():
    print("=" * 70); print("OTE v2 — EXAMPLE"); print("=" * 70)
    d5 = generate_synthetic_fx_data(freq_minutes=5, seed=42)
    d30 = generate_synthetic_fx_data(freq_minutes=30, seed=42)
    d1h = generate_synthetic_fx_data(freq_minutes=60, seed=42)
    print(f"5m: {len(d5):,}  30m: {len(d30):,}  1hr: {len(d1h):,}")
    df, diag, sw = build_ote_labels(d5, d30, df_1hr=d1h, verbose=True)
    plot_swings(df, sw, df["label_long_ote"].values, df["label_short_ote"].values,
                "2023-01-09", "2023-01-20", "/home/claude/ote_v2_plot.png")
    return df, diag, sw

if __name__ == "__main__":
    run_example()

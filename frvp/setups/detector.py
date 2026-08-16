from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from features.config import FeatureBuilderConfig


SETUP_1 = 1
SETUP_2 = 2
SETUP_3 = 3
SETUP_4 = 4
SETUP_5 = 5
SETUP_6 = 6

SETUP2_LOOKBACK_BARS = 15
SETUP4_MAX_OUTSIDE_BARS = 3
SETUP4_SWEEP_LOOKBACK_BARS = 3
SETUP4_MAX_REENTRY_VOLUME_ZSCORE = 0.75
SETUP4_MIN_REENTRY_DEPTH_ATR = 0.05
SETUP_TOUCH_ATR = 0.65
SETUP1_REENTRY_OVERSHOOT_ATR = 0.30
SETUP3_VOLUME_ZSCORE = 1.25
REAL_DISPLACEMENT_MIN_CLOSE_LOCATION = 0.75
SETUP3_MIN_OVERSHOOT_ATR = 0.05
SETUP5_LVN_DISTANCE_ATR = 0.3
SETUP6_MIN_HVN_DISTANCE_ATR = 0.75
SETUP6_MAX_HVN_DISTANCE_ATR = 2.5
SETUP6_MIN_HVN_SEPARATION_ATR = 0.30
SETUP6_MIN_FAR_NEAR_RATIO = 1.35
SETUP6_MAX_VOLUME_ZSCORE = 0.50
RETEST_TOLERANCE_ATR = 0.30
SETUP2_MAX_INSIDE_CLOSE_ATR = 0.05

SETUP_PRIORITY = {
    SETUP_4: 0,
    SETUP_2: 1,
    SETUP_3: 2,
    SETUP_5: 3,
    SETUP_1: 4,
    SETUP_6: 5,
}


@dataclass(frozen=True)
class _BreakoutState:
    index: int
    session_key: object


@dataclass(frozen=True)
class _SetupCandidate:
    setup_type: int
    setup_side: int
    confidence: float


def detect_frvp_setups(df: pd.DataFrame) -> pd.DataFrame:
    """Detect the FRVP rule-based setups using completed-bar information only."""

    required = {
        "open",
        "high",
        "low",
        "close",
        "frvp_profile_shape",
        "frvp_open_type",
        "frvp_open_drive_flag",
        "frvp_in_va",
        "frvp_above_vah",
        "frvp_below_val",
        "frvp_dist_vah_atr",
        "frvp_dist_val_atr",
        "frvp_dist_nearest_lvn_atr",
        "frvp_hvn_above_close",
        "frvp_hvn_below_close",
        "volume_zscore_50",
        "displacement_bullish",
        "displacement_bearish",
        "bars_since_sweep_high",
        "bars_since_sweep_low",
        "atr_14",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"FRVP setup detection is missing required columns: {missing_text}")

    working = df.reset_index(drop=True)
    fired: list[bool] = []
    setup_type: list[int] = []
    setup_side: list[int] = []
    confidence: list[float] = []

    previous_candidates: dict[tuple[int, int], bool] = {}
    last_breakout_above: _BreakoutState | None = None
    last_breakout_below: _BreakoutState | None = None
    outside_run_side = 0
    outside_run_length = 0
    previous_session_key: object | None = None
    previous_contract_id: str | None = None

    for index, row in working.iterrows():
        session_key = row["session_date"] if "session_date" in working.columns else None
        contract_id = str(row["contract_id"]) if "contract_id" in working.columns else None
        if index == 0 or session_key != previous_session_key or contract_id != previous_contract_id:
            previous_candidates = {}
            last_breakout_above = None
            last_breakout_below = None
            outside_run_side = 0
            outside_run_length = 0

        atr = _to_float(row.get("atr_14"))
        current_candidates: list[_SetupCandidate] = []

        candidate_1 = _candidate_setup_1(row)
        if candidate_1 is not None:
            current_candidates.append(candidate_1)

        candidate_2 = _candidate_setup_2(
            row=row,
            index=index,
            atr=atr,
            last_breakout_above=last_breakout_above,
            last_breakout_below=last_breakout_below,
            session_key=session_key,
        )
        if candidate_2 is not None:
            current_candidates.append(candidate_2)

        candidate_3 = _candidate_setup_3(row)
        if candidate_3 is not None:
            current_candidates.append(candidate_3)

        candidate_4 = _candidate_setup_4(row, outside_run_side=outside_run_side, outside_run_length=outside_run_length)
        if candidate_4 is not None:
            current_candidates.append(candidate_4)

        candidate_5 = _candidate_setup_5(row)
        if candidate_5 is not None:
            current_candidates.append(candidate_5)

        candidate_6 = _candidate_setup_6(row)
        if candidate_6 is not None:
            current_candidates.append(candidate_6)

        current_active = {(candidate.setup_type, candidate.setup_side): True for candidate in current_candidates}
        eligible = [
            candidate
            for candidate in current_candidates
            if not previous_candidates.get((candidate.setup_type, candidate.setup_side), False)
        ]
        selected = _select_candidate(eligible)

        if selected is None:
            fired.append(False)
            setup_type.append(0)
            setup_side.append(0)
            confidence.append(0.0)
        else:
            fired.append(True)
            setup_type.append(int(selected.setup_type))
            setup_side.append(int(selected.setup_side))
            confidence.append(float(selected.confidence))

        if candidate_3 is not None:
            if candidate_3.setup_side > 0:
                last_breakout_above = _BreakoutState(index=index, session_key=session_key)
            else:
                last_breakout_below = _BreakoutState(index=index, session_key=session_key)

        row_above_vah = _as_bool(row.get("frvp_above_vah"))
        row_below_val = _as_bool(row.get("frvp_below_val"))
        if row_above_vah and not row_below_val:
            if outside_run_side == -1:
                outside_run_length += 1
            else:
                outside_run_side = -1
                outside_run_length = 1
        elif row_below_val and not row_above_vah:
            if outside_run_side == 1:
                outside_run_length += 1
            else:
                outside_run_side = 1
                outside_run_length = 1
        else:
            outside_run_side = 0
            outside_run_length = 0

        previous_candidates = current_active
        previous_session_key = session_key
        previous_contract_id = contract_id

    return pd.DataFrame(
        {
            "fired": pd.Series(fired, dtype=bool),
            "setup_type": pd.Series(setup_type, dtype="int64"),
            "setup_side": pd.Series(setup_side, dtype="int64"),
            "confidence": pd.Series(confidence, dtype="float64"),
        },
        index=df.index,
    )


def summarize_setup_fire_rates(
    df: pd.DataFrame,
    *,
    config: FeatureBuilderConfig | None = None,
    instrument: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize setup fire counts per session and side on a sample window."""

    setup_input = _ensure_setup_input_frame(df, config=config, instrument=instrument)
    detections = detect_frvp_setups(setup_input)
    if "session_date" not in setup_input.columns:
        raise ValueError("Setup fire-rate diagnostic requires a session_date column.")

    total_sessions = int(pd.Series(setup_input["session_date"]).dropna().nunique())
    if total_sessions <= 0:
        raise ValueError("Setup fire-rate diagnostic requires at least one completed session.")

    events = pd.concat(
        [
            setup_input.loc[:, ["session_date"]].reset_index(drop=True),
            detections.reset_index(drop=True),
        ],
        axis=1,
    )
    fired = events.loc[events["fired"]].copy()
    if fired.empty:
        summary = pd.DataFrame(
            [
                {
                    "setup_type": setup,
                    "setup_side": side,
                    "total_fires": 0,
                    "sessions_with_fire": 0,
                    "avg_fires_per_session_side": 0.0,
                    "within_target_band": False,
                }
                for setup in range(1, 7)
                for side in (-1, 1)
            ]
        )
    else:
        per_session = (
            fired.groupby(["setup_type", "setup_side", "session_date"], sort=True)
            .size()
            .reset_index(name="fires")
        )
        summary = (
            per_session.groupby(["setup_type", "setup_side"], sort=True)
            .agg(
                total_fires=("fires", "sum"),
                sessions_with_fire=("session_date", "nunique"),
            )
            .reset_index()
        )
        summary["avg_fires_per_session_side"] = summary["total_fires"] / float(total_sessions)
        summary["within_target_band"] = summary["avg_fires_per_session_side"].between(0.5, 2.0, inclusive="both")
        template = pd.DataFrame(
            [(setup, side) for setup in range(1, 7) for side in (-1, 1)],
            columns=["setup_type", "setup_side"],
        )
        summary = template.merge(summary, on=["setup_type", "setup_side"], how="left")
        summary["total_fires"] = summary["total_fires"].fillna(0).astype(int)
        summary["sessions_with_fire"] = summary["sessions_with_fire"].fillna(0).astype(int)
        summary["avg_fires_per_session_side"] = summary["avg_fires_per_session_side"].fillna(0.0)
        summary["within_target_band"] = summary["within_target_band"].where(
            summary["within_target_band"].notna(),
            False,
        ).astype(bool)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    return summary


def _candidate_setup_1(row: pd.Series) -> _SetupCandidate | None:
    if _to_int(row.get("frvp_profile_shape")) != 0:
        return None
    if pd.isna(row.get("frvp_open_type")):
        return None
    if _to_int(row.get("frvp_open_type")) != 0:
        return None

    dist_vah = _to_float(row.get("frvp_dist_vah_atr"))
    dist_val = _to_float(row.get("frvp_dist_val_atr"))
    volume_zscore = _to_float(row.get("volume_zscore_50"))
    open_drive = _to_int(row.get("frvp_open_drive_flag"), default=0)
    quiet_volume_bonus = _clamp01(1.0 - max(volume_zscore, 0.0) / SETUP3_VOLUME_ZSCORE) if np.isfinite(volume_zscore) else 0.5
    drive_bonus = 0.0 if open_drive == 1 else 1.0

    if np.isfinite(dist_vah) and (-SETUP1_REENTRY_OVERSHOOT_ATR) <= dist_vah <= SETUP_TOUCH_ATR:
        if dist_vah < 0.0 and _has_real_displacement(row, direction=1) and _outside_overshoot_atr(row, direction=1) >= SETUP3_MIN_OVERSHOOT_ATR:
            return None
        touch_score = _clamp01(1.0 - (max(dist_vah, 0.0) / SETUP_TOUCH_ATR))
        reentry_bonus = _clamp01((-dist_vah) / SETUP1_REENTRY_OVERSHOOT_ATR) if dist_vah < 0.0 else 0.0
        return _SetupCandidate(
            setup_type=SETUP_1,
            setup_side=-1,
            confidence=_bounded_confidence(
                0.42 + 0.14 * touch_score + 0.08 * reentry_bonus + 0.08 * quiet_volume_bonus + 0.06 * drive_bonus
            ),
        )
    if np.isfinite(dist_val) and (-SETUP1_REENTRY_OVERSHOOT_ATR) <= dist_val <= SETUP_TOUCH_ATR:
        if dist_val < 0.0 and _has_real_displacement(row, direction=-1) and _outside_overshoot_atr(row, direction=-1) >= SETUP3_MIN_OVERSHOOT_ATR:
            return None
        touch_score = _clamp01(1.0 - (max(dist_val, 0.0) / SETUP_TOUCH_ATR))
        reentry_bonus = _clamp01((-dist_val) / SETUP1_REENTRY_OVERSHOOT_ATR) if dist_val < 0.0 else 0.0
        return _SetupCandidate(
            setup_type=SETUP_1,
            setup_side=1,
            confidence=_bounded_confidence(
                0.42 + 0.14 * touch_score + 0.08 * reentry_bonus + 0.08 * quiet_volume_bonus + 0.06 * drive_bonus
            ),
        )
    return None


def _candidate_setup_2(
    *,
    row: pd.Series,
    index: int,
    atr: float,
    last_breakout_above: _BreakoutState | None,
    last_breakout_below: _BreakoutState | None,
    session_key: object,
) -> _SetupCandidate | None:
    if not np.isfinite(atr) or atr <= 0.0:
        return None

    close_price = _to_float(row.get("close"))
    low_price = _to_float(row.get("low"))
    high_price = _to_float(row.get("high"))
    dist_vah = _to_float(row.get("frvp_dist_vah_atr"))
    dist_val = _to_float(row.get("frvp_dist_val_atr"))

    if last_breakout_above is not None and last_breakout_above.session_key == session_key:
        age = index - last_breakout_above.index
        if 1 <= age <= SETUP2_LOOKBACK_BARS and np.isfinite(dist_vah):
            vah = close_price + (dist_vah * atr)
            touched = low_price <= (vah + (RETEST_TOLERANCE_ATR * atr))
            held = close_price >= (vah - (SETUP2_MAX_INSIDE_CLOSE_ATR * atr))
            if touched and held:
                retest_score = _clamp01(1.0 - min(abs(close_price - vah), abs(low_price - vah)) / (0.35 * atr))
                freshness_score = _clamp01(1.0 - ((age - 1) / max(SETUP2_LOOKBACK_BARS - 1, 1)))
                return _SetupCandidate(
                    setup_type=SETUP_2,
                    setup_side=1,
                    confidence=_bounded_confidence(0.63 + 0.19 * retest_score + 0.12 * freshness_score),
                )

    if last_breakout_below is not None and last_breakout_below.session_key == session_key:
        age = index - last_breakout_below.index
        if 1 <= age <= SETUP2_LOOKBACK_BARS and np.isfinite(dist_val):
            val = close_price - (dist_val * atr)
            touched = high_price >= (val - (RETEST_TOLERANCE_ATR * atr))
            held = close_price <= (val + (SETUP2_MAX_INSIDE_CLOSE_ATR * atr))
            if touched and held:
                retest_score = _clamp01(1.0 - min(abs(close_price - val), abs(high_price - val)) / (0.35 * atr))
                freshness_score = _clamp01(1.0 - ((age - 1) / max(SETUP2_LOOKBACK_BARS - 1, 1)))
                return _SetupCandidate(
                    setup_type=SETUP_2,
                    setup_side=-1,
                    confidence=_bounded_confidence(0.63 + 0.19 * retest_score + 0.12 * freshness_score),
                )

    return None


def _candidate_setup_3(row: pd.Series) -> _SetupCandidate | None:
    volume_zscore = _to_float(row.get("volume_zscore_50"))
    if not np.isfinite(volume_zscore) or volume_zscore < SETUP3_VOLUME_ZSCORE:
        return None
    if _as_bool(row.get("frvp_above_vah")):
        if not _has_real_displacement(row, direction=1):
            return None
        overshoot_atr = _outside_overshoot_atr(row, direction=1)
        if not np.isfinite(overshoot_atr) or overshoot_atr < SETUP3_MIN_OVERSHOOT_ATR:
            return None
        expansion_score = _clamp01((volume_zscore - SETUP3_VOLUME_ZSCORE) / 2.0)
        overshoot_score = _clamp01((overshoot_atr - SETUP3_MIN_OVERSHOOT_ATR) / 0.35)
        return _SetupCandidate(
            setup_type=SETUP_3,
            setup_side=1,
            confidence=_bounded_confidence(
                0.58 + 0.18 * expansion_score + 0.12 * overshoot_score + 0.05 * _as_int(row.get("frvp_open_drive_flag"))
            ),
        )
    if _as_bool(row.get("frvp_below_val")):
        if not _has_real_displacement(row, direction=-1):
            return None
        overshoot_atr = _outside_overshoot_atr(row, direction=-1)
        if not np.isfinite(overshoot_atr) or overshoot_atr < SETUP3_MIN_OVERSHOOT_ATR:
            return None
        expansion_score = _clamp01((volume_zscore - SETUP3_VOLUME_ZSCORE) / 2.0)
        overshoot_score = _clamp01((overshoot_atr - SETUP3_MIN_OVERSHOOT_ATR) / 0.35)
        return _SetupCandidate(
            setup_type=SETUP_3,
            setup_side=-1,
            confidence=_bounded_confidence(
                0.58 + 0.18 * expansion_score + 0.12 * overshoot_score + 0.05 * _as_int(row.get("frvp_open_drive_flag"))
            ),
        )
    return None


def _candidate_setup_4(
    row: pd.Series,
    *,
    outside_run_side: int,
    outside_run_length: int,
) -> _SetupCandidate | None:
    if not _as_bool(row.get("frvp_in_va")):
        return None
    if outside_run_side == 0 or not (1 <= outside_run_length <= SETUP4_MAX_OUTSIDE_BARS):
        return None
    if _as_bool(row.get("frvp_open_drive_flag")):
        return None

    volume_zscore = _to_float(row.get("volume_zscore_50"))
    if np.isfinite(volume_zscore) and volume_zscore > SETUP4_MAX_REENTRY_VOLUME_ZSCORE:
        return None

    if outside_run_side < 0:
        same_side_sweep = _within_bars(row.get("bars_since_sweep_high"), SETUP4_SWEEP_LOOKBACK_BARS)
        reentry_depth = _to_float(row.get("frvp_dist_vah_atr"))
    else:
        same_side_sweep = _within_bars(row.get("bars_since_sweep_low"), SETUP4_SWEEP_LOOKBACK_BARS)
        reentry_depth = _to_float(row.get("frvp_dist_val_atr"))
    if not same_side_sweep:
        return None
    if not np.isfinite(reentry_depth) or reentry_depth <= SETUP4_MIN_REENTRY_DEPTH_ATR:
        return None

    duration_score = _clamp01(1.0 - ((outside_run_length - 1) / max(SETUP4_MAX_OUTSIDE_BARS - 1, 1)))
    quiet_reentry_bonus = (
        _clamp01(1.0 - max(volume_zscore, 0.0) / SETUP4_MAX_REENTRY_VOLUME_ZSCORE)
        if np.isfinite(volume_zscore)
        else 0.5
    )
    reentry_score = _clamp01((reentry_depth - SETUP4_MIN_REENTRY_DEPTH_ATR) / 0.25)
    side = outside_run_side
    return _SetupCandidate(
        setup_type=SETUP_4,
        setup_side=side,
        confidence=_bounded_confidence(0.66 + 0.10 * duration_score + 0.10 * quiet_reentry_bonus + 0.10 * reentry_score),
    )


def _candidate_setup_5(row: pd.Series) -> _SetupCandidate | None:
    lvn_distance = _to_float(row.get("frvp_dist_nearest_lvn_atr"))
    if not np.isfinite(lvn_distance) or lvn_distance > SETUP5_LVN_DISTANCE_ATR:
        return None

    if _as_bool(row.get("displacement_bullish")):
        proximity_score = _clamp01(1.0 - (lvn_distance / SETUP5_LVN_DISTANCE_ATR))
        return _SetupCandidate(
            setup_type=SETUP_5,
            setup_side=1,
            confidence=_bounded_confidence(0.61 + 0.24 * proximity_score + 0.05 * _as_int(row.get("frvp_open_drive_flag"))),
        )
    if _as_bool(row.get("displacement_bearish")):
        proximity_score = _clamp01(1.0 - (lvn_distance / SETUP5_LVN_DISTANCE_ATR))
        return _SetupCandidate(
            setup_type=SETUP_5,
            setup_side=-1,
            confidence=_bounded_confidence(0.61 + 0.24 * proximity_score + 0.05 * _as_int(row.get("frvp_open_drive_flag"))),
        )
    return None


def _candidate_setup_6(row: pd.Series) -> _SetupCandidate | None:
    hvn_above = _to_float(row.get("frvp_hvn_above_close"))
    hvn_below = _to_float(row.get("frvp_hvn_below_close"))
    if not np.isfinite(hvn_above) or not np.isfinite(hvn_below):
        return None
    if _to_int(row.get("frvp_profile_shape")) != 0:
        return None
    if _to_int(row.get("frvp_open_type")) != 0:
        return None
    if not _as_bool(row.get("frvp_in_va")):
        return None
    if _as_bool(row.get("frvp_open_drive_flag")):
        return None
    if _as_bool(row.get("displacement_bullish")) or _as_bool(row.get("displacement_bearish")):
        return None

    volume_zscore = _to_float(row.get("volume_zscore_50"))
    if np.isfinite(volume_zscore) and volume_zscore > SETUP6_MAX_VOLUME_ZSCORE:
        return None

    nearest = min(hvn_above, hvn_below)
    farthest = max(hvn_above, hvn_below)
    if nearest <= SETUP6_MIN_HVN_DISTANCE_ATR:
        return None
    if nearest >= SETUP6_MAX_HVN_DISTANCE_ATR:
        return None
    if (farthest - nearest) < SETUP6_MIN_HVN_SEPARATION_ATR:
        return None
    if farthest / max(nearest, 1e-8) < SETUP6_MIN_FAR_NEAR_RATIO:
        return None

    direction = 1 if hvn_above < hvn_below else -1
    distance_score = _clamp01((nearest - SETUP6_MIN_HVN_DISTANCE_ATR) / 1.0)
    asymmetry_score = _clamp01((farthest - nearest) / 1.0)
    quiet_volume_bonus = (
        _clamp01(1.0 - max(volume_zscore, 0.0) / SETUP6_MAX_VOLUME_ZSCORE)
        if np.isfinite(volume_zscore)
        else 0.5
    )
    return _SetupCandidate(
        setup_type=SETUP_6,
        setup_side=direction,
        confidence=_bounded_confidence(0.50 + 0.12 * distance_score + 0.12 * asymmetry_score + 0.08 * quiet_volume_bonus),
    )


def _select_candidate(candidates: list[_SetupCandidate]) -> _SetupCandidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            -SETUP_PRIORITY[candidate.setup_type],
            -candidate.setup_type,
            candidate.setup_side,
        ),
    )


def _ensure_setup_input_frame(
    df: pd.DataFrame,
    *,
    config: FeatureBuilderConfig | None,
    instrument: str | None,
) -> pd.DataFrame:
    required = {
        "open",
        "high",
        "low",
        "close",
        "session_date",
        "atr_14",
        "volume_zscore_50",
        "displacement_bullish",
        "displacement_bearish",
        "bars_since_sweep_high",
        "bars_since_sweep_low",
        "frvp_profile_shape",
        "frvp_open_type",
        "frvp_open_drive_flag",
        "frvp_in_va",
        "frvp_above_vah",
        "frvp_below_val",
        "frvp_dist_vah_atr",
        "frvp_dist_val_atr",
        "frvp_dist_nearest_lvn_atr",
        "frvp_hvn_above_close",
        "frvp_hvn_below_close",
    }
    if required.issubset(df.columns):
        return df.reset_index(drop=True)

    if config is None:
        from features.config import FeatureBuilderConfig

        config = FeatureBuilderConfig(
            feature_sets=["frvp_context"],
            warmup_rows=0,
            drop_warmup_rows=False,
            fillna_numeric=False,
            enable_lags=False,
            enable_rolling_stats=False,
            enable_zscores=False,
            enable_winsorization=False,
            enable_percentile_ranks=False,
            enable_atr_normalization=False,
            enable_sigma_normalization=False,
            enable_interactions=False,
        )

    from ..feature_sets.frvp_context import _prepare_frvp_context, build_frvp_context_features

    prepared = _prepare_frvp_context(df, config, instrument=instrument)
    features = build_frvp_context_features(df, config)
    return pd.concat(
        [
            prepared.market.reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )


def _to_float(value, *, default: float = np.nan) -> float:
    if value is None or value is pd.NA or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, *, default: int = 0) -> int:
    if value is None or value is pd.NA or pd.isna(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value) -> bool:
    return _to_int(value, default=0) == 1


def _as_int(value) -> int:
    return 1 if _as_bool(value) else 0


def _within_bars(value, maximum: int) -> bool:
    numeric = _to_float(value)
    return np.isfinite(numeric) and numeric <= maximum


def _clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _bounded_confidence(value: float) -> float:
    return _clamp01(value)


def _has_real_displacement(row: pd.Series, *, direction: int) -> bool:
    if direction > 0:
        displacement_flag = _as_bool(row.get("displacement_bullish"))
    else:
        displacement_flag = _as_bool(row.get("displacement_bearish"))
    if not displacement_flag:
        return False

    volume_zscore = _to_float(row.get("volume_zscore_50"))
    if not np.isfinite(volume_zscore) or volume_zscore < SETUP3_VOLUME_ZSCORE:
        return False

    close_location = _close_location_fraction(row, direction=direction)
    if not np.isfinite(close_location) or close_location < REAL_DISPLACEMENT_MIN_CLOSE_LOCATION:
        return False
    return True


def _close_location_fraction(row: pd.Series, *, direction: int) -> float:
    high_price = _to_float(row.get("high"))
    low_price = _to_float(row.get("low"))
    close_price = _to_float(row.get("close"))
    candle_range = high_price - low_price
    if not np.isfinite(candle_range) or candle_range <= 0.0:
        return np.nan
    if direction > 0:
        return _clamp01((close_price - low_price) / candle_range)
    return _clamp01((high_price - close_price) / candle_range)


def _outside_overshoot_atr(row: pd.Series, *, direction: int) -> float:
    if direction > 0:
        distance = _to_float(row.get("frvp_dist_vah_atr"))
    else:
        distance = _to_float(row.get("frvp_dist_val_atr"))
    if not np.isfinite(distance):
        return np.nan
    return max(-distance, 0.0)

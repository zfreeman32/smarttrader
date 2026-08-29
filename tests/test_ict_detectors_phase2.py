from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.config import FeatureBuilderConfig  # noqa: E402
from ict.detectors.displacement import detect_ict_displacement  # noqa: E402
from ict.detectors.fvg import detect_ict_fvg  # noqa: E402
from ict.detectors.order_blocks import detect_ict_order_blocks  # noqa: E402
from ict.detectors.premium_discount import detect_ict_premium_discount  # noqa: E402
from ict.detectors.sweeps import detect_ict_sweeps  # noqa: E402
from ict.structure.swings import detect_ict_swings  # noqa: E402


def _frame(
    *,
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float] | None = None,
    atr: float = 1.0,
) -> pd.DataFrame:
    rows = len(close)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-03 09:30:00", periods=rows, freq="5min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume if volume is not None else [100.0] * rows,
            "atr_14": [atr] * rows,
        }
    )


def test_detect_ict_swings_records_confirmation_tax_causally() -> None:
    frame = _frame(
        open_=[9.5, 10.5, 14.0, 12.0, 10.0],
        high=[10.0, 11.0, 15.0, 12.0, 11.0],
        low=[9.0, 10.0, 11.0, 10.0, 9.0],
        close=[9.5, 10.5, 14.0, 12.0, 10.0],
    )

    features = detect_ict_swings(frame, FeatureBuilderConfig(instrument="es", swing_window=1))

    assert int(features.loc[3, "ict_confirmed_swing_high"]) == 1
    assert float(features.loc[3, "ict_confirmed_swing_high_level"]) == 15.0
    assert float(features.loc[3, "ict_confirmed_swing_high_pivot_index"]) == 2.0
    assert float(features.loc[3, "ict_swing_high_confirmation_tax_price"]) == 3.0
    assert float(features.loc[3, "ict_swing_high_confirmation_tax_atr"]) == 3.0
    assert float(features.loc[4, "ict_latest_swing_high_index"]) == 2.0


def test_detect_ict_fvg_tracks_ce_touch_and_ifvg_flip() -> None:
    frame = _frame(
        open_=[99.5, 100.0, 102.2, 101.8, 101.0],
        high=[100.0, 100.8, 103.0, 101.6, 101.2],
        low=[99.0, 99.8, 102.0, 100.8, 99.0],
        close=[99.5, 100.5, 102.5, 101.2, 99.5],
    )

    features = detect_ict_fvg(frame, FeatureBuilderConfig(instrument="es"))
    zones = features.attrs["fvg_zones"]

    assert int(features.loc[3, "ict_inside_bull_fvg"]) == 1
    assert int(features.loc[3, "ict_bull_fvg_ce_tapped"]) == 1
    assert int(features.loc[4, "active_bear_ifvg_count"]) == 1
    assert int(features.loc[4, "ict_nearest_bear_fvg_is_ifvg"]) == 1
    inverted = zones.loc[zones["inverted"].fillna(False)]
    assert len(inverted) == 1
    assert int(inverted.iloc[0]["inversion_index"]) == 4


def test_detect_ict_sweeps_requires_reclaim_within_k_and_flags_failed_acceptance() -> None:
    frame = _frame(
        open_=[104.5, 105.0, 105.0, 105.1, 105.3],
        high=[104.8, 105.6, 105.2, 105.8, 106.0],
        low=[104.3, 104.9, 104.6, 105.0, 105.2],
        close=[104.6, 105.2, 104.8, 105.4, 105.6],
    )
    swing_features = pd.DataFrame(
        {
            "ict_latest_swing_high": [105.0] * len(frame),
            "ict_latest_swing_low": [float("nan")] * len(frame),
        },
        index=frame.index,
    )

    features = detect_ict_sweeps(
        frame,
        FeatureBuilderConfig(
            instrument="es",
            ict_sweep_close_back_bars=1,
            ict_sweep_buffer_ticks=1.0,
            ict_sweep_buffer_atr=0.0,
        ),
        swing_features=swing_features,
    )

    assert int(features.loc[2, "ict_buy_side_sweep"]) == 1
    assert int(features.loc[2, "ict_sweep_level_code"]) == 1
    assert float(features.loc[2, "ict_sweep_reclaim_bars"]) == 1.0
    assert int(features.loc[4, "ict_failed_buy_side_sweep"]) == 1
    assert int(features.loc[4, "ict_sweep_any"]) == 0


def test_detect_ict_displacement_requires_volume_confirmation_when_volume_exists() -> None:
    base = _frame(
        open_=[100.0] * 25,
        high=([100.5] * 24) + [103.5],
        low=([99.5] * 24) + [99.5],
        close=([100.1] * 24) + [103.1],
        volume=[100.0 + ((index % 5) * 5.0) for index in range(24)] + [110.0],
    )
    high_volume = base.copy()
    high_volume.loc[24, "volume"] = 200.0

    low_features = detect_ict_displacement(base, FeatureBuilderConfig(instrument="es"))
    high_features = detect_ict_displacement(high_volume, FeatureBuilderConfig(instrument="es"))

    assert int(low_features.loc[24, "displacement_bullish"]) == 0
    assert int(high_features.loc[24, "displacement_bullish"]) == 1
    assert float(high_features.loc[24, "ict_displacement_volume_zscore"]) > float(
        low_features.loc[24, "ict_displacement_volume_zscore"]
    )


def test_detect_ict_order_blocks_track_retests_after_displacement_break() -> None:
    frame = _frame(
        open_=[101.0, 100.0, 100.4, 100.0],
        high=[101.5, 103.5, 101.2, 100.1],
        low=[99.5, 99.8, 99.9, 98.9],
        close=[100.0, 103.0, 100.5, 99.5],
    )
    displacement = pd.DataFrame(
        {
            "displacement_bullish": [0, 1, 0, 0],
            "displacement_bearish": [0, 0, 0, 0],
            "ict_displacement_score": [0.0, 2.0, 0.0, 0.0],
        },
        index=frame.index,
    )
    structure = pd.DataFrame(
        {
            "ict_bos_bull": [0, 1, 0, 0],
            "ict_choch_bull": [0, 0, 0, 0],
            "ict_bos_bear": [0, 0, 0, 0],
            "ict_choch_bear": [0, 0, 0, 0],
        },
        index=frame.index,
    )

    features = detect_ict_order_blocks(
        frame,
        FeatureBuilderConfig(instrument="es"),
        displacement_features=displacement,
        structure_features=structure,
    )

    assert int(features.loc[1, "ict_active_bull_order_block_count"]) == 1
    assert float(features.loc[2, "dist_to_bull_order_block_atr"]) == 0.0
    assert float(features.loc[2, "ict_bull_order_block_retest_count"]) == 1.0
    assert int(features.loc[3, "ict_active_bull_order_block_count"]) == 0


def test_detect_ict_premium_discount_emits_ote_band_and_dol_levels() -> None:
    frame = _frame(
        open_=[103.0],
        high=[104.0],
        low=[102.5],
        close=[103.5],
    )
    swing_features = pd.DataFrame(
        {
            "ict_latest_swing_high": [110.0],
            "ict_latest_swing_low": [100.0],
            "ict_latest_swing_high_index": [8.0],
            "ict_latest_swing_low_index": [4.0],
        },
        index=frame.index,
    )
    liquidity_features = pd.DataFrame(
        {
            "ict_prior_rth_high": [115.0],
            "ict_overnight_high": [113.0],
            "ict_ib_high": [112.0],
            "ict_prior_week_high": [120.0],
            "ict_prior_rth_low": [98.0],
            "ict_overnight_low": [99.0],
            "ict_ib_low": [97.5],
            "ict_prior_week_low": [97.0],
        },
        index=frame.index,
    )

    features = detect_ict_premium_discount(
        frame,
        FeatureBuilderConfig(instrument="es"),
        swing_features=swing_features,
        liquidity_features=liquidity_features,
    )

    assert int(features.loc[0, "ict_impulse_direction"]) == 1
    assert int(features.loc[0, "ict_in_ote_band"]) == 1
    assert int(features.loc[0, "ict_ote_bucket_code"]) == 0
    assert float(features.loc[0, "ict_dol_level_up"]) == 110.0
    assert float(features.loc[0, "ict_dol_level_down"]) == 100.0

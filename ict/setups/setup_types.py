from __future__ import annotations

from enum import IntEnum, StrEnum

import pandas as pd


class ICTSetupFamily(StrEnum):
    NONE = "none"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class ICTSetupType(StrEnum):
    NONE = "none"
    SWEEP_RECLAIM = "sweep_reclaim"
    SWEEP_DISPLACEMENT_FVG = "sweep_displacement_fvg"
    OB_RETEST_AFTER_MSS = "ob_retest_after_mss"
    IFVG_REVERSAL = "ifvg_reversal"
    PREMIUM_DISCOUNT_CONTINUATION = "premium_discount_continuation"
    SESSION_OPEN_MANIPULATION_PRE_IB = "session_open_manipulation_pre_ib"
    SESSION_OPEN_MANIPULATION_POST_IB = "session_open_manipulation_post_ib"
    DISPLACEMENT_CONTINUATION_AFTER_RAID = "displacement_continuation_after_raid"


class ICTSetupSide(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


SETUP_OUTPUT_COLUMNS = (
    "event_time",
    "fired",
    "setup_type",
    "setup_family",
    "setup_side",
    "confidence",
    "anchor_level",
    "entry_price",
    "stop_reference",
    "target_reference",
    "reference_level",
    "reference_level_type",
    "sweep_type",
    "htf_context",
    "fvg_id",
    "ce_price",
    "order_block_id",
    "displacement_id",
    "displacement_volume_z",
    "session_phase",
)


def build_empty_setup_frame(index: pd.Index, *, event_time: pd.Series | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": event_time if event_time is not None else pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"),
            "fired": pd.Series(False, index=index, dtype=bool),
            "setup_type": pd.Series(ICTSetupType.NONE.value, index=index, dtype="string"),
            "setup_family": pd.Series(ICTSetupFamily.NONE.value, index=index, dtype="string"),
            "setup_side": pd.Series(int(ICTSetupSide.FLAT), index=index, dtype="Int64"),
            "confidence": pd.Series(0.0, index=index, dtype=float),
            "anchor_level": pd.Series(float("nan"), index=index, dtype=float),
            "entry_price": pd.Series(float("nan"), index=index, dtype=float),
            "stop_reference": pd.Series(float("nan"), index=index, dtype=float),
            "target_reference": pd.Series(float("nan"), index=index, dtype=float),
            "reference_level": pd.Series(float("nan"), index=index, dtype=float),
            "reference_level_type": pd.Series("", index=index, dtype="string"),
            "sweep_type": pd.Series("", index=index, dtype="string"),
            "htf_context": pd.Series("", index=index, dtype="string"),
            "fvg_id": pd.Series(pd.NA, index=index, dtype="Int64"),
            "ce_price": pd.Series(float("nan"), index=index, dtype=float),
            "order_block_id": pd.Series(pd.NA, index=index, dtype="Int64"),
            "displacement_id": pd.Series(pd.NA, index=index, dtype="Int64"),
            "displacement_volume_z": pd.Series(float("nan"), index=index, dtype=float),
            "session_phase": pd.Series(pd.NA, index=index, dtype="Int64"),
        },
        index=index,
    )

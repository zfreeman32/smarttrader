from __future__ import annotations

from enum import StrEnum
from typing import Any

from .setups.setup_types import ICTSetupFamily, ICTSetupType


class ICTTradeType(StrEnum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    META = "meta"


ICT_REVERSAL_LABEL_FAMILY = "ict_reversal"
ICT_CONTINUATION_LABEL_FAMILY = "ict_continuation"
ICT_META_LABEL_FAMILY = "ict_meta"

ICT_TRADE_TYPE_TO_LABEL_FAMILY = {
    ICTTradeType.REVERSAL.value: ICT_REVERSAL_LABEL_FAMILY,
    ICTTradeType.CONTINUATION.value: ICT_CONTINUATION_LABEL_FAMILY,
    ICTTradeType.META.value: ICT_META_LABEL_FAMILY,
}
ICT_LABEL_FAMILY_TO_TRADE_TYPE = {
    label_family: trade_type
    for trade_type, label_family in ICT_TRADE_TYPE_TO_LABEL_FAMILY.items()
}
ICT_TRADE_TYPE_TO_SETUP_FAMILIES = {
    ICTTradeType.REVERSAL.value: (ICTSetupFamily.REVERSAL.value,),
    ICTTradeType.CONTINUATION.value: (ICTSetupFamily.CONTINUATION.value,),
    ICTTradeType.META.value: (
        ICTSetupFamily.REVERSAL.value,
        ICTSetupFamily.CONTINUATION.value,
    ),
}
ICT_SETUP_FAMILY_TO_TRADE_TYPE = {
    ICTSetupFamily.REVERSAL.value: ICTTradeType.REVERSAL.value,
    ICTSetupFamily.CONTINUATION.value: ICTTradeType.CONTINUATION.value,
}
ICT_SETUP_FAMILY_TO_LABEL_FAMILY = {
    setup_family: ICT_TRADE_TYPE_TO_LABEL_FAMILY[trade_type]
    for setup_family, trade_type in ICT_SETUP_FAMILY_TO_TRADE_TYPE.items()
}
ICT_LABEL_FAMILY_TO_SETUP_FAMILY = {
    label_family: setup_family
    for setup_family, label_family in ICT_SETUP_FAMILY_TO_LABEL_FAMILY.items()
}
ICT_META_COMPONENT_LABEL_FAMILIES = (
    ICT_REVERSAL_LABEL_FAMILY,
    ICT_CONTINUATION_LABEL_FAMILY,
)
ICT_SETUP_TYPE_TO_FAMILY = {
    ICTSetupType.SWEEP_RECLAIM.value: ICTSetupFamily.REVERSAL.value,
    ICTSetupType.SWEEP_DISPLACEMENT_FVG.value: ICTSetupFamily.REVERSAL.value,
    ICTSetupType.OB_RETEST_AFTER_MSS.value: ICTSetupFamily.REVERSAL.value,
    ICTSetupType.IFVG_REVERSAL.value: ICTSetupFamily.REVERSAL.value,
    ICTSetupType.SESSION_OPEN_MANIPULATION_PRE_IB.value: ICTSetupFamily.REVERSAL.value,
    ICTSetupType.SESSION_OPEN_MANIPULATION_POST_IB.value: ICTSetupFamily.REVERSAL.value,
    ICTSetupType.PREMIUM_DISCOUNT_CONTINUATION.value: ICTSetupFamily.CONTINUATION.value,
    ICTSetupType.DISPLACEMENT_CONTINUATION_AFTER_RAID.value: ICTSetupFamily.CONTINUATION.value,
}


def normalize_ict_setup_type(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "" if text in {"", "none", "nan"} else text


def normalize_ict_setup_family(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "" if text in {"", "none", "nan"} else text


def normalize_ict_trade_type(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "" if text in {"", "none", "nan"} else text


def normalize_ict_label_family(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "" if text in {"", "none", "nan"} else text


def infer_ict_setup_family(setup_type: object) -> str | None:
    normalized = normalize_ict_setup_type(setup_type)
    if not normalized:
        return None
    return ICT_SETUP_TYPE_TO_FAMILY.get(normalized)


def infer_ict_label_family_from_setup_type(setup_type: object) -> str | None:
    setup_family = infer_ict_setup_family(setup_type)
    if setup_family is None:
        return None
    return ICT_SETUP_FAMILY_TO_LABEL_FAMILY.get(setup_family)


def infer_ict_trade_type_from_setup_type(setup_type: object) -> str | None:
    setup_family = infer_ict_setup_family(setup_type)
    if setup_family is None:
        return None
    return ICT_SETUP_FAMILY_TO_TRADE_TYPE.get(setup_family)


def infer_ict_trade_type_from_label_family(label_family: object) -> str | None:
    normalized = normalize_ict_label_family(label_family)
    if not normalized:
        return None
    return ICT_LABEL_FAMILY_TO_TRADE_TYPE.get(normalized)


def get_ict_taxonomy_snapshot() -> dict[str, Any]:
    return {
        "setup_types": [
            setup_type.value
            for setup_type in ICTSetupType
            if setup_type is not ICTSetupType.NONE
        ],
        "setup_families": list(ICT_SETUP_FAMILY_TO_LABEL_FAMILY.keys()),
        "trade_types": list(ICT_TRADE_TYPE_TO_LABEL_FAMILY.keys()),
        "label_families": [
            ICT_REVERSAL_LABEL_FAMILY,
            ICT_CONTINUATION_LABEL_FAMILY,
            ICT_META_LABEL_FAMILY,
        ],
        "meta_label_family": ICT_META_LABEL_FAMILY,
        "meta_component_trade_types": [
            ICTTradeType.REVERSAL.value,
            ICTTradeType.CONTINUATION.value,
        ],
        "meta_component_label_families": list(ICT_META_COMPONENT_LABEL_FAMILIES),
        "setup_type_to_setup_family": dict(ICT_SETUP_TYPE_TO_FAMILY),
        "setup_family_to_trade_type": dict(ICT_SETUP_FAMILY_TO_TRADE_TYPE),
        "setup_family_to_label_family": dict(ICT_SETUP_FAMILY_TO_LABEL_FAMILY),
        "trade_type_to_setup_families": {
            trade_type: list(setup_families)
            for trade_type, setup_families in ICT_TRADE_TYPE_TO_SETUP_FAMILIES.items()
        },
        "trade_type_to_label_family": dict(ICT_TRADE_TYPE_TO_LABEL_FAMILY),
    }


__all__ = [
    "ICT_CONTINUATION_LABEL_FAMILY",
    "ICT_LABEL_FAMILY_TO_SETUP_FAMILY",
    "ICT_LABEL_FAMILY_TO_TRADE_TYPE",
    "ICT_META_COMPONENT_LABEL_FAMILIES",
    "ICT_META_LABEL_FAMILY",
    "ICT_REVERSAL_LABEL_FAMILY",
    "ICT_SETUP_FAMILY_TO_TRADE_TYPE",
    "ICT_SETUP_FAMILY_TO_LABEL_FAMILY",
    "ICT_SETUP_TYPE_TO_FAMILY",
    "ICT_TRADE_TYPE_TO_LABEL_FAMILY",
    "ICT_TRADE_TYPE_TO_SETUP_FAMILIES",
    "ICTTradeType",
    "get_ict_taxonomy_snapshot",
    "infer_ict_label_family_from_setup_type",
    "infer_ict_setup_family",
    "infer_ict_trade_type_from_label_family",
    "infer_ict_trade_type_from_setup_type",
    "normalize_ict_label_family",
    "normalize_ict_setup_family",
    "normalize_ict_trade_type",
    "normalize_ict_setup_type",
]

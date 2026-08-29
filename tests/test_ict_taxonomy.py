from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.taxonomy import (  # noqa: E402
    ICTTradeType,
    ICT_LABEL_FAMILY_TO_TRADE_TYPE,
    ICT_META_LABEL_FAMILY,
    ICT_TRADE_TYPE_TO_LABEL_FAMILY,
    get_ict_taxonomy_snapshot,
    infer_ict_setup_family,
    infer_ict_trade_type_from_label_family,
    infer_ict_trade_type_from_setup_type,
    normalize_ict_trade_type,
)


def test_ict_taxonomy_snapshot_separates_setup_families_from_trade_types() -> None:
    snapshot = get_ict_taxonomy_snapshot()

    assert snapshot["setup_families"] == ["reversal", "continuation"]
    assert snapshot["trade_types"] == ["reversal", "continuation", "meta"]
    assert snapshot["setup_family_to_trade_type"] == {
        "reversal": "reversal",
        "continuation": "continuation",
    }
    assert snapshot["trade_type_to_setup_families"] == {
        "reversal": ["reversal"],
        "continuation": ["continuation"],
        "meta": ["reversal", "continuation"],
    }
    assert snapshot["trade_type_to_label_family"] == {
        "reversal": "ict_reversal",
        "continuation": "ict_continuation",
        "meta": "ict_meta",
    }


def test_ict_trade_type_inference_maps_setup_and_label_layers() -> None:
    assert infer_ict_setup_family("session_open_manipulation_post_ib") == "reversal"
    assert infer_ict_trade_type_from_setup_type("session_open_manipulation_post_ib") == "reversal"
    assert infer_ict_trade_type_from_setup_type("displacement_continuation_after_raid") == "continuation"
    assert infer_ict_trade_type_from_label_family(" ICT_META ") == ICTTradeType.META.value
    assert normalize_ict_trade_type(" Meta ") == ICTTradeType.META.value
    assert ICT_LABEL_FAMILY_TO_TRADE_TYPE[ICT_META_LABEL_FAMILY] == ICTTradeType.META.value
    assert ICT_TRADE_TYPE_TO_LABEL_FAMILY[ICTTradeType.REVERSAL.value] == "ict_reversal"

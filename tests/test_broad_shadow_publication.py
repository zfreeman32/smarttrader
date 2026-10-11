from __future__ import annotations

import json

import pytest

from pruned_archive import is_pruned
from scripts import publish_broad_full_session_shadow as publication


def test_selected_order_matches_v0_core_order() -> None:
    selection = json.loads(publication.SELECTION_V0.read_text(encoding="utf-8"))
    actual = tuple(
        selection["order"]["minimum_core_order"]
        + selection["order"]["expanded_core_if_latency_permits_order"]
    )
    assert actual == publication.SELECTED_IDS


def test_publication_refuses_missing_or_unreviewed_selected_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(publication.MODEL_ROOTS, 21, tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="ID 21"):
        publication.candidate_dir(21)


def test_published_freeze_rebuilds_from_exact_bytes() -> None:
    if not publication.SELECTION_V1.exists():
        pytest.skip("Publication is created only after all 12 replacement reviews pass")
    try:
        audit = publication.verify()
    except FileNotFoundError as exc:
        if is_pruned(exc.filename):
            pytest.skip(f"freeze input pruned on 2026-10-11 (docs/audits/data_prune_20261011.csv): {exc.filename}")
        raise
    assert audit["replacement_count"] == 12
    assert audit["all_exact_threshold_decisions_verified"] is True
    assert audit["all_full_policies_not_defined"] is True
    assert audit["scored_collection_eligible"] is False

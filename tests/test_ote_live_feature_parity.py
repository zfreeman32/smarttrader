from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.parity_replay import replay_feature_parity

REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_V2_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "long_ote_tcn_v2_candidate" / "live_runtime_manifest.json"
SHORT_V2_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "short_ote_tcn_v2_candidate" / "live_runtime_manifest.json"
LONG_LSTM_V2_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "long_ote_lstm_v2_candidate" / "live_runtime_manifest.json"
)


def _load_manifest(path: Path) -> LiveRuntimeManifest:
    return LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _subset_manifest(
    manifest: LiveRuntimeManifest,
    selected_feature_names: list[str],
) -> LiveRuntimeManifest:
    feature_manifest = manifest.feature_manifest.model_copy(
        update={
            "selected_feature_names": list(selected_feature_names),
            "selected_feature_count": len(selected_feature_names),
            "strategy_feature_count": sum(name.startswith("strategy__") for name in selected_feature_names),
        }
    )
    return manifest.model_copy(update={"feature_manifest": feature_manifest})


def test_replay_feature_parity_matches_reference_for_single_manifest_subset() -> None:
    manifest = _subset_manifest(
        _load_manifest(LONG_V2_MANIFEST_PATH),
        [
            "strategy__smi_signals__smi",
            "bb_width_20",
            "close_vs_ema_8_atr_lag_1",
        ],
    )

    report = replay_feature_parity(
        [manifest],
        evaluation_bars=4,
        rolling_window_bars=300,
    )

    assert report.rows_evaluated == 4
    assert report.missing_reference_rows == 0
    assert report.mismatched_cells == 0
    assert report.matched_cells == report.compared_cells == 4 * 3
    assert report.matched_cell_ratio == 1.0
    assert report.sample_mismatches == ()


def test_replay_feature_parity_matches_reference_for_multi_manifest_union_subset() -> None:
    long_manifest = _subset_manifest(
        _load_manifest(LONG_V2_MANIFEST_PATH),
        [
            "strategy__smi_signals__smi",
            "bb_width_20",
        ],
    )
    short_manifest = _subset_manifest(
        _load_manifest(SHORT_V2_MANIFEST_PATH),
        [
            "strategy__smi_signals__smi",
            "price_position_50_zscore_100",
            "atr_ratio_14_50_roll_mean_100",
        ],
    )

    report = replay_feature_parity(
        [long_manifest, short_manifest],
        evaluation_bars=4,
        rolling_window_bars=300,
    )

    assert report.rows_evaluated == 4
    assert report.missing_reference_rows == 0
    assert report.mismatched_cells == 0
    assert report.matched_cells == report.compared_cells == 4 * 4
    assert report.matched_cell_ratio == 1.0
    assert set(report.selected_feature_names) == {
        "strategy__smi_signals__smi",
        "bb_width_20",
        "price_position_50_zscore_100",
        "atr_ratio_14_50_roll_mean_100",
    }


def test_replay_feature_parity_matches_reference_for_cumulative_strategy_subset_over_multiple_bars() -> None:
    manifest = _subset_manifest(
        _load_manifest(SHORT_V2_MANIFEST_PATH),
        [
            "strategy__pvi_signals__pvi",
        ],
    )

    report = replay_feature_parity(
        [manifest],
        evaluation_bars=6,
        rolling_window_bars=300,
    )

    assert report.rows_evaluated == 6
    assert report.missing_reference_rows == 0
    assert report.mismatched_cells == 0
    assert report.matched_cells == report.compared_cells == 6
    assert report.matched_cell_ratio == 1.0

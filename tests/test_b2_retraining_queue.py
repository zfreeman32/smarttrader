"""Guards for extending the B2 queue without losing weighting or validation isolation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_b2_validation_views as prepare
from scripts import run_b2_retraining as runner
from scripts.review_b2_first_pass import contains_saved
from model_training.ote_training import ote_xgboost_pipeline as t


def test_shared_meta_union_selection_uses_earliest_recipe():
    jobs = [(21, Path("long_ote"), Path("meta"), {"cv_initial_train_rows": 180000}),
            (23, Path("long_ote"), Path("union"), {"cv_initial_train_rows": 200000})]
    assert prepare.selection_boundaries(jobs) == prepare.selection_boundaries(list(reversed(jobs))) == {"long_ote": 180000}
    assert len(runner.GROUPS["all"]) == len(set(runner.PRIORITY_IDS + runner.GROUPS["later"])) == 27


def test_recency_retains_half_life_floor_and_unweighted_test(tmp_path, monkeypatch):
    target = "long_frvp_reversal"
    prepared = tmp_path / "new"
    (prepared / target).mkdir(parents=True)
    timestamps = pd.Series(pd.to_datetime(["2016-01-01", "2024-01-02", "2026-01-01"], utc=True))
    dataset = tmp_path / "source.parquet"
    pd.DataFrame({"datetime": timestamps}).to_parquet(dataset)
    frame = pd.DataFrame(dict(source_row_idx=[0, 1, 2], target=[1, 0, 1], sample_weight=[.8, .4, 1.], feature=[3., 2., 1.]))
    for split in ("train", "val", "test"):
        frame.to_csv(prepared / target / f"{split}.csv", index=False)
    test_bytes = (prepared / target / "test.csv").read_bytes()
    (prepared / "summary.json").write_text("{}")
    weighting = dict(half_life_days=730, floor=.20, latest_ts_utc=timestamps.max().isoformat())
    summary = {"recency_weighting": weighting}
    contract = prepare.preserve_recency(summary, dataset, prepared, target)
    for split in ("train", "val"):
        actual = pd.read_csv(prepared / target / f"{split}.csv")
        np.testing.assert_allclose(actual.sample_weight, [.16, .20, 1.])
        pd.testing.assert_frame_equal(actual.drop(columns="sample_weight"), frame.drop(columns="sample_weight"))
    assert (prepared / target / "test.csv").read_bytes() == test_bytes
    old = tmp_path / "old"
    old.mkdir()
    for split in ("train", "val", "test"):
        (old / f"{split}.csv").write_bytes((prepared / target / f"{split}.csv").read_bytes())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    job = dict(recency_weighting=contract, old_prepared_dir="old")
    new_summary = json.loads((prepared / "summary.json").read_text())
    runner.validate_recency_weights(job, prepared / target, new_summary)
    # A second weighting pass or dropped weighting must fail preservation checks.
    for bad_weights in ([.032, .10, 1.], [.8, .4, 1.]):
        frame.assign(sample_weight=bad_weights).to_csv(prepared / target / "train.csv", index=False)
        with pytest.raises(AssertionError):
            runner.validate_recency_weights(job, prepared / target, new_summary)
    with pytest.raises(ValueError, match="requires"):
        prepare.preserve_recency({}, dataset, prepared, target)
    with pytest.raises(ValueError, match="Unexpected recency"):
        prepare.preserve_recency({"recency_weighting": dict(weighting, half_life_days=365)}, dataset, prepared, target)


def test_preflight_rejects_later_positive_shortfalls_and_selection_overlap():
    rows = 2000
    data = t.PreparedTargetDataset("long_frvp_continuation_setup3", runner.ROOT / "unused", ["x"], ["x"],
        {"split_geometry": {"applied_embargo_bars": 1071},
         "selection_contract": {"mode": "frozen_development_prefix", "fit_rows": 400,
                                "fit_end_source_row_idx": 3990, "required_gap_bars": 1071}},
        np.zeros((1600, 1)), np.arange(1600) % 2, np.ones(1600),
        np.zeros((400, 1)), np.arange(400) % 2, np.ones(400),
        np.zeros((100, 1)), np.arange(100) % 2, np.ones(100),
        np.arange(1600) * 10, np.arange(1600, rows) * 10, np.arange(rows, rows + 100) * 10)
    config = t.OTETrainingConfig(backend="xgboost", window_min=8, window_max=16,
        cv_initial_train_rows=400, cv_val_rows=200, cv_step_rows=200, cv_min_folds=2,
        label_max_holding_bars=1071, min_train_positive_rows=10,
        min_val_positive_rows=10, min_val_true_events=5)
    checked = runner.check_job(data, config, 6)
    assert len(checked["windows"]) == 3
    data.dev_y[1500:] = 0
    with pytest.raises(ValueError, match="positive/event safeguards"):
        runner.check_job(data, config, 6)
    data.report["selection_contract"].update(fit_rows=600, fit_end_source_row_idx=5990)
    with pytest.raises(ValueError, match="overlaps"):
        runner.check_job(data, config, 6)


def test_historical_preflight_comparison_accepts_additions_but_rejects_changed_gaps():
    saved = {"windows": [{"window": 16, "final_holdout": {"source_gap": 1071}}]}
    added = {"windows": [{"window": 16, "final_holdout": {"source_gap": 1071}, "counts": {}}], "calibration": "none"}
    assert contains_saved(added, saved)
    added["windows"][0]["final_holdout"]["source_gap"] = 0
    assert not contains_saved(added, saved)


@pytest.mark.parametrize("model_id,key,saved,effective", [(10, "cv_val_rows", 300, 400),
                                                       (19, "cv_initial_train_rows", 320, 352)])
def test_geometry_adjustments_preserve_search_and_safeguards(model_id, key, saved, effective):
    from dataclasses import asdict
    config = t.OTETrainingConfig(**{key: saved})
    before = asdict(config)
    runner.apply_geometry_adjustment(config, model_id)
    assert asdict(config) == dict(before, **{key: effective})
    with pytest.raises(ValueError, match="stale"):
        runner.apply_geometry_adjustment(config, model_id)

# Model Testing

This directory contains the post-training evaluation stack for OTE models. The code here sits after `model_training/ote_training`: it takes cross-validation and test predictions, rejoins them to source rows, adds regime labels, searches operating policies, and runs walk-forward trade simulations.

## What Lives Here

- `ote_prediction_joiner.py`
  - Reattaches exported prediction rows to the original feature dataset.
  - Can use prepared split metadata and `source_row_idx` when available.
- `ote_regime_labeler.py`
  - Adds deterministic market-state labels such as `trend_regime`, `vol_regime`, `session_regime`, `stress_regime`, and `composite_regime`.
- `ote_regime_slices.py`
  - Builds per-regime evaluation tables, average precision summaries, bootstrap intervals, and winner tables.
- `ote_threshold_policy.py`
  - Searches score thresholds and regime-aware policy variants.
- `ote_abstain_policy.py`
  - Adds hard abstain filters such as stress/session filters, expected-move checks, probability quantiles, and cooldown logic.
- `ote_policy_backtest.py`
  - Runs walk-forward policy backtests with trade-level outputs, acceptance gates, and summary files.
- `ote_policy_metrics.py`
  - Shared reporting helpers for equity, drawdown, trade grouping, and fold aggregation.

## Typical Workflow

Most runs are orchestrated from `scripts/`, with this directory providing the reusable library code underneath.

1. Join predictions back to source rows.
2. Label each row with deterministic regime metadata.
3. Generate regime-slice reports.
4. Search thresholds and abstain policies.
5. Backtest selected policies in walk-forward mode.

Common entry points:

- `scripts/run_ote_regime_slice_report.py`
- `scripts/run_ote_threshold_policy_search.py`
- `scripts/run_ote_policy_backtest.py`

## Outputs

Reports are usually written under `model_testing/reports/`, including:

- regime slice runs under `reports/ote_regime_slices/...`
- threshold and abstain policy searches under `reports/ote_threshold_policies/...`
- walk-forward policy backtests under `reports/ote_policy_backtests/...`

The repo also includes checked-in summaries that reflect more recent comparison work, for example:

- `reports/MODEL_LEADERBOARDS_20260527.md`
- `reports/ote_regime_slices/.../SLICE_REPORT_SUMMARY.md`
- `reports/ote_threshold_policies/.../THRESHOLD_POLICY_SUMMARY.md`

## Relationship To Training

The OTE trainer now exports richer artifacts than the older docs assumed, including:

- `cv_fold_manifest.csv` and `cv_fold_manifest.json`
- `oof_predictions.csv`
- `test_predictions.csv`
- `training_summary.json`

Those outputs are the main inputs to the testing stack documented here.

## Recommended Starting Point

If you are tracing the current OTE workflow end to end:

1. Read [`../model_training/ote_training/README.md`](../model_training/ote_training/README.md).
2. Use the scripts in `scripts/` to produce regime reports and threshold policies.
3. Review [`POST_TRAINING_PRODUCTION_PLAN.md`](./POST_TRAINING_PRODUCTION_PLAN.md) for the longer-term deployment path.

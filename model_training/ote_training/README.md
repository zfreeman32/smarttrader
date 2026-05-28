# OTE Training Pipeline

This is the current training stack for prepared OTE datasets under `data/prepared/...`. It is the main maintained path for training OTE entry models in this repo.

## What It Trains

The primary entry point is `ote_xgboost_pipeline.py`. Despite the filename, it now supports multiple backends from the same prepared target folders:

- `xgboost`
- `torch` with `--model-type tcn`
- `torch` with `--model-type lstm`

The trainer is not limited to only `long_ote` and `short_ote`. If you omit `--targets`, it will auto-discover target directories under the prepared root.

## Expected Input Layout

Each prepared target folder should contain chronological train/validation/test splits plus metadata, for example:

```text
data/prepared/eurusd_5min_ote_full/
  summary.json
  long_ote/
    train.csv
    val.csv
    test.csv
    report.json
    report.txt
    features.json
    feature_importance.csv
  short_ote/
    ...
```

If `source_row_idx` is present in the split CSVs, the trainer carries it into the exported OOF and test predictions for later re-joins in `model_testing/`.

## Feature Ranking Behavior

The trainer loads ranked features from backend-aware files when available:

- XGBoost: `feature_importance_merged_xgboost.csv`
- Torch TCN: `feature_importance_merged_tcn.csv`
- Torch LSTM: `feature_importance_merged_lstm.csv`
- Fallbacks: `feature_importance_merged.csv`, then `feature_importance.csv`

This matters because the ranking files can now differ by backend instead of forcing every model family to share one merged ranking.

## Training Flow

At a high level, the pipeline:

1. Loads a prepared target folder and feature ranking metadata.
2. Drops obviously dangerous leakage-style feature names.
3. Builds either sparse lag windows or true sequence windows depending on backend.
4. Fits fold-safe scaling inside each training fold.
5. Runs purged walk-forward cross-validation.
6. Tunes model and window settings.
7. Optionally calibrates probabilities.
8. Selects operating thresholds with event-aware metrics.
9. Refits and exports OOF, test, and summary artifacts.

## Current Cross-Validation Interface

The old `--cv-splits` flag is deprecated and the script now errors if you pass it. Use explicit row-geometry controls instead:

- `--cv-initial-train-rows`
- `--cv-val-rows`
- `--cv-step-rows`
- `--cv-max-train-rows`
- `--cv-min-folds`

Current defaults in code are:

- initial train rows: `250000`
- validation rows: `100000`
- step rows: `100000`
- minimum folds: `2`

## Example Commands

Train the default XGBoost path:

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full `
  --output-root models/ote_full_xgb_v2 `
  --backend xgboost `
  --trials 20 `
  --cv-initial-train-rows 250000 `
  --cv-val-rows 100000 `
  --cv-step-rows 100000
```

Train a TCN sequence model:

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full `
  --output-root models/ote_full_tcn_v2 `
  --backend torch `
  --model-type tcn `
  --targets long_ote short_ote `
  --trials 12 `
  --window-size 24 `
  --epochs 18 `
  --batch-size 256 `
  --cv-initial-train-rows 250000 `
  --cv-val-rows 100000 `
  --cv-step-rows 100000
```

## Main Outputs

Each target writes a folder under the chosen `--output-root`, including:

- `model.json` for XGBoost or `model.pt` plus `best_checkpoint.pt` for torch models
- `scaler.joblib`
- optional `calibrator.joblib`
- `model_config.json`
- `optuna_trials.csv`
- `training_history.csv`
- `training_history.json`
- `window_feature_importance.csv`
- `cv_fold_manifest.csv`
- `cv_fold_manifest.json`
- `oof_predictions.csv`
- `test_predictions.csv`
- `training_summary.json`

`training_summary.json` is now one of the most important artifacts because it records fold geometry, threshold selection, calibration settings, and whether row-identity information was preserved.

## Testing And Post-Training Analysis

The trainer has targeted coverage in `tests/test_ote_xgboost_pipeline.py`.

Post-training evaluation now lives in [`../../model_testing/README.md`](../../model_testing/README.md), including:

- regime slice analysis
- threshold and abstain policy search
- walk-forward policy backtests

The current longer-term deployment notes are in [`../../model_testing/POST_TRAINING_PRODUCTION_PLAN.md`](../../model_testing/POST_TRAINING_PRODUCTION_PLAN.md).

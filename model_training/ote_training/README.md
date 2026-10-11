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

The current longer-term deployment notes are in [`../../docs/OTE_journal.md#post_training_production_plan`](../../docs/OTE_journal.md#post_training_production_plan).

<a id="full_pipeline_readme"></a>

## Full OTE Pipeline

*Merged 2026-10-09 from `model_training/ote_training/FULL_PIPELINE_README.md`.*

This runbook describes the current end-to-end OTE workflow from raw EURUSD data through prepared datasets, training, and post-training evaluation.

### Pipeline Shape

```text
Raw OHLCV
  -> OTE labels
  -> feature dataset
  -> prepared target folders
  -> OTE model training
  -> regime / policy / backtest evaluation
```

### 1. Environment

Dependencies are in `requirements.txt`. A typical setup is:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install xgboost
```

Optional:

```powershell
pip install torch
pip install pytest
```

Use `torch` only if you plan to run the sequence backends.

### 2. Generate Labels

The generic entry point is `data/labeling/labeling_engine.py`. For a basic full-dataset run:

```powershell
python data/labeling/labeling_engine.py `
  --input data/currency_data/EURUSD_5min.csv `
  --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv
```

There are also strategy-specific labeling engines in `data/labeling/` such as:

- `ote_breakout_labeling_engine.py`
- `ote_continuation_pullback_labeling_engine.py`
- `reversal_labeling_engine.py`

Use those only when you intentionally want a different labeling family.

### 3. Build Features

The current feature CLI lives in `features/cli.py`.

Recommended build:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --output data/features/eurusd_5min_ote_full.csv `
  --recipe features/recipes/ote_extended.json
```

Smaller baseline build:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --output data/features/eurusd_5min_ote_full_base.csv `
  --recipe features/recipes/ote_base.json
```

### 4. Preprocess Into Prepared Targets

Prepared target folders are created from the feature dataset:

```powershell
python -m features.cli preprocess `
  data/features/eurusd_5min_ote_full.csv `
  --output-dir data/prepared/eurusd_5min_ote_full_v1 `
  --scaler none `
  --corr-threshold 0.98 `
  --similarity-threshold 0.995 `
  --max-analysis-rows 10000
```

`--scaler none` is still the recommended setting here because the OTE trainer fits fold-safe scaling internally.

Typical prepared outputs include target directories such as:

- `long_ote/`
- `short_ote/`
- sometimes additional target folders if those labels exist in the feature dataset

Each target folder contains `train.csv`, `val.csv`, `test.csv`, `report.json`, `report.txt`, `features.json`, and feature-ranking files.

### 5. Train OTE Models

#### Default XGBoost pass

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_xgb_v2 `
  --backend xgboost `
  --targets long_ote short_ote `
  --trials 40 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 128 `
  --window-min 8 `
  --window-max 40 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 250000 `
  --cv-val-rows 100000 `
  --cv-step-rows 100000 `
  --seed 42
```

#### Torch sequence pass

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_tcn_v2 `
  --backend torch `
  --model-type tcn `
  --targets long_ote short_ote `
  --trials 20 `
  --window-size 24 `
  --epochs 18 `
  --batch-size 256 `
  --calibration-method platt `
  --cv-initial-train-rows 250000 `
  --cv-val-rows 100000 `
  --cv-step-rows 100000 `
  --seed 42
```

Important change from the older docs:

- `--cv-splits` is deprecated and should not be used anymore.
- The trainer now expects explicit CV row geometry instead.

### 6. Review Training Artifacts

Modern training runs write more than just a model file. Per target, expect artifacts such as:

- `cv_fold_manifest.csv`
- `cv_fold_manifest.json`
- `oof_predictions.csv`
- `test_predictions.csv`
- `training_summary.json`
- `window_feature_importance.csv`
- backend-specific model checkpoints

Those files are the bridge into the testing stack.

### 7. Post-Training Evaluation

Use the `model_testing` workflow after training rather than stopping at validation metrics.

Main orchestration scripts:

- `scripts/run_ote_regime_slice_report.py`
- `scripts/run_ote_threshold_policy_search.py`
- `scripts/run_ote_policy_backtest.py`

That stack produces regime comparisons, threshold policies, abstain filters, and walk-forward trade simulations under `model_testing/reports/`.

### 8. Smoke Testing

Before a longer run:

```powershell
pytest tests/test_ote_xgboost_pipeline.py -q
```

You can also point the trainer at the smaller sample prepared dataset for a quick sanity check:

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_2000_v2 `
  --output-root models/ote_smoke_xgb `
  --backend xgboost `
  --trials 2 `
  --cv-initial-train-rows 800 `
  --cv-val-rows 300 `
  --cv-step-rows 300
```

### 9. Where To Read Next

- [`README.md`](./README.md) for the trainer-specific details
- [`../../docs/OTE_journal.md#ote_training_workflow_report`](./../../docs/OTE_journal.md#ote_training_workflow_report) for the methodology write-up
- [`../../model_testing/README.md`](../../model_testing/README.md) for the post-training stack

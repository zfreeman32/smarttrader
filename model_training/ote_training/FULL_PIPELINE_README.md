# Full OTE Pipeline

This runbook describes the current end-to-end OTE workflow from raw EURUSD data through prepared datasets, training, and post-training evaluation.

## Pipeline Shape

```text
Raw OHLCV
  -> OTE labels
  -> feature dataset
  -> prepared target folders
  -> OTE model training
  -> regime / policy / backtest evaluation
```

## 1. Environment

The repo still includes `ote_requirements.txt` for the OTE workflow. A typical setup is:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ote_requirements.txt
pip install xgboost
```

Optional:

```powershell
pip install torch
pip install pytest
```

Use `torch` only if you plan to run the sequence backends.

## 2. Generate Labels

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

## 3. Build Features

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

## 4. Preprocess Into Prepared Targets

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

## 5. Train OTE Models

### Default XGBoost pass

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

### Torch sequence pass

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

## 6. Review Training Artifacts

Modern training runs write more than just a model file. Per target, expect artifacts such as:

- `cv_fold_manifest.csv`
- `cv_fold_manifest.json`
- `oof_predictions.csv`
- `test_predictions.csv`
- `training_summary.json`
- `window_feature_importance.csv`
- backend-specific model checkpoints

Those files are the bridge into the testing stack.

## 7. Post-Training Evaluation

Use the `model_testing` workflow after training rather than stopping at validation metrics.

Main orchestration scripts:

- `scripts/run_ote_regime_slice_report.py`
- `scripts/run_ote_threshold_policy_search.py`
- `scripts/run_ote_policy_backtest.py`

That stack produces regime comparisons, threshold policies, abstain filters, and walk-forward trade simulations under `model_testing/reports/`.

## 8. Smoke Testing

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

## 9. Where To Read Next

- [`README.md`](./README.md) for the trainer-specific details
- [`OTE_TRAINING_WORKFLOW_REPORT.md`](./OTE_TRAINING_WORKFLOW_REPORT.md) for the methodology write-up
- [`../../model_testing/README.md`](../../model_testing/README.md) for the post-training stack

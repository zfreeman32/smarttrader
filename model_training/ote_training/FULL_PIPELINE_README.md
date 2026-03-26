# Full OTE Pipeline README

## Purpose

This runbook explains how to execute the current OTE workflow on a full EUR/USD 5-minute dataset, starting from raw OHLCV data and ending with trained OTE models and exported artifacts.

The pipeline order is:

```text
Raw 5m OHLCV
  -> OTE labels
  -> engineered feature dataset
  -> target-specific prepared train/val/test folders
  -> XGBoost or PyTorch OTE models
```

## 1. What the Current Code Requires

The implemented OTE training path currently needs one primary raw input:

- a EUR/USD 5-minute OHLCV CSV

The labeling and feature code will derive 30-minute and 1-hour structure internally by resampling the 5-minute series. You do not need separate 30-minute or 1-hour raw files for the current code path.

Accepted timestamp formats include:

- `Date` + `Time`
- `timestamp`
- `datetime`

Accepted price columns include common aliases for:

- `open`
- `high`
- `low`
- `close`
- `volume`

## 2. Recommended Directory Convention

You can place the raw full dataset anywhere and pass `--input`, but the repository defaults assume:

```text
data/currency_data/EURUSD_5min.csv
```

Recommended output layout for a full run:

```text
data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv
data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv
data/features/eurusd_5min_ote_full.csv
data/prepared/eurusd_5min_ote_full_v1/
models/ote_full_xgb/
models/ote_full_torch/
```

Version your prepared-data and model directories explicitly. That makes it much easier to compare labeling or feature changes later.

## 3. Environment Setup

### PowerShell example

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ote_requirements.txt
pip install xgboost
```

Optional packages:

```powershell
pip install torch
pip install pytest
```

Optional strategy-feature dependencies if you plan to use `strategy_signals` or `--all-strategies`:

```powershell
pip install ta
pip install TA-Lib
```

### Notes

- `xgboost` is required for the default training backend.
- `torch` is required only if you want the `torch` backend (`tcn` or `lstm`).
- The repository already contains `ote_requirements.txt` for the OTE workflow, but `xgboost` and `torch` are intentionally left commented there and should be installed explicitly.

## 4. Step 1: Generate OTE Labels from Raw 5-Minute Data

Basic full-dataset labeling run:

```powershell
python data/labeling/labeling_engine.py `
  --input data/currency_data/EURUSD_5min.csv `
  --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv
```

If you also want a diagnostic chart:

```powershell
python data/labeling/labeling_engine.py `
  --input data/currency_data/EURUSD_5min.csv `
  --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv `
  --plot `
  --plot-output data/labeling/plots/eurusd_5min_ote_full.png
```

What this step produces:

- bar-level zone labels
- precise entry labels
- quality scores
- exclusion masks
- safe-negative masks
- higher-timeframe confluence features
- sample weights
- swing-level audit metadata

## 5. Optional Review Layer

If you want to review or override labels before feature generation:

Launch the manual swing annotation app:

```powershell
python data/labeling/manual_labeling/manual_data_labeler_v2.py
```

Launch the OTE review app:

```powershell
python data/labeling/ote_label_review_app.py
```

If you use a reviewed export instead of the raw auto-labeled file, point the feature builder at the reviewed CSV instead.

## 6. Step 2: Build the Feature Dataset

### Recommended baseline build

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --output data/features/eurusd_5min_ote_full.csv `
  --recipe features/recipes/ote_extended.json
```

This command writes:

- `data/features/eurusd_5min_ote_full.csv`
- `data/features/eurusd_5min_ote_full.metadata.json`

### Lighter build

If you want a smaller feature table first:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --output data/features/eurusd_5min_ote_full_base.csv `
  --recipe features/recipes/ote_base.json
```

### Build with selected strategy-module features

Only do this if you specifically want strategy-derived features and have the extra dependencies available.

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --output data/features/eurusd_5min_ote_full_with_strategies.csv `
  --recipe features/recipes/ote_extended.json `
  --strategy acc_dist_strat `
  --strategy adx_breakouts_signals `
  --skip-strategy-errors
```

Notes:

- The default OTE recipes do not require strategy features.
- `--all-strategies` is possible but usually too broad for a first full-dataset run.

## 7. Step 3: Preprocess the Feature Dataset into Prepared Targets

Recommended preprocessing command:

```powershell
python -m features.cli preprocess `
  data/features/eurusd_5min_ote_full.csv `
  --output-dir data/prepared/eurusd_5min_ote_full_v1 `
  --scaler none `
  --corr-threshold 0.98 `
  --similarity-threshold 0.995 `
  --max-analysis-rows 10000
```

Why `--scaler none` is recommended here:

- the OTE trainer already fits fold-safe scalers inside each training fold
- leaving preprocessing unscaled avoids double scaling

Prepared target folders will be created automatically for any supported label columns found in the dataset, typically:

- `long_ote`
- `short_ote`
- `long_entry`
- `short_entry`

Each target directory contains:

- `train.csv`
- `val.csv`
- `test.csv`
- `report.json`
- `report.txt`
- `features.json`
- `feature_importance.csv`

## 8. Step 4: Train the Default XGBoost OTE Models

### Train both directional OTE models

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_xgb `
  --backend xgboost `
  --targets long_ote short_ote `
  --trials 40 `
  --cv-splits 3 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 128 `
  --window-min 8 `
  --window-max 40 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --seed 42
```

### Train only one target

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_xgb_long_only `
  --backend xgboost `
  --targets long_ote `
  --trials 40 `
  --cv-splits 3
```

### Recommended first full run settings

For a first real full-dataset training pass, these are reasonable defaults:

- backend: `xgboost`
- targets: `long_ote short_ote`
- trials: `30` to `60`
- cv splits: `3`
- calibration: `platt`

Increase trials only after the full pipeline is stable on the full dataset.

## 9. Step 5: Train the Sequence Backend (Optional)

### LSTM example

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_lstm `
  --backend torch `
  --model-type lstm `
  --targets long_ote short_ote `
  --trials 20 `
  --cv-splits 3 `
  --batch-size 256 `
  --epochs 18 `
  --hidden-size 64 `
  --num-layers 2 `
  --dropout 0.20 `
  --learning-rate 0.001 `
  --window-size 24 `
  --calibration-method platt `
  --seed 42
```

### TCN example

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_tcn `
  --backend torch `
  --model-type tcn `
  --targets long_ote short_ote `
  --trials 20 `
  --cv-splits 3 `
  --batch-size 256 `
  --epochs 18 `
  --hidden-size 64 `
  --num-layers 4 `
  --dropout 0.20 `
  --learning-rate 0.001 `
  --window-size 24 `
  --calibration-method platt `
  --seed 42
```

## 10. What Training Writes

For each target, the trainer writes a model folder under your chosen `--output-root`.

Expected outputs include:

- model checkpoint
- scaler
- optional calibrator
- `optuna_trials.csv`
- `training_history.csv`
- `training_history.json`
- `model_config.json`
- `window_feature_importance.csv`
- `test_predictions.csv`
- `training_summary.json`

## 11. Suggested End-to-End Command Set

If you want a single sequence of commands for a full XGBoost run on the full dataset, this is the cleanest starting point:

```powershell
python data/labeling/labeling_engine.py `
  --input data/currency_data/EURUSD_5min.csv `
  --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv

python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv `
  --output data/features/eurusd_5min_ote_full.csv `
  --recipe features/recipes/ote_extended.json

python -m features.cli preprocess `
  data/features/eurusd_5min_ote_full.csv `
  --output-dir data/prepared/eurusd_5min_ote_full_v1 `
  --scaler none

python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_full_v1 `
  --output-root models/ote_full_xgb `
  --backend xgboost `
  --targets long_ote short_ote `
  --trials 40 `
  --cv-splits 3 `
  --calibration-method platt `
  --seed 42
```

## 12. Full-Dataset Practical Advice

### Start with a reproducible directory version

Use explicit names such as:

- `eurusd_5min_ote_full_v1`
- `eurusd_5min_ote_full_v2`
- `ote_full_xgb_v1`

This prevents accidental overwrites and makes it easy to compare labeling or feature changes later.

### Keep the first full run simple

For the first full-dataset pass:

- use `ote_extended.json`
- keep preprocessing scaling at `none`
- train `xgboost` first
- skip strategy-module features
- skip the torch backend until the full XGBoost run is stable

### Do not expect timestamps inside the prepared split CSVs

Prepared split files are intentionally numeric-only. Temporal order is preserved by row order and documented in the target reports. If you need timestamp-level joins later, keep the original feature and labeled datasets.

### CUSUM is not the current sample filter

The labeling engine computes CUSUM events, but the current OTE labels are driven by structural ATR swing detection, outcome validation, and quality scoring. Do not assume the prepared samples are CUSUM-filtered events.

## 13. Smoke Testing and Validation

Before a long full-dataset run, you can smoke test the trainer with the included test suite:

```powershell
pytest tests/test_ote_xgboost_pipeline.py -q
```

You can also run a quick sample-data training pass by pointing the trainer at the included prepared sample dataset:

```powershell
python -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root data/prepared/eurusd_5min_ote_2000_v2 `
  --output-root models/ote_smoke_xgb `
  --backend xgboost `
  --targets long_ote short_ote `
  --trials 2 `
  --cv-splits 2
```

## 14. Where to Read the Methodology

The research-style description of the full workflow lives here:

- `model_training/ote_training/OTE_TRAINING_WORKFLOW_REPORT.md`

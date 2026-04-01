# OTE Training Pipeline

This directory contains the current training pipeline for EURUSD Optimal Trade Entry detection on the prepared datasets under `data/prepared/...`.

The pipeline is built for the two directional zone-label targets:

- `long_ote`
- `short_ote`

It is designed to work with the newer prepared-data structure rather than the older buy/sell classification scripts.

## What This Pipeline Does

The trainer in [ote_xgboost_pipeline.py](/C:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/model_training/ote_training/ote_xgboost_pipeline.py) loads one prepared target folder at a time and trains a direction-specific OTE model with:

- time-ordered train/validation/test data from `data/prepared/<dataset>/<target_name>/`
- ranked feature loading from `features.json` and `feature_importance_merged.csv` when present, otherwise `feature_importance.csv`
- leakage filtering for obviously future-looking columns
- backend-aware temporal representations:
  - causal sparse-window lag features for XGBoost
  - dense sequence windows for PyTorch TCN/LSTM models
- fold-safe robust scaling
- purged walk-forward cross-validation
- focal-loss training through a custom XGBoost objective
- hard-negative weighting near positive OTE zones
- progressive boosting phases
- probability calibration
- threshold selection using event-level metrics

The current default backend is XGBoost, but the same prepared target folders can now also feed PyTorch TCN/LSTM models without changing the preprocessing output format.

## Why We Built It This Way

OTE prediction is not a normal binary classification problem.

- Positive labels are rare.
- Labels are zone-based, not single-bar exact hits.
- Temporal leakage is easy to introduce.
- Near-miss negatives matter more than random negatives.
- Bar-level accuracy is not a useful optimization target.

The implementation therefore focuses on methods that fit the structure of the task:

- AUPRC instead of accuracy.
- Event-level thresholding instead of raw 0.5 cutoff assumptions.
- Purged temporal validation instead of random folds.
- Focal loss and hard-negative emphasis instead of naive class balancing.
- Sparse lag windows instead of flattening the entire history into one giant dense table.

## End-to-End Flow

### 1. Prepared Dataset Creation

Prepared datasets are created by [preprocessing.py](/C:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/features/preprocessing.py).

For each target, preprocessing:

- standardizes and sorts the source data by `datetime`
- resolves candidate feature columns from metadata or a fallback selector
- excludes known label/helper columns from the model feature set
- encodes non-numeric features
- removes duplicate and constant columns
- builds target-specific usable rows
- applies chronological train/validation/test splits
- writes per-target CSVs with:
  - selected feature columns
  - `target`
  - `sample_weight`
- writes companion metadata and reports:
  - `report.json`
  - `report.txt`
  - `features.json`
  - `feature_importance.csv`

### 2. Training Data Load

The OTE trainer loads:

- `train.csv`
- `val.csv`
- `test.csv`
- `report.json`
- `features.json`
- `feature_importance_merged.csv` when present, otherwise `feature_importance.csv`

It uses the feature ranking file to cap the number of loaded base features before window expansion. This keeps memory usage under control for larger datasets.

### 3. Leakage Filtering

The trainer drops obviously dangerous feature names before training, including patterns such as:

- `target_profit`
- `future`
- `lookahead`
- `mfe`
- `mae`
- `take_profit`
- `tp_hit`
- `sl_hit`
- `exit_signal`

This is a safety layer on top of the preprocessing stage.

### 4. Causal Windowing

The model does not consume one-row snapshots only.

The pipeline now supports two temporal views built from the same prepared CSVs:

- XGBoost path: a sparse lag view
- PyTorch path: true sequence windows shaped like `(samples, window, features)`

The sparse lag view still:

- choose a window size
- choose a number of lag anchors inside that window
- keep the current row plus selected historical lag rows
- optionally add delta features between the current row and older lag rows

The sequence path keeps the full causal window for TCN/LSTM backends.

### 5. Scaling

Scaling is fit only on the training fold and then applied to validation/test rows.

The scaler is:

- `RobustScaler`
- using configurable quantiles
- followed by value clipping

This reduces sensitivity to outliers and preserves time-series hygiene.

### 6. Cross-Validation

The trainer uses a purged walk-forward splitter:

- expanding training window
- forward-only validation windows
- purge gap between train and validation
- minimum train/validation row constraints

This is intended to reduce leakage from nearby bars and overlapping temporal context.

### 7. Imbalance Handling

OTE labels are highly imbalanced, so the trainer uses multiple mechanisms together:

- custom binary focal loss
- sample weights from preprocessing
- hard-negative upweighting near positive zones
- optional balanced downsampling for tuning

The goal is to focus the model on hard discriminations instead of learning to predict “not an entry” everywhere.

### 8. Progressive Training

Each backend is trained in phases:

1. Warmup on an early subset.
2. Main training on the full fold.
3. Fine-tuning with a reduced learning rate.

For XGBoost this is staged boosting. For PyTorch this is phased optimization with early stopping on validation AUPRC.

### 9. Calibration and Thresholding

Raw probabilities are optionally calibrated with:

- Platt scaling
- isotonic regression
- or no calibration

The operating threshold is not fixed at `0.5`.

Instead the trainer searches a threshold grid using event-level metrics that better match zone labels:

- event precision
- event recall
- event F1
- event F-beta with beta = 0.5

### 10. Final Refit and Export

After tuning, the trainer:

- reruns cross-validation with the best params
- fits a final model on the development set
- scores the held-out test split
- exports artifacts to `models/ote_xgboost/<target_name>/`

Saved outputs include:

- `model.json` for XGBoost or `model.pt` / `best_checkpoint.pt` for PyTorch
- `scaler.joblib`
- `calibrator.joblib` when calibration is enabled
- `model_config.json`
- `optuna_trials.csv`
- `training_history.csv`
- `training_history.json`
- `window_feature_importance.csv`
- `test_predictions.csv`
- `training_summary.json`

## Dataset Layout

Expected layout:

```text
data/prepared/eurusd_5min_ote_2000_v2/
  summary.json
  summary_report.txt
  long_ote/
    train.csv
    val.csv
    test.csv
    report.json
    report.txt
    features.json
    feature_importance.csv
  short_ote/
    train.csv
    val.csv
    test.csv
    report.json
    report.txt
    features.json
    feature_importance.csv
```

Each split CSV is a model-ready table:

- feature columns only
- `target`
- `sample_weight`

No raw datetime column is stored in the split CSVs.

## Current Limitations

### 1. Prepared split CSVs do not carry timestamps

This is intentional, but it means exported prediction files are row-index keyed unless you rejoin them to the original source dataset using split metadata and row order.

### 2. This sample dataset is small

The included `eurusd_5min_ote_2000_v2` dataset is useful for testing the pipeline, but real tuning quality will depend on the much larger dataset you mentioned.

## Running Training

Example:

```bash
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_2000_v2 ^
  --output-root models/ote_xgboost ^
  --backend xgboost ^
  --targets long_ote short_ote ^
  --trials 20 ^
  --cv-splits 3
```

Useful flags:

- `--backend`
- `--model-type`
- `--max-loaded-features`
- `--top-feature-min`
- `--top-feature-max`
- `--window-min`
- `--window-max`
- `--window-size`
- `--epochs`
- `--batch-size`
- `--event-tolerance-bars`
- `--event-cooldown-bars`
- `--calibration-method`

## Tests

Coverage for the new trainer lives in [test_ote_xgboost_pipeline.py](/C:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/tests/test_ote_xgboost_pipeline.py).

It validates:

- lag-step generation
- sparse lag feature alignment
- sequence window alignment
- event-level scoring for zone labels
- focal-loss gradient behavior
- a small cross-validation smoke path
- torch forward passes when PyTorch is installed
- a torch backend cross-validation smoke path when PyTorch is installed

Run:

```bash
pytest tests/test_ote_xgboost_pipeline.py -q
```

## Post-Training Roadmap

The concrete repo plan for moving the current OTE models toward regime-aware production deployment lives in:

- [POST_TRAINING_PRODUCTION_PLAN.md](/C:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/model_training/ote_training/POST_TRAINING_PRODUCTION_PLAN.md)

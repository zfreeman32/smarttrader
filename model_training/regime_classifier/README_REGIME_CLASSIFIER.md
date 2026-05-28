# Regime Classifier

This folder contains a standalone market-regime workflow built from two parts:

1. deterministic regime labeling from engineered market-state features
2. an LSTM classifier trained to predict those regime classes from rolling sequences

It is separate from the OTE post-training regime analysis in `model_testing/ote_regime_labeler.py`.

## Current Regime Classes

The labeler in `regime_classifier.py` produces seven classes:

- strong uptrend
- weak uptrend
- ranging
- weak downtrend
- strong downtrend
- high volatility
- low volatility

The thresholds are driven by ADX, directional indicators, ROC, and ATR percentile logic. The trainer exposes the key thresholds as CLI flags.

## Main Files

- `regime_classifier.py`
  - `RegimeFeatureEngineering` builds the technical feature set.
  - `RegimeLabeler` creates deterministic class labels.
  - `RegimeClassifierModel` defines the stacked LSTM classifier.
- `train_regime_classifier.py`
  - End-to-end training entry point.
  - Selects a 24-feature subset for model training.

## Training CLI

The training script currently supports:

- `--data_path`
- `--sample_size`
- `--adx_trend`
- `--adx_ranging`
- `--atr_high_pct`
- `--atr_low_pct`
- `--lookback`
- `--lstm_units`
- `--dropout`
- `--learning_rate`
- `--epochs`
- `--batch_size`
- `--train_ratio`
- `--val_ratio`
- `--use_class_weights`
- `--output_dir`
- `--model_name`
- `--plot`

Example:

```powershell
python model_training/regime_classifier/train_regime_classifier.py `
  --data_path data/currency_data/EURUSD_1min.csv `
  --output_dir model_training/regime_classifier/outputs `
  --model_name regime_classifier.keras `
  --epochs 100 `
  --lookback 100 `
  --use_class_weights
```

## Outputs

A training run writes:

- `<model_name>` such as `regime_classifier.keras`
- matching scaler file `<model_name without .keras>_scaler.npy`
- `regime_model_metadata.txt`
- optional plots when `--plot` is enabled

## Notes

- This workflow is self-contained and useful for standalone regime research.
- If you want the regime labels currently used in OTE evaluation, use the `model_testing` stack instead of assuming the two label systems are interchangeable.

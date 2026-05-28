# Regression Training

This folder contains the older multi-step regression pipeline for direct price forecasting. The current code is centered on forecasting future `Close` values from sliding windows of market features; the previous README references to `returns_1` and `returns_5` are no longer the main default behavior.

## Main Entry Point

- `regression_model_train.py`
  - Main training script and orchestration layer.
- `regression_model_build.py`
  - Model builder library used by the trainer.

## Current Default Configuration

The `ConfigManager` in `regression_model_train.py` currently defaults to:

- `data_file = "Close_dataset.csv"`
- `target_col = "Close"`
- `lookback_window = 240`
- `forecast_horizon = 15`

That means the baseline workflow is a 240-step input window predicting the next 15 `Close` values.

## Active Model Set

The current trainer actively uses these neural architectures:

- `LSTM`
- `GRU`
- `SimpleRNN`
- `Conv1D`
- `TCN`
- `ResNet`
- `MultiStream_Hybrid`
- `Transformer`
- `LSTM_CNN_Hybrid`
- `Conv1D_LSTM`
- `Conv1DPooling`

`regression_model_build.py` still contains additional experimental builders, but they are not all included in the trainer's active `MODEL_BUILDERS` list.

## Training Flow

The implemented pipeline does the following:

1. Load the source CSV and clean numeric features.
2. Build lagged and transformed inputs for time-series forecasting.
3. Frame the data into supervised sliding windows.
4. Run Keras Tuner Hyperband over the supported model families.
5. Re-evaluate top candidates with time-series cross-validation.
6. Perform progressive training and final model export.

The code emphasizes robust preprocessing and training stability over lightweight experimentation.

## Outputs

Typical artifacts include:

- trained models saved as `models/<model_name>_regressor.keras`
- training logs written to `regression_training_results.txt`
- plots and validation summaries produced during training

## Notes

- This is a standalone forecasting pipeline, not part of the newer prepared-data OTE classifier stack.
- If your current work is OTE-focused, start with [`../ote_training/README.md`](../ote_training/README.md) instead.

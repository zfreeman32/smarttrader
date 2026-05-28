# Multi-Timeframe Confluence

This folder contains a research model that scores multi-timeframe agreement and directional bias from 1-minute source data.

## What The Current Code Trains

The pipeline is regression-style with two continuous outputs:

- `confluence_score`
- `directional_bias`

Those labels are derived in `mtf_confluence_trainer.py` from forward returns and cross-horizon agreement, not from a discrete class label.

## Timeframes And Inputs

`MultiTimeframeDataPreparator` currently builds aligned inputs for:

- `1m`
- `5m`
- `15m`
- `1h`
- `4h`

It resamples the 1-minute dataframe into higher timeframes, selects up to 50 priority features, and creates per-timeframe rolling sequences before training.

## Model Shape

The current model combines:

- per-timeframe sequence encoders
- transformer-style blocks
- one head for `confluence_score`
- one head for `directional_bias`

The compiled loss is MSE on both outputs, with MAE/MSE metrics tracked per head.

## Training Entry Points

- `mtf_confluence_trainer.py`
  - Core data preparation and model definition.
- `train_confluence.py`
  - Full training and reporting pipeline.
- `mtf_confluence_analyzer.py`
  - Inference and decision helper layer.

## Important Current Caveat

`train_confluence.py` is not a flexible CLI in its current state. The `main()` function still uses hard-coded constants such as:

- `DATA_PATH`
- `OUTPUT_DIR`

Update those values in the script before running a new training pass.

## Outputs

The default training script writes under `./mtf_confluence_output/`, including:

- `models/confluence_model_best.keras`
- `models/confluence_scalers.pkl`
- training reports
- evaluation summaries
- generated plots

## Analyzer Thresholds

`mtf_confluence_analyzer.py` currently uses these practical cutoffs:

- `details['tradeable']` becomes true when confluence is at least `0.65` and absolute bias is at least `0.20`
- `should_trade()` defaults to `min_confluence=0.70` and `min_bias=0.30`

Those thresholds are a useful reminder that the model is intended as a trade filter, not just a forecasting exercise.

## Notes

- This folder is still valuable research code, but it is not wired into the newer prepared-data OTE training path.
- For the actively maintained OTE workflow, start with [`../ote_training/README.md`](../ote_training/README.md).

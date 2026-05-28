# Model Training

This directory contains the repo's offline model-building pipelines. The folders are not all at the same maturity level:

- `ote_training/` is the current OTE workflow and the main path for prepared-data training.
- `classification/` and `regression/` are older direct-from-CSV training stacks that still work but are no longer the center of the OTE pipeline.
- `pattern_validation/`, `regime_classifier/`, and `mtf_confluence/` are specialized research models that feed broader trade-selection ideas.

## Directory Map

- `classification/`
  - Binary and multiclass direction classifiers trained from filtered CSV datasets.
  - Includes long/short entry models plus direction-horizon variants.
- `regression/`
  - Multi-step price forecasting pipeline for future `Close` values.
  - Uses Keras Tuner, progressive training, and cross-validation over a model zoo.
- `ote_training/`
  - Current prepared-data OTE trainer.
  - Supports `xgboost` plus PyTorch sequence backends (`tcn`, `lstm`) from the same prepared target folders.
- `pattern_validation/`
  - Multi-input pattern-quality model for scoring ICT-style detections.
- `regime_classifier/`
  - Deterministic regime labeling plus an LSTM classifier trained on engineered market-state features.
- `mtf_confluence/`
  - Multi-timeframe sequence model that predicts directional bias and confluence strength.

## Shared Helpers

Several older training stacks reuse common utilities at this level:

- `model_layers.py`
- `model_training_utils.py`

These support the legacy classification and regression pipelines more than the newer prepared-data OTE workflow.

## Which Pipeline Is Current

If you are training the actively maintained OTE models, start in [`ote_training/README.md`](./ote_training/README.md). That pipeline assumes:

- features have already been built under `data/features/...`
- target-specific prepared splits already exist under `data/prepared/...`
- post-training evaluation happens in [`../model_testing/README.md`](../model_testing/README.md)

The other subdirectories remain useful references and still contain runnable code, but many of them reflect earlier experimentation paths rather than the current end-to-end OTE production flow.

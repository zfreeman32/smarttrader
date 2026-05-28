# Classification Training

This folder contains the older deep-learning classification pipelines that train directly from filtered CSV files. They are still runnable, but they are separate from the newer prepared-data OTE workflow in `model_training/ote_training/`.

## Current Scripts

- `buy_classification_model_train.py`
  - Trains a binary classifier for `long_signal`.
  - Default input file: `long_EURUSD_1min_filtered.csv`
  - Current default selected model list: `Conv1D_LSTM`
  - Saves models under `models/buy_models/`
- `sell_classification_model_train.py`
  - Trains a binary classifier for `short_signal`.
  - Default input file: `short_EURUSD_1min_filtered.csv`
  - Current default selected model list: `LSTM`, `GRU`, `Conv1D`, `Conv1D_LSTM`
  - Saves models under `models/sell_models/`
- `direction_classification_model_train.py`
  - Trains binary direction targets for multiple horizons.
  - Current targets: `direction_1`, `direction_5`, `direction_14`
  - Saves each target to its own directory such as `models/direction_1_models/`
- `direction_3class_5_classification.py`
  - Trains the three-class direction target `direction_3class_5`
  - Saves models under `models/direction_3class_5_models/`

## Shared Building Blocks

- `classification_model_build.py`
  - Builder library for the available architectures.
- `classification_utils.py`
  - Shared preprocessing, losses, metrics, and training helpers.

The builder file still exposes a larger model zoo than the scripts use by default, including:

- `LSTM`
- `GRU`
- `Conv1D`
- `Conv1D_LSTM`
- `BiLSTM_Attention`
- `Transformer`
- `MultiStream_Hybrid`
- `ResNet`
- `TCN`

Some legacy tree-model builders also exist, but the current training scripts focus on the neural architectures above.

## What The Pipelines Do

Across the different scripts, the workflow is broadly:

1. Load a filtered CSV dataset.
2. Build sliding windows over time-series features.
3. Scale features with robust preprocessing.
4. Tune and train one or more sequence architectures.
5. Save trained `.h5` models plus summary logs.

The direction trainer also includes distributed training support for larger multi-target runs.

## Outputs

Typical outputs include:

- trained models under `models/...`
- text summaries such as `buy_class_model_training_results.txt`
- target-specific model directories for the direction scripts

## Notes

- These scripts predate the prepared target-folder structure used by the current OTE stack.
- If you are training modern OTE entry models from `data/prepared/...`, use [`../ote_training/README.md`](../ote_training/README.md) instead.

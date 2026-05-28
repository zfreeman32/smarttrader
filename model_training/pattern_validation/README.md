# Pattern Validation

This folder contains a multi-input pattern-quality model for scoring ICT-style detections after the rule-based detector has already found a candidate setup.

## What The Current Code Does

The current implementation is a quality-score model, not just a plain binary classifier.

- `model_architecture.py`
  - Builds a four-input network with chart, volume, pattern-metadata, and indicator branches.
  - Final output is a single sigmoid quality score.
  - Default compile path uses `loss='mse'` with metrics `['mae', 'mse']`.
- `training_pipeline.py`
  - Loads `X_chart.npy`, `X_volume.npy`, `X_pattern.npy`, `X_indicators.npy`, and `y.npy`.
  - Splits chronologically into train/validation/test sets.
  - Supports progressive training with a warmup phase and a full-data fine-tune phase.
- `inference_pipeline.py`
  - Loads the trained model and returns `quality_score`, `tradeable`, and `confidence_level`.
  - Default `quality_threshold` is `0.75`.
- `main_pattern_validation.py`
  - Orchestrates `generate`, `train`, `deploy`, and `full` modes.

## Directory Layout

The orchestrator creates and uses a local working area under:

```text
pattern_validation_system/
  data/
  models/
  outputs/
  logs/
```

By default:

- generated training datasets live under `pattern_validation_system/data/`
- deployed models live under `pattern_validation_system/models/`
- training plots and evaluation files live under `pattern_validation_system/outputs/`

## Training Inputs

The trainer expects a saved dataset directory containing:

- `X_chart.npy`
- `X_volume.npy`
- `X_pattern.npy`
- `X_indicators.npy`
- `y.npy`

Those files are produced by the data-generation layer driven from detector outputs and price data.

## Training Outputs

Current training runs write artifacts such as:

- `best_model.keras`
- `evaluation_metrics.json`
- `training_history.png`
- `predictions.png`
- `error_distribution.png`

The training wrapper then copies the best model into the system `models/` directory for deployment.

## CLI Modes

Generate training data:

```powershell
python model_training/pattern_validation/main_pattern_validation.py `
  --mode generate `
  --ict-detections path/to/detections.pkl `
  --price-data path/to/price_data.csv `
  --data-name training_data_v1
```

Train a model:

```powershell
python model_training/pattern_validation/main_pattern_validation.py `
  --mode train `
  --data-name training_data_v1 `
  --model-name pattern_validator_v1
```

Deploy the validator:

```powershell
python model_training/pattern_validation/main_pattern_validation.py `
  --mode deploy `
  --model-name pattern_validator_v1
```

## Important Current Caveat

`--mode full` is not a true generate-train-deploy end-to-end wrapper right now. It still requires `--ict-detections` and `--price-data`, but the current code path jumps straight into training and deployment without calling `generate_training_data()` first.

Use `generate` explicitly before `train` if you need a fresh dataset build.

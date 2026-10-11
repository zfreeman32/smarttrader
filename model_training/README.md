# Model Training

Offline model training for every family (OTE/EURUSD, FRVP/ES, ICT/ES).

- `ote_training/`: the prepared-data trainer used by all current models. `ote_xgboost_pipeline.py` trains XGBoost and
  PyTorch sequence backends (`tcn`, `lstm`) with purged walk-forward CV, Optuna search, calibration and OOF-only
  threshold selection. See [`ote_training/README.md`](./ote_training/README.md) for inputs, commands and outputs.
- `source_history.py`: builds consecutive source-bar history so training tensors match runtime inference (added in the
  B2 source-history repair; see `docs/B2_model_repair.md`).

Post-training evaluation lives in [`../model_testing/README.md`](../model_testing/README.md). The B2 and v2 retraining
runners are in `scripts/` (`run_b2_retraining.py`, `run_b2_v2_source_history.py`, `run_b2_v2_r3.py`).

The older direct-from-CSV stacks (`classification/`, `regression/`, `pattern_validation/`, `regime_classifier/`,
`mtf_confluence/`, `model_layers.py`, `model_training_utils.py`) were removed on 2026-10-09; they are in git history and
on the `backup/pre-cleanup-20261009` branch.

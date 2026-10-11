# trade_bot

Intraday ML trading research and a live shadow-signal app for two streams:

- **ES (E-mini S&P 500, 5m):** FRVP (fixed-range volume profile) and ICT/SMC setup families, meta-labelled by XGBoost.
- **EURUSD (5m):** OTE (optimal trade entry) reversal / continuation / breakout models (XGBoost, TCN, LSTM).

Nothing here places broker orders. The live app records shadow predictions, setups and simulated outcomes.

**Where things stand** is in [`notes.txt`](notes.txt) (the active plan) and
[`docs/section4_progress.md`](docs/section4_progress.md) (the latest results). Every document is indexed in
[`docs/README.md`](docs/README.md).

## Layout

| Path | What it is |
|---|---|
| `data/` | Raw bars (`currency_data/`, `futures_data/`), the label engines (`data/labeling/`), and the EURUSD OTE feature/prepared datasets the live models were trained on. Large files are gitignored. |
| `features/` | Feature builder (`python -m features.cli build`): 16 feature families, the standalone strategy library under `features/strategies/`, recipes, and the causal feature contract v2 (`causal_contract.py`). |
| `frvp/`, `ict/` | ES strategy packages: sessions, profiles/structure, setup detectors, labeling and pipelines. |
| `preprocessing/` | Chronological splits, feature selection and attribution (`python -m preprocessing prepare`). |
| `model_training/` | The prepared-data trainer for all families (`ote_training/`) and source-history construction. |
| `model_testing/` | Policy/threshold search, walk-forward backtests, and the B1 outcome-contract engine. |
| `configs/` | Frozen, hash-locked research contracts: outcome contracts, scopes, B3/B4 freezes, route bindings, the section 4 parity protocol. Never edited in place; new versions get new files. |
| `models/` | Trained model bundles and registries (gitignored). See `models/README.md`. |
| `ote_live/` | The live app: IBKR/FMP ingestion, incremental features, model runners, policies, SQLite storage and the Dash dashboard. See `ote_live/README.md`. |
| `scripts/` | Pipeline runners (training, post-training evaluation, policy search/backtest), the B2/v2 retraining and B1-outcome research scripts, and the section 3/4 replay and parity harness. |
| `research/` | The 2026-09-11 ES live audit and the Anti-Filter Effect paper (`research/papers/anti_filter_effect/`). |
| `artifacts/` | Research outputs (gitignored): versioned feature tapes, prepared roots, source histories, sweeps. |
| `tests/` | pytest suite. |

## Common commands

All commands use the repo's interpreter, `ote_venv/Scripts/python.exe`. Dependencies are in `requirements.txt`.

```powershell
# Live app (dashboard + ES and OTE collectors; see the script's parameters for ICT/setup modes)
./app_start_cmd.ps1
./app_stop_cmd.ps1

# Feature build and preprocessing
ote_venv/Scripts/python.exe -m features.cli build --help
ote_venv/Scripts/python.exe -m preprocessing prepare --help

# Training and post-training evaluation
ote_venv/Scripts/python.exe -m model_training.ote_training.ote_xgboost_pipeline --help
./scripts/run_frvp_post_training_eval.ps1
./scripts/run_ict_post_training_eval.ps1

# Tests (the progress doc's convention: no cache provider and a short basetemp)
ote_venv/Scripts/python.exe -m pytest tests -p no:cacheprovider --basetemp C:/Users/<you>/AppData/Local/Temp/tbpt
```

## History

The original 2025 system-architecture vision that used to live in this README is preserved at the top of
[`docs/OTE_journal.md`](docs/OTE_journal.md). A full pre-cleanup snapshot (2026-10-09) is on the local branch
`backup/pre-cleanup-20261009`.

# OTE End-to-End Workflow

This README is the operational runbook for the EURUSD 5-minute OTE stack from raw bars to model-testing evidence.

It reflects the repo state and recent runs as of March 31, 2026.

The short version is:

1. Label the raw EURUSD 5-minute bars.
2. Build the feature dataset.
3. Preprocess into model-ready target folders.
4. Build backend-aware attribution rankings.
5. Train XGBoost, TCN, and LSTM against the correct merged ranking for each backend.
6. Run regime slices.
7. Run threshold and abstain policy search.
8. Run the walk-forward backtest with frictions.
9. Promote only the models and policies that survive the full stack.

## Canonical Timezone Contract

For the current full EURUSD training flow, treat the historical raw feed as:

- raw source timezone: `GMT-6`
- canonical storage/runtime timezone: `UTC`
- feature/session clock timezone: `America/New_York`
- FX market close boundary: `17:00 America/New_York`

The current pipeline now persists that contract from:

- raw label outputs via `.metadata.json` sidecars
- feature builds via `data/features/*.metadata.json`
- prepared datasets via `summary.json` and target `report.json`
- trained model artifacts via `training_summary.json` and `model_config.json`

## Current Workflow Decision

Before the backend-aware attribution module existed, your SHAP flow was effectively XGBoost-centric:

- raw XGBoost SHAP outputs were written under `data/prepared/eurusd_5min_ote_full/shap_outputs/`
- the merged ranking was written back as the generic `feature_importance_merged.csv`
- TCN and LSTM training did not have their own backend-specific merged rankings

That means:

- XGBoost is the closest to already being aligned with the old ranking flow.
- TCN and LSTM should be retrained with the new backend-aware ranking files.
- After retraining TCN and LSTM, you should rerun regime slicing, threshold search, abstain policy evaluation, and walk-forward backtesting.
- If you want perfect apples-to-apples parity across all three backends, rerun XGBoost too.

## Recent Artifact Roots

These are the important paths from your recent runs:

- Label output: `data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv`
- Feature output: `data/features/eurusd_5min_ote_full.csv`
- Prepared root: `data/prepared/eurusd_5min_ote_full`
- Old generic merged rankings:
  - `data/prepared/eurusd_5min_ote_full/long_ote/feature_importance_merged.csv`
  - `data/prepared/eurusd_5min_ote_full/short_ote/feature_importance_merged.csv`
- Existing trained model roots:
  - `models/ote_full_xgb_v2`
  - `models/ote_full_tcn_v3_tail`
  - `models/ote_full_lstm_v1`
- Existing regime slice run:
  - `model_testing/reports/ote_regime_slices/full_registry_20260330_200iter`
- Existing threshold-policy run:
  - `model_testing/reports/ote_threshold_policies/active_champions_cost_aware_20260330`
- Existing backtest run:
  - `model_testing/reports/ote_policy_backtests/full_run_v2`

## Directory Map

- Raw bars: `data/currency_data/eurusd-5m.csv`
- Labeled bars: `data/labeling/labeled_data/`
- Feature dataset: `data/features/`
- Prepared training folders: `data/prepared/eurusd_5min_ote_full/`
- Model artifacts: `models/`
- Registry: `models/ote_model_registry.json`
- Testing reports: `model_testing/reports/`

## Stage 1: Labeling

The labeler converts raw EURUSD 5-minute OHLCV data into OTE targets plus helper columns. The current labeler uses a fixed `warmup_bars=50` internally.

This is the exact command shape already used in your batch pipeline:

```cmd
python data/labeling/labeling_engine.py ^
  --input data/currency_data/eurusd-5m.csv ^
  --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv ^
  --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv ^
  --source-timezone GMT-6 ^
  --canonical-timezone UTC ^
  --bar-timestamp-semantics as_provided
```

Primary outputs:

- `data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv`
- `data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv`

## Stage 2: Feature Generation

This expands the labeled bars into the full engineered feature set.

This is the exact command shape from your recent full-data run:

```cmd
python -m features.cli build data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv ^
  --output data/features/eurusd_5min_ote_full.csv ^
  --recipe features/recipes/ote_extended.json ^
  --all-strategies ^
  --transform-workers 4 ^
  --optimize-memory ^
  --skip-strategy-errors ^
  --progress ^
  --strategy-timeout-seconds 600 ^
  --source-timezone UTC ^
  --canonical-timezone UTC ^
  --feature-clock-timezone America/New_York ^
  --market-close-timezone America/New_York
```

Primary outputs:

- `data/features/eurusd_5min_ote_full.csv`
- `data/features/eurusd_5min_ote_full.metadata.json`

## Stage 3: Preprocessing

This step converts the full feature CSV into chronological train/val/test target folders and computes the base feature-importance table.

This is the exact command shape from your recent full-data run:

```cmd
python -m preprocessing prepare data/features/eurusd_5min_ote_full.csv ^
  --output-dir data/prepared/eurusd_5min_ote_full ^
  --scaler none ^
  --corr-threshold 0.98 ^
  --similarity-threshold 0.995 ^
  --max-analysis-rows 10000
```

Notes:

- The prepared root also includes `long_entry` and `short_entry`, but the OTE model-training flow here is for `long_ote` and `short_ote`.
- Each OTE target folder contains `train.csv`, `val.csv`, `test.csv`, `features.json`, `feature_importance.csv`, and `report.json`.
- `summary.json` and each target `report.json` now carry the timezone contract plus source-lineage metadata.

If you want one command for stages 1 to 3, use the existing batch file:

```cmd
scripts\run_eurusd_ote_pipeline.bat
```

## Stage 4: Backend-Aware Attribution Rankings

This is the new step that now belongs between preprocessing and model training.

It keeps one common prepared dataset, but writes backend-specific ranking files for:

- XGBoost
- TCN
- LSTM

The trainer now automatically prefers these files when they exist:

- `feature_importance_merged_xgboost.csv`
- `feature_importance_merged_tcn.csv`
- `feature_importance_merged_lstm.csv`

Recommended command:

```cmd
python -m preprocessing backend-attribution ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --backend xgboost ^
  --backend tcn ^
  --backend lstm ^
  --max-features 160 ^
  --base-weight 0.20 ^
  --shap-weight 0.55 ^
  --shap-positive-weight 0.25 ^
  --attribution-floor-fraction 0.15 ^
  --attribution-cumulative-importance 0.90
```

Important implementation detail:

- XGBoost uses TreeSHAP.
- TCN and LSTM use integrated gradients internally, but the module writes SHAP-compatible ranking columns so the existing merge logic still works.
- The merged backend ranking is now filtered before training with two gates:
  - floor gate: keep only features with `mean_abs_shap_all >= 15%` of the max attribution
  - cumulative gate: from those survivors, keep only the features needed to reach `90%` cumulative attribution share

Primary outputs per target:

- `shap_feature_stats_xgboost.csv`
- `shap_feature_stats_tcn.csv`
- `shap_feature_stats_lstm.csv`
- `feature_importance_merged_xgboost.csv`
- `feature_importance_merged_tcn.csv`
- `feature_importance_merged_lstm.csv`
- `backend_attribution_summary_<backend>.json`

## Stage 5: Model Training

### General Rules

- Use the same prepared root: `data/prepared/eurusd_5min_ote_full`
- Keep new retrains in new output roots instead of overwriting the old artifacts
- Only update the registry after the new candidates clear the testing stack
- The trainer now uses the full filtered backend ranking instead of tuning a fixed top-N feature count.

### XGBoost

Your previous XGBoost run lived under `models/ote_full_xgb_v2`. That run is probably the least urgent to redo because the old generic merged ranking came from XGBoost-based SHAP. If you want parity with the new ranking flow, rerun it too.

Command based on your past XGBoost setup:

```cmd
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --output-root models/ote_full_xgb_v3_backend_ranked ^
  --backend xgboost ^
  --targets long_ote short_ote ^
  --trials 20 ^
  --max-loaded-features 160 ^
  --window-min 8 ^
  --window-max 40 ^
  --batch-size 256 ^
  --calibration-method platt
```

Notes:

- Your prior XGBoost model root was `models/ote_full_xgb_v2`.
- XGBoost phase scheduling is handled internally through tuned `warmup_rounds`, `main_rounds`, `fine_rounds`, and `fine_lr_scale`, not via explicit CLI epoch-phase flags.
- New model artifacts now persist the prepared-dataset timezone contract and upstream source lineage in `training_summary.json` and `model_config.json`.

### TCN

This is the backend that most clearly needs to be retrained now that backend-aware ranking files exist.

Your previous TCN run used an explicit tail schedule under `models/ote_full_tcn_v3_tail`.

Recommended rerun command:

```cmd
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --output-root models/ote_full_tcn_v4_backend_ranked ^
  --backend torch ^
  --model-type tcn ^
  --targets long_ote short_ote ^
  --trials 20 ^
  --max-loaded-features 160 ^
  --window-min 16 ^
  --window-max 32 ^
  --epochs 40 ^
  --batch-size 256 ^
  --hidden-size 64 ^
  --num-layers 3 ^
  --dropout 0.20 ^
  --learning-rate 0.001 ^
  --torch-warmup-epochs 4 ^
  --torch-main-epochs 18 ^
  --torch-fine-epochs 10 ^
  --torch-tail-epochs 8 ^
  --torch-fine-lr-scale 0.20 ^
  --torch-tail-lr-scale 0.07 ^
  --calibration-method platt
```

That matches the schedule style of your existing `ote_full_tcn_v3_tail` run:

- `epochs = 40`
- `warmup = 4`
- `main = 18`
- `fine = 10`
- `tail = 8`
- `4 + 18 + 10 + 8 = 40`

### LSTM

Your previous LSTM benchmark lived under `models/ote_full_lstm_v1` and was long-only. Since LSTM did not have backend-specific merged rankings at the time, it should be retrained if you want a valid comparison under the new flow.

Command matching your previous long-side benchmark style:

```cmd
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --output-root models/ote_full_lstm_v2_backend_ranked ^
  --backend torch ^
  --model-type lstm ^
  --targets long_ote ^
  --trials 20 ^
  --max-loaded-features 160 ^
  --window-min 8 ^
  --window-max 40 ^
  --epochs 40 ^
  --batch-size 256 ^
  --hidden-size 64 ^
  --num-layers 2 ^
  --dropout 0.20 ^
  --learning-rate 0.001 ^
  --calibration-method platt
```

If you now want a full short-side LSTM benchmark too, add `short_ote`:

```cmd
--targets long_ote short_ote
```

## Torch Warmup, Fine, And Tail Schedule Rules

These rules matter for TCN and LSTM:

- If you do not set explicit torch phase flags, the trainer auto-builds a warmup/main/fine schedule from `--epochs`.
- If you set any explicit phase count, or set `--torch-tail-epochs` above zero, then you must set all of:
  - `--torch-warmup-epochs`
  - `--torch-main-epochs`
  - `--torch-fine-epochs`
  - and the sum with `--torch-tail-epochs` must equal `--epochs`
- `--torch-fine-lr-scale` controls the lower-LR fine phase.
- `--torch-tail-lr-scale` controls the final tail phase.

Practical guidance:

- TCN: keep the explicit tail schedule you already used successfully.
- LSTM: your previous benchmark used the automatic schedule with no tail. That is still valid. If you want an explicit tail phase for LSTM now, choose a schedule that sums cleanly to `--epochs`.

Example of a valid explicit 40-epoch torch schedule:

```cmd
--epochs 40 ^
--torch-warmup-epochs 4 ^
--torch-main-epochs 20 ^
--torch-fine-epochs 10 ^
--torch-tail-epochs 6
```

## Stage 6: Registry Update

The testing scripts operate off `models/ote_model_registry.json`, so new candidate models should either:

- be added to the registry with new `model_id` values, or
- temporarily replace old artifact paths if you are intentionally superseding them

Recommended approach:

- keep the old registry entries intact
- add new candidate entries for the backend-ranked retrains
- only promote them after regime slicing, threshold policy search, and backtesting

## Stage 7: Regime Slice Report

This is your first post-training evidence layer.

Your existing full-registry run used:

- output root: `model_testing/reports/ote_regime_slices/full_registry_20260330_200iter`
- `bootstrap_iterations = 200`
- `min_positive_events = 50`

Command in that style:

```cmd
python scripts/run_ote_regime_slice_report.py ^
  --registry-path models/ote_model_registry.json ^
  --output-root model_testing/reports/ote_regime_slices/full_registry_20260330_200iter ^
  --bootstrap-iterations 200 ^
  --min-positive-events 50
```

If you only want to test fresh backend-ranked candidates first, run with repeated `--model-id` values after adding them to the registry.

Primary outputs:

- `slice_report.csv`
- `composite_bucket_winners.csv`
- per-model `oof_regime_labeled_predictions.csv`
- per-model `test_regime_labeled_predictions.csv`

## Stage 8: Threshold And Abstain Policy Search

This stage searches regime-aware thresholds and checks whether abstention improves the trading decision layer.

Your existing champion-focused run used:

- output root: `model_testing/reports/ote_threshold_policies/active_champions_cost_aware_20260330`
- `min_positive_events = 50`
- `min_events_per_month = 3.0`
- `min_trades_per_week = 3.0`
- optional: `--write-registry-policies` only after manual review of the saved threshold/backtest artifacts

Command in that style:

```cmd
python scripts/run_ote_threshold_policy_search.py ^
  --regime-report-root model_testing/reports/ote_regime_slices/full_registry_20260330_200iter ^
  --registry-path models/ote_model_registry.json ^
  --output-root model_testing/reports/ote_threshold_policies/active_champions_cost_aware_20260330 ^
  --include-role champion ^
  --min-positive-events 50 ^
  --min-events-per-month 3.0 ^
  --min-trades-per-week 3.0
```

What this does:

- searches composite-regime thresholds
- compares `global_threshold`, `regime_threshold`, and abstain-aware variants
- writes reviewable threshold/backtest artifacts first; registry write-back is now a deliberate follow-up step rather than the default research flow

Primary outputs:

- `policy_table.csv`
- `policy_evaluation.csv`
- optional per-model decision files if `--write-policy-decisions` is used

## Stage 9: Walk-Forward Policy Backtest

This is the final offline promotion gate before paper trading.

Your existing `full_run_v2` backtest used:

- regime report root: `model_testing/reports/ote_regime_slices/full_registry_20260330_200iter`
- output root: `model_testing/reports/ote_policy_backtests/full_run_v2`
- `min_train_years = 2`
- `test_window_months = 3`
- `rolling_step_months = 3`
- `min_folds = 8`
- `min_positive_events = 50`
- `min_events_per_month = 3.0`
- `min_trades_per_week = 3.0`
- `maximum_drawdown_pct = 12.0`
- `drawdown_starting_balance_units = 10000.0`
- `fixed_slippage_pips_per_trade = 0.3`
- `commission_pips_per_trade = 0.35`
- `--targeted-filter-preset full_run_v2`

Command in that style:

```cmd
python scripts/run_ote_policy_backtest.py ^
  --regime-report-root model_testing/reports/ote_regime_slices/full_registry_20260330_200iter ^
  --registry-path models/ote_model_registry.json ^
  --output-root model_testing/reports/ote_policy_backtests/full_run_v2 ^
  --include-role champion ^
  --min-train-years 2 ^
  --test-window-months 3 ^
  --rolling-step-months 3 ^
  --min-folds 8 ^
  --min-positive-events 50 ^
  --min-events-per-month 3.0 ^
  --min-trades-per-week 3.0 ^
  --maximum-drawdown-pct 12.0 ^
  --drawdown-starting-balance-units 10000 ^
  --fixed-slippage-pips-per-trade 0.3 ^
  --commission-pips-per-trade 0.35 ^
  --targeted-filter-preset full_run_v2
```

Walk-forward acceptance now uses account-equity `max_drawdown_pct_below_threshold`, computed from cumulative strategy performance plus `--drawdown-starting-balance-units`. The drawdown gate is advisory risk context rather than a hard promotion veto, so downstream registry/bundle decisions should preserve both the accepted flag and the explicit drawdown readout.

Primary outputs:

- `model_summary.csv`
- `run_summary.json`
- per-model `fold_summary.csv`
- per-model `selected_test_trades.csv`
- per-model `selected_test_equity_curve.csv`
- per-model breakdowns by year, quarter, session, composite regime, and confidence bucket

## Recommended Rerun Order Right Now

Given the new backend-aware ranking step, this is the clean rerun order I recommend:

1. Rebuild backend-aware attribution rankings on `data/prepared/eurusd_5min_ote_full`.
2. Retrain TCN with the new `feature_importance_merged_tcn.csv`.
3. Retrain LSTM with the new `feature_importance_merged_lstm.csv`.
4. Optionally rerun XGBoost so all three backends are using the same new attribution pipeline.
5. Add the new model artifacts to `models/ote_model_registry.json` as candidates.
6. Rerun regime slice reporting.
7. Rerun threshold policy search.
8. Rerun the walk-forward backtest.
9. Only then decide promotion, challenger status, or benchmark-only status.

Practical interpretation of your current belief:

- Yes, it is reasonable to treat the old XGBoost stack as mostly aligned already.
- Yes, TCN and LSTM are the backends that most clearly need retraining and retesting after the backend-aware attribution update.

## Definition Of Done

Your complete model-training workflow is only finished when all of the following exist for the candidate generation you want to promote:

- labeled dataset
- full feature dataset
- prepared OTE target folders
- backend-specific merged ranking files for the trained backends
- trained model artifacts
- registry entries pointing to the new artifacts
- regime slice report
- threshold-policy evaluation
- abstain-policy metadata where justified
- walk-forward backtest report with frictions
- final champion/challenger decision recorded in the registry

## Typical Full Run Order

```cmd
python data/labeling/labeling_engine.py ...
python -m features.cli build ...
python -m preprocessing prepare ...
python -m preprocessing backend-attribution ...
python -m model_training.ote_training.ote_xgboost_pipeline ...    rem xgboost
python -m model_training.ote_training.ote_xgboost_pipeline ...    rem tcn
python -m model_training.ote_training.ote_xgboost_pipeline ...    rem lstm
python scripts/run_ote_regime_slice_report.py ...
python scripts/run_ote_threshold_policy_search.py ...
python scripts/run_ote_policy_backtest.py ...
```

That is now the complete offline OTE workflow from labeling to model-testing evidence.

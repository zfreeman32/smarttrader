# FRVP Setup-Specific Long Rerun Plan

Date: 2026-08-29

## 1. Readiness Check

Status: code path is implemented. Do not open another pooled FRVP retrain branch before this setup-specific lane is materialized and evaluated.

- `frvp/target_lanes.py` defines 6 setup types, 12 setup-specific direct targets, 4 pooled direct targets, and 2 meta controls.
- `data/labeling/frvp_labeling_engine.py` writes pooled and setup-specific helper columns from the same events.
- Setup targets inherit pooled-family concurrency and sample weights. This keeps the first comparison focused on routing only.
- `frvp/pipelines/es_primary_phase04.py` now prepares all `FRVP_TARGET_COLUMNS`.
- `scripts/run_frvp_training_stack.ps1` supports `-XgbTargetMode setup`, but this plan uses direct target lists so only long lanes run.
- `scripts/build_frvp_candidate_registry.py` infers setup model IDs such as `frvp_long_reversal_setup1_xgb_v1`.

Verified locally:

```powershell
ote_venv\Scripts\python.exe -m pytest tests\test_frvp_per_setup_targets.py tests\test_frvp_setups.py
ote_venv\Scripts\python.exe -m pytest tests\test_build_frvp_candidate_registry.py tests\test_frvp_labeling.py
```

Result: 34 passed.

Current artifact gap: `artifacts\frvp_es_primary_current\phase04\prepared` still contains only the four pooled direct targets. The next run must materialize a new setup-aware Phase 4 root before training.

## 2. Setup Logic Read

The setup logic is sound for the current Fixed Range Volume Profile contract.

- Setup 1: balanced D-profile, open in value, fade VAH/VAL. It rejects real outside displacement so continuation breakouts do not get mislabeled as fades.
- Setup 2: continuation retest after a real value-area breakout. It owns the shallow near-edge hold.
- Setup 3: initiative value-area breakout. Requires real volume expansion, close-location efficiency, and at least 0.05 ATR acceptance beyond VAH/VAL.
- Setup 4: failed auction. Requires 1-3 outside bars, same-side sweep, quiet reentry, and more than 0.05 ATR reentry into value.
- Setup 5: LVN / thin-zone continuation, gated by displacement and proximity to nearest LVN.
- Setup 6: balanced in-value HVN magnet. Correctly avoids open-drive, displacement, high-volume, and ambiguous symmetric-HVN contexts.

Caveats:

- Setup 4 is structurally thin: prior replay found 380 usable long events and 412 usable short events. Treat S4 as research-only until economics prove otherwise.
- Setup 6 still lacks gamma-context and Setup 6b naked-VPOC event logic. Do not block this rerun on those deferred items.
- Current targeted policy presets are keyed to pooled model IDs. Do no-preset setup economics first, then port filters only after setup-level regime tables justify it.

## 3. Branches To Rerun

Rerun only the strongest long FRVP branches:

- Long continuation full-span XGBoost: split into S2, S3, S5.
- Long reversal full-span XGBoost: split into S1, S4, S6.
- Long reversal recency-weighted XGBoost: split into S1 and S6 first, using the proven 730-day half-life and 0.20 floor. S4 recency is optional research after the full-span S4 read.
- Do not run TCNs in wave 1.
- Do not train short lanes in this wave.

## 4. Materialize Setup-Aware Prepared Root

Run from repo root.

```powershell
ote_venv\Scripts\python.exe -m frvp.pipelines.es_primary_phase04 `
  --input data\futures_data\ES-5m-tagged.csv `
  --output-root artifacts\frvp_es_primary_setup_targets_20260829 `
  --feature-csv artifacts\frvp_es_primary_current\phase02\es_primary_frvp_features_full.csv.gz `
  --feature-metadata artifacts\frvp_es_primary_current\phase02\es_primary_frvp_features_full.csv.metadata.json `
  --instrument es `
  --backend-max-features 160 `
  --attribution-max-rows 100000 `
  --top-n-features 25 `
  --min-usable-rows 250 `
  --min-train-rows 100 `
  --min-positive-samples 25 `
  --phase02-compression gzip
```

Verify target directories:

```powershell
Get-ChildItem artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared -Directory | Select-Object Name
```

Expected long setup dirs:

- `long_frvp_reversal_setup1`
- `long_frvp_continuation_setup2`
- `long_frvp_continuation_setup3`
- `long_frvp_reversal_setup4`
- `long_frvp_continuation_setup5`
- `long_frvp_reversal_setup6`

## 5. Train Full-Span Long Continuation Setups

Use reduced setup-lane CV geometry instead of pooled 900/300/300. This keeps S2 from being over-constrained while still giving S3/S5 enough folds.

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-root models\frvp_long_continuation_setup_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_continuation_setup2 long_frvp_continuation_setup3 long_frvp_continuation_setup5 `
  --trials 28 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 40 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 500 `
  --cv-val-rows 150 `
  --cv-step-rows 150 `
  --cv-min-folds 2 `
  --min-train-positive-rows 100 `
  --min-val-positive-rows 25 `
  --min-val-true-events 10 `
  --seed 42
```

Note: continuation calibration resolves to `none` by target contract even when `--calibration-method platt` is passed.

## 6. Train Full-Span Long Reversal Setups S1/S6

S1 and S6 get a smaller but still multi-fold geometry. Keep S4 separate.

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup1 long_frvp_reversal_setup6 `
  --trials 32 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 32 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 350 `
  --cv-val-rows 100 `
  --cv-step-rows 100 `
  --cv-min-folds 2 `
  --min-train-positive-rows 70 `
  --min-val-positive-rows 15 `
  --min-val-true-events 8 `
  --seed 42
```

## 7. Train Setup 4 Standalone

Research-only geometry. Do not compare S4 promotion status to the regular lanes without noting the one-fold constraint.

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup4_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup4 `
  --trials 40 `
  --max-loaded-features 160 `
  --top-feature-min 16 `
  --top-feature-max 64 `
  --window-min 8 `
  --window-max 24 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 140 `
  --cv-val-rows 40 `
  --cv-step-rows 40 `
  --cv-min-folds 1 `
  --min-train-positive-rows 20 `
  --min-val-positive-rows 5 `
  --min-val-true-events 3 `
  --final-eval-min-rows 40 `
  --seed 42
```

## 8. Materialize Recency-Weighted Reversal Setup Roots

Use the prior winning full-history recency contract: 730-day half-life, 0.20 floor.

```powershell
ote_venv\Scripts\python.exe scripts\materialize_frvp_recency_prepared_root.py `
  --base-prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-prepared-root artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared `
  --target long_frvp_reversal_setup1 `
  --half-life-days 730 `
  --floor 0.20
```

```powershell
ote_venv\Scripts\python.exe scripts\materialize_frvp_recency_prepared_root.py `
  --base-prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-prepared-root artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared `
  --target long_frvp_reversal_setup6 `
  --half-life-days 730 `
  --floor 0.20
```

## 9. Train Recency-Weighted Reversal Setups

S1:

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup1_recency730_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup1 `
  --trials 32 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 32 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 350 `
  --cv-val-rows 100 `
  --cv-step-rows 100 `
  --cv-min-folds 2 `
  --min-train-positive-rows 70 `
  --min-val-positive-rows 15 `
  --min-val-true-events 8 `
  --seed 42
```

S6:

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup6_recency730_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup6 `
  --trials 32 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 32 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 350 `
  --cv-val-rows 100 `
  --cv-step-rows 100 `
  --cv-min-folds 2 `
  --min-train-positive-rows 70 `
  --min-val-positive-rows 15 `
  --min-val-true-events 8 `
  --seed 42
```

## 10. Build Registries

Full-span setup registry:

```powershell
ote_venv\Scripts\python.exe scripts\build_frvp_candidate_registry.py `
  --model-root models\frvp_long_continuation_setup_xgb_20260829 `
  --model-root models\frvp_long_reversal_setup_xgb_20260829 `
  --model-root models\frvp_long_reversal_setup4_xgb_20260829 `
  --source-registry-path models\frvp_es_primary_model_registry_current.json `
  --output-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json
```

Recency reversal setup registry:

```powershell
ote_venv\Scripts\python.exe scripts\build_frvp_candidate_registry.py `
  --model-root models\frvp_long_reversal_setup1_recency730_xgb_20260829 `
  --model-root models\frvp_long_reversal_setup6_recency730_xgb_20260829 `
  --source-registry-path models\frvp_es_primary_model_registry_current.json `
  --output-path models\frvp_es_primary_model_registry_long_reversal_setup_recency730_20260829.json
```

Use separate registries because full-span and recency models share the same inferred setup model IDs.

## 11. Evaluate Full-Span Regular Lanes

No targeted preset on the first pass.

```powershell
.\scripts\run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_setup_fullspan_20260829 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_setup_fullspan_20260829 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829 `
  -ModelIds frvp_long_reversal_setup1_xgb_v1,frvp_long_reversal_setup6_xgb_v1,frvp_long_continuation_setup2_xgb_v1,frvp_long_continuation_setup3_xgb_v1,frvp_long_continuation_setup5_xgb_v1 `
  -BacktestMinTrainYears 2 `
  -BacktestMinFolds 8 `
  -SpreadCostMode session_schedule
```

## 12. Evaluate Setup 4 Separately

Research contract, low-frequency thresholds.

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_regime_slice_report.py `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --bootstrap-iterations 200 `
  --min-positive-events 10
```

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_threshold_policy_search.py `
  --regime-report-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_threshold_policies\frvp_long_setup4_fullspan_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --instrument es `
  --spread-cost-mode session_schedule `
  --min-positive-events 10 `
  --min-events-per-month 0.5 `
  --min-trades-per-week 0.5 `
  --evaluation-contract-mode research `
  --write-policy-decisions
```

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_policy_backtest.py `
  --regime-report-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --instrument es `
  --spread-cost-mode session_schedule `
  --min-train-years 2 `
  --test-window-months 3 `
  --rolling-step-months 3 `
  --min-folds 4 `
  --min-positive-events 10 `
  --min-events-per-month 0.5 `
  --min-trades-per-week 0.5 `
  --evaluation-contract-mode research `
  --minimum-sharpe 0.8 `
  --maximum-drawdown-pct 12.0 `
  --minimum-dsr 0.3
```

## 13. Evaluate Recency Reversal Lanes

First pass: full evaluation window.

```powershell
.\scripts\run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_setup_recency730_20260829.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_20260829 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_20260829 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_setup_recency730_20260829 `
  -ModelIds frvp_long_reversal_setup1_xgb_v1,frvp_long_reversal_setup6_xgb_v1 `
  -BacktestMinTrainYears 2 `
  -BacktestMinFolds 8 `
  -SpreadCostMode session_schedule
```

Second pass: recent-regime control matching the prior accepted reversal branch geometry.

```powershell
.\scripts\run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_setup_recency730_20260829.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_recent2y_20260829 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_recent2y_20260829 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_setup_recency730_recent2y_20260829 `
  -ModelIds frvp_long_reversal_setup1_xgb_v1,frvp_long_reversal_setup6_xgb_v1 `
  -BacktestMinTrainYears 2 `
  -BacktestMaxTrainYears 2 `
  -BacktestMinScheduledTestStart 2024-01-01 `
  -BacktestMinFolds 8 `
  -SpreadCostMode session_schedule
```

## 14. Decision Rules

Promote only if a setup lane beats its pooled parent on economics, not just classifier metrics.

- Continuation setup winner must beat or clearly simplify `frvp_long_continuation_gatefix_v3_20260715_accountdd`: 524 trades, +7190.40 ticks, Sharpe 1.226, DSR 1.061, WFE 2.459, 8/8 gates.
- Reversal full-span setup winner must repair the pooled recency branch's train-side instability: +6027.00 ticks and Sharpe 1.061 were not enough because WFE was -3.797.
- Reversal recent2y setup winner must beat the accepted selective contract: 63 trades, +3677.05 ticks, Sharpe 1.480, DSR 1.179, WFE 2.162, 8/8 gates.
- S4 must be judged as a standalone research lane unless it clears enough fold/trade breadth to become promotion-quality eligible.
- If S2/S3/S5 all trail pooled continuation, keep pooled continuation.
- If S1 or S6 wins but only after regime pruning, add setup-specific targeted filter presets in a separate code change and rerun the backtest. Do not reuse pooled presets blindly.

## 15. Experiment Continuation

Status after `frvp_setup_specific_long_rerun_20260829_output_v2.txt`: no setup-specific lane is promotable. Keep the current deployed/paper roster unchanged.

### 15.1 Clean Script Noise

Update `scripts\run_frvp_setup_specific_long_rerun_plan_20260829.ps1` before the next run.

- Skip recency prepared-root materialization when the target root already exists, matching the Step 4 skip behavior.
- Set pytest temp/cache paths inside the repo, or disable pytest cache, so Step 1A stops false-failing on local permission errors.

Implementation target:

```powershell
New-Item -ItemType Directory -Force tmp\pytest | Out-Null
$env:TMP = (Resolve-Path tmp\pytest).Path
$env:TEMP = (Resolve-Path tmp\pytest).Path
$env:PYTEST_ADDOPTS = "-p no:cacheprovider"
```

Recency-root skip rule:

```powershell
if (Test-Path "artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared") {
  Write-Output "Skipping S1 recency prepared root; already exists."
}
else {
  ote_venv\Scripts\python.exe scripts\materialize_frvp_recency_prepared_root.py `
    --base-prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
    --output-prepared-root artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared `
    --target long_frvp_reversal_setup1 `
    --half-life-days 730 `
    --floor 0.20
}
```

Apply the same pattern for `long_frvp_reversal_setup6`.

### 15.2 S2 Continuation Policy-Only Pass

Do not retrain S2 yet. It has real signal, but 2024 is the drag:

- `2024`: `-1861.65` ticks
- `2025`: `+3744.65` ticks
- `2026`: `+1616.35` ticks

First pass: use the existing v2 full-span outputs to identify bad regime/session pockets.

```powershell
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup2_xgb_v1\breakdown_by_year.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup2_xgb_v1\breakdown_by_session.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup2_xgb_v1\breakdown_by_composite.csv
```

Then create a setup-specific targeted filter preset only if the bad pockets are repeated enough to justify blocking. Rerun economics with the model fixed.

Executed continuation pass:

- Added preset `frvp_long_continuation_setup2_xgb_medium_session_prune_v1` for `frvp_long_continuation_setup2_xgb_v1`.
- Filtered composite/session pairs: `strong_down_medium/london`, `strong_down_medium/overlap`, `strong_up_medium/london`, `strong_up_medium/overlap`.
- Rerun output root: `model_testing\reports\frvp_backtests\frvp_long_setup2_medium_session_prune_20260829`.
- Result improved from `201` trades, `+3499.35` ticks, Sharpe `0.708`, max DD `24.52%`, WFE `-2.004` to `104` trades, `+4344.40` ticks, Sharpe `1.625`, DSR `1.153`, max DD `8.36%`, WFE `-4.127`.
- Gate read: economics improved and drawdown repaired, but still not promotable because WFE remains negative and largest-single-trade concentration still fails.
- Next S2 action: do not promote. If S2 remains interesting after S5, inspect train-window instability before any retrain or stronger pruning.

### 15.3 Drop S3

Drop `frvp_long_continuation_setup3_xgb_v1` from the active setup rerun path for now.

- Classifier metrics are good.
- Economics are negative: `485` trades, `-2964.25` ticks, Sharpe `-0.379`, max DD `49.96%`.
- Do not spend more training or policy time on S3 until S2/S5 are exhausted.

### 15.4 S5 Continuation Drawdown Policy Pass

Treat S5 as too broad and too drawdown-heavy.

- Result: `1217` trades, `+5255.95` ticks.
- Problem: Sharpe `0.281`, DSR `0.279`, max DD `82.26%`.

Run a targeted policy pass to reduce drawdown before considering more training.

```powershell
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup5_xgb_v1\breakdown_by_year.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup5_xgb_v1\breakdown_by_session.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup5_xgb_v1\breakdown_by_composite.csv
```

Main suspected drag: 2022 was `-6762.45` ticks. Confirm whether that is a year-only regime failure or a smaller composite/session pocket before coding a preset.

Executed continuation pass:

- Added preset `frvp_long_continuation_setup5_xgb_repeated_drawdown_prune_v1` for `frvp_long_continuation_setup5_xgb_v1`.
- Filtered repeated negative composite/session pockets: `strong_down_high/london`, `strong_up_high/london`, `strong_up_medium/asia`, `ranging_high/new_york`, `ranging_medium/asia`, `strong_down_low/london`, `strong_down_medium/overlap`, `strong_up_low/asia`, `strong_down_high/new_york`, `strong_up_medium/new_york`.
- Rerun output root: `model_testing\reports\frvp_backtests\frvp_long_setup5_repeated_drawdown_prune_promotion_20260829`.
- Result improved from `1217` trades, `+5255.95` ticks, Sharpe `0.281`, DSR `0.279`, max DD `82.26%`, WFE `1.110` to `724` trades, `+12966.40` ticks, Sharpe `0.912`, DSR `0.859`, max DD `38.59%`, WFE `1.723`.
- Post-pass slice read: all composite regimes are positive and all sessions are positive; 2022 remains the only negative calendar year at `-3303.55` ticks, led by `2022Q2` at `-2343.80` ticks.
- Gate read: S5 is materially better but still not promotable because max drawdown remains far above the `12%` advisory gate and largest-single-trade concentration still fails (`16.21%` of total PnL).
- Next S5 action: do not promote. If S5 remains interesting, the next pass should target train-window/date-regime instability around the 2022 drawdown rather than another broad composite prune.

### 15.5 Pause S6 Training

Do not spend more training budget on S6 yet.

- The recall-oriented hyperparameter run did not improve realized economics.
- Full-window result stayed negative at `-1961.25` ticks.
- Recent2y result stayed negative at `-2172.80` ticks.

Next action is policy/threshold slicing, not another blind training sweep.

```powershell
Get-Content model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_opt_20260829\policy_table.csv
Get-Content model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_opt_20260829\frvp_long_reversal_setup6_xgb_v1\test_slice_report.csv
```

Executed S6 pause review on 2026-08-30:

- Reviewed the optimized recency S6 policy table and test slice report.
- All S6 composite policy rows still used the `0.05` global fallback threshold; composite regime thresholds were not data-sufficient.
- Full-window optimized backtest stayed negative: `345` trades, `-1961.25` ticks, Sharpe `-0.342`, DSR `-0.291`, max DD `33.40%`, WFE `0.368`.
- Recent2y optimized backtest stayed negative: `192` trades, `-2172.80` ticks, Sharpe `-0.528`, DSR `-0.374`, max DD `33.30%`, WFE `1.605`.
- Full-window breakdown confirms the drag is not a single obvious pocket: `2025` was `-1757.50` ticks and `2026` was `-1199.95` ticks; London was `-1304.50` ticks and off-hours was `-618.60` ticks.
- Composite policy/backtest read remains mixed, with only `4/9` realized composite buckets positive and the larger negative buckets overwhelming the positive ones.
- Decision: keep S6 paused for training. If revisited, do policy research around threshold sufficiency and session/year instability first; do not run another blind hyperparameter sweep.

### 15.6 Keep S4 Research-Only

S4 trained, but the policy backtest only produced one fold while the command required four.

Keep S4 out of promotion comparisons. For the next S4 read, either run an explicit one-fold research report or lower `--min-folds 1`.

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_policy_backtest.py `
  --regime-report-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_1fold_research_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --instrument es `
  --spread-cost-mode session_schedule `
  --min-train-years 2 `
  --test-window-months 3 `
  --rolling-step-months 3 `
  --min-folds 1 `
  --min-positive-events 10 `
  --min-events-per-month 0.5 `
  --min-trades-per-week 0.5 `
  --evaluation-contract-mode research `
  --minimum-sharpe 0.8 `
  --maximum-drawdown-pct 12.0 `
  --minimum-dsr 0.3
```

Executed S4 one-fold research report on 2026-08-30:

- Output root: `model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_1fold_research_20260829`.
- The one-fold research run completed successfully with `effective_min_folds=1`, `evaluation_contract_mode=research`, and `promotion_quality_gate_eligible=false`.
- Result: `5` selected test trades, `+392.75` ticks, expectancy `+78.55` ticks, hit rate `100%`, Sharpe `4.409`, DSR `0.000`, max DD `0.00%`, WFE `8.152`.
- Breadth is too thin for promotion: the only realized test period was `2026-04-01` through `2026-06-30`, with `2` Asia trades and `3` New York trades.
- Concentration is too high: largest single trade share was `70.36%` of total PnL, so the concentration gate failed even though the tiny sample was profitable.
- Decision: keep S4 research-only and out of setup promotion comparisons. The positive one-fold read is worth preserving as a curiosity, but not enough to justify roster changes or more training budget without a broader event/fold design.

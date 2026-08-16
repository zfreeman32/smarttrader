# ICT Minimum-Trades Contract Audit

Date: `2026-07-19`

Compared three otherwise-matched ICT ES walk-forward / threshold stacks:

- `3 trades/week` strict baseline:
  - threshold: `model_testing/reports/ict_threshold_policies/ict_es_primary_20260718T002437Z/`
  - backtest: `model_testing/reports/ict_backtests/ict_es_primary_20260718T002437Z/`
- `2 trades/week` rerun:
  - threshold: `model_testing/reports/ict_threshold_policies/ict_es_primary_20260719_min2/`
  - backtest: `model_testing/reports/ict_backtests/ict_es_primary_20260719_min2/`
- `0 trades/week` rerun:
  - threshold: `model_testing/reports/ict_threshold_policies/ict_es_primary_20260719_min0/`
  - backtest: `model_testing/reports/ict_backtests/ict_es_primary_20260719_min0/`

## Findings

- Threshold selection did not improve under the looser cadence contracts. All six models still saved `qualified_policy_names = []` and `selected_policy_name = global_threshold` in both the `2/week` and `0/week` reruns.
- Five of six walk-forward summaries were unchanged across `3/week`, `2/week`, and `0/week`:
  - `ict_long_continuation_xgb_v1`: unchanged, still fails the individual-model gate.
  - `ict_long_meta_xgb_v1`: unchanged, still passes the current individual-model gate.
  - `ict_short_continuation_xgb_v1`: unchanged, still fails.
  - `ict_short_meta_xgb_v1`: unchanged, still fails concentration.
  - `ict_short_reversal_xgb_v1`: unchanged, still passes the current individual-model gate.
- `ict_long_reversal_xgb_v1` was the only branch that changed when the cadence floor dropped from `3/week` to `2/week` or `0/week`.
  - `3/week`: selected-policy mix stayed `{'global_threshold': 1286}` with `+7865.1` ticks, expectancy `6.116`, Sharpe `1.927`, DSR `0.848`, WFE `2.414`, and positive composite expectancy share `0.889`.
  - `2/week` and `0/week`: selected-policy mix became `{'global_threshold': 1075, 'global_threshold_plus_abstain': 147}` with `+7241.7` ticks, expectancy `5.926`, Sharpe `1.779`, DSR `0.995`, WFE `2.222`, and positive composite expectancy share `0.778`.
  - Net effect: looser cadence did not create a better branch; it slightly weakened the strongest non-meta family candidate on net PnL, expectancy, Sharpe, WFE, and slice breadth while still leaving it accepted by the current individual-model gate.

## Interpretation

- Lowering or removing the minimum-trades floor does not improve the current ICT roster.
- The current `min_trades_per_week` contract is still mostly a policy-variant selection constraint, not a realized cadence acceptance veto.
- That matters because some currently passing or nearly passing branches are still low-cadence in realized walk-forward usage:
  - `ict_long_meta_xgb_v1`: `336` selected test trades over the full window, about `0.96/week`
  - `ict_short_meta_xgb_v1`: `275` selected test trades over the full window, about `0.79/week`
  - `ict_short_reversal_xgb_v1`: `791` selected test trades over the full window, about `2.28/week`
  - `ict_long_reversal_xgb_v1`: `1286` selected test trades at `3/week`, about `3.70/week`

## Promotion Readout

- `ict_long_reversal_xgb_v1` remains the strongest non-meta family candidate.
- It is not the only branch close under the current individual-model gate: `ict_long_meta_xgb_v1` and `ict_short_reversal_xgb_v1` still pass that gate.
- `ict_short_meta_xgb_v1` still fails concentration.
- Both continuation families still fail on economics and/or robustness.

## Follow-Up

- If minimum cadence is supposed to matter operationally, add an explicit model-level or bundle-level selected-policy cadence gate. The current contract does not yet enforce that.

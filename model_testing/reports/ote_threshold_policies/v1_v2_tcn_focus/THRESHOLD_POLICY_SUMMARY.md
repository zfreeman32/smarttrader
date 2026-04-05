# v1 vs v2 TCN Focus Threshold Policy Summary

This summary covers the threshold and abstain policy search run in:

- `model_testing/reports/ote_threshold_policies/v1_v2_tcn_focus/run_summary.json`
- `model_testing/reports/ote_threshold_policies/v1_v2_tcn_focus/policy_evaluation.csv`
- `model_testing/reports/ote_threshold_policies/v1_v2_tcn_focus/policy_table.csv`

Source inputs:

- regime report root: `model_testing/reports/ote_regime_slices/v1_v2_comparison`
- registry: `models/ote_model_registry_v1_v2_candidates.json`
- models evaluated:
  - `long_ote_tcn_v2_candidate`
  - `short_ote_tcn_v2_candidate`
  - `long_ote_tcn_v1_candidate`
  - `short_ote_tcn_v1_candidate`

Run timestamp:

- generated at `2026-04-03T04:57:06.986990+00:00`
- localized to `2026-04-03 00:57:06 America/New_York`

Run settings:

- `min_positive_events = 50`
- `min_events_per_month = 3.0`
- `min_trades_per_week = 3.0`
- registry updates written: `false`

## Executive Summary

- Only one model found a qualified non-baseline policy: `long_ote_tcn_v2_candidate`.
- The selected policy for that model was `regime_threshold`.
- The improvement for `long_ote_tcn_v2_candidate` is real under the script's qualification rule, but small.
- No abstain-aware policy qualified for any model.
- The abstain variants consistently improved event F0.5, but they also cut throughput and post-cost dollars too aggressively.
- The current next step should be walk-forward backtesting, not registry writes or deployment of an abstain policy.

## Qualification Rule

A policy qualifies only if, on the `test` split, it:

- beats `global_threshold` on `event_f05`
- beats `global_threshold` on `post_cost_expectancy_pips`
- maintains at least `min_trades_per_week`

Under that rule, only one policy qualified in this run.

## Model Results

### Overall Outcome Table

| Model | Baseline Policy | Qualified Policy | Selected Policy | Conclusion |
| --- | --- | --- | --- | --- |
| `long_ote_tcn_v2_candidate` | `global_threshold` | `regime_threshold` | `regime_threshold` | Advance as the strongest threshold-search result |
| `short_ote_tcn_v2_candidate` | `global_threshold` | none | none | Keep baseline global policy for backtest |
| `long_ote_tcn_v1_candidate` | `global_threshold` | none | none | Keep baseline global policy for backtest |
| `short_ote_tcn_v1_candidate` | `global_threshold` | none | none | Keep baseline global policy for backtest |

### `long_ote_tcn_v2_candidate`

Baseline test metrics:

- `global_threshold`
- event F0.5: `0.3815`
- post-cost expectancy: `8.0917` pips
- net PnL: `14573.15` pips
- trades/week: `11.77`

Qualified result:

- `regime_threshold`
- event F0.5: `0.3828`
- post-cost expectancy: `8.1756` pips
- net PnL: `14536.22` pips
- trades/week: `11.62`

Interpretation:

- This is the only model whose regime-threshold variant beat the baseline on both selection metrics.
- The edge is narrow:
  - event F0.5 improved by `+0.0012`
  - post-cost expectancy improved by `+0.0839` pips
  - net PnL fell by about `-36.93` pips
- The regime search mostly confirmed the existing configuration rather than radically changing it.

Key threshold table observations:

- searched global threshold remained `0.71`
- most composite buckets stayed at `0.71`
- the main material tightening was `strong_down_high -> 0.80`
- `strong_up_low` remained a `global_fallback` bucket because it lacked enough positive events

Conclusion:

- `long_ote_tcn_v2_candidate` remains the best promotion favorite in this cohort.
- Its `regime_threshold` variant is worth carrying into walk-forward backtesting.

### `short_ote_tcn_v2_candidate`

Baseline test metrics:

- `global_threshold`
- event F0.5: `0.3850`
- post-cost expectancy: `6.7098` pips
- net PnL: `8910.55` pips
- trades/week: `8.68`

Regime-threshold test metrics:

- `regime_threshold`
- event F0.5: `0.3816`
- post-cost expectancy: `6.7762` pips
- net PnL: `8185.70` pips
- trades/week: `7.90`

Interpretation:

- Regime thresholds improved expectancy slightly, but event F0.5 got worse.
- That means the policy failed the qualification rule.
- The search wanted a meaningfully higher searched global threshold (`0.75`) than the candidate registry's stored threshold (`0.62`), but that did not translate into a qualified regime-policy upgrade.

Key threshold table observations:

- `strong_up_high` stayed at `0.75`
- many other buckets moved down to `0.62`
- `strong_down_low` remained `global_fallback`

Conclusion:

- `short_ote_tcn_v2_candidate` is still the strongest short model from the slice report.
- For now, its baseline `global_threshold` policy is safer than adopting the regime-threshold variant from this run.

### `long_ote_tcn_v1_candidate`

Baseline test metrics:

- `global_threshold`
- event F0.5: `0.3875`
- post-cost expectancy: `8.4512` pips
- net PnL: `11975.38` pips
- trades/week: `9.26`

Regime-threshold test metrics:

- `regime_threshold`
- event F0.5: `0.3897`
- post-cost expectancy: `8.3938` pips
- net PnL: `11356.80` pips
- trades/week: `8.84`

Interpretation:

- Event F0.5 improved slightly.
- Expectancy and net PnL worsened.
- The policy therefore did not qualify.

Conclusion:

- `long_ote_tcn_v1_candidate` remains a useful challenger, but this run did not produce a better policy than its baseline.

### `short_ote_tcn_v1_candidate`

Baseline test metrics:

- `global_threshold`
- event F0.5: `0.3777`
- post-cost expectancy: `4.8981` pips
- net PnL: `6803.45` pips
- trades/week: `9.08`

Regime-threshold test metrics:

- `regime_threshold`
- event F0.5: `0.3721`
- post-cost expectancy: `4.3626` pips
- net PnL: `5684.47` pips
- trades/week: `8.52`

Interpretation:

- Both selection metrics worsened.
- This is the weakest threshold-policy outcome among the four TCN candidates.

Conclusion:

- Keep `short_ote_tcn_v1_candidate` as a benchmark challenger only.

## Abstain Policy Findings

The abstain variants all followed the same pattern:

- event F0.5 improved sharply
- emitted signal count dropped heavily
- trades/week dropped heavily
- post-cost expectancy deteriorated
- total net PnL deteriorated even more

Examples on the `test` split:

- `long_ote_tcn_v2_candidate`
  - `global_threshold`: event F0.5 `0.3815`, expectancy `8.0917`, net PnL `14573.15`, trades/week `11.77`
  - `global_threshold_plus_abstain`: event F0.5 `0.8458`, expectancy `7.0942`, net PnL `4469.32`, trades/week `4.12`
- `short_ote_tcn_v2_candidate`
  - `global_threshold`: event F0.5 `0.3850`, expectancy `6.7098`, net PnL `8910.55`, trades/week `8.68`
  - `global_threshold_plus_abstain`: event F0.5 `0.7791`, expectancy `6.0691`, net PnL `2998.12`, trades/week `3.23`

Conclusion:

- The current abstain configuration is too blunt for production use.
- It is acting like a strong trade suppressor rather than a profitable risk filter.
- No abstain policy should be written into the registry from this run.

## Cross-Model Conclusions

- `long_ote_tcn_v2_candidate` is the only model with a threshold-policy improvement that survives the current qualification rule.
- That improvement is small enough that it still needs walk-forward confirmation before any registry write or promotion decision.
- `short_ote_tcn_v2_candidate` remains the best short-side model overall, but threshold search did not improve it enough to replace its baseline policy.
- `TCN v1` remains a challenger pair, but neither side earned a policy upgrade here.
- Threshold search was directionally useful, but the major failure mode is not thresholding. It is the current abstain configuration.

## Recommended Next Steps

### 1. Run The Walk-Forward Policy Backtest Next

This is the correct next stage for this report.

Recommended models:

- `long_ote_tcn_v2_candidate`
- `short_ote_tcn_v2_candidate`
- `long_ote_tcn_v1_candidate`
- `short_ote_tcn_v1_candidate`

Recommended command:

```text
python scripts/run_ote_policy_backtest.py --regime-report-root model_testing/reports/ote_regime_slices/v1_v2_comparison --registry-path models/ote_model_registry_v1_v2_candidates.json --output-root model_testing/reports/ote_policy_backtests/v1_v2_tcn_focus --model-id long_ote_tcn_v2_candidate --model-id short_ote_tcn_v2_candidate --model-id long_ote_tcn_v1_candidate --model-id short_ote_tcn_v1_candidate --min-train-years 2 --test-window-months 3 --rolling-step-months 3 --min-folds 8 --min-positive-events 50 --min-events-per-month 3.0 --min-trades-per-week 3.0 --fixed-slippage-pips-per-trade 0.3 --commission-pips-per-trade 0.35
```

### 2. Do Not Write Registry Policy Updates Yet

Reason:

- This run did not produce strong enough evidence to justify locking in new production policy metadata.
- The backtest code re-searches thresholds fold by fold anyway, so backtesting can proceed without first writing these thresholds into the registry.

Practical guidance:

- wait for walk-forward confirmation first
- only then consider rerunning threshold search with `--write-registry-policies`, or manually copying the validated policy metadata

### 3. Carry Forward Two Distinct Hypotheses Into Backtest

Primary hypothesis:

- `long_ote_tcn_v2_candidate` can preserve its small regime-threshold advantage under walk-forward friction.

Secondary hypothesis:

- `short_ote_tcn_v2_candidate` is still the best short model, but its best deployable policy may remain the baseline global-threshold form.

### 4. Do Not Promote Any Abstain Policy From This Run

Reason:

- no abstain variant qualified
- every abstain variant sacrificed too much throughput and dollar value

If another iteration is needed after backtest, tune abstain policy separately from threshold policy.

### 5. If We Need A Follow-Up Threshold Search, Focus On Abstain Logic

The next tuning target should not be the threshold grid first. It should be the abstain configuration.

Most likely improvement areas:

- relax or disable some default hard abstain behavior
- reduce trade suppression in otherwise profitable high-confidence regimes
- test narrower regime-specific abstain filters instead of broad high-stress/off-hours suppression
- only consider registry-level abstain metadata after a new run beats baseline on both event quality and post-cost expectancy

## Bottom Line

- Best current long policy candidate: `long_ote_tcn_v2_candidate` with `regime_threshold`
- Best current short policy candidate: `short_ote_tcn_v2_candidate` with baseline `global_threshold`
- Best operational next step: walk-forward backtest on the four TCN models
- Clear no-go from this run: do not deploy the current abstain policy configuration


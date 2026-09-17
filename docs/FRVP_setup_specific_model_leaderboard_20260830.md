# FRVP Setup-Specific Model Leaderboard

Date: 2026-08-30

Scope: long setup-specific FRVP targets after the second experimentation round. Metrics are from saved `model_summary.csv` / `summary.json` outputs under `model_testing\reports\frvp_backtests`.

## Executive Read

No setup-specific model should be promoted into the accepted paper roster yet.

Two lanes are useful enough for quarantined shadow paper monitoring:

- `frvp_long_continuation_setup2_xgb_v1` with `frvp_long_setup2_medium_session_prune_20260829`.
- `frvp_long_continuation_setup5_xgb_v1` with `frvp_long_setup5_repeated_drawdown_prune_promotion_20260829`.

Use "shadow paper" here to mean log-only signal observation with separate reporting, no roster replacement, no combined portfolio gating, and no pass/fail credit toward production until a fresh forward window accumulates. Do not treat these as accepted paper-trading models because each still fails at least one important promotion gate.

## Best Variant Leaderboard

| Rank | Setup | Best Saved Variant | Folds | Trades | Net ticks | Expectancy | PF | Sharpe | DSR | Max DD | WFE | Gate Read | Shadow Paper |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | S5 continuation | `frvp_long_setup5_repeated_drawdown_prune_promotion_20260829` | 20 | 724 | +12966.40 | +17.91 | 1.391 | 0.912 | 0.859 | 38.59% | 1.723 | Fails drawdown and single-trade concentration | Yes, shadow only |
| 2 | S2 continuation | `frvp_long_setup2_medium_session_prune_20260829` | 9 | 104 | +4344.40 | +41.77 | 2.265 | 1.625 | 1.153 | 8.36% | -4.127 | Fails WFE and single-trade concentration | Yes, shadow only |
| 3 | S1 reversal recency recent2y | `frvp_long_reversal_setup_recency730_recent2y_20260829` | 8 | 173 | +1524.55 | +8.81 | 1.192 | 0.284 | 0.240 | 24.12% | -5.943 | Broad instability; fails most quality gates | No |
| 4 | S4 reversal research | `frvp_long_setup4_fullspan_1fold_research_20260829` | 1 | 5 | +392.75 | +78.55 | n/a | 4.409 | 0.000 | 0.00% | 8.152 | Research-only, one fold, 70.36% single-trade concentration | Log-only curiosity |
| 5 | S1 reversal full-span | `frvp_long_setup_fullspan_20260829` | 11 | 234 | +178.90 | +0.76 | 1.016 | 0.025 | 0.025 | 34.89% | -0.031 | Essentially flat after costs | No |
| 6 | S6 reversal recency full-window | `frvp_long_reversal_setup_recency730_opt_20260829` | 12 | 345 | -1961.25 | -5.68 | 0.870 | -0.342 | -0.291 | 33.40% | 0.368 | Negative realized economics | No |
| 7 | S6 reversal recency recent2y | `frvp_long_reversal_setup_recency730_opt_recent2y_20260829` | 8 | 192 | -2172.80 | -11.32 | 0.794 | -0.528 | -0.374 | 33.30% | 1.605 | Negative realized economics | No |
| 8 | S6 reversal full-span | `frvp_long_setup_fullspan_20260829` | 12 | 251 | -2897.15 | -11.54 | 0.766 | -0.523 | -0.513 | 37.17% | 0.621 | Negative realized economics | No |
| 9 | S3 continuation | `frvp_long_setup_fullspan_20260829` | 16 | 485 | -2964.25 | -6.11 | 0.887 | -0.379 | -0.375 | 49.96% | -0.562 | Negative realized economics | No |

## Full-Round Notes

S5 is the highest economic result from the setup-specific work. The policy-pruned promotion-quality run improved the original S5 from `+5255.95` ticks, Sharpe `0.281`, DSR `0.279`, max DD `82.26%`, WFE `1.110` to `+12966.40` ticks, Sharpe `0.912`, DSR `0.859`, max DD `38.59%`, WFE `1.723`. That is a real improvement, but the drawdown is still far outside the `12%` advisory gate and largest-single-trade concentration still fails.

S2 is cleaner than S5 from a risk-stat perspective after medium/session pruning: `+4344.40` ticks, PF `2.265`, Sharpe `1.625`, DSR `1.153`, and max DD `8.36%`. The problem is train-side instability: WFE is `-4.127` because mean train annualized PnL is negative while test annualized PnL is positive. This could be a genuine recent-regime edge, but it is exactly the kind of thing paper shadowing should test before promotion.

S1 is not currently worth shadow paper. The recent2y recency variant is positive, but only modestly, with low Sharpe, high drawdown, negative WFE, and weak composite breadth.

S4 should stay research-only. The one-fold report was positive, but it only produced `5` trades in one realized test quarter. One trade contributed `70.36%` of total PnL, so the result is not broad enough to trust.

S6 and S3 should stay paused/dropped. S6 stayed negative in full-window and recent2y optimized recency tests, and the S6 policy table still mostly fell back to the `0.05` global threshold because composite thresholds were not data-sufficient. S3 has good classifier optics but negative trading economics.

## Paper Trading Recommendation

Do not put any setup-specific model into the normal accepted paper roster yet.

Do run a quarantined shadow-paper watchlist if operationally cheap:

| Candidate | Shadow Mode | Why |
| --- | --- | --- |
| S2 pruned | Highest priority | Best risk-adjusted read, drawdown repaired, strong expectancy; needs forward evidence that the negative WFE is not just instability. |
| S5 pruned promotion-quality | Medium priority | Best absolute PnL and positive WFE; needs live observation because historical drawdown is still too high for roster inclusion. |
| S4 one-fold | Optional log-only | Interesting failed-auction signal, but too rare and too concentrated for anything more than event collection. |

Shadow-paper guardrails:

- Keep these signals out of the accepted paper roster and out of any production allocation.
- Tag them with explicit experiment IDs so their fills, misses, and abstains cannot pollute current model reporting.
- Review after a minimum forward window, not after the first good/bad week. Suggested minimums: S2 at least `30` shadow signals, S5 at least `100` shadow signals, S4 at least `20` events or two additional quarters, whichever comes later.
- Stop early only for operational defects or obvious cost/model-routing bugs, not ordinary variance.

## Backend Attribution Clarification

Backend attribution was not skipped for the setup-specific XGBoost prepared root.

The setup-aware Phase 4 root contains XGBoost attribution artifacts, including:

- `artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared\backend_attribution_summary.json`
- per-target `backend_attribution_summary_xgboost.json`
- per-target `feature_importance_merged_xgboost.csv`
- per-target `shap_feature_stats_xgboost.csv`

What we skipped was standalone TCN backend attribution and any TCN training. That was intentional because this experiment was scoped as an XGBoost-only target-routing/policy experiment:

- Wave 1 explicitly said: do not run TCNs.
- Setup targets inherited pooled-family quality, weights, and concurrency so the first comparison changed target membership and policy, not model family.
- Recency prepared roots copied the base prepared target files and only reweighted `train` / `val` sample weights; they did not need a fresh attribution pass because the feature universe and target rows were intentionally held constant for comparability.

Next attribution action: before any third-round retrain, review the existing per-target XGBoost attribution summaries for S2 and S5. Only rerun attribution if the next experiment changes the feature universe, target definition, backend family, or recency weighting policy enough that copied rankings are no longer a fair comparison.

## Recommended Next Steps

1. Create a shadow-paper bundle for S2 pruned and S5 pruned promotion-quality only.
2. Add explicit experiment IDs and reporting separation for setup-specific shadow signals.
3. Run an S2 train-window instability audit before retraining: focus on why train expectancy is negative while test is strongly positive.
4. Run an S5 date-regime audit around the 2022 drawdown before any further pruning or retraining.
5. Do not spend more training budget on S3 or S6 until new setup logic or feature context changes.
6. Keep S4 as event-collection research, not promotion research.

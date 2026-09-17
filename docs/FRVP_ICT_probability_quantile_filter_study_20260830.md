# FRVP / ICT Probability Quantile Filter Study

Date: 2026-08-30

## Plain-English Plan

Goal: test whether weak FRVP / ICT branches should ignore their weaker "yes" signals and only take trades when the model is unusually confident by its own standards.

Steps:

1. Use the existing `abstain_policy.minimum_probability_quantile` lever.
2. Try several filter strengths: no filter, `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, and `0.60`.
3. Focus on the weak-discrimination branches from the 2026-08-11 audit:
   - `frvp_short_meta_xgb_v1`
   - `frvp_long_meta_xgb_v1`
   - `ict_long_continuation_xgb_v1`
   - `ict_short_continuation_xgb_v1`
4. Compare trades, net ticks, expectancy, profit factor, drawdown, and concentration.
5. Only recommend a setting if fewer trades look meaningfully better, not just different.

## What Was Run

Two scripts were added:

- `scripts/run_probability_quantile_sweep_20260830.py`
- `scripts/analyze_probability_quantile_trade_screen_20260830.py`

Output root:

`model_testing\reports\probability_quantile_sweeps\frvp_ict_weak_discrimination_20260830`

The full walk-forward sweep was too slow to finish for all four models in one run. It completed partial reruns for `frvp_short_meta_xgb_v1` through `q30`. To finish the study efficiently, I also ran a faster trade-level screen across all four models using the saved historical emitted trades.

Important caveat: the trade-level screen answers, "what if we dropped the lower-confidence trades that already fired?" It is a useful first pass, but not final deployment proof. Before changing live policy, the winning setting should get a full candidate-level walk-forward rerun.

## Results

| Model | Best Screened Filter | Baseline | Best Screen | Plain-English Read |
| --- | --- | ---: | ---: | --- |
| `ict_long_continuation_xgb_v1` | `q50` | `247` trades, `+7.45` ticks | `119` trades, `+366.65` ticks | Strongest improvement. This model should talk less. |
| `ict_short_continuation_xgb_v1` | `q60` | `165` trades, `+500.75` ticks | `66` trades, `+530.10` ticks | Fewer trades, slightly more PnL, much better expectancy. Worth validating. |
| `frvp_short_meta_xgb_v1` | `q40` to `q60` looked better in the fast screen | `1523` trades, `+1829.05` ticks | q40: `910` trades, `+4498.50` ticks | Interesting, but mixed. The partial full walk-forward rerun only reached q30 and did not confirm a clean win before q40. |
| `frvp_long_meta_xgb_v1` | none | `3621` trades, `-4468.65` ticks | Still negative at every tested value | Do not spend more time here unless the model/target changes. |

## Partial Full Walk-Forward Check

For `frvp_short_meta_xgb_v1`, the heavier walk-forward rerun produced:

| Filter | Trades | Net ticks | Sharpe | Max DD | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | `1713` | `+1994.55` | `0.067` | `55.97%` | Weak |
| q10 | `1215` | `+20.25` | `0.001` | `55.58%` | Worse |
| q20 | `1042` | `-215.30` | `-0.010` | `53.37%` | Worse |
| q30 | `887` | `+2915.45` | `0.131` | `44.77%` | Better, but still weak |

This says `frvp_short_meta` may need a stricter value like q40/q50, but we should not trust the fast-screen q40/q60 result until a full rerun finishes.

## Full Candidate-Level ICT Validation

Completed after the initial screen:

`model_testing\reports\ict_backtests\ict_probability_quantile_candidate_sweep_20260830`

This reran the actual candidate-level walk-forward backtest for the promising ICT continuation settings.

| Model | Filter | Trades | Net ticks | Expectancy | PF | Sharpe | DSR | Max DD | WFE | Gate Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ict_long_continuation_xgb_v1` | q40 | `107` | `+499.45` | `+4.67` | `1.794` | `0.926` | `0.835` | `0.87%` | `0.984` | Fails concentration only |
| `ict_long_continuation_xgb_v1` | q50 | `88` | `+392.80` | `+4.46` | `1.694` | `0.746` | `0.699` | `1.10%` | `0.791` | Fails Sharpe and concentration |
| `ict_long_continuation_xgb_v1` | q60 | `75` | `+210.25` | `+2.80` | `1.421` | `0.521` | `0.505` | `1.42%` | `0.688` | Fails Sharpe and concentration |
| `ict_short_continuation_xgb_v1` | q50 | `53` | `+48.55` | `+0.92` | `1.057` | `0.070` | `0.070` | `4.06%` | `0.214` | Fails most quality gates |
| `ict_short_continuation_xgb_v1` | q60 | `47` | `+102.45` | `+2.18` | `1.151` | `0.211` | `0.210` | `3.05%` | `0.570` | Still weak; fails Sharpe, DSR, breadth, and concentration |

Baseline comparison:

- `ict_long_continuation_xgb_v1` baseline was basically flat: `247` trades, `+7.45` ticks, Sharpe `0.009`, DSR `0.009`, WFE `0.059`.
- q40 changed it into a real, small positive branch: fewer trades, better expectancy, good drawdown, and much better WFE.
- Stricter q50/q60 filters removed too many trades and made the long-continuation result weaker.
- `ict_short_continuation_xgb_v1` baseline was already modestly positive at `165` trades and `+500.75` ticks, but weak on Sharpe and breadth. The quantile filters reduced trade count without improving the branch enough.

Updated plain-English read:

- The quantile filter is useful for `ict_long_continuation_xgb_v1`, but the right setting is q40, not the stricter q50/q60 suggested by the fast trade screen.
- q40 means: skip the weakest 40% of candidate signals and only let the upper 60% compete for trades.
- q40 is not promotion-ready because one trade still contributes too much of the total profit: largest-trade share is `24.90%`, above the `10%` concentration gate.
- The filter does not rescue `ict_short_continuation_xgb_v1`. It becomes quieter, but not meaningfully better.

## ICT Long Continuation q40 Concentration Read

Executed next-step concentration review on 2026-08-30.

Largest winner:

- Entry: `2022-01-26 21:05:00+00:00`
- Exit: `2022-01-26 23:45:00+00:00`
- Source row: `355615`
- Pocket: `strong_down_high / new_york`
- Probability: `0.6734` against a `0.65` threshold
- Result: `+124.35` ticks net

Plain-English read: the largest winner looks like a one-off event, not a repeatable pocket by itself. The full `strong_down_high / new_york` pocket had `8` trades and `+134.80` ticks, but after removing this one trade it had only `+10.45` ticks left. So that pocket is not carrying strong repeat evidence.

The branch is not fake, though. If we remove only the largest winner:

| Case | Trades | Net ticks | Expectancy | Hit rate | Largest remaining trade share |
| --- | ---: | ---: | ---: | ---: | ---: |
| q40 as run | `107` | `+499.45` | `+4.67` | `57.94%` | `24.90%` |
| q40 without largest winner | `106` | `+375.10` | `+3.54` | `57.55%` | `15.82%` |

So the model still makes money without the biggest trade, but it still fails concentration. The next largest trade is `+59.35` ticks, which is still `15.82%` of the reduced total.

Breadth check:

- Years: positive in `2022`, `2024`, `2025`, and `2026`; negative in `2023`.
- Without the largest winner, `2022` remains positive at `+201.00` ticks.
- Sessions: Asia is the main repeatable pocket with `64` trades and `+324.40` ticks. New York drops from `+146.35` ticks to only `+22.00` ticks without the largest winner.
- Folds: `13/18` folds were profitable, matching the saved `0.722` profitable-fold share.
- Top-trade stack is still heavy: top 1 trade is `24.90%` of total PnL, top 3 are `47.26%`, and top 5 are `66.02%`.

Conclusion:

`ict_long_continuation_xgb_v1` q40 is a real improvement over baseline, but not clean enough for paper yet. The largest winner is a one-off, while the broader edge appears to live mostly in Asia and several smaller repeat pockets. The problem is not one bad pocket to delete; the problem is that the branch is still small and a handful of winners dominate total PnL.

Next action: keep q40 as a research candidate, not a paper candidate. A useful follow-up would be a q40 concentration-control pass focused on reducing large-trade dependence without wiping out the Asia repeat edge.

## ICT Long Continuation q40 Concentration-Control Pass

Output root:

`model_testing\reports\ict_backtests\ict_long_continuation_q40_concentration_controls_20260830`

This pass tested whether simple pocket filters could keep the useful q40 signal while lowering dependence on the biggest winners.

| Variant | Plain-English Filter | Trades | Net ticks | PF | Sharpe | DSR | Max DD | WFE | Largest-trade share | Gate Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `q40_block_largest_pocket` | Remove the observed largest-winner pocket | `98` | `+337.30` | `1.566` | `0.767` | `0.716` | `0.90%` | `1.069` | `17.60%` | Fails Sharpe and concentration |
| `q40_no_new_york` | Remove all New York trades | `60` | `+354.00` | `2.266` | `1.120` | `0.955` | `0.62%` | `1.532` | `16.77%` | Fails concentration only |
| `q40_asia_only` | Keep only Asia trades | `59` | `+321.65` | `2.136` | `1.027` | `0.902` | `0.62%` | `1.545` | `18.45%` | Fails concentration only |
| `q40_loss_pocket_prune` | Remove observed losing pockets, keep largest-winner pocket | `77` | `+612.95` | `2.934` | `1.405` | `1.062` | `0.70%` | `1.211` | `20.29%` | Fails concentration only |
| `q40_loss_pocket_prune_no_largest` | Remove losing pockets plus the largest-winner pocket | `70` | `+532.50` | `2.914` | `1.603` | `1.066` | `0.47%` | `1.665` | `13.59%` | Fails concentration only |

Findings:

- None of the simple controls passed the `10%` largest-trade concentration gate.
- Blocking only the largest-winner pocket was not enough. It cut profit and Sharpe, and the next largest winner still made up `17.60%` of total profit.
- Removing all New York trades made the branch cleaner in drawdown and profit factor, but it also left only `60` trades and still failed concentration.
- Asia-only confirms there is repeat signal outside the one-off New York winner, but it is still too dependent on a few large wins.
- The best economics came from `q40_loss_pocket_prune`: `+612.95` ticks, Sharpe `1.405`, DSR `1.062`, and all composites positive. The problem is that concentration stayed high at `20.29%`.
- The best balance was `q40_loss_pocket_prune_no_largest`: `+532.50` ticks, Sharpe `1.603`, DSR `1.066`, max drawdown `0.47%`, and concentration improved to `13.59%`. That is much better than `24.90%`, but still above the gate.

Plain-English conclusion:

The q40 branch got cleaner, but not clean enough. The filters show that the model has a real signal, especially in Asia and in selected non-loss pockets, but the branch is still too small and too dependent on a few big wins. More hand-pruning may make the backtest look nicer, but it increases the risk of overfitting to this exact historical tape.

Recommendation after this pass: do not move `ict_long_continuation_xgb_v1` q40 to paper trading yet. Keep it research-only or log-only. The next productive work is not another small pocket prune; it should be either forward observation with no execution risk, more out-of-sample evidence, or a target/exit redesign that can create more repeatable medium-sized winners.

## Recommendations

1. Keep `ict_long_continuation_xgb_v1` q40 as the only meaningful winner from this filter study.
2. Do not promote q40 yet; keep it research-only or log-only because concentration still fails after the control pass.
3. Stop simple pocket-pruning for q40 for now. The best concentration-control variant improved largest-trade share from `24.90%` to `13.59%`, but it still missed the `10%` gate.
4. Stop quantile tuning for `ict_short_continuation_xgb_v1` for now. q50/q60 did not produce a strong enough branch.
5. Put `frvp_short_meta_xgb_v1` behind ICT long continuation; rerun only q40/q50/q60 if we still want to pursue this lever after the ICT q40 control pass.
6. Drop `frvp_long_meta_xgb_v1` from this quantile-filter experiment. The filter did not rescue it.

Layman summary: this filter is useful when a model has occasional good ideas buried inside too much noise. It appears most useful for ICT continuation. It does not magically fix a model that is broadly wrong, which is what the FRVP long-meta result looks like here.

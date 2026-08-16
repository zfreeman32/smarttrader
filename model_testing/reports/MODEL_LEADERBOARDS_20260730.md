# Model Leaderboards

Generated on `2026-07-30` from the saved training and walk-forward artifacts for OTE, FRVP, and ICT models.

The previous full OTE snapshot remains available in `MODEL_LEADERBOARDS_20260527.md`. This update adds the completed 1-minute OTE, breakout-sustained, FRVP, and leakage-safe ICT experiments without treating unlike instruments, bar sizes, labels, and evaluation windows as one directly comparable score.

## Ranking Method

- Models are ranked only inside comparable cohorts.
- Training tables rank by mean CV `average_precision`, with mean CV event `F0.5` as the tiebreaker.
- Post-training tables put accepted/passing branches first, then use monthly Sharpe, walk-forward efficiency (`WFE`), profitable-quarter breadth, and concentration as the main ordering evidence.
- OTE results use EUR/USD pips. FRVP and ICT results use ES ticks. These units must not be compared directly.
- The older OTE summaries use a six-gate contract. The current FRVP and ICT summaries use an eight-gate promotion-quality contract, including the newer concentration and account-drawdown checks.
- A recent-regime or short-window result is marked explicitly and is not promoted above a full-span result solely because its Sharpe is higher.

## Current Promotion-Relevant Leaders

| Cohort | Model / policy | Scope | Gates | Trades | Net PnL | Sharpe | WFE | Current status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OTE EUR/USD 5-minute | `short_reversal_xgb_v2_20260525` | Full saved OTE walk-forward | 6/6 | 561 | +5453.65 pips | 3.615 | 0.729 | Existing 5-minute OTE leader; no newer 5-minute experiment displaced it |
| OTE EUR/USD 1-minute | `short_reversal_xgb_1m_v1` | 2021-10 through 2026-03 | 6/6 | 3142 | +31943.90 pips | 2.872 | 0.885 | Strongest broad-window 1-minute result; still a separate candidate lane |
| OTE EUR/USD 1-minute | `long_reversal_tcn_1min_runpod_20260612_candidate` | 2024-01 through 2026-03 | 6/6 | 2295 | +19128.65 pips | 2.676 | 2.889 | Strong long 1-minute candidate; the higher-Sharpe one-year rerun is diagnostic, not the primary evidence |
| FRVP ES 5-minute | `frvp_long_continuation_xgb_v1` with `v3` policy | Full-span | 8/8 | 524 | +7190.40 ticks | 1.226 | 2.459 | Accepted promotion-near FRVP baseline |
| FRVP ES 5-minute | `frvp_long_reversal_xgb_recent_regime_prune_v2` | Recent-regime selective lane | 8/8 | 63 | +3677.05 ticks | 1.480 | 2.162 | Accepted selective-deployment reversal contract |
| ICT ES 5-minute | `ict_long_meta_xgb_v1` | Latest leakage-safe full walk-forward | 8/8 | 1555 | +7461.25 ticks | 1.586 | 1.997 | Current ICT research leader; packaged for shadow only |
| ICT ES 5-minute | `ict_long_reversal_xgb_v1` | Latest leakage-safe full walk-forward | 8/8 | 1275 | +5471.25 ticks | 1.269 | 3.013 | Accepted research branch; packaged for shadow only |

## New OTE Training Experiments

### EUR/USD 1-Minute

| Rank | Artifact | Backend | Folds | CV Mean AP | CV Mean Event F0.5 | Test AP | Test Event F0.5 | Post-training read |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `runpod_1min_tcn_long_ote_20260615_002054/long_ote` | TCN | 5 | 0.767551 | 0.491293 | 0.863161 | 0.522754 | Training leader; no saved walk-forward policy backtest yet |
| 2 | `runpod_1min_tcn_long_reversal_20260612_171301/long_reversal` | TCN | 6 | 0.561999 | 0.447933 | 0.677881 | 0.434947 | Passed 6/6 downstream gates on the saved broad and strict windows |
| 3 | `ote_1min_reversal_wave1_short_xgb/short_reversal` | XGBoost | 18 | 0.498387 | 0.674257 | 0.531847 | 0.578512 | Best broad-window 1-minute economics result |
| 4 | `runpod_1min_tcn_short_reversal_20260614_193525/short_reversal` | TCN | 6 | 0.490051 | 0.446595 | 0.645085 | 0.413163 | Holdout AP improved over the XGBoost branch, but no saved walk-forward result promotes it |

### EUR/USD 5-Minute Breakout Improvements

| Rank | Artifact | Backend | Folds | CV Mean AP | CV Mean Event F0.5 | Test AP | Test Event F0.5 | Post-training read |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `experiment_a_breakout_sustained_pilot_20260621/long_breakout_sustained` | TCN | 3 | 0.982097 | 0.976017 | 0.996278 | 0.987699 | Failed downstream economics: 2/6 gates, -2646.05 pips, Sharpe -1.091 |
| 2 | `ote_multi_family_tcn_refresh_20260530/short_breakout` | TCN | 2 | 0.921215 | 0.878271 | 0.987240 | 0.852843 | Did not repair the short-breakout promotion case; latest event-metadata rerun remains post-cost negative |

The sustained-breakout pilot is now the clearest warning against using classifier metrics as the promotion leaderboard: it has the highest saved OTE CV AP in the repository but negative downstream economics.

## OTE 1-Minute Post-Training Leaderboard

| Rank | Model / run | Evaluation window | Gates | Trades | Net PnL (pips) | Sharpe | Profit Factor | WFE | Profitable Quarters | Read |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `long_reversal_tcn_1min_runpod_20260612_candidate` strict | 2025-04 through 2026-03 | 6/6 | 1856 | +17088.30 | 5.728 | 5.263 | 4.014 | 1.000 | Excellent one-year diagnostic, but too short to be the primary promotion record |
| 2 | `short_reversal_xgb_1m_v1` | 2021-10 through 2026-03 | 6/6 | 3142 | +31943.90 | 2.872 | 2.287 | 0.885 | 1.000 | Best broad-window 1-minute candidate |
| 3 | `long_reversal_tcn_1min_runpod_20260612_candidate` broad | 2024-01 through 2026-03 | 6/6 | 2295 | +19128.65 | 2.676 | 4.409 | 2.889 | 0.889 | Stronger robustness evidence than the strict one-year rerun |
| 4 | `long_breakout_sustained_tcn_expA_20260621_pilot` | 2023-04 through 2026-03 | 2/6 | 789 | -2646.05 | -1.091 | 0.737 | 0.652 | 0.250 | Reject; high AP did not convert to a tradeable policy |

## FRVP Research Leaderboard

This table carries forward the current saved FRVP keep-set. Full-span and recent-regime branches remain explicitly distinguished.

| Rank | Branch | Saved checkpoint / contract | Scope | Gates | Trades | Net PnL (ticks) | Sharpe | DSR | WFE | Current read |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Long continuation | `frvp_long_continuation_gatefix_v3_20260715_accountdd` | Full-span | 8/8 | 524 | +7190.40 | 1.226 | 1.061 | 2.459 | Best promotion-near direct FRVP checkpoint |
| 2 | Long reversal, recent regime | `frvp_long_reversal_xgb_recent_regime_prune_v2` | Recent-regime selective lane | 8/8 | 63 | +3677.05 | 1.480 | 1.179 | 2.162 | Accepted operational reversal contract |
| 3 | Long reversal | `frvp_long_reversal_recency_trial1_v3_20260715_accountdd` | Full-span recency weighted | 7/8 | 140 | +6027.00 | 1.061 | 0.956 | -3.797 | Best full-span reversal control; blocked by train-side stability |
| 4 | Long meta | `frvp_long_meta_gatefix_v3_20260715_accountdd` | Full-span | 5/8 | 603 | +9216.05 | 0.595 | 0.577 | 1.897 | Profitable research control; Sharpe, concentration, and drawdown remain weak |
| 5 | Long meta recency sentinel | `frvp_long_meta_recency_trial1_gatefix_v3_20260705` | Full-span recency weighted | Fail | 599 | +6938.65 | 0.478 | 0.469 | 1.974 | Weaker than the refresh `v3`; no family-wide recency rollout |
| 6 | Long meta post-2022 control | `frvp_es_primary_post2022_smallcv_20260703` | Post-2022 only | Fail | 1615 | +9347.25 | 0.647 | 0.624 | -3.532 | Hard history truncation did not fix robustness |

Canonical FRVP detail remains in `docs/FRVP_ES_primary_audit_report.md`, `docs/FRVP_experiment_journal.md`, and `docs/FRVP_operational_contract_20260721.md`.

## ICT Leakage-Safe Training Improvement

Baseline is `models/ict_es_primary_xgb_baseline_20260712_v2`. New is `models/ict_es_primary_xgb_bootstrap_20260726_full`, trained from the embargo-safe prepared root with ICT sequential bootstrap.

| Target | CV AP: base -> new | Delta | CV F0.5: base -> new | Test AP: base -> new | Test F0.5: base -> new |
| --- | ---: | ---: | ---: | ---: | ---: |
| `long_ict_meta` | 0.429046 -> 0.520778 | +0.091732 | 0.751544 -> 0.820435 | 0.460894 -> 0.492966 | 0.732946 -> 0.788423 |
| `long_ict_reversal` | 0.421209 -> 0.467281 | +0.046071 | 0.746927 -> 0.806525 | 0.380377 -> 0.450214 | 0.696662 -> 0.760870 |
| `short_ict_meta` | 0.406901 -> 0.438125 | +0.031224 | 0.727337 -> 0.775363 | 0.409874 -> 0.451301 | 0.679739 -> 0.754310 |
| `long_ict_continuation` | 0.544115 -> 0.563803 | +0.019688 | 0.844915 -> 0.891806 | 0.459948 -> 0.519435 | 0.842105 -> 0.833333 |
| `short_ict_continuation` | 0.518481 -> 0.530427 | +0.011946 | 0.965265 -> 0.922914 | 0.561203 -> 0.509766 | 0.840336 -> 0.841837 |
| `short_ict_reversal` | 0.417800 -> 0.418708 | +0.000908 | 0.718851 -> 0.743417 | 0.366322 -> 0.420523 | 0.638889 -> 0.760000 |

All six targets improved CV AP. Five of six improved holdout AP; `short_ict_continuation` was the exception. The improvement is meaningful, but post-training economics and concentration still decide promotion.

## Current ICT Post-Training Leaderboard

Source: `model_testing/reports/ict_backtests/ict_es_primary_bootstrap_20260726_full`.

| Rank | Model | Gates | Trades | Net PnL (ticks) | Sharpe | DSR | Profit Factor | WFE | Profitable Quarters | Max DD | Largest Trade / Net | Current read |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `ict_long_meta_xgb_v1` | 8/8 | 1555 | +7461.25 | 1.586 | 1.068 | 1.510 | 1.997 | 0.667 | 6.40% | 5.60% | Accepted; current ICT leader |
| 2 | `ict_long_reversal_xgb_v1` | 8/8 | 1275 | +5471.25 | 1.269 | 1.023 | 1.453 | 3.013 | 0.667 | 8.40% | 5.20% | Accepted; strongest WFE in the current roster |
| 3 | `ict_short_reversal_xgb_v1` | 6/8 | 429 | +3052.15 | 1.047 | 0.914 | 1.638 | 1.618 | 0.556 | 7.82% | 11.12% | Fails quarter breadth and single-trade concentration |
| 4 | `ict_short_meta_xgb_v1` | 6/8 | 1082 | +3906.70 | 0.905 | 0.820 | 1.300 | 1.442 | 0.593 | 7.13% | 10.02% | Fails quarter breadth and single-trade concentration |
| 5 | `ict_short_continuation_xgb_v1` | 4/8 | 165 | +500.75 | 0.326 | 0.322 | 1.199 | 0.858 | 0.438 | 7.01% | 78.15% | Positive only through a highly concentrated winner set |
| 6 | `ict_long_continuation_xgb_v1` | 3/8 | 247 | +7.45 | 0.009 | 0.009 | 1.003 | 0.059 | 0.667 | 5.06% | 1669.13% | Effectively flat; raw economics fail before concentration is meaningful |

The leakage-safe retrain did not create a general short-side winner. The two short leaders share `284` exact walk-forward events contributing `+2597.40` ticks to each branch, so the next ICT improvement should remain a frozen-model regime/concentration and roster-overlap study rather than another generic retrain.

## Promotion Decisions

- Keep the existing 5-minute OTE reversal core unchanged. None of the new 5-minute breakout experiments displaced it.
- Keep the 1-minute OTE models in a separate research lane and do not deploy them while OTE Live has no 1-minute stream. `short_reversal_xgb_1m_v1` has the strongest broad-window evidence; the long-reversal TCN is also strong, but its best headline result is from only one year.
- Treat FRVP long continuation `v3` and the recent-regime long-reversal `v2` contract as the accepted FRVP leaders.
- Treat ICT long meta and long reversal as the only promotion-eligible branches from the latest leakage-safe roster.
- Keep ICT short meta and short reversal in shadow/research until concentration and overlapping-event dependence improve.
- Reject the sustained-breakout pilot for promotion despite its exceptional classifier metrics.

## OTE Live Deployment Decision

| View | Runtime bundle | Long/short coverage | Decision |
| --- | --- | --- | --- |
| OTE | Existing EUR/USD 5-minute manifests | Four models per direction: reversal, union, meta, and breakout | No model or policy change |
| FRVP | `frvp_es_shadow_20260721` | One long and one short continuation, reversal, and meta model | No model or policy change; retain all six as candidate/shadow |
| ICT | `ict_es_shadow_20260730` | One long and one short continuation, reversal, and meta model | Replace the July 13 baseline bundle with all six leakage-safe July 26 retrain branches; retain candidate/shadow status |

The ICT shadow bundle intentionally does not promote the two 8/8 long branches
to active delivery. The saved static policy for long meta differs from the
walk-forward dominant policy, and both accepted long branches need
packaged-policy shadow parity before active promotion. The 5-minute-only
runtime selection also keeps every 1-minute artifact out of the live
manifests.

# Model Leaderboards

> Historical OTE snapshot. The current cross-family update is `MODEL_LEADERBOARDS_20260730.md`.

Generated on `2026-05-27` from saved artifacts in `models/**/cv_fold_manifest.json` and `model_testing/reports/ote_policy_backtests/**/summary.json`.

## Ranking Method

- Training leaderboard: ranked all `42` model artifacts by mean CV `average_precision`, with mean CV event `F0.5` as the tiebreaker.
- Post-training leaderboard: collapsed `63` saved backtests into the best saved run for each of `19` unique `model_id`s, then ranked by post-cost profitability flag, acceptance gates passed, monthly Sharpe, annualized net pips, and profit factor.

## Full Training Leaderboard

| Rank | Artifact | Family | Folds | CV Mean AP | CV Mean Event F0.5 | CV Mean ROC AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `champion_models/breakout_models/long_breakout_tcn_challenger` | `champion_models` | 2 | 0.968597 | 0.943288 | 0.999327 |
| 2 | `champion_models/breakout_models/short_breakout_tcn_champion` | `champion_models` | 2 | 0.962804 | 0.932696 | 0.999243 |
| 3 | `champion_models/breakout_models/long_breakout_tcn_champion/long_breakout` | `champion_models` | 3 | 0.956475 | 0.926166 | 0.999067 |
| 4 | `champion_models/breakout_models/long_breakout_tcn_repair_20260511_v1/long_breakout` | `champion_models` | 3 | 0.950439 | 0.875437 | 0.998995 |
| 5 | `champion_models/breakout_models/long_breakout_tcn_repair_20260513_v2/long_breakout` | `champion_models` | 3 | 0.950439 | 0.875437 | 0.998995 |
| 6 | `ote_multi_family_xgb_v1/long_continuation_entry` | `ote_multi_family_xgb_v1` | 2 | 0.935666 | 0.925758 | 0.998829 |
| 7 | `champion_models/breakout_models/long_breakout_xgb_v1` | `champion_models` | 2 | 0.932928 | 0.919254 | 0.998072 |
| 8 | `legacy_models/short_breakout_v2` | `legacy_models` | 3 | 0.924965 | 0.892155 | 0.998473 |
| 9 | `ote_multi_family_xgb_v1/short_breakout` | `ote_multi_family_xgb_v1` | 2 | 0.923411 | 0.906244 | 0.997843 |
| 10 | `legacy_models/short_breakout_challenger_v1_neighborhood_isotonic_20260506/short_breakout` | `legacy_models` | 3 | 0.916854 | 0.883077 | 0.998340 |
| 11 | `legacy_models/short_breakout_challenger_v1_neighborhood_platt_20260506/short_breakout` | `legacy_models` | 3 | 0.916854 | 0.883077 | 0.998340 |
| 12 | `champion_models/meta_models/long_tcn_champion` | `champion_models` | 3 | 0.869369 | 0.879057 | 0.973708 |
| 13 | `champion_models/meta_models/short_tcn_repair_20260511_v2/short_ote` | `champion_models` | 3 | 0.867386 | 0.880367 | 0.975783 |
| 14 | `long_ote_union_tcn_candidate_20260523/long_ote` | `long_ote_union_tcn_candidate_20260523` | 3 | 0.861896 | 0.885093 | 0.972721 |
| 15 | `short_ote_union_tcn_candidate_20260520/short_ote` | `short_ote_union_tcn_candidate_20260520` | 3 | 0.861455 | 0.876290 | 0.974121 |
| 16 | `champion_models/meta_models/short_tcn_champion` | `champion_models` | 3 | 0.856914 | 0.869772 | 0.975011 |
| 17 | `ote_multi_family_xgb_v1/long_ote` | `ote_multi_family_xgb_v1` | 2 | 0.790322 | 0.825132 | 0.966174 |
| 18 | `ote_multi_family_xgb_v1/short_ote` | `ote_multi_family_xgb_v1` | 2 | 0.773068 | 0.811962 | 0.963969 |
| 19 | `challenger_models/reversal_models/short_reversal_xgb_v2_20260525/short_reversal` | `challenger_models` | 3 | 0.694484 | 0.795344 | 0.973210 |
| 20 | `champion_models/reversal_models/short_reversal_xgb_v1` | `champion_models` | 2 | 0.683392 | 0.797314 | 0.972645 |
| 21 | `challenger_models/reversal_models/long_reversal_tcn_v2_20260525_narrow48/long_reversal` | `challenger_models` | 3 | 0.663246 | 0.763491 | 0.966971 |
| 22 | `challenger_models/reversal_models/long_reversal_tcn_v3_20260526_overlap_ny_dd_repair/long_reversal` | `challenger_models` | 3 | 0.653173 | 0.768331 | 0.962270 |
| 23 | `ote_multi_family_xgb_v1/short_continuation_entry` | `ote_multi_family_xgb_v1` | 2 | 0.650896 | 0.740767 | 0.992236 |
| 24 | `champion_models/reversal_models/long_reversal_tcn_challenger` | `champion_models` | 3 | 0.650884 | 0.764150 | 0.965368 |
| 25 | `champion_models/reversal_models/long_reversal_tcn_champion` | `champion_models` | 2 | 0.628486 | 0.774910 | 0.965257 |
| 26 | `ote_full_tcn_v2/long_ote` | `ote_full_tcn_v2` | 8 | 0.625416 | 0.773805 | 0.961238 |
| 27 | `ote_full_tcn_v2/short_ote` | `ote_full_tcn_v2` | 8 | 0.623925 | 0.747349 | 0.957147 |
| 28 | `champion_models/reversal_models/short_reversal_tcn_champion` | `champion_models` | 2 | 0.580101 | 0.707691 | 0.954236 |
| 29 | `ote_multi_family_xgb_v1/long_reversal` | `ote_multi_family_xgb_v1` | 2 | 0.571142 | 0.702358 | 0.967150 |
| 30 | `champion_models/reversal_models/short_reversal_tcn_challenger/short_reversal` | `champion_models` | 3 | 0.569423 | 0.692066 | 0.955722 |
| 31 | `ote_full_tcn_v1/long_ote` | `ote_full_tcn_v1` | 8 | 0.569290 | 0.723085 | 0.949258 |
| 32 | `ote_full_xgb_v2/short_ote` | `ote_full_xgb_v2` | 8 | 0.554363 | 0.684545 | 0.958649 |
| 33 | `ote_full_xgb_v1/short_ote` | `ote_full_xgb_v1` | 8 | 0.547941 | 0.700760 | 0.959550 |
| 34 | `ote_full_xgb_v1/long_ote` | `ote_full_xgb_v1` | 8 | 0.547554 | 0.706759 | 0.961961 |
| 35 | `ote_full_lstm_v1/long_ote` | `ote_full_lstm_v1` | 8 | 0.518598 | 0.690700 | 0.947020 |
| 36 | `ote_full_xgb_v2/long_ote` | `ote_full_xgb_v2` | 8 | 0.513801 | 0.637165 | 0.961435 |
| 37 | `ote_full_tcn_v1/short_ote` | `ote_full_tcn_v1` | 8 | 0.497507 | 0.667450 | 0.946955 |
| 38 | `ote_multi_family_tcn_v1/short_continuation_pullback` | `ote_multi_family_tcn_v1` | 2 | 0.304618 | 0.504349 | 0.960559 |
| 39 | `ote_multi_family_tcn_v1/long_continuation_pullback` | `ote_multi_family_tcn_v1` | 2 | 0.293755 | 0.465285 | 0.959182 |
| 40 | `ote_multi_family_tcn_v2/short_continuation_pullback` | `ote_multi_family_tcn_v2` | 3 | 0.279318 | 0.459477 | 0.964270 |
| 41 | `ote_full_lstm_v2/long_ote` | `ote_full_lstm_v2` | 8 | 0.277039 | 0.433646 | 0.916098 |
| 42 | `ote_multi_family_tcn_v2/long_continuation_pullback` | `ote_multi_family_tcn_v2` | 3 | 0.249718 | 0.459439 | 0.960173 |

## Full Post-Training Leaderboard

| Rank | Model ID | Direction | Backend | Gates Passed | Monthly Sharpe | Annualized Net Pips | Profit Factor | WFE | Profitable Quarter Share | Positive Composite Share | Best Summary |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `short_reversal_xgb_v2_20260525` | `short` | `xgboost` | 6/6 | 3.614743 | 2728.693 | 2.740156 | 0.729451 | 1.000000 | 0.875000 | `model_testing/reports/ote_policy_backtests/short_reversal_xgb_v2_20260525/short_reversal_xgb_v2_20260525/summary.json` |
| 2 | `long_reversal_tcn_v2_20260525_narrow48` | `long` | `tcn` | 6/6 | 3.356343 | 2736.148 | 2.802304 | 0.821250 | 1.000000 | 0.875000 | `model_testing/reports/ote_policy_backtests/long_reversal_tcn_v2_20260525_narrow48/long_reversal_tcn_v2_20260525_narrow48/summary.json` |
| 3 | `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair` | `long` | `tcn` | 6/6 | 3.281066 | 3557.985 | 2.581914 | 0.796016 | 0.875000 | 1.000000 | `model_testing/reports/ote_policy_backtests/long_reversal_tcn_v3_20260526_overlap_ny_dd_repair/long_reversal_tcn_v3_20260526_overlap_ny_dd_repair/summary.json` |
| 4 | `short_reversal_xgb_v1` | `short` | `xgboost` | 6/6 | 2.910341 | 1004.187 | 2.966201 | 1.038424 | 0.750000 | 0.750000 | `model_testing/reports/ote_policy_backtests/short_reversal_xgb_rm_london_hard_prune_20260525/short_reversal_xgb_v1/summary.json` |
| 5 | `short_reversal_tcn_champion` | `short` | `tcn` | 6/6 | 2.782246 | 2731.950 | 2.771188 | 2.003400 | 0.800000 | 1.000000 | `model_testing/reports/ote_policy_backtests/champion_models_20260508_v2_min5/short_reversal_tcn_champion/summary.json` |
| 6 | `long_reversal_tcn_champion` | `long` | `tcn` | 6/6 | 2.759203 | 2463.912 | 3.072065 | 0.801333 | 1.000000 | 1.000000 | `model_testing/reports/ote_policy_backtests/champion_models_20260508_v2_min5/long_reversal_tcn_champion/summary.json` |
| 7 | `short_ote_meta_tcn_champion` | `short` | `tcn` | 6/6 | 2.700026 | 1736.628 | 1.912748 | 0.988834 | 0.888889 | 0.800000 | `model_testing/reports/ote_policy_backtests/short_ote_meta_tcn_repair_20260511_v2_regime_prune_v1/short_ote_meta_tcn_champion/summary.json` |
| 8 | `long_ote_union_tcn_candidate_20260523` | `long` | `tcn` | 6/6 | 2.403597 | 2831.213 | 1.720431 | 1.216258 | 0.875000 | 0.888889 | `model_testing/reports/ote_policy_backtests/long_ote_union_prune_v1_pairs_20260525/long_ote_union_tcn_candidate_20260523/summary.json` |
| 9 | `short_ote_union_tcn_candidate_20260520` | `short` | `tcn` | 6/6 | 1.872381 | 1241.250 | 1.731540 | 1.103072 | 0.750000 | 0.800000 | `model_testing/reports/ote_policy_backtests/short_ote_union_regime_prune_v1_20260521/short_ote_union_tcn_candidate_20260520/summary.json` |
| 10 | `long_ote_tcn_v2_candidate` | `long` | `tcn` | 5/6 | 2.360779 | 3954.731 | 1.956244 | 1.977755 | 0.814815 | 0.777778 | `model_testing/reports/ote_policy_backtests/multifamily_live_v1/long_ote_tcn_v2_candidate/summary.json` |
| 11 | `long_ote_tcn_v1_candidate` | `long` | `tcn` | 5/6 | 2.225222 | 2932.728 | 1.904755 | 1.928717 | 0.814815 | 0.875000 | `model_testing/reports/ote_policy_backtests/v1_v2_tcn_focus/long_ote_tcn_v1_candidate/summary.json` |
| 12 | `long_ote_champion_v1` | `long` | `tcn` | 5/6 | 2.195336 | 2893.658 | 1.892054 | 1.901480 | 0.814815 | 0.750000 | `model_testing/reports/ote_policy_backtests/full_run_v1/long_ote_champion_v1/summary.json` |
| 13 | `short_ote_tcn_v2_candidate` | `short` | `tcn` | 5/6 | 1.900539 | 2524.804 | 1.700306 | 1.657368 | 0.777778 | 1.000000 | `model_testing/reports/ote_policy_backtests/multifamily_live_v1/short_ote_tcn_v2_candidate/summary.json` |
| 14 | `short_ote_tcn_v1_candidate` | `short` | `tcn` | 5/6 | 1.333977 | 1698.450 | 1.424885 | 8.565467 | 0.685185 | 0.750000 | `model_testing/reports/ote_policy_backtests/v1_v2_tcn_focus/short_ote_tcn_v1_candidate/summary.json` |
| 15 | `short_ote_candidate_tcn_v1` | `short` | `tcn` | 5/6 | 1.307833 | 1673.693 | 1.412550 | 9.228640 | 0.666667 | 0.750000 | `model_testing/reports/ote_policy_backtests/full_run_v1/short_ote_candidate_tcn_v1/summary.json` |
| 16 | `long_ote_meta_tcn_champion` | `long` | `tcn` | 4/6 | 1.010648 | 1709.001 | 1.240908 | 1.065332 | 0.555556 | 0.666667 | `model_testing/reports/ote_policy_backtests/champion_models_20260508_v2/long_ote_meta_tcn_champion/summary.json` |
| 17 | `long_breakout_tcn_champion` | `long` | `tcn` | 4/6 | 0.811649 | 1409.953 | 1.226200 | 23.603992 | 0.400000 | 0.666667 | `model_testing/reports/ote_policy_backtests/long_breakout_tcn_repair_20260513_v2_regime_prune_v1_q20_asia/long_breakout_tcn_champion/summary.json` |
| 18 | `short_breakout_tcn_champion` | `short` | `tcn` | 2/6 | -0.946546 | -2502.664 | 0.800269 | 1.627507 | 0.333333 | 0.333333 | `model_testing/reports/ote_policy_backtests/champion_models_20260508_v2_min5/short_breakout_tcn_champion/summary.json` |
| 19 | `long_breakout_xgb_v1` | `long` | `xgboost` | 1/6 | -0.413638 | -584.791 | 0.913013 | 0.856226 | 0.250000 | 0.222222 | `model_testing/reports/ote_policy_backtests/breakout_reversal_xgb_20260509/long_breakout_xgb_v1/summary.json` |

## Observations

- The training leaderboard is dominated by breakout-family models: `7` of the top `10` training artifacts are breakout variants, and the top `5` are all breakout artifacts.
- That edge does not carry through post-training. The best post-training breakout result is `long_breakout_tcn_champion`, which only lands at post-training rank `17` with `4/6` gates and Sharpe `0.812`. `short_breakout_tcn_champion` and `long_breakout_xgb_v1` are both post-cost unprofitable in their best saved runs.
- The post-training winners are mostly reversal and OTE/union models. The top `6` post-training slots are all reversal models, and each of those top `6` passed all `6` acceptance gates.
- Several of the best live-like performers were only mid-pack or lower in training rank: `short_reversal_xgb_v2_20260525` -> training rank `19`, `long_reversal_tcn_v2_20260525_narrow48` -> training rank `21`, `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair` -> training rank `22`, `short_reversal_xgb_v1` -> training rank `20`, `short_reversal_tcn_champion` -> training rank `28`, `long_reversal_tcn_champion` -> training rank `25`.
- `long_ote_tcn_v2_candidate` is the best pure PnL outlier in post-training with annualized net pips `3954.7`, but it passed `5/6` gates rather than `6/6`, so it is a stronger return candidate than robustness candidate.
- The strongest post-training Sharpe belongs to `short_reversal_xgb_v2_20260525` at `3.615`, while the strongest profit factor belongs to `long_reversal_tcn_champion` at `3.072`.
- A lot of the highest training scores come from smaller fold counts. `4` of the top `10` training artifacts were evaluated on only `2` folds, which makes the CV leaderboard more fragile than the post-training table.

## Recommendations

- Use the post-training leaderboard, not the raw training leaderboard, as the primary promotion surface. The saved data shows that CV AP alone is overvaluing breakout-family artifacts.
- For near-term promotion and retrain priority, anchor on the six-model post-training core: `short_reversal_xgb_v2_20260525`, `long_reversal_tcn_v2_20260525_narrow48`, `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair`, `short_reversal_xgb_v1`, `short_reversal_tcn_champion`, and `long_reversal_tcn_champion`.
- Keep `long_ote_tcn_v2_candidate` and `long_ote_tcn_v1_candidate` in the active challenger set because they still produce strong net pips and respectable Sharpe, but treat them as controlled-risk candidates until the missing gate is fixed.
- De-emphasize breakout models in the next retrain wave unless you also tighten the post-training objective. Their training scores are excellent, but the saved backtests show poor conversion into robust post-cost performance.
- Raise the selection bar on future training sweeps by adding minimum fold-count and post-training proxy constraints before a model is considered `top tier`. A practical rule would be: at least `3` folds, then require a downstream policy backtest pass before promotion.

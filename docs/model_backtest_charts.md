# Model leaderboards and backtest charts

Realized net P&L charts: the backtest atlas for every trained model with trade-level output, then the per-family charts for the bundles the live dashboard loads. Images are under `docs/assets/`.

Contents:
- [Backtest atlas: realized net P&L for every trained model](#model_backtest_atlas_20260927)
- [Realized net P&L by model family](#model_realized_pnl_charts_20260927)


<a id="model_backtest_atlas_20260927"></a>

## Backtest atlas: realized net P&L for every trained model

*Merged 2026-10-09 from `docs/model_backtest_atlas_20260927.md`.*

Generated 2026-09-27. Every backtest under `model_testing/reports` that produced trade-level output, including experiments, sweeps and superseded runs.

Each line is the running sum of per-trade net P&L from a `selected_test_trades.csv`, ordered by trade exit time. Out-of-sample folds only, after modelled spread, slippage and commission. One panel per experiment run; the legend gives each curve's trade count and final total. Flat stretches are gaps between rolling folds. Panels tagged `[LIVE]` feed a bundle the app currently loads.

OTE is EURUSD and reports pips. FRVP and ICT are ES and report ticks. Totals are not comparable across instruments, and experiment runs use different cost and fold settings, so curves are only directly comparable within a panel.

### Coverage

| Research line | Family | Curves | Panels |
|---|---|---:|---:|
| OTE | Reversal | 23 | 19 |
| OTE | Meta | 10 | 7 |
| OTE | Union | 8 | 7 |
| OTE | Breakout | 22 | 20 |
| OTE | General | 10 | 5 |
| FRVP | Reversal | 53 | 32 |
| FRVP | Continuation | 23 | 15 |
| FRVP | Meta | 16 | 11 |
| ICT | Reversal | 43 | 15 |
| ICT | Continuation | 33 | 15 |
| ICT | Meta | 23 | 12 |
| **Total** | | **264** | **158** |

Per-curve stats for all 264 curves are in [`backtest_curve_index.csv`](assets/backtest_atlas/backtest_curve_index.csv).

### OTE Reversal

23 curves across 19 panels.

![OTE Reversal page 1](assets/backtest_atlas/ote_reversal_p1.png)

![OTE Reversal page 2](assets/backtest_atlas/ote_reversal_p2.png)

![OTE Reversal page 3](assets/backtest_atlas/ote_reversal_p3.png)

### OTE Meta

10 curves across 7 panels.

![OTE Meta page 1](assets/backtest_atlas/ote_meta_p1.png)

### OTE Union

8 curves across 7 panels.

![OTE Union page 1](assets/backtest_atlas/ote_union_p1.png)

### OTE Breakout

22 curves across 20 panels.

![OTE Breakout page 1](assets/backtest_atlas/ote_breakout_p1.png)

![OTE Breakout page 2](assets/backtest_atlas/ote_breakout_p2.png)

![OTE Breakout page 3](assets/backtest_atlas/ote_breakout_p3.png)

### OTE General

10 curves across 5 panels.

![OTE General page 1](assets/backtest_atlas/ote_general_p1.png)

### FRVP Reversal

53 curves across 32 panels.

![FRVP Reversal page 1](assets/backtest_atlas/frvp_reversal_p1.png)

![FRVP Reversal page 2](assets/backtest_atlas/frvp_reversal_p2.png)

![FRVP Reversal page 3](assets/backtest_atlas/frvp_reversal_p3.png)

![FRVP Reversal page 4](assets/backtest_atlas/frvp_reversal_p4.png)

### FRVP Continuation

23 curves across 15 panels.

![FRVP Continuation page 1](assets/backtest_atlas/frvp_continuation_p1.png)

![FRVP Continuation page 2](assets/backtest_atlas/frvp_continuation_p2.png)

### FRVP Meta

16 curves across 11 panels.

![FRVP Meta page 1](assets/backtest_atlas/frvp_meta_p1.png)

![FRVP Meta page 2](assets/backtest_atlas/frvp_meta_p2.png)

### ICT Reversal

43 curves across 15 panels.

![ICT Reversal page 1](assets/backtest_atlas/ict_reversal_p1.png)

![ICT Reversal page 2](assets/backtest_atlas/ict_reversal_p2.png)

### ICT Continuation

33 curves across 15 panels.

![ICT Continuation page 1](assets/backtest_atlas/ict_continuation_p1.png)

![ICT Continuation page 2](assets/backtest_atlas/ict_continuation_p2.png)

### ICT Meta

23 curves across 12 panels.

![ICT Meta page 1](assets/backtest_atlas/ict_meta_p1.png)

![ICT Meta page 2](assets/backtest_atlas/ict_meta_p2.png)


<a id="model_realized_pnl_charts_20260927"></a>

## Realized net P&L by model family

*Merged 2026-10-09 from `docs/model_realized_pnl_charts_20260927.md`.*

Generated 2026-09-27. One chart per model family, for every model bundle the live app's dashboard views load.

Each line is the running sum of per-trade net P&L from that model's `selected_test_trades.csv`, ordered by trade exit time. Out-of-sample backtest folds only, after modelled spread, slippage and commission. Flat stretches are gaps between rolling folds, not flat performance. OTE is EURUSD and reports pips. FRVP and ICT are ES and report ticks, so totals are not comparable across instruments.

### Contents

- [OTE multi-family (EURUSD 5m)](#ote)
- [FRVP shadow and paper-signal bundles (ES 5m)](#frvp_shadow)
- [FRVP setup family 2026-08-29 (ES 5m, long only)](#frvp_setup)
- [ICT paper-signal bundle 2026-08-13 (ES 5m)](#ict_paper)
- [ICT setup family 2026-08-11 audit (ES 5m, short only)](#ict_setup)

<a id="ote"></a>

### OTE multi-family (EURUSD 5m)

Loaded by: OTE dashboard view. Instrument EURUSD, P&L in pips.

> Note: short_breakout_tcn_champion is listed in the live multi-family registry but was excluded from this backtest run, so it has no curve here.

#### Reversal family

![OTE multi-family (EURUSD 5m) Reversal family](assets/realized_pnl/ote_reversal.png)

| Model | Direction | Trades | Net pips | Max drawdown (pips) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `long_reversal_tcn_v2_20260525_narrow48` | long | 445 | +5,469 | -333 | 2024-04-03 | 2026-03-20 |
| `short_reversal_xgb_v2_20260525` | short | 561 | +5,454 | -289 | 2024-04-01 | 2026-03-23 |

#### Meta family

![OTE multi-family (EURUSD 5m) Meta family](assets/realized_pnl/ote_meta.png)

| Model | Direction | Trades | Net pips | Max drawdown (pips) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `long_ote_meta_tcn_champion` | long | 1,579 | +3,841 | -1,157 | 2024-01-03 | 2026-03-24 |
| `short_ote_meta_tcn_champion` | short | 577 | +3,904 | -337 | 2024-01-02 | 2026-03-20 |

#### OTE union family

![OTE multi-family (EURUSD 5m) OTE union family](assets/realized_pnl/ote_ote_union.png)

| Model | Direction | Trades | Net pips | Max drawdown (pips) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `long_ote_union_tcn_candidate_20260523` | long | 899 | +5,659 | -551 | 2024-04-01 | 2026-03-24 |
| `short_ote_union_tcn_candidate_20260520` | short | 422 | +2,481 | -320 | 2024-04-02 | 2026-03-20 |

#### Breakout family

![OTE multi-family (EURUSD 5m) Breakout family](assets/realized_pnl/ote_breakout.png)

| Model | Direction | Trades | Net pips | Max drawdown (pips) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `long_breakout_tcn_champion` | long | 2,235 | -4,704 | -6,663 | 2023-07-03 | 2026-03-23 |
| `long_breakout_xgb_v1` | long | 981 | -1,121 | -2,653 | 2024-07-02 | 2026-03-24 |

#### All families, OTE multi-family

![OTE multi-family (EURUSD 5m) overview](assets/realized_pnl/ote_all_families.png)

<a id="frvp_shadow"></a>

### FRVP shadow and paper-signal bundles (ES 5m)

Loaded by: FRVP dashboard view, plus the paper-signal promotion layer. Instrument ES, P&L in ticks.

> Note: The paper-signal bundle frvp_es_paper_signal_20260816 carries the same models and re-evaluates the same selected_test_trades.csv files as the shadow bundle frvp_es_shadow_live_bundle_20260721, so its curves are identical and are not repeated. frvp_short_reversal_xgb_v1 appears in both runs but is not carried on the paper-signal short manifest.

#### Reversal family

![FRVP shadow and paper-signal bundles (ES 5m) Reversal family](assets/realized_pnl/frvp_shadow_reversal.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `frvp_long_reversal_xgb_v1` | long | 63 | +3,677 | -805 | 2024-01-09 | 2026-05-28 |
| `frvp_short_reversal_xgb_v1` | short | 637 | -3,476 | -5,758 | 2022-07-11 | 2026-06-12 |

#### Continuation family

![FRVP shadow and paper-signal bundles (ES 5m) Continuation family](assets/realized_pnl/frvp_shadow_continuation.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `frvp_long_continuation_xgb_v1` | long | 512 | +7,001 | -1,670 | 2020-07-06 | 2026-06-18 |
| `frvp_short_continuation_tcn_v1` | short | 2,807 | -10,430 | -21,769 | 2022-01-03 | 2026-06-19 |

#### Meta family

![FRVP shadow and paper-signal bundles (ES 5m) Meta family](assets/realized_pnl/frvp_shadow_meta.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `frvp_long_meta_xgb_v1` | long | 603 | +9,216 | -3,755 | 2020-04-02 | 2026-06-15 |
| `frvp_short_meta_xgb_v1` | short | 1,523 | +1,829 | -14,207 | 2020-04-21 | 2026-06-19 |

#### All families, FRVP shadow and paper-signal bundles

![FRVP shadow and paper-signal bundles (ES 5m) overview](assets/realized_pnl/frvp_shadow_all_families.png)

<a id="frvp_setup"></a>

### FRVP setup family 2026-08-29 (ES 5m, long only)

Loaded by: FRVP Setup Models dashboard view. Instrument ES, P&L in ticks.

> Note: frvp_long_reversal_setup4_xgb_v1 comes from a separate single-fold research run and has only 5 trades, so its curve is not comparable to the others.

#### Reversal family

![FRVP setup family 2026-08-29 (ES 5m, long only) Reversal family](assets/realized_pnl/frvp_setup_reversal.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `frvp_long_reversal_setup1_xgb_v1` | long | 234 | +179 | -3,792 | 2023-04-06 | 2026-06-12 |
| `frvp_long_reversal_setup4_xgb_v1` | long | 5 | +393 | 0 | 2026-04-24 | 2026-06-08 |
| `frvp_long_reversal_setup6_xgb_v1` | long | 251 | -2,897 | -4,190 | 2023-01-17 | 2026-06-11 |

#### Continuation family

![FRVP setup family 2026-08-29 (ES 5m, long only) Continuation family](assets/realized_pnl/frvp_setup_continuation.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `frvp_long_continuation_setup2_xgb_v1` | long | 201 | +3,499 | -2,410 | 2024-04-17 | 2026-06-15 |
| `frvp_long_continuation_setup3_xgb_v1` | long | 485 | -2,964 | -5,399 | 2022-01-03 | 2026-06-12 |
| `frvp_long_continuation_setup5_xgb_v1` | long | 1,217 | +5,256 | -10,005 | 2021-07-01 | 2026-06-19 |

#### All families, FRVP setup family 2026-08-29

![FRVP setup family 2026-08-29 (ES 5m, long only) overview](assets/realized_pnl/frvp_setup_all_families.png)

<a id="ict_paper"></a>

### ICT paper-signal bundle 2026-08-13 (ES 5m)

Loaded by: ICT dashboard view. Instrument ES, P&L in ticks.

#### Reversal family

![ICT paper-signal bundle 2026-08-13 (ES 5m) Reversal family](assets/realized_pnl/ict_paper_reversal.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `ict_long_reversal_xgb_v1` | long | 1,275 | +5,471 | -1,141 | 2019-10-07 | 2026-06-18 |
| `ict_short_reversal_xgb_v1` | short | 429 | +3,052 | -957 | 2019-10-02 | 2026-06-09 |

#### Continuation family

![ICT paper-signal bundle 2026-08-13 (ES 5m) Continuation family](assets/realized_pnl/ict_paper_continuation.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `ict_long_continuation_xgb_v1` | long | 247 | +7 | -522 | 2022-01-04 | 2026-06-15 |
| `ict_short_continuation_xgb_v1` | short | 165 | +501 | -753 | 2022-07-12 | 2026-06-09 |

#### Meta family

![ICT paper-signal bundle 2026-08-13 (ES 5m) Meta family](assets/realized_pnl/ict_paper_meta.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `ict_long_meta_xgb_v1` | long | 1,555 | +7,461 | -652 | 2019-10-04 | 2026-06-18 |
| `ict_short_meta_xgb_v1` | short | 1,082 | +3,907 | -984 | 2019-10-02 | 2026-06-15 |

#### All families, ICT paper-signal bundle 2026-08-13

![ICT paper-signal bundle 2026-08-13 (ES 5m) overview](assets/realized_pnl/ict_paper_all_families.png)

<a id="ict_setup"></a>

### ICT setup family 2026-08-11 audit (ES 5m, short only)

Loaded by: ICT Setup Models dashboard view. Instrument ES, P&L in ticks.

#### Reversal family

![ICT setup family 2026-08-11 audit (ES 5m, short only) Reversal family](assets/realized_pnl/ict_setup_reversal.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `ict_short_reversal_ifvg_reversal_xgb_v1` | short | 298 | -66 | -666 | 2020-10-06 | 2026-06-17 |
| `ict_short_reversal_sweep_reclaim_xgb_v1` | short | 755 | +3,049 | -810 | 2020-01-03 | 2026-06-17 |

#### Continuation family

![ICT setup family 2026-08-11 audit (ES 5m, short only) Continuation family](assets/realized_pnl/ict_setup_continuation.png)

| Model | Direction | Trades | Net ticks | Max drawdown (ticks) | First exit | Last exit |
|---|---|---:|---:|---:|---|---|
| `ict_short_continuation_premium_discount_continuation_xgb_v1` | short | 70 | -324 | -536 | 2022-10-03 | 2026-06-09 |

#### All families, ICT setup family 2026-08-11 audit

![ICT setup family 2026-08-11 audit (ES 5m, short only) overview](assets/realized_pnl/ict_setup_all_families.png)


<a id="model_leaderboards_20260527"></a>

## Model Leaderboards

*Merged 2026-10-09 from `model_testing/reports/MODEL_LEADERBOARDS_20260527.md`.*

> Historical OTE snapshot. The current cross-family update is `#model_leaderboards_20260730`.

Generated on `2026-05-27` from saved artifacts in `models/**/cv_fold_manifest.json` and `model_testing/reports/ote_policy_backtests/**/summary.json`.

### Ranking Method

- Training leaderboard: ranked all `42` model artifacts by mean CV `average_precision`, with mean CV event `F0.5` as the tiebreaker.
- Post-training leaderboard: collapsed `63` saved backtests into the best saved run for each of `19` unique `model_id`s, then ranked by post-cost profitability flag, acceptance gates passed, monthly Sharpe, annualized net pips, and profit factor.

### Full Training Leaderboard

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

### Full Post-Training Leaderboard

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

### Observations

- The training leaderboard is dominated by breakout-family models: `7` of the top `10` training artifacts are breakout variants, and the top `5` are all breakout artifacts.
- That edge does not carry through post-training. The best post-training breakout result is `long_breakout_tcn_champion`, which only lands at post-training rank `17` with `4/6` gates and Sharpe `0.812`. `short_breakout_tcn_champion` and `long_breakout_xgb_v1` are both post-cost unprofitable in their best saved runs.
- The post-training winners are mostly reversal and OTE/union models. The top `6` post-training slots are all reversal models, and each of those top `6` passed all `6` acceptance gates.
- Several of the best live-like performers were only mid-pack or lower in training rank: `short_reversal_xgb_v2_20260525` -> training rank `19`, `long_reversal_tcn_v2_20260525_narrow48` -> training rank `21`, `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair` -> training rank `22`, `short_reversal_xgb_v1` -> training rank `20`, `short_reversal_tcn_champion` -> training rank `28`, `long_reversal_tcn_champion` -> training rank `25`.
- `long_ote_tcn_v2_candidate` is the best pure PnL outlier in post-training with annualized net pips `3954.7`, but it passed `5/6` gates rather than `6/6`, so it is a stronger return candidate than robustness candidate.
- The strongest post-training Sharpe belongs to `short_reversal_xgb_v2_20260525` at `3.615`, while the strongest profit factor belongs to `long_reversal_tcn_champion` at `3.072`.
- A lot of the highest training scores come from smaller fold counts. `4` of the top `10` training artifacts were evaluated on only `2` folds, which makes the CV leaderboard more fragile than the post-training table.

### Recommendations

- Use the post-training leaderboard, not the raw training leaderboard, as the primary promotion surface. The saved data shows that CV AP alone is overvaluing breakout-family artifacts.
- For near-term promotion and retrain priority, anchor on the six-model post-training core: `short_reversal_xgb_v2_20260525`, `long_reversal_tcn_v2_20260525_narrow48`, `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair`, `short_reversal_xgb_v1`, `short_reversal_tcn_champion`, and `long_reversal_tcn_champion`.
- Keep `long_ote_tcn_v2_candidate` and `long_ote_tcn_v1_candidate` in the active challenger set because they still produce strong net pips and respectable Sharpe, but treat them as controlled-risk candidates until the missing gate is fixed.
- De-emphasize breakout models in the next retrain wave unless you also tighten the post-training objective. Their training scores are excellent, but the saved backtests show poor conversion into robust post-cost performance.
- Raise the selection bar on future training sweeps by adding minimum fold-count and post-training proxy constraints before a model is considered `top tier`. A practical rule would be: at least `3` folds, then require a downstream policy backtest pass before promotion.


<a id="model_leaderboards_20260730"></a>

## Model Leaderboards

*Merged 2026-10-09 from `model_testing/reports/MODEL_LEADERBOARDS_20260730.md`.*

Generated on `2026-07-30` from the saved training and walk-forward artifacts for OTE, FRVP, and ICT models.

The previous full OTE snapshot remains available in `#model_leaderboards_20260527`. This update adds the completed 1-minute OTE, breakout-sustained, FRVP, and leakage-safe ICT experiments without treating unlike instruments, bar sizes, labels, and evaluation windows as one directly comparable score.

### Ranking Method

- Models are ranked only inside comparable cohorts.
- Training tables rank by mean CV `average_precision`, with mean CV event `F0.5` as the tiebreaker.
- Post-training tables put accepted/passing branches first, then use monthly Sharpe, walk-forward efficiency (`WFE`), profitable-quarter breadth, and concentration as the main ordering evidence.
- OTE results use EUR/USD pips. FRVP and ICT results use ES ticks. These units must not be compared directly.
- The older OTE summaries use a six-gate contract. The current FRVP and ICT summaries use an eight-gate promotion-quality contract, including the newer concentration and account-drawdown checks.
- A recent-regime or short-window result is marked explicitly and is not promoted above a full-span result solely because its Sharpe is higher.

### Current Promotion-Relevant Leaders

| Cohort | Model / policy | Scope | Gates | Trades | Net PnL | Sharpe | WFE | Current status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OTE EUR/USD 5-minute | `short_reversal_xgb_v2_20260525` | Full saved OTE walk-forward | 6/6 | 561 | +5453.65 pips | 3.615 | 0.729 | Existing 5-minute OTE leader; no newer 5-minute experiment displaced it |
| OTE EUR/USD 1-minute | `short_reversal_xgb_1m_v1` | 2021-10 through 2026-03 | 6/6 | 3142 | +31943.90 pips | 2.872 | 0.885 | Strongest broad-window 1-minute result; still a separate candidate lane |
| OTE EUR/USD 1-minute | `long_reversal_tcn_1min_runpod_20260612_candidate` | 2024-01 through 2026-03 | 6/6 | 2295 | +19128.65 pips | 2.676 | 2.889 | Strong long 1-minute candidate; the higher-Sharpe one-year rerun is diagnostic, not the primary evidence |
| FRVP ES 5-minute | `frvp_long_continuation_xgb_v1` with `v3` policy | Full-span | 8/8 | 524 | +7190.40 ticks | 1.226 | 2.459 | Accepted promotion-near FRVP baseline |
| FRVP ES 5-minute | `frvp_long_reversal_xgb_recent_regime_prune_v2` | Recent-regime selective lane | 8/8 | 63 | +3677.05 ticks | 1.480 | 2.162 | Accepted selective-deployment reversal contract |
| ICT ES 5-minute | `ict_long_meta_xgb_v1` | Latest leakage-safe full walk-forward | 8/8 | 1555 | +7461.25 ticks | 1.586 | 1.997 | Current ICT research leader; packaged for shadow only |
| ICT ES 5-minute | `ict_long_reversal_xgb_v1` | Latest leakage-safe full walk-forward | 8/8 | 1275 | +5471.25 ticks | 1.269 | 3.013 | Accepted research branch; packaged for shadow only |

### New OTE Training Experiments

#### EUR/USD 1-Minute

| Rank | Artifact | Backend | Folds | CV Mean AP | CV Mean Event F0.5 | Test AP | Test Event F0.5 | Post-training read |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `runpod_1min_tcn_long_ote_20260615_002054/long_ote` | TCN | 5 | 0.767551 | 0.491293 | 0.863161 | 0.522754 | Training leader; no saved walk-forward policy backtest yet |
| 2 | `runpod_1min_tcn_long_reversal_20260612_171301/long_reversal` | TCN | 6 | 0.561999 | 0.447933 | 0.677881 | 0.434947 | Passed 6/6 downstream gates on the saved broad and strict windows |
| 3 | `ote_1min_reversal_wave1_short_xgb/short_reversal` | XGBoost | 18 | 0.498387 | 0.674257 | 0.531847 | 0.578512 | Best broad-window 1-minute economics result |
| 4 | `runpod_1min_tcn_short_reversal_20260614_193525/short_reversal` | TCN | 6 | 0.490051 | 0.446595 | 0.645085 | 0.413163 | Holdout AP improved over the XGBoost branch, but no saved walk-forward result promotes it |

#### EUR/USD 5-Minute Breakout Improvements

| Rank | Artifact | Backend | Folds | CV Mean AP | CV Mean Event F0.5 | Test AP | Test Event F0.5 | Post-training read |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `experiment_a_breakout_sustained_pilot_20260621/long_breakout_sustained` | TCN | 3 | 0.982097 | 0.976017 | 0.996278 | 0.987699 | Failed downstream economics: 2/6 gates, -2646.05 pips, Sharpe -1.091 |
| 2 | `ote_multi_family_tcn_refresh_20260530/short_breakout` | TCN | 2 | 0.921215 | 0.878271 | 0.987240 | 0.852843 | Did not repair the short-breakout promotion case; latest event-metadata rerun remains post-cost negative |

The sustained-breakout pilot is now the clearest warning against using classifier metrics as the promotion leaderboard: it has the highest saved OTE CV AP in the repository but negative downstream economics.

### OTE 1-Minute Post-Training Leaderboard

| Rank | Model / run | Evaluation window | Gates | Trades | Net PnL (pips) | Sharpe | Profit Factor | WFE | Profitable Quarters | Read |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `long_reversal_tcn_1min_runpod_20260612_candidate` strict | 2025-04 through 2026-03 | 6/6 | 1856 | +17088.30 | 5.728 | 5.263 | 4.014 | 1.000 | Excellent one-year diagnostic, but too short to be the primary promotion record |
| 2 | `short_reversal_xgb_1m_v1` | 2021-10 through 2026-03 | 6/6 | 3142 | +31943.90 | 2.872 | 2.287 | 0.885 | 1.000 | Best broad-window 1-minute candidate |
| 3 | `long_reversal_tcn_1min_runpod_20260612_candidate` broad | 2024-01 through 2026-03 | 6/6 | 2295 | +19128.65 | 2.676 | 4.409 | 2.889 | 0.889 | Stronger robustness evidence than the strict one-year rerun |
| 4 | `long_breakout_sustained_tcn_expA_20260621_pilot` | 2023-04 through 2026-03 | 2/6 | 789 | -2646.05 | -1.091 | 0.737 | 0.652 | 0.250 | Reject; high AP did not convert to a tradeable policy |

### FRVP Research Leaderboard

This table carries forward the current saved FRVP keep-set. Full-span and recent-regime branches remain explicitly distinguished.

| Rank | Branch | Saved checkpoint / contract | Scope | Gates | Trades | Net PnL (ticks) | Sharpe | DSR | WFE | Current read |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Long continuation | `frvp_long_continuation_gatefix_v3_20260715_accountdd` | Full-span | 8/8 | 524 | +7190.40 | 1.226 | 1.061 | 2.459 | Best promotion-near direct FRVP checkpoint |
| 2 | Long reversal, recent regime | `frvp_long_reversal_xgb_recent_regime_prune_v2` | Recent-regime selective lane | 8/8 | 63 | +3677.05 | 1.480 | 1.179 | 2.162 | Accepted operational reversal contract |
| 3 | Long reversal | `frvp_long_reversal_recency_trial1_v3_20260715_accountdd` | Full-span recency weighted | 7/8 | 140 | +6027.00 | 1.061 | 0.956 | -3.797 | Best full-span reversal control; blocked by train-side stability |
| 4 | Long meta | `frvp_long_meta_gatefix_v3_20260715_accountdd` | Full-span | 5/8 | 603 | +9216.05 | 0.595 | 0.577 | 1.897 | Profitable research control; Sharpe, concentration, and drawdown remain weak |
| 5 | Long meta recency sentinel | `frvp_long_meta_recency_trial1_gatefix_v3_20260705` | Full-span recency weighted | Fail | 599 | +6938.65 | 0.478 | 0.469 | 1.974 | Weaker than the refresh `v3`; no family-wide recency rollout |
| 6 | Long meta post-2022 control | `frvp_es_primary_post2022_smallcv_20260703` | Post-2022 only | Fail | 1615 | +9347.25 | 0.647 | 0.624 | -3.532 | Hard history truncation did not fix robustness |

Canonical FRVP detail remains in `docs/FRVP_ES_primary_audit_report.md`, `docs/FRVP_experiment_journal.md`, and `docs/FRVP_experiment_journal.md#frvp_operational_contract_20260721`.

### ICT Leakage-Safe Training Improvement

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

### Current ICT Post-Training Leaderboard

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

### Promotion Decisions

- Keep the existing 5-minute OTE reversal core unchanged. None of the new 5-minute breakout experiments displaced it.
- Keep the 1-minute OTE models in a separate research lane and do not deploy them while OTE Live has no 1-minute stream. `short_reversal_xgb_1m_v1` has the strongest broad-window evidence; the long-reversal TCN is also strong, but its best headline result is from only one year.
- Treat FRVP long continuation `v3` and the recent-regime long-reversal `v2` contract as the accepted FRVP leaders.
- Treat ICT long meta and long reversal as the only promotion-eligible branches from the latest leakage-safe roster.
- Keep ICT short meta and short reversal in shadow/research until concentration and overlapping-event dependence improve.
- Reject the sustained-breakout pilot for promotion despite its exceptional classifier metrics.

### OTE Live Deployment Decision

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

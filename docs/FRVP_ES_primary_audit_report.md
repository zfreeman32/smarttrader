# FRVP ES Primary Audit Report

## 0. Refresh Update (2026-07-02)

This section supersedes the economics conclusions in Sections 1-5 below for the refresh branch `frvp_es_primary_refresh_20260701`. The older sections remain useful as the pre-refresh baseline audit, but they were written before the spread-cost fix, Setup 4 re-enable, and pooled meta-model wiring were reflected in the saved refresh artifacts. In this pass, only Phase 6/7 was rerun via `scripts/run_frvp_post_training_eval.ps1`, so the status notes below distinguish between code-complete changes and refresh-artifact validation.

### 0.1 Original Goal Status

| Goal | Status | Evidence | Current read |
|---|---|---|---|
| Use the extracted CPI calendar instead of mixed pre/post-2022 handling | `Partially complete` | `frvp/calendars/macro.py:19-35,160-180` now resolves CPI from `data/futures_data/CPI_release_dates.txt`, with recent in-code dates used only as gap-fill; `artifacts/frvp_es_primary_refresh_20260701/phase03/labeling_diagnostics.json` shows `todo_macro_flags_unavailable = false`, `macro_flag_columns_used` includes `cpi_flag`, and `events_excluded_macro = 3431` | The calendar sourcing fix is implemented and present in the refresh labels, but the non-stationarity problem is not solved by that change alone. On refresh OOF predictions, reversal AP still drops materially after 2022: long reversal `0.6011 -> 0.5218`, short reversal `0.5636 -> 0.4909`, and long meta `0.5804 -> 0.5114`. |
| Fix `approx_spread` being treated like full candle range in economics | `Complete and validated in Phase 6/7` | `features/feature_sets/microstructure.py:26,41` now routes `approx_spread` through `_resolve_approx_spread_proxy`, and `model_testing/ote_threshold_policy.py:48,394-416` now forces ES/6E to use session-schedule costs instead of feature-spread costs | This materially changed outcomes. The old all-red economics result no longer holds on the refresh branch. `frvp_long_continuation_xgb_v1` is now `+12465.15` ticks net and `frvp_long_reversal_xgb_v1` is `+2197.25` ticks net in `model_testing/reports/frvp_backtests/frvp_es_primary_refresh_20260701/model_summary.csv`. |
| Re-enable Setup 4 | `Complete and validated in Phase 3 artifacts` | `frvp/pipelines/es_primary_phase04.py:109` sets `enable_failed_auction_labels=True`; `artifacts/frvp_es_primary_refresh_20260701/phase03/labeling_diagnostics.json` shows `events_excluded_disabled_setup = 0`; `artifacts/frvp_es_primary_refresh_20260701/phase03/es_primary_frvp_events.csv` contains Setup 4 rows | Setup 4 is back in the saved refresh label set, not merely enabled in code. The refresh Phase 3 event file contains `1002` Setup 4 events, `792` of them usable. |
| Create pooled meta models for long/short FRVP | `Complete for XGBoost and validated end-to-end` | `preprocessing/config.py:28-29`, `frvp/pipelines/es_primary_phase04.py:44-45`, and `preprocessing/pipeline.py` now carry `label_long_frvp_meta` / `label_short_frvp_meta`; `artifacts/frvp_es_primary_refresh_20260701/phase04/prepared/summary.json` shows healthy pooled targets with `9349` long-meta usable rows at `0.4793` positive rate and `9851` short-meta usable rows at `0.4805`; registry and Phase 6/7 outputs include `frvp_long_meta_xgb_v1` and `frvp_short_meta_xgb_v1` | The pooled meta branch is now real and trackable. It is not yet deployment-ready: long meta is walk-forward negative, short meta is slightly positive but economically weak. Also note that the current training stack only trained pooled meta for XGBoost, not TCN. |

### 0.2 Refresh Evaluation Scoreboard

Saved refresh backtest summary from `model_testing/reports/frvp_backtests/frvp_es_primary_refresh_20260701/model_summary.csv`:

| Model | Backend | Net PnL (ticks) | Sharpe | DSR | WFE | Positive Composite Share | Accepted |
|---|---|---:|---:|---:|---:|---:|---|
| `frvp_long_continuation_xgb_v1` | XGBoost | `+12465.15` | `0.604` | `0.585` | `2.345` | `0.8889` | `False` |
| `frvp_long_reversal_tcn_v1` | TCN | `+5385.85` | `0.744` | `0.702` | `-158.178` | `0.5556` | `False` |
| `frvp_long_reversal_xgb_v1` | XGBoost | `+2197.25` | `0.178` | `0.177` | `-0.218` | `0.4444` | `False` |
| `frvp_short_meta_xgb_v1` | XGBoost | `+1829.05` | `0.067` | `0.067` | `0.138` | `0.5556` | `False` |
| `frvp_long_continuation_tcn_v1` | TCN | `-2712.05` | `-0.131` | `-0.131` | `-0.460` | `0.5556` | `False` |
| `frvp_short_reversal_tcn_v1` | TCN | `-2812.70` | `-0.484` | `-0.473` | `-9.944` | `0.2222` | `False` |
| `frvp_short_reversal_xgb_v1` | XGBoost | `-3476.05` | `-0.393` | `-0.388` | `1.139` | `0.4444` | `False` |
| `frvp_long_meta_xgb_v1` | XGBoost | `-4468.65` | `-0.133` | `-0.133` | `0.506` | `0.2222` | `False` |
| `frvp_short_continuation_tcn_v1` | TCN | `-10429.55` | `-0.378` | `-0.373` | `0.728` | `0.6667` | `False` |
| `frvp_short_continuation_xgb_v1` | XGBoost | `-22679.10` | `-0.693` | `-0.665` | `1.489` | `0.2222` | `False` |

What changed versus the old baseline:

- The refresh branch is no longer "all models negative after costs." Four models are now post-cost positive in walk-forward: `frvp_long_continuation_xgb_v1`, `frvp_long_reversal_tcn_v1`, `frvp_long_reversal_xgb_v1`, and `frvp_short_meta_xgb_v1`.
- The best current direct model is now `frvp_long_continuation_xgb_v1`, not `frvp_long_reversal_xgb_v1`. It has the strongest combination of net PnL, WFE, DSR, and positive composite share in the refresh backtest summary.
- No model is paper-trading ready yet. `accepted_for_paper_trading_gate = False` remains true for all ten models in `model_summary.csv`.

### 0.3 Meta Model Health

The new pooled XGBoost models are now fully in the registry and Phase 6/7 outputs, so they should be tracked explicitly rather than treated as side experiments.

| Model | CV AP | Test AP | Test event F0.5 | Threshold-search result | Walk-forward result | Health read |
|---|---:|---:|---:|---|---|---|
| `frvp_long_meta_xgb_v1` | `0.5648` | `0.5460` | `0.7353` | `model_testing/reports/frvp_threshold_policies/frvp_es_primary_refresh_20260701/run_summary.json` selected `regime_threshold` with `+18.41` post-cost expectancy and `+11356.7` ticks on the static test set | `-4468.65` ticks net, Sharpe `-0.133`, positive composite share `0.2222` in walk-forward | Ranking is adequate, but the static test result did not persist out of sample. This branch is overtrading (`3621` walk-forward trades) and remains economically unstable. |
| `frvp_short_meta_xgb_v1` | `0.5367` | `0.4940` | `0.6682` | Static threshold search selected `regime_threshold_plus_abstain` with `-15.84` post-cost expectancy and `-12356.25` ticks on the held-out test set | `+1829.05` ticks net, Sharpe `0.067`, WFE `0.138`, positive composite share `0.5556` in walk-forward | Slightly positive, but only barely. The short meta branch is interesting as a research aggregator, not as a live candidate. |

Current interpretation:

- The meta-model idea is now implemented correctly enough to evaluate.
- Long meta is not yet adding economic clarity; it blends too many distinct behaviors and gives up too much stability.
- Short meta is a weak positive aggregator, but its Sharpe/DSR are far below gate quality.
- If pooled meta stays in the stack, it should be tracked as a research-control branch rather than as the leading deployment candidate.

### 0.4 Key Observations from the Refresh Rerun

- The spread-cost correction was the highest-value fix. It changed the branch from an "all red" economics story to a mixed but genuinely promising one, especially for `frvp_long_continuation_xgb_v1`.
- The 0DTE / post-2022 stability question is still open in substance even though the CPI calendar source is now cleaner. Refresh OOF AP still weakens after 2022 for reversal-heavy models: long reversal `0.6011 -> 0.5218`, short reversal `0.5636 -> 0.4909`, long meta `0.5804 -> 0.5114`. Continuation is more stable, and short continuation even improves slightly post-2022 (`0.5074 -> 0.5299`).
- Setup 4 is no longer a disabled branch. The saved refresh Phase 3 artifact includes live failed-auction examples, so future attribution and per-setup analysis should treat Setup 4 as real rather than theoretical.
- The threshold-search summary and walk-forward summary now diverge in a useful way. For example, `frvp_long_continuation_xgb_v1` shows no qualified static policy in `model_testing/reports/frvp_threshold_policies/frvp_es_primary_refresh_20260701/run_summary.json`, yet the walk-forward backtest is clearly positive. That means the fold-by-fold policy selection path is extracting value that the one-shot static qualification summary is not capturing cleanly.

### 0.5 Next Course of Action

1. Treat `frvp_long_continuation_xgb_v1` as the primary optimization target. It is the strongest refresh candidate, but it still misses the paper-trading gate because Sharpe is only `0.604` and the acceptance contract remains red in `model_summary.csv`.
2. Run a focused Phase 6 policy study on `frvp_long_continuation_xgb_v1` and `frvp_long_reversal_xgb_v1` to explain the threshold/backtest mismatch. Specifically, compare the static `policy_evaluation.csv` outputs against the fold-selected policies in the backtest folders under `model_testing/reports/frvp_backtests/frvp_es_primary_refresh_20260701/...`.
3. Run the explicit post-2022 / recency-weighted training experiment for reversal and long-meta models. The CPI archive fix cleaned the calendar contract, but the reversal non-stationarity signal is still there in refresh OOF AP.
4. Keep Setup 4 in the next analysis pass and add per-setup-family reporting. The old audit hypothesis that Setup 4 was absent is no longer true for the refresh branch.
5. Keep the pooled meta models in the report going forward, but track them as research controls unless one of them materially improves Sharpe and DSR on a full walk-forward rerun.

### 0.6 Working Conclusion

We did accomplish three of the four original engineering goals in a verifiable way on the refresh branch: the spread-cost contract is fixed and reflected in Phase 6/7, Setup 4 is re-enabled and present in Phase 3, and pooled XGBoost meta models now exist end-to-end in preprocessing, training artifacts, the registry, and post-training evaluation. The CPI calendar source fix is also implemented, and the refresh labels do use macro flags correctly, but the explicit pre/post-2022 regime-shift problem is only partially solved: the data-source contract is better, while the reversal-family non-stationarity is still showing up in OOF performance.

That leaves us in a much better place than the pre-refresh audit. We are no longer diagnosing a universally broken economics layer. We now have one direct model that is clearly worth focused promotion work (`frvp_long_continuation_xgb_v1`), one TCN challenger that remains interesting (`frvp_long_reversal_tcn_v1`), a weaker but positive long-reversal XGBoost baseline, and two pooled meta branches whose status is now measurable rather than hypothetical.

## 1. Executive Summary

The FRVP ES-primary pipeline is not failing because Phases 0-4 are broken. The continuity stack is causal and well-tested, the setup detector behaves as coded, and the prepare/train stack produces ranking metrics that are genuinely above base rate. The main failure is that the economics layer does not match the research contract in the design paper. Two implementation choices are especially important: `frvp/pipelines/es_primary_phase04.py:93-110` tightens continuation barriers materially versus Section 7.3 of the paper, and `model_testing/ote_threshold_policy.py:392-418` prices trade friction off `approx_spread`, while `features/feature_sets/microstructure.py:23-27` defines `approx_spread` as the full candle range (`high - low`), not a bid/ask spread proxy. That turns the saved walk-forward backtests into an extremely punitive cost test.

That friction mismatch is the largest single reason the ranking-good models are economics-bad. On the saved selected test trades, mean realized cost is `29.85` units for `frvp_long_reversal_xgb_v1`, `28.56` for `frvp_short_reversal_xgb_v1`, `39.27` for `frvp_long_continuation_xgb_v1`, and `45.34` for `frvp_short_continuation_xgb_v1`, versus a nominal session-schedule-only round-turn cost of roughly `3.15-5.65` units implied by `model_testing/reports/frvp_threshold_policies/frvp_es_primary_current/run_summary.json`. Under a simple arithmetic counterfactual that keeps the saved gross PnL fixed and replaces the bar-range-driven spread charge with the stated session schedule, `frvp_long_reversal_xgb_v1` moves from `-8215.65` to `+816.10`, and `frvp_long_continuation_xgb_v1` moves from `-51450.55` to `+9034.95`. The two short models remain negative even under that milder cost assumption, which means they have a genuine edge-quality problem in addition to a friction problem.

The three highest-EV fixes are therefore: first, rerun Phase 7 with a falsifiable friction A/B that separates the nominal ES session-cost schedule from the current bar-range-as-spread implementation; second, run regime/day-type-gated threshold policies, because the current threshold search never actually populated any abstain regime lists and long reversal already shows a profitable `strong_down_medium` pocket; third, split the pooled family models by setup, because short reversal is being dragged by Setup 1, and short continuation is being dragged by Setups 3 and 5. Feature re-encoding for `frvp_open_type` and `frvp_day_type` is low priority: the saved contracts show no one-hot dilution, and the reversal targets in particular have already hard-coded `frvp_open_type == inside_value` upstream.

## 2. Phase-by-Phase Trace Findings

### Audit scope note

The design paper was read first in full from [docs/FRVP_ES_primary_6E_variant_design.md](/abs/path/C:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/docs/FRVP_ES_primary_6E_variant_design.md). The user-tab artifact path `artifacts/frvp_es_primary_phase04_full_reuse/...` is missing in this repo, so this audit uses `artifacts/frvp_es_primary_current/...` throughout and flags places where that blocks exact reproduction.

### Phase 0: Continuity

`frvp/continuity/roll_calendar.py:49-53` explicitly implements the design-paper rule that session `N` uses session `N-1` volume to choose the lead contract, and `frvp/continuity/roll_calendar.py:117-131` assigns the prior session's winner causally. `frvp/continuity/continuous_contract.py:55-57` forbids cross-contract profile slices under reset-at-roll mode, `frvp/continuity/continuous_contract.py:124-147` flags event windows that span a roll, and `frvp/continuity/continuous_contract.py:211-220` keeps raw profile construction on lead-contract bars only while accumulating roll spreads separately.

`tests/test_continuity.py` does assert the things Section 4.3 requires. The test file contains direct assertions for:

- single-contract profile slices and rejection of cross-roll slices: `tests/test_continuity.py:66-76`
- absolute-level coordinate mismatches: `tests/test_continuity.py:82-97`
- causal lead assignment from completed sessions: `tests/test_continuity.py:97-117`
- cumulative back-adjustment across rolls: `tests/test_continuity.py:122-145`
- roll-spanning event-window exclusion: `tests/test_continuity.py:148-164`
- tagged-series roll consistency: `tests/test_continuity.py:191-235`

The continuity/profile/setup smoke tests passed locally:

```text
pytest tests/test_continuity.py tests/test_profiles.py tests/test_frvp_context.py tests/test_frvp_setups.py -q
27 passed in 11.28s
```

The saved roll schedule also looks plausible against quarterly third-Friday ES expiries. In `data/futures_data/es_roll_schedule.json`, `ESH7`, `ESM7`, `ESU7`, and `ESZ7` all expire on the calendar third Friday, and their roll boundaries land about `4.56` days before expiration, which is consistent with a volume-led switch rather than an expiry-day switch. `data/futures_data/es_roll_reconstruction_report.json` reports `all_checks_passed = true`, full-year roll counts of `4` for 2018-2025, and plausible days-before-expiration. The caveat is that the same reconstruction report still flags `12` large seam steps around 2023-2026 rolls, so Q1 is not closed; the seams are not evidence of leakage, but they are evidence that roll sensitivity is still worth A/B testing.

### Phase 1: Profiles and Setups

The setup detector is internally consistent with the design intent, and Setup 4 is thin because its literal condition set is strict, not because I found an obvious implementation bug.

Key rule checks from `frvp/setups/detector.py`:

- Setup 1 requires `frvp_profile_shape == D` and `frvp_open_type == 0` at `frvp/setups/detector.py:283-289`.
- Setup 4 requires: back inside value, an outside run of `1-3` bars, no open drive, quiet re-entry volume, and a same-side sweep within `3` bars at `frvp/setups/detector.py:405-430`, with thresholds defined at `frvp/setups/detector.py:22-26`.
- Setup 6 also requires `frvp_profile_shape == D`, `frvp_open_type == 0`, in-value, no open drive, and no displacement impulse at `frvp/setups/detector.py:462-490`.

The saved fire-rate artifact at `artifacts/frvp_es_primary_current/phase01/setup_fire_rates.csv` matches the paper's narrative: Setup 4 fires at `0.1816` short and `0.1598` long fires per session-side, while most other setups land inside the `0.5-2.0` target band. I could not reproduce that CSV exactly from the broader Phase 2 full dump, and I could not rerun the full raw Phase 1 build inside a few minutes. Because the user-tabbed `phase04_full_reuse` artifact root is missing, I am treating exact fire-rate reproduction as unverified rather than pretending the saved CSV is exact.

The thinness itself is real. On the broad saved FRVP feature dump, the Setup 4 ladder collapses roughly as follows:

- rows already inside value area: `303,162`
- after requiring a recent outside-value run: `7,112`
- after removing open-drive cases: `2,539`
- after requiring a quiet re-entry: `1,321`
- after requiring the same-side sweep and final Setup 4 pattern: `227`

That final rate is about `0.075%` of in-value rows. The detector logic is simply highly selective.

### Phase 2: Features

The saved feature audit generally holds up, but there are two important clarifications.

First, `frvp_open_type` and `frvp_day_type` are not absent because of categorical encoding dilution. `artifacts/frvp_es_primary_current/phase04/prepared/summary.json` reports `encoded_categorical_columns: []`, and `artifacts/frvp_es_primary_current/phase04/prepared/encoders.json` is effectively empty. In other words, the saved prepare run did not one-hot explode those features.

Second, `frvp_open_type` and `frvp_day_type` really are weak on the saved dataset, and `frvp_open_type` is additionally structurally constrained on reversal targets:

- recomputed mutual information for `frvp_open_type` is `0.001429` long reversal, `0.001205` short reversal, `0.000027` long continuation, and `0.000081` short continuation
- recomputed mutual information for `frvp_day_type` is near zero on all four targets, between about `0.000005` and `0.000046`
- `frvp_open_type` is strongly correlated with `frvp_open_vs_prior_poc_atr` on the merged Phase 2 dataset, with correlation `0.7729`, but the categorical itself still adds little marginal signal

I also found one more near-duplicate pair beyond the paper's documented overlap pair:

- `frvp_va_overlap_pct` vs `frvp_rth_eth_value_overlap`: correlation `1.000000`
- `frvp_dist_poc_swing_atr` vs `frvp_swing_poc_vs_session_poc`: correlation `-0.999990`

The prepare stack already handles part of this. In the saved prepared reports, one member of the swing-POC pair is often dropped in the collinearity pass, while `frvp_rth_eth_value_overlap` is absent and `frvp_va_overlap_pct` is retained.

### Phase 3: Labeling

Phase 3 is where repo state diverges most clearly from the design paper.

Section 7.3 of the paper specifies the following target widths in [docs/FRVP_ES_primary_6E_variant_design.md](/abs/path/C:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/docs/FRVP_ES_primary_6E_variant_design.md:299):

| Family | Paper TP | Paper SL | Paper Max Hold |
|---|---:|---:|---:|
| Mean reversion (1, 6, 6b) | `0.8 ATR` | `1.0 ATR` | `60` bars |
| Continuation (2, 3, 5) | `1.5 ATR` | `1.0 ATR` | `96` bars |
| Failed auction (4) | `1.2 ATR` | `0.8 ATR` | `48` bars |

The code actually used is different in two layers:

- `data/labeling/frvp_labeling_engine.py:94-124` defaults continuation to `1.2/1.0/96` and failed auction to `1.0/0.9/48`
- the saved Phase 4 pipeline then overrides those further in `frvp/pipelines/es_primary_phase04.py:93-110` to `1.0 ATR` continuation default, `0.95` for long Setup 3, `0.90` for long Setup 5, `0.85` for short Setup 5, and `enable_failed_auction_labels=False`

That last override matters a lot. Setup 4 is not merely thin in the saved label set; it is fully disabled. In `artifacts/frvp_es_primary_current/phase03/es_primary_frvp_events.csv`, usable event counts by setup are:

- Setup 1: `2174` usable
- Setup 2: `2510` usable
- Setup 3: `3885` usable
- Setup 4: `0` usable
- Setup 5: `7280` usable
- Setup 6: `2579` usable

Most Setup 4 exclusions are tagged `failed_auction_setup_disabled`, which is exactly what `data/labeling/frvp_labeling_engine.py:840-843` implements when `enable_failed_auction_labels=False`.

The saved labeling diagnostics at `artifacts/frvp_es_primary_current/phase03/labeling_diagnostics.json` do support the quality statistics cited in the paper:

- overall `label_quality_mean = 0.6124`
- long reversal `0.6538`
- short reversal `0.6479`
- long continuation `0.6124`
- short continuation `0.5865`

Realized barrier economics are softer than the theoretical RR because of timeouts and stopped trades, but they are not tiny under the nominal ES schedule:

- mean-reversion usable-event median target: roughly `29.87-31.64` ticks
- continuation usable-event median target: roughly `33.02-35.94` ticks
- realized family RR from usable events: about `0.852` mean reversion and `0.931` continuation

That means Q7 is not answered by "the targets are only a couple of ticks." Under the nominal ES session-cost schedule, the target widths are not marginal. Under the saved Phase 7 cost implementation, they become marginal only because the backtest is charging bar-range-driven pseudo-spread costs.

### Phase 4: Preprocessing and Attribution

The saved prepare artifacts explain the open/day-type attribution gap more cleanly than the paper does.

For reversal targets, `frvp_open_type` is removed as low variance, not merely ignored by SHAP:

- long reversal removed columns include `frvp_open_type`, `frvp_open_drive_flag`, `frvp_gap_into_value`, and `frvp_failed_auction_with_sweep` in `artifacts/frvp_es_primary_current/phase04/prepared/long_frvp_reversal/report.json`
- short reversal shows the same pattern in `artifacts/frvp_es_primary_current/phase04/prepared/short_frvp_reversal/report.json`

That is structurally expected, because the reversal event universe has already been filtered upstream:

- Setup 1 requires `frvp_open_type == inside_value` at `frvp/setups/detector.py:283-289`
- Setup 6 requires `frvp_open_type == inside_value` at `frvp/setups/detector.py:466-469`
- Setup 4, the only reversal family that could have reintroduced open-type variety, is disabled in the saved labeling pipeline at `frvp/pipelines/es_primary_phase04.py:101`

So for the two reversal targets, `frvp_open_type` is not "missing signal the model forgot to use." The rule layer has already collapsed it.

For continuation targets, both `frvp_open_type` and `frvp_day_type` survive preprocessing, but neither ranks as an important marginal feature:

- `frvp_open_type` rank `49` for long continuation, absent from short continuation merged top rows
- `frvp_day_type` rank `42` for long continuation and `73` for short continuation
- no saved target shows evidence of categorical encoding dilution, because the prepare summary shows no categorical expansion

My conclusion is:

- `frvp_open_type` absence on reversal is structural
- `frvp_day_type` absence is mostly genuine low marginal predictive value on the saved contracts
- feature re-encoding is lower value than fixing economics and family pooling

### Phase 5: Training

The direct XGBoost models are not literal copy-pastes, but they do share a common scaffold and some common weak points.

| Model | CV AP | Test AP | CV F0.5 | Test F0.5 | Calibration |
|---|---:|---:|---:|---:|---|
| `frvp_long_reversal_xgb_v1` | `0.7088` | `0.7343` | `0.7680` | `0.7452` | `platt` |
| `frvp_short_reversal_xgb_v1` | `0.7266` | `0.7537` | `0.7681` | `0.7904` | `platt` |
| `frvp_long_continuation_xgb_v1` | `0.7460` | `0.6975` | `0.8195` | `0.7834` | `none` |
| `frvp_short_continuation_xgb_v1` | `0.7242` | `0.7263` | `0.8069` | `0.7384` | `none` |

These values come from the four training summaries under `models/frvp_es_primary_xgb_v1/.../training_summary.json`.

Findings:

- The continuation models are not obviously under-ranked; their AP and F0.5 are acceptable.
- `frvp_long_continuation_xgb_v1` shows the clearest train/test degradation, with AP dropping from `0.7460` to `0.6975`.
- No direct XGBoost target uses explicit class weighting: the training summaries show `class_weight_mode = None` and no active `scale_pos_weight`.
- Reversal models are calibrated with Platt scaling; continuation models have calibration intentionally disabled.
- Hyperparameters vary by target, so this is not a simple copy-paste from the OTE incumbent, but all four still share the same sequence-window framing around `window_size = 32`.

### Phase 6: Regime/Day-Type Slices and Threshold Policy

The saved slice artifacts do contain regime/day-type dimensions, but the saved threshold-policy run does not actually test the regime-gated deployment idea the paper says to prioritize.

In `model_testing/reports/frvp_threshold_policies/frvp_es_primary_current/run_summary.json`, all four direct XGBoost model outputs show:

- `abstain_session_regimes = []`
- `abstain_composite_regimes = []`
- `abstain_composite_session_pairs = []`
- `abstain_composite_stress_pairs = []`

So the search did not discover or enforce any real regime/day-type gating. It only compared global versus regime-threshold variants plus a generic abstain layer.

Best saved test-policy results are still negative for all four models:

- long reversal best test policy: `regime_threshold`, post-cost expectancy `-35.33`
- short reversal best test policy: `global_threshold_plus_abstain`, post-cost expectancy `-36.46`
- long continuation best test policy: `global_threshold`, post-cost expectancy `-34.90`
- short continuation best test policy: `regime_threshold_plus_abstain`, post-cost expectancy `-81.97`

The important nuance is that the saved backtests do show profitable sub-populations for some models, especially long reversal:

- `frvp_long_reversal_xgb_v1` is positive on `neutral` day types (`+316.65`, `+5.37` expectancy) and strongly positive in `strong_down_medium` composite regime (`+1628.20`, `+17.70` expectancy)
- `frvp_short_reversal_xgb_v1` has no robust profitable pocket under the current saved cost model; `asia` is roughly flat (`+13.05` total)
- both continuation models are negative across all major saved setup/day-type buckets under the current cost implementation

So Q6 is not confirmed by the current saved run, but regime mixing is still materially diluting at least the long-reversal model.

### Phase 7: Backtest and DSR Economics

This phase contains the main diagnosis.

Saved selected-test backtest outcomes from `model_testing/reports/frvp_backtests/frvp_es_primary_current/model_summary.csv`:

| Model | Trades | Net PnL | Expectancy | Profit Factor | Positive Composite Share | Accepted |
|---|---:|---:|---:|---:|---:|---|
| `frvp_long_reversal_xgb_v1` | `341` | `-8215.65` | `-24.09` | `0.577` | `0.2222` | `False` |
| `frvp_short_reversal_xgb_v1` | `476` | `-14394.40` | `-30.24` | `0.478` | `0.0000` | `False` |
| `frvp_long_continuation_xgb_v1` | `1687` | `-51450.55` | `-30.50` | `0.542` | `0.0000` | `False` |
| `frvp_short_continuation_xgb_v1` | `2429` | `-119360.85` | `-49.14` | `0.395` | `0.0000` | `False` |

Gross-vs-net decomposition on the saved selected trades:

- long reversal: gross `+1964.0`, net `-8215.65`
- short reversal: gross `-799.0`, net `-14394.40`
- long continuation: gross `+14799.0`, net `-51450.55`
- short continuation: gross `-9227.0`, net `-119360.85`

That immediately splits the models into two groups:

- long reversal and long continuation have positive gross edge that is being overwhelmed after costs
- short reversal and short continuation are already gross-negative, so friction is not the only problem

The largest specific killer is the cost model itself. `features/feature_sets/microstructure.py:23-27` defines `approx_spread = high - low`. Then `model_testing/ote_threshold_policy.py:392-418` treats that field as an entry and exit spread, divides by ES tick size, and adds more slippage on top. On the saved selected test trades this yields:

| Model | Mean Actual Cost | Mean Session-Only Schedule Cost | Actual / Schedule |
|---|---:|---:|---:|
| `frvp_long_reversal_xgb_v1` | `29.85` | `3.37` | `8.87x` |
| `frvp_short_reversal_xgb_v1` | `28.56` | `3.37` | `8.48x` |
| `frvp_long_continuation_xgb_v1` | `39.27` | `3.42` | `11.49x` |
| `frvp_short_continuation_xgb_v1` | `45.34` | `3.46` | `13.11x` |

This is not a small modeling choice. It dominates the economics.

I therefore computed a simple counterfactual using the same saved selected trades, the same saved gross PnL, and the same ES session-cost schedule from `run_summary.json`, but without the bar-range-driven `approx_spread` charge:

| Model | Saved Net | Session-Schedule-Only Net | Session-Schedule-Only Expectancy |
|---|---:|---:|---:|
| `frvp_long_reversal_xgb_v1` | `-8215.65` | `+816.10` | `+2.39` |
| `frvp_short_reversal_xgb_v1` | `-14394.40` | `-2402.15` | `-5.05` |
| `frvp_long_continuation_xgb_v1` | `-51450.55` | `+9034.95` | `+5.36` |
| `frvp_short_continuation_xgb_v1` | `-119360.85` | `-17630.85` | `-7.26` |

That arithmetic counterfactual is not a full gate rerun, so it does not prove promotion readiness. It does prove that the saved current-cost results materially overstate how much friction the long models must overcome.

Other Phase 7 decompositions still matter:

- Setup-family drag under current saved costs:
  - long reversal: both Setup 1 and Setup 6 are negative
  - short reversal: Setup 1 is much worse than Setup 6
  - long continuation: Setup 5 is the largest drag
  - short continuation: Setup 3 is the worst expectancy, Setup 5 the biggest total drag
- Session/time-of-day under current saved costs:
  - long reversal is worst in `midday` and `new_york`
  - short reversal is worst in `morning`, `midday`, and `overlap`
  - long continuation is negative in every major session bucket, least bad in `asia`
  - short continuation is especially weak in `london` and `morning`
- Roll-bracket trades:
  - long reversal roll-bracket trades are actually positive (`22` trades, `+43.08` expectancy)
  - continuation roll-bracket trades are worse than non-roll trades, especially short continuation (`197` trades, `-71.58` expectancy)

So roll leakage is not what is killing long reversal, but roll sensitivity still deserves a continuation-focused A/B under Q1.

## 3. Per-Model Root-Cause Diagnosis

### `frvp_long_reversal_xgb_v1`

Ranking is not the problem first. The model posts `0.7088` CV AP and `0.7343` test AP, with `0.7680` CV F0.5 and `0.7452` test F0.5 in `models/frvp_es_primary_xgb_v1/long_frvp_reversal/training_summary.json`. Phase 7 shows positive saved gross PnL (`+1964.0`) but negative saved net PnL (`-8215.65`). The dominant cause is `d) friction model mismatch`: mean saved trade cost is `29.85` units, `8.87x` the nominal session-cost schedule, because the backtest uses `approx_spread = high - low` as spread. Under the session-schedule-only arithmetic counterfactual, the same saved trades become `+816.10` net.

There is still a secondary `e) regime mixing` issue. Under current saved costs, the only robust positive pockets are `neutral` day type and `strong_down_medium` composite regime. Setup 1 is the better family under the session-only counterfactual (`+5.77` expectancy), while Setup 6 remains slightly negative (`-2.01`). That means even after a friction fix, this is not a universal long-reversal deployment; it is a selective one. The Phase 4 attribution gap for `frvp_open_type` is not the root problem here, because reversal events are already structurally inside-value only, and `frvp_open_type` is removed as low variance before training.

Root-cause chain: `d) friction mismatch` -> `e) profitable pocket diluted by broad deployment` -> minor `b) attribution gap is mostly structural, not a missing feature bug`.

### `frvp_short_reversal_xgb_v1`

This model also ranks well on paper, at `0.7266` CV AP, `0.7537` test AP, and `0.7904` test F0.5, but economics are bad for two reasons. First, `d) friction mismatch` still matters: saved mean trade cost is `28.56` units, `8.48x` the session schedule. Second, unlike long reversal, the model is already gross-negative (`-799.0`), so friction is not sufficient to explain the failure.

The saved family mix shows a clear `e) regime/setup mixing` problem. Under current saved costs, Setup 6 loses less than Setup 1. Under the session-only counterfactual, Setup 6 becomes positive (`+4.74` expectancy), while Setup 1 remains clearly negative (`-15.69`). The best pockets are also concentrated: `asia` session is positive (`+23.60` expectancy session-only), `normal_variation` day type is positive (`+11.08`), and `ranging_medium` is positive (`+13.56`), but `neutral`, `new_york`, and `strong_down_medium` remain bad.

This model therefore looks like a mix of `d) friction mismatch`, `e) pooled-family dilution`, and `f) genuine signal weakness` in Setup 1. The post-2022 stability check is also unfavorable for reversal models, which makes Q11 relevant here.

Root-cause chain: `e) Setup 1 drags pooled short reversal` + `d) friction mismatch magnifies losses` + `f) residual short-reversal weakness remains even under milder costs`.

### `frvp_long_continuation_xgb_v1`

This is the clearest example of a model that is ranking-good but economics-bad because the economics stack is harsher than the paper contract. The model records `0.7460` CV AP and `0.6975` test AP, with positive saved gross PnL of `+14799.0`, but saved net PnL of `-51450.55`. Mean saved trade cost is `39.27` units, `11.49x` the session schedule. Under the session-schedule-only arithmetic counterfactual, the same trades become `+9034.95` net with `+5.36` expectancy, and all three continuation setup families are positive.

There is a real secondary issue in `a) label/barrier construction`: the saved pipeline tightened continuation TP from the paper's `1.5 ATR` to an effective `1.0/0.95/0.90/0.85 ATR` family in `frvp/pipelines/es_primary_phase04.py:93-110`. That shrinks the gross edge per trade even before friction. There is also some `f) drift/overfit` signal, because this model has the largest CV-to-test AP drop of the four direct XGB targets.

Current saved Phase 6 results do not show a robust profitable pocket under the current cost implementation, but the session-only counterfactual shows the model is broad-based positive across Setup 2, Setup 3, Setup 5, and most major session buckets except `off_hours`. That strongly suggests the first experiment should isolate the friction layer before changing the classifier.

Root-cause chain: `d) friction mismatch` primary -> `a) continuation TP tightened versus paper reduces gross margin` -> secondary `f) drift/overfit` caution.

### `frvp_short_continuation_xgb_v1`

This is the weakest direct XGBoost target economically. Although test AP (`0.7263`) and test F0.5 (`0.7384`) are acceptable, the model is gross-negative (`-9227.0`) and massively net-negative (`-119360.85`). The saved friction layer makes it worse, with mean cost `45.34` units, but even the session-schedule-only arithmetic counterfactual remains negative at `-17630.85`.

The decomposition points to `e) regime/setup mixing` and `f) genuine lack of deployable signal after costs` in the pooled continuation family. Setup 2 is nearly flat under the session-only counterfactual (`+0.19` expectancy), but Setup 3 (`-16.27`) and Setup 5 (`-6.06`) remain negative. Some composite buckets are positive (`ranging_medium`, `strong_up_medium`), but the large buckets are still negative, especially `london`, `strong_down_high`, and `normal_variation`. Phase 4 also offers little help from `frvp_day_type` or `frvp_open_type`, which remain weak marginal features rather than hidden alpha sources.

Root-cause chain: `f) gross edge is weak/negative in the dominant setup families` + `e) pooled continuation model mixes unlike subfamilies` + `d) friction mismatch makes the already-bad result catastrophic`.

## 4. Ranked Experiment Backlog

### 4.1 Friction A/B: Session Schedule vs. Bar-Range Spread Proxy

**Hypothesis**

The saved net failures for the long FRVP models are primarily caused by the backtest pricing `approx_spread` as true spread, even though `approx_spread` is defined as full candle range in `features/feature_sets/microstructure.py:23-27`. This is causing Phase 7 to fail gate 2 and gate 3 for reasons that are partly implementation-driven rather than model-driven.

**Change**

Run a controlled Phase 6/7 A/B using the current saved models and current saved gross trade paths, but vary only the friction input in `model_testing/ote_threshold_policy.py:392-418`:

1. current implementation
2. session-schedule-only spread from `model_testing/reports/frvp_threshold_policies/frvp_es_primary_current/run_summary.json`
3. capped or alternative ES-specific spread proxy, if you want a third arm

Do not touch labels or model weights in this experiment.

**Control**

Keep models, thresholds, labels, and train/test splits fixed. Change only the trade-cost computation.

**Success Metric**

Section 10 gates:

- gate 2: walk-forward Sharpe after frictions `> 0.8`
- gate 3: walk-forward max drawdown `< 12%`
- gate 4: Deflated Sharpe `> 0.3`

**Cost/Risk**

Low compute. No leakage risk. High information value. This should be first because it tells you whether long reversal and long continuation are truly bad, or merely over-penalized by the current cost layer.

### 4.2 Regime-Gated Deployment and Joint Threshold Search

**Hypothesis**

At least `frvp_long_reversal_xgb_v1` has a real profitable sub-population that the global policy is diluting. The saved threshold search never actually populated regime abstain lists, so the paper's gating idea remains untested rather than disproven.

**Change**

Extend the threshold-policy search in `model_testing/ote_threshold_policy.py` and the corresponding runner so that `abstain_session_regimes`, `abstain_composite_regimes`, and composite-session pair filters are genuinely searched for FRVP models, using the slice dimensions already present in `model_testing/reports/frvp_regime_slices/frvp_es_primary_current/...`.

Priority candidates from the saved audit:

- long reversal: keep `strong_down_medium`, test `neutral` day-type emphasis
- short reversal: test Setup 6 / `asia` / `ranging_medium`
- short continuation: test whether `ranging_medium` and `strong_up_medium` are the only buckets worth keeping

**Control**

Keep models and labels fixed. Change only the policy layer.

**Success Metric**

Section 10 gates:

- gate 2: walk-forward Sharpe `> 0.8`
- gate 5: reversal models should win in at least two range/normal/neutral buckets; continuation models in at least two trend/high-vol buckets

**Cost/Risk**

Moderate compute. Main risk is policy overfitting to sparse buckets, so require the same walk-forward discipline and minimum-events-per-month constraints already in the saved threshold search.

### 4.3 Per-Setup-Family Models Instead of Pooled Direction Families

**Hypothesis**

The current per-direction family pooling is hiding materially different economics:

- short reversal Setup 1 is much worse than Setup 6
- short continuation Setup 3 and Setup 5 are dragging the family
- long continuation gross edge may be broad, but the pooled model still inherits tighter-than-paper continuation labels

**Change**

Create separate prepared targets and model branches by setup family rather than only by direction family. Concretely, this means modifying the Phase 3/4 target construction path driven by `data/labeling/frvp_labeling_engine.py` and `frvp/pipelines/es_primary_phase04.py`, then training per-setup models under the same registry/backtest framework.

Treat Setup 4 separately if it is re-enabled; do not immediately re-pool it into reversal.

**Control**

Keep the same feature set, same train/test windows, and same Phase 7 evaluation contract.

**Success Metric**

Section 10 gates:

- gate 2: Sharpe `> 0.8`
- gate 5: setup-aligned slice wins should become clearer rather than blurrier
- gate 1: OOF AUC-PR should remain above the `> 0.60` target

**Cost/Risk**

Moderate compute and some sample-size risk, especially for thin setups. Worth it because the current pooled models are economically heterogeneous.

### 4.4 Q11 Recency Weighting or Post-2022 Primary Window for Reversal

**Hypothesis**

Pre-2022 reversal data is hurting post-2022 generalization. The OOF stability check shows large pre/post-2022 ranking shifts for both reversal targets.

**Change**

Run a controlled training-window experiment for the reversal models only:

1. current window
2. recency-weighted full window
3. post-2020 or post-2022 primary window

Implement in the training pipeline and registry path used by the current direct XGBoost FRVP models.

**Control**

Keep features, labels, and Phase 7 evaluation unchanged. Change only the sample weighting or start date.

**Success Metric**

Section 10 gates:

- gate 1: OOF AUC-PR stays above `0.60`
- gate 2: walk-forward Sharpe after frictions `> 0.8`
- gate 6: placebo gap remains above `3%`

**Cost/Risk**

Moderate compute. Main risk is sample shrinkage. Based on the stability evidence, this is higher EV for reversal than for continuation.

### 4.5 Q7 Mean-Reversion Edge-Floor Retune, But Only After the Friction A/B

**Hypothesis**

After the friction layer is put on a defensible ES footing, reversal edge may still need more margin over costs, especially for Setup 6 and short reversal Setup 1. The saved data do not support "targets are only a handful of ticks" under nominal costs, but they do support "some mean-reversion subfamilies do not have enough cushion."

**Change**

In `data/labeling/frvp_labeling_engine.py` and the Phase 3 override block in `frvp/pipelines/es_primary_phase04.py`, test a small grid that widens or filters the mean-reversion family:

- raise TP from `0.8` to `0.9` or `1.0 ATR`
- optionally impose a minimum expected-move-to-cost ratio for reversal labels
- optionally filter low-ATR or high-proxy-spread sessions

**Control**

Do not change the continuation family in this experiment. Do not combine with a feature-set change.

**Success Metric**

Section 10 gates:

- gate 2: Sharpe `> 0.8`
- gate 3: max drawdown `< 12%`
- gate 1: AP should not collapse below the paper threshold

**Cost/Risk**

Moderate compute and moderate label-drift risk. Leakage risk is low if the same causal ATR and exclusion rules remain intact.

### 4.6 Q1 Roll Translation vs. Reset-at-Roll

**Hypothesis**

Roll handling is not leaking, but continuation economics may be more sensitive to reset-at-roll than reversal economics. Roll-bracket-adjacent continuation trades are materially worse in the saved selected trades.

**Change**

Run the paper's explicit Q1 A/B:

- current reset-at-roll path
- translate-by-spread path for cross-roll absolute levels

Target the same Phase 3 labels and Phase 7 backtests, with special attention to continuation families.

**Control**

Keep every non-roll component fixed.

**Success Metric**

Section 10 gates:

- gate 2: Sharpe `> 0.8`
- gate 3: max drawdown `< 12%`
- plus a reduction in roll-bracket-specific underperformance

**Cost/Risk**

Moderate implementation effort. Leakage risk is exactly why this must be isolated and audited carefully, but the continuity tests already give a good baseline.

### 4.7 Re-Enable Setup 4 as a Standalone Research Branch

**Hypothesis**

The saved repo has not actually tested the paper's failed-auction thesis because Setup 4 labels are disabled outright. Setup 4 may still be too thin to be production-worthy, but that is an empirical question, not a question the current saved model family can answer.

**Change**

Flip `enable_failed_auction_labels` back on in the Phase 3 parameter path in `frvp/pipelines/es_primary_phase04.py`, but evaluate Setup 4 as its own branch rather than immediately pooling it into reversal.

**Control**

Keep all other setup families unchanged.

**Success Metric**

Section 10 gates:

- gate 1: AP above threshold
- gate 6: placebo gap above `3%`
- if sample size is too low to support Phase 7 gating, that itself answers the question

**Cost/Risk**

Moderate compute and high sparsity risk. Worth doing because the current saved branch has not tested the hypothesis at all.

### 4.8 Feature Re-Encoding for `frvp_open_type` and `frvp_day_type`

**Hypothesis**

Low EV. The saved audit finds weak evidence that encoding is the real problem.

**Change**

Only consider this after the economics experiments above. If pursued, test target-specific encoding or interaction features for continuation models only, because reversal models already collapse `frvp_open_type` upstream.

**Control**

Keep labels and cost model fixed.

**Success Metric**

Section 10 gate 1 first, then gate 2. If AP improves but Sharpe does not, this experiment did not solve the real problem.

**Cost/Risk**

Low compute, low leakage risk, but probably low expected value. The saved contracts show no one-hot dilution bug, and `frvp_day_type` has near-zero MI.

## 5. Explicit Answers to Q1, Q6, Q7, Q11

### Q1: Roll method sensitivity

Still open. I verified that the current reset-at-roll implementation is causal and well-tested, and I did not find continuity leakage. I did not find a saved translate-by-spread A/B artifact, so I cannot close the question from repo state alone. What I can say is:

- long reversal is not being killed by roll-bracket trades
- continuation models do underperform on roll-bracket-adjacent trades

That makes Q1 secondary but still worth testing, especially for continuation.

### Q6: Day-type vs. setup-family interaction

Not confirmed by the saved evidence. The paper's hypothesis was "reversal on balance, continuation on trend." The saved artifacts do not support that cleanly:

- long reversal is strongest on `neutral`, which is directionally consistent
- short reversal is worst on `neutral`
- long continuation is least bad on `neutral` under current saved costs and broadly positive across many day types in the session-only counterfactual
- short continuation is negative across every major day type

The more defensible conclusion is that setup-family heterogeneity matters more than the simple reversal/balance versus continuation/trend split. If you simplify anything, simplify toward per-setup research, not toward trusting the current family split.

### Q7: Mean-reversion family vs. friction floor

Partially answered, but not closed. Under the current saved cost implementation, the mean-reversion family fails economically. Under a session-schedule-only arithmetic counterfactual, long reversal becomes positive and short reversal becomes much less negative, which means the saved repo cannot yet tell you whether mean reversion truly fails a realistic ES friction floor or whether it is being over-penalized by the current cost contract.

So the current answer is:

- mean reversion definitely fails under the saved current backtest implementation
- that result is not yet trustworthy as a clean economics answer, because the friction layer is pricing full candle range as spread
- once the friction A/B is run, revisit mean-reversion TP edge-floor tuning

### Q11: 0DTE regime non-stationarity pre/post-2022

Yes, especially for reversal. Using the saved OOF prediction artifacts joined back to source datetimes:

- long reversal AP drops from `0.5847` pre-2022 to `0.5177` post-2022
- short reversal AP drops from `0.6413` pre-2022 to `0.5070` post-2022
- continuation shifts are smaller: long continuation `0.5823 -> 0.5658`, short continuation `0.5142 -> 0.5258`

That makes recency weighting or a post-2020/post-2022 primary window a high-priority reversal experiment and only a medium-priority continuation experiment.

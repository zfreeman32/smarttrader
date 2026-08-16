# FRVP / ICT Live Signal Output Audit

Date: `2026-08-11`

This audit answers a specific operational question: live shadow predictions for the FRVP and ICT ES families cross their decision threshold far more often than the OTE EUR/USD champions do. Is that a real signal-quality problem, and if so, where does it come from? The read draws on the live audit database (`ote_live/runtime_data/live_market_data.sqlite3`), the live runtime manifests, saved training summaries, and the existing FRVP/ICT backtest and concentration reports. It is meant to sit alongside `FRVP_ES_primary_audit_report.md` and `ICT_SHORT_SIDE_CONCENTRATION_AUDIT_20260728.md` as a signal-layer companion piece, not a replacement for either.

## 0. Scope And Safety Context

All 12 FRVP/ICT models discussed here run in **shadow mode on ES futures** — a separate instrument from the EUR/USD OTE production pipeline. Shadow mode means every prediction is logged with a `shadow` decision and a `candidate_passed_threshold` / `probability_below_threshold` tag, but nothing is ever emitted as a live trade. None of the findings below describe live capital risk today. They describe signal-quality and promotion-readiness issues that should be resolved before any of these models leave shadow mode.

## 1. Headline Finding: Threshold-Crossing Rate By Family

Live shadow predictions since `2026-04-09` (`50,671` total predictions across all models):

| Family | Above-threshold rate (range across models) | Reference |
|---|---|---|
| FRVP (6 ES models) | `27% - 100%` of predictions | |
| ICT (6 ES models) | `5% - 22%` of predictions | |
| OTE EUR/USD champions (8 live models) | `0% - 1%` of predictions | baseline for comparison |

Per-model detail:

| Model | n predictions | % above threshold | avg threshold |
|---|---|---|---|
| `frvp_long_continuation_xgb_v1` | 265 | 41.1% | 0.700 |
| `frvp_long_meta_xgb_v1` | 265 | 62.6% | 0.450 |
| `frvp_long_reversal_xgb_v1` | 265 | 67.9% | 0.600 |
| `frvp_short_continuation_tcn_v1` | 264 | 27.3% | 0.527 |
| `frvp_short_meta_xgb_v1` | 265 | 82.3% | 0.431 |
| `frvp_short_reversal_xgb_v1` | 31 | 100.0% | 0.500 |
| `ict_long_continuation_xgb_v1` | 264 | 10.6% | 0.650 |
| `ict_long_meta_xgb_v1` | 264 | 10.6% | 0.388 |
| `ict_long_reversal_xgb_v1` | 264 | 0.0% | 0.400 |
| `ict_short_continuation_xgb_v1` | 264 | 8.7% | 0.550 |
| `ict_short_meta_xgb_v1` | 264 | 21.6% | 0.344 |
| `ict_short_reversal_xgb_v1` | 264 | 4.9% | 0.400 |
| OTE champions (8 models, EUR/USD) | 634 - 4086 each | 0.0% - 1.0% | 0.47 - 0.91 |

The OTE champions almost never cross threshold by design — they are tuned for roughly 1-3 trades/day. FRVP and ICT are crossing threshold an order of magnitude more often, which is the behavior that prompted this audit.

Sanity check on the rule engine itself: raw FRVP/ICT setup events are only about `1.1% - 1.5%` of all bars (`rows_usable` vs. total dataset rows in the training summaries), which is a normal event-driven fire rate. **The high threshold-crossing rate is not coming from an overzealous setup detector.** It is coming from the meta-labeling layer downstream of the setup detector.

## 2. Root Cause: Weak Discrimination Meeting High Base Rates

Two things compound:

1. **Label base rates are high.** FRVP meta/reversal/continuation targets carry a `29% - 51%` positive rate in training (vs. OTE's much rarer, harder-won positive class).
2. **Discrimination is weak on several branches.** ROC AUC ranges `0.51 - 0.68` across FRVP/ICT models. For comparison, the OTE short-meta TCN champion scores ROC AUC `0.98`, Brier `0.014`.

When a threshold sits close to a 30-50% base rate and the model is only weakly separating signal from noise, probabilities cluster near the threshold and cross it constantly — a high fire rate here does not mean "lots of good signals," it means the model is not confidently discriminating.

`AP lift over base rate` isolates real skill: it is average precision minus the label base rate, i.e. how much better than a random classifier the model actually is.

| Model | Base rate | Test AP | **AP lift** | ROC AUC | Threshold |
|---|---|---|---|---|---|
| `frvp_long_continuation_xgb_v1` | 48.5% | 0.714 | 0.229 | 0.595 | 0.680 |
| `frvp_long_meta_xgb_v1` | 47.9% | 0.546 | **0.067** | 0.597 | 0.500 |
| `frvp_long_reversal_xgb_v1` | 46.6% | 0.705 | 0.239 | 0.641 | 0.620 |
| `frvp_short_continuation_tcn_v1` | 48.5% | 0.711 | 0.226 | 0.514 | 0.620 |
| `frvp_short_meta_xgb_v1` | 48.0% | 0.494 | **0.014** | 0.525 | 0.500 |
| `frvp_short_reversal_xgb_v1` | 47.0% | 0.751 | 0.281 | 0.669 | 0.680 |
| `ict_long_continuation_xgb_v1` | 51.1% | 0.519 | **0.009** | 0.534 | 0.620 |
| `ict_short_continuation_xgb_v1` | 45.1% | 0.510 | **0.059** | 0.553 | 0.440 |
| `ict_long_meta_xgb_v1` | 35.5% | 0.493 | 0.138 | 0.675 | 0.320 |
| `ict_short_meta_xgb_v1` | 32.0% | 0.451 | 0.131 | 0.653 | 0.320 |
| `ict_long_reversal_xgb_v1` | 31.2% | 0.450 | 0.138 | 0.683 | 0.320 |
| `ict_short_reversal_xgb_v1` | 29.3% | 0.421 | 0.127 | 0.665 | 0.290 |

Reading this table:

- **Reversal branches carry the real edge.** Both FRVP reversal models (lift `0.24 - 0.28`, AUC `0.64 - 0.67`) and ICT reversal models (lift `0.13`, AUC `0.68`) show genuine skill above the base rate.
- **Meta and continuation branches are the weak links**, especially `frvp_short_meta` (lift `0.014`), `ict_long_continuation` (lift `0.009`), and `ict_short_continuation` (lift `0.059`) — these are close to indistinguishable from noise on a pure classifier-metric basis, even though their live backtests can still show a post-cost edge from policy-layer filtering (see Section 4).

## 3. Pooled-Model Dilution: The Same Disease In Both Codebases

This is the most actionable finding, and it independently shows up in the repo's own prior audits on both sides:

- **ICT** (`ICT_SHORT_SIDE_CONCENTRATION_AUDIT_20260728.md`): `ict_short_meta_xgb_v1` shares `284` exact walk-forward trades with `ict_short_reversal_xgb_v1`. Those shared trades account for `66.5%` of the meta model's net PnL and `85.1%` of the reversal model's net PnL. The pooled short-meta model also shares `84` trades with `ict_short_continuation_xgb_v1`, whose unique-only remainder is negative (`-63.65` ticks over `81` trades). The pooled model is acting as a catch-all that absorbs the profitable reversal and continuation subsets rather than behaving like an independent bet.
- **FRVP** (`frvp_deferred_research_audit_20260728/DEFERRED_RESEARCH_SUMMARY.md`): the long branches are dominated by a single Phase-3 setup type (Setup 1 or Setup 5 carrying `64% - 72%` of trade share), with wide expectancy variance across setup types pooled into the same branch (e.g. best-setup expectancy `108.43` vs. worst-setup expectancy `18.79` within the same reversal branch; best `52.51` vs. worst `-0.81` within the same continuation branch).

**Same mechanism in both families: pooling heterogeneous setup types into one meta-label averages away the good setups' edge and lets the bad setups' noise leak into the aggregate probability.** This is consistent with the low AP-lift numbers in Section 2 for the meta/continuation branches specifically, and it is already the top-priority open item in `notes.txt` for FRVP (per-setup-family split) — this audit adds the ICT-side quantitative confirmation of the same pattern.

## 4. Concentration And Crisis-Day Dependency (ICT Short Side)

Both `ict_short_meta_xgb_v1` and `ict_short_reversal_xgb_v1` fail the promotion-quality gate on the same two criteria:

- single largest trade above the `10%` cap (`10.02%` and `11.12%` respectively)
- profitable-quarter share below the `60%` floor (`59.3%` and `55.6%`)

Top-5 trades contribute `33% - 38%` of total net PnL; top-10 contribute `55% - 61%`. `2020-03-17` (the COVID crash) shows up as a top-3 winning day for both branches. A meaningful chunk of the backtest edge is shock-day dependent rather than a steady repeatable edge — already documented in the ICT concentration audit, restated here because it directly explains part of why the meta/reversal probabilities look confident in backtest despite weak standalone AUC: a handful of very large, very correlated events are doing a disproportionate share of the work.

## 5. FRVP Short Family Is Currently Non-Viable, Not Just Under-Threshold

From `frvp_short_family_leaders_refresh_20260715_accountdd/model_summary.csv`:

| Model | Sharpe | Deflated Sharpe | Max account DD | Profit factor |
|---|---|---|---|---|
| `frvp_short_continuation_tcn_v1` | -0.38 | -0.37 | **109.2%** | 0.93 |
| `frvp_short_meta_xgb_v1` | 0.07 | 0.07 | 54.6% | 1.02 |
| `frvp_short_reversal_xgb_v1` | -0.39 | -0.39 | 48.7% | 0.88 |

A `109%` max drawdown means this branch would blow the reference account outright. All three fail the drawdown gate. This is not a threshold-tuning problem — the continuation and reversal branches do not currently have tradable edge on the short side, and no amount of threshold/filtering work fixes that. Only `frvp_short_meta_xgb_v1` is even marginally alive (Sharpe `~0`), consistent with its status as "cleanest short sentinel" in the existing FRVP experiment record.

## 6. Two ICT Models Already Clear All 8 Promotion-Quality Gates

From the July 30, 2026 ICT backtest (`ict_es_backtests/ict_es_shadow_20260730/model_summary.csv`):

| Model | Sharpe | Deflated Sharpe | Max DD | `accepted_for_paper_trading_gate` |
|---|---|---|---|---|
| `ict_long_meta_xgb_v1` | 1.586 | 1.068 | 6.40% | **True** — 8/8 gates |
| `ict_long_reversal_xgb_v1` | 1.269 | 1.023 | 8.40% | **True** — 8/8 gates |
| `ict_short_meta_xgb_v1` | 0.905 | 0.820 | 7.13% | False (concentration/quarter-breadth) |
| `ict_short_reversal_xgb_v1` | 1.047 | 0.914 | 7.82% | False (concentration/quarter-breadth) |
| `ict_short_continuation_xgb_v1` | 0.326 | 0.322 | 7.01% | False |
| `ict_long_continuation_xgb_v1` | 0.009 | 0.009 | 5.06% | False (flat economics) |

`ict_long_meta_xgb_v1` and `ict_long_reversal_xgb_v1` both show `accepted_for_paper_trading_gate = True` with 8/8 promotion-quality gates passed, yet both remain at generic `candidate` status in the live runtime manifest with no promotion action recorded. This is the one place in this audit where the finding is "there may be a validated model sitting on the shelf" rather than "filter this harder" — worth an explicit decision (promote to paper trading, or document the specific reason it's being held back).

## 7. Filtering Tips

1. **Don't trust raw `calibrated_probability >= threshold` as a signal-quality proxy for meta/continuation branches.** For `frvp_short_meta`, `frvp_long_meta`, `ict_long_continuation`, and `ict_short_continuation` specifically, the probability is barely more informative than the base rate (AP lift under `0.07`). If a live-usable confidence signal is needed from these branches, use **probability rank/quantile within a rolling window** instead of the raw calibrated value — a `0.52` in a distribution that swings `0.36-0.54` is not the same thing as a `0.52` in a distribution that swings `0.0001-0.03`.
2. **Weight conviction by AP-lift-over-base-rate per branch, not by family label.** Trust FRVP short-reversal (`0.28` lift), FRVP long-reversal (`0.24`), and ICT long/short reversal and meta (`0.13`) well ahead of `frvp_short_meta`, `frvp_long_meta`, `ict_long_continuation`, or `ict_short_continuation` (all under `0.07` lift).
3. **Check trade-level overlap before treating "meta" and "family" signals firing together as independent confirmation.** With `85%` overlap between ICT short-meta and short-reversal, two signals firing together on the short side is very likely the same underlying event counted twice, not two independent confirmations stacking conviction.
4. **Session/regime-gate the short-side ICT models manually if paper-traded ahead of a policy fix.** Documented net-loser pockets: for `ict_short_meta_xgb_v1` — `ranging_medium/asia`, `ranging_high/new_york`, `strong_down_medium/off_hours`; for `ict_short_reversal_xgb_v1` — `ranging_high/new_york`, `strong_down_medium/off_hours`, `ranging_medium/off_hours`.
5. **Treat single-day/single-trade-dominated backtests with suspicion.** If a candidate's trade log shows `2020-03-17` or another crisis day in the top-3 PnL contributors, discount the headline Sharpe/DSR until performance excluding that date has been checked.
6. **There is already an unused lever for this in the live policy contract.** `abstain_policy.minimum_probability_quantile` in the FRVP/ICT `live_policy.json` files is currently `null` on every model checked. This is exactly the quantile-based filter described in tip 1, already wired into the runtime abstain path — it just needs a value and a validation pass instead of new code.

## 8. Summary Of Correction Work

Ranked by leverage, cross-referenced against what is already tracked in `notes.txt` / the experiment journals so this list does not duplicate open work:

1. Split pooled meta models into per-setup-family models (FRVP: already top-priority in `notes.txt`; ICT: this audit's Section 3 is the quantitative confirmation the same disease exists on the ICT side).
2. Run the short-side regime-gated concentration/overlap cleanup (already parked in `notes.txt`; this audit reinforces it with the AP-lift and shock-day evidence).
3. Stop investing further threshold/filtering work in the FRVP short family until the underlying setup logic changes — the continuation branch's `109%` drawdown is an economics problem, not a filtering problem (already reflected in `notes.txt`; this audit adds the concrete drawdown number).
4. **New:** decide on `ict_long_meta_xgb_v1` and `ict_long_reversal_xgb_v1` — both already clear 8/8 promotion-quality gates. Promote to paper trading or document the specific blocker.
5. **New:** add an explicit discrimination gate (minimum ROC AUC and/or AP-lift-over-base-rate) to the promotion-quality contract, so a future retrain cannot clear the financial gates on a backtest while having near-zero real discriminative skill, as currently happens with the continuation branches.
6. **New:** populate and validate `abstain_policy.minimum_probability_quantile` for the low-AP-lift branches (`frvp_short_meta`, `frvp_long_meta`, `ict_long_continuation`, `ict_short_continuation`) as a low-cost filtering experiment before any retraining work on those branches.

# ICT Short-Side Concentration Audit

Date: `2026-07-28`

Compared the latest leakage-safe ICT ES refresh against the last pre-refresh contract:

- latest leakage-safe full refresh:
  - backtests: `model_testing/reports/ict_backtests/ict_es_primary_bootstrap_20260726_full/`
  - threshold policies: `model_testing/reports/ict_threshold_policies/ict_es_primary_bootstrap_20260726_full/`
  - regime slices: `model_testing/reports/ict_regime_slices/ict_es_primary_bootstrap_20260726_full/`
  - models: `models/ict_es_primary_xgb_bootstrap_20260726_full/`
- comparison baseline from the July 19, 2026 cadence audit:
  - backtests: `model_testing/reports/ict_backtests/ict_es_primary_20260719_min0/`
  - threshold policies: `model_testing/reports/ict_threshold_policies/ict_es_primary_20260719_min0/`

## Headline Read

- The short-side issue is no longer only `ict_short_meta_xgb_v1`.
- On the July 27, 2026 leakage-safe full refresh, both `ict_short_meta_xgb_v1` and `ict_short_reversal_xgb_v1` fail the promotion-quality gate on the same two items:
  - single-trade concentration above the `10%` cap
  - profitable-quarter breadth below the `60%` floor
- The continuation families still do not justify another generic retrain cycle yet:
  - `ict_short_continuation_xgb_v1` turned positive, but its PnL is still dominated by one or two trades
  - `ict_long_continuation_xgb_v1` is effectively flat and fails on raw economics plus robustness

## Current Short-Side Gate Read

### `ict_short_meta_xgb_v1`

- walk-forward summary:
  - `1082` selected test trades
  - `+3906.7` ticks net
  - expectancy `3.611`
  - Sharpe `0.905`
  - DSR `0.820`
  - WFE `1.442`
  - max drawdown `% = 7.13`
- gate misses:
  - largest single trade share `0.10017`
  - profitable-quarter share `16 / 27 = 0.5926`
- concentration profile:
  - top `5` trades contribute `32.66%` of total net PnL
  - top `10` trades contribute `55.33%`
  - best day is `2020-03-17` at `+437.7` ticks, or `11.20%` of total net

### `ict_short_reversal_xgb_v1`

- walk-forward summary:
  - `429` selected test trades
  - `+3052.15` ticks net
  - expectancy `7.115`
  - Sharpe `1.047`
  - DSR `0.914`
  - WFE `1.618`
  - max drawdown `% = 7.82`
- gate misses:
  - largest single trade share `0.11118`
  - profitable-quarter share `15 / 27 = 0.5556`
- concentration profile:
  - top `5` trades contribute `37.54%` of total net PnL
  - top `10` trades contribute `61.06%`
  - best day is `2020-03-17` at `+643.05` ticks, or `21.07%` of total net

## What Changed Versus July 19, 2026

- `ict_short_meta_xgb_v1`:
  - selected trades `275 -> 1082`
  - net PnL `+3191.25 -> +3906.7`
  - profitable-quarter share `20/27 -> 16/27`
  - the branch stayed profitable, but it became much broader and materially less time-stable
- `ict_short_reversal_xgb_v1`:
  - selected trades `791 -> 429`
  - net PnL `+5475.85 -> +3052.15`
  - profitable-quarter share `20/27 -> 15/27`
  - the branch became narrower and more dependent on a smaller winner set

Interpretation:

- The leakage-safe retrain did not simply "kill the short edge."
- It reallocated where the short edge appears:
  - the pooled short-meta lane became much more active
  - the family-specific short-reversal lane became much more selective
  - both ended up leaning on the same historical event subset too heavily

## Overlap Findings

- `ict_short_meta_xgb_v1` and `ict_short_reversal_xgb_v1` share `284` exact walk-forward trades when matched on `(entry_datetime, source_row_idx)`.
- Those shared trades have identical realized PnL in both branches and contribute `+2597.4` ticks to each model.
- Share of total net PnL from the shared subset:
  - `ict_short_meta_xgb_v1`: `66.5%`
  - `ict_short_reversal_xgb_v1`: `85.1%`
- Unique-only remainder after removing the shared subset:
  - `ict_short_meta_xgb_v1`: `798` trades, `+1309.3` ticks
  - `ict_short_reversal_xgb_v1`: `145` trades, `+454.75` ticks

Interpretation:

- The current short meta and short reversal branches are not behaving like independent family bets.
- The reversal branch is now mostly a concentrated subset of the pooled short-meta edge.
- That makes the concentration problem a roster-structure problem as much as a pure threshold problem.

## Short Meta Is Also Absorbing Continuation

- `ict_short_meta_xgb_v1` and `ict_short_continuation_xgb_v1` share `84` exact trades for `+564.4` ticks.
- `ict_short_continuation_xgb_v1` unique-only remainder is negative:
  - `81` trades
  - `-63.65` ticks
- `ict_short_reversal_xgb_v1` and `ict_short_continuation_xgb_v1` share `0` exact trades under the same event-key match.

Interpretation:

- The pooled short-meta model is acting like a catch-all short router that captures:
  - most of the profitable reversal subset
  - the profitable continuation subset
- The standalone continuation branch is not independently robust yet; it stays positive only because the shared subset with short meta is strong enough to offset the rest.

## Regime / Session Pocket Read

### `ict_short_meta_xgb_v1`

Main positive pockets:

- `strong_down_medium / asia`: `+752.45` ticks across `107` trades
- `strong_down_high / new_york`: `+626.55` across `73`
- `strong_up_high / new_york`: `+513.2` across `52`

Main negative pockets:

- `ranging_medium / asia`: `-222.3` ticks across `62` trades
- `ranging_high / new_york`: `-211.35` across `59`
- `strong_down_medium / off_hours`: `-143.15` across `91`

Read:

- The branch is profitable because a few directional shock pockets are strong.
- It still spends a lot of trade count in broad, low-quality ranging or off-hours pockets that weaken quarter breadth.

### `ict_short_reversal_xgb_v1`

Main positive pockets:

- `strong_down_high / off_hours`: `+583.2` ticks across `12` trades
- `strong_down_high / new_york`: `+456.35` across `21`
- `strong_down_medium / asia`: `+438.95` across `37`

Main negative pockets:

- `ranging_high / new_york`: `-368.85` ticks across `29` trades
- `strong_down_medium / off_hours`: `-257.9` across `26`
- `ranging_medium / off_hours`: `-201.4` across `16`

Read:

- The reversal branch is now even more pocket-dependent than the pooled meta branch.
- The best pockets are sparse but large; the worst pockets are broad enough to knock quarter share below the gate.

## Date-Level Concentration

### `ict_short_meta_xgb_v1`

Largest winners:

- `2022-02-25 21:30 UTC`: `+391.35` ticks, `10.02%` of total net
- `2026-02-27 21:05 UTC`: `+243.35`
- `2020-03-17 22:30 UTC`: `+222.35`
- `2020-03-17 22:00 UTC`: `+215.35`

Largest winning days:

- `2020-03-17`: `+437.7`
- `2022-02-25`: `+391.35`
- `2025-04-08`: `+341.7`

### `ict_short_reversal_xgb_v1`

Largest winners:

- `2025-06-12 23:45 UTC`: `+339.35` ticks, `11.12%` of total net
- `2020-03-17 22:30 UTC`: `+222.35`
- `2020-03-17 22:00 UTC`: `+215.35`
- `2020-03-17 20:00 UTC`: `+205.35`

Largest winning days:

- `2020-03-17`: `+643.05`
- `2025-06-12`: `+339.35`
- `2026-06-01`: `+187.4`

Interpretation:

- Both short branches still make a disproportionate amount of money on shock days and crisis-like continuation pockets.
- The same dates repeatedly show up across branches, which reinforces the overlap read above.

## Feature-Level Clues

Prepared-root and trained-model importance both point to a broader pooled short read rather than a clean family split:

- `artifacts/ict_es_primary_refresh_20260724_spacing_refit_final_confirm/phase04_prepared/prepared/short_ict_meta/report.txt` ranks `htf_confluence_short_ict_continuation` as one of the strongest short-meta features.
- `models/ict_es_primary_xgb_bootstrap_20260726_full/short_ict_meta/window_feature_importance.csv` keeps `htf_confluence_short_ict_continuation__lag_0` near the top of the final trained model.
- The July 12, 2026 baseline short-meta importance was more sweep / distance / structure-driven; the July 27, 2026 leakage-safe refresh adds more broad state variables and explicit continuation confluence.

Inference:

- The pooled short-meta lane is not just "reversal plus a little continuation."
- It currently looks like a general short-side state model that is swallowing profitable pieces of both family branches.

## Continuation Readout

### `ict_short_continuation_xgb_v1`

- improved versus July 19, 2026:
  - net PnL `-312.45 -> +500.75`
  - trades `73 -> 165`
- still not promotion-quality:
  - Sharpe `0.326`
  - WFE `0.858`
  - profitable-quarter share `7 / 16 = 0.4375`
  - top trade share `0.7815`
  - top `5` trades contribute `205.0%` of total net
- the best pocket `strong_down_high / new_york` contributes `+763.15` ticks, or `152.4%` of total net
- the worst pocket `strong_down_high / asia` gives back `-506.85` ticks, or `-101.2%` of total net

### `ict_long_continuation_xgb_v1`

- effectively flat:
  - `247` trades
  - `+7.45` ticks total
  - Sharpe `0.009`
  - WFE `0.059`
  - positive composite expectancy share `0.333`
- concentration is not the main story because raw net is near zero:
  - top trade share `16.6913`

Interpretation:

- Short continuation is still a concentration problem even after the refresh turned it positive.
- Long continuation is still an economics / robustness problem first.

## Highest-Value Next Steps

1. Run an explicit short-side regime-gated deployment / concentration study on the frozen July 27, 2026 leakage-safe roster, analogous to the FRVP concentration pass.
2. Start with abstain candidates that already show up as broad losers in the latest walk-forward:
   - `ict_short_meta_xgb_v1`: `ranging_medium / asia`, `ranging_high / new_york`, `strong_down_medium / off_hours`
   - `ict_short_reversal_xgb_v1`: `ranging_high / new_york`, `strong_down_medium / off_hours`, `ranging_medium / off_hours`
3. Treat pooled-vs-family disentanglement as the next train-side question only after the policy-layer concentration pass:
   - compare frozen short meta against a family-routed contract
   - test whether overlap-heavy trades should belong only to one active short branch
4. Keep both continuation families below promotion priority until the short-side concentration / overlap problem is clearer.

## Bottom Line

- The short-side failure is now well localized.
- The latest leakage-safe refresh did not expose a generic short-edge collapse.
- It exposed a short roster that is too dependent on the same event cluster showing up in multiple branches, with the pooled short-meta lane now absorbing profitable pieces of both reversal and continuation.
- The next best experiment is therefore not another generic continuation retrain. It is a frozen-model short-side regime-gated concentration study plus a pooled-vs-family overlap cleanup.

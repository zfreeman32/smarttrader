# FRVP Operational Contract

**Date:** 2026-07-21
**Purpose:** Record the exact FRVP operating rules that were finalized on Tuesday, July 21, 2026.

This note turns the recent FRVP decisions into one explicit contract for code, shadow ops, and review.

## Documentation Provenance

The operational decisions in this note are the contract-level output of the canonical experiment record in [FRVP_experiment_journal.md](FRVP_experiment_journal.md).

- The continuation baseline decision comes from `E02`.
- The full-span reversal control history comes from `E06`.
- The recent-regime reversal lane comes from `E07` and `E11`.
- The default cost-contract decision comes from `E10`.

Future FRVP contract changes should be documented in the journal first, using [FRVP_experiment_record_template.md](FRVP_experiment_record_template.md), before this note is revised.

## 1. Anchor And Range Selection

The FRVP anchor order is now explicit in code via `frvp/profiles/anchors.py`.

- `prior_rth` is the primary decision range.
  - Use its `POC`, `VAH`, `VAL`, `HVN`, and `LVN` as the default live trading reference levels.
  - `frvp_dist_vah_atr`, `frvp_dist_val_atr`, `frvp_in_va`, `frvp_above_vah`, `frvp_below_val`, and `frvp_profile_shape` are all interpreted off this anchor first.
- `overnight_eth` is the secondary inventory range.
  - Use it for overnight inventory context, RTH-vs-ETH overlap, and naked overnight POC tracking.
- `initial_balance` is the intraday confirmation range.
  - It becomes valid only after the first 60 RTH minutes are complete, at 10:30 ET.
- `swing_to_swing` is reactive context only.
  - It is not the default trigger anchor for `POC` / `VAH` / `VAL` decisions.
- `rolling_composite` is higher-timeframe context only.
  - It is a context range, not the live trigger range for Setup 1-6 decisions.

All anchors remain causal and single-contract:

- No anchor may mix contracts.
- Cross-roll naked-POC tracking stays reset-at-roll.
- No `POC` / `VAH` / `VAL` level is compared across contract coordinates without translation.

## 2. Real Displacement Vs. Noise

For FRVP operations, "real displacement" is no longer a qualitative phrase.

A bullish or bearish displacement is treated as real only if all of these are true:

1. The existing directional displacement flag is already on.
   - `displacement_bullish == 1` for bullish.
   - `displacement_bearish == 1` for bearish.
2. `volume_zscore_50 >= 1.25`.
3. The close finishes in the outer 25% of the bar on the move side.
   - Bullish: `(close - low) / (high - low) >= 0.75`
   - Bearish: `(high - close) / (high - low) >= 0.75`

For Setup 3 specifically, the move must also close at least `0.05 ATR` beyond the relevant value-area edge:

- Bullish breakout: `-frvp_dist_vah_atr >= 0.05`
- Bearish breakout: `-frvp_dist_val_atr >= 0.05`

If those conditions are not met, the move is treated as noise for FRVP setup selection.

## 3. Setup Tiebreakers

### Setup 1 vs. Setup 3

This tie is now resolved by measurable acceptance outside value.

- Choose `Setup 3` only if there is real displacement and the close is at least `0.05 ATR` beyond `VAH` or `VAL`.
- Otherwise keep the event in `Setup 1` territory if price is only touching or making a small reentry / overshoot around the edge.

In practice:

- A small poke outside value without real displacement is still fadeable.
- A real, high-participation close outside value is continuation, not a fade.

### Setup 2 vs. Setup 4

This tie is now resolved by measurable reentry depth.

- `Setup 2` owns the near-edge hold:
  - after a breakout, the retest may close back inside by at most `0.05 ATR`.
- `Setup 4` owns the failed auction:
  - price must reenter value by more than `0.05 ATR`
  - after `1-3` outside bars
  - with quiet reentry volume
  - and a same-side sweep inside the lookback window

That makes the two contracts disjoint:

- shallow hold near the broken edge -> `Setup 2`
- deeper return back into value -> `Setup 4`

## 4. Extended Shadow Window

The extended FRVP shadow window starts on **2026-07-21** from the `frvp_es_shadow_20260721` bundle.

Primary baseline:

- `frvp_long_continuation_xgb_v1`
- saved `v3` targeted-policy contract
- this remains the promotion-near FRVP branch

Observation focus:

- contract-switch behavior
- dashboard continuity
- signal persistence
- audit-replay consistency

Operational expectation:

- keep the continuation baseline frozen
- use the new July 21, 2026 shadow bundle as the default FRVP runtime package
- treat new shadow evidence as operational validation, not classifier-research pressure

## 5. Reversal Contract Decision

Decision on **2026-07-21**:

- accept `frvp_long_reversal_xgb_recent_regime_prune_v2` as the operational FRVP reversal contract
- keep the full-span recency `v3` branch as the research control, not the live contract

Important naming note:

- the accepted live contract is aliased in repo language as `frvp_long_reversal_xgb_recent_regime_prune_v2`
- the saved July 21, 2026 backtest / threshold artifact directory that produced it is still named `long_reversal_recent2y_sdh_overlap_prune_v1`

Saved evidence for the accepted contract from the July 21, 2026 backtest artifact:

- `63` trades
- `+3677.05` ticks
- Sharpe `1.480`
- DSR `1.179`
- `WFE = 2.162`
- profitable-quarter share `0.60`
- accepted paper-trading gate `True`

That is sufficient to treat the recent-regime lane as the live selective-deployment answer unless live shadow evidence later proves otherwise.

## 6. Conditional Train-Side Follow-Up

No new train-side branch is opened on **2026-07-21** because the reversal contract decision above is a "yes."

Only reopen full-span reversal train-side work if one of these becomes true:

- the accepted recent-regime live contract proves insufficient in shadow operation
- the repo explicitly decides it still needs a better full-span reversal answer

If that happens:

- keep the current full-span control frozen
- test only new train-side stability levers that can improve `WFE`
- do not reopen same-environment micro-pruning

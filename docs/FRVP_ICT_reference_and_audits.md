# FRVP / ICT / OTE trading logic and cross-family audits

The trader-facing trading-logic summary, followed by the FRVP/ICT concept-to-code audit, the 2026-08-11 live signal-output audit and the 2026-08-30 probability quantile filter study.

Contents:
- [FRVP, ICT, and OTE Trading Logic](#frvp_ict_ote_trading_logic)
- [FRVP and ICT/SMC Concept-to-Code Audit](#frvp_ict_concept_to_code_audit)
- [FRVP / ICT Live Signal Output Audit](#frvp_ict_live_signal_output_audit_20260811)
- [FRVP / ICT Probability Quantile Filter Study](#frvp_ict_probability_quantile_filter_study_20260830)


<a id="frvp_ict_ote_trading_logic"></a>

## FRVP, ICT, and OTE Trading Logic

*Merged 2026-10-09 from `docs/FRVP_ICT_OTE_TRADING_LOGIC.md`.*

This note is a trader-facing summary of how the repo thinks about markets.

It is intentionally focused on trading logic, not model training, data prep, or backtesting mechanics.

### 1. The Big Picture

The workflow is built around a simple idea:

- Do not trade random bars.
- Wait for price to reach an area that matters.
- Read whether price is rejecting that area or accepting it.
- Trade either the reversal, the continuation pullback, or the breakout.

In plain English, the stack is trying to answer:

1. Where is price relative to meaningful structure?
2. Is the market balanced, trending, or running stops?
3. Is this a place to fade, follow, or stand aside?

### 2. What "OTE Workflow" Means Here

In this repo, "OTE workflow" is best understood as the decision process around a trade, not just one narrow pattern.

The important executable trade types used in that workflow are:

| Trade type | Plain-English meaning | Typical job |
| --- | --- | --- |
| Reversal | Price overextends into a key level and rejects | Fade the move back into balance or back toward the opposite side |
| Continuation pullback | Price shows strength, pulls back into a good area, then resumes | Join the trend after the pullback |
| Breakout | Price leaves a balance area with commitment | Go with the expansion if acceptance is real |

`Confirmation / confluence` is still important, but it is a filter layer rather than a fourth trade type. It helps decide whether a reversal, continuation, or breakout idea is worth taking.

So the workflow is not "buy because a model said so."

It is:

1. Find context.
2. Find a setup.
3. Check whether price behavior supports that setup.
4. Define the invalidation point.
5. Aim for the next logical target.

### 3. The Core OTE Reading Sequence

This is the simplest trader version of the workflow:

1. Start with context.
   Look at session, higher-timeframe direction, and obvious reference levels.
2. Mark the important prices.
   Prior highs/lows, session levels, value-area edges, fair value gaps, order blocks, and obvious swing points.
3. Decide what type of day it looks like.
   Balanced day, trend day, stop-run day, breakout day, or choppy/noisy day.
4. Wait for one of the three trade types.
   Reversal, continuation pullback, or breakout.
5. Enter near structure, not in the middle.
   Good entries usually happen on a retest, reclaim, pullback, or clean break with follow-through.
6. Define the "wrong" point first.
   If price gets back through the level that made the trade idea valid, the trade is wrong.
7. Target the next magnet.
   That might be POC, VAH/VAL, the next HVN, an opposing liquidity pool, or the next obvious swing level.

### 4. FRVP Logic in Plain English

FRVP stands for Fixed Range Volume Profile.

It is a way of answering:

- Where did the market actually spend the most business?
- Where did it reject price quickly?
- Where is "fair value" versus "too far"?

#### FRVP concepts that matter

| Term | Plain-English meaning |
| --- | --- |
| POC | The price where the most business was done |
| VAH / VAL | The upper and lower edges of the main value area |
| HVN | A heavy-volume area; price often gets attracted to it or slows there |
| LVN | A thin-volume area; price often moves through it quickly |
| Balanced profile | The market found agreement and rotated around fair value |
| Trend profile | The market accepted higher or lower prices and did not stay balanced |

#### What FRVP is trying to trade

FRVP is mostly about auction behavior:

- If price is still inside value, fade the edges back toward fair value.
- If price cleanly leaves value and accepts higher or lower prices, go with the move.
- If price pokes outside value and fails, fade it back inside.

#### Main FRVP setups used here

1. Mean reversion inside a balanced profile.
   If price reaches VAH or VAL and there is no real acceptance outside, the idea is a move back toward POC.
2. Breakout retest after acceptance.
   If price breaks out of value, then comes back and holds the old edge, the idea is continuation.
3. Value-area breakout with force.
   If price leaves value with strong participation, treat it as initiative activity and follow the move.
4. Failed auction.
   If price looks above or below value, cannot hold there, and falls back inside, that is a reversal setup.
5. LVN traverse.
   Thin zones often act like air pockets. If price enters one with momentum, it can travel quickly through it.
6. HVN magnet.
   If price is between accepted volume nodes, the nearest node can act like a pull target.

#### The trader's FRVP mindset

Ask:

- Are we rotating inside value or accepting new value?
- Is price being rejected at the edge, or is it building above/below it?
- Is the next move likely back to POC, or onward to the next node?

### 5. ICT Logic in Plain English

ICT logic is built around liquidity, structure, and displacement.

It is trying to answer:

- Where are stops likely sitting?
- Did price run those stops?
- Did price reclaim the move, or did it actually accept the new level?
- Is there a clean imbalance or order block to enter from?

#### ICT concepts that matter

| Term | Plain-English meaning |
| --- | --- |
| Liquidity sweep | Price runs a prior high or low to trigger stops |
| Displacement | A strong push away from a level; shows urgency and intent |
| FVG | A fast-move imbalance area that price may revisit |
| Order block | The last meaningful opposing candle area before displacement |
| MSS / CHoCH | A structure shift that hints the move may be changing direction |
| Premium / discount | Expensive area for shorts, cheap area for longs, relative to the current swing |
| Session manipulation | A false move around the open that sets up the real move |

#### What ICT is trying to trade

ICT logic is less about "fair value" in the auction sense and more about this sequence:

1. Price attacks a pool of stops.
2. The market reveals whether that raid was a trap or a true breakout.
3. The trade comes from the reclaim, the structure shift, or the pullback into imbalance.

#### Main ICT setups used here

1. Sweep reclaim.
   Price runs a high or low, then closes back the other way. That suggests the raid failed.
2. Sweep, displacement, then FVG retrace.
   Stops get run, price launches away, then pulls back into the imbalance for entry.
3. Order block retest after structure shift.
   Structure changes, then price returns to the origin area before continuing.
4. IFVG reversal / classic breaker research branch.
   A failed gap zone flips and becomes a reversal area.
   The classic failed-order-block breaker now has an audited causal state surface, but it is still not part of current labels until an explicit Phase 3 setup/label branch is implemented and reviewed.
5. Premium/discount continuation pullback.
   The move is already directional; the entry comes from a better-priced retrace, not from chasing the impulse.
6. Session-open manipulation reversal.
   The open runs an obvious level, traps traders, then reverses.
7. Displacement continuation after a raid.
   Price raids one side, then drives hard in the opposite direction and keeps going.

#### The trader's ICT mindset

Ask:

- Which side's stops just got taken?
- Did price reclaim the level or keep accepting beyond it?
- Do we have real displacement, or just noise?
- Is the pullback returning into a useful entry zone?

### 6. How FRVP and ICT Work Together

They are different lenses on the same market.

FRVP tells you:

- where business was accepted
- where price is balanced
- where price is likely to rotate or get pulled

ICT tells you:

- where liquidity sits
- whether stops were raided
- whether a move is a trap, reversal, or continuation

When both line up, the trade is usually cleaner.

Examples:

- ICT sweep reversal at the same place FRVP shows rejection from value edge
- ICT continuation pullback into an FVG while FRVP shows acceptance outside value
- Session-open stop raid that also rejects a known HVN or value-area extreme

### 7. What Matters Most to a Trader

If you strip all the tooling away, the important parts are:

- Context first
- Entry near structure
- Clear invalidation
- Trade type clarity
- Patience when the market is in the middle of nowhere

The workflow is strongest when it can clearly say one of these:

- "Price rejected a key area, so fade it."
- "Price accepted a new area, so follow it."
- "Price is pulling back into a good spot inside an existing move, so join it."

If it cannot say one of those clearly, the best trade is often no trade.

### 8. Short Summary

- OTE is the overall trade-selection workflow.
- FRVP is the volume-profile lens: value, rejection, acceptance, node-to-node movement.
- ICT is the liquidity-and-structure lens: sweeps, displacement, gaps, order blocks, and session traps.
- The important executable trade types are reversal, continuation pullback, and breakout; confirmation through confluence is a filter, not a fourth trade type.
- The goal is not to predict every bar. The goal is to wait for a meaningful area, read the reaction, and act only when the story is clean.


<a id="frvp_ict_concept_to_code_audit"></a>

## FRVP and ICT/SMC Concept-to-Code Audit

*Merged 2026-10-09 from `docs/FRVP_ICT_CONCEPT_TO_CODE_AUDIT.md`.*

Scope: canonical FRVP setup generation under `frvp/`, canonical ICT/SMC setup generation under `ict/`, shared primitives used by those generators, and current/recent ES artifacts under `artifacts/`.

This audit focuses on whether the implemented setup events correspond to the intended trading concepts, whether primitive features are valid, whether the signal is causal at the bar where it fires, and whether artifact behavior matches the code.

### Executive Verdict

FRVP is mostly causal at bar-close signal time, but its core auction primitive is materially wrong by standard volume-profile convention: value area is built from globally ranked high-volume bins and then spanned from min to max, rather than by contiguous expansion from POC. On actual ES RTH sessions, this changes VAH/VAL in 89.7% of profiles and usually widens value. Because every FRVP setup depends on VAH/VAL, POC, HVN, or LVN, this is the highest-priority concept-to-code defect.

ICT/SMC primitives are mostly causal when interpreted as closed-bar signals, but several core concepts are simplified enough to change setup meaning: equal-high/low pools can disappear before sweep detection, premium/discount uses the latest confirmed high/low pair rather than a coherent dealing range, order blocks only use the immediate opposite candle before a displacement break, and compound sequence setups are not linked by event identity. Current prepared ICT model inputs also include non-causal raw swing flags and pre-08:30 access to the 08:30 open.

The empirical artifacts support the code audit:

| Family | Diagnostic | Result |
| --- | ---: | --- |
| FRVP | ES RTH sessions compared for ranked VA vs contiguous POC-expansion VA | 2,438 sessions |
| FRVP | VA edges different | 2,187 sessions, 89.7% |
| FRVP | Ranked VA wider than contiguous VA | 2,104 sessions, 86.3% |
| FRVP | Median absolute edge difference | 5.64 ES points |
| FRVP | `frvp_open_drive_flag` true before first 30 minutes complete | 18 first-30 RTH bars in current artifact |
| FRVP | Current artifact Setup 4 exclusion | 1002/1002 events excluded as disabled |
| FRVP | Refresh artifact Setup 4 exclusion | 210/1002 events excluded, no disabled flag |
| ICT | Base artifact `sweep_displacement_fvg` events | 1 event |
| ICT | Base artifact `ob_retest_after_mss` events | 7 events |
| ICT | Base artifact session-open manipulation events | 10 events |
| ICT | Intermediate ICT sweep/FVG/confluence flags | Many, including 17,936 displacement-after-sweep rows and 20,277 FVG-retrace-after-displacement rows |
| ICT | Prepared ICT feature lists include `ict_raw_swing_*` | All six prepared target folders |
| ICT | Prepared ICT feature lists include `ict_open_0830` or distance | All six prepared target folders |
| ICT | Pre-08:30 rows with populated `ict_open_0830` | 249,245/249,245 pre-08:30 rows |

### FRVP Shared Primitive Audit

#### Volume profile construction

Implementation:

- Histogram bars are distributed uniformly across overlapped price bins in `VolumeProfileBuilder._build_histogram` at `frvp/profiles/builder.py:151`.
- Bin width is `max(tick_size, ATR / atr_bin_divisor)` in `_resolve_bin_width` at `frvp/profiles/builder.py:142`.
- POC is `histogram.idxmax()` in `build` at `frvp/profiles/builder.py:59`.
- Value area is selected by sorting all bins descending by volume, accumulating to 70%, then returning the min/max price of selected bins in `_extract_value_area` at `frvp/profiles/builder.py:193`.
- HVN/LVN levels are global mean/std cutoffs over positive-volume bins in `_extract_volume_nodes` at `frvp/profiles/builder.py:206`.
- P/b/D shape is a skew/concentration heuristic in `_classify_shape` at `frvp/profiles/builder.py:247`.

Concept mapping:

- Uniform OHLCV allocation is a reasonable approximation when tick-level volume-at-price is unavailable.
- Value area does not match standard market-profile / volume-profile VA expansion. It should expand contiguously outward from POC by repeatedly including the higher-volume adjacent price row(s). Current ranked selection can include separated high-volume islands and then declare all prices between them to be value.
- Bin width is not rounded to the ES tick grid. Actual diagnostics showed POC/VA levels such as `5044.112500`, which are not tradable ES ticks. Distances can still be numeric, but the levels are not market-native.
- HVN/LVN primitives are not local nodes. They are "above/below global distribution cutoff" bins, so Setup 5 and Setup 6 do not truly know whether a level is a local low-volume void or local high-volume magnet.
- Shape classification is a useful heuristic, but it is not a robust D/P/b/double-distribution profile classifier.

Causality:

- Completed prior RTH, overnight, IB, and swing-to-swing windows are causal in general. Prior RTH/overnight are resolved with strict completed-session indexing in `frvp/feature_sets/frvp_context.py:84` and `frvp/profiles/anchors.py:211`.
- IB is only exposed once `current_ts >= ib_end` in `frvp/feature_sets/frvp_context.py:378`.
- Swing-to-swing uses confirmed swings with prefix indexing in `frvp/feature_sets/frvp_context.py:837`.
- Boundary note: strict `side="left"` indexing can delay just-completed anchors by one bar exactly at session end/open boundaries.

Required fixes:

- Replace ranked value area with contiguous POC expansion.
- Make profile bin widths tick-grid aligned: `bin_width = tick_size * max(1, round(raw_width / tick_size))`.
- Keep zero-volume bins inside the profile price range if LVN/single-print logic needs them.
- Replace HVN/LVN global std cutoffs with local-extrema/grouped-node logic, with node width and prominence thresholds.
- Add fixture tests with hand-built histograms for POC, VAH/VAL, HVN/LVN, and profile shape.

#### FRVP session-open and setup feature timing

Implementation:

- `_build_session_open_context` precomputes first-30 and first-60 minute RTH session context at `frvp/feature_sets/frvp_context.py:1065`.
- The final `open_drive_flag` is assigned to all rows for that RTH session at `frvp/feature_sets/frvp_context.py:376`.
- Setups read `frvp_open_drive_flag` in `frvp/setups/detector.py:298`, `:396`, `:411`, `:427`, `:470`, `:477`, and `:493`.

Concept mapping:

- An open-drive flag based on the completed first 30 minutes is not known during the first 30 minutes.
- This affects Setup 1, 3, and 5 confidence, and blocks Setup 4 and 6.

Causality:

- Non-causal before the first-30 window has completed.
- Current artifact impact is small but real: 18 first-30 RTH bars have `frvp_open_drive_flag=True`, and 1 generated setup row in the first 30 minutes carried `open_drive_flag=True`.
- The false value is also future-informed before 10:00 ET, but in the current artifact no Setup 4 rows and no Setup 6 first-30 rows depended on that false value.

Required fixes:

- Expose a developing open-drive flag separately from a finalized first-30 open-drive flag.
- Set finalized `frvp_open_drive_flag` to unknown/0 until the first-30 window is complete, or block any setup rule that depends on it before completion.
- Add prefix-recompute tests: compute features on full data and on data truncated at each bar; pre-10:00 rows must match.

#### FRVP final selection

Implementation:

- Candidate priority is present, but `_select_candidate` sorts by confidence first and priority second in `frvp/setups/detector.py:528`.

Concept mapping:

- If setup taxonomy matters, priority should usually decide family first, with confidence breaking ties inside the family. ICT does this correctly.

Required fixes:

- Decide whether FRVP setup type is intended to be "highest-confidence event" or "highest-priority event." If taxonomy matters, sort by priority first.

### FRVP Setup Audit

#### Setup 1: VAH/VAL Rejection From Balanced Value

Intended concept:

- Prior session is balanced/D-shaped.
- Current session opens inside or near prior value.
- Price tags VAH/VAL and rejects back toward value/POC without acceptance outside value.
- Low/normal volume favors rotational mean reversion; initiative displacement outside value invalidates.

Current implementation:

- Candidate function: `frvp/setups/detector.py:287`.
- Requires D-shape and open inside value.
- Short triggers if `frvp_dist_vah_atr` is between roughly -0.30 and +0.65 ATR.
- Long triggers if `frvp_dist_val_atr` is between roughly -0.30 and +0.65 ATR.
- Rejects only if a real same-direction outside displacement is present.
- `frvp_open_drive_flag` and quiet volume add confidence but do not define the setup.

Primitive audit:

- Depends on VAH/VAL and D-shape, both weakened by the current value-area and shape algorithms.
- There is no explicit tag/touch-and-reject candle requirement.
- There is no explicit failure-to-accept test such as N closes outside value, quick return inside, or rejection wick.
- POC/other side of value is not emitted as a target primitive.

Causality/leakage:

- Causal at bar close if the signal is assumed to fire after the current bar closes.
- Open-drive confidence can be non-causal during the first 30 RTH minutes.

Empirical diagnostics:

- Current artifact: 3,487 Setup 1 events, roughly balanced by side.
- Exclusion rate: 37.7% in current artifact, mostly first RTH/macro/HTF gates.
- Because VAH/VAL are materially distorted in 89.7% of sampled RTH profiles, Setup 1 edge location is not concept-faithful.

Verdict:

- Conceptually incomplete and dependent on a flawed VA primitive.

Fixes/tests:

- Require actual VA edge interaction: high/low touches edge within tolerance, close returns inside value, and no multi-bar acceptance outside.
- Use contiguous tick-aligned VA.
- Add synthetic tests for: clean edge rejection, no-touch near-edge non-signal, accepted break outside value, and open-drive pre-10:00 behavior.

#### Setup 2: Value-Area Breakout Acceptance and Retest

Intended concept:

- Price breaks out of prior value with acceptance.
- A later retest of VAH/VAL holds from the outside.
- Continuation is expected toward next auction reference such as HVN, poor high/low, overnight extreme, or measured move.

Current implementation:

- Candidate function: `frvp/setups/detector.py:329`.
- Retest state is created only when Setup 3 fires, via `last_breakout_above`/`last_breakout_below` in `detect_frvp_setups`.
- Retest must occur within `SETUP2_LOOKBACK_BARS = 15`.
- Long hold: close remains at or above `VAH - 0.05 ATR`.
- Short hold: close remains at or below `VAL + 0.05 ATR`.

Primitive audit:

- The breakout memory is not a general "breakout and acceptance" state. It is "a prior Setup 3 initiative break fired."
- Acceptance is not measured directly. There is no time-above-value count, volume building outside value, or rejection of return into value.
- Retest can close slightly back inside value and still pass.

Causality/leakage:

- Causal. The breakout memory is previous-row state, and candidate evaluation happens before current-row state update.
- Same VA primitive weakness applies.

Empirical diagnostics:

- Current artifact: 3,279 Setup 2 events.
- Exclusion rate: 23.5%.
- Event count is not starved, but it represents the code's Setup3-derived retest concept, not a full auction acceptance/retest concept.

Verdict:

- Conceptually incomplete. It is a narrow continuation-after-Setup3 rule, not a full value-acceptance retest.

Fixes/tests:

- Track breakout state independently from Setup 3.
- Require acceptance metrics before retest: at least N closes outside value, volume outside value, or time outside without immediate reclaim.
- Require retest wick/touch and close hold from the outside.
- Add tests where price breaks without Setup 3 but accepts, and where Setup 3 fires but no acceptance follows.

#### Setup 3: Initiative Breakout From Value

Intended concept:

- Initiative participant breaks out of value with displacement and above-normal volume.
- Close is outside VAH/VAL with enough excursion to reject rotational behavior.

Current implementation:

- Candidate function: `frvp/setups/detector.py:380`.
- Requires `volume_zscore_50 >= 1.25`.
- Requires `frvp_above_vah` or `frvp_below_val`.
- Requires `_has_real_displacement`.
- Minimum outside overshoot is `SETUP3_MIN_OVERSHOOT_ATR = 0.05`.

Primitive audit:

- This is the closest FRVP setup to its intended primitive sequence.
- `volume_zscore_50` is causal but includes the current bar in its rolling mean/std baseline, so spike magnitude is somewhat damped relative to a strictly prior baseline.
- Overshoot threshold is very small; the displacement check carries most of the burden.
- VAH/VAL semantics remain flawed.

Causality/leakage:

- Causal at bar close.
- Open-drive confidence bonus can be non-causal before first-30 completion.

Empirical diagnostics:

- Current artifact: 5,408 Setup 3 events.
- Exclusion rate: 28.2%.

Verdict:

- Partially validated after the VA fix. Current implementation is conceptually recognizable, but it inherits invalid VA and uses loose overshoot.

Fixes/tests:

- Use prior-only volume z-score for all setup logic.
- Raise or instrument overshoot threshold by market regime.
- Add tests for outside close without displacement, displacement without volume, and valid initiative break.

#### Setup 4: Failed Auction Back Into Value

Intended concept:

- Price probes beyond VAH/VAL and fails.
- Reclaims value quickly, ideally after a liquidity sweep and on poor continuation volume.
- Trade is opposite the failed auction, back into value.

Current implementation:

- Candidate function: `frvp/setups/detector.py:417`.
- Tracks prior outside-value run length, max 3 bars.
- Requires current bar back inside value.
- Blocks open-drive sessions.
- Requires current `volume_zscore_50 <= 0.75`.
- Requires recent same-side generic sweep: `bars_since_sweep_high` or `bars_since_sweep_low`.
- Requires reentry depth greater than 0.05 ATR.

Primitive audit:

- The outside-run state is directionally sensible and causal.
- Sweep primitive is generic price-structure sweep, not specifically a VAH/VAL auction failure.
- Low-volume condition is on the reentry bar, not on the failed outside auction attempt.
- No explicit "no acceptance outside value" measure beyond run length.

Causality/leakage:

- Causal except for `frvp_open_drive_flag` before first-30 completion.

Empirical diagnostics:

- Current artifact: 1,002 generated events, but 100% excluded because failed auction labels were disabled in that artifact.
- Refresh artifact: same 1,002 events, 20.96% excluded, and no disabled flag.
- Source default in `frvp/pipelines/es_primary_phase04.py:103` now enables failed-auction labels.

Verdict:

- Conceptually plausible but incomplete. Artifact behavior depends on which pipeline run is used.

Fixes/tests:

- Make the sweep test VA-edge specific: pierced VAH/VAL, reclaimed value, and no accepted outside close sequence.
- Measure failure volume at the outside probe, not only on reentry.
- Add an artifact metadata check so reports cannot silently mix disabled and enabled Setup 4 regimes.

#### Setup 5: LVN Rejection or Fast Traverse

Intended concept:

- LVN can act as a rejection zone or an air pocket.
- A valid setup must distinguish rejection/stall at the LVN from fast traverse through it.
- Direction depends on the observed interaction, not only displacement direction.

Current implementation:

- Candidate function: `frvp/setups/detector.py:460`.
- Requires nearest LVN within 0.30 ATR.
- Bullish displacement creates a long.
- Bearish displacement creates a short.
- Open-drive adds confidence.

Primitive audit:

- LVN is not a local low-volume void; it is a global low-volume cutoff bin.
- Code only implements "displacement near LVN continues in displacement direction."
- It does not classify rejection, absorption, stall, or fast traverse speed across an LVN.

Causality/leakage:

- Causal at bar close except open-drive confidence before first-30 completion.

Empirical diagnostics:

- Current artifact: 9,091 Setup 5 events, the largest FRVP setup bucket.
- Exclusion rate: 19.9%.
- High count is consistent with a broad "displacement near any global LVN" rule.

Verdict:

- Semantically overloaded. It is not a faithful LVN rejection/traverse detector.

Fixes/tests:

- Split into `lvn_fast_traverse` and `lvn_rejection`.
- Define fast traverse as entry into LVN zone, large body/range, close beyond zone, and low dwell time.
- Define rejection as tag of LVN zone, failure to close through, wick rejection, and reversal close.
- Replace LVN primitive first.

#### Setup 6: HVN Magnet / Rotation Between Accepted Nodes

Intended concept:

- In balanced conditions, price rotates toward accepted high-volume nodes.
- Direction should be toward a meaningful HVN/POC magnet with no strong initiative flow against it.

Current implementation:

- Candidate function: `frvp/setups/detector.py:482`.
- Requires D-shape, open inside value, current price inside value, no open-drive, no displacement, and quiet volume.
- Requires both HVN above and below.
- Chooses direction toward nearest HVN if distance is between 0.75 and 2.5 ATR and far/near asymmetry is sufficient.

Primitive audit:

- HVNs are global cutoff bins, not local accepted nodes.
- POC is not explicitly prioritized over a lesser HVN.
- No check for intervening LVN, session VWAP, developing value, or order-flow conflict.
- It is a magnet heuristic, not a robust rotational auction model.

Causality/leakage:

- Causal after first-30 completion. Before then, `frvp_open_drive_flag` is not finalized.

Empirical diagnostics:

- Current artifact: 6,400 events.
- Exclusion rate: 59.7%, partly due HTF/reversal gates and cooldown.

Verdict:

- Conceptually incomplete and dependent on weak HVN/D-shape primitives.

Fixes/tests:

- Use grouped local HVN nodes, with POC as the primary magnet.
- Require no intervening LVN or require target path quality.
- Add tests where nearest global-high bin is not a local HVN, where POC should dominate, and where open-drive blocks only after it is known.

### ICT/SMC Shared Primitive Audit

#### Swings and equal highs/lows

Implementation:

- Raw swing highs/lows use symmetric future windows, then confirmed swings are delayed by `window` bars in `ict/structure/swings.py`.
- Raw swing flags are still emitted as `ict_raw_swing_high` and `ict_raw_swing_low` at `ict/structure/swings.py:82`.
- Equal-high/low pools are created from confirmed swing pairs, but active pools are removed before current-row stats are emitted if the current high/low violates the zone in `ict/structure/swings.py:134`.

Concept mapping:

- Confirmed swings are causal.
- Raw swing flags are non-causal at the pivot row.
- Equal-pool lifecycle can hide exactly the sweep that should be detected when price pierces beyond the equal-pool zone.

Causality/leakage:

- Setup generation mainly uses confirmed swings and latest confirmed levels.
- Prepared model features include raw swing flags in all six ICT prepared targets.

Required fixes:

- Drop `ict_raw_swing_high` and `ict_raw_swing_low` from all model recipes/prepared features.
- Emit equal-pool state from the previous bar into sweep detection, then invalidate after sweep/reclaim processing.
- Add prefix-recompute tests for all swing-derived columns.

#### Liquidity reference levels

Implementation:

- Prior RTH high/low are shifted completed sessions in `ict/structure/liquidity.py`.
- Overnight high/low are developing during overnight and final during RTH.
- IB high/low are exposed only when complete.
- `ict_open_0830` is mapped to all rows on the local date at `ict/structure/liquidity.py:153`.

Concept mapping:

- Prior RTH, overnight, and IB are mostly causal.
- The 08:30 open is not known before 08:30, but it is present on all pre-08:30 rows of the same date.

Causality/leakage:

- Base artifact: 249,245/249,245 pre-08:30 rows have populated `ict_open_0830`.
- Prepared ICT feature lists include `ict_open_0830` and/or `ict_open_0830_dist_atr` in all six targets.
- This is model-input leakage. It is not directly in final setup selection unless those context features feed a rule, but it contaminates downstream ML.

Required fixes:

- Null `ict_open_0830` and distance until local time is at or after 08:30.
- Add a forbidden-feature list for known future/context columns.

#### Sweeps

Implementation:

- Sweep levels include swing highs/lows, equal highs/lows, prior RTH, overnight, IB, and prior week in `ict/detectors/sweeps.py:16`.
- A pierce is detected at bar `i`, then a reclaim can fire at bar `j` up to `sweep_close_back_bars` later in `ict/detectors/sweeps.py`.
- Output columns keep one aggregate sweep code/value per bar.

Concept mapping:

- Reclaim-based sweep is causal if the event is timestamped at the reclaim close.
- Multiple levels can be swept on the same bar, but the scalar output can only preserve one level code/value.
- Equal-pool removal can create false negatives for equal-high/low sweeps.

Required fixes:

- Add deterministic level priority for same-bar multi-level sweeps.
- Preserve multi-level sweep records or expose primary and secondary levels.
- Consume previous-row equal-pool state before invalidating it.

#### Displacement

Implementation:

- Displacement requires range ATR, body-to-range, close location, and volume z-score in `ict/detectors/displacement.py`.
- Volume gate passes when volume is missing: `(volume_ok | volume.isna())` at `ict/detectors/displacement.py:66`.

Concept mapping:

- ES should generally require real volume participation. Missing-volume bypass is useful for FX/CFD but should be instrument-specific.
- Rolling ICT volume z-score is prior-only through `rolling_group_zscore`, so it is causal.

Required fixes:

- For ES, fail displacement if volume is missing unless explicitly configured.
- Keep missing-volume bypass only for instruments where volume is not reliable.

#### FVG / IFVG

Implementation:

- Three-candle FVG detection follows the common definition: bullish `low[t] > high[t-2] + min_gap`, bearish mirror.
- IFVG flips an active zone when a close fully crosses the opposite side in `ict/detectors/fvg.py:142`.
- IFVG state remains active and can flip direction again rather than being invalidated after re-inversion.

Concept mapping:

- FVG detection is faithful enough.
- IFVG lifecycle needs stricter state handling: original FVG, inverted FVG, invalidated/re-inverted should be distinct.

Required fixes:

- Add IFVG states and invalidation rules.
- Prevent repeated polarity flips from being treated as fresh IFVG quality.

#### Order blocks and breakers

Implementation:

- Order block creation is tied to displacement plus structure break in `ict/detectors/order_blocks.py`.
- Bullish OB uses the immediately previous bearish candle; bearish OB uses the immediately previous bullish candle.
- Breaker logic requires invalidation plus displacement/shift and recent sweep.

Concept mapping:

- This is a strict subset of OB theory.
- It misses candle clusters, last down/up candle searches farther back, body/wick refinements, and OB quality filters.

Required fixes:

- Search back through the displacement impulse for the last opposite candle/cluster.
- Add quality score: displacement strength, break level, imbalance/FVG overlap, mitigation depth, and age.

#### Market structure and dealing range

Implementation:

- BOS/CHoCH use latest confirmed swing levels and wick breaks in `ict/structure/market_structure.py`.
- Premium/discount range uses the latest confirmed swing high and latest confirmed swing low, regardless of whether they are a coherent impulse leg, in `ict/detectors/premium_discount.py:49`.

Concept mapping:

- Wick-only structure breaks can classify liquidity probes as BOS.
- Latest-high/latest-low pairing is not a robust dealing range.
- OTE continuation logic inherits this weak dealing range.

Required fixes:

- Use close-through or configurable wick/close structure modes.
- Build dealing ranges from a confirmed impulse leg: BOS/CHoCH anchor low-to-high for bullish, high-to-low for bearish, with invalidation.

### ICT/SMC Setup Audit

#### Sweep Reclaim

Intended concept:

- Price raids liquidity above/below a meaningful level and reclaims back through it.
- Optional confluence from displacement, FVG/OB, premium/discount, or structure improves quality.

Current implementation:

- Candidate function: `ict/setups/detector.py:354`.
- Fires on current `ict_sweep_direction`.
- Uses reference level/code from sweep primitive.
- Confidence adds confluence, but a sweep alone can fire.

Primitive audit:

- Core concept is recognizable.
- Meaningful-level quality depends on sweep primitive and scalar level output.
- Equal-pool sweeps can be missed.

Causality/leakage:

- Causal at reclaim close.

Empirical diagnostics:

- Base artifact: 7,175 events, the largest ICT setup.
- Refresh artifact: 6,360 events.
- Outcomes are poor in raw artifact terms, but that is a label/model question rather than a concept-code failure.

Verdict:

- Partially validated. Primitive cleanup needed.

Fixes/tests:

- Add same-bar multi-level priority tests.
- Add equal-high/low sweep tests.
- Add tests that pierce without reclaim does not fire until a reclaim bar.

#### Session Open Manipulation, Pre-IB and Post-IB

Intended concept:

- Around RTH open/initial balance, price raids overnight/prior-day/IB liquidity and reclaims, implying reversal from engineered liquidity.

Current implementation:

- Candidate function: `ict/setups/detector.py:292`.
- Pre-IB allowed when `ict_is_rth` and `ict_ib_complete` is false.
- Post-IB allowed after IB is complete, with IB high/low as the required level.
- Displacement, VWAP, and premium/discount are confidence additions, not hard requirements.

Primitive audit:

- Pre-IB level set is reasonable.
- Post-IB has no upper time bound beyond RTH and spacing, so an IB sweep at 13:40 ET can still be classified as "session open manipulation."

Causality/leakage:

- Mostly causal. IB levels are available only after completion.

Empirical diagnostics:

- Base artifact: 7 pre-IB and 3 post-IB events.
- Refresh artifact: 6 pre-IB and 2 post-IB events.
- Base post-IB examples include 13:40 ET, which is too late for the session-open concept.

Verdict:

- Conceptually incomplete and empirically starved.

Fixes/tests:

- Limit post-IB to a defined opening window, for example 10:30-11:30 ET, or rename it as generic IB sweep reclaim.
- Decide which confluences are mandatory.
- Add fixture tests for pre-IB ONH/ONL raids, post-IB late raids, and IB incomplete behavior.

#### Sweep + Displacement + FVG

Intended concept:

- Liquidity raid, reclaim, displacement away from the raid, creation of an FVG by that displacement, then retracement/CE interaction and continuation.

Current implementation:

- Candidate function: `ict/setups/detector.py:402`.
- Requires recent opposite sweep, recent displacement, supportive nearest FVG, not IFVG, created by displacement, and CE touch/hold.
- Uses nearest supportive zone, not event-linked sweep/displacement/FVG IDs.

Primitive audit:

- Conditions are directionally sensible but not identity-linked.
- A valid FVG can be ignored if nearest supportive zone selection chooses a different zone kind.
- The FVG need not be the one created by the displacement after the sweep.

Causality/leakage:

- Causal at bar close.

Empirical diagnostics:

- Base artifact: 1 event.
- Refresh artifact: 1 event.
- Intermediate flags are not scarce: 17,936 displacement-after-sweep rows and 20,277 FVG-retrace-after-displacement rows in the base feature table.
- Therefore the final setup collapse is caused by linkage/selection rules, not lack of primitive events.

Verdict:

- Not currently viable as a canonical setup. High false-negative risk and weak event linkage.

Fixes/tests:

- Track event IDs: sweep_id -> displacement_id -> fvg_id -> retest_id.
- Require the FVG formed after the sweep and by the same displacement leg.
- Test one clean valid chain, one mismatched-FVG chain, and one nearest-zone false-negative case.

#### OB Retest After MSS

Intended concept:

- Liquidity raid, market structure shift, displacement, order block creation, then retest/mitigation of the OB for continuation/reversal.

Current implementation:

- Candidate function: `ict/setups/detector.py:460`.
- Requires recent opposite sweep, recent CHoCH/MSS, OB retest event, and compatible structure state.
- FVG overlap is a confidence bonus, not a hard condition.

Primitive audit:

- OB primitive only uses the immediately previous opposite candle.
- No explicit linkage from the OB to the MSS displacement leg.
- OB cluster and quality logic are missing.

Causality/leakage:

- Causal at bar close.

Empirical diagnostics:

- Base artifact: 7 events.
- Refresh artifact: 6 events.
- This is too sparse for a major SMC setup family.

Verdict:

- Conceptually recognizable but under-specified and empirically starved.

Fixes/tests:

- Link OB formation to the same sweep/MSS/displacement sequence.
- Expand OB detection to prior opposite candle/cluster inside the impulse.
- Add tests for valid OB retest, retest before OB formation, and OB from wrong impulse.

#### IFVG Reversal

Intended concept:

- FVG fails, becomes inverted, then retests/uses the inversion zone as support/resistance with structure confirmation.

Current implementation:

- Candidate function: `ict/setups/detector.py:504`.
- Requires nearest IFVG, retest after inversion index, CE interaction, close in reversal direction, and structure support.

Primitive audit:

- This is one of the more faithful ICT setup mappings.
- Main weakness is IFVG lifecycle: zones can repeatedly flip rather than invalidate after re-inversion.
- Nearest-zone selection can still mask better contextual zones.

Causality/leakage:

- Causal. Code requires `position > inversion_index`, so same-bar inversion/retest is avoided.

Empirical diagnostics:

- Base artifact: 6,487 events.
- Refresh artifact: 5,957 events.

Verdict:

- Partially validated. Needs IFVG state cleanup and better zone ranking.

Fixes/tests:

- Add IFVG invalidation/re-inversion states.
- Add tests for first inversion, same-bar non-retest, valid later retest, and invalidated second failure.

#### Premium/Discount Continuation

Intended concept:

- In an established impulse/trend, price retraces into discount for longs or premium for shorts, preferably OTE, then continues with institutional reference confluence.

Current implementation:

- Candidate function: `ict/setups/detector.py:554`.
- Requires continuation trend support, PD/OTE alignment, no recent opposite structure flip, and supportive zone within 0.75 ATR.
- Trend support can come from structure state or HTF EMA alignment.

Primitive audit:

- Dealing range is latest high/low pair, not a confirmed impulse leg.
- EMA alignment can substitute for a valid SMC impulse structure.
- Supportive zone need not be tightly linked to the impulse pullback sequence.

Causality/leakage:

- Causal, but depends on weak dealing range semantics.

Empirical diagnostics:

- Base artifact: 1,515 events.
- Refresh artifact: 2,883 events.
- This is the only compound continuation setup with meaningful count, but it is the one with the weakest dealing-range foundation.

Verdict:

- Conceptually weak until dealing range is rebuilt.

Fixes/tests:

- Build dealing range from BOS/CHoCH-confirmed impulse anchors.
- Require continuation after retracement into the range, not just EMA trend support.
- Add tests where latest high/low pair is not a valid impulse range.

#### Displacement Continuation After Raid

Intended concept:

- Liquidity raid, displacement in the true direction, shallow pullback or imbalance/OB interaction, then continuation.

Current implementation:

- Candidate function: `ict/setups/detector.py:604`.
- Requires trend support, recent opposite sweep, recent same-direction displacement, supportive zone, and close beyond latest displacement origin.

Primitive audit:

- No explicit shallow-pullback or mitigation/retest requirement.
- Supportive zone can be near price without being causally tied to the raid displacement.
- Latest displacement origin is a loose proxy for continuation confirmation.

Causality/leakage:

- Causal at bar close.

Empirical diagnostics:

- Base artifact: 59 events.
- Refresh artifact: 107 events.

Verdict:

- Conceptually incomplete and underfiring.

Fixes/tests:

- Require pullback into the FVG/OB/OTE zone after displacement.
- Link sweep, displacement, and supportive zone by IDs/indices.
- Add tests for no-pullback displacement, wrong-zone retest, and valid raid-displacement-pullback continuation.

### Setup Integrity Matrix

| Setup | Concept Fidelity | Primitive Validity | Causality | Artifact Health | Overall Integrity |
| --- | --- | --- | --- | --- | --- |
| FRVP 1 VA rejection | Medium-low | Low until VA fixed | Mostly causal | Adequate count | Incomplete |
| FRVP 2 breakout retest | Medium-low | Low until VA fixed | Causal | Adequate count | Incomplete |
| FRVP 3 initiative break | Medium-high | Medium after VA fix | Mostly causal | Adequate count | Repairable |
| FRVP 4 failed auction | Medium | Medium-low | Mostly causal | Artifact mismatch | Incomplete |
| FRVP 5 LVN traverse/reject | Low | Low | Mostly causal | Over-broad count | Weak |
| FRVP 6 HVN magnet | Low-medium | Low | Mostly causal after open window | High exclusion | Weak |
| ICT sweep reclaim | Medium-high | Medium | Causal | High count | Repairable |
| ICT session open manipulation | Medium-low | Medium | Causal | Starved, late post-IB | Incomplete |
| ICT sweep-displacement-FVG | Low-medium | Medium | Causal | Essentially absent | Weak |
| ICT OB retest after MSS | Medium-low | Low-medium | Causal | Starved | Weak |
| ICT IFVG reversal | Medium-high | Medium | Causal | High count | Repairable |
| ICT PD continuation | Low-medium | Low | Causal | Adequate count | Weak |
| ICT displacement continuation after raid | Low-medium | Medium-low | Causal | Starved | Weak |

### Highest-Priority Remediation Plan

1. Fix FRVP value area and bin-grid semantics first.
   - This is upstream of all FRVP setup logic.
   - Add hand-histogram tests and rerun event counts by setup.

2. Add a universal prefix-causality regression harness.
   - For every feature row, recompute on data truncated at that row and compare selected columns to full-run output.
   - Run this against FRVP open context, ICT swings, ICT reference levels, FVG/OB, sweeps, and setup outputs.

3. Remove or delay known non-causal ICT model inputs.
   - Drop `ict_raw_swing_high`, `ict_raw_swing_low`.
   - Null `ict_open_0830` and `ict_open_0830_dist_atr` before 08:30 local time.
   - Audit `ict_open_drive_flag` similarly if it is based on completed opening windows.

4. Repair ICT equal-pool sweep lifecycle.
   - Sweep detector should see previous active equal pools before invalidation.
   - Preserve multiple same-bar swept levels or define deterministic priority.

5. Rebuild FRVP HVN/LVN as local grouped nodes.
   - Split Setup 5 into rejection and traverse variants.
   - Rework Setup 6 around POC/HVN target quality.

6. Link ICT compound setups by event identity.
   - Use explicit sweep, displacement, FVG, OB, retest, and MSS IDs/indices.
   - Avoid nearest-zone substitutions for sequence setups.

7. Rebuild ICT dealing range/OTE from confirmed impulse anchors.
   - Use BOS/CHoCH-confirmed swing pairs, not latest high/latest low.
   - Add invalidation and stale-range rules.

8. Add setup-specific golden fixtures.
   - Each setup needs positive, near-miss, stale-state, wrong-side, and non-causal-prefix fixtures.

### Bottom Line

FRVP should not be treated as concept-faithful until value-area construction is replaced. The code is cleanly organized and mostly causal, but the central auction primitive is not the intended auction primitive.

ICT/SMC has a better causal foundation in several places, especially confirmed swings, FVGs, and IFVG retests, but the most sophisticated SMC sequences are either under-linked or empirically starved. The immediate model-leakage issue is also real: raw swing flags and pre-08:30 open features are present in prepared model inputs and should be removed or delayed before relying on downstream performance metrics.


<a id="frvp_ict_live_signal_output_audit_20260811"></a>

## FRVP / ICT Live Signal Output Audit

*Merged 2026-10-09 from `docs/FRVP_ICT_Live_Signal_Output_Audit_20260811.md`.*

Date: `2026-08-11`

This audit answers a specific operational question: live shadow predictions for the FRVP and ICT ES families cross their decision threshold far more often than the OTE EUR/USD champions do. Is that a real signal-quality problem, and if so, where does it come from? The read draws on the live audit database (`ote_live/runtime_data/live_market_data.sqlite3`), the live runtime manifests, saved training summaries, and the existing FRVP/ICT backtest and concentration reports. It is meant to sit alongside `FRVP_ES_primary_audit_report.md` and `ICT_SHORT_SIDE_CONCENTRATION_AUDIT_20260728.md` as a signal-layer companion piece, not a replacement for either.

### 0. Scope And Safety Context

All 12 FRVP/ICT models discussed here run in **shadow mode on ES futures** — a separate instrument from the EUR/USD OTE production pipeline. Shadow mode means every prediction is logged with a `shadow` decision and a `candidate_passed_threshold` / `probability_below_threshold` tag, but nothing is ever emitted as a live trade. None of the findings below describe live capital risk today. They describe signal-quality and promotion-readiness issues that should be resolved before any of these models leave shadow mode.

### 1. Headline Finding: Threshold-Crossing Rate By Family

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

### 2. Root Cause: Weak Discrimination Meeting High Base Rates

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

### 3. Pooled-Model Dilution: The Same Disease In Both Codebases

This is the most actionable finding, and it independently shows up in the repo's own prior audits on both sides:

- **ICT** (`ICT_SHORT_SIDE_CONCENTRATION_AUDIT_20260728.md`): `ict_short_meta_xgb_v1` shares `284` exact walk-forward trades with `ict_short_reversal_xgb_v1`. Those shared trades account for `66.5%` of the meta model's net PnL and `85.1%` of the reversal model's net PnL. The pooled short-meta model also shares `84` trades with `ict_short_continuation_xgb_v1`, whose unique-only remainder is negative (`-63.65` ticks over `81` trades). The pooled model is acting as a catch-all that absorbs the profitable reversal and continuation subsets rather than behaving like an independent bet.
- **FRVP** (`frvp_deferred_research_audit_20260728/DEFERRED_RESEARCH_SUMMARY.md`): the long branches are dominated by a single Phase-3 setup type (Setup 1 or Setup 5 carrying `64% - 72%` of trade share), with wide expectancy variance across setup types pooled into the same branch (e.g. best-setup expectancy `108.43` vs. worst-setup expectancy `18.79` within the same reversal branch; best `52.51` vs. worst `-0.81` within the same continuation branch).

**Same mechanism in both families: pooling heterogeneous setup types into one meta-label averages away the good setups' edge and lets the bad setups' noise leak into the aggregate probability.** This is consistent with the low AP-lift numbers in Section 2 for the meta/continuation branches specifically, and it is already the top-priority open item in `notes.txt` for FRVP (per-setup-family split) — this audit adds the ICT-side quantitative confirmation of the same pattern.

### 4. Concentration And Crisis-Day Dependency (ICT Short Side)

Both `ict_short_meta_xgb_v1` and `ict_short_reversal_xgb_v1` fail the promotion-quality gate on the same two criteria:

- single largest trade above the `10%` cap (`10.02%` and `11.12%` respectively)
- profitable-quarter share below the `60%` floor (`59.3%` and `55.6%`)

Top-5 trades contribute `33% - 38%` of total net PnL; top-10 contribute `55% - 61%`. `2020-03-17` (the COVID crash) shows up as a top-3 winning day for both branches. A meaningful chunk of the backtest edge is shock-day dependent rather than a steady repeatable edge — already documented in the ICT concentration audit, restated here because it directly explains part of why the meta/reversal probabilities look confident in backtest despite weak standalone AUC: a handful of very large, very correlated events are doing a disproportionate share of the work.

### 5. FRVP Short Family Is Currently Non-Viable, Not Just Under-Threshold

From `frvp_short_family_leaders_refresh_20260715_accountdd/model_summary.csv`:

| Model | Sharpe | Deflated Sharpe | Max account DD | Profit factor |
|---|---|---|---|---|
| `frvp_short_continuation_tcn_v1` | -0.38 | -0.37 | **109.2%** | 0.93 |
| `frvp_short_meta_xgb_v1` | 0.07 | 0.07 | 54.6% | 1.02 |
| `frvp_short_reversal_xgb_v1` | -0.39 | -0.39 | 48.7% | 0.88 |

A `109%` max drawdown means this branch would blow the reference account outright. All three fail the drawdown gate. This is not a threshold-tuning problem — the continuation and reversal branches do not currently have tradable edge on the short side, and no amount of threshold/filtering work fixes that. Only `frvp_short_meta_xgb_v1` is even marginally alive (Sharpe `~0`), consistent with its status as "cleanest short sentinel" in the existing FRVP experiment record.

### 6. Two ICT Models Already Clear All 8 Promotion-Quality Gates

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

### 7. Filtering Tips

1. **Don't trust raw `calibrated_probability >= threshold` as a signal-quality proxy for meta/continuation branches.** For `frvp_short_meta`, `frvp_long_meta`, `ict_long_continuation`, and `ict_short_continuation` specifically, the probability is barely more informative than the base rate (AP lift under `0.07`). If a live-usable confidence signal is needed from these branches, use **probability rank/quantile within a rolling window** instead of the raw calibrated value — a `0.52` in a distribution that swings `0.36-0.54` is not the same thing as a `0.52` in a distribution that swings `0.0001-0.03`.
2. **Weight conviction by AP-lift-over-base-rate per branch, not by family label.** Trust FRVP short-reversal (`0.28` lift), FRVP long-reversal (`0.24`), and ICT long/short reversal and meta (`0.13`) well ahead of `frvp_short_meta`, `frvp_long_meta`, `ict_long_continuation`, or `ict_short_continuation` (all under `0.07` lift).
3. **Check trade-level overlap before treating "meta" and "family" signals firing together as independent confirmation.** With `85%` overlap between ICT short-meta and short-reversal, two signals firing together on the short side is very likely the same underlying event counted twice, not two independent confirmations stacking conviction.
4. **Session/regime-gate the short-side ICT models manually if paper-traded ahead of a policy fix.** Documented net-loser pockets: for `ict_short_meta_xgb_v1` — `ranging_medium/asia`, `ranging_high/new_york`, `strong_down_medium/off_hours`; for `ict_short_reversal_xgb_v1` — `ranging_high/new_york`, `strong_down_medium/off_hours`, `ranging_medium/off_hours`.
5. **Treat single-day/single-trade-dominated backtests with suspicion.** If a candidate's trade log shows `2020-03-17` or another crisis day in the top-3 PnL contributors, discount the headline Sharpe/DSR until performance excluding that date has been checked.
6. **There is already an unused lever for this in the live policy contract.** `abstain_policy.minimum_probability_quantile` in the FRVP/ICT `live_policy.json` files is currently `null` on every model checked. This is exactly the quantile-based filter described in tip 1, already wired into the runtime abstain path — it just needs a value and a validation pass instead of new code.

### 8. Summary Of Correction Work

Ranked by leverage, cross-referenced against what is already tracked in `notes.txt` / the experiment journals so this list does not duplicate open work:

1. Split pooled meta models into per-setup-family models (FRVP: already top-priority in `notes.txt`; ICT: this audit's Section 3 is the quantitative confirmation the same disease exists on the ICT side).
2. Run the short-side regime-gated concentration/overlap cleanup (already parked in `notes.txt`; this audit reinforces it with the AP-lift and shock-day evidence).
3. Stop investing further threshold/filtering work in the FRVP short family until the underlying setup logic changes — the continuation branch's `109%` drawdown is an economics problem, not a filtering problem (already reflected in `notes.txt`; this audit adds the concrete drawdown number).
4. **New:** decide on `ict_long_meta_xgb_v1` and `ict_long_reversal_xgb_v1` — both already clear 8/8 promotion-quality gates. Promote to paper trading or document the specific blocker.
5. **New:** add an explicit discrimination gate (minimum ROC AUC and/or AP-lift-over-base-rate) to the promotion-quality contract, so a future retrain cannot clear the financial gates on a backtest while having near-zero real discriminative skill, as currently happens with the continuation branches.
6. **New:** populate and validate `abstain_policy.minimum_probability_quantile` for the low-AP-lift branches (`frvp_short_meta`, `frvp_long_meta`, `ict_long_continuation`, `ict_short_continuation`) as a low-cost filtering experiment before any retraining work on those branches.


<a id="frvp_ict_probability_quantile_filter_study_20260830"></a>

## FRVP / ICT Probability Quantile Filter Study

*Merged 2026-10-09 from `docs/FRVP_ICT_probability_quantile_filter_study_20260830.md`.*

Date: 2026-08-30

### Plain-English Plan

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

### What Was Run

Two scripts were added:

- `scripts/run_probability_quantile_sweep_20260830.py`
- `scripts/analyze_probability_quantile_trade_screen_20260830.py`

Output root:

`model_testing\reports\probability_quantile_sweeps\frvp_ict_weak_discrimination_20260830`

The full walk-forward sweep was too slow to finish for all four models in one run. It completed partial reruns for `frvp_short_meta_xgb_v1` through `q30`. To finish the study efficiently, I also ran a faster trade-level screen across all four models using the saved historical emitted trades.

Important caveat: the trade-level screen answers, "what if we dropped the lower-confidence trades that already fired?" It is a useful first pass, but not final deployment proof. Before changing live policy, the winning setting should get a full candidate-level walk-forward rerun.

### Results

| Model | Best Screened Filter | Baseline | Best Screen | Plain-English Read |
| --- | --- | ---: | ---: | --- |
| `ict_long_continuation_xgb_v1` | `q50` | `247` trades, `+7.45` ticks | `119` trades, `+366.65` ticks | Strongest improvement. This model should talk less. |
| `ict_short_continuation_xgb_v1` | `q60` | `165` trades, `+500.75` ticks | `66` trades, `+530.10` ticks | Fewer trades, slightly more PnL, much better expectancy. Worth validating. |
| `frvp_short_meta_xgb_v1` | `q40` to `q60` looked better in the fast screen | `1523` trades, `+1829.05` ticks | q40: `910` trades, `+4498.50` ticks | Interesting, but mixed. The partial full walk-forward rerun only reached q30 and did not confirm a clean win before q40. |
| `frvp_long_meta_xgb_v1` | none | `3621` trades, `-4468.65` ticks | Still negative at every tested value | Do not spend more time here unless the model/target changes. |

### Partial Full Walk-Forward Check

For `frvp_short_meta_xgb_v1`, the heavier walk-forward rerun produced:

| Filter | Trades | Net ticks | Sharpe | Max DD | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | `1713` | `+1994.55` | `0.067` | `55.97%` | Weak |
| q10 | `1215` | `+20.25` | `0.001` | `55.58%` | Worse |
| q20 | `1042` | `-215.30` | `-0.010` | `53.37%` | Worse |
| q30 | `887` | `+2915.45` | `0.131` | `44.77%` | Better, but still weak |

This says `frvp_short_meta` may need a stricter value like q40/q50, but we should not trust the fast-screen q40/q60 result until a full rerun finishes.

### Full Candidate-Level ICT Validation

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

### ICT Long Continuation q40 Concentration Read

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

### ICT Long Continuation q40 Concentration-Control Pass

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

### Recommendations

1. Keep `ict_long_continuation_xgb_v1` q40 as the only meaningful winner from this filter study.
2. Do not promote q40 yet; keep it research-only or log-only because concentration still fails after the control pass.
3. Stop simple pocket-pruning for q40 for now. The best concentration-control variant improved largest-trade share from `24.90%` to `13.59%`, but it still missed the `10%` gate.
4. Stop quantile tuning for `ict_short_continuation_xgb_v1` for now. q50/q60 did not produce a strong enough branch.
5. Put `frvp_short_meta_xgb_v1` behind ICT long continuation; rerun only q40/q50/q60 if we still want to pursue this lever after the ICT q40 control pass.
6. Drop `frvp_long_meta_xgb_v1` from this quantile-filter experiment. The filter did not rescue it.

Layman summary: this filter is useful when a model has occasional good ideas buried inside too much noise. It appears most useful for ICT continuation. It does not magically fix a model that is broadly wrong, which is what the FRVP long-meta result looks like here.

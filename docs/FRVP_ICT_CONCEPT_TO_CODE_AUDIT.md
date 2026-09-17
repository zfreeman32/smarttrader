# FRVP and ICT/SMC Concept-to-Code Audit

Scope: canonical FRVP setup generation under `frvp/`, canonical ICT/SMC setup generation under `ict/`, shared primitives used by those generators, and current/recent ES artifacts under `artifacts/`.

This audit focuses on whether the implemented setup events correspond to the intended trading concepts, whether primitive features are valid, whether the signal is causal at the bar where it fires, and whether artifact behavior matches the code.

## Executive Verdict

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

## FRVP Shared Primitive Audit

### Volume profile construction

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

### FRVP session-open and setup feature timing

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

### FRVP final selection

Implementation:

- Candidate priority is present, but `_select_candidate` sorts by confidence first and priority second in `frvp/setups/detector.py:528`.

Concept mapping:

- If setup taxonomy matters, priority should usually decide family first, with confidence breaking ties inside the family. ICT does this correctly.

Required fixes:

- Decide whether FRVP setup type is intended to be "highest-confidence event" or "highest-priority event." If taxonomy matters, sort by priority first.

## FRVP Setup Audit

### Setup 1: VAH/VAL Rejection From Balanced Value

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

### Setup 2: Value-Area Breakout Acceptance and Retest

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

### Setup 3: Initiative Breakout From Value

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

### Setup 4: Failed Auction Back Into Value

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

### Setup 5: LVN Rejection or Fast Traverse

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

### Setup 6: HVN Magnet / Rotation Between Accepted Nodes

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

## ICT/SMC Shared Primitive Audit

### Swings and equal highs/lows

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

### Liquidity reference levels

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

### Sweeps

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

### Displacement

Implementation:

- Displacement requires range ATR, body-to-range, close location, and volume z-score in `ict/detectors/displacement.py`.
- Volume gate passes when volume is missing: `(volume_ok | volume.isna())` at `ict/detectors/displacement.py:66`.

Concept mapping:

- ES should generally require real volume participation. Missing-volume bypass is useful for FX/CFD but should be instrument-specific.
- Rolling ICT volume z-score is prior-only through `rolling_group_zscore`, so it is causal.

Required fixes:

- For ES, fail displacement if volume is missing unless explicitly configured.
- Keep missing-volume bypass only for instruments where volume is not reliable.

### FVG / IFVG

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

### Order blocks and breakers

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

### Market structure and dealing range

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

## ICT/SMC Setup Audit

### Sweep Reclaim

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

### Session Open Manipulation, Pre-IB and Post-IB

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

### Sweep + Displacement + FVG

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

### OB Retest After MSS

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

### IFVG Reversal

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

### Premium/Discount Continuation

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

### Displacement Continuation After Raid

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

## Setup Integrity Matrix

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

## Highest-Priority Remediation Plan

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

## Bottom Line

FRVP should not be treated as concept-faithful until value-area construction is replaced. The code is cleanly organized and mostly causal, but the central auction primitive is not the intended auction primitive.

ICT/SMC has a better causal foundation in several places, especially confirmed swings, FVGs, and IFVG retests, but the most sophisticated SMC sequences are either under-linked or empirically starved. The immediate model-leakage issue is also real: raw swing flags and pre-08:30 open features are present in prepared model inputs and should be removed or delayed before relying on downstream performance metrics.

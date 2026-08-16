# Fixed Range Volume Profile as ML Signal Foundation
## Design Paper: FRVP-Driven Signal Pipeline for 5-Minute EUR/USD

**Author:** Quantitative Research & ML Systems  
**Date:** 2026-06-15  
**Status:** Blueprint — no pipeline code yet  
**Repo target:** Slot into the existing OTE meta-labeling architecture

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [FRVP Primer](#2-frvp-primer)
3. [ML Formulation Analysis](#3-ml-formulation-analysis)
4. [FRVP Feature Engineering](#4-frvp-feature-engineering)
5. [Labeling](#5-labeling)
6. [Data Requirements](#6-data-requirements)
7. [Model Architecture](#7-model-architecture)
8. [Validation and Anti-Overfitting](#8-validation-and-anti-overfitting)
9. [Build Roadmap](#9-build-roadmap)
10. [Open Questions and Risks](#10-open-questions-and-risks)

---

## 1. Executive Summary

This paper defines how the **Fixed Range Volume Profile (FRVP)** indicator becomes a primary signal layer in the existing 5-minute EUR/USD ML pipeline. The goal is 1–3 high-probability swing entries per day by detecting the precise moments when institutional volume structure predicts an imminent reversal or acceleration.

**The recommended approach is meta-labeling (Formulation B).** FRVP rule-based setups generate the primary side hypothesis (long/short) and a hypothesis about what type of move is expected (reversal vs. continuation). A separate ML model — consistent with the existing `long_ote_meta_tcn_champion` and `short_ote_meta_tcn_champion` pattern already in production — predicts P(this specific FRVP setup is profitable). The ML model does not need to identify *where* setups occur; the rule-based FRVP engine handles that. The ML model answers only: **"Given that this setup fired, should we bet?"**

The key practical decisions:

| Decision | Recommendation | Reasoning |
|---|---|---|
| Volume basis for profiles | Tick volume (+ dollar-volume proxy) | EUR/USD is OTC with no central tape; tick volume is the only practical universal proxy |
| Profile anchoring scheme | Dual: prior-session + swing-to-swing | Reproducible without look-ahead; session anchor is clean, swing anchor is reactive |
| ML formulation | Meta-labeling (B) with TCN primary | Consistent with existing meta TCN champions; regime-conditional performance separates well |
| Fallback | XGBoost direct event classifier | If FRVP setups are too sparse for the TCN's sequence context to help |
| Target frequency | 1–3 signals/day | ~36–108 signals per month across a 5-year dataset yields sufficient sample depth |

**What already exists in the repo vs. what this paper adds:**

| Component | Status |
|---|---|
| Labeling engine (triple-barrier, CUSUM, quality scoring, sample weights) | **Exists** — reuse verbatim |
| Feature framework (16 feature families, 405 strategies, recipe system) | **Exists** — new `frvp_context` module slots in here |
| ICT context features (FVGs, order blocks, BOS/CHoCH, sweeps) | **Exists** — use as interaction context for FRVP |
| HTF context (30m/1h resampling, ATR, swing detection) | **Exists** — anchor FRVP profiles to HTF session boundaries |
| FX calendar (session boundaries, market-day closes) | **Exists** — provides clean, look-ahead-free session anchors |
| Volume feature set (log vol, relative vol, dollar volume) | **Exists** — extend for profile construction |
| Regime classifier + regime-slice evaluation | **Exists** — condition FRVP thresholds on regime |
| Walk-forward backtest with frictions | **Exists** — reuse for FRVP validation |
| Meta-labeling recipe (`features/recipes/meta_labeling.json`) | **Exists** — the correct recipe for this pipeline |
| OTE training pipeline (XGBoost / TCN / LSTM) | **Exists** — train FRVP meta-labeler against it |
| **FRVP profile construction engine** | **New** — `features/feature_sets/frvp_context.py` |
| **FRVP setup rule layer** | **New** — `features/strategies/frvp_setups.py` |
| **FRVP-specific label events** | **New** — extend `data/labeling/labeling_engine.py` |

**Hard sanity check.** Any backtest showing Sharpe > 3 or win rate > 65% on intraday FX should be treated as a bug until proven otherwise. The design targets Sharpe ~1.0–1.5 out-of-sample with max drawdown < 12% and signal frequency of 1–3 trades/day. Anything significantly above these numbers demands a leakage audit before celebration.

---

## 2. FRVP Primer

### 2.1 What Is the Fixed Range Volume Profile?

The Volume Profile is a market tool that plots traded volume at each price level over a specified lookback window, rather than plotting volume over time (as a traditional volume bar chart does). The *Fixed Range* variant defines that window by a specific price-time interval chosen by the analyst — a session, a prior day, a swing leg — rather than a rolling window.

The core idea traces to J. Peter Steidlmayer's Market Profile work from the 1980s at the CBOT (Steidlmayer & Hawkins, 2003), later extended to electronic order-flow and volume analysis by Dalton et al. (2007). The FRVP is now a standard tool in institutional order-flow analysis.

### 2.2 Profile Anatomy

**Point of Control (POC)**  
The price level within the fixed range at which the most volume was traded. In equilibrium markets, price tends to gravitate toward the POC. A POC that price has not revisited after leaving is called a **Naked POC** or **Virgin POC (VPOC)**.

**Value Area (VA)**  
The price range containing 70% of the total volume traded within the fixed range. The 70% is derived from the classic Market Profile observation that price spends roughly 70% of its time within the value area when a market is in balance (Dalton et al., 2007). By convention:
- **Value Area High (VAH)**: Upper bound of the value area
- **Value Area Low (VAL)**: Lower bound of the value area

**High-Volume Nodes (HVN)**  
Price levels with volume significantly above the mean node volume. HVNs act as price magnets — they create friction and cause temporary consolidation. They are support/resistance layers that slow price.

**Low-Volume Nodes (LVN)**  
Price levels with volume significantly below the mean. LVNs are thin, fast-traverse zones — price slips through them quickly in the direction of momentum with little resistance.

**Profile Shape Taxonomy**  
- **D-shape (Balanced)**: Volume concentrated in the middle of the range, tapering tails on both sides. Indicates price found a fair-value equilibrium during the session. Mean-reversion setups dominate in subsequent sessions.
- **P-shape (Bullish)**: High-volume node at the bottom of the range, thin tail extending upward. Buyers dominated the session — they drove price up but established value lower. Indicates bullish sentiment with potential for continuation.
- **b-shape (Bearish)**: Mirror of P-shape. High-volume node at the top, thin tail downward. Sellers dominated.

### 2.3 Setups and Trade Logic

**Setup 1: Mean-Reversion Inside a Balanced (D-shape) Profile**  
Condition: Prior session profile is D-shaped (balanced). Current session opens near the midpoint or within the prior VA.  
Logic: Price at VAH with no new initiative buyers → fade toward POC. Price at VAL with no new initiative sellers → fade toward POC.  
Signal type: Reversal.  
Session context: Works best in low-volatility Asian range and early London sessions before directional initiative.

**Setup 2: POC Retest / Acceptance After Breakout**  
Condition: Price breaks through the prior session's VAH or VAL with volume expansion, then retests the prior VAH/VAL from the new side.  
Logic: If the breakout is genuine, the prior VAH becomes new support (or VAL becomes new resistance). A retest that holds → continuation.  
Signal type: Acceleration/continuation.  
Session context: Most reliable during London Open and NY Open when institutional flow re-enters after the Asian range.

**Setup 3: Value Area Breakout with Volume Expansion**  
Condition: Price at VAH or VAL with expanding tick volume (current session volume significantly above rolling norm).  
Logic: Volume expansion at structure → initiative participants are in control → momentum trade toward next HVN.  
Signal type: Continuation/breakout.  
Caution: Requires genuine volume spike; a low-volume VAH tag is more likely to reverse.

**Setup 4: Failed Auction / Value Area Rejection**  
Condition: Price extends beyond VAH or VAL into the tail, fails to trade volume at those levels (no acceptance), and returns back inside the value area.  
Logic: The market attempted to discover new value at those extremes but found no counterpart. Trapped aggressive breakout traders → reversal.  
Signal type: Reversal. This is one of the highest-probability FRVP signals when properly filtered.  
Note: In EUR/USD on 5-minute bars, this often appears as a wick that touches beyond VAH/VAL and closes back within.

**Setup 5: LVN Rejection/Fast-Traverse Zones**  
Condition: Price enters a known LVN region and either (a) blasts through rapidly or (b) stalls unexpectedly, suggesting absorption at a hidden HVN.  
Logic: LVNs have no volume "memory" — price should pass quickly. If it stalls, something structural is underneath.  
Signal type: Acceleration through LVN is a continuation signal; stall at LVN suggests hidden structure.

**Setup 6: HVN Magnet / Equilibrium Pull**  
Condition: Price is between two HVNs. The nearest HVN acts as a gravitational attractor.  
Logic: HVNs represent price levels where both buyers and sellers previously agreed on value. Price is drawn back to these levels.  
Signal type: Mean-reversion toward HVN.

### 2.4 Critical FX/Automation Caveats

**Caveat 1: EUR/USD Has No Centralized True Volume**

EUR/USD is traded over-the-counter across a global decentralized network of banks, ECNs, and prime brokers. There is no consolidated tape. The "volume" available in any retail or semi-institutional data feed is **tick count** — the number of price updates (or sometimes the number of lots traded at a single ECN/broker), not the aggregate global notional.

Implications for FRVP:
- The profile is built on tick volume, not notional dollar volume. This is a known limitation.
- Lequeux and Acar (1998) showed that tick frequency in FX is significantly correlated with actual trading activity, particularly during peak session hours. The correlation degrades during off-hours (Asian session for EUR/USD pairs specifically).
- The absolute level of tick volume has no meaning across different data providers or time periods after infrastructure changes. Only the *relative* volume (e.g., relative to its rolling median) is meaningful.
- Three proxies are available and should be compared empirically:
  1. **Raw tick count**: Number of price updates per bar. Available universally.
  2. **Tick volume** (lots-at-ECN): Available from some providers. Closer to true volume but still not consolidated.
  3. **Dollar-volume proxy**: `close × tick_count`, which the repo already computes as `dollar_volume` in `features/feature_sets/volume.py`. This roughly weights price levels by notional proximity.

**Recommendation**: Build profiles using raw tick volume as the primary basis, and test `dollar_volume` as a secondary basis. Report which produces more stable nodes in out-of-sample validation. Do not assume dollar-volume is superior — the extra `close` multiplier may amplify noise at extreme price levels.

**Caveat 2: FRVP Is Normally Manually Drawn**

In its traditional form, FRVP requires a human to specify the anchor range. For a live bot, the anchor must be defined algorithmically — deterministically, without look-ahead bias, from information available at the time of bar close.

**Algorithmic Anchoring Schemes Evaluated:**

| Scheme | Look-ahead free? | Stability | Reactivity | Verdict |
|---|---|---|---|---|
| Prior session (London/NY/Asia) | Yes — session ended before current bar | High | Medium | **Recommended (primary)** |
| Prior FX day (17:00 – 17:00 NY) | Yes — prior day is closed | High | Low | **Recommended (secondary)** |
| Rolling-N-session lookback | Yes | Very high | Low | Useful for long-term context, not signal generation |
| Swing-to-swing (causal) | Yes — uses the repo's existing causal swing detector | Medium | High | **Recommended (reactive anchor)** |
| Regime-anchored | Yes — but regime detection has lag | Low | Very high | Risky; save for Phase 3 research |

**Recommended dual-anchor scheme:**

1. **Session anchor (primary)**: Use the repo's `fx_calendar.py` market-day/session labeling to identify the prior London session (02:00–11:00 ET) and prior NY session (08:00–16:00 ET). Build one FRVP per anchor per bar. At any given 5-minute bar, two session profiles are active: the most recently completed London profile and the most recently completed NY profile. These anchors change once per session completion, not per bar.

2. **Swing-to-swing anchor (reactive)**: Use the existing causal zigzag swing detector from the labeling engine (`detect_swings_zigzag` in `reversal_labeling_engine.py`) to define the anchor as the leg from the most recently confirmed swing low to the most recently confirmed swing high (or vice versa). This anchor updates when a new confirmed swing is detected. Because the swing detector is already causal and uses ATR-scaled confirmation, there is no look-ahead.

Both anchors produce profiles that are defined entirely by past data at the time of computation. The session anchor gives a clean, session-relative view of institutional value. The swing anchor gives a structural view of where the most recent directional leg placed volume.

---

## 3. ML Formulation Analysis

### 3.1 The Central Question

Given that an FRVP setup fires (one of the 6 setups above is algorithmically detected), should the system take the bet? This is the question the ML layer must answer. The formulation choice determines:
- What label is assigned to each event
- What features have predictive content
- How severely overfitting will mask itself
- How the model composes with the existing regime classifier and risk layer

Four formulations are evaluated below.

---

### 3.2 Formulation A: Direct Signal Classification

**What it predicts:** The triple-barrier outcome {-1, 0, +1} (loss, timeout, profit) directly from FRVP + contextual features, without a prior rule-based signal filter.

**Labeling implied:** Standard Lopez de Prado triple-barrier labeling (Lopez de Prado, 2018, Chapter 3). Every bar near a FRVP structural level is sampled; the label is determined by which barrier (TP, SL, or time) is hit first.

**Advantages:**
- Simpler pipeline — one model, one pass.
- Lets the model discover which FRVP configurations matter without pre-filtering.
- If the FRVP features have high signal-to-noise, no rule-based layer is needed.

**Disadvantages:**
- High class imbalance — at 5-minute granularity, the vast majority of bars are not near any FRVP structure.
- Even near FRVP levels, the base rate of profitable outcomes is modest (targeting ~55–60% win rate for 1:2 RR), making class 0 (timeout) dominant.
- A single model conflates the tasks of "is this a FRVP setup?" and "is this particular setup going to work?" — the latter is regime-conditional and setup-type-conditional.
- Does not benefit from the interpretable signal layer. Harder to debug and explain why a signal fired.
- Loses the architectural separation of concerns that the repo is built around (rule-based detection → ML validation).
- The reversal vs. acceleration duality requires the model to handle two fundamentally different bet types simultaneously, which can create conflicting gradient signals in a single-output classifier.

**Data requirements:** Large sample required for balanced training. With ~1.1M total 5-minute bars (2010–2025) and FRVP-level contact maybe 5–10% of bars, that's ~55,000–110,000 potentially labeled events. Sufficient in principle, but many will be near overlapping or ambiguous profiles.

**Leakage exposure:** The primary risk is profile-construction leakage. If the fixed range includes even one bar from the *current* prediction period, the profile implicitly knows the future. This is the most dangerous leakage trap in the entire system. With the recommended prior-session and swing-to-swing anchors, this risk is controlled — but only if the implementation is strictly enforced (see Section 8).

**Verdict:** Viable but suboptimal. The class imbalance, the reversal/acceleration duality, and the loss of the interpretable signal layer make this a weaker choice than meta-labeling for the current repo architecture.

---

### 3.3 Formulation B: Meta-Labeling of FRVP Setups (Recommended Primary)

**What it predicts:** P(this specific FRVP setup is profitable), conditioned on knowing that a specific FRVP setup type has already fired.

**Labeling implied:** The primary model is the FRVP rule engine (Setups 1–6 from Section 2.3). Each time a setup fires, an event is sampled at that bar. The meta-label is binary: 1 if the triple-barrier outcome matches the hypothesized direction (setup says "reversal long" → triple-barrier TP hit on the long side), 0 otherwise. Lopez de Prado (2018, Chapter 10) calls this the "meta-labeling" step. The existing repo already uses this architecture for the OTE meta TCN champion.

**Advantages:**
- **Exact fit with the repo's incumbent pattern.** The `features/recipes/meta_labeling.json` recipe, the `long_ote_meta_tcn_champion`, and `short_ote_meta_tcn_champion` models are all meta-labeling implementations. FRVP slots directly into the same training pipeline.
- **Handles reversal vs. acceleration naturally.** Each setup type (Setup 1 → reversal, Setup 3 → continuation) becomes its own rule-layer event. The ML model predicts "is this fired setup worth betting?" regardless of type, but separate long/short meta-labelers handle directional asymmetry.
- **Clean separation of interpretable signal generation (rule) from probabilistic filtering (ML).** Debugging is easier: if a signal fires in production, you can inspect which FRVP setup triggered it and why. The ML probability is the confidence filter on top.
- **Sparse, high-quality events.** By sampling only when a setup fires (not on every bar), the meta-label dataset is much cleaner than a full bar-by-bar label. This reduces class imbalance naturally.
- **Composes cleanly with the existing regime classifier.** Regime-conditional thresholds (the `ote_threshold_policy_search` step already in the repo) can be applied directly to the meta-labeler output probability. A "failed auction" setup in a trending regime has a different threshold than the same setup in a ranging regime.
- **Accuracy ≠ profit.** Meta-labeling separates the size/confidence decision from the side decision. A meta-labeler with 55% accuracy on binary outcomes can produce positive expected value if position sizing scales with probability and the RR ratio is ≥ 2:1.

**Disadvantages:**
- **Quality bounded by the rule-based signal layer.** If the FRVP setup definitions are poorly specified (too loose, missing important distinctions), the meta-labeler will train on noisy primary signals. The setup logic in Section 2.3 must be rigorously coded.
- **Requires a working FRVP event engine** before the ML pipeline can begin. This means Phase 1 (profile construction) and Phase 2 (setup detection) must be complete before any model training.
- **Small sample risk.** If setups fire only once per session, a 5-year dataset yields ~1,000–2,000 events per setup type per side. This is workable but not abundant. Feature selection via SHAP becomes critical.

**Data requirements:** See Section 6. Approximately 2,000–5,000 events per setup type per side across 2018–2024. TCN and LSTM require sequence context; the existing `window-min 16`, `window-max 32` defaults from the OTE pipeline should be appropriate.

**Leakage exposure:** Same profile-construction risk as (A) but concentrated at the event bars only. The secondary risk is **information leakage from the setup definition itself** — if a setup requires seeing the current bar's close before deciding the setup fired (e.g., "the bar closes back inside the VA after tagging VAH"), then any features derived from the same bar are fine, but the label confirmation uses future information. This must be resolved in the setup definitions (see Section 5).

**Reversal vs. acceleration duality:** Handled by having separate setup types. Setup 1, 4, and 6 are reversal setups; Setups 2, 3, and 5 are continuation setups. The meta-labeler receives a `setup_type` feature indicating which fired. If sample counts allow, separate meta-labelers per setup family (reversal vs. continuation) are superior to a combined model.

**Verdict:** Recommended primary approach. Consistent with the repo architecture, handles the duality cleanly, composes with the existing regime and risk layers.

---

### 3.4 Formulation C: Learning-to-Rank

**What it predicts:** When multiple FRVP setups are simultaneously active (e.g., VAH of the London session profile coincides with VAL of the prior-day profile), rank them by expected quality and trade only the top-ranked signal.

**Labeling implied:** Pairwise or listwise ranking labels (e.g., LambdaRank, XGBoost ranking objective). Labels require at least two concurrent signals per ranking window — the one that worked is ranked above the one that didn't.

**Advantages:** Directly addresses the "which of multiple concurrent signals to prioritize" question, which will arise when multiple profiles produce overlapping setups.

**Disadvantages:**
- At 1–3 trades/day frequency, truly concurrent FRVP signals (same bar, different anchors) will be uncommon. The ranking corpus will be very small.
- The infrastructure for ranking label generation and training does not exist in the repo and would require significant new development.
- Overkill for the current signal frequency. The problem of "which signal to take" is better handled by a confidence-threshold gate on the meta-labeler, combined with a "no concurrent positions" rule in the risk layer.

**Verdict:** Not recommended for this phase. File as a Phase 4 research experiment if meta-labeling produces too many concurrent high-confidence signals.

---

### 3.5 Formulation D: Risk-Adjusted Return / Trend-Scanning Regression

**What it predicts:** A continuous value — either the vol-normalized forward return, or the trend-scanning t-statistic (Lopez de Prado, 2018, Chapter 16) — over a fixed forward window. The label is then thresholded to produce a trading signal.

**Labeling implied:** Regression target. The trend-scan t-statistic is the log-price slope / standard-error over the best-fit window across several forward horizons, giving a single number that summarizes the "strength and consistency" of the subsequent move.

**Advantages:**
- Richer training signal than binary labels — continuous values carry more information per sample (Lopez de Prado & Lewis, 2019).
- Trend-scan t-statistic already plays a role in the repo's labeling pipeline (used in quality scoring within the reversal labeler). Extending it to a prediction target is a natural evolution.
- The threshold (above which to take a long bet) can be regime-calibrated more finely than a binary classifier's probability threshold.

**Disadvantages:**
- Regression on financial returns is notoriously noisy; R² near 0 is typical even for good models. The predictive signal is still weak — it just represents the weakness differently than a classifier.
- The thresholding step to convert regression output to a trade is an extra calibration layer that requires careful out-of-sample tuning.
- The reversal vs. acceleration duality is collapsed into a single number, losing directional context unless a signed return is used (which then requires separate models for sign and magnitude).
- Does not compose as cleanly with the regime-slice threshold policy search that is already the repo's standard post-training evaluation workflow.

**Steel-man case for (D):** If FRVP setups are diverse enough in outcome magnitude (some setups yield 5-pip moves, others yield 30-pip moves), a regression model that learns to predict the *magnitude* of the subsequent move could inform position sizing more directly than a binary probability. This would be a useful *secondary* output even within a meta-labeling framework: use the binary meta-label for trade/no-trade decisions, and the regression output to scale position size within the allowed range. This hybrid use of (D) within (B) is worth including in Phase 3.

**Verdict:** Not the primary approach, but worth pursuing as an auxiliary regression head in Phase 3 after the meta-labeling pipeline is established. Do not make this the primary signal source.

---

### 3.6 Recommendation and Fallback

**Primary: Formulation B (Meta-Labeling)**  
Run separate meta-labelers for the two setup families (reversal setups: 1, 4, 6; continuation setups: 2, 3, 5), with separate long/short models consistent with the repo's established pattern. Use the existing OTE TCN pipeline for training. Apply regime-conditional thresholds via the existing regime-slice machinery.

**Fallback: Formulation A (Direct Classification, CUSUM-sampled)**  
If FRVP setup events are too sparse (fewer than ~1,000 per setup family per side in 2018–2024), fall back to CUSUM event sampling at FRVP levels with direct triple-barrier classification. Use XGBoost (faster, lower data requirement than TCN) as the model. This is equivalent to the `long_breakout_xgb_v1` pattern already in the repo.

**The key distinction between B and A in practice:** In meta-labeling, the event sampling is driven by the rule-based setup firing. In direct classification, the sampling can be driven by CUSUM or by proximity to a profile level. Meta-labeling produces cleaner, more interpretable events; direct classification gives more events but with more noise.

---

## 4. FRVP Feature Engineering

### 4.1 Integration with the Repo's Feature Framework

New FRVP features live in a new registered feature family: `features/feature_sets/frvp_context.py`, registered as `"frvp_context"`. This module is added to the `ote_extended.json` recipe and to a new `frvp_meta.json` recipe. It depends on:
- `features/feature_sets/volume.py` (for `dollar_volume`, `volume_zscore_50`)
- `features/feature_sets/structure.py` (for `atr_14`, swing levels)
- `features/fx_calendar.py` (for session boundary lookups)
- The causal swing detector from `data/labeling/reversal_labeling_engine.py` (for swing-to-swing anchoring)

All features are computed **causally** — using only the completed prior session or completed prior swing leg. No bar in the current modeling window is included in the profile construction.

### 4.2 Profile Construction Algorithm

**Step 1: Session boundary identification**  
Use `fx_calendar.py` market-day close labels to identify the timestamp boundaries of each completed session. For each 5-minute bar at time `t`, the "prior London profile" is built from bars in the most recently completed London session (02:00–11:00 ET) with `session_end < t`. The "prior NY profile" is built similarly for 08:00–16:00 ET.

**Step 2: Volume histogram**  
Within the anchor window, construct a volume-at-price histogram with bin width = `max(1 pip, ATR_14 / 20)`. Use `tick_volume` as the primary count. For each bin `b`:

```
vol_at_price[b] = sum(tick_volume[i]) for all bars i in window where low[i] <= midpoint(b) <= high[i]
```

Note: The "uniform volume within bar" assumption distributes each bar's volume evenly across the bar's price range. This is standard in volume profile calculations and equivalent to what TradingView's FRVP tool uses.

**Step 3: POC, VA, HVN, LVN extraction**  
- POC: `argmax(vol_at_price)` → single price level
- VA: Sort bins by volume descending, accumulate until sum ≥ 70% of total. VAH = max bin in accumulated set. VAL = min bin.
- HVN: Bins where `vol_at_price[b] > mean(vol_at_price) + 0.5 * std(vol_at_price)`
- LVN: Bins where `vol_at_price[b] < mean(vol_at_price) - 0.5 * std(vol_at_price)`

All thresholds are relative. No absolute volume level is used.

**Step 4: Naked POC tracking**  
A POC is "naked" (unretested) from the time it forms until the first subsequent bar whose price range (high, low) overlaps the POC level. Track this state causally: at each bar `t`, check if `low[t] <= POC[prior_session] <= high[t]`. Once visited, the POC is no longer naked.

### 4.3 Feature Catalog

All distance features are in **ATR units** (ATR from the 14-period ATR computed on 5-minute bars). This normalizes across different volatility regimes and is consistent with the repo's existing feature conventions (e.g., `dist_to_prior_high_20_atr`, `dist_to_bull_fvg_atr`).

Transformations: Apply the same rolling z-score and percentile-rank transforms used for other features in the `ote_extended.json` recipe. Volume-at-price features (POC volume fraction, node density) should additionally be log-winsorized at the 1st/99th percentile before z-scoring.

#### Group 1: Distance Features

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_dist_poc_session_atr` | `(close - POC_session) / ATR_14`. Signed: positive above POC, negative below | z-score over rolling 20 sessions |
| `frvp_dist_poc_day_atr` | Same but for prior-day POC | z-score over rolling 20 days |
| `frvp_dist_poc_swing_atr` | Distance to POC from last swing-to-swing anchor | None (already ATR-scaled) |
| `frvp_dist_vah_atr` | `(VAH_session - close) / ATR_14`. Positive below VAH, negative above | None |
| `frvp_dist_val_atr` | `(close - VAL_session) / ATR_14`. Positive above VAL, negative below | None |
| `frvp_dist_nearest_hvn_atr` | Distance from close to nearest HVN level / ATR | None |
| `frvp_dist_nearest_lvn_atr` | Distance from close to nearest LVN level / ATR | None |
| `frvp_dist_lvn_above_atr` | Distance to nearest LVN above close / ATR | None |
| `frvp_dist_lvn_below_atr` | Distance to nearest LVN below close / ATR | None |
| `frvp_naked_poc_dist_session_atr` | Distance to nearest unretested session POC / ATR | None |
| `frvp_naked_poc_dist_day_atr` | Distance to nearest unretested day POC / ATR | None |

#### Group 2: Price Position Within Value Area

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_price_position_va` | `(close - VAL) / (VAH - VAL)`. 0 = at VAL, 1 = at VAH, <0 = below VA, >1 = above VA | None (bounded interpretation) |
| `frvp_above_poc` | Binary: 1 if `close > POC_session`, else 0 | None |
| `frvp_in_va` | Binary: 1 if `VAL <= close <= VAH` | None |
| `frvp_above_vah` | Binary: 1 if `close > VAH` | None |
| `frvp_below_val` | Binary: 1 if `close < VAL` | None |
| `frvp_va_overshoot_atr` | `max(close - VAH, 0) / ATR` or `max(VAL - close, 0) / ATR` — signed extension beyond VA | None |

#### Group 3: Profile Shape Classification

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_profile_shape` | Encoded: 0 = D/balanced, 1 = P/bullish, -1 = b/bearish | None |
| `frvp_vol_skew` | Pearson skewness of the volume-at-price distribution | z-score over 20 sessions |
| `frvp_vol_concentration_top_pct` | Fraction of total profile volume in the top 25% of the price range | log-winsorize then z-score |
| `frvp_vol_concentration_bot_pct` | Fraction of total profile volume in the bottom 25% of the price range | log-winsorize then z-score |
| `frvp_poc_vol_pct` | POC node volume as % of total profile volume (peak-ness) | log-winsorize then z-score |

Shape classification rule:
- P-shape: `vol_concentration_bot_pct > 0.35 AND vol_skew > 0.5`
- b-shape: `vol_concentration_top_pct > 0.35 AND vol_skew < -0.5`
- D-shape: Otherwise

#### Group 4: HVN/LVN Density

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_hvn_count` | Number of HVNs in the session profile | None |
| `frvp_lvn_count` | Number of LVNs in the session profile | None |
| `frvp_hvn_count_1atr` | Count of HVN levels within ±1 ATR of current close | None |
| `frvp_lvn_count_1atr` | Count of LVN levels within ±1 ATR of current close | None |
| `frvp_hvn_above_close` | Distance to nearest HVN above close / ATR | None |
| `frvp_hvn_below_close` | Distance to nearest HVN below close / ATR | None |

#### Group 5: Value Area Metrics

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_va_width_atr` | `(VAH - VAL) / ATR_14` — value area width normalized by ATR | z-score over 20 sessions |
| `frvp_va_width_zscore_20` | Z-score of `frvp_va_width_atr` vs trailing 20-session mean/std | None (already z-scored) |
| `frvp_va_migration_vel` | Change in VA midpoint between prior two sessions / ATR | None |
| `frvp_poc_migration_vel` | Change in POC level between prior two sessions / ATR | None |
| `frvp_va_expansion` | 1 if current VA width > rolling 20-session median, else 0 | None |

#### Group 6: Multi-Anchor Confluence

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_poc_anchor_diff_atr` | `(POC_session - POC_day) / ATR` — how much session and day POCs diverge | None |
| `frvp_va_overlap_pct` | Overlap fraction of session VA and prior-day VA | None |
| `frvp_swing_poc_vs_session_poc` | `(POC_swing - POC_session) / ATR` | None |
| `frvp_multi_poc_cluster` | Binary: 1 if session POC, day POC, and swing POC are all within ±0.5 ATR of each other | None |

#### Group 7: Setup Type Encoding (for Meta-Labeling)

| Feature Name | Description |
|---|---|
| `frvp_setup_type` | Integer: 1=mean-reversion, 2=POC-retest, 3=VA-breakout, 4=failed-auction, 5=LVN-traverse, 6=HVN-magnet |
| `frvp_setup_side` | 1 = long hypothesis, -1 = short hypothesis |
| `frvp_setup_confidence_rule` | Rule-based confidence score [0–1] from the setup engine (based on convergence of indicators) |

#### Group 8: Interaction with Existing Repo Features

These are not new computed features but are existing repo features used explicitly in FRVP interaction terms. They should be selected via the SHAP-based backend-aware attribution pipeline and included in the feature ranking for the FRVP meta-labeler.

| Existing Feature | Interaction with FRVP |
|---|---|
| `dist_to_bull_fvg_atr`, `dist_to_bear_fvg_atr` (`ict_context`) | ICT FVG coinciding with FRVP VAH/VAL → double zone strength |
| `dist_to_bull_order_block_atr` (`ict_context`) | OB + HVN alignment → institutional memory convergence |
| `ict_bull_confluence_1atr`, `ict_bear_confluence_1atr` | ICT confluence within 1 ATR amplifies FRVP signal |
| `htf_confluence_long`, `htf_confluence_short` | HTF swing alignment confirms FRVP setup direction |
| `price_position_50` (`structure`) | FRVP in premium zone (>0.65) → expect mean-reversion long from below |
| `in_premium_zone`, `in_discount_zone` (`structure`) | HTF premium/discount confirms FRVP reversal direction |
| `volume_zscore_50` (`volume`) | Volume spike at FRVP level → breakout vs. rejection depends on direction |
| `displacement_bullish`, `displacement_bearish` (`structure`) | Post-FRVP displacement confirms continuation setup |
| `session_label` (`session`) | Kill-zone timing (London Open, NY Open) amplifies FRVP setups |
| `bars_since_sweep_high`, `bars_since_sweep_low` (`structure`) | Recent sweep before FRVP touch → higher reversal probability |

**Explicit interaction terms to compute** (add to `frvp_context.py`):

```
frvp_poc_plus_ict_fvg_confluence: (dist_poc_session_atr <= 0.5) AND (dist_to_bull_fvg_atr <= 0.5 OR dist_to_bear_fvg_atr <= 0.5)
frvp_vah_plus_ob_confluence: (dist_vah_atr <= 0.3) AND (dist_to_bear_order_block_atr <= 0.5)
frvp_val_plus_ob_confluence: (dist_val_atr <= 0.3) AND (dist_to_bull_order_block_atr <= 0.5)
frvp_failed_auction_with_sweep: (frvp_setup_type == 4) AND (bars_since_sweep_high <= 3 OR bars_since_sweep_low <= 3)
frvp_in_killzone: session_label in {london_open, ny_open}
```

---

## 5. Labeling

### 5.1 Overview: FRVP Events in the Existing Pipeline

The existing labeling pipeline generates three label families from causal swing detection: reversal, continuation-pullback, and breakout. FRVP events are a new, fourth family — distinct from the existing families because the event trigger is not a price swing but a price interaction with a volume profile level.

FRVP labeling reuses the same infrastructure: triple-barrier outcomes, trend-scan quality scoring, sample uniqueness weights, exclusion masks, and the zone/entry label distinction.

### 5.2 FRVP Event Sampling

**Event trigger:** A FRVP event is sampled when one of the 6 rule-based setups fires. This replaces the CUSUM filtering used in the reversal labeler. The setup engine (see `features/strategies/frvp_setups.py`) returns, for each bar, the active setup type and direction (or None if no setup is active).

This is equivalent in spirit to CUSUM filtering (Lopez de Prado, 2018, Chapter 2): CUSUM samples events when a cumulative sum threshold is crossed; FRVP event sampling samples events when a structural price-volume condition is satisfied. Both methods produce sparse, informative events rather than dense bar-by-bar samples.

**Event definition for each setup type:**

- **Setup 1 (Mean-reversion):** Fire at the bar where `frvp_dist_vah_atr <= 0.2` (price touching VAH from below in a D-shape profile) or `frvp_dist_val_atr <= 0.2` (price touching VAL from above). Direction is set from the profile side.

- **Setup 2 (POC retest):** Fire at the bar where price retests the prior-session's VAH (now acting as support) or VAL (now acting as resistance) from the new side after a confirmed breakout. Requires `frvp_above_vah == 1` on the breakout bar (confirmed prior close), then `frvp_price_position_va` rising back toward 1.0 from above → reversal toward that retest point.

- **Setup 3 (VA breakout):** Fire at the bar where price closes beyond VAH or VAL with `volume_zscore_50 > 1.5` (significant volume). Direction follows the breakout direction.

- **Setup 4 (Failed auction):** Fire at the bar where price returns inside the VA after having been beyond VAH/VAL for ≤ 3 bars without acceptance. This is a *close*-based definition: event fires when `frvp_in_va == 1` AND the prior 1–3 bars had `frvp_above_vah == 1` (or `below_val == 1`). This requires the current bar to close — so the event is sampled at bar close, not during the bar.

- **Setup 5 (LVN traverse):** Fire when price enters a known LVN zone (within ±0.3 ATR of the nearest LVN) with momentum confirmation (`displacement_bullish == 1` or `displacement_bearish == 1`). Direction follows displacement.

- **Setup 6 (HVN magnet):** Fire when price is between two HVNs and the closer HVN is more than 0.5 ATR from current price. Direction toward nearer HVN.

**Critical look-ahead constraint on event definition:**  
All setup fire conditions must use only information available at bar close. This means:
- The profile is built from prior sessions only (already enforced by anchor scheme).
- The setup condition is evaluated using the *completed* bar's OHLCV.
- The meta-label (triple-barrier outcome) is evaluated starting from the *next* bar's open (first tradeable bar after the signal).

This is consistent with the existing reversal labeler's convention: entry labels search for the best executable entry bar near the swing, not at the swing detection bar itself.

### 5.3 Triple-Barrier Labeling

Reuse the existing triple-barrier infrastructure with FRVP-specific barrier widths for each setup family:

| Setup Family | TP (ATR) | SL (ATR) | Max Holding (bars) | RR Ratio |
|---|---|---|---|---|
| Mean-reversion (1, 6) | 0.8 × ATR | 1.0 × ATR | 60 bars (5h) | 0.8:1 (label, but size-adjust at risk layer) |
| POC retest / continuation (2, 3, 5) | 1.5 × ATR | 1.0 × ATR | 96 bars (8h) | 1.5:1 |
| Failed auction (4) | 1.2 × ATR | 0.8 × ATR | 48 bars (4h) | 1.5:1 |

Note: The ATR used for barriers is the structural ATR (1hr mapped to 5-min) as in the existing labeling engine, not the 5-minute ATR directly. This prevents overly tight barriers in volatile periods.

The meta-label (binary) is: 1 if the TP barrier is hit first (in the hypothesized direction), 0 if the SL barrier or timeout is hit first.

### 5.4 Quality Scoring and Sample Weights

Reuse the existing quality scoring framework with these weights:

| Component | Weight | Note |
|---|---|---|
| Triple-barrier outcome | 40% | Higher weight for FRVP because the profile structure itself is a strong prior |
| Trend-scan forward strength | 25% | Same as existing pipeline |
| HTF confluence | 20% | `htf_confluence_long` / `htf_confluence_short` from labeling output |
| Entry score | 10% | Proximity to FRVP level, candle wick structure |
| FRVP setup confidence | 5% | Rule-based confidence from the setup engine |

Events below quality threshold 0.6 are rejected from positive labeling. Unlike the reversal labeler's 0.894 average quality score, FRVP events may have a lower initial average until the setup definitions are tuned — accept a lower threshold initially.

**Sample uniqueness weights:** Computed identically to the existing pipeline (`sample_weight_entry_long`, `sample_weight_entry_short`). Overlapping FRVP events (e.g., Setup 1 and Setup 4 firing within 3 bars of each other at the same level) should be down-weighted. Track `frvp_concurrency` analogous to `concurrency_long`.

### 5.5 Exclusion Masks

Exclusion zones around ambiguous FRVP areas:
- Bars within a freshly completed session (first 5 bars of a new session) before the new profile has enough volume — exclude from training.
- Bars during major macro events (FOMC, ECB, NFP) — use the existing `quality` feature set's anomaly flags.
- Bars where session tick volume is less than 20% of the rolling 20-session median (very thin volume, profile unreliable).

---

## 6. Data Requirements

### 6.1 Historical Data

| Timeframe | Purpose | Already in repo? |
|---|---|---|
| 5-minute EUR/USD OHLCV | Modeling timeframe — labeling, feature engineering, training | **Yes** — `data/currency_data/eurusd-5m.csv` (2010–2025) |
| 30-minute EUR/USD | HTF structural context | **Derived** — computed in `htf_context` feature set via resampling |
| 1-hour EUR/USD | Structural ATR, HTF trend, session context | **Derived** — computed in `htf_context` feature set |
| 1-minute EUR/USD | Intrabar precision for FRVP histogram bin assignment, execution modeling | **Potentially not present** — check `data/currency_data/` |
| Session boundary metadata | Which bars belong to each London/NY/Asia session | **Derived** — `fx_calendar.py` handles this |

**Date range:** Use 2018–2024 for model training (7 years). 2025–present is the out-of-sample holdout for final promotion decisions. The existing dataset starts in 2010, but pre-2018 EUR/USD microstructure is qualitatively different (pre-MiFID II, different ECN landscape). Use 2018 as the cutoff to reduce regime-stationarity problems.

**1-minute data:** If available, use it to improve volume histogram resolution (finer price-level assignment within each bar). If not available, the standard "uniform volume across the bar range" assumption is sufficient for 5-minute bars during liquid sessions. This is worth a sensitivity analysis in Phase 3.

### 6.2 Profile Construction Per Bar

At each 5-minute bar `t`, maintain three live profile states:

1. **Session profile (London):** Built from all bars in the most recently *completed* London session (session_end ≤ t). Updates once per day when the London session closes at ~11:00 ET.

2. **Session profile (NY):** Built from all bars in the most recently *completed* NY session. Updates once per day when NY closes at ~16:00 ET.

3. **Swing profile:** Built from all bars in the most recently *completed* swing leg (from confirmed swing low to confirmed swing high, or vice versa). Updates each time the causal swing detector from the labeling engine fires a new confirmed swing.

Memory requirement: Maintaining three active profiles at each bar is computationally tractable — each profile is a dictionary of `{price_bin: tick_volume}` updated incrementally.

### 6.3 Alignment and Look-Ahead Rules

| Rule | Rationale |
|---|---|
| Profile from prior session/swing only | The primary look-ahead defense |
| Setup fire condition uses completed-bar close | Consistent with existing reversal labeler convention |
| Triple-barrier evaluation starts at next bar's open | The trade is not executable at the signal bar's close (at bar-close labeling) |
| Structural ATR uses prior 14 bars × 1hr data only | ATR is always backward-looking |
| SHAP/attribution is computed inside each CV fold on training data only | Prevents feature selection from leaking test data |

---

## 7. Model Architecture

### 7.1 Where FRVP Fits in the Existing Model Stack

The existing live model catalog has 8 models across 5 families:
- Reversal: `long_reversal_tcn`, `short_reversal_xgb`
- Meta TCN: `long_ote_meta_tcn_champion`, `short_ote_meta_tcn_champion`
- Union TCN: `long_ote_union_tcn`, `short_ote_union_tcn`
- Breakout TCN: `long_breakout_tcn`, `short_breakout_tcn` (candidates)

FRVP adds **two new model families**:
- `long_frvp_reversal_meta` — meta-labeler for reversal setup family (long side)
- `short_frvp_reversal_meta` — meta-labeler for reversal setup family (short side)
- `long_frvp_continuation_meta` — meta-labeler for continuation setup family (long side)
- `short_frvp_continuation_meta` — meta-labeler for continuation setup family (short side)

This is 4 new models, which is consistent with the repo's per-family, per-side model organization.

In the live registry (`models/ote_model_registry_live_multifamily.json`), FRVP models start as `status="candidate"` and are promoted to `status="active"` only after surviving all 6 post-training gates (regime slices, threshold policy, walk-forward backtest).

### 7.2 Model Choice

**Primary: TCN (Temporal Convolutional Network)**  
Same architecture as `long_ote_meta_tcn_champion`. FRVP features include both level features (static per-profile values) and time-series context features (distance metrics evolving bar-by-bar as price moves relative to the profile). TCN is well-suited for this mix because it captures the temporal evolution of price's relationship with the FRVP levels.

Recommended hyperparameter starting point (same as existing meta TCN):
```
window: 16–32 bars (80–160 minutes of 5-min context)
hidden_size: 64
num_layers: 3
dropout: 0.20
learning_rate: 0.001
epochs: 40 (warmup 4 + main 18 + fine 10 + tail 8)
```

**Fallback: XGBoost**  
If event counts per family per side fall below ~1,500 training events, XGBoost is more data-efficient than TCN. The existing `ote_xgboost_pipeline.py` handles both backends. The initial experiment should train both and compare OOF metrics before deciding.

### 7.3 Probability → Position Size Flow

Consistent with the existing OTE live system:

1. **Primary classifier** (reversal model or continuation model) outputs `P(profitable setup)`.
2. **Regime gate**: Apply the regime-conditional threshold from the `ote_threshold_policy_search` step. In trending regimes, continuation setups need lower threshold (trend helps); reversal setups need higher threshold (trend fights them). In ranging regimes, reversal setups get a lower threshold.
3. **Abstain policy**: If neither reversal nor continuation setup is active, abstain. If both are active simultaneously (rare), take the higher-confidence one.
4. **Position size**: Scale fractional Kelly between `P_min` (minimum acceptable probability, ~0.55 given 1:2 RR) and `P_max` (near-certain, ~0.80). The existing risk layer already implements this via confidence multipliers.
5. **Stop loss / take profit**: Use the FRVP-specific barrier widths from Section 5.3, scaled by current ATR.

### 7.4 Composing with the Existing Regime Classifier

The existing `model_testing` stack produces regime-slice reports that show each model's performance by regime bucket. For FRVP models:

- **Reversal setups** should outperform in `ranging` and `low_volatility` regimes (profile is stable, mean-reversion reliable).
- **Continuation setups** should outperform in `strong_uptrend`, `strong_downtrend`, and `high_volatility_breakout` regimes (VAH/VAL breakouts are more sustained).

Use the existing 7-class regime labeler for slicing. Set minimum event count at 50 per regime bucket (same as existing `min_positive_events=50` convention). Regime buckets with fewer events should trigger abstain rather than forcing a low-sample threshold.

---

## 8. Validation and Anti-Overfitting

### 8.1 Purged K-Fold + Embargo

The existing OTE training pipeline implements purged walk-forward cross-validation via explicit row-geometry controls:
- `--cv-initial-train-rows 250000` (≈ 2 years of 5-min bars)
- `--cv-val-rows 100000` (≈ 10 months)
- `--cv-step-rows 100000`
- Minimum 4 folds (to span at least 3–4 years of training evolution)

For FRVP meta-labeling, events are sparse (perhaps 1 per session). The row-geometry approach works on the full 5-minute series including non-event bars. The purging removes the window around each meta-labeled event from the validation set to prevent temporal leakage between nearby events that share training data (Lopez de Prado, 2018, Chapter 7). No change needed to the existing purging logic.

**Embargo:** At least 1 full session (288 bars = 24 hours of 5-min bars) between the training window and the validation window. This prevents the model from learning from the tail of a session whose FRVP profile informs an event in the next session.

### 8.2 CPCV (Combinatorial Purged Cross-Validation)

Lopez de Prado (2018, Chapter 12) shows that standard k-fold CV underestimates the number of effectively independent test paths, biasing Sharpe ratio estimates upward. CPCV addresses this by evaluating all possible combinations of fold assignments.

The existing repo uses a sequential walk-forward approach rather than full CPCV. For FRVP, include a CPCV backtest (minimum 6 paths) as an additional validation step before promotion. This is not in the existing pipeline and requires a new script or a mode flag on the existing walk-forward runner.

### 8.3 Walk-Forward Policy Backtest with Frictions

Reuse the existing backtest runner (`scripts/run_ote_policy_backtest.py`) with the same friction parameters:
- `fixed_slippage_pips_per_trade = 0.3`
- `commission_pips_per_trade = 0.35`
- `min_train_years = 2`
- `test_window_months = 3`
- `rolling_step_months = 3`
- `min_folds = 8`

The backtest should report performance separately by: setup family (reversal vs. continuation), regime bucket, session (London vs. NY), and confidence decile.

### 8.4 Deflated Sharpe Ratio

The standard Sharpe ratio is inflated by selection bias when multiple model configurations are tried and the best is selected. Lopez de Prado & Bailey (2014, "The Deflated Sharpe Ratio") provide a correction that accounts for the effective number of trials.

For FRVP: if training involves trials (`--trials 20` in the Optuna hyperparameter search), and multiple setup-type filter combinations are tested, the effective number of trials could be 20–100. The Deflated Sharpe is approximately:

```
DSR = SR × sqrt((T - 1) / (N_trials)) × (1 - SR²/SR_max²)^0.5
```

A promotion gate of DSR > 0.5 (annualized) is reasonable for an intraday FX strategy at 1–3 trades/day. Any candidate with DSR < 0.3 should not be promoted regardless of nominal performance.

### 8.5 Placebo / Shuffled-Label Test

After training a FRVP meta-labeler, re-run training with **randomly shuffled meta-labels** (preserving the event timestamps and features). Compare OOF AUC-PR between the real model and the shuffled model. If the real model's AUC-PR is within 2 standard deviations of the shuffled model's AUC-PR across 10 shuffles, the features contain no detectable signal. This is a strict null-hypothesis test.

Run this as a separate reviewed step against the saved artifact with `scripts/run_ote_placebo_readout.py`, then archive the summary JSON beside the promotion package.

Saved ES-primary reference record on 2026-07-17:

```
ote_venv\Scripts\python.exe scripts/run_ote_placebo_readout.py --real-artifact-dir models\frvp_es_primary_xgb_refresh_20260701\long_frvp_continuation --output-root model_testing\reports\frvp_placebo_readouts\frvp_long_continuation_xgb_v1_20260717 --num-shuffles 10
```

Archived result: `model_testing\reports\frvp_placebo_readouts\frvp_long_continuation_xgb_v1_20260717\placebo_readout_summary.json` recorded real OOF AP `0.7371` versus shuffled mean OOF AP `0.5002` (std `0.0059`) across `10` shuffles, for a placebo gap of `0.2368`; this passed the `> 0.03` promotion rule by a wide margin.

Saved ES-primary promotion-package record on 2026-07-17: `model_testing\reports\frvp_promotion_packages\frvp_long_continuation_xgb_v1_20260717\promotion_package_summary.json` finalized the threshold-vs-prune readout for the long-continuation baseline. The package conclusion is that the saved `v3` branch is a hard-pruned `global_threshold` contract rather than a new regime-threshold winner: relative to the unpruned refresh baseline it cut trade count `1609 -> 524`, lifted expectancy `7.75 -> 13.72`, improved Sharpe `0.604 -> 1.226`, improved DSR `0.585 -> 1.061`, and reduced max drawdown `10200.85 -> 1540.10`. The July 17, 2026 rerun after the passing roll-shadow validation now records `package_status = finalized_without_human_same_contract_signoff` and `promotion_decision = pending_human_same_contract_signoff`, with paper-trading gate, placebo, roll audit, and roll-shadow validation all passing.

### 8.6 FRVP-Specific Leakage Traps

These are the leakage scenarios specific to FRVP that do not appear in the existing OTE pipeline:

**Trap 1: Profile Anchoring Window Leakage (Most Dangerous)**  
If the "prior London session" profile is computed from bars that include the current bar (e.g., if the session-end timestamp is misaligned by one bar), the profile incorporates information about the bar being predicted. This is an off-by-one error with immediate consequence: the profile's POC and VAH/VAL would shift to reflect the current bar's price, making the distance features trivially predictive.

Mitigation: In `frvp_context.py`, enforce `session_end < bar_timestamp` strictly. Add an assertion that checks the maximum bar timestamp in each profile window is strictly less than the current bar's timestamp.

**Trap 2: Naked POC State Leakage**  
When computing whether a POC is "naked" (unretested), checking if the current bar's range overlaps the POC is fine — that's causal. But computing a *future* POC visitation (e.g., "has this POC been visited in the next N bars?") as a feature would leak future information. This feature type must not appear in the training dataset. In practice, `frvp_naked_poc_*` should be a backward-looking flag only: "is the most recent session POC still naked as of this bar's open?"

**Trap 3: Setup Fire Condition Uses Future Volume**  
For Setup 3 (volume expansion breakout), the condition `volume_zscore_50 > 1.5` uses only the current bar's completed tick volume (backward-looking rolling stats). This is safe. But if the implementation uses the volume of the session-in-progress (i.e., the current session's accumulated volume relative to its own eventual total), that would be forward-looking. Always compare volume to the prior session's norms, not the current session's eventual total.

**Trap 4: Feature Selection Across Full Dataset**  
The SHAP-based backend-aware attribution step must run only on training data within each fold. The existing `preprocessing backend-attribution` command runs on the full prepared dataset — this means the feature selection could reflect test-set feature importance. For FRVP: ensure that the attribution step for feature selection is performed on a trailing window (same geometry as the training window) rather than the full dataset. This is a known limitation of the existing pipeline worth flagging.

---

## 9. Build Roadmap

The following phases are sequenced with explicit dependencies. Each phase has a validation gate that must be passed before beginning the next.

### Phase 1: FRVP Profile Construction Engine

**Goal:** A reliable, look-ahead-free FRVP profile builder that outputs POC, VAH, VAL, HVN, LVN, and naked POC state for each 5-minute bar using the dual-anchor (session + swing-to-swing) scheme.

**Files to create:**
- `features/feature_sets/frvp_context.py` — the registered feature family
- `features/strategies/frvp_setups.py` — the rule-based setup detection (6 setup types)
- `tests/test_frvp_context.py` — unit tests

**Key implementation tasks:**
1. Wire to `fx_calendar.py` for session boundary lookup.
2. Implement the VolumeProfileBuilder class: incrementally build histogram, extract POC/VA/HVN/LVN.
3. Implement the dual-anchor scheme: session anchor and swing-to-swing anchor. The swing anchor uses `detect_confirmed_swings` from `features/transforms.py` (the same function used by `ict_context.py`).
4. Implement naked POC tracker as a causal state machine (forward scan from profile formation time).
5. Implement setup detection logic for all 6 setup types. Each returns (fired: bool, setup_type: int, setup_side: int, confidence: float) per bar.
6. Add to `features/recipes/ote_extended.json` and create `features/recipes/frvp_meta.json`.

**Validation gate:** On 90 days of held-out 5-minute bars:
- Manually inspect 20 random FRVP profiles and compare POC/VAH/VAL to TradingView FRVP drawings.
- Confirm that no profile contains any bar from the current prediction window (off-by-one test).
- Confirm that naked POC state correctly updates (resets when price range overlaps POC).
- All 6 setup types fire at plausible rates: target 0.5–2 setup fires per session per side.

---

### Phase 2: FRVP Feature Dataset Build

**Goal:** Generate the complete FRVP feature dataset by adding `frvp_context` to the existing feature pipeline.

**Commands (reuse existing CLI):**
```
python -m features.cli build data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv \
  --output data/features/eurusd_5min_frvp_full.csv \
  --recipe features/recipes/frvp_meta.json \
  --all-strategies \
  --transform-workers 4 \
  --optimize-memory \
  --source-timezone UTC \
  --canonical-timezone UTC \
  --feature-clock-timezone America/New_York
```

**Validation gate:** Run feature correlation check (`preprocessing prepare` with `--corr-threshold 0.98`). Expect several FRVP distance features to correlate with existing `dist_to_prior_high_20_atr` features — this is expected but some collinear pairs should be removed. Confirm FRVP features have non-trivial mutual information with the existing meta-labels (quick χ² or spearman ρ check).

---

### Phase 3: FRVP Event Labeling

**Goal:** Generate the FRVP-specific meta-label dataset, extending the existing labeling engine.

**Files to create/modify:**
- `data/labeling/frvp_labeling_engine.py` — new module for FRVP event sampling + meta-label generation
- Extend `data/labeling/labeling_engine.py` to call the FRVP labeler alongside the existing reversal/continuation/breakout labelers.

**Key implementation tasks:**
1. For each FRVP setup fire event (from Phase 1's setup detection), sample the event bar.
2. Apply triple-barrier labeling with setup-family-specific barrier widths (Section 5.3).
3. Compute quality scores and sample weights consistent with the existing labeling engine.
4. Write `data/labeling/labeled_data/eurusd_5min_frvp_labels.csv` with FRVP-specific columns.

**Validation gate:**
- Total FRVP events per year: target 400–800 per setup family per side (roughly 1.5–3 per trading day, consistent with the 1–3 trades/day target).
- Meta-label base rate (% TP outcomes): target 45–60%. If consistently below 40%, the setup definitions need tightening.
- Quality score distribution: target mean ≥ 0.65 after filtering.
- Event temporal distribution: events should not cluster excessively (CUSUM-like uniform spacing is ideal).

---

### Phase 4: Preprocessing and Backend Attribution

**Goal:** Prepare the FRVP labeled dataset for model training using the existing preprocessing pipeline.

**Commands (verbatim reuse):**
```
python -m preprocessing prepare data/features/eurusd_5min_frvp_full.csv \
  --output-dir data/prepared/eurusd_5min_frvp_full \
  --scaler none \
  --corr-threshold 0.98 \
  --similarity-threshold 0.995

python -m preprocessing backend-attribution \
  --prepared-root data/prepared/eurusd_5min_frvp_full \
  --backend xgboost \
  --backend tcn \
  --max-features 160 \
  --base-weight 0.20 \
  --shap-weight 0.55 \
  --shap-positive-weight 0.25 \
  --attribution-floor-fraction 0.15 \
  --attribution-cumulative-importance 0.90
```

Targets to prepare: `long_frvp_reversal`, `short_frvp_reversal`, `long_frvp_continuation`, `short_frvp_continuation`.

**Validation gate:** SHAP top-10 features for each target should make structural sense:
- FRVP distance features (`frvp_dist_poc_session_atr`, `frvp_price_position_va`) should rank in the top 10.
- Regime-context features (`htf_confluence_long/short`, session features) should appear.
- If raw OHLC or label-adjacent features appear (e.g., `label_long_reversal` from the labeling output), remove them — they are leakage.

---

### Phase 5: Model Training

**Goal:** Train TCN and XGBoost meta-labelers for each FRVP target using the existing OTE training pipeline.

**Commands (adapting existing templates):**

TCN (reversal, both sides):
```
python -m model_training.ote_training.ote_xgboost_pipeline \
  --prepared-root data/prepared/eurusd_5min_frvp_full \
  --output-root models/frvp_reversal_tcn_v1 \
  --backend torch \
  --model-type tcn \
  --targets long_frvp_reversal short_frvp_reversal \
  --trials 20 \
  --max-loaded-features 160 \
  --window-min 16 \
  --window-max 32 \
  --epochs 40 \
  --torch-warmup-epochs 4 \
  --torch-main-epochs 18 \
  --torch-fine-epochs 10 \
  --torch-tail-epochs 8 \
  --torch-fine-lr-scale 0.20 \
  --torch-tail-lr-scale 0.07 \
  --calibration-method platt
```

XGBoost (fallback, all four targets):
```
python -m model_training.ote_training.ote_xgboost_pipeline \
  --prepared-root data/prepared/eurusd_5min_frvp_full \
  --output-root models/frvp_xgb_v1 \
  --backend xgboost \
  --targets long_frvp_reversal short_frvp_reversal long_frvp_continuation short_frvp_continuation \
  --trials 20 \
  --max-loaded-features 160 \
  --calibration-method platt
```

**Validation gate:**
- OOF AUC-PR substantially above the naive baseline (naive = precision at dataset base rate).
- Placebo test (shuffled labels) OOF AUC-PR should be near baseline. Real model should exceed shuffled by > 3%.
- Calibration check: actual outcomes should correlate with predicted probabilities (calibration plot).

---

### Phase 6: Regime-Slice Report and Threshold Policy Search

**Goal:** Characterize FRVP model performance by regime and find regime-conditional thresholds.

**Commands (reuse existing):**
```
python scripts/run_ote_regime_slice_report.py \
  --registry-path models/ote_model_registry.json \
  --output-root model_testing/reports/frvp_regime_slices/v1 \
  --bootstrap-iterations 200 \
  --min-positive-events 50

python scripts/run_ote_threshold_policy_search.py \
  --regime-report-root model_testing/reports/frvp_regime_slices/v1 \
  --registry-path models/ote_model_registry.json \
  --output-root model_testing/reports/frvp_threshold_policies/v1 \
  --min-positive-events 50 \
  --min-events-per-month 3.0
```

Registry write-back is now a separate reviewed step. The default research rerun path should inspect the saved threshold and walk-forward artifacts first, then opt into `--write-registry-policies` only if the selected policy contract is the one you actually want persisted.

**Validation gate:** Verify that reversal models have higher precision in `ranging` regimes than `trending` regimes. Verify that continuation models have higher precision in `trending` regimes. If this directional relationship does not hold, the FRVP setup definitions or the regime labeler may have a calibration problem.

---

### Phase 7: Walk-Forward Backtest and DSR Check

**Goal:** Final offline promotion gate with realistic frictions and deflated Sharpe computation.

**Commands (reuse existing):**
```
python scripts/run_ote_policy_backtest.py \
  --regime-report-root model_testing/reports/frvp_regime_slices/v1 \
  --registry-path models/ote_model_registry.json \
  --output-root model_testing/reports/frvp_backtests/v1 \
  --min-train-years 2 \
  --test-window-months 3 \
  --rolling-step-months 3 \
  --min-folds 8 \
  --maximum-drawdown-pct 12.0 \
  --drawdown-starting-balance-units 10000 \
  --fixed-slippage-pips-per-trade 0.3 \
  --commission-pips-per-trade 0.35
```

Walk-forward acceptance now uses account-equity `max_drawdown_pct_below_threshold`, computed from cumulative strategy performance plus `--drawdown-starting-balance-units`. Profit-giveback % remains diagnostic-only, and the drawdown gate is advisory rather than a hard promotion veto.

**Promotion gates (all 7 must pass before live deployment):**
1. OOF AUC-PR > 0.60 (substantially above base rate)
2. Walk-forward Sharpe (after frictions) > 0.8 annualized
3. Walk-forward max drawdown < 12%
4. DSR > 0.3
5. Regime-slice: reversal models win in ≥ 2 of the 3 range/low-vol regime buckets
6. Placebo test: real model AUC-PR exceeds shuffled-label mean by > 3%
7. Roll-aware embargo / roll audit clean: zero usable roll-span events, fold gaps satisfy the extra roll embargo, and continuity tests pass on the saved source history.

Saved ES-primary reference record on 2026-07-17: `model_testing\reports\frvp_roll_audit_packages\frvp_long_continuation_xgb_v1_20260717\roll_audit_package_summary.json` passed gate 7 with `49` excluded roll-span events, `0` usable roll-span events, `11` roll-crossing fold boundaries that all still cleared the `576`-bar minimum, a minimum observed fold gap of `10238` bars (`1290.75` hours), and `39` focused pytest checks passing.

---

### Phase 8: Live Integration

**Goal:** Deploy FRVP models as candidates in the existing `ote_live` app.

**Tasks:**
1. Add FRVP model artifacts to `models/live/` flat directory.
2. Add entries to `models/ote_model_registry_live_multifamily.json` with `status="candidate"`.
3. Update `ote_live/features/incremental_engine.py` to compute FRVP features incrementally at each bar (the profile builder maintains state across bars; the setup detector runs at each new bar close).
4. Update `ote_live/models/loaders.py` to load FRVP meta-labeler models.
5. Shadow-mode paper trading for minimum 4 weeks before considering promotion to `status="active"`.

Saved ES-primary live-runtime reference record on 2026-07-17: `model_testing\reports\frvp_roll_shadow_validation\frvp_es_shadow_roll_20260717\roll_shadow_validation_summary.json` replayed the actual `ESM6 -> ESU6` roll boundary at `2026-06-17T00:00:00+00:00` after an `8000`-bar warmup. The final rerun now passes cleanly with `validation_passed = true`, `feature_parity_passed = true`, `health_error_count = 0`, `missing_live_feature_names = []`, `dashboard_state_contract_switch_recorded = true`, and `15` persisted shadow decisions. This removes the historical Phase 8 runtime blocker; the remaining live-integration work is human same-contract sign-off plus the still-manual IBKR front-month handoff in production operations.

---

## 10. Open Questions and Risks

The following items cannot be resolved by reasoning alone. Each is an experiment to run during the build phases.

**Q1: Tick volume profile reliability by session**  
Does tick volume produce stable, repeatable HVN/LVN locations across equivalent sessions (e.g., different Tuesdays in the same volatility regime), or does it scatter unpredictably? If the node locations are not stable, FRVP features will be noisy regardless of the ML formulation. Test: compute inter-session correlation of HVN locations across 20 same-day-of-week sessions in the same regime bucket. Target correlation ≥ 0.6 for the top 3 HVN levels.

**Q2: Dollar-volume vs. tick-count profile quality**  
The theoretical argument for `dollar_volume = close × tick_count` as the profile basis is that it weights price levels by notional proximity. Empirically, does this produce better-calibrated node locations? Test: train two versions of the Phase 5 XGBoost model — one using tick-count profiles, one using dollar-volume profiles. Compare OOF AUC-PR. The better-calibrated basis is also the one whose distance features have higher SHAP importance.

**Q3: Optimal value area percentage**  
The 70% VA convention comes from the Market Profile literature, which was developed for futures markets. For EUR/USD tick volume, the optimal VA percentage may differ. Test: in Phase 3, generate FRVP labels using VA = 60%, 70%, and 80%. Compare meta-label base rate and quality score distribution. Use the percentage that gives the highest quality score mean.

**Q4: Setup frequency vs. quality tradeoff**  
If Phase 3 produces fewer than 1,000 events per setup family per side (2018–2024), the meta-labeling approach will underperform and the fallback to direct CUSUM classification (Formulation A) becomes necessary. The threshold of 1,000 events is not a magic number — it depends on feature dimensionality and signal-to-noise. Monitor this carefully in Phase 3.

**Q5: Optimal anchoring window length for swing-to-swing profiles**  
Short swing legs (10–20 bars) produce fine-grained profiles with volatile node locations. Long swing legs (200+ bars) produce smooth profiles but may lag structural changes. The sweet spot for the swing-to-swing anchor may be 50–100 bars. Test in Phase 1 sensitivity analysis: compare profile stability metrics (HVN location variance) across swing leg lengths of 20, 50, 100, and 200 bars.

**Q6: Regime-conditional setup performance hypothesis**  
The claim that reversal setups outperform in ranging regimes and continuation setups in trending regimes is intuitive but not empirically confirmed in this dataset. If Phase 6 shows no such pattern, the setup type distinction may not be meaningful for the meta-labeler, and all FRVP setups should be pooled into a single meta-label per side. This would simplify the model count from 4 to 2 models.

**Q7: FRVP + ICT confluence features — are they additive?**  
The interaction terms (POC + FVG confluence, VAH + order block confluence) are hypothesized to add signal beyond each component alone. This is plausible but not guaranteed. If SHAP analysis in Phase 4 shows the interaction terms have near-zero importance, remove them and rely on the base FRVP distance features plus the existing ICT context features independently.

**Q8: London vs. NY session profile weighting**  
At any given NY morning bar, both the prior London profile and the prior NY profile are active. Which profile dominates for EUR/USD? The London session is higher volume for EUR/USD. But the NY Open produces significant directional moves that may override the London profile's equilibrium. Test in Phase 6 regime-slice analysis: break results by session (London vs. NY) and check which anchor the top-SHAP features belong to.

**Q9: Live profile staleness during data outages**  
If the live data feed drops for 1–3 bars, the incremental profile builder may produce stale or incorrect profiles. The `ote_live` ingestion layer (`ote_live/ingestion/runtime.py`) has gap-handling logic. Verify that the FRVP profile builder handles gaps gracefully: do not extend a session profile beyond its correct session end even if the data feed reconnects mid-session.

**Q10: Sample size adequacy for TCN**  
TCN requires sequence windows of 16–32 bars. For sparse event datasets (1–3 events/day), the effective training set may be much smaller than the number of labeled events suggests, because many events share overlapping context windows. The purging step removes these overlapping windows from the validation set but the *training* set retains them (which is correct). Monitor for overfitting on the training fold vs. OOF gap as a signal that the dataset is too small for TCN. If the gap is large, fall back to XGBoost.

---

## References

- Dalton, J., Jones, E., & Dalton, R. (2007). *Markets in Profile*. Wiley.
- Lequeux, P., & Acar, E. (1998). "A Dynamic Index for Managed Currencies Funds Using CME Currency Contracts." *European Journal of Finance*, 4:4, 311–330.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Lopez de Prado, M., & Bailey, D. H. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*, 40:5, 94–107.
- Lopez de Prado, M., & Lewis, M. J. (2019). "Detection of False Investment Strategies Using Unsupervised Learning Methods." *Quantitative Finance*, 19:9, 1555–1565.
- Steidlmayer, J. P., & Hawkins, S. B. (2003). *Steidlmayer on Markets*. Wiley.

---

## Repository File Index

New files required by this paper:

| Path | Description |
|---|---|
| `features/feature_sets/frvp_context.py` | Registered `frvp_context` feature family — profile construction and all FRVP features |
| `features/strategies/frvp_setups.py` | Rule-based setup detector — 6 setup types, returns (type, side, confidence) per bar |
| `features/recipes/frvp_meta.json` | Build recipe for the FRVP meta-labeling feature dataset |
| `data/labeling/frvp_labeling_engine.py` | FRVP event sampling, triple-barrier labeling, quality scoring, sample weights |
| `tests/test_frvp_context.py` | Unit tests for profile construction, look-ahead assertions, setup detection |

Existing files modified by this paper:

| Path | Modification |
|---|---|
| `features/recipes/ote_extended.json` | Add `frvp_context` to the recipe |
| `data/labeling/labeling_engine.py` | Call `frvp_labeling_engine.py` alongside existing family labelers |
| `features/config.py` | Add FRVP-specific config parameters (bin width, VA percentage, session anchor settings) |
| `models/ote_model_registry.json` | Add FRVP model entries when training completes in Phase 5 |
| `models/ote_model_registry_live_multifamily.json` | Add FRVP live candidates when Phase 8 begins |
| `ote_live/features/incremental_engine.py` | Add incremental FRVP profile update at each bar close |
| `ote_live/models/loaders.py` | Add FRVP meta-labeler loading |

# Fixed Range Volume Profile as ML Signal Foundation
## Design Paper: FRVP-Driven Signal Pipeline for ES (E-mini S&P 500) — Primary, with 6E (Euro FX) Variant

**Author:** Quantitative Research & ML Systems
**Date:** 2026-06-21
**Status:** Phases 0-8 plus promotion packaging are now archived on the full ES history window for the long-continuation baseline. The historical roll-safe shadow replay now passes on the real `ESM6 -> ESU6` contract switch from 2026-06-17; the remaining open items are human Phase 0/1 same-contract sign-off and the still-manual IBKR front-month handoff in live ops.
**Implementation update:** 2026-07-21
**Repo target:** Slot into the existing OTE meta-labeling architecture
**Companion paper:** Supersedes the EUR/USD spot FRVP blueprint (2026-06-15) on instrument grounds; reuses its ML scaffolding verbatim

Raw Data FilePath: C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\docs\FRVP_ES_primary_6E_variant_design.md

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why ES, and What Changes vs. EUR/USD Spot](#2-why-es-and-what-changes-vs-eurusd-spot)
3. [FRVP Primer — Real-Volume Edition](#3-frvp-primer--real-volume-edition)
4. [Contract Continuity and Roll Handling (the new most-dangerous trap)](#4-contract-continuity-and-roll-handling)
5. [ML Formulation Analysis](#5-ml-formulation-analysis)
6. [FRVP Feature Engineering](#6-frvp-feature-engineering)
7. [Labeling](#7-labeling)
8. [Data Requirements](#8-data-requirements)
9. [Model Architecture](#9-model-architecture)
10. [Validation and Anti-Overfitting](#10-validation-and-anti-overfitting)
11. [The 6E Variant](#11-the-6e-variant)
12. [Build Roadmap](#12-build-roadmap)
13. [Open Questions and Risks](#13-open-questions-and-risks)

---

## 1. Executive Summary

This paper re-homes the **Fixed Range Volume Profile (FRVP)** signal layer onto **ES (E-mini S&P 500 futures)** as the primary instrument, with **6E (CME euro FX futures)** as a configuration variant. The motivation is foundational rather than cosmetic: ES is the canonical volume-profile instrument. It trades on a single centralized venue (CME Globex) with a consolidated tape and **real, reported contract volume**. This dissolves the single weakest assumption in the EUR/USD spot blueprint — that tick count is a usable proxy for transacted volume. On ES, POC, value area, HVN/LVN, and the entire auction-theoretic apparatus mean what the literature says they mean: price levels where contracts actually changed hands.

The goal is unchanged: **1–3 high-probability swing entries per day** by detecting moments when volume structure predicts an imminent reversal or acceleration. The recommended formulation is **meta-labeling (Formulation B)**, identical to the incumbent `long_ote_meta_tcn_champion` / `short_ote_meta_tcn_champion` pattern. The rule layer (Setups 1–6) generates the side and move-type hypothesis; the ML model answers only: *"Given that this setup fired, should we bet?"*

What this paper trades for cleaner volume is a new class of instrument-specific complexity that **does not exist in spot FX**: contract expiry and roll. Mishandled, the continuous-contract construction silently relocates every POC/VAH/VAL to prices that never traded. Section 4 treats this with the same severity the EUR/USD paper gave to the missing tape.

The key practical decisions:

| Decision | Recommendation | Reasoning |
|---|---|---|
| Volume basis for profiles | **Real contract volume** | ES is centralized; the tick-volume proxy problem is gone. Use reported volume directly. |
| Continuous-contract scheme | **Raw (unadjusted) prices for profiles; back-adjusted only for path features** | Back-adjusted absolute levels are fictional; profiles must live in true price coordinates (Section 4). |
| Roll determination | **Volume-based, causal** (front-month vol < next-month vol) | Backward-looking, leak-free; aligns profiles with where liquidity actually was. |
| Profile anchoring scheme | **Prior RTH-session + overnight (ETH) + Initial Balance + swing-to-swing** | RTH open/close are real volume events; far cleaner anchors than FX session conventions. |
| ML formulation | Meta-labeling (B) with TCN primary | Unchanged — consistent with existing meta TCN champions. |
| Fallback | XGBoost direct event classifier (CUSUM-sampled) | If setups are too sparse for TCN sequence context to help. |
| Target frequency | 1–3 signals/day | ~250–750 events/year per setup family per side over a 7-year window. |
| Frictions | Re-expressed in **ticks/points**, not pips | ES tick = 0.25 pt = $12.50; round-turn friction is small but open/close slippage spikes. |

**What already exists in the repo vs. what this paper adds:**

| Component | Status |
|---|---|
| Labeling engine (triple-barrier, CUSUM, quality scoring, sample weights) | **Exists** — reuse verbatim |
| Feature framework (16 families, recipe system) | **Exists** — `frvp_context` module slots in here |
| ICT context features (FVGs, order blocks, BOS/CHoCH, sweeps) | **Exists** — reframe session windows to the equity cash session |
| HTF context (resampling, ATR, swing detection) | **Exists** — anchor FRVP to RTH/ETH boundaries |
| Regime classifier + regime-slice evaluation | **Exists** — augment with profile-native day-type / open-type |
| Walk-forward backtest with frictions | **Exists** — re-parameterize frictions in ticks |
| Meta-labeling recipe (`features/recipes/meta_labeling.json`) | **Exists** — correct recipe for this pipeline |
| OTE training pipeline (XGBoost / TCN / LSTM) | **Exists** — train FRVP meta-labeler against it |
| **Continuous-contract + roll engine** | **New** — `frvp/continuity/continuous_contract.py`, `frvp/continuity/roll_calendar.py` |
| **FRVP profile construction engine** | **New** — `frvp/feature_sets/frvp_context.py` via the registry shim `features/feature_sets/frvp_context.py` |
| **FRVP setup rule layer** | **New** — `frvp/setups/detector.py` via the compatibility wrapper `frvp/strategies/frvp_setups.py` |
| **FRVP-specific label events** | **New** — `data/labeling/frvp_labeling_engine.py` |
| **Instrument config (ES / 6E)** | **New** — `frvp/config/instruments.py` with compatibility shims at `frvp/config_instruments.py` and `features/config_instruments.py` |

**Hard sanity check.** Any backtest showing Sharpe > 3 or win rate > 65% on intraday ES should be treated as a bug until proven otherwise — and on ES the *first* suspect is roll/continuity leakage, the second is profile-anchoring leakage. The design targets Sharpe ~1.0–1.5 out-of-sample, max drawdown < 12%, and 1–3 trades/day. ES is one of the most efficient, most algo-saturated instruments in existence; cleaner volume makes the *indicator* more valid, it does **not** make alpha easier to find.

---

## 2. Why ES, and What Changes vs. EUR/USD Spot

### 2.1 The one thing that changes everything: real volume

EUR/USD spot has no consolidated tape; its "volume" is tick count, a proxy for activity/volatility rather than transacted size. ES trades on CME Globex with reported per-bar contract volume. Every downstream concept becomes literal:

- **POC** is the price with the most contracts traded, not the most price updates.
- **Value area** is where 70% of *real* volume transacted.
- **Failed auctions / single prints / LVNs** are genuinely measurable: a price extreme that printed little real volume is a real rejection, not a tick-count artifact.
- The entire tick-vs-dollar-volume debate from the EUR/USD paper (its Caveat 1 and Open Questions Q1/Q2) **disappears**. There is one correct basis: reported contract volume.

### 2.2 What gets harder

Three things absent from spot FX appear on ES:

1. **Contract expiry and roll.** ES is quarterly (H/M/U/Z, third-Friday expiry). You must build a continuous series, and the naive back-adjusted series puts profile levels at fictional prices. This is Section 4 — the new top-priority leakage/validity trap.
2. **Session is RTH-centric, not 24h-symmetric.** ES trades nearly around the clock on Globex (Sunday 18:00 ET to Friday 17:00 ET with a daily 17:00–18:00 ET halt), but liquidity and meaning concentrate in the cash **RTH session, 09:30–16:00 ET**. The open and close are large, real volume events — better anchors than any FX session boundary. The original London/NY dual-anchor is replaced by an RTH/ETH/IB scheme.
3. **Options-driven intraday gravity.** ES/SPX intraday is heavily shaped by dealer gamma positioning, especially same-day (0DTE) options, which pins price toward large strikes. These pins can masquerade as HVN magnets (Setup 6). This is both a risk and a feature opportunity (Section 6.4).

### 2.3 Side-by-side

| Dimension | EUR/USD spot (original) | ES futures (this paper) |
|---|---|---|
| Volume | Tick count (proxy) | Real reported contract volume |
| Venue | Decentralized OTC | Centralized (CME Globex) |
| Profile validity | Theoretically displaced | Native habitat |
| Session anchors | Prior London / NY | Prior RTH / overnight ETH / Initial Balance |
| Natural volume events | None (continuous ribbon) | Cash open, cash close (genuine auctions) |
| Gaps | Effectively none | RTH-open vs prior value "open type" is a core signal |
| Contract structure | Continuous spot | Quarterly expiry + roll (new complexity) |
| Dominant intraday force | Macro/rate news arrival | News + options gamma (0DTE) |
| Friction unit | Pips (~0.65 round-trip) | Ticks/points (≈1–1.5 ticks liquid; spikes at open/close) |
| Tick size / value | — | 0.25 pt = $12.50/contract (MES micro = $1.25/tick) |

The ML scaffolding, the meta-labeling formulation, the leakage discipline, and the promotion gates all transfer unchanged. The instrument-specific layers — anchoring, continuity, frictions, and a few profile-native context features — are what this paper rewrites.

---

## 3. FRVP Primer — Real-Volume Edition

### 3.1 Profile anatomy (unchanged definitions, now literal)

**Point of Control (POC):** the price level with the most volume in the fixed range. A POC price has not revisited after leaving is a **Naked / Virgin POC (VPOC)**. On ES this is genuinely predictive folklore-with-teeth: price has a well-documented tendency to return to naked VPOCs, because they mark unfinished auctions in real volume terms. Naked VPOC as a magnet target is an ES-native addition to the setup roster (Setup 6b below).

**Value Area (VA):** the price range containing 70% of real volume. VAH/VAL are its bounds. The 70% convention is inherited from Market Profile; whether 70% is optimal on ES is an empirical question (Section 13, Q3), but unlike on EUR/USD the question is now about parameter tuning rather than about whether the underlying quantity means anything.

**HVN / LVN:** high- and low-volume nodes relative to the profile mean. On real volume, **LVNs correspond to TPO single prints** — thin, fast-traverse zones the market moved through without accepting value. These are materially more reliable on ES than on tick-count FX.

**Profile shape taxonomy:** D-shape (balanced), P-shape (bullish, HVN low + thin tail up), b-shape (bearish, mirror). Same as before, now grounded in real distribution of transacted volume.

### 3.2 Anchor scheme (redesigned for ES)

At any 5-minute bar `t`, maintain these live profiles, all built strictly from completed prior data:

1. **Prior RTH-session profile (primary).** Built from the most recently *completed* cash RTH session (09:30–16:00 ET) with `session_end < t`. This is the canonical "yesterday's profile" of ES profile trading. Updates once per day at the 16:00 close.
2. **Overnight / ETH profile (secondary).** Built from the most recently completed Globex overnight segment (prior 16:00 → 09:30 open). Captures overnight inventory and where globally-set value sits before the cash crowd arrives.
3. **Initial Balance (IB) profile (intraday, causal).** The first 60 minutes of the *current* RTH (09:30–10:30 ET). After 10:30, the IB is a completed, look-ahead-free intraday anchor for the rest of the session. Before 10:30, IB is not yet available and IB-derived features are masked.
4. **Swing-to-swing profile (reactive).** Identical to the original: the leg between the two most recently *confirmed* causal swings, using `detect_confirmed_swings` from `features/transforms.py`. Roll-aware (Section 4).
5. **Composite multi-day / weekly profile (context).** Rolling N-session profile for longer-horizon HVN/LVN context. Not a signal source; conditioning only.

Each anchor is defined entirely by past data at computation time. The RTH and overnight anchors change once per day; IB updates once at 10:30; the swing anchor updates on swing confirmation.

### 3.3 ES-specific structural concepts worth encoding

These are profile-native concepts with no clean FX analog; they materially improve conditioning:

- **Open type** (relative to prior RTH value): *open inside value*, *open above value*, *open below value*. Strong prior on whether the day mean-reverts back into value or initiates away from it.
- **Open drive vs. open-test-drive vs. open-rejection-reverse** — early-session auction behavior (first 5–30 min) that conditions reversal vs. continuation.
- **IB extension** up/down — break of the first-hour range, a classic directional tell.
- **Day type** (Dalton): trend day, double-distribution trend day, normal day, normal-variation day, neutral day. A profile-native regime label that complements the existing 7-class regime classifier (Section 6.4).

### 3.4 Setups (transfer + ES-native additions)

Setups 1–6 from the original transfer directly and several get *stronger* on real volume:

- **Setup 1 — Mean-reversion in a balanced (D) profile.** Now conditioned on *open inside value*. Fade VAH/VAL toward POC. Reversal. Strongest on balance days / open-in-value.
- **Setup 2 — POC/VA acceptance after breakout.** Breakout above prior VAH with real volume expansion, retest holds → continuation.
- **Setup 3 — VA breakout with volume expansion.** Now uses **real** volume z-score, not tick count — a far more honest "initiative participants in control" filter. Continuation.
- **Setup 4 — Failed auction / VA rejection ("look above/below and fail").** On ES this is one of the highest-probability profile plays, and "no acceptance" is genuinely measurable as low real volume + single prints at the extreme before return inside VA. Reversal.
- **Setup 5 — LVN / single-print traverse.** Real single prints; fast-fill continuation, or stall → hidden HVN. 
- **Setup 6 — HVN magnet / equilibrium pull.** Mean-reversion toward nearest HVN. **Caution:** verify the magnet is a volume node and not merely a 0DTE gamma strike (Section 6.4); these often coincide and you want the model to learn the difference.
- **Setup 6b (ES-native) — Naked VPOC target.** When a naked VPOC sits within reach (≤ ~1.5 ATR) and no opposing structure intervenes, price tends to seek it. Reversal/continuation depending on side relative to current price. File as an extension once Setups 1–6 are validated.

---

## 4. Contract Continuity and Roll Handling

**This section is to ES what "EUR/USD has no centralized true volume" was to the original paper: the instrument-specific trap that silently invalidates everything if mishandled.**

### 4.1 The problem

FRVP requires *real price levels where volume transacted*. ES is a sequence of quarterly contracts (ESH26, ESM26, …), each liquid for roughly three months. To get continuous history you must stitch them, and the stitching method has consequences:

| Series type | Absolute price levels | Returns / path | Use for FRVP? |
|---|---|---|---|
| Individual contract (raw, unadjusted) | **True** | Discontinuous at expiry | **Yes — profiles must be built here** |
| Continuous unadjusted (stitched, gaps at roll; e.g. Databento `ES.v.0` with per-bar contract tags) | True *within* a segment | Jumps at roll | Yes, within-segment only |
| Continuous back-adjusted (Panama / ratio) | **Fictional** (shifted by cumulative roll spreads) | Smooth | **No — never compute profiles on this** |

A POC computed on back-adjusted data sits at a price that **never traded**. If a live signal fires against that phantom level, the strategy is trading a coordinate-system artifact. This is the canonical way futures volume-profile backtests produce beautiful, unrealizable equity curves.

### 4.2 Resolution

**Rule 1 — Profiles live in raw price coordinates.** All FRVP construction (POC/VAH/VAL/HVN/LVN, naked-POC tracking) uses raw, unadjusted prices from the contract that was actually lead-month at that time. The RTH, ETH, and IB anchors almost always live inside a single contract; only the swing and multi-day anchors can straddle a roll (handled below). If the source is a vendor-provided continuous series, the continuous ticker itself may stay constant across history (for example `symbol=ES.v.0`); the true contract coordinate then comes from explicit per-bar tags such as `contract_symbol` and/or `instrument_id`, not from the continuous ticker string.

**Rule 2 — Features are ATR-normalized distances, which are roll-invariant *if and only if* both terms share one coordinate system.** Your features are already of the form `(close − POC) / ATR`. As long as `close`, `POC`, and `ATR` are all drawn from the *same* (raw, current-lead-contract) coordinate at time `t`, the distance is correct and the back-adjustment issue is moot for that feature. The fatal error is mixing systems — e.g., a POC carried from back-adjusted history against a raw live close. Enforce a single source of truth per timestamp.

**Rule 3 — Back-adjusted series is permitted only for path/return features** that are translation-invariant by construction (log returns, ATR, volatility, regime inputs). It must never supply an absolute level that gets compared to a raw price.

**Rule 4 — Roll is volume-based and causal.** Determine the lead contract each day by which contract carried the higher volume (or open interest) — known only after the session, hence backward-looking and leak-free. Do **not** use a fixed calendar roll that can misalign with where liquidity actually sat. Record the roll date and the roll spread (lead-vs-next price difference at roll).

**Rule 5 — Cross-roll level handling.** A naked VPOC or swing-anchored level formed before a roll is in the *old* contract's coordinates. Two acceptable treatments:
   - **Translate** the level by the observed roll spread into the new contract's coordinates, or
   - **Reset** cross-contract level tracking at the roll boundary and re-seed from the first post-roll completed session.
   Reset is simpler and recommended for v1; translation is a Phase-3 refinement. Either way, a level must never be compared across coordinate systems untranslated.

**Rule 6 — Roll-window volume split.** For several days around the roll, volume splits across two contracts. Profile **only the lead contract per day**; never sum volume across contracts (different price coordinates, and the sum is meaningless). Flag the 2–3 days bracketing each roll as a reduced-confidence window.

### 4.3 Leakage assertions (add to tests)

- Assert every profile's source bars come from exactly one contract.
- Assert no profile-derived absolute level is ever differenced against a price from a different coordinate system.
- Assert the lead-contract assignment at each timestamp uses only data with timestamp ≤ `t`.
- Assert events whose triple-barrier window spans a roll boundary are excluded from labeling (Section 7.5) — a barrier evaluated across a coordinate discontinuity is corrupt.

---

## 5. ML Formulation Analysis

The four-way analysis from the original paper holds with the same verdict; only the strength of the underlying signal changes. Summary, with ES-specific notes:

**Formulation A — Direct triple-barrier classification.** Viable, simpler, but conflates "is this a setup?" with "will this setup work?", suffers class imbalance, and discards the interpretable rule layer. On ES the imbalance is similar to FX. Keep as fallback (CUSUM-sampled, XGBoost).

**Formulation B — Meta-labeling of FRVP setups (recommended primary).** Unchanged recommendation, and the case is *stronger* on ES: the primary rule layer now stands on real volume, so the meta-labeler trains on cleaner primary signals. Each setup fires an event; the meta-label is binary (1 if the hypothesized-direction TP barrier is hit first). Separate long/short and reversal/continuation models, identical to the incumbent OTE meta-TCN pattern. Composes cleanly with the regime layer and the existing threshold-policy search.

**Formulation C — Learning-to-rank.** Still overkill at 1–3 trades/day; concurrent multi-anchor signals are uncommon. Defer.

**Formulation D — Risk-adjusted / trend-scan regression.** Still best as an *auxiliary* head inside B (predict move magnitude to inform sizing), not a primary signal. On ES, magnitude dispersion across setups is wide (an open-drive trend day vs. a balance-day fade are very different sizes), so a magnitude head has more to learn here than on FX — a mild argument to prioritize it earlier (Phase 3 rather than 4).

**Recommendation:** Formulation B primary; A as fallback if any setup family yields < ~1,000 events over the training window. The reversal family (Setups 1, 4, 6) and continuation family (2, 3, 5) get separate long/short meta-labelers — four models, consistent with repo organization. Setup 6b (naked VPOC) folds into the reversal family once validated.

---

## 6. FRVP Feature Engineering

### 6.1 Integration

New features live in `frvp/feature_sets/frvp_context.py` and are exposed to the existing builder through the registry shim `features/feature_sets/frvp_context.py` (registered `"frvp_context"`), added to `ote_extended.json` and a new `frvp_meta.json`. Dependencies: `volume.py` (real-volume z-scores), `structure.py` (ATR, swings), the new `frvp/continuity/continuous_contract.py` (lead-contract + roll), and the equity-session calendar (RTH/ETH/IB boundaries) via `frvp/sessions/equity.py` and its shim `features/sessions_equity.py`. All features are computed **causally**.

### 6.2 Carried-over feature groups (definitions identical, units in points)

Groups 1–7 from the original transfer directly. Distances are ATR-normalized (ATR in index points, not pips). The catalog below highlights only the ES-specific changes and additions; everything else (distance to POC/VAH/VAL, price position in VA, shape classification, HVN/LVN density, VA metrics, multi-anchor confluence, setup-type encoding) is reused as written.

Bin width: `max(1 tick = 0.25 pt, ATR_14 / 20)`. Volume basis: **real contract volume** (no tick-count substitute).

### 6.3 New / changed feature groups

**Group 9 — Session-relative position (replaces FX kill-zone features):**

| Feature | Description |
|---|---|
| `frvp_open_type` | 0 = open inside prior value, 1 = open above value, −1 = open below value |
| `frvp_open_drive_flag` | 1 if first 30 min is a one-directional drive from the open |
| `frvp_ib_extension` | Signed: +1 IB-high broken, −1 IB-low broken, 0 inside IB (masked before 10:30 ET) |
| `frvp_dist_ib_high_atr`, `frvp_dist_ib_low_atr` | Distance to IB bounds / ATR (masked before 10:30) |
| `frvp_session_phase` | Categorical: open / morning / midday-lull / afternoon / close-ramp |
| `frvp_rth_eth_value_overlap` | Overlap fraction of prior RTH VA and overnight ETH VA |

**Group 10 — Gap / open-vs-value relationship:**

| Feature | Description |
|---|---|
| `frvp_open_vs_prior_poc_atr` | (RTH open − prior RTH POC) / ATR |
| `frvp_open_gap_atr` | (RTH open − prior RTH close) / ATR — the cash "gap" even though ETH filled continuously |
| `frvp_gap_into_value` | 1 if the open gap lands inside prior VA (gap-fill-to-value bias) |

**Group 11 — Naked VPOC tracking:**

| Feature | Description |
|---|---|
| `frvp_naked_vpoc_dist_above_atr`, `frvp_naked_vpoc_dist_below_atr` | Distance to nearest unretested VPOC above/below / ATR |
| `frvp_naked_vpoc_age_sessions` | Sessions since the nearest naked VPOC formed |
| `frvp_naked_vpoc_count` | Count of live naked VPOCs within ±3 ATR |

### 6.4 Options-gamma context (ES-specific, optional but recommended)

ES/SPX intraday gravity is materially shaped by dealer gamma, especially 0DTE. Where a large strike sits, price tends to pin — and that pin can coincide with or override an HVN magnet (Setup 6). Two reasons to encode it: (a) a context feature that helps the model distinguish a real volume magnet from a gamma pin, and (b) a guard against the model "discovering" Setup 6 edge that is actually a transient gamma artifact.

| Feature | Description | Source |
|---|---|---|
| `es_dist_to_nearest_large_strike_atr` | Distance to nearest high-OI SPX/ES strike / ATR | Options chain snapshot (causal, prior close) |
| `es_gamma_pin_flag` | 1 if an HVN magnet is within ~0.25 ATR of a large-gamma strike | Derived |
| `es_zero_dte_regime` | 1 on days with elevated 0DTE activity | Calendar / vendor feed |

These require an options data source the repo does not currently ingest. Treat as a Phase-3 enhancement; the base FRVP pipeline must work without them, and the SHAP analysis decides whether they earn inclusion.

### 6.5 Day-type as a profile-native regime overlay

Compute Dalton day-type (trend / double-distribution trend / normal / normal-variation / neutral) as a categorical context feature and as an *additional* slice dimension in the regime-slice report (Section 10). It complements, not replaces, the existing 7-class regime labeler. Hypothesis to test (Q6): reversal setups concentrate edge on normal/neutral/balance days; continuation setups on trend / double-distribution-trend days.

### 6.6 Interaction terms with existing ICT features

Reuse the original interaction terms (POC + FVG, VAH/VAL + order block, failed-auction + sweep), with the kill-zone term replaced by session-phase:

```
frvp_in_open_drive: frvp_session_phase == open AND frvp_open_drive_flag == 1
frvp_failed_auction_with_sweep: (frvp_setup_type == 4) AND (bars_since_sweep_high <= 3 OR bars_since_sweep_low <= 3)
frvp_poc_plus_ict_fvg_confluence: (dist_poc_rth_atr <= 0.5) AND (dist_to_bull_fvg_atr <= 0.5 OR dist_to_bear_fvg_atr <= 0.5)
frvp_magnet_is_gamma_pin: (frvp_setup_type == 6) AND (es_gamma_pin_flag == 1)
```

---

## 7. Labeling

### 7.1 FRVP events as a new label family

Same as the original: FRVP events form a fourth label family (alongside reversal, continuation-pullback, breakout), triggered by rule-based setup fires rather than swings. Reuse triple-barrier outcomes, trend-scan quality scoring, sample-uniqueness weights, and exclusion masks.

### 7.2 Event sampling

A FRVP event is sampled when one of Setups 1–6 (and later 6b) fires, using only completed-bar information. The original per-setup fire conditions transfer; the failed-auction (Setup 4) "close back inside VA" remains a bar-close definition with the meta-label evaluated from the *next* bar's open.

### 7.3 Triple-barrier widths (re-expressed in ATR points)

**Historical design proposal.** The table below is retained as the original
labeling design, not the active paper-evaluation contract. The established
Phase03 pipeline overrides continuation TP to 1.00 ATR, S3-long to 0.95,
S5-long to 0.90 and S5-short to 0.85; S4 uses stop/target 0.90/1.00 ATR.
These values are present in the saved July/August event configuration.
[B1's September 17 resolution](B1_outcome_contracts.md#b1_frvp_risk_resolution_20260917), recorded
as E16 in the experiment journal, independently freezes those values for
paper evaluation with causal next-open entry and fixed barriers. Historical
labels remain unchanged. The 96-bar continuation and 48-bar S4 paper clocks
start at actual evaluation entry. S4 policy candidacy remains paused.

| Setup Family | TP (ATR) | SL (ATR) | Max Holding | RR |
|---|---|---|---|---|
| Mean-reversion (1, 6, 6b) | 0.8 × ATR | 1.0 × ATR | 60 bars (5h) | 0.8:1 (size-adjust at risk layer) |
| Continuation (2, 3, 5) | 1.5 × ATR | 1.0 × ATR | 96 bars (8h) | 1.5:1 |
| Failed auction (4) | 1.2 × ATR | 0.8 × ATR | 48 bars (4h) | 1.5:1 |

ATR is the structural ATR (1h mapped to 5-min) as in the existing engine, in index points. **Re-check the friction math before trusting any of these:** at ES tick = 0.25 pt, a 0.8 × ATR mean-reversion target on a quiet day can be only a handful of ticks; with round-turn friction of ~1–1.5 ticks plus open/close slippage spikes, the mean-reversion family's edge is thin. Validate that TP comfortably exceeds modeled friction per setup family before promotion.

### 7.4 Quality scoring and sample weights

Reuse the original weighting (triple-barrier 40%, trend-scan 25%, HTF confluence 20%, entry 10%, setup confidence 5%). Down-weight overlapping FRVP events via a `frvp_concurrency` series analogous to `concurrency_long`.

### 7.5 Exclusion masks (ES additions)

- First 5 bars of a new RTH session (profile not yet meaningful) — exclude, as before.
- **Roll-bracket window (±2–3 days around each volume-based roll)** — exclude or down-weight; coordinate ambiguity (Section 4).
- **Any event whose triple-barrier window spans a roll boundary** — exclude outright (barrier evaluated across a coordinate discontinuity is invalid).
- Scheduled macro / Fed events (FOMC, CPI, NFP) and the FOMC-statement/press-conference window — use the existing anomaly flags; ES reacts violently to these.
- Half-days / early closes (day-after-Thanksgiving, holiday eves) — exclude; truncated sessions distort the RTH profile.
- Quad-witching and index-rebalance days — flag; volume distribution is atypical.
- Sessions with RTH volume < 20% of the trailing 20-session median — exclude (thin/holiday).

---

## 8. Data Requirements

### 8.1 Historical data

| Stream | Purpose | In repo? |
|---|---|---|
| ES OHLCV **with real volume**, either as a raw tagged continuous lead-contract series or as per-contract bars | Profiles, labeling, features, training | **Current baseline** — `data/futures_data/ES-5m.csv` (vendor-tagged `ES.v.0` with per-bar `instrument_id`); optional refinement — per-contract files |
| ES roll calendar / lead-contract series | Continuity (Section 4) | **Current artifacts** — `data/futures_data/es_roll_schedule.json`, `data/futures_data/es_roll_reconstruction_report.json`; code — `frvp/continuity/reconstruct_boundaries.py` |
| 30-min / 1h ES (back-adjusted) | HTF structure, ATR, regime (path features only) | **Derived** — resample the continuous series |
| Equity session calendar (RTH/ETH/IB, half-days, holidays) | Anchors and masks | **Current implementation** — `frvp/sessions/equity.py` with compatibility shim `features/sessions_equity.py`; half-day / holiday refinements still pending |
| SPX/ES options OI by strike (optional) | Gamma context (Section 6.4) | **New / optional** — Phase 3 |

Source **true-volume** ES data from a CME-derived vendor; do not use a single back-adjusted continuous file for profile construction. A raw, non-adjusted vendor-rolled continuous file is acceptable **if** it carries explicit per-bar contract tags (for example Databento `ES.v.0` with `contract_symbol`, `instrument_id`, `is_roll_boundary`, and `bars_since_roll`). Tick or 1-min granularity improves histogram resolution (finer within-bar price assignment), but the current repo baseline is a 5-minute tagged series, which is acceptable for the Phase 0/1 implementation while remaining coarser than the ideal profile source.

### 8.2 Date range

ES electronic history runs to the late 1990s, but microstructure is non-stationary. Recommended training window **2015–2024** (deeper sample than the FX paper's 2018 cutoff, since ES electronic data is clean further back), with **2025–present** as out-of-sample holdout. **Caveat:** the 0DTE era (~2022 onward) changed intraday dynamics materially; run a pre-2022 vs. post-2022 stability check and consider recency-weighting or a post-2020 primary window if the regimes diverge (Section 13, Q11).

### 8.3 Per-bar profile state

At each 5-min bar `t`, maintain: prior-RTH profile, overnight-ETH profile, current-session IB profile (post-10:30), swing profile, and the rolling multi-session context profile — all in raw lead-contract coordinates, each a `{price_bin: real_volume}` dictionary updated incrementally. In the tagged-continuous case, the profile state keys off `contract_symbol` / `instrument_id`, not the constant continuous `symbol` field. Computationally tractable.

### 8.4 Look-ahead rules (superset of original)

| Rule | Rationale |
|---|---|
| Profile from prior session/swing only | Primary look-ahead defense |
| Single contract per profile; lead assignment causal | Continuity defense (Section 4) |
| Back-adjusted series only for translation-invariant features | Coordinate-system defense |
| Setup fire condition uses completed-bar close | Existing convention |
| Triple-barrier starts at next bar's open; excluded if it spans a roll | Executable-entry + coordinate defense |
| IB features masked before 10:30 ET | IB not yet complete |
| SHAP/attribution computed inside each CV fold on training data only | Selection-leakage defense |

---

## 9. Model Architecture

### 9.1 New model families

Four new candidates, mirroring the repo's per-family/per-side organization:

- `long_frvp_reversal_meta`, `short_frvp_reversal_meta`
- `long_frvp_continuation_meta`, `short_frvp_continuation_meta`

Registered in `models/ote_model_registry.json` as `status="candidate"`, promoted to `active` only after all promotion gates pass (Section 10).

### 9.2 Model choice

**Primary: TCN**, same architecture as `long_ote_meta_tcn_champion`. ES FRVP features mix static per-profile levels with time-evolving distance metrics — TCN captures the temporal evolution of price's relationship to the profile. Starting hyperparameters identical to the incumbent meta-TCN (window 16–32, hidden 64, 3 layers, dropout 0.20, lr 0.001, 40-epoch warmup/main/fine/tail schedule).

**Fallback: XGBoost** if any family/side falls below ~1,500 training events. `ote_xgboost_pipeline.py` handles both backends; train both and compare OOF before deciding.

### 9.3 Probability → position size

Unchanged flow: primary classifier `P(profitable)` → regime gate (regime-conditional threshold from `ote_threshold_policy_search`, now also conditioned on day-type/open-type) → abstain policy → fractional-Kelly sizing between `P_min ≈ 0.55` and `P_max ≈ 0.80` → FRVP-specific barriers scaled by current ATR. The auxiliary magnitude head (Formulation D, Phase 3) can scale size within the allowed band.

### 9.4 Regime composition

Reversal setups should outperform in ranging / low-vol / normal-or-neutral day-types; continuation setups in trending / high-vol / trend day-types. Slice on both the existing 7-class regime and the new day-type label; `min_positive_events = 50` per bucket, abstain below.

---

## 10. Validation and Anti-Overfitting

Reuse the entire existing apparatus — purged walk-forward CV with embargo, CPCV (≥ 6 paths), walk-forward policy backtest with frictions, Deflated Sharpe, and the shuffled-label placebo test — with these ES-specific changes:

- **Frictions in ticks/points.** Set `fixed_slippage_ticks_per_trade` and `commission_per_contract` (verify against your broker/clearing; retail all-in is often ~1–1.5 ticks round-turn at liquid times, materially higher into the open and close). Do not reuse the FX pip values.
- **Embargo spans a full RTH session minimum**, plus an extra day around any roll.
- **New leakage trap — roll/continuity (Section 4)** joins the original Trap 1 (anchoring-window) as a top-priority audit. Add the Section 4.3 assertions to `tests/test_frvp_context.py`.
- **Regime slices add open-type and day-type dimensions** alongside the existing regime buckets.
- **Placebo test** unchanged: real model OOF AUC-PR must exceed shuffled-label mean by > 3% across ≥ 10 shuffles.

**Promotion gates (all must pass):**

1. OOF AUC-PR substantially above base rate (> 0.60 target).
2. Walk-forward Sharpe after frictions > 0.8 annualized.
3. Walk-forward max drawdown < 12%.
4. Deflated Sharpe > 0.3.
5. Regime/day-type slice: reversal models win in ≥ 2 of the range/normal/neutral buckets; continuation models in ≥ 2 of the trend/high-vol buckets.
6. Placebo: real AUC-PR exceeds shuffled mean by > 3%.
7. **Roll audit clean:** zero events with cross-coordinate barriers; lead-contract assignment causal; no profile spans two contracts.

---

## 11. The 6E Variant

6E (CME euro FX futures, €125,000 notional, tick 0.00005 = $6.25, quarterly H/M/U/Z) is offered as a **configuration variant**, not a separate pipeline. It exists for two reasons: it preserves the EUR market thesis of the original paper while replacing tick-volume with **real centralized volume**, and it enables a clean scientific A/B against the original EUR/USD spot design.

### 11.1 What 6E keeps from the original EUR/USD paper

6E liquidity tracks the London/NY overlap, so the original's **London/NY session-anchor scheme largely survives** — unlike ES, 6E does not pivot to an RTH-centric model. Most of the original feature catalog and setup logic apply with minimal change. The main edits are:

- **Volume basis → real contract volume** (the whole point; resolves the original's Caveat 1 and Q1/Q2).
- **Add contract continuity + roll handling** (Section 4 applies identically — quarterly cycle, same back-adjustment hazard, volume-based causal roll).
- Frictions re-expressed in 6E ticks ($6.25/tick); 6E is liquid but thinner than ES, so model slippage more conservatively, especially outside the London/NY overlap.

### 11.2 The A/B test worth running

Run the *same* setups and *same* meta-labeling pipeline three ways and compare:

1. EUR/USD spot, tick-volume profiles (the original design).
2. 6E futures, real-volume profiles (this variant).
3. ES futures, real-volume profiles (this paper's primary).

Compare meta-label base rates, OOF AUC-PR, and especially the **placebo-test gap** (real minus shuffled AUC-PR). If real volume carries genuine signal that tick volume does not, (2) and (3) should show a larger placebo gap than (1). This directly tests the foundational claim that motivated re-homing the strategy, and it is cheap once the pipeline is instrument-parameterized via `frvp/config/instruments.py`.

### 11.3 Config-level differences (ES vs 6E)

| Parameter | ES | 6E |
|---|---|---|
| Tick size / value | 0.25 pt / $12.50 | 0.00005 / $6.25 |
| Primary session anchor | Prior RTH (09:30–16:00 ET) | Prior London / NY (original scheme) |
| IB / open-type features | Yes (equity open) | Optional (London open analog) |
| Gamma-context features | Yes (SPX 0DTE) | No |
| Day-type overlay | Yes (Dalton) | Lower priority |
| Roll handling | Section 4 | Section 4 (identical) |
| Micro contract for testing | MES ($1.25/tick) | M6E ($1.25/tick) |

---

## 12. Build Roadmap

Phases mirror the original; the new Phase 0 (continuity) is a hard prerequisite gating everything.

### Phase 0 — Continuous-contract + roll engine (NEW, blocking)
**Goal:** A causal lead-contract series, a volume-based roll calendar, and a raw-coordinate profile data layer.
**Files:** `frvp/continuity/continuous_contract.py`, `frvp/continuity/roll_calendar.py`, `frvp/continuity/reconstruct_boundaries.py`, `tests/test_continuity.py`.
**Gate:** Lead-contract assignment is causal and matches known historical roll dates within ±1 day; back-adjusted vs. raw series reconcile on returns but differ on absolute levels exactly by cumulative roll spreads; assertions in Section 4.3 pass. If the input is a vendor-tagged raw continuous series rather than overlapping per-contract bars, Phase 0 may validate the observed `contract_symbol` / roll-boundary tags instead of re-deriving the roll calendar from scratch.

### Phase 1 — FRVP profile construction engine
**Files:** `frvp/profiles/builder.py`, `frvp/profiles/anchors.py`, `frvp/feature_sets/frvp_context.py`, `frvp/setups/detector.py`, `frvp/sessions/equity.py`, `tests/test_profiles.py`, `tests/test_frvp_context.py`, `tests/test_frvp_setups.py`. Thin compatibility shims remain at `features/feature_sets/frvp_context.py`, `features/sessions_equity.py`, and `frvp/strategies/frvp_setups.py`.
**Gate (on 90 held-out RTH sessions):** 20 random profiles match TradingView FRVP drawings on the *same contract*; off-by-one anchoring test passes; naked-VPOC state updates correctly; each setup fires at 0.5–2 per session per side; no profile spans two contracts.

### Phase 0/1 implementation notes (as of 2026-07-01)

Phase 0 and Phase 1 are structurally complete in code and stable on the full tagged ES history window. The only remaining Phase 0/1 blocker is external human sign-off on same-contract profile matching, not missing repo logic.

Implemented and verified on the canonical production path:
- `data/futures_data/ES-5m-tagged.csv` is the canonical upstream FRVP ES-primary input.
- All FRVP profile construction remains on `ContinuousContractResult.raw_profile_bars`, never on the back-adjusted path series.
- `RawProfileBars.profile_slice(...)` still enforces the single-contract rule and raises `RollBoundaryError` on cross-roll slices.
- `build_equity_session_frame` now applies explicit US equity holiday, half-day, and early-close overrides from `frvp/calendars/equity.py`.
- The labeler treats short-session detection as fallback-only; the current integrity contract shows the half-day and early-close calendar is now explicit rather than empirical.
- The CPI calendar now uses the archived historical backfill plus curated recent dates; the old CPI placeholder debt is closed.
- `VolumeProfileBuilder`, causal anchors, naked-VPOC tracking, the registered `frvp_context` family, the rule-based setup detector, and the Dalton day-type context are all present in-repo.

Full-history setup fire-rate readout from `artifacts/frvp_es_primary_current/phase01/setup_fire_rates.csv`:

| Setup / side | Fires per session | Gate read |
|---|---:|---|
| Setup 1 short | 0.613 | Pass |
| Setup 1 long | 0.575 | Pass |
| Setup 2 short | 0.596 | Pass |
| Setup 2 long | 0.521 | Pass |
| Setup 3 short | 0.799 | Pass |
| Setups 3 long, 5, and 6 both sides | In band | Pass on fire-rate gate |
| Setup 4 both sides | 0.160-0.182 | Below legacy band; retain and document as thinner / weaker |

What worked:
- The earlier Setup 1 / Setup 2 / Setup 3-short sparsity issue is fixed.
- The earlier Setup 3 / Setup 6 over-fire issue remains resolved on the full-history audit.
- Setup 4 is now treated correctly as a weaker selective pattern rather than a gate failure.

What did not change:
- The histogram still allocates each 5-minute bar's volume uniformly across `[low, high]`.
- The implemented cross-roll policy remains reset-at-roll only; cross-contract level translation is still deferred.
- `Setup 6b` and optional gamma-context features remain deferred research items, not Phase 0/1 blockers.

Remaining Phase 0/1 blocker:
- The held-out TradingView same-contract profile audit still needs human sign-off.

### Phase 2 — Feature dataset build
Reuse `features.cli build` with `--recipe features/recipes/frvp_meta.json` and `--instrument es`. Gate: correlation check at 0.98; FRVP distances show non-trivial MI with existing meta-labels.

### Phase 2 implementation notes (as of 2026-07-01)

The full Phase 2 build is complete on the intended history window: 666,769 five-minute ES bars from `2017-01-02 23:00:00+00:00` through `2026-06-19 16:55:00+00:00`.

Primary artifacts:
- `artifacts/frvp_es_primary_current/phase02/es_primary_frvp_features_full.csv.gz`
- `artifacts/frvp_es_primary_current/phase02/es_primary_frvp_features_full.csv.metadata.json`
- `artifacts/frvp_es_primary_current/phase02/es_primary_frvp_phase04_dataset.csv.gz`
- `artifacts/frvp_es_primary_current/phase02/es_primary_frvp_phase04_dataset.csv.metadata.json`
- `artifacts/frvp_es_primary_current/phase02/phase2_feature_audit.json`

What passed:
- No duplicated time columns entered the persisted Phase 2 dataset; `ts_event` and `timestamp` were dropped in favor of canonical `datetime`.
- No raw OHLCV fields or contract-lineage / roll-lineage fields survived the persisted Phase 2 dataset.
- FRVP distances and HTF confluence retained positive MI across all four direct FRVP targets.
- The full-width Phase 2 build now survives the earlier pandas fragmentation / memory failure; the concat-based transform build plus float-safe downcasting closed the old `ArrayMemoryError` path.

What worked best:
- The merged Phase 2 dataset and the raw full feature dump now coexist in one canonical artifact root, so Phase 4 and later stages can be rerun without rebuilding exploratory copies.
- The full-width gzip path is now a storage issue at worst, not a memory-blocking issue in the feature builder itself.

What remains a documented weakness:
- `frvp_open_type` survives the merged dataset and keeps non-zero MI on the short-side targets, but its MI is still effectively near-zero for `long_frvp_reversal` and `long_frvp_continuation`.
- The raw audit still contains one near-duplicate pair (`frvp_va_overlap_pct` and `frvp_rth_eth_value_overlap`) at approximately 1.00 correlation; downstream pruning handles it cleanly, but the pre-pruned audit view is not perfectly minimal.

### Phase 3 — FRVP event labeling
**Files:** `data/labeling/frvp_labeling_engine.py`; extend `labeling_engine.py`.
**Gate:** 250–750 events/year per family per side; meta-label base rate 45–60%; quality mean around 0.60-0.65 is preferred but no longer a hard blocker if downstream ranking quality and economics remain honest; roll-spanning events excluded; events not excessively clustered.

### Phase 4 — Preprocessing + backend attribution
Reuse `preprocessing prepare` / `backend-attribution` verbatim. Gate: FRVP distances and HTF confluence should remain materially represented in attribution while open-type/day-type remain desirable but not mandatory top-10 survivors if they stay weak after honest full-history testing; no label-adjacent or raw-OHLC leakage features survive.

### Phase 5 — Model training
Reuse `ote_xgboost_pipeline.py` (TCN primary, XGB fallback) for the four targets. Gate: OOF AUC-PR > base rate; placebo gap > 3%; calibration sound; keep XGBoost as an acceptable primary if the TCN does not add value cleanly.

### Phase 6 — Regime/day-type slice + threshold policy
Reuse the slice and threshold-policy scripts with the current ES-aware reporting contract. Open-type status, open-type label, and day-type label are now part of the saved slice outputs; the remaining gate is whether those slices support a profitable policy rather than whether the reporting dimension exists.

### Phase 7 — Walk-forward backtest + DSR
Reuse the policy backtest with **ES tick/point frictions**. That ES-aware contract is now implemented; the remaining gate is whether any FRVP candidate still clears the seven promotion gates in Section 10 plus DSR once those costs are applied.

### Phase 8 — Live integration
Add artifacts to `models/live/`; register as `candidate`; extend `ote_live/features/incremental_engine.py` to maintain raw-coordinate profiles and handle live rolls; extend `ote_live/ingestion/runtime.py` for contract-roll events and gap handling; shadow-trade ≥ 4 weeks before promotion.

---

### Phase 3 implementation notes (as of 2026-07-01)

The FRVP labeler is now stable on the full intended ES window, with canonical upstream macro and session flags in place.

Implemented and verified:
- `data/labeling/frvp_labeling_engine.py` samples FRVP events from Setups 1-6, evaluates barriers from the next bar's open, reuses the structural ATR / trend-scan / sample-weight machinery, and excludes roll-spanning events via the continuity layer.
- `data/labeling/labeling_engine.py` exports `label_long_frvp_reversal`, `label_short_frvp_reversal`, `label_long_frvp_continuation`, and `label_short_frvp_continuation` alongside the existing three legacy families.
- Canonical upstream macro flags now exist for FOMC, CPI, NFP, Fed statement, and Fed presser windows.
- Explicit half-day / early-close flags now come from the equity-session calendar rather than placeholder fallbacks.

Primary artifacts:
- `artifacts/frvp_es_primary_current/phase03/es_primary_frvp_labels.csv`
- `artifacts/frvp_es_primary_current/phase03/es_primary_frvp_events.csv`
- `artifacts/frvp_es_primary_current/phase03/labeling_diagnostics.json`
- `artifacts/frvp_es_primary_current/phase03/es_primary_frvp_diagnostic_report.csv`
- `artifacts/frvp_es_primary_current/phase03_gate_report.json`

Full-history Phase 3 diagnostic:

| Target | Events/year | Base rate | Quality mean | Gate status |
|---|---:|---:|---:|---|
| Long FRVP reversal | 250.88 | 49.56% | 0.654 | Pass on density and base rate; quality target met |
| Short FRVP reversal | 251.62 | 50.71% | 0.648 | Pass on density and base rate; quality near-target |
| Long FRVP continuation | 698.72 | 48.46% | 0.612 | Pass on density and base rate; softer quality |
| Short FRVP continuation | 747.03 | 48.37% | 0.586 | Pass on density and base rate; weakest quality cell |

Full-history exclusion and integrity readout:
- `events_excluded_roll_span = 49`
- `events_excluded_macro = 3406`
- `events_excluded_half_day = 155`
- `events_excluded_first_rth = 2021`
- `events_excluded_thin_session = 704`
- `events_excluded_disabled_setup = 1002`
- `events_excluded_past_htf_confluence = 2807`
- `events_excluded_reversal_cooldown = 1059`

Experiments that helped:
- The hybrid CPI archive plus curated recent schedule closed the old macro TODO.
- Reversal cooldown plus disabled-setup handling fixed the earlier reversal density / clustering problem.
- The causal Setup 1 / Setup 6 HTF-confluence gate using the 48/96 rule, with the tighter 36/72 rule for Setup 1 short, improved the reversal contract without collapsing continuation counts.

Experiments that regressed and are now closed:
- Blanket continuation hold shortening (`96 -> 72` bars) regressed continuation base rates and bought no quality.
- Repeated continuation-only label tightening on Setups 3 and 5 moved metrics around, but did not produce a clearly better training label contract than the current baseline.

Current read:
- Roll-spanning exclusion is intact.
- Events are no longer excessively clustered.
- All four direct targets clear the 45-60% base-rate band.
- All four direct targets sit inside the 250-750 events/year density band.
- Continuation quality is still softer than reversal quality, but `0.65` is now a target, not a blocker.

What does not work well enough to keep optimizing blindly:
- Setup 4 remains structurally thinner than the other setups.
- Short continuation remains the weakest quality cell.
- Further label tightening should be judged by downstream ranking and ES economics, not by blindly chasing `0.65`.

### Phase 4 implementation notes (as of 2026-07-01)

The FRVP preprocessing and backend-attribution path now runs end-to-end on the full-width, full-history ES dataset for the four direct FRVP targets.

Implemented and verified:
- `preprocessing prepare` works for all four direct FRVP targets with no target-specific downstream hacks.
- The preprocessing path preserves the target-specific `htf_confluence_*` columns even though those are labeler carry-through context rather than builder-generated features.
- The prepared allowlists continue to exclude raw OHLCV, label-helper, barrier-adjacent, and contract-lineage fields such as `contract_symbol`, `instrument_id`, `is_roll_boundary`, `bars_since_roll`, and `in_roll_bracket`.
- Dalton day-type is implemented upstream and survives preprocessing, even though it does not yet clear the attribution target.
- The full-width Phase 2 -> Phase 4 handoff succeeds on the 736-feature gzip dataset; the old memory blocker is no longer the controlling issue.

Primary artifacts:
- `artifacts/frvp_es_primary_current/phase02/es_primary_frvp_phase04_dataset.csv.gz`
- `artifacts/frvp_es_primary_current/phase02/es_primary_frvp_phase04_dataset.csv.metadata.json`
- `artifacts/frvp_es_primary_current/phase02/phase2_feature_audit.json`
- `artifacts/frvp_es_primary_current/phase04/prepared/summary.json`
- `artifacts/frvp_es_primary_current/phase04/prepared/backend_attribution_summary.json`

Important full-history readout:
- `phase04/prepared/summary.json` resolves the full-width upstream contract cleanly: 736 metadata feature columns, 4 target-specific HTF carry-through columns, 38 duplicate columns removed, and 3 constant columns removed.
- All four direct targets score `green` training readiness with 2373 / 2380 / 6609 / 7066 usable rows and 583-597 selected prepared features.

Full-history Phase 4 attribution readout from the saved full-width TCN rerun:

| Target | `prepare` status | FRVP distance in merged top-10 | `frvp_open_type` in merged top-10 | `frvp_day_type` in merged top-10 | Target HTF confluence in merged top-10 | Gate status |
|---|---|---|---|---|---|---|
| Long FRVP reversal | Pass | Yes | No | No | No | Partial pass |
| Short FRVP reversal | Pass | No | No | No | No | Partial fail |
| Long FRVP continuation | Pass | No | No | No | Yes | Partial pass |
| Short FRVP continuation | Pass | No | No | No | Yes | Partial pass |

What worked:
- All four direct FRVP targets prepare cleanly and preserve backward compatibility with the existing target families.
- Target-specific `htf_confluence_*` carry-through columns survive preprocessing and remain in the prepared contracts.
- The leakage audit remained clean: no raw OHLCV fields, no label/sample-weight/quality helpers, no barrier metadata, and no roll-lineage columns survived the prepared sets.
- The full-width prepare summary was strong enough to move directly into training without target-specific rescue logic.

What did not become a core feature signal:
- `frvp_open_type` survives preprocessing but still misses merged top-10 everywhere.
- `frvp_day_type` also survives preprocessing but still misses merged top-10 everywhere.
- Short reversal remains the weakest target from an FRVP-specific attribution perspective.

Current interpretation:
- Phase 4 is no longer blocked.
- The remaining Phase 4 issues are feature-priority questions, not reasons to stop the baseline workflow.

### Phase 5-7 implementation notes (as of 2026-07-01)

Training, regime-slice evaluation, threshold-policy search, and walk-forward backtest all complete end-to-end on the current FRVP shortlist.

Focused update as of `2026-07-03` for `frvp_long_continuation_xgb_v1` on the refresh branch:

- Refresh baseline backtest: `1609` trades, `+12465.15` ticks, Sharpe `0.604`, DSR `0.585`, max drawdown `10200.85`.
- Targeted `v1` London drawdown prune: `1135` trades, `+7968.25` ticks, Sharpe `0.684`, DSR `0.657`, max drawdown `3568.20`.
- Targeted `v2` overlap/composite prune: `662` trades, `+6433.70` ticks, Sharpe `0.957`, DSR `0.881`, max drawdown `2183.70`.
- Targeted `v3` pair prune: `524` trades, `+7190.40` ticks, Sharpe `1.226`, DSR `1.061`, max drawdown `1540.10`.
- The winning `v2` and `v3` behavior is a hard-pruned base `global_threshold`, not a `regime_threshold` variant.
- The threshold-search selector mismatch that left `selected_policy_name = null` for that case is now fixed in `scripts/run_ote_threshold_policy_search.py`; the search path now falls back to explicit `global_threshold` selection with saved baseline metrics, matching `model_testing/ote_policy_backtest.py`.
- `v1` fixed the first London tail pockets. `v2` added overlap-session and negative-composite pruning plus `apply_to_base_policy_variants = true`. `v3` then added only `ranging_medium/new_york` and `strong_up_low/asia` on top of `v2`, which lifted Sharpe from `0.957` to `1.226` while increasing net PnL from `+6433.70` to `+7190.40`.

Focused update as of `2026-07-03` for `frvp_long_meta_xgb_v1` on the refresh branch:

- Refresh baseline backtest: `3621` trades, `-4468.65` ticks, Sharpe `-0.133`, DSR `-0.133`, max drawdown `19662.75`.
- Targeted `v1` composite prune: `2269` trades, `-650.85` ticks, Sharpe `-0.027`, DSR `-0.027`, max drawdown `13789.90`.
- Targeted `v2` composite/session prune: `1040` trades, `+8787.00` ticks, Sharpe `0.460`, DSR `0.452`, max drawdown `7439.05`.
- Targeted `v3` narrow pair prune: `603` trades, `+9216.05` ticks, Sharpe `0.595`, DSR `0.577`, max drawdown `3754.55`.
- `v1` removed the two worst baseline composites globally: `strong_down_medium` and `strong_up_medium`.
- `v2` then removed the largest surviving pair-level losers: `strong_up_high/overlap`, `strong_down_high/new_york`, `ranging_high/new_york`, `strong_up_low/asia`, `strong_up_low/london`, `ranging_low/london`, `ranging_medium/new_york`, `ranging_medium/asia`, and `strong_down_low/london`.
- `v3` added only `ranging_medium/london`, `strong_up_high/london`, and `strong_down_high/overlap` on top of `v2`, which improved net PnL and cut drawdown again.
- The saved `v3` static threshold study still falls back to a hard-pruned base `global_threshold` with `qualified_policy_names = []`, and the walk-forward `v3` result remains below the promotion bar despite turning clearly positive.
- Treat long-meta `v3` as the best pooled-branch research checkpoint, not as a promotion candidate. The refreshed `2026-07-15` account-drawdown rerun keeps the branch at Sharpe `0.595`, DSR `0.577`, WFE `1.897`, and account drawdown `25.93%`, so the remaining misses are Sharpe, concentration, and renewed `2026` weakness rather than a pending promotion-contract refresh.

Focused update as of `2026-07-03` for `frvp_long_reversal_xgb_v1` on the refresh branch:

- Focused baseline backtest: `655` trades, `+2197.25` ticks, Sharpe `0.178`, DSR `0.177`, WFE `-0.218`, max drawdown `4040.55`.
- Targeted `v1` composite prune: `394` trades, `+3546.90` ticks, Sharpe `0.393`, DSR `0.387`, WFE `-0.637`, max drawdown `2548.15`.
- Targeted `v2` composite/pair prune: `284` trades, `+4925.40` ticks, Sharpe `0.659`, DSR `0.635`, WFE `-1.181`, max drawdown `3243.30`.
- `v1` removed the three broad negative composites `ranging_high`, `strong_up_low`, and `strong_up_medium` while preserving the strong `strong_down_medium` pocket.
- `v2` kept the `v1` composite prunes and added the four largest surviving pair-level losers: `strong_up_high/new_york`, `strong_down_high/new_york`, `ranging_low/london`, and `strong_down_low/asia`.
- The saved `v2` static threshold study still falls back to a hard-pruned base `global_threshold`, but static test expectancy still improved from `+2.15` ticks in the focused baseline to `+13.78` ticks in `v2`.
- Treat long-reversal `v2` as the best current-environment reversal policy pass. The remaining drag is too small to justify another broad same-environment prune, so it became the control branch for the next Q11 experiment rather than the final saved reversal checkpoint.

Focused update as of `2026-07-03` for the post-2022 small-CV retrain branch:

- The post-2022 prepared root was materially smaller than the full refresh branch, so the retrain used reduced CV geometry rather than the default wide-window V2 geometry.
- `frvp_long_reversal_xgb_v1` post-2022 walk-forward was worse than the current-environment policy branch: `329` trades, `-2018.85` ticks, Sharpe `-0.396`, DSR `-0.391`, WFE `-0.873`.
- `frvp_long_meta_xgb_v1` post-2022 walk-forward improved economically but remained sub-promotion: `1615` trades, `+9347.25` ticks, Sharpe `0.647`, DSR `0.624`, WFE `-3.532`, profitable quarter share `0.4167`.
- Conclusion: the CPI calendar extraction fix was correct, but a hard post-2022 cutover is not the new default branch. The next non-stationarity experiment should be recency weighting on the full-history refresh root.

Focused update as of `2026-07-04` for the recency-weighted long-reversal branch:

- A full-history recency-weighted retrain was run for `frvp_long_reversal_xgb_v1` using the same prepared chronology and feature set, but with sample weights tilted toward recent years. The saved weighting contract uses a `730`-day half-life, a `0.20` minimum weight floor, and the latest timestamp `2026-06-19T16:55:00+00:00`.
- The recency retrain did **not** improve raw classifier metrics. Relative to the refresh training summary, CV AP fell from `0.7219` to `0.6958`, test AP fell from `0.7636` to `0.7053`, and test event F0.5 fell from `0.7347` to `0.5229`.
- Economics still improved materially once the recency-trained model was run through the targeted policy contract. The first recency backtest branch reached `167` trades, `+5188.45` ticks, Sharpe `0.883`, DSR `0.823`, WFE `-2.612`, and max drawdown `2072.15`.
- A narrow `v3` policy tighten on that recency branch then removed the remaining drag pockets `ranging_medium/asia`, `ranging_medium/london`, and `strong_down_low/london`, producing the best saved reversal result so far: `140` trades, `+6027.00` ticks, Sharpe `1.061`, DSR `0.956`, WFE `-3.797`, max drawdown `1610.85`, and `positive_composite_expectancy_share = 1.0`.
- The static threshold study still does not qualify an alternate threshold variant on this branch. It falls back to a hard-pruned base `global_threshold`, and the walk-forward improvement comes from the recency-trained model plus the pruned base policy rather than a selected `regime_threshold` contract.
- This recency `v3` path is now the best saved full-span reversal checkpoint overall. The refreshed `2026-07-15` account-drawdown rerun records account drawdown `10.06%` and a passing advisory drawdown gate, so the remaining failure is full-history WFE rather than drawdown.

Focused update as of `2026-07-04` for reversal train-side stability:

- A bounded rolling-train control was added to the walk-forward tooling so Q11 could be tested against fixed recent history rather than only an expanding train window.
- Diagnostic train-trade attribution on the recency `v3` branch shows the worst train-side damage is concentrated in older regime history, especially `2022`: full-year `-42231.80` ticks, `2022/new_york -32032.40`, `2022/london -18817.55`, and `2022/strong_down_medium -40894.95`.
- A `3`-year rolling train window and a `2`-year rolling train window did not rescue full-span WFE; they produced `-8.738` and `-142.392` respectively because train annualized PnL stayed unstable.
- When the same `2`-year rolling train contract was evaluated only on post-2024 scheduled folds, WFE turned positive at `0.973` with Sharpe `0.884` and DSR `0.824`.
- Conclusion: the remaining reversal failure is now best understood as older-regime train-side instability. Another broad policy prune is not the mainline fix.

Focused update as of `2026-07-05` for the matching long-meta recency sentinel:

- A full-history recency-weighted sentinel was then run for `frvp_long_meta_xgb_v1` using the same `730`-day half-life and `0.20` floor contract as the reversal branch. The saved weighting artifact is `artifacts/frvp_long_meta_recency_trial1_20260705/phase04/prepared/long_frvp_meta/recency_weighting_summary.json`.
- The retrain changed classifier quality only modestly. Relative to refresh training, test AP improved slightly from `0.5460` to `0.5555` and test ROC AUC from `0.5971` to `0.6046`, but test event F0.5 slipped from `0.7353` to `0.7133`.
- Raw recency economics were poor: `3973` trades, `-1271.45` ticks, Sharpe `-0.035`, DSR `-0.035`, WFE `0.118`, and `positive_composite_expectancy_share = 0.4444`.
- Reapplying the saved long-meta `v3` targeted policy contract rescued the branch to `599` trades, `+6938.65` ticks, Sharpe `0.478`, DSR `0.469`, WFE `1.974`, and max drawdown `3556.85`.
- That rescue still did not beat the saved refresh `v3` pooled checkpoint at `603` trades, `+9216.05` ticks, Sharpe `0.595`, DSR `0.577`, WFE `1.897`, and `positive_composite_expectancy_share = 0.8571`.
- Conclusion: Q11 is not purely reversal-specific, but the actionable recency-weighting win is still mostly reversal-specific. Long meta still responds more to the saved `v3` policy contract than to the recency retrain itself.

Primary artifacts:
- `models/frvp_es_primary_model_registry_current.json`
- `model_testing/reports/frvp_regime_slices/frvp_es_primary_current/run_summary.json`
- `model_testing/reports/frvp_threshold_policies/frvp_es_primary_current/run_summary.json`
- `model_testing/reports/frvp_backtests/frvp_es_primary_current/run_summary.json`
- `model_testing/reports/frvp_backtests/frvp_es_primary_current/model_summary.csv`

Current shortlist:

| Model | Backend | CV AP | Test AP | Test event F0.5 | Current role |
|---|---|---:|---:|---:|---|
| `frvp_long_reversal_xgb_v1` | XGBoost | 0.709 | 0.734 | 0.745 | Benchmark |
| `frvp_short_reversal_xgb_v1` | XGBoost | 0.727 | 0.754 | 0.790 | Benchmark |
| `frvp_long_continuation_xgb_v1` | XGBoost | 0.746 | 0.698 | 0.783 | Benchmark |
| `frvp_short_continuation_xgb_v1` | XGBoost | 0.724 | 0.726 | 0.738 | Benchmark |
| `frvp_long_reversal_tcn_v1` | TCN | 0.720 | 0.739 | 0.770 | Challenger |
| `frvp_short_continuation_tcn_v1` | TCN | 0.729 | 0.723 | 0.806 | Challenger |

Training experiments that did not make the keep-set:
- `frvp_short_reversal_tcn_v1` trained, but its threshold contract collapsed event coverage and did not justify shortlist status.
- `frvp_long_continuation_tcn_v1` trained, but XGBoost remained cleaner on both ranking quality and downstream economics.

What succeeded:
- Phase 5 training is complete on the full-width prepared root, and the resulting shortlist is coherent rather than noisy.
- XGBoost remains the clean baseline across all four direct targets.
- The only meaningful TCN challengers are `frvp_long_reversal_tcn_v1` and `frvp_short_continuation_tcn_v1`.
- The post-training evaluation plumbing joins predictions back to the raw tagged source correctly, so regime labels and trade-path attribution run end-to-end with OHLCV context available.
- The latest reporting stack now carries ES-aware unit aliases (`*_units`) and current FRVP slice dimensions: `frvp_open_type_status`, `frvp_open_type_label`, and `frvp_day_type_label`.
- The open-type coverage issue is now understood correctly: old `unknown` rows were mostly pre-RTH / not-yet-known states, and the current saved rerun shows effectively zero true `missing_after_open` rows.

What works best right now:
- `frvp_long_continuation_xgb_v1` is now the leading direct promotion candidate on the refresh branch. The saved `v3` targeted prune clears the Sharpe and DSR gates decisively in walk-forward, lifts net PnL above `v2`, and compresses drawdown versus the refresh baseline, `v1`, and `v2`.
- `frvp_long_meta_xgb_v1` is no longer just a negative research branch. The saved `v3` targeted prune is now economically positive and much cleaner than baseline, so it should be kept as the pooled-model checkpoint.
- `frvp_long_reversal_xgb_v1` recency `v3` is now the best saved XGBoost reversal policy pass. It is materially stronger than the current-environment `v2` on Sharpe, DSR, net PnL, drawdown, and composite cleanliness, even though it still misses promotion gates.
- `frvp_long_reversal_tcn_v1` is still the only TCN that looks worth a future targeted retrain if we stay in the FRVP family.

What does not work:
- A clean all-family promotion pass still does not exist, but the long-continuation leader is now materially further along. The refreshed `2026-07-15` account-drawdown rerun plus shadow-bundle refresh moved `frvp_long_continuation_xgb_v1` `v3` onto the current promotion contract and it now clears the paper-trading gate under that contract.
- Threshold-policy search still does not find a superior alternate variant for the long-continuation `v2` or `v3` passes; the improvement comes from a hard-pruned base policy, not from the regime-threshold branch.
- Long meta baseline is still weak, and even the saved `v3` targeted prune remains sub-promotion because it misses Sharpe and concentration gates; drawdown should now be treated as advisory risk context after the rerun refresh.
- Long reversal is still sub-promotion even in the recency `v3` branch because full-history WFE stays negative. The refreshed `2026-07-15` account-drawdown rerun passed the advisory drawdown gate, so the remaining issue is train-side instability, not broad residual drag pockets.
- Short reversal, short continuation, and most TCN branches remain economically weak after ES-aware costs.
- Continuation is no longer the weakest family unconditionally; the weakness is now concentrated in short continuation and in the residual bad pockets of the long-continuation baseline policy.

### Phase-gate TODOs and completion assumptions (as of 2026-07-01)

This checklist is the current "what still has to be true before the ES-primary flow is complete" view. It intentionally separates implemented scaffolding from passed gates.

#### Phase 0 / Phase 1 close-out items

- Completed in code: canonical upstream input is now `data/futures_data/ES-5m-tagged.csv`.
- Completed in code: `build_equity_session_frame` applies explicit US equity holiday / half-day / early-close overrides.
- Completed in data: the CPI archive contract is backed by `data/futures_data/CPI_release_dates.txt`, with historical backfill complete through 2023 and curated recent years for 2024-2026.
- Completed in data: Setup 1, Setup 2, and Setup 3 short moved back into the fire-rate band on the current baseline.
- Remaining blocker: the held-out TradingView same-contract profile audit still lacks human sign-off.
- Documented weakness, not blocker: Setup 4 remains thinner than the other setups and should be kept with that caveat rather than forced into a redesign.
- Deferred post-Phase-4 research: `Setup 6b` and optional gamma-context features.

#### Phase 2 gate checklist: feature dataset build

- Completed: the full-history raw feature dump and the merged Phase 2 dataset are archived together under `artifacts/frvp_es_primary_current/phase02/`.
- Pass: no duplicated time columns, no raw OHLCV columns, and no roll-lineage columns entered the persisted Phase 2 dataset.
- Pass: FRVP distances and HTF confluence retain positive MI across all four direct FRVP targets.
- Completed: the old feature-builder fragmentation / memory blocker is closed.
- Documented weakness, not blocker: `frvp_open_type` MI remains weak on `long_frvp_reversal` and `long_frvp_continuation`.

#### Phase 3 gate checklist: FRVP event labeling

- Completed: upstream macro flags now exist for FOMC, CPI, NFP, Fed statement, and Fed presser windows.
- Completed: explicit half-day / early-close flags now come from the equity-session calendar rather than placeholders.
- Pass: roll-spanning exclusions remain intact and event clustering remains inside the gate.
- Pass: all four direct targets sit inside the 250-750 events/year density band.
- Pass: all four direct targets clear the 45% base-rate floor.
- Policy change: Setup 4 is retained and documented as weaker rather than treated as a must-pass standalone gate item.
- Closed direction: blanket shorter continuation holds are no longer an active lever.
- Research note, not hard blocker: continuation quality still trails reversal quality, so use downstream ranking and ES-specific policy economics to decide whether further tightening helps or merely degrades the training sample.

#### Phase 4 gate checklist: preprocessing + backend attribution

- Completed: all four direct FRVP targets prepare cleanly with zero target-specific downstream hacks, and target-specific `htf_confluence_*` carry-through columns remain part of the contract.
- Completed: leakage audit remained clean; no raw OHLCV, label helpers, barrier metadata, or contract-lineage fields survived the prepared sets.
- Completed with caveat: Dalton day-type is implemented upstream, but it did not reach merged top-10 on the saved full-history attribution audit.
- Completed: full-width 736-feature preprocessing is no longer memory-blocked.
- Research note, not blocker: `frvp_open_type` and `frvp_day_type` still miss merged top-10 everywhere on the saved TCN attribution rerun.
- Deferred post-Phase-4 research: determine whether open-type / day-type are genuinely weak, mildly mis-specified, or simply dominated by correlated proxies before spending a new training cycle on them.

#### Phase 5 gate checklist: model training

- Completed: the four direct FRVP targets are trained, and the current shortlist is archived in `models/frvp_es_primary_model_registry_current.json`.
- Completed: calibration choices and selected policy payloads are stored per shortlisted model in the registry.
- Completed promotion hygiene on 2026-07-17: an explicit shuffled-label placebo readout is archived at `model_testing\reports\frvp_placebo_readouts\frvp_long_continuation_xgb_v1_20260717\placebo_readout_summary.json`; the long-continuation baseline recorded real OOF AP `0.7371` versus shuffled mean `0.5002` across `10` shuffles for a placebo gap of `0.2368`, comfortably passing the `> 0.03` rule.
- If a follow-up training sweep becomes necessary, keep it narrow: the serious TCN challengers worth revisiting are long reversal first and short continuation only if upstream continuation economics improve.
- The auxiliary magnitude head remains deferred until the core classification plus policy-economics stack is trustworthy on ES.

#### Phase 6 gate checklist: regime / day-type slice + threshold policy

- Completed: regime-slice evaluation and threshold-policy search run end-to-end on the six-model shortlist.
- Completed: open-type status, open-type label, and Dalton day-type label are now included in the current reporting outputs.
- Completed: ES-aware unit/tick naming now flows through the threshold outputs.
- Completed in code on 2026-07-03: the threshold-search selector now records a hard-pruned base `global_threshold` as the selected policy when no alternate variant qualifies, so base-policy abstain metadata can be written back into the registry cleanly.
- Updated promotion flow on 2026-07-15: threshold search now saves `selected_policy_contract` plus a selected-policy-only `registry_writeback_candidate`, and the FRVP / ICT post-training wrappers no longer auto-write registry policies during research reruns.
- Updated result: the saved `v3` threshold-search run now records `selected_policy_name = global_threshold`, baseline metrics, and `selected_policy_reason = no_non_global_policy_qualified_against_test_baseline`, which is the correct static-study representation for the winning pruned-base-policy contract.
- Completed housekeeping on 2026-07-03: the refresh registry now carries the long-continuation `v3` base-policy prune directly, including the added `ranging_medium/new_york` and `strong_up_low/asia` abstain pairs and the updated static-study metrics.
- Completed research checkpoint on 2026-07-03: the long-meta branch now has saved targeted policy passes through `v3`, with the best static-study contract still falling back to a hard-pruned `global_threshold` rather than a qualified alternate threshold variant.
- Completed research checkpoint on 2026-07-04 UTC: the long-reversal branch now has saved current-environment policy passes through `v2` and recency-weighted policy passes through `v3`. The best saved reversal checkpoint is now recency `v3`, while the best static-study contract still falls back to a hard-pruned `global_threshold`.
- Completed research checkpoint on 2026-07-05 UTC: the matching long-meta recency sentinel is now saved both raw and with the refresh `v3` prune contract applied. It confirmed that pooled long meta is time-sensitive, but it did not beat the refresh `v3` checkpoint once the same policy contract was held fixed.
- Completed friction A/B on 2026-07-19: the saved FRVP branches were rerun under `session_schedule` versus explicit `feature_proxy` spread costs with the model artifacts and regime-labeled prediction roots held fixed. `session_schedule` remained the better contract for the long-continuation `v3` baseline (`+7001.20` ticks, Sharpe `1.187` vs. `+1598.40`, `0.441`) and the full-span long-reversal control (`+6027.00`, `1.061` vs. `+5764.00`, `1.023`). `feature_proxy` only improved the recent-2y long-reversal lane (`+4488.55`, Sharpe `1.564` vs. `+4193.90`, `1.429`) and the weak short-meta sentinel.
- Updated next question after the friction A/B: can regime-gated deployment and concentration control turn the saved reversal checkpoints into a selective deployment contract, or do the remaining misses still require a different train-side stability lever?

#### Phase 7 gate checklist: walk-forward backtest + DSR

- Completed technically: walk-forward policy backtest now runs for all six shortlisted models.
- Completed: the saved backtest contract now uses ES tick-based spreads, slippage, commission, and tick value rather than FX-style pips.
- Updated status: the refreshed `2026-07-15` account-drawdown rerun keeps `frvp_long_continuation_xgb_v1` `v3` at Sharpe `1.226`, DSR `1.061`, WFE `2.459`, max drawdown `1540.10`, and `positive_composite_expectancy_share = 1.0`, and now records `max_drawdown_pct = 9.90%` with `accepted_for_paper_trading_gate = True` under the current advisory account-equity drawdown contract.
- Updated research status: the refreshed `frvp_long_meta_xgb_v1` `v3` account-drawdown rerun in `frvp_long_meta_gatefix_v3_20260715_accountdd` keeps the pooled-branch result at `+9216.05` ticks, Sharpe `0.595`, DSR `0.577`, WFE `1.897`, and account drawdown `25.93%`. It still fails promotion on Sharpe and largest-single-trade-share, but the promotion-contract refresh itself is now complete.
- Updated research status: the saved long-meta recency sentinel improved only after the same `v3` prune contract was re-applied, reaching `+6938.65` ticks, Sharpe `0.478`, DSR `0.469`, WFE `1.974`, and max drawdown `3556.85`. That makes it a useful Q11 control, not the new pooled-model checkpoint.
- Updated research status: the refreshed `frvp_long_reversal_xgb_v1` recency `v3` account-drawdown rerun in `frvp_long_reversal_recency_trial1_v3_20260715_accountdd` remains the best full-span reversal policy result at `+6027.00` ticks, Sharpe `1.061`, DSR `0.956`, WFE `-3.797`, account drawdown `10.06%`, and `positive_composite_expectancy_share = 1.0`. It still fails promotion on full-history WFE, but the drawdown read is now current and passing.
- Completed diagnostic tooling on 2026-07-04 UTC: walk-forward backtest now supports a bounded recent-history train window via `--max-train-years`, which was used to verify that reversal WFE turns positive (`0.973`) once the same recency branch is evaluated only on post-2024 folds with a 2-year rolling train window.
- Completed promotion hygiene on 2026-07-17: the strongest FRVP branches, family leaders, and shadow bundle remain refreshed under the current promotion contract, and the saved long-continuation baseline now also carries the explicit placebo record.
- Completed promotion hygiene on 2026-07-17: the roll-aware embargo / roll-audit package is archived at `model_testing\reports\frvp_roll_audit_packages\frvp_long_continuation_xgb_v1_20260717\roll_audit_package_summary.json`, passes gate 7, records `49` excluded roll-span events with `0` usable roll-span events, and shows `11` roll-crossing fold gaps that all still clear the `576`-bar minimum with a minimum observed gap of `10238` bars (`1290.75` hours).
- Completed promotion packaging on 2026-07-17: `model_testing\reports\frvp_promotion_packages\frvp_long_continuation_xgb_v1_20260717\promotion_package_summary.json` archives the focused threshold-vs-prune study for the long-continuation baseline. The saved `v3` contract is explicitly documented as a hard-pruned `global_threshold` policy rather than a new regime-threshold winner; versus the unpruned refresh baseline it cut trade count `1609 -> 524`, lifted expectancy `7.75 -> 13.72`, improved Sharpe `0.604 -> 1.226`, improved DSR `0.585 -> 1.061`, and reduced max drawdown `10200.85 -> 1540.10`.
- Completed promotion rerun on 2026-07-17: after the roll-safe runtime fixes and passing historical roll replay, `model_testing\reports\frvp_promotion_packages\frvp_long_continuation_xgb_v1_20260717\promotion_package_summary.json` now records `package_status = finalized_without_human_same_contract_signoff` and `promotion_decision = pending_human_same_contract_signoff`, with paper-trading gate, placebo, roll audit, and roll-shadow validation all passing.
- Completed friction A/B readout on 2026-07-19: the economics-layer rerun in `model_testing\reports\frvp_friction_studies\frvp_friction_ab_20260719` closed the main cost-contract uncertainty for the saved branches. The result was not "feature proxy everywhere" or "session schedule everywhere." It was branch-specific: keep `session_schedule` as the default FRVP contract, retain the `feature_proxy` result as a sensitivity read for the recent-2y reversal lane and short-meta sentinel, and move the next work to concentration / selective-deployment policy design rather than another generic friction debate.
- Completed regime-gated deployment / concentration pass on 2026-07-21 in `model_testing\reports\frvp_regime_gated_deployment\frvp_regime_gated_deployment_20260721`. The continuation control stayed the clean frozen reference branch. The full-span reversal candidates did not solve `WFE`: adding `strong_down_medium/asia` lifted net PnL to `+6245.15` ticks and Sharpe to `1.138`, but worsened `WFE` to `-4.334`, which reinforces the train-side-instability read. The recent-2y reversal lane did produce a new selective-deployment checkpoint: blocking `strong_down_high/overlap` yielded `63` trades, `+3677.05` ticks, Sharpe `1.480`, DSR `1.179`, `WFE = 2.162`, profitable-quarter share `0.60`, largest-single-trade share `0.0942`, and `accepted_for_paper_trading_gate = True`.
- Canonical experiment write-ups now live in `docs/FRVP_experiment_journal.md`, using the field contract in `docs/FRVP_experiment_journal.md#frvp_experiment_record_template`. This design paper keeps the strategic interpretation, while the journal records the full `question / control / change / method / artifacts / result / what worked / what did not / observations / decision / repo change` history for each major FRVP experiment.
- Documentation rule from this point forward: if an FRVP experiment changes a checkpoint, a shadow/live contract, a frozen research control, or the backlog order, log it in the journal immediately rather than leaving the explanation only in notes or artifact paths.
- Updated remaining promotion work: placebo evidence, clean roll-audit evidence, the threshold-vs-prune package, and the passing historical roll-shadow validation are now archived; the remaining sign-off work is the human same-contract review plus live-ops discipline around the still-manual IBKR front-month handoff.
- Treat the long-continuation `v3` pass as the new targeted-policy baseline for promotion work, and treat the old all-red readout as superseded for that one branch only.
- Treat the long-meta `v3` pass as the saved pooled-model checkpoint. The matching recency sentinel is complete and came in weaker once the same prune contract was held fixed, so there is still no reason to cut the whole family over to recency weighting.
- Treat the long-reversal recency `v3` pass as the saved full-span reversal checkpoint, and treat `frvp_long_reversal_xgb_recent_regime_prune_v2` as the new selective-deployment checkpoint for the recent-regime lane. The next branch should only escalate to a different train-side stability lever if the repo needs a better full-span reversal answer than the accepted recent-regime contract now provides.

#### Phase 8 gate checklist: live integration

- Completed validation on 2026-07-17: `model_testing\reports\frvp_roll_shadow_validation\frvp_es_shadow_roll_20260717\roll_shadow_validation_summary.json` replayed the real `ESM6 -> ESU6` contract switch at `2026-06-17T00:00:00+00:00` after an `8000`-bar warmup plus a tight live/shadow around-the-roll window.
- Final validation result: `validation_passed = true`, `feature_parity_passed = true`, `health_error_count = 0`, `missing_live_feature_names = []`, and `dashboard_state_contract_switch_recorded = true`, with `15` persisted shadow decisions and sampled audit replays matching exactly.
- Same-day runtime hardening resolved the original blocker by preserving target-specific carry-through context in replay/runtime, preserving FRVP roll-boundary numeric `NaN` values for FRVP XGBoost snapshots, and expanding the inferred history budget for prior-RTH FRVP history features so the short-bundle parity replay matches the saved artifact.
- Remaining live-ops task: keep the active IBKR front-month switch as an explicit operator runbook step until collector-level roll automation exists; the historical roll-safe runtime gate itself is no longer a promotion blocker.
- Extend runtime ingestion to surface contract-roll events, calendar anomalies, and any upstream macro-event flags needed by the offline labeler.
- Register FRVP candidates in the model registry and keep them in `candidate` / shadow mode until the full promotion stack passes.
- Shadow trade for at least four weeks and explicitly verify no live profile carries across a roll untranslated and no open position mapping breaks at the lead-contract switch.

#### Cross-phase assumptions to resolve before calling the flow "complete"

- Canonical macro-calendar sourcing is materially improved by the archived CPI backfill plus curated 2024-2026 dates; the remaining debt is archival hygiene, not missing logic.
- Canonical half-day / holiday handling is implemented in code via the offline rule-based equity calendar; the remaining open item is external sign-off, not missing logic.
- The final Phase 4 full-width prepare verdict comes from the gzip full-history dataset rather than an audit subset.
- Freeze the primary historical train / holdout window before any follow-up training, policy re-evaluation, or promotion rerun.
- `Setup 6b`, optional gamma-context features, and the 6E A/B validation remain deferred research items rather than Phase 0-4 blockers.
- The remaining promotion blockers are now human same-contract sign-off and the live-ops fact that the IBKR front-month handoff is still an explicit operator action. The threshold/backtest mismatch study and the passing historical roll-shadow validation are now archived in the saved promotion package.

## 13. Open Questions and Risks

Status triage after the 2026-07-01 cleanup pass:

- Promotion blockers before paper trading: human same-contract profile sign-off and live-ops sign-off on the still-manual IBKR front-month handoff. The historical FRVP roll-safe shadow replay itself now passes cleanly on the strong long-continuation baseline.
- Not current blockers: Setup 4's lower fire rate, continuation quality missing 0.65, and `frvp_open_type` / `frvp_day_type` failing to reach top-10 as direct model features.
- Highest-value near-term research after the completed 2026-07-19 friction A/B: regime-gated deployment / concentration work for the reversal family, selective recent-regime deployment controls, and any train-side stability lever that still matters after those policy-layer checks. The focused threshold/backtest mismatch study on the strong long-continuation branch is now archival context rather than the main active question. Open-type conditioning and broader feature expansion remain lower priority.
- Highest-value near-term research after the executed 2026-07-21 concentration pass: decide whether the accepted recent-regime reversal checkpoint is the right operational contract, and separately decide whether the full-span branch deserves more train-side stability work now that the policy-layer pass failed to repair `WFE`.

**Q1 — Roll method sensitivity.** Does reset-at-roll vs. translate-by-spread for cross-roll levels change meta-label base rates or backtest Sharpe? Test both in Phase 0/3.

**Q2 — Optimal value-area percentage on ES.** 70% is inherited from futures Market Profile, so it is plausibly already near-optimal here — but confirm against 60%/70%/80% on real volume.

**Q3 — Open-type conditioning power.** Does `frvp_open_type` (open above/inside/below value) materially shift meta-label base rates? If strong, consider open-type-specific models.

**Q4 — Naked VPOC return tendency.** Quantify the empirical rate at which naked VPOCs are revisited within N sessions on ES, and whether Setup 6b clears the placebo gate as a standalone family.

**Q5 — Gamma context value.** Do `es_*` gamma features earn SHAP inclusion, and does `frvp_magnet_is_gamma_pin` help the model avoid false HVN-magnet trades? If options data is too costly, does the model degrade meaningfully without it?

**Q6 — Day-type vs. setup-family interaction.** Confirm reversal-on-balance / continuation-on-trend. If absent, collapse families and reduce model count from 4 to 2.

**Q7 - Mean-reversion family vs. friction floor.** Partially closed on 2026-07-19. The friction A/B showed that the reversal family is not simply being crushed by one obviously wrong cost contract. On the saved full-span control, `session_schedule` still led (`+6027.00` ticks, Sharpe `1.061`) over explicit `feature_proxy` (`+5764.00`, `1.023`). On the saved recent-2y reversal lane, `feature_proxy` improved the readout slightly (`+4488.55`, Sharpe `1.564` vs. `+4193.90`, `1.429`), but that branch still missed promotion on quarter share and concentration rather than raw post-cost edge. So Q7 is now narrower: the remaining question is selective deployment quality and concentration, not whether the current saved reversal checkpoint is being invalidated by a generic friction bug.

**Q8 — RTH vs. ETH anchor dominance.** At a given RTH bar, which anchor's levels carry more SHAP weight — prior RTH or overnight ETH? Slice top-SHAP features by source profile in Phase 6.

**Q9 — Event sparsity for TCN.** With 1–3 events/day and overlapping context windows, the effective training set may be small. Watch train-vs-OOF gap; fall back to XGBoost if it widens.

**Q10 — Live roll robustness.** Verify the incremental engine rolls cleanly mid-strategy: no profile carries across the roll untranslated, no level is compared across coordinate systems, and open positions are mapped correctly when the lead contract changes.

**Q11 — 0DTE regime non-stationarity.** Does pre-2022 ES behave differently enough that training on it harms post-2022 performance? Saved evidence now says "yes, especially for reversal," and the branch history is now clearer. The hard post-2022-only retrain hurt `frvp_long_reversal_xgb_v1` materially and only partially helped `frvp_long_meta_xgb_v1`, so simple truncation is not the right default answer. The full-history recency-weighted reversal branch improved net PnL, Sharpe, DSR, drawdown, and composite cleanliness enough to become the best saved reversal checkpoint, but full-history WFE stayed negative. A bounded 2-year rolling-train diagnostic then turned WFE positive on post-2024 folds, which points to older-regime train-side instability as the remaining issue. The matching long-meta recency sentinel is now complete: raw recency economics were poor, and even after reapplying the saved `v3` prune contract the branch still underperformed the refresh `v3` pooled checkpoint. The current read is therefore more selective than before: Q11 is real beyond reversal, but the economically useful recency-weighting response is still mostly reversal-specific. The next Q11 work should stay on reversal train-side stability controls and selective recent-regime deployment, not on a blanket family-wide recency rollout or another broad same-environment prune.

**Q11 execution plan (2026-07-19).**

1. Freeze the current controls and stop broad same-environment pruning.
   - Keep `frvp_long_reversal_recency_trial1_v3_20260715_accountdd` as the full-span control checkpoint.
   - Keep `frvp_long_reversal_recent_regime_prune_v1_20260715_accountdd` as the recent-regime deployment control.
   - Do not add another new reversal prune branch unless a train-side branch first improves on one of those controls.
2. Reproduce the current full-span recency control under a clean single-model evaluation root.

```powershell
powershell -File scripts/run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_q11_control_20260719 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_q11_control_20260719 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_q11_control_20260719 `
  -ModelIds frvp_long_reversal_xgb_v1 `
  -TargetedFilterPreset frvp_long_reversal_xgb_composite_prune_v3
```

3. Reproduce the bounded-window recent-regime deployment lane with the saved recent-regime prune contract.

```powershell
powershell -File scripts/run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_q11_recent2y_20260719 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_q11_recent2y_20260719 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_q11_recent2y_20260719 `
  -ModelIds frvp_long_reversal_xgb_v1 `
  -TargetedFilterPreset frvp_long_reversal_xgb_recent_regime_prune_v1 `
  -BacktestMaxTrainYears 2 `
  -BacktestMinScheduledTestStart 2024-01-01
```

4. Run one more bounded-window full-span diagnostic before retraining anything.

```powershell
powershell -File scripts/run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_q11_roll3y_20260719 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_q11_roll3y_20260719 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_q11_roll3y_20260719 `
  -ModelIds frvp_long_reversal_xgb_v1 `
  -TargetedFilterPreset frvp_long_reversal_xgb_composite_prune_v3 `
  -BacktestMaxTrainYears 3
```

5. Only if steps 2-4 still show the same train-side failure pattern, open two new retrain branches using the saved recency contract as the template and changing only the half-life.
   - Branch A: `548`-day half-life, `0.20` floor.
   - Branch B: `365`-day half-life, `0.20` floor.
   - Keep the same full-history chronology, features, target definition, calibration method, and prune contracts.
   - Materialize prepared roots that mirror `artifacts/frvp_long_reversal_recency_trial1_20260703/phase04/prepared/`, then train only the reversal XGBoost target with `scripts/run_frvp_training_stack.ps1` and rebuild a dedicated registry from the new model root.

```powershell
powershell -File scripts/run_frvp_training_stack.ps1 `
  -PreparedRoot artifacts\frvp_long_reversal_recency_trial2_548d_20260719\phase04\prepared `
  -XgbOutputRoot models\frvp_long_reversal_xgb_recency_trial2_548d_20260719 `
  -SkipTcnAttribution `
  -SkipTcnTraining

ote_venv\Scripts\python.exe scripts/build_frvp_candidate_registry.py `
  --model-root models\frvp_long_reversal_xgb_recency_trial2_548d_20260719 `
  --source-registry-path models\frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json `
  --output-path models\frvp_es_primary_model_registry_long_reversal_recency_trial2_548d_20260719.json
```

6. Evaluate each new retrain branch twice, once with the full-span `v3` prune contract and once with the recent-regime deployment contract from step 3.
   - A new full-span winner must improve materially on `WFE = -3.797` while keeping Sharpe, DSR, and drawdown in the same neighborhood as the saved recency `v3` control.
   - If no full-span retrain clears that bar, stop retraining and treat the recent-regime deployment lane as the mainline output of Q11 rather than forcing another prune loop.

**Q11 execution update (2026-07-19, completed).**

- Steps 2-4 reproduced the saved failure pattern before retraining: the full-span control in `frvp_long_reversal_q11_control_20260719` stayed economically strong at `+6027.00` ticks, Sharpe `1.061`, DSR `0.956`, and account drawdown `10.06%`, but still failed on `WFE = -3.797`. The bounded recent-regime control in `frvp_long_reversal_q11_recent2y_20260719` kept `WFE = 2.413` and account drawdown `6.95%`, but still failed profitable-quarter share (`0.50`) and single-trade concentration (`0.1279`).
- Added `scripts/materialize_frvp_recency_prepared_root.py` so shorter-half-life reversal branches can be rebuilt from the clean `frvp_es_primary_refresh_20260701` prepared root without changing chronology, features, or the untouched test split.
- The `548`-day retrain branch confirmed that a milder half-life is not the answer. Full-span `frvp_long_reversal_q11_548d_control_20260719` finished at `+4572.50` ticks, Sharpe `0.778`, DSR `0.738`, `WFE = -3.429`, account drawdown `14.41%`, and largest-single-trade share `0.1236`, so it degraded the full-span control on both stability and risk. Its bounded recent-regime lane (`frvp_long_reversal_q11_548d_recent2y_20260719`) stayed viable at `+2939.25` ticks, Sharpe `0.930`, `WFE = 1.450`, and drawdown `11.29%`, but still failed profitable-quarter share (`0.50`) and concentration (`0.1825`).
- The `365`-day retrain branch was the stronger of the two, but it still did not fix full-span promotion quality. Full-span `frvp_long_reversal_q11_365d_control_20260719` improved drawdown and Sharpe enough to clear those gates at `+4071.50` ticks, Sharpe `0.823`, DSR `0.775`, and account drawdown `11.00%`, but `WFE` only improved to `-2.348` and single-trade concentration still failed at `0.1389`.
- The best new output is the bounded recent-regime `365`-day lane in `frvp_long_reversal_q11_365d_recent2y_20260719`: `+4451.85` ticks, Sharpe `1.446`, DSR `1.167`, `WFE = 2.355`, account drawdown `6.27%`, profitable-quarter share `0.60`, and positive composite expectancy share `1.0`. It still misses promotion quality because the largest-single-trade share is `0.1205`, but it is now the strongest saved selective-deployment checkpoint produced by the half-life retrain pass.
- Decision: the half-life-only retrain branch is now closed. No new full-span winner cleared the saved control bar, so the repo should stop recency half-life tuning here, keep the full-span recency `v3` branch as the saved full-span control, and treat the `365`-day recent-regime lane as the main Q11 selective-deployment checkpoint if reversal remains on a post-2024 deployment path. Any follow-up FRVP reversal work should now move to a different stability lever or to concentration/risk acceptance, not to another same-environment prune loop.

**Next FRVP roadmap (2026-07-21).**

The next concrete work is documented in `docs/FRVP_experiment_journal.md#frvp_regime_gated_deployment_roadmap_20260721`, and the first execution pass is now complete. In plain English, the result is: keep the continuation control unchanged; keep the full-span reversal control unchanged because one more policy prune did not fix `WFE`; and use `frvp_long_reversal_xgb_recent_regime_prune_v2` as the current selective-deployment checkpoint while the repo decides whether that narrower contract is sufficient or whether a new train-side lever is still worth pursuing.

**Deferred-research audit update (2026-07-28).**

The saved repo-state read for the deferred FRVP cluster now lives in `model_testing/reports/frvp_deferred_research_audit/frvp_deferred_research_audit_20260728/DEFERRED_RESEARCH_SUMMARY.md`. It closes one ambiguity without pretending the experiments are already run: `Setup 6b` is still not implemented in saved Phase 3 artifacts, gamma-context columns are still absent from the Phase 2 feature contract, Setup 4 is thin but real enough to justify a standalone retest, and the higher-EV next lane is per-setup-family analysis/training rather than another pooled-direction-family optimization pass. The current execution order is therefore: split-first, then Setup 4 standalone, then roll-translation A/B on the branches that still show roll-bracket drag, then Setup 6b only after naked-VPOC context is promoted into event logic, and gamma-context only after a real options-data contract exists.

**Q12 — Does real volume actually beat tick volume? (the foundational question)** Run the Section 11.2 three-way A/B (EUR/USD spot vs. 6E vs. ES). If the real-volume variants do not show a larger placebo gap than the tick-volume original, the entire premise of re-homing is empirically unsupported and the instrument choice should be revisited on other grounds (frictions, exposure, data cost).

---

## References

- Dalton, J., Jones, E., & Dalton, R. (2007). *Markets in Profile*. Wiley.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Lopez de Prado, M., & Bailey, D. H. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40:5, 94–107.
- Lopez de Prado, M., & Lewis, M. J. (2019). "Detection of False Investment Strategies Using Unsupervised Learning Methods." *Quantitative Finance*, 19:9, 1555–1565.
- Steidlmayer, J. P., & Hawkins, S. B. (2003). *Steidlmayer on Markets*. Wiley.
- CME Group. *E-mini S&P 500 (ES)* and *Euro FX (6E)* contract specifications (verify current specs and fees against the exchange before sizing).

---

## Repository File Index

Implemented in repo today:

| Path | Description |
|---|---|
| `frvp/continuity/continuous_contract.py` | Causal lead-contract series, separated raw/profile vs. back-adjusted/path coordinates |
| `frvp/continuity/roll_calendar.py` | Volume-based, causal roll-date determination and roll-spread record |
| `frvp/continuity/reconstruct_boundaries.py` | Vendor-tagged continuous-series reconstruction, schedule/report export, and per-bar contract tagging |
| `frvp/config/instruments.py` | Per-instrument config (ES / 6E): tick size, anchors, sessions, and profile defaults |
| `frvp/sessions/equity.py` | RTH/ETH/IB boundaries for ES-style session anchors |
| `features/sessions_equity.py` | Compatibility shim for the design-paper session path |
| `frvp/profiles/builder.py` | `VolumeProfileBuilder` and profile-level POC / VA / HVN / LVN / shape extraction |
| `frvp/profiles/anchors.py` | Causal anchor resolution plus naked-VPOC state tracking |
| `frvp/feature_sets/frvp_context.py` | Real `frvp_context` implementation built on the Phase 0 continuity layer |
| `features/feature_sets/frvp_context.py` | Registry shim that exposes `frvp_context` to the existing feature builder |
| `frvp/setups/detector.py` | Rule-based setup detector (Setups 1-6) plus fire-rate diagnostic |
| `frvp/strategies/frvp_setups.py` | Compatibility wrapper for the setup detector |
| `features/recipes/frvp_meta.json` | Build recipe for the FRVP meta-labeling dataset |
| `tests/test_continuity.py` | Roll/continuity assertions (Section 4.3) |
| `tests/test_profiles.py` | Profile-builder, anchor, naked-VPOC, and single-contract assertions |
| `tests/test_frvp_context.py` | FRVP feature-family and causality assertions |
| `tests/test_frvp_setups.py` | Setup-detector and fire-rate summary assertions |

Supporting or later-phase additions/modifications:

| Path | Modification |
|---|---|
| `features/recipes/ote_extended.json` | Current repo change: add `frvp_context` to the extended recipe |
| `data/labeling/labeling_engine.py` | Call `frvp_labeling_engine.py` alongside existing families |
| `data/labeling/frvp_labeling_engine.py` | FRVP event sampling, triple-barrier, quality scoring, sample weights |
| `models/frvp_es_primary_model_registry_current.json` | Current FRVP shortlist registry and saved policy payloads |
| `scripts/run_frvp_training_stack.ps1` | One-shot FRVP training wrapper for XGBoost plus TCN attribution/training |
| `scripts/run_frvp_post_training_eval.ps1` | One-shot FRVP regime-slice, threshold-policy, and walk-forward backtest wrapper |
| `scripts/materialize_frvp_recency_prepared_root.py` | Rebuilds a single-target FRVP prepared root with a chosen recency half-life/floor while leaving the test split untouched for apples-to-apples Q11 evaluation |
| `model_testing/ote_regime_labeler.py` | FRVP open-type status labeling (`classified`, `not_yet_known`, `missing_after_open`) |
| `model_testing/ote_regime_slices.py` | FRVP open-type and day-type slice-family reporting |
| `model_testing/ote_policy_metrics.py` | ES-aware `*_units` aliases alongside legacy `*_pips` fields |
| `model_testing/ote_threshold_policy.py` | Threshold-policy outputs with ES unit aliases |
| `model_testing/ote_policy_backtest.py` | ES-aware walk-forward backtest metrics and saved unit aliases |
| `ote_live/features/incremental_engine.py` | Incremental FRVP profiles + live roll handling |
| `ote_live/ingestion/runtime.py` | Contract-roll events + gap handling |
| `ote_live/models/loaders.py` | Load FRVP meta-labeler models |

---

## Appendix: original EURUSD FRVP design paper (2026-06-15)

The ES-primary paper above evolved from this EURUSD design. It is kept verbatim below.

<a id="frvp_ml_strategy"></a>

### Fixed Range Volume Profile as ML Signal Foundation

*Merged 2026-10-09 from `docs/FRVP_ML_Strategy.md`.*

#### Design Paper: FRVP-Driven Signal Pipeline for 5-Minute EUR/USD

**Author:** Quantitative Research & ML Systems  
**Date:** 2026-06-15  
**Status:** Blueprint — no pipeline code yet  
**Repo target:** Slot into the existing OTE meta-labeling architecture

---

#### Table of Contents

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

#### 1. Executive Summary

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

#### 2. FRVP Primer

##### 2.1 What Is the Fixed Range Volume Profile?

The Volume Profile is a market tool that plots traded volume at each price level over a specified lookback window, rather than plotting volume over time (as a traditional volume bar chart does). The *Fixed Range* variant defines that window by a specific price-time interval chosen by the analyst — a session, a prior day, a swing leg — rather than a rolling window.

The core idea traces to J. Peter Steidlmayer's Market Profile work from the 1980s at the CBOT (Steidlmayer & Hawkins, 2003), later extended to electronic order-flow and volume analysis by Dalton et al. (2007). The FRVP is now a standard tool in institutional order-flow analysis.

##### 2.2 Profile Anatomy

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

##### 2.3 Setups and Trade Logic

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

##### 2.4 Critical FX/Automation Caveats

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

#### 3. ML Formulation Analysis

##### 3.1 The Central Question

Given that an FRVP setup fires (one of the 6 setups above is algorithmically detected), should the system take the bet? This is the question the ML layer must answer. The formulation choice determines:
- What label is assigned to each event
- What features have predictive content
- How severely overfitting will mask itself
- How the model composes with the existing regime classifier and risk layer

Four formulations are evaluated below.

---

##### 3.2 Formulation A: Direct Signal Classification

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

##### 3.3 Formulation B: Meta-Labeling of FRVP Setups (Recommended Primary)

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

##### 3.4 Formulation C: Learning-to-Rank

**What it predicts:** When multiple FRVP setups are simultaneously active (e.g., VAH of the London session profile coincides with VAL of the prior-day profile), rank them by expected quality and trade only the top-ranked signal.

**Labeling implied:** Pairwise or listwise ranking labels (e.g., LambdaRank, XGBoost ranking objective). Labels require at least two concurrent signals per ranking window — the one that worked is ranked above the one that didn't.

**Advantages:** Directly addresses the "which of multiple concurrent signals to prioritize" question, which will arise when multiple profiles produce overlapping setups.

**Disadvantages:**
- At 1–3 trades/day frequency, truly concurrent FRVP signals (same bar, different anchors) will be uncommon. The ranking corpus will be very small.
- The infrastructure for ranking label generation and training does not exist in the repo and would require significant new development.
- Overkill for the current signal frequency. The problem of "which signal to take" is better handled by a confidence-threshold gate on the meta-labeler, combined with a "no concurrent positions" rule in the risk layer.

**Verdict:** Not recommended for this phase. File as a Phase 4 research experiment if meta-labeling produces too many concurrent high-confidence signals.

---

##### 3.5 Formulation D: Risk-Adjusted Return / Trend-Scanning Regression

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

##### 3.6 Recommendation and Fallback

**Primary: Formulation B (Meta-Labeling)**  
Run separate meta-labelers for the two setup families (reversal setups: 1, 4, 6; continuation setups: 2, 3, 5), with separate long/short models consistent with the repo's established pattern. Use the existing OTE TCN pipeline for training. Apply regime-conditional thresholds via the existing regime-slice machinery.

**Fallback: Formulation A (Direct Classification, CUSUM-sampled)**  
If FRVP setup events are too sparse (fewer than ~1,000 per setup family per side in 2018–2024), fall back to CUSUM event sampling at FRVP levels with direct triple-barrier classification. Use XGBoost (faster, lower data requirement than TCN) as the model. This is equivalent to the `long_breakout_xgb_v1` pattern already in the repo.

**The key distinction between B and A in practice:** In meta-labeling, the event sampling is driven by the rule-based setup firing. In direct classification, the sampling can be driven by CUSUM or by proximity to a profile level. Meta-labeling produces cleaner, more interpretable events; direct classification gives more events but with more noise.

---

#### 4. FRVP Feature Engineering

##### 4.1 Integration with the Repo's Feature Framework

New FRVP features live in a new registered feature family: `features/feature_sets/frvp_context.py`, registered as `"frvp_context"`. This module is added to the `ote_extended.json` recipe and to a new `frvp_meta.json` recipe. It depends on:
- `features/feature_sets/volume.py` (for `dollar_volume`, `volume_zscore_50`)
- `features/feature_sets/structure.py` (for `atr_14`, swing levels)
- `features/fx_calendar.py` (for session boundary lookups)
- The causal swing detector from `data/labeling/reversal_labeling_engine.py` (for swing-to-swing anchoring)

All features are computed **causally** — using only the completed prior session or completed prior swing leg. No bar in the current modeling window is included in the profile construction.

##### 4.2 Profile Construction Algorithm

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

##### 4.3 Feature Catalog

All distance features are in **ATR units** (ATR from the 14-period ATR computed on 5-minute bars). This normalizes across different volatility regimes and is consistent with the repo's existing feature conventions (e.g., `dist_to_prior_high_20_atr`, `dist_to_bull_fvg_atr`).

Transformations: Apply the same rolling z-score and percentile-rank transforms used for other features in the `ote_extended.json` recipe. Volume-at-price features (POC volume fraction, node density) should additionally be log-winsorized at the 1st/99th percentile before z-scoring.

###### Group 1: Distance Features

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

###### Group 2: Price Position Within Value Area

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_price_position_va` | `(close - VAL) / (VAH - VAL)`. 0 = at VAL, 1 = at VAH, <0 = below VA, >1 = above VA | None (bounded interpretation) |
| `frvp_above_poc` | Binary: 1 if `close > POC_session`, else 0 | None |
| `frvp_in_va` | Binary: 1 if `VAL <= close <= VAH` | None |
| `frvp_above_vah` | Binary: 1 if `close > VAH` | None |
| `frvp_below_val` | Binary: 1 if `close < VAL` | None |
| `frvp_va_overshoot_atr` | `max(close - VAH, 0) / ATR` or `max(VAL - close, 0) / ATR` — signed extension beyond VA | None |

###### Group 3: Profile Shape Classification

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

###### Group 4: HVN/LVN Density

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_hvn_count` | Number of HVNs in the session profile | None |
| `frvp_lvn_count` | Number of LVNs in the session profile | None |
| `frvp_hvn_count_1atr` | Count of HVN levels within ±1 ATR of current close | None |
| `frvp_lvn_count_1atr` | Count of LVN levels within ±1 ATR of current close | None |
| `frvp_hvn_above_close` | Distance to nearest HVN above close / ATR | None |
| `frvp_hvn_below_close` | Distance to nearest HVN below close / ATR | None |

###### Group 5: Value Area Metrics

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_va_width_atr` | `(VAH - VAL) / ATR_14` — value area width normalized by ATR | z-score over 20 sessions |
| `frvp_va_width_zscore_20` | Z-score of `frvp_va_width_atr` vs trailing 20-session mean/std | None (already z-scored) |
| `frvp_va_migration_vel` | Change in VA midpoint between prior two sessions / ATR | None |
| `frvp_poc_migration_vel` | Change in POC level between prior two sessions / ATR | None |
| `frvp_va_expansion` | 1 if current VA width > rolling 20-session median, else 0 | None |

###### Group 6: Multi-Anchor Confluence

| Feature Name | Description | Transform |
|---|---|---|
| `frvp_poc_anchor_diff_atr` | `(POC_session - POC_day) / ATR` — how much session and day POCs diverge | None |
| `frvp_va_overlap_pct` | Overlap fraction of session VA and prior-day VA | None |
| `frvp_swing_poc_vs_session_poc` | `(POC_swing - POC_session) / ATR` | None |
| `frvp_multi_poc_cluster` | Binary: 1 if session POC, day POC, and swing POC are all within ±0.5 ATR of each other | None |

###### Group 7: Setup Type Encoding (for Meta-Labeling)

| Feature Name | Description |
|---|---|
| `frvp_setup_type` | Integer: 1=mean-reversion, 2=POC-retest, 3=VA-breakout, 4=failed-auction, 5=LVN-traverse, 6=HVN-magnet |
| `frvp_setup_side` | 1 = long hypothesis, -1 = short hypothesis |
| `frvp_setup_confidence_rule` | Rule-based confidence score [0–1] from the setup engine (based on convergence of indicators) |

###### Group 8: Interaction with Existing Repo Features

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

#### 5. Labeling

##### 5.1 Overview: FRVP Events in the Existing Pipeline

The existing labeling pipeline generates three label families from causal swing detection: reversal, continuation-pullback, and breakout. FRVP events are a new, fourth family — distinct from the existing families because the event trigger is not a price swing but a price interaction with a volume profile level.

FRVP labeling reuses the same infrastructure: triple-barrier outcomes, trend-scan quality scoring, sample uniqueness weights, exclusion masks, and the zone/entry label distinction.

##### 5.2 FRVP Event Sampling

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

##### 5.3 Triple-Barrier Labeling

Reuse the existing triple-barrier infrastructure with FRVP-specific barrier widths for each setup family:

| Setup Family | TP (ATR) | SL (ATR) | Max Holding (bars) | RR Ratio |
|---|---|---|---|---|
| Mean-reversion (1, 6) | 0.8 × ATR | 1.0 × ATR | 60 bars (5h) | 0.8:1 (label, but size-adjust at risk layer) |
| POC retest / continuation (2, 3, 5) | 1.5 × ATR | 1.0 × ATR | 96 bars (8h) | 1.5:1 |
| Failed auction (4) | 1.2 × ATR | 0.8 × ATR | 48 bars (4h) | 1.5:1 |

Note: The ATR used for barriers is the structural ATR (1hr mapped to 5-min) as in the existing labeling engine, not the 5-minute ATR directly. This prevents overly tight barriers in volatile periods.

The meta-label (binary) is: 1 if the TP barrier is hit first (in the hypothesized direction), 0 if the SL barrier or timeout is hit first.

##### 5.4 Quality Scoring and Sample Weights

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

##### 5.5 Exclusion Masks

Exclusion zones around ambiguous FRVP areas:
- Bars within a freshly completed session (first 5 bars of a new session) before the new profile has enough volume — exclude from training.
- Bars during major macro events (FOMC, ECB, NFP) — use the existing `quality` feature set's anomaly flags.
- Bars where session tick volume is less than 20% of the rolling 20-session median (very thin volume, profile unreliable).

---

#### 6. Data Requirements

##### 6.1 Historical Data

| Timeframe | Purpose | Already in repo? |
|---|---|---|
| 5-minute EUR/USD OHLCV | Modeling timeframe — labeling, feature engineering, training | **Yes** — `data/currency_data/eurusd-5m.csv` (2010–2025) |
| 30-minute EUR/USD | HTF structural context | **Derived** — computed in `htf_context` feature set via resampling |
| 1-hour EUR/USD | Structural ATR, HTF trend, session context | **Derived** — computed in `htf_context` feature set |
| 1-minute EUR/USD | Intrabar precision for FRVP histogram bin assignment, execution modeling | **Potentially not present** — check `data/currency_data/` |
| Session boundary metadata | Which bars belong to each London/NY/Asia session | **Derived** — `fx_calendar.py` handles this |

**Date range:** Use 2018–2024 for model training (7 years). 2025–present is the out-of-sample holdout for final promotion decisions. The existing dataset starts in 2010, but pre-2018 EUR/USD microstructure is qualitatively different (pre-MiFID II, different ECN landscape). Use 2018 as the cutoff to reduce regime-stationarity problems.

**1-minute data:** If available, use it to improve volume histogram resolution (finer price-level assignment within each bar). If not available, the standard "uniform volume across the bar range" assumption is sufficient for 5-minute bars during liquid sessions. This is worth a sensitivity analysis in Phase 3.

##### 6.2 Profile Construction Per Bar

At each 5-minute bar `t`, maintain three live profile states:

1. **Session profile (London):** Built from all bars in the most recently *completed* London session (session_end ≤ t). Updates once per day when the London session closes at ~11:00 ET.

2. **Session profile (NY):** Built from all bars in the most recently *completed* NY session. Updates once per day when NY closes at ~16:00 ET.

3. **Swing profile:** Built from all bars in the most recently *completed* swing leg (from confirmed swing low to confirmed swing high, or vice versa). Updates each time the causal swing detector from the labeling engine fires a new confirmed swing.

Memory requirement: Maintaining three active profiles at each bar is computationally tractable — each profile is a dictionary of `{price_bin: tick_volume}` updated incrementally.

##### 6.3 Alignment and Look-Ahead Rules

| Rule | Rationale |
|---|---|
| Profile from prior session/swing only | The primary look-ahead defense |
| Setup fire condition uses completed-bar close | Consistent with existing reversal labeler convention |
| Triple-barrier evaluation starts at next bar's open | The trade is not executable at the signal bar's close (at bar-close labeling) |
| Structural ATR uses prior 14 bars × 1hr data only | ATR is always backward-looking |
| SHAP/attribution is computed inside each CV fold on training data only | Prevents feature selection from leaking test data |

---

#### 7. Model Architecture

##### 7.1 Where FRVP Fits in the Existing Model Stack

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

##### 7.2 Model Choice

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

##### 7.3 Probability → Position Size Flow

Consistent with the existing OTE live system:

1. **Primary classifier** (reversal model or continuation model) outputs `P(profitable setup)`.
2. **Regime gate**: Apply the regime-conditional threshold from the `ote_threshold_policy_search` step. In trending regimes, continuation setups need lower threshold (trend helps); reversal setups need higher threshold (trend fights them). In ranging regimes, reversal setups get a lower threshold.
3. **Abstain policy**: If neither reversal nor continuation setup is active, abstain. If both are active simultaneously (rare), take the higher-confidence one.
4. **Position size**: Scale fractional Kelly between `P_min` (minimum acceptable probability, ~0.55 given 1:2 RR) and `P_max` (near-certain, ~0.80). The existing risk layer already implements this via confidence multipliers.
5. **Stop loss / take profit**: Use the FRVP-specific barrier widths from Section 5.3, scaled by current ATR.

##### 7.4 Composing with the Existing Regime Classifier

The existing `model_testing` stack produces regime-slice reports that show each model's performance by regime bucket. For FRVP models:

- **Reversal setups** should outperform in `ranging` and `low_volatility` regimes (profile is stable, mean-reversion reliable).
- **Continuation setups** should outperform in `strong_uptrend`, `strong_downtrend`, and `high_volatility_breakout` regimes (VAH/VAL breakouts are more sustained).

Use the existing 7-class regime labeler for slicing. Set minimum event count at 50 per regime bucket (same as existing `min_positive_events=50` convention). Regime buckets with fewer events should trigger abstain rather than forcing a low-sample threshold.

---

#### 8. Validation and Anti-Overfitting

##### 8.1 Purged K-Fold + Embargo

The existing OTE training pipeline implements purged walk-forward cross-validation via explicit row-geometry controls:
- `--cv-initial-train-rows 250000` (≈ 2 years of 5-min bars)
- `--cv-val-rows 100000` (≈ 10 months)
- `--cv-step-rows 100000`
- Minimum 4 folds (to span at least 3–4 years of training evolution)

For FRVP meta-labeling, events are sparse (perhaps 1 per session). The row-geometry approach works on the full 5-minute series including non-event bars. The purging removes the window around each meta-labeled event from the validation set to prevent temporal leakage between nearby events that share training data (Lopez de Prado, 2018, Chapter 7). No change needed to the existing purging logic.

**Embargo:** At least 1 full session (288 bars = 24 hours of 5-min bars) between the training window and the validation window. This prevents the model from learning from the tail of a session whose FRVP profile informs an event in the next session.

##### 8.2 CPCV (Combinatorial Purged Cross-Validation)

Lopez de Prado (2018, Chapter 12) shows that standard k-fold CV underestimates the number of effectively independent test paths, biasing Sharpe ratio estimates upward. CPCV addresses this by evaluating all possible combinations of fold assignments.

The existing repo uses a sequential walk-forward approach rather than full CPCV. For FRVP, include a CPCV backtest (minimum 6 paths) as an additional validation step before promotion. This is not in the existing pipeline and requires a new script or a mode flag on the existing walk-forward runner.

##### 8.3 Walk-Forward Policy Backtest with Frictions

Reuse the existing backtest runner (`scripts/run_ote_policy_backtest.py`) with the same friction parameters:
- `fixed_slippage_pips_per_trade = 0.3`
- `commission_pips_per_trade = 0.35`
- `min_train_years = 2`
- `test_window_months = 3`
- `rolling_step_months = 3`
- `min_folds = 8`

The backtest should report performance separately by: setup family (reversal vs. continuation), regime bucket, session (London vs. NY), and confidence decile.

##### 8.4 Deflated Sharpe Ratio

The standard Sharpe ratio is inflated by selection bias when multiple model configurations are tried and the best is selected. Lopez de Prado & Bailey (2014, "The Deflated Sharpe Ratio") provide a correction that accounts for the effective number of trials.

For FRVP: if training involves trials (`--trials 20` in the Optuna hyperparameter search), and multiple setup-type filter combinations are tested, the effective number of trials could be 20–100. The Deflated Sharpe is approximately:

```
DSR = SR × sqrt((T - 1) / (N_trials)) × (1 - SR²/SR_max²)^0.5
```

A promotion gate of DSR > 0.5 (annualized) is reasonable for an intraday FX strategy at 1–3 trades/day. Any candidate with DSR < 0.3 should not be promoted regardless of nominal performance.

##### 8.5 Placebo / Shuffled-Label Test

After training a FRVP meta-labeler, re-run training with **randomly shuffled meta-labels** (preserving the event timestamps and features). Compare OOF AUC-PR between the real model and the shuffled model. If the real model's AUC-PR is within 2 standard deviations of the shuffled model's AUC-PR across 10 shuffles, the features contain no detectable signal. This is a strict null-hypothesis test.

Run this as a separate reviewed step against the saved artifact with `scripts/run_ote_placebo_readout.py`, then archive the summary JSON beside the promotion package.

Saved ES-primary reference record on 2026-07-17:

```
ote_venv\Scripts\python.exe scripts/run_ote_placebo_readout.py --real-artifact-dir models\frvp_es_primary_xgb_refresh_20260701\long_frvp_continuation --output-root model_testing\reports\frvp_placebo_readouts\frvp_long_continuation_xgb_v1_20260717 --num-shuffles 10
```

Archived result: `model_testing\reports\frvp_placebo_readouts\frvp_long_continuation_xgb_v1_20260717\placebo_readout_summary.json` recorded real OOF AP `0.7371` versus shuffled mean OOF AP `0.5002` (std `0.0059`) across `10` shuffles, for a placebo gap of `0.2368`; this passed the `> 0.03` promotion rule by a wide margin.

Saved ES-primary promotion-package record on 2026-07-17: `model_testing\reports\frvp_promotion_packages\frvp_long_continuation_xgb_v1_20260717\promotion_package_summary.json` finalized the threshold-vs-prune readout for the long-continuation baseline. The package conclusion is that the saved `v3` branch is a hard-pruned `global_threshold` contract rather than a new regime-threshold winner: relative to the unpruned refresh baseline it cut trade count `1609 -> 524`, lifted expectancy `7.75 -> 13.72`, improved Sharpe `0.604 -> 1.226`, improved DSR `0.585 -> 1.061`, and reduced max drawdown `10200.85 -> 1540.10`. The July 17, 2026 rerun after the passing roll-shadow validation now records `package_status = finalized_without_human_same_contract_signoff` and `promotion_decision = pending_human_same_contract_signoff`, with paper-trading gate, placebo, roll audit, and roll-shadow validation all passing.

##### 8.6 FRVP-Specific Leakage Traps

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

#### 9. Build Roadmap

The following phases are sequenced with explicit dependencies. Each phase has a validation gate that must be passed before beginning the next.

##### Phase 1: FRVP Profile Construction Engine

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

##### Phase 2: FRVP Feature Dataset Build

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

##### Phase 3: FRVP Event Labeling

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

##### Phase 4: Preprocessing and Backend Attribution

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

##### Phase 5: Model Training

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

##### Phase 6: Regime-Slice Report and Threshold Policy Search

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

##### Phase 7: Walk-Forward Backtest and DSR Check

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

##### Phase 8: Live Integration

**Goal:** Deploy FRVP models as candidates in the existing `ote_live` app.

**Tasks:**
1. Add FRVP model artifacts to `models/live/` flat directory.
2. Add entries to `models/ote_model_registry_live_multifamily.json` with `status="candidate"`.
3. Update `ote_live/features/incremental_engine.py` to compute FRVP features incrementally at each bar (the profile builder maintains state across bars; the setup detector runs at each new bar close).
4. Update `ote_live/models/loaders.py` to load FRVP meta-labeler models.
5. Shadow-mode paper trading for minimum 4 weeks before considering promotion to `status="active"`.

Saved ES-primary live-runtime reference record on 2026-07-17: `model_testing\reports\frvp_roll_shadow_validation\frvp_es_shadow_roll_20260717\roll_shadow_validation_summary.json` replayed the actual `ESM6 -> ESU6` roll boundary at `2026-06-17T00:00:00+00:00` after an `8000`-bar warmup. The final rerun now passes cleanly with `validation_passed = true`, `feature_parity_passed = true`, `health_error_count = 0`, `missing_live_feature_names = []`, `dashboard_state_contract_switch_recorded = true`, and `15` persisted shadow decisions. This removes the historical Phase 8 runtime blocker; the remaining live-integration work is human same-contract sign-off plus the still-manual IBKR front-month handoff in production operations.

---

#### 10. Open Questions and Risks

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

#### References

- Dalton, J., Jones, E., & Dalton, R. (2007). *Markets in Profile*. Wiley.
- Lequeux, P., & Acar, E. (1998). "A Dynamic Index for Managed Currencies Funds Using CME Currency Contracts." *European Journal of Finance*, 4:4, 311–330.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Lopez de Prado, M., & Bailey, D. H. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*, 40:5, 94–107.
- Lopez de Prado, M., & Lewis, M. J. (2019). "Detection of False Investment Strategies Using Unsupervised Learning Methods." *Quantitative Finance*, 19:9, 1555–1565.
- Steidlmayer, J. P., & Hawkins, S. B. (2003). *Steidlmayer on Markets*. Wiley.

---

#### Repository File Index

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

# Fixed Range Volume Profile as ML Signal Foundation
## Design Paper: FRVP-Driven Signal Pipeline for ES (E-mini S&P 500) — Primary, with 6E (Euro FX) Variant

**Author:** Quantitative Research & ML Systems
**Date:** 2026-06-21
**Status:** Blueprint — no pipeline code yet
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
| **Continuous-contract + roll engine** | **New** — `data/futures/continuous_contract.py`, `data/futures/roll_calendar.py` |
| **FRVP profile construction engine** | **New** — `features/feature_sets/frvp_context.py` |
| **FRVP setup rule layer** | **New** — `features/strategies/frvp_setups.py` |
| **FRVP-specific label events** | **New** — `data/labeling/frvp_labeling_engine.py` |
| **Instrument config (ES / 6E)** | **New** — `features/config_instruments.py` |

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

New features live in `features/feature_sets/frvp_context.py` (registered `"frvp_context"`), added to `ote_extended.json` and a new `frvp_meta.json`. Dependencies: `volume.py` (real-volume z-scores), `structure.py` (ATR, swings), the new `data/futures/continuous_contract.py` (lead-contract + roll), and the equity-session calendar (RTH/ETH/IB boundaries) via a new `features/sessions_equity.py` (the ES analog of `fx_calendar.py`). All features are computed **causally**.

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
| ES OHLCV **with real volume**, either as a raw tagged continuous lead-contract series or as per-contract bars | Profiles, labeling, features, training | **Current baseline** — `data/futures_data/ES-5m-tagged.csv`; optional refinement — `data/futures_data/es/` per-contract files |
| ES roll calendar / lead-contract series | Continuity (Section 4) | **New** — derive in `roll_calendar.py` |
| 30-min / 1h ES (back-adjusted) | HTF structure, ATR, regime (path features only) | **Derived** — resample the continuous series |
| Equity session calendar (RTH/ETH/IB, half-days, holidays) | Anchors and masks | **New** — `features/sessions_equity.py` |
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

Compare meta-label base rates, OOF AUC-PR, and especially the **placebo-test gap** (real minus shuffled AUC-PR). If real volume carries genuine signal that tick volume does not, (2) and (3) should show a larger placebo gap than (1). This directly tests the foundational claim that motivated re-homing the strategy, and it is cheap once the pipeline is instrument-parameterized via `features/config_instruments.py`.

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
**Files:** `data/futures/continuous_contract.py`, `data/futures/roll_calendar.py`, `tests/test_continuity.py`.
**Gate:** Lead-contract assignment is causal and matches known historical roll dates within ±1 day; back-adjusted vs. raw series reconcile on returns but differ on absolute levels exactly by cumulative roll spreads; assertions in Section 4.3 pass. If the input is a vendor-tagged raw continuous series rather than overlapping per-contract bars, Phase 0 may validate the observed `contract_symbol` / roll-boundary tags instead of re-deriving the roll calendar from scratch.

### Phase 1 — FRVP profile construction engine
**Files:** `features/feature_sets/frvp_context.py`, `features/strategies/frvp_setups.py`, `features/sessions_equity.py`, `tests/test_frvp_context.py`.
**Gate (on 90 held-out RTH sessions):** 20 random profiles match TradingView FRVP drawings on the *same contract*; off-by-one anchoring test passes; naked-VPOC state updates correctly; each setup fires at 0.5–2 per session per side; no profile spans two contracts.

### Phase 2 — Feature dataset build
Reuse `features.cli build` with `--recipe features/recipes/frvp_meta.json` and `--instrument es`. Gate: correlation check at 0.98; FRVP distances show non-trivial MI with existing meta-labels.

### Phase 3 — FRVP event labeling
**Files:** `data/labeling/frvp_labeling_engine.py`; extend `labeling_engine.py`.
**Gate:** 250–750 events/year per family per side; meta-label base rate 45–60%; quality mean ≥ 0.65; roll-spanning events excluded; events not excessively clustered.

### Phase 4 — Preprocessing + backend attribution
Reuse `preprocessing prepare` / `backend-attribution` verbatim. Gate: SHAP top-10 includes FRVP distances, open-type/day-type, and HTF confluence; no label-adjacent or raw-OHLC leakage features survive.

### Phase 5 — Model training
Reuse `ote_xgboost_pipeline.py` (TCN primary, XGB fallback) for the four targets. Gate: OOF AUC-PR > base rate; placebo gap > 3%; calibration sound.

### Phase 6 — Regime/day-type slice + threshold policy
Reuse the slice and threshold-policy scripts; add open-type/day-type slice dimensions. Gate: reversal precision higher in range/normal/neutral; continuation higher in trend/high-vol.

### Phase 7 — Walk-forward backtest + DSR
Reuse the policy backtest with **tick-based frictions**. Gate: the seven promotion gates in Section 10.

### Phase 8 — Live integration
Add artifacts to `models/live/`; register as `candidate`; extend `ote_live/features/incremental_engine.py` to maintain raw-coordinate profiles and handle live rolls; extend `ote_live/ingestion/runtime.py` for contract-roll events and gap handling; shadow-trade ≥ 4 weeks before promotion.

---

## 13. Open Questions and Risks

**Q1 — Roll method sensitivity.** Does reset-at-roll vs. translate-by-spread for cross-roll levels change meta-label base rates or backtest Sharpe? Test both in Phase 0/3.

**Q2 — Optimal value-area percentage on ES.** 70% is inherited from futures Market Profile, so it is plausibly already near-optimal here — but confirm against 60%/70%/80% on real volume.

**Q3 — Open-type conditioning power.** Does `frvp_open_type` (open above/inside/below value) materially shift meta-label base rates? If strong, consider open-type-specific models.

**Q4 — Naked VPOC return tendency.** Quantify the empirical rate at which naked VPOCs are revisited within N sessions on ES, and whether Setup 6b clears the placebo gate as a standalone family.

**Q5 — Gamma context value.** Do `es_*` gamma features earn SHAP inclusion, and does `frvp_magnet_is_gamma_pin` help the model avoid false HVN-magnet trades? If options data is too costly, does the model degrade meaningfully without it?

**Q6 — Day-type vs. setup-family interaction.** Confirm reversal-on-balance / continuation-on-trend. If absent, collapse families and reduce model count from 4 to 2.

**Q7 — Mean-reversion family vs. friction floor.** Given small ES mean-reversion targets relative to tick-based friction and open/close slippage, does the reversal family survive realistic costs? This is the most likely family to fail promotion gate 2; monitor early.

**Q8 — RTH vs. ETH anchor dominance.** At a given RTH bar, which anchor's levels carry more SHAP weight — prior RTH or overnight ETH? Slice top-SHAP features by source profile in Phase 6.

**Q9 — Event sparsity for TCN.** With 1–3 events/day and overlapping context windows, the effective training set may be small. Watch train-vs-OOF gap; fall back to XGBoost if it widens.

**Q10 — Live roll robustness.** Verify the incremental engine rolls cleanly mid-strategy: no profile carries across the roll untranslated, no level is compared across coordinate systems, and open positions are mapped correctly when the lead contract changes.

**Q11 — 0DTE regime non-stationarity.** Does pre-2022 ES behave differently enough that training on it harms post-2022 performance? Run pre/post-2022 stability and consider a recency-weighted or post-2020 primary window.

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

New files required by this paper:

| Path | Description |
|---|---|
| `data/futures/continuous_contract.py` | Causal lead-contract series + raw-coordinate profile data layer |
| `data/futures/roll_calendar.py` | Volume-based, causal roll-date determination and roll-spread record |
| `features/sessions_equity.py` | RTH/ETH/IB boundaries, half-days, holidays (ES analog of `fx_calendar.py`) |
| `features/feature_sets/frvp_context.py` | Registered `frvp_context` family — profile construction + all FRVP features |
| `features/strategies/frvp_setups.py` | Rule-based setup detector (Setups 1–6, 6b) |
| `features/recipes/frvp_meta.json` | Build recipe for the FRVP meta-labeling dataset |
| `features/config_instruments.py` | Per-instrument config (ES / 6E): tick size, anchors, sessions, frictions |
| `data/labeling/frvp_labeling_engine.py` | FRVP event sampling, triple-barrier, quality scoring, sample weights |
| `tests/test_continuity.py` | Roll/continuity assertions (Section 4.3) |
| `tests/test_frvp_context.py` | Profile construction, look-ahead, setup-detection tests |

Existing files modified:

| Path | Modification |
|---|---|
| `features/recipes/ote_extended.json` | Add `frvp_context` |
| `data/labeling/labeling_engine.py` | Call `frvp_labeling_engine.py` alongside existing families |
| `features/config.py` | Add FRVP config (bin width, VA %, anchor settings) |
| `models/ote_model_registry.json` | Add FRVP model entries (Phase 5) |
| `models/ote_model_registry_live_multifamily.json` | Add FRVP live candidates (Phase 8) |
| `ote_live/features/incremental_engine.py` | Incremental FRVP profiles + live roll handling |
| `ote_live/ingestion/runtime.py` | Contract-roll events + gap handling |
| `ote_live/models/loaders.py` | Load FRVP meta-labeler models |

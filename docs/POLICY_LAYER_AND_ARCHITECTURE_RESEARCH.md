# The Policy Layer Is the Bottleneck, Not the Backbone
## A Diagnostic of the Abstain Policy, and a Prioritized Architecture Research Program

**Author:** Quantitative Research & ML Systems
**Date:** 2026-07-12
**Status:** Research design paper — diagnostic + experiment program. No implementation yet.
**Repo target:** `model_testing/ote_abstain_policy.py`, `model_testing/ote_threshold_policy.py`, `model_training/ote_training/*`
**Companion papers:** `EURUSD_OTE_Framework` (2026), `FRVP_ES_primary_6E_variant_design` (2026-06-21), `ICT_ES_META_LABELING_DESIGN`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Part I — The Abstain Policy Diagnostic](#2-part-i--the-abstain-policy-diagnostic)
   - 2.1 [The Smoking Gun](#21-the-smoking-gun)
   - 2.2 [Root Cause: The Objective Function Is Wrong](#22-root-cause-the-objective-function-is-wrong)
   - 2.3 [Rule-by-Rule Failure Analysis](#23-rule-by-rule-failure-analysis)
   - 2.4 [Structural Defects in the Policy Layer](#24-structural-defects-in-the-policy-layer)
   - 2.5 [The Diagnostic Protocol](#25-the-diagnostic-protocol)
   - 2.6 [The Redesign: From Veto to Sizing](#26-the-redesign-from-veto-to-sizing)
   - 2.7 [Acceptance Criteria and the Null Result](#27-acceptance-criteria-and-the-null-result)
3. [Part II — Model Architecture Research Program](#3-part-ii--model-architecture-research-program)
   - 3.1 [What the Literature Says at Our Scale](#31-what-the-literature-says-at-our-scale)
   - 3.2 [The Meta-Labeling Architecture Literature](#32-the-meta-labeling-architecture-literature)
   - 3.3 [Tier 1 — Cheap, High-Probability Wins](#33-tier-1--cheap-high-probability-wins)
   - 3.4 [Tier 2 — Sequence Architectures With Structural Fit](#34-tier-2--sequence-architectures-with-structural-fit)
   - 3.5 [Tier 3 — Pretraining, the Real Lever on ~2k Labels](#35-tier-3--pretraining-the-real-lever-on-2k-labels)
   - 3.6 [Deprioritized and Rejected](#36-deprioritized-and-rejected)
4. [Part III — Formulation Changes That Outrank Backbone Changes](#4-part-iii--formulation-changes-that-outrank-backbone-changes)
5. [Part IV — Experiment Program](#5-part-iv--experiment-program)
6. [Part V — Evaluation Discipline](#6-part-v--evaluation-discipline)
7. [Open Questions and Risks](#7-open-questions-and-risks)
8. [References](#8-references)

---

## 1. Executive Summary

This paper does two things.

**Part I** diagnoses the hard-abstain policy, which the `EURUSD_OTE_Framework` paper already flags as "mis-tuned" and names as "the largest available lever" for future work, but does not dissect. The diagnosis: the abstain layer is not merely mis-tuned, it is **structurally mis-specified**. It optimizes precision when the economics require expectancy; it applies binary vetoes where it should modulate size; its individual rules are unvalidated magic numbers; and — critically — the aggregate evidence shows it is removing *above-average-expectancy* trades. Under hard abstain, throughput fell 65% while net PnL fell 69%. That asymmetry is a mathematical proof that the vetoed trades were, on average, better than the retained ones. The layer is anti-selective.

**Part II** surveys the model-architecture landscape and proposes a prioritized experiment program. The conclusion there is deflationary in a useful way: at our event counts (~1,500–5,000 per family per side, ~5% positive rate), the published evidence does not support large gains from swapping backbones. The literature and our own leaderboard both show gradient boosting and small causal-convolutional models sitting near the ceiling of what the data supports. The architectures worth testing are those that either (a) attack the **sample-size constraint** (pretraining, tabular foundation models), (b) exploit **structure we currently discard** (the volume profile's shape), or (c) buy **ensemble diversity**.

**The load-bearing claim of this paper:** fixing the policy layer is a higher-expected-value use of engineering time than any architecture on the list, and it is roughly a week of work versus months. Part I should be executed before Part II is started.

---

# 2. Part I — The Abstain Policy Diagnostic

## 2.1 The Smoking Gun

From the TCN focus cohort in the OTE framework evaluation, comparing the global-threshold policy against its hard-abstain-augmented counterpart on `long_ote_tcn_v2`:

| Metric | Threshold only | + Hard abstain | Change |
|---|---|---|---|
| Event F0.5 | 0.382 | 0.846 | **+121%** |
| Trades / week | 11.8 | 4.1 | **−65%** |
| Net PnL (pips) | 14,573 | 4,469 | **−69%** |

The precision statistic more than doubled. The profit collapsed by more than two thirds. That much is already recorded in the framework paper, and the conclusion drawn there was that abstain "acts as a blunt trade suppressor."

But the paper stops one arithmetic step short of the real finding. Compare the two percentage drops:

- Trades removed: **65%**
- PnL removed: **69%**

If the abstain layer were removing *random* trades, PnL would fall by the same 65%. If it were removing *bad* trades — which is its entire purpose — PnL would fall by **less** than 65%, and could even rise. Instead PnL falls by **more** than trade count.

> **The abstained trades had a higher mean expectancy than the trades that survived. The abstain policy is not a filter. It is an anti-filter.**

Quantitatively: mean expectancy of retained trades ≈ 4,469 / 4.1 ≈ **1,090 pips per trade-week-unit**, versus a pre-abstain blend of 14,573 / 11.8 ≈ **1,235**. The removed cohort therefore carried a mean expectancy of roughly (14,573 − 4,469) / (11.8 − 4.1) ≈ **1,312** — about **20% higher** than the trades the policy chose to keep.

This is not a tuning problem. A tuning problem would show precision up and PnL down *proportionally less*, and would be fixed by loosening thresholds. This shows precision up and PnL down *disproportionately more*, which means the selection axis itself is pointing the wrong way.

**Why this happens is not mysterious.** The label geometry is triple-barrier with a target reward:risk of roughly 2:1. At 2:1, break-even hit rate is 33%. A candidate with a 45% hit rate is *comfortably profitable* and *precision-destructive*. Any objective that rewards precision will throw it away. The abstain policy is doing exactly what it was asked to do; it was asked for the wrong thing.

## 2.2 Root Cause: The Objective Function Is Wrong

The threshold search selects on:

```
selection_score = 0.60 · event_F0.5 + 0.40 · normalized_post_cost_expectancy
```

with the F-beta parameter set to **β = 0.55**, which the design explicitly describes as weighting precision over recall.

The abstain layer then sits **downstream** of that already precision-weighted threshold and applies *further* precision-improving vetoes.

So the architecture is:

```
model probability
      ↓
threshold selected on a 60%-precision-weighted objective
      ↓
abstain rules that improve precision
      ↓
trade
```

**At no point in this chain does anything maximize post-cost profit, Sharpe, or expectancy as a primary objective.** Expectancy appears once, at 40% weight, inside a score that is dominated by an F-measure. The abstain layer has no objective at all — it is a hand-written rule stack with no fitted parameters and no selection criterion.

The framework paper's own discussion section reaches the right verdict without following it to its conclusion: *"This is a cautionary tale about optimizing the wrong objective — precision is not P&L."* Correct. The fix is not to re-tune the rules. The fix is to delete the objective.

### 2.2.1 Why precision was chosen, and why the reasoning fails

The motivation for a precision-weighted objective is intuitive and, in a different setting, sound: a signal system that fires constantly and is right slightly more than half the time is operationally miserable and psychologically unusable. Event F0.5 was chosen to reflect "how a trader actually consumes signals."

But this conflates two distinct desiderata:

1. **Signal parsimony** — we want few, meaningful signals rather than a firehose.
2. **Signal accuracy** — we want the signals we emit to be right.

The labeling engine and event de-duplication already deliver (1). Structural-ATR-scaled swing detection, quality scoring, and event-level evaluation *are* the parsimony mechanism. Layering a precision objective on top to deliver parsimony a second time is double-counting — and it pays for that redundant parsimony in expectancy.

## 2.3 Rule-by-Rule Failure Analysis

The `HardAbstainConfig` / `apply_abstain_policy()` layer applies five veto families in first-match precedence order. Each is analyzed below.

### Rule 1 — `abstain_high_stress` (blocks all bars where `stress_regime == 'high'`)

**Stated rationale:** during abnormal volatility spikes, spread costs and slippage are likely to exceed the expected move.

**Why it is probably the single most destructive rule:** it contradicts the regime-slice evidence in the same framework. The per-regime analysis found that model skill is **highest in trend-aligned, high-volatility, London/overlap conditions** and **collapses in low-volatility ranges where structure is faint**. The framework's own summary of the regime table: *"the models are skilful precisely where the labeling philosophy predicts they should be — in volatile, trend-aligned, liquid-session conditions."*

Stress regime and volatility regime are not identical, but they are strongly correlated by construction. Blocking high-stress bars therefore blocks a large fraction of exactly the regime where the edge is concentrated. The rule is not filtering noise; it is filtering signal.

**The rationale is also an assumption, not a measurement.** "Spread costs are likely to exceed the expected move" is a testable proposition, and the cost model already has session-dependent spreads to test it with. It has never been tested. It should be replaced by an explicit EV computation (Rule 2, done correctly), which subsumes it: if high-stress spreads really do eat the move, the EV filter will abstain automatically, on evidence rather than on prior.

**Recommendation:** delete. Let the EV filter decide.

### Rule 2 — Expected-move filter

**Current implementation:** abstain if `probability × regime_expected_move < 2.0 × session_spread`.

Three independent defects:

**(a) It is a gross-move filter, not an expected-value filter.** It never subtracts the loss branch. It asks "is the upside big enough?" and never asks "is the upside big enough *net of the downside*?" The economically correct test, using quantities all known at signal time from the triple-barrier configuration:

```
EV = p_cal · TP_distance
   − (1 − p_cal) · SL_distance
   − entry_spread − exit_spread
   − slippage − commission

abstain iff EV ≤ margin
```

This is the *only* principled veto in the entire system, and it is not the one being applied.

**(b) `regime_expected_move` is the wrong conditioning set.** It is an unconditional average move for the regime bucket. But we are not asking "how far does price typically travel in a strong_up_high regime?" — we are asking "how far does price travel *given that FRVP Setup 3 fired at the VAH of the prior-RTH profile with this displacement magnitude*?" Those are different distributions. The regime mean is a crude prior that ignores every setup-specific feature we spent the entire feature-engineering effort constructing.

**(c) Double-counting of probability, and an unvalidated constant.** `probability × regime_expected_move` multiplies by a probability that has *already* been used once, at the threshold gate. And the `2.0×` multiplier on session spread is a magic number. No sweep of it appears anywhere in the artifacts. It could be 1.2 or 3.5 and nobody would know.

**Recommendation:** replace entirely with the EV expression above. Estimate `E[TP | hit]` and `E[SL | stop]` from the realized barrier outcomes in the training fold, conditioned on setup family and volatility regime, not from a regime-level mean move.

### Rule 3 — Cooldown (`cooldown_bars = 4`, unconditional)

**Stated rationale:** prevent burst trading around a single structural event.

**Why it is solving the wrong problem:** the backtest assumes a **fixed 120-bar exit**. The real exposure risk is therefore *concurrent overlapping positions* held across a 10-hour window — not two signals firing 20 minutes apart. A 4-bar cooldown is far too short to control concurrency (it blocks 3% of the holding period) and just deletes the second-best signal from a legitimate cluster.

Worse, signal clusters around a structural event are not necessarily redundant. In the FRVP/ICT formulation, a sweep followed by a displacement followed by a pullback into the resulting FVG is a *sequence of distinct, individually-labeled setups* around one structural event, and the later ones are often the higher-quality entries. A cooldown keyed to the first accepted signal systematically prefers the earliest, least-confirmed member of each cluster.

**Recommendation:** delete the cooldown. Replace with (i) an explicit **concurrency cap** (max N simultaneous open positions per side), and (ii) **overlap-aware position sizing** — if a new signal arrives while a correlated position is open, scale the new position down rather than vetoing it. If de-duplication within a structural event is genuinely wanted, do it at the *event* level in the labeler (where the machinery already exists), not with a blind bar counter at the policy layer.

### Rule 4 — Probability-quantile filter

**Current implementation:** abstain when probability falls below a configurable quantile of the candidate distribution.

**Why it is not a decision rule:** it is a *quota*. It removes a fixed fraction of candidates regardless of whether those candidates are good. In a month where every signal is excellent, it deletes the bottom decile of excellent signals. In a month where every signal is marginal, it retains 90% of marginal signals. It has the sign of a filter and the behavior of a rationing scheme.

It is also **redundant** with the calibrated absolute threshold (which is the correct way to express "only take signals above quality X"), and — depending on the window over which the quantile is computed — potentially **leaky**. If the quantile is computed over the full candidate distribution including the test period, the policy has seen the test set's difficulty level. This needs an audit of `apply_abstain_policy`'s quantile window.

**Recommendation:** delete. The absolute calibrated threshold already does this job, correctly, and without look-ahead risk.

### Rule 5 — Off-hours / explicit session and composite-regime blocks

**Assessment: probably the only defensible rule in the stack**, and even it should be earned rather than assumed. Off-hours spreads (3.0 pips default) are genuinely punitive and the block is cheap. But note that even here, the EV filter subsumes it: if a 3.0-pip spread kills the trade's EV, the EV filter abstains without needing a hard-coded session block. Keep it as a cheap short-circuit; do not rely on it as the mechanism.

The explicit `abstain_composite_regimes` / `abstain_composite_session_pairs` tuples are a different matter — these are hand-curated block lists. Each entry needs a counterfactual PnL justification (Section 2.5) or it should be removed.

## 2.4 Structural Defects in the Policy Layer

Beyond the individual rules, four defects are architectural.

**(a) Binary veto instead of size modulation.** A hard veto throws away all option value in a marginal trade. Given that the FRVP design already specifies fractional-Kelly sizing scaled between `P_min` and `P_max`, and the framework already computes calibrated probabilities via Platt scaling on out-of-fold predictions, we have every ingredient needed to convert vetoes into size reductions and do not use them. A trade with EV slightly above zero should be taken at 0.2× size, not skipped.

**(b) First-match precedence with no attribution.** `apply_abstain_policy()` records only the *first* matching reason in `policy_abstain_reason`. Rules later in the precedence order never get credit or blame for anything a prior rule already caught. With five compounding vetoes, we have no idea which one is doing the damage — only that collectively they remove 65% of throughput.

**(c) The rules are unfitted.** Every threshold and constant in the abstain layer (`2.0×` spread, `4` cooldown bars, the block lists) is hand-set. None is the output of a search. None has an in-fold/out-of-fold split. The rest of the pipeline is rigorous about purged walk-forward validation, embargo windows, and in-fold scaler fitting — and then the final gate before a trade is a set of hand-typed constants.

**(d) Sequential compounding against a threshold that already priced in regime quality.** The regime-conditional threshold map already raises the bar in adverse regimes (e.g. `strong_down_high` tightened to 0.80; `ranging_low` at 0.70 versus `strong_up_low` at 0.55 for longs). The abstain layer then blocks many of those same adverse-regime signals *again*. The same information is being used to reject the same trades twice. That is precisely the mechanism by which a mildly conservative system becomes a crippled one.

## 2.5 The Diagnostic Protocol

Before any redesign, run this. It is roughly one day of work, and it converts the entire preceding argument from inference into measurement.

**The counterfactual PnL attribution table.**

`policy_abstain_reason` is already recorded per candidate. For every abstained candidate in the walk-forward backtest, compute the realized post-cost PnL **as if the trade had been taken** — the outcome is already known from the triple-barrier evaluation, so no new simulation is required. Then attribute by reason.

| Abstain reason | Signals blocked | Counterfactual net PnL (pips) | Mean expectancy | Hit rate | Verdict |
|---|---|---|---|---|---|
| `high_stress` | ? | ? | ? | ? | ? |
| `off_hours` | ? | ? | ? | ? | ? |
| `blocked_composite` | ? | ? | ? | ? | ? |
| `expected_move` | ? | ? | ? | ? | ? |
| `prob_quantile` | ? | ? | ? | ? | ? |
| `cooldown` | ? | ? | ? | ? | ? |

**Decision rule:** any reason with **positive** counterfactual net PnL is destroying money and is deleted immediately, without further debate.

**Predicted result, stated in advance so it can be falsified:** `high_stress` and `cooldown` both come back strongly positive. `prob_quantile` comes back mildly positive. `expected_move` is ambiguous — it will block some genuinely uneconomic trades and some good ones, because it is directionally reasonable and quantitatively wrong. `off_hours` comes back near zero or mildly negative (i.e. correctly blocking).

**Second diagnostic — greedy ablation.** Having deleted the clearly-positive rules, evaluate each remaining rule's *incremental* contribution to net Sharpe across the 49 walk-forward folds, adding one rule at a time in order of measured value. Any rule that does not improve out-of-fold Sharpe by a margin exceeding fold-to-fold noise does not ship. Note that this must be done **out-of-fold** — the abstain rules were never subjected to purged validation, and a rule tuned on the same data it is evaluated on will always look good.

**Third diagnostic — the precision/expectancy decoupling plot.** For the retained cohort, plot realized expectancy against calibrated probability decile. If the relationship is monotone (as Platt calibration and the existing confidence-quintile analysis suggest it should be), then a *threshold* is the correct and sufficient control, and the entire hard-abstain rule stack is redundant machinery. If it is non-monotone — if, say, the top decile has *lower* expectancy because it corresponds to trades entered late in an already-extended move — that is a genuinely important finding and the abstain layer should be redesigned to target the non-monotone region specifically.

## 2.6 The Redesign: From Veto to Sizing

The replacement architecture:

```
model probability p
      ↓
Platt/isotonic calibration (already exists)
      ↓
EV = p·TP − (1−p)·SL − spread − slippage − commission
      ↓
abstain iff EV ≤ margin          ← the ONLY veto
      ↓
size = f(EV, variance)           ← fractional Kelly, already specced
      ↓
concurrency cap / exposure limit  ← replaces cooldown
      ↓
trade
```

**Design principles:**

1. **One veto, and it is economic.** `EV ≤ 0` is the only condition under which not trading is provably correct. Everything else is a sizing question. The `margin` term absorbs estimation error in `p` and should itself be fitted, not assumed.

2. **Size is continuous.** Marginal trades get small size. This preserves the right tail of the payoff distribution — which is precisely what the current precision-optimizing veto destroys.

3. **The policy objective becomes post-cost Sharpe (or expectancy with a drawdown penalty), not F0.5.** This is the single most important change in the paper. The threshold search's `0.60 · event_F0.5 + 0.40 · expectancy` should become something closer to a pure risk-adjusted return objective. Event F0.5 remains valuable as a **diagnostic** — it tells us about signal quality — but it must stop being an **objective**.

4. **Every constant is fitted, in-fold, under the existing purged walk-forward geometry.** The abstain/sizing layer gets the same validation discipline as the model.

5. **Concurrency, not cooldown.** Exposure is managed by capping simultaneous positions and scaling overlapping entries, not by blinding the system for 20 minutes.

### 2.6.1 The learned abstain (the better version of Rule 2)

The EV expression above requires `E[TP | hit]` and `E[SL | stop]`. Rather than estimating those from regime averages, **learn them**. Train a second-stage regressor on the **realized R-multiple** of each event — not the binary meta-label — and abstain when predicted R falls below the cost hurdle.

This is not an extra model. It is an **auxiliary head on the model we already train** (see Section 4.1). The MFE/MAE head *is* the abstain model. This unifies Parts I and II of this paper: the highest-value architectural change and the highest-value policy change are the same change.

## 2.7 Acceptance Criteria and the Null Result

The redesigned policy layer ships only if, across the full purged walk-forward backtest (49 folds), it beats the **no-abstain, threshold-only, Kelly-sized** baseline on:

- Net post-cost PnL, **and**
- Monthly Sharpe, **and**
- Maximum drawdown (not worse by more than a stated tolerance), **and**
- Fold-level stability (positive in ≥ 70% of folds)

**And it must be prepared to fail.** It is entirely possible — arguably the modal outcome — that the correct answer is:

> *Take every signal above the calibrated threshold. Size it by fractional Kelly. Cap concurrency. Abstain only when EV ≤ 0. There is no additional rule that beats this.*

That is a legitimate, valuable, publishable finding, and it should be accepted rather than tuned around. The failure mode to guard against is the one the current abstain policy already fell into: continuing to add rules until the metric you happen to be looking at improves.

**Present live status, for the record:** no abstain policy is written into any entry in the active production registry. The mis-tuned layer is therefore not currently costing money in live — it is an *unrealized lever*, not an active wound. This lowers the urgency but not the value.

---

# 3. Part II — Model Architecture Research Program

## 3.1 What the Literature Says at Our Scale

The operating regime is: **~1,500–5,000 events per family per side, ~5% positive rate, 5-minute bars, engineered tabular-plus-sequence features, purged walk-forward validation.** This is the exact regime in which the published evidence is most hostile to architectural ambition.

**The tabular/GBDT ceiling is real and well-documented:**

- A study of deep incremental learning on **financial temporal tabular datasets with distribution shifts** found that increasing MLP complexity does not improve performance, that the best result came from a *two-layer* MLP providing the minimum non-linearity needed to avoid degenerating into ridge regression, that **XGBoost beat MLPs across a wide hyperparameter range**, and that modern deep tabular architectures such as TabNet do not reliably beat plain MLPs.
- A 2025 comparative evaluation across XGBoost, LightGBM, N-BEATS, N-HiTS, and the Temporal Fusion Transformer on sparse, intermittent data concluded that **localized tree-based methods achieved superior performance** and that model selection should prioritize *alignment with problem characteristics over architectural sophistication*.
- A practitioner study running **13,500 model fits** across XGBoost, LSTM, self-attention variants, and LSTM→XGBoost hybrids on crypto found XGBoost won **consistently across every asset and every fold**, with the decisive factor being small walk-forward training windows (~2,000 candles). Their observation is directly applicable: at ~2,000 samples, even a 35,000-parameter transformer is already prone to overfitting.
- A 2026 study on **MNQ futures** — same instrument class, same 5-minute bar frequency, ~944 trading days — explicitly compared gradient boosting on engineered features against an LSTM on raw 5-minute return sequences, motivated by the Kronos foundation-model results, and asked whether sequential structure even exists at single-instrument scale over four years. This is the closest published analogue to our problem and should be read before any sequence-model work is authorized.

**Our own leaderboard agrees.** The training-phase leaderboard shows TCN dominating on CV AP, but the post-training, cost-aware, walk-forward Sharpe leaderboard shows **`short_reversal_xgb_v2` at monthly Sharpe 3.61 / profit factor 2.74**, ahead of `long_reversal_tcn_v3` — and shows Breakout TCN with a spectacular CV AP of 0.969 and a post-training Sharpe of 0.81. Also on record: version-to-version retraining is non-monotonic and occasionally destructive (long LSTM v2 regressed −0.311 test AP), and headline ROC-AUC of ~0.96 badly overstates usefulness at a 5.6% base rate.

**The honest prior, therefore:**

> Architecture is not currently the binding constraint. Sample size, label quality, and the policy layer are. Any architecture proposal must justify itself against that prior, not against a leaderboard of CV AP.

The architectures below are shortlisted precisely because each attacks the actual constraint — sample size, discarded structure, or ensemble diversity — rather than simply being a fancier function approximator.

## 3.2 The Meta-Labeling Architecture Literature

There is a body of work that is *specifically about the architecture of the secondary model in a meta-labeling system*, and this repo — which is a meta-labeling system — does not currently draw on it.

**Meta-Labeling: Theory and Framework** (Joubert, *Journal of Financial Data Science*, 2022). Consolidates the technique into a single framework, explains the relationship between binary-classification metrics and strategy performance, and — critically for Part I of this paper — deconstructs meta-labeling into three components, using a controlled experiment to show how each contributes to strategy metrics and what feature types belong in the model specification. The metrics-to-performance mapping is the exact material needed to justify replacing event F0.5 as an objective.

**Meta-Labeling: Architecture** (Meyer, Joubert & Alfeus, *JFDS*, 2022). The central reference for this section. It establishes several *heterogeneous architectures* for the secondary model, organized along two axes:
- **Feature-driven architectures** — exploiting how information in the data is structured and how the selected models consume that information.
- **Strategy-driven architectures** — explicitly incorporating characteristics unique to the underlying primary strategy.

It also introduces **inverse meta-labeling**, a technique for improving the *quantity and quality of the side forecasts* produced by the primary layer.

The strategy-driven framing is directly load-bearing for us. Our `setup_type`, `setup_family`, `anchor_level`, `sweep_type`, `session_phase`, and `htf_context` are **strategy artifacts**, not generic market features. The architecture paper's argument is that these deserve *dedicated input pathways* rather than being flattened into a single concatenated feature vector alongside 128 lagged technical indicators. Our current pipeline does the flattening.

**Ensemble Meta-Labeling** (Thumm, Barucca & Joubert, *JFDS*, 2023). Systematically investigates ensemble methods for the meta-labeling secondary model and finds that **ensembles are especially beneficial when the underlying data consist of multiple regimes and are nonlinear in nature.** That sentence is a description of our ES setup with its 15-cell composite regime cross-product. We currently train multiple backends and then *discard all but one* via champion selection. This paper says that is the wrong move.

Reference implementation: `github.com/hudson-and-thames/meta-labeling`.

## 3.3 Tier 1 — Cheap, High-Probability Wins

### 3.3.1 Heterogeneous ensembling instead of champion selection

**The change:** stop picking a champion. Fuse the calibrated probabilities of XGBoost + TCN (+ a third diverse learner) via rank-averaging and via a stacked meta-learner trained out-of-fold under the existing purged geometry.

**Why:** we already train all these models. We already calibrate them. We then throw away the ensemble and keep one. The ensemble meta-labeling literature says this is exactly backwards in multi-regime, nonlinear data. Our own results reinforce it: TCN and XGBoost have *different* strengths (TCN wins CV AP; reversal XGB wins post-training Sharpe), which is the textbook signature of decorrelated errors — the precondition for ensemble gain.

**Cost:** near zero. This is the first thing to run.

**Risk:** the stacker is another model with another opportunity to overfit. It must be trained strictly out-of-fold, and its parameters counted against the deflated-Sharpe correction (Section 6).

### 3.3.2 TabPFN-v2 as a tabular backend

**The change:** feed the existing XGBoost feature matrix (top-N features × window × lags, plus delta features) to TabPFN-v2 as an in-context classifier.

**Why this is the best architecture/scale fit currently available:** TabPFN-v2 is a transformer pretrained on synthetic priors that performs in-context classification with no gradient training, and it is **state-of-the-art specifically on small datasets** — the regime where classical tabular regressors start to run out. Its backbone alternates **attention across features (columns)** with **attention across samples (rows)**, meaning it reasons over the ~2,000 events as a *set* rather than fitting a fixed function to them. That is a genuinely different inductive bias from gradient boosting, and it is aimed at our actual constraint.

The TabPFN-TS work further shows the same backbone handles *temporal* structure when supplied with lightweight temporal features, at only **11M parameters**, achieving state-of-the-art covariate-informed forecasting on GIFT-Eval and fev-bench. Since we already build a temporally-featurized matrix for XGBoost, the integration is close to a drop-in.

**Critical implementation constraint:** TabPFN's "context set" *is* training data. It must be constructed strictly from the in-fold training rows, respecting the purge and embargo. A naive implementation that passes the full dataset as context is a catastrophic leak. This must be an explicit assertion in the trainer, in the same family as the existing anti-leakage assertions.

**Secondary constraints:** column and row limits on the model; inference cost scales with context size. Check both against our matrix dimensions before committing.

### 3.3.3 CatBoost / LightGBM for diversity

**The change:** add CatBoost and LightGBM as backends alongside XGBoost.

**Why:** not as replacements — as *decorrelation*. Different split-finding algorithms produce genuinely different error surfaces, which is what the ensemble consumes. CatBoost's ordered boosting and native categorical handling is a particularly good fit for the high-cardinality interactions we care about: `setup_type × session_phase × composite_regime`, which XGBoost currently sees only as one-hot noise.

**Cost:** low. Both slot into the existing Optuna + purged-CV trainer with minimal change.

## 3.4 Tier 2 — Sequence Architectures With Structural Fit

### 3.4.1 The FRVP profile-shape encoder (the highest-novelty item on this list)

**The problem with the current representation.** The Fixed Range Volume Profile is a **distribution over price** — a histogram. We currently compress it into a handful of scalar features: distance to POC, distance to VAH/VAL, HVN/LVN flags, value-area width. This discards **shape**.

A trader looking at a profile does not read "distance to POC = 1.3 ATR." They read *morphology*: is this a b-shape (long liquidation tail), a p-shape (short covering), a double distribution (two accepted regions with a thin middle), a balanced bell, or a thin LVN shelf sitting directly above price with nothing to slow a breakout? None of those are recoverable from our scalar features, and every one of them is the actual content of the auction-theoretic argument that motivated the whole FRVP program.

**The change.** Represent the profile as a fixed-length vector of volume-per-bin over price bins, **expressed relative to current price and normalized by structural ATR** (so the representation is translation- and scale-invariant, which also solves the contract-coordinate problem flagged in the ES roll-handling section). Then run a small 1D CNN — or a DeepSets / attention encoder over the set of significant levels — over that vector.

Architecture becomes two-branch:

```
[ profile-shape branch ]         [ sequence branch ]
  volume-per-relative-bin          window of bar features
  → 1D CNN / set encoder           → TCN (existing)
            \                       /
             \                     /
              → concat → head → P(profitable)
```

**Why this is the item most worth doing:** it is the only architecture on this list that is *specific to our signal* and not available off the shelf. Every other proposal here is a better general-purpose function approximator. This one adds *information the model currently cannot see*. In our experience that is where the untapped edge usually is.

**It also applies directly to ICT.** The same relative-price-bin encoding trick works for the FVG/order-block/liquidity-level landscape: encode the map of untapped levels above and below price as a vector over relative-price bins, and let a CNN learn the geometry of the liquidity landscape rather than hand-crafting distance-to-nearest-FVG scalars.

### 3.4.2 Dual-attention transformer (TLOB-style), adapted to bars

TLOB proposes a transformer whose blocks contain **both** self-attention over the temporal axis **and** attention over the feature axis, on the argument that market data requires learning temporal *and* spatial/feature dependencies simultaneously. It also replaces conventional z-score normalization with a **bilinear normalization layer**, explicitly because standard normalization fails under distribution shift at inference time.

**Two things to steal:**

1. **Feature-axis attention.** Our TCN reads out only the final timestep of the last dilated conv layer (`x[:, :, -1]`). It learns temporal patterns within each feature channel, and cross-channel mixing only implicitly through the pointwise convolutions. It structurally *cannot* learn an explicit, input-dependent interaction between, say, "distance to LVN" and "displacement magnitude" at a given bar. Feature-axis attention can.

2. **The normalization critique.** We use RobustScaler fit strictly in-fold — which is correct, and prevents leakage — but it is *static*. It is fit on the training fold and applied unchanged to a test fold that may live in a different volatility world. An adaptive normalization layer is worth an ablation on its own, independent of the attention architecture.

**Expected outcome:** modest. This is still a transformer at 2,000 events. Test it, do not bet on it.

### 3.4.3 Temporal Fusion Transformer

**Recommended not for accuracy but for architecture.** TFT has first-class, distinct slots for:

- **Static covariates** → `setup_type`, `setup_family`, `side`, `htf_context`, `anchor_level` type
- **Time-varying known** inputs → session phase, time-of-day, calendar
- **Time-varying observed** inputs → the price/feature sequence

plus a **variable selection network** yielding per-event feature attribution natively, rather than post-hoc via SHAP.

This is the closest off-the-shelf match to the **strategy-driven architecture** the Meyer/Joubert paper argues for: the strategy artifacts get their own pathway. It also gives an interpretable *gate* rather than a post-hoc explanation, which matters for a system whose entire design philosophy is "interpretable rule layer + probabilistic filter."

Expect it to be data-hungry and to underperform on raw metrics at our event counts. Run it as a candidate and an interpretability instrument, not a favorite.

### 3.4.4 Mamba / structured state-space (S4/S6) as a TCN swap

Linear-time sequence modeling with a long effective receptive field and fewer parameters than attention. Practically relevant for us: our TCN training had to **disable AMP entirely** (`resolve_amp_usage()`) after observing non-finite logits during full-data refits — a known pathology of deep dilated convolutions with float16 gradient accumulators. A state-space encoder is a drop-in replacement for the TCN encoder in the existing trainer, may not exhibit that pathology, and buys ensemble diversity even if it only matches the TCN.

**Cost:** low (encoder swap). **Expected gain:** small but positive, mostly via the ensemble.

## 3.5 Tier 3 — Pretraining, the Real Lever on ~2k Labels

**The core insight of this section:** our binding constraint is *labeled event count*, not model capacity. The literature's answer to a labeled-data constraint is not a bigger supervised model. It is **self-supervised pretraining on the unlabeled data we already have** — millions of 5-minute bars — followed by fine-tuning a small head on the sparse labeled events.

We have ~2,000 labeled events per family/side and roughly **500,000+ unlabeled ES 5-minute bars** (2015–2024). We currently use approximately 0.4% of our data.

### 3.5.1 Self-supervised pretraining of the encoder

**Options:**

- **Masked reconstruction** (MAE-style): mask random spans of the bar sequence, reconstruct them. Simple, robust, no negative-sampling machinery.
- **TF-C (Time-Frequency Consistency)**: pretrains by enforcing that time-domain and frequency-domain views of the same window embed close together. Reported to outperform eight SOTA baselines by ~15% F1 on average when fine-tuned on downstream classification. The frequency view is attractive for us because volatility-regime structure is naturally spectral.
- **TS2Vec-style hierarchical contrastive learning**: contrastive at multiple temporal resolutions, which matches our multi-timeframe (5m / 30m / 1h) design.

**The protocol:**

1. Pretrain the TCN (or Mamba, or dual-attention) encoder on **all** ES 5-minute bars 2015–2024, self-supervised, no labels. This is a one-time cost.
2. Freeze or lightly fine-tune the encoder; train a **small classification head** on the ~2,000 FRVP/ICT meta-label events.
3. Compare OOF AP and post-cost walk-forward Sharpe against the from-scratch TCN baseline.

**Leakage discipline:** the pretraining corpus must respect the walk-forward geometry. An encoder pretrained on 2015–2024 and then evaluated on a 2019 test fold has seen the future. Either (a) pretrain separately per fold on the fold's training window only (expensive but correct), or (b) pretrain once on a strictly pre-backtest window and hold everything after it as a genuine out-of-sample era. Option (b) is cleaner and cheaper; it costs some data.

**This is the highest-ceiling item in Part II.** It directly attacks the ~1,500-event floor that currently forces XGBoost as a fallback in both the FRVP and ICT design papers.

### 3.5.2 Kronos as a frozen feature extractor

Kronos is the first open-source foundation model for financial candlesticks (K-lines), a family of **decoder-only** models using a two-stage framework: a specialized tokenizer that quantizes continuous OHLCV into hierarchical discrete tokens, then an autoregressive transformer pretrained on **over 12 billion K-line records from 45 global exchanges**. Reported zero-shot results: **+93% RankIC over the leading general time-series foundation model** and **+87% over the best non-pretrained baseline** on price-series forecasting, plus 9% lower MAE on volatility forecasting. Accepted to AAAI 2026; weights and fine-tuning scripts are public.

Its motivating premise is directly relevant to us: generic TSFMs, despite scale, **underperform dedicated architectures on financial data**, because financial series have extreme noise, non-stationarity, heavy tails and regime changes that generic pretraining corpora do not contain.

**The cheap experiment (one week):**

1. At each event bar, run the preceding N-bar window through **frozen** Kronos.
2. Take the final hidden state (or a pooled representation) as an embedding vector.
3. Append the embedding to the existing XGBoost feature matrix.
4. Measure the change in OOF AP and walk-forward Sharpe.

**Reading the result:**
- Embedding adds nothing → strong evidence that our engineered features already saturate the available signal, and that the whole Tier-3 pretraining program is not worth pursuing. **This is a valuable negative result and it is cheap to obtain.**
- Embedding adds 2–3 points of AP → fine-tune Kronos on our corpus and promote this to a primary track.

**Run this before the expensive SSL program in 3.5.1.** It is a cheap read on whether pretraining helps *at all* on our instrument and timeframe, and it costs a week instead of a quarter.

## 3.6 Deprioritized and Rejected

| Candidate | Verdict | Reason |
|---|---|---|
| **LSTM** | Retain as diversity benchmark only | v2 regressed **−0.311 test AP** from v1. Documented as fragile and non-monotonic under retraining. Not a contender. |
| **Transformer trained from scratch** | Reject | 2,000 events. Overfits before it learns. The 13,500-fit crypto study found even 35k-parameter transformers overfit at ~2,000 samples. |
| **Reinforcement learning / policy-gradient sizing** | Reject for now | Sample-inefficient by orders of magnitude, and we have a well-specified supervised formulation with a clean EV expression. Revisit only after Part I is done and the sizing layer is a proven bottleneck. |
| **Limit-order-book models (TLOB, LiT, DeepLOB, HLOB)** | Reject as models; mine for ideas | We do not have LOB data. Their *architectural* ideas (dual attention, bilinear normalization) transfer; their models do not. |
| **Graph neural networks / cross-asset** | Defer | Interesting for an ES/NQ/6E cross-asset extension. Out of scope until single-instrument performance is settled. |

---

# 4. Part III — Formulation Changes That Outrank Backbone Changes

These are cheaper than any architecture swap and are likely to move the needle more. They are listed separately because they are changes to *what we predict*, not *what predicts it*.

## 4.1 Multi-task auxiliary heads (highest value/cost ratio in the paper)

**The change:** instead of a single binary logit, give the existing TCN/XGB multiple heads trained jointly:

- `P(TP before SL)` — the current meta-label (primary)
- `E[MFE / MAE ratio]` — regression (auxiliary)
- `E[bars to barrier]` — regression (auxiliary)
- `E[realized R-multiple]` — regression (auxiliary)

**Why this matters at 2,000 events.** A binary label extracts **one bit** of information per event. An event where price ran 3.4R before reversing and an event where it scraped 1.01R past the target are labeled identically. Auxiliary regression targets extract far more signal from each event and act as a powerful regularizer — the shared encoder must learn representations that support *all* the heads, which is much harder to overfit than a single bit.

**And it unifies with Part I.** The `E[realized R]` head **is the abstain model** described in Section 2.6.1. One architectural change delivers the auxiliary-task regularization *and* the economically-correct veto. This is the single highest-priority item across both parts of this paper.

## 4.2 Predict R, threshold afterward

The current labeling collapses **timeout → 0**, which is a lossy and slightly dishonest encoding: a trade that drifted sideways for 20 bars and one that got stopped out immediately are treated as identical failures. Predicting expected R directly and binarizing at inference preserves the distinction and gives the policy layer a continuous quantity to size on.

The ICT design already flags this as an open question ("timeout should initially map to 0 unless existing conventions prove otherwise"). This is the answer: don't map it to anything. Regress on R.

## 4.3 Regime-gated mixture of experts

**The evidence:** the framework found that regime-conditional thresholds "help only marginally over a well-chosen global threshold, because the global threshold already sits near the regime-blended optimum."

**The correct inference from that finding is not "regime doesn't matter."** It is that **regime information does not belong in the threshold — it belongs inside the model.** A threshold is a scalar; it can only shift a decision boundary up or down. It cannot change *which features matter* in a given regime, which is what actually differs between a strong_up_high and a ranging_low market.

**The change:** a mixture-of-experts head gated on the composite regime (or on a learned soft regime embedding), so different expert sub-networks specialize in different market states, with the gate learned end-to-end. This is the architecturally-correct expression of what the regime-threshold map was trying and failing to do, and it is exactly the setting where the ensemble meta-labeling paper says ensembles pay: multiple regimes, nonlinear data.

## 4.4 Inverse meta-labeling

The meta-model's ceiling is bounded by the primary rule layer — a limitation both the FRVP and ICT design papers explicitly flag ("quality bounded by the rule-based signal layer"). Inverse meta-labeling, introduced in the Meyer/Joubert architecture paper, is the published technique for improving the **quantity and quality of the side forecasts** coming out of that primary layer. If the FRVP/ICT setup definitions are the ceiling, this is the tool for raising the ceiling.

## 4.5 Sample weighting and the CUSUM sampler

Noted for completeness: the framework records that **CUSUM events are computed but unused as a sampler**. That is free, already-built machinery for improving event-sampling quality, sitting idle. It should be either used or removed.

---

# 5. Part IV — Experiment Program

Ordered by expected value per unit of engineering time. **Do not start Part II before Phase A is complete.**

### Phase A — Policy layer (≈ 1–2 weeks). *Blocking.*

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| A1 | Counterfactual PnL attribution table by `policy_abstain_reason` | 1 day | Identifies which rules destroy money. Predicted: `high_stress`, `cooldown` positive. |
| A2 | Delete every rule with positive counterfactual PnL | 1 day | Immediate PnL recovery. |
| A3 | Implement the correct EV filter: `EV = p·TP − (1−p)·SL − costs` | 2 days | Replaces expected-move filter with an economically sound veto. |
| A4 | Replace cooldown with concurrency cap + overlap-aware sizing | 2 days | Recovers the suppressed cluster trades. |
| A5 | Re-tune the policy objective to post-cost Sharpe (drop F0.5 from the objective; retain as diagnostic) | 3 days | The root-cause fix. |
| A6 | Fractional-Kelly sizing on calibrated `p` | 2 days | Converts remaining vetoes into size reductions. |
| A7 | Full purged walk-forward re-backtest against the no-abstain baseline | 2 days | Ship/no-ship gate (Section 2.7). |

### Phase B — Cheap architecture wins (≈ 2 weeks)

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| B1 | Ensemble existing XGB + TCN calibrated probabilities (rank-avg + OOF stack) | 1 day | Likely positive. Free. |
| B2 | TabPFN-v2 on the existing XGB feature matrix | 3 days | Biggest cheap upside. Watch context-set leakage. |
| B3 | CatBoost + LightGBM backends for ensemble diversity | 3 days | Small direct gain, real ensemble gain. |
| B4 | Multi-task auxiliary heads on the incumbent TCN (**§4.1**) | 5 days | **Highest value/cost item in the paper.** Also produces the abstain model. |

### Phase C — Structural architecture (≈ 3–4 weeks)

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| C1 | FRVP profile-shape CNN branch (**§3.4.1**) | 2 weeks | Most novel. Adds information the model currently cannot see. |
| C2 | Same encoding applied to the ICT liquidity landscape | 1 week | Extends C1 to the ICT package. |
| C3 | Regime-gated mixture of experts (**§4.3**) | 1 week | The correct version of regime conditioning. |

### Phase D — Pretraining (≈ 1 week gate, then a quarter if it passes)

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| D1 | Kronos frozen embeddings → XGBoost feature matrix | 1 week | **Cheap gate.** If AP doesn't move, kill the whole Tier-3 program. |
| D2 | *(gated on D1)* SSL pretrain encoder on all unlabeled ES bars, fine-tune head on events | 4–6 weeks | Highest ceiling. Attacks the sample-size floor directly. |
| D3 | *(gated on D1)* Mamba / dual-attention encoder swaps | 1 week each | Diversity + AMP-stability. |

---

# 6. Part V — Evaluation Discipline

Every experiment in Part IV is judged on the same gate, and the gate is not CV AP.

**Primary metrics:**
1. **Cost-aware, purged walk-forward, post-cost net PnL and monthly Sharpe.** This is the only metric that has ever agreed with reality in this repo.
2. **Event F0.5** — retained as a *diagnostic*, never as an *objective*. It goes on the report; it does not go in the selection score.
3. **Fold stability** — positive in ≥ 70% of walk-forward folds.
4. **Maximum drawdown and underwater duration.**

**Why CV AP is disqualified as the primary gate:** the record is unambiguous. Breakout TCN achieved **CV AP 0.969** and **post-training Sharpe 0.81**. Reversal XGB achieved **CV AP 0.694** and **post-training Sharpe 3.61**. The correlation between the training leaderboard and the deployment leaderboard is, on our own data, close to zero or negative. AP tells us the model ranks events well. Sharpe tells us whether we make money. They are different questions and we have been optimizing the first one.

**Deflated Sharpe is mandatory.** This paper proposes roughly a dozen architectures across four model families, two instruments, and multiple policy configurations. That is a very large multiple-testing surface. Every reported Sharpe must be **deflated** for the number of configurations tried (Bailey, Borwein, López de Prado & Zhu, 2014). Without this, we will find something that looks good, and it will be noise. The existing walk-forward machinery must be extended to track the trial count.

**Leakage assertions extend to every new component.** The existing anti-leakage assertion list (closed-bar-only FVGs, causal swings, completed HTF bars, in-fold scaler fitting, purged folds, embargo windows) must be extended with:
- TabPFN context set contains **only** in-fold training rows.
- SSL pretraining corpus respects the walk-forward boundary.
- Kronos embeddings are computed from a **causal** window only.
- The abstain/sizing layer's fitted constants are fit **in-fold**.
- Profile-shape bin vectors use only completed prior-session/prior-swing profiles, in single-contract coordinates.

---

# 7. Open Questions and Risks

**Q1. Is the abstain layer's damage confined to the OTE/EUR-USD cohort, or does it carry into ES?**
The measured numbers are from the EUR/USD OTE framework. The ES FRVP pipeline inherits the same policy code. The counterfactual attribution (A1) must be run separately on ES; the roll-boundary and 0DTE-era effects could change the answer.

**Q2. Is `stress_regime` actually correlated with `vol_regime`?**
The high-stress abstain critique in §2.3 depends on this. If stress is measured by something orthogonal to volatility (e.g. a spread-widening or gap-frequency proxy), the rule may be more defensible than argued. Measure it before deleting the rule.

**Q3. Is the probability-quantile filter leaky?**
The quantile window is not documented in the artifacts available. If it is computed over the full candidate distribution including test rows, this is a live look-ahead bug affecting every reported abstain-variant number. **Audit this first — it is a correctness issue, not a tuning issue.**

**Q4. Does the confidence-expectancy relationship remain monotone at the top?**
The confidence-quintile analysis reports monotone expectancy, which — if it holds at the decile level — implies the threshold is a sufficient control and most of the abstain stack is redundant. If it *breaks* at the top decile, that is a genuinely important non-monotonicity and the redesign in §2.6 needs an extra term.

**Q5. Will the ~2,000-event floor survive the split into four FRVP families and six-plus ICT setup types?**
Both design papers already flag small-sample risk. Adding architectures does not add events. If per-family/per-side counts fall below ~1,000, most of Part II becomes moot and the correct move is Tier-3 pretraining or family pooling with a `setup_type` feature — not a better backbone.

**Q6. Is there any abstain rule that beats "take everything above threshold, size by Kelly"?**
Stated as a risk because the answer may be no, and the project must be prepared to accept that. See §2.7.

**Q7. Multiple-testing burden.**
Executing this entire program without a deflated-Sharpe discipline will manufacture a false champion with near-certainty. The trial-count tracking (Section 6) is not optional.

---

# 8. References

**Meta-labeling**
- Joubert, J. F. (2022). *Meta-Labeling: Theory and Framework.* Journal of Financial Data Science, 4(3), 31–44.
- Meyer, M., Joubert, J. F., & Alfeus, M. (2022). *Meta-Labeling: Architecture.* Journal of Financial Data Science, 4(4), 10–24.
- Thumm, D., Barucca, P., & Joubert, J. F. (2023). *Ensemble Meta-Labeling.* Journal of Financial Data Science.
- Hudson & Thames. *meta-labeling* code base. `github.com/hudson-and-thames/meta-labeling`
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (Ch. 3 triple-barrier; Ch. 7 purged CV; Ch. 10 meta-labeling.)

**Architecture — tabular / foundation models**
- Hollmann, N., et al. (2025). *TabPFN-v2* — tabular foundation model; row-attention + column-attention backbone; SOTA on small datasets.
- Hoo, S. B., Müller, S., Salinas, D., & Hutter, F. (2025–2026). *From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting.* arXiv:2501.02945. 11M parameters; SOTA on covariate-informed forecasting (GIFT-Eval, fev-bench). Package: `tabpfn-time-series`.
- Shi, Y., Fu, Z., Chen, S., Zhao, B., Xu, W., Zhang, C., & Li, J. (2025). *Kronos: A Foundation Model for the Language of Financial Markets.* arXiv:2508.02739. AAAI 2026. 12B K-lines, 45 exchanges; +93% RankIC vs. leading TSFM. `github.com/shiyu-coder/Kronos`

**Architecture — sequence models**
- Berti, L., & Kasneci, G. (2025). *TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data.* arXiv:2502.15757. Dual (temporal + feature) attention; bilinear normalization for distribution shift.
- Bai, S., Kolter, J. Z., & Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271. (The TCN reference.)
- Lim, B., et al. *Temporal Fusion Transformer* — static covariates, variable selection networks.

**Architecture — pretraining**
- Zhang, X., Zhao, Z., Tsiligkaridis, T., & Zitnik, M. (2022). *Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency (TF-C).* NeurIPS 2022. arXiv:2206.08496. +15.4% F1 average over eight baselines.

**Evidence that deep learning does not automatically win**
- *Deep incremental learning models for financial temporal tabular datasets with distribution shifts.* arXiv:2303.07925. XGBoost > MLP across a wide hyperparameter range; TabNet does not reliably beat plain MLP.
- *Comparative Analysis of Modern Machine Learning Models for Retail Sales Forecasting.* arXiv:2506.05941. Tree-based methods beat N-BEATS / N-HiTS / TFT on sparse data; "prioritize alignment with problem characteristics over architectural sophistication."
- Mesfin, M. (2026). *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ.* arXiv:2605.17724. Closest published analogue to our problem: 5-minute futures bars, single instrument, ~944 days.
- *Why We Chose XGBoost Over LSTM for Crypto Prediction.* 13,500 model fits; XGBoost beat LSTM, attention variants, and LSTM→XGBoost hybrids consistently at ~2,000-sample training windows.

**Evaluation**
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management, 40(5), 94–107.

---

## Appendix A — The Single-Paragraph Version

The abstain policy removes 65% of trades and 69% of profit, which means the trades it removes are *better* than the trades it keeps; this happens because the entire policy stack — a `0.60·event_F0.5 + 0.40·expectancy` threshold objective followed by five hand-written precision-improving vetoes — optimizes precision in a system with a 2:1 payoff ratio, where the break-even hit rate is 33% and precision-destroying trades are routinely profitable. The fix is to replace the veto stack with a single economic condition (`EV ≤ 0`), size everything else by fractional Kelly, replace the 4-bar cooldown with a concurrency cap, and re-tune the whole layer against post-cost Sharpe. On the model side, the literature and our own leaderboard both say architecture is not the binding constraint at ~2,000 events — sample size is — so the shortlist is: ensemble what we already train (free), TabPFN-v2 (built for exactly this data scale), multi-task auxiliary heads (which double as the abstain model), a CNN over the volume-profile *shape* we currently throw away (the one genuinely novel idea here), and a one-week Kronos-embedding probe to decide whether the expensive pretraining program is worth starting at all. Judge everything on deflated, cost-aware, purged walk-forward Sharpe, because CV average precision has been actively misleading us: Breakout TCN scored 0.969 AP and a Sharpe of 0.81; Reversal XGB scored 0.694 AP and a Sharpe of 3.61.

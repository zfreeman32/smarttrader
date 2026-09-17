# Research Audit Memo: Abstention, Thresholds, and Economic Promotion in This Repository

Date: 2026-09-10. Scope: every saved artifact, journal, doc, script, and test in the repo that bears on the paper
"When Abstention Destroys Alpha: Counterfactual Abstain Attribution and Signal Promotion Contracts for Financial ML".

Classification labels used throughout: `DIRECTLY OBSERVED` (number read from a saved artifact or recomputed from a saved
trade tape), `SUPPORTED INTERPRETATION` (follows from observed numbers plus a stated assumption), `HYPOTHESIS`
(plausible, untested), `NOT ENOUGH EVIDENCE`. The evidence map (`evidence_map.csv`) carries one row per claim.

## A. Relevant research discovered

### A.1 Abstention and threshold machinery (code)

- `model_testing/ote_abstain_policy.py`: hard-abstain rule stack (high stress, off-hours, explicit regime/session/pair
  blocks, expected-move-vs-spread, probability quantile, cooldown). First-match precedence; one reason recorded per row.
- `model_testing/ote_threshold_policy.py`: global and per-regime threshold search. Selection score is
  `0.60 * event_F0.5 + 0.40 * normalized post-cost expectancy` with leave-one-year-out averaging. Event precision is a
  tolerance-matched event count, not a row-level label rate.
- `model_testing/policy_selection.py`: a non-baseline policy qualifies per fold only if its post-cost expectancy beats the
  baseline expectancy, its net PnL is positive, its event F0.5 is within 0.02 of baseline, and it clears the
  trades-per-week floor. It never requires the candidate's *total* net PnL to beat the baseline's total net PnL.
- `model_testing/ote_policy_backtest.py`: walk-forward with 3-month test windows, 2-year minimum train, 40-bar purge gap;
  thresholds and policy variant are chosen on the fold's training split and applied to the untouched test window.
  Eight acceptance gates: post-cost profitable, monthly Sharpe >= 0.80, WFE > 0.50, DSR >= 0.30, profitable-quarter
  share >= 0.60, positive-composite-expectancy share >= 0.60, largest-single-trade share < 0.10, max drawdown < 12%
  (advisory). `scripts/evaluation_contracts.py` adds the promotion-quality contract (3 trades/week floor, no
  auto-relaxed folds).
- `ote_live/policies/abstain.py`: the live abstain path. The probability-quantile cutoff is computed over a rolling
  deque of the last 512 candidate probabilities. The research path computes it over the whole evaluation frame handed to
  `apply_abstain_policy` (for a walk-forward test decision, the whole 3-month test fold). The two are not the same rule.
- `ote_live/storage/replay.py`, `ote_live/scripts/run_feature_parity_replay.py`, `scripts/audit_frvp_paper_signal_readiness.py`,
  `scripts/audit_ict_paper_signal_readiness.py`: feature-, prediction-, and decision-level replay audits, SHA-256 pinned
  artifact bundles, readiness lifecycle.

### A.2 Saved experiments that bear on the thesis

| # | Artifact | What it is |
| --- | --- | --- |
| 1 | `model_testing/reports/ote_threshold_policies/v1_v2_tcn_focus/policy_evaluation.csv` (2026-04-03) | Static held-out test split: four OTE TCN models, four policy variants each. Source of the "hard abstain doubled F0.5 and cut PnL by 69%" finding in `docs/EURUSD_OTE_Framework.docx` and the "anti-filter" diagnosis in `docs/POLICY_LAYER_AND_ARCHITECTURE_RESEARCH.md`. |
| 2 | `model_testing/reports/ote_policy_backtests/v1_v2_tcn_focus/*/policy_evaluation.csv` | Same four models, 49 walk-forward folds, per-fold test evaluation of all four policies. |
| 3 | `model_testing/reports/ote_policy_backtests/multifamily_live_v1/*/policy_evaluation.csv` (2026-05-28) | Eight OTE live-registry models, per-fold four-policy evaluation. |
| 4 | `model_testing/reports/frvp_backtests/frvp_es_primary_refresh_20260701/` | Ten FRVP ES models, walk-forward with per-fold four-policy evaluation and saved trade tapes (journal E01). |
| 5 | `model_testing/reports/ict_backtests/ict_es_primary_bootstrap_20260726_full/` | Six leakage-safe ICT ES models, same layout. |
| 6 | `model_testing/reports/probability_quantile_sweeps/frvp_ict_weak_discrimination_20260830/` | 2026-08-30 probability-quantile study: retrospective trade-level screen (4 models x 6 quantiles) plus a partial FRVP short-meta walk-forward rerun (baseline, q10, q20, q30). |
| 7 | `model_testing/reports/ict_backtests/ict_probability_quantile_candidate_sweep_20260830/` | Candidate-level walk-forward reruns for ICT long continuation q40/q50/q60 and short continuation q50/q60. |
| 8 | `model_testing/reports/ict_backtests/ict_long_continuation_q40_concentration_controls_20260830/` | Five pocket-pruning variants stacked on q40; none clears the 10% largest-trade gate. |
| 9 | `model_testing/reports/frvp_backtests/frvp_long_*_gatefix_*` (journal E02-E04) | Regime/session pair prunes on frozen FRVP models: continuation v3, long meta v3, long reversal v2. |
| 10 | `model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721/` (journal E11) | Frozen-model concentration study: recent-2y reversal baseline, q10 floor, sparse-pocket prune (the accepted contract), full-span prunes. |
| 11 | `model_testing/reports/frvp_paper_signal_bundles/frvp_es_paper_signal_20260816/run_summary.json` and `ote_live/runtime_manifests/frvp_es_paper_signal_20260816/paper_signal_validation_summary.json` (journal E12) | Fixed-live-policy sensitivity versus adaptive walk-forward; feature/prediction/decision replay parity results. |
| 12 | `ote_live/runtime_manifests/ict_es_paper_signal_20260813/paper_signal_validation_summary.json` and `model_testing/reports/ict_contract_audits/ICT_LONG_PAPER_SIGNAL_DECISION_20260813.md` | ICT replay parity, coordinate-feature mismatch that blocks the reversal model. |
| 13 | `model_testing/reports/ote_policy_backtests/experiment_a_breakout_sustained_pilot_20260621/` and `models/experiment_a_breakout_sustained_pilot_20260621/` | Classifier with CV AP 0.982 / test AP 0.996 / ROC AUC 0.9999 that loses 2,646 pips post-cost over 789 trades. |
| 14 | `model_testing/reports/ict_contract_audits/ICT_SHORT_SIDE_CONCENTRATION_AUDIT_20260728.md`, `ICT_MIN_TRADES_CONTRACT_AUDIT_20260719.md` | Overlap and concentration diagnostics; the cadence audit where admitting the +abstain variant reduced net PnL. |
| 15 | `model_testing/reports/frvp_placebo_readouts/frvp_long_continuation_xgb_v1_20260717/` | Label-shuffle placebo: real OOF AP 0.737 vs shuffled 0.500 +- 0.006 (10 shuffles). |
| 16 | `model_testing/reports/ict_leakage_control_audits/*after_embargo/` | Per-target realized event windows and embargo checks, zero boundary failures. |
| 17 | `docs/FRVP_ICT_Live_Signal_Output_Audit_20260811.md` | AP-lift-over-base-rate table; identifies the four weak-discrimination branches and recommends the quantile lever. |
| 18 | `*/breakdown_by_confidence_quintile.csv` in items 2, 4, 5 | Expectancy by within-tape confidence quintile. |
| 19 | `docs/POLICY_LAYER_AND_ARCHITECTURE_RESEARCH.md` | Prior internal diagnosis that used the phrase "anti-filter" for item 1 and proposed (but did not run) a counterfactual attribution table by abstain reason. |

Items 1 through 12 and 18 were re-read programmatically by `scripts/analyze_attribution.py`; every number quoted from
them in the paper is recomputed from the saved file and checked against the saved `summary.json` totals
(`artifacts/source_integrity_checks.csv`, `artifacts/attribution_source_manifest.csv` with SHA-256).

### A.3 Strategy families and instruments (for the reader who does not know the repo)

- OTE: EUR/USD 5-minute, Optimal Trade Entry pullback events, triple-barrier style labels with a 120-bar horizon,
  TCN and XGBoost meta-labelers. Costs in pips: session spread 1.0/1.5/1.5/2.5/3.0 (overlap/London/New York/Asia/off-hours),
  slippage 0.3, commission 0.35 per trade.
- FRVP: ES futures 5-minute, fixed-range volume-profile setups (six numbered setups, pooled into continuation, reversal,
  and meta direction families), XGBoost and TCN. Costs in ticks: session spread 1.0/1.0/1.0/1.5/2.0, slippage 0.25,
  commission 0.40 per trade; tick = 0.25 index points = $12.50.
- ICT: ES futures 5-minute, smart-money-concept setups (sweeps, fair value gaps, order blocks), same cost contract as
  FRVP, sequential-bootstrap and embargo-safe training.

## B. Strongest Anti-Filter evidence (ranked)

1. **OTE hard-abstain layer, static held-out test and 49-fold walk-forward** (`DIRECTLY OBSERVED`). Long TCN v2:
   1,801 -> 630 trades, event precision 0.354 -> 0.987, net 14,573 -> 4,469 pips; the 1,171 rejected trades carried
   +10,104 pips at 8.6 pips per trade versus 7.1 for the accepted trades. All eight TCN-focus pairs (two base policies x
   four models) show precision up, net down, rejected expectancy above accepted expectancy. Per fold, the long v2 pair is
   an anti-filter in 45 of 49 folds; the rejected population is 4,096 trades and +39,004 of +53,644 pips.
   Multifamily live-registry models repeat the pattern on the profitable branches (short reversal XGB v2: 8 of 8 folds).
2. **FRVP short meta, probability quantile q10/q20, full candidate-level walk-forward** (`DIRECTLY OBSERVED`).
   1,713 -> 1,215 / 1,042 trades, label precision 0.534 -> 0.537 / 0.546, net +1,995 -> +20 / -215 ticks. The rejected
   trades carried +1,974 / +2,210 ticks at 4.0 / 3.3 ticks each. Caveat: baseline Sharpe 0.067; the branch is a weak
   sentinel, so this is an anti-filter on a strategy that had little alpha to destroy.
3. **ICT short continuation, probability quantile q50/q60, candidate-level walk-forward** (`DIRECTLY OBSERVED`).
   165 -> 53 / 47 trades, label precision 0.521 -> 0.566 / 0.574, net +501 -> +49 / +102 ticks; rejected trades carried
   +452 / +398 ticks. Caveat: the baseline is itself dominated by one trade (largest-trade share 0.78; the +391.35 tick
   trade on 2025-01-31 has probability 0.555 and is rejected by every quantile >= q20). The filter removes the one trade
   the strategy depended on. Both baseline and filtered policies fail concentration.
4. **Cutoff-definition sensitivity** (`DIRECTLY OBSERVED`, new in this paper). For ICT short continuation, the accepted
   population's net PnL is negative or near zero at every quantile when the cutoff is taken from the fold's training
   window (-196, -85, -26, -80, -235, -3 ticks for q10..q60); the apparent q60 improvement (+530) exists only under the
   retrospective within-test-fold cutoff and the rolling prior-trade cutoff. For FRVP short meta, the same nominal q10
   gives +1,592 (retrospective), +1,611 (causal), +5,096 (rolling) ticks against a 1,829 baseline: the sign of the
   filter's effect depends on how the cutoff is computed. The saved research runs used the within-fold definition; the
   live runtime uses a rolling window.
5. **Abstain variant admitted by a looser cadence floor, ICT long reversal** (`DIRECTLY OBSERVED`). 1,286 -> 1,222
   trades; 64 rejected trades carried +623 ticks at 9.7 ticks each versus 5.9 accepted; Sharpe 1.927 -> 1.779. Precision
   did not rise (0.404 -> 0.403), so this is anti-additive rather than a strict Anti-Filter Effect.

Supporting: OTE short meta prune_v1 + q20 (rejected 192 trades, +1,542 pips at 8.0 vs 6.4 accepted; precision
0.972 -> 0.996; non-nested, 62 trades added); FRVP recent-2y reversal sparse-pocket prune (11 rejected trades, +517
ticks, precision 0.595 -> 0.619, net -12%, adopted because Sharpe rose and largest-trade share fell below 0.10, then
failed concentration again under the fixed live contract).

## C. Contradictory evidence (filtering genuinely improved economics)

1. **FRVP long meta regime/session prune v3** (`DIRECTLY OBSERVED`): 3,621 -> 603 trades, net -4,469 -> +9,216 ticks;
   the 3,062 rejected trades lost 13,651 ticks. Label precision *fell* (0.505 -> 0.491). Economics and classifier
   metrics decoupled in the opposite direction.
2. **ICT long continuation q40** (`DIRECTLY OBSERVED`): 247 -> 107 trades, net +7 -> +499 ticks, rejected -492; the
   sign is stable across all three cutoff rules and all six quantiles (every one of 18 variants beats the flat
   baseline). Not promotable: largest-trade share 0.249, and no pocket-control variant reaches 0.10.
3. **FRVP Setup-2 continuation medium/session prune** (`DIRECTLY OBSERVED`): 201 -> 104 trades, net +3,499 -> +4,344,
   Sharpe 0.708 -> 1.625, max drawdown 24.5% -> 8.4%. Fails WFE (-4.1) and concentration; shadow-only.
4. **OTE short meta regime prune v1** (`DIRECTLY OBSERVED`): 1,540 -> 577 trades, net +2,022 -> +3,904 pips; this is
   the accepted live champion policy. In the multifamily walk-forward, hard abstain also helps the weak short meta and
   short union branches (baseline net small or negative).
5. **FRVP long continuation prune v3** (`DIRECTLY OBSERVED`, utility-dependent): net -42% (rejected 1,165 trades carried
   +5,095 ticks), but Sharpe 0.604 -> 1.226, drawdown 10,201 -> 1,540 ticks, all eight gates pass. Additive under a
   drawdown-constrained or Sharpe utility, anti-additive under total net PnL. The paper treats this as the case that
   forces the utility to be declared.
6. **Confidence-quintile monotonicity where the model discriminates** (`DIRECTLY OBSERVED`): OTE long TCN v2 expectancy
   rises monotonically from 3.6 to 11.9 pips across Q1..Q5; ICT long reversal, short reversal, short meta rise mostly
   monotonically. For these models a *threshold* on the score is economically meaningful; the hard abstain rules are
   what remove profitable trades.

Census counts (nested pairs, base policy vs base policy + hard abstain, test split): walk-forward 96 comparisons, 83
precision up, 40 precision up and net down, 43 precision up and net up; static 58 comparisons, 29 anti-filter, 21
pro-filter. The effect is common, not universal, and is concentrated in the profitable branches.

## D. Evidence gaps

- **No trade-level tapes for the hard-abstain variants.** The saved backtests store trades only for the selected policy.
  For the OTE and FRVP/ICT +abstain pairs, rejected-population economics are exact aggregate differences (nested
  sets), but their distribution, concentration, and per-reason attribution cannot be reconstructed without rerunning.
- **Per-reason attribution** (which abstain rule removed which trade) was proposed in the policy-layer doc and never run.
- **No causal walk-forward rerun of the quantile lever.** The candidate-level reruns used the within-fold cutoff. The
  causal and rolling variants in this paper are trade-level re-screens of already-emitted trades, not full reruns.
- **Baseline drift between runs of the same model.** FRVP short meta produced 1,523 trades / +1,829 ticks in the
  2026-07-02 refresh run and 1,713 trades / +1,995 ticks in the 2026-08-30 sweep baseline with the same registry. The
  cause is not recorded (code changes to trade timing or spread mode between July and August are the likely source).
- **Small samples.** ICT continuation branches have 165-247 test trades and one trade can be 78% of baseline PnL.
  FRVP recent-2y reversal has 63-74 trades. Significance claims are not made for these.
- **Replay parity is demonstrated for features, predictions, and decisions on frozen bars, and for the fixed policy on
  historical predictions. It is not demonstrated for a live confirmation window**: the FRVP and ICT paper-signal
  trials are authorized but `not_started` (stale/unhealthy collector heartbeat).
- **No cross-strategy portfolio evaluation** of abstention; all results are per-branch.
- **OTE row-level label rate is not a precision proxy** (many consecutive bars inside an event zone are labelled
  positive); event precision from `policy_evaluation.csv` is used for OTE instead.

## E. Proposed headline result

The evidence supports this main empirical conclusion:

> In this repository, abstention policies that raise classifier precision frequently lower post-cost net PnL, and
> the trades they reject are often at least as profitable per trade as the trades they keep. The effect appears in
> both hard regime/session abstains (OTE: 45 of 49 folds for the flagship long model; all eight static TCN pairs) and
> in confidence-quantile filters (FRVP short meta q10/q20; ICT short continuation q50/q60), and it is not universal
> (40 of 96 nested walk-forward pairs are anti-filters, 43 are pro-filters). Whether a quantile filter helps or hurts
> can depend on how its cutoff is computed, which is precisely the detail that differs between the research and live
> code paths. The repository's existing eight-gate contract already tests post-cost expectancy, robustness, and
> concentration, but it selects alternate policies on per-trade expectancy rather than incremental value, has no
> explicit replay-parity gate inside `acceptance`, and has promoted a policy (recent-2y reversal prune) whose
> concentration pass did not survive the fixed live contract.

The paper's contribution is therefore: (i) naming and formalizing the failure mode relative to a declared utility;
(ii) Counterfactual Abstain Attribution as the diagnostic, including the cutoff-rule sensitivity check; (iii) a
Signal Promotion Contract that unifies the repo's existing gates with the two missing conditions (additive value
against the correct baseline, replay parity of the exact deployable policy).

Claims that follow from this audit and are made in the paper: B.1-B.5, C.1-C.6, the census counts, the parity table,
the sample-size ladder, and the leakage-design description. Claims deliberately not made: universality, statistical
significance of any single-branch difference, live-trading confirmation, per-reason attribution.

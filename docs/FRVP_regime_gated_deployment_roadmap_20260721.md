# FRVP Regime-Gated Deployment and Concentration Roadmap

**Date:** 2026-07-21
**Purpose:** Turn the post-friction-A/B FRVP next step into one concrete ordered work plan.

Detailed historical context for why this roadmap exists now lives in [FRVP_experiment_journal.md](FRVP_experiment_journal.md). This roadmap is the operational extraction of the latest friction, Q11, and concentration experiments rather than the full experiment history itself.

Documentation provenance:

- the half-life sweep and selective recent-regime checkpoint context comes from `E09`
- the cost-contract decision comes from `E10`
- the concentration / regime-gated deployment decision comes from `E11`
- future updates should use `docs/FRVP_experiment_record_template.md` before the roadmap is revised

## Execution update

The sequence in this roadmap was executed on July 21, 2026 in `model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721`.

What changed:

- The frozen continuation control stayed clean. `frvp_long_continuation_xgb_v1` `v3` remained the strong reference branch at `+7001.20` ticks, Sharpe `1.187`, `WFE = 2.344`, profitable-quarter share `0.625`, and paper-trading gate accepted.
- The full-span long-reversal control did not get solved by one more policy prune. Adding `strong_down_medium/asia` lifted net PnL to `+6245.15` ticks, Sharpe to `1.138`, and drawdown down to `8.57%`, but `WFE` worsened to `-4.334`. That is a cleaner sign that the full-span failure is still older-regime train-side instability, not just one last live deployment pocket.
- The recent-2y long-reversal lane did produce a real selective-deployment winner. Blocking the sparse `strong_down_high/overlap` pocket turned the branch into a new checkpoint: `63` trades, `+3677.05` ticks, Sharpe `1.480`, DSR `1.179`, `WFE = 2.162`, profitable-quarter share `0.60`, largest-single-trade share `0.0942`, and `accepted_for_paper_trading_gate = True`.
- That winning contract is now codified as `frvp_long_reversal_xgb_recent_regime_prune_v2`.

## Why this is next

The July 19, 2026 friction A/B already answered the first big question: the saved FRVP branches are not mainly waiting on one generic cost-model fix. `session_schedule` should stay the default FRVP contract, and the only notable `feature_proxy` improvements were the recent-2y long-reversal lane and the weak short-meta sentinel.

That means the next question is simpler and more practical: are the saved reversal profits broad enough to trust, or are they too concentrated in a few trades, a few quarters, or a few regime pockets?

## What "regime-gated deployment / concentration work" means

It means we do not retrain or change classifiers first. We keep the saved models fixed and test whether the policy layer can deploy them more selectively and more honestly.

In plain English:

- "Regime-gated deployment" means only allowing a model to trade in the market conditions where the saved evidence is actually good.
- "Concentration work" means checking whether the result depends too much on a tiny number of trades, quarters, sessions, or composite regimes.

## Ordered roadmap

1. Freeze the controls before changing anything.

Use these as the reference branches:

- `frvp_long_continuation_xgb_v1` with the saved `v3` prune contract
- `frvp_long_reversal_xgb_v1` full-span recency `v3` control
- `frvp_long_reversal_xgb_v1` recent-regime control

Keep `session_schedule` as the default economics contract. Keep `feature_proxy` only as a sensitivity lane for the recent-2y reversal checkpoint and the short-meta sentinel.

2. Measure exactly where the saved edge is concentrated.

For each frozen control, produce a plain summary of:

- largest-single-trade share
- top-5 and top-10 trade contribution share
- profitable-quarter share
- yearly contribution split
- session contribution split
- composite-regime contribution split
- composite-session pair contribution split

The goal is to answer two questions:

- Which pockets are carrying the result?
- Which pockets are dragging the result or making it too fragile?

3. Separate "good selective deployment" from "overfit lucky pockets."

Use the concentration readout to classify each pocket:

- keep: repeated positive contribution with enough trades
- block: repeated negative contribution or clear drawdown drag
- watch: positive but too sparse or too dependent on one outsized trade

Do not call something a deployable regime just because it had one good run.

4. Turn the slice readout into real policy rules.

Convert the keep/block decisions into explicit policy-layer filters:

- `abstain_composite_regimes`
- `abstain_session_regimes`
- composite-session pair blocks

This is the point where the repo stops using regime reports as commentary and starts using them as a tested deployment contract.

5. Rerun threshold search and walk-forward backtest with models held fixed.

Do not retrain the model here. Change only the deployment policy.

Compare every candidate against the frozen control on:

- net PnL
- Sharpe
- DSR
- max drawdown
- WFE
- profitable-quarter share
- largest-single-trade share
- positive composite expectancy share

6. Make a branch-level deployment decision.

Use the rerun results to decide, branch by branch:

- Long continuation: keep the current promotion path unless the concentration pass reveals a hidden fragility that the current package missed.
- Full-span long reversal: keep only if the selective-deployment contract improves stability or concentration enough to matter.
- Recent-regime long reversal: keep if it becomes the cleanest selective deployment, even if it stays narrower than the full-span branch.
- Short-side FRVP: keep shadow-only unless the policy-layer rerun changes the result materially.

7. Only then decide whether classifier or train-side changes are still needed.

If the best policy-layer result still fails because the edge is too concentrated or unstable, then escalate to:

- a different train-side stability lever
- risk-acceptance decisions
- setup-family splits
- classifier changes

Do not do those first. The policy-layer answer is cheaper, cleaner, and more falsifiable.

## Resulting decision

- Keep `frvp_long_continuation_xgb_v1` `v3` unchanged as the promotion-near continuation branch.
- Keep the full-span long-reversal control unchanged and treat its remaining miss as a train-side stability problem.
- Use `frvp_long_reversal_xgb_recent_regime_prune_v2` as the current selective-deployment checkpoint for reversal.
- Defer new classifier changes until the repo either accepts that recent-regime selective deployment is the right reversal contract or finds a train-side lever that improves the full-span branch without giving back the clean recent-regime behavior.

## Definition of done for this phase

This phase is complete when the repo can answer all three of these clearly:

1. Which FRVP branches have broad enough saved edge to trust operationally?
2. Which branches are only viable as selective deployment?
3. Which branches still fail even after the best honest selective-deployment policy is applied?

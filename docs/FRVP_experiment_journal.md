# FRVP Experiment Journal

This is the canonical plain-English record for FRVP research experiments. The design paper holds the strategic thesis, the audit report holds the diagnostic read, and this journal explains each saved experiment in a consistent format:

- why we ran it
- what we changed
- how we ran it
- which artifacts were produced
- what worked
- what did not work
- observations
- the result and resulting decision
- what changed in the repo or operating contract because of it

Every experiment that changes a branch decision, a live or shadow contract, or a frozen research control should get one entry here. Future FRVP experiments should be added here instead of leaving the reasoning spread across one-off notes or artifact folders.

## Documentation Standard For Future FRVP Experiments

For every new FRVP experiment, capture these fields:

- `Question:` what uncertainty we were trying to close.
- `Control:` which saved branch or policy contract we treated as the baseline.
- `Change:` the exact model, policy, cost, or train-window change we introduced.
- `Method:` how we evaluated it, including the artifact root or report root.
- `Artifacts:` the saved report roots, manifests, registries, scripts, or prepared roots that someone else should open to replay the work.
- `Result:` the key metrics and the delta versus the control.
- `What worked:` the part of the idea that genuinely improved the branch.
- `What did not work:` what stayed broken or got worse.
- `Observations:` what we learned about the family, not just the one branch.
- `Decision:` what became the new checkpoint and what the repo should do next.
- `Repo change:` what actually changed in saved branch status, live/shadow defaults, or the experiment backlog because of the result.

### Canonical Entry Template

Use this exact shape for future entries:

- `Question:`
- `Control:`
- `Change:`
- `Method:`
- `Artifacts:`
- `Result:`
- `What worked:`
- `What did not work:`
- `Observations:`
- `Decision:`
- `Repo change:`

## Timeline

The entries below cover the full FRVP experiment sequence from July 1, 2026 through July 21, 2026.

### E01. Refresh Rerun (2026-07-01)

- `Question:` were the earlier all-red FRVP economics mostly coming from the broken spread-cost contract and missing refresh validations rather than from dead signal families?
- `Control:` the pre-refresh audit conclusions and the saved refresh training artifacts.
- `Change:` reran Phase 6/7 on `frvp_es_primary_refresh_20260701` after the spread-cost fix, Setup 4 re-enable, and pooled XGBoost meta-model wiring were in place.
- `Method:` post-training evaluation on the refresh registry and prepared roots in `model_testing/reports/frvp_backtests/frvp_es_primary_refresh_20260701`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_es_primary_refresh_20260701` and the saved refresh model registries under `models/frvp_es_primary_*_20260701.json`.
- `Result:` `frvp_long_continuation_xgb_v1` became the best direct baseline at `+12465.15` ticks and Sharpe `0.604`; `frvp_long_reversal_xgb_v1` turned positive at `+2197.25`; `frvp_short_meta_xgb_v1` turned slightly positive at `+1829.05`; `frvp_long_meta_xgb_v1` stayed negative at `-4468.65`.
- `What worked:` the spread-cost correction reopened real post-cost edge in the continuation and reversal families; Setup 4 and pooled long/short meta were confirmed as real saved branches rather than code-only changes.
- `What did not work:` no branch was promotion-ready from the raw refresh rerun; long meta still overtraded badly and short-side quality remained weak.
- `Observations:` the economics layer had been masking viable FRVP branches more than the feature/label stack had. Continuation was immediately the cleanest family once costs were fixed.
- `Decision:` treat the refresh branch as the new research control and move next to targeted policy passes instead of questioning the entire FRVP stack.
- `Repo change:` the old universally negative FRVP readout was retired as the active baseline, and all later FRVP experiments now anchor to the refreshed 2026-07-01 artifacts instead.

### E02. Long-Continuation Gatefix Sequence (2026-07-02 to 2026-07-15)

- `Question:` could targeted policy-layer pruning turn the strong but noisy long-continuation baseline into a promotion-near branch without changing the classifier?
- `Control:` the refresh continuation baseline at `1609` trades, `+12465.15` ticks, Sharpe `0.604`, and max drawdown `10200.85`.
- `Change:` `v1` removed five bad London pairs; `v2` added an `overlap` abstain plus global prunes for `strong_down_medium` and `strong_up_high`; `v3` added only `ranging_medium/new_york` and `strong_up_low/asia`; the branch was later rerun under the account-equity drawdown contract on 2026-07-15.
- `Method:` policy-layer reruns on the same saved model in `frvp_long_continuation_gatefix_20260702`, `frvp_long_continuation_gatefix_v2_20260703`, `frvp_long_continuation_gatefix_v3_20260703`, and `frvp_long_continuation_gatefix_v3_20260715_accountdd`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_long_continuation_gatefix_20260702`, `..._v2_20260703`, `..._v3_20260703`, `..._v3_20260715_accountdd`, and the later promotion package / shadow bundle artifacts that reuse the saved `v3` contract.
- `Result:` `v1` lifted Sharpe `0.604 -> 0.684`; `v2` lifted Sharpe `0.684 -> 0.957`; `v3` finished at `524` trades, `+7190.40` ticks, Sharpe `1.226`, DSR `1.061`, `WFE = 2.459`, and account max drawdown `9.90%`, with `accepted_for_paper_trading_gate = True` on the refreshed 2026-07-15 rerun.
- `What worked:` base-policy pruning worked extremely well. Narrower pair-level pruning improved not just drawdown and Sharpe but also absolute net PnL versus `v2`.
- `What did not work:` the static threshold study still did not produce a qualified non-global policy winner, so threshold selection itself was not the source of the improvement.
- `Observations:` the continuation edge was broad once the obvious drag pockets were removed. The biggest gain came from better deployment discipline, not from new model signal.
- `Decision:` keep `frvp_long_continuation_xgb_v1` `v3` as the FRVP promotion baseline and treat later work on this branch as promotion hygiene, not core research rescue.
- `Repo change:` the saved `v3` continuation contract became the FRVP promotion baseline and later the July 21, 2026 extended-shadow baseline.

### E03. Long-Meta Gatefix Sequence (2026-07-03 to 2026-07-15)

- `Question:` did the pooled long-meta model have any real economic signal once the worst overtrading pockets were blocked?
- `Control:` the refresh long-meta baseline at `3621` trades, `-4468.65` ticks, Sharpe `-0.133`, and max drawdown `19662.75`.
- `Change:` `v1` globally pruned `strong_down_medium` and `strong_up_medium`; `v2` added the largest surviving bad composite/session pairs; `v3` added three narrower pairs; the saved `v3` branch was rerun under the account-equity drawdown contract on 2026-07-15.
- `Method:` targeted policy reruns in `frvp_long_meta_gatefix_v1_20260703`, `frvp_long_meta_gatefix_v2_20260703`, `frvp_long_meta_gatefix_v3_20260703`, and `frvp_long_meta_gatefix_v3_20260715_accountdd`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_long_meta_gatefix_v1_20260703`, `..._v2_20260703`, `..._v3_20260703`, and `..._v3_20260715_accountdd`.
- `Result:` baseline `-4468.65` became `v1 -650.85`, `v2 +8787.00`, and `v3 +9216.05`; the refreshed `v3` rerun stayed at Sharpe `0.595`, DSR `0.577`, `WFE = 1.897`, and account max drawdown `25.93%`.
- `What worked:` targeted pruning clearly rescued the pooled long-meta branch and proved there was real economic content in the model.
- `What did not work:` even the best saved branch still missed the Sharpe gate, largest-single-trade-share gate, and time-stability tests. The branch also stayed weak again in `2026`.
- `Observations:` long meta improved much more from a cleaner policy contract than from any evidence of a stronger classifier. The branch is valid as a research control, not as a live candidate.
- `Decision:` keep `frvp_long_meta_xgb_v1` `v3` as the saved pooled-model checkpoint and stop spending time on more micro-prunes unless the broader stability evidence changes.
- `Repo change:` long meta kept a documented saved checkpoint for comparison, but it was explicitly downgraded to research-control status rather than promotion status.

### E04. Long-Reversal Same-Environment Gatefix Sequence (2026-07-03)

- `Question:` could same-environment policy pruning alone rescue the long-reversal XGBoost branch?
- `Control:` the focused baseline at `655` trades, `+2197.25` ticks, Sharpe `0.178`, DSR `0.177`, and `WFE = -0.218`.
- `Change:` `v1` globally pruned `ranging_high`, `strong_up_low`, and `strong_up_medium`; `v2` added four pair-level prunes: `strong_up_high/new_york`, `strong_down_high/new_york`, `ranging_low/london`, and `strong_down_low/asia`.
- `Method:` policy-layer reruns in `frvp_long_reversal_gatefix_v1_20260703` and `frvp_long_reversal_gatefix_v2_20260703`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_long_reversal_gatefix_v1_20260703` and `..._v2_20260703`.
- `Result:` baseline `+2197.25` became `v1 +3546.90` and `v2 +4925.40`; Sharpe improved `0.178 -> 0.393 -> 0.659`, but `WFE` worsened to `-0.637` and then `-1.181`.
- `What worked:` broad drag pockets really were hurting the branch and pruning them improved net PnL, Sharpe, DSR, and concentration.
- `What did not work:` pruning alone did not fix the real stability problem. Full-span `WFE` stayed negative and profitable-quarter share remained weak.
- `Observations:` same-environment pruning can clean obvious deployment mistakes, but it cannot solve an older-regime train-side problem by itself.
- `Decision:` stop treating more same-environment pruning as the main answer and move the next branch to Q11-style train-side stability work.
- `Repo change:` the repo closed the same-environment reversal micro-prune loop as a primary research path and treated later policy work as supporting evidence only.

### E05. Post-2022 Small-CV Retrain (2026-07-03)

- `Question:` was pre-2022 ES history hurting post-2022 performance enough that the repo should simply retrain FRVP on post-2022 data only?
- `Control:` the refresh full-history branches for long reversal and long meta.
- `Change:` retrained `frvp_long_reversal_xgb_v1` and `frvp_long_meta_xgb_v1` on `artifacts/frvp_es_primary_post2022_20260703/phase04/prepared` using reduced CV geometry because the prepared root was much smaller than the full-history refresh branch.
- `Method:` focused post-training evaluation in `model_testing/reports/frvp_backtests/frvp_es_primary_post2022_smallcv_20260703`.
- `Artifacts:` `artifacts/frvp_es_primary_post2022_20260703/phase04/prepared` and `model_testing/reports/frvp_backtests/frvp_es_primary_post2022_smallcv_20260703`.
- `Result:` long reversal collapsed to `329` trades, `-2018.85` ticks, Sharpe `-0.396`, and `WFE = -0.873`; long meta improved to `1615` trades, `+9347.25` ticks, Sharpe `0.647`, but still had `WFE = -3.532` and profitable-quarter share `0.4167`.
- `What worked:` the experiment confirmed that the non-stationarity question was real and that long meta could sometimes improve under a more recent window.
- `What did not work:` hard truncation was a bad answer for long reversal and not a clean answer for long meta either.
- `Observations:` Q11 was real, but the correct response was not "delete old history." The repo needed a softer recency lever.
- `Decision:` do not adopt a hard post-2022 cutover. Move next to full-history recency weighting.
- `Repo change:` hard post-2022 retraining was explicitly removed from the mainline backlog and kept only as a diagnostic control.

### E06. Full-History Recency-Weighted Long-Reversal Branch (2026-07-04 to 2026-07-15)

- `Question:` could the repo keep the full history but weight recent years more heavily to reduce reversal train-side instability?
- `Control:` the same-environment `v2` reversal checkpoint at `+4925.40` ticks, Sharpe `0.659`, DSR `0.635`, and `WFE = -1.181`.
- `Change:` trained `frvp_long_reversal_xgb_v1` with a `730`-day half-life and `0.20` minimum weight floor, using the same chronology, features, and target definition; then reapplied the saved `v3` prune contract and reran the branch under the account-equity drawdown contract on 2026-07-15.
- `Method:` recency-weighted training artifacts in `artifacts/frvp_long_reversal_recency_trial1_20260703/phase04/prepared` and backtests in `frvp_long_reversal_recency_trial1_20260703`, `frvp_long_reversal_recency_trial1_v3_20260704`, and `frvp_long_reversal_recency_trial1_v3_20260715_accountdd`.
- `Artifacts:` `artifacts/frvp_long_reversal_recency_trial1_20260703/phase04/prepared`, `model_testing/reports/frvp_backtests/frvp_long_reversal_recency_trial1_20260703`, `..._v3_20260704`, and `..._v3_20260715_accountdd`.
- `Result:` the raw recency branch reached `167` trades, `+5188.45` ticks, Sharpe `0.883`; the `v3` recency branch reached `140` trades, `+6027.00` ticks, Sharpe `1.061`, DSR `0.956`, `WFE = -3.797`, max drawdown `1610.85`, and account drawdown `10.06%`.
- `What worked:` economics, drawdown, year stability, and composite cleanliness improved materially. Every saved test year from `2022` through `2026` turned positive on the recency `v3` branch.
- `What did not work:` raw classifier metrics got worse and full-history `WFE` stayed negative, so the branch still failed the full promotion read.
- `Observations:` the useful gain came from weighting the same signal toward the present, not from discovering a new feature set. This was a regime-emphasis win more than a classifier win.
- `Decision:` keep recency `v3` as the saved full-span long-reversal checkpoint and use it as the full-span control for later Q11 and concentration work.
- `Repo change:` the recency `v3` branch replaced same-environment `v2` as the saved full-span reversal control used in later Q11, friction, and concentration studies.

### E07. Rolling-Train and Recent-Regime Reversal Diagnostics (2026-07-04 to 2026-07-15)

- `Question:` was the remaining reversal miss mainly a train-side instability problem, and could a recent-regime deployment lane already be clean enough even if the full-span branch was not?
- `Control:` the recency `v3` full-span branch.
- `Change:` added bounded train windows (`3` years and `2` years), a post-2024 recent-fold evaluation, and then a recent-regime prune contract `frvp_long_reversal_xgb_recent_regime_prune_v1` that added `strong_down_medium/asia`, `ranging_low/asia`, and `ranging_medium/new_york`.
- `Method:` diagnostics in `frvp_long_reversal_recency_trial1_v3_rolltrain3y_20260704`, `frvp_long_reversal_recency_trial1_v3_rolltrain2y_20260704`, `frvp_long_reversal_recency_trial1_v3_rolltrain2y_post2024_20260704`, `frvp_long_reversal_recent_regime_prune_v1_20260705`, and `frvp_long_reversal_recent_regime_prune_v1_20260715_accountdd`.
- `Artifacts:` the bounded-window diagnostics in `model_testing/reports/frvp_backtests/frvp_long_reversal_recency_trial1_v3_rolltrain*_20260704`, the recent-fold diagnostic in `..._post2024_20260704`, and the saved recent-regime policy roots `frvp_long_reversal_recent_regime_prune_v1_20260705` and `..._20260715_accountdd`.
- `Result:` the `3`-year rolling train still had `WFE = -8.738`; the `2`-year rolling train had an unusable official `WFE = -142.392` because train annualized PnL collapsed toward zero; the post-2024 `2`-year diagnostic lifted `WFE` to `0.973`; the recent-regime `v1` branch reached `60` trades, `+3262.00` ticks, Sharpe `1.226`, DSR `1.061`, `WFE = 1.643`, and later passed the drawdown gate on the 2026-07-15 rerun.
- `What worked:` recent-regime evaluation showed the branch could behave much more cleanly when older-regime folds were removed, and the recent-regime `v1` contract produced a credible selective-deployment checkpoint.
- `What did not work:` the full-span bounded-window diagnostics did not create a new all-history winner, and the recent-regime `v1` branch still failed profitable-quarter share and concentration.
- `Observations:` the worst instability was coming from older training history, especially the 2022 regime pockets. The problem was narrower and more train-side than the broad same-environment policy studies had suggested.
- `Decision:` keep the recent-regime lane as a real checkpoint, but do not call it done yet. The next controls needed to include a matching long-meta recency sentinel and later concentration-specific work.
- `Repo change:` the repo gained a documented recent-regime reversal lane and a formal train-side-stability diagnosis, which redirected later work toward concentration control instead of more generic pruning.

### E08. Long-Meta Recency Sentinel (2026-07-05)

- `Question:` was the recency-weighting story broad enough that pooled long meta should also move to a recency-trained default?
- `Control:` the refresh long-meta `v3` checkpoint at `603` trades, `+9216.05` ticks, Sharpe `0.595`, and `WFE = 1.897`.
- `Change:` trained `frvp_long_meta_xgb_v1` with the same `730`-day half-life and `0.20` floor used in the reversal branch, then evaluated it raw and with the saved long-meta `v3` prune contract reapplied.
- `Method:` backtests in `frvp_long_meta_recency_trial1_20260705` and `frvp_long_meta_recency_trial1_gatefix_v3_20260705`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_long_meta_recency_trial1_20260705` and `..._gatefix_v3_20260705`.
- `Result:` raw recency selected `3973` trades and lost `-1271.45` ticks with Sharpe `-0.035`; reapplying the `v3` prune contract rescued it to `599` trades, `+6938.65` ticks, Sharpe `0.478`, DSR `0.469`, and `WFE = 1.974`, but that was still weaker than refresh `v3`.
- `What worked:` the saved long-meta prune contract still rescued a weak raw branch, which confirmed that the long-meta deployment contract itself was meaningful.
- `What did not work:` recency training did not beat the existing long-meta checkpoint and did not justify a family-wide rollout.
- `Observations:` long meta responds more to disciplined deployment than to the same recency retrain that helped reversal.
- `Decision:` keep recency weighting as mostly reversal-specific and keep refresh `v3` as the saved long-meta checkpoint.
- `Repo change:` the repo explicitly kept long meta on the refresh `v3` control and treated long-meta recency weighting as a closed sentinel, not a rollout candidate.

### E09. Q11 Half-Life Sweep: 548-Day and 365-Day Reversal Retrains (2026-07-19)

- `Question:` if the saved recency `v3` control still failed full-span `WFE`, would a shorter recency half-life fix the branch without changing the rest of the stack?
- `Control:` the recency `v3` full-span control and the saved recent-regime lane reproduced under the clean 2026-07-19 evaluation roots.
- `Change:` added `scripts/materialize_frvp_recency_prepared_root.py`, rebuilt reversal-only prepared roots with `548`-day and `365`-day half-lives, retrained the same target, and evaluated each branch in both full-span and recent-2y lanes.
- `Method:` evaluation roots `frvp_long_reversal_q11_control_20260719`, `frvp_long_reversal_q11_recent2y_20260719`, `frvp_long_reversal_q11_548d_control_20260719`, `frvp_long_reversal_q11_548d_recent2y_20260719`, `frvp_long_reversal_q11_365d_control_20260719`, and `frvp_long_reversal_q11_365d_recent2y_20260719`.
- `Artifacts:` `scripts/materialize_frvp_recency_prepared_root.py` plus the six saved evaluation roots under `model_testing/reports/frvp_backtests/frvp_long_reversal_q11_*_20260719`.
- `Result:` the `548`-day full-span control finished at `+4572.50` ticks, Sharpe `0.778`, `WFE = -3.429`, and account drawdown `14.41%`; the `548`-day recent-2y lane reached `+2939.25` ticks, Sharpe `0.930`, `WFE = 1.450`, but largest-single-trade share `0.1825`; the `365`-day full-span control reached `+4071.50` ticks, Sharpe `0.823`, `WFE = -2.348`, and largest-single-trade share `0.1389`; the `365`-day recent-2y lane reached `+4451.85` ticks, Sharpe `1.446`, DSR `1.167`, `WFE = 2.355`, profitable-quarter share `0.60`, and largest-single-trade share `0.1205`.
- `What worked:` the `365`-day recent-2y lane became the strongest selective-deployment checkpoint produced by the half-life sweep, and the `365`-day full-span branch improved Sharpe and drawdown relative to the longer-half-life alternatives.
- `What did not work:` no shorter-half-life retrain produced a new full-span winner, and concentration still failed on the strongest recent-2y branch.
- `Observations:` half-life tuning could reshape where the edge landed, but it did not solve the core full-span stability problem. The branch had moved from "need another generic retrain" to "need concentration or a different stability lever."
- `Decision:` close half-life-only retraining. Keep recency `v3` as the saved full-span control. Treat the `365`-day recent-2y branch as an interim selective-deployment checkpoint that could still be superseded by a cleaner policy-only concentration pass.
- `Repo change:` a reusable recency-root materialization script was added, but half-life-only retraining was removed from the mainline backlog after the 365-day lane still failed concentration.

### E10. Friction A/B: Session Schedule vs. Feature Proxy (2026-07-19)

- `Question:` were the saved FRVP branches still mainly sensitive to one wrong spread-cost assumption, and should the repo switch the default economics contract before changing classifiers?
- `Control:` the saved FRVP branches under the standard `session_schedule` cost contract.
- `Change:` held the model artifacts and regime-labeled prediction roots fixed, then reran only the economics layer under two cost arms: `session_schedule` and explicit `feature_proxy`.
- `Method:` branch comparison in `model_testing/reports/frvp_friction_studies/frvp_friction_ab_20260719`.
- `Artifacts:` `model_testing/reports/frvp_friction_studies/frvp_friction_ab_20260719`.
- `Result:` long continuation `v3` strongly preferred `session_schedule` at `+7001.20` ticks and Sharpe `1.187` versus `+1598.40` and `0.441`; the full-span long-reversal control also preferred `session_schedule` at `+6027.00` and `1.061` versus `+5764.00` and `1.023`; the recent-2y long-reversal lane improved under `feature_proxy` from `+4193.90`, Sharpe `1.429` to `+4488.55`, Sharpe `1.564`; `frvp_short_meta_xgb_v1` also improved under `feature_proxy`, while short reversal stayed negative in both arms.
- `What worked:` the A/B closed the biggest remaining economics-contract uncertainty and showed exactly where cost sensitivity mattered.
- `What did not work:` `feature_proxy` was not a universal fix and did not rescue the weak short-reversal family.
- `Observations:` cost sensitivity was branch-specific. The strongest long branches still wanted `session_schedule`, while the recent-regime reversal lane and short-meta sentinel were the only meaningful places where `feature_proxy` helped.
- `Decision:` keep `session_schedule` as the default FRVP cost contract, keep `feature_proxy` only as a sensitivity lane, and move the next FRVP work to concentration / regime-gated deployment before any classifier change.
- `Repo change:` later FRVP docs, studies, and shadow packaging now treat `session_schedule` as the default cost contract and keep `feature_proxy` as branch-specific sensitivity evidence only.

### E11. Regime-Gated Deployment and Concentration Pass (2026-07-21)

- `Question:` with friction uncertainty narrowed, could honest selective deployment and concentration control turn the saved reversal checkpoints into an operational contract without retraining the models again?
- `Control:` the three frozen `session_schedule` branches: long continuation `v3`, full-span long-reversal recency `v3`, and the recent-2y long-reversal control.
- `Change:` measured trade, quarter, session, regime, and pair concentration; reran explicit abstain lists with models held fixed; added `scripts/run_frvp_regime_gated_concentration_study.py`; codified the winning recent-regime contract as `frvp_long_reversal_xgb_recent_regime_prune_v2`.
- `Method:` study outputs in `model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721`.
- `Artifacts:` `scripts/run_frvp_regime_gated_concentration_study.py`, `model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721`, `models/frvp_es_shadow_live_registry_20260721.json`, and `ote_live/runtime_manifests/frvp_es_shadow_20260721`.
- `Result:` the continuation control stayed strong at `+7001.20` ticks, Sharpe `1.187`, `WFE = 2.344`, and accepted gate `True`; the full-span reversal prune that added `strong_down_medium/asia` improved net PnL to `+6245.15`, Sharpe to `1.138`, and drawdown to `8.57%`, but worsened `WFE` to `-4.334`; the winning recent-2y contract blocked `strong_down_high/overlap` and reached `63` trades, `+3677.05` ticks, Sharpe `1.480`, DSR `1.179`, `WFE = 2.162`, profitable-quarter share `0.60`, largest-single-trade share `0.0942`, and `accepted_for_paper_trading_gate = True`.
- `What worked:` selective deployment solved the recent-regime concentration problem cleanly enough to produce an accepted checkpoint, and the continuation control remained stable as a frozen reference branch.
- `What did not work:` one more policy prune did not repair the full-span reversal branch. `WFE` got worse even when net PnL and drawdown improved.
- `Observations:` the full-span reversal miss is still a train-side stability problem rooted in older regimes, not one last bad live pocket. The recent-regime concentration problem was real but solvable with an explicit sparse-pocket block.
- `Decision:` keep the full-span reversal control unchanged, use `frvp_long_reversal_xgb_recent_regime_prune_v2` as the current selective-deployment checkpoint, and only escalate to a new train-side lever if the repo still needs a better full-span reversal answer than this narrower accepted contract provides.
- `Repo change:` the July 21, 2026 shadow bundle became the default FRVP runtime package, the recent-regime reversal lane became the operational selective-deployment contract, and new full-span reversal work stayed closed unless shadow evidence later proves the narrower contract insufficient.

## Recurring Lessons From The FRVP Experiment Cycle

- The highest-value FRVP fixes were policy-layer and economics-contract fixes, not classifier swaps.
- Long continuation became promotion-near once the repo stopped overdeploying it.
- Long reversal split into two separate problems: a recent-regime deployment problem that can be cleaned up at the policy layer, and a full-span stability problem that still lives on the training side.
- Long meta proved the pooled-model idea is real, but it also showed that not every positive research branch deserves more optimization.
- Broad statements like "friction is wrong" or "just cut to post-2022" were weaker than branch-specific evidence. The useful answers were narrower than that.

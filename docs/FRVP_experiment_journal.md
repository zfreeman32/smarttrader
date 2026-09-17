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
- `Result:` the continuation control stayed strong at `+7001.20` ticks, Sharpe `1.187`, `WFE = 2.344`, and accepted gate `True`; the full-span reversal prune that added `strong_down_medium/asia` improved net PnL to `+6245.15`, Sharpe to `1.138`, and drawdown to `8.57%`, but worsened `WFE` to `-4.334`; the winning recent-2y contract blocked `strong_down_high/overlap` and reached `63` trades, `+3677.05` ticks, Sharpe `1.480`, DSR `1.179`, `WFE = 2.162`, profitable-quarter share `0.60`, canonical largest-single-trade share `0.099985`, and `accepted_for_paper_trading_gate = True`.
- `What worked:` selective deployment solved the recent-regime concentration problem cleanly enough to produce an accepted checkpoint, and the continuation control remained stable as a frozen reference branch.
- `What did not work:` one more policy prune did not repair the full-span reversal branch. `WFE` got worse even when net PnL and drawdown improved.
- `Observations:` the full-span reversal miss is still a train-side stability problem rooted in older regimes, not one last bad live pocket. The recent-regime concentration problem was real but solvable with an explicit sparse-pocket block.
- `Decision:` keep the full-span reversal control unchanged, use `frvp_long_reversal_xgb_recent_regime_prune_v2` as the current selective-deployment checkpoint, and only escalate to a new train-side lever if the repo still needs a better full-span reversal answer than this narrower accepted contract provides.
- `Repo change:` the July 21, 2026 shadow bundle became the default FRVP runtime package, the recent-regime reversal lane became the operational selective-deployment contract, and new full-span reversal work stayed closed unless shadow evidence later proves the narrower contract insufficient.

### E12. Controlled Paper-Signal Roster Decision (2026-08-16)

- `Question:` which frozen FRVP contract can move from the July shadow bundle into a controlled, no-order paper-signal confirmation window?
- `Control:` the July 21 continuation `v3` and recent-regime reversal walk-forward contracts, plus the immutable July shadow artifacts.
- `Change:` audited exact trade overlap, replayed each final fixed live threshold and hard-filter contract, repaired the missing reversal `strong_down_high/overlap` filter, and built the immutable `frvp_es_paper_signal_20260816` bundle with readiness and ledger controls.
- `Method:` exact `(entry_datetime, source_row_idx)` overlap audit; fixed-policy sensitivity over frozen labeled OOF/test predictions using original chronological folds, ES scheduled costs, and 120-bar close markouts; 8,000-bar roll-boundary live-feature replay; persisted audit replay; immutable builder and focused runtime tests.
- `Artifacts:` `scripts/build_frvp_paper_signal_bundle.py`, `scripts/audit_frvp_paper_signal_readiness.py`, `models/frvp_es_paper_signal_registry_20260816.json`, `ote_live/runtime_manifests/frvp_es_paper_signal_20260816`, `ote_live/policy_artifacts/frvp_es_paper_signal_20260816`, and `model_testing/reports/frvp_paper_signal_bundles/frvp_es_paper_signal_20260816`.
- `Result:` continuation and reversal share zero exact entries. The fixed continuation global-`0.70` contract produced `180` trades, `+2473.00` ticks, Sharpe `0.738`, DSR `0.704`, drawdown `7.64%`, profitable-quarter share `0.5833`, and largest-trade share `0.1453`, missing three gates. The corrected fixed reversal global-`0.60` contract produced `50` trades, `+2567.50` ticks, Sharpe `1.398`, DSR `1.148`, drawdown `5.39%`, profitable-quarter share `0.60`, and largest-trade share `0.1287`, missing concentration only. Historical feature parity matched `96/96` cells and the exact persisted prediction/signal replay matched after resolved regime context was added to audit metadata.
- `What worked:` the reversal policy bug was repaired from the authoritative accepted backtest; the corrected reversal remained economically strong under the final static policy; exact feature, prediction, and decision replay passed; the 120-bar confirmation ledger and exact-bundle launch guard are ready.
- `What did not work:` continuation's headline walk-forward result was not robust to the exact final static live contract, and reversal still did not earn full promotion because its fixed-policy largest-trade share remained above `0.10`. The current ES heartbeat is stale and unhealthy, so no confirmation clock could start.
- `Observations:` promotion-quality walk-forward results and final fixed live-policy sensitivity answer different questions. The latter is now recorded explicitly instead of allowing a mixed walk-forward policy headline to stand in for the deployable contract.
- `Decision:` make `frvp_long_reversal_xgb_v1` the sole active controlled paper-signal lane, specifically to validate concentration for at least 28 calendar days. Keep continuation and the other nondeprecated branches candidate/shadow; keep short reversal deprecated. Active permits signals, notifications, and independent markouts only—never broker orders or fills. Defer the human same-contract TradingView signoff only for this controlled trial; full promotion remains unauthorized.
- `Repo change:` the decision is encoded in a new immutable bundle, a reversal-only 120-bar markout ledger, a read-only readiness audit, and content-aware collector guards. The active model/scaler/calibrator/config/training-summary bytes are SHA-256 pinned before deserialization; restart restores the four-bar cooldown from the last exact-manifest emit; and the controlled dashboard is exact-manifest scoped and reads the 120-bar ES ledger. Lifecycle is authorized pending readiness but remains `not_started` until a fresh healthy authenticated paper feed passes the final guard.

### E13. Per-Setup Target and Artifact Plumbing (2026-08-27)

- `Question:` could the repo isolate the six numbered FRVP setups for clean training/evaluation without changing the existing pooled controls or accidentally launching a retrain?
- `Control:` the existing long/short reversal, continuation, and meta targets; pooled-family sample weights/concurrency; saved Phase-3 labels and event tape from `frvp_es_primary_refresh_20260701`.
- `Change:` added 12 family-rich targets: `frvp_reversal_setup1`, `frvp_continuation_setup2`, `frvp_continuation_setup3`, `frvp_reversal_setup4`, `frvp_continuation_setup5`, and `frvp_reversal_setup6`, each split long/short. The four pooled direct targets and two meta targets remain available as controls.
- `Method:` additive event routing in `data/labeling/frvp_labeling_engine.py`; centralized target contract in `frvp/target_lanes.py`; expanded Phase-4 preprocessing/backend attribution and Phase-2 setup reporting; explicit `pooled | setup | all` XGBoost target selection; generic candidate-registry ID verification. Setup targets deliberately inherit pooled-family quality, weight, and concurrency so this first comparison changes only target membership.
- `Artifacts:` code and tests only. No prepared root, model, policy, registry artifact, or live bundle was generated or replaced.
- `Result:` artifact-backed replay covered `666,769` label rows and `28,667` events. All `28` legacy pooled label/helper columns matched with zero mismatches, and all `12` setup lanes matched exact usable/positive event counts. Focused labeling/Phase-4/registry tests passed `28/28`; preprocessing/backend tests passed `23/23`.
- `What worked:` the generic preprocessing, trainer, registry, and post-training evaluation contracts already supported family-rich target names once the hard-coded target inventories were expanded. Embedding `reversal`/`continuation` in setup slugs also preserved the trainer's current continuation-calibration rule.
- `What did not work:` the initial baseline test collection exposed a pre-existing eager-import cycle in the FRVP feature shim; the shim now imports the real FRVP context lazily. The broader verification run still has one unrelated existing failure in threaded feature-builder numeric comparison because the current feature matrix contains object-typed numeric columns.
- `Observations:` Setup 4 remains the only clearly thin lane (`380` usable long and `412` usable short events). Its separate training geometry must be chosen explicitly rather than weakening all setup-model controls. Setup-specific uniqueness weighting is also a separate A/B; it was intentionally not bundled into this routing change.
- `Decision:` close the split-first plumbing item. Keep all pooled lanes as regression controls, run the standalone Setup 4 retest next, and do not combine setup targets until separate walk-forward evidence supports it.
- `Repo change:` setup-specific branches can now be prepared, attributed, trained selectively, registered as IDs such as `frvp_long_reversal_setup1_xgb_v1`, and evaluated through the existing post-training stack. No live/paper roster or execution authority changed.

### E14. Setup-Specific Long Rerun Wave 1 and V2 Optimization (2026-08-29)

- `Question:` after the per-setup FRVP target plumbing landed, could the strongest long branches beat their pooled controls when rerun on setup-specific targets?
- `Control:` pooled long continuation `frvp_long_continuation_gatefix_v3_20260715_accountdd`, pooled full-span long reversal recency `v3`, the accepted recent-regime reversal checkpoint, and the first setup-specific rerun log `frvp_setup_specific_long_rerun_20260829_output.txt`.
- `Change:` materialized the setup-aware Phase-4 prepared root once, then reran long continuation as S2/S3/S5 and long reversal as S1/S4/S6. The v2 runner reduced continuation/S4 CV geometry enough for thin setup lanes, split continuation training by setup, kept S1 recency-730 as the benchmark/challenger, and added an S6 recall-oriented recency challenger.
- `Method:` sequential PowerShell runner `scripts/run_frvp_setup_specific_long_rerun_plan_20260829.ps1`, v2 log `frvp_setup_specific_long_rerun_20260829_output_v2.txt`, setup-aware prepared root `artifacts/frvp_es_primary_setup_targets_20260829/phase04/prepared`, full-span setup registry `models/frvp_es_primary_model_registry_long_setup_fullspan_20260829.json`, and optimized recency registry `models/frvp_es_primary_model_registry_long_reversal_setup_recency730_opt_20260829.json`.
- `Artifacts:` `models/frvp_long_continuation_setup2_xgb_20260829`, `models/frvp_long_continuation_setup3_xgb_20260829`, `models/frvp_long_continuation_setup5_xgb_20260829`, `models/frvp_long_reversal_setup_fullspan_rerun2_xgb_20260829`, `models/frvp_long_reversal_setup4_xgb_20260829`, `models/frvp_long_reversal_setup1_recency730_opt_xgb_20260829`, `models/frvp_long_reversal_setup6_recency730_recall_xgb_20260829`, and matching reports under `model_testing/reports/frvp_backtests/frvp_long_setup_fullspan_20260829`, `.../frvp_long_reversal_setup_recency730_opt_20260829`, and `.../frvp_long_reversal_setup_recency730_opt_recent2y_20260829`.
- `V1 result:` preprocessing completed and S1/S6 recency trained, but continuation failed because S2 had `849` development rows versus `932` required, S4 failed with `278` development rows versus `312` required, and downstream full-span/S4 evaluation failed because missing model IDs were not in the registry.
- `V2 result:` continuation and S4 training completed. Full-span S2 continuation was the best new setup lane at `201` trades, `+3499.35` ticks, expectancy `+17.41`, PF `1.399`, Sharpe `0.708`, DSR `0.614`, `WFE = -2.004`, max DD `24.5%`, and gate `False`. S5 continuation reached `1217` trades and `+5255.95` ticks but had Sharpe `0.281` and max DD `82.3%`. S3 continuation lost `-2964.25` ticks. Full-span S1 reversal was only `+178.90` ticks, and S6 reversal lost `-2897.15` ticks.
- `Recency result:` the optimized S1/S6 recency rerun did not improve realized economics. S1 remained `+922.05` ticks full-window and `+1524.55` ticks recent2y, but Sharpe, DSR, WFE, profitable-quarter share, positive composite-expectancy share, and drawdown all failed promotion requirements. S6 remained negative in both windows (`-1961.25` full-window and `-2172.80` recent2y).
- `Failures:` the v2 test step still false-failed on local pytest temp/cache permissions; recency prepared-root materialization false-failed because roots already existed; S4 policy backtest produced only one fold while the command required four. None of these changed the training/economic read, but they need cleanup before the next run.
- `What worked:` the setup-specific target stack is operational end to end, and S2 continuation appears to contain real signal. Its yearly read was `2024 = -1861.65`, `2025 = +3744.65`, and `2026 = +1616.35`, which points to a policy-layer cleanup candidate rather than immediate retraining.
- `What did not work:` no setup lane beat the deployed continuation benchmark (`524` trades, `+7190.40` ticks, Sharpe `1.226`, DSR `1.061`, `WFE = 2.459`, accepted gate `True`) or the accepted recent-regime reversal checkpoint (`63` trades, `+3677.05` ticks, Sharpe `1.480`, DSR `1.179`, `WFE = 2.162`, accepted gate `True`). S6 recall-oriented hyperparameters did not fix realized economics.
- `Observations:` splitting families by setup improved diagnosis more than deployment quality. S2 is promising but needs targeted deployment constraints. S5 is too broad and drawdown-heavy. S3 is classifier-good but economics-bad. S4 is too sparse for promotion-quality fold breadth. S1/S6 setup-specific reversal did not repair the pooled reversal stability problem.
- `Decision:` keep current deployed/paper roster unchanged. Do not promote any setup-specific model from this wave. Move next to script-noise cleanup, S2 policy-only pruning, S5 drawdown-reduction pruning, S6 threshold/slice analysis, and one-fold explicit S4 research reporting. Do not spend more blind training budget on S6 until policy/threshold slicing has been exhausted.
- `Repo change:` setup-specific prepared/training/evaluation artifacts now exist and should remain research candidates only. The next setup-specific work is policy evaluation and runner hygiene, not live/paper deployment.

### E15. Setup-Specific Round 2 Policy Pruning and Leaderboard Decision (2026-08-30)

- `Question:` after the second setup-specific experimentation round, were any numbered FRVP setup targets strong enough for promotion or at least controlled paper observation?
- `Control:` E14 setup-specific full-span and recency results, the deployed/paper roster from the August controlled paper-signal bundle, pooled long continuation `frvp_long_continuation_gatefix_v3_20260715_accountdd`, and the accepted recent-regime reversal checkpoint.
- `Change:` held the trained setup-specific XGBoost models fixed, reviewed S6 threshold/slice behavior, ran an explicit one-fold S4 research backtest, and built a setup-specific model leaderboard across the saved full-span, recency, optimized-recency, and policy-pruned report roots.
- `Method:` policy-only S2 medium/session pruning, S5 repeated-drawdown pruning, S6 optimized recency policy/slice review, S4 `--min-folds 1` research backtest, and consolidated leaderboard review in `docs/FRVP_setup_specific_model_leaderboard_20260830.md`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_long_setup2_medium_session_prune_20260829`, `model_testing/reports/frvp_backtests/frvp_long_setup5_repeated_drawdown_prune_promotion_20260829`, `model_testing/reports/frvp_backtests/frvp_long_reversal_setup_recency730_opt_20260829`, `model_testing/reports/frvp_backtests/frvp_long_reversal_setup_recency730_opt_recent2y_20260829`, `model_testing/reports/frvp_backtests/frvp_long_setup4_fullspan_1fold_research_20260829`, and `docs/FRVP_setup_specific_model_leaderboard_20260830.md`.
- `Result:` S5 pruned became the top absolute-PnL setup lane at `724` trades, `+12966.40` ticks, expectancy `+17.91`, PF `1.391`, Sharpe `0.912`, DSR `0.859`, max DD `38.59%`, and `WFE = 1.723`, but still failed drawdown and single-trade concentration. S2 pruned became the cleanest risk-adjusted setup lane at `104` trades, `+4344.40` ticks, expectancy `+41.77`, PF `2.265`, Sharpe `1.625`, DSR `1.153`, max DD `8.36%`, but failed `WFE = -4.127` and concentration. S1 recency recent2y stayed modest at `+1524.55` ticks with low Sharpe and negative WFE. S4 one-fold research was positive at `5` trades and `+392.75` ticks, but the sample was too thin and one trade contributed `70.36%` of total PnL. S6 remained negative in both optimized recency windows, and S3 stayed economics-negative.
- `What worked:` policy-only pruning created useful forward-observation candidates without spending more training budget. S2's drawdown and expectancy profile became credible enough to shadow, and S5's broad raw continuation signal became much more economically concentrated after repeated-drawdown pruning.
- `What did not work:` no setup-specific lane cleared the full promotion/paper gate stack. S5 still carries unacceptable historical drawdown, S2's negative train-side WFE remains unresolved, S4 lacks fold/trade breadth, and S6 threshold slicing did not reveal a credible rescue path because composite thresholds were not data-sufficient and realized economics stayed negative.
- `Observations:` the right next move is not accepted paper deployment, but quarantined shadow-paper observation for the two informative continuation lanes. S2 should test whether its apparently recent-regime edge is real forward behavior or instability; S5 should test whether the pruned high-PnL lane can behave without reproducing its historical drawdown. S4 can be logged only for event collection.
- `Backend attribution note:` XGBoost backend attribution was not skipped for the setup-specific prepared root. The root contains `backend_attribution_summary.json`, per-target `backend_attribution_summary_xgboost.json`, `feature_importance_merged_xgboost.csv`, and `shap_feature_stats_xgboost.csv`. What was skipped was standalone TCN attribution/training and extra post-hoc attribution reruns, intentionally, because this wave was an XGBoost-only target-routing/policy experiment and the recency roots only reweighted `train`/`val` sample weights.
- `Decision:` do not promote any setup-specific FRVP model and do not add them to the accepted paper roster. Create a separate shadow-paper watchlist only if it is isolated from accepted paper reporting: S2 pruned as highest priority, S5 pruned promotion-quality as medium priority, and S4 one-fold as optional log-only event collection. Keep S1, S3, and S6 out of shadow paper for now.
- `Repo change:` added `docs/FRVP_setup_specific_model_leaderboard_20260830.md` as the traceable leaderboard/recommendation artifact. The setup-specific continuation lanes are now classified as shadow-observation candidates, while setup-specific reversal remains research-only or paused.

## Recurring Lessons From The FRVP Experiment Cycle

- The highest-value FRVP fixes were policy-layer and economics-contract fixes, not classifier swaps.
- Long continuation became promotion-near once the repo stopped overdeploying it.
- Long reversal split into two separate problems: a recent-regime deployment problem that can be cleaned up at the policy layer, and a full-span stability problem that still lives on the training side.
- Long meta proved the pooled-model idea is real, but it also showed that not every positive research branch deserves more optimization.
- Broad statements like "friction is wrong" or "just cut to post-2022" were weaker than branch-specific evidence. The useful answers were narrower than that.

## 2026-09-13 - A1 collection baseline preservation

- `Why:` prevent legacy observations, corrected inputs, and replacement model artifacts from silently sharing performance reports.
- `What / how:` verified the existing 250-file snapshot and captured a separate 952-file current snapshot spanning all 21 audited ES models, with SHA-256 object verification; expanded collection identity inputs and enforced collection separation in dashboard markout reports.
- `Result / what worked:` both snapshots have no missing objects; 78 focused collection/dashboard/paper-guard tests pass.
- `What did not work:` four older operator-tooling tests require a missing local OTE champion manifest. Self-contained collection-report tests pass.
- `Observations:` the current preservation bytes do not prove September audit-era artifact identity; that remains B2. Full evidence, manifest hash anchors, and report usage are in `docs/ES_A1_collection_baseline_20260913.md`.
- `Decision:` A1 is complete. Keep ES shadow-only and all existing economics, drawdown, concentration, promotion, and paper-trial readiness restrictions. No scored collection or paper trial was started; no thresholds were tuned. A2 onward and B1-B5 remain separate prerequisites.

## 2026-09-14 - A3 HTF alignment and live input verification

- `Why:` distinguish real event-dependent missingness from absent helper producers and stale HTF inputs before prospective shadow qualification.
- `What / how:` audited the shared HTF producer, training-only confluence helpers, cached engine metadata, lag/sequence inputs, and persisted runtime decisions. Hardened source-slot/finite-value checks, transform-history classification, and latest-source timestamp matching; versioned the contract as `es-causal-inputs-v2`.
- `Result / what worked:` 156 broad regression tests passed, followed by 33 focused contract tests after final attestation coverage (157 distinct tests). The real processor/SQLite test preserves a fallback score with its diagnostic reasons and forces shadow mode even with an otherwise eligible bar. Eight non-retired roster models select missing live helpers, including six FRVP models.
- `What did not work:` count/endpoints alone could accept malformed HTF constituents, and event-like transformed names could excuse missing warmup. Both were corrected. One old test expected the column list before existing provenance additions; its expectation was updated.
- `Observations:` the inventory covers 20 non-retired models plus retained deprecated FRVP short reversal. Eighteen non-retired models lack corrected-input lineage approval; S1/S4 still require runtime and policy checks. A fallback zero cannot prove a producer observed no confluence. Full evidence and commands: `docs/ES_A3_input_contract_20260914.md`.
- `Decision:` A3 is complete; affected models remain diagnostic-only pending B2. Preserve economics, drawdown, concentration, promotion, S4 policy, and paper-trial readiness restrictions. No collector, paper trial, or qualified period was started; no model artifacts or thresholds were changed.

## 2026-09-15 - A9/A10 app research priorities and S4 candidacy pause

- `Why:` make the intended FRVP comparisons visible while preserving older economics, drawdown, concentration and readiness evidence; avoid treating sparse specialist coverage as failure or short-S5 results as long-model evidence.
- `What / how:` added the versioned `es-frvp-research-priorities-v1` roster. Highlighted short-continuation TCN and long-continuation XGB, exploratory long S3 and matching long-S5 collection. Moved long reversal, both meta models and long S1/S2/S6 to background Research presentation. Persisted roster metadata with shadow evaluations and collection identity. Added an explicit S4 candidate rejection independent of its provisional policy flag; retained retired short reversal and its artifacts.
- `Result / what worked:` 183 regression tests passed, including SQLite persistence for all twelve roster entries, a complete-policy S4 rejection, opposite-side S5 exclusion, default versus Research rendering, audit replay, research ledger and paper guards. Headless Chrome verified the visible priorities and Research toggle using an isolated synthetic database.
- `What did not work:` S4 still has no reviewed model-specific policy: its current provisional policy uses S2 through direction fallback. This task pauses candidacy and retains diagnostics; it does not invent or approve a replacement policy.
- `Observations:` coverage is selected same-side setup observations, not resolved simulated outcomes. Lack of matching S1/S2 examples remains insufficient evidence. Short-continuation TCN's older approximately -0.38 Sharpe and 109.2% reference-account maximum drawdown still bar promotion despite research prominence. Long S3 is exploratory only. Detailed implementation, limits and validation: `docs/ES_A9_A10_research_priorities_20260915.md`.
- `Decision:` A9/A10 are complete as app changes. Keep all prerequisite, economics, drawdown, concentration and promotion gates. Preserve frozen artifacts/policies and the parked four-week paper trial. No collector restart, scored period, threshold adjustment, broker order or paper activation occurred. B5's full prospective eligibility and promotion contract remains pending.

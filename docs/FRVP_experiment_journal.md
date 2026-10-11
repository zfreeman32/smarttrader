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

<a id="frvp_experiment_record_template"></a>

### FRVP Experiment Record Template

*Merged 2026-10-09 from `docs/FRVP_experiment_record_template.md`.*

Use this template whenever an FRVP experiment changes a branch decision, a live or shadow contract, a saved research control, or the backlog order.

#### Required Fields

- `Question:` what uncertainty are we trying to close?
- `Control:` which saved branch, policy, or cost contract is the baseline?
- `Change:` what exactly changed in the model, policy, training window, cost layer, or evaluation contract?
- `Method:` how was it run?
- `Artifacts:` which report roots, prepared roots, scripts, manifests, or registries should someone open to verify the work?
- `Result:` what happened numerically versus the control?
- `What worked:` what genuinely improved?
- `What did not work:` what stayed broken or regressed?
- `Observations:` what did we learn about the family or stack?
- `Decision:` what checkpoint, contract, or backlog decision follows from the result?
- `Repo change:` what changed in saved defaults, frozen controls, shadow packaging, or documentation because of this experiment?

#### Copy/Paste Skeleton

```text
### EXX. Experiment Name (YYYY-MM-DD or date range)

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
```

#### Logging Rule

If an experiment produces a new checkpoint, closes a research path, changes a live or shadow contract, or materially changes the backlog order, add it to `docs/FRVP_experiment_journal.md` immediately instead of leaving the reasoning only in artifact folders.

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
- `Method:` policy-only S2 medium/session pruning, S5 repeated-drawdown pruning, S6 optimized recency policy/slice review, S4 `--min-folds 1` research backtest, and consolidated leaderboard review in `docs/FRVP_experiment_journal.md#frvp_setup_specific_model_leaderboard_20260830`.
- `Artifacts:` `model_testing/reports/frvp_backtests/frvp_long_setup2_medium_session_prune_20260829`, `model_testing/reports/frvp_backtests/frvp_long_setup5_repeated_drawdown_prune_promotion_20260829`, `model_testing/reports/frvp_backtests/frvp_long_reversal_setup_recency730_opt_20260829`, `model_testing/reports/frvp_backtests/frvp_long_reversal_setup_recency730_opt_recent2y_20260829`, `model_testing/reports/frvp_backtests/frvp_long_setup4_fullspan_1fold_research_20260829`, and `docs/FRVP_experiment_journal.md#frvp_setup_specific_model_leaderboard_20260830`.
- `Result:` S5 pruned became the top absolute-PnL setup lane at `724` trades, `+12966.40` ticks, expectancy `+17.91`, PF `1.391`, Sharpe `0.912`, DSR `0.859`, max DD `38.59%`, and `WFE = 1.723`, but still failed drawdown and single-trade concentration. S2 pruned became the cleanest risk-adjusted setup lane at `104` trades, `+4344.40` ticks, expectancy `+41.77`, PF `2.265`, Sharpe `1.625`, DSR `1.153`, max DD `8.36%`, but failed `WFE = -4.127` and concentration. S1 recency recent2y stayed modest at `+1524.55` ticks with low Sharpe and negative WFE. S4 one-fold research was positive at `5` trades and `+392.75` ticks, but the sample was too thin and one trade contributed `70.36%` of total PnL. S6 remained negative in both optimized recency windows, and S3 stayed economics-negative.
- `What worked:` policy-only pruning created useful forward-observation candidates without spending more training budget. S2's drawdown and expectancy profile became credible enough to shadow, and S5's broad raw continuation signal became much more economically concentrated after repeated-drawdown pruning.
- `What did not work:` no setup-specific lane cleared the full promotion/paper gate stack. S5 still carries unacceptable historical drawdown, S2's negative train-side WFE remains unresolved, S4 lacks fold/trade breadth, and S6 threshold slicing did not reveal a credible rescue path because composite thresholds were not data-sufficient and realized economics stayed negative.
- `Observations:` the right next move is not accepted paper deployment, but quarantined shadow-paper observation for the two informative continuation lanes. S2 should test whether its apparently recent-regime edge is real forward behavior or instability; S5 should test whether the pruned high-PnL lane can behave without reproducing its historical drawdown. S4 can be logged only for event collection.
- `Backend attribution note:` XGBoost backend attribution was not skipped for the setup-specific prepared root. The root contains `backend_attribution_summary.json`, per-target `backend_attribution_summary_xgboost.json`, `feature_importance_merged_xgboost.csv`, and `shap_feature_stats_xgboost.csv`. What was skipped was standalone TCN attribution/training and extra post-hoc attribution reruns, intentionally, because this wave was an XGBoost-only target-routing/policy experiment and the recency roots only reweighted `train`/`val` sample weights.
- `Decision:` do not promote any setup-specific FRVP model and do not add them to the accepted paper roster. Create a separate shadow-paper watchlist only if it is isolated from accepted paper reporting: S2 pruned as highest priority, S5 pruned promotion-quality as medium priority, and S4 one-fold as optional log-only event collection. Keep S1, S3, and S6 out of shadow paper for now.
- `Repo change:` added `docs/FRVP_experiment_journal.md#frvp_setup_specific_model_leaderboard_20260830` as the traceable leaderboard/recommendation artifact. The setup-specific continuation lanes are now classified as shadow-observation candidates, while setup-specific reversal remains research-only or paused.

### E16. B1 FRVP S2/S3/S5 and S4 Outcome-Contract Resolution (2026-09-17)

- `Question:` which pre-existing risk rules resolve the original design table, generic labeler defaults, saved pipeline overrides, and S4 enable/disable history without choosing rules from evaluation returns?
- `Control:` the July research control/refresh and August setup-specialist event configurations; archived B1 v1 remains unchanged. This is a configuration audit, not a new performance experiment.
- `Change:` freeze independent v2 paper contracts for ES and the configured 6E variant: S2 stop/target 1/1 ATR, S3 1/0.95 long and 1/1 short, S5 1/0.90 long and 1/0.85 short, all 96 bars; S4 0.90/1 ATR, 48 bars. Keep causal next-open execution, fixed entry-time geometry, adverse-first collision fallback, and separate position costs. Add side-specific dispatch for S3/S5.
- `Method:` read configuration columns only from the three saved event tapes; trace `_default_phase03_labeling_params` and `_barrier_spec`, the existing pipeline assertions, historical source revisions, training-summary prepared-root pointers, and E01/E13/E14. No P&L search, label regeneration, training, policy fitting or deployment.
- `Artifacts:` [resolution and evidence](B1_outcome_contracts.md#b1_frvp_risk_resolution_20260917), `docs/audits/b1_frvp_risk_evidence_20260917.json`, `configs/outcome_contracts/b1_v2.json` and lock, [active matrix](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260917), and `tests/test_outcome_contracts.py`.
- `Result:` all four requested FRVP setups now have executable, immutable paper economics on both sides, including pooled/meta and specialist dispatch. The active catalogue has 45 entries: 32 resolved and 13 unresolved. No economic-performance delta was calculated.
- `What worked:` saved event configuration resolves the third layer missed by the initial B1 audit: the pipeline's continuation target is 1.00 ATR with explicit S3/S5 overrides, not the generic labeler's 1.20. All three saved event roots agree on these risk values.
- `What did not work:` the old design widths, generic defaults and pooled 120-bar report horizon cannot be used interchangeably. Old training metadata alone does not establish causal input lineage or a paper evaluation contract.
- `Observations:` S4 was disabled in the old control and re-enabled in E01's refresh. Its current policy-candidacy pause is a separate restriction. The same pipeline configures 6E; these ES records do not establish 6E model validation or promotion.
- `Decision:` adopt the established per-setup/per-side configuration as the independent B1 v2 paper baseline, with B1 execution and outcome semantics. Keep all training artifacts/labels and archived v1 contracts intact. Retain S4's model-specific-policy pause and all existing B2/B3/B4/B5, economics and promotion restrictions.
- `Repo change:` add the locked v2 catalogue, enforce side compatibility in both routing and direct evaluation, preserve hashes in all four artifact surfaces, and update the design, operational addendum, audit, matrix, live change journal and B1 checklist. OTE, classic breaker and unmatched pooled outcomes remain open B1 decisions.

### E17. All-Pooled Diagnostic Scope and Causal Structural ATR (2026-09-18)

- `Question:` how to retain model-versus-setup research data while excluding pooled predictions from baseline trades, and whether the FRVP ATR denominator is actually a causally available hourly value.
- `Control:` E16's frozen risk catalogue and existing immutable setup/prediction logging; trained labels, detectors and model artifacts remain untouched.
- `Change:` user-approved `b1_baseline_scope_v1` makes all pooled predictions diagnostic-only, even with a matching setup; classic breaker/unmatched predictions are explicitly excluded. Add model/setup/context and coverage artifacts, read-only live collection export, and a completed-hour structural ATR producer for B1 evaluation.
- `Method:` trace inherited ATR timeframe/period/smoothing and hourly timestamp mapping; reproduce future-hour contamination synthetically; test scope admission, explicit costs, market-response independence, missing data/rolls and immutable replay. No evaluation-period P&L or threshold search.
- `Artifacts:` [scope/ATR/audit design](B1_outcome_contracts.md#b1_baseline_scope_and_model_setup_audit_20260917), `docs/audits/b1_scope_atr_audit_20260918.json`, `baseline_scope_v1.json` and lock, [current matrix](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260918), and the live change journal.
- `Result:` paper/position admission now excludes pooled models while retaining their gross forward responses, scores, setup relationships and slice/coverage evidence. FRVP risk uses price-unit, completed 1-hour Wilder ATR14 mapped to the decision clock with source provenance. The v1/v2 economics catalogues and saved FRVP event/training evidence remain unchanged.
- `What worked:` explicit scope separated a specialist's setup economics from a pooled model sharing that setup. Tests proved future 5m constituents cannot change earlier hourly ATR values or source hashes; raw diagnostics survive missing ATR provenance without invented normalized values.
- `What did not work:` historical FRVP resampling stamps full-hour ATR at the hour open before forward mapping, so it is not a causal evaluation source. The initial B1 ATR audit was incomplete. Existing labels must remain frozen; B2 must trace affected training lineage before any correction/replacement.
- `Observations:` model/setup slices are exploratory gross-response evidence, not trade P&L or a winner-selection policy. ICT still uses signal-bar ATR14; this task does not migrate its strategy geometry. OTE standalone economics remains open.
- `Decision:` retain all pooled scores for research and future common-opportunity model/setup comparisons; no pooled position admission in this baseline. Preserve S4's policy pause, B2/B3/B4/B5, economics and promotion restrictions. No collection or trial activation.
- `Repo change:` new scope, ATR and audit modules; schema/manifest/replay extensions; runtime regime-context persistence; read-only SQLite export; tests and checklist updates. E16's numerical barriers stay frozen, while its earlier pooled-paper eligibility is superseded by this explicit scope decision.

## Supporting FRVP records (merged 2026-10-09)

- [FRVP Operational Contract](#frvp_operational_contract_20260721)
- [FRVP Regime-Gated Deployment and Concentration Roadmap](#frvp_regime_gated_deployment_roadmap_20260721)
- [FRVP Setup-Specific Long Rerun Plan](#frvp_setup_specific_long_rerun_plan_20260829)
- [FRVP Setup-Specific Model Leaderboard](#frvp_setup_specific_model_leaderboard_20260830)

<a id="frvp_operational_contract_20260721"></a>

### FRVP Operational Contract

*Merged 2026-10-09 from `docs/FRVP_operational_contract_20260721.md`.*

**Date:** 2026-07-21
**Purpose:** Record the exact FRVP operating rules that were finalized on Tuesday, July 21, 2026.

This note turns the recent FRVP decisions into one explicit contract for code, shadow ops, and review.

#### August 16, 2026 Controlled-Trial Addendum

The July shadow contract below remains historical context, but the controlled
paper-signal roster is now superseded by `frvp_es_paper_signal_20260816`:

- `frvp_long_reversal_xgb_v1` is the sole active paper-signal model, representing
  the `frvp_long_reversal_xgb_recent_regime_prune_v2` selective contract.
- Its packaged global `0.60` policy now restores the accepted
  `strong_down_high/overlap` abstain pair; all 11 accepted pair filters are present.
- `frvp_long_continuation_xgb_v1` remains candidate/shadow. Its exact final
  global-`0.70` sensitivity missed Sharpe, profitable-quarter breadth, and
  single-trade concentration gates even though the archived adaptive
  walk-forward snapshot passed all eight gates.
- The reversal fixed-policy sensitivity stayed strong but missed the `0.10`
  largest-trade-share limit (`0.1287`). The 28-day window is therefore a
  concentration-validation trial, not full promotion.
- Entry is the signal-bar close and settlement is the close after 120 completed
  `5m` bars. Events may overlap; stop and target are not applicable. No broker
  order or position path is authorized.
- Human same-contract TradingView signoff is explicitly deferred only for this
  controlled no-order trial and remains mandatory for full promotion.
- The trial is authorized pending the exact readiness audit, but it has not
  started. Its clock begins only on the final collector's first healthy running
  heartbeat from an authenticated non-delayed paper feed.

The immutable decision details and validation evidence live in
`ote_live/runtime_manifests/frvp_es_paper_signal_20260816/`.

#### Documentation Provenance

##### September 17, 2026 B1 outcome-contract addendum

September 18 scope update (journal E17): all pooled FRVP predictions are now
diagnostics-only, including setup matches. The geometry below continues to
apply to setup specialists subject to prerequisites. See the
[scope/ATR contract](B1_outcome_contracts.md#b1_baseline_scope_and_model_setup_audit_20260917): new
evaluation uses attested completed-hour ATR, and the historical label path's
hour-open mapping remains a B2 lineage issue. Historical trials are unchanged.

Experiment-journal E16 resolves S2/S3/S5 and S4 paper economics using the
established pipeline and saved event configurations. The independent
[B1 v2 contracts](B1_outcome_contracts.md#b1_frvp_risk_resolution_20260917) use S2 stop/target 1/1 ATR;
S3 1/0.95 long and 1/1 short; S5 1/0.90 long and 1/0.85 short; all 96 bars.
S4 uses 0.90/1 ATR, 48 bars. Use causal completed 1-hour ATR, strict next-open
entry, fixed barriers, and separate position costs. Split S3/S5 contract IDs
enforce side; pooled models dispatch by the causal selected setup.

This is the new research evaluation baseline, not a change to the frozen
August trial, historical labels, saved policies or model activation. S4's
model-specific-policy pause remains. The [active B1 matrix](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260917)
and [resolution evidence](audits/b1_frvp_risk_evidence_20260917.json) are the
references for new B1 outcomes; prior 120-bar markouts retain their old meaning.

##### Earlier operational decisions

The operational decisions in this note are the contract-level output of the canonical experiment record in [FRVP_experiment_journal.md](FRVP_experiment_journal.md).

- The continuation baseline decision comes from `E02`.
- The full-span reversal control history comes from `E06`.
- The recent-regime reversal lane comes from `E07` and `E11`.
- The default cost-contract decision comes from `E10`.

Future FRVP contract changes should be documented in the journal first, using [#frvp_experiment_record_template](#frvp_experiment_record_template), before this note is revised.

#### 1. Anchor And Range Selection

The FRVP anchor order is now explicit in code via `frvp/profiles/anchors.py`.

- `prior_rth` is the primary decision range.
  - Use its `POC`, `VAH`, `VAL`, `HVN`, and `LVN` as the default live trading reference levels.
  - `frvp_dist_vah_atr`, `frvp_dist_val_atr`, `frvp_in_va`, `frvp_above_vah`, `frvp_below_val`, and `frvp_profile_shape` are all interpreted off this anchor first.
- `overnight_eth` is the secondary inventory range.
  - Use it for overnight inventory context, RTH-vs-ETH overlap, and naked overnight POC tracking.
- `initial_balance` is the intraday confirmation range.
  - It becomes valid only after the first 60 RTH minutes are complete, at 10:30 ET.
- `swing_to_swing` is reactive context only.
  - It is not the default trigger anchor for `POC` / `VAH` / `VAL` decisions.
- `rolling_composite` is higher-timeframe context only.
  - It is a context range, not the live trigger range for Setup 1-6 decisions.

All anchors remain causal and single-contract:

- No anchor may mix contracts.
- Cross-roll naked-POC tracking stays reset-at-roll.
- No `POC` / `VAH` / `VAL` level is compared across contract coordinates without translation.

#### 2. Real Displacement Vs. Noise

For FRVP operations, "real displacement" is no longer a qualitative phrase.

A bullish or bearish displacement is treated as real only if all of these are true:

1. The existing directional displacement flag is already on.
   - `displacement_bullish == 1` for bullish.
   - `displacement_bearish == 1` for bearish.
2. `volume_zscore_50 >= 1.25`.
3. The close finishes in the outer 25% of the bar on the move side.
   - Bullish: `(close - low) / (high - low) >= 0.75`
   - Bearish: `(high - close) / (high - low) >= 0.75`

For Setup 3 specifically, the move must also close at least `0.05 ATR` beyond the relevant value-area edge:

- Bullish breakout: `-frvp_dist_vah_atr >= 0.05`
- Bearish breakout: `-frvp_dist_val_atr >= 0.05`

If those conditions are not met, the move is treated as noise for FRVP setup selection.

#### 3. Setup Tiebreakers

##### Setup 1 vs. Setup 3

This tie is now resolved by measurable acceptance outside value.

- Choose `Setup 3` only if there is real displacement and the close is at least `0.05 ATR` beyond `VAH` or `VAL`.
- Otherwise keep the event in `Setup 1` territory if price is only touching or making a small reentry / overshoot around the edge.

In practice:

- A small poke outside value without real displacement is still fadeable.
- A real, high-participation close outside value is continuation, not a fade.

##### Setup 2 vs. Setup 4

This tie is now resolved by measurable reentry depth.

- `Setup 2` owns the near-edge hold:
  - after a breakout, the retest may close back inside by at most `0.05 ATR`.
- `Setup 4` owns the failed auction:
  - price must reenter value by more than `0.05 ATR`
  - after `1-3` outside bars
  - with quiet reentry volume
  - and a same-side sweep inside the lookback window

That makes the two contracts disjoint:

- shallow hold near the broken edge -> `Setup 2`
- deeper return back into value -> `Setup 4`

#### 4. Extended Shadow Window

The extended FRVP shadow window starts on **2026-07-21** from the `frvp_es_shadow_20260721` bundle.

Primary baseline:

- `frvp_long_continuation_xgb_v1`
- saved `v3` targeted-policy contract
- this remains the promotion-near FRVP branch

Observation focus:

- contract-switch behavior
- dashboard continuity
- signal persistence
- audit-replay consistency

Operational expectation:

- keep the continuation baseline frozen
- use the new July 21, 2026 shadow bundle as the default FRVP runtime package
- treat new shadow evidence as operational validation, not classifier-research pressure

#### 5. Reversal Contract Decision

Decision on **2026-07-21**:

- accept `frvp_long_reversal_xgb_recent_regime_prune_v2` as the operational FRVP reversal contract
- keep the full-span recency `v3` branch as the research control, not the live contract

Important naming note:

- the accepted live contract is aliased in repo language as `frvp_long_reversal_xgb_recent_regime_prune_v2`
- the saved July 21, 2026 backtest / threshold artifact directory that produced it is still named `long_reversal_recent2y_sdh_overlap_prune_v1`

Saved evidence for the accepted contract from the July 21, 2026 backtest artifact:

- `63` trades
- `+3677.05` ticks
- Sharpe `1.480`
- DSR `1.179`
- `WFE = 2.162`
- profitable-quarter share `0.60`
- accepted paper-trading gate `True`

That is sufficient to treat the recent-regime lane as the live selective-deployment answer unless live shadow evidence later proves otherwise.

#### 6. Conditional Train-Side Follow-Up

No new train-side branch is opened on **2026-07-21** because the reversal contract decision above is a "yes."

Only reopen full-span reversal train-side work if one of these becomes true:

- the accepted recent-regime live contract proves insufficient in shadow operation
- the repo explicitly decides it still needs a better full-span reversal answer

If that happens:

- keep the current full-span control frozen
- test only new train-side stability levers that can improve `WFE`
- do not reopen same-environment micro-pruning

<a id="frvp_regime_gated_deployment_roadmap_20260721"></a>

### FRVP Regime-Gated Deployment and Concentration Roadmap

*Merged 2026-10-09 from `docs/FRVP_regime_gated_deployment_roadmap_20260721.md`.*

**Date:** 2026-07-21
**Purpose:** Turn the post-friction-A/B FRVP next step into one concrete ordered work plan.

Detailed historical context for why this roadmap exists now lives in [FRVP_experiment_journal.md](FRVP_experiment_journal.md). This roadmap is the operational extraction of the latest friction, Q11, and concentration experiments rather than the full experiment history itself.

Documentation provenance:

- the half-life sweep and selective recent-regime checkpoint context comes from `E09`
- the cost-contract decision comes from `E10`
- the concentration / regime-gated deployment decision comes from `E11`
- future updates should use `docs/FRVP_experiment_journal.md#frvp_experiment_record_template` before the roadmap is revised

#### Execution update

The sequence in this roadmap was executed on July 21, 2026 in `model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721`.

What changed:

- The frozen continuation control stayed clean. `frvp_long_continuation_xgb_v1` `v3` remained the strong reference branch at `+7001.20` ticks, Sharpe `1.187`, `WFE = 2.344`, profitable-quarter share `0.625`, and paper-trading gate accepted.
- The full-span long-reversal control did not get solved by one more policy prune. Adding `strong_down_medium/asia` lifted net PnL to `+6245.15` ticks, Sharpe to `1.138`, and drawdown down to `8.57%`, but `WFE` worsened to `-4.334`. That is a cleaner sign that the full-span failure is still older-regime train-side instability, not just one last live deployment pocket.
- The recent-2y long-reversal lane did produce a real selective-deployment winner. Blocking the sparse `strong_down_high/overlap` pocket turned the branch into a new checkpoint: `63` trades, `+3677.05` ticks, Sharpe `1.480`, DSR `1.179`, `WFE = 2.162`, profitable-quarter share `0.60`, largest-single-trade share `0.0942`, and `accepted_for_paper_trading_gate = True`.
- That winning contract is now codified as `frvp_long_reversal_xgb_recent_regime_prune_v2`.

#### Why this is next

The July 19, 2026 friction A/B already answered the first big question: the saved FRVP branches are not mainly waiting on one generic cost-model fix. `session_schedule` should stay the default FRVP contract, and the only notable `feature_proxy` improvements were the recent-2y long-reversal lane and the weak short-meta sentinel.

That means the next question is simpler and more practical: are the saved reversal profits broad enough to trust, or are they too concentrated in a few trades, a few quarters, or a few regime pockets?

#### What "regime-gated deployment / concentration work" means

It means we do not retrain or change classifiers first. We keep the saved models fixed and test whether the policy layer can deploy them more selectively and more honestly.

In plain English:

- "Regime-gated deployment" means only allowing a model to trade in the market conditions where the saved evidence is actually good.
- "Concentration work" means checking whether the result depends too much on a tiny number of trades, quarters, sessions, or composite regimes.

#### Ordered roadmap

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

#### Resulting decision

- Keep `frvp_long_continuation_xgb_v1` `v3` unchanged as the promotion-near continuation branch.
- Keep the full-span long-reversal control unchanged and treat its remaining miss as a train-side stability problem.
- Use `frvp_long_reversal_xgb_recent_regime_prune_v2` as the current selective-deployment checkpoint for reversal.
- Defer new classifier changes until the repo either accepts that recent-regime selective deployment is the right reversal contract or finds a train-side lever that improves the full-span branch without giving back the clean recent-regime behavior.

#### Definition of done for this phase

This phase is complete when the repo can answer all three of these clearly:

1. Which FRVP branches have broad enough saved edge to trust operationally?
2. Which branches are only viable as selective deployment?
3. Which branches still fail even after the best honest selective-deployment policy is applied?

<a id="frvp_setup_specific_long_rerun_plan_20260829"></a>

### FRVP Setup-Specific Long Rerun Plan

*Merged 2026-10-09 from `docs/FRVP_setup_specific_long_rerun_plan_20260829.md`.*

Date: 2026-08-29

#### 1. Readiness Check

Status: code path is implemented. Do not open another pooled FRVP retrain branch before this setup-specific lane is materialized and evaluated.

- `frvp/target_lanes.py` defines 6 setup types, 12 setup-specific direct targets, 4 pooled direct targets, and 2 meta controls.
- `data/labeling/frvp_labeling_engine.py` writes pooled and setup-specific helper columns from the same events.
- Setup targets inherit pooled-family concurrency and sample weights. This keeps the first comparison focused on routing only.
- `frvp/pipelines/es_primary_phase04.py` now prepares all `FRVP_TARGET_COLUMNS`.
- `scripts/run_frvp_training_stack.ps1` supports `-XgbTargetMode setup`, but this plan uses direct target lists so only long lanes run.
- `scripts/build_frvp_candidate_registry.py` infers setup model IDs such as `frvp_long_reversal_setup1_xgb_v1`.

Verified locally:

```powershell
ote_venv\Scripts\python.exe -m pytest tests\test_frvp_per_setup_targets.py tests\test_frvp_setups.py
ote_venv\Scripts\python.exe -m pytest tests\test_build_frvp_candidate_registry.py tests\test_frvp_labeling.py
```

Result: 34 passed.

Current artifact gap: `artifacts\frvp_es_primary_current\phase04\prepared` still contains only the four pooled direct targets. The next run must materialize a new setup-aware Phase 4 root before training.

#### 2. Setup Logic Read

The setup logic is sound for the current Fixed Range Volume Profile contract.

- Setup 1: balanced D-profile, open in value, fade VAH/VAL. It rejects real outside displacement so continuation breakouts do not get mislabeled as fades.
- Setup 2: continuation retest after a real value-area breakout. It owns the shallow near-edge hold.
- Setup 3: initiative value-area breakout. Requires real volume expansion, close-location efficiency, and at least 0.05 ATR acceptance beyond VAH/VAL.
- Setup 4: failed auction. Requires 1-3 outside bars, same-side sweep, quiet reentry, and more than 0.05 ATR reentry into value.
- Setup 5: LVN / thin-zone continuation, gated by displacement and proximity to nearest LVN.
- Setup 6: balanced in-value HVN magnet. Correctly avoids open-drive, displacement, high-volume, and ambiguous symmetric-HVN contexts.

Caveats:

- Setup 4 is structurally thin: prior replay found 380 usable long events and 412 usable short events. Treat S4 as research-only until economics prove otherwise.
- Setup 6 still lacks gamma-context and Setup 6b naked-VPOC event logic. Do not block this rerun on those deferred items.
- Current targeted policy presets are keyed to pooled model IDs. Do no-preset setup economics first, then port filters only after setup-level regime tables justify it.

#### 3. Branches To Rerun

Rerun only the strongest long FRVP branches:

- Long continuation full-span XGBoost: split into S2, S3, S5.
- Long reversal full-span XGBoost: split into S1, S4, S6.
- Long reversal recency-weighted XGBoost: split into S1 and S6 first, using the proven 730-day half-life and 0.20 floor. S4 recency is optional research after the full-span S4 read.
- Do not run TCNs in wave 1.
- Do not train short lanes in this wave.

#### 4. Materialize Setup-Aware Prepared Root

Run from repo root.

```powershell
ote_venv\Scripts\python.exe -m frvp.pipelines.es_primary_phase04 `
  --input data\futures_data\ES-5m-tagged.csv `
  --output-root artifacts\frvp_es_primary_setup_targets_20260829 `
  --feature-csv artifacts\frvp_es_primary_current\phase02\es_primary_frvp_features_full.csv.gz `
  --feature-metadata artifacts\frvp_es_primary_current\phase02\es_primary_frvp_features_full.csv.metadata.json `
  --instrument es `
  --backend-max-features 160 `
  --attribution-max-rows 100000 `
  --top-n-features 25 `
  --min-usable-rows 250 `
  --min-train-rows 100 `
  --min-positive-samples 25 `
  --phase02-compression gzip
```

Verify target directories:

```powershell
Get-ChildItem artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared -Directory | Select-Object Name
```

Expected long setup dirs:

- `long_frvp_reversal_setup1`
- `long_frvp_continuation_setup2`
- `long_frvp_continuation_setup3`
- `long_frvp_reversal_setup4`
- `long_frvp_continuation_setup5`
- `long_frvp_reversal_setup6`

#### 5. Train Full-Span Long Continuation Setups

Use reduced setup-lane CV geometry instead of pooled 900/300/300. This keeps S2 from being over-constrained while still giving S3/S5 enough folds.

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-root models\frvp_long_continuation_setup_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_continuation_setup2 long_frvp_continuation_setup3 long_frvp_continuation_setup5 `
  --trials 28 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 40 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 500 `
  --cv-val-rows 150 `
  --cv-step-rows 150 `
  --cv-min-folds 2 `
  --min-train-positive-rows 100 `
  --min-val-positive-rows 25 `
  --min-val-true-events 10 `
  --seed 42
```

Note: continuation calibration resolves to `none` by target contract even when `--calibration-method platt` is passed.

#### 6. Train Full-Span Long Reversal Setups S1/S6

S1 and S6 get a smaller but still multi-fold geometry. Keep S4 separate.

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup1 long_frvp_reversal_setup6 `
  --trials 32 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 32 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 350 `
  --cv-val-rows 100 `
  --cv-step-rows 100 `
  --cv-min-folds 2 `
  --min-train-positive-rows 70 `
  --min-val-positive-rows 15 `
  --min-val-true-events 8 `
  --seed 42
```

#### 7. Train Setup 4 Standalone

Research-only geometry. Do not compare S4 promotion status to the regular lanes without noting the one-fold constraint.

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup4_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup4 `
  --trials 40 `
  --max-loaded-features 160 `
  --top-feature-min 16 `
  --top-feature-max 64 `
  --window-min 8 `
  --window-max 24 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 140 `
  --cv-val-rows 40 `
  --cv-step-rows 40 `
  --cv-min-folds 1 `
  --min-train-positive-rows 20 `
  --min-val-positive-rows 5 `
  --min-val-true-events 3 `
  --final-eval-min-rows 40 `
  --seed 42
```

#### 8. Materialize Recency-Weighted Reversal Setup Roots

Use the prior winning full-history recency contract: 730-day half-life, 0.20 floor.

```powershell
ote_venv\Scripts\python.exe scripts\materialize_frvp_recency_prepared_root.py `
  --base-prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-prepared-root artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared `
  --target long_frvp_reversal_setup1 `
  --half-life-days 730 `
  --floor 0.20
```

```powershell
ote_venv\Scripts\python.exe scripts\materialize_frvp_recency_prepared_root.py `
  --base-prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
  --output-prepared-root artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared `
  --target long_frvp_reversal_setup6 `
  --half-life-days 730 `
  --floor 0.20
```

#### 9. Train Recency-Weighted Reversal Setups

S1:

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup1_recency730_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup1 `
  --trials 32 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 32 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 350 `
  --cv-val-rows 100 `
  --cv-step-rows 100 `
  --cv-min-folds 2 `
  --min-train-positive-rows 70 `
  --min-val-positive-rows 15 `
  --min-val-true-events 8 `
  --seed 42
```

S6:

```powershell
ote_venv\Scripts\python.exe -m model_training.ote_training.ote_xgboost_pipeline `
  --prepared-root artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared `
  --output-root models\frvp_long_reversal_setup6_recency730_xgb_20260829 `
  --backend xgboost `
  --targets long_frvp_reversal_setup6 `
  --trials 32 `
  --max-loaded-features 160 `
  --top-feature-min 24 `
  --top-feature-max 96 `
  --window-min 8 `
  --window-max 32 `
  --event-tolerance-bars 2 `
  --event-cooldown-bars 4 `
  --calibration-method platt `
  --cv-initial-train-rows 350 `
  --cv-val-rows 100 `
  --cv-step-rows 100 `
  --cv-min-folds 2 `
  --min-train-positive-rows 70 `
  --min-val-positive-rows 15 `
  --min-val-true-events 8 `
  --seed 42
```

#### 10. Build Registries

Full-span setup registry:

```powershell
ote_venv\Scripts\python.exe scripts\build_frvp_candidate_registry.py `
  --model-root models\frvp_long_continuation_setup_xgb_20260829 `
  --model-root models\frvp_long_reversal_setup_xgb_20260829 `
  --model-root models\frvp_long_reversal_setup4_xgb_20260829 `
  --source-registry-path models\frvp_es_primary_model_registry_current.json `
  --output-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json
```

Recency reversal setup registry:

```powershell
ote_venv\Scripts\python.exe scripts\build_frvp_candidate_registry.py `
  --model-root models\frvp_long_reversal_setup1_recency730_xgb_20260829 `
  --model-root models\frvp_long_reversal_setup6_recency730_xgb_20260829 `
  --source-registry-path models\frvp_es_primary_model_registry_current.json `
  --output-path models\frvp_es_primary_model_registry_long_reversal_setup_recency730_20260829.json
```

Use separate registries because full-span and recency models share the same inferred setup model IDs.

#### 11. Evaluate Full-Span Regular Lanes

No targeted preset on the first pass.

```powershell
.\scripts\run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_setup_fullspan_20260829 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_setup_fullspan_20260829 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829 `
  -ModelIds frvp_long_reversal_setup1_xgb_v1,frvp_long_reversal_setup6_xgb_v1,frvp_long_continuation_setup2_xgb_v1,frvp_long_continuation_setup3_xgb_v1,frvp_long_continuation_setup5_xgb_v1 `
  -BacktestMinTrainYears 2 `
  -BacktestMinFolds 8 `
  -SpreadCostMode session_schedule
```

#### 12. Evaluate Setup 4 Separately

Research contract, low-frequency thresholds.

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_regime_slice_report.py `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --bootstrap-iterations 200 `
  --min-positive-events 10
```

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_threshold_policy_search.py `
  --regime-report-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_threshold_policies\frvp_long_setup4_fullspan_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --instrument es `
  --spread-cost-mode session_schedule `
  --min-positive-events 10 `
  --min-events-per-month 0.5 `
  --min-trades-per-week 0.5 `
  --evaluation-contract-mode research `
  --write-policy-decisions
```

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_policy_backtest.py `
  --regime-report-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --instrument es `
  --spread-cost-mode session_schedule `
  --min-train-years 2 `
  --test-window-months 3 `
  --rolling-step-months 3 `
  --min-folds 4 `
  --min-positive-events 10 `
  --min-events-per-month 0.5 `
  --min-trades-per-week 0.5 `
  --evaluation-contract-mode research `
  --minimum-sharpe 0.8 `
  --maximum-drawdown-pct 12.0 `
  --minimum-dsr 0.3
```

#### 13. Evaluate Recency Reversal Lanes

First pass: full evaluation window.

```powershell
.\scripts\run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_setup_recency730_20260829.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_20260829 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_20260829 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_setup_recency730_20260829 `
  -ModelIds frvp_long_reversal_setup1_xgb_v1,frvp_long_reversal_setup6_xgb_v1 `
  -BacktestMinTrainYears 2 `
  -BacktestMinFolds 8 `
  -SpreadCostMode session_schedule
```

Second pass: recent-regime control matching the prior accepted reversal branch geometry.

```powershell
.\scripts\run_frvp_post_training_eval.ps1 `
  -RegistryPath models\frvp_es_primary_model_registry_long_reversal_setup_recency730_20260829.json `
  -RegimeOutputRoot model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_recent2y_20260829 `
  -ThresholdOutputRoot model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_recent2y_20260829 `
  -BacktestOutputRoot model_testing\reports\frvp_backtests\frvp_long_reversal_setup_recency730_recent2y_20260829 `
  -ModelIds frvp_long_reversal_setup1_xgb_v1,frvp_long_reversal_setup6_xgb_v1 `
  -BacktestMinTrainYears 2 `
  -BacktestMaxTrainYears 2 `
  -BacktestMinScheduledTestStart 2024-01-01 `
  -BacktestMinFolds 8 `
  -SpreadCostMode session_schedule
```

#### 14. Decision Rules

Promote only if a setup lane beats its pooled parent on economics, not just classifier metrics.

- Continuation setup winner must beat or clearly simplify `frvp_long_continuation_gatefix_v3_20260715_accountdd`: 524 trades, +7190.40 ticks, Sharpe 1.226, DSR 1.061, WFE 2.459, 8/8 gates.
- Reversal full-span setup winner must repair the pooled recency branch's train-side instability: +6027.00 ticks and Sharpe 1.061 were not enough because WFE was -3.797.
- Reversal recent2y setup winner must beat the accepted selective contract: 63 trades, +3677.05 ticks, Sharpe 1.480, DSR 1.179, WFE 2.162, 8/8 gates.
- S4 must be judged as a standalone research lane unless it clears enough fold/trade breadth to become promotion-quality eligible.
- If S2/S3/S5 all trail pooled continuation, keep pooled continuation.
- If S1 or S6 wins but only after regime pruning, add setup-specific targeted filter presets in a separate code change and rerun the backtest. Do not reuse pooled presets blindly.

#### 15. Experiment Continuation

Status after `frvp_setup_specific_long_rerun_20260829_output_v2.txt`: no setup-specific lane is promotable. Keep the current deployed/paper roster unchanged.

##### 15.1 Clean Script Noise

Update `scripts\run_frvp_setup_specific_long_rerun_plan_20260829.ps1` before the next run.

- Skip recency prepared-root materialization when the target root already exists, matching the Step 4 skip behavior.
- Set pytest temp/cache paths inside the repo, or disable pytest cache, so Step 1A stops false-failing on local permission errors.

Implementation target:

```powershell
New-Item -ItemType Directory -Force tmp\pytest | Out-Null
$env:TMP = (Resolve-Path tmp\pytest).Path
$env:TEMP = (Resolve-Path tmp\pytest).Path
$env:PYTEST_ADDOPTS = "-p no:cacheprovider"
```

Recency-root skip rule:

```powershell
if (Test-Path "artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared") {
  Write-Output "Skipping S1 recency prepared root; already exists."
}
else {
  ote_venv\Scripts\python.exe scripts\materialize_frvp_recency_prepared_root.py `
    --base-prepared-root artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared `
    --output-prepared-root artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared `
    --target long_frvp_reversal_setup1 `
    --half-life-days 730 `
    --floor 0.20
}
```

Apply the same pattern for `long_frvp_reversal_setup6`.

##### 15.2 S2 Continuation Policy-Only Pass

Do not retrain S2 yet. It has real signal, but 2024 is the drag:

- `2024`: `-1861.65` ticks
- `2025`: `+3744.65` ticks
- `2026`: `+1616.35` ticks

First pass: use the existing v2 full-span outputs to identify bad regime/session pockets.

```powershell
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup2_xgb_v1\breakdown_by_year.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup2_xgb_v1\breakdown_by_session.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup2_xgb_v1\breakdown_by_composite.csv
```

Then create a setup-specific targeted filter preset only if the bad pockets are repeated enough to justify blocking. Rerun economics with the model fixed.

Executed continuation pass:

- Added preset `frvp_long_continuation_setup2_xgb_medium_session_prune_v1` for `frvp_long_continuation_setup2_xgb_v1`.
- Filtered composite/session pairs: `strong_down_medium/london`, `strong_down_medium/overlap`, `strong_up_medium/london`, `strong_up_medium/overlap`.
- Rerun output root: `model_testing\reports\frvp_backtests\frvp_long_setup2_medium_session_prune_20260829`.
- Result improved from `201` trades, `+3499.35` ticks, Sharpe `0.708`, max DD `24.52%`, WFE `-2.004` to `104` trades, `+4344.40` ticks, Sharpe `1.625`, DSR `1.153`, max DD `8.36%`, WFE `-4.127`.
- Gate read: economics improved and drawdown repaired, but still not promotable because WFE remains negative and largest-single-trade concentration still fails.
- Next S2 action: do not promote. If S2 remains interesting after S5, inspect train-window instability before any retrain or stronger pruning.

##### 15.3 Drop S3

Drop `frvp_long_continuation_setup3_xgb_v1` from the active setup rerun path for now.

- Classifier metrics are good.
- Economics are negative: `485` trades, `-2964.25` ticks, Sharpe `-0.379`, max DD `49.96%`.
- Do not spend more training or policy time on S3 until S2/S5 are exhausted.

##### 15.4 S5 Continuation Drawdown Policy Pass

Treat S5 as too broad and too drawdown-heavy.

- Result: `1217` trades, `+5255.95` ticks.
- Problem: Sharpe `0.281`, DSR `0.279`, max DD `82.26%`.

Run a targeted policy pass to reduce drawdown before considering more training.

```powershell
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup5_xgb_v1\breakdown_by_year.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup5_xgb_v1\breakdown_by_session.csv
Get-Content model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829\frvp_long_continuation_setup5_xgb_v1\breakdown_by_composite.csv
```

Main suspected drag: 2022 was `-6762.45` ticks. Confirm whether that is a year-only regime failure or a smaller composite/session pocket before coding a preset.

Executed continuation pass:

- Added preset `frvp_long_continuation_setup5_xgb_repeated_drawdown_prune_v1` for `frvp_long_continuation_setup5_xgb_v1`.
- Filtered repeated negative composite/session pockets: `strong_down_high/london`, `strong_up_high/london`, `strong_up_medium/asia`, `ranging_high/new_york`, `ranging_medium/asia`, `strong_down_low/london`, `strong_down_medium/overlap`, `strong_up_low/asia`, `strong_down_high/new_york`, `strong_up_medium/new_york`.
- Rerun output root: `model_testing\reports\frvp_backtests\frvp_long_setup5_repeated_drawdown_prune_promotion_20260829`.
- Result improved from `1217` trades, `+5255.95` ticks, Sharpe `0.281`, DSR `0.279`, max DD `82.26%`, WFE `1.110` to `724` trades, `+12966.40` ticks, Sharpe `0.912`, DSR `0.859`, max DD `38.59%`, WFE `1.723`.
- Post-pass slice read: all composite regimes are positive and all sessions are positive; 2022 remains the only negative calendar year at `-3303.55` ticks, led by `2022Q2` at `-2343.80` ticks.
- Gate read: S5 is materially better but still not promotable because max drawdown remains far above the `12%` advisory gate and largest-single-trade concentration still fails (`16.21%` of total PnL).
- Next S5 action: do not promote. If S5 remains interesting, the next pass should target train-window/date-regime instability around the 2022 drawdown rather than another broad composite prune.

##### 15.5 Pause S6 Training

Do not spend more training budget on S6 yet.

- The recall-oriented hyperparameter run did not improve realized economics.
- Full-window result stayed negative at `-1961.25` ticks.
- Recent2y result stayed negative at `-2172.80` ticks.

Next action is policy/threshold slicing, not another blind training sweep.

```powershell
Get-Content model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_opt_20260829\policy_table.csv
Get-Content model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_opt_20260829\frvp_long_reversal_setup6_xgb_v1\test_slice_report.csv
```

Executed S6 pause review on 2026-08-30:

- Reviewed the optimized recency S6 policy table and test slice report.
- All S6 composite policy rows still used the `0.05` global fallback threshold; composite regime thresholds were not data-sufficient.
- Full-window optimized backtest stayed negative: `345` trades, `-1961.25` ticks, Sharpe `-0.342`, DSR `-0.291`, max DD `33.40%`, WFE `0.368`.
- Recent2y optimized backtest stayed negative: `192` trades, `-2172.80` ticks, Sharpe `-0.528`, DSR `-0.374`, max DD `33.30%`, WFE `1.605`.
- Full-window breakdown confirms the drag is not a single obvious pocket: `2025` was `-1757.50` ticks and `2026` was `-1199.95` ticks; London was `-1304.50` ticks and off-hours was `-618.60` ticks.
- Composite policy/backtest read remains mixed, with only `4/9` realized composite buckets positive and the larger negative buckets overwhelming the positive ones.
- Decision: keep S6 paused for training. If revisited, do policy research around threshold sufficiency and session/year instability first; do not run another blind hyperparameter sweep.

##### 15.6 Keep S4 Research-Only

S4 trained, but the policy backtest only produced one fold while the command required four.

Keep S4 out of promotion comparisons. For the next S4 read, either run an explicit one-fold research report or lower `--min-folds 1`.

```powershell
ote_venv\Scripts\python.exe scripts\run_ote_policy_backtest.py `
  --regime-report-root model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829 `
  --registry-path models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json `
  --output-root model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_1fold_research_20260829 `
  --status candidate `
  --model-id frvp_long_reversal_setup4_xgb_v1 `
  --instrument es `
  --spread-cost-mode session_schedule `
  --min-train-years 2 `
  --test-window-months 3 `
  --rolling-step-months 3 `
  --min-folds 1 `
  --min-positive-events 10 `
  --min-events-per-month 0.5 `
  --min-trades-per-week 0.5 `
  --evaluation-contract-mode research `
  --minimum-sharpe 0.8 `
  --maximum-drawdown-pct 12.0 `
  --minimum-dsr 0.3
```

Executed S4 one-fold research report on 2026-08-30:

- Output root: `model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_1fold_research_20260829`.
- The one-fold research run completed successfully with `effective_min_folds=1`, `evaluation_contract_mode=research`, and `promotion_quality_gate_eligible=false`.
- Result: `5` selected test trades, `+392.75` ticks, expectancy `+78.55` ticks, hit rate `100%`, Sharpe `4.409`, DSR `0.000`, max DD `0.00%`, WFE `8.152`.
- Breadth is too thin for promotion: the only realized test period was `2026-04-01` through `2026-06-30`, with `2` Asia trades and `3` New York trades.
- Concentration is too high: largest single trade share was `70.36%` of total PnL, so the concentration gate failed even though the tiny sample was profitable.
- Decision: keep S4 research-only and out of setup promotion comparisons. The positive one-fold read is worth preserving as a curiosity, but not enough to justify roster changes or more training budget without a broader event/fold design.

<a id="frvp_setup_specific_model_leaderboard_20260830"></a>

### FRVP Setup-Specific Model Leaderboard

*Merged 2026-10-09 from `docs/FRVP_setup_specific_model_leaderboard_20260830.md`.*

Date: 2026-08-30

Scope: long setup-specific FRVP targets after the second experimentation round. Metrics are from saved `model_summary.csv` / `summary.json` outputs under `model_testing\reports\frvp_backtests`.

#### Executive Read

No setup-specific model should be promoted into the accepted paper roster yet.

Two lanes are useful enough for quarantined shadow paper monitoring:

- `frvp_long_continuation_setup2_xgb_v1` with `frvp_long_setup2_medium_session_prune_20260829`.
- `frvp_long_continuation_setup5_xgb_v1` with `frvp_long_setup5_repeated_drawdown_prune_promotion_20260829`.

Use "shadow paper" here to mean log-only signal observation with separate reporting, no roster replacement, no combined portfolio gating, and no pass/fail credit toward production until a fresh forward window accumulates. Do not treat these as accepted paper-trading models because each still fails at least one important promotion gate.

#### Best Variant Leaderboard

| Rank | Setup | Best Saved Variant | Folds | Trades | Net ticks | Expectancy | PF | Sharpe | DSR | Max DD | WFE | Gate Read | Shadow Paper |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | S5 continuation | `frvp_long_setup5_repeated_drawdown_prune_promotion_20260829` | 20 | 724 | +12966.40 | +17.91 | 1.391 | 0.912 | 0.859 | 38.59% | 1.723 | Fails drawdown and single-trade concentration | Yes, shadow only |
| 2 | S2 continuation | `frvp_long_setup2_medium_session_prune_20260829` | 9 | 104 | +4344.40 | +41.77 | 2.265 | 1.625 | 1.153 | 8.36% | -4.127 | Fails WFE and single-trade concentration | Yes, shadow only |
| 3 | S1 reversal recency recent2y | `frvp_long_reversal_setup_recency730_recent2y_20260829` | 8 | 173 | +1524.55 | +8.81 | 1.192 | 0.284 | 0.240 | 24.12% | -5.943 | Broad instability; fails most quality gates | No |
| 4 | S4 reversal research | `frvp_long_setup4_fullspan_1fold_research_20260829` | 1 | 5 | +392.75 | +78.55 | n/a | 4.409 | 0.000 | 0.00% | 8.152 | Research-only, one fold, 70.36% single-trade concentration | Log-only curiosity |
| 5 | S1 reversal full-span | `frvp_long_setup_fullspan_20260829` | 11 | 234 | +178.90 | +0.76 | 1.016 | 0.025 | 0.025 | 34.89% | -0.031 | Essentially flat after costs | No |
| 6 | S6 reversal recency full-window | `frvp_long_reversal_setup_recency730_opt_20260829` | 12 | 345 | -1961.25 | -5.68 | 0.870 | -0.342 | -0.291 | 33.40% | 0.368 | Negative realized economics | No |
| 7 | S6 reversal recency recent2y | `frvp_long_reversal_setup_recency730_opt_recent2y_20260829` | 8 | 192 | -2172.80 | -11.32 | 0.794 | -0.528 | -0.374 | 33.30% | 1.605 | Negative realized economics | No |
| 8 | S6 reversal full-span | `frvp_long_setup_fullspan_20260829` | 12 | 251 | -2897.15 | -11.54 | 0.766 | -0.523 | -0.513 | 37.17% | 0.621 | Negative realized economics | No |
| 9 | S3 continuation | `frvp_long_setup_fullspan_20260829` | 16 | 485 | -2964.25 | -6.11 | 0.887 | -0.379 | -0.375 | 49.96% | -0.562 | Negative realized economics | No |

#### Full-Round Notes

S5 is the highest economic result from the setup-specific work. The policy-pruned promotion-quality run improved the original S5 from `+5255.95` ticks, Sharpe `0.281`, DSR `0.279`, max DD `82.26%`, WFE `1.110` to `+12966.40` ticks, Sharpe `0.912`, DSR `0.859`, max DD `38.59%`, WFE `1.723`. That is a real improvement, but the drawdown is still far outside the `12%` advisory gate and largest-single-trade concentration still fails.

S2 is cleaner than S5 from a risk-stat perspective after medium/session pruning: `+4344.40` ticks, PF `2.265`, Sharpe `1.625`, DSR `1.153`, and max DD `8.36%`. The problem is train-side instability: WFE is `-4.127` because mean train annualized PnL is negative while test annualized PnL is positive. This could be a genuine recent-regime edge, but it is exactly the kind of thing paper shadowing should test before promotion.

S1 is not currently worth shadow paper. The recent2y recency variant is positive, but only modestly, with low Sharpe, high drawdown, negative WFE, and weak composite breadth.

S4 should stay research-only. The one-fold report was positive, but it only produced `5` trades in one realized test quarter. One trade contributed `70.36%` of total PnL, so the result is not broad enough to trust.

S6 and S3 should stay paused/dropped. S6 stayed negative in full-window and recent2y optimized recency tests, and the S6 policy table still mostly fell back to the `0.05` global threshold because composite thresholds were not data-sufficient. S3 has good classifier optics but negative trading economics.

#### Paper Trading Recommendation

Do not put any setup-specific model into the normal accepted paper roster yet.

Do run a quarantined shadow-paper watchlist if operationally cheap:

| Candidate | Shadow Mode | Why |
| --- | --- | --- |
| S2 pruned | Highest priority | Best risk-adjusted read, drawdown repaired, strong expectancy; needs forward evidence that the negative WFE is not just instability. |
| S5 pruned promotion-quality | Medium priority | Best absolute PnL and positive WFE; needs live observation because historical drawdown is still too high for roster inclusion. |
| S4 one-fold | Optional log-only | Interesting failed-auction signal, but too rare and too concentrated for anything more than event collection. |

Shadow-paper guardrails:

- Keep these signals out of the accepted paper roster and out of any production allocation.
- Tag them with explicit experiment IDs so their fills, misses, and abstains cannot pollute current model reporting.
- Review after a minimum forward window, not after the first good/bad week. Suggested minimums: S2 at least `30` shadow signals, S5 at least `100` shadow signals, S4 at least `20` events or two additional quarters, whichever comes later.
- Stop early only for operational defects or obvious cost/model-routing bugs, not ordinary variance.

#### Backend Attribution Clarification

Backend attribution was not skipped for the setup-specific XGBoost prepared root.

The setup-aware Phase 4 root contains XGBoost attribution artifacts, including:

- `artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared\backend_attribution_summary.json`
- per-target `backend_attribution_summary_xgboost.json`
- per-target `feature_importance_merged_xgboost.csv`
- per-target `shap_feature_stats_xgboost.csv`

What we skipped was standalone TCN backend attribution and any TCN training. That was intentional because this experiment was scoped as an XGBoost-only target-routing/policy experiment:

- Wave 1 explicitly said: do not run TCNs.
- Setup targets inherited pooled-family quality, weights, and concurrency so the first comparison changed target membership and policy, not model family.
- Recency prepared roots copied the base prepared target files and only reweighted `train` / `val` sample weights; they did not need a fresh attribution pass because the feature universe and target rows were intentionally held constant for comparability.

Next attribution action: before any third-round retrain, review the existing per-target XGBoost attribution summaries for S2 and S5. Only rerun attribution if the next experiment changes the feature universe, target definition, backend family, or recency weighting policy enough that copied rankings are no longer a fair comparison.

#### Recommended Next Steps

1. Create a shadow-paper bundle for S2 pruned and S5 pruned promotion-quality only.
2. Add explicit experiment IDs and reporting separation for setup-specific shadow signals.
3. Run an S2 train-window instability audit before retraining: focus on why train expectancy is negative while test is strongly positive.
4. Run an S5 date-regime audit around the 2022 drawdown before any further pruning or retraining.
5. Do not spend more training budget on S3 or S6 until new setup logic or feature context changes.
6. Keep S4 as event-collection research, not promotion research.

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

## 2026-09-18 - B2 saved lineage inventory and dependency trace

- `Why:` determine which historical FRVP artifacts inherited the hour-open structural-ATR defect and label-derived live-helper mismatch before rebuilding or retraining anything.
- `What / how:` traced all 34 FRVP bundles under `models/`, ten trained placebo bundles and two saved temporary smoke runs. Recorded parents through label/feature merges, prepared targets, recency materializations, selected features, fitted artifacts and runtime manifests. Hashed current files and ran bounded read-only saved-data comparisons. The wider inventory covers 113 primary bundles plus eight auxiliary weight files across all three strategy families.
- `Result / what worked:` the current, July refresh, August setup-target and post-2022 label tapes all reproduce early hourly ATR publication: 3,708 comparable rows per first-4,000-row prefix in the first three tapes, and 3,828 in the post-2022 tape, match the hour-open reconstruction and differ from the completion-stamped timing control. All 34 non-placebo model bundles and ten placebo bundles declare one of those label parents. Thirteen FRVP model bundles plus the ten placebos select label-produced confluence helpers; six FRVP model IDs in the frozen live snapshot have only zero values for those helpers despite nonzero prepared training values.
- `What did not work / limits:` the two temporary smoke runs reference missing old feature/prepared roots. The witnesses do not measure full-history feature/label changes or prove which filtered training rows changed. Current hashes do not certify training-time input identity. Complete inventory, per-model matrix and source anchors: `docs/B2_model_repair.md#b2_lineage_inventory_and_dependency_trace_20260918`.
- `Observations:` structural ATR affects event/label calculations, while the detector's feature ATR remains a separate dependency. The shared swing-match helper accepts future swing anchors, so simply reproducing that retrospective helper in live inference is not a causal repair. Recency weighting and setup specialization preserve their defective parent lineage.
- `Decision:` inventory/tracing complete; B2 remains open for consumed-row impact review and minimum corrective replacements. No historical artifact, training job, calibration, policy, runtime manifest or eligibility gate was changed. Keep pooled models diagnostic-only, S4 paused, short reversal retired and all existing economics/drawdown/concentration/promotion restrictions intact.

## 2026-09-19 - B2 timing impact and proposed minimum repair

- Full-history timing diagnostics and fixed-event replay find changed training targets in all 34 non-placebo bundles with available parents: 1-107 per model, with zero old-target mismatches on joined comparable rows. Reproduced, non-excluded events change target-hit flags by 1.64-1.69% across the four overlapping label batches and holding times by 14.66-15.30%. The comparison preserves saved setups, exclusions and historical barrier rules; it is not a full relabeling, quality/weight replay or performance estimate. Unreproduced ATR rows remain unverified.
- Proposed repair: publish completed hourly ATR, recalculate affected event/label/quality/weight products, rebuild prepared inputs and retrain retained models. Preserve the independent local-ATR detector/features; remove retrospective confluence predictors or replace them with a proven causal offline/live producer. Keep ten placebo controls archived unless their experiment is needed; two smoke bundles have missing parents.
- Evidence: `docs/B2_model_repair.md#b2_minimum_repairs_20260919`, per-model repair CSV and `docs/audits/b2_impact_20260918/`. No production code, historical data, model, calibration, threshold or runtime change; no training or baseline release. B2 remains open. Pooled models stay diagnostic-only, S4 paused and short reversal retired.


## 2026-09-21 - B2 retained shared-data repair

- Corrected ICT references, FRVP completed-hour ATR/confirmed-swing gates and OTE completed-HTF publication. Historical label controls reproduced before correction; label-derived helper predictors are excluded from new preparation.
- Rebuilt and checked 25 prepared datasets for 27 retained models; preserved independent FRVP profiles/setup inputs and the existing long-reversal recency recipe. The nine missing-history bundles remain quarantined.
- Validation: 127 tests passed; chronological split gaps, finite features/weights and target coverage checked. 1,489 historical model/companion files, the OTE registry and 12 key data parents are unchanged. No trading model was trained or activated; existing research/shadow/diagnostic restrictions remain.
- Report and replacement-data paths: [B2 shared-data repair](B2_model_repair.md#b2_shared_data_repair_20260921). B2 remains open for selected-model retraining/calibration, versioning and B1 evaluation.

## 2026-09-24 - B2 long S3/S5 replacement review

- `Why:` assess the completed exploratory replacements before scheduling further training or evaluating outcomes.
- `What / how:` reviewed IDs 6/7 against pinned corrected data, frozen selection, saved recipes and seeds, CV/final-holdout geometry, OOF thresholds and all 24 output-file hashes; replayed predictions using saved models and scalers. Both runs retain their 36/40 trial budgets, with 18/12 completed trials and 18/28 pruned trials, respectively; none failed.
- `Result / what worked:` both artifact reviews pass; prediction replay errors are below 0.00000006 and threshold decisions match. Six queue regression tests pass. S3 test weighted AUC is 0.5808 and AP 0.6523 versus prevalence 0.5962; all eight CV folds have AUC above 0.5.
- `What did not work / limits:` S5 pooled OOF AUC is 0.5051 and test AUC 0.4884; test AP 0.6386 is below prevalence 0.6498. S3's reported 100% event precision is tolerance-based matching on sparse retained rows, not a trade win rate. The reused historical tests do not establish independent net outcomes or a common-opportunity comparison.
- `Observations:` S3 has modest signal worth evaluating first; S5 remains useful for the planned same-side coverage/comparison, without evidence supporting another search budget. Full evidence and reproducible review: [B2 exploratory review](B2_model_repair.md#b2_exploratory_review_20260924).
- `Decision:` freeze the six completed candidate bindings and predeclare B1/B3 setup-only, frozen-threshold and full-policy comparisons. Preserve S3/S5 thresholds 0.71/0.65 and continuation calibration behavior; defer background/OTE training until a named comparison needs it. This review fitted no trading models and changed no model, threshold or live registry. B2 outcome evaluation and other readiness/promotion prerequisites remain open.

## 2026-09-27 - B2 frozen S3 historical outcome backtest

- `Why:` test whether S3's modest classifier discrimination improves the economics of matching long S3 setups under the existing B1 contract.
- `What / how:` predeclared 0.71/no calibration, the entire original held-out source interval, all matching detector opportunities, completed-hour ATR, existing B1 exits/costs, one ES contract and one position at a time. Reconstructed frozen-prefix imputation and used consecutive source-bar inference; no label-based opportunity selection. Compared setup-only and fixed-threshold branches, including doubled-cost stress.
- `Result / what worked:` 471 observations, 449 with common readiness. Setup-only: 374 closed trades, net -$37,501.25, -$100.27/trade, profit factor 0.722. Model-filtered: 148 closed trades, net -$19,346.25, -$130.72/trade, profit factor 0.627. No unresolved admitted positions or same-bar barrier collisions. Both saved runs replay identically; 189 tests passed.
- `What did not work / limits:` the filter reduced exposure but worsened per-trade loss. The historical period is already inspected, and the full-policy branch has no frozen replacement policy or observed arrival/feature-parity evidence. Independent calendar missing slots remain explicit; 22 observations lack causal ATR readiness. Consecutive-bar history changes 54/289 threshold decisions relative to original retained-row replay, which itself still reproduces successfully.
- `Observations:` original label metrics and event precision did not imply tradable profitability. The backtest preserves all rejected opportunities and the distinction between independent signals and nonoverlapping positions. Evidence, contracts, costs, equity plot and reproduction: [S3 historical backtest](B2_model_repair.md#b2_s3_backtest_20260927).
- `Decision:` keep S3 research-only and outside qualified shadow collection. Audit the history/imputation deployment contract before considering another version; reserve untouched data for any later validation. Preserve 0.71 and the negative historical result without tuning hours, exits or thresholds on it. B2 clean evaluation and other readiness gates remain open. No fitting, model activation, registry changes or ES collector changes occurred.

## 2026-09-27 - B2 S3 source-history pipeline repair

- `Why:` label-selected training rows compressed the lookback timeline and changed 54/289 threshold decisions when the old model received consecutive market bars.
- `What / how:` separated labeled endpoints from full source history in XGB CV, holdout, refit and test tensors. Prepared all 666,769 source bars with frozen-prefix missing-value fills; saved the contract with the scaler and enforced it at runtime. Added a dedicated S3 preparation/check/train/review workflow and blocked the old ID 6 training route.
- `Result / what worked:` preserved labels, weights, selection prefix, seed, 36-trial recipe and separation gates. All eight window choices pass preflight, with 2,536 exact training/runtime input comparisons. Synthetic model save/load replay tests exercise raw missing values and reject filtered history. All 73 original outputs across six completed models retain their saved hashes.
- `What did not work / limits:` broader validation has 107 passes, two skips and the previously documented unsupported cv_max_folds failure. No replacement trading model has been trained; exact real-data model-score replay must follow retraining. S3 XGB alone is migrated; market-clock gaps, roll/freshness eligibility and full live feature production are not certified here.
- `Observations:` this fixes input meaning, not economic performance. Old 0.71 and negative backtest results remain archived. Long-reversal 730-day/0.20 weighting is unchanged. Evidence and commands: [S3 source-history repair](B2_model_repair.md#b2_s3_source_history_repair_20260927).
- `Decision:` next run the corrected S3 36-trial job and its artifact/runtime review, then freeze a replacement-specific B1 comparison and untouched/prospective evaluation. Select the new threshold only from new development OOF predictions. No live registry, ES collection or qualification change; remaining B2/B3-B5/A12/A13 gates remain open.

## 2026-09-27 - Corrected S3 retraining review and frozen three-branch backtest

- `Why:` assess the corrected source-history model at its own frozen 0.65 threshold, including the declared full-policy branch, before considering qualified ES shadow collection.
- `What / how:` the user's 36-trial run completed with 10 complete, 26 pruned and zero failed trials. Artifact/runtime replay passes all 301 saved predictions (maximum error 8.84e-08), with identical decisions and OOF-only threshold reproduction. Test weighted AUC is 0.6190, versus 0.5808 previously, but pooled OOF AUC fell to 0.5655 from 0.5889. Froze a new historical comparison on all 471 matching long S3 opportunities, identical B1 exits/costs, independent calendar, common readiness, one contract/position, and the existing packaged S3 exclusions rebound offline to the new model.
- `Result / what worked:` 449 common-ready opportunities. Setup-only: 374 trades, net -$37,501.25, -$100.27/trade, PF 0.722. Corrected threshold: 207 trades, net -$11,244.38, -$54.32/trade, PF 0.822, realized max drawdown $13,746.88; double-cost net -$18,276.25. Ten of twenty traded months are positive. All three branch replays match; independent accounting/capacity/input-hash checks pass and 202 tests pass. Zero threshold disagreements among 289 matching saved-test observations, resolving the prior 54-decision discrepancy.
- `What did not work / limits:` model-filtered trading remains negative even before costs. Full policy admits zero trades because expected_move_pips is absent for all 471 observations; production shadow gates correctly fail closed. Its zero ledger P&L is not a profitable result. Existing session-spread gate units also need an explicit ES contract. No proxy, outcome-tuned filter or bypass was used. The inspected historical interval, one-second timing assumption and 408 explicit calendar/data missing slots do not certify fresh validation or observed live availability.
- `Observations:` the repair improves this historical average loss per trade by approximately 45.8%, but does not establish a trading edge. Full-policy economics remain unassessed until its expected-move producer and units are resolved. Detailed declarations, equity, branch ledgers, rejected opportunities, test evidence and reproduction: [Corrected S3 backtest](B2_model_repair.md#b2_s3_corrected_backtest_20260927).
- `Decision:` preserve the corrected model/0.65, archived old model/0.71 and both historical results. Keep S3 research-only. Next define and verify causal offline/live expected-move inputs and ES units, freeze and evaluate the completed policy in a new version, then follow existing live-parity, prospective comparison and collection-readiness prerequisites. No fitting, live installation, registry/collector changes, broker orders or paper-trial activation occurred during this backtest.

## 2026-09-28 - Causal ES expected-move contract and S3 full-policy v2

- `Why:` complete the missing movement/spread gate without using profit-target distance, disabling the gate or selecting parameters from inspected outcomes.
- `What / how:` froze es_abs_close_move_20x5m_v1: mean absolute five-minute close change over the last 20 valid returns, expressed in ES ticks, requiring 21 completed raw same-contract bars. Gaps/closures/rolls/invalid data reset warmup; valid zero differs from unknown. The shared producer runs before app model imputation/trimming. Explicit opt-in policies use B1 ES session spreads in ticks and preserve inherited multipliers. S3 keeps 0.65, ratio 2, stress/session exclusions, cooldown and fixed B1 exits/costs.
- `Result / what worked:` 258 regression tests pass. Six-candidate gate-input audit covers 3,254 endpoints and 514 actual app-frame witnesses. All 471 S3 opportunity estimates are ready. Frozen v2 full policy admits 169 closed trades: gross +$4,212.50, costs $5,673.13, net -$1,460.63, -$8.64/trade, PF 0.970, realized max drawdown $6,547.50; doubled-cost net -$7,133.75. Ten of twenty traded months are positive. All three branch replays match; setup-only/model-threshold ledgers reproduce v1 exactly and every blocked-v1 artifact hash is preserved.
- `What did not work / limits:` full policy still loses after costs. Its 195 independent overlapping signals average +$17.74, while actual one-position outcomes average -$8.64; do not report independent signals as portfolio P&L. This is an already inspected historical sample with assumed availability, not fresh validation or live feature certification. Other models retain separate history/qualification gates. IDs 1/11 already disable abstention in their legacy policy sources; the audit records that fact without changing them.
- `Observations:` movement magnitude is not directional profit expectancy. The full package improves this diagnostic but does not isolate the movement gate's contribution or establish promotion readiness. Gate reasons across all observations include 21 below the movement/spread hurdle, 61 high stress, 22 cooldown and four off-hours; reasons overlap with threshold/readiness failures.
- `Decision:` expected-move prerequisite and historical comparison complete; retain all versions and keep S3 research-only. Do not tune lookback, multiplier, hours, exits or thresholds to these outcomes. Continue other-candidate B2 history/outcome work, live parity and B3-B7/A12/A13 gates before any qualified collection. No model fitting, installed policy/registry changes, collector restart, broker orders or paper-trial activation. Evidence: [ES movement contract](B2_model_repair.md#es_expected_move_contract_20260927), [S3 full-policy v2 report](B2_model_repair.md#b2_s3_full_policy_backtest_20260928).

## 2026-09-29 - Selected FRVP candidate history repair and retraining

- `Why:` audit each remaining selected FRVP branch independently of S3. Consecutive source-bar histories change all held-out inputs for IDs 2/1/7 and change 123/215/178 archived threshold decisions. Frozen training fills were also absent from their runtime scalers; TCN rejects 56 raw windows containing missing values.
- `What / how:` prepared full source history and frozen prefix fills, then retrained only long continuation XGB, short continuation TCN and retained long S5 XGB. Preserved selected-feature order, split identities, labels, weights, search budgets, validation gaps and development-only selection. TCN sequence sampling now selects endpoints without compressing the underlying market history. Archived S3 and the earlier S3 replacement remain unchanged.
- `Result / what worked:` all 72 FRVP trials attempted: 32 complete, 40 pruned, zero failed. All 2,527 saved test predictions replay through the production backend/runner with identical decisions. New OOF-only thresholds for IDs 2/1/7 are 0.62/0.56/0.62. Maximum replay error is below 1.23e-7. The first TCN GPU-helper replay failed tolerance; correcting the reviewer to use the actual CPU loader resolves it without relaxing tolerance or changing fitting code.
- `What did not work / limits:` this is input/lineage verification, not new B1 profitability evidence or actual app feature-producer parity. Focused tests pass 33; broader pipeline passes 37, skips two and reproduces the documented unsupported cv_max_folds failure. The inspected test periods are not untouched evaluation data.
- `Observations:` all 85 checked old outputs across FRVP/ICT and both S3 versions retain their hashes. Deferred long reversal retains the independently checked 730-day/0.20 weighting contract and original split weights; it was not trained.
- `Decision:` freeze these offline replacements, retain diagnostic-only status and all short-side economics/drawdown/promotion restrictions. Remaining outcomes, live parity and collection gates stay open. Background/OTE fitting is not a blanket prerequisite. No live installation, collector restart, qualified collection, paper trial or broker order. Evidence: [candidate history audit](B2_model_repair.md#b2_candidate_history_audit_20260928), [completion pins](audits/b2_candidate_history_completion_20260928.json).

## 2026-09-29 - B1 named-candidate historical outcome review

- `Question:` determine whether the remaining FRVP candidates have candidate-specific post-cost evidence under the frozen B1 contract, using the same matching opportunities for setup-only and frozen-threshold comparisons.
- `Control / method:` bind ID 7 to `es_frvp_setup5_long_eval_v2`, use the exact replacement-model held-out named S5 endpoints, one ES position, full B1 costs and the predeclared B3/B4 comparison. Two endpoints whose contemporaneous detector row was setup 3 were excluded rather than relabeled.
- `Result:` 490 common S5 opportunities. Setup-only admitted 347 trades for **-$8,756.88** net; frozen 0.62 threshold admitted 101 for **-$12,358.13** net. The full-policy leg is insufficient evidence because no exact replacement-version policy binding was frozen.
- `What did not work / limits:` this is historical held-out economics under a declared offline timing assumption, not live feature parity, qualified collection or promotion evidence. The negative result must not trigger threshold or policy tuning on this inspected sample.
- `Decision / repo change:` ID 7 is reviewed but not promotion-cleared. It may be considered for prospective research only after live parity and B5-B7/A12/A13 gates. Pooled FRVP IDs 1 and 2 are explicitly excluded from the qualified baseline by `baseline_scope_v1` and remain diagnostic-only. Evidence: [B1 candidate outcome review](B1_outcome_contracts.md#b1_candidate_outcome_review_20260929), [audit JSON](audits/b1_candidate_outcome_review_20260929.json), [frozen comparison](../configs/b1_candidate_outcome_review_20260929_v1.json).

## 2026-09-29 - B5 baseline eligibility, FRVP priorities, and promotion contract

- `Question:` which FRVP branches are eligible for the next prospective scored shadow period, and what evidence must still be true before any branch can be promoted?
- `Control / evidence:` the B4 freeze in `configs/b4_baseline_freeze_20260929.json`, the B1 candidate outcome review, the B5/B6 parity audit in `docs/B5_B6_live_feature_parity_20260929.md`, and the previously recorded FRVP contracts and promotion findings below. The B4 freeze is a research binding; it is not an installation or promotion decision.
- `Result:` no FRVP branch is currently eligible for prospective scored shadow research. The six-branch B5/B6 audit recorded zero full-parity passes and zero promotion passes: raw pre-imputation selected features were not persisted, completion and availability were unknown for every recorded candidate snapshot, unresolved non-closure gaps remained, and the bounded actual `IncrementalFeatureEngine` replay was not run. IDs 1 and 2 are also pooled and excluded from the qualified baseline by `baseline_scope_v1`.

### FRVP eligibility and research-priority roster

| Branch | Collection status now | Research priority | Binding reason / unresolved gate |
|---|---|---|---|
| ID 1, pooled short continuation TCN | Diagnostic-only; excluded from qualified baseline | Background comparison | Pooled-model scope restriction; B5 raw producer parity and completion/availability evidence are missing. |
| ID 2, pooled long continuation XGB | Diagnostic-only; excluded from qualified baseline | Background comparison | Pooled-model scope restriction; B5 raw producer parity and completion/availability evidence are missing. |
| ID 6, long S3 setup specialist | Diagnostic-only; deferred from scored collection | Retain for causal/model-versus-setup diagnostics | Corrected full-policy historical result is net negative (`-$1,460.63`, PF `0.970`); exact replacement-version collection binding and B5 parity are unresolved. |
| ID 7, long S5 LVN Burst specialist | Diagnostic-only; not promotion-cleared | Highest FRVP named-candidate priority, conditional on clearing gates | B4 frozen at threshold `0.62`, but B1 setup-only was `-$8,756.88` and frozen-threshold was `-$12,358.13`; B5 parity is not proven and the exact replacement full-policy leg is insufficient evidence. |
| Other FRVP families, including S4 and short reversal | Background/deferred or excluded per prior roster | No change | S4 remains paused pending a model-specific policy; short reversal remains retired; prior drawdown, concentration, economics, and readiness restrictions remain binding. |

There are currently zero rows in the `eligible for prospective scored shadow research` category. A named candidate can become eligible only after the B5/B6, B7, A12 and A13 gates pass under a new immutable collection identity; eligibility for research will still not mean promotion to trading.

### Promotion criteria retained for FRVP

Promotion is a conjunctive decision against the same common opportunities and must use all three B3/B4 surfaces: setup-only, setup plus the frozen model threshold, and setup plus the intended full policy. A branch cannot pass by showing a model-only result, a zero-trade full-policy ledger, or an incomplete policy binding.

1. **Contract and readiness:** exact replacement model, scaler/history, setup definition, threshold/calibration, policy, instrument, and B1 outcome-contract hashes must be bound; actual live feature-producer parity must cover raw selected features, causal history, completion/availability, warmup, gaps, closures, rolls and fallbacks; and B7/A12/A13 must pass.
2. **Intended exits and costs:** the full policy must be evaluated with the frozen FRVP B1/B3 exits, entry timing, timeout/management, adverse-first resolution, one-position/concurrency rules, and realistic ES scheduled costs. Gross movement, target hits, setup labels, or setup-only P&L are not substitutes for closed net position economics. Cost stress must be reported and cannot be hidden by changing the exit contract.
3. **Incremental economics:** after realistic costs, the full-policy branch must be profitable and economically improve on the unchanged setup-only benchmark on the same opportunities (net expectancy/return and risk must not be worse without an explicitly reviewed trade-off). A positive setup-only control with a negative model-filtered branch is a model failure for promotion, as shown by ID 17 in the ICT companion review; a negative setup-only control is not rescued by a small selected pocket.
4. **Time stability and discrimination:** use strict chronological, leakage-safe folds with the existing minimum fold/cadence requirements; require the configured Sharpe/DSR, positive WFE, profitable-quarter breadth and positive-composite-expectancy gates, plus evidence of discrimination over the setup/base-rate control. A recent-window or one-hour result cannot replace full-span stability.
5. **Concentration and drawdown:** largest-single-trade share must be at or below the existing `10%` gate, with day/session/regime overlap reported; the declared drawdown advisory/risk limit must be respected or the branch remains non-promoted pending explicit risk review. The older FRVP findings remain active: continuation's exact fixed live policy had `7.64%` drawdown but `14.53%` largest-trade share, the accepted recent-regime reversal checkpoint had `12.87%` concentration, S5 had `38.59%` drawdown, S2 had `WFE -4.127`, and S4's five-trade result had `70.36%` concentration.

The September one-hour audit, tiny hourly samples, retrospective profitable-hour pockets, or any other small positive pocket cannot alter these gates, thresholds, setup definitions, or full-session coverage. Any session restriction is a separately frozen experiment with the full-session control retained.

### Prospective collection boundary

If a branch later clears the prerequisite gates, its clean C1 sample starts at the first healthy post-B7 timestamp under a new immutable candidate/runtime identity. Collection remains full-session across eligible hours, retains both emitted and rejected same-side setup opportunities, and is reviewed after approximately 20 fully collected sessions with 50-100 resolved matches as a practical target (rare specialists may require longer). A model, threshold, calibration, setup matcher, policy, cost, session scope, or exit change creates a new version and restarts the clean boundary; it cannot be patched into the observed sample.

- `Decision / repo change:` B5 is recorded as a gate-setting decision, not a clearance. FRVP keeps no scored-shadow-eligible branch, keeps ID 7 as the conditional named research priority, and preserves pooled-model diagnostic-only scope, S4 pause, retired short reversal, older full-contract economics, and all drawdown/concentration/readiness restrictions. No model, registry, threshold, collector, paper trial, or order state changed.

## 2026-10-01 - B2 catch-up: ID 5 `frvp_long_continuation_setup2_xgb_v1` source-history repair

- `Why:` ID 5 (reserve tier 1) must reach the same B2 stage as the 12-model core before the 16-model B4 freeze and route manifest.
- `What:` New prepared root `artifacts/b2_candidate_history_catchup_es_20261001_v1/5` and model root `models/b2_candidate_history_catchup_es_20261001_v1/05_long_frvp_continuation_setup2`; legacy bundle untouched and not used.
- `How:` `scripts/run_b2_candidate_history.py prepare/check/train/review 5` on the 09-21 repaired FRVP data (completed-hour ATR, confirmed-only HTF gates), full consecutive 666,769-bar tape, frozen 160 features / 400-row selection / frozen fills; original recipe, folds, purge, purged holdout, seeds and 32-trial search.
- `Result:` Technical review passed (`docs/audits/b2_catchup_5_review_20261001.json`, supplement `…_review_supplement_20261001.json`): 7 windows x 162 endpoints exact; trials 17 complete / 15 pruned / 0 failed; OOF threshold 0.62 reproduced; 146 test predictions replayed through the production runner, max error 8.5e-08, identical decisions, 0 history-driven disagreements.
- `What worked:` The catch-up roster admission and the 09-30 runner needed no fitting or review changes.
- `What didn't:` Nothing technical. Advisory quality is weak out of sample: test ROC AUC 0.356, AP 0.561 (OOF AP 0.640) on 146 rows.
- `Observations:` Calibration resolves to `none` for this continuation target, as in the original recipe. Sample is small (11 predicted / 21 true test events).
- `Decision:` `reviewed_pass`, diagnostic-only, `collection_eligible=false`. Advisory metrics do not gate B2; no threshold or recipe change. Details in `docs/B2_model_repair.md#b2_candidate_history_catchup_5_14_18_23_20261001`.

## 2026-10-01 - ID 5 in the 16-model B4 freeze v2 and route bindings v2

- `Why:` ID 5 must sit at the same stage (section 3, item 1) as the 12 core models before the remaining gates run for all 16.
- `What:` ID 5 frozen in `broad_full_session_shadow_v2` / B4 freeze v2 and routed in route bindings v2.
- `How:` `scripts/publish_broad_full_session_shadow.py publish --version 2` (re-hashes the 12 v1 entries, then adds 5/23/18/14); route follows ID 19's specialist pattern: `continuation_setup2`, exact contemporaneous same-side, `setup2` → `es_frvp_setup2_eval_v2`, `es_session_cost_v1`.
- `Result:` v1 entries and v1 routes carried forward identical; both manifests load and resolve; 175 focused tests pass.
- `What worked:` the v1 publisher and route loader needed only version constants and the catch-up review path.
- `What didn't:` nothing.
- `Observations:` threshold frozen at the OOF value 0.62, no calibrator, 160 features; full policy `not_defined`.
- `Decision:` identity and route freeze only; no collection, registry, paper, alert or broker authority. Details in `docs/B3_B4_freeze_and_routes.md#broad_full_session_shadow_route_bindings_v2_20261001`.

## 2026-10-02 - Section 3 surfaces, ledgers and session coverage (shared, ES FRVP models)

- `Why:` section 3 needs the four surfaces, both ledger kinds and full-session coverage bound before any smoke replay.
- `What:` shared module `model_testing/outcome_surfaces.py` (schema `section3-surfaces-v1`) and the D9 capacity ordering key in the b1-v5 simulator.
- `How:` one implementation for ES and EURUSD, kept in separate stores; fixture uses FRVP IDs 10 (pooled) and 5 (specialist) on the frozen route bindings v2.
- `Result:` `tests/test_section3_surfaces_ledgers.py` 16 passed. Synthetic bars only; no FRVP model was run on real data.
- `What worked:` the existing engine, route resolver and pooled-routing audit were reused unchanged apart from the ordering key.
- `What didn't:` nothing technical.
- `Observations:` all FRVP full policies stay `not_defined` (status only, no ledger). No P&L was produced or inspected.
- `Decision:` infrastructure only; no collection, registry, paper, alert or broker authority. Details in `docs/section3_simulation_rules.md#section3_progress`.

## 2026-10-02 - Section 3 deterministic smoke replay (FRVP IDs 2, 10, 3, 5)

- `Why:` section 3 needs one deterministic replay per exact frozen version to prove routing, fills, costs, ledger accounting and replay hashes.
- `What:` roster-driven harness `scripts/run_section3_smoke_replay.py` on engine b1-v5, B4 v2, route bindings v2 and B3 v3.
- `How:` windows chosen from setup presence and the calendar only (at most 4 weeks, hashed before outcomes); frozen bytes hash-checked; production runner on consecutive history; surfaces and ledgers built twice in separate processes.
- `Result:` IDs 2, 10, 3 and 5 passed: identical hashes across both runs, accounting checks clean, runner scores equal to the saved held-out predictions. All FRVP routes exercised (2: 3, 10: 3, 3: 6, 5: 1).
- `What worked:` the shared surfaces module and B1 review harness helpers were reused without change.
- `What didn't:` nothing technical. No position was left open in any window, so open-exposure carrying is proven by the synthetic tests only.
- `Observations:` ID 5 has 30 held-out rows without a setup; they stay in the census and never enter a ledger. Full policy `not_defined` for all four. P&L recorded as advisory, not interpreted.
- `Decision:` contract verification only; no collection, registry, paper, alert or broker authority. Details in `docs/section3_simulation_rules.md#section3_smoke_replay_20261002`.

## 2026-10-02 - Section 4 R1 raw snapshots and the D8 HTF finding (FRVP IDs 2, 3, 5, 10)

- `Why:` B5 parity needs raw pre-imputation values with explicit provenance and state from the actual app path.
- `What:` a copy-only capture (`s4-raw-preimpute-snapshot-v1`) after every producer and before any fill, attachable per group. It is disabled in the running app.
- `How:` the ES_FRVP group was built from hash-verified B4 v2 bytes and run on/off on isolated stores over 3 endpoints.
- `Result:` features, scaled inputs, scores and threshold decisions were bit-identical with snapshotting on and off.
- `What worked:` all four models warm and score on the shared group pass. NaN is preserved and missing/fallback states are explicit.
- `What didn't:` the paper-signal guard treats any manifest with the installed registry path as the controlled model, so staged manifests need a staging path.
- `Observations:` the store-fetch path drops `is_complete`, so `htf_context` rejects every bucket. On recorded bars, carrying provenance (D8) changes 111 of 362 group features (HTF, `continuation_*`, interactions), and endpoint NaN falls from 77 to 10. Today the app feeds frozen fills where the offline reference has HTF values.
- `Decision:` D8 stays opt-in and off. Whether to enable it is a user decision; see `docs/section4_progress.md`.
- `Update (2026-10-02, Q-D8 approved):` protocol v2 locked. Section 4 parity runs the app path with D8 on, so FRVP HTF and continuation inputs follow the offline gate-absent semantics on complete bars. Historical parity bars are copied as complete. B6 must enable D8.

## 2026-10-05 - Section 4 full live-parity run (FRVP IDs 2, 3, 5, 10)

- `Why:` B5 must prove that the real app path reproduces each frozen model's features, preprocessing, scores and threshold decisions.
- `What:` protocol v2 matrix (classified under v3). 15 ES windows covered recorded, historical and fault cases (RTH open, maintenance, DST, holiday, roll, weekend, gaps, backfill, revision, restart, cold start, producer failure), plus composition, as one shared ES_FRVP producer pass per window. Reports: `docs/audits/section4_parity_*_20261005.json`; summary: `docs/section4_progress.md#section4_parity_20261005`.
- `How:` app = `LiveSignalProcessor` on isolated parity DBs, with B4 v2 bytes hash-verified. Reference = frozen `source_history.parquet` (historical) or the frozen dataset config recomputed on recorded bars. Every mismatch group was classified with a recompute diagnostic.
- `Result:` all four fail.
  - IDs 2, 3, 10: `fail:training_side_noncausal`. `frvp_swing_poc_vs_session_poc` is revised once later bars arrive (141/145 bars), so the frozen training values used future bars.
  - ID 5: `fail:app_producer_bug`. Window truncation: the app's 8,000-bar history cannot hold the long-lived ICT/FRVP structures that frozen full-history values use; the app equals the offline config recomputed on its exact window.
- `What worked:`
  - restart (incl. after the roll and the maintenance break) and chunk resumes are bitwise equal to continuous;
  - each model alone equals the group engine;
  - backfill/revision behind the cursor is absorbed correctly on ES;
  - producer failures surface as errors, never silent fills;
  - S2: all reference FRVP events matched except one first-endpoint pair.
- `What didn't:` recorded ESU6 bars have unknown completion, so the HTF gate (declared behavior) differs from offline on those windows; they are not known-completion evidence. Source parity is incomplete (IBKR down).
- `Observations:` truncation and the swing-POC lookahead dominate the S3 mismatches. Decision flips across all windows: ID 2 122, ID 3 3, ID 5 46, ID 10 65 (most in completion-gate windows).
- `Decision:` all four branches stay diagnostic and unscored; no fix, retrain or recalibration in section 4.
  - Repair options for the user: a versioned history-length or state fix for truncation (prompt 5 candidate), and a retrain without the non-causal swing POC (2, 3, 10).

## 2026-10-07 - Follow-up proposal: training-data sufficiency, hard negatives and an every-bar model (FRVP)

- `Status:` proposed, not started. Runs after the v2 retrain (R3), B3 and B4; out of section 4 scope. Changes labels, so it needs new versioned label roots and its own retrain.
- `Why:` each FRVP model is a meta-label filter trained only on setup events. 9.5 years of ES history (2017-01-02 to 2026-06-19, 666,769 5-minute bars) yields few events per target, and the excluded ~660k bars carry no label for "did this setup win". The question raised (user, 2026-10-07): would training on every bar teach the model where to trade and where not to?
- `Evidence (v2 validation views, R2c):`
  - events (usable rows) per target: long_frvp_meta (ID 3) 9,080 (train 6,356 / 3,036 wins); long_frvp_continuation (2) 6,608 (3,213 wins); long_frvp_reversal (10) 2,472; long_frvp_continuation_setup2 (5) 999 (train 699 / 369 wins, test 146).
  - ID 5 has several hundred inputs (loaded features x lags) against 699 training rows: overfitting risk. Its test sample is too small to separate, e.g., 55% from 75% win rate (95% CI about +/-12 pp at ~60 trades).
  - Treat ID 5 as low-confidence until B3/B4 shows fold-stable results; IDs 2 and 3 have reasonable sample sizes.
- `Why not simply add the excluded bars as 0s:` an FRVP label means "this setup hit target before stop"; a non-setup bar has no entry, stop or target, so 0 there means "no setup", not "bad trade". It teaches setup recognition (which the rules already do exactly) and buries the ~0.4% positives. Adjacent 5-minute bars are nearly identical with overlapping outcomes, so the effective sample is far below the row count.
- `What (two experiments):`
  - E-A hard negatives: add near-miss bars (an FRVP setup partly formed but did not trigger, e.g. a value-area test without acceptance) labelled with the same entry/stop/target geometry as if entered. Needs a per-setup near-miss definition and the usual causal-prefix and leakage audits.
  - E-B every-bar model: a fixed triple-barrier label from every bar (family horizon, ATR-scaled target/stop), setup flags as features, uniqueness weights / sequential bootstrap and purged, embargoed walk-forward CV. This is the design the EURUSD OTE models already use (180k-540k rows).
- `How to compare:` the same walk-forward windows, costs and exit contract as the meta model; score on the meta model's setup opportunities (post-cost expectancy per trade, AP lift over base rate, concentration, time stability), and separately report any off-setup trades E-B adds and their own post-cost expectancy. Promotion still follows the existing contract.
- `Decision:` none yet; the R3 retrain keeps the event-conditioned (meta-label) design unchanged. Cross-reference: `docs/ICT_execution_journal.md#ict_experiment_journal` (same proposal for ICT), `notes.txt` backlog.

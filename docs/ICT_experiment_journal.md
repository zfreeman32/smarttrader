# ICT Experiment Journal

## E1. Short Setup-Family Split-First Audit Experiment (2026-08-31)

- `Question:` does the pooled short ICT meta branch dilute or overfit family-specific behavior, and can setup-specific short ICT models beat the pooled short meta/reversal/continuation controls after the 2026-08-11 overlap audit found heavy exact-trade sharing?
- `Control:` July 18 strict ICT pooled controls from `model_testing/reports/ict_backtests/ict_es_primary_20260718T002437Z`: `ict_short_meta_xgb_v1`, `ict_short_reversal_xgb_v1`, and `ict_short_continuation_xgb_v1`. The audit premise was that `ict_short_meta_xgb_v1` shared `284` exact walk-forward trades with `ict_short_reversal_xgb_v1` and `84` with `ict_short_continuation_xgb_v1`, echoing the FRVP pooled-vs-family dilution pattern.
- `Template:` FRVP E13-E15 split-first workflow: add setup/family-rich targets without replacing pooled controls, preserve pooled-family weighting/leakage controls where possible, materialize a separate prepared root, train only sufficiently broad setup lanes, evaluate with the existing regime/threshold/backtest stack, and judge on post-cost walk-forward economics instead of classifier metrics alone.

### Plan

1. Add additive ICT setup-target plumbing rather than changing the existing six pooled ICT target columns.
2. Materialize a fresh short-only setup prepared root from the saved leakage-safe ICT root `artifacts/ict_es_primary_refresh_20260724_spacing_refit_final_confirm`.
3. Preserve event-window leakage/ICT sequential-bootstrap support for setup-specific target names.
4. Train only setup lanes with sufficient event breadth; log thin setup types as non-trainable in this wave.
5. Build a standalone registry and run strict ICT post-training evaluation under the usual promotion-quality contract.
6. Compare setup-specific economics against pooled short meta, reversal, and continuation controls.
7. Leave live/paper roster unchanged unless a setup-specific branch clears the gate stack and beats its pooled parent.

### Plumbing

- Added `scripts/materialize_ict_setup_prepared_root.py`.
- Extended `ict.reports.leakage_control.build_ict_event_window_frame` to emit setup-specific aliases such as `short_ict_reversal_sweep_reclaim`, while preserving pooled family and meta rows.
- Added a focused test to `tests/test_ict_phase6_pipeline.py`.
- Verification: `ote_venv\Scripts\python.exe -m pytest tests\test_ict_phase6_pipeline.py tests\test_ict_taxonomy.py` passed `5/5`.

### Materialization

- Command target root: `artifacts/ict_short_setup_targets_20260811_audit/phase04_prepared/prepared`.
- Source root: `artifacts/ict_es_primary_refresh_20260724_spacing_refit_final_confirm`.
- Usable short event rows: `7405`; missing signal-row joins: `0`.
- Setup target read:
  - `short_ict_continuation_premium_discount_continuation`: `1237` usable, `556` positive, green.
  - `short_ict_reversal_ifvg_reversal`: `2935` usable, `960` positive, green.
  - `short_ict_reversal_sweep_reclaim`: `3188` usable, `836` positive, green.
  - `short_ict_continuation_displacement_continuation_after_raid`: `40` usable, `20` positive, too thin for wave 1.
  - `short_ict_reversal_ob_retest_after_mss`: `2` usable, `1` positive, non-trainable.
  - `short_ict_reversal_session_open_manipulation_post_ib`: `1` usable, `0` positive, non-trainable.
  - `short_ict_reversal_session_open_manipulation_pre_ib`: `2` usable, `0` positive, non-trainable.

### Training

- Model root: `models/ict_short_setup_family_xgb_20260811_audit`.
- Registry: `models/ict_short_setup_family_model_registry_20260811_audit.json`.
- Trainable targets:
  - `ict_short_continuation_premium_discount_continuation_xgb_v1`
  - `ict_short_reversal_ifvg_reversal_xgb_v1`
  - `ict_short_reversal_sweep_reclaim_xgb_v1`
- `premium_discount_continuation` and `ifvg_reversal` used the standard ICT setup-lane geometry with `16` XGBoost trials and ICT sequential bootstrap.
- `sweep_reclaim` required reduced setup-lane geometry because the standard `min_train_positive_rows=115` pruned every trial; rerun used `cv_initial_train_rows=320`, `cv_val_rows=96`, `min_train_positive_rows=80`, `min_val_positive_rows=12`, and `min_val_true_events=8`.
- Classifier read:
  - `premium_discount_continuation`: study best `0.628`, CV AP `0.525`, test AP `0.507`, ROC AUC `0.555`, event F0.5 `0.868`.
  - `ifvg_reversal`: study best `0.548`, CV AP `0.406`, test AP `0.438`, ROC AUC `0.602`, event F0.5 `0.759`.
  - `sweep_reclaim`: study best `0.550`, CV AP `0.429`, test AP `0.407`, ROC AUC `0.715`, event F0.5 `0.712`.

### Strict Evaluation

- Report roots:
  - `model_testing/reports/ict_regime_slices/ict_short_setup_family_20260811_audit`
  - `model_testing/reports/ict_threshold_policies/ict_short_setup_family_20260811_audit`
  - `model_testing/reports/ict_backtests/ict_short_setup_family_20260811_audit`
- Contract: promotion-quality, `min_trades_per_week=3.0`, requested min folds `7`, available min folds `13`, no auto-relaxed folds.
- Threshold selection found no qualified non-global policy for any setup-specific model.
- Walk-forward economics:
  - `premium_discount_continuation`: `70` trades, `-324.50` ticks, expectancy `-4.64`, PF `0.716`, Sharpe `-0.393`, DSR `-0.388`, WFE `-0.900`, max DD `5.29%`, gate `False`.
  - `ifvg_reversal`: `298` trades, `-65.70` ticks, expectancy `-0.22`, PF `0.982`, Sharpe `-0.036`, DSR `-0.036`, WFE `0.376`, max DD `6.60%`, gate `False`.
  - `sweep_reclaim`: `755` trades, `+3049.25` ticks, expectancy `+4.04`, PF `1.377`, Sharpe `0.750`, DSR `0.711`, WFE `1.213`, max DD `5.84%`, gate `False`.

### Control Comparison

- `ict_short_meta_xgb_v1`: `275` trades, `+3191.25` ticks, expectancy `+11.60`, PF `2.591`, Sharpe `1.664`, DSR `1.051`, WFE `1.932`, max DD `2.00%`, gate `False` only because largest-single-trade concentration failed.
- `ict_short_reversal_xgb_v1`: `791` trades, `+5475.85` ticks, expectancy `+6.92`, PF `1.795`, Sharpe `1.048`, DSR `0.915`, WFE `1.306`, max DD `3.44%`, gate `True`.
- `ict_short_continuation_xgb_v1`: `73` trades, `-312.45` ticks, expectancy `-4.28`, PF `0.731`, Sharpe `-0.567`, DSR `-0.547`, WFE `1.210`, max DD `4.58%`, gate `False`.

### Observations

- The dilution concern is real diagnostically: the short meta branch overlaps heavily with short reversal, and setup splitting shows most usable short reversal volume is concentrated in `ifvg_reversal` and `sweep_reclaim`.
- Setup-specific training did not beat the accepted pooled short reversal parent. `sweep_reclaim` is the only economically positive setup model, but it still trails pooled `ict_short_reversal_xgb_v1` on net PnL, Sharpe, DSR, WFE, drawdown, and the acceptance gate.
- `sweep_reclaim` is worth preserving as a research read because it posts `+3049.25` ticks with positive WFE and low account drawdown, but it fails Sharpe, profitable-quarter share, and largest-single-trade concentration.
- `ifvg_reversal` has classifier signal but no realized edge after costs.
- `premium_discount_continuation` confirms the pooled short continuation weakness rather than rescuing it.
- The tiny setup types are not model candidates under the current event design. They should stay as event-collection diagnostics unless the detector is broadened.

### Decision

- Do not promote any ICT setup-specific short model.
- Keep `ict_short_reversal_xgb_v1` as the accepted short-side family control.
- Keep `ict_short_meta_xgb_v1` out of accepted deployment unless the existing concentration problem is solved separately.
- Keep `ict_short_continuation_xgb_v1` and the `premium_discount_continuation` setup lane out of promotion work for now.
- Next useful pass, if budget is allocated, is policy-only research on `ict_short_reversal_sweep_reclaim_xgb_v1` around 2023-2026 deterioration and session filtering, not another blind retrain.

## E2. Short Sweep-Reclaim Policy-Only Deterioration / Session Pass (2026-09-04)

- `Question:` can the positive but non-promotable `ict_short_reversal_sweep_reclaim_xgb_v1` branch be made operationally credible with policy-only session filters, especially after the visible 2023-2026 deterioration?
- `Scope:` policy-only research. No model retraining, no canonical label change, no live/paper registry write-back, and no acceptance change.
- `Model / registry:` `ict_short_reversal_sweep_reclaim_xgb_v1` from `models/ict_short_setup_family_model_registry_20260811_audit.json`.
- `Source reports:` reused the setup-family regime labels and strict walk-forward baseline under `model_testing/reports/ict_regime_slices/ict_short_setup_family_20260811_audit` and `model_testing/reports/ict_backtests/ict_short_setup_family_20260811_audit`.

### Plan

1. Reuse the frozen setup-specific model and existing regime-labeled predictions.
2. Diagnose the prior selected walk-forward trades by calendar year, session, and year/session.
3. Run only abstain overlays: no-abstain baseline, cooldown-only, high-stress/off-hours filters, keep-Asia/New-York filters, New-York-only, and Asia-only.
4. Run each overlay twice: full walk-forward span (`2020-2026`) and deterioration span starting with scheduled test folds on `2023-01-01`.
5. Evaluate with ES session-schedule costs, the same global threshold (`0.29`), `16` DSR trials, `7` minimum folds, and research-mode gate metadata.
6. Keep any result as diagnostic unless it survives post-cost expectancy, Sharpe/DSR, WFE, quarter breadth, composite breadth, concentration, and practical cadence review.

### Execution

- Added `scripts/run_ict_sweep_reclaim_policy_research_20260904.py`.
- Command: `ote_venv\Scripts\python.exe scripts\run_ict_sweep_reclaim_policy_research_20260904.py`.
- Output root: `model_testing/reports/ict_policy_research/ict_sweep_reclaim_20260904`.
- Main artifacts:
  - `model_testing/reports/ict_policy_research/ict_sweep_reclaim_20260904/RESEARCH_SUMMARY.md`
  - `model_testing/reports/ict_policy_research/ict_sweep_reclaim_20260904/candidate_walk_forward_results.csv`
  - `model_testing/reports/ict_policy_research/ict_sweep_reclaim_20260904/diagnostics/baseline_by_year_session.csv`
  - per-candidate walk-forward outputs under `model_testing/reports/ict_policy_research/ict_sweep_reclaim_20260904/wf`

### Results

- Baseline full-span reproduced the prior strict result: `755` trades, `+3049.25` ticks, expectancy `+4.04`, PF `1.377`, Sharpe `0.750`, DSR `0.711`, WFE `1.213`, max DD `5.84%`, profitable-quarter share `0.538`, largest-trade share `12.83%`; gate remains `False`.
- Calendar-year deterioration is real:
  - `2020`: `101` trades, `+1186.35` ticks, expectancy `+11.75`.
  - `2021`: `91` trades, `+70.85` ticks, expectancy `+0.78`.
  - `2022`: `145` trades, `+2222.75` ticks, expectancy `+15.33`.
  - `2023`: `113` trades, `-81.45` ticks, expectancy `-0.72`.
  - `2024`: `105` trades, `-197.25` ticks, expectancy `-1.88`.
  - `2025`: `144` trades, `+172.40` ticks, expectancy `+1.20`.
  - `2026`: `56` trades, `-324.40` ticks, expectancy `-5.79`.
- Full-span session read:
  - `asia`: `354` trades, `+1601.90` ticks, expectancy `+4.53`.
  - `new_york`: `191` trades, `+1560.85` ticks, expectancy `+8.17`.
  - `off_hours`: `203` trades, `-70.95` ticks, expectancy `-0.35`.
  - `london` and `overlap` were too small to matter materially (`7` trades combined, `-42.55` ticks).
- 2023-2026 session read:
  - `new_york`: `114` trades, `+507.90` ticks, expectancy `+4.46`.
  - `asia`: `187` trades, `-80.55` ticks, expectancy `-0.43`.
  - `off_hours`: `111` trades, `-764.15` ticks, expectancy `-6.88`.
  - `london` plus `overlap`: `6` trades, `-93.90` ticks.
- Best full-span overlay by net ticks was `keep_asia_new_york_no_cooldown`: `545` trades, `+3162.75` ticks, expectancy `+5.80`, PF `1.573`, Sharpe `1.037`, DSR `0.931`, WFE `1.501`, profitable-quarter share `0.654`, positive-composite share `0.667`, max DD `4.31%`, largest-trade share `12.37%`; raw acceptance still `False` because concentration misses the `10%` gate.
- The operational keep-Asia/New-York version with cooldown was close but slightly lower: `534` trades, `+3129.90` ticks, expectancy `+5.86`, Sharpe `1.021`, DSR `0.920`, WFE `1.480`, largest-trade share `12.50%`; still fails concentration.
- The 2023+ baseline was poor: `418` trades, `-430.70` ticks, expectancy `-1.03`, PF `0.915`, Sharpe `-0.299`, DSR `-0.296`, WFE `-0.252`, profitable-quarter share `0.429`, positive-composite share `0.333`.
- Best 2023+ overlay was `new_york_only_recent_2023`: `110` trades, `+496.50` ticks, expectancy `+4.51`, PF `1.399`, Sharpe `0.602`, DSR `0.582`, WFE `0.964`, max DD `2.50%`, profitable-quarter share `0.643`, positive-composite share `0.833`; raw acceptance remains `False` because Sharpe is below `0.80` and largest-trade share is `27.26%`.
- Off-hours removal helps the recent window but does not make a promotion candidate: `drop_off_hours_no_cooldown_recent_2023` reached `+333.45` ticks over `307` trades, and `drop_off_hours_recent_2023` reached `+253.70` ticks over `302` trades, but Sharpe stayed below `0.31` and WFE below `0.25`.

### Observations

- The branch's full-span profitability is heavily supported by `2020` and `2022`; once the test window starts in `2023`, the unfiltered policy is net negative.
- The main recent damage is off-hours, not London/overlap. London and overlap are too thin to drive the branch-level result.
- Asia was useful in the full span, but it is not stable in the recent window. Keeping Asia plus New York still improves the recent baseline, but most of the rescue comes from New York.
- New-York-only is the cleanest deterioration-window clue, but it is not deployment-quality: only `110` trades across `14` folds, actual cadence about `0.60` trades/week, Sharpe below gate, and largest-trade concentration far above the `10%` limit.
- Cooldown by itself does not repair the branch; it slightly worsens the 2023+ result (`-510.85` ticks versus `-430.70`).
- High-stress removal is not the answer here. It worsened recent performance when applied alone and diluted the off-hours filter when both were used.

### Decision

- Keep `ict_short_reversal_sweep_reclaim_xgb_v1` research-only.
- Do not write a policy back to the setup-family registry and do not add this branch to live/paper accepted slots.
- Treat `off_hours` exclusion and New-York-only routing as forward-shadow hypotheses, not promotion policies.
- Do not spend retraining budget on this branch before the broader ICT setup-logic audit and explicit classic-breaker Phase 3 label-design branch are complete.

## E3. Classic-Breaker Phase 3 Label-Design / Expansion Pass (2026-09-05)

- `Question:` can the isolated ICT classic-breaker research lane produce a causally clean, trainable setup target without leaking into canonical ICT family/meta labels?
- `Scope:` Phase 3 label-design refresh, Phase 3 gate audit, short-side Phase 4 materialization, and prepared-root leakage audit. No model training and no live/paper registry change.
- `Baseline artifact:` `artifacts/ict_es_classic_breaker_research_20260904/phase03_labeling`.
- `Expanded artifact:` `artifacts/ict_es_classic_breaker_research_20260904_expanded_firstretest_noreject_v1/phase03_labeling`.
- `Expanded short prepared root:` `artifacts/ict_es_classic_breaker_research_20260904_expanded_firstretest_noreject_v1_phase4_short/phase04_prepared/prepared`.

### Plan

1. Keep classic breaker isolated as research label family `ict_classic_breaker`.
2. Audit label causality, breaker lineage, duplicate firing, canonical isolation, and prepared-feature leakage.
3. Materialize only the short target first because the baseline long side had only `1` positive.
4. Loosen detector settings only in a new artifact, then gate the expanded sample before considering training.
5. Stop before model training unless the expanded sample becomes large enough for purged CV geometry.

### Execution

- Added classic-breaker support to the setup-target materializer so canonical feature roots can be paired with a separate classic-breaker Phase 3 source directory.
- Added `--source-phase03-dir` and `--artifact-base-dir` to `scripts/materialize_ict_setup_prepared_root.py`.
- Preserved target naming as `short_ict_classic_breaker`, avoiding the bad duplicate form `short_ict_classic_breaker_classic_breaker`.
- Tightened the classic-breaker prepared-feature audit so it scans actual feature names from `features.json` and `feature_importance.csv`, not benign report metadata fields such as `target_column`.
- Exposed classic-breaker detector knobs in `scripts/run_ict_classic_breaker_phase3_refresh.py`:
  - `--classic-breaker-max-age`
  - `--classic-breaker-allow-later-retests`
  - `--classic-breaker-allow-non-rejection-close`
- Verification:
  - Focused suite after materializer/audit patches: `59 passed`.
  - Focused suite after refresh knob tests: `40 passed`.
  - `git diff --check` passed.

### Baseline Results

- Phase 3 gate report: `model_testing/reports/ict_classic_breaker_phase3_audits/ict_es_classic_breaker_research_20260904_phase3_gate`.
- Status: `PASS WITH CAVEATS`.
- Events: `33` total, `31` usable, `6` positives, base rate `19.35%`.
- Direction split:
  - long: `13` usable, `1` positive, base rate `7.69%`.
  - short: `18` usable, `5` positives, base rate `27.78%`.
- Integrity checks passed: breaker/source lineage present, source order block before activation, activation before retest, signal equals retest index, valid zone bounds, first-retest-only, displacement not after signal, and entry/stop/target geometry.
- Canonical isolation passed: classic labels were present, canonical positives stayed `0` in the classic-only artifact, and classic events did not leak into canonical family/meta labels.
- Baseline short Phase 4 materialization:
  - root: `artifacts/ict_es_classic_breaker_research_20260904_phase4_short/phase04_prepared/prepared`.
  - target: `short_ict_classic_breaker`.
  - usable rows: `18`; positives: `5`; train/val/test: `12/3/3`; selected features: `346`.
  - leakage report: `prepared_targets=1`, `failures=0`, matched `18/18`.

### Expansion Results

- Broadest loose run:
  - settings: `max_age=360`, later retests allowed, rejection close not required.
  - artifact: `artifacts/ict_es_classic_breaker_research_20260904_expanded_loose_v1`.
  - produced `59` events and `55` usable rows, but failed the Phase 3 gate.
  - failures: `first_retest_only_failed`; duplicates: `14` same-breaker-side events.
  - decision: reject this variant because later retests break the label-design contract.
- Gate-passing expanded run:
  - settings: `max_age=360`, first-retest-only preserved, rejection close not required.
  - artifact: `artifacts/ict_es_classic_breaker_research_20260904_expanded_firstretest_noreject_v1`.
  - Phase 3 status: `PASS WITH CAVEATS`.
  - Events: `45` total, `42` usable, `9` positives, base rate `21.43%`.
  - Direction split:
    - long: `19` usable.
    - short: `23` usable, `7` positives, base rate `30.43%`.
  - No Phase 3 failures; caveats remain `sparse_long_sample` and `sparse_short_sample`.
  - Duplicate same-breaker-side events: `0`; first-retest-only pct: `100%`.
- Expanded short Phase 4 materialization:
  - root: `artifacts/ict_es_classic_breaker_research_20260904_expanded_firstretest_noreject_v1_phase4_short/phase04_prepared/prepared`.
  - target: `short_ict_classic_breaker`.
  - usable rows: `23`; positives: `7`; train/val/test: `16/3/4`.
  - selected features: `373`; readiness: `green`, score `100.0`.
  - prepared-root leakage gate: `prepared_targets=1`, `failures=0`, matched `23/23`.
  - final pretraining classic audit inspected `746` prepared feature references and found `0` suspicious features.

### Observations

- Classic breaker is causally clean when first-retest-only is preserved.
- Allowing later retests expands count but corrupts the event contract by introducing duplicate same-breaker-side events.
- Dropping the rejection-close requirement is a safer expansion knob than allowing later retests, but it only expands short usable rows from `18` to `23`.
- The label design is not the blocking issue now; event frequency is. The current expanded short sample is still too small for the trainer's purged CV/final-fit geometry.
- A direct training attempt against the placeholder `EXPANDED` path failed because that path did not exist. The real expanded root exists, but even it is still too small for meaningful model training.

### Decision

- Keep the gate-passing expanded classic-breaker artifact as a research/plumbing artifact only.
- Do not train or add `short_ict_classic_breaker` to the live app yet.
- Reject the later-retest expansion variant.
- Preserve first-retest-only as a classic-breaker contract invariant unless a future branch explicitly redefines the setup.
- Next useful ICT work should follow `notes.txt`: audit setup logic before spending training budget on long setup-based models or live integration.

# ICT Execution Journal

## Ordered Plan

Execution order for the remaining ICT work as of `2026-07-21`:

1. Finish the ICT labeler paper contract.
2. Run tighter ICT event-sample audits for `htf_context`, `stop_reference`, `target_reference`, and per-family barrier geometry.
3. Refit ICT setup spacing and cooldowns from realized time-to-first-barrier and event-clustering diagnostics.
4. Finish leakage-control integration: sequential bootstrap plus explicit ICT fold-geometry / embargo validation.

Why this order is the most efficient:

- The event-sample audit is only worth doing after the label contract is stable enough to trust the sampled events.
- The spacing/cooldown refit depends on realized barrier timing and overlap patterns from the upgraded labeler.
- The leakage-control work depends on the final horizon and spacing geometry, because embargo width and overlap treatment should be pinned to the actual label windows, not a superseded heuristic contract.

## Step 1

Date: `2026-07-21`
Item: Finish the full ICT labeler paper contract: vol/session-aware horizons, richer continuation exit logic, and broader exclusions for macro, half-days, lunch, and thin-session windows.

### What We Did

- Upgraded `ict/labeling/ict_labeling_engine.py` to enrich the 5-minute market frame with reusable ES session and macro-calendar context from the existing FRVP infrastructure.
- Replaced the old fixed-bar horizon behavior with volatility-scaled, session-aware horizon sizing.
- Added lunch-clock freezing for vertical-barrier accounting and a lunch-dominated window exclusion.
- Added macro-window, half-day / holiday, and thin-session exclusion rules at the realized event-window level.
- Added richer continuation management: continuation events now treat target touch as activation of a trailing/structure-style exit path instead of forcing an immediate fixed-target exit.
- Added extra event diagnostics fields that will be useful for the next audit step, including `horizon_scale`, `target_hit_index`, `target_hit_time`, and `exit_reason`.
- Expanded `tests/test_ict_labeling_phase4.py` to cover:
  - volatility-scaled horizons
  - continuation trailing exits after target activation
  - macro-event exclusions
  - half-day exclusions
  - lunch-dominated window exclusions

### Why We Did It

- The previous labeler still used fixed integer bar horizons, which the ICT design doc explicitly called out as a v1 simplification.
- Continuation trades were being labeled with fixed target behavior that truncated the run phase the paper contract wants to preserve.
- Macro windows, half-days, lunch, and thin-session windows were still allowed through the label path even though the paper contract already marked them as exclusions or special handling cases.
- This step reduces the chance that later audits and spacing refits are performed on a label sample that is structurally distorted by low-liquidity or incorrectly truncated events.

### Results

- The labeler now derives session and macro flags automatically even when they are not precomputed upstream.
- Reversal and session-open families now use volatility-scaled horizons instead of raw fixed bars.
- Continuation families now support target activation plus trailing continuation management, so the exit economics are no longer forced into an overly short fixed-target contract.
- Broader exclusion counts are now emitted through the existing diagnostics path via reasons such as:
  - `macro_event_window`
  - `half_day_or_holiday_window`
  - `lunch_dominated_window`
  - `thin_session_window`
- Verification passed with:

```text
pytest tests/test_ict_labeling_phase4.py tests/test_ict_setup_detector_phase3.py tests/test_ict_taxonomy.py tests/test_build_ict_shadow_live_bundle.py
```

- Result: `19 passed`

### Observations

- The new continuation path gives us a cleaner distinction between “target reached” and “final managed exit,” which should make the upcoming event-sample audit more informative.
- The lunch handling is now materially better, but the next audit should explicitly inspect how often lunch freezing versus lunch exclusion is driving the final sample.
- Thin-session logic is intentionally conservative; the next audit should quantify whether it is removing only the intended tails or too much otherwise-usable ICT activity.
- The new `horizon_scale` and `exit_reason` fields should make the next step much faster, because we can inspect event samples without first reverse-engineering why a window was longer or how a continuation exit finalized.

### Next Step

Run the tighter ICT event-sample audits on the upgraded label sample, focusing on `htf_context`, `stop_reference`, `target_reference`, and per-family barrier geometry.

## Step 2

Date: `2026-07-22`
Item: Run tighter ICT event-sample audits for `htf_context`, `stop_reference`, `target_reference`, and per-family barrier geometry.

### What We Did

- Added a reusable ICT audit module in `ict/reports/event_sample_audit.py`.
- Added a runnable wrapper in `scripts/run_ict_event_sample_audit.py` that can:
  - refresh `phase03_labeling` artifacts from the current labeler
  - reuse the saved `phase02_features/ict_es_features.csv` surface to regenerate setup detection more cheaply
  - emit a JSON summary, markdown summary, and manual-review CSV
- Added coverage in `tests/test_ict_event_sample_audit.py` so the audit works against both the newer event schema and the older saved event schema.
- Executed the audit against the current saved ICT phase-3 sample with:

```text
python scripts/run_ict_event_sample_audit.py --skip-refresh --report-id ict_es_primary_20260722_event_audit_from_phase03
```

- Saved the report package at:
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_20260722_event_audit_from_phase03/ict_event_sample_audit_summary.json`
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_20260722_event_audit_from_phase03/ict_event_sample_audit_summary.md`
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_20260722_event_audit_from_phase03/ict_event_sample_audit_review_rows.csv`

### Why We Did It

- The ICT design doc explicitly called for a real event-sample audit before trusting downstream training conclusions.
- We needed a direct read on whether `htf_context`, raw stop/target references, and final barrier geometry were internally consistent by branch and setup type.
- A reusable audit runner is much better than a one-off notebook pass because we will need to rerun this after the upgraded labeler artifacts are regenerated.

### Results

- Verification passed with:

```text
pytest tests/test_ict_event_sample_audit.py tests/test_ict_labeling_phase4.py
```

- Result: `13 passed`
- Audit headline on the current saved phase-3 sample:
  - `15254` total events
  - `15220` usable events
  - `34` excluded events
  - base rate `26.69%`
  - mean label quality `0.7244`
- `htf_context` is not sparse or missing on the saved sample:
  - both continuation branches are `100%` `structure_aligned`
  - reversal branches split roughly `55%` `structure_aligned` / `45%` `structure_mixed`
- Raw reference geometry is mostly directional-correct, but not perfect:
  - raw stop on expected side: `99.57%`
  - raw target on expected side: `99.11%`
  - raw stop+target pair valid together: `98.68%`
  - final stop/target geometry after labeler normalization: `100.00%`
- Barrier adjustments are asymmetric:
  - stop adjusted from raw reference: `11.61%`
  - target adjusted from raw reference: `64.87%`
  - median target adjustment: about `3.43` ticks
  - 90th percentile target adjustment: about `32.30` ticks
- Branch-level barrier geometry on the saved sample:
  - `long ict_continuation`: median RR `1.25`, median stop `32.0` ticks, median target `48.39` ticks, base rate `44.70%`
  - `short ict_continuation`: median RR `1.25`, median stop `20.0` ticks, median target `37.5` ticks, base rate `43.09%`
  - `long ict_reversal`: median RR `3.69`, median stop `5.0` ticks, median target `18.0` ticks, base rate `25.15%`
  - `short ict_reversal`: median RR `4.00`, median stop `4.5` ticks, median target `19.57` ticks, base rate `24.26%`
- Setup-level audit highlighted where the geometry pressure is concentrated:
  - continuation setups have the heaviest target normalization pressure:
    - `premium_discount_continuation`: target adjusted on `89.01%` of long rows and `82.50%` of short rows
    - `displacement_continuation_after_raid`: target adjusted on `100%` of both long and short rows
  - reversal raw-reference failures cluster mostly in `ifvg_reversal` and `sweep_reclaim`, but the failure rate is still low enough that final geometry remains valid after normalization.
- All `34` exclusions in the saved sample are `ambiguous_5m_without_1m`.

### Observations

- The saved July 12, 2026 event sample is clearly pre-upgrade:
  - it has no `horizon_scale`
  - it has no `exit_reason`
  - it records `0` continuation target activations
  - it records `34` unresolved ambiguous bars without 1-minute resolution
- That means the audit already surfaced useful structural issues, but it is still auditing the old event contract rather than the upgraded labeler we finished in Step 1.
- Continuation target references appear to be the main weak point in the saved sample. The labeler is frequently forced to normalize them back to the minimum-RR / fallback-target contract, which suggests the raw continuation target anchor is often too close, wrongly sided, or otherwise not economically representative.
- Reversal geometry looks directionally healthier than continuation geometry: reversals keep tight stops plus much farther targets, which is consistent with the intended paper contract.
- The manual-review rows show several extreme continuation target adjustments and a smaller pocket of wrong-sided raw references in `ifvg_reversal` / `sweep_reclaim`. Those rows should be the first place to inspect when tightening the event logic.
- Two full-history refresh attempts against the upgraded labeler were launched on `2026-07-22`, including a faster path that reuses the saved phase-2 ICT setup surface, but both exceeded the current runtime window and did not rewrite `artifacts/ict_es_primary/phase03_labeling/`. The next rerun should either use a longer wall-clock window or break Phase 3 into a more explicitly checkpointed refresh path.

### Refresh Addendum

Date: `2026-07-23`
Artifact refresh: `artifacts/ict_es_primary_refresh_20260722/phase03_labeling`
Audit package: `model_testing/reports/ict_event_sample_audits/ict_es_primary_refresh_20260722_event_audit/`

#### What We Did

- Reviewed the completed refreshed Phase 3 rerun produced from:
  - `data/futures_data/ES-5m-tagged.csv`
  - `data/futures_data/ES-1m.csv`
  - `artifacts/ict_es_primary/phase02_features/ict_es_features.csv` as the setup-surface input
- Ran the same audit tool against the refreshed artifact tree with:

```text
python scripts/run_ict_event_sample_audit.py --skip-refresh --artifact-run-id ict_es_primary_refresh_20260722 --artifact-base-dir artifacts --report-id ict_es_primary_refresh_20260722_event_audit
```

#### Results

- The refresh completed successfully and materially changed the diagnosis:
  - `15331` total sampled events
  - `28` usable events
  - `15303` excluded events
  - base rate `21.43%`
- The refreshed event schema now reflects the upgraded labeler:
  - `horizon_scale` is populated
  - `exit_reason` is populated
  - ambiguous handling now records `6` `unresolved_intrabar_1m` rows instead of the older `34` `ambiguous_5m_without_1m` rows
- The collapse is overwhelmingly driven by one gate:
  - `events_excluded_thin_session_window = 14776`
  - `events_excluded_half_day_or_holiday_window = 511`
  - `events_excluded_lunch_dominated_window = 8`
  - `events_excluded_macro_event_window = 2`
  - `events_excluded_unresolved_intrabar_1m = 6`
- If `thin_session_window` were not firing, the refreshed sample would still have on the order of `14804` potentially usable events before the other exclusions, so the dataset collapse is not coming from the other new gates.
- Thin-session exclusions cluster exactly where the current rule would be expected to over-fire:
  - most excluded rows occur in the overnight and close-adjacent UTC buckets (`20` through `5` UTC)
  - the excluded setups are primarily:
    - `sweep_reclaim` (`6066`)
    - `ifvg_reversal` (`5705`)
    - `premium_discount_continuation` (`2894`)
- The surviving refreshed sample is too small and distorted to trust for downstream spacing/cooldown refits:
  - usable branches: `25` reversal events and `3` continuation events total
  - continuation target activations remain `0`, which is not a meaningful basis for continuation-exit diagnostics

#### Observations

- This refreshed rerun resolved the earlier uncertainty about whether the sample collapse was due to runtime failure versus true label-contract behavior. It is true label-contract behavior.
- The bottleneck is not raw stop/target geometry. On the refreshed sample, raw reference pair validity is still `98.83%` and final geometry remains `100.00%` valid after normalization.
- The bottleneck is the current `thin_session_window` definition in `ict/labeling/ict_labeling_engine.py`, which excludes any event window with insufficient RTH share or any bar inside the final `60` RTH minutes. On a 24-hour ES market with horizons that often span outside pure RTH, that rule is currently acting more like a near-global veto than a tail-window filter.
- Because of that, the efficient work order changed after the refreshed audit. Spacing/cooldown refits should wait until the thin-session exclusion is narrowed to the intended behavior.

### Next Step

Refine the `thin_session_window` contract first, rerun the refreshed Phase 3 labeling and event-sample audit, and only then move on to the spacing/cooldown refit.

### Thin-Session Refinement Addendum

Date: `2026-07-23`
Branch: `ict-thin-session-refine-20260723`

#### What We Did

- Narrowed `thin_session_window` in `ict/labeling/ict_labeling_engine.py` so it is now anchored to the actual entry bar and preserves the RTH close-tail exclusion without treating low full-window RTH share as a blanket overnight veto.
- Added focused tests in `tests/test_ict_labeling_phase4.py` covering:
  - overnight event allowance under the refined gate
  - preserved exclusion for late-RTH close-tail entries
- Validation passed with:

```text
pytest tests/test_ict_labeling_phase4.py tests/test_ict_event_sample_audit.py
```

- Result: `15 passed`
- A full refined Phase 3 refresh plus audit was then completed on `2026-07-23` for:
  - `artifacts/ict_es_primary_refresh_20260723_thin_session_full/phase03_labeling/`
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_refresh_20260723_thin_session_full_event_audit/`
- The completed full rerun matched the earlier event-level thin-session recast exactly on the headline counts and branch-level diagnostics, so the recovered event contract is now confirmed both in `ict_es_events.csv` and in the rebuilt `ict_es_labels.csv`.
- Saved the refined-gate recast audit package at:
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_refresh_20260722_thin_session_recast/ict_event_sample_audit_summary.json`
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_refresh_20260722_thin_session_recast/ict_event_sample_audit_summary.md`
  - `model_testing/reports/ict_event_sample_audits/ict_es_primary_refresh_20260722_thin_session_recast/ict_event_sample_audit_review_rows.csv`

#### Results

- Removing the overnight blanket veto restored the sample immediately:
  - `15331` total events
  - `14804` usable events
  - `527` excluded events
  - `5026` positive events
  - base rate `33.95%`
- Remaining exclusions are now narrow and plausible:
  - `511` half-day / holiday windows
  - `8` lunch-dominated windows
  - `6` unresolved 1-minute intrabar windows
  - `2` macro windows
  - `0` thin-session exclusions in the recast contract
- Branch-level geometry is back in a usable range:
  - `long ict_continuation`: `1670` usable, base rate `51.14%`, median RR `1.25`
  - `short ict_continuation`: `1331` usable, base rate `45.23%`, median RR `1.25`
  - `long ict_reversal`: `5725` usable, base rate `31.28%`, median RR `3.13`
  - `short ict_reversal`: `6078` usable, base rate `29.27%`, median RR `3.28`
- Continuation management is now visible again on the refreshed sample:
  - `3001` usable continuation events
  - `285` continuation target activations
  - activation share `9.50%`
  - median extension after target `1.0` bar
  - exit mix:
    - `2115` `timeout_pnl_sign`
    - `601` `stop_touch`
    - `285` `continuation_trailing_stop`

#### Observations

- The refined thin-session contract resolves the main blocker from the prior refreshed audit. The event sample is no longer collapsing under the overnight veto, and the remaining exclusion counts are in a believable range.
- This also confirms that the earlier collapse was not a structural problem with the upgraded labeler itself. It was almost entirely a thin-session policy problem.
- The continuation family is now back in scope for real diagnostics, and the sample again supports the next research steps.
- One important issue remains active after the thin-session fix: continuation target normalization pressure is still high. The refined-gate recast still shows target adjustment on `68.93%` of events overall, with the heaviest concentration in continuation setups.
- The longer offline rerun is now complete, so the refined thin-session contract is no longer just an event-level recast conclusion. It is the confirmed full Phase 3 artifact state.

### Next Step

Move to the spacing/cooldown refit. The remaining high-priority label-contract issue after session gating is continuation target-anchor pressure, but the immediate next work item should be spacing/cooldown diagnostics because the refreshed artifact set is now stable enough to support that refit.

## Step 3

Date: `2026-07-23`
Item: Refit ICT setup spacing and cooldowns from realized time-to-first-barrier and event-clustering diagnostics.

### What We Did

- Added a reusable spacing/cooldown diagnostics module in `ict/reports/spacing_cooldown_diagnostics.py`.
- Added a runnable wrapper in `scripts/run_ict_spacing_cooldown_diagnostics.py` so the spacing refit can be rerun directly from saved `phase03_labeling` artifacts.
- Added focused coverage in `tests/test_ict_spacing_cooldown_diagnostics.py` for:
  - continuation setups that should widen spacing from realized barrier timing
  - structural session-open setups that should keep their once-per-session spacing
  - event frames that do not carry an explicit `excluded` column
- Ran the spacing diagnostics against the confirmed full refined Phase 3 artifact set:

```text
python scripts/run_ict_spacing_cooldown_diagnostics.py --artifact-run-id ict_es_primary_refresh_20260723_thin_session_full --report-id ict_es_primary_refresh_20260723_thin_session_full_spacing_refit_baseline
```

- Refit `SETUP_MIN_SPACING` in `ict/setups/detector.py` from the realized July 23 baseline:
  - `premium_discount_continuation`: `6 -> 12` on the first pass, then refined to `10` after the validated July 24 sweep
  - `displacement_continuation_after_raid`: `8 -> 16`
- Added a fail-fast guardrail in `scripts/run_ict_event_sample_audit.py` so an explicit missing `--setup-feature-csv` now raises instead of silently falling back to raw 5-minute setup detection.
- Added `scripts/run_ict_spacing_refit_sweep.py` to sweep premium-continuation spacing against a fixed saved Phase 02 setup surface.

### Why We Did It

- The ICT design doc already called out cooldown and horizon as coupled, but the detector was still using guessed constants rather than the realized barrier-timing distribution from the upgraded labeler.
- The thin-session refinement unblocked a trustworthy event sample, so this was the first point where the spacing refit could be done on a stable contract instead of on a distorted label set.
- A reusable runner is important because the spacing refit now needs an explicit before/after validation loop, not another one-off notebook calculation.

### Results

- Validation passed with:

```text
pytest tests/test_ict_spacing_cooldown_diagnostics.py tests/test_ict_event_sample_audit.py
```

- Result: `5 passed`
- Baseline spacing report package saved at:
  - `model_testing/reports/ict_spacing_cooldown_diagnostics/ict_es_primary_refresh_20260723_thin_session_full_spacing_refit_baseline/ict_spacing_cooldown_summary.json`
  - `model_testing/reports/ict_spacing_cooldown_diagnostics/ict_es_primary_refresh_20260723_thin_session_full_spacing_refit_baseline/ict_spacing_cooldown_summary.md`
  - `model_testing/reports/ict_spacing_cooldown_diagnostics/ict_es_primary_refresh_20260723_thin_session_full_spacing_refit_baseline/ict_spacing_cooldown_setup_rows.csv`
- Only two setup types warranted a spacing change on the July 23 refined artifact set:
  - `premium_discount_continuation`: current spacing `6`, recommended `12`, `p60_first_barrier=18.00`, `p25_same_setup_side_gap=12.00`, overlap `29.01%`
  - `displacement_continuation_after_raid`: current spacing `8`, recommended `16`, `p60_first_barrier=16.00`, overlap `2.86%`
- The reversal families and session-open patterns stayed on their existing spacing:
  - `ifvg_reversal` held at `8`
  - `sweep_reclaim` held at `6`
  - session-open setups stayed at `60`
- The first post-refit confirmation rerun on July 23, 2026 (`ict_es_primary_refresh_20260723_spacing_refit_full`) is not a valid comparison point and should not be used for final spacing decisions:
  - the command pointed at `artifacts\ict_es_primary_refresh_20260723_thin_session_full\phase02_features\ict_es_features.csv`, which does not exist
  - the refresh path therefore fell back to rebuilding setups from raw `ES-5m-tagged.csv`
  - the resulting `htf_context` taxonomy collapsed into `structure_aligned` / `structure_mixed`, proving the run was not apples-to-apples with the confirmed July 23 baseline
- The validated follow-up rerun on July 23, 2026 (`ict_es_primary_refresh_20260723_spacing_refit_full_v2`) reused the same saved Phase 02 setup surface as the confirmed thin-session baseline via `artifacts\ict_es_primary\phase02_features\ict_es_features.csv`:
  - `15322` total events
  - `14796` usable events
  - `4979` positive events
  - base rate `33.65%`
  - `2818` usable continuation events
  - `265` continuation target activations
  - `68.46%` target-adjusted share
  - `htf_aligned_long` / `htf_aligned_short` and `htf_mixed` context taxonomy restored
- The July 24, 2026 premium-spacing sweep held `displacement_continuation_after_raid` at `16` and tested `premium_discount_continuation` at `8`, `10`, and `12` against the validated Phase 02 setup surface:
  - `premium=8`: `14799` usable events, base rate `33.79%`, `2924` continuation events, premium overlap `26.61%`
  - `premium=10`: `14796` usable events, base rate `33.75%`, `2869` continuation events, premium overlap `24.79%`
  - `premium=12`: `14796` usable events, base rate `33.65%`, `2818` continuation events, premium overlap `22.87%`
- Current detector baseline after the validated sweep:
  - `premium_discount_continuation = 10`
  - `displacement_continuation_after_raid = 16`
- The final confirmation rerun on `2026-07-24` (`ict_es_primary_refresh_20260724_spacing_refit_final_confirm`) matched the validated `premium=10 / displacement=16` sweep row exactly on the key headline metrics:
  - `15322` total events
  - `14796` usable events
  - `4994` positive events
  - base rate `33.75%`
  - `2869` usable continuation events
  - `268` continuation target activations
  - target-adjusted share `68.56%`
  - premium overlap `24.79%`
- Relative to the confirmed thin-session baseline (`ict_es_primary_refresh_20260723_thin_session_full`), the final spacing baseline is a small and acceptable contraction rather than a structural regime change:
  - usable events `14804 -> 14796`
  - positive events `5026 -> 4994`
  - overall base rate `33.95% -> 33.75%`
  - continuation events `3001 -> 2869`
  - continuation base rate `48.52% -> 48.41%`
- Experiment cleanup completed on `2026-07-24`:
  - active artifact roots now keep the confirmed thin-session baseline, the final spacing confirmation run, and the Phase 02 setup surface
  - active reports now keep the thin-session baseline audit, the final confirmation audit/follow-up, and the high-level sweep summary
  - invalid or superseded spacing-refit artifacts were archived under:
    - `artifacts/_archive/ict_spacing_refit_20260724/`
    - `model_testing/reports/ict_event_sample_audits/_archive/ict_spacing_refit_20260724/`
    - `model_testing/reports/ict_spacing_cooldown_diagnostics/_archive/ict_spacing_refit_20260724/`

### Observations

- `premium_discount_continuation` is the only family showing materially heavy same-setup-side overlap in the baseline report, so widening it from `6` to `12` is the highest-signal part of the refit.
- `displacement_continuation_after_raid` is not overlap-heavy, but its realized time-to-first-barrier is much slower than the old `8`-bar heuristic, so moving it to `16` brings the detector spacing in line with the actual holding geometry.
- The later validated sweep showed that spacing alone will only reduce, not eliminate, premium-continuation overlap. Even at `12`, the premium overlap rate remained `22.87%`, so leakage-control work still matters after the spacing refit.
- `premium=10` is the best current compromise. It improves premium overlap materially versus the July 23 baseline without paying the larger continuation-count penalty of `12`.
- The spacing diagnostics are now a reusable regression check. The final confirmation rerun already matched the validated `10 / 16` sweep row, so spacing is no longer the main blocker.
- The major unresolved issue after the spacing refit is no longer event density. It is residual continuation overlap plus the still-high continuation target-adjustment share, both of which now point more naturally to leakage-control and fold-geometry work than to another immediate spacing rerun.

### Next Step

Move to leakage-control integration next. The spacing baseline is now confirmed on a validated saved Phase 02 setup surface, and another immediate rerun is unlikely to change the main remaining risks. The next efficient step is to implement sequential bootstrap plus explicit ICT fold-geometry / embargo validation using the confirmed `10 / 16` spacing contract.

## Step 4

Date: `2026-07-25`
Item: Finish leakage-control integration: sequential bootstrap plus explicit ICT fold-geometry / embargo validation.

### What We Did

- Added `ict/reports/leakage_control.py` with reusable ICT leakage-control utilities:
  - realized event-window normalization by target
  - per-target embargo resolution from realized windows plus `swing_confirm_bars`
  - average-uniqueness and sequential-bootstrap diagnostics
  - prepared split boundary auditing against `source_row_idx`
- Added `scripts/run_ict_leakage_control_audit.py` so the contract can be rerun directly against any ICT Phase 3 / Phase 6 artifact pair.
- Extended the shared preprocessing split builder so prepared datasets can reserve target-specific source-row embargo gaps via `target_split_embargo_bars`.
- Patched `ict/pipelines/es_primary_phase06.py` to infer the ICT leakage contract automatically from `phase03_labeling/ict_es_events.csv` plus the saved Phase 02 metadata and then pass those embargo widths into the prepared-root build.
- Added regression coverage in:
  - `tests/test_ict_leakage_control.py`
  - `tests/test_ict_phase6_pipeline.py`
  - `tests/test_features_preprocessing.py`
- Rebuilt the confirmed spacing-baseline artifact `ict_es_primary_refresh_20260724_spacing_refit_final_confirm` so the refreshed Phase 6 prepared root uses the new ICT embargo contract.

### Why We Did It

- The first real leakage audit on the confirmed July 24, 2026 spacing baseline showed that the old prepared-root geometry still used simple fractional chronological splits on usable rows. That preserved time order, but it did not reserve the ICT embargo gap in source-bar space.
- The resulting failure mode was broader than literal overlap. Only one boundary showed actual event-window overlap, but five of six targets still violated the required post-window embargo distance, which means the fold geometry was too optimistic even where no direct collision was visible.
- Spacing was already validated. At that point the remaining leakage question had to be answered in the split builder itself, not by another setup-spacing rerun.

### Results

- The first real audit on `ict_es_primary_refresh_20260724_spacing_refit_final_confirm` failed `5 / 6` prepared targets with an overall recommended embargo of `49` bars.
- The resolved target-specific contract from the saved Phase 02 metadata plus the confirmed Phase 3 events was:
  - `long_ict_reversal = 36`
  - `short_ict_reversal = 36`
  - `long_ict_continuation = 49`
  - `short_ict_continuation = 47`
  - `long_ict_meta = 49`
  - `short_ict_meta = 47`
- After rebuilding Phase 6 on `2026-07-25`, the follow-up audit at `model_testing/reports/ict_leakage_control_audits/ict_es_primary_refresh_20260724_spacing_refit_final_confirm_leakage_audit_after_embargo/` passed all `6 / 6` targets.
- The confirmed post-patch leakage audit headline was:
  - max realized window `46` bars
  - overall recommended embargo `49` bars
  - boundary failures `0`
  - overlap events `0` at every audited train/val and val/test boundary
- The patch fixed the leakage contract without materially damaging row counts. For example, `long_ict_meta` still kept `7391` usable rows overall and only dropped `1` usable row at the train/val boundary plus `2` usable rows at the val/test boundary to satisfy the embargo.
- Sequential-bootstrap diagnostics are now reproducible from the same audit runner and showed a real uniqueness lift on continuation labels versus the same-size chronological prefix:
  - `long_ict_continuation`: `0.86 -> 0.97` average uniqueness (`+0.11`)
  - `short_ict_continuation`: `0.85 -> 0.97` average uniqueness (`+0.13`)

### Observations

- The main leakage problem was usually not obvious event overlap. It was under-sized fold spacing after the last train or validation event window. Auditing only for direct overlap would have missed most of the real contract failures.
- ICT needs target-specific fold geometry. Reversal windows topped out at `33` realized bars, while continuation and meta targets needed `44-46` realized bars, so a single small fixed split gap would still have been wrong.
- The new split geometry is cheap in practice. The prepared-root rebuild cleared every boundary while sacrificing only a few usable rows at the split edges instead of forcing a broad retrim of the dataset.
- Sequential bootstrap is now implemented as an ICT utility and diagnostic, but it is not yet consumed directly inside an ICT trainer because the repo still does not have a dedicated ICT training wrapper analogous to the older FRVP stack.

### Decision

- Treat explicit ICT fold geometry / embargo validation as integrated and verified on the confirmed July 24, 2026 spacing baseline.
- Keep one leakage-control subtask open: wire the new sequential-bootstrap sampler into the eventual ICT training / CV path so overlap handling is applied during model fitting, not only in diagnostics and prepared-root auditing.
- The next efficient step is no longer another spacing rerun. It is either:
  - build the thin ICT training wrapper that consumes this leakage-safe prepared root, or
  - run the next ICT model-training / economics refresh against this prepared artifact if an existing trainer entry point is already sufficient.

## Step 5

Date: `2026-07-26`
Item: Wire sequential bootstrap into the actual ICT training / CV path and validate the leakage-safe retrain flow.

### What We Did

- Patched `model_training/ote_training/ote_xgboost_pipeline.py` so ICT training now consumes the event-window contract directly:
  - load target-specific `signal_index -> barrier_end_index` windows from `phase03_labeling/ict_es_events.csv`
  - apply sequential bootstrap in each fold after hard-negative weighting and any balanced-tuning subsample
  - reuse the same bootstrap path in the final pre-eval fit and final refit-on-dev stage
  - carry bootstrap diagnostics into `cv_fold_manifest`, `training_summary.json`, and `model_config.json`
  - upgrade ICT CV purge spacing from the old generic horizon heuristic to the realized event-window embargo contract derived from Phase 3 plus the saved Phase 02 metadata
- Added focused regression coverage in `tests/test_ote_xgboost_pipeline.py` for:
  - ICT event-window lookup resolution from the artifact layout
  - ICT-aware purge-bar resolution
  - in-fold sequential-bootstrap diagnostics
  - existing source-row-index artifact writing
- Added a one-shot runner at `scripts/run_ict_training_refresh.ps1` so the full leakage-safe refresh can be relaunched with the July 26, 2026 contract instead of reconstructing the command by hand.
- Ran focused validation:
  - `python -m py_compile` on the trainer, leakage helper, and touched tests
  - targeted pytest coverage for the new ICT bootstrap / purge / event-window path
- Ran a real leakage-safe smoke retrain on `long_ict_meta`:
  - output root: `models/ict_es_primary_xgb_bootstrap_20260726_smoke/long_ict_meta`
  - prepared root: `artifacts/ict_es_primary_refresh_20260724_spacing_refit_final_confirm/phase04_prepared/prepared`
  - config: `n_trials = 1`, `sequential_bootstrap_mode = ict`
- Rebuilt a registry from that smoke root and ran the post-training economics stack end to end for `ict_long_meta_xgb_v1`:
  - registry: `models/ict_es_primary_model_registry_bootstrap_20260726_smoke.json`
  - regime slices: `model_testing/reports/ict_regime_slices/ict_es_primary_bootstrap_20260726_smoke/`
  - threshold policies: `model_testing/reports/ict_threshold_policies/ict_es_primary_bootstrap_20260726_smoke/`
  - walk-forward backtests: `model_testing/reports/ict_backtests/ict_es_primary_bootstrap_20260726_smoke/`

### Why We Did It

- The July 25, 2026 leakage-control work fixed the prepared-root boundary contract, but the trainer itself still treated overlapping ICT labels like ordinary tabular rows.
- Without this patch, the next retrain would still under-handle overlapping continuation / meta labels inside folds even though the Phase 6 root itself had already become leakage-safe at the train/val/test boundary level.
- The fastest trustworthy validation was not a full six-target rerun first. It was a real one-target smoke on the leakage-safe root that could prove the fold geometry, bootstrap diagnostics, final refit path, registry rebuild, and economics stack all still worked together.

### Results

- The trainer now records the new leakage-safe contract in artifacts. On the July 26, 2026 smoke retrain:
  - `training_summary.json` shows `sequential_bootstrap_mode = ict`, `sequential_bootstrap_final_refit = true`, and `event_window_source = artifacts\ict_es_primary_refresh_20260724_spacing_refit_final_confirm\phase03_labeling\ict_es_events.csv`
  - `cv_fold_manifest.json` shows `purge_bars = 49` for `long_ict_meta`, matching the July 25 confirmed embargo contract rather than the old `34`-bar heuristic
  - sequential bootstrap applied on every recorded fold with meaningful duplication / uniqueness control; for example fold 1 kept `389` sampled rows but only `278` unique rows (`duplicate_share = 0.2853`)
- The smoke `long_ict_meta` retrain completed successfully with:
  - test AP `0.49297`
  - test event F0.5 `0.78842`
  - threshold `0.32`
- The smoke economics refresh also completed successfully for `ict_long_meta_xgb_v1`:
  - selected policy: `regime_threshold_plus_abstain`
  - selected-policy test post-cost expectancy: `9.64` ticks
  - selected-policy trade rate: `3.55` trades per week
  - walk-forward fold count: `27`
  - monthly Sharpe: `1.586`
  - approximate deflated Sharpe: `1.586`
  - max drawdown: `6.40%`
- Focused validation passed:
  - targeted pytest slice: `7 passed`
  - `py_compile` passed on all touched modules

### Observations

- The trainer-side leakage gap is now closed in the places that matter most operationally:
  - fold purge spacing now reflects realized ICT windows
  - overlap handling is no longer only an audit-side statistic; it changes the actual training rows
  - final refit uses the same overlap-aware selection path instead of silently reverting to chronological full-row fitting
- The runtime cost is real. A single-target, single-trial smoke retrain on `long_ict_meta` took roughly eight and a half minutes, which implies the full six-target, ten-trial refresh is a multi-hour job rather than something to expect from a short interactive shell session.
- Because of that runtime, the full July 26, 2026 six-target refresh should be treated as an explicit long-run command using the new runner, not as a quick follow-up after code edits.

### Decision

- Treat the trainer-side leakage-control integration as implemented and validated on a real ICT target as of July 26, 2026.
- Move the next open ICT action from infrastructure to execution:
  - run the full six-target bootstrap retrain from `ict_es_primary_refresh_20260724_spacing_refit_final_confirm`
  - rebuild the full ICT registry from that root
  - review whether `ict_short_meta_xgb_v1` and the continuation families materially improve under the leakage-safe trainer

Date: `2026-07-28`
Item: Audit why the short-side ICT branches still concentrate PnL after the full leakage-safe bootstrap refresh.

### What We Did

- Audited the latest leakage-safe short-side report family:
  - backtests: `model_testing/reports/ict_backtests/ict_es_primary_bootstrap_20260726_full/`
  - threshold policies: `model_testing/reports/ict_threshold_policies/ict_es_primary_bootstrap_20260726_full/`
  - regime slices: `model_testing/reports/ict_regime_slices/ict_es_primary_bootstrap_20260726_full/`
  - model artifacts: `models/ict_es_primary_xgb_bootstrap_20260726_full/`
- Compared the July 27, 2026 leakage-safe outputs against the July 19, 2026 pre-refresh cadence-audit baseline in:
  - `model_testing/reports/ict_backtests/ict_es_primary_20260719_min0/`
  - `model_testing/reports/ict_threshold_policies/ict_es_primary_20260719_min0/`
- Measured trade-level concentration, day-level concentration, regime/session pockets, and exact event overlap across:
  - `ict_short_meta_xgb_v1`
  - `ict_short_reversal_xgb_v1`
  - `ict_short_continuation_xgb_v1`
  - `ict_long_continuation_xgb_v1`
- Cross-checked the backtest read with prepared-target reports and trained-model importance outputs.
- Saved the detailed readout in `model_testing/reports/ict_contract_audits/ICT_SHORT_SIDE_CONCENTRATION_AUDIT_20260728.md`.

### Why We Did It

- The top open ICT queue item after the leakage-safe full refresh was to explain why the short-side branches still concentrated PnL so heavily before spending another cycle on the continuation family.
- The repo needed to separate three possibilities:
  - the short edge broadly collapsed under the leakage-safe trainer
  - the edge survived but moved into different family buckets
  - the edge remained real only inside a small number of overlapping regime pockets

### Results

- The latest short-side miss is no longer only `ict_short_meta_xgb_v1`. Under the July 27, 2026 leakage-safe refresh, both short leaders now fail the promotion-quality gate on the same two items:
  - `ict_short_meta_xgb_v1`: largest-trade share `0.10017`, profitable-quarter share `16 / 27 = 0.5926`
  - `ict_short_reversal_xgb_v1`: largest-trade share `0.11118`, profitable-quarter share `15 / 27 = 0.5556`
- The leakage-safe retrain materially reshuffled the short roster versus July 19, 2026:
  - `ict_short_meta_xgb_v1`: trades `275 -> 1082`, net `+3191.25 -> +3906.7`, positive quarters `20/27 -> 16/27`
  - `ict_short_reversal_xgb_v1`: trades `791 -> 429`, net `+5475.85 -> +3052.15`, positive quarters `20/27 -> 15/27`
- The biggest structural finding is overlap, not just threshold weakness:
  - `ict_short_meta_xgb_v1` and `ict_short_reversal_xgb_v1` share `284` exact walk-forward events when matched on `(entry_datetime, source_row_idx)`
  - those shared events contribute `+2597.4` ticks to each branch
  - that is `66.5%` of short-meta net and `85.1%` of short-reversal net
- The pooled short-meta lane is also absorbing the profitable continuation subset:
  - `ict_short_meta_xgb_v1` and `ict_short_continuation_xgb_v1` share `84` exact events for `+564.4` ticks
  - the continuation unique-only remainder is negative at `81` trades and `-63.65` ticks
- The continuation read improved but did not become promotion-quality:
  - `ict_short_continuation_xgb_v1` moved from `-312.45` to `+500.75` ticks, but still fails on Sharpe, WFE, quarter share, and extreme concentration (`top trade share = 0.7815`)
  - `ict_long_continuation_xgb_v1` remains effectively flat at `+7.45` ticks with Sharpe `0.009` and WFE `0.059`
- Feature-level clues support the overlap read:
  - the prepared short-meta report ranks `htf_confluence_short_ict_continuation` as a strong driver
  - the trained short-meta model also keeps `htf_confluence_short_ict_continuation__lag_0` near the top of `window_feature_importance.csv`

### Observations

- The leakage-safe trainer did not expose a generic short-edge collapse. It exposed a short roster that is too dependent on the same shock-event subset appearing in multiple branches.
- The pooled short-meta branch now behaves more like a broad short-side state model than a clean "meta wrapper" over separable family edges.
- The remaining continuation question is not "try another generic continuation retrain immediately." The more urgent question is whether the current short-side roster can be made honest through regime-gated deployment / concentration controls before another train-side redesign.

### Decision

- Treat the current short-side diagnosis as complete enough to change priorities.
- Move the next highest-value ICT step to a frozen-model short-side regime-gated deployment / concentration study on the July 27, 2026 leakage-safe roster.
- Keep both continuation families below promotion priority until that concentration / overlap pass is complete.
- If a policy-layer concentration pass cannot clean up the shared short pockets honestly enough, then reopen the train-side question as pooled short meta vs. family-routed short models rather than another same-shape continuation retrain.

## Step 6

Date: `2026-08-13`
Item: Execute the long ICT paper-signal lifecycle decision.

### What We Did

- Selected `ict_long_meta_xgb_v1` as the sole controlled paper-signal leader under the exact walk-forward `global_threshold = 0.40` contract with no regime thresholds or abstention.
- Kept `ict_long_reversal_xgb_v1` candidate/shadow and recorded the quantified non-additive overlap hold.
- Built a separate immutable `ict_es_paper_signal_20260813` registry, policy package, direction manifests, lifecycle summary, validation record, and idempotent builder instead of changing the frozen July 30 shadow bundle.
- Pointed ICT dashboard/runtime defaults to the new bundle and limited the dashboard's active-weight set to long-meta.
- Added fail-closed collector guards for the paper account mode, paper endpoint, disabled all-model activation override, and an explicit trial-enable switch.
- Switched the local non-secret IBKR mode/port settings from live/`4001` to paper/`4002`, left the trial-enable switch false, and did not start the collector.
- Fixed the generic live abstention-context bug so raw policy frames are labeled before off-hours/high-stress abstention is evaluated.
- Added a read-only readiness checker that validates the exact bundle contract, environment safety facts, saved validation evidence, and shared collector health without connecting to IBKR or exposing secrets.
- Added the dedicated `ict_paper_signal_events` confirmation ledger and runtime hooks for emitted long-meta signals. It records independent overlapping signal-close markouts, settles on the twentieth completed bar, applies the accepted spread/slippage/commission contract, and has no order or position semantics.

### Why We Did It

- Both long branches passed all eight saved financial promotion gates, so leaving them indefinitely at generic candidate status no longer expressed an actual lifecycle decision.
- Activating both would double-count mostly the same historical edge. The exact overlap audit found 612 shared trades producing `+5,758.20` ticks, while the 663 reversal-only trades lost `286.95` ticks.
- The prior meta live package used a static regime-threshold-plus-abstain policy even though every walk-forward fold selected the global `0.40` policy. A paper trial needed the exact evaluated contract.
- Runtime `active` status permits `emit` decisions and notifications; it does not provide broker orders, fills, positions, or execution PnL. That distinction required a fail-closed staged launch.

### Results

- The generated registry has exactly one active ICT model: `ict_long_meta_xgb_v1`.
- Meta packages global `0.40`, null regime thresholds, disabled abstention, and `selected_policy_name = global_threshold`.
- Reversal packages the same comparison threshold but remains candidate/shadow with the overlap hold in its promotion reason.
- Direct model-artifact replay passed 150 rows per selected long model across offsets `0`, `100`, and `300` with zero raw, calibrated, or label mismatches.
- Focused bundle, readiness, launch-guard, dashboard, policy, serving, and audit-replay tests passed.
- The ledger migration/repository test passed exact twentieth-bar settlement, duplicate-open idempotence, `new_york` total cost of 2.65 ticks, nullable stop/target, and repeat-settlement no-op.
- Correcting the live XGBoost missing-value contract improved historical live-feature replay from 377/414 to 405/414 matching cells on the final three frozen bars. The remaining 9 raw-value differences are limited to three arbitrary rolling coordinates. Long-meta's two affected inputs have zero Booster splits/gain and its reference-vs-live predictions are bit-identical; the third belongs only to shadow reversal and has a small nonzero prediction effect.
- The current shared ES heartbeat is stale and unhealthy, with its last generated snapshot on `2026-08-04`.

### Observations

- A clean model-file replay does not establish live parity when selected inputs contain stateful full-history identifiers.
- Raw FVG/displacement IDs are especially fragile model inputs because their numeric values change with the beginning of the recomputation window even when the underlying market structure is the same.
- The live numeric fill behavior must match the training artifact for XGBoost too; carrying event-only values can silently change probabilities even though XGBoost natively handles missing values.
- Promotion readiness should follow the active model's prediction contract, not block on raw mismatches in provably unused Booster inputs; candidate reversal still requires a clean retrain because its inversion index is used once.
- The accepted economics are a 20-bar close-to-close markout with no stop/target barrier and 385 overlapping entries in 1,555 trades. The pre-launch bundle was amended to mirror that evidence: independent overlapping event markouts are allowed, broker-position tracking is false, and stop/target are explicitly not applicable.
- Promotion status and trial start are separate lifecycle facts. The decision can be recorded while the launch remains honestly blocked.

### Decision

- Treat the model-selection decision as complete: long-meta is the sole paper-signal leader and long-reversal is the shadow challenger.
- Advance the bundle lifecycle to `authorized_pending_paper_feed_not_started` after active-meta parity and the faithful event-markout ledger pass; keep the final launch switch false and the four-week clock unset until a healthy authenticated paper-feed handoff.
- Accept active long-meta prediction parity with the documented zero-gain ID warning, but do not promote reversal until its arbitrary coordinate input is removed through retraining.
- Use the accepted event-markout contract for the dedicated confirmation ledger; do not substitute an unevaluated barrier or one-open-position strategy.
- Do not enable or launch the active ICT path until healthy authenticated paper-feed readiness passes.
- Use `model_testing/reports/ict_contract_audits/ICT_LONG_PAPER_SIGNAL_DECISION_20260813.md` as the detailed decision/audit record.

# ICT Execution Journal

<a id="ict_experiment_journal"></a>

> Merged 2026-10-09: the former `docs/ICT_experiment_journal.md` (experiments E1-E3 and its dated entries) is interleaved below by date; each moved entry is marked.


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

## Step 7

Date: `2026-08-31`
Item: Execute the short-side pooled-meta versus setup-family split-first experiment from the 2026-08-11 audit queue.

### What We Did

- Used the completed FRVP setup-specific workflow as the template for the ICT short-side overlap problem.
- Added additive setup-specific ICT target materialization without replacing the six pooled ICT targets.
- Extended ICT event-window lookup so setup-specific target names preserve leakage-aware purge / sequential-bootstrap support.
- Materialized a short-only setup prepared root at `artifacts/ict_short_setup_targets_20260811_audit/phase04_prepared/prepared`.
- Trained the three setup lanes with enough event breadth:
  - `ict_short_continuation_premium_discount_continuation_xgb_v1`
  - `ict_short_reversal_ifvg_reversal_xgb_v1`
  - `ict_short_reversal_sweep_reclaim_xgb_v1`
- Logged the detailed plan, commands, artifacts, metrics, observations, and decision in `docs/ICT_execution_journal.md#ict_experiment_journal`.

### Why We Did It

- The 2026-08-11 audit showed that `ict_short_meta_xgb_v1` shared `284` exact walk-forward trades with `ict_short_reversal_xgb_v1` and `84` with `ict_short_continuation_xgb_v1`.
- That overlap looked like the same pooled-vs-family dilution pattern the FRVP split-first work was designed to diagnose.
- The goal was to test whether setup-specific short ICT models could isolate a cleaner edge than pooled short meta or pooled short reversal.

### Results

- Setup target breadth was highly uneven:
  - trainable: `premium_discount_continuation`, `ifvg_reversal`, and `sweep_reclaim`
  - too thin for wave 1: `displacement_continuation_after_raid`, `ob_retest_after_mss`, and both session-open manipulation setup types
- Strict promotion-quality evaluation completed with requested min folds `7`, available min folds `13`, and no auto-relaxed folds.
- No qualified non-global policy was selected for any setup-specific model.
- Walk-forward economics:
  - `premium_discount_continuation`: `70` trades, `-324.50` ticks, Sharpe `-0.393`, WFE `-0.900`, gate `False`
  - `ifvg_reversal`: `298` trades, `-65.70` ticks, Sharpe `-0.036`, WFE `0.376`, gate `False`
  - `sweep_reclaim`: `755` trades, `+3049.25` ticks, Sharpe `0.750`, DSR `0.711`, WFE `1.213`, gate `False`
- The accepted pooled control `ict_short_reversal_xgb_v1` remained stronger at `791` trades, `+5475.85` ticks, Sharpe `1.048`, DSR `0.915`, WFE `1.306`, and gate `True`.

### Observations

- The overlap concern is real diagnostically, but the setup-specific challengers did not beat the pooled short reversal parent.
- `sweep_reclaim` is the only setup-specific short branch worth preserving as a research candidate, mainly for possible policy-only slicing around 2023-2026 deterioration and session behavior.
- `ifvg_reversal` has classifier signal but no realized post-cost edge.
- `premium_discount_continuation` confirms the weak short-continuation read rather than rescuing it.
- The tiny setup types should be treated as detector/event-design diagnostics, not model candidates.

### Decision

- Do not promote any ICT setup-specific short model.
- Keep `ict_short_reversal_xgb_v1` as the accepted short-side family control.
- Keep `ict_short_meta_xgb_v1` out of accepted deployment unless its concentration problem is solved separately.
- Do not run a broad full ICT setup/family sweep from this result alone.
- Move the main queue forward to the next `notes.txt` item after this documentation sync.

## Step 8

Date: `2026-08-31`
Item: Investigate the deferred classic-breaker version of Setup 4.

### What We Did

- Reviewed the Setup 4 note against public ICT breaker-block definitions: a classic breaker is a failed order block that flips after a liquidity sweep, displacement/CHoCH or MSS, and a later retest of the failed zone.
- Confirmed the saved Phase 2 ICT surface has active order-block retest columns and IFVG inversion columns, but no failed-order-block / breaker columns.
- Added a causal breaker state surface to `ict/detectors/order_blocks.py` and wired sweep context into `ict/feature_sets/ict_context.py`.
- Kept the classic breaker out of labels/default setup selection.

### Why We Did It

- The old implementation could detect an active order-block retest, but it did not preserve invalidated order blocks as flipped breaker zones.
- Treating any broken OB as a breaker would be too loose for ICT logic; the new state path requires close-through invalidation, same-direction displacement, CHoCH/MSS, recent opposite-side sweep, and a later retest.

### Result

- Future regenerated Phase 2 artifacts will include causal breaker columns such as `dist_to_bull_breaker_atr`, `dist_to_bear_breaker_atr`, `ict_bull_breaker_retest_event`, `ict_bear_breaker_retest_event`, and source order-block ids.
- Current saved artifacts remain missing those columns, so the classic breaker Setup 4 label stays disabled until regeneration and event-sample diagnostics are complete.

### Decision

- Keep IFVG as the active Setup 4 reversal-after-failure primitive for current labels.
- Do not train or promote a classic breaker branch from current artifacts.
- Next valid action is a targeted Phase 2 regeneration/audit of breaker event breadth, not model training.

## Step 9

Date: `2026-08-31`
Item: Regenerate and audit the causal classic-breaker Phase 2 state surface.

### What We Did

- Added a narrow `features/recipes/ict_setup_surface.json` recipe for setup-surface regeneration with only `ict_context` and no post-feature transforms.
- Added `ict/reports/breaker_state_audit.py` and `scripts/run_ict_breaker_state_audit.py` to audit breaker-state column presence, event breadth, and causal integrity.
- Audited the old canonical Phase 2 surface at `artifacts/ict_es_primary/phase02_features/ict_es_features.csv`; it is missing all `24` required breaker-state columns.
- Regenerated a fresh breaker-aware setup surface at `artifacts/ict_es_primary_breaker_phase02_audit_20260831/phase02_features/ict_es_features.csv`.
- Optimized FVG and order-block state scans by pruning expired/invalidated zones from active detector lists; the full ES setup-surface build dropped from a 10-minute timeout to about 63 seconds.
- Ran the breaker audit at `model_testing/reports/ict_breaker_state_audits/ict_es_primary_breaker_phase02_audit_20260831/`.

### Why We Did It

- The classic ICT breaker cannot be reviewed professionally from the old active order-block retest alone; it needs failed order-block state with source order-block lineage, formation bar, later retest event, and direction flip.
- The saved canonical artifact still predates those columns, so we needed a fresh Phase 2 surface before deciding whether classic breaker was merely theoretical or ready for label design.

### Result

- New surface rows: `666,769`, covering `2017-01-02T23:00:00Z` through `2026-06-19T16:55:00Z`.
- Breaker event bars: `67` creations and `163` later retests.
- Side split: `64` bullish breaker retests and `99` bearish breaker retests.
- Coverage: retests appear across `10` calendar years and `38` months.
- Integrity: `0` same-bar create/retest events, `100%` retests after breaker formation, `100%` source order-block presence, and `100%` finite breaker bounds.
- Audit decision: `ready_for_label_design`.

### Observations

- The raw retest sample is real but selective: `163` retest bars map to `45` unique breaker ids on retest rows.
- Some years are thin (`2018`, `2022`, and partial `2026`), so the first classic-breaker label pass should stay separate from pooled reversal rather than being merged directly into existing reversal labels.
- The current CSV surface stores nearest breaker ids on event rows, not a full per-bar multi-breaker ledger. That is enough for an initial setup branch but should be remembered if future work needs multiple simultaneous breaker objects per bar.

### Decision

- The blocker in the original note is cleared: classic breaker is no longer missing a clean causal state machine or a basic event-breadth audit.
- Do not silently add it to canonical labels yet.
- Next valid action is an explicit classic-breaker setup type / Phase 3 label-design branch, followed by a label/event audit before any model training.

## E1. Short Setup-Family Split-First Audit Experiment (2026-08-31)

*From the former `ICT_experiment_journal.md`.*

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

*From the former `ICT_experiment_journal.md`.*

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

*From the former `ICT_experiment_journal.md`.*

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

## 2026-09-13 - A1 collection baseline preservation

- `Why:` prevent legacy observations, corrected inputs, and replacement model artifacts from silently sharing performance reports.
- `What / how:` verified the existing 250-file snapshot and captured a separate 952-file current snapshot spanning all 21 audited ES models, with SHA-256 object verification; expanded collection identity inputs and enforced collection separation in dashboard markout reports.
- `Result / what worked:` both snapshots have no missing objects; 78 focused collection/dashboard/paper-guard tests pass.
- `What did not work:` four older operator-tooling tests require a missing local OTE champion manifest. Self-contained collection-report tests pass.
- `Observations:` the current preservation bytes do not prove September audit-era artifact identity; that remains B2. Full evidence, manifest hash anchors, and report usage are in `docs/ES_A1_collection_baseline_20260913.md`.
- `Decision:` A1 is complete. Keep ES shadow-only and all existing economics, drawdown, concentration, promotion, and paper-trial readiness restrictions. No scored collection or paper trial was started; no thresholds were tuned. A2 onward and B1-B5 remain separate prerequisites.

## 2026-09-14 - A2 causal session and week references

- `Why:` the original shifted-aggregate lookup needed future current-session/week RTH keys to expose known overnight references, while whole-session open mapping leaked future opening prices into earlier rows.
- `What / how:` reviewed and retained the existing strictly earlier observed-period lookup and forward-only opening-price correction in the shared ICT producer; excluded unknown timestamp keys from aggregate lookup and removed the unused whole-session open aggregate. Added reference-surface prefix regressions and a real live-builder/cache parity check.
- `Result / what worked:` 119 tests passed across causal reference/input-contract checks, ICT scaffold/detectors/setups, collection boundaries, and both paper-signal runtime guard suites. Every tested reference prefix stays unchanged as future bars arrive, including missing history, session/week boundaries, both DST changes, and year rollover.
- `What did not work:` an unknown timestamp previously sorted after all aggregate keys and could select a future reference. The new missing-key guard and regression fix that edge case.
- `Observations:` last-observed historical aggregates do not attest to complete source coverage. A missing 09:30 bar cannot be substituted with a later RTH open. Live parity retains the unverified-lineage diagnostic restriction; it does not validate deployed training artifacts. Full contract and reproducible commands: `docs/ES_A2_causal_references_20260914.md`.
- `Decision:` A2 is complete. Keep ICT research-only and preserve all existing economics, drawdown, concentration, promotion, and paper-trial restrictions. B2 still owns artifact lineage and corrective regeneration/retraining; A3 onward and other evaluation prerequisites remain queued. No collector, qualified collection period, or paper trial was started, and no model thresholds or artifacts were changed.

## 2026-09-14 - A3 HTF alignment and live input verification

- `Why:` corrected reference values are insufficient if other required inputs are stale, absent, or substituted by an unverified zero fallback.
- `What / how:` verified shared HTF publication and live-builder prefix parity; tightened malformed-bucket and transform/history checks, checked all consumed lag/sequence rows, and retained per-model contract metadata through cache, health events, and persisted observations. The contract is now `es-causal-inputs-v2`.
- `Result:` 156 broad tests passed, followed by all 33 focused contract tests after final multi-row producer-attestation coverage (157 distinct tests). The integration check proves that a fresh bar plus a fallback score stays diagnostic/shadow, with reasons recoverable from SQLite and the health query.
- `Observations:` all nine non-retired ICT manifests require corrected-lineage review. Long meta selects two confluence helpers and short meta selects one; their training label-output producers are absent from the live feature path. No generic alignment proxy was substituted for those event-specific definitions. Full inventory, findings, and reproduction: `docs/ES_A3_input_contract_20260914.md`.
- `Decision:` A3 is complete. Keep ICT research-only; B2 owns helper timing/parity, prepared-data and artifact lineage, and any regeneration/retraining. Preserve concentration, economics, promotion, and paper-trial restrictions. No collector, qualified collection period, paper trial, artifact replacement, or threshold change was introduced.

## 2026-09-15 - A11 ICT research-only roster and pending priorities

- `Why:` the A2 live feature correction does not certify training/artifact lineage. All nine packaged ICT models still need B2 review, and dashboard priority must not imply activation or promotion.
- `What / how:` added `es-ict-research-priorities-v1` with an explicit quarantine for every ICT model, including unknown replacement IDs. The ES processor forces diagnostic shadow evaluation even with passing input checks and an active binding. A6 persists roster metadata and rejection reasons, retains raw/setup/policy diagnostics, and prevents candidate state advancement. Collection identity includes the roster version and implementation hash.
- `Result / what worked:` 266 tests passed across two suites. All nine roster models plus an unknown replacement retain their scores and persisted rejections with passing synthetic inputs and complete policies. A real processor/SQLite integration case proves that a fresh bar, passing stamped contract and active binding cannot bypass B2. Both ICT Dash tabs retain background diagnostics and show pending focused priorities without active-weight badges.
- `What did not work / limits:` no artifact has been cleared by this app task. B2 still owns causal regeneration, affected-label review, corrective training/calibration and artifact-specific evidence. UI verification exercised Dash callbacks/layout endpoints, not an interactive browser or running collector.
- `Observations:` after required contracts/artifacts pass, long continuation and the short premium/discount specialist have focused shadow-comparison priority. Other families, both meta models, IFVG and sweep remain background comparisons. Long continuation's q40 largest-trade share of 13.59% still exceeds the 10% gate. Short meta/reversal concentration, profitable-quarter breadth and shock-day exposure remain unresolved; meta helper producers also require review. Research priority overrides none of these findings.
- `Decision:` A11 is complete; keep the whole ICT roster research-only. Release requires a branch-specific, evidence-backed B2 review and versioned roster change, followed by the existing input/policy/baseline prerequisites. B5's full promotion contract remains open. No collector restart, artifact/threshold change, qualified collection period, paper-trial activation or broker order was introduced. Evidence and commands: `docs/ES_A11_ICT_research_roster_20260915.md`.

## 2026-09-18 - B2 saved lineage inventory and dependency trace

- `Why:` identify which saved datasets and trained artifacts inherited the reference/HTF inconsistencies before deciding on corrective work.
- `What / how:` inventoried 113 primary trained bundles across ICT/FRVP/OTE, eight auxiliary weight files, 120 registrations and 18 prepared roots; saved dependency edges, selected-feature routes and current-file hashes. Queried the frozen September snapshot in read-only mode and compared bounded historical feature/ATR samples in memory.
- `Result:` all 17 ICT bundles reference the old base feature tape and select affected reference inputs or transforms. The tape publishes the January 3, 2017 RTH open at the January 2 overnight row. In the 4,000-row witness, 2,694 RTH-open values differ from the causal producer. The refreshed and short-specialist prepared datasets retain the same base feature ancestry. Long/short meta helper values are zero in the frozen live observations while the corresponding prepared training helpers contain nonzero values.
- `Observations:` reference dependence flows through sweeps, DOL, MSS/breakers, setup geometry and potentially labels/masks/weights; it does not make every ICT-named feature reference-dependent. A witnessed defective parent does not quantify changes among a model's actually consumed rows. Exact historical training input hashes are still missing. See `docs/B2_model_repair.md#b2_lineage_inventory_and_dependency_trace_20260918` and its per-model CSV/JSON evidence.
- `Decision:` inventory/tracing complete; B2 remains open for full consumed-row impact and minimum repair decisions. No source producer, historical dataset, label, model, calibration, threshold or runtime eligibility was changed. No training or evaluation activation. Keep all nine packaged ICT models research-only and preserve the existing concentration/promotion restrictions.

## 2026-09-18 - B1 classic-breaker exclusion and all-pooled diagnostics

*From the former `ICT_experiment_journal.md`.*

- `Question:` settle classic-breaker's baseline scope without inventing economics for its sparse research branch, and retain pooled-model versus setup evidence.
- `Decision:` explicitly exclude classic breaker from the baseline while keeping detector state, existing research/prepared artifacts and diagnostic logging. All ICT pooled reversal/continuation/meta predictions remain diagnostics-only even when a setup matches. Named specialists retain their own contracts subject to current restrictions.
- `Implementation:` locked `b1_baseline_scope_v1`, enforced by paper evaluation and independently by the position simulator; no paper barrier outcome or P&L for excluded rows. Preserve causal gross forward responses, raw/calibrated scores, thresholds, all setup relationships, regimes, sample counts and missingness. Add read-only single-collection exports and model/setup audit artifacts.
- `ATR distinction:` ICT's existing contract uses signal-bar ATR14, not FRVP's structural hourly ATR. The confirmed FRVP hour-open publication defect is documented for B2; this change does not silently alter ICT risk geometry or any historical labels.
- `Evidence:` [scope, ATR and audit design](B1_outcome_contracts.md#b1_baseline_scope_and_model_setup_audit_20260917), [current matrix](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260918), and live change journal validation.
- `Restrictions:` no classic-breaker training/promotion, pooled baseline positions, threshold tuning or activation. Existing ICT quarantine and economics/concentration gates remain. Prior research results and decisions below remain historical records.

## 2026-09-19 - B2 measured impact and proposed minimum repair

- Compared the full 666,769-row reference tape and all available ICT prepared splits. Every one of the 17 saved ICT bundles has changed selected reference inputs; applying the train-only missing-value rule to corrected references changes 99.27-99.95% of each bundle's training rows. Finite saved reference values reproduce the prepared values. Counts describe prepared inputs, not reconstructed individual bootstrap draws or new prediction performance.
- Proposed repair: rebuild shared causal references and their dependent features; compare setup/event/label/weight paths and retain unchanged label products; rebuild prepared inputs and retrain only retained affected models. Remove retrospective confluence predictors or supply a proven identical causal producer offline and live. Complete corrected ICT label counts remain pending.
- Evidence: `docs/B2_model_repair.md#b2_minimum_repairs_20260919`, per-model repair CSV and `docs/audits/b2_impact_20260918/`. This is a review proposal. No production code, historical data, model, calibration, threshold or runtime was changed; no training or baseline release. B2 stays open and all existing ICT restrictions remain.

## 2026-09-21 - B2 retained shared-data repair

- Corrected ICT references, FRVP completed-hour ATR/confirmed-swing gates and OTE completed-HTF publication. Historical label controls reproduced before correction; label-derived helper predictors are excluded from new preparation.
- Rebuilt and checked 25 prepared datasets for 27 retained models; preserved independent FRVP profiles/setup inputs and the existing long-reversal recency recipe. The nine missing-history bundles remain quarantined.
- Validation: 127 tests passed; chronological split gaps, finite features/weights and target coverage checked. 1,489 historical model/companion files, the OTE registry and 12 key data parents are unchanged. No trading model was trained or activated; existing research/shadow/diagnostic restrictions remain.
- Report and replacement-data paths: [B2 shared-data repair](B2_model_repair.md#b2_shared_data_repair_20260921). B2 remains open for selected-model retraining/calibration, versioning and B1 evaluation.

## 2026-09-28 - Proposed ICT candidates: ES movement-gate input audit

- `Why:` verify the same expected-move/spread inputs needed by the S3 repair for proposed ICT long continuation and short premium/discount candidates.
- `What / how:` shared raw-bar es_abs_close_move_20x5m_v1 estimates next-five-minute absolute movement from 20 trailing returns in ES ticks; missing/gap/roll/incomplete contexts remain unavailable. New opt-in policies use the existing B1 ES tick-spread schedule. No labels/outcomes fit the calculation. Audit uses held-out identities only as witnesses, without model fitting or policy installation.
- `Result:` ID 11 has 134 ready / 107 warmup observations among 241 endpoints, with 123 app witnesses; ID 17 has 104 ready / 81 warmup among 185 endpoints, with 96 app witnesses. Offline/app movement and checked session/stress inputs match. The six-candidate audit totals 3,254 endpoints / 514 witnesses; 258 regression tests pass.
- `Observations / limits:` frequent ICT warmup is preserved rather than imputed. ID 11's examined legacy policy already disables abstention with multiplier zero; its proposed unit binding does not implicitly activate filtering. No ICT economics, all-feature parity or lag/history certification is supplied by this input audit.
- `Decision:` retain all ICT lineage/concentration/promotion restrictions and pending model-history/qualification work. No installed policy, model, registry, collector or paper trial changed. Evidence: [ES expected-move contract and candidate audit](B2_model_repair.md#es_expected_move_contract_20260927).

## 2026-09-29 - Selected ICT candidate history repair and retraining

- `What / why:` independently audited IDs 11 and 17. Every held-out lag history differed between retained labeled rows and consecutive source bars, changing 36/241 and 9/185 archived threshold decisions with frozen training fills. Raw missing-value inference also differed from training imputation.
- `How:` rebuilt full source-history views and carried frozen prefix fills with each scaler. Retrained only these two selected XGB branches, preserving features, labels, weights, corrected ICT event windows, sequential bootstrap, seeds, search budgets, purged folds/holdout and validation-free final refits.
- `Result:` all 26 trials attempted: 18 complete, eight pruned, zero failed. Production backend/runner replay passes all 426 held-out predictions with exact threshold decisions and maximum score error below 8.92e-8. New OOF-only thresholds are 0.56 for ID 11 and 0.53 for ID 17; archived thresholds/weights remain frozen. All model, source, preparation and code snapshot pins pass review.
- `Observations:` the corrected history contract does not establish net economics, resolve concentration findings or certify actual live feature producers. The technical suite passes 33 focused tests; the broader pipeline has 37 passes, two skips and the known unsupported cv_max_folds failure. No test outcomes selected thresholds or trading hours.
- `Decision:` retain diagnostic-only status and all ICT readiness/concentration/promotion restrictions. Freeze new B1/B3/B4 bindings before candidate outcome review; live parity and prospective eligibility remain open. Background/OTE training is not a blanket prerequisite for selected ES collection. No runtime installation, registry change, collector restart, paper trial, scored collection or broker order. Evidence: [candidate history audit](B2_model_repair.md#b2_candidate_history_audit_20260928), [diagnostic-only roster](../configs/b2_candidate_history_roster_20260928.json).

## 2026-09-29 - B1 named-candidate historical outcome review

- `Question:` test whether the corrected ICT premium/discount specialist has candidate-specific post-cost evidence under its frozen B1 contract, with common matching opportunities and no outcome-based threshold selection.
- `Control / method:` bind ID 17 to `es_ict_premium_discount_continuation_eval_v1`, use its exact replacement-model held-out named events, one ES position, full B1 costs and the predeclared B3/B4 comparison. The offline historical clock is completed signal bar plus one second; live producer parity remains a separate gate.
- `Result:` 185 common opportunities. Setup-only admitted 87 and closed 86 for **+$3,901.25** net; frozen 0.53 threshold admitted 53 and closed 52 for **-$1,678.75** net, with one unresolved exposure. The full-policy leg is insufficient evidence because no exact replacement-version policy binding was frozen.
- `What did not work / limits:` the model-filtered leg did not beat the unchanged setup control. This is not a reason to tune the inspected sample, and it is not a promotion pass. ID 11 is a pooled ICT model and is explicitly excluded from the B1 qualified baseline even when matched.
- `Decision / repo change:` ID 17 is reviewed but not promotion-cleared. It may be considered for prospective research after live parity and B5-B7/A12/A13 gates; future C1 outcomes are not required to start an otherwise eligible research collection. Evidence: [B1 candidate outcome review](B1_outcome_contracts.md#b1_candidate_outcome_review_20260929), [audit JSON](audits/b1_candidate_outcome_review_20260929.json), [frozen comparison](../configs/b1_candidate_outcome_review_20260929_v1.json).

## 2026-09-29 - B5 baseline eligibility, ICT priorities, and promotion contract

- `Question:` which ICT branches may enter the next prospective scored shadow period, and what evidence is required before an ICT result can be called promotion-ready?
- `Control / evidence:` B4's frozen named baseline `17` in `configs/b4_baseline_freeze_20260929.json`, the B1 candidate outcome review, the B5/B6 parity audit in `docs/B5_B6_live_feature_parity_20260929.md`, and the prior ICT execution and setup-family findings. The B4 binding is for research comparison only; it does not authorize installation, promotion, or a paper trial.
- `Result:` no ICT branch is currently eligible for prospective scored shadow research. B5/B6 reviewed six branches and recorded zero full-parity passes and zero promotion passes. The recorded app archive contains post-imputation values rather than raw producer features; completion, availability, source identity and full causal gap state are not proven, and the actual incremental producer rebuild was resource-bounded and not run.

### ICT eligibility and research-priority roster

| Branch | Collection status now | Research priority | Binding reason / unresolved gate |
|---|---|---|---|
| ID 11, pooled long continuation XGB | Diagnostic-only; excluded from qualified baseline | Background comparison | Pooled-model scope restriction; B5 raw producer parity and completion/availability evidence are missing. The earlier long-continuation q40 pocket reached only `70` trades and `13.59%` largest-trade share, above the `10%` gate. |
| ID 17, short premium/discount continuation specialist | Diagnostic-only; not promotion-cleared | Highest ICT named-candidate priority, conditional on clearing gates | B4 frozen at threshold `0.53`, but B1 setup-only was `+$3,901.25` while the frozen-threshold branch was `-$1,678.75` with one unresolved exposure; the exact replacement full-policy leg is insufficient evidence and B5 parity is not proven. |
| Other ICT continuation, reversal, meta, IFVG and sweep branches | Background/diagnostic-only or deferred per prior roster | Preserve for comparison; no activation priority | Prior full-contract evidence remains binding: short-meta and short-reversal were concentrated in the same `284` exact events; short continuation remained weak/extremely concentrated; setup-specific sweep-reclaim was positive but failed the promotion gate. |
| Classic breaker | Excluded; diagnostics retained | Deferred until a separate label/model contract | Baseline scope explicitly excludes it; sparse first-retest-only artifacts do not establish trainable or promotion-quality economics. |

There are currently zero rows in the `eligible for prospective scored shadow research` category. ID 17 is a conditional research priority, not a promotion candidate. Pooled ICT models remain diagnostic-only even when a setup matches, and all earlier ICT concentration, overlap, economics and readiness restrictions continue to apply.

### Promotion criteria retained for ICT

The ICT promotion decision uses the same conjunctive contract as FRVP and is evaluated against the same named-setup opportunities: setup-only, setup plus the frozen threshold, and setup plus the intended full policy. No model-only metric, target-hit count, empty/full-policy zero-trade result, or classifier replay can stand in for a complete position-level comparison.

1. **Contract and readiness:** exact replacement artifact, scaler/history, selected-feature order, setup matcher, threshold/calibration, policy, signal-bar ATR14 geometry, instrument and B1 outcome-contract hashes must be bound. Raw live producer parity, causal history, completion/availability, warmup, gaps, closures, rolls and fallback state must pass, followed by B7/A12/A13.
2. **Intended exits and realistic costs:** use the frozen ICT stop/target/timeout/continuation-management contract and its causal entry clock, with one-position/concurrency rules and realistic ES scheduled costs. A positive label, target touch, gross markout, or setup-only economics is not a net-profitable trade. Missing or unresolved exits remain unresolved, not wins or zero trades.
3. **Incremental economics:** the full-policy branch must be positive after costs and add value over the unchanged setup-only benchmark on the same opportunities, with no unjustified worsening of expectancy, drawdown or exposure concentration. ID 17 currently fails this comparison because model filtering turns a positive setup control negative.
4. **Time stability and discrimination:** require strict chronological/leakage-safe folds, the existing minimum fold/cadence requirements, configured Sharpe/DSR, positive WFE, profitable-quarter breadth, positive-composite-expectancy breadth, and discrimination above the setup/base-rate control. The full-span result must support the claim; a recent or one-hour pocket cannot substitute for it.
5. **Concentration and drawdown:** largest-single-trade share must meet the existing `10%` gate, with exact-event overlap, day/session/regime concentration and shock-day dependence reported. The declared drawdown advisory/risk limit must be respected or the branch remains non-promoted pending explicit risk review. Retained older findings include short-meta largest-trade share `10.017%` with `16/27` positive quarters, short-reversal largest-trade share `11.118%` with `15/27` positive quarters, and the q40 long-continuation pocket at `13.59%` concentration.

The September one-hour audit, tiny hourly buckets, retrospective profitable-hour filters and any small positive pocket are descriptive diagnostics only. They cannot change the frozen threshold, setup, exit, cost, session or concentration gates. Any session restriction must be a separately frozen prospective experiment with full-session control.

### Prospective collection boundary

If an ICT branch later clears the prerequisites, its clean C1 sample starts at the first healthy post-B7 timestamp under a new immutable candidate/runtime identity, with full eligible session coverage and all matching/rejected named setup opportunities retained. Review after approximately 20 fully collected sessions and 50-100 resolved matches as a practical target; rare specialists may need longer. Any model, threshold, calibration, setup matcher, policy, cost, session scope or exit change creates a new version and a new clean boundary.

- `Decision / repo change:` B5 is recorded as a gate-setting decision, not a clearance. ICT keeps no scored-shadow-eligible branch, keeps ID 17 as the conditional named research priority, and preserves pooled diagnostic-only scope, classic-breaker exclusion, prior overlap/concentration findings, and all economics/readiness restrictions. No model, registry, threshold, collector, paper trial or order state changed.

## 2026-09-29 - B5 eligibility and promotion-contract cross-reference

*From the former `ICT_experiment_journal.md`.*

- The canonical B5 roster, retained full-contract economics/concentration/readiness findings, promotion gates, and clean prospective-collection boundary are recorded in [ICT execution journal B5](ICT_execution_journal.md#2026-09-29---b5-baseline-eligibility-ict-priorities-and-promotion-contract).
- Current disposition is unchanged for this experiment journal: zero ICT branches are eligible for scored shadow research; ID 17 is only a conditional named-candidate research priority, pooled branches remain diagnostic-only, classic breaker remains excluded, and no one-hour or small positive pocket can override the full contract, cost, time-stability, drawdown, or concentration gates.

## 2026-10-01 - B2 catch-up: ID 14 source-history backfill

- `What:` brought ID 14 (`ict_short_continuation_xgb_v1`) to the same B2 source-history stage as the 12-model core: new prepared root from the 09-21 repaired ICT data (666,769 consecutive bars, 96 frozen features, frozen-prefix fills), window preflight, retraining into a new versioned model root, technical review.
- `Why:` the broad full-session shadow roster needs IDs 5/23/14/18 at the B2-reviewed stage before the 16-model B4 freeze and route manifest.
- `Result:` preflight 1,035 exact training-vs-runtime tensor comparisons across windows 8-24; 10 trials = 7 complete / 3 pruned / 0 failed; 6 purged folds (purge 47) and purged holdout reproduced; calibration none; OOF-only threshold 0.53 reproduced; production-runner replay of all 191 test predictions within 8.04e-8 with identical decisions; legacy bundles and frozen configs unchanged. Technical review passed.
- `Observations:` advisory only - OOF AP 0.442, test AP 0.456, test ROC AUC 0.456 on a small sample; not a gate and not used for any selection. Original recipe, folds, purge, seeds and search budget unchanged.
- `Decision:` ID 14 is `reviewed_pass`, diagnostic-only (`collection_eligible=false`); no registry, app, collection, paper, alert or broker change. ID 18 backfill is next. Evidence: [catch-up report](B2_model_repair.md#b2_candidate_history_catchup_5_14_18_23_20261001), [review](audits/b2_catchup_14_review_20261001.json).

## 2026-10-01 - B2 catch-up: ID 18 source-history backfill

- `What:` brought ID 18 (`ict_short_reversal_ifvg_reversal_xgb_v1`) to the same B2 source-history stage as the 12-model core: new prepared root from the 09-21 repaired ICT specialists data (666,769 consecutive bars, 96 frozen features, frozen-prefix fills), window preflight, retraining into a new versioned model root, technical review.
- `Why:` last of IDs 5/23/14/18 needed at the B2-reviewed stage before the 16-model B4 freeze and route manifest.
- `Result:` preflight 2,280 exact training-vs-runtime tensor comparisons across windows 8-24; 16 trials = 5 complete / 11 pruned / 0 failed; 21 purged folds (purge 36) and purged holdout reproduced; calibration platt; OOF-only threshold 0.32 reproduced; production-runner replay of all 440 test predictions within 8.48e-8 with identical decisions; legacy bundles and frozen configs unchanged. Technical review passed.
- `Observations:` advisory only - OOF AP 0.372, test AP 0.414, test ROC AUC 0.573; not a gate and not used for any selection. Original recipe, folds, purge, seeds and search budget unchanged.
- `Decision:` ID 18 is `reviewed_pass`, diagnostic-only (`collection_eligible=false`); no registry, app, collection, paper, alert or broker change. All four catch-up IDs are now B2-reviewed; the 16-model selection, B4 freeze and route manifest follow as new versions. Evidence: [catch-up report](B2_model_repair.md#b2_candidate_history_catchup_5_14_18_23_20261001), [review](audits/b2_catchup_18_review_20261001.json).

## 2026-10-01 - IDs 14/18 in the 16-model B4 freeze v2 and route bindings v2

- `What:` froze IDs 14 and 18 in `broad_full_session_shadow_v2` / B4 freeze v2 and bound their outcome routes in route bindings v2. ID 14 (short continuation, detector-selected): `premium_discount_continuation` and `displacement_continuation_after_raid` v1 contracts. ID 18 (short IFVG specialist, exact match): `es_ict_ifvg_reversal_eval_v1`. Both use `es_session_cost_v1`.
- `Why:` bring both to the same stage (section 3, item 1) as the 12 core models so the remaining gates run for all 16 together.
- `Result:` the 12 v1 freeze entries re-hashed from disk and carried forward identical; v1 routes identical in v2; both manifests load and resolve; 175 focused tests pass. v0/v1 selection, v1 freeze, v1 routes and B1 v3 unchanged.
- `Observations:` thresholds frozen at the OOF values (ID 14 0.53, no calibrator; ID 18 0.32, platt). Full policy stays `not_defined` for both.
- `Decision:` identity and route freeze only; no collection, registry, paper, alert or broker authority. Evidence: [route bindings v2](B3_B4_freeze_and_routes.md#broad_full_session_shadow_route_bindings_v2_20261001), [freeze v2](B3_B4_freeze_and_routes.md#b4_broad_full_session_shadow_freeze_v2_20261001).

## 2026-10-02 - Section 3 surfaces, ledgers and session coverage (shared, ES ICT models)

- `What:` the shared surfaces and ledger module `model_testing/outcome_surfaces.py` (schema `section3-surfaces-v1`) and the D9 instrument-capacity ordering key in the b1-v5 simulator now cover every ES route, ICT included.
- `Why:` bind the four surfaces, independent and instrument-capacity ledgers, and full-session coverage before the smoke replay.
- `Result:` `tests/test_section3_surfaces_ledgers.py` 16 passed on synthetic FRVP and OTE fixtures; no ICT-specific fixture and no ICT model run on real data.
- `Observations:` ICT models share the single ES instrument-capacity ledger with FRVP in frozen selection order; all ICT full policies stay `not_defined`. No P&L was produced or inspected.
- `Decision:` infrastructure only; no collection, registry, paper, alert or broker authority. Details in [progress](section3_simulation_rules.md#section3_progress).

## 2026-10-02 - Section 3 deterministic smoke replay (ICT IDs 12, 13, 15, 16, 19, 18, 14)

- `What:` ran the roster-driven smoke replay `scripts/run_section3_smoke_replay.py` (engine b1-v5, B4 v2, route bindings v2, B3 v3) for the seven ICT models, with the shared ES instrument-capacity ledger.
- `Why:` prove setup routing, entries/exits, costs, ledger accounting and replay hashes per exact frozen version.
- `Result:` all seven passed: identical hashes across two independent runs, accounting checks clean, capacity and independent ledgers reconcile by trade identity with 0 mismatches.
- `Observations:` routes with no held-out fired event are unexercised: IDs 12 and 15 (4 of 8 each), 13 and 16 (4 of 6 each). Every held-out ICT event opens outside 09:30-16:00 New York, so the RTH element is unexercised for all seven. No open position and no counterfactual markout occurred. Full policy `not_defined` for all. P&L recorded as advisory, not interpreted.
- `Decision:` contract verification only; no collection, registry, paper, alert or broker authority. Details in [smoke replay](section3_simulation_rules.md#section3_smoke_replay_20261002).

## 2026-10-02 - Section 4 R1 raw snapshots and the D8 HTF finding (ICT IDs 12–19)

- `What:` a copy-only raw pre-imputation snapshot (`s4-raw-preimpute-snapshot-v1`) for the ES_ICT group, run on/off with B4 v2 bytes on isolated stores.
- `Why:` B5 parity needs raw values with explicit provenance, warmup, missing-producer and fallback state.
- `Result:` all seven models scored. Features, scaled inputs, scores and threshold decisions were bit-identical with snapshotting on and off.
- `Observations:` with completion dropped on the store-fetch path, the app's 30m/1h HTF features are NaN. Carrying provenance (D8) changes 25 of 300 group features on recorded bars (HTF and interactions), and endpoint NaN falls from 27 to 7.
- `Decision:` D8 stays opt-in and off pending the user decision recorded in `docs/section4_progress.md`.
- `Update (2026-10-02, Q-D8 approved):` protocol v2 locked. The parity app path runs with D8 on, historical parity bars are copied as complete, and B6 must enable D8.

## 2026-10-05 - Section 4 full live-parity run (ICT IDs 12–19)

- `What:` protocol v2 matrix (classified under v3) over 15 ES windows, plus composition, as one shared ES_ICT producer pass per window. Reports: `docs/audits/section4_parity_*_20261005.json`; summary: `docs/section4_progress.md#section4_parity_20261005`.
- `Why:` B5 must prove that the real app path reproduces each frozen model's features, preprocessing, scores and threshold decisions.
- `Result:`
  - `fail:training_side_noncausal`:
    - 14 and 16: positional row-index predictors (`*_formed_index`, `*_inversion_index`, `*_displacement_id`; Q-T1);
    - 18: both positional and `ict_raw_swing_*`;
    - 13 and 19: `ict_raw_swing_high/low`, a centered pivot that needs the next bars (`ict/structure/swings.py:66-82`).
  - `fail:app_producer_bug` (window truncation only): 12 and 15.
  - Declared under protocol v3 (Q-P1): float32 tree-split score differences at 14 endpoints (12, 13, 15), with no decision flips.
- `Observations:`
  - recorded ESU6 windows differ only through the HTF completion gate (declared; not known-completion evidence);
  - restart, chunk resume and composition are bitwise equal;
  - S2: every reference ICT event matched with exact geometry; app-only events are outside the fired-event reference;
  - source parity incomplete (IBKR down).
- `Decision:` all seven branches stay diagnostic and unscored.
  - Repair options for the user: retrain with causal forms (bars-since instead of row indexes; `ict_confirmed_swing_*` instead of `ict_raw_swing_*`) for 13, 14, 16, 18, 19.
  - A versioned truncation fix is a prompt 5 candidate for 12 and 15.

## 2026-10-07 - Follow-up proposal: training-data sufficiency, hard negatives and an every-bar model (ICT)

*From the former `ICT_experiment_journal.md`.*

### Status

- Proposed, not started. Runs after the v2 retrain (R3), B3 and B4; out of section 4 scope. It changes labels, so it needs new versioned label roots and its own retrain.

### Why

- Each ICT model is a meta-label filter trained only on setup events. 9.5 years of ES history (2017-01-02 to 2026-06-19, 666,769 5-minute bars) yields few events per target. The excluded ~660k bars (`exclude_<target>`) carry no label for "did this setup win".
- The question raised (user, 2026-10-07): would training on every bar teach the model where to trade and where not to?

### Evidence (v2 validation views, R2c)

- Events (usable rows) per target: long_ict_meta (ID 12) 7,406 (train 5,184 / 1,839 wins); short_ict_meta (15) 7,438; short_ict_reversal (16) 6,170; long_ict_reversal (13) 5,800; short_ict_reversal_sweep_reclaim (19) 3,226 (train 2,258 / 593 wins); short_ict_reversal_ifvg_reversal (18) 2,939; short_ict_continuation (14) 1,268 (train 887 / 380 wins, test 191 / 90 wins).
- ID 14 loads 96 features; with lags and deltas that is several hundred inputs against 887 training rows (overfitting risk). Its R3 pilot made 60 positive test calls, so a 95% interval on its win rate is about +/-12 pp.
- Treat IDs 14 and, to a lesser degree, 18/19 as low-confidence until B3/B4 shows fold-stable results. The meta models (12, 15) have reasonable sample sizes. This matches the earlier classic-breaker finding that event frequency, not label design, is the binding constraint.

### Why not simply add the excluded bars as 0s

- An ICT label means "this setup hit target before stop". A non-setup bar has no entry, stop or target, so 0 there means "no setup", not "bad trade". It teaches setup recognition (which the rules already do exactly) and buries the ~0.4% positives.
- Adjacent 5-minute bars are nearly identical with overlapping outcomes, so the effective sample is far below the row count.

### Proposed experiments

- E-A hard negatives: add near-miss bars (an ICT setup partly formed but did not trigger, e.g. a liquidity sweep without displacement or FVG, or a breaker without a qualifying retest) labelled with the same entry/stop/target geometry as if entered. Needs a per-setup near-miss definition, first-retest-only kept where it is a contract invariant, and the causal-prefix and leakage audits.
- E-B every-bar model: a fixed triple-barrier label from every bar (family horizon, ATR-scaled target/stop), setup flags as features, uniqueness weights / sequential bootstrap and purged, embargoed walk-forward CV. This is the design the EURUSD OTE models already use (180k-540k rows).

### How to compare

- Same walk-forward windows, costs and exit contract as the meta models. Score on the meta model's setup opportunities (post-cost expectancy per trade, AP lift over base rate, concentration, shock-day exclusion, time stability), and separately report any off-setup trades E-B adds and their own post-cost expectancy. Promotion still follows the existing contract.

### Decision

- None yet. The R3 retrain keeps the event-conditioned (meta-label) design unchanged. Cross-reference: `docs/FRVP_experiment_journal.md` (same proposal for FRVP), `notes.txt` backlog.

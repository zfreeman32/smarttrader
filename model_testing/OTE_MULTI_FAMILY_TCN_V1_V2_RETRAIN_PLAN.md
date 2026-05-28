# OTE Multi-Family TCN v1/v2 Retrain, Retune, and Recalibration Plan

## Purpose

This document turns the last two multi-family TCN training trials into a concrete next-step plan for:

- `models/ote_multi_family_tcn_v1`
- `models/ote_multi_family_tcn_v2`

The goal is to decide, for each of the six target models, whether to:

- keep the current artifact,
- recalibrate it,
- retune and retrain it,
- or retire it from the active candidate set.

This plan uses the saved `training_summary.json`, `training_history.json`, and cross-version comparisons from the two latest runs.

---

## Executive Summary

| Model | Keep Now | Why | Next Step |
|---|---|---|---|
| `short_breakout` | `v1` | Best CV and test metrics in both trials | Freeze `v1`, optional bounded challenger run |
| `long_breakout` | `v1` | `v2` collapsed to 7 test events and F0.5 `0.0513` | Retune `v2` from the `v1` neighborhood |
| `long_reversal` | `v1` active, `v2` challenger | `v2` improved AP slightly, but `v1` still wins on test event F0.5 | Recalibrate `v2` first, retrain only if needed |
| `short_reversal` | `v1` | `v2` lost too much recall and test F0.5 | Retune `v2` from the `v1` neighborhood |
| `long_continuation_pullback` | keep neither active | Both versions emit 0 test events | Full retune plus continuation calibration fix |
| `short_continuation_pullback` | keep neither active | `v2` improved AP, but thresholded behavior is still unusable | Use `v2` as the research base and retune |

---

## What The Last Two Trials Tell Us

### 1. Most models did not run out of stability. They ran out of epoch budget.

Both `v1` and `v2` used a fixed 18-epoch PyTorch budget. The current trainer already has patience-based early stopping and best-checkpoint restore in `model_training/ote_training/torch_trainer.py`, so the issue is not "missing early stopping." The issue is that validation AUPRC kept improving through the end of training for most targets.

Observed pattern:

- In `v2`, 5 of 6 targets hit best validation AUPRC on the final epoch.
- In `v1`, 5 of 6 targets also hit best validation AUPRC on the final epoch.

Interpretation:

- The trainer is behaving as designed.
- The 18-epoch schedule is usually too short for these TCN runs.
- The current auto phase split is effectively too compressed for several targets.

### 2. Continuation targets have a pipeline-level handicap.

In `model_training/ote_training/ote_xgboost_pipeline.py`, `resolve_calibration_method()` forces any target containing `"continuation"` to use calibration method `none`, even when `platt` was requested.

That means:

- continuation targets are not being allowed to use the same probability correction path as breakout/reversal targets,
- their thresholds are selected on uncalibrated OOF probabilities,
- and both continuation models are currently failing on held-out thresholded behavior.

### 3. Threshold selection currently has no explicit coverage floor.

The threshold selector in `model_training/ote_training/ote_xgboost_pipeline.py` optimizes:

1. event F0.5
2. event precision
3. closeness of predicted event count to true event count

That is reasonable for strong models, but it does not explicitly require:

- a minimum number of emitted events,
- a minimum recall floor,
- or a minimum monthly trade count.

For continuation targets, where score distributions are weaker and calibration is disabled, this can still produce thresholds that are unusable on held-out data.

### 4. `v2` improved search geometry, but not every target family improved.

`v2` brought useful changes:

- 3 CV folds instead of 2
- smaller purge-aware geometry steps
- cleaner artifact summaries

But the actual model quality was mixed:

- `short_breakout` remained excellent, but `v1` is still better.
- `long_reversal` was roughly flat to slightly mixed.
- `long_breakout` regressed sharply.
- continuation remained unusable at deployment thresholds.
- `short_reversal` regressed materially.

---

## Global Fixes Before The Next Retrain Batch

These should happen before we judge the next training run.

### A. Increase epoch budget and make the phase schedule explicit

The stronger reference point is not the generic 18-epoch multi-family schedule. It is the proven schedule used by `models/ote_full_tcn_v2`:

```text
epochs = 40
warmup = 4
main   = 18
fine   = 10
tail   = 8
fine_lr_scale = 0.20
tail_lr_scale = 0.07
use_amp = false
```

That schedule produced strong saved results:

- `long_ote`: CV AP/F0.5 `0.6254 / 0.7738`, test AP/F0.5 `0.7625 / 0.8518`
- `short_ote`: CV AP/F0.5 `0.6239 / 0.7473`, test AP/F0.5 `0.7375 / 0.7871`

More importantly, the tail phase appears to have mattered:

- `long_ote` hit best validation AUPRC at epoch `39/40`
- `short_ote` hit best validation AUPRC at epoch `37/40`
- both improved materially during the tail phase rather than flattening early

So the next multi-family retrain batch should borrow this proven shape instead of using a shorter generic schedule.

Recommended default:

```text
--epochs 40
--torch-warmup-epochs 4
--torch-main-epochs 18
--torch-fine-epochs 10
--torch-tail-epochs 8
--torch-fine-lr-scale 0.20
--torch-tail-lr-scale 0.07
```

Recommended runtime defaults:

- `use_amp = false` for TCN runs, matching the stronger `ote_full_tcn_v2` artifacts
- keep the existing best-checkpoint restore logic
- keep the explicit phase schedule so the late tail behavior is visible in `training_history.json`

Important comparison caveat:

- `ote_full_tcn_v2` was trained on much larger OTE datasets with 8 CV folds, so it is not a perfect apples-to-apples comparison with the six multi-family targets.
- even so, it is the best on-disk evidence we have for what a successful TCN schedule looks like in this repo.

Review gate:

- If a retrained model still peaks in the final `1-2` epochs under the `40` epoch schedule, raise it to `48` for that target on the next pass.
- If a target clearly flattens before the tail, keep `40` but consider shrinking tail epochs for that specific family later.

### B. Allow continuation calibration experiments

For the next continuation experiments, do not hard-disable calibration.

Test these variants on OOF predictions:

- `none`
- `platt`
- `isotonic`

Pick the winner by:

1. OOF average precision
2. OOF event F0.5
3. coverage sanity, meaning the threshold must emit a non-trivial number of events

### C. Add a coverage constraint to threshold selection for weak targets

For continuation and any future weak-recall model, add a threshold acceptance floor such as:

- `predicted_events >= max(25, 0.15 * true_events)` on OOF

or

- `event_recall >= 0.10` on OOF

The exact floor can be tuned, but the next run should not accept a threshold that produces effectively zero live decisions.

### D. Add feature-count guardrails by target family

The `long_breakout v2` collapse from `106` features to `6` is too aggressive.

Use family-specific feature guardrails:

- breakout/reversal: floor at `48` selected features
- continuation: keep in the `16-64` range until coverage is fixed

### E. Separate "ranking quality" from "deployment quality"

A model can have good test AP and still be unusable as a deployed classifier.

For the next round, every target should be judged on both:

- ranking metrics: AP, ROC AUC, Brier
- thresholded event metrics: precision, recall, F0.5, predicted event count

---

## Model-by-Model Plan

## `short_breakout`

### Current read

- Keep `v1`.
- `v1` is stronger than `v2` on every saved metric:
  - CV AP: `0.9628` vs `0.9250`
  - CV event F0.5: `0.9327` vs `0.8922`
  - Test AP: `0.9950` vs `0.9871`
  - Test event F0.5: `0.9736` vs `0.9457`
  - Test recall: `0.8857` vs `0.7910`

### Recommendation

- Freeze `v1` as the active short-breakout artifact.
- Do not spend the next cycle on a full replacement attempt unless we are explicitly running a challenger check.

### Next action

- Optional challenger run only.
- Re-run the `v1` parameter neighborhood under the `v2` 3-fold geometry to verify the advantage survives the stricter CV setup.

### Schedule choice

- Do not make `short_breakout` the first consumer of a brand-new schedule idea.
- If a challenger run is executed, use the proven full TCN schedule from `ote_full_tcn_v2` unchanged:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This target is already excellent, so the goal is schedule verification, not aggressive exploration.

### Next-run target settings

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: `20-24`
- selected features: `80-120`
- layers: `2`
- learning rate: `0.0008-0.0015`
- focal alpha: `0.72-0.78`
- focal gamma: `3.0-3.4`
- hard negative radius: `1`
- hard negative multiplier: `2.2-2.6`
- calibration: compare `platt` vs `isotonic`

### Acceptance gate

Only replace `v1` if the challenger:

- matches or beats `0.9736` test event F0.5,
- keeps recall within `3` percentage points of `v1`,
- and does not degrade AP.

---

## `long_breakout`

### Current read

- Keep `v1`.
- Retune `v2`.
- `v2` regressed badly:
  - feature count fell from `106` to `6`
  - test predicted events fell from `280` to `7`
  - test event F0.5 fell from `0.7864` to `0.0513`

### Likely issue

`v2` over-compressed feature selection and ended up with a model that ranks okay but is far too conservative at deployment threshold.

### Recommendation

- Do not keep `v2` as-is.
- Retrain from the `v1` neighborhood, not the `v2` neighborhood.

### Next action

- Full retune and retrain.

### Schedule choice

- Use the full proven TCN schedule from `ote_full_tcn_v2` as the default retrain schedule:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This target is the clearest example that the shorter multi-family run shape is not enough.
- Because `v2` collapsed to `7` test events, this retrain should keep the full tail phase intact rather than shortening it.

### Next-run target settings

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: `20-28`
- selected features: `64-120`
- hard floor: reject trials with fewer than `32` selected features
- layers: `3`
- learning rate: `0.0035-0.0050`
- focal alpha: `0.70-0.78`
- focal gamma: `2.7-3.1`
- hard negative radius: `1-2`
- hard negative multiplier: `1.8-2.3`
- calibration: `platt` and `isotonic` comparison

### Additional checks

- Review feature-ranking output for why the selected set collapsed to `6`.
- Compare raw OOF score percentiles and test score percentiles to see whether the issue is ranking drift, calibration drift, or both.

### Acceptance gate

The next candidate should not be considered healthy unless it:

- clears `0.70` test event F0.5,
- emits materially more than `7` test events,
- and keeps recall above `0.20`.

---

## `long_reversal`

### Current read

- Keep `v1` active.
- Keep `v2` as challenger only.
- `v2` slightly improved ranking metrics:
  - test AP: `0.6995` vs `0.6880`
- but `v1` still wins on deployed event behavior:
  - test event F0.5: `0.6951` vs `0.6753`
  - test recall: `0.3187` vs `0.2989`

### Likely issue

This is close enough that the first fix should be recalibration and threshold re-search, not immediate architecture change.

### Recommendation

- Try recalibration-first on `v2`.
- Retrain only if recalibration does not recover thresholded performance.

### Next action

- Recalibration study on the saved `v2` OOF/test predictions.

### Schedule choice

- First pass: no retrain, recalibration only.
- If recalibration fails, retrain with the full proven OTE schedule:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This target is close enough that we should not widen the search space until probability scaling and thresholding have been re-checked.

### Recalibration plan

- Compare:
  - raw probabilities
  - `platt`
  - `isotonic`
- Re-search thresholds in the `0.45-0.70` range
- Prefer the version that improves event F0.5 without losing AP

### If retraining is needed

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: keep at `24`
- selected features: `70-95`
- layers: `3`
- learning rate: `0.0040-0.0050`
- focal alpha: `0.70-0.78`
- focal gamma: `2.6-3.0`
- hard negative radius: `1`
- hard negative multiplier: `1.9-2.2`
- calibration: `platt` vs `isotonic`

### Acceptance gate

Promote a new `long_reversal` only if it at least matches `v1` on test event F0.5, or slightly trails it while clearly improving AP and recall together.

---

## `short_reversal`

### Current read

- Keep `v1`.
- Retune `v2`.
- `v2` became too conservative:
  - test AP: `0.5257` vs `0.6435`
  - test event F0.5: `0.5229` vs `0.7644`
  - test predicted events: `82` vs `190`
  - test recall: `0.1831` vs `0.4188`

### Likely issue

Relative to `v1`, the `v2` neighborhood moved toward:

- shorter window
- more aggressive focal setup
- softer hard-negative multiplier

That combination appears to have hurt coverage and recall.

### Recommendation

- Retune from the `v1` neighborhood.

### Next action

- Full retune and retrain.

### Schedule choice

- Use the full proven OTE schedule as the starting schedule:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- Keep the schedule stable and move the search back toward the `v1` parameter neighborhood.
- This target does not need a continuation-specific schedule; it needs more room plus less aggressive parameter drift.

### Next-run target settings

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: `24-28`
- selected features: `80-100`
- layers: `3`
- learning rate: `0.0040-0.0050`
- focal alpha: `0.70-0.75`
- focal gamma: `2.8-3.2`
- hard negative radius: `1`
- hard negative multiplier: `1.9-2.2`
- calibration: `platt` vs `isotonic`
- threshold search emphasis: recover recall, not just precision

### Acceptance gate

Do not accept a new version unless it restores:

- test event F0.5 above `0.70`
- recall above `0.35`

---

## `long_continuation_pullback`

### Current read

- Keep neither `v1` nor `v2` active.
- Both versions emit `0` test events.
- Both versions have test event F0.5 of `0.0`.
- `v2` also expanded feature count from `19` to `142` while still failing on coverage.

### Likely issue

This is both a modeling problem and a pipeline problem:

- continuation calibration is disabled,
- thresholding is being done on uncalibrated probabilities,
- the target is likely too precision-biased at the current focal/threshold combination,
- and the current model family is not producing stable actionable probabilities.

### Recommendation

- Do not choose between `v1` and `v2` for deployment.
- Start a continuation recovery track.

### Next action

1. Run a recalibration-only study on saved artifacts.
2. Then run a smaller, more disciplined retrain.

### Schedule choice

- First pass: no retrain, recalibration only.
- If retraining is still needed after recalibration, use a continuation-specific long-tail variant instead of the raw breakout/reversal template:
  - `40` epochs
  - `4/14/12/10` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- Reason:
  - continuation needs more low-LR refinement once coverage is restored,
  - and this family should bias more of the budget toward fine and tail phases than toward the high-LR middle.

### Recalibration plan

Using saved OOF predictions from both `v1` and `v2`, test:

- raw
- `platt`
- `isotonic`

Then re-search thresholds with a coverage floor.

If either existing artifact starts emitting sensible events after recalibration, use that as the next training baseline.

### Next-run target settings

- schedule: continuation variant `40` epochs with `4/14/12/10`
- window: `32-48`
- selected features: `16-48`
- layers: prefer `3`; only use `4` if `3` clearly underfits
- learning rate: `0.0025-0.0040`
- focal alpha: `0.68-0.76`
- focal gamma: `1.5-2.5`
- hard negative radius: `2-4`
- hard negative multiplier: `1.0-1.4`
- calibration: test `platt` and `isotonic`

### Why these settings

- Longer windows fit continuation better than short breakout-style windows.
- Smaller feature sets reduce noisy feature sprawl.
- Lower gamma should make the loss less precision-only and more recall-tolerant.
- Moderate hard-negative settings should avoid over-penalizing near-miss continuation setups.

### Acceptance gate

The next continuation candidate must clear these basic gates before any champion discussion:

- non-zero held-out predicted events
- OOF threshold with coverage floor satisfied
- test event F0.5 materially above `0.0`

---

## `short_continuation_pullback`

### Current read

- Keep neither version active.
- Use `v2` as the research base.
- `v2` improved ranking quality:
  - test AP: `0.4188` vs `0.1522`
- but thresholded behavior is still unusable:
  - test event F0.5: `0.0478`
  - test predicted events: `3`

### Likely issue

The model is learning some ranking signal, but thresholding and probability scaling are not producing usable decision coverage.

### Recommendation

- Use `v2` as the starting point for continuation recovery.
- Fix calibration and coverage first.

### Next action

1. Recalibration-only study on saved `v2` predictions.
2. Then retrain with tighter feature and threshold guardrails.

### Schedule choice

- First pass: no retrain, recalibration only.
- If retraining is still required, use the same continuation-specific long-tail variant:
  - `40` epochs
  - `4/14/12/10` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This keeps the late low-LR behavior that worked in `ote_full_tcn_v2`, but pushes even more budget into fine and tail for a weaker target family.

### Next-run target settings

- schedule: continuation variant `40` epochs with `4/14/12/10`
- window: `20-32`
- selected features: `24-64`
- layers: prefer `3`
- learning rate: `0.0015-0.0030`
- focal alpha: `0.72-0.82`
- focal gamma: `1.4-2.2`
- hard negative radius: `2-3`
- hard negative multiplier: `1.0-1.3`
- calibration: `platt` vs `isotonic`

### Why these settings

- `v2` already improved AP, so keep its directionally better signal base.
- A smaller feature budget should make the signal less noisy.
- Lower-to-moderate gamma should help keep recall from collapsing.
- Re-enabling calibration is the most important single change.

### Acceptance gate

Before we compare this target to any older artifact, the next version must first show:

- non-zero held-out predicted events
- event F0.5 above `0.25`
- stable OOF threshold behavior with a coverage floor

---

## Recommended Run Order

### Wave 1: low-cost recalibration studies

Run recalibration-only comparisons on saved predictions for:

1. `long_reversal v2`
2. `long_continuation_pullback v1`
3. `long_continuation_pullback v2`
4. `short_continuation_pullback v1`
5. `short_continuation_pullback v2`

Goal:

- see whether threshold failure is mostly a calibration problem before paying for full retraining

### Wave 2: highest-value retrains

Retrain in this order:

1. `long_breakout`
2. `short_reversal`
3. `long_reversal` only if Wave 1 recalibration does not recover `v2`

### Wave 3: continuation recovery

Retrain:

1. `long_continuation_pullback`
2. `short_continuation_pullback`

These should be treated as research targets, not deployment targets, until they emit stable non-zero held-out events.

### Wave 4: optional verification challenger

Run one bounded `short_breakout` challenger using the `v1` parameter neighborhood under the `v2` CV geometry.

---

## Success Criteria For The Next Round

The next round should be considered successful if it delivers all of the following:

1. At least one retrained non-continuation target that no longer peaks at the final epoch.
2. `long_breakout` recovers usable thresholded behavior and no longer collapses to single-digit event count.
3. `short_reversal` recovers recall and moves back above `0.70` test event F0.5.
4. At least one continuation target emits non-zero held-out events after recalibration or retraining.
5. The continuation path is no longer forced into `calibration = none` without an explicit comparison.

---

## Final Recommendation Snapshot

- Keep:
  - `short_breakout v1`
  - `long_breakout v1`
  - `long_reversal v1`
  - `short_reversal v1`

- Keep as challenger only:
  - `long_reversal v2`
  - `short_breakout v2`

- Retune aggressively:
  - `long_breakout v2`
  - `short_reversal v2`
  - `long_continuation_pullback v1`
  - `long_continuation_pullback v2`
  - `short_continuation_pullback v1`
  - `short_continuation_pullback v2`

- Highest-priority pipeline fix:
  - allow continuation calibration experiments and add a threshold coverage floor

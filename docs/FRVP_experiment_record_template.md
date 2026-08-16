# FRVP Experiment Record Template

Use this template whenever an FRVP experiment changes a branch decision, a live or shadow contract, a saved research control, or the backlog order.

## Required Fields

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

## Copy/Paste Skeleton

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

## Logging Rule

If an experiment produces a new checkpoint, closes a research path, changes a live or shadow contract, or materially changes the backlog order, add it to `docs/FRVP_experiment_journal.md` immediately instead of leaving the reasoning only in artifact folders.

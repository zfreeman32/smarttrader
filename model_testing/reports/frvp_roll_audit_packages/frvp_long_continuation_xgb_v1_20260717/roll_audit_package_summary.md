# FRVP Roll Audit Package

- Generated at UTC: `2026-07-17T05:00:13.103967+00:00`
- Model artifact: `models\frvp_es_primary_xgb_refresh_20260701\long_frvp_continuation`
- Target: `long_frvp_continuation`

## Roll Gate Summary

- Gate 7 overall pass: `True`
- Roll reconstruction all checks passed: `True`
- Zero usable roll-span events: `True`
- Fold embargo audit pass: `True`
- Continuity/profile smoke tests passed: `True`

## Event Audit

- Reported roll-span exclusions: `49`
- Usable roll-span rows in saved event file: `0`
- Reported roll-bracket events: `3010`
- Usable roll-bracket rows in saved event file: `1914`

## Fold Boundary Audit

- Resolved purge bars in training artifact: `132`
- Minimum raw-source gap across folds: `10238` bars / `1290.75` hours
- Folds with a roll boundary inside the train-to-validation gap: `11`
- Required gap rule used here: `288` bars minimum, or `576` when the gap crosses a roll boundary

## Roll Reconstruction

- Schedule roll count: `38`
- Flagged seam boundaries: `12`

## Linked Placebo Readout

- Placebo pass: `True`
- Real OOF AP: `0.737051`
- Shuffled mean OOF AP: `0.500226`
- Placebo gap: `0.236825`

# FRVP Deferred Research Audit

Generated: `2026-07-28T07:41:51.887747+00:00`

This audit revisits the deferred FRVP research cluster using only saved repo artifacts.

## Repo-State Snapshot

- Saved Phase 3 setup types present: `1, 2, 3, 4, 5, 6`
- Setup 6b implemented in saved events: `False`
- Gamma-context columns present in Phase 2 feature metadata: `False`
- Gamma columns found: none

## Pooled Vs. Per-Setup Read

- `long_continuation_v3_baseline`: dominant `Setup 5` share `0.639`, best `Setup 2` expectancy `52.51`, worst `Setup 3` expectancy `-0.81`, split recommended `True`.
- `long_reversal_fullspan_baseline`: dominant `Setup 1` share `0.664`, best `Setup 4` expectancy `77.58`, worst `Setup 1` expectancy `36.75`, split recommended `True`.
- `long_reversal_recent2y_baseline`: dominant `Setup 1` share `0.716`, best `Setup 6` expectancy `108.43`, worst `Setup 4` expectancy `18.79`, split recommended `True`.
- `long_reversal_recent2y_operational`: dominant `Setup 1` share `0.667`, best `Setup 6` expectancy `108.43`, worst `Setup 4` expectancy `18.79`, split recommended `True`.

## Branch Snapshot

- `long_continuation_v3_baseline`: trades `512`, net `7001.20`, expectancy `13.67`, roll-bracket share `0.082`, directional naked-VPOC share `0.016`.
- `long_reversal_fullspan_baseline`: trades `140`, net `6027.00`, expectancy `43.05`, roll-bracket share `0.071`, directional naked-VPOC share `0.057`.
- `long_reversal_recent2y_baseline`: trades `74`, net `4193.90`, expectancy `56.67`, roll-bracket share `0.068`, directional naked-VPOC share `0.054`.
- `long_reversal_recent2y_operational`: trades `63`, net `3677.05`, expectancy `58.37`, roll-bracket share `0.063`, directional naked-VPOC share `0.016`.

## Setup 4 Standalone Retest

- Usable Setup 4 `long` events: `380` with TP rate `0.276` and quality `0.648`.
- Usable Setup 4 `short` events: `412` with TP rate `0.260` and quality `0.599`.
- `long_reversal_fullspan_baseline` Setup 4 selected trades: `13` with net `1008.55` and expectancy `77.58`.
- `long_reversal_recent2y_baseline` Setup 4 selected trades: `9` with net `169.15` and expectancy `18.79`.
- `long_reversal_recent2y_operational` Setup 4 selected trades: `9` with net `169.15` and expectancy `18.79`.

## Setup 6b Readiness Proxy

- `frvp_continuation`: directional naked-VPOC-in-reach share `0.040` (548/13666), TP in-reach `0.473` vs. not-in-reach `0.485`.
- `frvp_reversal`: directional naked-VPOC-in-reach share `0.033` (184/5534), TP in-reach `0.418` vs. not-in-reach `0.470`.

## Recommended Order

1. Per-setup-family audit/training lane before any new pooled-family optimization.
1. Standalone Setup 4 retest once the per-setup path exists.
1. Roll translation vs reset A/B on the control branches that still show roll-bracket drag, especially continuation and the full-span reversal control.
1. Setup 6b only after the naked-VPOC target path is promoted from context to event logic.
1. Gamma-context features only after a concrete options-data contract exists.

# ICT Long Paper-Signal Decision — 2026-08-13

## Decision

- Select `ict_long_meta_xgb_v1` as the sole controlled ICT paper-signal leader.
- Package the exact accepted walk-forward policy: global threshold `0.40`, no regime thresholds, and abstention disabled.
- Keep `ict_long_reversal_xgb_v1` at `candidate`/shadow status as a non-additive challenger.
- Do not stack meta and reversal or allocate a second position to the same event.
- Do not authorize broker orders. In this repository, `active` enables normal `emit` decisions and configured notifications only.
- Keep the four-week confirmation clock at `not_started` until the readiness audit passes.

## Evidence

The canonical July 30 leakage-safe walk-forward results remain valid:

| Model | Trades | Net ticks | Sharpe | Approx. DSR | Max DD | WFE | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ict_long_meta_xgb_v1` | 1,555 | 7,461.25 | 1.586 | 1.068 | 6.40% | 1.997 | 8/8 |
| `ict_long_reversal_xgb_v1` | 1,275 | 5,471.25 | 1.269 | 1.023 | 8.40% | 3.013 | 8/8 |

The two branches are not additive when matched on `(entry_datetime, source_row_idx)`:

- 612 exact shared trades produced `+5,758.20` ticks in each branch.
- Shared trades account for `77.17%` of meta net PnL and `105.24%` of reversal net PnL.
- Reversal's 663 unique trades produced `-286.95` ticks.

That makes meta the stronger single-slot leader and reversal useful as a shadow comparator, not a second concurrent allocation.

## Implemented Contract

The immutable bundle is `ict_es_paper_signal_20260813`:

- Registry: `models/ict_es_paper_signal_registry_20260813.json`
- Policies: `ote_live/policy_artifacts/ict_es_paper_signal_20260813/`
- Runtime manifests: `ote_live/runtime_manifests/ict_es_paper_signal_20260813/`
- Lifecycle summary: `model_testing/reports/ict_paper_signal_bundles/ict_es_paper_signal_20260813/run_summary.json`
- Saved validation: `ote_live/runtime_manifests/ict_es_paper_signal_20260813/paper_signal_validation_summary.json`

The dashboard and shared ES collector defaults now point to this bundle. The local IBKR configuration was moved from live mode/port `4001` to paper mode/port `4002`, `ES_LIVE_ALL_MODELS_ACTIVE=false`, and the fail-closed launch switch remains `ICT_PAPER_SIGNAL_TRIAL_ENABLED=false`.

The frozen July 30 shadow builder and artifacts were not modified. The new builder refuses to mutate an existing dated artifact unless regenerated content is byte-identical.

## Verification

- Bundle construction and lifecycle tests passed.
- Runtime status semantics and notification tests confirm candidate decisions remain shadow-only and only `emit` decisions notify.
- The live abstention path now materializes regime/session context before applying high-stress or off-hours abstains; focused regression tests passed.
- Direct model-artifact prediction replay passed for both selected long models at row offsets `0`, `100`, and `300`: 150 rows per model, zero raw-probability, calibrated-probability, or predicted-label mismatches.
- Historical live-feature replay matches 405 of 414 selected-feature cells across the final three frozen bars after correcting the XGBoost missing-value contract. The remaining 9 raw-value mismatches do not invalidate the active long-meta prediction contract.

The residual mismatch is recorded, not hidden behind a tolerance:

- the live XGBoost feature and model paths now preserve the frozen training-time nulls, removing all 28 holiday and sparse-event mismatches;
- the remaining nine differences are exactly three coordinate features across three bars: `ict_nearest_bull_fvg_id`, `ict_latest_bear_displacement_id`, and `ict_nearest_bull_fvg_inversion_index`;
- long-meta selects the first two fields, but its frozen Booster has zero splits and zero gain across all eight lag slots for both; raw and calibrated predictions are bit-identical between frozen-reference and live-window inputs on all three replay bars;
- only candidate/shadow reversal selects the inversion index. It contributes one of 181 Booster splits and 0.4608% of total gain; the live coordinate changes calibrated probability by 0.00394 to 0.00495 in the three-bar sample, although all remain below 0.40;
- reversal therefore remains blocked from promotion until retrained without raw positional identifiers. A persisted offset would reproduce an arbitrary chronology proxy, not improve the scientific contract.

The shared ES collector is also not ready: its last heartbeat is from `2026-08-04`, marked unhealthy and source-stale. No collector process was started during this decision because the paper TWS/IB Gateway session is not available for verification.

## Current Lifecycle

The promotion decision is executed in configuration, but launch is fail-closed:

- lifecycle: `authorized_pending_paper_feed_not_started`
- sole active manifest model: `ict_long_meta_xgb_v1`
- reversal status: `candidate`
- trial start authorized: `true`, contingent on the readiness-bound healthy paper-feed handoff
- broker order submission authorized: `false`
- confirmation start: `null`
- minimum confirmation window after an honest start: 28 calendar days

The readiness checker must report `ready_to_start` before enabling the launch switch. Active long-meta prediction parity is accepted with a documented dead-input warning, and the dedicated event-markout ledger is ready. Trial start is internally authorized but not active; launch remains blocked by the disabled final switch and stale/unhealthy collector health.

## Next Required Work

1. Keep reversal shadow-only and retrain it without raw `*_id`/`*_index` fields before any future promotion. Do not qualify it with a coordinate seed offset.
2. Start an authenticated paper TWS/IB Gateway session with socket API and read-only mode, then run the one-cycle ICT-excluded bootstrap to establish a fresh clean handoff snapshot. Port `4002` was not listening during this audit.
3. Rerun `scripts/audit_ict_paper_signal_readiness.py --preflight --allow-clean-stopped-handoff`. Only after the paper-feed blockers are gone, set `ICT_PAPER_SIGNAL_TRIAL_ENABLED=true`, launch with `--include-ict --allow-ict-clean-handoff`, and record the first healthy running heartbeat as the actual confirmation start.

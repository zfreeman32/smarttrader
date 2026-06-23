# 8-Model OTE Live Promotion Plan

## Summary

- Final live runtime target: 8 models total, grouped as 4 long and 4 short, with `meta` as the third family and no continuation models in scope.
- Final `active` emitters: `long_ote_tcn_v2_candidate`, `short_ote_tcn_v2_candidate`, `long_reversal_tcn_champion`, `short_reversal_xgb_v1`.
- Final `candidate` shadow models: `long_breakout_tcn_champion`, `short_breakout_tcn_champion`, `long_ote_meta_tcn_champion`, `short_ote_meta_tcn_champion`.
- Research-only models kept out of the 8-model live runtime: `short_reversal_tcn_champion`, `long_breakout_xgb_v1`, legacy OTE v1/XGB/LSTM variants.

## Interface Changes

- Keep `models/ote_model_registry_live_multifamily.json` as the single live 8-model registry for reevaluation, packaging, and runtime export.
- Add `models/ote_model_registry_live_multifamily.json` as the exact 8-model live registry used by `ote_live`.
- Make live runtime semantics depend on existing registry `status`:
  - `active` = live emitter and alert-eligible
  - `candidate` = shadow-only, persisted to DB/dashboard, no operator alerts
  - `deprecated` = excluded from live export/runtime
- Keep `recommended_primary_model_id` per direction, but lock it to the OTE v2 models so the dashboard and legacy fallbacks stay stable.
- Generalize policy packaging and manifest export defaults so they no longer assume only the two OTE TCN v2 models exist.

## Implementation Steps

1. Canonicalize the model set and naming. Use `long_ote_tcn_v2_candidate` and `short_ote_tcn_v2_candidate` as the canonical live OTE IDs, stop using the older active-registry aliases in runtime config, and build the two new multifamily registries around the exact 8 live targets plus the 2 research-only comparison models.

2. Do model-side work before reevaluation. Leave `long_ote_tcn_v2_candidate`, `short_ote_tcn_v2_candidate`, `long_reversal_tcn_champion`, and `short_reversal_xgb_v1` unchanged at the artifact level. Repackage `long_ote_meta_tcn_champion` without retraining first. Retrain `short_ote_meta_tcn_champion`. Run the `long_breakout_tcn_champion` 48-epoch rescue pass. Treat `short_breakout_tcn_champion` as a pipeline repair first, then retrain it after fixing target/trade-conversion assumptions. Write all repaired artifacts to new date-stamped directories and only repoint registry `artifact_path` after reevaluation passes.

3. Rerun `model_testing` from the new multifamily candidate registry in one consistent pass. Run regime slices first, then threshold policy search, then walk-forward policy backtests. Use stable output roots such as `model_testing/reports/ote_regime_slices/multifamily_live_v1`, `model_testing/reports/ote_threshold_policies/multifamily_live_v1`, and `model_testing/reports/ote_policy_backtests/multifamily_live_v1` so live packaging can reference non-date-fragile paths.

4. Apply promotion rules after the unified rerun. Promote to `status=active` only if post-cost profitability, WFE, profitable-quarter share, composite expectancy share, and trade concentration gates pass. Keep `long_ote_tcn_v2_candidate` and `short_ote_tcn_v2_candidate` active on explicit legacy override if the rerun still shows they are profitable and the drawdown gate remains the only failing flag. Keep all other failing models at `status=candidate` so they load shadow-only.

5. Package live policy artifacts for all 8 models from the new threshold-search run, not from OTE-only defaults or direction fallbacks. Every live-target model must get a real `live_policy.json`, including threshold and abstain metadata, under `ote_live/policy_artifacts/<model_id>/`.

6. Export fresh long and short runtime manifests from `models/ote_model_registry_live_multifamily.json`. The long manifest must contain exactly `long_ote_tcn_v2_candidate`, `long_reversal_tcn_champion`, `long_breakout_tcn_champion`, and `long_ote_meta_tcn_champion`. The short manifest must contain exactly `short_ote_tcn_v2_candidate`, `short_reversal_xgb_v1`, `short_breakout_tcn_champion`, and `short_ote_meta_tcn_champion`.

7. Refactor the live collector binding logic so one run can mix active emitters and shadow models at the same time. The current primary-vs-rest logic is not enough for this mission. After the change, all `active` models emit normally, all `candidate` models run with `shadow_mode=True`, and `all_models_active` remains only as an override for special testing.

8. Refactor the dashboard from 2 primary-only inspection panels to an 8-card layout grouped Long and Short. Each card should show model id, family, status badge, latest decision, latest probability, active threshold, last signal timestamp, and a link or embedded image for the latest stored chart when available. Keep the existing all-signals price chart and recent-signal history, but drive them from the new 8-model manifests.

9. Update live-stack defaults and docs so `run_live_stack`, manifest export, and policy packaging all point at the new multifamily live registry and the stable multifamily report roots. The repo should no longer boot the live app from the old v1/v2-only candidate registry.

## Test Plan

- Extend `pytest` coverage around `ote_live` so status-driven mixed loading is verified: active models emit, candidate models shadow, deprecated models are skipped, and the exported manifests still validate.
- Keep `tests/test_ote_live_runtime_manifests.py` and `tests/test_ote_live_signal_runtime_integration.py` green, then add new cases for the 8-model long/short manifests and the mixed active-shadow collector binding path.
- Run feature parity replay on the 4 active emitters first, then on the 4 candidate shadow models after their manifests are exported.
- Run packaging and manifest export in dry-run form, then run one short live collector cycle and verify that only active models send alerts while shadow models persist `shadow` decisions for the dashboard.
- Final acceptance check: the runtime manifests contain exactly 4 long and 4 short models, the dashboard resolves all 8 cards, and the collector starts without falling back to old OTE-only defaults.

## Assumptions

- `mera` means the existing `meta` model family.
- Continuation models are fully out of scope for this mission.
- Stable model IDs are preserved; retrains change artifact directories, not model IDs.
- `short_reversal_tcn_champion` stays as a research challenger only, and `long_breakout_xgb_v1` stays as a benchmark/deprecated recovery reference only.
- The user wants 8 models visible in the live app/runtime even if some remain shadow-only after reevaluation.

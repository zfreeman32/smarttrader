# Model Testing

Post-training evaluation for every model family. It holds two stacks: the OTE-era policy stack, which turns saved
predictions into threshold/abstain policies and walk-forward backtests, and the B1 outcome stack, which evaluates signals
under frozen, causal trading contracts (entries, barriers, costs, ledgers).

## Policy stack (prediction -> policy -> backtest)

- `ote_prediction_joiner.py`: reattaches exported prediction rows to the source feature dataset (uses prepared split
  metadata and `source_row_idx` when available).
- `ote_regime_labeler.py`: deterministic market-state labels (`trend_regime`, `vol_regime`, `session_regime`,
  `stress_regime`, `composite_regime`).
- `ote_regime_slices.py`: per-regime evaluation tables, average-precision summaries, bootstrap intervals, winner tables.
- `ote_threshold_policy.py`, `policy_selection.py`: threshold and regime-aware policy search and selection.
- `ote_abstain_policy.py`: hard abstain filters (stress/session, expected move, probability quantiles, cooldowns).
- `ote_policy_backtest.py`, `ote_policy_metrics.py`, `evaluation_costs.py`, `promotion_gates.py`: walk-forward trade
  simulation, frictions, equity/drawdown reporting and promotion gates.
- `ote_breakout_event_metadata.py`: breakout event metadata used by the breakout targets.

Entry points (in `scripts/`): `run_ote_regime_slice_report.py`, `run_ote_threshold_policy_search.py`,
`run_ote_policy_backtest.py`, `run_outcome_evaluation.py`, and the per-family `run_frvp_post_training_eval.ps1` /
`run_ict_post_training_eval.ps1`. Reports go to `reports/` (regime slices, threshold policies, policy backtests per
family).

## Outcome stack (B1 contracts, sections 3-4)

- `outcome_contracts.py`, `outcome_engine.py`, `outcome_atr.py`: the frozen B1 contracts
  (`configs/outcome_contracts/b1_v*.json`) and the causal outcome engine with separate position accounting.
- `outcome_scope.py`, `outcome_routes.py`: research-scope membership and the model/setup-to-contract route bindings.
- `outcome_surfaces.py`: the four section 3 surfaces and the independent / instrument-capacity ledgers.
- `outcome_diagnostics.py`, `outcome_artifacts.py`, `outcome_lineage.py`, `outcome_compatibility.py`: model/setup
  diagnostics, immutable replay artifacts, lineage adapters and hash-addressed access to pre-B1 evaluators.
- `b3_comparison.py`: predeclared B3 comparison plans and full-policy result guards.
- `b1_fast_outcomes.py`: vectorized B1 outcomes for research censuses (verified equal to the engine).
- `artifact_lineage_audit.py`: B2 artifact inventory and the `digest` helper used by the B2 data/retraining scripts.

`examples/outcome_b1/` holds a small worked example. Design and results: `docs/B1_outcome_contracts.md`,
`docs/section3_simulation_rules.md` and `docs/section4_progress.md`.

## Inputs from training

The trainer (`model_training/ote_training`) writes `cv_fold_manifest.{csv,json}`, `oof_predictions.csv`,
`test_predictions.csv` and `training_summary.json`; those are the inputs to the policy stack. Historical leaderboards and
backtest charts are collected in `docs/model_backtest_charts.md`.

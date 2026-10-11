# Synthetic B1 four-surface replay

These prices, labels, identities, policies and evidence are invented fixtures,
not a strategy recommendation or approved B3 comparison. The expected schedule
is independent of the OHLC file; removing an OHLC row must not remove its
expected timestamp.

Run from the repository root into new output directories:

```powershell
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --signals model_testing/examples/outcome_b1/signals.jsonl --bars model_testing/examples/outcome_b1/bars.csv --schedule model_testing/examples/outcome_b1/schedule.csv --calendar model_testing/examples/outcome_b1/calendar.json --position-contract model_testing/examples/outcome_b1/position_contract.json --output-dir tmp/b1_example
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --replay-run tmp/b1_example --output-dir tmp/b1_example_replay
ote_venv/Scripts/python.exe scripts/run_ote_policy_backtest.py --b1-input-manifest model_testing/examples/outcome_b1/input_manifest.json --output-root tmp/b1_phase7
```

Under the current scope, all three original ES fixture signals are **pooled**
predictions and therefore diagnostics-only: paper NOT_EVALUATED, no position,
and independent forward returns. Their old event names do not override scope.
Earlier pre-scope descriptions of target hits and position P&L no longer apply.
The original input bytes are retained for archived replay.

The standard ES contract rounds futures barriers outward to the tick grid;
the 0.08-point analytic target is 0.25 points above entry. No model was loaded,
trained, or relabeled. Both manifests keep scored collection disabled.

See [B1 design](../../../docs/B1_outcome_contracts.md#b1_outcome_contract_design_20260916) for the
input evidence schema, exclusions, costs, and migration boundary.

## Independent OTE example

`ote_input_manifest.json` supplies entirely synthetic EURUSD OHLC, an independent
schedule, four named OTE forecasts and one pooled forecast. It derives causal
1h/30m ATR from the included warmup history rather than importing label ATR.

```powershell
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --input-manifest model_testing/examples/outcome_b1/ote_input_manifest.json --output-dir tmp/b1_ote_example
ote_venv/Scripts/python.exe scripts/run_outcome_evaluation.py --replay-run tmp/b1_ote_example --output-dir tmp/b1_ote_example_replay
```

All four named forecasts reach their paper target. Only the first named signal
is admitted by the explicit one-position fixture policy; the other three remain
independent paper targets. The pooled forecast receives forward diagnostics
only. Reversal's training outcome is 0 while its paper outcome is TARGET; its
first-bar forward close move differs from its barrier exit. The simulated
position separately reports pip P&L, each cost and net P&L. Forward diagnostics
continue after every paper exit. The fixture contains no real model/setup audit
collection and explicitly reports missing context instead of inventing coverage.

# Synthetic A7 execution replay

These invented prices and identities exercise execution accounting. The draft
contract is diagnostic-only and is not a finalized B1/B3 contract.

Run from the repository root:

```powershell
ote_venv/Scripts/python.exe -m ote_live.scripts.replay_research_execution --contract ote_live/examples/research_execution/synthetic_contract.json --events ote_live/examples/research_execution/synthetic_events.jsonl --database tmp/es_a7_example/research.sqlite --report tmp/es_a7_example/report.json --run-id synthetic-a7 --started-at 2026-09-14T14:30:00+00:00
```

Expect six independent resolved records, one for each selection/cost case.
The decision at 14:35:01 enters at 14:40, then reaches its target during the
14:45 bar. Gross ticks are 8; net ticks are 6.1 at baseline costs or 4.2 at
doubled costs. The report explicitly disables scored collection.

Full semantics and limitations: [A7 evidence](../../../docs/live_app_change_journal.md#imported-a7).

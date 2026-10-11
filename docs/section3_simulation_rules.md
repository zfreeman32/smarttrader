# Section 3: exact trading and simulation rules (2026-10-02)

Consolidated record of section 3 ("Bind exact trading and simulation rules"): the progress log, the gap matrix and the final (v3) deterministic smoke replay. The two earlier smoke-replay passes are kept losslessly as unified diffs against v3 at the end.

Contents:
- [Section 3 progress — "Bind exact trading and simulation rules"](#section3_progress)
- [Section 3 gap matrix — 2026-10-02](#section3_gap_matrix_20261002)
- [Section 3 smoke replay - 2026-10-02](#section3_smoke_replay_20261002_v3)


<a id="section3_progress"></a>

## Section 3 progress — "Bind exact trading and simulation rules"

*Merged 2026-10-09 from `docs/section3_progress.md`.*

Last updated: 2026-10-02 (prompt 6: IDs 22 and 26, section 3 closed for all 16). Resume from this file alone.

### Status

| Item | Status | Evidence | Notes |
|---|---|---|---|
| B3v2 | done: published and locked as v3 | `configs/b3_b4_candidate_review_v3_20261002.json` (sha256 `7e5189cdd1543005e5035e940a168d3d22e233b3039aac899077a303b0a7dcc4`), lock `…v3_20261002.lock.json` (sha256 `211844470a589d4294462f2f907ecb764502520d3ac042770d07b1f0c004f67e`), `scripts/publish_b3_protocol_v3.py` | Numbered v3 because a B3 v2 was already locked on 2026-10-02 (D10). All eight v2 sections carried byte-equal; v1, v2 and their locks unchanged. D1-D10 approved as recommended on 2026-10-02. Draft input kept at `configs/b3_b4_candidate_review_v3_draft_20261002.json`. |
| R1 four surfaces | done in the shared module (synthetic fixtures); not yet run on the 16 models' real data | `model_testing/outcome_surfaces.py`, `tests/test_section3_surfaces_ledgers.py` (16 passed), gap matrix R1 rows | Schema `section3-surfaces-v1`. See "Prompt 3" below. |
| R2 ledgers | done (independent, instrument-capacity with D9, reporting guard) | same | D9 landed inside b1-v5 as `simulate_positions(priority=...)`. |
| R3 execution enforcement | done in the shared simulator (b1-v5); EURUSD calendar pin still open (D6) | `tests/test_section3_execution_rules.py` (56 passed), gap matrix R3 rows | Every R3 row is `exists` for both streams except planned closures on EURUSD, which stays `partial` only for the missing calendar pin. See "Prompt 2" below. |
| R4 smoke replay | done: all 16 passed (11 ES, 5 EURUSD) | `docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v3`, `docs/audits/section3_smoke_<model_id>_20261002.json` (ES), `..._v3.json` (EURUSD) | See "Prompt 4" to "Prompt 6" below. |
| R5 session coverage | done for the simulator and shared module | same | Three guard tests. Runtime `ote_live` policy gates are section 5. |

### Inputs (read-only, never edit)

- Roster: `configs/broad_full_session_shadow_route_bindings_v2.json` + lock (16 models, 59 routes: 45 ES, 14 EURUSD).
- Freeze: `configs/b4_broad_full_session_shadow_freeze_v2.json` (thresholds, artifact roots; all 16 `full_policy` = `not_defined`).
- Selection order: 2, 10, 12, 13, 15, 16, 22, 26, 3, 19, 21, 25, 5, 23, 18, 14 (`configs/broad_full_session_shadow_selection_v1.json`).
- Catalogue: `configs/outcome_contracts/b1_v3.json`; scope `b2_research_scope_v1.json`.
- Engine b1-v4: `model_testing/outcome_engine.py`, `outcome_artifacts.py`, `outcome_routes.py`, `outcome_diagnostics.py`.
- Harness to generalize: `scripts/review_b1_candidate_outcomes.py` (hardcoded to candidates 7 and 17, ES only).
- Held-out predictions per model: `<artifact_root>/test_predictions.csv` under the B4 v2 artifact roots.

### Approved decisions (2026-10-02, as recommended)

- D1 capacity feed: full policy where exact, else setup_threshold, labeled per entry (all 16 are setup_threshold today).
- D2 priority: earliest known_at, then frozen selection order, then opportunity key.
- D3 blocking: an open or unresolved position blocks every entry on that instrument until it closes.
- D4 independent ledgers: one position at a time, one unit, no opposite side, zero cooldown.
- D5 smoke window: whole trading weeks, at most 4 per model, earliest-week tie-break, held-out period only.
- D6 EURUSD smoke stays `not_run` until a versioned EURUSD 5m calendar is pinned.
- D7 ES clock spellings are aliases of one clock; no behavior change.
- D8 all 16 full policies stay `not_defined` in section 3.
- D9 capacity ordering key goes into the shared simulator as b1-v5; b1-v4 stays reproducible.
- D10 published as v3.

### Validation (prompt 1b)

- `scripts/validate`-equivalent check ran inside the publisher (`build_b3_comparison_plan` on v3 with selection v1): 16 candidates, both streams.
- `tests/test_b3_comparison_plan.py` + `tests/test_outcome_route_bindings.py`: 15 passed. Needs `-p no:cacheprovider --basetemp <writable dir>`; the default pytest temp dir is access-denied on this machine.
- No test covers the v3 file itself yet.

### What was not done

- No engine or runtime code changed; no journal entries.
- The EURUSD bar source and calendar for a B1 tape are not located or pinned (D6).

### Prompt 2 (2026-10-02): R3 execution-rule enforcement

B3 lock check: v2 and v3 are both locked; v3 is the operative protocol.

Engine is now `b1-v5` (`ENGINE_VERSION` in `model_testing/outcome_engine.py`, used by `outcome_artifacts.py`). Changes, all additive:

- `EvaluationTape(require_completion=True)`: a bar without `is_complete == True` is unusable, also when the column is absent. Default `False` keeps b1-v4 behavior. The flag is written to `calendar.json` and honored by replay. **The smoke harness must pass `require_completion=True`.**
- Open at path end: outcome keeps `DATA_GAP` / `paper_resolved=False` and gains `paper_mark_ts`, `paper_mark_price`, `paper_unrealized_price_movement`. The position row stays `UNRESOLVED` with `mark_ts`, `mark_price`, `unrealized_gross_pnl`; `gross_pnl`, `net_pnl` and costs stay null; capacity is kept.
- Cost cases: `COST_CASES = {base: 1, double_all_costs: 2}`; every position row has `cost_case="base"`; `reprice_positions(rows, "double_all_costs")` re-prices the identical admitted trades and refuses a non-base input.

Hashes (sha256):

- `model_testing/outcome_engine.py` `2f5a8871601023336d87e2a970146a14b2ace009d86c1e0b6660c46ba6d41674`
- `model_testing/outcome_artifacts.py` `dc61106bae46567eeb4ffb3a2503605adbf44ed338a1bb15a8ed840651f516d9`
- `tests/test_section3_execution_rules.py` `bec28b1c132ec598e27675ea73fd094090bfa2b54dc3eefab0da12be9317ae14`

b1-v4 reproducibility: the engine files are untracked in git. The b1-v4 source is the `engine_source/` folder inside each b1-v4 run (engine `cf0377a70b59…`, scope `bcc3a3d6d853…`, diagnostics `1ebb4d3fb5d2…`, identical across all 34 snapshots). Finding: the working tree had already diverged from that source on 2026-09-29 (pooled-routing scope hook in engine, scope and diagnostics) while still labeled b1-v4, so `replay_outcome_run` already refused those runs before this prompt. b1-v5 now names that state honestly.

Prior evidence affected (not rewritten; replay needs the archived source in an isolated checkout; no numeric change expected because defaults reproduce b1-v4 fills and P&L, only new columns appear):

- `artifacts/b1_candidate_outcome_review_20260929_v1` to `_v4` and `docs/B1_outcome_contracts.md#b1_candidate_outcome_review_20260929`
- `artifacts/b2_s3_backtest_20260927_v1`, `b2_s3_corrected_backtest_20260927_v1`, `b2_s3_full_policy_backtest_20260927_v2` and `docs/B2_S3_*`

No frozen B1 contract, cost contract, B3 protocol, route manifest or bundle changed.

Notes for later prompts:

- Gap-through rule: the frozen `exit_price` term is `barrier_or_adverse_stop_gap_open_timeout_close`. A stop gap fills at the gap open. A target gap fills at the target (no favorable improvement), not at the gap open. Tested as frozen; changing it would change a B1 contract.
- "Stale" in the historical simulator means the expected entry bar is absent: `DATA_GAP`, no forward search, recorded `no_trade_reason`. The 90 s / 30 min / 60 min wall-clock freshness limits live only in A7 and belong to section 5.
- The later-setup prohibition is tested at engine level (evidence dated after the decision is `INVALID`); pooled routing is covered by the existing route and scope tests.
- D9 (capacity ordering key) is not done. It can join b1-v5 as long as no b1-v5 evidence is published first; otherwise it is b1-v6.
- No v3 protocol test was added (not in the prompt 2 scope).
- Tests need `-p no:cacheprovider --basetemp <writable dir>` and the `ote_venv` interpreter (system Python has no pyarrow).

Test results: focused `tests/test_section3_execution_rules.py` 56 passed. Broader suite: full `tests/` run once with `ote_venv`: 1176 passed, 2 skipped, 20 failed. Two failures were b1-v4 version pins in `tests/test_outcome_contracts.py` and `tests/test_outcome_ote_contracts.py`, updated to b1-v5; the seven outcome/B3/route/scope/section-3 test files then pass (261 passed). The other 18 failures are in files that do not import the outcome modules (`test_backend_attribution`, `test_es_collection_baseline`, `test_ict_phase6_pipeline`, `test_ict_setup_prepared_materializer`, `test_ote_live_feature_parity`, `test_ote_live_feature_seed_state`, `test_ote_live_phase5_operator_tooling`, `test_ote_live_prediction_replay`, `test_ote_xgboost_pipeline`): 17 `FileNotFoundError` for missing data or temp files and one `OTETrainingConfig` `cv_max_folds` TypeError. They were not compared against a pre-change baseline and were left untouched. The full suite was not re-run after the pin update.

### Prompt 3 (2026-10-02): R1 surfaces, R2 ledgers, R5 session coverage

R3 gate: R3 is complete; the only open item is the EURUSD calendar pin, which is approved decision D6 and not an engine gap, so prompt 3 proceeded.

Shared module `model_testing/outcome_surfaces.py` (new; imported by the A7 ledger in `ote_live` and to be used by the prompt 4 harness):

- `build_stream_surfaces(stream, signals, tape, thresholds=, model_hashes=, selection_order=, protocol_sha256=, full_policy=)`: one stream per call. Resolves every signal through route bindings v2 (`resolve_model_setup_route`), evaluates it once on the research scope, then builds stores. Refuses a tape without `require_completion=True`, a model outside the stream's roster, a foreign unit and duplicate event IDs.
- Markout stores `mk.{stream}.{queue_id}.raw_score_gross_forward_response` with two tables: `forward_response` (prediction x observed setup, routing flags `executable_route` / `counterfactual_markout_only`) and `paper_contract_outcome` (per signal, `executable` flag). No cost or P&L columns.
- Independent ledgers `ind.{stream}.{queue_id}.{variant}.{cost_case}`: one `simulate_positions` run per model x variant; `double_all_costs` via `reprice_positions` on the identical admitted trades.
- Instrument-capacity ledgers `cap.{stream}.instrument_policy.{cost_case}`: all of the stream's models, feed = exact full policy else `setup_threshold`, labeled in `surface` on every row, with `admission_sequence`.
- Surface 4: `full_policy[model_id] = None` gives the `not_defined` status only (no ledger id, no file, no rows). An exact policy is a `FullPolicyBinding(policy_id, policy_hash, rejections)`; its exclusions are recorded `no_trade_reason` values in surface 4 only.
- Every row carries the 14 B3 v3 attribution fields plus `store_kind`, `unit`, `schema_version`. `decision_session` and `decision_hour_new_york` are added by `report_slices` as facets only.
- `guard_report_table` / `ledger_summary` raise `SurfaceReportError` on mixed ledger kinds, markouts with positions, two streams or units, two cost cases, more than one ledger in a total, or a `not_defined` result.
- `persist_stream_surfaces(result, output_dir)` writes `{stream}/markouts/<store>/*.csv`, `{stream}/ledgers/<ledger>.csv` and `{stream}/manifest.json` (file hashes, per-ledger summaries, `surface_hash`); refuses an existing directory.

Engine (still `b1-v5`; no b1-v5 evidence had been published): `simulate_positions(..., priority=None)`. With `priority` (model_id -> frozen selection rank) the order is known_at, rank, event ID, and an exit at a bar's open does not free capacity for an entry at that same open. Without it, ordering and release are unchanged. Reading recorded for review: "opportunity key" is the event ID; "a same-bar exit does not free capacity at that bar's open" is applied to gap-open exits in the capacity ledger only (independent ledgers keep the b1-v4 release rule).

App side: `ote_live/storage/research_execution.py` `report()` returns `surface_schema` (schema identity and the `setup_full_policy` -> `full_policy` alias). Nothing else in the app changed; scored registration stays disabled; no installed manifest or collection touched.

Hashes (sha256):

- `model_testing/outcome_engine.py` `e39ba0c0318bb4b668ae0e21dddcb39e8e58d136f75fdd78c8f5aa31f41fe991`
- `model_testing/outcome_surfaces.py` `eac06b369a346b277a6b20b4ee9d2392ed464d839c2b7ee080d17049304dcfd7`
- `tests/test_section3_surfaces_ledgers.py` `da40507a79c68584ac2ce01782fb77b6c44a1f0ecdcb1c734e7e27089223afec`
- `ote_live/storage/research_execution.py` `f319dd6a92360a7689775f140ee9d00cd6caaeca3b4e126b13255693592d200f`

Tests: focused `tests/test_section3_surfaces_ledgers.py` 16 passed (two models x two streams, route -> surfaces -> both ledger kinds, per-trade sums equal ledger totals after a CSV round trip, costs applied once, determinism, guard, session shift through all five sessions, AST guard). Eight outcome / section 3 / B3 / route / A7 files together: 313 passed. Broader suite, run once with `ote_venv`: 1197 passed, 2 skipped, 15 failed (log `logs/section3_prompt3_full_suite_20261002.log`). All 15 are in six files already on the prompt 2 list of failures that do not import the outcome modules (`test_backend_attribution` 2, `test_ote_live_feature_parity` 3, `test_ote_live_feature_seed_state` 2, `test_ote_live_phase5_operator_tooling` 4, `test_ote_live_prediction_replay` 3, `test_ote_xgboost_pipeline` 1); left untouched and not compared against a pre-change baseline. No outcome, section 3, B3, route or A7 ledger test failed.

Known limits and notes for prompt 4:

- Nothing was run on real data; no surface or ledger exists yet for any of the 16 models.
- ES unit defaults to ticks; `unit="cash"` (USD) is accepted for ES only. EURUSD is pips only.
- `run_outcome_evaluation` / `replay_outcome_run` do not carry `priority`; deterministic replay of these stores is a rebuild compared on `surface_hash` and file hashes.
- A model that fails inside `build_stream_surfaces` raises for the whole stream call; the harness must isolate a failing model as diagnostic/unscored and rebuild without it.
- Windows path length: store names are long, so use a short output root (pytest needs a short `--basetemp`, e.g. `%TEMP%\claude\s3a`).
- The fixture uses FRVP and OTE models only; ICT routes go through the same code with no ICT-specific fixture.

### Prompt 4 (2026-10-02): R4 deterministic smoke replay

Gate: R1, R2, R3 and R5 are complete (EURUSD calendar pin is decision D6, not a gap), so prompt 4 proceeded.

Harness: `scripts/run_section3_smoke_replay.py` (new; reuses `build_schedule` / `source_bars` from the B1 review harness, `runtime` from `scripts/audit_b2_candidate_history.py`, and `build_stream_surfaces` / `persist_stream_surfaces`). Stages, each safe to re-run with `ote_venv\Scripts\python.exe`:

1. `setup` - done. Windows `artifacts/s3smoke_20261002_v1/windows.json` sha256 `9db6421a06bcf04b9e83b0453204ef0d13f3f029773509da0e8e0f4d66f0956b`, written before any outcome. Log `logs/section3_smoke_setup_20261002.log`.
2. `run --run a` then `run --run b` (two separate processes; a finished model is skipped via its `m<id>/report.json`). Logs `logs/section3_smoke_run_{a,b}_20261002.log`; completion marker `RUN a COMPLETE` / `RUN b COMPLETE`.
3. `report` - compares a with b, writes `docs/audits/section3_smoke_<model_id>_20261002.json`, `docs/section3_simulation_rules.md#section3_smoke_replay_20261002`, and the per-model status in `artifacts/s3smoke_20261002_v1/state.json`. Marker `REPORT COMPLETE`.

Per-model status lives in `artifacts/s3smoke_20261002_v1/state.json` (pending -> running -> passed | failed | not_run).

Readings recorded for review (also inside `windows.json`):

- Trading week = sessions of the pinned ES calendar grouped by the Monday of the session date. Candidate week = every expected open present in the source bars, lying inside the model's first-to-last held-out endpoint.
- Overnight / RTH element = a same-side setup match whose decision bar opens outside / inside 09:30-16:00 New York. Planned closure = a week with fewer expected opens than the modal week. DST transition = first open on a different New York UTC offset than the previous week. ES roll = a contract-coordinate change in the source bars.
- ES ledgers use the default unit, ticks (USD not produced).
- Every ES held-out prediction has at most one detector-selected same-side setup, so no pooled route is ambiguous. ID 5 has 30 held-out rows with no setup; they stay in the per-model `census.csv` and never enter a ledger.
- ICT fired events in the held-out period all open outside 09:30-16:00 New York, so `rth_segment` is unexercised for the seven ICT models. ICT routes with no held-out setup match are unexercised: 12 and 15 (4 of 8 each), 13 and 16 (4 of 6 each).
- EURUSD (22, 26, 21, 25, 23): `not_run` per D6 / B3 v3 `eurusd_calendar_rule`; frozen bytes are still hash-verified.
- Surface 1 limit: the shared module only accepts predictions on a declared route, so a prediction without a setup match is recorded in `census.csv` with its score and match state but has no forward-response row.

#### Prompt 4 result (2026-10-02) - COMPLETE

All three stages finished (`RUN a COMPLETE`, `RUN b COMPLETE`, `REPORT COMPLETE`).

| Stream | IDs | Status |
|---|---|---|
| ES | 2, 10, 12, 13, 15, 16, 3, 19, 5, 18, 14 | passed |
| EURUSD | 22, 26, 21, 25, 23 | not_run (D6: no pinned EURUSD 5m calendar); frozen bytes verified against B4 v2 |

- Per model: bytes equal to the B4 v2 hashes; production-runner scores equal to the saved held-out predictions (threshold decisions identical); canonical hash identical across runs a and b; per-trade P&L sums to ledger totals; costs recomputed once at contract rates (and exactly doubled in `double_all_costs`); independent ledgers byte-identical whether built alone or beside the other models.
- ES instrument-capacity ledger: 280 opportunities, 56 admitted, 56 closed, 0 open; identical across runs, canonical hash `c8b820719b5859d066956944e4d1599b4d89cd0be893833f0946b67a89c020e7`; 56 trades reconcile with the independent `setup_threshold` ledgers by event ID, 0 mismatches.
- Routes: 29 of 45 ES routes exercised; 16 unexercised (IDs 12, 15, 13, 16: no held-out setup match). 14 EURUSD routes unexercised (stream not run).
- Not exercised on real data: no position stayed open in any window and no counterfactual markout occurred (every match is the single selected setup). Both paths are covered by the synthetic tests only.
- Full policy: `not_defined` for all 16, no ledger file.
- Evidence: `docs/section3_simulation_rules.md#section3_smoke_replay_20261002` (sha256 `365e61b386ffb7e78a56c292e427d91f5942c61c327a7f2dd0c01d8e8c9855d3`), 16 files `docs/audits/section3_smoke_<model_id>_20261002.json`, outputs `artifacts/s3smoke_20261002_v1/{a,b}`, `summary.json`, `state.json`. Harness sha256 `ac674b67884fb6d341a34ec7291df71686682ca5417a065169626d8376a1e796`.
- Tests: `tests/test_section3_surfaces_ledgers.py`, `tests/test_section3_execution_rules.py`, `tests/test_outcome_route_bindings.py`: 83 passed. No shared code changed in prompt 4 (one new script), so the broader suite was not re-run. The harness has no unit test of its own.
- Close-out: notes.txt section 3 - session coverage ticked; surfaces, ledgers, enforcement and smoke replay annotated as partial with the EURUSD IDs instead of ticked; carry-IDs item annotated and left unchecked. FRVP and ICT journal entries added. No runtime code changed, so no `live_app_change_journal.md` entry.

### Section 3 status after prompt 4 (superseded by prompt 5 below): COMPLETE for ES; EURUSD open on one input

Section 3 work is finished for the 11 ES models. The five EURUSD models need one thing: a versioned EURUSD 5m expected-open calendar (D6). After it is pinned, extend `scripts/run_section3_smoke_replay.py` with the EURUSD inputs (OTE setup census, tape) under a new output root and a new date, then tick the four annotated notes.txt items.

### Next action

None inside section 3 without the EURUSD calendar decision. Section 4 has not been started.

Superseded plan text: roster-driven smoke harness on b1-v5 (`require_completion=True`, B4 v2 thresholds and artifact hashes, D5 window) calling `build_stream_surfaces` / `persist_stream_surfaces` per stream - R4 deterministic replay, canonical hashing and the per-route coverage report; EURUSD stays `not_run` until a calendar is pinned (D6).
### Prompt 5 (2026-10-02): EURUSD smoke replay after the user lifted D6 - COMPLETE

User instruction (2026-10-02): pick a standard FX calendar or derive it from the data, run the same smoke for the five EURUSD models, close section 3 for all 16.

- Calendar pinned: `configs/eurusd_5m_expected_open_calendar_v1.json` (`eurusd_5m_feed_clock_week_v1`, sha256 `5db00e42d5513af92943184a7c566a5f44782700258a72a1f889c2d9d884d139`) plus `.lock.json`. Rule-based: Sunday 21:00 UTC to Friday 22:00 UTC, closed Christmas Day and New Year's Day, fixed UTC (the feed's clock). Weekly hours were set to the feed's observed regular hours; no individual closure was inferred from bars. Provisional; B3 v3 is unchanged (its rule only required a versioned calendar to be pinned).
- Harness extended (same file, `--stream EURUSD`); ES code paths and ES evidence untouched. EURUSD root `artifacts/s3smoke_eurusd_20261002_v1`; windows sha256 `af937aeb4e9b5293a2c2a3d1a10ffa40bcccdacdd660cc45beffa4b66090aaea`. New evidence is published as `docs/audits/section3_smoke_<model_id>_20261002_v2.json` and `docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v2`; the first-pass files are kept.
- Stages: `setup --stream EURUSD` done; `run --run a --stream EURUSD`, `run --run b --stream EURUSD`, `report --stream EURUSD` (logs `logs/section3_smoke_eurusd_*_20261002.log`, markers `RUN a COMPLETE`, `RUN b COMPLETE`, `REPORT COMPLETE`).
- EURUSD inputs: bars = pinned B2 source tape `artifacts/b2_ote_labels_20260920_v2/labels.csv.gz` (hash from each prepared manifest); setups = OTE base events `events.csv.gz` in the same artifact (reversal, continuation pullback, breakout), matched when the event's confirmation is the close of the decision bar; side = swing low -> long, swing high -> short. Only family, side and confirmation bar are read (no label, tier, entry or outcome column).
- Readings: exactly one same-side family on a bar = detector-selected setup; two or more = ambiguous, kept as non-executable counterfactual markouts. `breakout_sustained` is never emitted by the detector, so that route is unexercised for 21/25/23.
- Finding: IDs 22 and 26 (reversal specialists) have no saved held-out prediction on any reversal confirmation bar (0 of 644 for ID 22; their training rows exclude exactly those bars), so no route can be exercised under the exact-match rule. They stay diagnostic. Open question for the user: allow the production runner to score those confirmation bars even though they are not in the saved held-out predictions?
- Only 6 of 59 EURUSD calendar weeks have every expected bar (4 candidates per model), so planned-closure and DST elements are unexercised for EURUSD.

#### Prompt 5 result

All stages finished (`RUN a COMPLETE`, `RUN b COMPLETE`, `REPORT COMPLETE`).

- IDs 21, 25, 23: passed. Bytes equal to B4 v2; runner scores equal to saved predictions; canonical hashes identical across runs; accounting clean in pips. Windows: 21 and 23 week 2025-03-03, 25 week 2025-09-08. Signals 58 / 47 / 58.
- EURUSD instrument-capacity ledger: 163 opportunities, 3 admitted, 3 closed, 0 open; identical across runs, hash `a3546ac91dfe735246d975d568e8092f9562c40512a00c38d705eb7a327ad9c3`; reconciles with the independent ledgers, 0 mismatches. ID 21 had no threshold pass in its window, so its `setup_threshold` ledger and its capacity overlap are empty (a coverage result, not a pass on trades).
- Newly exercised on real data: one open position carried as open (each of 21, 25, 23 `setup_only`), and 12 non-executable counterfactual markouts on ID 25 (3 ambiguous bars).
- IDs 22, 26: failed, diagnostic/unscored. Bytes verified. No window could be chosen because no saved held-out prediction sits on a reversal confirmation bar.
- Routes: EURUSD 9 of 14 exercised. Unexercised: `breakout_sustained` for 21/25/23 (no detector emits it) and `reversal` for 22/26.
- Evidence: `docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v2` (sha256 `e5662fcf96d2c9608d37f80e5b4ebef3d89c1c393e3060eae59b2b489c760f17`), five `docs/audits/section3_smoke_<model_id>_20261002_v2.json`, `artifacts/s3smoke_eurusd_20261002_v1/{a,b}`. First-pass summary unchanged (sha256 `365e61b3...`). Harness sha256 now `1e7cbf7281ae597cdd8171f3933869955fe7ade6e22f6c2002a0b63154091411` (the first-pass evidence was produced by revision `ac674b67...`).
- Tests: no shared module changed (harness and one new config only); the 83 focused tests were not re-run and the harness still has no unit test.
- notes.txt section 3: execution enforcement and session coverage ticked; surfaces, ledgers and smoke replay annotated with failing IDs 22/26; carry-IDs item notes its section 3 portion done, left unchecked.

### Section 3 status after prompt 5 (superseded by prompt 6 below): 14 of 16; IDs 22 and 26 diagnostic

Open question (one): for 22 and 26, may the production runner score the reversal confirmation bars that are absent from their saved held-out predictions? If yes, re-run those two under a new output root. Also still provisional: the EURUSD calendar v1. Section 4 not started.

### Prompt 6 (2026-10-02): IDs 22 and 26 on confirmation bars - COMPLETE

User decision (2026-10-02): yes, the production runner may score the reversal confirmation bars that are absent from the saved held-out predictions of IDs 22 and 26.

- Harness flag `--stream EURUSD --pass 2`; new root `artifacts/s3smoke_eurusd_20261002_v2`; windows sha256 `7e40460772dd06123dbf3b47cc99418629d2ca8f5272f93548f791373792fac8` (22 and 26: week 2025-03-03; 21/25/23 windows carried verbatim from the first EURUSD root).
- Prediction universe for 22/26 = saved held-out predictions plus declared-route (reversal) confirmation bars strictly inside the saved endpoint range. Scores on the added bars cannot be compared with saved predictions; every saved bar in the same window is compared.
- 21/25/23 model outputs are read from `artifacts/s3smoke_eurusd_20261002_v1` (unchanged); the EURUSD capacity ledger is rebuilt with all five models in the new root.
- Stages: setup done; `run --run a`, `run --run b`, `report` (all with `--stream EURUSD --pass 2`); logs `logs/section3_smoke_eurusd_p2_*_20261002.log`; markers `RUN a COMPLETE`, `RUN b COMPLETE`, `REPORT COMPLETE`.
- Outputs: `docs/audits/section3_smoke_<model_id>_20261002_v3.json` (five EURUSD models) and `docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v3`. Earlier files are kept.
- After the report: tick notes.txt surfaces, ledgers and smoke replay if all 16 pass; update this file.

#### Prompt 6 result

- IDs 22 and 26: passed. 22: 1373 bars scored, 1363 matched against saved predictions, 10 confirmation bars unverifiable; 10 signals. 26: 1329 scored, 1314 matched, 15 unverifiable; 15 signals. Neither had a threshold pass in its window, so their `setup_threshold` ledgers and capacity overlap are empty (coverage result).
- EURUSD capacity ledger rebuilt with all five models: 188 opportunities, 3 admitted, 3 closed, 0 open; identical across runs; hash `b2964bdca2db74ea0e91774eadc97857fed02a735aa469fd0d29904a002e6848`; 0 mismatches. It supersedes the three-model ledger `a3546ac9...` of prompt 5.
- IDs 21/25/23: model outputs unchanged from prompt 5 (same canonical hashes); re-audited as `_v3` against the five-model capacity ledger.
- Harness defect found and fixed: the resume path re-read `scores.csv` with the default float parser, changing the last digit of two score columns, so one markout file differed byte-for-byte between the model build and the stream build (ledgers and outcomes were identical). Fixed with `float_precision="round_trip"`. The invalid stream outputs are kept at `artifacts/s3smoke_eurusd_20261002_v2/{a,b}/stream_EURUSD_invalid_csv_float_parse`; the `_v3` audits and summary were regenerated after the fix (logs `logs/section3_smoke_eurusd_p2_*_rerun_20261002.log`). The ES and first EURUSD passes did not use that path for a stream build, so they are unaffected.
- Evidence: `docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v3` (sha256 `f2792e2b0e40835b02b6cae8c756d7521d2677094a8622923c97eadfece797a2`), five `docs/audits/section3_smoke_<model_id>_20261002_v3.json`, `artifacts/s3smoke_eurusd_20261002_v2`. Harness sha256 `3b001ba7b353e34faa361018496497d5d4e54236daa9a1e30933eb065d57362c`.
- Routes exercised: 29 of 45 ES, 11 of 14 EURUSD (`breakout_sustained` unexercised for 21/25/23).
- notes.txt section 3: all six items ticked; carry-IDs item notes its section 3 portion done and stays unchecked.
- Tests: no shared module changed; focused tests not re-run; the harness has no unit test.

### Section 3 status: COMPLETE for all 16

Standing caveats: the EURUSD calendar v1 is provisional; 22/26 confirmation-bar scores are not comparable with saved predictions; EURUSD ran on one week per model, with planned-closure and DST weeks unexercised; ES ledgers are in ticks only. Section 4 not started.


<a id="section3_gap_matrix_20261002"></a>

## Section 3 gap matrix — 2026-10-02

*Merged 2026-10-09 from `docs/section3_gap_matrix_20261002.md`.*

Inventory from prompt 1. R3 rows were updated on 2026-10-02 (prompt 2) after the b1-v5 enforcement fixes; `t_s3` = `tests/test_section3_execution_rules.py` (one test per rule, parametrized over ES and EURUSD). R1, R2 and R5 rows were updated on 2026-10-02 (prompt 3); `surf` = `model_testing/outcome_surfaces.py`, `t_surf` = `tests/test_section3_surfaces_ledgers.py` (synthetic two-model fixture per stream). R4 rows are unchanged. Status is exists / partial / missing per stream.

Abbreviations: `eng` = `model_testing/outcome_engine.py`, `art` = `model_testing/outcome_artifacts.py`, `diag` = `model_testing/outcome_diagnostics.py`, `harness` = `scripts/review_b1_candidate_outcomes.py`, `A7` = `ote_live/storage/research_execution.py`, `t_oc` = `tests/test_outcome_contracts.py`, `t_ote` = `tests/test_outcome_ote_contracts.py`, `t_scope` = `tests/test_outcome_scope_audit_atr.py`, `t_route` = `tests/test_outcome_route_bindings.py`.

Cross-cutting facts behind most rows:

- The B1 engine (b1-v4) is instrument-agnostic and tested on both ES and EURUSD contracts. The only runnable harness is hardcoded to candidates 7 and 17, neither of which is in the 16-model roster, and to ES bars and the CME calendar.
- b1-v4's "four surfaces" are training label, paper contract outcome, forward returns and simulated position. They are not the section-3 four surfaces.
- The A7 live ledger is ES-only (`asset: Literal["ES"]`, FRVP/ICT ids), uses fixed tick geometry and a flat spread instead of the B1 contract ATR and session costs, and has three variants. It is not used for the historical smoke replay.
- All 16 full policies are `not_defined` in B4 v2.

| Req | Sub-requirement | Code location | Existing tests | ES | EURUSD | Minimal fix |
|---|---|---|---|---|---|---|
| R1 | Raw score / gross forward response | `surf` `build_stream_surfaces` (markout store `mk.{stream}.{queue_id}.raw_score_gross_forward_response`, table `forward_response`); source `eng` forward_return_*, `diag` | `t_surf::test_route_to_surfaces_to_both_ledger_kinds_end_to_end` | exists | exists | Done (prompt 3): one row per prediction x observed setup, above or below threshold, with routing state. Not yet produced for the 16 on real data (prompt 4 harness). |
| R1 | Setup-only trade | `surf` (variant `setup_only`, ledger `ind.{stream}.{queue_id}.setup_only.{cost_case}`) | `t_surf` | exists | exists | Done: no probability filter, no policy gate. Real-data run is prompt 4. |
| R1 | Setup + threshold | `surf` `_eligibility` (calibrated probability >= caller-supplied frozen threshold, equality accepted) | `t_surf` | exists | exists | Done. The harness must pass the exact B4 v2 threshold float per model. |
| R1 | Intended full policy or `not_defined` | `surf` (`full_policy` map; `FullPolicyBinding` for an exact policy); `model_testing/b3_comparison.py` 206-229 | `t_surf`, `test_b3_comparison_plan.py` | exists (as `not_defined`) | exists (as `not_defined`) | Done: `not_defined` is persisted as a status in `manifest.json` with no ledger file, no rows and no totals; totalling it raises. |
| R1 | Event markouts stored separately from position P&L | `surf` `persist_stream_surfaces` (`{stream}/markouts/` versus `{stream}/ledgers/`) | `t_surf` | exists | exists | Done: markout tables (`forward_response`, `paper_contract_outcome`) carry no cost or P&L columns; pooled counterfactuals are markout rows flagged non-executable and never a ledger row. |
| R1 | Attribution fields | `surf` `ATTRIBUTION_FIELDS` (the 14 B3 v3 fields) plus `store_kind`, `unit`, `schema_version` | `t_surf::test_schema_matches_the_frozen_b3_protocol` and the end-to-end test | exists | exists | Done on every markout and ledger row; schema `section3-surfaces-v1`. |
| R2 | Independent ledger per model × variant | `surf` (one `simulate_positions` run per model × variant, `concurrency_scope="model"`, then `reprice_positions` per cost case) | `t_surf` | exists | exists | Done: limits per B3 v3 (1 position, 1 unit, no opposite side, zero cooldown). |
| R2 | Separately named one-position-per-instrument ledger | `surf` (`cap.{stream}.instrument_policy.{cost_case}`); `eng` `simulate_positions(priority=…)` | `t_surf` (tie, earlier decision, gap-open exit, default ordering unchanged) | exists | exists | Done (D9, inside b1-v5): order is known_at, then frozen selection order, then event ID; an exit at a bar's open does not free capacity for an entry at that same open. Feed variant is labeled on every row. |
| R2 | Reporting guard (ledger kinds, markout vs position, ES vs EURUSD) | `surf` `guard_report_table`, `ledger_summary` (`SurfaceReportError`) | `t_surf::test_reporting_guard_fails_closed`, `test_streams_models_and_units_fail_closed` | exists | exists | Done: raises on mixed ledger kinds, markouts with positions, two streams or units, two cost cases, more than one ledger in a total, and `not_defined` passed for totals. |
| R3 | Next eligible open after actual decision time | `eng` 338-349 (`searchsorted(known, "right")`) | `t_oc::test_close_boundary_is_strict…`, `test_no_entry_at_end_of_calendar` | exists | exists | Historical known_at is bar close + 1 s (labeled offline). None. |
| R3 | Completed / fresh-bar availability | `eng` 197 (only if `is_complete` column present), `eng` 157-191 (ordering only) | `t_ote::test_provisional_execution_path_cannot_resolve_a_trade` | exists | exists | Fixed in b1-v5: `EvaluationTape(require_completion=True)` rejects a bar with no `is_complete == True`, including when the column is absent; the flag is saved in `calendar.json` for replay. Tested in `t_s3` (incomplete, unknown, stale). The harness must pass `require_completion=True`. The 90 s / 30 min / 60 min wall-clock limits live only in A7 (ES-only) and stay flagged for section 5. |
| R3 | Fixed entry geometry from the contract's causal ATR | `eng` 207-250, `model_testing/outcome_atr.py` | `t_oc` 389-416, `t_ote` 59, 155, `t_scope` 136-218 | exists | exists | None. |
| R3 | Entry-relative timeout | `eng` 391-393 | `t_oc::test_timeout_does_not_start_from_retrospective_anchor`, `t_ote` 59 | exists | exists | None. |
| R3 | Management rules | `outcome_contracts.py` 105 (`fixed_barriers` only) | `t_oc::test_structural_barriers_freeze_initial_references…` | exists | exists | None; no other management is frozen. |
| R3 | Adverse-first collision fallback | `eng` 253-295 | `t_oc` 149, 157 | exists | exists | None. |
| R3 | Gap-through fills | `eng` 255-259 | `t_oc::test_stop_gap_pays_open_not_stop`, `t_s3` | exists | exists | None. Per the frozen `exit_price` term a stop gap fills at the gap open; a target gap fills at the target with no favorable improvement. |
| R3 | Planned closures / weekends | `eng` 99-131 (calendar), 410-411; `harness` 68-81 (CME) | `t_oc::test_planned_weekend_and_forbidden_overnight` | exists | partial | Calendar handling is one implementation and passes for both streams on synthetic calendars (`t_s3`). EURUSD stays partial only because no versioned EURUSD expected-open calendar is pinned (D6); that is a data pin, not an engine gap. |
| R3 | ES rolls | `eng` 199-203 | `t_scope` 159 | exists | n/a | None. |
| R3 | Missing bars | `eng` 344-348, 403-406 | `t_oc` 113, `t_ote` 235 | exists | exists | None. |
| R3 | Costs and cost stress | `eng` 445-457 (base only); stress ad hoc in `scripts/backtest_b2_s3_*` | `t_oc` 218, 240; `t_ote` 76 | exists | exists | Fixed in b1-v5: `COST_CASES` and `reprice_positions` re-price the identical admitted trades and fills at 2× every component; every position row carries `cost_case`. Tested in `t_s3`. |
| R3 | Quantity / concurrency | `eng` 496-506 | `t_oc` 205; A7 279, 294 | exists | exists | None beyond R2. |
| R3 | Unresolved / open exposure | `eng` 496, 510-511 | `t_oc::test_unknown_exit_consumes_capacity…` | exists | exists | b1-v5 adds a mark: `paper_mark_ts/price`, and `mark_ts`, `mark_price`, `unrealized_gross_pnl` on the UNRESOLVED position row; `gross_pnl` and `net_pnl` stay null. Tested in `t_s3`. Harness must report it as open exposure. |
| R3 | Prohibitions (label close, best entry, target hit, later setup never a fill) | `eng` 298-300, 349; `outcome_scope.py` 67-104; `diag` 100-180 | `t_ote` 163, `t_oc` 126, 205, 306, `t_scope` 106, `t_route` 145 | exists | exists | None. |
| R4 | Deterministic replay | `art` 140-166 | `t_oc` 348, 356; `t_ote` 175 | partial | partial | Exists per run; not run for any of the 16. Replay refuses if engine source changed, which matters for D9. |
| R4 | Canonical hashing | `outcome_contracts.py` 17-30, `art` 133-135 | `t_oc` 251, 270 | exists | exists | `run_hash` covers python and pandas versions; record the environment with each smoke. |
| R4 | Per-route coverage reporting | `art` 79-82 (per model, not per route id) | none | missing | missing | Add the per-route table defined in the draft. |
| R5 | No hour/session filtering outside frozen full-policy exclusions | `surf` `report_slices` (facets only); `eng` has none; `ote_live/policies/abstain.py` 58-78 can filter sessions | `t_surf::test_surfaces_1_to_3_and_all_ledgers_ignore_session_and_hour`, `test_frozen_full_policy_exclusion_is_only_a_surface_4_rejection_reason`, `test_no_session_or_hour_filter_exists_outside_surface_4` | exists | exists | Done for the simulator and the shared module. Runtime policy gates in `ote_live` belong to section 5. |

### Engine-version impact

Update 2026-10-02 (prompt 2): the engine is now b1-v5 for the three R3 fixes above (opt-in mandatory completion, open-position mark, shared cost cases). All are additive: default arguments reproduce b1-v4 fills and P&L, and only new columns appear. The working tree had already drifted from the archived b1-v4 source on 2026-09-29 (pooled-routing scope hook in `outcome_engine.py`, `outcome_scope.py`, `outcome_diagnostics.py`) without a version bump; b1-v5 covers that drift too. b1-v4 is reproducible from the `engine_source` folder archived inside each b1-v4 run. D9 landed in prompt 3 as the optional `priority` argument of `simulate_positions`, still b1-v5 because no b1-v5 evidence had been published; without `priority` the ordering and capacity release are unchanged.

Original note: Only D9 (capacity-ledger ordering key) touches `simulate_positions`. If approved it becomes b1-v5. Prior evidence that would then need archived b1-v4 source to replay, with no numeric change expected: the S3 backtests (`docs/B2_S3_*`) and the B1 candidate outcome review (`docs/B1_outcome_contracts.md#b1_candidate_outcome_review_20260929`). No frozen B1 contract changes. Everything else is harness-level.


<a id="section3_smoke_replay_20261002_v3"></a>

## Section 3 smoke replay - 2026-10-02

*Merged 2026-10-09 from `docs/section3_smoke_replay_20261002_v3.md`.*

Deterministic smoke replay (R4) of the 16 selected versions on engine `b1-v5`, B4 v2 freeze, route bindings v2 and B3 protocol v3. Windows were chosen from setup presence and the calendar only and hashed before any outcome was computed. P&L and trade counts are advisory: recorded, not interpreted. All 16 full policies are `not_defined` (no ledger, no rows, no totals). ES and EURUSD are reported separately and never combined.

- ES windows: `artifacts/s3smoke_20261002_v1/windows.json` sha256 `9db6421a06bcf04b9e83b0453204ef0d13f3f029773509da0e8e0f4d66f0956b`
- EURUSD windows: `artifacts/s3smoke_eurusd_20261002_v2/windows.json` sha256 `7e40460772dd06123dbf3b47cc99418629d2ca8f5272f93548f791373792fac8`
- Protocol sha256 `7e5189cdd1543005e5035e940a168d3d22e233b3039aac899077a303b0a7dcc4`; B4 v2 sha256 `b9a7bc674b2b38118fcc2187b8c94c9f1aec6b2d84869572244f33f5abd6cb5b`; route bindings v2 sha256 `2473e44e4054068eac4fda1219586401c300b3a2e7a3aa54202e08bf873045d6`
- Outputs: `<stream root>/a` and `/b` (two independent processes); per-model audits `docs/audits/section3_smoke_<model_id>_20261002.json` (ES) and `..._20261002_v2.json` (EURUSD, when run)

EURUSD calendar: `configs/eurusd_5m_expected_open_calendar_v1.json` (`eurusd_5m_feed_clock_week_v1`, sha256 `5db00e42d5513af92943184a7c566a5f44782700258a72a1f889c2d9d884d139`), a provisional rule-based calendar authorized by the user on 2026-10-02 and pinned before any EURUSD outcome was computed. The ES section below restates the unchanged first-pass evidence.

Pass 2 (user-authorized 2026-10-02): for IDs 22 and 26 the production runner also scored the reversal confirmation bars that their saved held-out predictions omit. Those scores cannot be compared with saved predictions; every saved bar in the same window was compared and matched. IDs 21, 25 and 23 model outputs are unchanged from the first EURUSD pass; the EURUSD instrument-capacity ledger was rebuilt with all five models.

### ES (ticks)

Instrument-capacity ledger: models [2, 10, 12, 13, 15, 16, 3, 19, 5, 18, 14]; replay identical: True; hash `c8b820719b5859d066956944e4d1599b4d89cd0be893833f0946b67a89c020e7`; problems: 0.

- advisory `cap.ES.instrument_policy.base`: opportunities 280, admitted 56, closed 56, open 0, gross 43.00, cost 196.40, net -153.40 ticks
- advisory `cap.ES.instrument_policy.double_all_costs`: opportunities 280, admitted 56, closed 56, open 0, gross 43.00, cost 392.80, net -349.80 ticks

| ID | Model | Status | Weeks | Routes exercised | Routes unexercised | Replay hash (a = b) |
|---|---|---|---|---|---|---|
| 2 | `frvp_long_continuation_xgb_v1` | passed | 2024-12-16, 2024-12-23, 2025-03-10 | 3 | 0 | `95ed05c4c9233b0b` = `95ed05c4c9233b0b` |
| 10 | `frvp_long_reversal_xgb_v1` | passed | 2024-12-16, 2024-12-30, 2025-03-10 | 3 | 0 | `f750796ac0b5efd7` = `f750796ac0b5efd7` |
| 12 | `ict_long_meta_xgb_v1` | passed | 2025-03-10, 2025-03-17, 2025-05-26 | 4 | 4 | `1b68bcb2e60f37ae` = `1b68bcb2e60f37ae` |
| 13 | `ict_long_reversal_xgb_v1` | passed | 2025-03-10, 2025-06-16 | 2 | 4 | `08aa61c7074f5dcb` = `08aa61c7074f5dcb` |
| 15 | `ict_short_meta_xgb_v1` | passed | 2025-03-10, 2025-05-05, 2025-06-16 | 4 | 4 | `7872651421ec8ad8` = `7872651421ec8ad8` |
| 16 | `ict_short_reversal_xgb_v1` | passed | 2025-03-10, 2025-06-16 | 2 | 4 | `bcaf287c4a515219` = `bcaf287c4a515219` |
| 3 | `frvp_long_meta_xgb_v1` | passed | 2024-12-16, 2024-12-30, 2025-03-10 | 6 | 0 | `a5a99838ce59db30` = `a5a99838ce59db30` |
| 19 | `ict_short_reversal_sweep_reclaim_xgb_v1` | passed | 2025-03-10, 2025-06-16 | 1 | 0 | `d843d1794c689bf4` = `d843d1794c689bf4` |
| 5 | `frvp_long_continuation_setup2_xgb_v1` | passed | 2024-11-04, 2024-11-25, 2024-12-16 | 1 | 0 | `79a85f520a43e3ec` = `79a85f520a43e3ec` |
| 18 | `ict_short_reversal_ifvg_reversal_xgb_v1` | passed | 2025-03-10, 2025-06-16 | 1 | 0 | `7475479faf49bb11` = `7475479faf49bb11` |
| 14 | `ict_short_continuation_xgb_v1` | passed | 2025-03-10, 2025-05-05, 2025-06-16 | 2 | 0 | `cdc65f74050c2218` = `cdc65f74050c2218` |

#### ID 2 `frvp_long_continuation_xgb_v1` - passed

- Exercised routes: `2:setup2:es_frvp_setup2_eval_v2`, `2:setup3:es_frvp_setup3_long_eval_v2`, `2:setup5:es_frvp_setup5_long_eval_v2`
- Unexercised routes: none
- Unexercised calendar elements: none
- Signals 35; census {'selected_same_side_setup': 35}; open-position rows 0; capacity reconcile {'capacity_rows': 35, 'capacity_admitted': 9, 'independent_admitted': 9, 'trades_in_both': 9, 'trades_mismatched': 0}
- advisory `ind.ES.2.setup_only.base`: admitted 24, closed 24, open 0, gross 182.00, cost 69.60, net 112.40 ticks
- advisory `ind.ES.2.setup_threshold.base`: admitted 9, closed 9, open 0, gross 112.00, cost 25.35, net 86.65 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 10 `frvp_long_reversal_xgb_v1` - passed

- Exercised routes: `10:setup1:es_frvp_setup1_eval_v1`, `10:setup4:es_frvp_setup4_eval_v2`, `10:setup6:es_frvp_setup6_eval_v1`
- Unexercised routes: none
- Unexercised calendar elements: none
- Signals 15; census {'selected_same_side_setup': 15}; open-position rows 0; capacity reconcile {'capacity_rows': 15, 'capacity_admitted': 1, 'independent_admitted': 1, 'trades_in_both': 1, 'trades_mismatched': 0}
- advisory `ind.ES.10.setup_only.base`: admitted 12, closed 12, open 0, gross 290.00, cost 37.80, net 252.20 ticks
- advisory `ind.ES.10.setup_threshold.base`: admitted 1, closed 1, open 0, gross 64.00, cost 2.65, net 61.35 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 12 `ict_long_meta_xgb_v1` - passed

- Exercised routes: `12:sweep_reclaim:es_ict_sweep_reclaim_eval_v1`, `12:ifvg_reversal:es_ict_ifvg_reversal_eval_v1`, `12:premium_discount_continuation:es_ict_premium_discount_continuation_eval_v1`, `12:displacement_continuation_after_raid:es_ict_displacement_continuation_after_raid_eval_v1`
- Unexercised routes: `12:sweep_displacement_fvg:es_ict_sweep_displacement_fvg_eval_v1` (no_eligible_week_in_held_out_period), `12:ob_retest_after_mss:es_ict_ob_retest_after_mss_eval_v1` (no_eligible_week_in_held_out_period), `12:session_open_manipulation_pre_ib:es_ict_session_open_manipulation_pre_ib_eval_v1` (no_eligible_week_in_held_out_period), `12:session_open_manipulation_post_ib:es_ict_session_open_manipulation_post_ib_eval_v1` (no_eligible_week_in_held_out_period)
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 46; census {'selected_same_side_setup': 46}; open-position rows 0; capacity reconcile {'capacity_rows': 46, 'capacity_admitted': 16, 'independent_admitted': 18, 'trades_in_both': 16, 'trades_mismatched': 0}
- advisory `ind.ES.12.setup_only.base`: admitted 38, closed 38, open 0, gross -96.00, cost 142.70, net -238.70 ticks
- advisory `ind.ES.12.setup_threshold.base`: admitted 18, closed 18, open 0, gross -96.00, cost 66.20, net -162.20 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 13 `ict_long_reversal_xgb_v1` - passed

- Exercised routes: `13:sweep_reclaim:es_ict_sweep_reclaim_eval_v1`, `13:ifvg_reversal:es_ict_ifvg_reversal_eval_v1`
- Unexercised routes: `13:sweep_displacement_fvg:es_ict_sweep_displacement_fvg_eval_v1` (no_eligible_week_in_held_out_period), `13:ob_retest_after_mss:es_ict_ob_retest_after_mss_eval_v1` (no_eligible_week_in_held_out_period), `13:session_open_manipulation_pre_ib:es_ict_session_open_manipulation_pre_ib_eval_v1` (no_eligible_week_in_held_out_period), `13:session_open_manipulation_post_ib:es_ict_session_open_manipulation_post_ib_eval_v1` (no_eligible_week_in_held_out_period)
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 27; census {'selected_same_side_setup': 27}; open-position rows 0; capacity reconcile {'capacity_rows': 27, 'capacity_admitted': 6, 'independent_admitted': 11, 'trades_in_both': 6, 'trades_mismatched': 0}
- advisory `ind.ES.13.setup_only.base`: admitted 24, closed 24, open 0, gross 34.00, cost 90.60, net -56.60 ticks
- advisory `ind.ES.13.setup_threshold.base`: admitted 11, closed 11, open 0, gross 62.00, cost 42.15, net 19.85 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 15 `ict_short_meta_xgb_v1` - passed

- Exercised routes: `15:sweep_reclaim:es_ict_sweep_reclaim_eval_v1`, `15:ifvg_reversal:es_ict_ifvg_reversal_eval_v1`, `15:premium_discount_continuation:es_ict_premium_discount_continuation_eval_v1`, `15:displacement_continuation_after_raid:es_ict_displacement_continuation_after_raid_eval_v1`
- Unexercised routes: `15:sweep_displacement_fvg:es_ict_sweep_displacement_fvg_eval_v1` (no_eligible_week_in_held_out_period), `15:ob_retest_after_mss:es_ict_ob_retest_after_mss_eval_v1` (no_eligible_week_in_held_out_period), `15:session_open_manipulation_pre_ib:es_ict_session_open_manipulation_pre_ib_eval_v1` (no_eligible_week_in_held_out_period), `15:session_open_manipulation_post_ib:es_ict_session_open_manipulation_post_ib_eval_v1` (no_eligible_week_in_held_out_period)
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 46; census {'selected_same_side_setup': 46}; open-position rows 0; capacity reconcile {'capacity_rows': 46, 'capacity_admitted': 14, 'independent_admitted': 17, 'trades_in_both': 14, 'trades_mismatched': 0}
- advisory `ind.ES.15.setup_only.base`: admitted 35, closed 35, open 0, gross -118.00, cost 133.25, net -251.25 ticks
- advisory `ind.ES.15.setup_threshold.base`: admitted 17, closed 17, open 0, gross 89.00, cost 64.55, net 24.45 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 16 `ict_short_reversal_xgb_v1` - passed

- Exercised routes: `16:sweep_reclaim:es_ict_sweep_reclaim_eval_v1`, `16:ifvg_reversal:es_ict_ifvg_reversal_eval_v1`
- Unexercised routes: `16:sweep_displacement_fvg:es_ict_sweep_displacement_fvg_eval_v1` (no_eligible_week_in_held_out_period), `16:ob_retest_after_mss:es_ict_ob_retest_after_mss_eval_v1` (no_eligible_week_in_held_out_period), `16:session_open_manipulation_pre_ib:es_ict_session_open_manipulation_pre_ib_eval_v1` (no_eligible_week_in_held_out_period), `16:session_open_manipulation_post_ib:es_ict_session_open_manipulation_post_ib_eval_v1` (no_eligible_week_in_held_out_period)
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 20; census {'selected_same_side_setup': 20}; open-position rows 0; capacity reconcile {'capacity_rows': 20, 'capacity_admitted': 4, 'independent_admitted': 9, 'trades_in_both': 4, 'trades_mismatched': 0}
- advisory `ind.ES.16.setup_only.base`: admitted 17, closed 17, open 0, gross -225.00, cost 66.05, net -291.05 ticks
- advisory `ind.ES.16.setup_threshold.base`: admitted 9, closed 9, open 0, gross 35.00, cost 34.85, net 0.15 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 3 `frvp_long_meta_xgb_v1` - passed

- Exercised routes: `3:setup1:es_frvp_setup1_eval_v1`, `3:setup2:es_frvp_setup2_eval_v2`, `3:setup3:es_frvp_setup3_long_eval_v2`, `3:setup4:es_frvp_setup4_eval_v2`, `3:setup5:es_frvp_setup5_long_eval_v2`, `3:setup6:es_frvp_setup6_eval_v1`
- Unexercised routes: none
- Unexercised calendar elements: none
- Signals 53; census {'selected_same_side_setup': 53}; open-position rows 0; capacity reconcile {'capacity_rows': 53, 'capacity_admitted': 2, 'independent_admitted': 6, 'trades_in_both': 2, 'trades_mismatched': 0}
- advisory `ind.ES.3.setup_only.base`: admitted 36, closed 36, open 0, gross 589.00, cost 107.40, net 481.60 ticks
- advisory `ind.ES.3.setup_threshold.base`: admitted 6, closed 6, open 0, gross -134.00, cost 15.90, net -149.90 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 19 `ict_short_reversal_sweep_reclaim_xgb_v1` - passed

- Exercised routes: `19:sweep_reclaim:es_ict_sweep_reclaim_eval_v1`
- Unexercised routes: none
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 13; census {'selected_same_side_setup': 13}; open-position rows 0; capacity reconcile {'capacity_rows': 13, 'capacity_admitted': 2, 'independent_admitted': 5, 'trades_in_both': 2, 'trades_mismatched': 0}
- advisory `ind.ES.19.setup_only.base`: admitted 12, closed 12, open 0, gross -140.00, cost 47.30, net -187.30 ticks
- advisory `ind.ES.19.setup_threshold.base`: admitted 5, closed 5, open 0, gross -61.00, cost 18.75, net -79.75 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 5 `frvp_long_continuation_setup2_xgb_v1` - passed

- Exercised routes: `5:setup2:es_frvp_setup2_eval_v2`
- Unexercised routes: none
- Unexercised calendar elements: none
- Signals 5; census {'no_same_side_setup': 2, 'selected_same_side_setup': 5}; open-position rows 0; capacity reconcile {'capacity_rows': 5, 'capacity_admitted': 0, 'independent_admitted': 1, 'trades_in_both': 0, 'trades_mismatched': 0}
- advisory `ind.ES.5.setup_only.base`: admitted 3, closed 3, open 0, gross 47.00, cost 7.95, net 39.05 ticks
- advisory `ind.ES.5.setup_threshold.base`: admitted 1, closed 1, open 0, gross 26.00, cost 2.65, net 23.35 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 18 `ict_short_reversal_ifvg_reversal_xgb_v1` - passed

- Exercised routes: `18:ifvg_reversal:es_ict_ifvg_reversal_eval_v1`
- Unexercised routes: none
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 7; census {'selected_same_side_setup': 7}; open-position rows 0; capacity reconcile {'capacity_rows': 7, 'capacity_admitted': 0, 'independent_admitted': 2, 'trades_in_both': 0, 'trades_mismatched': 0}
- advisory `ind.ES.18.setup_only.base`: admitted 5, closed 5, open 0, gross -85.00, cost 18.75, net -103.75 ticks
- advisory `ind.ES.18.setup_threshold.base`: admitted 2, closed 2, open 0, gross -54.00, cost 7.30, net -61.30 ticks
- Surface 4 `full_policy`: `not_defined`

#### ID 14 `ict_short_continuation_xgb_v1` - passed

- Exercised routes: `14:premium_discount_continuation:es_ict_premium_discount_continuation_eval_v1`, `14:displacement_continuation_after_raid:es_ict_displacement_continuation_after_raid_eval_v1`
- Unexercised routes: none
- Unexercised calendar elements: rth_segment (no_eligible_week_in_held_out_period)
- Signals 13; census {'selected_same_side_setup': 13}; open-position rows 0; capacity reconcile {'capacity_rows': 13, 'capacity_admitted': 2, 'independent_admitted': 8, 'trades_in_both': 2, 'trades_mismatched': 0}
- advisory `ind.ES.14.setup_only.base`: admitted 10, closed 10, open 0, gross -91.00, cost 38.00, net -129.00 ticks
- advisory `ind.ES.14.setup_threshold.base`: admitted 8, closed 8, open 0, gross -130.00, cost 29.70, net -159.70 ticks
- Surface 4 `full_policy`: `not_defined`

### EURUSD (pips)

Instrument-capacity ledger: models [22, 26, 21, 25, 23]; replay identical: True; hash `b2964bdca2db74ea0e91774eadc97857fed02a735aa469fd0d29904a002e6848`; problems: 0.

- advisory `cap.EURUSD.instrument_policy.base`: opportunities 188, admitted 3, closed 3, open 0, gross 24.14, cost 14.95, net 9.19 pips
- advisory `cap.EURUSD.instrument_policy.double_all_costs`: opportunities 188, admitted 3, closed 3, open 0, gross 24.14, cost 29.90, net -5.76 pips

| ID | Model | Status | Weeks | Routes exercised | Routes unexercised | Replay hash (a = b) |
|---|---|---|---|---|---|---|
| 22 | `long_reversal_tcn_v2_20260525_narrow48` | passed | 2025-03-03 | 1 | 0 | `25b35bfb1a151ace` = `25b35bfb1a151ace` |
| 26 | `short_reversal_xgb_v2_20260525` | passed | 2025-03-03 | 1 | 0 | `83b5dde72781788c` = `83b5dde72781788c` |
| 21 | `long_ote_meta_tcn_champion` | passed | 2025-03-03 | 3 | 1 | `f151e50318b25057` = `f151e50318b25057` |
| 25 | `short_ote_meta_tcn_champion` | passed | 2025-09-08 | 3 | 1 | `fe45981bc47177d6` = `fe45981bc47177d6` |
| 23 | `long_union_tcn_champion` | passed | 2025-03-03 | 3 | 1 | `c776dd4ff1be1210` = `c776dd4ff1be1210` |

#### ID 22 `long_reversal_tcn_v2_20260525_narrow48` - passed

- Exercised routes: `22:reversal:eurusd_ote_reversal_eval_v2`
- Unexercised routes: none
- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
- Signals 10; census {'no_same_side_setup': 1318, 'selected_same_side_setup': 10, 'undeclared_setup_family': 45}; open-position rows 2; capacity reconcile {'capacity_rows': 10, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
- advisory `ind.EURUSD.22.setup_only.base`: admitted 9, closed 8, open 1, gross 137.91, cost 30.70, net 107.21 pips
- advisory `ind.EURUSD.22.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
- Surface 4 `full_policy`: `not_defined`

#### ID 26 `short_reversal_xgb_v2_20260525` - passed

- Exercised routes: `26:reversal:eurusd_ote_reversal_eval_v2`
- Unexercised routes: none
- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
- Signals 15; census {'no_same_side_setup': 1298, 'selected_same_side_setup': 15, 'undeclared_setup_family': 16}; open-position rows 0; capacity reconcile {'capacity_rows': 15, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
- advisory `ind.EURUSD.26.setup_only.base`: admitted 13, closed 13, open 0, gross -66.98, cost 47.95, net -114.93 pips
- advisory `ind.EURUSD.26.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
- Surface 4 `full_policy`: `not_defined`

#### ID 21 `long_ote_meta_tcn_champion` - passed

- Exercised routes: `21:reversal:eurusd_ote_reversal_eval_v2`, `21:continuation:eurusd_ote_continuation_eval_v2`, `21:breakout:eurusd_ote_breakout_eval_v2`
- Unexercised routes: `21:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2` (no_detector_emits_this_setup_family)
- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
- Signals 58; census {'no_same_side_setup': 1394, 'selected_same_side_setup': 58}; open-position rows 2; capacity reconcile {'capacity_rows': 58, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
- advisory `ind.EURUSD.21.setup_only.base`: admitted 41, closed 40, open 1, gross 151.51, cost 164.00, net -12.49 pips
- advisory `ind.EURUSD.21.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
- Surface 4 `full_policy`: `not_defined`

#### ID 25 `short_ote_meta_tcn_champion` - passed

- Exercised routes: `25:reversal:eurusd_ote_reversal_eval_v2`, `25:continuation:eurusd_ote_continuation_eval_v2`, `25:breakout:eurusd_ote_breakout_eval_v2`
- Unexercised routes: `25:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2` (no_detector_emits_this_setup_family)
- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
- Signals 47; census {'ambiguous_multiple_same_side_setups': 3, 'no_same_side_setup': 1408, 'selected_same_side_setup': 41}; open-position rows 2; capacity reconcile {'capacity_rows': 47, 'capacity_admitted': 1, 'independent_admitted': 1, 'trades_in_both': 1, 'trades_mismatched': 0}
- advisory `ind.EURUSD.25.setup_only.base`: admitted 31, closed 30, open 1, gross -13.16, cost 127.00, net -140.16 pips
- advisory `ind.EURUSD.25.setup_threshold.base`: admitted 1, closed 1, open 0, gross 4.10, cost 5.65, net -1.55 pips
- Surface 4 `full_policy`: `not_defined`

#### ID 23 `long_union_tcn_champion` - passed

- Exercised routes: `23:reversal:eurusd_ote_reversal_eval_v2`, `23:continuation:eurusd_ote_continuation_eval_v2`, `23:breakout:eurusd_ote_breakout_eval_v2`
- Unexercised routes: `23:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2` (no_detector_emits_this_setup_family)
- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
- Signals 58; census {'no_same_side_setup': 1394, 'selected_same_side_setup': 58}; open-position rows 2; capacity reconcile {'capacity_rows': 58, 'capacity_admitted': 2, 'independent_admitted': 2, 'trades_in_both': 2, 'trades_mismatched': 0}
- advisory `ind.EURUSD.23.setup_only.base`: admitted 41, closed 40, open 1, gross 151.51, cost 164.00, net -12.49 pips
- advisory `ind.EURUSD.23.setup_threshold.base`: admitted 2, closed 2, open 0, gross 20.04, cost 9.30, net 10.74 pips
- Surface 4 `full_policy`: `not_defined`


<a id="section3_smoke_replay_20261002_v2"></a>

## Earlier pass: `section3_smoke_replay_20261002_v2.md` (diff against v3)

*Merged 2026-10-09. Applying this diff to the v3 text above reproduces `docs/section3_smoke_replay_20261002_v2.md` exactly.*

```diff
--- section3_smoke_replay_20261002_v3.md
+++ section3_smoke_replay_20261002_v2.md
@@ -3,13 +3,11 @@
 Deterministic smoke replay (R4) of the 16 selected versions on engine `b1-v5`, B4 v2 freeze, route bindings v2 and B3 protocol v3. Windows were chosen from setup presence and the calendar only and hashed before any outcome was computed. P&L and trade counts are advisory: recorded, not interpreted. All 16 full policies are `not_defined` (no ledger, no rows, no totals). ES and EURUSD are reported separately and never combined.
 
 - ES windows: `artifacts/s3smoke_20261002_v1/windows.json` sha256 `9db6421a06bcf04b9e83b0453204ef0d13f3f029773509da0e8e0f4d66f0956b`
-- EURUSD windows: `artifacts/s3smoke_eurusd_20261002_v2/windows.json` sha256 `7e40460772dd06123dbf3b47cc99418629d2ca8f5272f93548f791373792fac8`
+- EURUSD windows: `artifacts/s3smoke_eurusd_20261002_v1/windows.json` sha256 `af937aeb4e9b5293a2c2a3d1a10ffa40bcccdacdd660cc45beffa4b66090aaea`
 - Protocol sha256 `7e5189cdd1543005e5035e940a168d3d22e233b3039aac899077a303b0a7dcc4`; B4 v2 sha256 `b9a7bc674b2b38118fcc2187b8c94c9f1aec6b2d84869572244f33f5abd6cb5b`; route bindings v2 sha256 `2473e44e4054068eac4fda1219586401c300b3a2e7a3aa54202e08bf873045d6`
 - Outputs: `<stream root>/a` and `/b` (two independent processes); per-model audits `docs/audits/section3_smoke_<model_id>_20261002.json` (ES) and `..._20261002_v2.json` (EURUSD, when run)
 
 EURUSD calendar: `configs/eurusd_5m_expected_open_calendar_v1.json` (`eurusd_5m_feed_clock_week_v1`, sha256 `5db00e42d5513af92943184a7c566a5f44782700258a72a1f889c2d9d884d139`), a provisional rule-based calendar authorized by the user on 2026-10-02 and pinned before any EURUSD outcome was computed. The ES section below restates the unchanged first-pass evidence.
-
-Pass 2 (user-authorized 2026-10-02): for IDs 22 and 26 the production runner also scored the reversal confirmation bars that their saved held-out predictions omit. Those scores cannot be compared with saved predictions; every saved bar in the same window was compared and matched. IDs 21, 25 and 23 model outputs are unchanged from the first EURUSD pass; the EURUSD instrument-capacity ledger was rebuilt with all five models.
 
 ## ES (ticks)
 
@@ -144,38 +142,32 @@
 
 ## EURUSD (pips)
 
-Instrument-capacity ledger: models [22, 26, 21, 25, 23]; replay identical: True; hash `b2964bdca2db74ea0e91774eadc97857fed02a735aa469fd0d29904a002e6848`; problems: 0.
-
-- advisory `cap.EURUSD.instrument_policy.base`: opportunities 188, admitted 3, closed 3, open 0, gross 24.14, cost 14.95, net 9.19 pips
-- advisory `cap.EURUSD.instrument_policy.double_all_costs`: opportunities 188, admitted 3, closed 3, open 0, gross 24.14, cost 29.90, net -5.76 pips
+Instrument-capacity ledger: models [21, 25, 23]; replay identical: True; hash `a3546ac91dfe735246d975d568e8092f9562c40512a00c38d705eb7a327ad9c3`; problems: 0.
+
+- advisory `cap.EURUSD.instrument_policy.base`: opportunities 163, admitted 3, closed 3, open 0, gross 24.14, cost 14.95, net 9.19 pips
+- advisory `cap.EURUSD.instrument_policy.double_all_costs`: opportunities 163, admitted 3, closed 3, open 0, gross 24.14, cost 29.90, net -5.76 pips
 
 | ID | Model | Status | Weeks | Routes exercised | Routes unexercised | Replay hash (a = b) |
 |---|---|---|---|---|---|---|
-| 22 | `long_reversal_tcn_v2_20260525_narrow48` | passed | 2025-03-03 | 1 | 0 | `25b35bfb1a151ace` = `25b35bfb1a151ace` |
-| 26 | `short_reversal_xgb_v2_20260525` | passed | 2025-03-03 | 1 | 0 | `83b5dde72781788c` = `83b5dde72781788c` |
+| 22 | `long_reversal_tcn_v2_20260525_narrow48` | failed (diagnostic) | - | 0 | 1 | - |
+| 26 | `short_reversal_xgb_v2_20260525` | failed (diagnostic) | - | 0 | 1 | - |
 | 21 | `long_ote_meta_tcn_champion` | passed | 2025-03-03 | 3 | 1 | `f151e50318b25057` = `f151e50318b25057` |
 | 25 | `short_ote_meta_tcn_champion` | passed | 2025-09-08 | 3 | 1 | `fe45981bc47177d6` = `fe45981bc47177d6` |
 | 23 | `long_union_tcn_champion` | passed | 2025-03-03 | 3 | 1 | `c776dd4ff1be1210` = `c776dd4ff1be1210` |
 
-### ID 22 `long_reversal_tcn_v2_20260525_narrow48` - passed
-
-- Exercised routes: `22:reversal:eurusd_ote_reversal_eval_v2`
-- Unexercised routes: none
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 10; census {'no_same_side_setup': 1318, 'selected_same_side_setup': 10, 'undeclared_setup_family': 45}; open-position rows 2; capacity reconcile {'capacity_rows': 10, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.22.setup_only.base`: admitted 9, closed 8, open 1, gross 137.91, cost 30.70, net 107.21 pips
-- advisory `ind.EURUSD.22.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
-- Surface 4 `full_policy`: `not_defined`
-
-### ID 26 `short_reversal_xgb_v2_20260525` - passed
-
-- Exercised routes: `26:reversal:eurusd_ote_reversal_eval_v2`
-- Unexercised routes: none
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 15; census {'no_same_side_setup': 1298, 'selected_same_side_setup': 15, 'undeclared_setup_family': 16}; open-position rows 0; capacity reconcile {'capacity_rows': 15, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.26.setup_only.base`: admitted 13, closed 13, open 0, gross -66.98, cost 47.95, net -114.93 pips
-- advisory `ind.EURUSD.26.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
-- Surface 4 `full_policy`: `not_defined`
+### ID 22 `long_reversal_tcn_v2_20260525_narrow48` - failed
+
+- Failures: ValueError('no eligible window: no recorded held-out prediction has a detector-selected same-side setup match on a declared route, so no route can be exercised'); ValueError('no eligible window: no recorded held-out prediction has a detector-selected same-side setup match on a declared route, so no route can be exercised')
+- Exercised routes: none
+- Unexercised routes: `22:reversal:eurusd_ote_reversal_eval_v2` (no_eligible_week_in_held_out_period)
+- Unexercised calendar elements: overnight_segment (no_eligible_week_in_held_out_period), london_new_york_segment (no_eligible_week_in_held_out_period), planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
+
+### ID 26 `short_reversal_xgb_v2_20260525` - failed
+
+- Failures: ValueError('no eligible window: no recorded held-out prediction has a detector-selected same-side setup match on a declared route, so no route can be exercised'); ValueError('no eligible window: no recorded held-out prediction has a detector-selected same-side setup match on a declared route, so no route can be exercised')
+- Exercised routes: none
+- Unexercised routes: `26:reversal:eurusd_ote_reversal_eval_v2` (no_eligible_week_in_held_out_period)
+- Unexercised calendar elements: overnight_segment (no_eligible_week_in_held_out_period), london_new_york_segment (no_eligible_week_in_held_out_period), planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
 
 ### ID 21 `long_ote_meta_tcn_champion` - passed
 
```


<a id="section3_smoke_replay_20261002"></a>

## Earlier pass: `section3_smoke_replay_20261002.md` (diff against v3)

*Merged 2026-10-09. Applying this diff to the v3 text above reproduces `docs/section3_smoke_replay_20261002.md` exactly.*

```diff
--- section3_smoke_replay_20261002_v3.md
+++ section3_smoke_replay_20261002.md
@@ -2,14 +2,9 @@
 
 Deterministic smoke replay (R4) of the 16 selected versions on engine `b1-v5`, B4 v2 freeze, route bindings v2 and B3 protocol v3. Windows were chosen from setup presence and the calendar only and hashed before any outcome was computed. P&L and trade counts are advisory: recorded, not interpreted. All 16 full policies are `not_defined` (no ledger, no rows, no totals). ES and EURUSD are reported separately and never combined.
 
-- ES windows: `artifacts/s3smoke_20261002_v1/windows.json` sha256 `9db6421a06bcf04b9e83b0453204ef0d13f3f029773509da0e8e0f4d66f0956b`
-- EURUSD windows: `artifacts/s3smoke_eurusd_20261002_v2/windows.json` sha256 `7e40460772dd06123dbf3b47cc99418629d2ca8f5272f93548f791373792fac8`
+- Windows: `artifacts/s3smoke_20261002_v1/windows.json` sha256 `9db6421a06bcf04b9e83b0453204ef0d13f3f029773509da0e8e0f4d66f0956b`
 - Protocol sha256 `7e5189cdd1543005e5035e940a168d3d22e233b3039aac899077a303b0a7dcc4`; B4 v2 sha256 `b9a7bc674b2b38118fcc2187b8c94c9f1aec6b2d84869572244f33f5abd6cb5b`; route bindings v2 sha256 `2473e44e4054068eac4fda1219586401c300b3a2e7a3aa54202e08bf873045d6`
-- Outputs: `<stream root>/a` and `/b` (two independent processes); per-model audits `docs/audits/section3_smoke_<model_id>_20261002.json` (ES) and `..._20261002_v2.json` (EURUSD, when run)
-
-EURUSD calendar: `configs/eurusd_5m_expected_open_calendar_v1.json` (`eurusd_5m_feed_clock_week_v1`, sha256 `5db00e42d5513af92943184a7c566a5f44782700258a72a1f889c2d9d884d139`), a provisional rule-based calendar authorized by the user on 2026-10-02 and pinned before any EURUSD outcome was computed. The ES section below restates the unchanged first-pass evidence.
-
-Pass 2 (user-authorized 2026-10-02): for IDs 22 and 26 the production runner also scored the reversal confirmation bars that their saved held-out predictions omit. Those scores cannot be compared with saved predictions; every saved bar in the same window was compared and matched. IDs 21, 25 and 23 model outputs are unchanged from the first EURUSD pass; the EURUSD instrument-capacity ledger was rebuilt with all five models.
+- Outputs: `artifacts/s3smoke_20261002_v1/a` and `/b` (two independent processes); per-model audits `docs/audits/section3_smoke_<model_id>_20261002.json`
 
 ## ES (ticks)
 
@@ -144,65 +139,32 @@
 
 ## EURUSD (pips)
 
-Instrument-capacity ledger: models [22, 26, 21, 25, 23]; replay identical: True; hash `b2964bdca2db74ea0e91774eadc97857fed02a735aa469fd0d29904a002e6848`; problems: 0.
-
-- advisory `cap.EURUSD.instrument_policy.base`: opportunities 188, admitted 3, closed 3, open 0, gross 24.14, cost 14.95, net 9.19 pips
-- advisory `cap.EURUSD.instrument_policy.double_all_costs`: opportunities 188, admitted 3, closed 3, open 0, gross 24.14, cost 29.90, net -5.76 pips
+Stream status: `not_run`. EURUSD smoke replay is not_run until one versioned EURUSD 5m expected-open calendar is pinned (B3 v3 smoke_replay.eurusd_calendar_rule, decision D6). Frozen bytes were verified; nothing was scored or simulated.
 
 | ID | Model | Status | Weeks | Routes exercised | Routes unexercised | Replay hash (a = b) |
 |---|---|---|---|---|---|---|
-| 22 | `long_reversal_tcn_v2_20260525_narrow48` | passed | 2025-03-03 | 1 | 0 | `25b35bfb1a151ace` = `25b35bfb1a151ace` |
-| 26 | `short_reversal_xgb_v2_20260525` | passed | 2025-03-03 | 1 | 0 | `83b5dde72781788c` = `83b5dde72781788c` |
-| 21 | `long_ote_meta_tcn_champion` | passed | 2025-03-03 | 3 | 1 | `f151e50318b25057` = `f151e50318b25057` |
-| 25 | `short_ote_meta_tcn_champion` | passed | 2025-09-08 | 3 | 1 | `fe45981bc47177d6` = `fe45981bc47177d6` |
-| 23 | `long_union_tcn_champion` | passed | 2025-03-03 | 3 | 1 | `c776dd4ff1be1210` = `c776dd4ff1be1210` |
+| 22 | `long_reversal_tcn_v2_20260525_narrow48` | not_run | - | 0 | 1 | - |
+| 26 | `short_reversal_xgb_v2_20260525` | not_run | - | 0 | 1 | - |
+| 21 | `long_ote_meta_tcn_champion` | not_run | - | 0 | 4 | - |
+| 25 | `short_ote_meta_tcn_champion` | not_run | - | 0 | 4 | - |
+| 23 | `long_union_tcn_champion` | not_run | - | 0 | 4 | - |
 
-### ID 22 `long_reversal_tcn_v2_20260525_narrow48` - passed
+### ID 22 `long_reversal_tcn_v2_20260525_narrow48` - not_run
 
-- Exercised routes: `22:reversal:eurusd_ote_reversal_eval_v2`
-- Unexercised routes: none
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 10; census {'no_same_side_setup': 1318, 'selected_same_side_setup': 10, 'undeclared_setup_family': 45}; open-position rows 2; capacity reconcile {'capacity_rows': 10, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.22.setup_only.base`: admitted 9, closed 8, open 1, gross 137.91, cost 30.70, net 107.21 pips
-- advisory `ind.EURUSD.22.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
-- Surface 4 `full_policy`: `not_defined`
+- Unexercised routes (stream not run): `22:reversal:eurusd_ote_reversal_eval_v2`
 
-### ID 26 `short_reversal_xgb_v2_20260525` - passed
+### ID 26 `short_reversal_xgb_v2_20260525` - not_run
 
-- Exercised routes: `26:reversal:eurusd_ote_reversal_eval_v2`
-- Unexercised routes: none
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 15; census {'no_same_side_setup': 1298, 'selected_same_side_setup': 15, 'undeclared_setup_family': 16}; open-position rows 0; capacity reconcile {'capacity_rows': 15, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.26.setup_only.base`: admitted 13, closed 13, open 0, gross -66.98, cost 47.95, net -114.93 pips
-- advisory `ind.EURUSD.26.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
-- Surface 4 `full_policy`: `not_defined`
+- Unexercised routes (stream not run): `26:reversal:eurusd_ote_reversal_eval_v2`
 
-### ID 21 `long_ote_meta_tcn_champion` - passed
+### ID 21 `long_ote_meta_tcn_champion` - not_run
 
-- Exercised routes: `21:reversal:eurusd_ote_reversal_eval_v2`, `21:continuation:eurusd_ote_continuation_eval_v2`, `21:breakout:eurusd_ote_breakout_eval_v2`
-- Unexercised routes: `21:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2` (no_detector_emits_this_setup_family)
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 58; census {'no_same_side_setup': 1394, 'selected_same_side_setup': 58}; open-position rows 2; capacity reconcile {'capacity_rows': 58, 'capacity_admitted': 0, 'independent_admitted': 0, 'trades_in_both': 0, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.21.setup_only.base`: admitted 41, closed 40, open 1, gross 151.51, cost 164.00, net -12.49 pips
-- advisory `ind.EURUSD.21.setup_threshold.base`: admitted 0, closed 0, open 0, gross 0.00, cost 0.00, net 0.00 pips
-- Surface 4 `full_policy`: `not_defined`
+- Unexercised routes (stream not run): `21:reversal:eurusd_ote_reversal_eval_v2`, `21:continuation:eurusd_ote_continuation_eval_v2`, `21:breakout:eurusd_ote_breakout_eval_v2`, `21:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2`
 
-### ID 25 `short_ote_meta_tcn_champion` - passed
+### ID 25 `short_ote_meta_tcn_champion` - not_run
 
-- Exercised routes: `25:reversal:eurusd_ote_reversal_eval_v2`, `25:continuation:eurusd_ote_continuation_eval_v2`, `25:breakout:eurusd_ote_breakout_eval_v2`
-- Unexercised routes: `25:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2` (no_detector_emits_this_setup_family)
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 47; census {'ambiguous_multiple_same_side_setups': 3, 'no_same_side_setup': 1408, 'selected_same_side_setup': 41}; open-position rows 2; capacity reconcile {'capacity_rows': 47, 'capacity_admitted': 1, 'independent_admitted': 1, 'trades_in_both': 1, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.25.setup_only.base`: admitted 31, closed 30, open 1, gross -13.16, cost 127.00, net -140.16 pips
-- advisory `ind.EURUSD.25.setup_threshold.base`: admitted 1, closed 1, open 0, gross 4.10, cost 5.65, net -1.55 pips
-- Surface 4 `full_policy`: `not_defined`
+- Unexercised routes (stream not run): `25:reversal:eurusd_ote_reversal_eval_v2`, `25:continuation:eurusd_ote_continuation_eval_v2`, `25:breakout:eurusd_ote_breakout_eval_v2`, `25:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2`
 
-### ID 23 `long_union_tcn_champion` - passed
+### ID 23 `long_union_tcn_champion` - not_run
 
-- Exercised routes: `23:reversal:eurusd_ote_reversal_eval_v2`, `23:continuation:eurusd_ote_continuation_eval_v2`, `23:breakout:eurusd_ote_breakout_eval_v2`
-- Unexercised routes: `23:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2` (no_detector_emits_this_setup_family)
-- Unexercised calendar elements: planned_closure (no_eligible_week_in_held_out_period), dst_transition (no_eligible_week_in_held_out_period)
-- Signals 58; census {'no_same_side_setup': 1394, 'selected_same_side_setup': 58}; open-position rows 2; capacity reconcile {'capacity_rows': 58, 'capacity_admitted': 2, 'independent_admitted': 2, 'trades_in_both': 2, 'trades_mismatched': 0}
-- advisory `ind.EURUSD.23.setup_only.base`: admitted 41, closed 40, open 1, gross 151.51, cost 164.00, net -12.49 pips
-- advisory `ind.EURUSD.23.setup_threshold.base`: admitted 2, closed 2, open 0, gross 20.04, cost 9.30, net 10.74 pips
-- Surface 4 `full_policy`: `not_defined`
+- Unexercised routes (stream not run): `23:reversal:eurusd_ote_reversal_eval_v2`, `23:continuation:eurusd_ote_continuation_eval_v2`, `23:breakout:eurusd_ote_breakout_eval_v2`, `23:breakout_sustained:eurusd_ote_breakout_sustained_eval_v2`
```

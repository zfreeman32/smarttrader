# Section 4 progress: "Close the B5 live-parity evidence gap"

Last updated: 2026-10-08 (B1-outcome label research complete: no setup-gated or every-bar variant reaches >= 6 trades/week with positive B1 returns; see "B1-outcome label research").

Operative protocol: **v2**, `configs/section4_parity_protocol_v2.json` (sha256 `e25b5e48905d44de5d6c4369a5ac24b49bf711074722710b149494fccb34eff5`) with lock `configs/section4_parity_protocol_v2.lock.json` (sha256 `f9b65ced680b65e46ca8f84167c4abd5de15a9372d3d88f98369d2304acbff38`). Publisher: `scripts/publish_section4_parity_protocol_v2.py`.

- v2 = v1 plus the Q-D8 changes; tolerances and segment shapes are unchanged.
- v1 (`55f9d226…`, lock `d9550d28…`) is superseded but kept.
- Changing tolerances or shapes needs a v3 that the user approves.

## Scope and rules

- Roster: ES FRVP 2, 3, 5, 10; ES ICT 12, 13, 14, 15, 16, 18, 19; EURUSD OTE 21, 22, 23, 25, 26.
- Reserves 6, 7, 11, 17 and 1 are out of scope. ID 1 may appear only as the composition negative control.
- Full policy is `not_defined` for all 16 models, so the decision compared is the threshold decision.
- Do not start section 5+ (B6, A12, A13, B7).
- Immutable: the B4 v2 freeze and its bytes, selection v0/v1, route manifests v1/v2 and their locks, B3 v1/v2, B1 contracts, b1-v4/b1-v5, repaired prepared data, all section 3 artifacts, and the 2026-09-29 B5 audit (`docs/B5_B6_live_feature_parity_20260929.md`, `docs/audits/b5_b6_live_feature_parity_20260929_v2.json`).
- Never touch the running app, installed manifests or the registry. Open the live DB `ote_live/runtime_data/live_market_data.sqlite3` read-only (`file:...?mode=ro`); it was last written 2026-09-22.
- Interpreter: `ote_venv/Scripts/python.exe`. Run pytest with `-p no:cacheprovider --basetemp <short dir>`.
- Journals: FRVP → `docs/FRVP_experiment_journal.md`; ICT → `docs/ICT_execution_journal.md`; OTE → the section 4 summary doc; runtime changes → `docs/live_app_change_journal.md`.

## Requirement status

| Req | Status | Evidence | Next |
|---|---|---|---|
| P0 protocol | **done: v1 locked** (D1–D9 approved as recommended, 2026-10-02) | `configs/section4_parity_protocol_v1.json` + `.lock.json` (hashes above). Publisher `scripts/publish_section4_parity_protocol_v1.py` (sha256 `6d5e8906…`) refuses to overwrite. The approved draft stays unchanged and is hashed as an input. | — |
| R1 raw snapshots | **done** (snapshot schema `s4-raw-preimpute-snapshot-v1`, value-neutral in all 3 groups). D8 and D6 are implemented opt-in. Q-D8 is approved: the parity path runs with D8 on (protocol v2). | See "Prompt 2" below | — |
| R2 app-path replay | **harness built and validated (prompt 3)**; timing measured | Per-build medians (8,000-bar window, 3 builds each): ES_FRVP {2,3,5,10} 12.7 s, ES_ICT {12–19} 1.9 s, EURUSD union 78.6 s (`strategy_signals` ~77 s). Files: `docs/audits/section4_build_timing_{ES_FRVP,ES_ICT,EURUSD}_20261002.json`, tool `docs/audits/section4_tools_20261002/time_builds.py`, log `logs/section4_timing_20261002.log`. | Prompt 4: windows.json by rule, anchor check, segments |
| R3 row-level comparison | **comparator built and validated (prompt 3)** | `scripts/section4_parity_compare.py`; protocol §3, §4, §9 | Prompt 4 runs and classification |
| R4 edge-case coverage | matrix drafted; F injections implemented and tested (prompt 3) | Protocol §5. No overlapping data exists for the S layer (D2). No live OTE event producer exists (D5). | — |
| R5 per-version reports | schema drafted | Protocol §11 | — |
| R6 retrain guard | rule drafted | Protocol §12 | — |

## Per-model status

Prompt 4 matrix running (ES first, then EURUSD); live state is in `matrix_state.json`. The 09-29 status (diagnostic) still stands until superseded. The "Stage-manifest base" column gives the installed manifest family to clone with B4 v2 artifacts.

| ID | Model | Stream | Backend | Stage-manifest base | Section 4 status | Notes |
|---:|---|---|---|---|---|---|
| 2 | frvp_long_continuation_xgb_v1 | ES | xgboost | frvp_es_paper_signal_20260816 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 3 | frvp_long_meta_xgb_v1 | ES | xgboost | frvp_es_paper_signal_20260816 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 5 | frvp_long_continuation_setup2_xgb_v1 | ES | xgboost | frvp_es_setup_family_20260829 | **fail:app_producer_bug** (report 2026-10-05) | |
| 10 | frvp_long_reversal_xgb_v1 | ES | xgboost | frvp_es_paper_signal_20260816 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 12 | ict_long_meta_xgb_v1 | ES | xgboost | ict_es_paper_signal_20260813 | **fail:app_producer_bug** (report 2026-10-05) | |
| 13 | ict_long_reversal_xgb_v1 | ES | xgboost | ict_es_paper_signal_20260813 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 14 | ict_short_continuation_xgb_v1 | ES | xgboost | ict_es_paper_signal_20260813 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 15 | ict_short_meta_xgb_v1 | ES | xgboost | ict_es_paper_signal_20260813 | **fail:app_producer_bug** (report 2026-10-05) | |
| 16 | ict_short_reversal_xgb_v1 | ES | xgboost | ict_es_paper_signal_20260813 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 18 | ict_short_reversal_ifvg_reversal_xgb_v1 | ES | xgboost | ict_short_setup_family_20260811_audit | **fail:training_side_noncausal** (report 2026-10-05) | |
| 19 | ict_short_reversal_sweep_reclaim_xgb_v1 | ES | xgboost | ict_short_setup_family_20260811_audit | **fail:training_side_noncausal** (report 2026-10-05) | |
| 21 | long_ote_meta_tcn_champion | EURUSD | tcn | long_ote_meta_tcn_champion | **fail:training_side_noncausal** (report 2026-10-05) | S2 blocked (no live OTE producer, D5) |
| 22 | long_reversal_tcn_v2_20260525_narrow48 | EURUSD | tcn | long_reversal_tcn_v2_20260525_narrow48 | **fail:app_producer_bug** (report 2026-10-05) | confirmation bars are app-vs-reference only |
| 23 | long_union_tcn_champion | EURUSD | tcn | none installed; D9 → long_ote_union_tcn_candidate_20260523 | **fail:training_side_noncausal** (report 2026-10-05) | |
| 25 | short_ote_meta_tcn_champion | EURUSD | tcn | short_ote_meta_tcn_champion | **fail:training_side_noncausal** (report 2026-10-05) | |
| 26 | short_reversal_xgb_v2_20260525 | EURUSD | xgboost | short_reversal_xgb_v2_20260525 | **fail:training_side_noncausal** (report 2026-10-05) | confirmation bars are app-vs-reference only |

## Prompt 1 (2026-10-02): inventory and protocol draft - COMPLETE

Read-only, except for these new files:

- `docs/section4_inventory_20261002.md` (sha256 `15be36a5e4e2239ed416ce3ba36627d62b50107c0d1bac87ade377aeb391485b`)
- `docs/audits/section4_feature_inventory_20261002.json` (sha256 `d97a949be8150c4a5f9513f3ae4a2fe6f53ab0aaf4e8c68a284dbbf49923bb9f`): per feature, the producer, module sha256, native timeframe, HTF flag, tolerance class and frozen fill.
- Generator scripts in `docs/audits/section4_tools_20261002/`:
  - `attrib.py` `ab13451c…`
  - `inventory.py` `16ce5c1e…`
  - `attrib_es.json` `e6bb3b0f…`
  - rerun with `ote_venv/Scripts/python.exe <tools>/inventory.py <tools>/attrib_es.json <out>`
- `docs/section4_parity_protocol_v1_draft_20261002.md` (unlocked)
- this file

No runtime code, manifest, journal or test changed, so no tests were run.

Key findings, none fixed or classified yet:

1. Evaluated bars carry no provenance: 400/400 recent snapshots per stream have `is_complete=None`, because the store-fetch SELECT omits those columns.
2. The app has no live OTE setup-event producer.
3. ES fill ordering depends on engine composition.
4. A EURUSD strategy timeout falls back to fill 0.
5. Recorded and training sources do not overlap (ES 06-19 vs 07-29; EURUSD 03-25 vs 04-08).
6. A working-tree producer edit (`continuation_pullback.py`) needs a recompute anchor check.
7. EURUSD live weekend boundaries differ from calendar v1.
8. All 16 models need 8,000-bar windows. `frvp_context` costs ~21 s per build, which explains the 09-29 resource bound.

## Prompt 1b (2026-10-02): protocol v1 approved, published and locked - COMPLETE

The user approved D1–D9 as recommended. That includes:

- D2: the outward backfill fetch for source-parity overlap is approved. It uses a separate client ID and an isolated DB, never the running app's session.
- D5: EURUSD S2 stays incomplete until a live OTE event producer exists.
- D8: the provenance fix goes into R1.

Settled while locking (the draft reserved these for after timing):

- **Engine composition.** An ES all-model union engine cannot be built. `build_runtime_feature_config` raises "multiple z-score windows (60, 100)", because the runtime pruning path supports one window per transform family. Every single model, the FRVP group {2,3,5,10}, the ICT group {12–19} and the EURUSD union {21–26} all build.
  - Protocol v1 therefore uses one shared producer pass per **group** per segment. This matches the app's real group topology and still avoids 16 separate passes.
  - Deviation from D7's "stream-union" wording: the intent (compute once, prove composition invariance, B6 groups FRVP/ICT all-XGB) is unchanged.
  - It is also a B6 constraint: ES must run as at least two groups.
- **Segment shape.**
  - ES: 48 bars before the event, 96 after (145 endpoints).
  - EURUSD: 32 before, 64 after (97 endpoints).
  - The pre-event count covers the longest consumed window (ES 40 rows, EURUSD 32).
- **Chunk size.** Each group uses at most 48 endpoints per chunk and at most 25 minutes: FRVP 48 (~10 min), ICT 48 (~1.5 min), EURUSD 19 (~25 min).
- **Estimated segment time.**
  - ES: ~35 min (FRVP 30.6 + ICT 4.5).
  - EURUSD: ~127 min.
  - Rough total for the matrix: ES ~12 segments ≈ 7 h, EURUSD ~8 segments ≈ 17 h, plus F re-runs. Every segment runs in the background and is checkpointed.

No runtime code, manifest, journal or frozen artifact changed. New files: the protocol JSON, its lock, the publisher, the timing tool, and three timing JSONs. No tests were run, because no shared code changed.

## Prompt 2 (2026-10-02): R1 raw pre-imputation snapshots - COMPLETE, stopped on Q-D8

### Runtime change

Versioned, disabled by default. The running app was not restarted and no manifest or registry was touched.

| File | sha256 (prefix) | Change |
|---|---|---|
| `features/builder.py` | `060fc6b4` | `build(..., pre_fill_observer=None)`: a read-only hook after all producers (feature sets, transforms, inf→NaN) and before `fillna_numeric`; passes a column→producer map |
| `ote_live/features/raw_snapshots.py` (new) | `9f506cde` | `capture_pre_fill`, `RawSnapshotRecorder`, `consumed_rows`, `es_app_calendar`, `eurusd_calendar_v1` (lock-verified, provisional) |
| `ote_live/features/incremental_engine.py` | `76b5c815` | `enable_raw_capture(rows)` (0 = off, and `build()` is called exactly as before), `last_raw_capture`, `fill_mode` |
| `ote_live/ingestion/signals.py` | `d0839157` | see below |

`signals.py` changes:

- `attach_raw_snapshot_recorder`, and `MultiGroupLiveSignalProcessor.attach_raw_snapshot_recorders({group: recorder})`.
- The record call sits in `prepare_ingested_bar_snapshot`, after the feature build. Its failures go to `recorder.errors` and never reach scoring.
- `_store_provenance_for` reads canonical-row provenance and `source_row_idx` read-only.
- D8 opt-in: `carry_bar_provenance=False`, and `_bar_from_row(row, carry_provenance=False)`. The SELECTs now include the provenance columns.

The collectors are not wired. B6 attaches recorders through the API, and the caller flushes per chunk.

### Snapshot schema `s4-raw-preimpute-snapshot-v1`

Per group, `recorder.flush(dir, prefix)` writes these tables:

| Table | Content |
|---|---|
| `<prefix>.builds.parquet` | build_id = snapshot_id, endpoint, `source_row_idx`, `collection_version`, cache_hit, fill_mode, missing-producer/fallback/nonnumeric/skipped-strategy lists, calendar_id |
| `rows` | `capture_rows` tail rows per build; values in `f:<feature>` columns with NaN preserved. Also: store_* provenance (completion, version, feed, first/last observed, kind, contract, `source_row_idx`), engine_* bar fields as the engine saw them, HTF helper age/complete columns, gap_before_slots / gap_expected_open_slots / gap_state (none, closure, gap, window_start, calendar_unavailable, sequence_invalid), roll_boundary |
| `feature_states` | producer, producer module and sha256, `produced`, group_state (produced, missing_producer, fallback_zero_default) |
| `model_states` | warmup fields, contract status and reasons, consumed rows, `consumed_rows_captured` |
| `model_feature_states` | every input-contract category per model and feature (available, event_dependent_missing, not_yet_observed, missing_producer, unverified_producer, missing_history_or_input, stale_input) |
| `<prefix>.index.json` | the per-model selected-feature index (feature_positions, consumed/capture rows, threshold, B4 v2 artifact sha256 from the A1 collection contract), producer modules, errors, table hashes |

- Values are stored once per group, so nothing is duplicated across models.
- `capture_rows` = max over the group's models of (consumed rows, `context_rows`+1).

Notes:

- The HTF as-of alignment inside a producer is part of that producer and is not undone.
- Seed offsets apply after capture; no roster model selects one.
- No D6 rule exists, so EURUSD completion stays unknown.

### Value-neutrality evidence (B4 v2 bytes, all 16 models)

`scripts/run_section4_r1_value_neutrality.py` (sha256 `430bd208`):

- Staged in-memory manifests: an installed base, plus the B4 v2 bytes verified by `verify_bytes` and loaded with the frozen runtime constructors, plus a staging `registry_path`. The staging path is needed because the FRVP paper-signal guard fails closed on the installed registry path.
- Isolated SQLite stores under `artifacts/s4parity_20261002_v1/r1_value_neutrality/<group>/{off,on}.sqlite3`, with an 8,000-bar training-source tail and 3 endpoints.

**Result:** in all three groups, feature frames, scaled inputs, raw and calibrated scores, threshold decisions and persisted snapshots are bit-identical (`value_neutrality.json` sha256 `bb7914e76129d8a1194910b8ec678c63a1087c92306c00849216615c60ace316`; log `logs/section4_r1_value_neutrality_20261002.log`).

| Group | Predictions | Captured rows | Feature columns | Raw NaN cells | Recorder errors |
|---|---:|---:|---:|---:|---:|
| ES_ICT | 21 | 72 | 300 | 2,100 | 0 |
| ES_FRVP | 12 | 120 | 362 | 9,096 | 0 |
| EURUSD_OTE | 15 | 96 | 253 | 342 | 0 |

- EURUSD gap state uses the **provisional** calendar v1.

Tests: `tests/test_ote_live_raw_snapshots.py` (6 focused tests) pass: copy-only capture, observer neutrality, processor on/off bit identity, explicit fallback state with NaN (never zero), NaN preserved where the builder fills, and D8 off = legacy bar. Broader suite: run once, `logs/section4_prompt2_full_tests_20261002.log`: 1203 passed, 2 skipped, 15 failed. All 15 are unrelated to this change: the 8 known (feature_parity 3, prediction_replay 3, seed_state 2); phase5_operator_tooling 4 (missing installed manifest `long_ote_champion_v1`); backend_attribution 2 (same failures with the HEAD builder); and ote_xgboost_pipeline 1 (`OTETrainingConfig` kwarg). None of these touch the changed code paths.

### Finding and open decision Q-D8

**Finding.** `features/feature_sets/htf_context.py:47` keeps an HTF bucket only if every constituent has `is_complete == True`. The store-fetch path drops completion (inventory c), so the app computes all 30m/1h HTF features as NaN. Consumers inherit it: `continuation_pullback` reads `htf_*_ema_alignment`, and so do the interactions. Offline sources have no `is_complete` column, so the gate never applies there.

Probe (`docs/audits/section4_tools_20261002/d8_effect_probe.py`, read-only live DB, last runtime window):

| Group | Features changed | Changed cells | NaN at the endpoint |
|---|---|---:|---|
| ES_ICT | 25 of 300 | 33,263 | 27 → 7 |
| ES_FRVP | 111 of 362 | 135,555 | 77 → 10 |
| EURUSD_OTE | 0 | 0 | — |

- The ES window was 6,152 bars, 1,739 of them complete.
- EURUSD has 8,000 bars and every one has `is_complete` NULL. D6 is required before anything changes there.
- Evidence: `docs/audits/section4_d8_effect_{ES_ICT,ES_FRVP,EURUSD_OTE}_20261002.json`.

This contradicts the protocol's D8 premise ("OHLCV/features unchanged"). Under the mismatch taxonomy, the HTF gap is a candidate `app_producer_bug` (provenance loss), not a training-side issue.

Related: historical P-layer windows copied into a parity DB from training sources would carry no completion either. Their HTF would be all-NaN unless completion is set for archived historical bars (also part of Q-D8/D6).

**Q-D8 resolved (2026-10-02):** the user answered "whatever you recommend", so the recommendation is approved and implemented.

- **Protocol v2** (hashes at the top), with the reason in `changes_from_v1`:
  - the parity app path runs with `carry_bar_provenance=True`;
  - B6 must enable D8 and D6;
  - historical training-source bars copied into parity DBs carry `is_complete=True`, `observation_kind='historical'`, `feed_type='historical'`, `source='section4_training_source:<file>'`, and are never live evidence;
  - archived EURUSD rows stay completion-unknown, so their gated HTF inputs are `declared_contract_behavior` candidates.
- **D6** (opt-in, `fmp-historical-chart-completion-v1`):
  - `ote_live/ingestion/provenance.py` (`af60afa3`) adds `apply_fmp_completion_rule`. Completion is set once observed at or after bar end + 90 s; otherwise it stays unknown. `bar_version` is recomputed.
  - `ote_live/ingestion/connector_backfill.py` (`fe984756`) adds `FMPConfig.apply_completion_rule=False`.
- **Before/after:** `tests/test_ote_live_section4_provenance_rules.py` (`b8f5498a`), 3 tests:
  - D8 on with complete bars is bit-identical to the gate-absent (offline) build;
  - D8 off gives all-NaN HTF features with every non-HTF feature identical;
  - D6 is opt-in, and OHLCV is unchanged under it.
- **Focused re-run:** 101 passed (new tests, raw snapshots, normalizer, bar provenance, runtime integration, ES causal contract).
- The collectors were not changed, and the running app was not restarted.
- Journal entries added: live-app change journal, FRVP, ICT.

## Prompt 3 (2026-10-02): parity harness and fault injection - COMPLETE

Harness built; unit validation passes (21 of 21); plumbing for IDs 2 and 22 done. No runtime code, manifest, frozen artifact or journal changed in this prompt.

| File | sha256 (prefix) | Role |
|---|---|---|
| `scripts/section4_parity_harness.py` | `7eb4b3ea` | Extends the R1 script. Protocol v2 lock check; windows (`write_windows`: immutable, `.sha256` sidecar); isolated parity DB; `AppDriver` (real `LiveSignalProcessor`, `carry_bar_provenance=True`, recorder on, runner taps for S4 input/imputed/raw, S5, S6, S7); `run_app_segment` (bar by bar, chunk parts plus `.done` markers, checkpoint via SQLite backup, resume = restart allowed only with `allow_resume`); `feed_program` faults; `producer_failure`; `FrozenReference` and `build_reference` (frozen source history, or `recompute` via `recompute_history`); `plumbing` CLI |
| `scripts/section4_parity_compare.py` | `47602816` | `compare_segment`/`compare_model` (S1, S3 + states, S4 + mask, S5, S6, S7, near band, flips, checks, diffs to `diff/`, summary); `compare_app_runs` (bitwise restart/continuous); `reference_setup_events` + `compare_setup_events` (S2); `compare_source` + `source_impact` (S layer) |
| `tests/test_section4_parity_harness.py` | `2e8ff6d2` | 21 tests |

Harness conventions (record; not protocol changes):

- **Training sources.** ES uses `data/futures_data/ES-5m-tagged.csv`. EURUSD uses `artifacts/b2_ote_labels_20260920_v2/labels.csv.gz` (UTC bar-open). Row position equals the frozen `source_row_idx` (checked per model).
  - The R1 script read `eurusd-5m.csv` as UTC. That file is not UTC: it ends 20:45 against 02:45 UTC in the history. This was harmless for R1 value-neutrality, but the file must not be used for parity.
- **Replay clock.** Copied bars are observed at bar close + 90 s. Faults use later observation times. This holds in isolated DBs only and is never live evidence.
- **Alignment.** Rows are aligned by position, which is what the model consumes. S1 explains any shift. The app-vs-reference `source_row_idx` offset is anchored at the segment's first comparable row.
- **S3 reference stage.** ES compares raw pre-fill values. For EURUSD (`fillna_numeric=True`), app NaN cells are counted `not_compared_ref_post_fill` and compared at the S4 input.
- **Revision fault.** A provisional version (+1 tick, completion unknown) is processed first, then the final version equal to the source is upserted. This happens for the latest bar and for an earlier bar behind the cursor, so the frozen reference stays valid.
- **Producer exceptions.** The app catches a producer exception (`feature_build_failed` health event) and skips the bar. The harness collects health events per part. The comparator fails on `health_errors` and on delivered endpoints that were `not_evaluated`.
- **Evidence flags per mismatch.** `injected_row`, `injected_window`, `injected_before_endpoint`, `after_restart`, `part`/`instance`, `gap_state`, `roll_boundary`, completion and observation kind, HTF helper ages, `app_prefill`, `group_state`, `contract_category`.

Validation (`tests/test_section4_parity_harness.py`):

- **Clean run.** Zero mismatches on every stage.
- **Synthetic defects, each detected.** Value change, identity change, missing snapshot, stale helper, unexplained fallback, wrong artifact hash, score/decision flip, and tampering with a chunk file without a marker update.
- **Equality runs.** Restart mid-session, restart at a chunk boundary, restart every chunk and crash-resume are each bitwise equal to continuous. Resume is refused without `allow_resume`.
- **Faults.** Dropped bars, backfill behind the cursor, provisional-then-final revision, reconnect batch, cold start and producer failure are each logged and detected or surfaced.
- **Other comparators.** S2 and S tests pass. Windows are immutable, and the protocol lock is verified.

### Plumbing (tiny real-data runs; recorded, NOT classified or acted on)

- **Shared setup.**
  - Segment rule: the last 3 bars of the pinned training source (H), with a prefix of 8,000 bars. These are not protocol segments.
  - Execution: chunks of 2 endpoints, run as the full group engine; only the named ID is compared.
  - Windows files: `artifacts/s4parity_20261002_v1/plumbing/windows_{02,22}.json` (`0992b3f8`, `0d22966c`).
- **ID 2 (ES_FRVP group).**
  - Output: `artifacts/s4parity_20261002_v1/plumbing/ES/ES_FRVP/plumbing_02_ES_tail/` (`summary.json` `4e014541`); log `logs/section4_plumbing_02_20261002.log`.
  - Run: the app took 51 s; 3 of 3 endpoints were comparable.
  - Matches: S1 0 mismatches, S3 states 0, S7 0 flips, 0 near-threshold, no failed checks. The B4 v2 artifact hashes were OK on both sides.
  - Mismatches: S3 243 of 19,200 cells across 6 features (producers: `ict_context` 3, `interactions`, `sigma_normalization`, `temporal_context`), max abs 35.0. Also S4_mask 16 cells (1 feature), S4 243, S5 243, and S6 6 (max abs 2.8e-3).
  - S2 compared: 0 app and 0 reference events in the window.
  - Health: 5 `model_input_contract` warnings.
- **ID 22 (EURUSD_OTE union).**
  - Output: `artifacts/s4parity_20261002_v1/plumbing/EURUSD/EURUSD_OTE/plumbing_22_EURUSD_tail/` (`summary.json` `f99b6eb4`); log `logs/section4_plumbing_22_20261002.log`.
  - Run: the app took 231 s; 3 of 3 endpoints were comparable.
  - Result: **no mismatch** on any stage. Cells compared: S1 780, S3 5,746 (14 counted `not_compared_ref_post_fill`), S4/S5 5,760, S6 6, S7 3.
  - S2 is incomplete (D5). The EURUSD calendar v1 is **provisional**.
- **Harness fixes found by plumbing.**
  - The freeze pins the source-history contract as an object plus its canonical `contract_sha256`, not a file hash. The check was corrected.
  - A completed app run is reused, not reopened.

### Tests

- **Focused.** `tests/test_section4_parity_harness.py` + `test_ote_live_raw_snapshots.py` + `test_ote_live_section4_provenance_rules.py`: 30 passed.
- **Broader suite, run once.** `logs/section4_prompt3_full_tests_20261002.log`: 1227 passed, 2 skipped, 15 failed. These are the same 15 pre-existing failures as prompt 2 (same files and counts). The pass count rose by 24 (21 harness tests plus 3 provenance-rule tests).

No runtime code, manifest, registry, frozen artifact or journal changed. No journal entries were made, because no finding was classified and no app change was made.

## Prompt 4 (2026-10-02): full parity run - IN PROGRESS (background job)

### How to resume (read this first)

- Job: `ote_venv/Scripts/python.exe -u scripts/section4_parity_matrix.py run`, started detached (PowerShell `Start-Process`, pid 14240) at about 2026-10-02 run start. It holds `artifacts/s4parity_20261002_v1/matrix.lock` (pid) and refuses a second instance.
- Log: `logs/section4_matrix_20261002.log` (stderr: `.err.log`). Check with a 5-20 line tail only.
- Markers: `SEGMENT <id> <group> COMPLETE`, `STREAM ES COMPLETE`, `STREAM EURUSD COMPLETE`, `MATRIX COMPLETE`. Failures: `SEGMENT ... FAILED`, `ANCHOR ... FAILED`, `RESTART GATE ... FAILED`.
- State: `artifacts/s4parity_20261002_v1/matrix_state.json` (per segment|group: running -> compared | failed). `python scripts/section4_parity_matrix.py status` prints counts plus running/failed items.
- If the process is dead before `MATRIX COMPLETE` (no live pid in the lock, or the lock is gone without the marker): re-run the same command detached. Finished segments are skipped (state `compared` + `summary.json`). An interrupted segment resumes from its verified checkpoint only if that stream's restart gate passed; otherwise it fails with "restart test" and needs a decision (record it and ask).
- Estimated wall time: ES ~8 h, then EURUSD ~21 h (EURUSD ~127 min per 97-endpoint segment; cold start is cheap).
- After `MATRIX COMPLETE`: classify mismatch groups (start with the anchor results and the ID 2 plumbing groups), write the 16 reports and the manifest, then the summary (prompt 4 steps 2-4). No fixes.

### Built in this prompt (no runtime code, manifest, registry, frozen artifact or journal changed)

| File | sha256 (prefix) | Change |
|---|---|---|
| `scripts/section4_parity_matrix.py` (new) | `7488808d` | Matrix driver: windows by rule, B4 v2 byte verification per group, anchor check, restart gate, segments, ES composition, state/lock/markers |
| `scripts/section4_parity_harness.py` | `f14ee406` | `feed_program`: `multi` fault (non-overlapping single faults; overlap refused) and restart `points`; single faults unchanged. `recompute_history`: one producer pass per dataset config per window (`_recompute_full` cache), then per-model selection |
| `scripts/section4_parity_compare.py` | `c8868b3c` | `compare_app_runs(models=, common_endpoints=, row_columns=)` for the composition check; defaults unchanged (bitwise, all keys) |
| `tests/test_section4_parity_harness.py` | `8bca6828` | +2 tests (multi-fault equals singles and refuses overlap; multi-point restart equals continuous; subset compare). 23 passed |

Broader suite not run: only section 4 scripts and their tests changed (no shared runtime code).

### Windows (written by rule before any comparison; immutable)

`artifacts/s4parity_20261002_v1/windows.json` sha256 `60a5dc5bea265525a748e706be52fe46acf6b4497b327ca1439a3d1e3a287e17` (sidecar `.json.sha256`), 26 segments. Shapes per protocol (ES 48+1+96 = 145, EURUSD 32+1+64 = 97), except the composition sample (6 endpoints; protocol `composition_invariance` says "sampled endpoints").

| Segment | Src | Event (UTC) | Scenarios | Reference |
|---|---|---|---|---|
| ES_H_rth_maint (anchor) | H | 2026-06-16 13:30 | overnight/RTH, maintenance break, HTF | frozen |
| ES_H_spring_dst | H | 2026-03-08 22:00 | spring DST, weekend reopen, HTF | frozen |
| ES_H_fall_dst | H | 2025-11-02 23:00 | fall DST, weekend reopen, HTF | frozen |
| ES_H_holiday | H | 2025-11-27 23:00 | holiday closure | frozen |
| ES_H_roll | H | 2026-06-17 00:00 | quarterly roll, maintenance | frozen |
| ES_R_rth_maint (base) | R | 2026-09-15 13:30 | overnight/RTH, maintenance, HTF (ESZ6 complete) | recompute, everything recorded (prefix 4,827) |
| ES_R_roll_weekend (base) | R | 2026-09-13 22:00 | roll, weekend reopen | recompute |
| ES_R_labor_day | R | 2026-09-07 22:00 | holiday (ESU6, completion unknown) | recompute |
| ES_R_gap_backfill | R | 2026-09-18 11:40 | missing bars, backfill, revision (repair-stored bars) | recompute |
| ES_F_restart_rth / _roll | F | as base | restart: every chunk boundary + mid-RTH 84, lag window 100, after break 139 / after roll 50 | recompute |
| ES_F_faults | F | R base | dropped 1@8, 3@20, 12@100; backfill@30x4 (+2); revision@52 (lag 3); reconnect@66x5 | recompute on final delivered series |
| ES_F_cold_start | F | R base | 100 prefix bars (< 132) | recompute on app view |
| ES_F_producer_failure | F | R base | frvp_context / ict_context raise, endpoints 60-65 | recompute |
| ES_F_composition | F | R base idx 60-65 | each ES model alone vs its group engine (bitwise) | app vs app |
| EU_H_weekend_dst | H | 2026-03-08 21:00 | weekend, US/EU DST mismatch, ATR/HTF | frozen |
| EU_H_session_dst (anchor) | H | 2026-03-10 12:00 | London->NY, DST mismatch, ATR/HTF | frozen |
| EU_H_holiday | H | 2025-12-25 23:00 | holiday (calendar v1) | frozen |
| EU_R_session (base) | R | 2026-09-15 12:00 | sessions, ATR/HTF | recompute, full recorded series |
| EU_R_weekend (base) | R | 2026-09-20 21:05 | weekend | recompute |
| EU_R_outage_edge | R | 2026-05-14 22:10 | missing bars / outage edge | recompute |
| EU_F_restart_session / _weekend | F | as base | every chunk boundary + London 8, TCN window 20, NY 36 / after reopen 34 | recompute |
| EU_F_faults | F | R base | dropped 1@6, 3@12, 12@64; backfill@20x3 (+2); revision@32; reconnect@44x4 | recompute on final delivered series |
| EU_F_cold_start | F | R base | 200 prefix bars (< 250) | recompute on app view |
| EU_F_strategy_timeout | F | R base | strategy_signals near-zero timeout, endpoints 40-45 | recompute |

Conventions (record; not protocol changes):

- **F reference.** Recomputed over the final delivered series (dropped bars absent, revised bars final), so mismatches are expected only for endpoints processed before a backfill/revision/reconnect resolved. A mismatch after resolution is a stale-history candidate (`app_producer_bug`).
- **S2.** Offline detector events exist only for training-period (H) windows, so S2 is compared on ES H segments; for R/F it is `not_compared` (no offline events). EURUSD S2 is incomplete (D5).
- **Restart gate.** Each restart segment is compared bitwise with its continuous base, per group. It covers both explicit restart points and restart-at-every-chunk-boundary (= chunk resume). Base and restart segments run without resume.
- **S layer (D2): incomplete for all 16 (protocol fallback).** IBKR is not running (nothing listens on 4001), so ES 2026-06-01..06-19 cannot be fetched. The app's FMP connector (`FMPRESTClient.fetch_historical_chart`) sends no from/to and filters the latest response, so EURUSD 2026-03-02..03-24 is unreachable through the app's own connector. No outward fetch was made.

### Anchor check (ES done 2026-10-03 02:34Z; EURUSD runs when the ES stream finishes)

Recompute (frozen dataset config over the same 8,000-bar span the app uses) vs frozen `source_history.parquet` on ES_H_rth_maint. Files: `artifacts/s4parity_20261002_v1/anchor/ES/{ES_FRVP,ES_ICT}/ES_H_rth_maint/anchor.json` + `<model>.anchor_diff.parquet`.

| ID | Features mismatched | Score mismatches (of 145) | Flips |
|---:|---:|---:|---:|
| 2 | 9 | 144 | 7 |
| 3 | 10 | 0 | 0 |
| 5 | 5 | 1 | 0 |
| 10 | 6 | 15 | 2 |
| 12 | 0 | 2 | 0 |
| 13 | 0 | 0 | 0 |
| 14 | 3 | 145 | 6 |
| 15 | 0 | 1 | 0 |
| 16 | 2 | 55 | 1 |
| 18 | 3 | 137 | 4 |
| 19 | 1 | 1 | 1 |

Not anchored, NOT yet classified. First look (IDs 2, 3): `*_order_block_age_bars` is NaN only in the recompute, and `ict_*confluence*`, `ict_zone_balance*`, `frvp_swing_poc_vs_session_poc` and the interactions built on them drift from the first rows. That pattern points to **window truncation** (long-lived ICT/FRVP structures older than 8,000 bars), not code drift. It matches the ID 2 plumbing groups (`ict_context`, interactions, sigma normalization).

- Classification step to do first: recompute over a much longer span (for example 60k bars). If the differences vanish, the cause is truncation: the taxonomy calls that `app_producer_bug (window truncation)` on the app side, and the R comparisons stay valid because app and recompute see identical recorded bars. If they persist, it is producer drift (check the working-tree `continuation_pullback.py` edit).
- Also explain the score mismatches where features match (IDs 12, 15, 19).

### Prompt 4, re-paste 1 (2026-10-03): disk-full stop, restart-gate input fix, relaunch, first classification

**Current job (supersedes the pid/log above):** pid 2668, started 2026-10-03 15:45Z with the same command. Log `logs/section4_matrix_20261003.log` (stderr `.err.log`). The first log `logs/section4_matrix_20261002.log` is kept and ends with a relaunch note. Markers and state file unchanged.

**What happened in run 1** (02:32Z-08:14Z): 13 segment×group runs compared (ES_R_rth_maint, ES_R_roll_weekend, ES_F_restart_rth, ES_F_restart_roll, ES_H_rth_maint, both groups; ES_H_spring_dst ES_FRVP). At 08:14Z every new segment failed with `sqlite3.OperationalError: database or disk is full`, and the process died while starting ES_H_holiday ES_ICT.
- The run's own artifacts total 2.9 GB, and C: had 184 GB free at 15:42Z, so the full disk was transient and caused by something outside the run (unknown).
- Fix: the driver now stops with `BLOCKED: <n> GB free` (exit 3) when free space < 5 GB, instead of failing every later segment.
- Failed partial runs (ES_H_spring_dst ES_ICT, ES_H_fall_dst, ES_H_holiday) re-run on relaunch.

**Restart gate ES failed in run 1, but because of the harness, not the app:**
- ES_F_restart_rth: equal for both groups.
- ES_F_restart_roll: unequal for both groups, but only in 4 columns (`engine_/store_first_observed_at`, `engine_/store_last_observed_at`) on prefix rows. Every value, S4-S7 and score was bitwise equal.
- Cause: all 4,413 recorded ESU6 rows (and 21,856 of 24,383 EURUSD rows) have NULL `first_observed_at_utc` in the live DB, so the store stamps the wall clock when the harness copies them (base 03:28Z vs restart 05:13Z). The inputs differed, so the gate did not test equal inputs.
- Fix (no comparison scope or tolerance change): `segment_bars` gives recorded rows with NULL observed-at the replay clock (bar close + 90 s, the existing H convention); everything else stays as recorded. Verified deterministic: two copies are identical, with 0 NULL left.
- The four roll runs and `restart_gate_ES.json` were moved aside (kept): `*__superseded_wallclock_copy_stamps`. Their state is `superseded`; they re-run first on relaunch, then the gate is re-evaluated.
- Note: ES_R_rth_maint and ES_F_restart_rth keep their run-1 results. Their copy-stamped rows are ESU6 prefix rows outside every captured window, and that gate was bitwise equal.
- Files: `scripts/section4_parity_harness.py` `f583a3d8`, `scripts/section4_parity_matrix.py` `efb3fbe5`. Focused tests: 23 passed.

**Classification, in progress** (no fixes made):

1. **training_side_noncausal candidate: positional row-index predictors (IDs 14, 16, 18).**
   - `ict/detectors/displacement.py:86` (`event_id = arange(len(df)) + 1`, sha `35eab821`) and `ict/detectors/fvg.py:263` (`formed_index`, sha `9ca56200`) emit row positions within the frame they are given.
   - Models select them: 14 `ict_nearest_bull_fvg_formed_index`, `ict_latest_bear_displacement_id`, `ict_nearest_bull_fvg_inversion_index`; 16 `ict_nearest_bear_fvg_inversion_index`, `ict_nearest_bear_fvg_formed_index`; 18 `ict_nearest_bull_fvg_formed_index`, `ict_nearest_bear_fvg_formed_index`.
   - Evidence on ES_H_rth_maint at 2026-06-16 07:35: `ict_latest_bear_displacement_id` app 7,932 / 8k recompute 7,933 / frozen 665,784.
   - The frozen value is the bar's ordinal in the 2017-onward training file. It is unavailable at decision time for any app window length. 7,329 of 7,329 S3 mismatch cells for ID 14 are these features.
   - Retrain guard: the branches are diagnostic and recalibration/retraining is not allowed here. **Repair recommendation:** drop or replace the positional features with origin-free forms (bars since the event = current index - formed index) in a retrain outside section 4.
2. **H-window FRVP mismatches mostly match the history span.**
   - `docs/audits/section4_tools_20261002/truncation_test.py` (`60b7f796`), output `artifacts/s4parity_20261002_v1/classification/truncation_test_ES_H_rth_maint.json` (`2ebb093f`).
   - App cells equal to the 8k-span recompute: ID 5 7,467/7,467; ID 2 20,994/26,594; ID 3 10,359/13,159; ID 10 2,099/3,779.
   - The remainder (`frvp_swing_poc_vs_session_poc`) also differs from the anchor. The anchor's span starts at a fixed bar while the app's 8,000-bar window slides, so a per-endpoint sliding recompute or the R windows (identical starts) must settle it.
   - If it is truncation, the class is `app_producer_bug (window truncation)` per the taxonomy, a fail.
   - ICT `ict_raw_swing_high/low` (IDs 13, 18, 19; ~36-46 cells) are app ≠ offline where recompute == frozen: open.
3. **R windows (identical history start for app and recompute):** S3 groups are dominated by `htf_context` (ESU6 completion unknown, so the app's HTF gate gives NaN, the D8 mechanism), plus rolling_stats/lags/zscores/continuation_pullback. Analysis is still to do after ES completes.

**Q-T1 RESOLVED 2026-10-03: user answered "whatever you recommend", so (a) applies.** IDs 14, 16 and 18 get `fail:training_side_noncausal` (diagnostic, unscored), with the repair recommendation recorded and their reports finished. No retraining or recalibration in section 4. Original question: IDs 14, 16 and 18 depend on positional row-index features, which is training-side non-causal. Should I (a) record them as `fail:training_side_noncausal` (diagnostic, unscored) with the repair recommendation and finish their reports, or (b) something else?

**Re-paste 2 (2026-10-03 16:10Z): Q-T1 recorded; the job is healthy** (pid 2668, roll base at chunk 2/4).

4. **R-window profile, ES_R_rth_maint** (`docs/audits/section4_tools_20261002/r_window_profile.py`, output `artifacts/s4parity_20261002_v1/classification/s3_profile_ES_R_rth_maint.json`):
   - Every S3 mismatch row is a complete ESZ6 row, and values are finite on both sides (except 2,700 app-NaN z-score cells for ID 5).
   - Producers: ICT models are `htf_context` only (plus a few interactions, and 27 `ict_context` cells for ID 13). FRVP models are `htf_context`, `continuation_pullback`, `frvp_context` and rolling_stats/lags/zscores/interactions.
   - App and recompute share the identical history start, so this is not truncation.
   - Leading hypothesis: the HTF completion gate (`features/feature_sets/htf_context.py:47`) engages over the ESU6 prefix (completion unknown), and HTF state then differs downstream.
5. **Deferred diagnostics queue** (run only when no matrix job runs, one at a time):
   - (i) `ote_venv/Scripts/python.exe docs/audits/section4_tools_20261002/gate_recompute.py ES ES_R_rth_maint`: recompute with the recorded `is_complete` column (as `ote_live/features/runtime_state.py:100` passes it). If gate-explained equals cells, the class is `declared_contract_behavior` (gate is pre-existing; protocol v2 D8 / eurusd_completion) and the segment is not known-completion evidence. Otherwise the remainder is open. Run the same for the other R segments.
   - (ii) A sliding-window or long-span recompute for H truncation (FRVP `frvp_swing_poc_vs_session_poc`; ICT `ict_raw_swing_high/low`).

Next after the run: wait for `MATRIX COMPLETE`; run the queue; classify the R-window groups; run the long-span or sliding recompute diagnostic (as its own job after the matrix); write reports, manifest and summary.

### Per-stream / per-model status

| Stream | Status |
|---|---|
| ES (11 models) | compared (STREAM ES COMPLETE 2026-10-04 02:10Z; restart gate passed after the deterministic re-run). Earlier note: Anchor done. ES_R_rth_maint ES_FRVP compared 02:58Z (1,441 s; 145/145 comparable for 2, 3, 5, 10; S1 clean; no failed checks; S3-S7 mismatches unclassified: S3 27k-90k cells, S7 flips 2: 11, 5: 2, 10: 9). Measured: FRVP ~7.6 min per 48-endpoint chunk on R (~4,800 bars), so ES is likely ~6-7 h |
| EURUSD (5 models) | compared (STREAM EURUSD COMPLETE 2026-10-05 02:35Z; restart gate passed; anchor: features mismatched 21:8, 22:1, 23:6, 25:3, 26:1, 0 flips) |

## Next action (prompt 4, superseded by the resume block above)

1. Write `artifacts/s4parity_20261002_v1/windows.json` by rule from protocol scenarios and candidates, using `historical_segment` and an R equivalent. Hash it before any comparison.
2. Run the anchor check: `build_reference(mode="recompute")` vs the frozen source history, one H segment per group.
3. Run the restart test per stream (F restart and chunked vs continuous), then segments in the background with `allow_resume=True` only after the restart test passes. Use `compare_segment` (with the ES `reference_setup_events`) and `compare_app_runs`.
4. Classify mismatch groups per the taxonomy, starting with the ID 2 plumbing groups. Never edit reference data, frozen bytes or preprocessing.
5. The D2 S-layer backfill fetch remains to do (isolated DB, separate client ID), then `compare_source` + `source_impact`.

## Prompt 4, re-paste 3 (2026-10-05): matrix complete, classification

- `MATRIX COMPLETE` at 2026-10-05 02:35Z (log `logs/section4_matrix_20261003.log`). State: 54 compared, 0 failed; restart gates ES and EURUSD passed; B4 v2 bytes verified for all 3 groups.
- ES composition (`artifacts/s4parity_20261002_v1/composition_ES.json`): equal for 2, 12 and 14; unequal for 3, 5, 10, 13, 15, 16, 18 and 19 (being analyzed).

### Classification status (2026-10-05 05:20Z) - IN PROGRESS, no fixes made

**Running job:** `sh docs/audits/section4_tools_20261002/run_diagnostics.sh` (detached). Log `logs/section4_diagnostics_20261005.log`, marker `DIAGNOSTICS COMPLETE`.
- ES "plain" recomputes (5 H segments) are done.
- EURUSD "htflegacy" recomputes: EU_H_weekend_dst done; EU_H_holiday, EU_R_session, EU_R_weekend and EU_R_outage_edge remain. The R ones run on ~23k bars (~15-25 min each).
- Outputs: `artifacts/s4parity_20261002_v1/classification/variant_{plain,htflegacy}_<segment>.json`.
- If the job died before the marker: re-run the same script; finished outputs are simply recomputed.

**Established findings (evidence in `classification/`):**

1. **Composition: equal for all 11 ES models.** `composition_recheck_ES.json`: zero differing columns on shared (endpoint, row) keys, identical window positions, and io/scores bitwise equal. The earlier "unequal" for 8 models was only capture depth (the group engine captures more rows).
2. **Restart gates passed** (ES after the deterministic re-run; EURUSD).
3. **ES R windows (`gate_test_ES_R_*.json`, 4 segments):** every S3 mismatch cell is either reproduced by recompute with the recorded `is_complete` (completion gate), or is one of 3 lookahead features. Class: `declared_contract_behavior/completion_gate` (htf_context.py:47; recorded ESU6 completion is NULL). These segments are not known-completion evidence.
4. **Training-side non-causal predictors (training_side_noncausal):**
   - positional: IDs 14, 16, 18 (Q-T1, approved);
   - lookahead `frvp_swing_poc_vs_session_poc` (IDs 2, 3, 10): on ES_R_rth_maint (fixed history start), 141/145 bars' values are revised once later bars arrive;
   - lookahead `ict_raw_swing_high/low` (IDs 13, 18, 19): `ict/structure/swings.py:66-82` centered pivot; mismatches only at row offsets -2..0;
   - EURUSD strategy lookahead (IDs 21, 23, 25, 26): `strategy__high_volume_points__*` (`features/strategies/high_volume_points.py:24-25` rolling center=True; offsets -14..0), `strategy__tesla_stock_boom_signals__*` (`tesla_stock_boom_signals.py:18` shift(-6); offsets -5..0), `strategy__ichimoku_signals_ichimoku_strategy_*` (offsets -25..0, no repaint).
5. **app_producer_bug/window_truncation (ES H):** app equals the 8,000-bar-span recompute (`variant_plain_ES_H_*.json`) where it differs from frozen. FRVP 5 is fully explained; FRVP 2/3/10 apart from the lookahead feature.
6. **app_producer_bug/htf_timestamp_semantics (EURUSD):**
   - `ote_live/features/incremental_engine.py:313-336` never sets `htf_bar_timestamp_semantics`, so `features/feature_sets/htf_context.py:140-142` falls back to `legacy_right_closed` for EURUSD.
   - The frozen config `artifacts/b2_ote_prepared_20260921_v1/dataset.metadata.json` says `bar_open`. Effect: app hourly buckets are 11:05-12:00, published one bar later.
   - On EU_H_session_dst, recompute with legacy semantics reproduces 34,944/35,869 (21), 220/220 (22), 19,305/19,934 (23), 8,446/9,218 (25) and 37/130 (26). The residuals are the strategy lookahead features above.
   - EURUSD has 0 decision flips in all segments.
7. **F segments:** producer failure / strategy timeout surfaced as `feature_build_failed` errors with endpoints skipped, never a silent fill (as required). S1 differences only in injected windows. not_evaluated endpoints = withheld backfill bars. Classifier rule: only endpoints processed before an injection resolved are `source_difference`; later cells not explained by the base segment are stale-history candidates (unexplained).
8. **stale_helper flags:** ES R windows = completion gate (helper incomplete). H windows = the first hour after a weekend/holiday reopen (ages 2,890/310 min), where HTF values equal the reference: `declared_contract_behavior/post_closure_helper_age`.
9. **S2 (ES H only):** all reference events matched with geometry exact, except one pair (ES_H_fall_dst FRVP 2025-10-31 17:05, setup_type app 1 vs ref 4, first endpoint, likely truncation; exact-window check pending). App-only events are outside the reference population (section-3 fired events). EURUSD S2 incomplete (D5).
10. **Precision residual (IDs 12, 15; 7 endpoints on H windows, no decision flips):**
    - S6 differs by 0.001-0.005 while S3/S4/S5 are within tolerance.
    - Rebuilding the XGB input from each side's recorded `sc:` rows reproduces each side's score exactly. The inputs differ by 1 float32 ULP: the frozen `source_history.parquet` is float32 (schema `float`, `model_training/source_history.py:40`), computed in float32; the app computes float64 and casts.
    - A tree split sits between the neighbours. Strictly under protocol v2 this is `unexplained` (no class or clause covers it).
11. **Other open residuals:**
    - `bear_volume_exhaustion_sigma_norm_60`: 1 row (ES_H_roll 2026-06-17 05:55) in 6 models, app 3,324,110 vs ref 319,086 (near-zero denominator).
    - ID 5 `continuation_bull_retest_quality_zscore_100`: 720 cells on ES_H_spring_dst, app raw 0 vs ref NaN over the weekend.
    - Next: `exact_window_check.py` (app's exact sliding window) for ES_H_spring_dst ID 5, ES_H_roll ID 2 at 2026-06-17 05:55+, and ES_H_fall_dst ID 2 at the first endpoint.

**Tools:** `classify.py` (rules 1-7), `variant_recompute.py`, `exact_window_check.py`, `composition_recheck.py`, `run_diagnostics.sh`, `gate_recompute.py` in `docs/audits/section4_tools_20261002/`; report writer `scripts/write_section4_parity_reports.py` (immutable, manifest, sidecars). The classifier dry run works; EURUSD numbers await the diagnostics.

**Next steps (in order):** (1) wait for `DIAGNOSTICS COMPLETE`; (2) run `exact_window_check.py` for the residuals in 11; (3) `classify.py`, then review the unexplained list; (4) `scripts/write_section4_parity_reports.py`; (5) summary `docs/section4_progress.md#section4_parity_20261005` plus FRVP/ICT journal entries; (6) final reply with question Q-P1.

**Pending question Q-P1 (asked at the end of prompt 4):** the float32 1-ULP tree-split residual (finding 10). Keep it as `unexplained` under v2 (IDs 12 and 15 become fail:unexplained), or approve a protocol v3 clause that treats S6-only differences reproduced exactly from each side's within-tolerance S5 inputs as declared precision behavior?

## Prompt 4, re-paste 4 (2026-10-05): Q-P1 resolved, protocol v3, classification closing

- **Q-P1 RESOLVED: user chose (b).**
  - **Protocol v3** published and locked: `configs/section4_parity_protocol_v3.json` sha256 `7ece6bbfa9f5bdf96d4adfa7c55052a31f41fde4968eccd0a8704d29e57cc939`, lock `configs/section4_parity_protocol_v3.lock.json` `914647a99ad071aa9567ea7117c5b91acceb87076bc64a6b2ec8e0b13ed7e6ce`. Publisher `scripts/publish_section4_parity_protocol_v3.py` (`29ad02da`).
  - v3 = v2 plus `mismatch_taxonomy.declared_contract_behavior.float32_tree_split`. Numeric tolerances are unchanged (asserted). The reason is recorded in `changes_from_v2`.
  - The clause requires no S1-S5 mismatch at the endpoint, and each side's recorded S5 rows must reproduce that side's raw score exactly. S7 flips are never covered.
  - The run executed under v2 and is classified under v3.
- **DIAGNOSTICS COMPLETE** at 05:47Z (all `variant_plain_ES_H_*` and `variant_htflegacy_EU_*`).
- **Precision check:** `docs/audits/section4_tools_20261002/precision_check.py`, output `classification/precision_check.json`. 14 candidate endpoints (IDs 12, 13, 15 on ES H windows), all verified exactly. EURUSD has no candidates.
- **Exact app-window check (ES):** `exact_window_check.py`, outputs `classification/exact_window_ES_H_spring_dst_5.json` and `exact_window_ES_H_roll_2.json`. Every checked cell is reproduced by recomputing on the app's exact sliding 8,000-bar window. ID 5 `continuation_bull_retest_quality_zscore_100` and `bear_volume_exhaustion_sigma_norm_60` are therefore `app_producer_bug/window_truncation`.
- **NEW finding: `app_producer_bug/stale_history_after_backfill` (EURUSD).**
  - On EU_F_faults, after the backfill is upserted behind the cursor (endpoint index 25), the app's consumed rows still lack the 3 backfilled bars (endpoint indexes 26, 32, 50 show gap rows). ES_F_faults contains all 4.
  - Mismatches persist at 2,779 cells/endpoint, decaying to 1,203 by index 96.
  - Cause: `ote_live/ingestion/signals.py:997` `_reconcile_setup_history` returns unless asset == "ES" (signals.py sha `d0839157`), so `ote_live/ingestion/setup_revisions.py:149-178` (engine state clear + extend from the store; sha `fad015e0`) never refreshes EURUSD history.
- **Classifier updated** (`classify.py`):
  - per-endpoint downstream attribution (S4-S7 count as downstream only where the endpoint has upstream mismatches);
  - float32 clause;
  - exact-window evidence;
  - EU_F_faults stale-history class.
- **Running now:** `sh docs/audits/section4_tools_20261002/run_exact_eurusd.sh` (detached). Log `logs/section4_exact_eurusd_20261005.log`, marker `EXACT EURUSD COMPLETE`, ~25 min. It checks the last EURUSD residuals:
  - EU_H_session_dst `strategy__dual_momentum_*` (IDs 21, 23, 26; 29 cells);
  - EU_H_holiday ID 21 `continuation_bear_retest_quality_zscore_100` (320 cells);
  - EU_R_weekend `strategy__rocwb_*` / `strategy__svesc_*` (IDs 22, 26; 46 cells, same bars in EU_F_restart_weekend).
- **Next:** after the marker, run `classify.py` (exact-window JSONs are picked up automatically), then `scripts/write_section4_parity_reports.py` (now protocol v3), the summary `docs/section4_progress.md#section4_parity_20261005`, and the journals.

## PROMPT 4 COMPLETE (2026-10-05)

- **Classification:** `classify.py` (`b4b23436`) produced `artifacts/s4parity_20261002_v1/classification/classified.json` (`975d209d`). Of 209 segment×model results, **0 contain an unexplained group**.
  - Second exact-window pass: `logs/section4_exact_eurusd2_20261005.log` (svesc and dual-momentum long reproduced).
  - EURUSD exact-window evidence: `exact_window_EU_*.json`. The dual-momentum, continuation z-score, rocwb and svesc residuals all reproduce on the app's exact window, so the class is window truncation.
- **Reports** (immutable, `.sha256` sidecars, refuse overwrite): `docs/audits/section4_parity_<model_id>_<reviewed_set_sha12>_20261005.json`, 16 files.
  - Manifest: `docs/audits/section4_parity_manifest_20261005.json`.
  - Writer: `scripts/write_section4_parity_reports.py`.
  - Protocol reference: v3, executed under v2.
- **Verdicts:**
  - `fail:training_side_noncausal`: 2, 3, 10, 13, 14, 16, 18, 19, 21, 23, 25, 26;
  - `fail:app_producer_bug`: 5, 12, 15, 22;
  - pass: none; incomplete-only: none.
  - Source parity is incomplete for all 16; EURUSD S2 is incomplete (D5).
- **Summary:** `docs/section4_progress.md#section4_parity_20261005` (`7fb1b2a5`): per-model table, coverage matrix, classes, provisional EURUSD calendar note, OTE journal. It supersedes the 2026-09-29 B5 docs, which are kept unchanged.
- **Journals:** FRVP and ICT entries appended (2026-10-05). OTE is in the summary.
- **Tests:** no runtime or shared code changed in prompt 4. Only section 4 scripts and tools changed; the focused harness tests passed (23) after the harness edits. The broader suite was not re-run.
- **Next action:** prompt 5. Candidate minimal versioned runtime fixes (none made yet):
  - (a) history truncation;
  - (b) EURUSD `htf_bar_timestamp_semantics` carried into the runtime config;
  - (c) EURUSD backfill/revision history refresh.
  Training-side branches need the user's repair decision. Do not start section 5+.


<a id="section4_parity_20261005"></a>

## Section 4 live-parity results (B5), 2026-10-05

*Merged 2026-10-09 from `docs/section4_parity_20261005.md`.*

Supersedes the B5 evidence in `docs/B5_B6_live_feature_parity_20260929.md` and `docs/audits/b5_b6_live_feature_parity_20260929_v2.json` (zero passes, `not_run_resource_bounded`). Both are kept unchanged.

**Bottom line:** none of the 16 roster versions passes live parity.

- 12 fail on the training side (`training_side_noncausal`): their frozen models were trained on values the live app cannot reproduce without lookahead, or on row positions. IDs 2, 3, 10, 13, 14, 16, 18, 19, 21, 23, 25, 26.
- 4 fail only on app-side defects (`app_producer_bug`) that a versioned runtime change could fix. IDs 5, 12, 15, 22.
- No model is merely `incomplete`. Source parity (S layer) is incomplete for all 16 under the protocol fallback, so no model could pass in this run anyway.

A parity verdict is not scored eligibility: B6, A12, A13 and B7 are still required for every model. Every failing branch stays diagnostic and unscored, and does not block the others. No fix, retrain or recalibration was made in this prompt.

### Run

| Item | Value |
|---|---|
| Run | `artifacts/s4parity_20261002_v1` (matrix 2026-10-02 02:32Z to 2026-10-05 02:35Z; classification 2026-10-05) |
| Protocol | executed under v2 (`e25b5e48…`); classified under **v3** (`7ece6bbf…`, lock `914647a9…`). v3 adds only the user-approved `float32_tree_split` clause (Q-P1); numeric tolerances are unchanged |
| Windows | `artifacts/s4parity_20261002_v1/windows.json` `60a5dc5b…`: 26 segments written by rule before any comparison (ES 15, EURUSD 11; H = training source, R = recorded, F = fault injection) |
| Model bytes | B4 v2 bytes hash-verified for every group before load (`matrix_state.json` `bytes|*`); artifact-hash check OK on both sides in every segment |
| Engine | one shared producer pass per group per segment (ES_FRVP, ES_ICT, EURUSD_OTE union), then per-model selection; app path = real `LiveSignalProcessor` on isolated parity DBs, D8 provenance on, scored registration off |
| Code | `scripts/section4_parity_matrix.py`, `scripts/section4_parity_harness.py`, `scripts/section4_parity_compare.py`; classification tools in `docs/audits/section4_tools_20261002/`; reports by `scripts/write_section4_parity_reports.py` |
| Logs | `logs/section4_matrix_20261002.log` (run 1, stopped by a transient full disk), `logs/section4_matrix_20261003.log` (`MATRIX COMPLETE`), `logs/section4_diagnostics_20261005.log`, `logs/section4_exact_eurusd_20261005.log` |
| Reports | `docs/audits/section4_parity_<model_id>_<reviewed_set_sha12>_20261005.json` (+ `.sha256`), manifest `docs/audits/section4_parity_manifest_20261005.json` |

### Per-model status

Status is the primary fail class. `training_side_noncausal` takes precedence over `app_producer_bug`; every class present is listed. Decision flips are app vs reference threshold decisions outside the near-threshold band, summed over all windows. Most ES flips occur in the recorded windows, where the declared completion gate applies. EURUSD has **0 flips** in every window.

Manifest: `docs/audits/section4_parity_manifest_20261005.json` (sha256 `6337a9370cbb6ae8413e8617ece3fc8a290dadd940af21f89f911d1c9ccdacb4`).

| ID | Model | Stream | Verdict | Fail classes (cells/checks) | Comparable endpoints | Decision flips | Report sha256 (12) |
|---:|---|---|---|---|---:|---:|---|
| 2 | `frvp_long_continuation_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (107,366)<br>training_side_noncausal/lookahead (73,105) | 1,973/2,030 | 122 | `a42c8d781f8f` |
| 3 | `frvp_long_meta_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (45,425)<br>training_side_noncausal/lookahead (36,605) | 1,973/2,030 | 3 | `205ad91959f6` |
| 5 | `frvp_long_continuation_setup2_xgb_v1` | ES | **fail:app_producer_bug** | app_producer_bug/window_truncation (34,218) | 1,973/2,030 | 46 | `f0e80dcaa411` |
| 10 | `frvp_long_reversal_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (11,323)<br>training_side_noncausal/lookahead (22,002) | 1,973/2,030 | 65 | `b84dc3b38922` |
| 12 | `ict_long_meta_xgb_v1` | ES | **fail:app_producer_bug** | app_producer_bug/window_truncation (24) | 2,004/2,030 | 16 | `d2164f1eef11` |
| 13 | `ict_long_reversal_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (8)<br>training_side_noncausal/lookahead (456) | 2,004/2,030 | 150 | `5a9c2e7dce8a` |
| 14 | `ict_short_continuation_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (1,752)<br>training_side_noncausal/positional_index (38,548) | 2,004/2,030 | 118 | `b9bcae66bc6a` |
| 15 | `ict_short_meta_xgb_v1` | ES | **fail:app_producer_bug** | app_producer_bug/window_truncation (160) | 2,004/2,030 | 27 | `69ad36d0b421` |
| 16 | `ict_short_reversal_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (1,284)<br>training_side_noncausal/positional_index (8,067) | 2,004/2,030 | 45 | `698d936e3976` |
| 18 | `ict_short_reversal_ifvg_reversal_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (32)<br>training_side_noncausal/lookahead (491)<br>training_side_noncausal/positional_index (18,069) | 2,004/2,030 | 113 | `51585b148b4e` |
| 19 | `ict_short_reversal_sweep_reclaim_xgb_v1` | ES | **fail:training_side_noncausal** | app_producer_bug/window_truncation (2,312)<br>training_side_noncausal/lookahead (487) | 2,004/2,030 | 144 | `3d8564d9ab86` |
| 21 | `long_ote_meta_tcn_champion` | EURUSD | **fail:training_side_noncausal** | app_producer_bug/htf_timestamp_semantics (189,249)<br>app_producer_bug/stale_history_after_backfill (97,157)<br>app_producer_bug/window_truncation (333)<br>training_side_noncausal/lookahead (11,092) | 961/1,067 | 0 | `a1a5181fca7f` |
| 22 | `long_reversal_tcn_v2_20260525_narrow48` | EURUSD | **fail:app_producer_bug** | app_producer_bug/htf_timestamp_semantics (1,609)<br>app_producer_bug/stale_history_after_backfill (50,076)<br>app_producer_bug/window_truncation (48) | 973/1,067 | 0 | `54a1c341d356` |
| 23 | `long_union_tcn_champion` | EURUSD | **fail:training_side_noncausal** | app_producer_bug/htf_timestamp_semantics (85,001)<br>app_producer_bug/stale_history_after_backfill (44,308)<br>app_producer_bug/window_truncation (7)<br>training_side_noncausal/lookahead (7,821) | 973/1,067 | 0 | `3da44a135a4b` |
| 25 | `short_ote_meta_tcn_champion` | EURUSD | **fail:training_side_noncausal** | app_producer_bug/htf_timestamp_semantics (89,890)<br>app_producer_bug/stale_history_after_backfill (42,917)<br>training_side_noncausal/lookahead (9,525) | 969/1,067 | 0 | `f533e71ba0ec` |
| 26 | `short_reversal_xgb_v2_20260525` | EURUSD | **fail:training_side_noncausal** | app_producer_bug/htf_timestamp_semantics (2,987)<br>app_producer_bug/stale_history_after_backfill (55,548)<br>app_producer_bug/window_truncation (53)<br>training_side_noncausal/lookahead (468) | 971/1,067 | 0 | `935b5e209ef5` |

### Per-stream coverage matrix

Every protocol scenario was exercised with comparable endpoints for every model:
- historical (H) windows: DST, holidays, roll, weekend, sessions;
- recorded (R) windows: RTH open, maintenance, roll, Labor Day, gap/backfill, weekend, outage;
- fault (F) windows: dropped 1/3/12 bars, backfill behind the cursor, revision, reconnect, restart, cold start, producer failure / strategy timeout, ES composition.

S2 and the S layer are reported separately; they keep every model from reaching `pass` regardless of the fail classes.


**ES coverage** (E = exercised with comparable endpoints for that model, - = not)

| Scenario | 2 | 3 | 5 | 10 | 12 | 13 | 14 | 15 | 16 | 18 | 19 | Segments |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overnight_rth_transitions | E | E | E | E | E | E | E | E | E | E | E | ES_H_rth_maint, ES_R_rth_maint |
| daily_maintenance_break | E | E | E | E | E | E | E | E | E | E | E | ES_H_roll, ES_H_rth_maint, ES_R_rth_maint |
| spring_dst | E | E | E | E | E | E | E | E | E | E | E | ES_H_spring_dst |
| fall_dst | E | E | E | E | E | E | E | E | E | E | E | ES_H_fall_dst |
| holiday_planned_closure | E | E | E | E | E | E | E | E | E | E | E | ES_H_holiday, ES_R_labor_day |
| quarterly_roll | E | E | E | E | E | E | E | E | E | E | E | ES_H_roll, ES_R_roll_weekend |
| weekend_reopen | E | E | E | E | E | E | E | E | E | E | E | ES_H_fall_dst, ES_H_spring_dst, ES_R_roll_weekend |
| missing_bars | E | E | E | E | E | E | E | E | E | E | E | ES_F_faults, ES_R_gap_backfill |
| reconnect_backfill | E | E | E | E | E | E | E | E | E | E | E | ES_F_faults, ES_R_gap_backfill |
| revision | E | E | E | E | E | E | E | E | E | E | E | ES_F_faults, ES_R_gap_backfill |
| restart | E | E | E | E | E | E | E | E | E | E | E | ES_F_restart_roll, ES_F_restart_rth |
| cold_start_warmup | E | E | E | E | E | E | E | E | E | E | E | ES_F_cold_start |
| producer_failure | E | E | E | E | E | E | E | E | E | E | E | ES_F_producer_failure |
| htf_publication | E | E | E | E | E | E | E | E | E | E | E | ES_H_fall_dst, ES_H_rth_maint, ES_H_spring_dst, ES_R_rth_maint |
| group_composition | E | E | E | E | E | E | E | E | E | E | E | single-model engine equal to group engine on 240 shared rows; io/scores bitwise equal (classification/composition_recheck_ES.json) |
| S2 setup events | compared | compared | compared | compared | compared | compared | compared | compared | compared | compared | compared | ES: H windows only; EURUSD: D5 |
| Source parity (S) | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | protocol fallback |

**EURUSD coverage** (E = exercised with comparable endpoints for that model, - = not)

| Scenario | 21 | 22 | 23 | 25 | 26 | Segments |
|---|---|---|---|---|---|---|
| weekend_close_open | E | E | E | E | E | EU_H_weekend_dst, EU_R_weekend |
| session_transitions_asia_london_ny | E | E | E | E | E | EU_H_session_dst, EU_R_session |
| us_eu_dst_mismatch | E | E | E | E | E | EU_H_session_dst, EU_H_weekend_dst |
| completed_30m_1h_atr_publication | E | E | E | E | E | EU_H_session_dst, EU_H_weekend_dst, EU_R_session |
| holiday_closure_calendar_v1 | E | E | E | E | E | EU_H_holiday |
| missing_bars | E | E | E | E | E | EU_F_faults, EU_R_outage_edge |
| reconnect_backfill | E | E | E | E | E | EU_F_faults, EU_R_outage_edge |
| revision | E | E | E | E | E | EU_F_faults |
| restart | E | E | E | E | E | EU_F_restart_session, EU_F_restart_weekend |
| cold_start_warmup | E | E | E | E | E | EU_F_cold_start |
| strategy_timeout_producer_failure | E | E | E | E | E | EU_F_strategy_timeout |
| S2 setup events | incomplete | incomplete | incomplete | incomplete | incomplete | ES: H windows only; EURUSD: D5 |
| Source parity (S) | incomplete | incomplete | incomplete | incomplete | incomplete | protocol fallback |

### Mismatch classes

Truth is the causal contract. For each group, the causally correct side was established first, and no reference data, frozen preprocessing, model bytes or app producer was changed. Evidence files are under `artifacts/s4parity_20261002_v1/classification/`, and every report lists per-group evidence paths into the segment `diff/` files.

#### training_side_noncausal (branch diagnostic; repair = retrain outside section 4, user decision)

| Cause | Feature(s) | Models | Evidence |
|---|---|---|---|
| Positional row index: the frozen value is the bar's ordinal in the 2017-onward training file (~665k); live values count from the app's window start (~7.9k) | `ict_nearest_*_fvg_formed_index`, `ict_nearest_*_fvg_inversion_index`, `ict_latest_bear_displacement_id` | 14, 16, 18 | `ict/detectors/displacement.py:86`, `ict/detectors/fvg.py:263`; 98.5–99.4% of train rows and 100% of val/test rows exceed 7,900; Q-T1 approved |
| Centered swing pivot (needs the next bars) | `ict_raw_swing_high`, `ict_raw_swing_low` | 13, 18, 19 | `ict/structure/swings.py:66-82` (`confirmed_*` is the shifted causal form); mismatches only at row offsets −2..0 |
| Swing-to-swing profile revised by later bars | `frvp_swing_poc_vs_session_poc` | 2, 3, 10 | On ES_R_rth_maint (fixed history start), 141 of 145 bars' values change after later bars arrive |
| Strategy signals using future bars | `strategy__high_volume_points__*`, `strategy__tesla_stock_boom_signals__*`, `strategy__ichimoku_signals_ichimoku_strategy_*` | 21, 23, 25, 26 | `features/strategies/high_volume_points.py:24-25` (`rolling(center=True)`; offsets −14..0), `tesla_stock_boom_signals.py:18` (`shift(-6)`; offsets −5..0), ichimoku offsets −25..0 with no app repaint |

The app is the causally correct side in every row above. Making an app producer reproduce these values is forbidden.

**Repair recommendation:** retrain without these predictors, or with their causal forms:
- bars-since-event instead of row indexes;
- `ict_confirmed_swing_*` instead of `ict_raw_swing_*`;
- the swing POC only from swings confirmed at or before t;
- strategy signals without negative shifts or centered windows.

The offline builders that produce them should be fixed and re-hashed before any retrain.

#### app_producer_bug (fail until a minimal versioned runtime change; rerun under a new run id)

| Cause | Scope | Evidence |
|---|---|---|
| **Window truncation.** The app keeps `runtime_history_bars` = 8,000 bars. Long-lived ICT/FRVP structures (order blocks, FVGs, confluence, swing structure, rolling sigma) differ from frozen full-history values | ES H windows; all ES models carry some cells | App equals a recompute of the frozen config on its exact sliding window (`exact_window_ES_H_*.json`, every checked cell) and on the fixed 8,000-bar span (`variant_plain_ES_H_*.json`) |
| **HTF timestamp semantics.** `build_runtime_feature_config` never sets `htf_bar_timestamp_semantics`, so EURUSD falls back to `legacy_right_closed` (hours 11:05–12:00, published one bar late); training used `bar_open` | EURUSD; all 5 models (HTF EMA spreads, `continuation_*` built on them, structure interactions) | `ote_live/features/incremental_engine.py:313-336`, `features/feature_sets/htf_context.py:140-142`, `artifacts/b2_ote_prepared_20260921_v1/dataset.metadata.json`; recompute with legacy semantics reproduces the app (`variant_htflegacy_EU_*.json`) |
| **Stale history after backfill or revision.** Bars upserted behind the cursor never enter the EURUSD in-memory history | EURUSD; all 5 models (EU_F_faults) | `ote_live/ingestion/signals.py:997` returns unless `asset == "ES"`, so `ote_live/ingestion/setup_revisions.py:149-178` (state clear + extend) never runs for EURUSD. App windows at endpoint indexes 26/32/50 still show the 3 backfilled bars as gaps, while ES holds all 4 |

#### declared_contract_behavior (pass with citation)

| Behavior | Citation |
|---|---|
| HTF completion gate on recorded ES bars whose completion is unknown (all ESU6 rows); also the `stale_helper` flags there | `features/feature_sets/htf_context.py:47`; protocol v2/v3 D8 and `eurusd_completion` analogue. Recompute with the recorded `is_complete` reproduces the app (`gate_test_ES_R_*.json`). These windows are therefore **not known-completion evidence** |
| Producer failure / strategy timeout surfaced as `feature_build_failed`; endpoint skipped, never silently filled | protocol `scenarios.F_injections.producer_failure` |
| HTF helper age after a planned closure (first hour after a weekend or holiday reopen; HTF values equal the reference) | HTF completeness rules in `htf_context.py`; ES app calendar closures |
| float32 tree split: S6-only differences at 14 endpoints (IDs 12, 13, 15) with S1–S5 within tolerance; each side's recorded S5 rows reproduce its own score exactly | protocol v3 `declared_contract_behavior.float32_tree_split` (Q-P1); `precision_check.json` |

#### source_difference (not a P failure)

Fault-injection endpoints processed before a backfill or revision resolved, plus the withheld bars that are never re-scored. ES resolves correctly after release. EURUSD does not: see stale history above.

#### unexplained

None. After the diagnostics, 0 of the 209 segment×model results contain an unexplained group (`classification/classified.json`).

### Other checks

- **Restart (protocol `restart`):** passed for both streams. Explicit restarts (mid-RTH, inside a lag window, after the maintenance break, after the roll; London, New York, inside a TCN window, after the weekend reopen) plus a restart at every chunk boundary were bitwise equal to continuous runs.
  - The first ES gate failed only because recorded bars without observation times were stamped with the wall clock at copy time. The copy was made deterministic and the pair re-run; the superseded runs are kept as `*__superseded_wallclock_copy_stamps`.
- **Engine composition (ES):** each of the 11 models alone equals its group engine on every shared row; inputs, scores and decisions bitwise (`composition_recheck_ES.json`).
- **S2 setup events (ES H windows only):** every reference event was matched with exact geometry except one FRVP pair (ES_H_fall_dst, 2025-10-31 17:05, first endpoint: setup type app 1 vs ref 4; FRVP context there is truncation-affected).
  - App-only events are outside the reference population (section 3 fired events only).
  - EURUSD S2 is incomplete: no live OTE event producer exists (D5).
- **Source parity (S layer): incomplete for all 16 (protocol fallback).** IBKR was not running (nothing listens on port 4001). The app's own FMP connector sends no from/to and only returns recent bars, so the ES 2026-06 and EURUSD 2026-03 overlaps could not be fetched. No outward fetch was made.
- **Run incidents:**
  - a transient full disk stopped run 1 at 08:14Z on 2026-10-03; the driver now stops cleanly below 5 GB free;
  - the restart-gate copy-stamp fix above.

### EURUSD calendar (provisional)

All EURUSD windows used the pinned **provisional** calendar `eurusd_5m_feed_clock_week_v1` (`configs/eurusd_5m_expected_open_calendar_v1.json`, sha256 `5db00e42…`). It is stated in each EURUSD report.
- Recorded EURUSD weekend boundaries differ from calendar v1 (inventory finding 7).
- Every recorded EURUSD bar has unknown completion, compared as the exact value "unknown". None of it is known-completion evidence.
- The EURUSD verdicts do not depend on the calendar: the HTF-semantics, stale-history and lookahead findings reproduce on historical windows with complete bars, or on fault windows.

### OTE (EURUSD) journal

- **What:** 11 EURUSD windows (3 H, 3 R, 5 F) through the EURUSD_OTE union engine (IDs 21, 22, 23, 25, 26), one shared producer pass per window, B4 v2 bytes verified.
- **Result:**
  - `fail:training_side_noncausal`: 21, 23, 25, 26. Strategy signals with lookahead: high-volume pivots, tesla boom, ichimoku.
  - `fail:app_producer_bug`: 22 (HTF semantics, stale history after backfill, window truncation).
  - Zero decision flips in every window.
- **Observations:**
  - The app's EURUSD HTF buckets follow `legacy_right_closed` while training used `bar_open` (config not carried into the runtime).
  - Backfills and revisions behind the cursor never enter the EURUSD in-memory history; the ES path refreshes it.
  - Restart and chunk resume are bitwise equal.
  - S2 is incomplete (no live OTE event producer, D5).
- **Decision:** diagnostic and unscored. Prompt 5 candidates (versioned, value-neutral elsewhere):
  - carry `htf_bar_timestamp_semantics` from the frozen dataset config into the runtime config;
  - run the setup-revision history refresh for EURUSD too;
  - address history truncation.
  Retraining without the lookahead strategy signals is a user decision.

### Next

Prompt 5. No fixes were made in prompt 4.
- App-side fixes would need minimal versioned runtime changes, logged in `docs/live_app_change_journal.md` and re-run under a new run id:
  - history truncation (all ES models; EURUSD 21–23, 26);
  - EURUSD HTF semantics;
  - EURUSD stale history.
- Training-side branches (12 models) need the user's repair decision; no retraining in section 4.
- Recorded ES windows remain completion-unknown evidence until bars with known completion are recorded.
- Source parity needs IBKR running, or a historical connector with a date range.

## Prompt 5 (2026-10-05): remediation and close-out - IN PROGRESS

### 1. app_producer_bug

- **Fixed** (opt-in, versioned; logged in `docs/live_app_change_journal.md` 2026-10-05):
  - `s4-htf-semantics-v1`: `IncrementalFeatureEngine` / `LiveSignalProcessor` `htf_bar_timestamp_semantics=` (EURUSD frozen value `bar_open`);
  - `s4-history-refresh-v1`: `LiveSignalProcessor(refresh_history_behind_cursor=True)` rebuilds history via `setup_revisions._refresh_live_state` when `source_bar_history` shows a first observation or revision at or before the cursor.
  - Files: `ote_live/features/incremental_engine.py` (`b3bcd636`), `ote_live/ingestion/signals.py` (`b5f212de`).
  - Causality: both HTF conventions are causal and the contract is `bar_open`; including backfilled bars is causal, as ES already does.
- **Tests:**
  - `tests/test_ote_live_section4_prompt5_fixes.py` (`10ceee7c`): 3 passed. Off = default config bit-for-bit; on changes only `htf_*`/`continuation_*`; the refresh equals an uninterrupted feed; neutral without behind-cursor writes.
  - Focused: 35 passed.
  - **Broader suite (once):** `logs/section4_prompt5_full_tests_20261005.log`: 1232 passed, 2 skipped, 15 failed. Same 15 pre-existing failures as prompts 2 and 3 (feature_parity 3, prediction_replay 3, seed_state 2, phase5_operator_tooling 4, backend_attribution 2, ote_xgboost_pipeline 1).
- **Harness** (`scripts/section4_parity_harness.py`, `eb19812d`): `S4_RUN_DATE` env var (run root `artifacts/s4parity_<date>_v1`) and `S4_APP_FIXES=p5`, which gives EURUSD groups both flags (semantics read from the frozen dataset configs, which must agree). `run.json` records `app_fixes`.
- **Not fixable without a contract or training change:** window truncation (all ES; EURUSD 21/22/23/26). Recommendation in `docs/section4_progress.md#section4_training_side_recommendation_20261005` §2.
- **RUNNING: EURUSD parity re-run** for all 5 EURUSD models (every model consuming htf_context / continuation / history).
  - Command: `S4_RUN_DATE=20261005 S4_APP_FIXES=p5 ote_venv/Scripts/python.exe -u scripts/section4_parity_matrix.py run EURUSD` (detached, pid 4068, started 16:42Z).
  - Run root: `artifacts/s4parity_20261005_v1` (windows.json copied byte-identical, `60a5dc5b...`).
  - Log: `logs/section4_matrix_20261005_eurusd_p5.log`. Markers: `STREAM EURUSD COMPLETE`. `MATRIX COMPLETE` will not print, because the ES stream is not re-run (no ES fix).
  - Estimate ~24 h (11 segments × ~2 h). If it dies: re-run the same command with the same env vars (resume rules as before).

### 2-4

- `docs/section4_progress.md#section4_training_side_recommendation_20261005`:
  - training_side_noncausal for 12 models, with features, minimal repair and the cascade (B2 → B3 → B4 v3 → route manifest v3 → section 3 smoke → section 4 re-run);
  - the truncation contract options (a/b/c, recommending b);
  - the source_difference flips (4, all ES_F_faults: IDs 2 ×2, 13, 18; `classification/source_difference_flips.json`);
  - unexplained: none.
- **Pending user decisions:** retrains (12 models) and the truncation option.

### Next steps (after `STREAM EURUSD COMPLETE`)

1. Classify the new EURUSD run.
   - Adapt the diagnostics: the HTF-legacy variant no longer applies. Use a plain bar_open recompute on the app span (H) and exact app-window checks for non-lookahead residuals. On R windows the reference uses all recorded bars, so app-span checks are needed for truncation.
   - Confirm EU_F_faults has no stale-history cells and no HTF-semantics cells remain.
2. Write new report versions for 21-26 (date of the run; old 20261005 reports kept) and mark the old ones superseded in a new manifest version (the old manifest stays unchanged).
3. Update the summary doc (new version file) and close-out notes.txt section 4: tick items 1-6 with completion notes; annotate the priorities item and the B2 sub-item with pass/fail IDs; note the parity portion on "Carry IDs 5/23/14/18".
4. Journals: per-model parity status; parity pass is not scored eligibility.
5. Mark this file complete.


<a id="section4_training_side_recommendation_20261005"></a>

## Section 4 remediation recommendation (training side and contract), 2026-10-05

*Merged 2026-10-09 from `docs/section4_training_side_recommendation_20261005.md`.*

Status: **awaiting the user's decision.** No retraining, recalibration, manifest or contract change was made in section 4. Every branch listed here stays diagnostic and unscored. A parity pass is never scored eligibility; B6, A12, A13 and B7 remain required.

Evidence:
- `docs/section4_progress.md#section4_parity_20261005`
- per-model reports `docs/audits/section4_parity_*_20261005.json`, manifest `docs/audits/section4_parity_manifest_20261005.json`
- classification `artifacts/s4parity_20261002_v1/classification/`

### 1. training_side_noncausal: 12 models

The frozen models were trained on predictor values the live app cannot reproduce causally. In every case the app is the causally correct side, so no app producer may be changed to imitate these values.

| ID | Model | Non-causal predictor(s) | Mechanism |
|---:|---|---|---|
| 2 | `frvp_long_continuation_xgb_v1` | `frvp_swing_poc_vs_session_poc` | swing-to-swing profile revised once later bars arrive (141/145 bars on a fixed-start window) |
| 3 | `frvp_long_meta_xgb_v1` | `frvp_swing_poc_vs_session_poc` | as above |
| 10 | `frvp_long_reversal_xgb_v1` | `frvp_swing_poc_vs_session_poc` | as above |
| 13 | `ict_long_reversal_xgb_v1` | `ict_raw_swing_low` | centered pivot needs the next bars (`ict/structure/swings.py:66-82`) |
| 14 | `ict_short_continuation_xgb_v1` | `ict_latest_bear_displacement_id`, `ict_nearest_bull_fvg_formed_index`, `ict_nearest_bull_fvg_inversion_index` | row position in the input frame (`ict/detectors/displacement.py:86`, `ict/detectors/fvg.py:263`); training values ≈ ordinal in the 2017-onward file |
| 16 | `ict_short_reversal_xgb_v1` | `ict_nearest_bear_fvg_formed_index`, `ict_nearest_bear_fvg_inversion_index` | row position |
| 18 | `ict_short_reversal_ifvg_reversal_xgb_v1` | `ict_nearest_bear_fvg_formed_index`, `ict_nearest_bull_fvg_formed_index`, `ict_raw_swing_high` | row position + centered pivot |
| 19 | `ict_short_reversal_sweep_reclaim_xgb_v1` | `ict_raw_swing_high` | centered pivot |
| 21 | `long_ote_meta_tcn_champion` | `strategy__high_volume_points__high_volume_pivot`, `strategy__ichimoku_signals_ichimoku_strategy_strategy__signal__long/short`, `strategy__tesla_stock_boom_signals__signal__neutral` | `rolling(center=True)` pivots (`features/strategies/high_volume_points.py:24-25`); `shift(-6)` (`tesla_stock_boom_signals.py:18`); ichimoku depends on bars t+1..t+25 |
| 23 | `long_union_tcn_champion` | high-volume pivot; ichimoku long/short | as above |
| 25 | `short_ote_meta_tcn_champion` | high-volume pivot; ichimoku long/short | as above |
| 26 | `short_reversal_xgb_v2_20260525` | `strategy__high_volume_points__high_volume_pivot` | as above |

#### Minimal repair (per model, smallest first)

1. **Drop the listed predictors and retrain** with the frozen hyperparameters, selection and seeds otherwise unchanged. This is the smallest change and touches no producer code.
2. **Or replace them with causal forms** and retrain:
   - bars-since-event (`current_index - formed_index`) instead of `*_formed_index`, `*_inversion_index` and `*_displacement_id`;
   - `ict_confirmed_swing_high/low` (the 3-bar-shifted form) instead of `ict_raw_swing_*`;
   - a swing POC computed only from swings confirmed at or before t;
   - strategy pivots and signals without centered windows or negative shifts.
   Fix the offline producers first, under a new producer version, and re-hash.

Before any retrain, add a guard that refuses a predictor whose value at row t changes when rows after t are appended. It would have caught all four mechanisms: a causal-prefix test on the B2 feature builder.

#### Cascade per repaired model

1. new B2 dataset/feature version, a prepared rebuild through `scripts/rebuild_b2_*_features.py` with the repaired producers (the current prepared data stays immutable);
2. retrain → B3 review;
3. new B4 freeze (v3) with the replacement bytes;
4. new route manifest (v3) and lock;
5. section 3 smoke replay (b1-v5 simulator) for that model;
6. section 4 parity re-run for that model under a new run id, then B6/A12/A13/B7.

Changing a shared offline producer changes the prepared features of every model that selects that feature. Only the models listed above select them.

### 2. app_producer_bug that needs a contract decision: window truncation

All 11 ES models and EURUSD 21, 22, 23 and 26 carry window-truncation cells. The app keeps `runtime_history_bars` = 8,000 bars (the B4 v2 staged manifest value), while the frozen features were computed over full history since 2017. Long-lived ICT/FRVP structures (unmitigated order blocks and FVGs, confluence, swing structure, rolling sigma) can be older than 8,000 bars. Recomputing the frozen config on the app's exact window reproduces every affected app cell, so the code is right and the history length is the gap.

This is the only remaining cause for **IDs 5, 12 and 15** (ID 22 also has it; its EURUSD defects were fixed, see §3).

It cannot be fixed with a value-neutral runtime patch. Options, each a contract or training change for the user:
- **(a)** raise `runtime_history_bars` in a new manifest version and seed live state from historical bars. This only bounds the error unless the features have bounded memory; the costs are build time (ES FRVP ~12.7 s per 8,000 bars) and the need for historical seeding.
- **(b)** retrain with bounded-memory features (for example, zone age capped by `ict_zone_max_age`, structure lookback ≤ the runtime window), so frozen values equal any window ≥ N bars. This is the durable fix.
- **(c)** carry incremental producer state across bars, so the app never truncates. A larger runtime change, with its own parity proof.

My recommendation is (b), together with the §1 retrains, since it removes the gap for every model at once.

### 3. Fixed in prompt 5 (EURUSD app defects, opt-in runtime flags)

- `s4-htf-semantics-v1`: EURUSD HTF buckets follow the frozen `bar_open`.
- `s4-history-refresh-v1`: bars backfilled or revised behind the cursor enter the history.

See `docs/live_app_change_journal.md` (2026-10-05). EURUSD parity is being re-run under run id `s4parity_20261005_v1` for all five EURUSD models.

### 4. source_difference decision flips (for review; D2 policy, never auto pass/fail)

Fault-injection endpoints processed before the backfill or revision resolved, where app and reference decisions differ (`artifacts/s4parity_20261002_v1/classification/source_difference_flips.json`):

| Segment | ID | Endpoint (UTC) | App cal. | Ref cal. | App / ref decision |
|---|---:|---|---:|---:|---|
| ES_F_faults | 2 | 2026-09-15 12:20 (bars withheld, backfill pending) | 0.5898 | 0.6391 | no / yes |
| ES_F_faults | 2 | 2026-09-15 13:35 (provisional revision) | 0.6238 | 0.6010 | yes / no |
| ES_F_faults | 13 | 2026-09-15 12:20 | 0.3496 | 0.4229 | no / yes |
| ES_F_faults | 18 | 2026-09-15 12:20 | 0.3187 | 0.3236 | no / yes |

These are expected under the injected feed: the app correctly scored on the bars it had. After resolution, ES matches the final series.

### 5. unexplained

None remain: 0 of 209 segment×model results after one bounded investigation pass. No branch fails as unexplained, so no extra shadow-collection inclusion is needed for that reason. All 16 branches stay diagnostic and unscored, and can remain in shadow collection under the existing rules.

### Decisions requested

1. Retrain the 12 training-side models: drop or replace the listed predictors (the §1 cascade).
2. Truncation: choose (a), (b) or (c) for all ES models and EURUSD 21/22/23/26.

## Prompt 5 decisions (user, 2026-10-05)

1. **Truncation: option (b).** Retrain with bounded-memory features (for example, zone age capped by `ict_zone_max_age`, structure lookback ≤ the runtime window), so frozen values equal those from any window ≥ N bars.
2. **Training side: replace the predictors and retrain**, after fixing the offline producers under a new producer version and re-hashing:
   - bars-since-event (`current_index - formed_index`) instead of `*_formed_index`, `*_inversion_index` and `*_displacement_id`;
   - `ict_confirmed_swing_high/low` instead of `ict_raw_swing_*`;
   - a swing POC computed only from swings confirmed at or before t;
   - strategy pivots and signals without centered windows or negative shifts.
3. **EURUSD p5 re-run STOPPED** at 2026-10-05 ~16:55Z (user request), during the first segment (EU_R_session) after the anchor check.
   - Partial run root `artifacts/s4parity_20261005_v1` is kept, unused; the stale lock's pid is dead. Log `logs/section4_matrix_20261005_eurusd_p5.log`.
   - Re-run EURUSD parity only after the retrain cascade, under a new run id.

This is a retrain workstream outside section 4's "no retraining" rule, explicitly decided by the user. It does not start section 5+. Cascade per model: producer fix (new version) → B2 prepared rebuild (new version) → retrain → B3 review → B4 freeze v3 → route manifest v3 → section 3 smoke → section 4 parity re-run (new run id).

**Step R0 (next):** a producer audit over the five frozen dataset configs. Causal-prefix: is the value at t unchanged when later bars are appended? Window invariance: is the endpoint value equal for 8,000 vs 16,000 bars of history? The result is the complete list of non-causal and unbounded-memory features, and the same audit proves the fixes.

## Retrain step R0 (2026-10-05): producer audit - COMPLETE (results below)

- **Tool:** `scripts/audit_feature_causality.py` (`204adc2e`). For one frozen dataset config and end bar T, it builds raw pre-fill producer output and runs three tests.
  - **Causal prefix:** fixed start, 12 cuts every 23 bars plus T, last 40 rows each vs the build extended by 30 bars; exact. Gives the lookahead depth.
  - **Bounded memory:** the 8,000-bar window vs 8,500/9,500/11,000/16,000/24,000-bar windows ending at T; C1 tolerance.
  - **Frozen contract:** the 8,000-bar window vs the roster models' frozen full-history `source_history.parquet` on the same 300 rows (C1, NaN == NaN). This is the (b) acceptance test: after the fix, frozen values must equal any window ≥ N.
- **Tool fixes during validation:**
  - a NaN-safe `changed()` (the first version reported all-NaN columns as changed);
  - multi-cut causality (a single end bar misses intermittent lookahead);
  - tolerance-aware memory (float decay is not unbounded memory);
  - the frozen-contract test (some order blocks predate any window start).
- **Validated on ICT config at 2026-06-16 13:30:**
  - non-causal: `ict_raw_swing_high` (depth 3; IDs 18, 19), `ict_raw_swing_low` (depth 2; ID 13), and `ict_failed_sell_side_sweep` (unselected);
  - frozen-contract mismatch: the positional indexes (14, 16, 18) and `bear_volume_exhaustion_sigma_norm_60` (3 cells).
- **Running:** `sh scripts/run_feature_causality_audit.sh` (`2053c7e8`, detached). Log `logs/feature_causality_audit_r0_20261005.log`, marker `AUDIT R0 COMPLETE`. It covers the 4 ES configs at 3 end bars each, then OTE at 2026-03-10 12:00 (longs=9000, cuts=4; ~1 h). Outputs: `artifacts/feature_causality_audit/<dataset>_<end>_frozen.json`.
- **Config map:** frvp_july → 2, 3, 10; frvp_setups → 5; ict → 12-16; ict_specialists → 18, 19; ote → 21-26.

**Next (R1):** from the R0 union, fix the producers under new versions. Each fix must be opt-in or versioned, and the frozen configs and data stay immutable.
- positional indexes → bars-since;
- raw swings: switch selection to `ict_confirmed_swing_*`;
- `frvp_swing_poc_vs_session_poc` from confirmed swings at or before t;
- strategy centered and negative-shift signals;
- bounded memory: cap order-block/FVG/zone ages and structure lookback, and stabilize sigma-norm denominators.

Re-run the R0 audit on the new producer version until there are 0 non-causal and 0 frozen-contract mismatches for the selected features. Then R2: B2 prepared rebuild in new roots; R3: retrain one job at a time; R4: B3 review → B4 v3 → route manifest v3 → section 3 smoke → section 4 parity (new run id).

## Retrain step R0 results (2026-10-05)

**`AUDIT R0 COMPLETE` 18:49Z** (`logs/feature_causality_audit_r0_20261005.log`): 13 files in `artifacts/feature_causality_audit/` (4 ES configs × 3 end bars + OTE).
- Union over selected features: `artifacts/feature_causality_audit/r0_union_selected.json`.
- Unselected non-causal producers: `r0_unselected_noncausal.json` (ict_context 2, frvp_context 2, strategy_signals 23).

**Selected features by finding** (NC = non-causal, lookahead depth in rows; MEM = differs between 8,000 and 8,500-24,000-bar windows; FRZ = 8,000-bar window ≠ frozen):

- **Non-causal (fix the producer or swap the predictor):**
  - `ict_raw_swing_high/low` (d3; 13, 18, 19);
  - `frvp_swing_poc_vs_session_poc` (d40, i.e. every row; 2, 3, 10);
  - `strategy__high_volume_points__high_volume_pivot` (d6; 21, 23, 25, 26);
  - ichimoku long/short (d23-26; 21, 23, 25);
  - `strategy__tesla_stock_boom_signals__signal__neutral` (d6; 21).
- **Positional plus memory** (MEM ≈ FRZ ≈ all rows): `ict_latest_bear_displacement_id` (14), `ict_nearest_*_fvg_formed_index` (14, 16, 18), `ict_nearest_*_fvg_inversion_index` (14, 16).
- **Memory beyond 24,000 bars or path-dependent** (MEM 0 but FRZ > 0; examples):
  - `active_bull_fvg_count`, `*_order_block_age_bars`, `dist_to_bear_fvg_atr`, `ict_bars_since_bos_any/sweep_any` (FRVP and OTE configs only), confluence, `ict_*_sigma_norm_60`, `major_event_*`, `cooldown_since_major_event`, many interactions, `frvp_setup_*`;
  - plus `continuation_bull_retest_quality_zscore_100` (MEM 300) and the volume-exhaustion sigma norms (1-8 cells, near-zero denominator).

**R0 follow-up: is the frozen tape reproducible from the dataset config?** `scripts/audit_frozen_full_history.py` (`970d90b1`; runner `scripts/run_frozen_full_history.sh` `ffbcd847`; audit tool now `204adc2e`) builds the frozen config on the FULL source history and compares it with frozen on the same 3×300 rows (C1).
- **ICT configs: 0 mismatches** (250 and 156 selected features compared). The ICT frozen tape equals config + current code on full history, so the 8,000-bar differences are genuine memory, positional or lookahead effects. No reference drift.
- **FRVP configs: RUNNING.** `sh scripts/run_frozen_full_history.sh` (detached). Log `logs/frozen_full_history_20261005.log`, marker `FULL HISTORY COMPLETE`. A full ES FRVP build takes >27 min and ~3 GB per config. It decides whether the FRVP-config FRZ cells (e.g. `ict_bars_since_bos_any` differing only in the FRVP config) are long memory or a different frozen build path (B2 repair scripts `scripts/rebuild_b2_*_features.py`).
- OTE is not run: a full EURUSD build is hours long. If needed, use a long window (e.g. 100k bars) instead.

**Design notes for R1:**
- Per-zone age caps already exist (`ict/detectors/order_blocks.py:78`, `fvg.py:75`, `swings.py:63`; 120 bars), yet values still depend on very old history. The unbounded memory comes from state machines (swing, BOS and structure state, active-zone sets) whose state depends on the whole path, so caps alone will not give window invariance.
- Candidate generic design: an **anchored warm-start**. Stateful producers restart at fixed calendar anchors (for example the weekly open) after a warm-up of W bars, so every value depends only on bars since (anchor - W). Offline and live compute it identically whenever the window covers anchor - W (≤ 8,000 bars).
- `frvp_swing_poc_vs_session_poc` repaint: swing confirmation (`features/transforms.py:137` shifts by the window), the swing window (`frvp/profiles/anchors.py:258`) and the bin width (`frvp/profiles/builder.py:142`) are causal on inspection. The repaint source must be located empirically in R1 (diff the intermediate profile for one bar between prefix and extended builds).

**Next:** after `FULL HISTORY COMPLETE`, write the R1 design (per-producer fixes plus the bounded-memory mechanism) and get user approval of the mechanism before coding; it sets the feature semantics of every retrained model.

## R0 follow-up complete (2026-10-05 20:11Z) and R1 design

- **`FULL HISTORY COMPLETE`** (`logs/frozen_full_history_20261005.log`).
  - ICT: 0/250 and 0/156 mismatched.
  - **FRVP: frvp_july 22/303 and frvp_setups 5/160 mismatched** (`artifacts/feature_causality_audit/*_fullhistory_start0.json`). The frozen FRVP tape is not reproducible from config + current code: `scripts/rebuild_b2_frvp_features.py` ("Repair FRVP's shared HTF dependencies while preserving unrelated features") patched an older tape. So the prompt 4 FRVP truncation findings are partly reference drift as well. The retrain rebuilds FRVP from scratch through one path.
- **R1 design:** `docs/section4_progress.md#retrain_r1_design_20261005`.
  - A `feature_version=2` config gate (v1 byte-identical) covers positional → bars-since, dropping raw swings, the swing-POC repaint, the strategy lookaheads and the sigma-norm floor.
  - Bounded memory by **anchored warm-start** (weekly anchor, W = 2,000 bars, max dependency ~3,500 bars < 8,000).
  - Acceptance = R0 audit clean on v2 plus v1 before/after identity.
  - All 16 models retrain.
- **OPEN QUESTION Q-R1:** approve the anchored warm-start (weekly anchor, W = 2,000), name another anchor or W, or choose per-producer caps.


<a id="retrain_r1_design_20261005"></a>

## Retrain R1 design: causal and bounded-memory feature producers (v2), 2026-10-05

*Merged 2026-10-09 from `docs/retrain_r1_design_20261005.md`.*

Status: **approved 2026-10-05 (Q-R1: anchored warm-start, weekly anchor, W = 2,000 bars); implemented as feature contract v2 (`features/causal_contract.py`); acceptance audit running (see `docs/section4_progress.md`, Retrain R1).** Implementation notes:

- The anchor is Saturday 12:00 America/New_York, inside the weekend closure of both streams, so blocks split exactly at the weekly open.
- The z-score and sigma-norm memory was traced to pandas' online rolling variance; v2 computes rolling statistics per window instead of the floor in §2.5.
- The swing-POC look-ahead was a datetime unit bug (microseconds vs nanoseconds) in `frvp_context._resolve_swing_window`. Decisions already made (2026-10-05): truncation option (b); replace the non-causal predictors; fix the producers first under a new version; retrain.

### 1. What R0 established

Tools:
- `scripts/audit_feature_causality.py`: causal prefix over 13 cuts, multi-start memory, frozen contract.
- `scripts/audit_frozen_full_history.py`: the frozen config on full history vs frozen.

Outputs are in `artifacts/feature_causality_audit/`.

| Finding | Selected features (models) |
|---|---|
| **Look-ahead** (value at t changes when later bars arrive) | `ict_raw_swing_high/low` (13, 18, 19; depth 3); `frvp_swing_poc_vs_session_poc` (2, 3, 10; every row repaints); `strategy__high_volume_points__high_volume_pivot` (21, 23, 25, 26; depth 6); ichimoku long/short (21, 23, 25; depth 23-26); `strategy__tesla_stock_boom_signals__signal__neutral` (21; depth 6). Unselected: 2 ict_context, 2 frvp_context, 23 strategy columns |
| **Positional** (row index in the frame) | `ict_latest_bear_displacement_id`, `ict_nearest_*_fvg_formed_index`, `ict_nearest_*_fvg_inversion_index` (14, 16, 18) |
| **Unbounded memory** (state depends on the whole price path; 8,000- to 24,000-bar windows agree with each other but not with full history) | FVG counts and distances, order-block ages, `ict_bars_since_bos/sweep_any`, confluence, `*_sigma_norm_60` of those, `major_event_*`, `cooldown_since_major_event`, structure interactions, `frvp_setup_*`, `continuation_bull_retest_quality_zscore_100` (all ES models and EURUSD 21-23, 25, 26) |
| **Numerically unstable** | `bear/bull_volume_exhaustion_sigma_norm_60` (near-zero rolling std; 1-8 cells) |
| **Frozen tape not reproducible** (FRVP only) | Full-history rebuild ≠ frozen for 22/303 (frvp_july) and 5/160 (frvp_setups) selected features. `scripts/rebuild_b2_frvp_features.py` rebuilt only the HTF dependencies and preserved the rest from an older tape. The ICT configs reproduce frozen exactly (0/250, 0/156) |

Per-zone age caps already exist (120 bars: `ict/detectors/order_blocks.py:78`, `fvg.py:75`, `swings.py:63`), yet values still depend on years of history. The memory sits in the **state machines** (swing, BOS and CHoCH state, active zone sets, event history) that feed those zones. Capping ages alone cannot make values window-invariant.

### 2. Producer fixes (feature version v2; v1 stays byte-identical for the frozen configs)

A new `FeatureBuilderConfig.feature_version` (default 1 = today's behavior, so every frozen config and the live app are unchanged) selects the v2 producers:

1. **Positional → bars-since:** `ict_nearest_{bull,bear}_fvg_bars_since_formed`, `..._bars_since_inversion`, `ict_bars_since_latest_{bull,bear}_displacement` (current index - event index). The v2 output drops the index columns.
2. **Raw swings:** v2 drops `ict_raw_swing_*` and keeps `ict_confirmed_swing_*` (already causal: shifted by the swing window).
3. **`frvp_swing_poc_vs_session_poc`:** locate the repaint empirically (diff the intermediate swing/session profiles for one bar between a prefix and an extended build), then compute it only from swings confirmed at or before t. The same goes for the 2 unselected non-causal FRVP columns and the 2 ICT ones (`ict_failed_sell_side_sweep` …).
4. **Strategy signals:**
   - high-volume pivots become trailing pivots, confirmed `left_bars` after the pivot (no `center=True`);
   - tesla-boom drops `shift(-6)` (that column is an exit label, not a signal);
   - ichimoku drops the forward-displaced component found by the prefix test;
   - the other 23 unselected non-causal strategy columns are fixed or removed in v2.
5. **Sigma-norm stability:** a floor on the rolling std (for example `max(std, 1e-9·scale)`), so near-zero denominators can't amplify float noise.

### 3. Bounded memory: anchored warm-start (decision requested)

**Proposal:** each stateful producer (ict_context, frvp_context, temporal_context, and any strategy with unbounded state) is computed per weekly block.
- Its state resets at a fixed calendar anchor: the weekly session open (ES Sunday 18:00 ET; EURUSD the calendar-v1 weekly open).
- Before each anchor it runs a warm-up of **W bars**.
- Each value at t therefore depends only on bars from (latest anchor ≤ t) − W to t, at most about 1 week + W.

Rolling transforms on top (`zscore_100`, `sigma_norm_60`, lags ≤ 23, TCN window 32) add ≤ 100 rows.

With W = 2,000 bars:
- maximum dependency ≈ 1,380 (an ES week) + 2,000 + 100 ≈ 3,500 bars;
- that is well inside the app's 8,000-bar window, so the live value equals the offline value exactly.

The same builder code runs offline and live, so no separate runtime algorithm is needed. Only the live config switches to v2 when the retrained models are installed (B6).

| | Anchored warm-start (recommended) | Per-producer lookback caps |
|---|---|---|
| Guarantee | exact window invariance for any window ≥ ~3,500 bars, provable with the R0 audit | each state machine must be bounded by hand; easy to miss one |
| Semantics change | structure, zones and events older than ~1 week + W are forgotten at each weekly reset | the same in effect, but feature by feature |
| Offline cost | about one build per week of history: ICT ~10 min, FRVP ~1 h, EURUSD strategy blocks several hours (strategy producers without state need no blocking) | similar |

**Acceptance (R1 done when):** the R0 audit on the v2 configs shows, for every selected v2 feature:
- 0 non-causal (13 cuts × 3 end bars);
- 0 memory differences between 8,000 and 8,500-24,000-bar windows and a full-history v2 build (C1 tolerance);
- v1 output bit-identical to today (before/after test).

### 4. Then (R2-R4, unchanged from the approved cascade)

1. **R2.** Rebuild the B2 prepared data in new versioned roots with the v2 config, through one build path. The current tapes, including the patched FRVP tape, stay immutable.
2. **R3.** Retrain one job at a time with the frozen recipes, re-running feature selection on the v2 columns.
3. **R4.** B3 review → B4 freeze v3 → route manifest v3 → section 3 smoke replay → section 4 parity under a new run id.

Affected models:
- **all 16 retrain**, because the bounded-memory change touches features every ES model and most EURUSD models select;
- ID 22 only selects HTF/interaction and rocwb features, but the v2 tape changes its inputs too.

### Decision requested

Approve **anchored warm-start, weekly anchor, W = 2,000 bars**, or name another anchor or W, or choose per-producer caps.

## Retrain R1 (2026-10-05): feature contract v2 - IMPLEMENTED, acceptance audit RUNNING

**Q-R1 approved:** anchored warm-start, weekly anchor, W = 2,000 bars.

**v1 guard (byte identity):** `scripts/feature_version_baseline.py` (`50171e96`) hashes raw pre-fill output of the 5 frozen configs on fixed windows.
- Baseline before any change: `artifacts/feature_v2_r1/v1_baseline_before.json`; repeat `v1_baseline_repeat.json`.
- After R1: `v1_after_r1_code_2.json` → **BASELINE IDENTICAL** in all 5 cases.
- 7 OTE columns are ignored because they differ between two identical v1 runs: `features/strategies/inflation_trading_signals.py:21` uses `np.random.uniform`, and the membership-plan india/GDX strategies are likewise random. None is selected; v2 drops them.

**Code** (all gated on `FeatureBuilderConfig.feature_version >= 2`; default 1):

| File | sha (8) | Change |
|---|---|---|
| `features/causal_contract.py` (new) | `0569eb65` | v2 contract: bounded sets (ict_context, frvp_context, temporal_context, strategy_signals); weekly anchor Saturday 12:00 America/New_York (DST-safe naive-date arithmetic); strategy replacements and drops; predictor exclusions (positional `ict_*_index`, identity `ict_*_id`, raw swings, failed sweeps, frvp_open_drive_flag, plus their transforms); `add_bars_since_columns`; `exact_rolling_mean_std` |
| `features/config.py` | `b99caf99` | `feature_version=1`, `bounded_memory_warmup_bars=2000`; `to_dict` omits both at v1 defaults (frozen config digests unchanged) |
| `features/builder.py` | `f634524a` | `_build_feature_set`: v2 runs bounded sets per weekly block with a W-bar warm-up; ict_context gains `*_bars_since`; exact rolling for rolling stats, z-scores and sigma norms |
| `features/transforms.py` | `abe86911` | `exact=` option: per-window mean/std, constant windows exact. pandas' online variance leaks the scale of earlier history (std ~0.045 for a constant window after 1e6-scale values); that is the root cause of the R0 z-score/sigma-norm memory |
| `frvp/feature_sets/frvp_context.py` | `1cebb4d5` | **bug found:** `_resolve_swing_window` compared `anchor_engine._timestamp_ns` (asi8 in **us**) with `Timestamp.value` (**ns**), so every bar searched past all swings in the frame (the swing-POC look-ahead). v2 passes the unit |
| `features/feature_sets/strategy_signals.py` | `5d20f79e` | v2 strategy id remap and drop |
| `features/strategies/{high_volume_points,ichimoku_strategy,tesla_stock_boom}_causal_v2.py` | `984fb3f5`, `bc363163`, `e7f13155` | trailing pivot reported at confirmation; close vs close 26 bars ago instead of `shift(-26)`; exit `shift(+6)` |
| `tests/test_feature_v2_causal_contract.py` | `94e858db` | 9 passed: DST-correct anchors, window invariance v2 (and v1 not), bars-since invariance, exclusions/ids, 3 strategies causal (prefix-equal), exact rolling, v1 serialization |

**First v2 audit (ICT, 2026-06-16 13:30):** 622 columns, 33 predictor-excluded; **eligible non-causal 0, eligible memory-unbounded 0** (v1: 2 and 7 selected). Build 3.9 s per 8,000 bars (v1 1.7 s).

**RUNNING: v2 acceptance:** `sh scripts/run_feature_v2_acceptance.sh` (`9ea23928`, detached), started 21:26Z. Log `logs/feature_v2_acceptance_20261005.log`, marker `V2 ACCEPTANCE COMPLETE`.
- Covers ICT ×3 end bars, ICT specialists ×1, FRVP july ×3, FRVP setups ×1, then OTE ×1 (longs=9000, cuts=3; ~2 h).
- Outputs: `artifacts/feature_causality_audit/*_feature_version2.json`. Each line reports eligible non-causal and memory-unbounded columns; both must be 0.

**Live-latency note (for B6/A12, not R1):** v2 blocks the bounded sets per week inside the app's 8,000-bar window (~6 blocks + warm-up), so builds are ~2.3x slower; EURUSD strategy_signals is the large one (~78 s per bar in v1). Mitigation for later: compute only the blocks covering the consumed tail rows.

**Next:**
- After the marker: any eligible non-causal or memory column gets fixed or excluded, then re-audit. Then the full-history v2 equality check is folded into R2: the rebuilt tapes' tails must equal 8,000-bar builds at the audit end bars.
- **R2:** rebuild B2 prepared data with `feature_version=2` in new versioned roots, through one build path. This also fixes the FRVP patched-tape drift.
- **R3:** retrain one job at a time.
- **R4:** cascade.

## R1 acceptance results (2026-10-06 03:22Z) and EURUSD performance split

`V2 ACCEPTANCE COMPLETE` (`logs/feature_v2_acceptance_20261005.log`; outputs `artifacts/feature_causality_audit/*_feature_version2.json`).
- **ES: PASS.** ict ×3 end bars, ict_specialists, frvp_july ×3 and frvp_setups: **eligible non-causal 0, eligible memory-unbounded 0**. 33-34 columns are predictor-excluded. Builds per 8,000 bars: ICT 3.9 s (v1 1.7), FRVP ~40 s (v1 ~13).
- **EURUSD:** eligible memory-unbounded 0, but **eligible non-causal 6, none selected**: `adaptive_cyber_cycle_signals` (depth 40), `warren_buffett_etf_signals` (depth 40; `features/strategies/warren_buffett_etf_signals.py:22` uses `.iloc[-1]` for every row) and `eurcad_trading_signals` (1 cell). All three were **added to `V2_DROPPED_STRATEGIES`**.
- **EURUSD performance:** a v2 build of 8,000 bars took 2,884 s (v1 ~500 s). Blocking all ~400 strategies weekly multiplies the per-strategy call overhead (full-history rebuild ≈ 40 h).
  - Fix (implemented, `features/builder.py`): under v2, strategy_signals runs non-memory strategies once on the whole frame (window-invariant) and only `V2_MEMORY_STRATEGIES` per weekly block. The two split reports are merged.
  - The list is still empty, to be filled from the audit below.
- **RUNNING:** `sh scripts/run_strategy_memory_audit.sh` (detached). Log `logs/strategy_memory_audit_20261006.log`, marker `STRATEGY MEMORY AUDIT COMPLETE`, ~1.5 h.
  - Tool `scripts/audit_strategy_memory.py`: strategy_signals alone, v2 ids, 8,000 vs 16,000/24,000-bar windows, 2 end bars (2026-03-10 12:00, 2025-12-23 15:00).
  - Output `artifacts/feature_causality_audit/strategy_memory_*.json` → `memory_strategy_ids`.

**Next:**
1. Set `V2_MEMORY_STRATEGIES` to the union of `memory_strategy_ids`.
2. v1 identity guard (`feature_version_baseline.py` + compare).
3. Re-run the EURUSD v2 audit (expect eligible non-causal 0 and memory 0; much faster).
4. Run the focused tests.
5. Then R2.

## R1 strategy memory result (2026-10-06)

- **`STRATEGY MEMORY AUDIT COMPLETE`** (`logs/strategy_memory_audit_20261006.log`): 9 and 22 strategies at the two end bars, union 25 of 382. The sets disagree.
  - Most are plain rolling-mean crossovers (e.g. `copper_trading_signals`, `gbpnzd_trading_signals`).
  - Cause: pandas rolling means carry float residue from earlier history, so near-tie comparisons flip with the window start. A partial memory list cannot be complete; static scans (`strategy_memory_union.json`) also miss most of them.
- **Decision (implementation, within the approved Q-R1 mechanism):** v2 blocks strategy_signals as a whole (the original design); the split was removed.
  - Absent one-hot dummy columns in a block are exactly 0.
  - v2 builds run strategies **in-process** (`strategy_timeout_seconds=None`). The frozen EURUSD config's 600 s timeout runs every strategy in a subprocess. Measured on 8,000 bars: unblocked with subprocesses ~417 s, unblocked in-process 14.6 s, **blocked in-process 40.4 s**. Full-history EURUSD v2 ≈ 45 min.
  - **B6/A12 constraint:** the live EURUSD engine also pays the subprocess cost (~77 s per bar in v1). A v2 runtime must run strategies in-process (with a whole-build timeout if needed) and compute only the blocks covering the consumed tail.
- **Code:** `features/causal_contract.py` (`8024a856`): drop list +3 (adaptive_cyber_cycle, eurcad, warren_buffett_etf). `features/builder.py` (`295e3020`). Tests 9 passed. **v1 guard: BASELINE IDENTICAL** (`artifacts/feature_v2_r1/v1_after_r1_code_3.json`).
- **RUNNING:** `sh scripts/run_feature_v2_acceptance_eurusd.sh` (detached). Log `logs/feature_v2_acceptance_eurusd_20261006.log`, marker `V2 EURUSD ACCEPTANCE COMPLETE`, ~1 h. EURUSD v2 with timeout None and longs 9,000/16,000/24,000 at 2026-03-10 12:00 and the held-out 2025-11-20 14:00.
- **Next:** if eligible non-causal 0 and memory 0 → **R1 COMPLETE**, then R2 (B2 prepared rebuild, v2, new roots, timeout None).

## R1 COMPLETE (2026-10-06)

- **`V2 EURUSD ACCEPTANCE COMPLETE`** (`logs/feature_v2_acceptance_eurusd_20261006.log`), in-process strategies, memory windows 9,000/16,000/24,000.
  - Eligible memory-unbounded **0** at both end bars (2026-03-10 12:00, held-out 2025-11-20 14:00).
  - Eligible non-causal: `anomaly_detection_signals` (both ends) and `membership_plans_signals__big_moves_trading_strategy_strategy` (one end); both use a whole-frame mean/std (`anomaly_detection_signals.py:13-14`). Neither is selected; both were added to `V2_DROPPED_STRATEGIES` (`features/causal_contract.py` `af7638ed`).
  - Strategies are independent per strategy, so dropping them removes exactly these columns and nothing else: **EURUSD eligible non-causal 0**.
- **ES:** eligible non-causal 0 and memory 0 on all 4 configs and 8 end-bar runs (prior section).
- **Final checks:** `tests/test_feature_v2_causal_contract.py` 9 passed. v1 guard **BASELINE IDENTICAL** (`artifacts/feature_v2_r1/v1_after_r1_final.json`).
- **v2 build settings for R2:** `feature_version=2`, `bounded_memory_warmup_bars=2000`, `strategy_timeout_seconds=None` (in-process).
- **Next: R2 (B2 prepared rebuild, v2, new versioned roots, one build path).** Lineage per group: labeled source → features tape (`artifacts/b2_*_features_*`, partly patched by B2 repair scripts) → preprocessing (`run_config.json`) → prepared root → validation preparation manifest → `scripts/run_b2_source_history.py` → `scripts/run_b2_retraining.py`. The R2 plan (commands, roots, the v2 predictor-exclusion hook in feature selection) is written before any rebuild.

## Retrain R2 (2026-10-06): B2 rebuild under feature contract v2 - R2a, R2b, R2c COMPLETE

**Plan** (one build path; every old root is immutable; all outputs are new versioned roots):

- **R2a. v2 feature tapes.** `scripts/build_b2_v2_features.py` (`8aebdef7`) builds from the pinned training source (the same bars the app and the audits use) with the frozen dataset config + `feature_version=2`, W=2,000 and in-process strategies. Row count must equal the base tape (label alignment: ES 666,769; EURUSD 543,191 after its 250 warm-up rows).
  - `feature_columns` = v2-eligible predictors only, so the existing `scripts/rebuild_b2_prepared.py` (which selects from `metadata['feature_columns']`) can never select an excluded column.
  - Acceptance: tape = fresh 8,000-bar v2 build on the last 300 rows of every eligible feature at 3 ES and 2 EURUSD end bars (C1). This is the full-history equality check.
  - Roots: `artifacts/b2_{ict,frvp_july,frvp_setups,ote}_features_v2_20261006` (ict_specialists reuses the ict root).
  - Note: v1 base tapes had 578 ICT predictors (HTF-confluence helpers and repair-removed helpers excluded); v2 builds 622 columns, 589 eligible. Feature selection reruns per target.
- **R2b. Prepared roots.** `scripts/rebuild_b2_prepared.py --group <g> --features-root <v2 root> --labels <same> --events <same> --output-root artifacts/b2_<g>_prepared_v2_<date>`. The same label/event files and preprocessing recipes come from `scripts/run_b2_retained_preparation.py` JOBS (frvp_july, ict, ict_specialists, frvp_setups); the OTE labels/events are to be located the same way. The FRVP recency root is re-materialized (`scripts/materialize_frvp_recency_prepared_root.py`) on the new frvp_july root.
- **R2c.** Validation preparation (frozen per-target selection prefix, imputation and scaling) and source history (`scripts/run_b2_source_history.py`) on the new roots, mirroring `b2_validation_prepared_20260922_v2` / `b2_source_history_20260927_v1`.
- **R3.** Retrain the 16 roster models one job at a time with their saved recipes (`scripts/run_b2_retraining.py`), into new model roots.
- **R4.** B3 review → B4 freeze v3 → route manifest v3 → section 3 smoke → section 4 parity (new run id, v2 runtime config: in-process strategies, `htf_bar_timestamp_semantics=bar_open`, history refresh behind the cursor).

**R2a run 1** (`sh scripts/run_b2_v2_features.sh`, `fcd7fe49`, 05:20Z-09:34Z, log `logs/b2_v2_features_20261006.log`):

| Group | Root | Eligible / excluded | Acceptance |
|---|---|---|---|
| ict | `artifacts/b2_ict_features_v2_20261006` | 589 / 33 | **PASS** (0 differ at 3 ES ends) |
| frvp_july | `artifacts/b2_frvp_july_features_v2_20261006` | 898 / 34 | **PASS** |
| frvp_setups | `artifacts/b2_frvp_setups_features_v2_20261006` | 898 / 34 | **PASS** |
| ote | `artifacts/b2_ote_features_v2_20261006` (superseded, kept) | 1793 / 33 | **FAIL**: 37 cols at 2026-03-10, 34 at 2025-11-20 |

**OTE FAIL diagnosis** (all in `strategy__*`; ES has no strategies):

1. *Within-week look-ahead (missed by both R1 audits).* `rocwb_signals` computes `RMS = np.mean(...)` over the whole weekly block. In the tape, the block holding 2026-03-09 11:25 runs to Fri 2026-03-13 21:55 (983 later bars); a window ending at the current bar cannot see them (3 flipped cells). rocwb alone on full history reproduces the tape exactly (0 / 543,191 diffs); 20k / 8k / 3k tails all agree with each other. A source scan of all strategy modules for whole-frame reductions found three more of the same class: `volume_flow_signals` (`np.std` of the VFI over the block), `london_breakout_signals` (Asian high/low over every day of the block), `end_of_month_effect_signals__..._nifty_50_...` (`TradingDay.max()` over the frame). Others flagged were causal (row-wise `axis=1`, block-start anchors, `valid` convolution, rolling) or already replaced/dropped.
2. *Window-dependent.* `dollar_cost_averaging_signals` writes one value per month; `_fill_numeric_columns_inplace` (whole-frame `ffill`) carries it across blocks, so the tape had 100 / 1.17487 / 85.1158 while a window starting 2026-01-29 (no month-end bar) had 0. Residual note: that global `ffill` is unbounded in principle; it only bites columns with NaN runs longer than the window, and no other column showed it at any of the 5 ES/EURUSD end bars.
3. *One-hot schema (~34 cols).* `get_dummies` emits a column only for categories seen in the frame. The tape is exactly 0 on all 300 rows of every such column, but in a window the column is absent, and the app turns an absent selected feature into NaN + missing producer (`ote_live/features/incremental_engine.py` 475-487), i.e. the model would go diagnostic-only.

**Fix** (`features/causal_contract.py` `e2ee43a2`, `features/config.py`, `features/builder.py`, `scripts/build_b2_v2_features.py` `7afd1295`; v1 unchanged): the 5 strategies join `V2_DROPPED_STRATEGIES`; new v2 config field `v2_strategy_one_hot_columns` freezes the tape's one-hot vocabulary (740 for OTE), and the blocked strategy build adds a listed column as 0 only when a sibling of the same strategy output exists (else the strategy did not build and the column stays missing). The field is omitted from `to_dict` when unset. Tests: `tests/test_feature_v2_causal_contract.py` + ES contract 44 passed. Pre-check on the old tape with the fixed config: **0 of 1,781 eligible columns differ** at both EURUSD ends.

**OTE rerun: PASS.** `sh scripts/run_b2_v2_features_ote_rerun.sh` (`9e8297c7`, 10:57Z-12:33Z, build 4,285 s) -> `artifacts/b2_ote_features_v2_20261006b`: (543,191 x 1,820), 1,781 eligible / 33 excluded; **0 of 1,781 differ** at 2026-03-10, 2025-11-20 and held-out 2025-07-16. Log `logs/b2_v2_features_ote_rerun_20261006.log`.

**R2b inputs located (OTE):** `artifacts/b2_ote_prepared_20260921_v1/dataset.metadata.json` records `label_source_path` `artifacts/b2_ote_labels_20260920_v2/labels.csv.gz` and `event_source_path` `.../events.csv.gz`; `scripts/rebuild_b2_prepared.py` supports `--group ote` (6 targets).

**R2b COMPLETE** (`sh scripts/run_b2_v2_prepared.sh`, 12:37Z-14:26Z; ES groups 2-5 min each, OTE 98 min; log `logs/b2_v2_prepared_20261006.log`). Roots `artifacts/b2_{frvp_july,ict,ict_specialists,frvp_setups,ote}_prepared_v2_20261006` (all `manifest.json` present) and `artifacts/b2_frvp_recency_prepared_v2_20261006`. Two earlier launches were stopped within minutes and their partial roots deleted (12:34Z under the 2 h tool limit; plain `sh` without Git `/usr/bin` on PATH).

Check (`docs/audits/retrain_r2_tools_20261006/check_v2_prepared.py` -> `docs/audits/b2_v2_prepared_check_20261006.json`), all 25 targets:

- Merged rows, usable rows and positive counts identical to the v1 roots (ES 666,769; EURUSD 543,191): label alignment held.
- All readiness grades green; 0 selected features are v2-excluded, from a v2-dropped strategy, or `htf_confluence_*`.
- Run configs equal v1 except the new empty default `target_selection_fit_rows: {}`.
- **Pool expansion (decision needed before R3):** v2 pools contain columns the frozen v1 schemas never had (B2 repair kept the old saved schema): FRVP +163, ICT +36, OTE +246 (breaker / IFVG / `*_bars_since` / HTF source-age and -complete, plus strategy columns). None of v1's deliberately excluded columns returned (ES roll/identity 9; EURUSD 94 incl. `breakout_*` label helpers). Selected counts: FRVP ~590 -> ~700 (~588 of the old kept), ICT about the same, OTE ~1,050 -> ~1,065 (~970 kept).

**Pool expansion: ACCEPTED by the user (2026-10-06).** R3 trains on the v2 pools; each model still loads only its recipe's top `max_loaded_features` by the frozen-prefix ranking (FRVP 160, ICT 96, OTE 96-160 caps).

**R2c design** (new files only: the v1 runners stay byte-identical because earlier models' reviews compare their code snapshots with the live files):

- Roster `docs/audits/b2_v2_model_data_roster_20261006.csv` (sha256 `8d50d873f376c91def632db9eff3217bc3cdfa2c9398d12c822faf641e72e5ae`), built by `scripts/build_b2_v2_roster.py` (`baa099c4`) from the pinned v1 validation manifest (`e99d03f0…`): the 16 roster IDs, their unchanged recipes (hash-checked), stream, backend (ES all xgboost; 21/22/23/25 tcn, 26 xgboost), v1 dirs, v2 R2b dirs and the selection boundary pinned from v1 (ES 900/400/320, OTE 180k/225k; 21 and 23 share `long_ote`).
- Validation views `scripts/prepare_b2_v2_validation_views.py` (`295eb006`) -> `artifacts/b2_validation_prepared_v2_20261006`: per root, the recipe reruns with the pinned per-target fit rows (frozen-prefix fills, pruning and ranking). Per model: split identity (`source_row_idx`, `target`, `sample_weight`) equals the R2b v2 split **and** the v1 validation view; finite; frozen-prefix contract; FRVP long reversal recency kept. Resumable per root; writes `configs/b2_v2_validation_views_lock_20261006.json` when all 16 pass. **ES done before launch: 11/11 models pass** (incl. model 10 recency weights equal to v1).
- Source history `scripts/run_b2_v2_source_history.py` (`f8671116`) `prepare|check|train`, all 16 IDs, both backends. Reuses `run_b2_retraining.load_job/check_job`, `audit_b2_candidate_history.source_frame` and `run_b2_candidate_history.preflight` by import. ES: per-model tape from the v2 R2b `dataset.parquet`. EURUSD: one union tape for 21/22/23/25/26 with the 250 warm-up bars as an unavailable prefix (label-file bars after the prefix must equal the feature tape), split ids shifted by 250, per-model frozen fills. Outputs `artifacts/b2_v2_source_history_20261006/<id>`, training (R3) into `models/b2_v2_retrained_20261006`. A `review` mode (B3) is still to write. Dry run on model 18 (temp root, deleted): prepare + load-time endpoint equality + preflight over all 5 search windows passed.

**R2c COMPLETE** (`sh scripts/run_b2_v2_r2c.sh`, 2026-10-06 23:15Z to 2026-10-07 01:16Z; log `logs/b2_v2_r2c_20261006.log`). No model trained.

- Views: EURUSD root 23:16Z-00:27Z; all 16 models pass split identity (== R2b v2 == v1 view), finiteness and the frozen-prefix contract. Manifest `artifacts/b2_validation_prepared_v2_20261006/manifest.json` (`afba2f9c…`), lock `configs/b2_v2_validation_views_lock_20261006.json` (`30f42408`).
- Source history `artifacts/b2_v2_source_history_20261006/<id>`: ES 666,769 bars per model; EURUSD union tape 543,441 bars (250-bar warm-up offset), 257 union features. Load-time endpoint equality passed for every model.
- Checks `docs/audits/b2_v2_source_history_check_20261006/<id>.json`: **16/16 PASS**. Training tensors == runtime tensors for every search window (ES xgboost 5-9 windows; TCN 21: 9, 22/23/25: 3; 26: 3). Loaded features equal the recipe caps (FRVP 160, ICT 96, OTE 160/96/128/96/96). Fit rows equal v1. Minimum walk-forward source gap >= 1,071 bars for FRVP/OTE (ICT: 3,068-19,461 bars, plus corrected event windows for all 1,268-7,438 usable rows per target). Model 10 recency weighting kept. Trials per recipe: 10-60.

**R3 (retrain + review).**

- Review: `scripts/review_b2_v2_models.py` (`a99f295b`), separate from the training code so reviewer fixes never change the hashes each model pins. Mirrors the v1 candidate-history reviews for xgboost and TCN: training manifest, recipe, preparation and every pinned code file (live and snapshot) unchanged; outputs unchanged; selected features = frozen-ranking prefix; fitted history/fill contract; report; purged final holdout; scaler fit scope; full search budget, best trial and params; CV fold plans and frozen selection; OOF rows/folds; calibration contract; threshold reselected from OOF only; TCN fixed-duration refit; runtime replay of the test endpoints through the production backend loader + `RuntimeModelRunner` on the raw source history (raw and calibrated probabilities, every threshold decision).
- Orchestrator: `scripts/run_b2_v2_r3.py` (`1e4e5549`): per model, train (separate process; skipped if already trained; an interrupted output dir is reported FAILED, never deleted) then review (report `docs/audits/b2_v2_review_20261007/<id>.json`); a failure moves on to the next model. Status `docs/audits/b2_v2_review_20261007/status.json`; per-step logs `logs/b2_v2_r3_20261007/<id>_{train,review}.log`. Order: ICT 12-19, FRVP 2/3/5/10, OTE 26, 22, 21, 23, 25.
- Pilot: model 14 trained 01:26-01:28Z and **review PASS** (191 test rows, max |runtime - saved| 6.8e-8, all decisions equal).
- **R3 COMPLETE** (`python scripts/run_b2_v2_r3.py`, 01:29Z-13:51Z; log `logs/b2_v2_r3_20261007.log`): **16/16 trained and review PASS**, no failures. Runtime replay max |error| 6.8e-8 to 8.9e-8 (xgboost) and 1.8e-7 to 5.4e-7 (TCN); every threshold decision equal. Train time: ICT 8-46 min, FRVP 2.5-26 min, OTE 26 58 min, TCN 22/21/23/25 64/69/142/117 min. Metrics: `docs/audits/b2_v2_r3_metrics_summary_20261007.json`.

**R3 results (test split; v1 numbers come from the non-causal training the 2026-10-05 report rejected):**

| Group | Test AP v2 (v1) | Lift over base rate | ROC AUC v2 | Test trades |
|---|---|---|---|---|
| FRVP 2/3/5/10 | 0.68/0.49/0.54/0.73 (0.71/0.55/0.65/0.71) | 1.08-1.57x | 0.38-0.65 | 23-238 |
| ICT 12-19 | 0.28-0.45 (0.41-0.51) | 0.96-1.22x | 0.47-0.59 | 60-516 |
| EURUSD 21/22/23/25/26 | 0.22-0.32 (0.71-0.90) | 6.1-12.1x | 0.88-0.93 | **0, 0, 0, 14, 0** |

- *EURUSD: ranking skill but degenerate thresholds.* The OOF threshold selector picked cut-offs that only 1-2 OOF rows clear (thresholds 0.50-0.83 against OOF q99 0.18-0.42). Test scores never reach them, so 4 of 5 models never trade (25 trades 14 times, event recall 0.006). v1's AP 0.9 is the leak the causal fix removed. The selector has no minimum-support constraint; this is a policy problem, not a model one.
- *ICT: near-zero lift.* AP is within 0.96-1.22x of the base rate and ROC AUC 0.47-0.59. Event precision ~0.78-0.92 reflects high base rates and event tolerance, not skill.
- *FRVP:* modest lift (1.08-1.57x); 10 is slightly better than v1, 5 has ROC AUC 0.38.

**Plan approved (user, 2026-10-07):** EURUSD thresholds via the B3 threshold path; ICT 12-19 excluded from the v3 freeze (no edge); FRVP 2/3/10 carried, 5 flagged. Sweep goal: per model, the threshold with the most trades/week that keeps returns positive, floor 6 trades/week.

**B3 threshold sweep** (`scripts/sweep_b2_v2_thresholds.py` `5b65ff9d`; outputs `artifacts/b2_v2_threshold_sweep_20261007/<id>/{outcomes.pkl,sweep_select.csv,result.json}`; summary `docs/audits/b2_v2_threshold_sweep_20261007.json`; logs `logs/b2_v2_threshold_sweep_20261007*.log`):

- Opportunities = detector-fired same-side setups on each model's declared route (route bindings v2; same matching as the section 3 smoke replay). Returns = B1 engine on b1_v3 contracts, base costs, research scope b2_research_scope_v1, B1 `simulate_positions` one position at a time per model, ledger restarted weekly (an unresolved data-gap trade blocks only its week). 101 thresholds (score quantiles of the selection period).
- Selection on out-of-fold dev scores, evaluation once on the test split. 22/26: training rows exclude reversal confirmation bars (no OOF on setups), so test-period confirmation bars were scored by the final model via the production runtime (checked on 50 saved test bars) and split in time: first half selects, second half evaluates.
- Feasible = >= 6 trades/week, net > 0, PF > 1, positive in >= half the years (quarters for 22/26). ES tape: 67 bars off the pinned CME calendar dropped.
- **Result: 0 of 9 models has a feasible threshold.**

| ID | Max trades/wk (no threshold) | Setup-only dev: gross / cost / net per trade | Positive thresholds (of 101) | Best positive threshold on dev -> test |
|---|---|---|---|---|
| 2 FRVP cont | 5.8 | +0.17 / 2.83 / -2.66 ticks | 1 (0.14/wk) | net +83 -> -41 |
| 3 FRVP meta | 7.3 | -0.42 / 2.84 / -3.26 ticks | 1 (0.11/wk) | +196 -> -58 |
| 5 FRVP setup2 | 1.3 | +1.90 / 2.72 / -0.83 ticks | 4 (<=0.08/wk) | +42 -> -82 |
| 10 FRVP rev | 3.1 | -4.03 / 2.92 / -6.95 ticks | 1 (0.06/wk) | +15 -> -29 |
| 21 OTE long | 26.2 | -0.39 / 4.10 / -4.49 pips | 0 | - |
| 23 OTE long union | 26.5 | -0.34 / 4.11 / -4.45 pips | 0 | - |
| 25 OTE short | 27.0 | -0.22 / 4.09 / -4.31 pips | 0 | - |
| 22 OTE long rev | 9.2 | -0.31 / 3.93 / -4.24 pips | 5 (<=0.68/wk) | +17 -> -1 |
| 26 OTE short rev | 8.6 | -0.70 / 3.89 / -4.59 pips | 18 (<=1.78/wk) | +69 -> -391 |

- *Why:* (1) setup supply: 2, 5, 10 cannot reach 6/week even untested; (2) the B1 gross edge per setup is ~0, and for 2/3/5/10/21/23/25 the model score does not raise gross per trade (21 win rate falls 40% -> 31% as the threshold rises); 22/26 do rank (gross/trade -0.3/-0.7 at no threshold -> +4.3/+5.8 pips in their top 5%), but on 16-19 trades and far below 6/week; (3) costs: EURUSD b1_v3 charges the full configured spread per side (1.0-3.0 pips each) + 0.3 slippage + 0.35 commission, ~4 pips round trip; ES ~2.8 ticks.
- *Root cause:* the models predict their training labels (setup-quality targets), not the B1 contract outcome they are scored on.

- *Correction (2026-10-08):* the root cause above holds for OTE, but not for FRVP. The FRVP label agrees with the B1 outcome on the same setup bar (see the next section); the FRVP problem is that nothing predicts it.

## B1-outcome label research (2026-10-08): option A - COMPLETE, nothing frozen or activated

User chose option A (retrain on B1 outcomes) and asked to first settle the label and the eligible candidate bars. All work is offline research under new versioned roots; b1_v3, the B1 engine, the scope, the routes and every earlier artifact are unchanged.

**Tools.**

- `model_testing/b1_fast_outcomes.py` (`5e60acf1`) is a vectorized copy of `evaluate_signal` and `simulate_positions` for the sweep configuration.
  - Equality was checked against the engine on 1,197 real samples (`artifacts/b1_outcome_research_20261008/verify.json`): ATR on every period, outcome, barriers, exit, cost, net and ledger admission all equal.
  - It reproduces all 16,285 cached sweep outcomes of models 3 and 25.
  - `tests/test_b1_fast_outcomes.py` (`aa2e88f2`): 2 passed (synthetic tapes with gaps, corrupt bars, gap opens and an ES roll).
- `scripts/research_b1_outcome_labels.py` (`cafea70d`): stages `inputs | verify | census | everybar | report`. Caches are in `artifacts/b1_outcome_research_20261008/cache`.
- `scripts/report_b1_outcome_research.py` (`83f73c30`) writes `docs/audits/b1_outcome_research_20261008.json` (`fb0fd756`).
- `scripts/train_b1_outcome_models.py` (`3bd205a8`) writes models to `models/b1_outcome_research_20261008/<stream>/<K>_<label>` (research only, `manifest.json` status `research_only_not_frozen_not_installed`) and the report `docs/audits/b1_outcome_models_20261008.json` (`75e09f6d`). Logs: `logs/b1_outcome_models_20261008_{ES,EURUSD}.log`.

**Periods.**

- Development: ES before 2024-10-01 11:15Z; EURUSD before 2025-02-20 15:05Z. These are the earliest 2026-10-07 test starts of the roster models.
- After that is labelled "inspected" (the sweep already looked at it) and is never called a fresh holdout.
- A fresh holdout needs forward bars: live DB `canonical_bars`, read-only, has EURUSD 2026-04-08 to 09-22 and ES 07-29 to 09-22 (gappy). The detectors would have to be run on them. It was not spent, because nothing passed development.

**Label audit.**

- *FRVP 2/3/5/10.*
  - The label sits on the setup bar, and the training rows are the usable setup events (37% are excluded by macro, first-RTH, thin-session, past-HTF and cooldown rules).
  - Label = triple barrier from the next open. Barriers per setup are the same as b1_v3; the ATR is the label engine's 1h structural ATR mapped to 5m.
  - Positive = target first, with the target checked before the stop in the same bar. Timeout counts as negative, and there are no costs.
  - B1 differs as follows:
    - entry one bar later (known_at = close + 1 s, so entry at the open after next)
    - stop first on a collision; a gap open beyond the stop exits at the open
    - timeout earns its signed close
    - full costs
    - completed-hour ATR, with a missing bar giving INVALID: 28% of development setups, 2% of inspected
    - no label exclusions; roll and gap censoring
  - Agreement: label target hit → B1 net > 0 in 89%; otherwise 17%.
- *OTE 21/23/25 (long_ote / short_ote = OR of reversal, continuation and breakout) and 22/26 (reversal).*
  - Every bar is a row. Label 1 sits on a zone around the swing extreme: reversal swing −2..0, continuation −1..+1, breakout 0..+2.
  - The swing is only known at confirmation: reversal median 5 bars later (p90 19), continuation median 2.
  - Share of label-1 bars that come before the setup is knowable: reversal 100%, continuation 84%, breakout 0%.
  - Positive = no stop from the confirmation close (timeout counts as positive), plus a forward trend-scan pass and a quality floor. No costs.
  - At the confirmation bar: label target hit → B1 net > 0 in 95% (reversal) and 69% (continuation, breakout).
  - The OTE models therefore learn hindsight swing timing on every bar, which is primary-signal style, and are then scored at confirmation bars they never trained on.

**Setup-only baseline** (k = 0, each setup's own b1_v3 contract, full costs, one position, weekly ledger; development):

| Stream | Setups/wk | Executable/wk | B1 trades/wk | Gross/trade | Cost/trade | Net/trade | Positive years |
|---|---|---|---|---|---|---|---|
| ES FRVP 1-6, both sides | 56.8 | 40.7 | 15.2 | −0.18 t | 2.8 t | −2.97 t | 0 of 8 |
| EURUSD OTE 3 families, both sides | 84.8 | 79.7 | 42.1 | −0.17 p | 4.1 p | −4.27 p | 0 of 7 |

- Every family and side has gross per setup between −7.2 and +1.2 ticks (ES) and between −0.26 and +0.05 pips (EURUSD). Inspected is the same (ES −5.69 t/trade, EURUSD −4.25 p).
- Roster routes on development: 2: 7.2/wk, 3: 9.8, 5: 1.4, 10: 3.6, 21/23: 28.1, 25: 28.3, 22: 9.3, 26: 9.2.
  - IDs 5 and 10 have a hard setup shortage on their own routes.
  - Every route is net negative.
- *Is the contract too tight?*
  - ES: no. Target is ~30-40 ticks, cost ~2.8 (8%), geometry about 1:1; the issue is ~0 gross.
  - EURUSD continuation/breakout: yes. Target is 7-9 pips against ~4.3 pips of cost (50-60% of the target); breakeven win rate is 66-69% against an actual 30-36%.
  - EURUSD reversal: stop 2.0 ATR (~26 p) against target 1.0 ATR (~13 p); breakeven 72-73% against an actual 60%.
  - Even at zero cost, gross is ~0.
- *Binary vs net label:* wins and losses are unequal (reversal +8.9 / −23.9 p; breakout +4.3 / −8.5 p), so a win-rate label can rise while expectancy falls. Both labels were trained.

**Timing (development).**

- Pre-setup bars look good only in hindsight. ES entering at or before the setup-bar open gains +2.8 to +6.7 net; entering after the setup bar's close (k = −1) is −3.8. EURUSD shows the same pattern: k = −4..−3 is +0.1 net, k = −1 is −4.5. The move that defines the setup happens inside the setup bar.
- No causal precursor signal is defined, so negative offsets stay diagnostic.
- Post-setup valid bars (same session, no opposite or newer setup, no gap, own stop or target not touched): net is flat.
  - ES: k = 1..5 between −3.1 and −3.6.
  - EURUSD: about −4.4.
- Fixed-delay policies 0-5 bars: ES best at a 4-bar delay, 11.0/wk at −2.59; EURUSD all about −4.3. All infeasible.
- Hindsight best valid bar per setup: ES +4.8, EURUSD −1.7. This is a diagnostic upper bound only.

**Every bar** (template contracts, both sides; development; a primary-signal diagnostic):

- ES Globex day: 1,351 bars/wk, gross −1.4 to +1.1, net −1.6 to −4.2.
- ES RTH: 380 bars/wk, similar.
- EURUSD FX day (17:00 to 17:00 ET): 1,442 bars/wk, net −3.9 to −4.4.
- EURUSD London open to NY close: net −3.5 to −3.7.
- EURUSD London/NY overlap 08-12 ET: 183 bars/wk, net −2.7 to −3.2 (lowest cost).
- The existing every-bar models 21 and 25 do not rank every-bar B1 outcomes: gross per trade is about 0 at every threshold from the median to the 99.9th percentile.

**Retraining on B1 outcomes.**

- Grid: XGBoost (fixed parameters) × K ∈ {0, 1, 3} × label {net > 0, net units}.
- Rows: executable valid candidates. Predictors: decision-time raw v2 features plus cost and ATR context.
- Validation: 5 purged, setup-grouped walk-forward folds with a 1-day embargo. Thresholds are chosen on out-of-fold scores only.
- **0 of 12 feasible.**
  - ES: out-of-fold Spearman (score vs net) 0.00-0.01; net by score decile is flat.
  - EURUSD: Spearman 0.16-0.17, coming from cost and family context (net improves from about −6 to −3.3 p across deciles); gross stays about 0 and no decile is positive.
  - At about 6 trades/week the best out-of-fold result is EURUSD K1_binary at −3.84 p/trade (gross −0.25).
  - The inspected period is consistent (best at about 4.6/wk: −2.47 p/trade).
- IDs 22 and 26: the new models have out-of-fold scores on every setup, so their routes compare on equal terms. The old baseline for 22 and 26 stays the half-split runtime replay.

**Conclusion.** The setup-gated supply is enough at stream level (ES 41 executable/wk, EURUSD 80), but the gross edge per candidate is about 0. Relabelling with B1 outcomes, picking the entry bar, or scanning every bar does not change that. The required change is primary opportunity generation: a causal proposal generator whose proposals show positive gross B1 outcomes before any model. Alternatively, for EURUSD, a reviewed contract version (geometry or execution cost), which needs user approval. No B4 v3 freeze.

## Per-model floor of 1 trade/week, portfolio check (2026-10-08) - COMPLETE, nothing frozen or activated

User: drop 6/week as a per-model floor; take what each model can give at >= 1/week and check whether the models together reach 6/week. `scripts/portfolio_b1_floor_research.py` -> `docs/audits/b1_floor_portfolio_20261008.json`, research-model scores in `artifacts/b1_floor_portfolio_20261008/scores`, logs `logs/b1_floor_portfolio_20261008*.log`. Rule per model (development scores only, 200-quantile grid): most trades/week with >= 1/week, net > 0, PF > 1, positive in >= half the years (quarters for 22/26). Candidates: the 9 v2 roster models + the 12 B1-outcome research models.

- **2 of 21 qualify**: roster 26 (1.78/wk, +1.44 p/trade, PF 1.19, 50 trades) and research ES K3_net (1.38/wk, +0.15 t/trade, PF 1.01, 335 trades). Development rates sum to **3.16/wk < 6**. The other 19 have no threshold with >= 1/week and positive returns (EURUSD research: no positive threshold at any rate).
- Inspected period (already seen; not a fresh holdout): 26 at its threshold 2.34/wk, -6.41 p/trade, PF 0.44; ES K3_net 1.94/wk, +4.40 t/trade, PF 1.11, but positive in only 1 of 3 calendar years. Combined 4.3/wk. The report field `combined_net_per_model_scope` adds ticks and pips and is not a meaningful number.
- Conclusion: even without the per-model floor, the portfolio does not reach 6/week with positive returns. ES K3_net is the only candidate with positive development and inspected net; its development edge is break-even (PF 1.01) and was picked from 4 positive thresholds out of 200, so it is at most a record-only shadow candidate.

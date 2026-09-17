# Live app change journal

This is the single ongoing journal for live app changes. It consolidates the full ES A1 through A11 records from September 13?15, 2026, in milestone order. Consolidated September 16, 2026.

Record future live app changes in this document under **New entries**, including the date, scope, behavior changes, validation, and any remaining restrictions or follow-up work. The imported entries below preserve their original content and describe the state at the time each entry was written; later entries may supersede earlier ones.

## Imported entry index

- [ES A1: frozen evidence and collection boundaries](#imported-a1) ? source: `ES_A1_collection_baseline_20260913.md`
- [ES A2: causal ICT session and week references](#imported-a2) ? source: `ES_A2_causal_references_20260914.md`
- [ES A3: higher-timeframe alignment and live input contracts](#imported-a3) ? source: `ES_A3_input_contract_20260914.md`
- [ES A4: source observations, prediction time, and bar eligibility](#imported-a4) ? source: `ES_A4_bar_provenance_20260914.md`
- [ES A5: immutable FRVP and ICT setup history](#imported-a5) ? source: `ES_A5_setup_history_20260914.md`
- [ES A6: separate shadow scores, setup decisions and policy qualification](#imported-a6) ? source: `ES_A6_shadow_policy_20260914.md`
- [ES A7: execution-aware research ledger](#imported-a7) ? source: `ES_A7_research_execution_20260914.md`
- [ES A8: setup-first chart and separate research view](#imported-a8) ? source: `ES_A8_chart_presentation_20260914.md`
- [ES A9/A10: FRVP research priorities and S4 candidacy pause](#imported-a9) ? source: `ES_A9_A10_research_priorities_20260915.md`
- [ES A11: ICT research-only roster](#imported-a11) ? source: `ES_A11_ICT_research_roster_20260915.md`

## New entries

Append future dated changes here.

## Imported records: September 13?15, 2026

---

<a id="imported-a1"></a>

Source: `ES_A1_collection_baseline_20260913.md`

# ES A1: frozen evidence and collection boundaries

A1 completed on September 13, 2026. This is an operational preservation and reporting change, not a new qualified evaluation period or model promotion.

## Preserved evidence

- The existing `artifacts/es_collection_baseline_20260912` remains intact: 250 files, 21 model manifests, no missing objects. All object hashes were verified.
- The supplemental `artifacts/es_collection_baseline_20260913_A1` captures the current state: 952 files, 21 model manifests, no missing objects. It includes the September 11 audit and frozen SQLite snapshot, nested appendices, model/scaler/calibrator bytes, embedded policies and referenced policy lineage files, and input/policy source files. Copies are addressed and verified by SHA-256; preservation refuses to overwrite an existing destination.
- Both baselines identify the audit observations as `legacy-unversioned`. The supplemental model and source bytes are those present at preservation time. They do not establish which bytes generated the September audit; B2 still owns that lineage review.

Baseline manifest SHA-256 anchors:

| Snapshot | SHA-256 of baseline_manifest.json |
| --- | --- |
| September 12 | `4fa785f7060083fe636beaf7c4615c4ddb8a8d67ebf5cae3ac3a7bbca574f9dc` |
| September 13 A1 | `3dfbaffdb484fad1615ae0cc0f442001093c03502fb05084d67fe802ae1b77a0` |

Snapshots live under the repository's ignored `artifacts/` directory. Keep those directories with the research evidence; a source-only Git checkout does not include them.

To verify without opening or changing the live database:

```powershell
ote_venv/Scripts/python.exe scripts/preserve_es_collection_baseline.py --verify --destination artifacts/es_collection_baseline_20260913_A1
```

## Collection contract

The existing runtime collection plumbing now hashes input contracts, JSON feature recipes, SQL schemas, and shared policy implementation in addition to manifests, artifact bytes, feature producers, detectors, inference, and live policy code. The identifier is `es-input-observation-v1:<SHA-256>`. Model, calibration, threshold/policy, or producer changes therefore create separate collection identities at processor initialization. Restart the collector after changing its inputs or artifacts; running processors retain their loaded model and collection identity.

SQLite migration `0006_collection_versions.sql` labels pre-existing feature snapshots, predictions, and decisions `legacy-unversioned` without rewriting their payloads. New runtime records carry their collection identifier. Collection contracts are registered immutably in SQLite. The migration is explicitly included despite the repository's general SQL ignore rule.

`fetch_collection_predictions` requires an explicit collection when its model population spans multiple versions. Dashboard `compute_signal_markouts` now applies the same rule to its full signal population before applying the row limit. Callers can pass `collection_version=...`; returned rows retain it. `summarize_signal_markouts` rejects combined versioned frames. The default dashboard displays an unavailable explanation for a mixed population and labels single-collection results. Raw diagnostic signal history remains accessible.

Example of a deliberately selected diagnostic report:

```python
markouts = compute_signal_markouts(
    store, audit_repository, model_ids=(model_id,), decisions=("shadow",),
    collection_version=selected_collection_version,
)
```

These remain existing event markouts, with their existing units and assumptions. They are not the execution ledger requested by A7. The frozen September research scripts continue to use their frozen audit database.

## Validation and decision

- 78 tests passed across `test_es_collection_baseline.py`, `test_ote_live_dashboard_app.py`, and both ICT/FRVP paper-signal runtime guard suites. Coverage includes legacy migration, immutable contracts, model/policy/producer version changes, reporting separation before limits, explicit selection, nested snapshot capture, overwrite refusal, and corruption detection.
- The additional older `test_ote_live_phase5_operator_tooling.py` suite has four failures caused by the absent local `ote_live/runtime_manifests/long_ote_champion_v1/live_runtime_manifest.json`. The new report regression uses self-contained fixtures and passes.
- No collector was started or restarted, no broker order was introduced, and no paper trial was activated. Existing shadow posture and paper-trial activation/readiness restrictions remain in force. Thresholds and model artifacts were not tuned or replaced.
- A2 onward and B1-B5 remain separate prerequisites. Neither this preservation snapshot nor a newly generated collection ID constitutes approval to begin scored collection.


---

<a id="imported-a2"></a>

Source: `ES_A2_causal_references_20260914.md`

# ES A2: causal ICT session and week references

A2 completed on September 14, 2026. The existing partial correction was reviewed and retained, the unknown-timestamp lookup was fixed, and regression coverage was extended through the actual live feature engine.

## Cause and resulting behavior

The original reference builder shifted aggregated RTH sessions and weeks, then mapped them back through keys that existed only after observing current-period RTH data. Overnight references could therefore disappear until a future RTH bar arrived. Mapping the current session's first opening price to every row also exposed that price before the opening bar existed.

`ict/structure/liquidity.py` now selects the latest observed strictly earlier session/week aggregate directly. Current opens are carried forward only from their opening bar. The shared registry path is `IncrementalFeatureEngine` -> `FeatureDatasetBuilder` -> `features.feature_sets.ict_context` -> `ict.feature_sets.ict_context` -> `build_reference_level_features`; historical rebuilds use the same producer.

- Prior RTH high, low, and close remain available overnight. Under the existing session convention, the normal 16:00 New York bar belongs to the next session date and can reference the RTH session that just ended.
- Prior-week high and low use RTH aggregates in `W-FRI` periods. They do not require a current-week RTH bar to exist.
- Missing sessions/weeks carry the latest observed earlier aggregate. With no earlier observed period, values remain NaN. An unknown timestamp also remains NaN instead of selecting the last, potentially future, aggregate.
- The RTH open is unknown until the actual 09:30 New York opening bar is present. A first observed bar at 09:35 cannot supply it. The value resets with the session date, and the RTH gap remains unknown whenever its open or prior close is unknown.
- Midnight and 08:30 opens likewise publish forward from their respective observed bars, grouped by New York calendar date.
- Distances retain the same reference-minus-current-close, divided-by-ATR definition and inherit reference missingness.

The contract assumes chronological, left-labeled input bars. Carrying an observed aggregate does not prove a complete historical session/week was received. Missing opening bars, missing history, and source revisions are not repaired by looking ahead. Dropping old history from a bounded window is also distinct from appending future bars to a fixed prefix.

## Validation

119 tests passed across these commands:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ict_reference_causality.py tests/test_es_causal_feature_contract.py -q
ote_venv/Scripts/python.exe -m pytest tests/test_ict_phase1_scaffold.py tests/test_ict_detectors_phase2.py tests/test_ict_setup_detector_phase3.py tests/test_ict_paper_signal_runtime_guard.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_es_collection_baseline.py -q
```

The first command passed 22 tests. New checks compare every column of the reference surface for every nonempty prefix across overnight rollover, midnight, 08:30, RTH open, initial-balance completion, missing sessions/weeks, cold starts, both DST transitions, and a year/week boundary. They preserve non-default row indexes and assert explicit expected high/low/close/open/gap/distance values. A separate check covers invalid timestamps.

The live integration check uses the real builder and ICT stack, compares its reference output with the producer, repeats each prefix through the engine cache, and verifies that unverified model lineage remains diagnostic-only. It uses the existing local FRVP shadow manifest as configuration without loading model weights; the standalone reference tests require no local model artifacts. The second command passed 97 tests covering downstream ICT behavior, paper-trial guards, and collection boundaries. `git diff --check` passed for the touched tracked files.

## Decision and remaining boundaries

A2 is complete. No collector was started or restarted, no model artifacts or thresholds were replaced, and no paper trial or qualified collection period was activated. The A1 collection identity includes ICT producer source, so this source revision receives a distinct identity when a collector is next initialized; the frozen baselines remain unchanged.

B2 still owns selected-feature/dataset/setup/model lineage review and any required regeneration, retraining, or recalibration. A3 owns the remaining HTF/input-contract audit; A4 owns source completion, freshness, provenance, and broader calendar/gap verification. Passing these A2 checks does not establish those tasks as complete or verify deployed artifacts. Keep the ICT roster research-only and preserve existing drawdown, concentration, promotion, and paper-trial restrictions.


---

<a id="imported-a3"></a>

Source: `ES_A3_input_contract_20260914.md`

# ES A3: higher-timeframe alignment and live input contracts

A3 was completed on September 14, 2026. The existing partial HTF correction and diagnostic metadata path were reviewed, hardened, and tested. This is a feature/runtime verification task, not a model retraining or prospective trading experiment.

## Findings and changes

The ES producer uses left-labeled source bars. A 30-minute bucket contains the six five-minute bars starting at :00 through :25, and publishes at :30, on the completed :25 source row. Hourly buckets follow the same convention. Historical and live features share `features/feature_sets/htf_context.py`; the live engine sets source-bar duration from its manifest timeframe. Other instruments retain the existing legacy alignment default.

The existing count/endpoints checks rejected truncated, gappy, and explicitly provisional buckets. A3 additionally rejects duplicate/off-grid interior timestamps and nonfinite OHLCV constituents, which could otherwise pass those checks. Source completeness and age now align to original row indexes, including unknown timestamps. Unknown completion is rejected before event-dependent missingness is considered. Thirty-minute/hourly values become stale when the last completed source bucket is at least 30/60 minutes old. A missing swing can be legitimate only after the source checks pass.

Daily/weekly observed aggregates retain the previously implemented forward-only publication at the period close. Tests verify publication of the just-completed period across a weekend without an extra period shift. These aggregates describe observed history; they do not prove full session/week coverage. Market-calendar, source revision, whole-period completeness, and executable-entry qualification remain A4/A13 responsibilities. No deployed daily/weekly-dependent artifact was certified by this task.

`ote_live/features/input_contract.py` now identifies its rules as `es-causal-inputs-v2`. The contract:

- Separates available values, recognized event absence, not-yet-observable session values, missing producers, unverified producers, missing history/input, and stale inputs.
- Checks raw values before the existing sequence-model zero imputation. Infinite event values and insufficient transform history do not count as valid event absence. A constant full z-score window is legitimately undefined; a lag can inherit a known event absence only when its source row exists. Other unproven transform missingness remains diagnostic.
- Propagates missing producer/fallback dependencies through transform names, and checks both timeframes for transformed alignment scores.
- Checks every row actually consumed by XGBoost sparse lags or TCN/LSTM sequences. Historical input failures include row offsets and affected features; model-wide lineage failures are reported once. A current valid row cannot excuse stale inputs elsewhere in the inference window.
- Requires helper producer identity, version, explicit parity verification, and matching source timestamps. `source_timestamp` attests one row; `source_timestamps` explicitly attests a history of rows for lag/sequence inputs. Attestation changes invalidate the engine cache. These fields are a trusted producer interface, not evidence that the deployed helper lineage has passed B2.
- Rejects qualification if the builder's final row does not match the requested source timestamp, including when OHLC validation removes the newest bar.

## Fallback producer audit

The fallback zero is a compatibility value for diagnostic inference. It cannot establish that a real producer evaluated a setup and found no confluence.

FRVP training helpers originate in `data/labeling/frvp_labeling_engine.py`: the label-output assembly writes event-specific 30-minute/hourly swing-match flags at event indexes. The FRVP Phase 4 feature pipeline deliberately carries these helpers through. ICT helpers originate in `ict/labeling/ict_labeling_engine.py`: output assembly writes the event's HTF-context alignment flag into family/meta columns. Neither label-output assembly is a live feature producer. Replacing these flags with generic trend alignment would introduce a different definition; A3 does not do that.

The reproducible [dependency inventory](../research/es_a3_input_contract_20260914/model_input_dependencies.json) contains manifest paths, hashes, selected HTF/helper names, status, and static blockers. It covers 21 packaged manifests: the September audit's 20-model roster plus the retained deprecated FRVP short reversal. It does not inspect a running collector or load model weights.

Eight non-retired models select helpers without a packaged live producer:

| Models | Selected helpers |
|---|---|
| FRVP long continuation, long meta, long S3, long S5 | `htf_confluence_long_frvp_continuation` |
| FRVP short continuation TCN | `htf_confluence_short_frvp_continuation` |
| FRVP short meta | `htf_confluence_short_frvp_continuation`, `htf_confluence_short_frvp_reversal` |
| ICT long meta | `htf_confluence_long_ict_continuation`, `htf_confluence_long_ict_reversal` |
| ICT short meta | `htf_confluence_short_ict_continuation` |

Of the 20 non-retired models, 18 select `ict_*`/`htf_*` inputs and lack the corrected input-contract version. They remain diagnostic-only pending lineage review. FRVP S1 and S4 have no static blocker from that particular prefix check; they still require runtime input checks and every existing policy/readiness restriction. This does not approve S4's provisional policy. The retired FRVP short-reversal model remains retired.

## Runtime visibility and qualification

The live engine publishes per-model contract metadata, including cache hits. The signal processor records contract changes as `signal_runtime.input_contract` health events and persists the contract and diagnostic reasons in each feature snapshot's observation metadata. The health-event dashboard query exposes that payload. ES input failures force shadow mode; raw scores remain available and no qualified shadow entry is assigned. A fresh bar alone cannot clear an input failure. An integration test exercises the real processor, decision engine, SQLite persistence, reconstruction, and health query with a deterministic test scorer.

## Validation and reproduction

The broad command passed 156 tests in 137.93 seconds. After adding explicit multi-row helper attestation coverage, the final focused `tests/test_es_causal_feature_contract.py` run passed all 33 tests (157 distinct tests across the two runs). Existing feature-builder warnings concerned undefined correlations and pandas copy-on-write deprecation. Diff whitespace checks passed for the touched tracked files.

```powershell
ote_venv/Scripts/python.exe scripts/audit_es_a3_input_dependencies.py --output research/es_a3_input_contract_20260914/model_input_dependencies.json
ote_venv/Scripts/python.exe -m pytest tests/test_es_causal_feature_contract.py tests/test_ote_live_feature_engine.py tests/test_features_builder.py tests/test_ict_reference_causality.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py tests/test_es_collection_baseline.py tests/test_ote_live_signal_runtime_integration.py tests/test_ote_live_runtime_manifests.py -q
```

Coverage includes exact HTF constituents/completion boundaries, prefix causality, missing/provisional/unknown source bars, invalid timestamps and values, transform/history missingness, helper attestation/cache changes, real live-builder parity, lag/sequence history, diagnostic persistence, and existing collection/paper-trial guards. The older policy-frame column assertion was updated for the seven provenance fields already present in the workspace; no production provenance schema was changed by A3.

## Decision

Mark A3 complete. Keep affected scores diagnostic-only and ICT research-only. B2 must review historical features, event-helper timing/definitions, prepared datasets, and deployed artifacts before any corrected manifest is attested. Do not fix that restriction by merely editing a version field. Other qualification prerequisites, promotion/economics/drawdown/concentration gates, and paper-trial launch guards continue to apply.

No collector was started or restarted; no broker orders, paper trial, qualified collection period, thresholds, model artifacts, or frozen manifests were changed. The A1 source-content identity will distinguish this revision when the collector is next initialized.


---

<a id="imported-a4"></a>

Source: `ES_A4_bar_provenance_20260914.md`

# ES A4: source observations, prediction time, and bar eligibility

A4 was completed on September 14, 2026. This task reviewed the existing provenance implementation, closed runtime/calendar/persistence gaps, and verified it with deterministic ES examples. It is an app contract change, not a new market experiment or the start of qualified collection.

## Preserved evidence

`MarketBar` carries the source timestamp, version, explicit completion flag, feed type, first and last observation times, and observation kind. Canonical SQLite rows retain these fields. The append-only `source_bar_history` retains provisional callbacks and revisions, bypassing dashboard snapshot throttling; SQLite triggers reject updates/deletes. A legacy canonical row is snapshotted before its first corrective overwrite. Unknown legacy completion/feed evidence remains unknown.

IBKR callbacks preserve their original feed classification. Changing the current subscription from delayed to live does not relabel older callbacks. Completion creates a new version; historical/backfill callbacks remain identified as such. The newest historical bucket stays provisional, including when history arrives out of order. Local aggregates retain constituent timestamps, versions, completion, feed, observation kind, and first/last observations, and require complete expected constituent coverage.

A4 additionally carries the earliest observed receipt from persisted provisional history into later callback revisions and canonical publication. This covers a collector reconnect before the bar was ever published canonically. The last receipt still describes the incoming version.

## Prediction time and freshness

The source bar's timestamp remains the prediction's `timestamp`; it is not the prediction observation time. Each model receives its own `prediction_recorded_at_utc` immediately after its inference returns, rather than sharing the bar timestamp or the first model's clock. SQLite separately records its actual insertion clock in `model_predictions.recorded_at_utc`. These distinct clocks are retained, not backdated. The integration test deliberately separates them by 20 milliseconds.

`complete-observed-live-bar-90s-dated-calendar-v2` requires explicit completion, a closed/aligned source bar, matching source timestamp, known ordered receipt times, known version, live feed and observation kind, and a prediction age of at most 90 seconds after bar close. Unknown last observation and completion observed before bar close are rejected. A second model finishing at 91 seconds fails even when the first model passed at 10 seconds. Backfill/replay/repair, delayed/frozen, and provisional evaluations have explicit classifications and rejection reasons.

The gate's `earliest_entry_at` is only a lower bound from the model observation clock, available when the bar gate passes. It does not assign a fill. Persisted observation metadata continues to set `qualified_shadow_entry=false`, `qualification_status=prerequisites_pending`, and executable entry time/price to null. Delayed/backfilled evaluations cannot acquire an earlier historical entry. A6/A7 and B1/B3 must define and check actual executable availability, including subsequent policy/persistence latency, before creating entries. A3/B2 model input and artifact restrictions still apply independently.

## Calendar and gaps

ES qualification now requires dated IBKR trading-hours coverage for the source interval and prediction time. The weekly/DST fallback and the existing September 7 closure remain useful for diagnostic gap detection, but cannot certify an unlisted holiday. Missing, expired, malformed, or invalid-timezone schedules fail qualification with `market_calendar_unverified`. The received schedule and timezone remain in the source snapshot. This is not a hardcoded all-holiday calendar.

The parser accepts legacy same-day hours, dated endpoints, multiple intervals, overnight sessions, explicit `CLOSED` dates, and trailing delimiters. It rejects partially malformed schedules rather than treating a partially parsed schedule as authoritative. Tests cover both DST Sunday reopens, the daily maintenance break, weekends, dated early closes, a halt inside a bar, and predictions recorded after an early close. The broker format is documented in [IBKR ContractDetails](https://interactivebrokers.github.io/tws-api/classIBApi_1_1ContractDetails.html); exchange holiday schedules are maintained on [CME's trading-hours page](https://www.cmegroup.com/trading-hours.html).

The live processor now passes the preceding source timestamp into the bar gate. It freezes this timestamp with the evaluated snapshot, including for a one-bar feature window. A missing expected market bar rejects eligibility; a covered scheduled closure does not. Missing calendar coverage across a gap produces `gap_calendar_unverified`. A first observation without a predecessor does not claim to prove earlier history; model warmup/input contracts remain separate prerequisites.

ES gap recovery requires explicitly complete bars matching the gap's asset and timeframe. Provisional/unknown completion and wrong instruments/timeframes cannot mark the gap resolved. Repaired observations remain labeled as backfill/repair.

The collection payload explicitly records provenance, freshness, and calendar contract versions. A1's source-content hashing also partitions these changes from earlier collections without altering frozen baseline artifacts.

## Validation

The final combined regression command is:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_bar_provenance.py tests/test_es_causal_feature_contract.py tests/test_ote_live_ingestion_health.py tests/test_ote_live_ingestion_service.py tests/test_ote_live_storage.py tests/test_ote_live_signal_runtime_integration.py tests/test_ote_live_parallel_signal_processor.py tests/test_ote_live_runtime_collector.py tests/test_ote_live_audit_replay.py tests/test_es_collection_baseline.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py tests/test_ote_live_ibkr.py tests/test_ote_live_ingestion_normalizer.py tests/test_ote_live_ingestion_aggregator.py tests/test_ote_live_setup_revisions.py -q --tb=short
```

The final combined run passed **234 tests in 67.55 seconds**. Tests exercise actual processor/decision-engine/SQLite persistence with deterministic scorers, source callback history, first-receipt recovery across database reopen, calendar/gap handling, aggregation, audit replay, collection isolation, and existing paper-trial authorization guards. Diff whitespace checks passed for the touched tracked files.

No collector was started or restarted, and no broker order, paper trial, model artifact, threshold, or frozen manifest was changed. Recorded-example end-to-end qualification and execution accounting remain A13 after their prerequisites.


---

<a id="imported-a5"></a>

Source: `ES_A5_setup_history_20260914.md`

# ES A5: immutable FRVP and ICT setup history

A5 was completed on September 14, 2026. This is an app persistence change, not a market experiment or the start of qualified shadow collection. It completes the existing setup-history implementation and connects its source-correction reconciler to the actual single-group and shared ES processors.

## Stored contract

`ote_live/storage/setup_events.py` maintains three append-only SQLite tables under `setup_history_v1`:

- `setup_events`: every observed FRVP/ICT rule candidate, including detector-priority, repeat/spacing and other detector rejections. Records preserve strategy, type, side, family, rule confidence, selected/research flags, geometry, detector identity and source digest, detector configuration where exposed, source timestamp/version and complete source-bar payload, actual observation time, first setup-observation time, collection version, and predecessor/revision identity.
- `setup_prediction_links`: associated prediction IDs, model IDs, existing decision/reasons, probability and threshold, selection/direction diagnostics, and copies of the prediction and its metadata. Links require the same collection, asset, timeframe and source timestamp. Rule confidence remains separate from model probability. These are evaluation associations; they do not claim that every candidate matches the model's setup policy.
- `setup_event_outcomes`: named, versioned outcome observations attached to each original or revised setup record. Corrections append records rather than overwriting earlier outcomes.

Event keys partition collection, instrument/timeframe, strategy, source timestamp, setup type/side/family and research status. Identical reobservations reuse their event ID. Changed content appends `revised`; disappearance after a successful detector recomputation appends `invalidated`. Reappearance appends another revision. SQL update/delete triggers protect all three tables, and insert-conflict triggers also prevent `INSERT OR REPLACE` from erasing records through another connection. A failed detector-result write rolls back the entire result.

The history has no dashboard-sized retention limit. `list_events`, `history`, `prediction_links`, and `outcomes` expose it for research queries. Ordinary audit retention does not prune these three tables. Prediction copies remain readable even if the separate prediction table is later archived/pruned. Backups must include the SQLite database; the existing monthly audit export does not yet export these new tables.

## Runtime and source corrections

Setup collection runs before model warmup, threshold and policy filtering for each enabled ES FRVP/ICT strategy. A detector failure preserves previous observations, records a health error and marks that evaluation diagnostic. Successful empty results are recorded in the source-evaluation cursor, so ordinary no-setup bars are not unnecessarily replayed.

The reconciler now runs on ordinary processing, startup seeding, and idle polls, including serial reconciliation before the shared processor's parallel feature work. A new historical source bar or correction triggers detector recomputation from causal prefixes through the last processed timestamp, including bars that previously produced no setup. It uses canonical bars and the runtime's bounded feature history, preserving supplied feature context. It refreshes the retained live source window before continuing. Provisional source revisions await canonical publication.

Work is bounded to 128 prefix evaluations per reconciliation call. Pending ranges and per-bar progress persist across process/database reopen; producer failures leave the failed prefix available for retry. Failed/pending reconciliation is surfaced in diagnostic reasons. Reconciliation appends setup history and raw follow-up observations only: it does not rerun old predictions, create earlier executable entries, send notifications or invoke paper ledgers.

Collection identity already hashes detector, feature, ingestion and storage sources, so these changes produce a different collection version from the preceding implementation. Existing observations are not silently imported into the new collection. Source history retained before an initial fresh collection is watermarked; this task does not manufacture a historical archive of events the app never observed.

## Rejected setups and later observations

Every subsequent observed candle within 60 elapsed minutes is retained for selected and rejected setups, including earlier revisions and setups later invalidated. Each follow-up stores its OHLC/source provenance and version, elapsed seconds, reference source close and directional close change. Repeated identical follow-ups are deduplicated; revised follow-ups append. Source-correction replay also fills follow-up observations for newly discovered historical candidates.

These are descriptive raw price observations (`raw_followup_bar_v1`, `diagnostic_raw_candles_elapsed_60m_v1`), with `executable_entry=false`. Gaps remain gaps; an absent follow-up is unknown, not a loss or a zero return. They do not infer intrabar stop/target ordering, training labels, fills, costs or realized P&L. B1/B3 and A7 still own the scored outcome/position contract and censoring. A6 still owns full prospective policy qualification and candidate-specific matching semantics. All qualified-entry flags remain false.

## Validation

The deterministic integration test uses the actual processors, feature-engine interface, decision engine and SQLite persistence with stubbed market feature production, detectors and scorers. Both separate and parallel processing retain 43 rejected setups per strategy, attach predictions and follow-up observations, consume a historical correction during an idle poll, and preserve the history after reopening the database without adding historical predictions.

Repository/reconciliation tests also cover actual FRVP/ICT detector candidate extraction, versions and invalidations, missing producers, atomic rollback, cross-collection/instrument/time linkage rejection, SQL replacement attempts from another connection, causal-prefix replay of no-fire/backfilled bars, and retry after a producer failure and database reopen.

The final combined regression run passed **182 tests in 28.47 seconds**:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_setup_event_history.py tests/test_ote_live_setup_revisions.py tests/test_ote_live_setup_history_runtime.py tests/test_ote_live_signal_runtime_integration.py tests/test_ote_live_parallel_signal_processor.py tests/test_ote_live_bar_provenance.py tests/test_es_collection_baseline.py tests/test_ote_live_storage.py tests/test_ote_live_archive.py tests/test_ote_live_ops.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py tests/test_frvp_setups.py tests/test_ict_setup_detector_phase3.py -q --tb=short
```

The archive regression initially exposed a pre-existing date-dependent fixture: it expected a manifest created at today's wall clock to be in an April archive. The test now pins the fixture's repository clock to April. Production archive behavior was not changed. Whitespace checks passed for the touched files.

No collector was started/restarted, no broker orders or paper trials were activated, and no model artifacts, thresholds, frozen manifests or promotion restrictions were changed.


---

<a id="imported-a6"></a>

Source: `ES_A6_shadow_policy_20260914.md`

# ES A6: separate shadow scores, setup decisions and policy qualification

A6 was completed on September 14, 2026. This is an app decision/audit change. It does not start a qualified evaluation period, activate a paper trial or change model artifacts, frozen manifests, thresholds or execution policies.

## Versioned observations

ES FRVP/ICT shadow decisions now carry `shadow_evaluation` under `es-selected-setup-full-policy-v1`. The signal remains `decision="shadow"`. The evaluation is stored in signal JSON, feature/prediction/signal audit metadata and immutable setup-prediction links. ES shadow observations are persisted even with the decision engine's legacy emit-only persistence default, provided an audit repository, snapshot and manifest are supplied.

| Field | Meaning |
| --- | --- |
| `raw_score`, `calibrated_probability` | All-bar model diagnostics; the raw model output and calibrated probability remain distinct. |
| `threshold_crossed` | Probability clears the existing resolved global/regime threshold. This is an above-threshold observation, not only the first bar of an upward crossing. |
| `setup_match` | Per-event matches/rejections and immutable matching setup-event IDs, independent of probability. |
| `setup_matched_decision` | Threshold passes and a selected setup matches. |
| `full_policy_passed` | Setup/threshold decision passes every enabled policy check and existing specialist hold reasons, using a complete policy. |
| `policy_candidate_eligible` | Full policy passes, and A3/A4 inputs, bar provenance/freshness and setup-history diagnostics also permit the observation. |
| `qualified_shadow_entry` | Always false until the separate outcome, lineage/baseline, comparison and research-ledger prerequisites are implemented and approved. |

`prerequisite_reasons` explicitly identifies B1, B2, B3 and A7. `executable_entry_at` and `executable_entry_price` remain null. A7 owns fills, costs, positions and simulated outcomes; acceptance state here is hypothetical policy cadence, not open exposure. Existing economics, concentration, readiness and promotion restrictions remain in force.

## Setup matching

`selected-same-bar-side-family-or-specialist-v1` is an explicit prospective experiment. The selected setup must have the same strategy, side, collection, instrument, timeframe, source timestamp and bar version, and must have been observed by prediction-recording time. Rejected, research-only and invalidated candidates cannot match, but retain their prediction associations and individual rejection reasons.

Pooled reversal/continuation models require a matching broad family. Pooled meta models accept either family on their own side. Specialists additionally require their exact setup type. The routes use the existing model naming convention and specialist parser; unknown routes fail closed. Matching is restricted to the current source bar and does not carry an earlier setup forward.

This pooled setup gate is **not the historical deployed policy**. The payload records that distinction explicitly. Existing non-shadow specialist matching and execution predicates remain unchanged. The existing specialist hold reasons are also honored in shadow evaluation.

## Policy gates and state

Every observation evaluates enabled stress, session, composite-regime, composite/session pair, composite/stress pair, probability-quantile, expected-move/spread and cooldown checks. Each check records `passed`, `rejected`, `unverified` or `disabled`, its relevant inputs and rejection reasons. Gates are evaluated independently through the existing `evaluate_live_abstain` predicate, so a high-stress rejection does not hide simultaneous spread or cooldown failures.

The spread gate retains the existing session spread assumptions, policy overrides, default fallback and expected-move units; it identifies the spread as a policy assumption. It does not replace that contract with a live-quote spread or infer a fill price. Missing/nonfinite inputs required by an enabled gate are unverified and fail this experimental variant closed. Legacy execution's optional-input behavior is unchanged. Disabled abstention remains disabled. Regime thresholds and global fallback use the existing resolver without tuning.

Shadow cooldown and probability histories are separate from execution state and partitioned by collection, model, asset/timeframe, experiment version and the complete policy SHA-256. Fresh, input-verified, selected-setup threshold candidates enter quantile history; an otherwise rejected candidate remains in that history, matching the existing candidate-history convention. Only eligible full-policy candidates advance hypothetical cooldown. Delayed/backfilled observations, off-setup scores and below-threshold scores do not consume that state. The existing inclusive cooldown boundary and 512-candidate history limit are preserved.

Each evaluation persists state before and after the decision. A new engine restores the latest matching state from SQLite, and state is committed to its in-memory cache only after the audit path succeeds. Audit replay rebuilds features/predictions and re-evaluates policy using recorded setup associations, observation contracts and preceding state. It does not reconstruct historical setup detection from today's revised bars. Legacy records without this contract retain their old replay behavior.

## Query and runtime integration

`LiveAuditRepository.list_shadow_evaluations(collection_version=..., model_id=..., stage=...)` exposes `raw`, `threshold`, `setup`, `policy` and `qualified` observation views. The policy view uses `policy_candidate_eligible`. It rejects an implicit selection across mixed collections and returns prediction/signal IDs for audit linkage. Qualified results are empty under the pending prerequisite contract.

Both separate and shared parallel ES processors use the immutable A5 candidates. Each setup-prediction link now distinguishes candidate-specific `shadow_setup_matched` from the older specialist gate diagnostic, which did not impose pooled matching. Collection identity already hashes policy, contract, storage and ingestion sources, so the implementation creates a new collection identity on the next collector startup. No collector was started or restarted as part of this task. Dashboard presentation changes remain A8.

## Validation

Deterministic tests cover both FRVP and ICT routes, selected/rejected and mismatched candidates, revision/source/observation-time checks, simultaneous exclusions, missing inputs and valid JSON persistence, all observation query stages, mixed-collection rejection, regime/global thresholds, disabled gates, quantile history, cooldown boundaries, execution-state isolation, SQLite reopen and actual audit replay with restored cooldown. Runtime tests exercise setup-specific prediction linkage through separate and parallel processors.

The broader regression also exposed a stale audit-store test fixture pointing to the removed `long_ote_champion_v1` directory. The fixture now uses the available `long_ote_meta_tcn_champion` manifest; production manifest selection is unchanged.

The combined regression passed **194 tests in 70.27 seconds**:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_shadow_policy.py tests/test_ote_live_setup_history_runtime.py tests/test_ote_live_model_serving.py tests/test_ote_live_signal_runtime_integration.py tests/test_ote_live_parallel_signal_processor.py tests/test_ote_live_bar_provenance.py tests/test_ote_live_audit_replay.py tests/test_ote_live_audit_store.py tests/test_ote_live_storage.py tests/test_ote_live_archive.py tests/test_es_collection_baseline.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py tests/test_frvp_paper_signal_ledger.py tests/test_ict_paper_signal_ledger.py tests/test_ote_live_setup_event_history.py tests/test_ote_live_setup_family_tabs.py -q --tb=short
```

Python compilation and whitespace checks also passed. After adding an unknown-setup-family rejection assertion, the final focused rerun of `tests/test_ote_live_shadow_policy.py` and `tests/test_ote_live_setup_history_runtime.py` passed **33 tests in 10.09 seconds**, including both runtime modes.


---

<a id="imported-a7"></a>

Source: `ES_A7_research_execution_20260914.md`

# ES A7: execution-aware research ledger

A7's app implementation was completed September 14, 2026. The ledger supports explicit diagnostic replay; it does **not** start scored collection. B1/B3 remain unchecked and no outcome or prospective comparison contract has been finalized by this work. No collector was started or restarted, no broker integration was added, and the frozen FRVP/ICT paper-trial ledgers and activation restrictions remain intact.

## Implementation and contract boundary

- `ote_live/contracts/research_execution.py`: validated, immutable run assumptions and opportunity inputs.
- `ote_live/storage/research_execution.py`: SQLite execution accounting, immutable evidence and lifecycle history, A5/A6 adapter, and partitioned reports.
- `ote_live/scripts/replay_research_execution.py`: explicit JSONL diagnostic replay command.
- `ote_live/examples/research_execution/`: synthetic contract and input stream for reproducing the smoke check.
- `tests/test_ote_live_research_execution.py`: execution, persistence, linkage, isolation and censoring checks.

`ResearchContract` accepts only `mode="diagnostic"`; supplying `scored` fails validation. The contract requires explicit outcome/comparison references, collection version, model roster, entry delay and wait limits, freshness, bracket exits, timeout, costs, quantity and concurrency assumptions. There are no default economic parameters. Each run binds its exact JSON and start time; changing those assumptions requires a new run ID. Collection identity already hashes the added storage/contract source files on subsequent collector startup.

This version implements a fixed stop/target bracket relative to the simulated fill and a completed-bar timeout. It has no trailing stops, scaling, partial fills, quote execution or inferred detector-specific exit geometry. Distinct family exit contracts can be tested in separate runs. If finalized B1 requires different management semantics, implement and version them before scoring. Draft reference strings do not constitute approval or readiness.

B1/B3 must still settle the family-specific outcome rules, comparison population, execution assumptions, reporting windows and stress cases. B2/B4/B5 lineage, baseline and promotion restrictions also remain applicable. Scored runtime registration and collection wiring must be introduced explicitly against those finalized prerequisites; this diagnostic API offers no activation switch. The historical A6 payload, including `research_ledger_A7_pending`, is retained unchanged to preserve audit replay. That gate remains closed until the finalized contract is connected to scored collection.

## Compared opportunities and evidence

`submit_audit(run_id, signal_decision_id=..., setup_event_id=...)` consumes persisted A6 shadow decisions and requires the immutable A5 setup-prediction association. It checks collection, instrument/timeframe, source version/time, actual prediction time, and the setup revision known at that time. It retains the setup payload, prediction, signal and observation metadata in the research opportunity, along with detector/policy identities and audit IDs.

Every submitted opportunity produces records for all three variants and every declared cost case:

| Variant | Selection |
| --- | --- |
| `setup_only` | Matching selected setup and common input/observation eligibility |
| `setup_threshold` | Same opportunity plus the existing resolved threshold |
| `setup_full_policy` | Same opportunity plus threshold and persisted A6 full-policy checks |

The setup-only variant is a **matched-cohort control**: it shares the model's actual prediction-recording availability with the filtered variants. It does not claim to measure earlier standalone rule execution or opportunities where no prediction was recorded. Unmatched/rejected candidates remain recorded with their reasons. A6's pooled setup requirement remains an explicit experimental gate, and its policy cooldown remains the existing hypothetical candidate cadence, independent of research fills.

The adapter deduplicates by setup-event identity and model. It rejects attempts to substitute different prediction/revision evidence for the same opportunity. The lower-level `submit` method accepts explicitly supplied diagnostic fixtures; it does not certify their provenance. It still enforces collection/model membership, causal timestamps, source-to-prediction freshness and immutable opportunity identity.

## Execution and positions

Availability equals actual prediction-recording time plus configured latency. Entry uses the first supplied eligible bar open at or after that availability, within the configured wait limit. For example, a decision recorded at 14:35:01 cannot fill at 14:35; on a five-minute stream its earliest possible open is 14:40. Both that hypothetical fill time and the later time when the completed bar reveals the fill are retained. Replay follows observation time and rejects retroactive submissions.

Completed/fresh live provenance and dated broker-calendar coverage are required for execution bars. The A4 assessment checks gaps, alignment, source/completion identity and calendar coverage. The single deliberate distinction is that a final session bar can resolve an already-resting hypothetical order when its completion is observed during the immediately following scheduled closure; the A4 `prediction_market_closed` reason applies to new decisions, not this exit evidence. Missing expected bars remain censored, while verified closures do not consume holding bars.

Each variant/cost case owns an independent book. Limits apply either per model or across the declared roster, as configured. Position count, aggregate contract quantity and opposite-side exposure are checked before entry. Simultaneous candidates are ordered deterministically by availability, opportunity key and position key. Existing positions reserve capacity at the bar open; an intrabar exit cannot make that capacity available retroactively at the same open. A capacity rejection is final for that opportunity.

The entry bar counts as the first holding bar. Stop gaps fill at the worse opening price. Target gaps receive the target price without favorable gap improvement. If both barriers occur in one OHLC bar, the contract chooses stop-first or censoring. Timeout exits use the last allowed bar's close. Exit evidence records the bar interval and observation time, without inventing an intrabar timestamp.

For each resolved position:

```text
gross_ticks = side * (exit_price - entry_price) / tick_size * quantity
cost_ticks = (spread_ticks + 2 * slippage_ticks_per_side
              + commission_ticks_round_trip) * cost_multiplier * quantity
net_ticks = gross_ticks - cost_ticks
net_dollars = net_ticks * tick_value
```

Costs are a round-trip debit, charged once at resolution; they are not also embedded in the simulated price. Baseline cost multiplier 1 is mandatory; stress cases must be unique and at least 1. No cross-case or cross-variant combined P&L is reported.

## Persistence, censoring and reporting

Research tables are created only when the explicit research API is instantiated. `research_runs`, `research_opportunities`, `research_bars` and `research_transitions` are append-only, including protection against `INSERT OR REPLACE`. A separate cursor persists observation order and terminal status. Each operation uses a transaction, and position state reconstructs from the latest durable lifecycle snapshots after a database reopen. Exact opportunity/bar retries are idempotent; conflicting evidence fails.

`report(run_id)` requires an explicit run and returns its contract/hash plus separate lists for:

- realized position outcomes;
- open exposure and pending entries;
- rejected/expired entries;
- censored outcomes and unresolved exposure;
- previously resolved outcomes tainted by later source/execution-bar revisions.

Missing, delayed, incomplete, backfilled or unverified bars censor affected pending/open positions instead of fabricating zero P&L. An uncertain position retains its capacity reservation. A pending order that could already have filled also reserves uncertain exposure, without inventing an entry price. An end-of-stream `finish` censors remaining positions rather than force-closing them; omitting it preserves open exposure for continuation. Unresolved reservations require a separate, explicitly reconciled future contract to release them; v1 never assumes they disappeared.

Revisions preserve original fills and lifecycle evidence. A revision to a resolved trade's source bar or entry-through-exit interval marks it separately in the report. Out-of-order/revised bars conservatively censor current pending/open positions. Backfill does not repair an uncertain outcome into a scored trade.

This ledger stores `position_level_simulated_pnl`. It neither reads outcomes from nor writes outcomes into the training-label, frozen paper-contract or A5 overlapping-event-markout stores. Use a dedicated database for JSONL replay. The ordinary live audit archive/prune mechanism does not archive these research tables; retain the research database separately. The in-database A5/A6 adapter is explicit and is not invoked by current live processors.

## Reproduce the synthetic smoke check

The supplied example contains invented ES prices and test identities. Its economics demonstrate accounting only; they are not B1/B3 assumptions or evidence about either strategy.

```powershell
ote_venv/Scripts/python.exe -m ote_live.scripts.replay_research_execution --contract ote_live/examples/research_execution/synthetic_contract.json --events ote_live/examples/research_execution/synthetic_events.jsonl --database tmp/es_a7_example/research.sqlite --report tmp/es_a7_example/report.json --run-id synthetic-a7 --started-at 2026-09-14T14:30:00+00:00
```

The expected report has six resolved records: three variants times two cost cases. All enter at 14:40 for 100 and hit the target at 102 in the next bar. Gross P&L is 8 ticks per independent position, with 1.9 baseline or 3.8 stressed cost ticks. `scored_collection_enabled` is false. No `finish` event is included; the CLI can resume an open run from another observation-ordered stream.

## Validation

The broader regression passed **196 tests in 53.47 seconds**, covering A7, A6, setup history/runtime linkage, bar provenance, audit replay/storage/archive, collection identity, and both frozen paper-ledger/runtime guards:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_research_execution.py tests/test_ote_live_shadow_policy.py tests/test_ote_live_setup_history_runtime.py tests/test_ote_live_setup_event_history.py tests/test_ote_live_bar_provenance.py tests/test_ote_live_audit_replay.py tests/test_ote_live_audit_store.py tests/test_ote_live_storage.py tests/test_ote_live_archive.py tests/test_es_collection_baseline.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py tests/test_frvp_paper_signal_ledger.py tests/test_ict_paper_signal_ledger.py -q --tb=short
```

Final additions cover source-bar revision tainting, potentially filled pending orders retaining capacity, and collection/freshness rejection. The final focused A7/A6 regression passed **69 tests in 3.69 seconds**. The explicit CLI smoke check also completed successfully. These are synthetic software validations, not new FRVP/ICT performance experiments or a qualified collection period.


---

<a id="imported-a8"></a>

Source: `ES_A8_chart_presentation_20260914.md`

# ES A8: setup-first chart and separate research view

A8 was completed on September 14, 2026. This is a dashboard presentation change for ES FRVP/ICT views, including their setup-family tabs. It is not a market experiment or the start of qualified collection.

## Default and optional views

The default chart shows price, the selected strategy's detector-selected setups, and matching persisted qualified shadow decisions. Model probability panels and off-setup threshold diagnostics are available by selecting **Research: thresholds and probability streams**. A high probability, a legacy `emit`, or an eligible policy candidate does not become a qualified entry through presentation logic.

Controls provide:

- All six FRVP setup types and all nine current ICT types, including the research classic breaker, as selectable layers. An empty type selection hides setup marks and their qualified decisions.
- Rejected and research setups, all revisions/invalidations, and strategy levels/zones as optional layers.
- A single collection selection. The default selects the collection with the latest recorded observation in the visible source-bar window and displays its identifier; explicit older collections remain selectable. Collections, assets and timeframes are not silently combined in research probability panels.

The dashboard reads the immutable A5 setup tables for the entire visible source-bar window, without the former latest-40 setup limit or a shared model-decision row cap. Display filtering does not change retention or delete events. Legacy runtime-state setup marks are not substituted for missing immutable history. Source-bar coverage still determines the chart window; this does not add a full-history browsing interface.

## Mark and timing contract

| Mark | Meaning |
| --- | --- |
| Green/red triangle | Long/short rule setup |
| Open triangle | Incomplete/provisional, rejected, or research setup; the legend and hover distinguish these states |
| Diamond | Revised setup |
| X | Invalidated setup |
| Open yellow square | Incomplete/provisional price bar |
| Blue star | Matching persisted qualified shadow decision, subject to the checks below |
| Open purple circle, Research only | Setup-matched or off-setup/unverified threshold diagnostic, explicitly labeled |

Setup hover text reports **rule confidence**, source-bar time/version, setup observation time, first bar/setup observation, actual delay relative to bar close, feed/observation kind, state/revision, geometry, collection and associated prediction IDs. A negative delay is described as time **before** bar close rather than being clamped to zero. Unknown provenance stays unknown. Model hover text separately reports **model probability**, raw model score, recorded threshold, actual prediction-recording time, decision delay, setup/prediction/decision IDs, rejection reasons and pending prerequisites.

Qualified markers require the recognized A6 policy and matching contracts, explicit persisted qualification and policy eligibility, eligible complete source provenance, and an exact immutable setup match known at prediction time. They do not rerun policy or infer qualification from a threshold. Later setup revisions/invalidations preserve an originally qualified decision's original event association and disclose the latest setup state in its hover text.

A qualified decision is drawn at its actual prediction-recording time using an explicitly labeled reference price unless it contains an executable entry time and price. An entry timestamp preceding prediction recording is not drawn as an executable entry. The chart extends its time range to keep the final bar's actual-time decisions visible. Research threshold marks remain at their source bars and are labeled diagnostics, with actual recording times and delay in the tooltip.

ES research probability panels use persisted thresholds for historical observations rather than retroactively applying today's configured threshold. Non-ES chart behavior and controlled FRVP bundle validation/markout scoping remain intact. Strategy-wide rule observations are identified separately from controlled-bundle model decisions.

## Qualification and operational boundary

A6/A7 still keep scored qualification disabled while B1/B2/B3 and baseline prerequisites are pending. An empty qualified layer is expected. This change does not promote models, tune thresholds, change policies, activate paper trials, submit orders, or alter the immutable event or research execution ledgers. No live collector was started or restarted.

The primary implementation is `ote_live/dashboard/es_chart.py`, integrated through `app.py`. `queries.py` retains bar completion/provenance and supports collection, asset and timeframe scoping for model history; `charts.py` uses explicit confidence/probability labels and recording-time hover details.

## Validation

The combined regression passed **236 tests in 18.74 seconds**:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_es_chart.py tests/test_ote_live_dashboard_app.py tests/test_ote_live_frvp_dashboard_contract.py tests/test_ote_live_ict_dashboard.py tests/test_ote_live_setup_family_tabs.py tests/test_ote_live_shadow_policy.py tests/test_ote_live_setup_event_history.py tests/test_ote_live_setup_history_runtime.py tests/test_ote_live_bar_provenance.py tests/test_ote_live_storage.py tests/test_ote_live_research_execution.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py -q --tb=short
```

The A8 tests exercise actual SQLite persistence, Plotly figures and the Dash refresh callback: 48 setups per strategy across database reopen, optional type/rejected/revision layers, distinct confidence fields, first-observation and prediction delays, off-setup and pending candidate exclusion, exact qualified linkage, actual-time visibility, later invalidation, mixed collections/assets/timeframes, empty model/manifest scopes, missing history, and read-only chart operations. Qualified examples in these tests are explicitly synthetic future records; production qualification remains disabled.

A headless Chrome smoke test served an isolated synthetic ES database through the real Dash HTTP endpoints. It verified the default chart with model panels hidden and clicked the Research control to reveal threshold diagnostics and probability panels. Visual inspection caught and corrected low-contrast control labels. The temporary fixture server was shut down after each check; it did not use the live database or broker feed. Temporary screenshots: `tmp/a8-browser-b57ef14f/default.png` (default) and `tmp/a8-browser-3f5a308c/default.png` (Research). Python compilation and whitespace checks passed.

After the control-label styling fix, the final dashboard regression rerun passed **54 tests in 3.87 seconds**.


---

<a id="imported-a9"></a>

Source: `ES_A9_A10_research_priorities_20260915.md`

# ES A9/A10: FRVP research priorities and S4 candidacy pause

Completed September 15, 2026. This implements app research prioritization; it does not begin a qualified evaluation period or change model promotion status.

## Roster

The versioned roster is `es-frvp-research-priorities-v1`, defined in `ote_live/models/frvp_research.py`.

| Model | Presentation and collection intent |
| --- | --- |
| Short-continuation TCN | Prominent shadow comparison; retain short-side economics/drawdown restrictions |
| Long-continuation XGB | Prominent shadow comparison, subject to input/policy/baseline qualification |
| Long S3 | Prominent exploratory specialist in FRVP Setup Models |
| Long S5 | Continue matching long observations; short-S5 setup results do not establish long-S5 model quality |
| Long-reversal family; long/short meta | Background research, retaining diagnostic logging |
| Long S1/S2/S6 | Background research; missing matching S1/S2 examples are insufficient evidence, not failure |
| Long S4 | Qualified candidacy paused; artifacts and diagnostic scores retained |
| Short-reversal family | Remains retired/deprecated; historical diagnostics and artifacts retained |

## Dashboard behavior

The default FRVP and FRVP Setup Models tabs show a research-priority section above the price chart. It contains no raw probabilities or implied trade signals. The default model-decision layer and recent model activity include only the foreground roster. The A8 setup layers remain available for every setup type.

Selecting **Research: thresholds and probability streams** exposes background models, their decisions and probability charts. Cards place priority models first and explain each model's research role, S4's pause, and the limits of S1/S2 evidence. Default shadow presentation no longer assigns an `ACTIVE WEIGHT` badge to the old family-model selection. The exact frozen controlled-bundle badge/content validation remains intact and conveys no new activation authority.

Long S3/S5 coverage reports selected same-side setup observations in the chosen collection/window, using the latest immutable setup states and excluding invalidations. These counts are explicitly not resolved outcomes or evidence of model quality. Three short S5 setups and one long S5 setup produce a long-S5 count of one. Historical selection, completion, provenance, matching and qualification checks continue to operate separately.

## Diagnostic and qualification contract

Every evaluated roster model receives persisted `research_priority` metadata in its A6 shadow evaluation, including the roster version, tier, note, research-only status and explicit absence of promotion authorization. Historical decision hover text displays the recorded priority. Collection identity includes the roster version and hashes its implementation, separating new runtime collections from earlier observations.

S4's current manifest is provisional and its policy lineage names S2 through `direction_fallback`. S4 now also receives the explicit `frvp_s4_model_specific_policy_pending` candidacy rejection. This block survives a generic `policy_status=complete` edit: resumption requires a reviewed model-specific policy and an explicit roster version change. Its threshold crossing, exact setup match, raw probability and isolated policy-check results remain diagnostic evidence. It cannot become an eligible shadow candidate or advance hypothetical candidate state while paused. Short reversal has an additional defensive retired-candidate rejection; existing runtime exclusion of deprecated models remains unchanged.

Background placement is a presentation/research priority, not a newly invented policy exclusion. Background scores and matching/rejected observations continue through existing logging. All branches retain A3/A4 input/freshness requirements and the existing B1/B2/B3/baseline gates; production `qualified_shadow_entry` remains false. No thresholds, calibration, artifacts, manifest files, execution policies, frozen paper ledgers or readiness restrictions were changed. No live collector was started/restarted and no broker order or paper-trial activation was introduced.

The older short-side economics findings remain binding, including the short-continuation TCN's approximately -0.38 Sharpe and 109.2% maximum reference-account drawdown under ES-aware costs. Research prominence does not override these results. B5's full prospective eligibility/promotion contract remains a separate unfinished prerequisite.

## Validation

The combined regression passed **183 tests in 49.67 seconds**:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_shadow_policy.py tests/test_ote_live_es_chart.py tests/test_ote_live_dashboard_app.py tests/test_ote_live_frvp_dashboard_contract.py tests/test_ote_live_setup_family_tabs.py tests/test_es_collection_baseline.py tests/test_ote_live_audit_replay.py tests/test_ote_live_research_execution.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py -q --tb=short
```

New tests cover all twelve roster entries through actual decision evaluation and SQLite persistence; S4's block even with an otherwise passing complete policy; retirement; unchanged thresholds/policy payloads; no qualified entries; long-S5 side matching; both dashboard tabs; background visibility; immutable record retention; and same-side setup coverage. Synthetic qualified records test presentation only.

Headless Chrome exercised the actual Dash HTTP callback in an isolated synthetic ES database. The default priority section rendered visibly with research panels hidden; selecting Research exposed probability panels and threshold diagnostics. Both browser processes and fixture servers exited. Temporary screenshots: `tmp/a8-browser-088868be/default.png` (default, visually inspected) and `tmp/a8-browser-a301b5f3/default.png` (Research). Python compilation and tracked-file whitespace checks passed.


---

<a id="imported-a11"></a>

Source: `ES_A11_ICT_research_roster_20260915.md`

# ES A11: ICT research-only roster

Completed September 15, 2026. A11 implements app restrictions and research priorities. B2's artifact-lineage review remains open; this change does not start a qualified evaluation period.

## Roster and release boundary

`ote_live/models/ict_research.py` defines `es-ict-research-priorities-v1`, covering all six pooled ICT models and all three packaged short specialists.

| Models | Current role | Role after required contracts/artifacts pass |
| --- | --- | --- |
| Long-continuation XGB | Research-only; focused comparison pending review | Focused shadow comparison |
| Short premium/discount continuation specialist | Research-only; focused comparison pending review | Focused shadow comparison |
| Long/short reversal and short continuation families | Background research | Background comparison |
| Long/short meta models | Background research | Background comparison |
| Short IFVG and sweep specialists | Background research | Background comparison |

All nine models receive `ict_feature_contract_and_B2_artifact_review_pending`. Unknown/new `ict_` model IDs also fail closed. A passing live input contract, a complete policy, a registry activation setting, or a new model name cannot release this block. The A2 causal producer fix is complete; that alone does not establish the lineage of prepared datasets, setup generation, labels, calibration, or deployed weights. A3 also identified absent live confluence-helper producers for the meta models.

B2 must supply branch-specific evidence and an explicit versioned roster change tied to reviewed artifacts before lifting the block. A11 deliberately provides no generic approval flag. Unaffected branches may be released individually through B2; they need not wait for all background models. Required input checks, bar freshness/completion, full policy, B1/B3 and frozen-baseline prerequisites continue to apply after any such release. A11 does not perform B2 regeneration, retraining, recalibration or baseline approval.

## Runtime and persistence

The ES signal processor adds the roster rejection to observation diagnostics and forces shadow mode even when a binding requests active evaluation and inputs pass. Raw scores and setup matching continue through the existing diagnostic path. The rejection persists in the feature snapshot and A6 evaluation, together with roster version, current tier, post-review priority, unverified artifact status and absence of promotion authorization.

Threshold crossings, selected setup matches and isolated full-policy results remain separate evidence. A quarantined observation cannot become a policy-eligible candidate or advance candidate cooldown/quantile history. It never receives a qualified entry, executable timestamp or executable price. Collection identity includes the ICT roster version and hashes its implementation, separating subsequent collections from earlier runtime contracts. Existing immutable records are retained.

## Dashboard

Both ES ICT tabs display the research-only restriction and the pending focused comparison priorities above the chart. The default chart retains setup observations; all quarantined model decisions remain outside its qualified-decision layer. The Research view retains all probability streams, threshold diagnostics and background comparisons. Focused candidates appear first in research cards, with their pending-review status; ICT cards show `RESEARCH-ONLY` rather than an active-weight badge. Frozen manifest/registry content remains intact.

## Existing restrictions

- Long continuation's best recorded q40 concentration control still has a 13.59% largest-trade share against the 10% gate. Research priority grants no promotion and does not reopen simple pocket-pruning.
- Short meta/reversal retain single-trade concentration and profitable-quarter breadth failures. Their 284 shared events contributed 2597.4 ticks to each branch; shock-day exposure, including March 17, 2020, remains unresolved.
- Short continuation's weak q50/q60 findings remain. Background placement retains its comparison evidence.
- Concentration, promotion, economics, readiness and paper-trial restrictions remain binding for the whole roster. B5 still owns the complete prospective eligibility/promotion contract.

No artifacts, calibration, thresholds, execution-policy settings, frozen paper ledgers or activation permissions were changed. No collector was started/restarted, and no paper trial or broker order was initiated.

## Validation

The focused suite passed **109 tests in 15.64 seconds**:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_shadow_policy.py tests/test_ote_live_es_chart.py tests/test_es_collection_baseline.py tests/test_es_causal_feature_contract.py -q --tb=short
```

New coverage exercises all nine ICT models plus an unknown replacement through decision evaluation and SQLite persistence, with passing inputs and complete policies. It verifies retained raw/setup/policy diagnostics, unchanged manifests/policies, no candidate state advancement, no qualified entry, exact roster coverage, intended priorities, collection rotation, both Dash tab callbacks, background visibility, research card ordering and read-only chart queries. A processor integration case supplies a fresh complete bar, a passing stamped input contract and an active binding; the persisted result remains diagnostic/shadow solely because of the A11 review block.

The broader regression passed **157 tests in 57.45 seconds**, for **266 passing tests** across both runs:

```powershell
ote_venv/Scripts/python.exe -m pytest tests/test_ote_live_dashboard_app.py tests/test_ote_live_ict_dashboard.py tests/test_ote_live_frvp_dashboard_contract.py tests/test_ote_live_setup_family_tabs.py tests/test_ote_live_audit_replay.py tests/test_ote_live_research_execution.py tests/test_frvp_paper_signal_runtime_guard.py tests/test_ict_paper_signal_runtime_guard.py tests/test_ote_live_signal_runtime_integration.py tests/test_ote_live_parallel_signal_processor.py -q --tb=short
```

This includes frozen bundle/paper guards, replay, research execution, runtime notifications using test transports, parallel processing and existing FRVP presentation. Dash callbacks and layout endpoints were tested; no interactive browser or running collector was used. Tracked-file whitespace checks passed.

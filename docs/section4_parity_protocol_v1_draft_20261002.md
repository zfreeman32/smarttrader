# Section 4 live-parity protocol v1: DRAFT, UNLOCKED (2026-10-02)

Status: `draft_unlocked`. Nothing here is binding until you approve the decisions at the end. After approval, prompt 2 publishes the approved content as `configs/section4_parity_protocol_v1.json` with a `.lock.json`. Tolerances are then fixed, and no comparison runs before the lock. A later change needs a new protocol version that you approve, with the reason recorded.

Inputs (immutable, verified by hash at every run):

- B4 v2 freeze `configs/b4_broad_full_session_shadow_freeze_v2.json` and its lock;
- route bindings v2 and lock;
- selection v1;
- EURUSD calendar `eurusd_5m_feed_clock_week_v1` (provisional; stated in every EURUSD report).

Inventory: `docs/section4_inventory_20261002.md`. The full policy is `not_defined` for all 16 models, so the decision compared is the threshold decision `calibrated_probability >= threshold` with the B4 v2 per-binding float.

## 1. Preconditions per run (fail closed)

1. Hash every B4 v2 artifact file and the prepared manifest (`verify_bytes` pattern). Stage a manifest per model outside `ote_live/runtime_manifests`. Any historical app artifact in its place fails that branch.
2. The app path runs in an isolated SQLite parity database built from the schema and migrations, holding copied canonical bars and source history. The live database is opened read-only to copy rows. Nothing touches the running app, installed manifests or the registry.
3. Scored registration stays disabled. Shadow mode is forced, and no broker, paper, alert or email output is produced.
4. The run records the code sha256 of the harness and of every runtime and producer module, as evidence identity.

## 2. Layers

- **(P) Producer parity.** Identical source bars go through the actual app path (`LiveSignalProcessor.process_new_bars_from_store` → `IncrementalFeatureEngine` → `RuntimeModelRunner` → decision engine) and through the offline reference:
  - historical windows: frozen `source_history.parquet` + frozen runner;
  - recorded live windows: repaired builder on the full contiguous series + frozen runner, after the anchor check (inventory d).
- **(S) Source parity.** App-recorded bars are compared with the historical training source on overlapping timestamps (policy D2). No overlap exists today in either stream.

## 3. Stages compared on every consumed row

Every app build at endpoint *t* is compared on all rows that model consumes: XGB uses `max(lag)+1` rows and TCN uses `window_size` rows, not only the endpoint.

| # | Stage | Fields |
|---|---|---|
| S1 | Source-bar identity / provenance | timestamp, source_row_idx (consecutive), instrument, contract_symbol, bar_version, is_complete, feed_type, observation_kind, first/last observed |
| S2 | Setup events consumed by routes (D5) | type, side, family, source timestamp, observed-at, detector identity/version, rule confidence, geometry, revision kind |
| S3 | Raw pre-imputation selected features | per model and feature, plus missing-producer / fallback / warmup state |
| S4 | Imputed values | the frozen-fill output plus the imputation mask (which cells were filled) |
| S5 | Scaled model input | after scaler and clip; the XGB lag/delta view or the TCN sequence tensor |
| S6 | Raw score and calibrated score | |
| S7 | Threshold decision | `calibrated >= threshold` |

Restart runs (§7) also compare app-restart with app-continuous on S1–S7.

## 4. Tolerance table (proposed; D1)

A cell passes if `|app − ref| ≤ atol + rtol·|ref|`. NaN matches NaN only, and ±inf matches the same sign only. A different missingness pattern is a mismatch.

| Class | Applies to | atol | rtol | Basis |
|---|---|---:|---:|---|
| E exact | S1 and S7 fields; S2 categorical and timestamp fields; states (warmup, gap, roll, missing producer, fallback, imputation mask); C0 features (integer-valued in the frozen reference) | 0 | 0 | identity |
| C1 continuous | ratios, ATR-normalized, z-scores, oscillators, ranks | 1e-6 | 1e-6 | B2 feature-matrix replay (`audit_b2_candidate_history.py:69`) |
| C2 price level | untransformed price-unit levels; S2 geometry | 1e-6 | 2e-7 | ≈3 float32 ULP: 0.0015 ES points (tick 0.25), 2.3e-7 EURUSD (pip 1e-4) |
| C3 volume scale | raw volume-scale features | 1e-6 | 1e-6 | as C1 |
| S4 imputed | the raw cell's class, after fill | class | class | fill constants are frozen |
| S5 scaled | float32 scaled input, `|x| ≤ 8` | 1e-5 | 1e-6 | float32 tensor after the scaler |
| S6 scores | raw and calibrated | 1e-6 | 1e-5 | B2 / section 3 score replay (`rtol=1e-5, atol=1e-6`); B4 v2 max errors 8e-8 to 5.8e-7 |
| S2 rule confidence | detector confidence | 1e-6 | 1e-6 | as C1 |
| Restart vs continuous (app vs app) | S1–S7 | 0 | 0 | same bars and same code, so bitwise identity is expected |

Per-feature classes are pre-assigned in `docs/audits/section4_feature_inventory_20261002.json` from frozen reference data only (rule recorded there). They are locked with the protocol.

**Near-threshold rule.** Let `band = 1e-6 + 1e-5·threshold`. Any endpoint where `|cal_ref − thr| ≤ band` or `|cal_app − thr| ≤ band` goes in a per-model `near_threshold` list with both scores and both decisions. Such endpoints are never counted as pass or fail. A model with a non-empty list cannot pass until you review the list. A decision flip outside the band is a S7 failure.

## 5. Windows and scenarios

Windows are chosen by calendar or event presence only, never by outcomes or scores. They are written to `windows.json` (hashed) before any comparison.

A scenario instance is a segment from 48 bars before the event to 96 bars after it. Each segment has its own prefix of up to 8,000 earlier bars; for recorded ES windows the prefix is everything recorded. Every endpoint in the segment is scored for every warm model. Segment lengths are fixed after the per-build timing measurement (§8) and before the lock.

Source codes: R = real app-recorded bars (live DB); H = real historical training-source bars (P layer only); F = deterministic fault injection on R or H bars.

**ES coverage matrix (IDs 2, 3, 5, 10, 12–16, 18, 19):**

| Scenario | Source | Instance (rule → candidate) |
|---|---|---|
| Overnight→RTH open, RTH close→overnight | R + H | 09:30 / 16:00 NY in R week 2026-09-14…18 (completion known); same in each H window |
| Daily maintenance break | R + H | 17:00–18:00 NY in R week 2026-09-14…18 |
| Spring DST | H | week containing 2026-03-08 |
| Fall DST | H | week containing 2025-11-02 |
| Holiday / planned closure | R + H | R: Labor Day early close 2026-09-07 (completion unknown, labeled so); H: Thanksgiving 2025-11-27/28 |
| Quarterly roll | R + H | R: ESU6→ESZ6 at 2026-09-13 22:00 (weekend; pre-roll completion unknown); H: first `contract_symbol` change in the source in June 2026 (and March 2026 if in another week) |
| Weekend reopen | R + H | R: 2026-09-20 22:00 |
| Missing bars (non-closure gap) | R + F | R: 2026-09-18 11:40–13:00 (16 bars); F: delete 1 / 3 / 12 consecutive RTH bars |
| Reconnect / backfill | R + F | R: the same 09-18 gap with its later backfill; F: withhold bars, then upsert them behind the cursor as `backfill` |
| Revision | R + F | R: latest-bar revisions in `source_bar_history`; F: revise the latest bar, and an earlier bar behind the cursor |
| Restart | F (real restart path) | 4 points: mid-RTH, just after the maintenance break, just after the roll boundary, inside a full lag window |
| Cold start / warmup | F | history shorter than `minimum_runtime_history_bars` |
| Producer failure | F | force one feature-set exception: must surface missing-producer state, never a silent fill |
| HTF 30m/1h publication | R + H | inherent in every segment (A3 :30 / :00 publication on the :25 / :55 row) |
| Group composition | F | union engine vs FRVP group {2,3,5,10} and ICT group {12–19}; negative control with reserve ID 1 to show the `ffill+0` switch (control only, not evaluated) |

**EURUSD coverage matrix (IDs 21, 22, 23, 25, 26; provisional calendar v1):**

| Scenario | Source | Instance |
|---|---|---|
| Weekend close / open | R + H | R: Friday 2026-09-18 → Sunday 09-20 (calendar v1 boundary difference recorded, inventory f); H: any week |
| Asia→London, London→NY, NY close | R + H | 07:00/08:00 London, 08:00/09:30 NY, 17:00 NY in R week 2026-09-14…18 |
| US/EU DST mismatch | H | 2026-03-08…03-24 (US shifted, EU not); 2025-10-26…11-01 (EU shifted, US not) |
| Completed 30m/1h ATR publication | R + H | every :00/:30 boundary in each segment. A value must change only on the completed-bucket row, never mid-bucket. The OTE event labeler uses the same publication (`data/labeling/causal_ote.py`). |
| Holiday closure (calendar v1) | H | 2025-12-25, 2026-01-01 |
| Missing bars | R + F | R: outage edge 2026-05-14 22:10; F: delete 1 / 3 / 12 bars in London |
| Reconnect / backfill | R + F | R: `repair` rows; F: as ES |
| Revision | F (+ R) | as ES; R revisions exist since 2026-09-09 |
| Restart | F | 4 points: London, NY, just after the weekend reopen, inside a TCN window |
| Cold start / warmup | F | as ES |
| Strategy timeout / producer failure | F | force a strategy timeout: must surface missing-producer state (inventory b) |

**Completion.** Every recorded EURUSD bar has unknown completion (D6). R-segments are compared with completion `unknown` as an exact identity value. That never counts as known-completion evidence.

## 6. Warmup handling

- A row is comparable only when the app's warmup status is ready and the reference row exists. On EURUSD the reference must also be past `source_row_offset` (250).
- Rows before that are counted, not compared. Their warmup state is an exact S3 state: app ready while the reference is not, or the reverse, is a mismatch.
- Every consumed history row of a comparable endpoint must itself be comparable. Otherwise the endpoint is `incomplete_history`, not a pass.

## 7. Restart test

- The real restart path: a fresh `LiveSignalProcessor` / `MultiGroupLiveSignalProcessor` on the same parity database at bar *k*, `warm_from_store`, then `process_new_bars_from_store` for the rest of the segment.
- Rows after the restart must equal the continuous run bitwise (§4) and the reference within tolerance.
- Chunk resumes (§8) count as restarts. They are allowed only after the restart test for that stream has passed. Until then, each segment runs in one process.

## 8. Chunking, checkpoints, storage and retention

- **Order of work.** First measure the per-build time: one stream-union build on 3 endpoints per stream. Then fix the segment lengths and the budget, record them, and lock.
- **Work unit.** (stream, segment) uses one stream-union processor, so producers are computed once and selected per model. Composition invariance is proven by the F scenario.
- **Chunks.** 48 endpoints each. Each chunk writes `chunk_<n>.parquet` and `chunk_<n>.done` with an sha256.
- **Runs.** Jobs over ~30 min run in the background with a log and a `SEGMENT <id> COMPLETE` marker. Resume skips finished chunks.
- **Layout.** `artifacts/s4parity_<rundate>_v1/<stream>/<segment>/{app,ref,diff}/` plus `summary.json`.
  - `app/` holds the raw union features for all consumed rows per build, together with S1/S2/state, S4–S7 per model, and per-stage sha256 per window.
  - `diff/` holds every out-of-tolerance cell (full diffs on disk).
  - Summaries hold counts per stage and feature, the first offending timestamp, max abs/rel error and the classification.
- **Reports.** `docs/audits/section4_parity_<model_id>_<rundate>.json` (immutable) and `docs/section4_parity_<rundate>.md`, superseding the 09-29 B5 evidence, which stays unchanged.
- **Retention.** Proposed in D4.

## 9. Mismatch taxonomy

| Class | Meaning | Branch effect |
|---|---|---|
| `app_producer_bug` | The app deviates from the causal contract. This includes window truncation, composition-dependent fill, dropped provenance and stale history after a backfill or revision. | Fail until fixed by a minimal versioned runtime change (with on/off tests and a journal entry) and re-run under a new run ID |
| `training_side_noncausal` | The frozen reference uses information unavailable at decision time | Branch diagnostic; record a repair recommendation; stop for your decision (no retraining in section 4) |
| `source_difference` | The inputs differ (S layer, or vendor/feed/roll convention) | Not a P failure. Reported with per-feature and decision impact; decision flips listed for review (D2) |
| `declared_contract_behavior` | The behavior follows a contract clause that existed before the run (cite file and version), e.g. an A3 stale-HTF NaN or A4 gap rejection | Pass only with the citation |
| `unexplained` | None of the above | Fail |

Truth is the causal contract. On any mismatch, first establish which side is causally correct. Never edit reference data, frozen preprocessing or model bytes, and never make an app producer reproduce a non-causal offline value.

## 10. Per-model verdict

`passed` requires all of the following:

1. B4 v2 bytes are verified and loaded by the staged manifest.
2. Every required scenario in the stream matrix was exercised with at least one comparable endpoint of this model.
3. Every compared cell in S1–S7 on every consumed row is within tolerance, or is classified `declared_contract_behavior` with a citation.
4. There is no `app_producer_bug`, `training_side_noncausal` or `unexplained` mismatch.
5. There is no S7 flip outside the near-threshold band, and the near-threshold list is reviewed.
6. Provenance and completion are present and exact. There is no unexplained fallback and no missing producer.
7. The restart test is equal.
8. The source-parity outcome follows D2.

Other verdicts:

- `incomplete`: a required scenario was unexercised, or evidence is missing (for example unknown completion where known completion is required). It is not a pass.
- `failed`: any fail class. The branch is diagnostic and unscored, and does not block the other branches.
- `diagnostic_training_side`: stop for your decision.

A pass is not scored eligibility. B6, A12, A13 and B7 still apply.

## 11. Immutable report schema (`section4-parity-report-v1`)

Fields:

- `model`: queue_id, model_id, `reviewed_artifact_set_sha256`, artifact file hashes, staged manifest sha256;
- `protocol`: version, sha256, lock sha256;
- `run`: id, date, code hashes, parity DB sha256;
- `windows[]`: segment ID, source R/H/F, rule, start/end, endpoints, comparable endpoints;
- `coverage`: scenario → exercised / unexercised, with reason;
- `stages`: per stage, cells compared, matched and mismatched, max abs/rel error, first offending timestamp;
- `features`: per feature, mismatch counts and class;
- `mismatches[]`: class, stage, feature, first timestamp, count, evidence path, and causal analysis note;
- `near_threshold[]`, `decision_flips[]`;
- `restart`: equal / not equal per restart point;
- `source_parity`: per D2;
- `verdict` and `reasons[]`;
- `eurusd_calendar`: identity and provisional status (EURUSD only);
- `report_sha256`, computed over the canonical JSON without this field and also written to a `.sha256` sidecar.

The writer refuses to overwrite.

## 12. Retrain guard (R6)

No retraining or recalibration happens in section 4. A `training_side_noncausal` finding makes the branch diagnostic, records the repair recommendation, and stops for your decision. IDs 2, 6, 7, 11, 17 and 1 are not retrained unless a completed parity audit finds a real selected-feature, history, preprocessing or producer mismatch.

## Decisions needing approval

| # | Decision | Recommended default |
|---|---|---|
| D1 | Tolerance table (§4) | Approve as written: exact for identity, states and C0; C1/C3 1e-6/1e-6 (B2 feature replay); C2 1e-6 abs / 2e-7 rel; scaled 1e-5/1e-6; scores 1e-6 abs / 1e-5 rel (B2 score replay); app restart vs continuous bitwise; near-threshold band listed for review, never auto-scored. |
| D2 | Source parity, and the lack of overlapping data | Policy as you proposed: identity, timestamps and completion must match exactly; price/volume differences are reported with per-feature and threshold-decision impact; flips are listed for your review. **Data:** no overlap exists (ES training ends 2026-06-19, live starts 07-29; EURUSD ends 03-25, live starts 04-08). To create overlap, run the app's own backfill connectors into an isolated parity DB, for windows inside the training source: ES IBKR historical 2026-06-01…06-19 (includes the June roll); EURUSD FMP historical 2026-03-02…03-24 (includes the DST mismatch). This uses a separate client ID, never the running app's session, and needs your go-ahead because it is an outward network call. Fallback if declined: S is `incomplete` for all 16. |
| D3 | Scenarios relying on fault injection | Restart, cold start, producer failure / strategy timeout and group composition use F only. Missing bars, backfill and revision use R where it exists plus F. DST (both streams) and EURUSD holidays use H only. Every F scenario injects on R bars where R exists. |
| D4 | Snapshot storage and retention | Keep every app/ref chunk, diff and summary for section 4 windows under `artifacts/s4parity_<date>_v1` until B7 closes (estimated <2 GB compressed). Reports in `docs/audits` are kept permanently and never overwritten. The isolated parity DBs are kept with their sha256. No live-DB rows are copied beyond the selected segments and their prefixes. |
| D5 | Include the setup-event comparison (S2) | Yes, for both streams. ES: the app's FRVP/ICT detectors on replayed bars vs the offline detector events used by section 3. **EURUSD: the app has no live OTE event producer**, so route events cannot be compared. Recommended: S2 is `incomplete` for 21–26 until a versioned live OTE event producer exists. It would reuse the offline labeler's completed-bar adapter (`data/labeling/causal_ote.py`) as the single producer and would be added in a later section 4 prompt as a minimal runtime change. Alternative: EURUSD S2 is out of scope here and becomes a B6 blocker. |
| D6 | EURUSD completion | FMP bars carry no completion or version. Recommended: an R1 versioned rule for FMP historical-chart bars (complete when bar end + grace ≤ observed_at), applied to new recordings and parity replays only. Archived rows stay `unknown`, and EURUSD R-segments count as completion-unknown evidence. |
| D7 | Engine composition | Use one stream-union engine per segment (producers once), plus the composition-invariance F scenario against the planned FRVP/ICT groups. B6 must not mix a non-`frvp_`/`ict_` XGB model into an ES group, or the composition-dependent `ffill+0` must be removed by a versioned fix. |
| D8 | Provenance-loss fix (inventory finding 1) | Approve as part of R1: select the provenance columns in `_fetch_signal_bars_after` and `_fetch_recent_signal_bars`, and carry them in `_bar_from_row`. OHLCV and feature values are unchanged (proved with on/off tests); only the bar-eligibility reasons change. Logged in `live_app_change_journal.md`. |
| D9 | ID 23 staged-manifest base (no installed manifest) | Base it on `ote_live/runtime_manifests/long_ote_union_tcn_candidate_20260523`, with every artifact, feature, context, lag and window field overridden from the B4 v2 training summary and hash-verified. |

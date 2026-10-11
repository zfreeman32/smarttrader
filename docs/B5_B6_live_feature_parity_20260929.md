# B5/B6 live feature-producer parity evidence — 2026-09-29

Evidence version: `b5-b6-live-feature-parity-v1` (final machine-readable run: `docs/audits/b5_b6_live_feature_parity_20260929_v2.json`).

## Result

All six proposed qualified branches remain `diagnostic_only`.

- six candidates reviewed; zero full-parity passes;
- zero promotion passes and no new scored collection is authorized;
- the frozen B2 imputation/scaling, causal-history, score, and threshold replay evidence is reused by candidate and remains passing where recorded;
- the live gate fails because the recorded app archive does not persist raw pre-imputation selected features, completion/availability is unknown for every recorded candidate snapshot, and the historical source windows contain unresolved non-closure gap observations;
- the actual `IncrementalFeatureEngine` rebuild was bounded as `not_run_resource_bounded`; it is not represented as a pass.

The recorded source archive is `research/es_live_audit_20260911/es_audit_snapshot.sqlite3`, cutoff `2026-09-12T02:02:03.011561+00:00`. Its snapshots were produced through the live signal path (`LiveSignalProcessor.prepare_ingested_bar_snapshot` → `IncrementalFeatureEngine.build_feature_frame` → `RuntimeModelRunner.predict_latest`), but the persisted feature values are post-imputation model inputs. They cannot prove raw producer parity or explain a fallback without a raw producer snapshot.

| ID | Candidate | Recorded snapshots | Completion unknown | Availability unknown | Gap observations | B2 score/threshold replay | Final |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | FRVP short continuation TCN | 2,159 | 2,159 | 2,159 | 182 | passed | diagnostic-only |
| 2 | FRVP long continuation XGB | 2,160 | 2,160 | 2,160 | 314 | passed | diagnostic-only |
| 6 | FRVP long continuation setup 3 XGB | 1,409 | 1,409 | 1,409 | 28 | passed | diagnostic-only |
| 7 | FRVP long continuation setup 5 XGB | 1,409 | 1,409 | 1,409 | 108 | passed | diagnostic-only |
| 11 | ICT long continuation XGB | 2,076 | 2,076 | 2,076 | 169 | passed | diagnostic-only |
| 17 | ICT short premium/discount continuation XGB | 1,409 | 1,409 | 1,409 | 92 | passed | diagnostic-only |

Roll/identity, warmup, missing-producer, and fallback fields are preserved per candidate in the JSON. No unknown or missing producer is converted into a qualification pass.

The canonical-bar windows did not show an observed contract transition in this retained cutoff, but endpoint source identity was unknown for every candidate snapshot; therefore “no roll observed” is not a roll-parity certification. Maintenance/weekend closure gaps are counted separately from unresolved non-closure gaps in the JSON.

## Clean evaluation and A13 rehearsal

B6 is evidence-only at this point. The prospective clean boundary remains the first healthy post-B7 collection timestamp under a new immutable candidate/runtime identity. Future C1 outcomes are not required before starting an otherwise eligible research collection. A13 rehearsal is therefore ready only after a branch clears the live parity and other prerequisite gates; this audit does not clear one.

Machine-readable evidence: [b5_b6_live_feature_parity_20260929_v2.json](C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\docs\audits\b5_b6_live_feature_parity_20260929_v2.json)

Replay implementation: [audit_b5_b6_live_feature_parity.py](C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\scripts\audit_b5_b6_live_feature_parity.py)

# B2 impact evidence

Measurements began on September 18, 2026. The review document is
[B2 minimum repairs](../../B2_model_repair.md#b2_minimum_repairs_20260919).

| Evidence | Contents |
|---|---|
| [impact.json](impact.json) | Full-history reference/ATR comparisons; all 113 model dispositions; available train/validation/test row measurements |
| [model_split_impact.csv](model_split_impact.csv) | Compact counts by model and split; counts overlap when models share data |
| [frvp_fixed_event_replay.csv](frvp_fixed_event_replay.csv) | Comparable saved events, reproduced original outcomes, timing-control outcomes and holding times |
| [OTE publication comparison](../b2_ote_publication_impact_20260918.json) | 17 selected HTF predictors, four continuation trend inputs and 43 model mappings |
| [Per-model repairs](../b2_model_minimum_repairs_20260918.csv) | All 113 versions; affected predictor rows and FRVP training/validation/test labels; proposed repair and release gate |
| [Validation](../b2_impact_validation_20260918.json) | Tests, protected-file checks and audit output identities |

**Method and interpretation**

- Full saved histories were read: 666,769 ICT rows, three full-history FRVP
  label batches of that length, 316,063 post-2022 FRVP rows, 543,441 OTE 5m
  labeled rows and 2,706,246 OTE 1m labeled rows. These are overlapping histories.
- Prepared `source_row_idx` was resolved against its declared input, then joined
  to timestamps. OTE's 250-row feature warmup was mapped by timestamp, not assumed
  to have the same indices as its label tape. Nine missing-input/split lineages
  remain unresolved; 104 bundles have available-split measurements.
- ICT retained the saved OHLCV and local ATR. Only reference construction was
  changed in memory. All finite saved selected reference values reproduced their
  prepared values. The corrected missing-value rule was fitted on each saved
  training split, with its existing row membership and feature selection fixed.
  Changed selected raw references/availability affect 93.14–99.95% of training
  rows; after refitting missing-value handling, 99.27–99.95% differ. These are
  per-model ranges, not pooled independent observations.
- ATR controls keep the historical Wilder ATR14 calculation and change its
  publication time. The primary comparison allows an open-stamped base bar to
  reach its close before prediction; the final bar of an hour may therefore use
  the hour that just completed. A row-open comparison is also recorded. Partial
  current and prior published buckets are excluded by expected bar count. This
  is a diagnostic, not a complete gap/roll/arrival-aware production producer.
- FRVP raw-price ATR reconstruction agrees with 574,414 of 629,038 comparable
  rows in each full-history batch and 271,839 of 303,147 post-2022 rows. The
  unmatched rows are **not cleared**; differences may involve the continuous
  price path/roll handling and require reconstruction during repair. Reported
  timing changes and event replay use only reproduced rows. OTE's saved ATR
  reproduces on every comparable row in both resolutions.
- FRVP replay preserves saved event selection, exclusion flags, entry, TP/SL
  multipliers, maximum holding period and historical target-first same-bar tie
  rule. Original outcomes and durations reproduce for all 25,327 / 25,327 /
  24,608 / 12,257 comparable events across the four batches. Non-excluded subsets
  produce the counts in the main report. It does not regenerate HTF swings,
  confluence, quality gates, concurrency, weights or all label outputs.
- Changed FRVP win flags were joined to each model's own prepared timestamps,
  side and setup/family. Original binary targets agree on all joined comparable
  rows. All 34 non-placebo FRVP bundles with parents available have at least one
  changed training target and changed development holding times. The ten shuffled
  placebo models remain controls; their source prepared labels are not proof of
  the labels actually supplied after experimental shuffling.
- OTE's original vendor timestamp convention is `as_provided`, not attested.
  Its 37-model selected-feature result is **conditional on bar-open semantics**.
  All 17 measured saved HTF predictor columns reproduce under the legacy
  calculation. The repaired control changes bucket membership, publication and
  partial-bucket rejection; almost-universal numerical changes must not be
  described as almost-universal future leakage or a quantified performance loss.
- Six available OTE bundles select none of those 17 direct HTF columns:
  `models/legacy_models/short_breakout_v2`, `models/live/long_breakout_xgb`,
  `models/live/short_breakout_tcn`, `models/ote_full_lstm_v2/long_ote`,
  `models/ote_multi_family_tcn_refresh_20260530/short_breakout`, and
  `models/ote_multi_family_xgb_v1/short_breakout`. Their label/derived-feature
  dependencies still require replay. The four measured continuation trend flags
  reproduce exactly and change on 1.56–1.65% of label-tape rows after publication
  delay; the stateful downstream continuation predictors were not rebuilt here.
- Helper counts compare saved selected prepared values to zero. Only the eight
  IDs in the prior frozen live audit have directly observed live-zero evidence.
  The other selected consumers inherit the producer/default risk by code lineage.
- Prepared-row measurements include inputs used for scaling and selection.
  They do not reconstruct individual stochastic/bootstrap draws, every lagged
  window, new predictions, or post-repair P&L. Exact full ICT/OTE label-change
  counts remain unmeasured. Existing hashes identify current files, not the exact
  bytes consumed by every historical training job.

**Reproduce into new audit paths**

```powershell
./ote_venv/Scripts/python.exe -B scripts/measure_b2_impact.py --output-dir docs/audits/NEW_IMPACT_RUN
./ote_venv/Scripts/python.exe -B scripts/measure_b2_ote_publication.py --output-file docs/audits/NEW_OTE_IMPACT.json
./ote_venv/Scripts/python.exe -B scripts/summarize_b2_minimum_repairs.py --impact-dir docs/audits/NEW_IMPACT_RUN --ote-impact docs/audits/NEW_OTE_IMPACT.json --output-file docs/audits/NEW_MODEL_REPAIRS.csv
```

These scripts write audit summaries only and refuse existing output destinations.
No feature dataset, label tape, prepared split, model, calibration or runtime
configuration is replaced. The main report is a proposed minimum repair, not a
completed B2 release or authorization to retrain.

# B3 comparison protocol, B4 freezes and route bindings (2026-09-29 to 2026-10-02)

Consolidated, chronological record of the prospective comparison declarations (B3 v1, v2), the B4 baseline freeze, the broad full-session shadow freezes (12-model v1, 16-model v2) and their route manifests (v1, v2). The frozen JSON and lock files live in `configs/`. The B5/B6 live-parity evidence stays in `B5_B6_live_feature_parity_20260929.md` because the section 4 protocol pins it by hash; its successor is the section 4 parity record in `section4_progress.md`.

Contents:
- [B3 prospective comparison declaration — 2026-09-29](#b3_prospective_comparison_20260929)
- [B4 baseline freeze — 2026-09-29](#b4_baseline_freeze_20260929)
- [Broad full-session shadow v1 / expanded B4 freeze — 2026-10-01](#b4_broad_full_session_shadow_freeze_v1_20261001)
- [Broad full-session shadow route bindings v1](#broad_full_session_shadow_route_bindings_v1_20261001)
- [Broad full-session shadow v2 / expanded B4 freeze — 2026-10-01](#b4_broad_full_session_shadow_freeze_v2_20261001)
- [Broad full-session shadow route bindings v2](#broad_full_session_shadow_route_bindings_v2_20261001)
- [B3 prospective comparison v2 (dual-stream) — 2026-10-02](#b3_prospective_comparison_v2_20261002)


<a id="b3_prospective_comparison_20260929"></a>

## B3 prospective comparison declaration — 2026-09-29

*Merged 2026-10-09 from `docs/B3_prospective_comparison_20260929.md`.*

The prospective C1 comparison is frozen in [`b3_b4_candidate_review_v1_20260929.json`](../configs/b3_b4_candidate_review_v1_20260929.json). It must be hashed and recorded before the first healthy post-B7 observation. This declaration contains no C1 outcomes.

### Primary comparison

For each eligible named setup opportunity, the same opportunity identity and execution tape are evaluated in three branches:

| Branch | Admission rule | Purpose |
|---|---|---|
| Setup only | The named setup match, with no score or policy filter | Setup-rule benchmark |
| Model threshold | The same opportunity plus the candidate's frozen B2 OOF threshold | Isolate the model threshold's incremental value |
| Full policy | The same opportunity plus the frozen threshold and exact intended candidate policy | Measure the complete declared policy package |

The common cohort is built before threshold, policy, capacity, cost, or outcome fields are inspected. Selected and rejected observations remain in the ledger. Setup-only is a matched-cohort control sharing actual prediction-recording availability; it does not claim to reconstruct an earlier standalone detector.

The comparison roster is every recipe in `configs/broad_full_session_shadow_selection_v0.json` (22 queue IDs: minimum core, expanded core, reserve, and diagnostic companion). IDs 7 and 17 are retained as historical B1 review examples, not as the B3 universe. Each selected recipe receives setup-only, model-threshold, and intended-full-policy branch declarations before fitting or evaluation. The research collection uses `b2_research_scope_v1`; the archived `baseline_scope_v1` treatment remains unchanged.

The full-policy branch uses the declared candidate threshold plus high-stress and off-hours abstention, a four-source-bar cooldown, and the causal `es_abs_close_move_20x5m_v1` gate requiring expected movement of at least twice the one-side B1 session spread. It has no regime-threshold or quantile filter. The exact candidate-specific replacement-version policy binding, including artifact and policy hashes, must be frozen before evaluation. If that binding is unavailable, the branch is `not_defined`: no full-policy ledger, P&L, zero-trade result, pass, or substituted historical policy may be reported.

### Shared execution and costs

All branches use the frozen B1 contract: first scheduled open strictly after causal `known_at`, one ES contract, one instrument position at a time, no opposite-side position, zero cooldown, fixed barriers, causal 1-minute resolution where available, adverse-first collision handling, and DATA_GAP/roll censoring without forward search. Capacity rejections are retained rather than removed from the opportunity universe.

The base case is `es_session_cost_v1`: full configured spread per side, session schedule 1/1/1/1.5/2 ticks, 0.25 tick round-trip slippage, and 0.40 tick round-trip commission. The required stress case doubles spread, slippage, and commission on the identical admitted trades. Both cases report gross, component costs, total costs, and net results.

### Freshness and reporting

Live C1 observations require a complete aligned closed bar, matching source identity and timestamp, known receipt time, immutable bar version, verified producer lineage, and actual availability of all causal inputs. Prediction age must be at most 90 seconds after bar close; input age limits are under 30 minutes for 30-minute HTF inputs and under 60 minutes for other completed inputs. Stale, provisional, delayed/backfilled, revised, unknown, or unverified observations are rejected or censored with a retained reason. The completed-bar-plus-one-second assumption is permitted only as explicitly labeled offline replay, not as live freshness evidence.

The fixed window is the first 20 consecutive fully eligible ES sessions after the healthy post-B7 start. Partial sessions are coverage-only. Results are reported for every complete session, calendar week, calendar month, session slice (overlap, London, New York, Asia, off-hours), and the full fixed window. The practical evidence target is approximately 50–100 resolved common opportunities per primary candidate; a shortfall is reported, not repaired by outcome-dependent extension.

### Experimental family-model gate

Any new family/model setup gate is a separate `family_model_setup_gate_v1` experimental variant. It is applied only after the base cohort is materialized, uses its own pre-frozen matcher identity, and receives a separate ledger and report. It is not one of the three primary branches, is not back-applied to historical outcomes, is not the historical deployed policy, and does not authorize installation, promotion, or a paper trial.

Historical inspected windows remain diagnostic. No threshold, gate, execution assumption, cost case, position limit, freshness rule, or reporting window may be changed after C1 outcomes are visible.


<a id="b4_baseline_freeze_20260929"></a>

## B4 baseline freeze — 2026-09-29

*Merged 2026-10-09 from `docs/B4_baseline_freeze_20260929.md`.*

The approved prospective baselines are frozen in [`b4_baseline_freeze_20260929.json`](../configs/b4_baseline_freeze_20260929.json), linked to the B3 comparison protocol. The freeze is for prospective research only; it does not install a model or authorize promotion, scored collection, a paper trial, or orders.

### Frozen baselines

| Candidate | Model | Calibration | Global threshold | Regime thresholds | Setup | Policy |
|---:|---|---|---:|---|---|---|
| 7 | `frvp_long_continuation_setup5_xgb_v1` | none | 0.62 | none | FRVP S5 long LVN Burst | B3/B4 full-policy v1 |
| 17 | `ict_short_continuation_premium_discount_continuation_xgb_v1` | none | 0.53 | none | ICT short premium/discount continuation | B3/B4 full-policy v1 |

The exact replacement model artifacts, setup-detector source hashes, calibration status, thresholds, setup definitions, policy gates, and policy version are recorded in the JSON. Pooled FRVP/ICT models remain diagnostic-only and are not approved baselines.

### Audit guardrails

The September audit's near-misses, tiny hourly samples, and retrospective profitable-hour pockets are descriptive evidence only. They do not change either threshold, create a regime threshold, alter setup definitions, or install an hourly/profitable-hours filter. The primary collection remains full-session. Any session restriction must be registered as a separately frozen prospective experiment with the full-session control retained.

Corrective calibration is owned by B2. It may use development/validation or out-of-fold data only and must produce a new model/calibrator hash and policy version. The new C1 evaluation sample, September audit markouts, near-misses, hourly buckets, and any prospective outcomes are prohibited calibration data. A changed calibration or threshold is a new baseline, not a modification of this freeze.

The new family-model setup gate remains a separate experimental variant under B3; it is not part of either approved baseline and is not the historical deployed policy.


<a id="b4_broad_full_session_shadow_freeze_v1_20261001"></a>

## Broad full-session shadow v1 / expanded B4 freeze — 2026-10-01

*Merged 2026-10-09 from `docs/B4_broad_full_session_shadow_freeze_v1_20261001.md`.*

All 12 selected replacement artifacts passed their candidate-specific B2 review. The freeze binds actual model, scaler, calibrator, selected-feature, fold/purge, OOF-threshold, code, data, and replay evidence. Historical quality metrics are retained as advisory evidence and were not used as a collection veto.

This is not scored-shadow admission. Every full-policy branch remains `not_defined` pending the exact B3 trading-rule binding, and B5/B6/B7 remain fail-closed. No legacy policy or artifact may be substituted.

| Queue | Model | Stream | Calibration | Exact threshold | Features | Model SHA-256 | Full policy |
|---:|---|---|---|---:|---:|---|---|
| 2 | `frvp_long_continuation_xgb_v1` | ES | none | 0.62 | 160 | `f8701763ac91` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 10 | `frvp_long_reversal_xgb_v1` | ES | platt | 0.55999999999999994 | 160 | `bb08d91f7324` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 12 | `ict_long_meta_xgb_v1` | ES | platt | 0.34999999999999992 | 96 | `076024ea07e9` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 13 | `ict_long_reversal_xgb_v1` | ES | platt | 0.34999999999999992 | 96 | `f447c796aa0b` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 15 | `ict_short_meta_xgb_v1` | ES | platt | 0.31999999999999995 | 96 | `0fd19ea888d9` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 16 | `ict_short_reversal_xgb_v1` | ES | platt | 0.28999999999999998 | 96 | `1e5ca95006cd` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 22 | `long_reversal_tcn_v2_20260525_narrow48` | EURUSD | platt | 0.43999999999999995 | 96 | `9c28d4366573` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 26 | `short_reversal_xgb_v2_20260525` | EURUSD | platt | 0.73999999999999999 | 96 | `c862c11002a0` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 3 | `frvp_long_meta_xgb_v1` | ES | platt | 0.49999999999999994 | 160 | `51e4e1115df5` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 19 | `ict_short_reversal_sweep_reclaim_xgb_v1` | ES | platt | 0.25999999999999995 | 96 | `f0a713c7bf6c` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 21 | `long_ote_meta_tcn_champion` | EURUSD | platt | 0.55999999999999994 | 160 | `f05cbd5528f8` | `not_defined_pending_exact_B3_trading_rule_binding` |
| 25 | `short_ote_meta_tcn_champion` | EURUSD | platt | 0.62 | 96 | `267c235872a1` | `not_defined_pending_exact_B3_trading_rule_binding` |

Selection: `configs/broad_full_session_shadow_v1.json` (`08729bea6852213d8afdea6c9928c727844bfde7c95a8eb7b73a6e37c5d61808`)

Expanded B4 freeze: `configs/b4_broad_full_session_shadow_freeze_v1.json`

Verification: `docs/audits/broad_full_session_shadow_v1_verification.json`


<a id="broad_full_session_shadow_route_bindings_v1_20261001"></a>

## Broad full-session shadow route bindings v1

*Merged 2026-10-09 from `docs/broad_full_session_shadow_route_bindings_v1_20261001.md`.*

Status: complete on 2026-10-01 for the 12 models frozen in `broad_full_session_shadow_v1`. This is an outcome/simulation binding only; it does not enable scored collection, paper trading, alerts, broker orders, or promotion.

### Frozen identities

- Selection: `configs/broad_full_session_shadow_v1.json`, SHA-256 `08729bea6852213d8afdea6c9928c727844bfde7c95a8eb7b73a6e37c5d61808`.
- B1 catalogue: `configs/outcome_contracts/b1_v3.json`, SHA-256 `a995a09d012ba36038d41f297495d467ca9e6a3ae0e72b32e7b21d813d7d8b63`.
- Route manifest: `configs/broad_full_session_shadow_route_bindings_v1.json`, file SHA-256 `d53af7931605e625842f350a6719465f161be7c8a356c5e166587e40e53b1868`, validated content SHA-256 `5651f14ec673ef2ffa9cca31d61e2e235fef4a72852b98189737415ddf17826f`; immutable lock SHA-256 `207ac656726512879bad8812031e166ebf85058ea18baa2e11609e85d237a262`.
- Pooled routing remains `pooled_detector_selected_same_side_v1`: only the detector-selected, contemporaneous, same-side setup may supply the executable contract. Other contemporaneous matches remain counterfactual markouts; later setups cannot be substituted.

### Result

The manifest covers all 12 selected models with 51 exact model/setup routes:

- 41 ES routes use an existing B1 FRVP or ICT outcome contract and `es_session_cost_v1`.
- The two OTE reversal models bind to `eurusd_ote_reversal_eval_v2` and `eurusd_session_cost_v1`.
- The two OTE meta models bind by the causal setup family to the existing reversal, continuation, breakout, or sustained-breakout v2 contract and `eurusd_session_cost_v1`. There is intentionally no OTE-meta exit contract.
- ICT classic breaker is absent from the selected route surface and remains excluded.

`model_testing/outcome_routes.py` verifies the selection and catalogue hashes, requires exact selected-roster coverage, resolves every declared route through the B1 catalogue, validates the cost identity, and fails closed for unknown models, undeclared setups, or instrument/side mismatches.

Validation: `ote_venv\Scripts\python.exe -m pytest tests\test_outcome_route_bindings.py tests\test_outcome_contracts.py tests\test_outcome_ote_contracts.py tests\test_pooled_routing_contract.py -q` -> `169 passed`.


<a id="b4_broad_full_session_shadow_freeze_v2_20261001"></a>

## Broad full-session shadow v2 / expanded B4 freeze — 2026-10-01

*Merged 2026-10-09 from `docs/B4_broad_full_session_shadow_freeze_v2_20261001.md`.*

All 16 selected replacement artifacts passed their candidate-specific B2 review. The freeze binds actual model, scaler, calibrator, selected-feature, fold/purge, OOF-threshold, code, data, and replay evidence. Historical quality metrics are retained as advisory evidence and were not used as a collection veto.

This is not scored-shadow admission. Every full-policy branch remains `not_defined` pending the exact B3 trading-rule binding, and B5/B6/B7 remain fail-closed. No legacy policy or artifact may be substituted.

| Queue | Model | Stream | Calibration | Exact threshold | Features | Model SHA-256 | Full policy |
|---:|---|---|---|---:|---:|---|---|
| 2 | `frvp_long_continuation_xgb_v1` | ES | none | 0.62 | 160 | `f8701763ac91` | not_defined_pending_exact_B3_trading_rule_binding |
| 10 | `frvp_long_reversal_xgb_v1` | ES | platt | 0.55999999999999994 | 160 | `bb08d91f7324` | not_defined_pending_exact_B3_trading_rule_binding |
| 12 | `ict_long_meta_xgb_v1` | ES | platt | 0.34999999999999992 | 96 | `076024ea07e9` | not_defined_pending_exact_B3_trading_rule_binding |
| 13 | `ict_long_reversal_xgb_v1` | ES | platt | 0.34999999999999992 | 96 | `f447c796aa0b` | not_defined_pending_exact_B3_trading_rule_binding |
| 15 | `ict_short_meta_xgb_v1` | ES | platt | 0.31999999999999995 | 96 | `0fd19ea888d9` | not_defined_pending_exact_B3_trading_rule_binding |
| 16 | `ict_short_reversal_xgb_v1` | ES | platt | 0.28999999999999998 | 96 | `1e5ca95006cd` | not_defined_pending_exact_B3_trading_rule_binding |
| 22 | `long_reversal_tcn_v2_20260525_narrow48` | EURUSD | platt | 0.43999999999999995 | 96 | `9c28d4366573` | not_defined_pending_exact_B3_trading_rule_binding |
| 26 | `short_reversal_xgb_v2_20260525` | EURUSD | platt | 0.73999999999999999 | 96 | `c862c11002a0` | not_defined_pending_exact_B3_trading_rule_binding |
| 3 | `frvp_long_meta_xgb_v1` | ES | platt | 0.49999999999999994 | 160 | `51e4e1115df5` | not_defined_pending_exact_B3_trading_rule_binding |
| 19 | `ict_short_reversal_sweep_reclaim_xgb_v1` | ES | platt | 0.25999999999999995 | 96 | `f0a713c7bf6c` | not_defined_pending_exact_B3_trading_rule_binding |
| 21 | `long_ote_meta_tcn_champion` | EURUSD | platt | 0.55999999999999994 | 160 | `f05cbd5528f8` | not_defined_pending_exact_B3_trading_rule_binding |
| 25 | `short_ote_meta_tcn_champion` | EURUSD | platt | 0.62 | 96 | `267c235872a1` | not_defined_pending_exact_B3_trading_rule_binding |
| 5 | `frvp_long_continuation_setup2_xgb_v1` | ES | none | 0.62 | 160 | `d0f41ad4c72d` | not_defined_pending_exact_B3_trading_rule_binding |
| 23 | `long_union_tcn_champion` | EURUSD | platt | 0.58999999999999997 | 128 | `e124186430fe` | not_defined_pending_exact_B3_trading_rule_binding |
| 18 | `ict_short_reversal_ifvg_reversal_xgb_v1` | ES | platt | 0.31999999999999995 | 96 | `6b4a7bdf323d` | not_defined_pending_exact_B3_trading_rule_binding |
| 14 | `ict_short_continuation_xgb_v1` | ES | none | 0.52999999999999992 | 96 | `f8e724fddca8` | not_defined_pending_exact_B3_trading_rule_binding |

Selection: `configs/broad_full_session_shadow_v2.json` (`a89e8d2622d9f6cf915eb80b1044a1e8af7dacd7cb99cca1fc6e2b9e23e6a430`)

Expanded B4 freeze: `configs/b4_broad_full_session_shadow_freeze_v2.json`

Verification: `docs/audits/broad_full_session_shadow_v2_verification.json`
16-model selection roster: `configs/broad_full_session_shadow_selection_v1.json` (`bd6c860088b3a92e945ccbbb10c41742431e226dde860484763052c826ddb853`), parent `configs/broad_full_session_shadow_selection_v0.json`.

The 12 v1 entries were re-hashed from disk and carried forward byte-identical to `configs/b4_broad_full_session_shadow_freeze_v1.json`; IDs 5/23/18/14 are the new entries.


<a id="broad_full_session_shadow_route_bindings_v2_20261001"></a>

## Broad full-session shadow route bindings v2

*Merged 2026-10-09 from `docs/broad_full_session_shadow_route_bindings_v2_20261001.md`.*

Status: complete on 2026-10-01 for the 16 models frozen in `broad_full_session_shadow_v2` (the 12 v1 core models plus the latency-approved extension IDs 5/23/18/14). This is an outcome/simulation binding only; it does not enable scored collection, paper trading, alerts, broker orders, or promotion. The v0 selection, `broad_full_session_shadow_v1`, the 12-model v1 B4 freeze, route bindings v1 and its lock, and the B1 v3 catalogue are unchanged (hashes re-checked before and after: `logs/catchup_p4_immutable_hashes_before_20261001.txt`).

### Frozen identities

- 16-model selection roster: `configs/broad_full_session_shadow_selection_v1.json`, SHA-256 `bd6c860088b3a92e945ccbbb10c41742431e226dde860484763052c826ddb853`; parent `configs/broad_full_session_shadow_selection_v0.json`. Each entry keeps its v0 role, instrument, side/family/setup and recipe/data parents and adds its review evidence.
- Selection bindings: `configs/broad_full_session_shadow_v2.json`, SHA-256 `a89e8d2622d9f6cf915eb80b1044a1e8af7dacd7cb99cca1fc6e2b9e23e6a430`.
- B4 freeze: `configs/b4_broad_full_session_shadow_freeze_v2.json`, SHA-256 `b9a7bc674b2b38118fcc2187b8c94c9f1aec6b2d84869572244f33f5abd6cb5b`; lock `configs/b4_broad_full_session_shadow_freeze_v2.lock.json`; parent B4 is the v1 freeze (`6c25d379…966fe`). Summary table: `docs/B3_B4_freeze_and_routes.md#b4_broad_full_session_shadow_freeze_v2_20261001`; verification: `docs/audits/broad_full_session_shadow_v2_verification.json`.
- B1 catalogue: `configs/outcome_contracts/b1_v3.json`, SHA-256 `a995a09d012ba36038d41f297495d467ca9e6a3ae0e72b32e7b21d813d7d8b63` (same pin as v1).
- Route manifest: `configs/broad_full_session_shadow_route_bindings_v2.json`, file SHA-256 `2473e44e4054068eac4fda1219586401c300b3a2e7a3aa54202e08bf873045d6`, validated content SHA-256 `18f49ce036c15dac14c9b1f48a5868011a0605a943e1207d4383356268785520`; lock SHA-256 `0d469596a6f69517ea36b5c800dfe2ad6e9c0b59b38840bfb11a7baf60aa482e`.

### Freeze carry-forward

`scripts/publish_broad_full_session_shadow.py publish --version 2` first checks the v1 selection and freeze against the v1 lock, then rebuilds all 12 v1 entries from the bytes on disk (model, scaler, calibrator, outputs, prepared manifest, source history, code snapshots) and requires each to be canonically identical to its v1 freeze entry before adding IDs 5/23/18/14. All 12 matched. `verify --version 1` and `verify --version 2` both pass.

| Queue | Model | Stream | Calibration | Threshold | Features | Full policy |
|---:|---|---|---|---:|---:|---|
| 5 | `frvp_long_continuation_setup2_xgb_v1` | ES | none | 0.62 | 160 | not_defined |
| 23 | `long_union_tcn_champion` | EURUSD | platt | 0.59 | 128 | not_defined |
| 18 | `ict_short_reversal_ifvg_reversal_xgb_v1` | ES | platt | 0.32 | 96 | not_defined |
| 14 | `ict_short_continuation_xgb_v1` | ES | none | 0.53 | 96 | not_defined |

Exact threshold floats and all artifact hashes are in the freeze. Every full-policy branch remains `not_defined_pending_exact_B3_trading_rule_binding`.

### Routes

The manifest covers 16 models with 59 exact model/setup routes. The first 12 bindings are the v1 bindings, asserted identical.

| Stream | v1 routes | v2 routes | Added |
|---|---:|---:|---|
| ES / FRVP | 12 | 13 | ID 5 |
| ES / ICT | 29 | 32 | IDs 18, 14 |
| EURUSD / OTE | 10 | 14 | ID 23 |
| Total | 51 | 59 | |

New routes (every contract ID verified present in the B1 v3 catalogue and resolved through it):

- ID 5 (long, `continuation_setup2`, `exact_contemporaneous_same_side`, as ID 19): `setup2` → `es_frvp_setup2_eval_v2`, `es_session_cost_v1`.
- ID 18 (short, `reversal_ifvg_reversal`, `exact_contemporaneous_same_side`, as ID 19): `ifvg_reversal` → `es_ict_ifvg_reversal_eval_v1`, `es_session_cost_v1`.
- ID 14 (short, `continuation`, `detector_selected_contemporaneous_same_side`, as the ID 16 family): `premium_discount_continuation` → `es_ict_premium_discount_continuation_eval_v1`; `displacement_continuation_after_raid` → `es_ict_displacement_continuation_after_raid_eval_v1`; `es_session_cost_v1`.
- ID 23 (long, `union`, `detector_selected_contemporaneous_same_side`): the same four routes as ID 21 — `reversal`, `continuation`, `breakout`, `breakout_sustained` → the matching `eurusd_ote_<family>_eval_v2`, `eurusd_session_cost_v1`. The detector-selected contemporaneous same-side setup is the primary route; other matches are non-executable counterfactual markouts only. Family v2 exits; there is no union or meta exit contract. EURUSD only, pips.

ID 23's label is the same three-component union as ID 21's (reversal, continuation pullback, breakout), and ID 21 binds four routes including `breakout_sustained`; ID 23 mirrors ID 21 exactly as instructed.

### Code and validation

- `model_testing/outcome_routes.py`: adds `ROUTE_BINDINGS_V1` / `ROUTE_BINDINGS_V2`; the default `ROUTE_BINDINGS` is now v2; the "no pooled exit contract" check covers `union` as well as `meta`. Loader and resolver are otherwise unchanged and still fail closed on unknown models, undeclared setups, and instrument/side mismatches. No `ote_live` or app code imports this module, so `docs/live_app_change_journal.md` is unchanged (as for v1).
- `scripts/publish_route_bindings_v2.py`: one-off publisher; refuses to overwrite.
- `tests/test_outcome_route_bindings.py`: existing tests parametrized over v1/v2, plus one v1→v2 carry-forward/extension test.
- Focused: `ote_venv\Scripts\python.exe -m pytest tests\test_outcome_route_bindings.py tests\test_outcome_contracts.py tests\test_outcome_ote_contracts.py tests\test_pooled_routing_contract.py -q` → `175 passed` (`logs/catchup_p4_tests_20261001.log`).
- Legacy bundle tree hashes for IDs 5/23/14/18 unchanged versus the progress file.


<a id="b3_prospective_comparison_v2_20261002"></a>

## B3 prospective comparison v2 (dual-stream) — 2026-10-02

*Merged 2026-10-09 from `docs/B3_prospective_comparison_v2_20261002.md`.*

The prospective C1 comparison for the 16-model roster is frozen in [`b3_b4_candidate_review_v2_20261002.json`](../configs/b3_b4_candidate_review_v2_20261002.json) (SHA-256 `4efbc9f50850115fa5a4ad3fb3d22042304261274461dac8ac0c8ebb940be0b7`), with lock `b3_b4_candidate_review_v2_20261002.lock.json` (SHA-256 `e6a54a71adf2bb48e0da17956283965353b6c1fa8c2f6708c5b94a30149173b1`). It must be hashed and recorded before the first healthy post-B7 observation. It contains no C1 outcomes and grants no collection, paper, alert, order or promotion authority.

The v1 protocol ([`#b3_prospective_comparison_20260929`](#b3_prospective_comparison_20260929)) and its lock are unchanged and remain the parent.

### What changed from v1

- v1 listed OTE IDs in its roster, but its execution, freshness, cost, session-window, expected-move and position text was ES-only. v2 splits those assumptions into two streams. `streams.ES` is the v1 text carried verbatim; `streams.EURUSD` is new.
- The roster is the 16-model selection with frozen replacement artifacts (`configs/broad_full_session_shadow_selection_v1.json`), not the 22-recipe v0 selection. v0 candidates 27, 6, 7, 11, 17 and 1 have no frozen replacement in the 16-model freeze and are not in this roster; v1 remains their declaration.
- The three branches (setup only, model threshold, full policy), the common-opportunity rule, the `not_defined` rule, the experimental family/model gate and the predeclared outputs are unchanged.

The lock pins the v2 protocol, the v1 parent, the 16-model selection, `broad_full_session_shadow_v2`, B4 freeze v2, route bindings v2 and the B1 v3 catalogue.

### Streams

| | ES | EURUSD |
|---|---|---|
| Candidates | 2, 10, 12, 13, 15, 16, 3, 19, 5, 18, 14 | 22, 26, 21, 25, 23 |
| Strategies | FRVP, ICT | OTE |
| Routes / contracts | 45 routes, 14 B1 contracts | 14 routes, 4 OTE family v2 contracts |
| Cost contract | `es_session_cost_v1` | `eurusd_session_cost_v1` |
| Units | ES ticks, USD | pips only |
| Window | first 20 fully eligible ES sessions | first 20 fully eligible EURUSD sessions, started and counted independently |
| Expected-move gate | `es_abs_close_move_20x5m_v1`, ticks | `not_defined` |

### EURUSD branch

- **Contracts.** The four frozen contracts bound by route bindings v2: `eurusd_ote_reversal_eval_v2` (completed 1h ATR, stop 2.0 / target 1.0, 120 bars), `eurusd_ote_continuation_eval_v2` (completed 30m ATR, 0.7 / 0.8, 36 bars), `eurusd_ote_breakout_eval_v2` and `eurusd_ote_breakout_sustained_eval_v2` (completed 30m ATR, 0.7 / 0.9, 24 bars). All hold overnight. Meta and union models use the family contract of the detector-selected contemporaneous same-side setup; there is no meta or union exit, and other matches are non-executable counterfactual markouts.
- **Execution.** First scheduled EURUSD open strictly after causal `known_at`, at most one 5-minute bar of delay, unadjusted open, unrounded FX barrier grid, causal 1-minute resolution where available, adverse-first collisions, DATA_GAP censoring, no forward search. Weekend and holiday closures are not inferred; there is no futures roll.
- **Position.** One position per instrument, one unit, no opposite side, zero cooldown, pip accounting. `eurusd_session_cost_v1` has no cash conversion, so there is no USD equity, USD P&L or lot conversion for EURUSD under v2.
- **Costs.** Full spread per side: 1.0 pip overlap, 1.5 London/New York, 2.5 Asia, 3.0 off-hours; 0.30 pip round-trip slippage; 0.35 pip round-trip commission. The double-all-costs stress is required on the identical admitted trades.
- **Freshness.** Complete aligned closed EURUSD 5-minute bar with matching series identity, known receipt time, immutable bar version and verified lineage. Prediction age at most 90 seconds after bar close; 30m structural inputs under 30 minutes, 1h structural and other completed inputs under 60 minutes. These limits mirror the ES values. ES bars are never an input.
- **Reporting.** Session clock `london_07_16_ny_08_17_asia_ny_19_02_v1` (the clock in the frozen cost contract); per complete EURUSD session, week, month, session slice and the fixed 20-session aggregate.
- **Expected move.** No versioned causal EURUSD expected-move contract is frozen. The ES contract and ES ticks must not be applied, and the unversioned `expected_move_pips` context field is not a contract. Until a pip-denominated contract and exact policy bindings are frozen in a new protocol version before C1, `full_policy` is `not_defined` for every EURUSD candidate. Setup-only and model-threshold branches are still collected.

### Cross-stream rules

Opportunity universes, ledgers, capacity, windows, units and reporting are separate per stream. Neither stream consumes the other's capacity. ES ticks/USD and EURUSD pips are never converted, summed, netted or ranked as one P&L series; the only cross-stream material allowed is coverage counts shown side by side. No OTE model runs on ES bars and no ES model runs on EURUSD bars.

### Code and validation

- `scripts/publish_b3_protocol_v2.py` builds v2 from the locked v1 text, the 16-model selection, route bindings v2 and the B1 v3 catalogue, and refuses to overwrite.
- `model_testing/b3_comparison.py`: `build_b3_comparison_plan` now validates `streams` when present (stream set equals the selected instruments, candidate partition, frozen cost contract per stream, no foreign outcome contracts, distinct units, expected-move status consistent, aggregation forbidden). v1 validation is unchanged.
- Validate: `ote_venv\Scripts\python.exe scripts\validate_b3_comparison_plan.py --protocol configs\b3_b4_candidate_review_v2_20261002.json --selection configs\broad_full_session_shadow_selection_v1.json`.
- Tests: `tests\test_b3_comparison_plan.py` (v1 test unchanged, two v2 tests added) with `tests\test_outcome_route_bindings.py` → 15 passed.

### Open points carried forward

- All 16 full policies are still `not_defined`; exact candidate policy bindings are separate work. EURUSD additionally needs a frozen expected-move contract first.
- v1 spells the ES reporting clock `london_07_16_ny_08_17_asia_ny_v1`, while the frozen cost contracts name `london_07_16_ny_08_17_asia_ny_19_02_v1`. The ES text was carried verbatim, so the v1 spelling is preserved in `streams.ES`.

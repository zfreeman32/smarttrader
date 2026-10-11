# ES archived setup evidence audit

Snapshot: `es_audit_snapshot.sqlite3`. This extraction never calls either setup detector, rebuilds features, or generates new historical model predictions. All times below use America/New_York unless marked UTC.

**Recoverable scope.** There are 87 unique retained setup events: 47 FRVP (31 short, 16 long) and 40 ICT (17 short, 23 long). FRVP type/side/confidence are present in all 19,284 FRVP prediction policy contexts; 433 prediction rows contain a fired setup, collapsing to 47 setup bars. All 40 retained FRVP dashboard markings exactly match their archived context close and confidence. The other seven FRVP events precede the rolling dashboard retention window. ICT type/side/family are absent from all 16,683 ICT prediction policy contexts. Only 40 retained ICT dashboard markings can be recovered as setup events; older ICT setup markings cannot be fully audited from this snapshot.

**What the dashboard evidence means.** Current source calls dashboard persistence on each processed bar before the model loop (`ote_live/ingestion/signals.py:641`). The dashboard detector result's latest row is appended to recent_setups only when fired, and each strategy retains at most 40 events (`ote_live/dashboard/view_state.py:14`, `:18`, `:94`). These are retained outputs from historical processing, not labels reconstructed by this audit. They may originate from delayed processing or catch-up: a bar timestamp is not its original wall-clock observation time. The current dashboard state's Sep 11 update timestamp is not the original timestamp when each older setup was marked. FVG history in the same ICT payload is structural-zone lifecycle data, not additional setup/trade events.

**Revision limitations.** No duplicate model/timestamp prediction records or conflicting FRVP labels/close values are present in this snapshot. The dashboard append helper ignores a new fire when the last event already has the same timestamp (`view_state.py:533`); an earlier fire is not removed if the revised bar subsequently reports no setup. Therefore absence of revisions in storage does not establish that the detector or incomplete bar never changed. ICT dashboard events do not carry per-event recorded_at timestamps or detector-version identifiers. `original_mark_recorded_at_utc` in setup_events.csv means the first archived FRVP prediction record containing that setup, not an exact chart-render timestamp; it is blank for ICT.

**Price and timing association.** Every recovered event has model outputs at its exact source bar timestamp: 772 event/model records, 433 FRVP and 339 ICT. Every model context close matches the recorded setup close. However, at least one associated context has ibkr_is_complete=False on 30/47 FRVP and 25/40 ICT marks. First associated model output lag from the source bar timestamp has median 17.33 minutes for FRVP and 17.15 minutes for ICT, maximum 291.38 and 419.07 minutes. These associations support studying model behavior at the source setup bar; they do not establish fills executable at that historical close. A five-minute bar's timestamp may be its opening timestamp, so the lag from nominal bar completion is five minutes smaller.

**Setup confidence and model probability are different fields.** Rule confidence ranges from 0.4831 to 0.96 for FRVP and 0.5325 to 0.87 for ICT. These values come from rule-based scoring formulas in the setup detectors; they are not calibrated probabilities of making money. Use `confidence` for rule score and `probability` for the separately archived model output. The family and specialist models may target different labels/horizons, so their probabilities are not interchangeable without checking manifest target semantics.

**The largest observed source of apparent model noise is off-setup shadow output.** All 8,454 FRVP specialist model predictions were evaluated across bars. There were 3,306 threshold passes, but only six of those passes occurred on that model's actual setup number and long direction. The other 3,300 (99.82%) were off its intended setup. Shadow decisions return immediately after checking the threshold (`ote_live/policies/decision_engine.py:105`), before setup holds, abstention rules, and cooldown. The caller also suppresses setup force-hold reasons for shadow mode (`ote_live/ingestion/signals.py:711`). Thus the archive's candidate_passed_threshold flag alone is not a strategy-qualified entry. This behavior is consistent with the current shadow implementation; it does not mean 3,300 unintended trades were placed.

**Families and specialists.** FRVP reversal pools S1/S4/S6; continuation pools S2/S3/S5 (`frvp/target_lanes.py:7`). A family model emits a score over the broader label family; a specialist targets one setup and direction. ICT reversal includes sweep reclaim and IFVG reversal here; premium/discount continuation belongs to continuation. Meta models combine family-level targets. The setup gate recognizes only specialist model IDs; broad family model IDs return a matched gate without checking the current setup (`ote_live/models/setup_family.py:131`). For meaningful setup comparisons, this audit explicitly requires strategy, direction, family (or meta), and specialist setup type to match.

**Coverage is asymmetric.** There are six FRVP long specialist models and three ICT short specialist models. No FRVP short specialist and no ICT long specialist appears in this archive. The FRVP short reversal family has only 31 archived scores, all outside the 47 fired FRVP setup bars, so no FRVP short reversal probability exists at these retained reversal marks. The 47 FRVP events have either five or eleven same-bar model scores, and the 40 ICT events have six or nine. Long S1 and long S2 specialists have zero matching event observations during their active coverage, despite producing 1,409 scores each. The count of raw scores substantially overstates the count of relevant validation examples.

**Recovered setup composition**

| Strategy | Setup | Family | Long | Short |
|---|---|---|---:|---:|
| FRVP | 1 | reversal | 0 | 3 |
| FRVP | 2 | continuation | 2 | 1 |
| FRVP | 3 | continuation | 6 | 8 |
| FRVP | 4 | reversal | 2 | 0 |
| FRVP | 5 | continuation | 3 | 10 |
| FRVP | 6 | reversal | 3 | 9 |
| ICT | ifvg_reversal | reversal | 13 | 10 |
| ICT | premium_discount_continuation | continuation | 3 | 2 |
| ICT | sweep_reclaim | reversal | 7 | 5 |

**Matching specialist output at the setup mark**

| Model | Matching marks | Threshold passes | Mean model probability |
|---|---:|---:|---:|
| frvp_long_continuation_setup2_xgb_v1 | 0 | 0 | — |
| frvp_long_continuation_setup3_xgb_v1 | 4 | 3 | 0.668 |
| frvp_long_continuation_setup5_xgb_v1 | 1 | 1 | 0.700 |
| frvp_long_reversal_setup1_xgb_v1 | 0 | 0 | — |
| frvp_long_reversal_setup4_xgb_v1 | 2 | 0 | 0.192 |
| frvp_long_reversal_setup6_xgb_v1 | 3 | 2 | 0.621 |
| ict_short_continuation_premium_discount_continuation_xgb_v1 | 2 | 1 | 0.443 |
| ict_short_reversal_ifvg_reversal_xgb_v1 | 8 | 1 | 0.273 |
| ict_short_reversal_sweep_reclaim_xgb_v1 | 5 | 0 | 0.176 |

Only 25 specialist observations match an archived setup: ten FRVP and fifteen ICT. Eight pass threshold. This is too small and uneven to rank all specialists reliably.

**Matching family output at the setup mark**

| Model | Matching marks | Threshold passes | Mean model probability |
|---|---:|---:|---:|
| frvp_long_continuation_xgb_v1 | 11 | 5 | 0.693 |
| frvp_long_meta_xgb_v1 | 16 | 12 | 0.456 |
| frvp_long_reversal_xgb_v1 | 5 | 2 | 0.566 |
| frvp_short_continuation_tcn_v1 | 19 | 10 | 0.503 |
| frvp_short_meta_xgb_v1 | 31 | 25 | 0.488 |
| ict_long_continuation_xgb_v1 | 3 | 3 | 0.680 |
| ict_long_meta_xgb_v1 | 23 | 2 | 0.339 |
| ict_long_reversal_xgb_v1 | 20 | 1 | 0.291 |
| ict_short_continuation_xgb_v1 | 2 | 0 | 0.437 |
| ict_short_meta_xgb_v1 | 17 | 0 | 0.239 |
| ict_short_reversal_xgb_v1 | 15 | 0 | 0.234 |

ICT family reversal filters agree with very few markings: long reversal passes one of twenty and short reversal zero of fifteen. Long continuation passes all three matching continuation marks; short continuation zero of two. On FRVP, the long/short meta models pass 12/16 and 25/31 respective setup marks. This describes model selectivity; it does not by itself prove which filters improve returns.

**Eight specialist threshold passes on their matching setup**

| Source bar ET | Model | Probability | Threshold |
|---|---|---:|---:|
| 2026-09-04 09:00 | frvp_long_reversal_setup6_xgb_v1 | 0.6386 | 0.6200 |
| 2026-09-04 09:20 | frvp_long_reversal_setup6_xgb_v1 | 0.6384 | 0.6200 |
| 2026-09-07 07:10 | frvp_long_continuation_setup5_xgb_v1 | 0.6997 | 0.6500 |
| 2026-09-08 18:50 | ict_short_continuation_premium_discount_continuation_xgb_v1 | 0.4718 | 0.4400 |
| 2026-09-09 16:00 | ict_short_reversal_ifvg_reversal_xgb_v1 | 0.3579 | 0.3500 |
| 2026-09-11 01:40 | frvp_long_continuation_setup3_xgb_v1 | 0.6800 | 0.6500 |
| 2026-09-11 02:45 | frvp_long_continuation_setup3_xgb_v1 | 0.6784 | 0.6500 |
| 2026-09-11 08:45 | frvp_long_continuation_setup3_xgb_v1 | 0.6727 | 0.6500 |

**Trade geometry and outcomes.** All 40 ICT retained marks contain entry, stop reference, and target reference with positive directional risk and reward. These are setup references, not recorded broker orders or fills. FRVP dashboard events do not contain an executable bracket. Outcome evaluation must use bars after the setup, flag gaps and same-bar stop/target ambiguity, and account for recorded-time availability. See the companion root audit for price-action markouts, simulated bracket behavior, trade ledger results, and profitability.

**Audit improvements indicated by the evidence.** Persist immutable setup events for both systems with detector/version hash, source bar version and completion flag, observed-at timestamp, exact model prediction IDs, gate match/result, setup geometry, and revision events. Keep threshold candidates separate from gate-qualified shadow entries in analytics, and evaluate the actual abstention/cooldown policy for shadow execution studies. Preserve more than the latest 40 ICT events. These recommendations follow concrete limitations above; no runtime changes were made by this audit.

Artifacts: `setup_events.csv` is one row per unique recovered setup; `setup_model_probabilities.csv` has all same-bar model outputs, including off-direction/family/type rows explicitly flagged; `setup_family_model_agreement.csv` and `setup_audit_summary.json` summarize coverage; `setup_audit.py` reproduces extraction from the read-only frozen snapshot.

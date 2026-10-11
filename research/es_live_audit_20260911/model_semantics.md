# ES model/setup semantics and feature-health audit

Evidence: read-only source review; frozen `es_audit_snapshot.sqlite3`; collector heartbeat copied to `es_heartbeat_evidence.json`; current-manifest extraction in `current_model_semantics.csv`. No collector, registry, model, order, or live database changes were made.

## What is currently running

The heartbeat captured at 2026-09-12 01:59 UTC (Friday September 11, 21:59 ET) reports **20 loaded models, all shadow, zero active models, zero emitted signals in the current process, delayed IBKR data, and both FRVP and ICT paper-signal ledgers disabled**. Its latest source bar and prediction are September 11 20:45 UTC /16:45 ET. `ict_delayed_test_mode=true`. The last heartbeat is stale after market close; that alone does not establish a market-hours outage. The account mode is paper, which describes the feed connection, not evidence of filled paper orders.

The runtime determines shadow status from both manifest status and force-shadow configuration (`ote_live/ingestion/signals.py:336`). The delayed ICT test overrides even the manifest-active ICT long-meta (`ote_live/scripts/run_es_live_collector.py:917`). The heartbeat's actual model roster is stronger evidence of current execution mode than the registry's `active` label. FRVP currently loads the July 21 shadow bundle, not the frozen August 16 paper-signal bundle.

The snapshot contains only `dashboard_view_state` and `ibkr.market_data` runtime-state scopes; there is no `frvp.paper_signal.confirmation` state. Therefore no stored FRVP confirmation start was found in this snapshot. The confirmation clock requires separate persisted start evidence (`ote_live/storage/frvp_confirmation_lifecycle.py:63`).

## Family models versus setup models

Pooled reversal/continuation models learn the corresponding broader event family; meta models pool reversal and continuation outcomes in one direction. Setup-specific models learn a selected setup/type and direction. The word `setup_family` in filenames refers to these setup-specific routes and should not be confused with pooled family models.

FRVP reversal comprises S1/S4/S6; continuation comprises S2/S3/S5 (`frvp/target_lanes.py:7`). The current roster has five pooled FRVP models and six long-only setup models. The deprecated pooled short-reversal has 31 historic deduplicated prediction rows but is absent from the current loaded roster. There are no current FRVP short setup-specific models to compare fairly against FRVP long setup models.

ICT has six pooled models, plus three short-only specialists: sweep reclaim, IFVG reversal, and premium/discount continuation. Therefore an aggregate family-versus-specialist comparison is also confounded by direction and different collection windows. Compare the same marked setup, direction, timestamp, and outcome horizon.

| Current model | Threshold | Post-score calibration artifact | Selected features |
|---|---:|---|---:|
| FRVP long reversal | 0.60 | Platt | 16 |
| FRVP long continuation | 0.70 | None | 8 |
| FRVP long meta | 0.45 | Platt | 9 |
| FRVP short continuation TCN | 0.55 | None | 42 |
| FRVP short meta | 0.50 | Platt | 21 |
| FRVP long S1 reversal | 0.68 | Platt | 5 |
| FRVP long S2 continuation | 0.71 | None | 15 |
| FRVP long S3 continuation | 0.65 | None | 15 |
| FRVP long S4 reversal | 0.23 | Platt | 13 |
| FRVP long S5 continuation | 0.65 | None | 10 |
| FRVP long S6 reversal | 0.62 | Platt | 17 |
| ICT long meta | 0.40 | Platt | 96 |
| ICT long reversal | 0.40 | Platt | 96 |
| ICT long continuation | 0.65 | None | 96 |
| ICT short reversal | 0.40 | Platt | 96 |
| ICT short meta | 0.40 | Platt | 96 |
| ICT short continuation | 0.55 | None | 96 |
| ICT short sweep reclaim | 0.29 | Platt | 96 |
| ICT short IFVG reversal | 0.35 | Platt | 96 |
| ICT short premium/discount continuation | 0.44 | None | 96 |

Eight currently loaded models explicitly have no calibrator artifact. For each, stored `raw_score` equals `calibrated_probability` on every deduplicated observed prediction. This is consistent with the loader, which returns raw probabilities when the calibrator is null (`ote_live/models/calibrators.py:29`), rather than evidence of a missing-file error. The common database column name does not establish that every model score was recalibrated. Different label prevalence and calibration methods make absolute score comparisons across models misleading.

## Setup marking and model gating

FRVP rule names are documented in `ote_live/dashboard/app.py:63`:

| Setup | Rule interpretation |
|---|---|
| S1 Value Edge Fade | Balanced profile/open auction; fade VAH short or VAL long without true displacement. |
| S2 Breakout Retest | Retest and hold VAH/VAL within 15 bars of a real breakout. |
| S3 Value-Area Breakout | High-volume displacement outside VAH/VAL; follow breakout direction. |
| S4 Failed Auction | Brief excursion and sweep outside value, then quiet re-entry; fade the failure. |
| S5 LVN Burst | Displacement within 0.30 ATR of a low-volume node. |
| S6 HVN Magnet | Quiet balanced inside-value environment; trade toward the nearer high-volume node. |

Rule confidence is a hand-coded score, not ML probability. FRVP formulas and winner selection are in `frvp/setups/detector.py:285` and `:534`. Multiple simultaneous candidates are reduced to one selected output, ranked by confidence then priority. FRVP fires on a newly eligible condition, not every bar that condition remains true (`:152`). ICT similarly reduces candidates to a primary event and applies anchor-specific spacing, plus a default maximum of three same-side fires per session (`ict/setups/detector.py:190`, `ict/config/setups.py:57`). A chart mark count is therefore not the count of all candidate patterns present in price action.

Only model names matching the setup-specific route regex are setup-gated (`ote_live/models/setup_family.py:90`). The model must match the current event's setup type and side, and ICT family too (`:257`). Pooled family models return matched unconditionally (`:133`), so a family-model threshold crossing does not prove a setup fired on that bar.

The shadow decision branch occurs before forced-hold and abstention logic (`ote_live/policies/decision_engine.py:109`). The ingestion path explicitly passes no setup force-hold reasons for shadow bindings (`ote_live/ingestion/signals.py:711`). Thus `shadow + candidate_passed_threshold` means only a score crossed its threshold; it is not a paper trade and does not mean the setup route, time filter, regime filter, spread filter, or cooldown passed.

The sampled ICT persisted policy context has no setup keys, and matched gate results are not persisted as a dedicated decision field. There is a current dashboard state payload, but no complete event-history table. Historical comparisons should state whether setup marks were persisted at the time or reconstructed later. Reconstructing an entire historical frame can differ from the original causal prefix, as the feature example below demonstrates.

## What probabilities predict versus what trade markouts measure

FRVP direct labels initially exclude all rows, unexclude usable event rows, and label a positive only when the modeled take-profit barrier was hit (`data/labeling/frvp_labeling_engine.py:996`, `:1043`). Both family and setup-specific labels receive the same event outcome (`:1023`). Label simulation enters at the following bar's open (`:452`) and uses setup-dependent ATR stop/target geometry (`:277`). A profitable timeout is not automatically a positive FRVP target.

ICT also simulates entry at the following bar's open (`ict/labeling/ict_labeling_engine.py:597`), using reference-based stop/target rules, timeout logic, and continuation extensions/trailing stops. Its positive outcome is `tp` or `timeout_profit` (`:1327`); setup-specific materialization preserves this definition (`scripts/materialize_ict_setup_prepared_root.py:82`). Consequently, neither strategy's model output is automatically the probability of a net-positive fixed-horizon markout entered at the signal bar close. A forward-return score association is useful, but it must not be labeled calibration against the original training target.

Training-horizon assumptions in current manifests are FRVP 120 bars and ICT 20 bars. These assumptions support purge/history contracts and chosen markout horizons; source label configurations can use other setup-dependent barriers. Avoid describing every FRVP setup as trained to predict a ten-hour un-stopped trade win.

## Paper event contracts and profitability limitations

The frozen paper ledger supports **only** FRVP long reversal and ICT long meta. Both are explicitly independent overlapping events, separate from broker positions (`ote_live/storage/frvp_paper_signal.py:60`, `ote_live/storage/ict_paper_signal.py:55`). FRVP enters at signal-close and exits at the close of the 120th future completed canonical bar; ICT uses the 20th bar. Neither ledger uses stops or targets. Settlement uses bar count, so ten hours/100 minutes of bars can span closures or missing periods.

Costs per one ES contract are `2 × session spread + 0.25 slippage + 0.40 commission` ticks. The session schedule produces 2.65 ticks/$33.125 normally, 3.65 ticks/$45.625 in Asia, and 4.65 ticks/$58.125 off hours. One tick is 0.25 index point/$12.50 (`ote_live/storage/frvp_paper_signal.py:24`, `:106`). These are recorded modeling assumptions, not verified brokerage fills or commissions. Portfolio P&L cannot be obtained by summing overlapping hypothetical events without accounting for concurrency, position limits, sizing, margin, and actual entry availability.

The frozen FRVP trial applies threshold 0.60 plus its accepted targeted regime/session exclusions; the currently loaded July 21 shadow contract is a different policy lineage. In particular the frozen August trial adds the `strong_down_high/overlap` exclusion noted as omitted from the old July bundle. ICT's frozen trial threshold is 0.40 with no abstention filters. Neither promotion annotation authorizes broker orders. Historical backtest profits quoted inside manifests are not collected-live profits.

Source timestamps have `bar_open` semantics. A 09:30 five-minute row is complete at 09:35; a delayed feed produces the prediction later still. Entry at the historical row close is therefore an analytical markout, not proof that an executable price was available when the model produced its score. Use prediction `recorded_at_utc` to audit observability/delay and timestamp_utc for market-bar alignment.

## Confirmed ICT live/offline reference parity defect

The frozen snapshot has 2,076 deduplicated ICT long-meta timestamps. The NY RTH flag is correct for all 2,076, with **zero timezone/RTH classification mismatches** after treating September 7 as Labor Day. All 623 RTH timestamps contain the tested prior-RTH/current-open/prior-week references. Outside RTH, prior-RTH high and prior-week high are null on 1,195/1,453 timestamps (82.2%); this is 57.6% of all collected model timestamps. The same missingness appears in selected inputs for multiple ICT models.

The cause is visible in source. ES's session-date boundary is 16:00 ET (`frvp/config/instruments.py:44`; `frvp/sessions/equity.py:105`). The ICT reference builder aggregates only observed RTH sessions, shifts that table, then maps using the current session-date key (`ict/structure/liquidity.py:78`). After 16:00 and before the next RTH, that next-session key does not yet exist in the table, so known prior-RTH and prior-week references become null. This is not a broad timezone failure or simply too little stored history.

`check_ict_reference_prefix_parity.py` reproduces the issue using only frozen collected ES bars. At September 11 08:00 ET, the input includes 4,306 prior bars, but the causal-prefix features are null. Append collected bars through 11:00 ET and recompute the same 08:00 row: prior-RTH high/low/close become 7620/7585.5/7598.75 and prior-week high/low become 7766.5/7621.5. The test also fills `ict_rth_open=7665` before the actual 09:30 open, which is a future-information hazard in full-frame historical calculation. Evidence is in `ict_reference_prefix_parity.json`.

The current-open field being unknown before the open is legitimate. Known previous-session/previous-week levels disappearing is avoidable. Fixing the latter requires mapping from the latest completed historical session/week, while preserving unknown current-session opens until observed. Current-model training exposure to this full-frame behavior has not been fully traced, so the demonstrated source hazard should not be stated as proof that every existing model's training set leaked those values.

Other sparse-feature observations are in `model_selected_feature_missingness.csv`: FRVP S3's pullback channel slope z-score is null on 1,136/1,409 observations (80.6%); FRVP S4's bear order-block distance is null on 1,301/1,409 (92.3%). Event-dependent absence can be valid, especially for XGBoost, and does not itself establish corruption. These features need training/live missing-rate comparison before attributing score behavior to them.

Finally, live fallback code explicitly supplies zero for unavailable `htf_confluence_*` features because the event-specific producer is absent (`ote_live/features/incremental_engine.py:841`). Several ICT selected-feature lists include such helpers. This is another documented live/training semantic difference requiring parity review, not a basis to infer profit or loss by itself.

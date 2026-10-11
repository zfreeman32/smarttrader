**ES live-collection audit — FRVP, ICT, setups, scores, and subsequent price action**

Frozen database extraction: September 11, 2026, 22:02 ET. Last collected five-minute bar: September 11, 16:45 ET. All times in this report are New York time, including during overnight trading. The report uses collected ESU6 bars and saved runtime outputs only; historical training/backtest results are excluded from the performance evidence.

**Profitability assessment.** The collected evidence does **not establish a profitable live trading system**. There are zero FRVP paper-ledger events, zero ICT paper-ledger events, and no broker fills or position accounting in the audited ES tables. Nearly all records are shadow evaluations. A consistent hypothetical one-hour test loses money for both sets of retained setups after costs. This is an audit of observed signal quality, not a finding that a brokerage account actually lost the dollar amounts below.

| System | Resolved setups | Win % | Net points | Mean points | Profit factor | USD at one ES/event |
| --- | --- | --- | --- | --- | --- | --- |
| FRVP | 45 | 42.22 | -144.00 | -3.20 | 0.52 | -7,200.00 |
| ICT | 26 | 34.62 | -15.00 | -0.58 | 0.73 | -750.00 |

The dollar conversion assumes one ES contract independently for each event. ES is $50 per point and $12.50 per 0.25-point tick; these contract values match the [CME contract rule](https://www.cftc.gov/filings/orgrules/rule030416cmedcm003.pdf). Independent event sums can contain overlapping exposure and are not portfolio returns, account ROI, or realized P&L.

**What was collected and what can actually be audited.**

- 4,411 ES five-minute bars, 734 half-hour bars, and 366 hourly bars; all source contracts are ESU6. There are 22 calendar dates with any bars and 15 fully observed 09:30–16:00 sessions. Approximately 48.9% of scheduled open five-minute slots in the full July 28–September 11 span are present under the documented calendar assumptions.
- 35,967 model evaluations: 19,284 FRVP across 2,160 source timestamps; 16,683 ICT across 2,076 timestamps. There are 21 historical model IDs and 20 currently loaded models. Nine specialists began producing saved scores on September 3 at 16:15 ET.
- 47 recoverable FRVP setups: 16 long and 31 short. The stored policy contexts preserve type, side, and rule confidence; the latest 40 also agree with retained dashboard markings.
- 40 recoverable ICT setups: 23 long and 17 short. These are the dashboard's latest 40 retained events. All 16,683 ICT prediction contexts omit setup type/side/family. **Older ICT setup markings are not recoverable as a full historical event ledger from this snapshot.** The 40 are a rolling retained subset, not the full number ever detected.
- Every recovered setup has same-source-bar model scores: 772 setup/model associations. There are no duplicate model/timestamp rows or contradictory archived FRVP labels in the snapshot. Only 25 of these associations match a specialist's exact type and side.
- The raw-market-event table is empty. The local runtime archive has no additional historical ES event archive. We have saved canonical bars, selected features, predictions, decisions, and recent chart state, rather than a raw tick/order replay.

The largest collection hole spans August 3 at 23:35 through August 25 before the 18:00 reopen; another spans August 26 at 19:10 through August 27 before 18:00. Scheduled closures inside these spans are not counted as missing tradable bars. The price appendix documents 4,611 missing open-market slots, calendar assumptions, and the nine explicitly incomplete rows. Final 30-minute/hourly aggregates match all comparable complete five-minute constituents. See [coverage](price_daily_coverage.csv), [gap ledger](price_gaps.csv), and [price audit](price_findings.md).

**How subsequent outcomes were measured.**

The main comparison buys or sells at the **next observed five-minute bar's open after the setup bar**, holds for exactly 15, 30, 60, or 100 minutes, then exits at the last included bar's close. Each round trip subtracts **one ES point total** for all costs, an explicit common audit assumption rather than a measured fill cost. For example, a setup bar labeled 09:30 finishes at 09:35; the 60-minute test enters the 09:35 open and exits the close of the 10:30 bar, at 10:35. All required bars, including the setup bar, must be complete, contiguous, and on the same contract. Missing, incomplete, end-of-sample, and closure-crossing windows are censored, not assigned zero profit or counted as losses.

This common horizon compares direction and follow-through across setups. It is **not a reproduction of the trained labels or the frozen paper policies**: FRVP labels use setup-dependent target/stop behavior and define a positive as target hit; ICT labels include target hits or profitable timeouts with additional management rules. The frozen paper ledger instead uses signal-close entry and 120 future bars for FRVP long reversal, or 20 for ICT long meta, without brackets. Its modeled costs are 2.65–4.65 ticks per round trip. These are different estimands and no exact strategy-performance claim is made by substituting the one-hour test.

| strategy | horizon_minutes | n | net_points | mean_points |
| --- | --- | --- | --- | --- |
| FRVP | 15 | 47 | -52.75 | -1.12 |
| FRVP | 30 | 47 | -155.50 | -3.31 |
| FRVP | 60 | 45 | -144.00 | -3.20 |
| FRVP | 100 | 44 | -110.00 | -2.50 |
| ICT | 15 | 35 | -38.75 | -1.11 |
| ICT | 30 | 33 | -32.00 | -0.97 |
| ICT | 60 | 26 | -15.00 | -0.58 |
| ICT | 100 | 26 | 3.75 | 0.14 |

FRVP remains negative even with zero assumed costs: **−99 points over 45 one-hour outcomes**. ICT is **+11 points gross over 26**, but becomes −2 points at 0.5-point costs, −15 at one point, and −41 at two points. Its gross margin is only 0.42 point per event. The day-cluster bootstrap 95% interval for one-hour mean net outcome spans −9.01 to +2.28 points for FRVP and −2.92 to +1.63 for ICT. These small, dependent samples do not establish a stable edge or a statistically certain negative expectation.

Keeping only the first nonoverlapping one-hour setup within each strategy reduces FRVP to 27 outcomes/−21 points and ICT to 13/−18.25 points. Thus repeated overlapping FRVP marks magnify the event-sum loss, but removing overlap does not make either positive. This restriction is per strategy; FRVP and ICT may still overlap each other. It is not a combined single-position ES portfolio.

FRVP's 600-minute test is positive (+215.25 points), but only ten of 47 events have an uninterrupted complete ten-hour window. That selection excludes most intraday and closure-crossing events. It is not a rescue of the one-hour result, nor evidence that the frozen 120-bar policy is profitable. Likewise ICT's 100-minute +3.75 points is marginal and does not validate its trained trade-management logic. [Cost sensitivity](setup_cost_sensitivity.csv), [daily results](setup_daily_summary.csv), and [all event outcomes](setup_outcomes.csv) preserve the denominators.

**Setup-by-setup behavior.**

FRVP S1/S4/S6 are reversal setups: value-edge fade, failed auction/re-entry, and high-volume-node magnet. S2/S3/S5 are continuation setups: breakout retest, value-area volume breakout, and low-volume-node traverse. The retained ICT types are sweep reclaim and inverse-FVG reversal, plus premium/discount continuation. Rule confidence is a hand-coded detector score; it is separate from the model probability.

| System | Setup | Side | Resolved N | Net points | Mean points |
| --- | --- | --- | --- | --- | --- |
| FRVP | balanced profile value-edge fade | Short | 3 | -35.00 | -11.67 |
| FRVP | breakout acceptance/retest | Short | 1 | 6.00 | 6.00 |
| FRVP | breakout acceptance/retest | Long | 2 | -10.25 | -5.12 |
| FRVP | value-area volume breakout | Short | 8 | 2.75 | 0.34 |
| FRVP | value-area volume breakout | Long | 6 | -61.75 | -10.29 |
| FRVP | failed auction/reentry | Long | 1 | -5.50 | -5.50 |
| FRVP | low-volume-node traverse | Short | 9 | 14.00 | 1.56 |
| FRVP | low-volume-node traverse | Long | 3 | 16.00 | 5.33 |
| FRVP | high-volume-node magnet | Short | 9 | -58.25 | -6.47 |
| FRVP | high-volume-node magnet | Long | 3 | -12.00 | -4.00 |
| ICT | ifvg reversal | Short | 6 | -0.75 | -0.12 |
| ICT | ifvg reversal | Long | 9 | -26.25 | -2.92 |
| ICT | premium discount continuation | Short | 1 | 6.75 | 6.75 |
| ICT | premium discount continuation | Long | 1 | 0.75 | 0.75 |
| ICT | sweep reclaim | Short | 4 | -4.50 | -1.12 |
| ICT | sweep reclaim | Long | 5 | 9.00 | 1.80 |

The better FRVP group was **S5/LVN continuation**: +30 points across 12 resolved outcomes, with nine short outcomes contributing +14 and three long outcomes +16. This is a small positive pocket, not a validated rule. The largest recurring weaknesses were **S6/HVN magnet (−70.25 points across 12)** and **S1 short value-edge fades (−35 across three)**. Long S3 breakouts lost −61.75 across six, dominated by the July 31 failed breakout; the four matching S3-specialist observations from the later September period instead total +11.25. Mixing those model coverage eras would give a misleading comparison. Across all FRVP setups from specialist activation onward, the one-hour result improves to −11.75 points over 32 outcomes, still negative after the common cost.

ICT **long IFVG reversal** was the weakest retained group: −26.25 points across nine resolved outcomes with one net winner. Short IFVG outcomes were roughly flat before the common cost adjustment's impact on small moves: −0.75 points net across six. Long sweep reclaim produced +9 points across five, but one +14-point event accounts for more than the total. Premium/discount continuation has only two resolved one-hour examples, so its +7.5 points is insufficient for ranking.

There is also an exit-management question distinct from entry direction: 14 of 26 losing/nonpositive FRVP one-hour outcomes first offered at least three favorable points. Some setups found an initial move but gave it back; others failed with virtually no favorable excursion. ICT had four such cases among 17 losing/nonpositive outcomes. Favorable/adverse excursions in the event ledger show these paths; capturing the maximum favorable excursion is not assumed executable.

The 40 ICT marks contain archived entry/stop/target references. A separate, explicitly idealized replay enters at the saved reference entry, checks subsequent complete candles for the reference barriers, uses a 20-bar cap, handles opening gaps, and gives stop precedence if both barriers are touched inside the same bar. It resolves 32: 21 stop outcomes and 11 target outcomes, with eight censored. Gross total is −13.07 points; after the same one-point cost it is **−45.07 points**, with 7/32 net winners. Several tiny target hits are losers after costs. These are reference-geometry diagnostics, not reproduced ICT labeling-engine exits or recorded trades.

**Which hours performed best, and where was the noise?**

These buckets use the **setup bar's start hour**, with one-hour outcomes after the mark. Scores became available later. They combine both retained systems but do not combine them into a portfolio. Counts and days are essential:

| Mark hour ET | Resolved N | Days | Net points | Mean points | Win % |
| --- | --- | --- | --- | --- | --- |
| 7 | 2 | 2.00 | 23.25 | 11.62 | 50.00 |
| 8 | 5 | 4.00 | 16.75 | 3.35 | 60.00 |
| 9 | 7 | 5.00 | -54.00 | -7.71 | 57.14 |
| 10 | 6 | 4.00 | -53.25 | -8.88 | 33.33 |
| 11 | 5 | 2.00 | -55.75 | -11.15 | 0.00 |
| 18 | 11 | 6.00 | 1.50 | 0.14 | 45.45 |
| 19 | 7 | 5.00 | -10.25 | -1.46 | 28.57 |
| 20 | 6 | 5.00 | 1.75 | 0.29 | 33.33 |
| 21 | 4 | 3.00 | -12.25 | -3.06 | 25.00 |

The largest positive total is 07:00–07:59 (+23.25 points) but there are only two observations. Among hours with at least five resolved setups, **08:00–08:59 is strongest** (+16.75 points, five observations on four dates). The **09:00–11:59 marks are the loss concentration: −163 points across 18 outcomes**. The 11:00 hour has zero winners in five observations, but only two dates support that result. This is a retrospective diagnostic; it does not justify hard-coding an hourly filter without further prospective evidence.

Evening 18:00 and 20:00 totals are barely positive (+1.5 and +1.75 points respectively), with negative median outcomes. Those totals are not broad-based profitability. Hours 19:00 and 21:00 are negative. All 16 marks in the 16:00 hour lack a strict full one-hour outcome because of a closure or incomplete constituent, so that hour is **unresolved**, not unprofitable. The one-hour sample thus systematically excludes late-session setups; shorter horizons remain available for many.

**Family models, meta models, and setup specialists.**

Family models pool reversal or continuation events within a direction. Meta models pool both broad families in a direction. Specialists target one setup and direction. FRVP's six specialists are long-only; ICT's three specialists are short-only. There are no observed long S1 or long S2 specialist matches despite 1,409 scores from each. FRVP short reversal has only 31 early historical scores and none on the recovered setup events, so it cannot be ranked here.

For this audit, a score matches only when strategy, direction, broad family (or meta), and specialist type agree with the archived setup. The following table shows the actual probability at those matching marks and the subset crossing the recorded threshold. A model with no resolved pass has an unavailable net total, not a zero-return trading strategy.

| Model | Matching marks | Passes | 60m N | 60m net points | Mean p at all matches |
| --- | --- | --- | --- | --- | --- |
| frvp_long_continuation_setup3_xgb_v1 | 4 | 3 | 3 | 9.50 | 0.67 |
| frvp_long_continuation_setup5_xgb_v1 | 1 | 1 | 1 | -7.50 | 0.70 |
| frvp_long_continuation_xgb_v1 | 11 | 5 | 5 | -30.00 | 0.69 |
| frvp_long_meta_xgb_v1 | 16 | 12 | 12 | -57.00 | 0.46 |
| frvp_long_reversal_setup4_xgb_v1 | 2 | 0 | 0 | — | 0.19 |
| frvp_long_reversal_setup6_xgb_v1 | 3 | 2 | 2 | -6.75 | 0.62 |
| frvp_long_reversal_xgb_v1 | 5 | 2 | 2 | -1.00 | 0.57 |
| frvp_short_continuation_tcn_v1 | 19 | 10 | 10 | 12.75 | 0.50 |
| frvp_short_meta_xgb_v1 | 31 | 25 | 24 | -81.50 | 0.49 |
| ict_long_continuation_xgb_v1 | 3 | 3 | 1 | 0.75 | 0.68 |
| ict_long_meta_xgb_v1 | 23 | 2 | 2 | -13.00 | 0.34 |
| ict_long_reversal_xgb_v1 | 20 | 1 | 1 | -0.75 | 0.29 |
| ict_short_continuation_premium_discount_continuation_xgb_v1 | 2 | 1 | 1 | 6.75 | 0.44 |
| ict_short_continuation_xgb_v1 | 2 | 0 | 0 | — | 0.44 |
| ict_short_meta_xgb_v1 | 17 | 0 | 0 | — | 0.24 |
| ict_short_reversal_ifvg_reversal_xgb_v1 | 8 | 1 | 0 | — | 0.27 |
| ict_short_reversal_sweep_reclaim_xgb_v1 | 5 | 0 | 0 | — | 0.18 |
| ict_short_reversal_xgb_v1 | 15 | 0 | 0 | — | 0.23 |

FRVP **short continuation TCN** is the most useful positive family observation in this sample: its ten accepted matching marks sum to +12.75 points. But all 18 resolved matching continuation events sum to +22.75, and nine-of-nineteen marks were rejected; filtering reduced total profit and did not improve win rate (61.1% all resolved versus 50% accepted). The score's one-hour win-proxy AUC on all matching events is about 0.48, so this does not establish strong ranking power. FRVP long continuation accepts five marks and loses −30 points; it also rejects a +28-point S5 move just below threshold. Threshold placement matters at the boundary.

FRVP **meta models are permissive**: they accept 12/16 long and 25/31 short marks. Their accepted one-hour outcomes total −138.5 points across 36, nearly the unfiltered loss. FRVP short-meta probabilities at matches vary relatively little, and a useful short breakout can be rejected at 0.4989 versus a 0.5000 threshold while nearby accepted values convey little practical separation. These are observations about this collected sample, not proof that calibration is universally defective.

ICT **reversal models are mostly abstaining at actual marks**: long reversal accepts 1/20, short reversal 0/15; short meta accepts 0/17. Long meta accepts 2/23 and both resolved one-hour outcomes lose, totaling −13 points. On its 15 resolved matching examples, higher long-meta probability is negatively associated with one-hour return (Spearman approximately −0.65), and win-proxy AUC is 0.25. That is a concerning small-sample observation; it is not original-label calibration. Long continuation accepts all three relevant marks, but only one has a valid one-hour window.

Eight current models explicitly have no post-score calibrator; their stored `calibrated_probability` is exactly the raw probability. In particular, do not interpret the FRVP continuation scores around 0.70 as directly comparable with ICT reversal scores around 0.25. They target different events, have different prevalence and calibration methods, and the one-hour net-win outcome is different from their training label.

The six FRVP specialist passes produce **−4.75 points** under ideal next-open one-hour entries. The two ICT specialist passes have only one resolved one-hour outcome, +6.75; the other has a −4.5-point 30-minute outcome and is censored at one hour. These counts are far too small to promote or discard a specialist.

| Mark ET | Model | p | Threshold | 15m net | 60m net | 100m net | 60m after model available |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-04 09:00 | frvp_long_reversal_setup6_xgb_v1 | 0.64 | 0.62 | 2.50 | 4.50 | -12.25 | -9.25 |
| 2026-09-04 09:20 | frvp_long_reversal_setup6_xgb_v1 | 0.64 | 0.62 | 2.75 | -11.25 | -27.00 | -20.00 |
| 2026-09-07 07:10 | frvp_long_continuation_setup5_xgb_v1 | 0.70 | 0.65 | -5.00 | -7.50 | 0.50 | -3.25 |
| 2026-09-08 18:50 | ict_short_continuation_premium_discount_continuation_xgb_v1 | 0.47 | 0.44 | -1.25 | 6.75 | 2.50 | -1.25 |
| 2026-09-09 16:00 | ict_short_reversal_ifvg_reversal_xgb_v1 | 0.36 | 0.35 | -3.50 | — | — | — |
| 2026-09-11 01:40 | frvp_long_continuation_setup3_xgb_v1 | 0.68 | 0.65 | 5.00 | 5.25 | 9.00 | 4.00 |
| 2026-09-11 02:45 | frvp_long_continuation_setup3_xgb_v1 | 0.68 | 0.65 | 1.25 | -6.50 | -0.25 | -8.00 |
| 2026-09-11 08:45 | frvp_long_continuation_setup3_xgb_v1 | 0.67 | 0.65 | 20.25 | 10.75 | 18.00 | -6.00 |

Applying only the archived static regime/session/stress exclusions to FRVP's matching family passes changes −18.25 points across 17 to +17.25 across 11. This is a potentially useful selection clue, but one day contributes +47 points, more than the entire net gain. The exercise does not reproduce cooldown, expected-move/spread rules, order sizing, or an approved strategy policy. Specialist policy costs were not packaged in the manifests, and S4 uses a provisional direction-fallback policy; raw threshold comparisons are therefore the primary reproducible evidence here. [Unique filtered-event summaries](unique_setup_filter_summary.csv) and [matched probability bins](matched_probability_bins.csv) give the detailed results.

**Actual examples: what the mark and probability were followed by.**

- **July 31, 09:30 — FRVP S3 long:** rule confidence 0.88; long continuation p=0.6792 fails the 0.70 threshold, while long meta p=0.4819 passes 0.45. The ideal one-hour setup loses 57 points after costs, with just 2.75 favorable versus 84.5 adverse points. The family filter avoids this failed breakout; meta does not. Scores were recorded around 13:41 ET, so this is a catch-up evaluation, not an executable 09:35 decision.
- **September 10, 07:50 — FRVP S5 short:** TCN p=0.8022 versus a regime threshold of 0.62; short meta p=0.5015 versus 0.50. One-hour net +30.75, favorable excursion 39.75 and adverse excursion zero. This is one of the clearest directional-follow-through examples. A model-availability entry still gives a hypothetical +24.75.
- **September 10, 08:15 — FRVP S3 short:** TCN p=0.8639 passes 0.62; meta p=0.4989 misses 0.50. One-hour net +25.75, favorable excursion 30.5 and adverse 1.75. The TCN recognized this observed continuation better than the meta threshold did; the delayed-entry result drops to +8 points.
- **September 10, 20:00 — ICT sweep-reclaim long:** long meta p=0.3062 and long reversal p=0.3530 both fail 0.40. One-hour net +14, favorable 16 and adverse 0.25. A strong retained move was filtered out, although the sample does not establish that thresholds should be lowered.
- **September 11, 08:45 — FRVP S3 long specialist:** p=0.6727 passes 0.65. It offers +20.25 at 15 minutes and +10.75 at one hour under ideal timing, but the model-availability one-hour replay is −6. The model score identified a historical move whose later entry behaved differently.

**Price action alone: the largest moves and most active hours.**

The largest directional five-minute candle is September 4, 08:30–08:35: −22.75 points open-to-close. The widest five-minute candle is September 11, 08:30–08:35: 34.75 points high-to-low, but only −10.5 net. This difference illustrates why volatility alone is not a directional trading edge.

The largest 15-minute directional window is July 29, 14:35–14:50, +51 points. The largest 30-minute window is July 29, 15:15–15:45, −77 points. The largest hour is July 29, 15:00–16:00, **−125 points**, following a +64-point hour from 13:55–14:55. This was violent reversal behavior rather than a smooth one-way session. The full cash session had a 137-point range and −106-point net move.

| Start ET | End ET | Net points | Range points | Source bars with scores | Retained setups (-15m to end) |
| --- | --- | --- | --- | --- | --- |
| 2026-07-29 15:00 | 2026-07-29 16:00 | -125.00 | 136.25 | 0 | 0 |
| 2026-07-29 13:55 | 2026-07-29 14:55 | 64.00 | 71.75 | 0 | 0 |
| 2026-07-31 09:35 | 2026-07-31 10:35 | -56.00 | 87.25 | 12 | 3 |
| 2026-07-29 09:45 | 2026-07-29 10:45 | -55.50 | 66.00 | 0 | 0 |
| 2026-08-28 11:30 | 2026-08-28 12:30 | -54.00 | 59.50 | 0 | 0 |
| 2026-08-03 09:30 | 2026-08-03 10:30 | 51.25 | 57.25 | 11 | 0 |
| 2026-09-03 10:35 | 2026-09-03 11:35 | 50.50 | 58.50 | 0 | 0 |
| 2026-07-29 12:15 | 2026-07-29 13:15 | 46.00 | 50.25 | 0 | 0 |
| 2026-09-10 07:45 | 2026-09-10 08:45 | -42.00 | 47.00 | 12 | 3 |
| 2026-07-31 11:30 | 2026-07-31 12:30 | 41.50 | 46.75 | 12 | 0 |
| 2026-07-30 09:00 | 2026-07-30 10:00 | 41.00 | 42.25 | 0 | 0 |
| 2026-07-30 11:30 | 2026-07-30 12:30 | 40.25 | 46.25 | 0 | 0 |

These windows are greedily selected without overlapping one-hour episodes, so the same shock is not repeatedly counted. Score coverage is by source timestamp, not score availability. Retained-setup counts include the prior 15 minutes and the episode itself; a missing ICT mark can mean lost retention, not necessarily no setup detection. **The July 29 largest moves predate the first saved ES model scores**, so the archive cannot show how those models would have handled them. Other large bars were backfilled with incomplete model coverage.

Hourly opportunity below uses exactly twelve complete contiguous bars per clock hour, and reports average range per observed hour rather than activity totals across uneven coverage:

| hour_et | observed_complete_hours | mean_range_points | median_range_points | mean_abs_net_points | mean_range_ex_july29 |
| --- | --- | --- | --- | --- | --- |
| 10 | 16 | 30.12 | 28.25 | 15.98 | 29.02 |
| 9 | 16 | 26.19 | 23.38 | 14.52 | 25.08 |
| 15 | 15 | 23.48 | 13.25 | 14.33 | 15.43 |
| 11 | 16 | 22.58 | 19.00 | 11.86 | 22.30 |
| 8 | 16 | 21.84 | 18.75 | 13.27 | 21.57 |
| 14 | 15 | 17.85 | 13.00 | 9.90 | 14.27 |
| 16 | 9 | 17.75 | 15.00 | 8.00 | 15.72 |
| 12 | 16 | 17.45 | 15.25 | 8.36 | 16.43 |

**10:00–10:59 has the highest typical price opportunity**, averaging 30.12 points of range across 16 observed hours; 09:00 is next at 26.19. These are also hours where the retained setup strategy lost, so the issue is selection/direction/management rather than simply a lack of movement. The 15:00 mean is inflated by July 29: excluding that day cuts its range average from 23.48 to 15.43 points. Noon and several overnight hours have low directional efficiency: 12:00 has 62.5% of observed hours below the defined 0.25 close-path-efficiency threshold; 03:00 has 68.75%. See [price charts and full statistics](price_findings.md).

**Operational/model issues that materially limit confidence.**

1. **Threshold-crossing noise:** FRVP specialists produced 3,306 threshold passes, but only six occurred on that specialist's matching setup and side. The other 3,300 (99.82%) were off-setup shadow scores. Shadow processing bypasses setup forced holds, abstention, and cooldown. They were not 3,300 unintended trades. Pooled families are not setup-gated either. The analytics must distinguish an all-bar score from a qualified setup decision.
2. **Delayed availability and revised inputs:** 35,805/35,967 predictions explicitly identify a delayed feed. Median record time is 11.93 minutes after nominal bar close; 95th percentile 74.11 minutes includes catch-up processing. 25,591 contexts have `ibkr_is_complete=False`, and the model-context close differs from the final canonical close on 21,439 rows (59.6%), by as much as 13.25 points. A final chart is not the exact bar version the model saw. The six FRVP specialist passes fall from −4.75 ideal-timing points to **−42.5** when entering at the first collected bar open at or after each model's recording time and holding a fresh hour. These remain hypothetical executions, but they expose the timing sensitivity.
3. **Confirmed ICT feature-prefix inconsistency:** at September 11 08:00, recomputing with only the 4,306 bars available through that row gives null prior-RTH references. Append bars through 11:00 and recompute that same 08:00 row: prior-RTH high/low/close become 7620/7585.5/7598.75, and the future 09:30 open 7665 appears in `ict_rth_open`. Known prior references drop out on 1,195/1,453 non-RTH model timestamps. The defect comes from mapping reference tables onto a next-session key before that RTH session exists, not from an incorrect NY timezone flag. This creates both live/offline mismatch and a demonstrated future-information hazard in full-frame feature generation. It does **not yet prove every deployed model was trained on leaked values**. [Reproduction evidence](ict_reference_prefix_parity.json), [reproduction script](check_ict_reference_prefix_parity.py), and [source analysis](model_semantics.md) preserve the finding.
4. **Audit retention and versioning:** only the latest 40 ICT events persist, with no per-event observation timestamp, detector hash, or complete revision history. An already appended chart mark is not necessarily removed when an incomplete bar revises. Lack of conflicting saved labels is not evidence of repaint-free behavior. Missingness in some other features can be legitimate event absence; it needs a training/live comparison rather than blanket imputation.

The two historical `emit` records are both at July 30 16:55: FRVP short reversal p=0.5929 versus 0.50 and ICT short continuation p=0.5592 versus 0.55. They were recorded about 17:55 and have no corresponding paper-ledger or broker-fill records. The current captured heartbeat has 20 shadow models, zero active models, delayed test mode, and both ledgers disabled. Neither the account's paper setting nor a manifest promotion note establishes executed P&L.

**What the evidence supports doing next.** First fix and causally test the ICT prior-session/reference construction while keeping the current-session open unavailable until it occurs. Persist immutable setup events, bar versions, observation timestamps, matched gate decisions, and later outcomes for both strategies. Evaluate shadow execution through the same setup, regime, spread, and cooldown gates as intended deployment, while keeping raw research scores available separately. Then collect a prospective sample with timely prices and an explicit position/exit/cost contract. Recheck S5 continuation, the short-continuation TCN, and the weak reversal groups on that frozen prospective contract; do not promote an hour or specialist on the tiny favorable pockets here.

**Audit artifacts and reproducibility.** [All 87 setup outcomes](setup_outcomes.csv), [all 772 same-bar setup/model probabilities](setup_model_probabilities.csv), [matched model outcomes](matched_model_setup_outcomes.csv), [all 35,967 prediction diagnostics](prediction_outcomes.csv), [setup extraction notes](setup_findings.md), [model semantics](model_semantics.md), and [price analysis](price_findings.md) are included. The standalone HTML adds a searchable per-setup explorer and candle chart. Re-run `model_outcomes.py`, `setup_audit.py`, `setup_outcomes.py`, `price_audit.py`, and `build_report.py` with the project virtual environment against the frozen snapshot; `extract_snapshot.py` refuses to overwrite an existing cutoff. No live settings, services, model artifacts, orders, or strategy code were changed.

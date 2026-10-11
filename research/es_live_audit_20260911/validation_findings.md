# Independent validation of ES outcome analysis

Reviewed model_outcomes.py and setup_outcomes.py and independently recalculated from the read-only frozen SQLite canonical_bars table. Checked all 87 setup events at five horizons, 81 predictions spanning all 21 models, all forty ICT brackets, all 187 matched setup/model joins, and 40 per-strategy nonoverlapping selections. The exported bar OHLCV values were checked against SQLite for every bar. Result: 0 failed checks across 27761 comparisons.

**Arithmetic and indexing.** A source bar stamped t is followed by entry at the next bar's open, t+5 minutes. Twelve future bars span that entry through the close of the bar stamped t+60, an exit at t+65: exactly 60 minutes held. Long and short signs, gross/net subtraction, future-only MFE/MAE, and exit timestamps are correct. Censoring requires all source-through-exit bars to be present, complete, and on the same contract. No test window bridges a missing five-minute slot or scheduled session break. Requiring a complete source bar is conservative even when entry is next open.

**Brackets.** The ICT branch uses the archived setup entry/stop/target references, future bars only, a twenty-bar maximum, one-point assumed round-trip cost, adverse opening-gap fills for stops, and target-level fills for favorable target gaps. Intrabar stop/target ambiguity is assigned to the stop. The forty cases independently match the exported branch classifications and numbers. This is a hypothetical setup-geometry study; those reference prices are not evidence of submitted or filled orders.

**Important interpretation limits; no root-script arithmetic correction is required.**

- The main setup totals add independent marked opportunities. Simultaneous and overlapping signals remain in those totals. The optional nonoverlapping file removes overlap separately inside FRVP and ICT, so combining both still does not establish a single ES position book. Model totals also repeat the same market episode across related models; they should never be added as portfolio P&L.
- Source-bar next-open and bracket-close studies use market prices before many archived predictions were actually recorded. Availability-adjusted setup results are a useful sensitivity, but ICT uses the first same-bar model timestamp as a proxy because the dashboard does not store the exact setup observed-at time. Following this review, the matched_model_setup_outcomes calculation was refined to use each selected model's own recorded_at. All 187 model/setup availability windows independently agree with direct SQLite calculations, and no invalid window retains a stale net result.
- Model all-bar diagnostics use an assumed source-session round-trip spread schedule, with fixed slippage/commission; they do not change the exit leg when a horizon crosses into another session. The specialist manifests have null/empty cost assumptions, so the schedule is an audit assumption for those models. Setup comparisons and brackets consistently use a separately stated one-point round-trip cost and have cost-sensitivity output. None of these figures establishes broker commissions or realized fills.
- Threshold_and_static_policy applies the archived static regime/session/stress blocks. It excludes neither the original stateful cooldown nor dynamic candidate-quantile/expected-move checks. Its name accurately describes a partial-policy sensitivity, not a full live-policy backtest. In these manifests probability-quantile thresholds are null; expected_move is not among archived context fields used by this audit.
- Summary days and hourly buckets are source-bar New York calendar dates/hours, not futures trade-date sessions or actual delayed execution hours. Censored late-afternoon rows make the valid-hours sample uneven. In particular all sixteen setup marks in the 16:00 hour lack uninterrupted sixty-minute windows; no return ranking can be inferred for that hour.
- The 600-minute FRVP sample is only ten uncensored observations, versus forty-five at sixty minutes. Positive long-horizon results must not be described as the same sample becoming profitable. Confidence intervals resample the available calendar days and remain small-sample descriptive uncertainty, not proof of independence or a validated edge.
- ROC AUC is correctly named auc_net_win_proxy: the target is this audit's next-open/net markout win, which differs from models' training labels. Probability bins and threshold selectivity are descriptive; these are not calibration tests against the original model target.

**Direct SQLite spot checks**

| Source time ET | Strategy/setup/side | Next open | 60m exit close | Gross points | Net points at 1pt cost | Bracket status |
|---|---|---:|---:|---:|---:|---|
| 2026-07-31T09:30:00-04:00 | FRVP 3 1 | 7512.0000 | 7456.0000 | -56.0000 | -57.0000 |  |
| 2026-07-31T10:00:00-04:00 | FRVP 5 -1 | 7454.2500 | 7476.7500 | -22.5000 | -23.5000 |  |
| 2026-09-02T09:25:00-04:00 | FRVP 5 1 | 7650.5000 | 7679.5000 | 29.0000 | 28.0000 |  |
| 2026-09-10T07:50:00-04:00 | FRVP 5 -1 | 7637.5000 | 7605.7500 | 31.7500 | 30.7500 |  |

Reproduction: run validate_outcomes.py from the audit directory or repository root. Details are in validation_summary.json. Validation wrote only new audit files and did not alter either outcome script, the snapshot, or runtime source.

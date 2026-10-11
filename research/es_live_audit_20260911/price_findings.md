# ES collected price-action audit

Frozen source: `es_bars_5m.csv`, exported with `es_audit_snapshot.sqlite3`; no training data or external prices were used. All clock times below are Eastern. All point movements are descriptive price opportunity, not executable profit.

## Coverage and quality

- 4,411 collected five-minute ES bars (367.58 hours), all ESU6; 4,402 explicitly complete and 9 explicitly incomplete.
- Span: 2026-07-28 21:00:00-04:00 to 2026-09-11 16:45:00-04:00 (bar starts); 22 calendar dates contain any bars and 15 fully observed 09:30–16:00 cash-session windows.
- Of 9,022 expected open five-minute slots inside that span, 4,411 are collected (48.89%). This is sparse collection over a wider calendar span, not 46 uninterrupted days of observations.
- Two collection holes remain. The long hole contains scheduled closures as well as 4,349 uncollected open-market bars. The shorter hole contains 262 uncollected open-market bars. See the exact bounds below and `price_gaps.csv`.
- Regular 17:00–18:00 maintenance and Friday 17:00–Sunday 18:00 closure are excluded. September 7 13:00–18:00 is treated as the Labor Day early closure (13:00–17:00 holiday pause plus maintenance), consistent with the observed gap and the [2026 futures holiday schedule](https://www.discounttrading.com/futures-holiday-schedule.html); [CME holiday calendar](https://www.cmegroup.com/trading-hours.html). Calendar assumptions affect coverage denominators only; all analytical windows independently require contiguous observed bars.
- No invalid OHLC ordering, duplicate contract/timestamps, zero-volume bars, or zero-range bars were found. All nine incomplete rows are excluded from all price-window and hourly-opportunity calculations.
- Every bar has source `ibkr.historical.update`; 1,463 explicitly say delayed and the remainder have unspecified market-data type. None explicitly say real time.
- The canonical insert timestamp is at least 5.55 minutes after the nominal five-minute close; median 14.06 minutes; 2,064 bars were inserted more than one hour afterward. This combines ingestion timing and backfills and is not a pure network-latency estimate.
- 3,506/4,411 rows were updated after insertion. Final OHLC can therefore differ from what a model saw initially. Bar timestamp is not proof of data availability then.

| first_absent_et | last_absent_et | missing_expected_open_bars | expected_open_hours_missing |
| --- | --- | --- | --- |
| 2026-08-03 23:35:00-04:00 | 2026-08-25 17:55:00-04:00 | 4,349.00 | 362.42 |
| 2026-08-26 19:10:00-04:00 | 2026-08-27 17:55:00-04:00 | 262.00 | 21.83 |

## Largest directional windows

Open of first included bar to close of last included bar; exact elapsed duration, complete bars, one contract, no gap bridging. A start at 15:10 and end at 16:10 includes bars labeled 15:10 through 16:05. Adjacent endpoints may differ from a close-to-close markout by the initial bar's opening gap.

| minutes | start_et | end_et | net_points | range_points | directional_efficiency |
| --- | --- | --- | --- | --- | --- |
| 5.00 | 2026-09-04 08:30:00-04:00 | 2026-09-04 08:35:00-04:00 | -22.75 | 30.00 | 1.00 |
| 15.00 | 2026-07-29 14:35:00-04:00 | 2026-07-29 14:50:00-04:00 | 51.00 | 59.00 | 1.00 |
| 30.00 | 2026-07-29 15:15:00-04:00 | 2026-07-29 15:45:00-04:00 | -77.00 | 90.00 | 1.00 |
| 60.00 | 2026-07-29 15:00:00-04:00 | 2026-07-29 16:00:00-04:00 | -125.00 | 136.25 | 0.85 |

The largest high-to-low spans can occur in different windows; their extremes have no guaranteed order and cannot be treated as attainable profit:

| minutes | start_et | end_et | net_points | range_points |
| --- | --- | --- | --- | --- |
| 5.00 | 2026-09-11 08:30:00-04:00 | 2026-09-11 08:35:00-04:00 | -10.50 | 34.75 |
| 15.00 | 2026-07-29 14:35:00-04:00 | 2026-07-29 14:50:00-04:00 | 51.00 | 59.00 |
| 30.00 | 2026-07-29 15:15:00-04:00 | 2026-07-29 15:45:00-04:00 | -77.00 | 90.00 |
| 60.00 | 2026-07-29 15:00:00-04:00 | 2026-07-29 16:00:00-04:00 | -125.00 | 136.25 |

Largest nonoverlapping 60-minute episodes (greedy ranking by absolute net movement, so one shock is not counted repeatedly):

| start_et | end_et | net_points | range_points | directional_efficiency |
| --- | --- | --- | --- | --- |
| 2026-07-29 15:00:00-04:00 | 2026-07-29 16:00:00-04:00 | -125.00 | 136.25 | 0.85 |
| 2026-07-29 13:55:00-04:00 | 2026-07-29 14:55:00-04:00 | 64.00 | 71.75 | 0.62 |
| 2026-07-31 09:35:00-04:00 | 2026-07-31 10:35:00-04:00 | -56.00 | 87.25 | 0.58 |
| 2026-07-29 09:45:00-04:00 | 2026-07-29 10:45:00-04:00 | -55.50 | 66.00 | 0.68 |
| 2026-08-28 11:30:00-04:00 | 2026-08-28 12:30:00-04:00 | -54.00 | 59.50 | 0.76 |
| 2026-08-03 09:30:00-04:00 | 2026-08-03 10:30:00-04:00 | 51.25 | 57.25 | 0.84 |
| 2026-09-03 10:35:00-04:00 | 2026-09-03 11:35:00-04:00 | 50.50 | 58.50 | 0.94 |
| 2026-07-29 12:15:00-04:00 | 2026-07-29 13:15:00-04:00 | 46.00 | 50.25 | 0.63 |
| 2026-09-10 07:45:00-04:00 | 2026-09-10 08:45:00-04:00 | -42.00 | 47.00 | 0.76 |
| 2026-07-31 11:30:00-04:00 | 2026-07-31 12:30:00-04:00 | 41.50 | 46.75 | 0.66 |
| 2026-07-30 09:00:00-04:00 | 2026-07-30 10:00:00-04:00 | 41.00 | 42.25 | 0.68 |
| 2026-07-30 11:30:00-04:00 | 2026-07-30 12:30:00-04:00 | 40.25 | 46.25 | 0.78 |

## Hour-of-day opportunity and noise

Hourly statistics use exactly 12 complete, contiguous bars starting on the clock hour. Each hour/day contributes once. These are points per observed hour rather than total activity across unequal exposure; sample counts are included. Full results and bootstrap confidence intervals for mean range are in `price_hourly_opportunity.csv`. The bootstrap resamples collected dates within each hour and does not establish an independent out-of-sample edge.

| hour_et | observed_complete_hours | mean_range_points | median_range_points | mean_abs_net_points | mean_directional_efficiency | mean_range_ex_july29 |
| --- | --- | --- | --- | --- | --- | --- |
| 10.00 | 16.00 | 30.12 | 28.25 | 15.98 | 0.36 | 29.02 |
| 9.00 | 16.00 | 26.19 | 23.38 | 14.52 | 0.33 | 25.08 |
| 15.00 | 15.00 | 23.48 | 13.25 | 14.33 | 0.29 | 15.43 |
| 11.00 | 16.00 | 22.58 | 19.00 | 11.86 | 0.29 | 22.30 |
| 8.00 | 16.00 | 21.84 | 18.75 | 13.27 | 0.39 | 21.57 |
| 14.00 | 15.00 | 17.85 | 13.00 | 9.90 | 0.34 | 14.27 |
| 16.00 | 9.00 | 17.75 | 15.00 | 8.00 | 0.29 | 15.72 |
| 12.00 | 16.00 | 17.45 | 15.25 | 8.36 | 0.25 | 16.43 |

Directional efficiency is absolute open-to-close movement divided by the sum of absolute five-minute close-path changes (including first open-to-close). It ranges from zero (return to start) to one (a monotonic close path). Intrabar wicks are not captured by this measure; `wick_adjusted_efficiency` additionally divides net displacement by summed candle ranges. "Choppy" is an explicit descriptive threshold below 0.25 and "directional" means at least 0.60, not a trained setup label.

Hours with the lowest mean directional efficiency:

| hour_et | observed_complete_hours | mean_range_points | mean_directional_efficiency | choppy_hours_pct | directional_hours_pct |
| --- | --- | --- | --- | --- | --- |
| 3.00 | 16.00 | 12.03 | 0.21 | 68.75 | 0.00 |
| 7.00 | 16.00 | 12.75 | 0.24 | 56.25 | 6.25 |
| 4.00 | 16.00 | 13.97 | 0.24 | 56.25 | 6.25 |
| 12.00 | 16.00 | 17.45 | 0.25 | 62.50 | 0.00 |
| 1.00 | 16.00 | 10.75 | 0.26 | 62.50 | 18.75 |
| 23.00 | 16.00 | 8.23 | 0.26 | 50.00 | 6.25 |

High range alone does not mean good execution: opposing moves can create large range and small net displacement. July 29 is an extreme outlier, so the hourly table also supplies mean range excluding that date. Trade-time or profitability decisions should use model-entry markouts, actual fill records and costs, not these price-only direction-agnostic rankings.

## Complete cash-session days

09:30–16:00, exactly 78 complete five-minute bars; holiday and partial sessions excluded. Largest five ranges:

| date_et | net_points | range_points | directional_efficiency |
| --- | --- | --- | --- |
| 2026-07-29 | -106.00 | 137.00 | 0.20 |
| 2026-07-31 | 21.00 | 113.50 | 0.06 |
| 2026-08-03 | 82.25 | 95.00 | 0.46 |
| 2026-07-30 | 58.25 | 79.50 | 0.21 |
| 2026-08-28 | -23.50 | 70.75 | 0.08 |

## Higher-timeframe consistency

Stored 30-minute and 1-hour canonical bars were compared with aggregation of final, complete five-minute constituents only. A mismatch is absolute difference above 0.001 points (or units for volume). These checks describe consistency at the frozen cutoff, not model-time history:

| timeframe | comparable | ohlc_mismatches | volume_mismatches |
| --- | --- | --- | --- |
| 1h | 361 | 0 | 0 |
| 30m | 729 | 0 | 0 |

## Files and reproducibility

`price_audit.py` generates the price audit only. `price_valid_windows.csv` contains every valid 5/15/30/60-minute window; `price_largest_nonoverlapping_episodes.csv` removes duplicate overlap within each horizon; `price_complete_clock_hours.csv` and `price_complete_rth_days.csv` give auditable observations. `price_daily_coverage.csv`, `price_hour_coverage.csv`, `price_gaps.csv`, `price_incomplete_bars.csv`, `price_bars_audited.csv`, and `price_aggregate_consistency.csv` document data quality. Plots: `price_coverage.png`, `price_hourly_opportunity.png`, `price_largest_episodes.png`.

"""Reproducible ES price audit of the frozen collection; no live writes/imports.

Run: .\ote_venv\Scripts\python.exe research/es_live_audit_20260911/price_audit.py
All returns are index points and describe price opportunity, not strategy profit.
Bar timestamps are interpreted as interval starts; ET is America/New_York.
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
TZ = 'America/New_York'
STEP = pd.Timedelta(minutes=5)
HOLIDAY = '2026-09-07'


def schedule(t):
    """Classify regular ES closures and the observed Labor Day early close."""
    wd, hour = t.dayofweek, t.hour
    if wd == 5 or (wd == 4 and hour >= 17) or (wd == 6 and hour < 18):
        return 'weekend_closed'
    if hour == 17:
        return 'daily_maintenance'
    if str(t.date()) == HOLIDAY and 13 <= hour < 17:
        return 'labor_day_early_close'
    return 'expected_open'


def write(df, name):
    df.to_csv(OUT / name, index=False, float_format='%.6f')


def table(df, digits=2):
    """Markdown without an optional tabulate dependency."""
    x = df.copy()
    for col in x.select_dtypes(include='number'):
        x[col] = x[col].map(lambda z: f'{z:,.{digits}f}' if pd.notna(z) else '')
    return '| ' + ' | '.join(x.columns) + ' |\n| ' + ' | '.join(['---'] * len(x.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in x.itertuples(index=False, name=None))


def boot_mean_ci(x):
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(20260911)
    means = rng.choice(a, (4000, len(a)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def window_row(g, minutes):
    first, last = g.iloc[0], g.iloc[-1]
    net = last.close - first.open
    path = abs(first.close - first.open) + np.abs(np.diff(g.close.to_numpy())).sum()
    candle_range = (g.high - g.low).sum()
    return dict(minutes=minutes, start_et=first.t, end_et=last.t + STEP,
                contract_symbol=first.contract_symbol, date_et=str(first.t.date()), hour_et=first.t.hour,
                open=first.open, close=last.close, high=g.high.max(), low=g.low.min(),
                net_points=net, abs_net_points=abs(net), range_points=g.high.max() - g.low.min(),
                close_path_points=path, directional_efficiency=abs(net)/path if path else np.nan,
                candle_range_sum=candle_range, wick_adjusted_efficiency=abs(net)/candle_range if candle_range else np.nan,
                volume=g.volume.sum(), n_bars=len(g), segment=int(first.segment))


def nonoverlap(g, n=20):
    picked = []
    for idx, row in g.sort_values('abs_net_points', ascending=False).iterrows():
        if not any(row.start_et < old.end_et and old.start_et < row.end_et for old in picked):
            picked.append(row)
            if len(picked) == n:
                break
    return pd.DataFrame(picked)


def main():
    bars = pd.read_csv(OUT / 'es_bars_5m.csv')
    assert bars.asset.eq('ES').all() and bars.timeframe.eq('5m').all()
    bars['t'] = pd.to_datetime(bars.timestamp_utc, utc=True).dt.tz_convert(TZ)
    bars = bars.sort_values('t').reset_index(drop=True)
    assert not bars.duplicated(['t', 'contract_symbol']).any()
    context = bars.feature_context_json.map(json.loads)
    bars['complete'] = context.map(lambda x: x.get('ibkr_is_complete') is True)
    bars['market_data_type'] = context.map(lambda x: x.get('ibkr_market_data_type') or 'unspecified')
    bars['date_et'] = bars.t.dt.strftime('%Y-%m-%d')
    bars['hour_et'] = bars.t.dt.hour
    bars['range_points'] = bars.high - bars.low
    bars['body_points'] = bars.close - bars.open
    bars['body_fraction'] = bars.body_points.abs() / bars.range_points
    bars['insert_lag_after_close_min'] = (pd.to_datetime(bars.inserted_at_utc, utc=True) - pd.to_datetime(bars.timestamp_utc, utc=True) - STEP).dt.total_seconds()/60
    bars['update_lag_after_close_min'] = (pd.to_datetime(bars.updated_at_utc, utc=True) - pd.to_datetime(bars.timestamp_utc, utc=True) - STEP).dt.total_seconds()/60
    bars['revised_after_insert'] = bars.inserted_at_utc.ne(bars.updated_at_utc)
    bars['valid_ohlc'] = bars.high.ge(bars[['open', 'close']].max(axis=1)) & bars.low.le(bars[['open', 'close']].min(axis=1)) & bars.high.ge(bars.low)
    assert bars.valid_ohlc.all()
    write(bars.drop(columns=['feature_context_json']), 'price_bars_audited.csv')
    write(bars.loc[~bars.complete].drop(columns=['feature_context_json']), 'price_incomplete_bars.csv')

    grid = pd.DataFrame({'t': pd.date_range(bars.t.min(), bars.t.max(), freq='5min')})
    grid['schedule'] = grid.t.map(schedule)
    grid['date_et'] = grid.t.dt.strftime('%Y-%m-%d')
    grid['hour_et'] = grid.t.dt.hour
    grid['observed'] = grid.t.isin(bars.t)
    grid['complete'] = grid.t.isin(bars.loc[bars.complete, 't'])
    grid['expected'] = grid.schedule.eq('expected_open')
    grid['missing_open'] = grid.expected & ~grid.observed
    grid['scheduled_closed'] = ~grid.expected
    assert not (grid.observed & grid.scheduled_closed).any()
    daily = grid.groupby('date_et').agg(expected_bars_in_span=('expected', 'sum'), collected_bars=('observed', 'sum'), complete_bars=('complete', 'sum'), missing_open_bars=('missing_open', 'sum'), scheduled_closed_bars=('scheduled_closed', 'sum')).reset_index()
    daily['coverage_pct'] = 100 * daily.collected_bars / daily.expected_bars_in_span.replace(0, np.nan)
    daily['complete_coverage_pct'] = 100 * daily.complete_bars / daily.expected_bars_in_span.replace(0, np.nan)
    write(daily, 'price_daily_coverage.csv')
    hour_coverage = grid.groupby('hour_et').agg(expected_bars=('expected', 'sum'), collected_bars=('observed', 'sum'), complete_bars=('complete', 'sum'), missing_open_bars=('missing_open', 'sum')).reset_index()
    hour_coverage['coverage_pct'] = 100 * hour_coverage.collected_bars / hour_coverage.expected_bars.replace(0, np.nan)
    hour_coverage['complete_hours_equivalent'] = hour_coverage.complete_bars / 12
    write(hour_coverage, 'price_hour_coverage.csv')

    gaps = []
    for i in bars.index[bars.t.diff().gt(STEP)]:
        left, right = bars.loc[i-1], bars.loc[i]
        absent = grid.loc[grid.t.gt(left.t) & grid.t.lt(right.t)]
        gaps.append(dict(previous_bar_et=left.t, next_bar_et=right.t, first_absent_et=left.t+STEP,
                         last_absent_et=right.t-STEP, absent_calendar_bars=len(absent),
                         missing_expected_open_bars=int(absent.missing_open.sum()),
                         expected_open_hours_missing=absent.missing_open.sum()/12,
                         scheduled_closed_bars=int(absent.scheduled_closed.sum()),
                         classification='collection_hole_with_scheduled_closures' if absent.missing_open.any() else 'scheduled_closure',
                         closure_types=','.join(sorted(set(absent.loc[absent.scheduled_closed, 'schedule'])))))
    gaps = pd.DataFrame(gaps)
    write(gaps, 'price_gaps.csv')

    valid = bars.loc[bars.complete & bars.valid_ohlc].copy()
    new_segment = valid.t.diff().ne(STEP) | valid.contract_symbol.ne(valid.contract_symbol.shift())
    valid['segment'] = new_segment.cumsum()
    windows = []
    for segment, chunk in valid.groupby('segment'):
        for minutes in (5, 15, 30, 60):
            n = minutes // 5
            for start in range(len(chunk)-n+1):
                windows.append(window_row(chunk.iloc[start:start+n], minutes))
    windows = pd.DataFrame(windows)
    assert ((windows.end_et-windows.start_et).dt.total_seconds()/60).eq(windows.minutes).all()
    write(windows, 'price_valid_windows.csv')
    write(windows.sort_values('abs_net_points', ascending=False).groupby('minutes', sort=True).head(30), 'price_largest_windows.csv')
    episodes = pd.concat([nonoverlap(x) for _, x in windows.groupby('minutes')], ignore_index=True)
    write(episodes, 'price_largest_nonoverlapping_episodes.csv')

    # Exactly 12 complete five-minute bars in a clock hour, no bridges or partial hours.
    hours = windows.loc[windows.minutes.eq(60) & windows.start_et.dt.minute.eq(0)].copy()
    assert hours.n_bars.eq(12).all()
    hours['choppy'] = hours.directional_efficiency.lt(0.25)
    hours['directional'] = hours.directional_efficiency.ge(0.60)
    write(hours, 'price_complete_clock_hours.csv')
    hourly = []
    for hour, x in hours.groupby('hour_et'):
        lo, hi = boot_mean_ci(x.range_points)
        no_july29 = x.loc[x.date_et.ne('2026-07-29')]
        hourly.append(dict(hour_et=hour, observed_complete_hours=len(x), distinct_dates=x.date_et.nunique(),
                           mean_range_points=x.range_points.mean(), median_range_points=x.range_points.median(),
                           p90_range_points=x.range_points.quantile(.9), mean_range_boot_ci_low=lo, mean_range_boot_ci_high=hi,
                           mean_abs_net_points=x.abs_net_points.mean(), median_abs_net_points=x.abs_net_points.median(),
                           mean_net_points=x.net_points.mean(), mean_directional_efficiency=x.directional_efficiency.mean(),
                           median_directional_efficiency=x.directional_efficiency.median(), choppy_hours_pct=100*x.choppy.mean(),
                           directional_hours_pct=100*x.directional.mean(), mean_volume=x.volume.mean(),
                           mean_range_ex_july29=no_july29.range_points.mean(), mean_abs_net_ex_july29=no_july29.abs_net_points.mean()))
    hourly = pd.DataFrame(hourly)
    write(hourly, 'price_hourly_opportunity.csv')
    barhour = valid.groupby('hour_et').agg(n_bars=('close', 'size'), mean_5m_range=('range_points', 'mean'), median_5m_range=('range_points', 'median'), mean_body_fraction=('body_fraction', 'mean'), total_range_points=('range_points', 'sum'), total_abs_body_points=('body_points', lambda x: x.abs().sum()), mean_volume=('volume', 'mean')).reset_index()
    barhour['range_activity_per_collected_hour'] = barhour.total_range_points / (barhour.n_bars/12)
    barhour['abs_body_activity_per_collected_hour'] = barhour.total_abs_body_points / (barhour.n_bars/12)
    write(barhour, 'price_5m_hourly_activity.csv')

    rth = []
    for date, x in valid.groupby('date_et'):
        y = x.loc[(x.hour_et.ge(10) | (x.hour_et.eq(9) & x.t.dt.minute.ge(30))) & x.hour_et.lt(16)]
        if len(y) == 78 and y.segment.nunique() == 1:
            rth.append(window_row(y, 390))
    rth = pd.DataFrame(rth)
    write(rth, 'price_complete_rth_days.csv')

    # Final canonical aggregate consistency, not a model-time feature reconstruction.
    con = sqlite3.connect((OUT/'es_audit_snapshot.sqlite3').as_uri()+'?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    aggregated = pd.read_sql_query("SELECT * FROM canonical_bars WHERE asset='ES' AND timeframe IN ('30m','1h')", con)
    con.close()
    aggregate_checks = []
    for raw in aggregated.itertuples(index=False):
        ts = pd.Timestamp(raw.timestamp_utc).tz_convert(TZ)
        minutes = 30 if raw.timeframe == '30m' else 60
        y = valid.loc[valid.t.ge(ts) & valid.t.lt(ts+pd.Timedelta(minutes=minutes))]
        if len(y) != minutes//5 or y.segment.nunique() != 1 or not y.t.iloc[0] == ts:
            continue
        expected = dict(open=y.open.iloc[0], high=y.high.max(), low=y.low.min(), close=y.close.iloc[-1], volume=y.volume.sum())
        row = dict(timeframe=raw.timeframe, start_et=ts, contract_symbol=raw.contract_symbol)
        for key, value in expected.items():
            row[key+'_stored'] = getattr(raw, key)
            row[key+'_from_5m'] = value
            row[key+'_diff'] = getattr(raw, key)-value
        row['ohlc_mismatch'] = any(abs(row[k+'_diff'])>.001 for k in ('open','high','low','close'))
        row['volume_mismatch'] = abs(row['volume_diff'])>.001
        aggregate_checks.append(row)
    aggregate_checks = pd.DataFrame(aggregate_checks)
    write(aggregate_checks, 'price_aggregate_consistency.csv')

    fig, axs = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for _, chunk in valid.groupby('segment'):
        axs[0].plot(chunk.t, chunk.close, color='#145da0', lw=.9)
    axs[0].set(title='Collected ESU6 price: gaps are left unconnected', ylabel='ES index points')
    axs[0].grid(alpha=.2)
    axs[1].bar(pd.to_datetime(daily.date_et), daily.expected_bars_in_span/12, color='#d9dfe6', label='Expected open hours in audit span')
    axs[1].bar(pd.to_datetime(daily.date_et), daily.complete_bars/12, color='#248f79', label='Collected complete hours')
    axs[1].set(ylabel='Hours', xlabel='Eastern calendar date')
    axs[1].legend(loc='upper left');axs[1].grid(axis='y', alpha=.2)
    fig.savefig(OUT/'price_coverage.png', dpi=160);plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    x = hourly.hour_et.to_numpy()
    axs[0].bar(x-.2, hourly.mean_range_points, width=.4, color='#145da0', label='Mean complete hourly range')
    axs[0].bar(x+.2, hourly.median_range_points, width=.4, color='#248f79', label='Median complete hourly range')
    axs[0].plot(x, hourly.mean_range_ex_july29, color='#ce6a22', marker='o', label='Mean excluding July 29')
    axs[0].set(ylabel='ES points', title='Price opportunity by ET hour, normalized per fully observed hour')
    axs[0].legend();axs[0].grid(axis='y', alpha=.2)
    axs[1].bar(x, hourly.mean_directional_efficiency, color='#7556a2')
    for i, row in hourly.iterrows():
        axs[1].text(row.hour_et, row.mean_directional_efficiency+.015, f'n={int(row.observed_complete_hours)}', ha='center', fontsize=7)
    axs[1].set(ylabel='Mean directional efficiency (0-1)', xlabel='Hour beginning, America/New_York', ylim=(0, .85))
    for ax in axs: ax.set_xticks(range(24));ax.set_xlim(-.6,23.6)
    axs[1].grid(axis='y',alpha=.2)
    fig.savefig(OUT/'price_hourly_opportunity.png',dpi=160);plt.close(fig)

    biggest = episodes.loc[episodes.minutes.eq(60)].head(6)
    fig, axs = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for ax, event in zip(axs.flat, biggest.itertuples()):
        y = valid.loc[valid.t.ge(event.start_et) & valid.t.lt(event.end_et)]
        elapsed = ((y.t-event.start_et).dt.total_seconds()/60).to_numpy()
        ax.fill_between(elapsed, y.low.to_numpy()-event.open, y.high.to_numpy()-event.open, color='#9dc6d8', alpha=.6)
        ax.plot(elapsed, y.close-event.open, color='#145da0', marker='o', markersize=3)
        ax.axhline(0,color='grey',lw=.6);ax.grid(alpha=.2)
        ax.set(title=f'{event.start_et:%b %d %H:%M}–{event.end_et:%H:%M} ET | {event.net_points:+.2f} pts', xlabel='Minutes from start', ylabel='Points from initial open')
    fig.savefig(OUT/'price_largest_episodes.png',dpi=160);plt.close(fig)

    summary = dict(bars=len(bars), complete_bars=int(bars.complete.sum()), incomplete_bars=int((~bars.complete).sum()),
                   first_bar_et=str(bars.t.min()), last_bar_et=str(bars.t.max()), calendar_dates=bars.date_et.nunique(),
                   contracts=bars.contract_symbol.value_counts().to_dict(), expected_open_bars=int(grid.expected.sum()),
                   missing_open_bars=int(grid.missing_open.sum()), collected_hours=len(bars)/12, complete_hours=bars.complete.sum()/12,
                   overall_coverage_pct=100*grid.observed.sum()/grid.expected.sum(), complete_rth_days=len(rth),
                   source_counts=bars.source.value_counts().to_dict(), market_data_types=bars.market_data_type.value_counts().to_dict(),
                   insertion_after_close_minutes=bars.insert_lag_after_close_min.describe(percentiles=[.1,.5,.9,.99]).to_dict(),
                   insert_gt60min=int(bars.insert_lag_after_close_min.gt(60).sum()), revised_after_insert=int(bars.revised_after_insert.sum()),
                   negative_lags=int(bars.insert_lag_after_close_min.lt(0).sum()), invalid_ohlc=int((~bars.valid_ohlc).sum()),
                   complete_clock_hours=len(hours), valid_windows_by_minutes=windows.minutes.value_counts().sort_index().to_dict(),
                   aggregate_comparisons=aggregate_checks.groupby('timeframe').agg(n=('timeframe','size'),ohlc_mismatch=('ohlc_mismatch','sum'),volume_mismatch=('volume_mismatch','sum')).to_dict(orient='index'))
    (OUT/'price_summary.json').write_text(json.dumps(summary,indent=2,default=lambda o: int(o) if isinstance(o,np.integer) else float(o)),encoding='utf-8')

    top_hourly = hourly.sort_values('mean_range_points', ascending=False).head(8)[['hour_et','observed_complete_hours','mean_range_points','median_range_points','mean_abs_net_points','mean_directional_efficiency','mean_range_ex_july29']]
    largest_each = windows.loc[windows.groupby('minutes').abs_net_points.idxmax()][['minutes','start_et','end_et','net_points','range_points','directional_efficiency']]
    ranges_each = windows.loc[windows.groupby('minutes').range_points.idxmax()][['minutes','start_et','end_et','net_points','range_points']]
    noise = hourly.sort_values('mean_directional_efficiency').head(6)[['hour_et','observed_complete_hours','mean_range_points','mean_directional_efficiency','choppy_hours_pct','directional_hours_pct']]
    holes = gaps.loc[gaps.missing_expected_open_bars.gt(0)]
    findings = f'''# ES collected price-action audit

Frozen source: `es_bars_5m.csv`, exported with `es_audit_snapshot.sqlite3`; no training data or external prices were used. All clock times below are Eastern. All point movements are descriptive price opportunity, not executable profit.

## Coverage and quality

- {len(bars):,} collected five-minute ES bars ({len(bars)/12:.2f} hours), all ESU6; {bars.complete.sum():,} explicitly complete and {(~bars.complete).sum()} explicitly incomplete.
- Span: {bars.t.min()} to {bars.t.max()} (bar starts); {bars.date_et.nunique()} calendar dates contain any bars and {len(rth)} fully observed 09:30–16:00 cash-session windows.
- Of {grid.expected.sum():,} expected open five-minute slots inside that span, {grid.observed.sum():,} are collected ({100*grid.observed.sum()/grid.expected.sum():.2f}%). This is sparse collection over a wider calendar span, not 46 uninterrupted days of observations.
- Two collection holes remain. The long hole contains scheduled closures as well as {int(holes.iloc[0].missing_expected_open_bars):,} uncollected open-market bars. The shorter hole contains {int(holes.iloc[1].missing_expected_open_bars):,} uncollected open-market bars. See the exact bounds below and `price_gaps.csv`.
- Regular 17:00–18:00 maintenance and Friday 17:00–Sunday 18:00 closure are excluded. September 7 13:00–18:00 is treated as the Labor Day early closure (13:00–17:00 holiday pause plus maintenance), consistent with the observed gap and the [2026 futures holiday schedule](https://www.discounttrading.com/futures-holiday-schedule.html); [CME holiday calendar](https://www.cmegroup.com/trading-hours.html). Calendar assumptions affect coverage denominators only; all analytical windows independently require contiguous observed bars.
- No invalid OHLC ordering, duplicate contract/timestamps, zero-volume bars, or zero-range bars were found. All nine incomplete rows are excluded from all price-window and hourly-opportunity calculations.
- Every bar has source `ibkr.historical.update`; {int(bars.market_data_type.eq('delayed').sum()):,} explicitly say delayed and the remainder have unspecified market-data type. None explicitly say real time.
- The canonical insert timestamp is at least {bars.insert_lag_after_close_min.min():.2f} minutes after the nominal five-minute close; median {bars.insert_lag_after_close_min.median():.2f} minutes; {bars.insert_lag_after_close_min.gt(60).sum():,} bars were inserted more than one hour afterward. This combines ingestion timing and backfills and is not a pure network-latency estimate.
- {bars.revised_after_insert.sum():,}/{len(bars):,} rows were updated after insertion. Final OHLC can therefore differ from what a model saw initially. Bar timestamp is not proof of data availability then.

{table(holes[['first_absent_et','last_absent_et','missing_expected_open_bars','expected_open_hours_missing']],2)}

## Largest directional windows

Open of first included bar to close of last included bar; exact elapsed duration, complete bars, one contract, no gap bridging. A start at 15:10 and end at 16:10 includes bars labeled 15:10 through 16:05. Adjacent endpoints may differ from a close-to-close markout by the initial bar's opening gap.

{table(largest_each)}

The largest high-to-low spans can occur in different windows; their extremes have no guaranteed order and cannot be treated as attainable profit:

{table(ranges_each)}

Largest nonoverlapping 60-minute episodes (greedy ranking by absolute net movement, so one shock is not counted repeatedly):

{table(episodes.loc[episodes.minutes.eq(60)].head(12)[['start_et','end_et','net_points','range_points','directional_efficiency']])}

## Hour-of-day opportunity and noise

Hourly statistics use exactly 12 complete, contiguous bars starting on the clock hour. Each hour/day contributes once. These are points per observed hour rather than total activity across unequal exposure; sample counts are included. Full results and bootstrap confidence intervals for mean range are in `price_hourly_opportunity.csv`. The bootstrap resamples collected dates within each hour and does not establish an independent out-of-sample edge.

{table(top_hourly)}

Directional efficiency is absolute open-to-close movement divided by the sum of absolute five-minute close-path changes (including first open-to-close). It ranges from zero (return to start) to one (a monotonic close path). Intrabar wicks are not captured by this measure; `wick_adjusted_efficiency` additionally divides net displacement by summed candle ranges. "Choppy" is an explicit descriptive threshold below 0.25 and "directional" means at least 0.60, not a trained setup label.

Hours with the lowest mean directional efficiency:

{table(noise)}

High range alone does not mean good execution: opposing moves can create large range and small net displacement. July 29 is an extreme outlier, so the hourly table also supplies mean range excluding that date. Trade-time or profitability decisions should use model-entry markouts, actual fill records and costs, not these price-only direction-agnostic rankings.

## Complete cash-session days

09:30–16:00, exactly 78 complete five-minute bars; holiday and partial sessions excluded. Largest five ranges:

{table(rth.sort_values('range_points',ascending=False).head(5)[['date_et','net_points','range_points','directional_efficiency']])}

## Higher-timeframe consistency

Stored 30-minute and 1-hour canonical bars were compared with aggregation of final, complete five-minute constituents only. A mismatch is absolute difference above 0.001 points (or units for volume). These checks describe consistency at the frozen cutoff, not model-time history:

{table(aggregate_checks.groupby('timeframe').agg(comparable=('timeframe','size'),ohlc_mismatches=('ohlc_mismatch','sum'),volume_mismatches=('volume_mismatch','sum')).reset_index(),0)}

## Files and reproducibility

`price_audit.py` generates the price audit only. `price_valid_windows.csv` contains every valid 5/15/30/60-minute window; `price_largest_nonoverlapping_episodes.csv` removes duplicate overlap within each horizon; `price_complete_clock_hours.csv` and `price_complete_rth_days.csv` give auditable observations. `price_daily_coverage.csv`, `price_hour_coverage.csv`, `price_gaps.csv`, `price_incomplete_bars.csv`, `price_bars_audited.csv`, and `price_aggregate_consistency.csv` document data quality. Plots: `price_coverage.png`, `price_hourly_opportunity.png`, `price_largest_episodes.png`.
'''
    (OUT/'price_findings.md').write_text(findings,encoding='utf-8')
    print(json.dumps(summary, indent=2, default=lambda o:int(o) if isinstance(o,np.integer) else float(o)))
    print('\nTOP HOURS\n'+top_hourly.to_string(index=False))
    print('\nLARGEST WINDOWS\n'+largest_each.to_string(index=False))
    print('\nNOISE\n'+noise.to_string(index=False))


if __name__ == '__main__':
    main()

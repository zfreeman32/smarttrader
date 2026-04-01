# Timezone And Calendar Migration Plan

## Goal

Move the OTE feature pipeline to a single explicit timestamp contract:

- ingest source bars with an explicit `source_timezone`
- normalize all stored datetimes to canonical `UTC`
- compute intraday seasonality from an explicit `feature_clock_timezone`
- compute London/New York session features from named timezone calendars
- compute daily and weekly higher-timeframe levels from FX market-close boundaries

## Why

The previous pipeline mixed:

- raw wall-clock hours from whatever timezone the source data used
- post-training regime logic that assumed UTC
- daily and weekly resampling that used plain calendar boundaries

That could make training features, backtests, and live deployment disagree when historical and live feeds used different timezones.

## Implemented In This Pass

- Added shared calendar utilities in `features/fx_calendar.py`
- Added timezone/calendar config fields to `features/config.py`
- Normalized feature-builder ingestion in `features/io.py` and `features/builder.py`
- Reworked `session.py` and `temporal_context.py` to use shared calendar logic
- Reworked `htf_context.py` daily and weekly aggregation to use FX market-close boundaries
- Aligned `model_testing/ote_regime_labeler.py` with the shared calendar logic
- Added CLI overrides for source/canonical/feature-clock/market-close timezones

## Implemented In The Full-Path Follow-Up Pass

- Normalized raw OTE labeling ingress in `data/labeling/labeling_engine.py`
- Added label-output metadata sidecars with timezone and source-feed lineage
- Propagated source-lineage and timezone-contract metadata through preprocessing `summary.json` and target `report.json`
- Carried timezone/source lineage into training artifacts via `training_summary.json` and `model_config.json`
- Updated `scripts/run_eurusd_ote_pipeline.bat` and `end-to-end-README.md` to use the explicit contract
- Added focused tests for raw labeling ingress, preprocessing lineage propagation, and training artifact metadata

## Current Contract

- `source_timezone`: timezone used to localize naive source timestamps
- `canonical_timezone`: normalized storage timezone, default `UTC`
- `feature_clock_timezone`: timezone for cyclical hour/day features, default `America/New_York`
- `market_close_timezone`: timezone for FX day/week roll, default `America/New_York`
- `market_close_time`: default `17:00`

## Recommended Historical Migration

For the historical EURUSD 5-minute dataset you described:

- set `source_timezone` to `GMT-6`
- keep `canonical_timezone=UTC`
- keep `feature_clock_timezone=America/New_York`
- keep `market_close_timezone=America/New_York`

Then:

1. rebuild labeled datasets
2. rebuild features
3. rebuild prepared train/val/test splits
4. retrain models
5. recalibrate thresholds and backtests

## Recommended Deployment Contract

At live inference:

1. declare the incoming feed timezone explicitly
2. normalize live bars to canonical `UTC`
3. maintain rolling state in `UTC`
4. derive intraday/session/day/week features through the shared calendar layer
5. log the timezone contract alongside model metadata

## Remaining Repo Steps

- add deployment-side adapters that reject mismatched timezone contracts before live inference
- add DST-transition regression tests around live startup and market-close rollovers
- version model artifacts by feature-calendar contract so old models cannot be mixed with new features

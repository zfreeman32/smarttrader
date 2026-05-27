# Feature Generation Toolkit

This package is the repo's modular feature-engineering layer for OHLCV-based dataset builds.
It now supports:

- registered feature families
- recipe-driven builds
- explicit timezone and calendar normalization
- reusable post-feature transforms
- standalone strategy-derived feature blocks
- metadata-rich handoff into preprocessing and model training

Current snapshot of the package:

- 16 registered feature sets
- 3 recipe JSONs
- 405 auto-discovered standalone strategy entries from 404 top-level scripts under `features/strategies/`

## What Lives Here

```text
features/
|-- __init__.py
|-- all_indicators.py
|-- all_strategies.py
|-- builder.py
|-- cli.py
|-- config.py
|-- feature_list.txt
|-- fx_calendar.py
|-- io.py
|-- preprocessing.py
|-- progress.py
|-- registry.py
|-- strategy_registry.py
|-- strategy_similarity.py
|-- strategy_subprocess_runner.py
|-- transforms.py
|-- feature_sets/
|   |-- __init__.py
|   |-- continuation_pullback.py
|   |-- exhaustion.py
|   |-- fractional_diff.py
|   |-- htf_context.py
|   |-- ict_context.py
|   |-- microstructure.py
|   |-- momentum.py
|   |-- price_action.py
|   |-- quality.py
|   |-- session.py
|   |-- strategy_signals.py
|   |-- structure.py
|   |-- temporal_context.py
|   |-- trend.py
|   |-- volatility.py
|   `-- volume.py
|-- recipes/
|   |-- meta_labeling.json
|   |-- ote_base.json
|   `-- ote_extended.json
|-- strategies/
|   |-- *.py
|   `-- misc/
`-- docs/
    `-- TIMEZONE_CALENDAR_PLAN.md
```

## Core Workflow

List the registered feature sets and recipe files:

```powershell
python -m features.cli list
```

List the discovered standalone strategy entries:

```powershell
python -m features.cli list --strategies
```

Build the default OTE feature dataset:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_features.csv `
  --recipe features/recipes/ote_base.json `
  --progress
```

Build with an explicit timezone contract for historical data:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_features_tz.csv `
  --recipe features/recipes/ote_base.json `
  --source-timezone GMT-6 `
  --canonical-timezone UTC `
  --feature-clock-timezone America/New_York `
  --market-close-timezone America/New_York `
  --progress
```

Build the larger extended recipe with parallel transform jobs and dtype downcasting:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_features_extended.csv `
  --recipe features/recipes/ote_extended.json `
  --transform-workers 4 `
  --optimize-memory `
  --progress
```

Build a dataset that also includes selected standalone strategy outputs:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_with_strategies.csv `
  --recipe features/recipes/ote_base.json `
  --strategy acc_dist_strat `
  --strategy adx_breakouts_signals `
  --strategy atr_trading_signals `
  --skip-strategy-errors `
  --progress
```

Build the widest available strategy-enriched dataset in best-effort mode:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_full_dataset.csv `
  --recipe features/recipes/ote_extended.json `
  --all-strategies `
  --skip-strategy-errors `
  --strategy-timeout-seconds 300 `
  --transform-workers 4 `
  --progress
```

Prepare a generated feature dataset for target-specific model training:

```powershell
python -m preprocessing prepare `
  data/features/eurusd_5min_ote_features.csv `
  --output-dir data/prepared/eurusd_5min_ote_features
```

The compatibility alias still works:

```powershell
python -m features.cli preprocess `
  data/features/eurusd_5min_ote_features.csv `
  --output-dir data/prepared/eurusd_5min_ote_features
```

Compare the legacy monolith against the standalone strategy folder:

```powershell
python -m features.strategy_similarity
```

## What The Builder Does Now

- standardizes common input columns such as `timestamp`, `datetime`, `date`, `time`, `open`, `high`, `low`, `close`, and `volume`
- creates and normalizes a canonical `datetime` column when only `date` and `time` are present
- localizes naive timestamps with `source_timezone` and stores them in `canonical_timezone`
- validates OHLC consistency and drops broken rows by default
- runs feature sets in recipe order through the central registry
- runs transform blocks for winsorization, percentile ranks, ATR normalization, sigma normalization, lags, rolling stats, z-scores, and curated interactions
- can parallelize independent transforms and strategy execution with `--transform-workers`
- can downcast generated numeric feature columns with `--optimize-memory`
- writes both the output CSV and a `.metadata.json` sidecar

That metadata sidecar now includes:

- feature-set and transform counts
- per-step timings
- memory usage
- timezone contract details
- invalid-row counts
- generated feature column names
- upstream source lineage when the input file already has its own metadata sidecar
- per-strategy build and skip details when `strategy_signals` is used

## Registered Feature Sets

- `price_action`: returns, candle anatomy, and close-location features
- `volatility`: ATR, rolling volatility, and range-normalized volatility features
- `trend`: EMA distance, slope, alignment, and MACD trend features
- `momentum`: RSI, ROC, acceleration, and exhaustion-style momentum features
- `volume`: volume intensity, imbalance, and money-flow features
- `structure`: causal structure, sweep, premium/discount, and displacement features
- `htf_context`: higher-timeframe swing, daily/weekly range, and HTF trend-alignment context
- `continuation_pullback`: pullback geometry, retest quality, and trend-resumption structure features
- `ict_context`: ICT proximity features for FVGs, order blocks, liquidity pools, and structure breaks
- `exhaustion`: momentum deceleration, divergence, compression, and failed-continuation features
- `microstructure`: spread, impact, illiquidity, and intrabar microstructure proxies
- `session`: session, overlap, kill-zone, and cyclical time features
- `temporal_context`: event-age, sequence-order, clustering, and session-phase timing features
- `fractional_diff`: fractionally differentiated close series for non-stationarity treatment
- `quality`: anomaly, bar-quality, and data-integrity flags
- `strategy_signals`: standalone strategy-module outputs encoded as dataset features when strategy ids are requested

## Recipes

- `features/recipes/ote_base.json`: default OTE recipe with the full current baseline feature-family stack, continuation-pullback context, and standard lag, rolling, and z-score settings
- `features/recipes/ote_extended.json`: a heavier OTE recipe with wider lag windows, broader rolling windows, and more continuation-oriented transform coverage
- `features/recipes/meta_labeling.json`: a lighter recipe for meta-labeling workflows with a narrower transform surface and no `continuation_pullback` block

You can also override recipe order directly:

```powershell
python -m features.cli build input.csv `
  --output output.csv `
  --feature-set price_action `
  --feature-set trend `
  --feature-set session
```

## Standalone Strategy Integration

- strategy discovery currently scans top-level `features/strategies/*.py`
- `features/strategies/misc/` is present in the repo, but it is not auto-registered by the current discovery pass
- if a strategy module exposes one top-level function, its file stem becomes the strategy id
- if a strategy module exposes multiple top-level functions, entries are registered as `file_stem.function_name`
- strategy outputs are renamed with a `strategy__<strategy_id>__...` prefix before being merged into the dataset
- boolean outputs are encoded as `0/1`
- numeric outputs stay numeric
- non-numeric categorical outputs are one-hot encoded
- `--skip-strategy-errors` records failures in metadata instead of stopping the full build
- `--strategy-timeout-seconds` enables subprocess-based timeouts so one bad strategy does not hang the whole dataset job

## Timezone And Calendar Contract

The feature pipeline now has an explicit timestamp contract:

- `source_timezone`: how naive source bars should be localized
- `canonical_timezone`: normalized storage timezone, default `UTC`
- `feature_clock_timezone`: timezone used for cyclical time features and strategy clock fields, default `America/New_York`
- `market_close_timezone`: timezone used to define FX market-day and market-week closes, default `America/New_York`
- `market_close_time`: default `17:00`

That contract is shared across:

- input normalization in `features/io.py`
- calendar helpers in `features/fx_calendar.py`
- session features in `features/feature_sets/session.py`
- temporal timing features in `features/feature_sets/temporal_context.py`
- higher-timeframe day and week aggregation in `features/feature_sets/htf_context.py`
- legacy strategy input adaptation in `features/strategy_registry.py`

See `features/docs/TIMEZONE_CALENDAR_PLAN.md` for the migration notes and recommended historical-data settings.

## Legacy And Reference Files

These files still matter, but they are no longer the main path for new work:

- `features/all_strategies.py`: legacy monolith and comparison reference
- `features/all_indicators.py`: older indicator reference code
- `features/feature_list.txt`: historical inventory/reference list

New feature work should usually go into:

- `features/feature_sets/` for reusable registered feature families
- `features/recipes/` for build presets
- top-level `features/strategies/*.py` for standalone strategy-derived features

## Extending The Package

To add a new feature family, create a module under `features/feature_sets/` and register it:

```python
import pandas as pd

from features.config import FeatureBuilderConfig
from features.registry import register_feature_set


@register_feature_set(
    name="my_custom_features",
    category="custom",
    description="Example custom factor block",
    required_columns=("close",),
)
def build_my_custom_features(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["my_factor"] = df["close"].pct_change(3)
    return out
```

Then add that feature-set name to a recipe JSON or pass it through repeated `--feature-set` flags.

To add a standalone strategy feature source, place a top-level `.py` file under `features/strategies/` that exposes one or more callables returning a `DataFrame` or `Series`. The strategy registry will discover it automatically on the next run.

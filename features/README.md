# Feature Generation Toolkit

This directory now has a modular feature-generation workflow built around:

- small, registered feature-set modules
- recipe-driven dataset builds
- reusable transformations for lags, rolling stats, z-scores, winsorization, percentile ranks, and interactions
- a documented roadmap for the higher-value missing features from your OTE and quantitative pipeline docs

The goal is simple: make it easy to add one new feature family without editing a giant monolithic script.

## What Changed

The new primary entry points are:

- `features/cli.py`
- `features/builder.py`
- `features/config.py`
- `features/registry.py`
- `features/strategy_registry.py`
- `features/strategy_similarity.py`
- `features/transforms.py`
- `features/feature_sets/`
- `features/recipes/`
- `features/docs/IMPLEMENTATION_PLAN.md`

The old monolithic reference code is still here for reuse and comparison:

- `features/all_indicators.py`
- `features/all_strategies.py`
- `features/features_analysis.py`
- `features/strategies/`

For strategy-script features, the source of truth is now the standalone files under `features/strategies/`.
`features/all_strategies.py` should be treated as a legacy aggregation/reference file rather than the place to pull strategy functions from.

## New Layout

```text
features/
|-- __init__.py
|-- builder.py
|-- cli.py
|-- config.py
|-- io.py
|-- registry.py
|-- transforms.py
|-- feature_sets/
|   |-- price_action.py
|   |-- volatility.py
|   |-- trend.py
|   |-- momentum.py
|   |-- volume.py
|   |-- structure.py
|   |-- htf_context.py
|   |-- ict_context.py
|   |-- exhaustion.py
|   |-- microstructure.py
|   |-- session.py
|   |-- temporal_context.py
|   |-- fractional_diff.py
|   `-- quality.py
|-- recipes/
|   |-- meta_labeling.json
|   |-- ote_base.json
|   `-- ote_extended.json
`-- docs/
    `-- IMPLEMENTATION_PLAN.md
```

## Quick Start

List available feature sets:

```powershell
python -m features.cli list
```

List discovered standalone strategy entries:

```powershell
python -m features.cli list --strategies
```

Build an OTE feature dataset from the auto-labeled or reviewed 5-minute file:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_features.csv `
  --recipe features/recipes/ote_base.json
```

Build a richer version with more lags and rolling windows:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_features_extended.csv `
  --recipe features/recipes/ote_extended.json
```

Each build writes:

- the feature CSV you requested
- a sidecar metadata file at the same path with `.metadata.json`

Prepare a generated feature dataset for target-specific model training:

```powershell
python -m preprocessing prepare `
  data/features/eurusd_5min_ote_2000.csv `
  --output-dir data/prepared/eurusd_5min_ote_2000
```

This preprocessing step:

- uses the builder metadata to keep the engineered feature set as the safe feature pool
- creates separate prepared outputs for detected long/short OTE and entry targets
- removes exact duplicate and low-information columns
- runs target-aware similarity / collinearity pruning
- reports class imbalance, feature importance, and dataset readiness per target

The legacy alias still works:

```powershell
python -m features.cli preprocess `
  data/features/eurusd_5min_ote_2000.csv `
  --output-dir data/prepared/eurusd_5min_ote_2000
```

Build a dataset that also includes selected standalone strategy outputs:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_with_strategies.csv `
  --recipe features/recipes/ote_base.json `
  --strategy acc_dist_strat `
  --strategy adx_breakouts_signals `
  --strategy average-true-range-percentage_strategy
```

Build the fullest version from your labeled 5-minute EURUSD file by keeping all core feature families, all transforms, and every discovered standalone strategy that can be executed in the current environment:

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_5min_ote_labels.csv `
  --output data/features/eurusd_5min_ote_full_dataset.csv `
  --recipe features/recipes/ote_extended.json `
  --all-strategies `
  --skip-strategy-errors `
  --progress
```

Notes:

- the output CSV already keeps your original label columns alongside the generated features
- `--all-strategies` currently runs in best-effort mode and records skipped strategies in the `.metadata.json` sidecar
- `--progress` prints live phase updates plus per-strategy status so long builds do not look stuck
- `--strategy-timeout-seconds 300` skips a standalone strategy if it runs longer than five minutes; use `0` to disable the timeout
- many standalone strategy scripts depend on the external `ta` package or on non-OHLCV inputs, so they will be skipped unless those dependencies and extra columns are available

Compare the monolithic `all_strategies.py` functions against the standalone strategy folder:

```powershell
python -m features.strategy_similarity
```

## Current Feature Families

The modular pipeline currently generates:

- price action and candle anatomy
- volatility and range normalization
- trend, EMA distance, and MACD structure
- momentum and exhaustion-style proxies
- volume and money-flow features
- causal structure and sweep context
- higher-timeframe structural context
- ICT / SMC proximity context
- turning-point / exhaustion context
- microstructure proxies
- session and kill-zone timing
- sequence-ready temporal context
- fractional differentiation
- anomaly and data-quality flags
- selected standalone strategy-module outputs
- lags, rolling means/stds, rolling z-scores, rolling winsorization, percentile ranks, and interaction features

That now covers the baseline transformation work from both pipeline docs, including the five previously highest-priority missing feature families.

## How To Add A New Feature Family

Add a new module under `features/feature_sets/` and register it:

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

Then add that feature-set name to a recipe JSON or pass it directly through the CLI.

## Why This Structure Matches Your Docs Better

The two reference docs push the feature pipeline in a direction that is:

- domain-aware instead of just indicator-heavy
- multi-stage instead of one giant script
- explicit about transformations, lags, rolling stats, and interactions
- aware of OTE-specific structure, confluence, and imbalance issues

This revamp gives you that scaffolding now, and the implementation roadmap in [`features/docs/IMPLEMENTATION_PLAN.md`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/features/docs/IMPLEMENTATION_PLAN.md) lays out the next feature families to add in priority order.

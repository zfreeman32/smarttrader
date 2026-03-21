# Data Labeling Method

This folder documents the labeling workflow used to create EURUSD 5-minute training labels for swing-based Optimal Trade Entry (OTE) modeling.

The method is intentionally not a naive "mark every local high/low" process. It is a causal, volatility-aware labeling pipeline designed to generate labels that are usable for machine learning without leaking future structure into the detection logic itself. Future information is only used where appropriate for training-target validation and entry scoring.

## Core Idea

The labeling system tries to answer two related questions:

1. Which 5-minute bars sit inside a valid long or short OTE zone?
2. Which single bar was the best executable entry near that swing?

To do that, the pipeline combines:

- causal swing detection on 5-minute data
- higher-timeframe structure from 30-minute and 1-hour context
- volatility-normalized thresholds using structural ATR
- post-event validation using triple-barrier outcomes
- forward trend-scan confirmation
- quality scoring, exclusion masks, and sample weights
- optional human review overrides

## Labeling Philosophy

The method is built around a few rules:

- Labels should reflect tradeable structure, not micro-noise.
- Thresholds should scale with volatility instead of using fixed pip distances.
- Swing labels should be stricter than chart annotations made by eye.
- Entry labels should be sparse and high quality, not broad.
- Negative samples near unresolved events should be excluded or down-weighted.
- Human review should correct edge cases without rewriting the whole dataset manually.

## End-to-End Workflow

### 1. Load and clean raw EURUSD 5-minute OHLCV

[`labeling_engine.py`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/labeling_engine.py) loads the base CSV, normalizes column names, builds a timestamp index, removes duplicate timestamps, fills missing volume if needed, and filters obvious anomalies.

Bars are dropped when they show:

- invalid OHLC relationships
- abnormally large ranges relative to ATR
- zero or non-positive volume

### 2. Build structural context

The 5-minute series is resampled into:

- 30-minute bars
- 1-hour bars

The engine computes normal ATR on 5-minute data, but uses a higher-timeframe structural ATR for swing thresholds. In the current default setup, 1-hour ATR is mapped down to the 5-minute index.

This is one of the main design choices in the method: structural thresholds are based on higher-timeframe volatility so the model labels meaningful moves instead of reacting to 5-minute noise.

### 3. Detect candidate swings causally

[`ote_labeling_engine.py`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/ote_labeling_engine.py) uses a causal zigzag-style detector with structural ATR-based confirmation.

Key rules:

- a swing must retrace by a minimum ATR-scaled amount before it is confirmed
- consecutive swings must be separated by a minimum ATR-scaled price distance
- consecutive swings must also be separated by a minimum number of bars
- warmup bars are ignored

This prevents consolidation clusters and reduces over-labeling.

### 4. Confirm multi-timeframe agreement

After 5-minute swings are detected, the pipeline also detects 30-minute and 1-hour swings and checks whether 5-minute swings align with higher-timeframe structure inside configurable time windows.

This produces confluence features such as:

- `htf_confluence_long`
- `htf_confluence_short`
- `htf_confluence_long_1hr`
- `htf_confluence_short_1hr`

It also stores timing features like bars since the last 30-minute or 1-hour swing of the same type.

### 5. Validate swings with outcome-based logic

Detected swings are not accepted blindly. Each swing is run through a triple-barrier test:

- take profit at `1.0 x structural ATR`
- stop loss at `2.0 x structural ATR`
- max holding window of `120` bars

Outcomes are recorded as:

- `tp`
- `sl`
- `timeout`
- `skip`

This step helps separate structurally meaningful swings from moves that immediately fail.

### 6. Cross-check with trend scanning

Each swing is also evaluated with a forward trend-scan over several windows. The engine looks for statistically meaningful follow-through in the expected direction using a linear trend t-statistic.

By default, a swing must pass this trend-scan filter to remain a positive label candidate.

### 7. Score label quality

Every swing receives a continuous quality score from `0.0` to `1.0`, then a tier:

- `A`
- `B`
- `C`
- `reject`

The quality score is a weighted blend of:

- triple-barrier outcome
- trend-scan strength
- higher-timeframe confluence
- entry quality

Current weighting in code:

- 35% triple barrier
- 30% trend scan
- 20% higher-timeframe agreement
- 15% entry score

Swings below the minimum quality threshold are rejected from positive labeling.

### 8. Create zone labels

Positive swings become bar-level OTE zone labels.

Current default zone definition:

- `2` bars before the swing extreme
- `0` bars after the swing extreme

This creates:

- `label_long_ote`
- `label_short_ote`

A long zone is anchored around a validated swing low. A short zone is anchored around a validated swing high.

### 9. Create precise entry labels

The pipeline also tries to identify one best executable entry bar per valid swing.

It searches a local window around the swing:

- up to `3` bars before the swing
- up to `4` bars after the swing
- but still before full confirmation

Candidate bars are scored using:

- risk/reward ratio
- follow-through in ATR units
- wick structure
- candle close location
- proximity to the swing price
- timing relative to the swing

This produces:

- `label_long_entry`
- `label_short_entry`

These labels are intentionally much sparser than the zone labels.

### 10. Mark exclusions and safe negatives

The method does not treat every unlabeled bar as a clean negative.

It creates exclusion masks around events so the downstream model can avoid learning from ambiguous areas:

- `exclude_long`
- `exclude_short`

It also marks bars that are considered safe negative samples:

- `neg_ok_long`
- `neg_ok_short`

This is important because unlabeled bars near unresolved or overlapping structure can poison a classifier if they are treated as ordinary negatives.

### 11. Compute concurrency and sample weights

When multiple events overlap, their information content is not equal. The pipeline therefore tracks concurrency and computes sample weights inspired by event uniqueness ideas from financial ML.

Important output columns:

- `concurrency_long`
- `concurrency_short`
- `sample_weight_long`
- `sample_weight_short`
- `sample_weight_entry_long`
- `sample_weight_entry_short`

Positive labels are weighted by both uniqueness and label quality.

### 12. Save both bar-level and swing-level outputs

Running the main engine writes:

- bar-level labels to [`eurusd_5min_ote_labels.csv`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/labeled_data/eurusd_5min_ote_labels.csv)
- swing metadata to [`eurusd_5min_ote_swings.csv`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/labeled_data/eurusd_5min_ote_swings.csv)

The bar-level file is what model training uses. The swing-level file is the audit trail for why those labels exist.

## Main Output Columns

The bar-level label file contains:

- raw market fields: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- volatility fields: `atr`, `structural_atr`
- zone labels: `label_long_ote`, `label_short_ote`
- entry labels: `label_long_entry`, `label_short_entry`
- quality fields: `label_quality_long`, `label_quality_short`, `entry_quality_long`, `entry_quality_short`
- exclusion fields: `exclude_long`, `exclude_short`, `neg_ok_long`, `neg_ok_short`
- higher-timeframe context fields
- concurrency and sample-weight fields
- `warmup_mask`

The swing metadata file contains one row per detected swing and stores:

- swing type and source timeframe
- swing and confirmation timestamps
- structural ATR-scaled swing size
- triple-barrier outcome
- chosen entry bar and entry score
- trend-scan stats
- higher-timeframe match flags
- final quality tier

## Human-in-the-Loop Layer

The pipeline is mostly automated, but not fully trust-based. Human review exists in two places.

### Manual swing annotation tool

[`manual_data_labeler_v2.py`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/manual_labeling/manual_data_labeler_v2.py) is a Dash app for manually drawing swing entry-to-exit lines on price data.

What it does:

- loads raw EURUSD price data
- lets you inspect large slices of history
- optionally overlays an EMA-smoothed line for easier visual swing reading
- snaps all drawn endpoints back to exact raw-price timestamps
- stores manual swing records with entry, exit, pips, bars, and notes

This tool is best understood as a research and calibration layer, not the main production label generator. It gives you a ground-truth-like manual view of what a meaningful swing looks like and helps validate whether the automatic engine is too loose or too strict.

Manual swings are currently saved to:

- [`swings_5min_manual.csv`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/manual_labeling/swings_5min_manual.csv) for historical manual labels
- or `swings_5min_smoothed.csv` when using the current app defaults

### OTE review and override app

[`ote_label_review_app.py`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/ote_label_review_app.py) is the second review layer.

What it does:

- loads the auto-labeled 5-minute dataset
- displays candlesticks with zone and entry labels
- lets you click a specific bar
- lets you force a label on or off
- stores overrides separately from the base label file
- exports a reviewed label file when you are satisfied

This means the labeling workflow stays auditable:

- base labels remain reproducible from code
- human corrections are stored separately
- final reviewed labels can be exported without losing the original auto-generated dataset

Override outputs live in:

- `data/labeling/review/ote_label_overrides.csv`
- `data/labeling/labeled_data/eurusd_5min_ote_labels_reviewed.csv`

## Current Default Parameters

The current production defaults are defined in [`labeling_engine.py`](/c:/Users/zebfr/Documents/All_Files/TRADING/trade_bot/data/labeling/labeling_engine.py) and include:

- ATR period: `14`
- structural ATR timeframe: `1hr`
- CUSUM threshold multiplier: `0.5`
- confirmation retrace: `0.8 x structural ATR`
- minimum swing size: `1.35 x structural ATR`
- minimum swing distance: `0.75 x structural ATR`
- minimum bars between swings: `18`
- zone pre-bars: `2`
- zone post-bars: `0`
- entry lookback bars: `3`
- max entry delay after swing: `4`
- triple-barrier profit: `1.0 x ATR`
- triple-barrier stop: `2.0 x ATR`
- triple-barrier max holding: `120` bars
- warmup bars: `50`

These defaults are opinionated and tuned toward cleaner training labels rather than maximum signal frequency.

## Why This Method Is Different From Simple Labeling

This approach is different from basic technical-analysis labeling because it does not:

- label every local pivot
- use fixed pip thresholds across all volatility regimes
- assume every unlabeled bar is a safe negative
- treat all positive labels as equally reliable
- rely on a purely visual discretionary workflow

Instead, it creates labels that are:

- causal in structural detection
- volatility-normalized
- outcome-validated
- quality-scored
- higher-timeframe aware
- reviewable and correctable by a human

## Typical Usage

Generate the base labels:

```powershell
python data/labeling/labeling_engine.py
```

Generate labels and save a representative diagnostic chart:

```powershell
python data/labeling/labeling_engine.py --plot
```

Launch the manual swing labeling tool:

```powershell
python data/labeling/manual_labeling/manual_data_labeler_v2.py
```

Launch the review and override app:

```powershell
python data/labeling/ote_label_review_app.py
```

## Snapshot of the Current Label Set

From the existing analysis outputs in this folder:

- total rows: `1,121,592`
- usable rows after warmup: `1,121,542`
- long zone rate: `1.544%`
- short zone rate: `1.559%`
- long entry rate: `0.499%`
- short entry rate: `0.504%`
- average long label quality: `0.894`
- average short label quality: `0.894`
- swing count: `16,575`

Those figures show the intended character of the dataset: sparse labels, relatively selective entries, and high average quality after filtering.

## Summary

The labeling method in this project is a layered process:

1. clean the raw market data
2. detect swings causally with structural ATR thresholds
3. validate them with outcome and follow-through tests
4. convert validated swings into zone and precise entry labels
5. add exclusion masks, confluence features, and sample weights
6. review edge cases with human overrides when needed

The result is a dataset designed for machine learning on tradeable swing structure rather than a loose collection of hand-marked pivots.

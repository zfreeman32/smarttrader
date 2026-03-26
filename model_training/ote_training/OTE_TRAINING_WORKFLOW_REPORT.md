# Optimal Trade Entry Model Training Workflow

## An End-to-End Methodology for Causal EUR/USD Swing-Entry Detection

### Abstract

This document describes the implemented end-to-end workflow used to train Optimal Trade Entry (OTE) models in this repository. The system is designed to detect high-quality long and short swing-entry opportunities on EUR/USD 5-minute data under severe class imbalance, strong temporal dependence, and high risk of leakage. The pipeline begins with raw OHLCV ingestion, proceeds through causal swing labeling, feature generation, target-aware preprocessing, and backend-specific model training, and terminates in calibrated probability models with event-level operating thresholds. The methodology emphasizes causal construction, volatility normalization, higher-timeframe structural context, and training procedures aligned to rare-event detection rather than generic bar classification. Because development has primarily used sampled datasets, this report intentionally documents methods, assumptions, and artifacts without presenting empirical performance results.

## 1. Research Goal and System Scope

The implemented OTE workflow is built to answer a narrow but difficult question:

> At the close of each 5-minute EUR/USD bar, is the current bar inside a high-quality long or short optimal trade entry zone, and if so, how confidently should that event be scored?

The current production code solves this problem as two independent binary detection tasks:

- `long_ote`
- `short_ote`

The workflow is centered on the following design principles:

| Principle | Implementation consequence |
| --- | --- |
| Causality first | Detection logic uses only information available up to the current bar; future information is used only for label validation and post hoc training targets |
| Volatility awareness | Thresholds are expressed in structural ATR units instead of fixed pip distances |
| Event-based thinking | Model selection and thresholding prioritize event detection metrics rather than bar accuracy |
| Temporal hygiene | Splits are chronological, folds are purged, and scaling is fit only on training windows |
| Rare-event robustness | Focal loss, hard-negative weighting, balanced tuning samples, and calibrated thresholds address extreme imbalance |
| Structural context | Higher-timeframe swings, ICT-style structure, and session features are built directly into the feature space |

### Implemented scope versus broader research scope

The broader research notes in this repository reference 1-minute, 5-minute, 30-minute, and 1-hour data. The implemented OTE training path described here currently requires only raw 5-minute OHLCV as its base input. The code resamples that 5-minute series into 30-minute and 1-hour structure internally for labeling and feature generation. The 1-minute layer belongs to the wider research roadmap but is not required by the current training code path.

## 2. Implemented System Components

The workflow is implemented across four stages:

| Stage | Primary code | Purpose |
| --- | --- | --- |
| Raw data loading and labeling | `data/labeling/labeling_engine.py`, `data/labeling/ote_labeling_engine.py` | Convert raw EUR/USD 5-minute OHLCV into zone labels, entry labels, exclusion masks, confluence fields, and sample weights |
| Feature generation | `features/builder.py`, `features/cli.py`, `features/feature_sets/`, `features/transforms.py` | Build a modeling table from labeled bars using modular feature families plus causal transforms |
| Target-aware preprocessing | `features/preprocessing.py` | Create clean, chronological, target-specific train/validation/test datasets for `long_ote`, `short_ote`, `long_entry`, and `short_entry` |
| Model training | `model_training/ote_training/ote_xgboost_pipeline.py`, `model_training/ote_training/torch_trainer.py`, `model_training/ote_training/torch_models.py` | Train XGBoost or PyTorch OTE models with purged walk-forward validation, calibration, thresholding, and artifact export |

## 3. Raw Data Foundation and Integrity Controls

### 3.1 Input contract

The labeling entry point accepts a CSV containing EUR/USD 5-minute OHLCV data. The loader supports several timestamp conventions:

- separate `Date` and `Time` columns
- a `timestamp` column
- a `datetime` column

OHLC aliases are normalized to canonical names:

- `open`
- `high`
- `low`
- `close`
- `volume`

If volume is missing, the loader inserts a default zero-volume column.

### 3.2 Timestamp normalization and deduplication

The raw file is converted into a timestamp-indexed DataFrame, sorted chronologically, and deduplicated by timestamp. This creates the strict temporal ordering required by both the labeling engine and later chronological train/validation/test splits.

### 3.3 Anomaly detection before labeling

Before any label construction occurs, the loader removes bars flagged by deterministic anomaly rules:

| Rule | Intent |
| --- | --- |
| Invalid OHLC relationships | Reject malformed bars where the high-low-open-close geometry is impossible |
| Excessive range relative to ATR | Remove bars whose range exceeds `10 x` a rolling ATR baseline |
| Non-positive volume | Exclude bars with zero or negative volume |

This is important because the OTE pipeline uses local windows extensively. A single broken bar can contaminate multiple subsequent windows, not just one sample.

### 3.4 Multi-timeframe construction

The current code derives higher-timeframe structure directly from the cleaned 5-minute series:

- 30-minute OHLCV is built by resampling with first-open, max-high, min-low, last-close, and summed volume
- 1-hour OHLCV is produced using the same aggregation rules

These higher-timeframe frames are used later for structural ATR estimation, swing confluence, and higher-timeframe context features.

## 4. Label Construction Methodology

The implemented labeling system is not a naive pivot-labeling routine. It is a multi-stage causal labeling engine that detects swing structure, validates it with forward outcome logic, scores label quality, and creates both positive zones and safe negatives for downstream training.

### 4.1 Labeling objectives

The labeling layer produces two different target types:

| Target type | Meaning |
| --- | --- |
| Zone labels | Bars inside a validated OTE region around a swing extreme |
| Entry labels | The single best executable bar near the validated swing |

In practice this yields four target columns:

- `label_long_ote`
- `label_short_ote`
- `label_long_entry`
- `label_short_entry`

### 4.2 Structural ATR and volatility normalization

The core labeling thresholds are scaled by structural ATR rather than raw price units. The current defaults compute:

- a standard 5-minute ATR for local context
- a structural ATR on 1-hour bars
- a causal forward-fill mapping of the 1-hour ATR back onto the 5-minute index

This design makes swing detection robust across changing volatility regimes. A reversal threshold that is meaningful in low-volatility conditions remains meaningful in high-volatility conditions because it is expressed in ATR units.

### 4.3 Causal swing detection

The main swing detector is a zigzag-style causal algorithm with four gatekeepers:

| Gatekeeper | Default behavior |
| --- | --- |
| Confirmation retrace | Swing confirmation requires a retrace of `0.8 x` structural ATR |
| Minimum swing size | Consecutive swings must span at least `1.35 x` structural ATR |
| Minimum price separation | Consecutive swings must differ by at least `0.75 x` structural ATR |
| Minimum temporal separation | Consecutive swings must be at least 18 bars apart |

The detector operates after a 50-bar warmup period and alternates between confirmed highs and lows. This prevents consolidation noise from producing clusters of false swing events.

### 4.4 Higher-timeframe structural confirmation

The engine also runs the same style of swing detection on:

- 30-minute bars
- 1-hour bars

Lower-timeframe swings are then annotated with:

- whether a same-direction higher-timeframe swing exists within a confluence window
- the number of bars since the previous 30-minute swing of the same type
- the number of bars since the previous 1-hour swing of the same type

In addition to per-swing matching, all 5-minute bars within a configurable window around higher-timeframe swings are marked as higher-timeframe confluence regions. This produces both event-level validation and bar-level context.

### 4.5 Triple-barrier outcome validation

Detected swings are not automatically accepted as positive labels. Each swing is validated by a triple-barrier procedure applied from the swing confirmation index:

| Barrier | Default |
| --- | --- |
| Take profit | `1.0 x` structural ATR |
| Stop loss | `2.0 x` structural ATR |
| Vertical barrier | 120 bars |

The outcome is recorded as:

- `tp`
- `sl`
- `timeout`
- `skip`

This stage is crucial because it filters out structurally plausible but economically weak turns.

### 4.6 Trend-scan cross-validation

Each swing is further evaluated by forward trend scanning. The engine computes a linear-trend t-statistic over multiple forward windows:

- 12 bars
- 24 bars
- 48 bars
- 96 bars

For a long swing, stronger positive trend t-statistics imply better follow-through. For a short swing, the sign is reversed accordingly. The best signed t-statistic is retained, and by default a minimum absolute t-statistic of 2.0 is required for the swing to pass the trend-scan filter.

### 4.7 Swing quality scoring

Validated swings receive a continuous quality score in `[0, 1]` and a quality tier (`A`, `B`, `C`, or `reject`). The implemented scoring rule is:

| Component | Weight |
| --- | --- |
| Triple-barrier outcome score | 35% |
| Trend-scan score | 30% |
| Higher-timeframe agreement score | 20% |
| Entry-score contribution | 15% |

By default, swings below a minimum quality threshold of `0.35` are rejected from positive labeling.

### 4.8 Zone labels

OTE zone labels are generated around validated swing extremes. The current defaults are:

| Parameter | Default |
| --- | --- |
| Pre-bars before swing extreme | 2 |
| Post-bars after swing extreme | 0 |

The result is intentionally zone-based rather than point-based. This acknowledges that a bar slightly before the exact swing extreme can still be an excellent trade entry.

### 4.9 Precise entry labels

The entry-labeling stage searches for the best executable bar around each accepted swing. Candidate bars are searched in a local window:

- up to 3 bars before the swing
- up to 4 bars after the swing
- never beyond the confirmation bar

Each candidate is scored using a weighted combination of:

| Entry criterion | Role |
| --- | --- |
| Risk-reward ratio | Prefer setups with strong upside relative to local adverse excursion |
| Follow-through in ATR units | Prefer candidates followed by clear directional movement |
| Wick structure | Reward reversal-style candle anatomy |
| Close location in the bar | Reward closes consistent with reversal intent |
| Proximity to the actual swing price | Favor bars near the true turning point |
| Timing bonus | Prefer entries closer to the ideal timing window |

Only candidates passing minimum follow-through and risk-reward constraints are retained.

### 4.10 Exclusion masks and safe negatives

The labeling engine explicitly separates ambiguous negatives from clean negatives.

For each direction it creates:

- `exclude_long` / `exclude_short`
- `neg_ok_long` / `neg_ok_short`

Exclusion masks are built around confirmation points and positive label zones. Only bars outside warmup and outside ambiguity zones are allowed to act as negative samples during preprocessing.

### 4.11 Concurrency-aware sample weights

The engine computes label concurrency and inverse-concurrency sample weights. Positive samples are weighted by:

1. inverse overlap with other active label windows
2. label quality or entry quality

This produces:

- `sample_weight_long`
- `sample_weight_short`
- `sample_weight_entry_long`
- `sample_weight_entry_short`

### 4.12 Labeling outputs

The labeling stage writes two complementary outputs:

| Output | Purpose |
| --- | --- |
| `eurusd_5min_ote_labels.csv` | Bar-level training table with labels, exclusion masks, qualities, confluence, and weights |
| `eurusd_5min_ote_swings.csv` | Swing audit trail containing confirmation metadata, outcomes, entry choice, and label tier |

Optional review layers exist through the manual labeling and review applications, allowing human overrides without destroying reproducibility of the base automated labels.

## 5. Feature Engineering Methodology

After labeling, the workflow transforms the labeled 5-minute dataset into a modeling table using a modular feature builder. The default OTE recipes are deliberately broad: they represent price behavior, volatility, trend, structure, higher-timeframe context, ICT context, exhaustion signals, microstructure proxies, and temporal/session state.

### 5.1 Feature builder entry conditions

The builder first standardizes column names, validates OHLC consistency, and removes invalid bars if requested. It then appends feature families in the order specified by the recipe.

The default OTE recipes preserve the original market columns and label columns while adding engineered features. Warmup rows are dropped after feature generation:

- `ote_base.json` uses 200 warmup rows
- `ote_extended.json` uses 250 warmup rows

After warmup removal, numeric columns are forward-filled and then remaining missing values are replaced with zero.

### 5.2 Default feature families

The default OTE recipes include the following feature sets:

| Feature family | Methodology | Representative features |
| --- | --- | --- |
| Price action | Returns, candle anatomy, gap behavior, candle-state flags | `close_return_1`, `candle_body`, `upper_wick`, `close_location`, `doji_like` |
| Volatility | ATR, realized volatility, volatility expansion, range shocks | `atr_14`, `atr_ratio_14_50`, `rolling_vol_20`, `volatility_expansion_20`, `range_shock_20` |
| Trend | EMA and SMA structure, moving-average spreads, slopes, MACD | `close_vs_ema_8_atr`, `ema_spread_21_50_atr`, `ema_alignment`, `macd_hist` |
| Momentum | Oscillator and price-momentum dynamics | `rsi_14`, `rsi_delta_3`, `roc_5`, `roc_10`, `price_acceleration_3` |
| Volume | Relative volume, imbalance, money-flow proxies | `log_volume`, `volume_relative_20`, `volume_imbalance_10`, `money_flow_10` |
| Structure | Prior highs and lows, premium/discount positioning, sweeps, displacement | `dist_to_prior_high_20_atr`, `price_position_50`, `sweep_low_20`, `displacement_bullish` |
| Higher-timeframe context | Resampled 30-minute and 1-hour swing/trend context plus daily/weekly range levels | `htf_30m_dist_to_swing_low_atr`, `htf_1h_ema_alignment`, `htf_dist_to_prev_day_high_atr`, `htf_alignment_score` |
| ICT context | FVG, order block, liquidity pool, BOS/CHOCH, and confluence features | `dist_to_bull_fvg_atr`, `dist_to_bear_order_block_atr`, `active_equal_low_pool_count`, `ict_total_confluence_1atr` |
| Exhaustion | Return deceleration, RSI deceleration, divergence, compression, failed breakout signals | `return_slope_decay_3_8`, `bullish_divergence_strength`, `compression_regime`, `bull_volume_exhaustion` |
| Microstructure | Spread, price impact, Amihud and Kyle-style proxies, efficiency | `approx_spread`, `relative_spread_bps`, `amihud_proxy`, `intrabar_efficiency` |
| Session | Clock, cyclical time encoding, session flags, kill-zone flags | `hour_sin`, `dow_cos`, `in_london_session`, `in_london_ny_overlap` |
| Temporal context | Bars since structural events, event density, multi-step event sequences | `bars_since_last_sweep`, `major_event_cluster_10`, `bull_sequence_sweep_displacement_retrace` |
| Fractional differentiation | Memory-preserving stationary transformation of close | `fracdiff_close`, `fracdiff_close_return_1` |
| Quality | Data quality and anomaly indicators | `anomaly_range_gt_10atr`, `zero_volume_flag`, `large_time_gap_flag` |

### 5.3 Higher-timeframe feature construction

The higher-timeframe feature block resamples the labeled 5-minute dataset into completed 30-minute and 1-hour bars using right-closed, right-labeled windows. Those higher-timeframe features are then forward-filled back to the 5-minute index. This ensures the 5-minute model never sees unfinished higher-timeframe candles.

The higher-timeframe block includes:

- latest higher-timeframe swing highs and lows
- distance to those swings in ATR units
- higher-timeframe EMA alignment and spread
- bars since higher-timeframe swing events
- distances to previous-day, rolling-daily, and rolling-weekly highs and lows

### 5.4 ICT-context feature construction

The ICT feature family encodes structural concepts directly from price action:

- fair value gaps defined by displacement-style three-candle geometry
- bullish and bearish order blocks defined from candle-body dominance and break conditions
- equal-high and equal-low liquidity pools using swing-level similarity tolerances
- break of structure and change of character states
- confluence counts inside a one-ATR neighborhood

These are implemented as continuous distance and age features whenever possible, rather than purely binary flags.

### 5.5 Transform library

Beyond the base feature families, the recipes add a second layer of engineered transforms:

| Transform type | Technique |
| --- | --- |
| Lags | Selected columns shifted by multiple past offsets |
| Rolling statistics | Rolling means and standard deviations |
| Rolling z-scores | Local normalization of regime-sensitive features |
| Rolling winsorization | Historical clipping using rolling quantiles for return-like features |
| Percentile ranks | Rolling percentile position of volatility and volume features |
| ATR normalization | Conversion of price-unit features into volatility-scaled units |
| Sigma normalization | Rolling standardization for selected higher-level signals |
| Curated interactions | Domain-guided combinations such as trend-momentum alignment, structure proximity, session-conditioned setups, and higher-timeframe alignment interactions |

### 5.6 Recipe configuration

Two default OTE recipes are provided:

| Recipe | Intended use |
| --- | --- |
| `features/recipes/ote_base.json` | Baseline OTE feature build with core transforms |
| `features/recipes/ote_extended.json` | Richer build with more lag columns, wider rolling windows, and additional temporal depth |

An optional `strategy_signals` family can also be added to incorporate outputs from standalone strategy modules, but it is not part of the default OTE recipe.

## 6. Target-Aware Preprocessing and Dataset Packaging

The feature dataset produced by the builder is not trained directly. It is first converted into target-specific prepared datasets for each label type. This stage is implemented by `FeaturePreprocessingPipeline`.

### 6.1 Feature-pool resolution

The preprocessing stage first determines which columns are eligible model features:

1. If feature-builder metadata exists, it uses the metadata-defined engineered feature list.
2. Otherwise it falls back to a heuristic selector that excludes labels, helper columns, timestamps, raw OHLCV, and other non-model fields.

This protects the pipeline against accidental inclusion of target or helper columns in the feature pool.

### 6.2 Feature encoding

Candidate features are converted to numeric form as follows:

| Input type | Preprocessing action |
| --- | --- |
| Boolean | Cast to integer |
| Numeric | Coerce to numeric |
| Categorical/string | Label-encode after null filling and trimming |
| Constant categorical | Replace with zero |

Encoder mappings are saved for auditability.

### 6.3 Duplicate and constant feature removal

Before target-specific pruning, the pipeline removes:

- exact duplicate columns using hash-based signatures plus equality confirmation
- globally constant columns

This reduces redundant dimensionality before later importance ranking and collinearity analysis.

### 6.4 Target discovery and usable-row construction

The pipeline automatically discovers supported targets such as:

- `long_ote`
- `short_ote`
- `long_entry`
- `short_entry`

For each target, it creates a usable mask using:

1. the target label
2. the warmup mask
3. the direction-specific exclusion mask
4. the direction-specific safe-negative mask

Only positive bars and clean negatives are retained. Ambiguous bars are intentionally excluded from the training universe.

### 6.5 Sample weight propagation

If the labeled dataset contains direction-specific sample-weight columns, those are propagated directly into the prepared splits. Weights are clipped to non-negative values and zero weights are replaced with one for valid negatives.

### 6.6 Chronological splitting

Prepared datasets are split chronologically:

- 70% train
- 15% validation
- 15% test

The splits preserve temporal order. Date ranges are recorded in the companion target reports even though the split CSVs intentionally exclude datetime columns.

### 6.7 Missing-value treatment

Missing-value fill rules are estimated from the training split only:

| Distribution shape | Fill strategy |
| --- | --- |
| Strong skew (`|skew| > 1`) | Median |
| Mild skew or near-symmetric | Mean |
| Entirely missing column | Zero |

These fill values are then applied consistently to train, validation, and test.

### 6.8 Low-variance and collinearity pruning

After filling, the preprocessing stage removes:

- low-variance columns under a configurable variance threshold
- highly collinear features using absolute correlation pruning

When two features are highly correlated, the pipeline keeps the one with the stronger association score against the target and drops the weaker one. Near-duplicate feature pairs are also recorded separately.

### 6.9 Feature ranking

For each prepared target, the pipeline computes a composite feature importance table from:

- absolute association with the target
- mutual information
- random-forest importance

These scores are rank-aggregated into a composite score, which is later used by the trainer to cap the number of loaded base features before window expansion.

### 6.10 Prepared output contract

Each prepared target directory contains:

- `train.csv`
- `val.csv`
- `test.csv`
- `report.json`
- `report.txt`
- `features.json`
- `feature_importance.csv`

The split CSVs contain only:

- selected numeric feature columns
- `target`
- `sample_weight`

This keeps the training contract simple and prevents timestamp leakage through the prepared tables.

## 7. Model Training Methodology

The OTE trainer consumes one prepared target directory at a time and supports two backend families:

- XGBoost using sparse causal lag windows
- PyTorch sequence models using dense causal windows (`TCN` or `LSTM`)

### 7.1 Training data loading and leakage blocking

Before training, the pipeline:

1. loads ranked features from `features.json` and `feature_importance.csv`
2. caps the number of loaded base features
3. removes features whose names indicate likely leakage

The leakage-name filter blocks columns containing terms such as:

- `future`
- `lookahead`
- `pnl`
- `mfe`
- `mae`
- `take_profit`
- `tp_hit`
- `sl_hit`
- `exit_signal`

This is a second leakage defense on top of the earlier preprocessing step.

### 7.2 Backend-specific temporal representations

The same prepared tabular dataset can be converted into two different temporal representations.

#### XGBoost path

The XGBoost backend builds a sparse lag view:

- choose top-ranked base features
- choose a window size
- choose a number of lag anchors inside that window
- include current and historical lag snapshots
- optionally add delta features between the current bar and selected older lags

This keeps tree-based training memory-efficient while preserving causal context.

#### PyTorch path

The PyTorch backend builds dense sliding windows of shape:

`(samples, window_size, features)`

These windows feed either:

- a causal temporal convolutional network (TCN)
- an LSTM classifier

### 7.3 Fold-safe scaling

Within each fold, the trainer fits a `RobustScaler` on the training rows only, using configurable low and high quantiles. After transformation, values are clipped to a bounded range. This creates a stable fold-local scaling regime and avoids leaking future distributional information into earlier folds.

### 7.4 Purged walk-forward cross-validation

Model selection uses an expanding-window, purged walk-forward splitter. Each fold contains:

- a growing historical training window
- a purge gap between training and validation
- a forward-only validation segment

The purge length is chosen as the maximum of:

- sequence-context length plus an additional purge buffer
- event tolerance plus event cooldown requirements

This is meant to reduce contamination from overlapping temporal context and nearby events.

### 7.5 Rare-event training controls

Because OTE labels are sparse, the trainer uses multiple imbalance controls simultaneously:

| Mechanism | Effect |
| --- | --- |
| Focal loss | Down-weights easy negatives and focuses on hard decision boundaries |
| Sample weights from labeling | Preserves event uniqueness and label-quality information |
| Hard-negative upweighting | Increases weight of negative bars near positives |
| Balanced tuning subsample | Downsamples easy negatives while preserving positives and hard negatives during tuning |

Hard negatives are identified by expanding positive zones with a configurable radius and emphasizing nearby negatives rather than uniformly random negatives.

### 7.6 XGBoost objective and progressive fitting

The XGBoost backend uses a custom binary focal-loss objective with tunable:

- positive-class alpha
- focusing gamma

Training is staged in three phases:

1. warmup on an early subset
2. main training on the full fold
3. fine-tuning at a reduced learning rate

The tuned XGBoost hyperparameter space includes:

- number of top features
- window size
- lag count
- delta feature count
- learning rate
- focal alpha and gamma
- tree depth and child weight
- subsample and column sample
- L1 and L2 regularization
- split loss and delta step
- histogram bin count
- warmup, main, and fine-tune rounds
- hard-negative radius and multiplier

### 7.7 PyTorch architectures and training schedule

The sequence backend supports two architectures:

| Model | Core design |
| --- | --- |
| `TCN` | Stacked residual causal 1D convolutions with dilation growth and GELU activations |
| `LSTM` | Sequence encoder using the final hidden state and dropout before classification |

PyTorch training also uses weighted focal loss and follows a phased schedule:

- warmup
- main
- fine

The trainer applies:

- AdamW optimization
- ReduceLROnPlateau scheduling on validation AUPRC
- gradient clipping
- AMP when CUDA is available and enabled
- early stopping based on validation performance

### 7.8 Hyperparameter optimization

Model selection is driven by Optuna with a TPE sampler and median pruning. The objective function is not raw validation loss. Instead, each fold contributes a weighted score:

`0.70 * average_precision + 0.30 * event_fbeta_0_5`

This keeps the search aligned with both class-ranking quality and event-level detectability.

### 7.9 Probability calibration

After cross-validation, out-of-fold probabilities are optionally calibrated using:

- Platt scaling
- isotonic regression
- or no calibration

Calibration is fit only on valid out-of-fold predictions, which preserves separation between model fitting and probability correction.

### 7.10 Event-level threshold selection

The final operating threshold is not fixed at 0.5. Instead, the trainer searches a threshold grid and chooses the threshold that optimizes event-level behavior under:

- event tolerance in bars
- event cooldown in bars
- event precision
- event recall
- event F1
- event F-beta with beta = 0.5

The selection rule prioritizes:

1. event F-beta
2. event precision
3. closeness between predicted event count and true event count

This is an explicit alignment between thresholding and the actual trading use case.

### 7.11 Final refit and held-out test scoring

After selecting the best trial:

1. the pipeline reruns cross-validation with the best parameters
2. it fits a final model on the development set
3. it applies the learned scaler and calibrator to the held-out test split
4. it exports prediction files and metadata

The held-out test stage reports metrics internally, but those numeric results are intentionally omitted from this paper.

### 7.12 Exported training artifacts

The final output directory stores:

- trained model checkpoint (`model.json` for XGBoost or `model.pt` for PyTorch)
- scaler
- optional calibrator
- Optuna trial log
- training history
- model configuration
- window-level feature importance
- test predictions
- training summary metadata

## 8. Evaluation Framework Without Empirical Results

The implemented evaluation protocol spans both bar-level probability scoring and event-level trading relevance.

### 8.1 Probability metrics

The trainer computes:

- average precision
- ROC AUC
- Brier score

These metrics characterize ranking quality, separability, and probability calibration.

### 8.2 Event metrics

OTE is fundamentally an event-detection problem, not a per-bar accuracy problem. The trainer therefore groups contiguous positive bars into true zones and converts probability forecasts into de-duplicated event timestamps using the selected threshold and cooldown rule.

It then reports:

- event precision
- event recall
- event F1
- event F-beta (beta = 0.5)
- predicted event count
- true event count
- matched event count

This event-centric evaluation is more consistent with the intended use of the model than a standard 0/1 bar classification report.

## 9. Reproducibility and Artifact Flow

The implemented artifact flow is:

```text
Raw EUR/USD 5m CSV
  -> cleaned 5m bars
  -> resampled 30m and 1h bars
  -> labeled bar table + swing audit table
  -> engineered feature dataset + metadata
  -> target-specific prepared train/val/test folders
  -> tuned and calibrated OTE model artifacts
```

Each stage writes intermediate outputs rather than only final models. This allows the workflow to be audited, restarted from intermediate checkpoints, and reviewed manually when label quality or feature behavior needs investigation.

## 10. Methodological Strengths

The current implementation has several strong properties:

1. It enforces causality where causality matters most: swing detection, higher-timeframe alignment, scaling, and cross-validation.
2. It treats label ambiguity explicitly through exclusion masks and safe negatives.
3. It propagates label quality and event uniqueness into training weights instead of pretending all positives are equivalent.
4. It aligns threshold selection and cross-validation scoring to rare-event detection rather than generic classification accuracy.
5. It supports both tabular and sequence backends without changing the prepared-data contract.

## 11. Current Limitations and Practical Notes

Several practical limitations should be kept in view when extending or interpreting the pipeline:

| Limitation | Practical implication |
| --- | --- |
| Prepared split CSVs do not keep timestamps | Downstream predictions are row-index keyed unless rejoined to original metadata |
| The current code computes CUSUM events but does not use them as the final sample universe | CUSUM currently acts as a diagnostic structural-break signal rather than the actual label filter |
| The broader research notes mention 1-minute execution refinement | The current training pipeline does not require 1-minute data |
| Strategy-signal features are optional and dependency-heavy | The default OTE recipes do not depend on the strategy catalog |
| This report excludes empirical results | Use the exported reports and training summaries for future full-dataset performance analysis |

## 12. Conclusion

The implemented OTE workflow in this repository is a complete rare-event financial ML pipeline, not only a model-training script. It begins with anomaly-controlled raw EUR/USD 5-minute data, constructs causal and volatility-aware swing labels, enriches the labeled bars with multi-family structural and temporal features, packages the data into target-specific chronological splits, and trains calibrated event-detection models using purged walk-forward validation. The system is therefore best understood as an integrated methodology for causal swing-entry modeling rather than a single classifier.

For full-dataset operation, the practical run instructions are provided in the companion document:

- `model_training/ote_training/FULL_PIPELINE_README.md`

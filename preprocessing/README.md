# Preprocessing Toolkit

This package turns a wide feature-engineering CSV into target-specific model-training datasets and backend-aware feature rankings.

It sits between `features/` and `model_training/`:

1. `features/` builds a large, metadata-rich feature table.
2. `preprocessing/` converts that table into clean train/validation/test splits for each target.
3. `preprocessing backend-attribution` refines feature ranking for the downstream model family (`xgboost`, `tcn`, or `lstm`).

The documentation below is derived from the current code in this folder, not from an aspirational design.

## What Lives Here

| File | Role |
| --- | --- |
| `__main__.py` | Module entry point for `python -m preprocessing` |
| `cli.py` | CLI argument parsing and command wiring |
| `config.py` | Dataclasses for preprocessing and target metadata |
| `feature_selection.py` | Feature resolution, encoding, filtering, splitting, scaling, and target discovery |
| `feature_importance.py` | Baseline feature ranking via association, mutual information, and random forest importance |
| `pipeline.py` | Main orchestration for per-target dataset preparation and artifact writing |
| `backend_attribution.py` | Backend-aware proxy modeling and attribution-driven feature re-ranking |
| `reporting.py` | Human-readable report generation and JSON helpers |

## End-to-End Workflow

### Stage 1: `prepare`

`python -m preprocessing prepare ...` reads a generated feature CSV and produces:

- one subdirectory per discovered target
- `train.csv`, `val.csv`, and `test.csv`
- `feature_importance.csv`
- `features.json`
- `report.json` and `report.txt`
- `scaler.joblib` when scaling is enabled
- top-level `summary.json`, `summary_report.txt`, and `encoders.json`

### Stage 2: `backend-attribution`

`python -m preprocessing backend-attribution ...` reads the prepared target directories and produces backend-specific attribution artifacts such as:

- `shap_feature_stats_xgboost.csv`
- `shap_feature_stats_tcn.csv`
- `shap_feature_stats_lstm.csv`
- `feature_importance_merged_<backend>.csv`
- `backend_attribution_summary_<backend>.json`
- a root-level `backend_attribution_summary.json`

## Quick Start

Prepare a feature dataset for target-specific training:

```powershell
python -m preprocessing prepare `
  data/features/eurusd_5min_ote_full.csv `
  --output-dir data/prepared/eurusd_5min_ote_full `
  --target label_long_ote `
  --target label_short_ote
```

Build backend-aware rankings for the prepared dataset:

```powershell
python -m preprocessing backend-attribution `
  --prepared-root data/prepared/eurusd_5min_ote_full `
  --backend xgboost `
  --backend tcn `
  --backend lstm `
  --max-features 160 `
  --attribution-max-rows 50000 `
  --attribution-floor-fraction 0.08 `
  --attribution-cumulative-importance 0.95
```

Optional dependencies:

- `xgboost` is required for the `xgboost` backend
- `torch` is required for `tcn` and `lstm` backends

## Command Surface

### `prepare`

Main CLI arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `input` | required | Input feature CSV |
| `--output-dir` | required | Target output root |
| `--metadata` | `INPUT.metadata.json` | Optional metadata sidecar override |
| `--target` | config default list | Repeatable target selection |
| `--scaler` | `none` | `none`, `robust`, or `standard` |
| `--corr-threshold` | `0.98` | Absolute correlation threshold for pruning |
| `--similarity-threshold` | `0.995` | Threshold for near-duplicate reporting |
| `--max-analysis-rows` | `100000` | Cap for expensive analysis operations |

Key defaults from `PreprocessingConfig`:

| Setting | Default |
| --- | --- |
| `time_column` | `datetime` |
| `train_size / val_size / test_size` | `0.70 / 0.15 / 0.15` |
| `use_time_based_split` | `True` |
| `variance_threshold` | `1e-9` |
| `min_usable_rows` | `250` |
| `min_train_rows` | `100` |
| `min_positive_samples` | `25` |
| `top_n_features` | `50` |
| `rf_n_estimators` | `250` |
| `rf_max_depth` | `8` |
| `rf_min_samples_leaf` | `8` |
| `mutual_info_neighbors` | `5` |
| `include_base_price_columns` | `False` |
| `save_scaler` | `True` |

### `backend-attribution`

Main CLI arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--prepared-root` | required | Prepared dataset root |
| `--target` | all discovered targets | Repeatable prepared target directory filter |
| `--backend` | all supported | Repeatable backend selection |
| `--max-features` | `160` | Top base-ranked prepared features used in proxy models |
| `--attribution-max-rows` | `100000` | Recent validation rows used for attribution |
| `--base-weight` | `0.20` | Base ranking weight in merge |
| `--shap-weight` | `0.55` | Overall attribution weight in merge |
| `--shap-positive-weight` | `0.25` | Positive-class attribution weight in merge |
| `--attribution-floor-fraction` | `0.15` | Minimum attribution floor as a fraction of the maximum |
| `--attribution-cumulative-importance` | `0.90` | Cumulative attribution share cutoff |
| `--top-n-features` | `25` | Summary output depth |

Important backend defaults:

| Setting | Default |
| --- | --- |
| `xgb_num_boost_round` | `300` |
| `xgb_early_stopping_rounds` | `40` |
| `xgb_learning_rate` | `0.05` |
| `xgb_max_depth` | `6` |
| `xgb_min_child_weight` | `5.0` |
| `xgb_subsample` | `0.85` |
| `xgb_colsample_bytree` | `0.85` |
| `window_size` | `24` |
| `batch_size` | `256` |
| `attribution_batch_size` | `128` |
| `torch_epochs` | `10` |
| `hidden_size` | `64` |
| `num_layers` | `2` |
| `dropout` | `0.20` |
| `torch_learning_rate` | `1e-3` |
| `weight_decay` | `1e-4` |
| `gradient_clip` | `1.0` |
| `scale_quantile_low / high` | `5 / 95` |
| `scale_clip` | `8.0` |
| `focal_alpha / gamma` | `0.75 / 2.0` |
| `ig_steps` | `16` |
| `random_seed` | `42` |

## Prepare Pipeline, Step by Step

### 1. Input normalization

The pipeline loads the feature CSV with pandas and passes it through `features.io.standardize_market_frame`, which:

- normalizes common OHLCV column names
- builds a canonical `datetime` column when possible
- localizes and normalizes timestamps
- sorts by time if the input is not monotonic
- coerces OHLCV columns to numeric

This means preprocessing assumes a market-frame contract, even though timestamps are not later fed into the training matrix.

### 2. Row identity preservation

The pipeline carries a stable `source_row_idx` column all the way into the final splits.

This is important because it allows downstream prediction exports to be joined back to the original feature frame without putting timestamps directly into the model matrix. The prepared CSVs therefore look like:

```text
source_row_idx, <selected feature columns...>, target, sample_weight
```

### 3. Metadata-aware feature resolution

If the feature sidecar metadata contains `feature_columns`, that list is used as the initial feature pool. Otherwise the pipeline falls back to a rule-based exclusion list that removes:

- label columns
- helper columns such as `sample_weight_*`, `exclude_*`, `neg_ok_*`, and quality flags
- timestamp-like columns
- `source_row_idx`
- base price columns by default (`open`, `high`, `low`, `close`, `volume`)

Base price columns can be reintroduced with `include_base_price_columns=True`.

### 4. Encoding candidate features

Feature encoding is intentionally lightweight:

- numeric columns are kept numeric and downcast when possible
- boolean columns are kept as booleans
- non-numeric categorical columns are label-encoded
- single-valued categorical columns are collapsed to zero

The fitted category maps are written to `encoders.json`.

### 5. Global structural cleanup

Before target-specific work begins, the package removes:

- globally constant columns
- exact duplicate columns

Exact duplicates are detected with a two-stage process:

1. A hash signature is built on a sampled probe frame.
2. Any hash collision candidates are confirmed with full-column equality.

This keeps duplicate detection efficient without trusting hashes blindly.

### 6. Target discovery and target semantics

Targets are discovered from configured label columns. The package supports two target styles:

#### Direct targets

These are ordinary label columns such as:

- `label_long_reversal`
- `label_short_breakout_entry`
- `label_long_entry`

For direct targets, the pipeline also attempts to locate helper columns for:

- sample weights
- label quality
- exclusion masks
- safe-negative masks

#### Synthetic OTE targets

Two synthetic targets are hard-coded:

- `label_long_ote`
- `label_short_ote`

They are unions of component targets:

- long OTE = `long_reversal OR long_continuation_pullback OR long_breakout`
- short OTE = `short_reversal OR short_continuation_pullback OR short_breakout`

Their helper semantics are:

- positive label rule: any component positive
- negative label rule: not positive, all component safe-negative masks true, and no component exclusion flags
- sample weight rule: rowwise maximum of component sample weights

This is a conservative construction: it widens the positive class while refusing to treat uncertain rows as clean negatives.

### 7. Warmup and usable-row filtering

Rows are removed from modeling if:

- `warmup_mask` is true
- the row is neither a valid positive nor a valid negative under the target's rules

The second category becomes "ambiguous" rather than automatically negative. This is one of the most important scientific design choices in the package because it explicitly trades sample count for cleaner class semantics.

### 8. Temporal ordering and data split

Usable rows are ordered by `datetime` if needed and split into train/validation/test according to the configured fractions.

Default behavior is time-based:

- first 70% -> train
- next 15% -> validation
- final 15% -> test

Random splitting is available but disabled by default because this package is built for time-series forecasting, where chronological separation is more statistically defensible.

### 9. Train-only imputation, variance filtering, and collinearity pruning

After the split, all train-derived preprocessing is fit on the training subset only.

#### Missing-value imputation

For each feature with missing values in the training data:

- if all values are missing, fill with `0.0`
- if `|skew| > 1`, fill with the median
- otherwise fill with the mean

The same fill values are then applied to validation and test.

#### Low-variance removal

Columns with variance `<= variance_threshold` are dropped.

#### Collinearity pruning

Pairwise absolute Pearson correlation is measured on the recent training subset capped by `max_analysis_rows`.

When two features exceed the configured correlation threshold:

- keep the feature with the stronger target association
- break ties deterministically by feature name

Near-duplicates above `similarity_threshold` are separately reported.

### 10. Optional scaling

Scaling is optional and is fit on the training split only.

Supported scalers:

- `none`
- `robust`
- `standard`

If enabled, the scaler is saved as `scaler.joblib`.

### 11. Baseline feature ranking and dataset reports

The package computes `feature_importance.csv` using three views of signal:

- absolute Pearson association with the target
- mutual information
- random forest feature importance

Each metric is converted into a percentile rank and the final `composite_score` is the average of those ranks. That rank-based ensemble is intentionally used instead of raw-score averaging so the three metrics can contribute on comparable scales.

The pipeline then emits:

- per-target JSON and text reports
- class-balance summaries
- validation checks
- a heuristic readiness score and grade
- a top-level summary across targets

## Output Layout

A typical prepared directory looks like this:

```text
data/prepared/<dataset_name>/
|-- encoders.json
|-- summary.json
|-- summary_report.txt
|-- backend_attribution_summary.json            # only after backend-attribution runs
|-- <target_name>/
|   |-- train.csv
|   |-- val.csv
|   |-- test.csv
|   |-- features.json
|   |-- feature_importance.csv
|   |-- report.json
|   |-- report.txt
|   |-- scaler.joblib                          # optional
|   |-- shap_feature_stats_xgboost.csv         # backend-specific, optional
|   |-- shap_feature_stats_tcn.csv             # backend-specific, optional
|   |-- shap_feature_stats_lstm.csv            # backend-specific, optional
|   |-- feature_importance_merged_xgboost.csv  # backend-specific, optional
|   |-- feature_importance_merged_tcn.csv      # backend-specific, optional
|   |-- feature_importance_merged_lstm.csv     # backend-specific, optional
|   `-- backend_attribution_summary_<backend>.json
```

The repo currently contains real examples under `data/prepared/eurusd_5min_ote_full/`.

## Backend Attribution Pipeline

The second command in this package exists because the best generic ranking is not always the best ranking for a specific model family.

### 1. Candidate feature set

For each prepared target directory, the pipeline:

- loads the allowed feature list from `features.json`
- loads the base ranking from `feature_importance.csv`
- sorts by the strongest available ranking column
- keeps the top `max_features` features

This step intentionally shrinks the candidate set before training backend proxies, which keeps attribution computation tractable.

### 2. Proxy datasets

The backend attribution stage reads:

- `train.csv`
- `val.csv`

and extracts only:

- selected feature columns
- `target`
- `sample_weight`

### 3. XGBoost backend

For `xgboost`, the pipeline trains a weighted binary logistic booster with:

- histogram tree method
- validation-based early stopping
- weighted class-imbalance correction via `scale_pos_weight`
- evaluation by validation average precision and ROC AUC

Attribution is then computed with Tree SHAP on the most recent validation rows, capped by `attribution_max_rows`.

The resulting statistics include:

- mean absolute attribution over all rows
- mean signed attribution over all rows
- class-conditional positive and negative attribution summaries
- mean feature values for the same groups

### 4. Sequence backends: TCN and LSTM

For `tcn` and `lstm`, the pipeline trains PyTorch proxy models on sliding windows.

Common sequence preprocessing:

- robust scaling using the training split only
- quantile range defaults of 5th to 95th percentile
- optional clipping after scaling
- causal sequence windows of length `window_size`

Validation windows are built with historical context from the tail of the training split so that the first validation windows still have the correct lookback without leaking future bars.

Model details:

- `TCN`: causal dilated convolutional residual blocks with a latest-timestep readout
- `LSTM`: stacked LSTM with final hidden-state readout
- both use weighted focal loss, gradient clipping, and validation AUPRC monitoring

Attribution for sequence models uses integrated gradients with a zero baseline. The package sums attributions across the time axis to produce one feature-level contribution score per window.

### 5. Ranking merge

Base ranking and backend attribution are merged with weighted rank scores:

- base ranking weight
- overall attribution weight
- positive-class attribution weight

Default merge weights:

- base: `0.20`
- overall attribution: `0.55`
- positive attribution: `0.25`

The positive-class term matters because in imbalanced event detection, a feature can be globally important yet mostly informative about the negative class. The extra positive-class channel reduces that failure mode.

### 6. Attribution gating

After merging, the ranking is filtered in two steps:

1. floor gate: drop features below a fraction of the maximum attribution
2. cumulative gate: keep only the top features needed to reach a target share of cumulative attribution

For rare targets with positive rate `<= 1.5%`, the gating becomes more permissive:

- floor fraction is relaxed down to at most `0.05`
- cumulative share is raised up to at least `0.98`

This protects sparse-label targets from being over-pruned by noisy attribution concentration.

## Deep Scientific Analysis

### 1. The package is built around temporal validity first

The strongest statistical choice in this package is that nearly all target-aware operations are fit on the training split only after chronological partitioning.

That includes:

- fill-value estimation
- variance filtering
- correlation pruning
- scaler fitting
- baseline feature ranking
- backend proxy training

This matters because in financial time series, even unsupervised transformations can leak regime information if they are estimated on the full sample and then applied backward in time. The code avoids that for the high-impact transformations.

Two exceptions are worth noting:

- exact duplicate removal is done before the split
- categorical label encoding is also fit before the split

Those are low-risk compared with target-aware leakage, but they still use full-sample structure. If you want the strictest possible historical simulation, those two steps are candidates for future walk-forward refactoring.

### 2. Negative-class quality is treated as a first-class problem

The package does not assume that "not positive" means "negative." Instead, it uses:

- exclusion masks
- safe-negative masks
- warmup masks

to decide whether a row is truly usable.

Scientifically, this is a label-noise control mechanism. In trading datasets, the negative class is often contaminated by:

- unfinished setups
- overlapping structures
- event windows where the label is undefined rather than false
- early warmup periods where indicators are not yet stable

Discarding ambiguous rows lowers sample size but usually improves the Bayes signal-to-noise ratio of the learning problem.

### 3. The redundancy controls target the covariance structure of indicator-heavy feature sets

Indicator libraries and transform stacks produce many features that are:

- exact duplicates
- low-variance aliases
- lagged or rolled near-copies
- highly collinear normalizations of the same underlying state

This package attacks redundancy in layers:

1. exact duplicate removal
2. constant-column removal
3. low-variance filtering
4. correlation-based pruning

That sequence reduces effective dimensionality and usually helps three different failure modes:

- tree split-credit dilution
- unstable variable importance estimates
- gradient noise and conditioning burden in neural models

The correlation-pruning tie-breaker is not arbitrary. It uses target association so that when two features encode almost the same information, the one that appears more directly aligned with the target is retained.

### 4. The baseline ranking is intentionally multi-view instead of single-metric

Each baseline importance component sees a different kind of structure:

- Pearson association sees linear marginal alignment
- mutual information sees broader nonlinear dependency
- random forest importance sees interaction-friendly nonlinear partitions

By converting each metric into a rank and averaging ranks, the pipeline avoids one metric dominating purely because of scale or variance. This is closer to a Borda-style ensemble than to raw-score fusion.

That said, the three metrics do not have identical statistical behavior:

- Pearson correlation is cheap and stable but only linear
- mutual information is flexible but noisy in finite samples
- random forest importance can favor high-cardinality continuous features

Rank averaging is a pragmatic way to dampen those biases rather than eliminate them.

### 5. Sample weighting enters the pipeline, but not uniformly

Sample weights influence:

- random forest fitting in the baseline ranking
- XGBoost proxy training
- focal-loss training for sequence proxies
- validation AUPRC and ROC AUC during backend attribution

Sample weights do not directly enter:

- the Pearson association term
- the mutual information term

So the package is partially weight-aware, not fully weight-consistent. That is a reasonable engineering compromise, but it matters scientifically if your weighting scheme is meant to encode economic utility rather than just class balancing.

### 6. Backend-aware ranking is the package's most important modeling idea

The package assumes that a feature that looks strong in a generic filter may not be the feature that a specific downstream architecture actually uses well.

That assumption is well motivated:

- tree ensembles prefer thresholdable, interaction-friendly predictors
- TCNs prefer causal local patterns and multi-step temporal motifs
- LSTMs can encode slower recurrent state but may diffuse importance across timesteps

Using backend-specific proxy models and attribution scores is therefore a form of architecture-conditioned feature selection.

This is stronger than generic filter ranking because it asks a more operational question:

> Which prepared features matter to the kind of model we actually plan to train?

### 7. Positive-class attribution is especially important under class imbalance

In rare-event classification, global importance can be dominated by how a feature helps reject negatives. That is useful, but it can bury features that are specifically important for detecting positives.

This package addresses that directly by storing:

- `mean_abs_shap_all`
- `mean_abs_shap_positive`

and giving both a role in the merged ranking.

Scientifically, that is a good fit for trade-trigger modeling, where the positive class is the scarce class we usually care about most.

### 8. Sequence attribution is informative, but it compresses time structure

For `tcn` and `lstm`, integrated gradients are summed across the time axis before feature ranking.

That gives a clean feature-level score, but it also discards temporal localization. In other words, the current output can tell you:

- which feature family mattered across the window

but not:

- whether its importance came from the earliest bars, the latest bars, or a specific motif position

That is a deliberate simplification for feature selection. It is acceptable for ranking, but it is not a full interpretability solution for sequence dynamics.

The zero baseline used for integrated gradients is another practical approximation. Zero is computationally convenient, but it is not always an economically neutral state in financial features.

### 9. The readiness score is a triage heuristic, not a statistical proof

The package emits a `readiness` score and grade. This is useful for quickly scanning many targets, but it is not a calibrated estimate of future model performance.

It is based on:

- validation pass rate
- sample count
- positive count for classification
- mutual-information signal strength

This is a good operational dashboard, but it should not be interpreted like a confidence interval or a formal out-of-sample guarantee.

### 10. Current limitations and scientifically meaningful next steps

The current design is strong, but there are still important limitations:

- no purged or embargoed cross-validation for overlapping-event leakage control
- no explicit regime-stratified evaluation in preprocessing
- no weighted mutual information or weighted correlation
- no stability-selection layer across repeated resamples
- no direct treatment of multiple-testing risk across very wide feature universes
- sequence attributions are collapsed over time
- global categorical encoding is pre-split

High-value future improvements would be:

- walk-forward or purged CV summaries in addition to the fixed train/val/test split
- stability metrics for feature ranks across folds or eras
- grouped feature pruning by family, not just pairwise correlation
- category handling that can gracefully represent unseen production values
- temporal attribution summaries for sequence models instead of time-collapsed totals

## Target Discovery Rules

The default target list in `PreprocessingConfig` is:

```text
label_long_reversal
label_short_reversal
label_long_reversal_entry
label_short_reversal_entry
label_long_continuation_pullback
label_short_continuation_pullback
label_long_continuation_entry
label_short_continuation_entry
label_long_breakout
label_short_breakout
label_long_breakout_entry
label_short_breakout_entry
label_long_ote
label_short_ote
label_long_entry
label_short_entry
```

Family alias handling is built in so helper columns can be resolved across naming variants such as:

- `continuation` vs `continuation_pullback`
- empty family vs `ote`

## Reports and Diagnostics

### Top-level summary

`summary.json` contains:

- input and metadata lineage
- detected upstream preprocessing metadata
- row-windowing decisions
- feature-pool counts
- timezone contract
- per-target readiness summary

### Per-target report

`report.json` contains:

- row counts and split counts
- split date ranges
- target construction details
- class balance
- fill strategy report
- low-variance and collinearity diagnostics
- baseline importance summary
- validation results
- readiness score and open issues

### Backend attribution summary

`backend_attribution_summary_<backend>.json` contains:

- proxy model metrics
- attribution-window stats
- merge weights
- rare-target gate adjustments
- top raw-attribution features
- top merged features
- artifact file paths

## Extension Points

Common places to modify behavior:

- add or remove default targets in `config.py`
- extend synthetic target definitions in `feature_selection.py`
- change helper-column naming assumptions in `feature_selection.py`
- alter ranking composition in `feature_importance.py`
- change attribution merge and filtering behavior in `backend_attribution.py`
- customize report formatting in `reporting.py`

If you are changing the scientific behavior of the pipeline, the most consequential files are:

- `feature_selection.py`
- `feature_importance.py`
- `pipeline.py`
- `backend_attribution.py`

## Practical Takeaway

This preprocessing package is not just a cleaning script. It is a target-construction, temporal-splitting, feature-pruning, ranking, and attribution layer designed for noisy, imbalanced trading labels.

Its strongest ideas are:

- conservative negative-class construction
- train-only fitting for most target-aware transformations
- layered redundancy control
- rank-ensemble baseline importance
- backend-conditioned attribution merging

Its main scientific limitations are:

- fixed single split rather than richer walk-forward validation
- heuristic readiness scoring
- approximate sequence attribution
- partial rather than universal sample-weight awareness

Within those boundaries, it is a thoughtful and fairly rigorous bridge between raw engineered features and backend-specific model training.

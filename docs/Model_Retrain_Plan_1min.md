# 1-Minute OTE Training Plan

## Goal

Train a new 1-minute OTE model stack on `data/currency_data/eurusd-1m.csv` using the same offline sequence we trust on 5-minute data:

1. labeling
2. feature generation
3. preprocessing
4. backend attribution
5. focused model training
6. post-training thresholding and backtests

The plan below is intentionally anchored to what already worked in this repo rather than reopening the full search space.

---

## Repo Reality

The current maintained OTE path is still 5-minute-first:

- `data/labeling/labeling_engine.py` loads a 5-minute base series and resamples to `30m` and `1h`
- `data/labeling/reversal_labeling_engine.py` contains several explicit 5-minute assumptions in bar-based settings
- `scripts/run_ote_training.ps1` defaults to `data/prepared/eurusd_5min_ote_full`
- `model_training/ote_training/OTE_TRAINING_WORKFLOW_REPORT.md` explicitly says the implemented OTE path currently requires 5-minute OHLCV as its base input

So this is not a simple dataset swap. We need a controlled 1-minute migration.

---

## What Past Runs Say To Copy

Use the post-training winners, not raw CV rank, as the guide rail.

### Highest-value families

1. `short_reversal_xgb_v2_20260525`
   - Best saved post-training rank
   - `6/6` gates passed
   - Monthly Sharpe `3.61`
   - Profit factor `2.74`
   - About `23.4` trades/month

2. `long_reversal_tcn_v2_20260525_narrow48`
   - `6/6` gates passed
   - Monthly Sharpe `3.36`
   - Profit factor `2.80`
   - About `18.6` trades/month

3. `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair`
   - `6/6` gates passed
   - Annualized net pips `3558`
   - Profit factor `2.58`
   - Positive composite share `1.0`

4. `short_ote_union_tcn_candidate_20260520`
   - `6/6` gates passed
   - WFE `1.10`
   - About `17.6` trades/month

5. `long_ote_union_tcn_candidate_20260523`
   - `6/6` gates passed
   - WFE `1.22`
   - Annualized net pips `2831`

### Families to de-prioritize

- Breakout models had excellent training leaderboard scores but poor post-training conversion.
- Do not spend the first 1-minute cycle on breakout retrains.
- Reversal and union OTE should be the first 1-minute production candidates.

---

## Dataset Scope

### Initial comparison window

For the first 1-minute migration, keep the historical window aligned to the current validated 5-minute workflow so results are comparable.

- Recommended initial window: `2019-01-01` through `2026-03-24`
- Do not use the full `2002-2026` 1-minute file for the first wave
- Expand history only after the 1-minute pipeline is stable and directly comparable to the 5-minute baseline

### Current 5-minute baseline we are trying to beat

- Current label analysis:
  - `1,121,592` total labeled rows
  - `16,575` swings
  - long/short entry rates near `0.50%`
- Current prepared OTE set:
  - about `443k-445k` usable OTE rows
  - about `24.8k` positives per side
  - about `1,050` selected features per OTE target

That means the 1-minute goal is not "more rows at any cost." The real goal is cleaner microstructure-aware labels and better post-training robustness.

---

## Phase 0: Control Baseline And Code Audit

### Objective

Create a safe comparison point before true 1-minute labeling.

### Tasks

1. Build a control dataset by resampling 1-minute data back to 5-minute and rerunning the current pipeline.
2. Snapshot current 5-minute metrics for:
   - label density
   - ambiguous-row rate
   - prepared positive rate
   - post-training metrics by family
3. Audit 5-minute assumptions in:
   - `data/labeling/*.py`
   - `features/feature_sets/htf_context.py`
   - `scripts/run_ote_training.ps1`
   - `scripts/run_eurusd_ote_pipeline.bat`
   - `model_training/ote_training/*`

### Exit criteria

- We can reproduce the existing 5-minute behavior from the 1-minute source after resampling.
- We have a file-by-file list of what must become timeframe-neutral.

---

## Phase 1: 1-Minute Labeling Migration

### Objective

Make the labeling stack treat 1-minute bars as the base series without destroying the time meaning of the existing thresholds.

### Design rule

Keep the first 1-minute label pass time-equivalent to the 5-minute defaults. Do not leave bar-count settings unchanged.

### Starting parameter conversion

Use the current 5-minute defaults as minutes, then convert them to 1-minute bars.

| Setting | Current 5m | Time Meaning | 1m Starting Value |
| --- | ---: | ---: | ---: |
| `warmup_bars` | 50 | 250 minutes | 250 |
| `min_bars_between_swings` reversal | 18 | 90 minutes | 90 |
| `min_bars_between_swings` continuation | 8 | 40 minutes | 40 |
| `min_bars_between_swings` breakout | 12 | 60 minutes | 60 |
| `tb_max_bars` reversal | 120 | 600 minutes | 600 |
| `tb_max_bars` continuation | 36 | 180 minutes | 180 |
| `tb_max_bars` breakout | 24 | 120 minutes | 120 |
| `zone_pre_bars` reversal | 2 | 10 minutes | 10 |
| `zone_pre_bars` continuation | 1 | 5 minutes | 5 |
| `zone_pre_bars` breakout | 0 | 0 minutes | 0 |
| `entry_lookback_bars` reversal | 3 | 15 minutes | 15 |
| `entry_lookback_bars` continuation | 2 | 10 minutes | 10 |
| `entry_lookback_bars` breakout | 0 | 0 minutes | 0 |
| `entry_max_delay_after_swing` reversal | 4 | 20 minutes | 20 |
| `label_exclusion_pre_bars` | 10 | 50 minutes | 50 |

### Structural context

Keep higher-timeframe structure in the first 1-minute version:

- base series: `1m`
- structural ATR: still `1h`
- HTF confluence: still `30m` and `1h`

This preserves the repo's current idea that local execution timing can get finer while structural thresholds stay higher-timeframe.

### Required code changes

1. Generalize `data/labeling/labeling_engine.py` naming and defaults from `5m` to base-timeframe-neutral.
2. Generalize `data/labeling/reversal_labeling_engine.py` so printed messages, defaults, and source tags are not hardcoded to `5m`.
3. Do the same for continuation and breakout labeling engines.
4. Add 1-minute output names:
   - `data/labeling/labeled_data/eurusd_1min_ote_labels_full.csv`
   - `data/labeling/labeled_data/eurusd_1min_ote_swings_full.csv`

### Labeling acceptance checks

Do not move forward if any of these fail:

- Entry labels become extremely dense relative to the 5-minute baseline
- Ambiguous rows collapse toward zero
- Swing counts explode without a corresponding quality increase
- Triple-barrier timeouts dominate because horizons were not rescaled correctly

---

## Phase 2: 1-Minute Feature Generation

### Objective

Build a narrower first-pass 1-minute feature table instead of repeating the full widest 5-minute feature explosion.

### First-pass recipe strategy

Start with repo-native feature families only:

- `price_action`
- `volatility`
- `trend`
- `momentum`
- `structure`
- `htf_context`
- `ict_context`
- `exhaustion`
- `microstructure`
- `session`
- `temporal_context`
- `quality`

Add `continuation_pullback` only for union OTE runs.

### What not to do in wave 1

- Do not start with `--all-strategies`
- Do not start with every historical strategy script mixed in
- Do not optimize for maximum feature count

### Output targets

- `data/features/eurusd_1min_ote_reversal_core.csv`
- `data/features/eurusd_1min_ote_union_core.csv`

### Feature-build checks

- Validate timezone lineage and metadata sidecars
- Check higher-timeframe features are mapped from completed bars only
- Compare feature counts against the 5-minute baseline and trim before training if the 1-minute set becomes much wider without clear value

---

## Phase 3: Preprocessing

### Objective

Create target-specific prepared roots for 1-minute reversal and union families.

### First-pass settings

- `scaler none`
- time-based splits only
- `corr-threshold 0.98`
- `similarity-threshold 0.995`
- `max-analysis-rows 100000`

### Prepared roots

- `data/prepared/eurusd_1min_ote_reversal_core`
- `data/prepared/eurusd_1min_ote_union_core`

### Targets to prepare first

1. `long_reversal`
2. `short_reversal`
3. `long_ote`
4. `short_ote`

### Preprocessing stop rules

Stop and retune labels or features before training if:

- reversal positive counts are too low for stable CV
- OTE positives jump mainly because negatives became ambiguous or duplicated
- prepared reports show weak readiness or severe class skew changes

---

## Phase 4: Backend Attribution

### Objective

Use backend-specific ranking early so 1-minute training is feature-disciplined from the start.

### Backends for wave 1

- `xgboost`
- `tcn`

Skip `lstm` in the first 1-minute wave.

### Attribution settings

#### Reversal prepared root

- `max-features 96`
- `attribution-max-rows 50000`
- `attribution-floor-fraction 0.08`
- `attribution-cumulative-importance 0.95`

#### Union OTE prepared root

- `max-features 128`
- `attribution-max-rows 75000`
- `attribution-floor-fraction 0.08`
- `attribution-cumulative-importance 0.95`

### Rule

Use the backend-aware merged rankings in training. Do not fall back to generic `feature_importance.csv` unless attribution fails.

---

## Phase 5: Focused Training Waves

### Time-scaling rule for trainer geometry

Several training controls are also time-based and should be scaled from 5-minute bars to 1-minute bars for the first pass:

- `label_max_holding_bars`
- `label_exclusion_pre_bars`
- `label_zone_pre_bars`
- `purge_buffer_bars`
- `event_tolerance_bars`
- `event_cooldown_bars`

Starting conversion:

| Setting | Current 5m Value | 1m Starting Value |
| --- | ---: | ---: |
| `label_max_holding_bars` | 120 | 600 |
| `label_exclusion_pre_bars` | 10 | 50 |
| `label_zone_pre_bars` | 2 | 10 |
| `purge_buffer_bars` | 12 | 60 |
| `event_tolerance_bars` | 2 | 10 |
| `event_cooldown_bars` reversal | 4 | 20 |
| `event_cooldown_bars` union | 5 | 25 |

### Wave 1: Reversal family first

Reversal is the best 1-minute candidate because it already converts well post-training and runs at lower turnover than the OTE union models.

#### 1. `short_reversal_xgb_1m_v1`

Use the `short_reversal_xgb_v2_20260525` neighborhood, with `short_reversal_xgb_v1` only as a secondary reference.

| Parameter | Focused 1m Range |
| --- | --- |
| backend | `xgboost` |
| trials | `48-60` |
| `cv_min_folds` | `3` |
| `max_loaded_features` | `96` |
| `top_feature_min/max` | `24 / 96` |
| `window_min/max` | `60-100` |
| `event_cooldown_bars` | `20` |
| `threshold_turnover_penalty_weight` | `0.45` |
| `threshold_turnover_target_ratio` | `0.85` |
| `objective_average_precision_weight` | `0.30` |
| `objective_threshold_score_weight` | `0.60` |
| `objective_brier_penalty_weight` | `0.10` |
| `focal_alpha` | `0.76-0.84` |
| `focal_gamma` | `1.9-2.3` |
| `hard_negative_radius` | `20-40` |
| `hard_negative_multiplier` | `1.6-1.9` |

#### 2. `long_reversal_tcn_1m_v1`

Use the `long_reversal_tcn_v2_20260525_narrow48` and `long_reversal_tcn_v3_20260526_overlap_ny_dd_repair` neighborhoods.

| Parameter | Focused 1m Range |
| --- | --- |
| backend | `torch` |
| model type | `tcn` |
| trials | `48-60` |
| `cv_min_folds` | `3` |
| `max_loaded_features` | `96` |
| `top_feature_min/max` | `24 / 96` |
| `window_min/max` | `90-130` |
| `event_cooldown_bars` | `20-25` |
| `threshold_turnover_penalty_weight` | `0.35-0.45` |
| `threshold_turnover_target_ratio` | `0.85-0.90` |
| `objective_average_precision_weight` | `0.30-0.35` |
| `objective_threshold_score_weight` | `0.55-0.60` |
| `objective_brier_penalty_weight` | `0.10` |
| `focal_alpha` | `0.70-0.77` |
| `focal_gamma` | `2.8-3.1` |
| `hard_negative_radius` | `5` |
| `hard_negative_multiplier` | `2.05-2.20` |
| epochs | `48` with `6/22/12/8` schedule |

### Wave 2: Union OTE family

Only move here after reversal labeling and training look stable.

#### 3. `short_ote_union_tcn_1m_v1`

Use the `short_ote_union_tcn_candidate_20260520` neighborhood.

| Parameter | Focused 1m Range |
| --- | --- |
| backend | `torch` |
| model type | `tcn` |
| trials | `48` |
| `cv_min_folds` | `3` |
| `max_loaded_features` | `128` |
| `top_feature_min/max` | `24 / 128` |
| `window_min/max` | `120-160` |
| `event_cooldown_bars` | `25` |
| `threshold_turnover_penalty_weight` | `0.65` |
| `threshold_turnover_target_ratio` | `0.78` |
| `objective_average_precision_weight` | `0.25` |
| `objective_threshold_score_weight` | `0.65` |
| `objective_brier_penalty_weight` | `0.10` |
| `focal_alpha` | `0.70-0.82` |
| `focal_gamma` | `2.6-3.0` |
| `hard_negative_radius` | `5-15` |
| `hard_negative_multiplier` | `1.7-2.0` |
| epochs | `48` with `6/22/12/8` schedule |

#### 4. `long_ote_union_tcn_1m_v1`

Use the `long_ote_union_tcn_candidate_20260523` neighborhood.

| Parameter | Focused 1m Range |
| --- | --- |
| backend | `torch` |
| model type | `tcn` |
| trials | `48` |
| `cv_min_folds` | `3` |
| `max_loaded_features` | `128` |
| `top_feature_min/max` | `48 / 96` |
| `window_min/max` | `100-140` |
| `event_cooldown_bars` | `25` |
| `threshold_turnover_penalty_weight` | `0.55` |
| `threshold_turnover_target_ratio` | `0.80` |
| `objective_average_precision_weight` | `0.30` |
| `objective_threshold_score_weight` | `0.60` |
| `objective_brier_penalty_weight` | `0.10` |
| `focal_alpha` | `0.70-0.80` |
| `focal_gamma` | `2.7-3.1` |
| `hard_negative_radius` | `5-15` |
| `hard_negative_multiplier` | `2.0-2.4` |
| epochs | `48` with `6/22/12/8` schedule |

### Wave 3: Only after wave 1 and 2 succeed

- continuation-entry specialists
- breakout specialists
- meta-label family refreshes

Breakout should stay out of the first 1-minute production cycle unless it proves it can convert post-training, not just in CV.

---

## Phase 6: Post-Training Evaluation

Use the exact same downstream stack already trusted in this repo:

1. regime slice report
2. threshold policy search
3. walk-forward backtest
4. acceptance-gate review

### Promotion rule

Do not promote a 1-minute model based on CV AP alone.

Promote only when it wins on the same post-training evidence surface used by the current saved leaderboards:

- profitable after costs
- acceptable WFE
- profitable quarter share
- positive composite expectancy share
- largest-trade concentration limits
- drawdown constraints

---

## Command Skeletons

These are the intended artifact paths after the 1-minute refactor is in place.

### Labeling

```powershell
python data/labeling/labeling_engine.py `
  --input data/currency_data/eurusd-1m.csv `
  --output data/labeling/labeled_data/eurusd_1min_ote_labels_full.csv `
  --swings-output data/labeling/labeled_data/eurusd_1min_ote_swings_full.csv `
  --start-date 2019-01-01 `
  --end-date 2026-03-24 `
  --source-timezone GMT-6 `
  --canonical-timezone UTC
```

### Feature generation

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_1min_ote_labels_full.csv `
  --output data/features/eurusd_1min_ote_reversal_core.csv `
  --recipe features/recipes/ote_1m_reversal_core.json
```

```powershell
python -m features.cli build `
  data/labeling/labeled_data/eurusd_1min_ote_labels_full.csv `
  --output data/features/eurusd_1min_ote_union_core.csv `
  --recipe features/recipes/ote_1m_union_core.json
```

### Preprocessing

```powershell
python -m preprocessing prepare `
  data/features/eurusd_1min_ote_reversal_core.csv `
  --output-dir data/prepared/eurusd_1min_ote_reversal_core `
  --scaler none
```

```powershell
python -m preprocessing prepare `
  data/features/eurusd_1min_ote_union_core.csv `
  --output-dir data/prepared/eurusd_1min_ote_union_core `
  --scaler none
```

### Backend attribution

```powershell
python -m preprocessing backend-attribution `
  --prepared-root data/prepared/eurusd_1min_ote_reversal_core `
  --backend xgboost `
  --backend tcn `
  --max-features 96 `
  --attribution-max-rows 50000 `
  --attribution-floor-fraction 0.08 `
  --attribution-cumulative-importance 0.95
```

```powershell
python -m preprocessing backend-attribution `
  --prepared-root data/prepared/eurusd_1min_ote_union_core `
  --backend tcn `
  --max-features 128 `
  --attribution-max-rows 75000 `
  --attribution-floor-fraction 0.08 `
  --attribution-cumulative-importance 0.95
```

### Training

```powershell
.\scripts\run_ote_training.ps1 `
  -PreparedRoot data/prepared/eurusd_1min_ote_reversal_core `
  -OutputRoot models/ote_1min_reversal_wave1 `
  -Backend xgboost `
  short_reversal
```

```powershell
.\scripts\run_ote_training.ps1 `
  -PreparedRoot data/prepared/eurusd_1min_ote_union_core `
  -OutputRoot models/ote_1min_union_wave1 `
  -Backend torch `
  -ModelType tcn `
  long_ote short_ote
```

---

## Recommended Execution Order

1. Build the 1m-to-5m control baseline.
2. Refactor labeling to support true 1-minute base bars.
3. Train `short_reversal_xgb_1m_v1`.
4. Train `long_reversal_tcn_1m_v1`.
5. If reversal wave passes downstream checks, train `short_ote_union_tcn_1m_v1`.
6. Then train `long_ote_union_tcn_1m_v1`.
7. Only after that, revisit continuation specialists and breakout.

---

## Bottom Line

The first 1-minute cycle should be a disciplined migration, not a broad search. The best prior evidence says:

- start with reversal and union OTE families
- copy the winning turnover-aware objective geometry
- scale all time-based bars from 5-minute meaning into 1-minute meaning
- keep backend attribution in the loop
- judge success on post-training robustness, not raw CV AP

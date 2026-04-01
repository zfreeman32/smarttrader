# OTE Post-Training Production Plan

## Purpose

This document is the implementation plan for moving the OTE stack from trained base models to a regime-aware, production-candidate trading system that can be trusted across changing market conditions.

The plan assumes the current codebase, artifacts, and directory layout in this repository as of March 29, 2026.

---

## Current Baseline Snapshot

### Base model status

| Direction | Current leader | Why |
|---|---|---|
| Long OTE | `models/ote_full_tcn_v3_tail/long_ote` | Best overall combination of CV and held-out test performance |
| Short OTE | Mixed: `models/ote_full_xgb_v2/short_ote` and `models/ote_full_tcn_v3_tail/short_ote` | XGBoost more stable in CV, TCN better on held-out test |
| LSTM | Benchmark only | Useful diversity signal, not a primary production candidate |

### Key numbers that drive the plan

| Model | CV mean AP | CV mean event F0.5 | Test AP | Test event F0.5 |
|---|---|---|---|---|
| `ote_full_tcn_v3_tail/long_ote` | 0.5693 | 0.7231 | 0.6987 | 0.8013 |
| `ote_full_xgb_v2/long_ote` | 0.5476 | 0.7068 | 0.6669 | 0.7041 |
| `ote_full_tcn_v3_tail/short_ote` | 0.4975 | 0.6675 | 0.6562 | 0.7878 |
| `ote_full_xgb_v2/short_ote` | 0.5479 | 0.7008 | 0.6330 | 0.7033 |

### Working conclusions

1. TCN is the best long-side base learner.
2. Short-side production selection cannot be finalized without regime slicing.
3. Regime-aware routing is more important than building a weighted ensemble right now.
4. The current ensemble layer in `ict_app/ensemble/ensemble_engine.py` should be treated as future integration code, not as the next production step.

---

## Production Definition of Done

The OTE system is not production-ready until all of the following are true:

1. Direction-by-direction champions exist with clear challenger tracking in a machine-readable registry.
2. Performance is measured by year, session, volatility regime, and trend regime.
3. Thresholds are selected by regime, with explicit abstain rules for low-confidence and hostile conditions.
4. The walk-forward backtest includes spread, slippage, cooldowns, and position sizing with Walk-Forward Efficiency above 0.50.
5. Performance is not carried by one narrow regime or one narrow year.
6. Paper trading over at least 4 weeks confirms that live behavior matches backtest expectations.
7. Model promotion and rollback are documented and repeatable.

---

## Execution Order

1. Freeze champion and challenger baselines in a model registry.
2. Build joinable prediction outputs, deterministic regime labels, and regime-sliced reports.
3. Add regime-conditioned thresholds and a selective abstain policy.
4. Build a regime router for model selection.
5. Backtest the full policy with realistic frictions using walk-forward protocol.
6. Paper trade with champion/challenger monitoring.
7. Only then test stacking or weighted ensemble methods if routing still leaves meaningful performance on the table.

---

## Research Foundation

The design choices in this plan are grounded in published research on forex regime classification, selective prediction, and production deployment of ML trading systems. Key findings are summarized here and referenced throughout the phases.

### Regime detection: deterministic labeling over learned classifiers

For EURUSD 5-minute intraday data, deterministic indicator-threshold regime labeling is preferred over trained classifiers (LSTM, HMM, etc.) for the following reasons:

- Published FX regime work uses ADX thresholds for trend strength and ATR percentiles for volatility as primary discriminants. These features are already computed in the OTE feature set.
- Deterministic labels produce identical outputs given identical inputs, which is critical for backtesting reproducibility and debugging. Heuristic HMM implementations in production trading tools explicitly cite this as an advantage over learned models.
- A trained regime classifier adds an entire model lifecycle (training, calibration, monitoring, retraining) to the production stack. A deterministic labeler adds a single function with no training data dependency.
- The regimes that matter for OTE routing (trend direction, volatility level, session, stress) are directly observable from features the pipeline already computes. There is no hidden state to infer.

HMMs remain a viable secondary method for cross-validation of regime assignments, particularly for volatility clustering detection. Research on EUR/USD specifically shows that 2-3 HMM states capture the dominant regime-switching dynamic, and that models with more states tend to overfit.

### Selective prediction: separating prediction from execution

Research on confidence-threshold trading frameworks shows that separating the directional prediction from the execution decision (trade or abstain) improves risk-adjusted returns compared to traditional all-trade classification. The model produces a calibrated probability; a separate policy layer decides whether the signal is strong enough to act on, conditioned on the current regime. This is the framework used in Phase 2.

### Walk-forward validation: the 78% false-positive rate

Research on walk-forward optimization demonstrates that up to 78% of strategies show negative out-of-sample Sharpe ratios despite positive in-sample results. Walk-Forward Efficiency (WFE), defined as the ratio of out-of-sample returns to in-sample returns, must exceed 0.50 to indicate a strategy that is not overfit. This threshold is enforced in Phase 4.

### Transaction cost dominance in intraday forex

Published comparative analyses of ML strategies for forex directional forecasting find that incorporating institutional-level transaction costs reverses the profitability of several strategies that looked profitable on raw accuracy metrics. The Mean Absolute Directional Loss (MADL) function was proposed specifically to align model optimization with trading profitability rather than classification accuracy. Phase 4 enforces post-cost evaluation.

### Regime overfitting from excessive slicing

Slicing performance by every combination of regime dimensions creates a combinatorial explosion where most cells have too few samples for reliable threshold estimation. The mitigation is to use composite regimes (trend × volatility only, 5-8 buckets maximum) rather than the full cross-product, and to require minimum event counts per bucket before allowing regime-specific thresholds.

### Complexity measures as features

Research on forex volatility forecasting (2024) found that incorporating the Hurst exponent and fuzzy entropy as features significantly enhanced deep learning model accuracy. Rolling Hurst exponent is a strong regime-discriminating feature because it directly measures whether the series is trending (H > 0.5), mean-reverting (H < 0.5), or random walk (H ≈ 0.5).

---

## Regime Taxonomy

Regime labels are computed deterministically from features that already exist in the OTE feature set. No trained model is required.

### Regime families and detection rules

#### Trend regime

| Bucket | Detection rule | Interpretation |
|---|---|---|
| strong_up | ADX > 30 AND EMA alignment > 0.7 | Directional uptrend with conviction |
| weak_up | ADX 20-30 AND close > EMA50 | Mild upward bias |
| ranging | ADX < 20 | No directional conviction |
| weak_down | ADX 20-30 AND close < EMA50 | Mild downward bias |
| strong_down | ADX > 30 AND EMA alignment < -0.7 | Directional downtrend with conviction |

Source features: `adx_14`, `ema_alignment`, `ema_spread_21_50_atr`

#### Volatility regime

| Bucket | Detection rule | Interpretation |
|---|---|---|
| low | ATR percentile rank < 25th over trailing 504 bars | Compression |
| medium | ATR percentile rank 25th-75th | Normal |
| high | ATR percentile rank > 75th | Expansion |

Source features: `atr_14`, `atr_ratio_14_50`, `rolling_vol_20`

#### Session regime

| Bucket | UTC hours | Interpretation |
|---|---|---|
| asia | 22:00-07:00 | Low liquidity, range-bound |
| london | 07:00-12:00 | Directional, liquidity sweeps |
| new_york | 12:00-17:00 | Reversals, FVG fills |
| overlap | 12:00-16:00 | Peak liquidity (subset of NY) |
| off_hours | 17:00-22:00 | Thin markets, avoid trading |

Source features: `hour_sin`, `hour_cos`, `in_london_session`, `in_london_ny_overlap`

#### Event stress regime

| Bucket | Detection rule | Interpretation |
|---|---|---|
| normal | range_shock_20 < 2.0 | Typical conditions |
| elevated | range_shock_20 2.0-3.0 | Heightened volatility |
| high | range_shock_20 > 3.0 | Extreme conditions, consider abstaining |

Source features: `range_shock_20`, `displacement_bullish`, `displacement_bearish`

### Composite regime

The composite regime is the primary key for routing and threshold selection. It is formed by concatenating trend and volatility regimes only:

```
composite_regime = trend_regime + "_" + vol_regime
```

This produces a maximum of 15 cells (5 trend × 3 volatility). Session and stress regimes are applied as separate filters on top of the composite.

Do not use the full cross-product of all four regime families. That creates 225 cells and most will have too few samples to support reliable threshold estimation.

### Minimum data requirements per regime bucket

- 50 positive events minimum before allowing a regime-specific threshold.
- If a bucket has fewer than 50 events, use the global threshold for that direction.
- Flag the bucket as "insufficient data for regime threshold" in reports.

---

## Phase 0: Freeze Baselines and Model Registry

### Goal

Create a stable production starting point before adding moving parts.

### Decisions

| Role | Model path |
|---|---|
| Long champion | `models/ote_full_tcn_v3_tail/long_ote` |
| Long challenger | `models/ote_full_xgb_v2/long_ote` |
| Short champion candidate A | `models/ote_full_xgb_v2/short_ote` |
| Short champion candidate B | `models/ote_full_tcn_v3_tail/short_ote` |
| Benchmark only | `models/ote_full_lstm_v1/long_ote` |

Short champion selection is deferred until Phase 1 regime slicing provides evidence.

### Implementation

Add an OTE-specific model registry that supersedes the older assumptions in `models/best_models.py`.

Registry schema:

```json
{
  "models": [
    {
      "model_id": "string — unique identifier, e.g. long_ote_champion_v1",
      "direction": "long | short",
      "role": "champion | challenger | candidate | benchmark",
      "backend": "xgboost | tcn | lstm",
      "artifact_path": "string — relative path to model directory",
      "cv_mean_ap": "float",
      "cv_mean_event_f05": "float",
      "test_ap": "float",
      "test_event_f05": "float",
      "global_threshold": "float | null — set after Phase 2",
      "regime_thresholds": "object | null — set after Phase 2",
      "calibration_method": "platt | isotonic | none",
      "promotion_date": "ISO date string",
      "promotion_reason": "string — human-readable justification",
      "status": "active | deprecated | candidate"
    }
  ],
  "promotion_rules": {
    "min_cv_splits": 3,
    "min_test_event_f05": 0.65,
    "require_regime_robustness": true,
    "require_post_cost_profitability": true,
    "require_paper_trading_confirmation": true
  }
}
```

### Proposed files

- New: `models/ote_model_registry.json`
- New: `models/ote_model_registry.schema.json`
- New: `models/ote_registry_loader.py`
- Update or deprecate: `models/best_models.py`

### Exit criteria

Every production-facing model path comes from the registry, not from ad hoc hard-coded file picks.

---

## Phase 1: Joinable Predictions, Regime Labels, and Regime Attribution

### Goal

Make existing prediction artifacts analyzable by regime without re-running full training, and resolve the short-side champion question with evidence.

### Why this comes next

Current prepared split CSVs intentionally exclude timestamps, which makes downstream prediction analysis row-index keyed. That is workable for training, but awkward for regime analysis, backtests, and policy debugging.

### Implementation: joinable predictions

Export enough row identity metadata to join `oof_predictions.csv` and `test_predictions.csv` back to source bars:

- Add a `source_row_idx` column to OOF and test prediction exports that maps each prediction back to the row index in the original feature dataset.
- Build a joiner utility that takes a prediction CSV and the original timestamped feature dataset and produces a timestamped prediction table.

### Implementation: deterministic regime labeler

Build a deterministic labeling function (not a trained model) that computes regime labels from features already present in the dataset:

```python
def label_regimes(df):
    """
    Compute regime labels for each bar.
    
    Expects a DataFrame with at minimum:
    - adx_14 or equivalent ADX column
    - ema_alignment or equivalent trend alignment score
    - atr_14 or equivalent ATR column
    - range_shock_20 or equivalent stress indicator
    - a datetime index in UTC
    
    Returns a DataFrame with columns:
    - trend_regime: strong_up | weak_up | ranging | weak_down | strong_down
    - vol_regime: low | medium | high
    - session_regime: asia | london | new_york | overlap | off_hours
    - stress_regime: normal | elevated | high
    - composite_regime: trend_regime + "_" + vol_regime
    """
```

Use the bucket boundaries defined in the Regime Taxonomy section above.

### Implementation: regime-sliced report

For each model × direction, produce a report table with one row per regime bucket:

| Column | Description |
|---|---|
| model_id | From registry |
| direction | long or short |
| regime_key | Composite regime label, or session, or year |
| n_positive | Count of positive events in bucket |
| n_negative | Count of negative samples in bucket |
| ap | Average precision |
| event_f05 | Event F-beta (beta=0.5) |
| event_precision | Event precision |
| event_recall | Event recall |
| optimal_threshold | Threshold maximizing event F0.5 in this bucket |
| ap_ci_lower | 95% bootstrap CI lower bound |
| ap_ci_upper | 95% bootstrap CI upper bound |

Generate this report sliced by: composite regime, session regime, stress regime, and year.

### Feature addition: rolling Hurst exponent

Add rolling Hurst exponent as a feature for future regime discrimination. Compute over 30-bar and 100-bar windows on the close series. This feature directly measures trending (H > 0.5) versus mean-reverting (H < 0.5) behavior and has been shown to improve forex volatility forecasting accuracy.

### Proposed files

- Update: `features/preprocessing.py` (add source_row_idx export)
- Update: `model_training/ote_training/ote_xgboost_pipeline.py` (propagate source_row_idx to predictions)
- New: `model_testing/ote_prediction_joiner.py`
- New: `model_testing/ote_regime_labeler.py`
- New: `model_testing/ote_regime_slices.py`
- New: `scripts/run_ote_regime_slice_report.py`

### Deliverables

- Joined prediction table with timestamps and source row keys.
- Regime-labeled OOF and test prediction tables.
- Report by model, direction, year, session, trend regime, and volatility regime with bootstrap CIs.

### Exit criteria

Can answer: "Which model wins for long and short inside each major composite regime bucket?" with statistical confidence.

---

## Phase 2: Threshold by Regime and Selective Abstain Policy

### Goal

Improve trustworthiness by changing when we trade, not only which model we use. Separate directional prediction from execution decisions.

### Design principle

The model produces a calibrated probability. A separate policy layer decides whether the signal is strong enough to act on. This separation follows the selective classification framework: the model predicts direction; the policy chooses coverage (trade) or abstention (skip) based on confidence and regime context.

### Hard abstain rules

These conditions produce no signal regardless of model output:

| Condition | Rule |
|---|---|
| Event stress | stress_regime = "high" (range_shock > 3.0) |
| Off-hours session | session_regime = "off_hours" (17:00-22:00 UTC) |
| Cooldown | Fewer than 4 bars (20 minutes) since last emitted signal in same direction |
| Insufficient data | Regime bucket flagged as "insufficient data" and no global fallback is above threshold |

### Soft abstain rules: regime-conditioned thresholds

Different regimes require different confidence levels. The starting values below are initial estimates to be refined by the threshold search:

| Composite regime | Long threshold | Short threshold | Rationale |
|---|---|---|---|
| strong_up + low_vol | 0.55 | 0.80 | Long has trend tailwind; short fights the trend |
| strong_up + medium_vol | 0.60 | 0.75 | Moderate directional edge for longs |
| strong_up + high_vol | 0.65 | abstain | High vol + countertrend short is dangerous |
| weak_up + any vol | 0.65 | 0.70 | Mild bias, need reasonable conviction both ways |
| ranging + low_vol | 0.70 | 0.70 | No directional edge, need strong signal |
| ranging + medium_vol | 0.70 | 0.70 | Same |
| ranging + high_vol | 0.75 | 0.75 | Worst conditions, be very selective |
| weak_down + any vol | 0.70 | 0.65 | Mild bias, short has slight edge |
| strong_down + low_vol | 0.80 | 0.55 | Short has trend tailwind; long fights the trend |
| strong_down + medium_vol | 0.75 | 0.60 | Moderate directional edge for shorts |
| strong_down + high_vol | abstain | 0.65 | High vol + countertrend long is dangerous |

### Threshold search procedure

For each composite regime bucket with at least 50 positive events:

1. Grid search threshold from 0.40 to 0.90 in steps of 0.05.
2. At each threshold, compute event F0.5 and post-cost expectancy on OOF predictions.
3. Select threshold maximizing `0.6 * event_f05 + 0.4 * normalized_expectancy`.
4. Reject any threshold that produces fewer than 3 events per month on average.
5. Use leave-one-year-out cross-validation for the threshold search: optimize on all years except one, evaluate on the held-out year, repeat for each year, average.

For buckets with fewer than 50 positive events:

- Fall back to the global threshold for that direction.
- Log the bucket as data-insufficient.

### Minimum pip distance filter

Reject any signal where the expected move (calibrated probability × average favorable excursion for that regime) is less than 2× the spread for the current session.

### Proposed files

- Update: `model_training/ote_training/ote_xgboost_pipeline.py` (propagate regime labels)
- New: `model_testing/ote_threshold_policy.py`
- New: `model_testing/ote_abstain_policy.py`
- New: `scripts/run_ote_threshold_policy_search.py`

### Deliverables

- Policy table: direction × composite regime → threshold, cooldown, abstain flag, minimum event count satisfied.
- Report comparing three policies: global threshold, regime-conditioned threshold, regime-conditioned threshold plus abstain rules.
- Coverage analysis: what percentage of signals does each policy act on versus abstain.

### Exit criteria

The regime-aware threshold policy beats the global threshold policy on event F0.5 and trading expectancy after costs, without collapsing trade counts below 3 per week per direction.

---

## Phase 3: Regime Router for Model Selection

### Goal

Select the best base learner by market state before considering averaging or stacking.

### Design principle

Use deterministic routing rules populated from Phase 1 evidence. Do not average model probabilities. Research shows that simple model selection by regime outperforms probability averaging when the models have different strengths across regimes. Averaging dilutes the specialist advantage.

### Routing logic

```python
def route_model(direction, composite_regime, registry):
    """
    Select model path based on direction and regime.
    
    Rules are populated from the Phase 1 regime slice report.
    Start with deterministic rules. Add a small learned router
    only if deterministic rules prove too brittle.
    """
    if direction == 'long':
        # TCN is the default long champion across all regimes
        # unless Phase 1 evidence shows XGBoost wins specific buckets
        return registry.get_model('long_ote_champion')
    
    elif direction == 'short':
        # Short routing depends on Phase 1 regime slicing results.
        # Placeholder rules — replace with evidence:
        if composite_regime.startswith('strong_down'):
            return registry.get_model('short_ote_trend_specialist')
        elif composite_regime.startswith('ranging'):
            return registry.get_model('short_ote_range_specialist')
        else:
            return registry.get_model('short_ote_default')
```

### Regime transition handling

When regime probabilities are ambiguous (the composite regime label changed within the last 6 bars), apply a conservative override:

- Use the more conservative of the two candidate models.
- Widen the threshold by 0.05 above the regime-specific threshold.
- Reduce position size to 50% of normal.

This addresses the documented pitfall that walk-forward optimization responds to regime changes with a lag.

### Router evaluation protocol

- Compare router policy against best single-model policy and against global threshold.
- Use walk-forward evaluation: 2-year minimum training window, 3-month test window, rolling quarterly.
- Primary metric: post-cost expectancy.
- Secondary metric: coefficient of variation of monthly returns across composite regime buckets (lower is better).

### Proposed files

- New: `model_training/ensemble/ote_regime_router.py`
- New: `model_training/ensemble/ote_router_config.json`
- Update: `model_testing/prediction.py`

### Deliverables

- Router config file with direction × composite regime → model_id mapping.
- Router evaluation report against single-model baselines.

### Exit criteria

Router policy beats or matches the best single-model policy in both walk-forward stability and post-cost expectancy.

---

## Phase 4: Trading Policy Backtest with Frictions

### Goal

Promote from prediction metrics to actual trading-system validation.

### Walk-forward protocol

| Parameter | Value | Rationale |
|---|---|---|
| Training window | Expanding, minimum 2 years | Sufficient history for regime coverage |
| Out-of-sample window | 3 months (1 quarter) | Long enough for statistical relevance |
| Purge gap | max(window_size, 40 bars) | Prevent contamination from temporal context |
| Number of folds | Determined by data length, minimum 8 | Sufficient for reliable WFE estimation |
| Re-optimization | Thresholds re-searched each fold; model weights fixed | Thresholds adapt; base models do not refit per fold |

### Walk-Forward Efficiency requirement

WFE = out-of-sample annualized return / in-sample annualized return

**Require WFE > 0.50.** If WFE is below 0.50, the policy is likely overfit and should not proceed to paper trading.

### Friction model

| Cost component | Specification |
|---|---|
| Spread: London/NY overlap | 1.0 pip |
| Spread: London or NY alone | 1.5 pip |
| Spread: Asia session | 2.5 pip |
| Spread: Off-hours | 3.0 pip |
| Slippage | 0.3 pip per trade |
| Commission | Per broker configuration; default $3.50 per round turn per 100k |
| Cooldown | 4 bars (20 minutes) minimum between signals in same direction |

### Required backtest outputs

**Core metrics:** expectancy per trade, profit factor, max drawdown, max drawdown duration, Sharpe ratio (annualized), Sortino ratio, trades per month, hit rate, average win / average loss ratio.

**Robustness metrics:**

| Metric | Requirement |
|---|---|
| Walk-Forward Efficiency | > 0.50 |
| Quarterly periods profitable | > 60% |
| Composite regime buckets with positive expectancy | > 60% |
| Largest single trade as % of total P&L | < 10% |
| Correlation of monthly returns with buy-and-hold EURUSD | Near zero |
| Herfindahl index of trade-level P&L | Low concentration |

**Breakdown dimensions:** by year, by quarter, by session regime, by composite regime, by signal confidence quintile.

### Proposed files

- New: `model_testing/ote_policy_backtest.py`
- New: `model_testing/ote_policy_metrics.py`
- New: `scripts/run_ote_policy_backtest.py`

### Deliverables

- Full backtest report with all metrics above.
- Regime-level P&L attribution table.
- WFE calculation with per-fold breakdown.
- Equity curve with regime annotations.

### Exit criteria

- Policy remains profitable after costs.
- Does not rely on one single year or regime.
- WFE > 0.50.
- No single trade accounts for more than 10% of total P&L.
- Positive expectancy in at least 60% of composite regime buckets.
- Maximum drawdown < 2× average monthly profit.

---

## Phase 5: Paper Trading Integration

### Goal

Validate real-time behavior of the policy before live capital is at risk.

### Implementation

Integrate the selected OTE policy into the execution stack in simulation mode. Use the existing `ict_app/execution/trading_orchestrator.py` and `ict_app/ensemble/ensemble_engine.py` only after policy selection is settled.

Run both champion and challenger models simultaneously during paper trading. For each signal, record:

- Timestamp, direction, all regime labels
- Champion model prediction, challenger model prediction
- Champion threshold decision, challenger threshold decision
- Regime router selection
- Actual market outcome

### Daily monitoring dashboard

| Metric | Check |
|---|---|
| Signal count by direction and regime | Compare to historical daily average ± 2σ |
| Fill rate | Percentage of signals that would have been executed |
| Skipped trade count and reasons | Categorize by abstain rule that triggered |
| Divergence from offline expectations | Flag if signal distribution deviates from backtest |
| Champion vs. challenger hit rate | Running comparison over trailing 2-week window |
| Calibration quality (ECE) | Binned predicted probability vs. actual hit rate |

### Alert thresholds

| Condition | Action |
|---|---|
| Signal count < 50% of historical daily average for 3 consecutive days | Investigate feature pipeline or regime distribution shift |
| Expected Calibration Error (ECE) > 0.10 | Flag for calibrator refit |
| Champion underperforms challenger by > 5% on event F0.5 over rolling 2-week window | Escalate for champion/challenger swap review |
| Any single-week drawdown exceeding worst backtest quarter | Halt and investigate |

### Proposed files

- Update: `ict_app/execution/trading_orchestrator.py`
- Update: `ict_app/ensemble/ensemble_engine.py`
- New: `ict_app/ensemble/ote_policy_adapter.py`
- New: `scripts/run_ote_paper_trading.py`
- New: `scripts/run_ote_paper_trading_monitor.py`

### Deliverables

- Paper trading mode using selected long and short OTE policies.
- Daily summary of signal count, fills, skipped trades, and divergence from offline expectations.
- Champion vs. challenger comparison log.

### Exit criteria

- Paper trading for minimum 4 weeks.
- Stable signal behavior: signal count within 2σ of backtest expectations.
- No major mismatch with backtest assumptions.
- Calibration ECE < 0.10 throughout paper trading period.
- No single-week drawdown exceeding worst backtest quarter.

---

## Phase 6: Ensemble Only If Needed

### Goal

Test stacking only if regime routing leaves meaningful performance on the table.

### Decision criterion for proceeding

Compute the oracle router improvement: the performance gain if the router always selected the model that turned out to be correct ex-post. If the actual router captures within 90% of the oracle router's performance, do not proceed to ensemble. The router is already extracting most of the available signal, and ensembling will not justify its complexity cost.

Only proceed to ensemble if the gap between actual router and oracle router exceeds 10% relative.

### Important constraint

Do not use the current `model_training/ensemble/stacking_model_train.py` as the direct path for OTE ensembling. It is built around an older workflow and objective.

### Implementation if warranted

- Build OTE-specific ensembling from OOF predictions only.
- Treat regime labels as meta-features.
- Use a simple logistic regression as the meta-learner (not a complex stacking architecture). Regularize with L2, C chosen by cross-validation.
- Compare: best single model, regime router, stacked ensemble.

### Proposed files

- New: `model_training/ensemble/ote_stacking.py`
- New: `model_training/ensemble/ote_meta_features.py`
- New: `scripts/run_ote_stacking_experiment.py`

### Deliverables

- OTE-specific stacking experiment report.
- Promotion rule that requires stacked policy to beat router policy on stability, not just headline AP.

### Exit criteria

Stacked model earns promotion only if it improves regime robustness and trading outcomes, not merely offline classification metrics.

---

## Known Pitfalls and Built-In Mitigations

These pitfalls are documented in published research on ML trading systems. The mitigations are embedded in the phases above, but collected here for reference.

### Regime overfitting

**Pitfall:** Optimizing thresholds per regime bucket when many buckets have too few samples creates false confidence. The full cross-product of regime families (trend × vol × session × stress) produces 225 cells, most with insufficient data.

**Mitigation:** Use composite regimes (trend × vol only, max 15 cells). Require 50 positive events per bucket. Fall back to global threshold when data is insufficient. Use leave-one-year-out CV for threshold search.

### Walk-forward lag on regime transitions

**Pitfall:** Walk-forward optimization adapts to regime changes reactively, not predictively. Performance deteriorates during transitions before parameters can adjust.

**Mitigation:** Phase 3 includes regime transition handling: conservative model selection, widened thresholds, and reduced position size during the 6-bar window after a regime label change.

### Transaction costs swamping signal

**Pitfall:** Models that appear profitable before costs become unprofitable after realistic spreads and slippage. This is the most common failure mode for intraday forex systems.

**Mitigation:** Phase 4 uses session-aware spread modeling and requires post-cost expectancy. Phase 2 includes a minimum pip distance filter that rejects signals where expected move < 2× spread.

### Calibration drift

**Pitfall:** Platt scaling or isotonic calibration fitted on development data degrades over time as market conditions shift.

**Mitigation:** Phase 5 monitors ECE weekly. If ECE > 0.10, the calibrator is refit on recent data without retraining the base model. Retraining cadence (see below) provides a structured schedule.

### Single-regime profitability carrying the result

**Pitfall:** A system that appears profitable overall but derives all profits from one regime or one year. When that condition doesn't appear, the system bleeds money.

**Mitigation:** Phase 4 exit criteria require positive expectancy in at least 60% of composite regime buckets and profitable in at least 60% of quarterly periods.

### Over-abstention collapsing trade count

**Pitfall:** Aggressive abstention from combined regime thresholds, confidence thresholds, and session filters reduces trade count to near zero, making performance statistically unreliable.

**Mitigation:** Phase 2 threshold search requires minimum 3 signals per week per direction. Phase 5 monitors signal count against historical baselines.

### Single-trade concentration

**Pitfall:** A small number of trades (less than 5%) accounting for more than 30% of total profit. The system appears profitable but depends on outlier events that may not recur.

**Mitigation:** Phase 4 requires no single trade to account for more than 10% of total P&L, and computes the Herfindahl index of trade-level P&L.

---

## Retraining and Monitoring Cadence

### Monthly monitoring

- Track calibration quality (ECE, reliability diagram).
- Track signal count distribution against historical baseline.
- Track regime distribution for structural shifts.
- Track champion vs. challenger performance.

### Quarterly retraining trigger

Retrain if any of:

- Calibration ECE exceeds 0.10 for a full quarter.
- Signal quality (event F0.5) drops below backtest expectations for a full quarter.
- Regime distribution shifts significantly (new regime appears that was absent in training data).

### Emergency retraining trigger

Retrain immediately if:

- Out-of-sample event F0.5 drops below 0.60 for two consecutive weeks during live trading.
- Maximum drawdown during live trading exceeds the worst backtest quarter by more than 50%.

### Retraining procedure

1. Retrain base models on expanded dataset (include recent data).
2. Refit calibrator.
3. Re-run regime slice report (Phase 1).
4. Re-run threshold search (Phase 2).
5. Re-evaluate router rules (Phase 3).
6. Re-run backtest (Phase 4).
7. Promote new models only if they pass all promotion rules.

---

## Promotion Rules

No model is promoted directly from headline test AP alone. Promotion requires favorable results on all of:

1. Walk-forward performance (WFE > 0.50).
2. Regime robustness (positive expectancy in at least 60% of composite regime buckets).
3. Post-cost expectancy (profitable after spread, slippage, commission).
4. Paper-trading consistency (4-week minimum with no alert threshold breached).

A challenger replaces a champion only if it improves at least one of:

- Stability across years.
- Stability across regimes.
- Drawdown-adjusted returns.

while not materially degrading the others.

---

## Immediate Ticket Backlog

These are the next concrete tickets to implement in order.

### Ticket 1: OTE model registry

- Add `models/ote_model_registry.json` with the schema specified in Phase 0.
- Add `models/ote_model_registry.schema.json` for validation.
- Add `models/ote_registry_loader.py` that loads registry entries and replaces `best_models.py` lookups.
- Record long champion and short champion candidates.
- **Acceptance criteria:** All model loading in training and inference code reads from registry. No hard-coded model paths remain outside the registry.

### Ticket 2: Joinable predictions

- Add `source_row_idx` column to OOF and test prediction exports in the training pipeline.
- Add `model_testing/ote_prediction_joiner.py` that joins predictions back to timestamped source data.
- **Acceptance criteria:** Can reconstruct timestamped predictions for any trained model without re-running training.

### Ticket 3: Deterministic regime labeler and slice report

- Implement `model_testing/ote_regime_labeler.py` with the indicator-threshold rules from the Regime Taxonomy section.
- Implement `model_testing/ote_regime_slices.py` that generates the per-model performance report.
- Add `scripts/run_ote_regime_slice_report.py` as the entry point.
- Optionally add rolling Hurst exponent (30-bar and 100-bar windows) to the feature set.
- **Acceptance criteria:** Can answer "which model wins for long and short inside each composite regime bucket" with 95% bootstrap confidence intervals.

### Ticket 4: Threshold policy search with abstain rules

- Implement `model_testing/ote_threshold_policy.py` with regime-conditioned threshold search.
- Implement `model_testing/ote_abstain_policy.py` with hard and soft abstain rules.
- Add `scripts/run_ote_threshold_policy_search.py` as the entry point.
- **Acceptance criteria:** Regime-aware policy beats global threshold on event F0.5 AND post-cost expectancy. Trade count remains at or above 3 per week per direction.

### Ticket 5: Router prototype

- Implement `model_training/ensemble/ote_regime_router.py` with deterministic routing rules.
- Populate routing rules from Ticket 3 evidence.
- Evaluate router policy against single-model baselines using walk-forward protocol.
- **Acceptance criteria:** Router WFE > 0.50. Router beats or matches best single-model policy on post-cost expectancy.

---

## Recommended First Sprint

If the goal is the highest-value next sprint, it should be:

1. Add OTE model registry (Ticket 1).
2. Add joinable prediction exports (Ticket 2).
3. Build deterministic regime labeler and slice report (Ticket 3).

That sprint answers the most important unresolved question in the current repo:

> Is the short-side production path best served by XGBoost, TCN, or a regime router between them?

---

## Explicit Non-Goals for Right Now

1. Building a large multi-model weighted ensemble before regime analysis is complete.
2. Training a separate ML regime classifier model (use deterministic labeling instead).
3. Wiring all ICT, confluence, and OTE components together before the OTE policy itself is stable.
4. Optimizing for maximum trade count at the expense of regime robustness.
5. Moving to live capital before paper-trading evidence is collected.
6. Averaging model probabilities across models (use model selection by regime instead).

---

## Appendix: Feature Mapping for Regime Labels

This table maps regime detection rules to specific feature columns expected to exist in the OTE feature dataset. If column names differ in the actual dataset, adjust the regime labeler accordingly.

| Regime rule | Required feature | Fallback if missing |
|---|---|---|
| ADX trend strength | `adx_14` | Compute from close, high, low using 14-period ADX |
| EMA alignment score | `ema_alignment` | Compute from EMA8, EMA21, EMA50 relative positions |
| Close vs EMA50 | `close`, EMA50 | Use `close_vs_ema_50_atr` if available |
| ATR for volatility percentile | `atr_14` | Compute from high, low, close using 14-period ATR |
| Range shock for stress | `range_shock_20` | Compute as current range / rolling 20-bar mean range |
| Session classification | datetime index (UTC) | Derive from `hour_sin`, `hour_cos` if index unavailable |
| Hurst exponent (optional) | Not yet in feature set | Add via `hurst` package or manual R/S computation |
# Production-Ready Real-Time Trading Signal System Plan

## Scope

This document reviews the current `trade_bot` codebase and lays out a phased plan for turning the existing offline OTE research stack into a local, production-ready, 24/7 real-time trading signal system running on a dedicated Windows PC.

The target is a signal system, not an auto-execution engine. The live system should:

- ingest real-time EURUSD market data
- normalize it into the same schema the offline OTE pipeline expects
- compute live features with backtest parity
- load promoted champion models and optional challengers
- emit auditable signals with thresholds and abstain rules
- store every raw input, feature snapshot, prediction, decision, and health event
- drive a local dashboard, alerts, and chart screenshots
- support quarterly retraining and controlled champion-challenger promotion

## Executive Assessment

The repo already has a strong offline OTE research pipeline:

- causal labeling is implemented
- feature engineering is broad and metadata-aware
- preprocessing, backend attribution, model training, threshold search, abstain logic, and walk-forward policy backtests are already in place
- model artifacts and a registry exist
- tests cover most of the offline pipeline

The repo does **not** yet have a true production live-service layer:

- there is no durable real-time ingestion subsystem
- there is no streaming feature engine
- there is no unified live database or audit store
- there is no operational service supervisor/health stack
- the current `ict_app` is a prototype track, not the canonical OTE runtime

The main technical risk for live parity is feature complexity:

- the full OTE feature build currently produces 1,497 columns
- `strategy_signals` alone contributes 947 feature columns
- promoted TCN/XGBoost models depend on a subset of those `strategy__...` columns
- exact live parity is possible, but the strategy feature catalog makes the first live version more fragile than the core handcrafted feature families

The main governance gap is policy packaging:

- model thresholds are stored in `models/ote_model_registry.json`
- abstain filters used by the walk-forward backtests are present in report outputs, not yet promoted into a runtime policy artifact
- production should promote a complete model-plus-policy package, not just the raw model artifact

## Relevant File Inventory

### Root docs, manifests, and runbooks

| Path | Role today | Production use |
| --- | --- | --- |
| `README.md` | General repo overview. | Low. Reference only. |
| `end-to-end-README.md` | Best current OTE runbook for the canonical offline pipeline. | High. Use as the canonical offline workflow reference for retraining automation. |
| `Optimal_Trade_Entry_Detection_Pipeline.docx.txt` | Narrative description of the OTE pipeline. | Medium. Reference for intent and assumptions. |
| `Quantitative_Trading_ML_Pipeline.docx.txt` | Broader project notes. | Medium. Reference only. |
| `requirements.txt` | Broad project dependency list. | Medium. Source for existing packages, but too broad for the live system. |
| `ote_requirements.txt` | Focused OTE pipeline dependency list. | High. Base requirements file to extend for the live stack. |

### Raw data, labeling, and lineage

| Path | Role today | Production use |
| --- | --- | --- |
| `data/README.md` | Data layout documentation. | Medium. Keep updated with live archive layout. |
| `data/currency_data/eurusd-5m.csv` | Canonical offline input: EURUSD 5-minute OHLCV bars. | High. This is the live schema target. |
| `data/currency_data/eurusd-1m.csv` | Additional historical bar data. | Medium. Useful for live replay tests and future bar aggregation checks. |
| `data/currency_data/eurusd-15m.csv` | Historical higher timeframe bars. | Low for v1 live. Not required if live resampling is local. |
| `data/currency_data/eurusd-1h.csv` | Historical higher timeframe bars. | Low for v1 live. Not required if live resampling is local. |
| `data/currency_data/eurusd-1d.csv` | Historical daily bars. | Low for v1 live. Not required if market-day levels are derived locally. |
| `data/currency_data/resample_data.py` | Helper for offline resampling. | Medium. Reuse logic ideas, but build a dedicated live bar aggregator. |
| `data/stock_data/SPY.csv` | Separate research data. | Out of scope for OTE live v1. |
| `data/stock_data/VIX.csv` | Separate research data. | Out of scope for OTE live v1. |
| `data/labeling/README.md` | Labeling module documentation. | High for quarterly retraining. |
| `data/labeling/labeling_engine.py` | Base labeling pipeline and raw data normalization patterns. | High. Reuse timezone/column normalization ideas and for retraining. |
| `data/labeling/ote_labeling_engine.py` | Canonical causal OTE labeling engine with ATR structure and triple-barrier logic. | High. Reuse directly for quarterly retraining and for live outcome attribution jobs. |
| `data/labeling/ote_label_deep_analysis.py` | Offline label diagnostics. | Medium. Useful for retraining review. |
| `data/labeling/ote_label_review_app.py` | Dash/Plotly label review UI. | Medium. Reuse charting/UI patterns for the live dashboard. |
| `data/labeling/labeled_data/*.metadata.json` | Label lineage and timezone metadata. | High. Reproduce this metadata discipline in the live database. |
| `data/labeling/analysis/summary.json` | Offline label analysis artifact. | Medium. Useful as a reporting pattern. |

### Feature engineering core

| Path | Role today | Production use |
| --- | --- | --- |
| `features/README.md` | Feature system overview. | High. Reference for live parity work. |
| `features/config.py` | Feature builder configuration and timezone contract. | High. Reuse and adapt into live feature settings. |
| `features/cli.py` | Offline feature build entrypoint. | High for retraining orchestration. |
| `features/builder.py` | Canonical batch feature builder. | High. Adapt for rolling-window and parity replay. |
| `features/io.py` | Market data loading, standardization, validation, save helpers. | High. Reuse standardization/validation logic in live ingestion. |
| `features/fx_calendar.py` | Timezone normalization, session regime classification, market day/week close labeling. | Very high. Reuse directly in live feature and policy layers. |
| `features/transforms.py` | Lag, rolling stats, z-scores, ATR/sigma normalization, interactions. | Very high. Reuse in live feature computation. |
| `features/registry.py` | Feature family registry. | Very high. Build live feature execution around the same registry boundaries. |
| `features/progress.py` | Progress reporting helpers. | Low for runtime, medium for retraining jobs. |
| `features/preprocessing.py` | Legacy helper module. | Low. Review before reuse. |
| `features/all_indicators.py` | Bulk indicator registry support. | Medium. Reference only. |
| `features/all_strategies.py` | Bulk strategy feature orchestration. | High. Important for current model dependency audit. |
| `features/strategy_registry.py` | Strategy feature registry. | High if current champion features are served as-is. |
| `features/strategy_subprocess_runner.py` | Support utility for strategy generation. | Medium. May help isolate fragile strategy features. |
| `features/strategy_similarity.py` | Strategy comparison tooling. | Low for v1 live, medium for future feature simplification work. |
| `features/feature_list.txt` | Feature catalog reference. | Medium. Useful for audits and manifests. |
| `features/docs/TIMEZONE_CALENDAR_PLAN.md` | Timezone/session design notes. | Medium. Reference only. |
| `features/recipes/ote_base.json` | Smaller OTE feature recipe. | Medium. Candidate for a production-simplified retrain track. |
| `features/recipes/ote_extended.json` | Canonical full OTE feature recipe. | Very high. Current live parity target. |
| `features/recipes/meta_labeling.json` | Meta-label feature recipe. | Low for OTE live v1. |

### Feature families used by the OTE stack

| Path | Role today | Production use |
| --- | --- | --- |
| `features/feature_sets/price_action.py` | Returns, candle anatomy, gap and range expansion features. | High. Direct live reuse. |
| `features/feature_sets/volatility.py` | ATR, rolling volatility, range shock, bands. | High. Direct live reuse. |
| `features/feature_sets/trend.py` | EMA/SMA distances, MACD, alignment. | High. Direct live reuse. |
| `features/feature_sets/momentum.py` | RSI, ROC, stochastic, acceleration. | High. Direct live reuse. |
| `features/feature_sets/volume.py` | Tick-volume style relative volume, imbalance, money flow. | High. Direct live reuse if live feed exposes usable volume/tick count. |
| `features/feature_sets/structure.py` | Prior highs/lows, sweeps, breakout distances, order-block style context. | High. Direct live reuse. |
| `features/feature_sets/htf_context.py` | 30m/1h context and daily/weekly market-level features. | Very high. Must match live resampling exactly. |
| `features/feature_sets/ict_context.py` | Causal ICT state features such as FVG/order-block/swing proximity. | Very high. Must be parity-tested carefully. |
| `features/feature_sets/exhaustion.py` | Divergence, compression, failed continuation. | High. Direct live reuse. |
| `features/feature_sets/microstructure.py` | Spread/impact proxy features. | High. Prefer replacing proxies with live bid/ask-derived values where available. |
| `features/feature_sets/session.py` | Session and killzone features. | Very high. Direct live reuse. |
| `features/feature_sets/temporal_context.py` | Event-age, event clustering, timing regime features. | High. Direct live reuse. |
| `features/feature_sets/fractional_diff.py` | Fractional differencing. | High. Direct live reuse. |
| `features/feature_sets/quality.py` | Data quality flags. | Very high. Reuse as live data quality guardrails. |
| `features/feature_sets/strategy_signals.py` | Bulk wrapper for the large strategy feature catalog. | Very high for current model parity, but also the main live fragility point. |

### Large strategy feature catalog

| Path | Role today | Production use |
| --- | --- | --- |
| `features/strategies/*.py` | Hundreds of strategy-specific feature generators. | High only because current promoted models still depend on them. Long-term goal should be to shrink this dependency surface. |
| `features/strategies/misc/*.py` | Miscellaneous strategy helpers. | Low to medium. Pull in only if selected live features actually require them. |

### Prepared datasets and metadata

| Path | Role today | Production use |
| --- | --- | --- |
| `data/features/eurusd_5min_ote_full.metadata.json` | Canonical feature build metadata; shows 1,497 columns and feature family counts. | Very high. Use as the live parity target and feature manifest source. |
| `data/features/eurusd_5min_ote_2000.metadata.json` | Smaller smoke-test feature metadata. | Medium. Useful for replay tests. |
| `data/prepared/eurusd_5min_ote_full/summary.json` | Canonical prepared-data summary and lineage. | Very high. Use for retraining automation and runtime manifests. |
| `data/prepared/eurusd_5min_ote_full/encoders.json` | Prepared-data encoder info. | Medium. Important if categorical encoding must be mirrored. |
| `data/prepared/eurusd_5min_ote_full/long_ote/features.json` | Final selected feature list for long OTE target. | Very high. Use to build live feature manifests. |
| `data/prepared/eurusd_5min_ote_full/short_ote/features.json` | Final selected feature list for short OTE target. | Very high. Use to build live feature manifests. |
| `data/prepared/eurusd_5min_ote_full/*/report.json` | Per-target preprocessing reports. | High. Useful for runtime assumptions and retraining validation. |
| `data/prepared/eurusd_5min_ote_full/*/backend_attribution_summary_*.json` | Backend-specific attribution summaries. | High for retraining and candidate comparison. |
| `data/prepared/eurusd_5min_ote_2000_smoketest_*/*` | Smaller prepared datasets. | Medium. Excellent for integration tests and live replay tests. |

### Preprocessing and backend attribution

| Path | Role today | Production use |
| --- | --- | --- |
| `preprocessing/config.py` | Preprocessing config model. | High for retraining automation. |
| `preprocessing/cli.py` | Offline prepare entrypoint. | High for retraining automation. |
| `preprocessing/pipeline.py` | Canonical OTE prepared-data build. | Very high for retraining and live feature parity audits. |
| `preprocessing/feature_selection.py` | Low-variance and collinearity filtering. | High for retraining. |
| `preprocessing/feature_importance.py` | Feature importance handling. | Medium. |
| `preprocessing/backend_attribution.py` | Backend-specific ranking attribution and merge logic. | High for retraining automation. |
| `preprocessing/reporting.py` | Prepare-time report generation. | Medium. Reuse reporting patterns. |
| `preprocessing/__main__.py` | Module entrypoint. | Low. |

### OTE training and model artifact generation

| Path | Role today | Production use |
| --- | --- | --- |
| `model_training/ote_training/README.md` | OTE training notes. | Medium. |
| `model_training/ote_training/FULL_PIPELINE_README.md` | Detailed training workflow notes. | High for retraining orchestration. |
| `model_training/ote_training/OTE_TRAINING_WORKFLOW_REPORT.md` | Training report. | Medium. |
| `model_training/ote_training/feature_ranking.py` | Resolves ranked feature files and attribution gates. | High for retraining. |
| `model_training/ote_training/ote_xgboost_pipeline.py` | Core training/evaluation pipeline for XGBoost and torch backends. | Very high. Reuse directly for quarterly retraining. |
| `model_training/ote_training/torch_models.py` | TCN and LSTM classifier definitions. | Very high for live torch inference. |
| `model_training/ote_training/torch_trainer.py` | Torch training loop and evaluation helpers. | High for retraining. |
| `model_training/model_training_utils.py` | Shared training helpers. | Medium. |
| `model_training/model_layers.py` | Shared layers for older model tracks. | Low for OTE live v1. |

### Model registry and promoted artifacts

| Path | Role today | Production use |
| --- | --- | --- |
| `models/README.md` | Registry/artifact notes, including naming drift caveats. | Very high. Use to avoid path confusion. |
| `models/ote_registry_loader.py` | Typed loader and validator for the OTE model registry. | Very high. Use directly in live serving and promotion automation. |
| `models/ote_model_registry.schema.json` | Registry schema. | High. Extend if live policy fields are added. |
| `models/ote_model_registry.json` | Current active/promoted model registry. | Very high. Current source of truth for live champion loading. |
| `models/ote_model_registry_v1_v2_candidates.json` | Candidate comparison registry. | High for shadow-mode challenger loading. |
| `models/ote_full_tcn_v1/long_ote/*` | Active long champion artifact set. Contains `model.pt`, `best_checkpoint.pt`, `scaler.joblib`, `calibrator.joblib`, `model_config.json`, `training_summary.json`. | Very high. Current primary long live artifact. |
| `models/ote_full_tcn_v1/short_ote/*` | Active short champion artifact set. | Very high. Current primary short live artifact. |
| `models/ote_full_xgb_v1/long_ote/*` | Active long challenger artifact set. | High. Use for shadow comparisons or fallback simpler serving path. |
| `models/ote_full_xgb_v1/short_ote/*` | Active short challenger artifact set. | High. |
| `models/ote_full_lstm_v1/long_ote/*` | Benchmark artifact set. | Medium. Shadow-only unless promoted. |
| `models/ote_full_tcn_v2/*`, `models/ote_full_xgb_v2/*`, `models/ote_full_lstm_v2/*` | Candidate v2 artifacts. | High for shadow/challenger testing and future promotion. |

### Post-training evaluation, thresholding, abstain, and backtest

| Path | Role today | Production use |
| --- | --- | --- |
| `model_testing/POST_TRAINING_PRODUCTION_PLAN.md` | Internal guidance saying deterministic regime routing is the preferred next production step. | Very high. Align live plan with this guidance. |
| `model_testing/ote_prediction_joiner.py` | Joins predictions back to source rows via `source_row_idx`. | Very high. Reuse the identity/join logic for live reproducibility. |
| `model_testing/ote_regime_labeler.py` | Deterministic regime labeling from features. | Very high. Reuse directly in the runtime policy layer. |
| `model_testing/ote_regime_slices.py` | Regime-slice reporting and event metrics. | High for shadow-mode and retraining evaluation. |
| `model_testing/ote_threshold_policy.py` | Composite-regime threshold search and policy application. | Very high. Adapt its apply/evaluate logic for live thresholding. |
| `model_testing/ote_abstain_policy.py` | Hard abstain logic and cooldown handling. | Very high. Reuse directly in live signal decisioning. |
| `model_testing/ote_policy_metrics.py` | Policy-level equity and performance metrics. | High. Reuse for live dashboard performance tracking. |
| `model_testing/ote_policy_backtest.py` | Quarterly walk-forward policy backtests. | Very high for retraining and promotion decisions. |
| `model_testing/reports/ote_regime_slices/v1_v2_comparison/SLICE_REPORT_SUMMARY.md` | Best current regime comparison summary. | High. Shows v2 TCN strength. |
| `model_testing/reports/ote_threshold_policies/v1_v2_tcn_focus/THRESHOLD_POLICY_SUMMARY.md` | Threshold search summary. | High. Shows limited incremental edge from new threshold variants. |
| `model_testing/reports/ote_policy_backtests/full_run_v2/run_summary.json` | Most important current walk-forward policy summary. | Very high. Use as baseline live policy expectations. |

### Scripts that define the current canonical offline path

| Path | Role today | Production use |
| --- | --- | --- |
| `scripts/run_eurusd_ote_pipeline.bat` | Canonical full offline pipeline from raw CSV through prepare step. | Very high. Treat this as the current reference orchestration. |
| `scripts/run_lstm_backend_attribution_and_training.bat` | Backend attribution plus training flow. | High for retraining automation. |
| `scripts/merge_ote_feature_rankings.py` | Ranking merge utility. | Medium to high for retraining. |
| `scripts/run_ote_regime_slice_report.py` | Regime reporting entrypoint. | High for retraining/shadow reporting. |
| `scripts/run_ote_threshold_policy_search.py` | Threshold and abstain policy search entrypoint. | Very high for retraining. |
| `scripts/run_ote_policy_backtest.py` | Walk-forward policy backtest entrypoint. | Very high for retraining and promotion. |
| `scripts/run_ote_recent_shap_analysis.py` | Recent explainability utility. | Medium for diagnostics. |

### Existing prototype live-ish app

| Path | Role today | Production use |
| --- | --- | --- |
| `ict_app/README.md` | Prototype live trading app docs. | Low as an architecture base. Reference only. |
| `ict_app/requirements.txt` | Prototype app dependencies. | Low. Do not use as the live OTE source of truth. |
| `ict_app/integration_guide.py` | Illustrative integration script with stale paths and simulated broker examples. | Low. Useful only as a cautionary reference. |
| `ict_app/data/data_utils.py` | Prototype data helpers. | Low to medium. Borrow isolated utilities only if needed. |
| `ict_app/ensemble/ensemble_engine.py` | Prototype ensemble engine. | Low for OTE live v1. The repo's own post-training doc says not to treat this as the next production step. |
| `ict_app/execution/trading_orchestrator.py` | Prototype orchestrator. | Low for OTE live v1. Signal-only system should not build on it. |
| `ict_app/risk/risk_management.py` | Prototype risk logic. | Low for signal-only v1. |
| `ict_app/ict_detection/*.py` | Prototype ICT detector stack. | Low for canonical OTE live v1 because the OTE feature pipeline already has its own causal ICT feature logic. |
| `ict_app/examples/*.py` | Prototype examples. | Low. |

### Adjacent research modules

| Path | Role today | Production use |
| --- | --- | --- |
| `model_training/regime_classifier/*` | Separate trained regime-classifier track. | Medium. Possible future ensemble signal, but not the recommended first production route. |
| `model_training/pattern_validation/*` | Pattern validation model track. | Medium. Optional future side-model. |
| `model_training/mtf_confluence/*` | MTF confluence model track. | Medium. Optional future side-model. |
| `model_training/classification/*` | Older classification tracks. | Low for OTE live v1. |
| `model_training/regression/*` | Older regression tracks. | Low for OTE live v1. |

### Tests already in place

| Path | What it validates | Production use |
| --- | --- | --- |
| `tests/test_labeling_engine.py` | Labeling correctness. | High for retraining safety. |
| `tests/test_features_builder.py` | Feature builder behavior. | Very high for live parity harness. |
| `tests/test_features_preprocessing.py` | Prepare-stage behavior. | High. |
| `tests/test_fx_calendar.py` | Timezone and calendar logic. | Very high for live parity. |
| `tests/test_backend_attribution.py` | Backend attribution logic. | High for retraining. |
| `tests/test_ote_xgboost_pipeline.py` | OTE training pipeline behavior. | Very high for retraining automation. |
| `tests/test_ote_registry_loader.py` | Registry validation/loading. | Very high for live serving. |
| `tests/test_ote_regime_labeler.py` | Deterministic regime labeling. | Very high for live policy parity. |
| `tests/test_ote_regime_slices.py` | Regime report calculations. | High. |
| `tests/test_ote_threshold_policy.py` | Threshold policy logic. | Very high. |
| `tests/test_ote_abstain_policy.py` | Abstain and cooldown logic. | Very high. |
| `tests/test_ote_policy_metrics.py` | Policy metric calculations. | High for dashboard performance tracker. |
| `tests/test_ote_policy_backtest.py` | Walk-forward backtest logic. | Very high for retraining. |
| `tests/test_ote_prediction_joiner.py` | Row identity join logic. | Very high for live auditability. |
| `ict_app/tests/test_ict_detectors.py` | Prototype detector tests. | Low for canonical OTE live v1. |

## Current Canonical Offline Pipeline

The current OTE pipeline is:

1. Start from `data/currency_data/eurusd-5m.csv`.
2. Run `data/labeling/ote_labeling_engine.py` to create causal OTE labels and metadata.
3. Run `features/builder.py` through `features/cli.py` using `features/recipes/ote_extended.json`.
4. Run `preprocessing/pipeline.py` through `preprocessing/cli.py` to create prepared target datasets.
5. Run `preprocessing/backend_attribution.py` and merge rankings if needed.
6. Train TCN, XGBoost, and optional LSTM with `model_training/ote_training/ote_xgboost_pipeline.py`.
7. Slice predictions by regime with `model_testing/ote_regime_slices.py`.
8. Search threshold and abstain policies with `model_testing/ote_threshold_policy.py` and `model_testing/ote_abstain_policy.py`.
9. Run walk-forward policy backtests with `model_testing/ote_policy_backtest.py`.
10. Promote winners into `models/ote_model_registry.json`.

Important current-state facts:

- canonical raw source timezone is `GMT-6`
- canonical internal timezone is `UTC`
- feature/session clock is `America/New_York`
- market close logic uses `America/New_York` at `17:00`
- full feature build currently has 1,497 columns
- `strategy_signals` contributes 947 of those columns
- active long champion is `long_ote_champion_v1` at global threshold `0.75`
- active short champion is `short_ote_candidate_tcn_v1` at global threshold `0.56`
- active registry entries have `abstain_policy: null`, even though the best walk-forward runs used explicit targeted abstain filters

## Backtest Data Sources Mapped To Real-Time Equivalents

| Current offline source | Current use | Real-time equivalent | Production recommendation |
| --- | --- | --- | --- |
| `data/currency_data/eurusd-5m.csv` | Primary 5-minute OHLCV bar source for labeling and features. | Broker/data-provider candle stream or locally aggregated bars from a tick/1-minute stream. | Prefer local aggregation from higher-frequency live data, then store canonical 5-minute bars in the same OHLCV schema. |
| `Volume` column in `eurusd-5m.csv` | Tick-volume style input used by volume features. | Provider tick count, quote count, or native volume field from broker feed. | Store raw provider field and map it to the canonical `volume` column. Document exact semantics. |
| 30m and 1h features derived in `features/feature_sets/htf_context.py` | Higher timeframe structure and EMA context. | Local resampling from the canonical live 5-minute bar store. | Do not subscribe to separate 30m/1h feeds for v1. Derive them locally for parity. |
| Daily/weekly levels from `features/fx_calendar.py` and `htf_context.py` | Rolling market-day and week context. | Local derivation using the same timezone and `17:00 America/New_York` market-close rules. | Derive locally, not from vendor daily candles. |
| Session regime from `features/fx_calendar.py` | Session, overlap, and killzone features. | Local calendar/timezone computation. | Reuse the same calendar code in live. |
| Structural event timing from `features/feature_sets/temporal_context.py` | Event age, clustering, cooldown-related features. | Local derivation from live-updated structural features. | Recompute locally. No external event feed is needed. |
| Spread assumptions in `model_testing/ote_abstain_policy.py` and backtests | Cost-aware abstain and policy evaluation. | Live bid/ask spread capture from broker feed, with session-based fallback. | Prefer real bid/ask snapshots; keep the current session spread table only as fallback. |
| `data/stock_data/SPY.csv`, `data/stock_data/VIX.csv` | Other research tracks. | Separate live market data if those features are ever promoted. | Out of scope for OTE live v1. |

## Recommended Production Architecture

The live system should stay Python-first and reuse the existing OTE logic wherever possible.

Text architecture:

```text
Live FX Data Feed / Broker API
    |
    v
Ingestion Connectors
    - stream listener
    - REST backfill client
    - heartbeat and reconnect logic
    |
    v
Canonical Normalizer
    - timestamp normalization
    - OHLCV schema mapping
    - bid/ask + spread capture
    |
    v
Local Bar Store
    - raw events
    - 1m bars if available
    - canonical 5m bars
    - derived 30m / 1h bars
    |
    +--------------------------+
    |                          |
    v                          v
Streaming Feature Engine   Health / Gap Monitor
    |                          |
    v                          v
Feature Snapshot Store     System Health Store
    |
    v
Model Runtime
    - champion loaders
    - challenger shadow loaders
    - calibration
    - multi-model voting if enabled
    |
    v
Regime + Threshold + Abstain Layer
    |
    v
Signal Decision Journal
    |
    +-------------------------------+-----------------------------+
    |                               |                             |
    v                               v                             v
Dashboard API / UI              Alerts Service               Screenshot Service
    |                               |                             |
    v                               v                             v
Local Web Dashboard            Email / SMS logs             PNG archive + attachments

Cold Path:
SQLite/WAL operational store --> Parquet archive --> Quarterly retrain pipeline
Historical CSV + archived live data --> existing labeling/features/preprocessing/training/testing --> challenger evaluation --> registry/policy promotion
```

## Recommended Production Folder Structure

```text
ote_live/
  __init__.py
  service.py
  config.py
  contracts/
    market_data.py
    feature_snapshot.py
    prediction.py
    signal.py
    health.py
  ingestion/
    base.py
    connector_stream.py
    connector_backfill.py
    normalizer.py
    aggregator.py
    gap_detector.py
    heartbeat.py
  features/
    manifest.py
    runtime_state.py
    incremental_engine.py
    strategy_adapter.py
    warmup.py
    parity_replay.py
  models/
    registry.py
    loaders.py
    runners.py
    calibrators.py
    ensemble.py
  policies/
    regime.py
    threshold.py
    abstain.py
    decision_engine.py
  storage/
    db.py
    schema.sql
    repositories.py
    archive.py
    retention.py
  dashboard/
    app.py
    queries.py
    charts.py
  alerts/
    emailer.py
    sms.py
    throttling.py
  media/
    chart_capture.py
    annotate.py
  ops/
    logging.py
    health.py
    startup.py
    shutdown.py
    supervisor_notes.md
  retraining/
    pipeline.py
    evaluator.py
    promotion.py
    scheduler.py
  tests/
    test_ingestion.py
    test_feature_parity.py
    test_model_runtime.py
    test_storage.py
    test_dashboard.py
    test_alerts.py
    test_retraining.py

runtime/
  db/
  logs/
  screenshots/
  archive/
  heartbeat/
```

## Recommended Runtime Database Design

Use SQLite in WAL mode for the hot operational store, then archive old partitions to Parquet.

Recommended tables:

| Table | Purpose | Minimum key fields |
| --- | --- | --- |
| `raw_market_events` | Raw feed messages or normalized quote/tick payloads. | `event_id`, `provider`, `asset`, `event_time_utc`, `payload_json` |
| `bars_1m` | Optional intermediate bar layer. | `asset`, `bar_end_time_utc`, `open`, `high`, `low`, `close`, `volume` |
| `bars_5m` | Canonical live bar layer matching offline schema. | `asset`, `bar_end_time_utc`, `open`, `high`, `low`, `close`, `volume`, `bid`, `ask`, `spread_pips` |
| `bars_30m` | Derived higher timeframe bars. | `asset`, `bar_end_time_utc`, OHLCV fields |
| `bars_1h` | Derived higher timeframe bars. | `asset`, `bar_end_time_utc`, OHLCV fields |
| `feature_snapshots` | Stored live feature vectors for every evaluated bar. | `feature_snapshot_id`, `asset`, `bar_end_time_utc`, `model_feature_manifest_id`, `features_json`, `quality_flags_json` |
| `model_predictions` | Raw model outputs before final policy decision. | `prediction_id`, `feature_snapshot_id`, `model_id`, `model_version`, `probability`, `raw_output_json` |
| `signal_decisions` | Final emitted or abstained signal record. | `signal_id`, `feature_snapshot_id`, `prediction_id`, `direction`, `decision`, `threshold`, `abstain_reason`, `regime_json` |
| `signal_outcomes` | Ex-post paper performance attribution for dashboard tracking. | `signal_id`, `outcome_status`, `gross_pnl_pips`, `net_pnl_pips`, `exit_time_utc` |
| `notifications` | Alert send history and cooldown tracking. | `notification_id`, `signal_id`, `channel`, `status`, `sent_time_utc` |
| `chart_artifacts` | Screenshot metadata and file paths. | `artifact_id`, `signal_id`, `file_path`, `created_time_utc`, `annotation_json` |
| `feed_health` | Feed connectivity, latency, heartbeat, and gap events. | `health_event_id`, `component`, `status`, `metric_json`, `event_time_utc` |
| `service_heartbeats` | Process liveness. | `service_name`, `heartbeat_time_utc`, `status_json` |
| `retrain_runs` | Quarterly retraining history. | `run_id`, `started_at_utc`, `completed_at_utc`, `status`, `config_json` |
| `promotion_decisions` | Champion-challenger results and rollbacks. | `decision_id`, `run_id`, `candidate_model_id`, `incumbent_model_id`, `decision`, `reason_json` |

Reproducibility rule:

- every emitted signal must link back to a stored `bars_5m` row, a `feature_snapshot_id`, a `model_id`, a threshold policy, and a calibrator version

Retention policy:

- keep `raw_market_events` and high-frequency intermediate bars in SQLite for 30 to 90 days, then archive to monthly Parquet files
- keep `bars_5m`, `feature_snapshots`, `model_predictions`, `signal_decisions`, and `promotion_decisions` indefinitely
- keep screenshots locally for 180 days, then archive/compress by month unless manually pinned
- keep logs in rolling files for 30 days hot, then zip monthly archives

## Recommended Signal Schema

Each final signal record should contain:

- `signal_id`
- `asset`
- `bar_end_time_utc`
- `direction` such as `long_ote` or `short_ote`
- `decision` as `emit`, `abstain`, or `candidate_only`
- `model_id`
- `model_backend`
- `model_artifact_hash` or artifact version stamp
- `feature_snapshot_id`
- `probability_raw`
- `probability_calibrated`
- `threshold_type` as `global` or `regime`
- `threshold_value`
- `composite_regime`
- `session_regime`
- `stress_regime`
- `abstain_reason`
- `cooldown_remaining_bars`
- `spread_pips`
- `quality_flags_json`
- `decision_metadata_json`

Recommendation:

- create a promoted `live_policy.json` per model or extend the registry schema so a live deployment always loads a complete package: model artifact path, calibrator, selected feature names, context rows, thresholds, abstain filters, and expected cost assumptions

## Dependencies To Add

Base on top of `ote_requirements.txt`.

Recommended additions for the live system:

- `pydantic` for runtime configs and message contracts
- `sqlalchemy` and `alembic` for the local operational database and migrations
- `httpx` and `websockets` for provider connectors
- `tenacity` for reconnect/backoff logic
- `apscheduler` for archival and quarterly retrain scheduling
- `pyarrow` for Parquet archival
- `fastapi` and `uvicorn` if you want a health/API layer in addition to Dash
- `dash` and `plotly` are already aligned with the repo and should remain the dashboard choice
- `playwright` for screenshot capture
- `Pillow` for PNG annotation
- `python-dotenv` for local secrets/config
- `twilio` if SMS delivery is used
- `pytest-asyncio` and `freezegun` for live-service testing

Use built-ins where possible:

- `sqlite3` is built into Python
- `smtplib` is built into Python, so email can be done without a third-party dependency if plain SMTP is acceptable

## Order Of Implementation

1. Phase 0: freeze runtime manifests and feature dependencies.
2. Phase 1: build live ingestion and canonical bar storage.
3. Phase 2: build streaming features and parity tests.
4. Phase 3: build model serving and policy decisioning.
5. Phase 4: build the audit database and archive workflows.
6. Phase 5: build dashboard, alerts, and screenshots.
7. Phase 6: add supervision, health, restart, and failure handling.
8. Phase 7: run shadow mode on live paper data and prove parity.
9. Phase 8: automate quarterly retraining and controlled promotion.

Dependency notes:

- Phase 2 depends on Phase 1.
- Phase 3 depends on Phase 2.
- Phase 4 should start during Phase 1 and complete before Phase 5.
- Phase 5 depends on Phase 3 and Phase 4.
- Phase 6 wraps all earlier runtime phases.
- Phase 7 requires Phases 1 through 6.
- Phase 8 can be implemented earlier, but it only becomes useful after live data begins accumulating.

## Phased Development Plan

### Phase 0: Freeze Runtime Contracts And Live Manifests

Objective:

- define exactly what the live runtime must reproduce before building any live service code

Reuse or adapt:

- `models/ote_registry_loader.py`
- `models/ote_model_registry.json`
- `models/ote_model_registry_v1_v2_candidates.json`
- `data/prepared/eurusd_5min_ote_full/summary.json`
- `data/prepared/eurusd_5min_ote_full/long_ote/features.json`
- `data/prepared/eurusd_5min_ote_full/short_ote/features.json`
- `models/ote_full_tcn_v1/*/model_config.json`
- `models/ote_full_tcn_v1/*/training_summary.json`
- `models/ote_full_tcn_v2/*/model_config.json`
- `models/ote_full_tcn_v2/*/training_summary.json`
- `model_testing/reports/ote_policy_backtests/full_run_v2/run_summary.json`

New modules/files:

- `ote_live/contracts/market_data.py`
- `ote_live/contracts/feature_snapshot.py`
- `ote_live/contracts/prediction.py`
- `ote_live/contracts/signal.py`
- `ote_live/features/manifest.py`
- `ote_live/models/registry.py`
- `scripts/export_live_runtime_manifests.py`

Responsibilities:

- export one manifest per deployed model containing selected feature names, context rows/window size, scaler path, calibrator path, thresholds, abstain config, timezone contract, and source lineage

Dependencies:

- `pydantic`

Key deliverables:

- `live_runtime_manifest_long.json`
- `live_runtime_manifest_short.json`
- a versioned `live_policy.json` format

Testing strategy:

- unit test registry-to-manifest export
- unit test that every selected feature named in model configs exists in the canonical feature metadata
- fail-fast test if a promoted live policy is missing abstain fields or context rows

Exit criteria:

- a single command can tell you exactly what fields, windows, thresholds, and artifacts live serving requires

### Phase 1: Real-Time Ingestion And Canonical Bar Store

Objective:

- build a resilient live data collector that turns provider events into the exact canonical bar schema expected by the offline stack

Reuse or adapt:

- `features/io.py`
- `features/fx_calendar.py`
- `data/labeling/labeling_engine.py`
- `data/currency_data/resample_data.py`
- timezone assumptions encoded in `scripts/run_eurusd_ote_pipeline.bat`

New modules/files:

- `ote_live/ingestion/base.py`
- `ote_live/ingestion/connector_stream.py`
- `ote_live/ingestion/connector_backfill.py`
- `ote_live/ingestion/normalizer.py`
- `ote_live/ingestion/aggregator.py`
- `ote_live/ingestion/gap_detector.py`
- `ote_live/ingestion/heartbeat.py`
- `ote_live/storage/db.py`
- `ote_live/storage/schema.sql`

Responsibilities:

- accept stream events from the chosen feed or broker API
- normalize columns to `timestamp`, `open`, `high`, `low`, `close`, `volume`
- optionally store bid, ask, and spread fields
- aggregate to canonical 5-minute bars if the provider sends ticks or 1-minute data
- derive 30-minute and 1-hour bars locally
- detect missing bars, stale feed heartbeats, and time jumps
- request REST backfill for detected gaps

Dependencies:

- `httpx`
- `websockets`
- `tenacity`
- `sqlalchemy` or built-in `sqlite3`

Testing strategy:

- unit tests for schema normalization and timestamp conversion
- unit tests for 5-minute aggregation from 1-minute or tick inputs
- integration tests that replay `data/currency_data/eurusd-1m.csv` into the collector and verify the resulting 5-minute bars match the historical reference
- fault-injection tests for disconnects, duplicate events, out-of-order events, and gap backfills

Exit criteria:

- the system can reconnect automatically, detect feed stalls, backfill gaps, and produce canonical 5-minute bars continuously

### Phase 2: Streaming Feature Engine With Batch Parity

Objective:

- reuse the existing feature code while making it safe and deterministic for live incremental computation

Reuse or adapt:

- `features/builder.py`
- `features/config.py`
- `features/registry.py`
- `features/transforms.py`
- `features/fx_calendar.py`
- `features/recipes/ote_extended.json`
- `features/feature_sets/*.py`
- `features/all_strategies.py`
- `features/strategy_registry.py`
- `tests/test_features_builder.py`
- `tests/test_fx_calendar.py`

New modules/files:

- `ote_live/features/runtime_state.py`
- `ote_live/features/incremental_engine.py`
- `ote_live/features/strategy_adapter.py`
- `ote_live/features/warmup.py`
- `ote_live/features/parity_replay.py`
- `ote_live/tests/test_feature_parity.py`

Design recommendation:

- v1 live should support two feature execution modes: `core_incremental` for efficient stateful computation of the core handcrafted feature families, and `rolling_batch_adapter` to recompute only the selected feature subset over a bounded trailing window for fragile `strategy__...` features

This gets you to shadow mode faster without rewriting 947 strategy features as bespoke state machines.

Longer-term recommendation:

- retrain a production-focused champion on a smaller, stable feature set so the live stack is not permanently coupled to the full strategy catalog

Warm-up guidance:

- feature recipe warm-up is currently 250 rows
- regime labeling uses longer rolling context, including a 504-bar ATR percentile window
- TCN sequence models need context rows, currently 23 to 27 for the active v1 champions and 19 for at least one strong v2 candidate
- live warm-up should therefore be manifest-driven and conservative; do not emit signals until all required feature values and policy inputs are valid

Dependencies:

- no major new dependency beyond Phase 1
- optional `numba` later if performance becomes a bottleneck

Testing strategy:

- replay historical CSV through the live feature engine and compare row-by-row with `features/builder.py` output
- assert no look-ahead by verifying that live features at bar `t` equal offline features at the same finalized bar `t`
- unit tests per feature family for warm-up behavior and NaN handling
- parity tests for selected model feature subsets only, because those are the fields that matter for live decisions

Exit criteria:

- the live feature engine matches the offline feature output within agreed numeric tolerances for every selected model feature and policy input

### Phase 3: Model Serving, Thresholding, Abstain, And Signal Generation

Objective:

- load champion models and make deterministic, auditable signal decisions on each finalized live bar

Reuse or adapt:

- `models/ote_registry_loader.py`
- promoted artifact folders in `models/ote_full_*`
- `model_training/ote_training/torch_models.py`
- `model_testing/ote_regime_labeler.py`
- `model_testing/ote_threshold_policy.py`
- `model_testing/ote_abstain_policy.py`
- `model_testing/ote_prediction_joiner.py`
- `tests/test_ote_registry_loader.py`
- `tests/test_ote_regime_labeler.py`
- `tests/test_ote_threshold_policy.py`
- `tests/test_ote_abstain_policy.py`

New modules/files:

- `ote_live/models/loaders.py`
- `ote_live/models/runners.py`
- `ote_live/models/calibrators.py`
- `ote_live/models/ensemble.py`
- `ote_live/policies/regime.py`
- `ote_live/policies/threshold.py`
- `ote_live/policies/abstain.py`
- `ote_live/policies/decision_engine.py`

Responsibilities:

- load active champions from the registry
- optionally load challengers in shadow mode
- build the exact ordered feature vector expected by each artifact
- apply scaler and calibrator
- run per-model inference
- compute deterministic regimes
- apply global or regime thresholds
- apply abstain filters and cooldown
- emit a final signal object and persist it

Signal behavior recommendation:

- treat each direction as a separate model decision stream
- store both `candidate` and `final` decisions
- allow multi-model shadow operation from day one
- keep ensemble logic optional in v1; champion plus challenger shadow is more important than early voting complexity
- Exclude long_ote_xgb_v1_candidate from live-eligible runtime use for now, it was trained on future looking feature and is invalid
  
Dependencies:

- none beyond earlier phases unless you add a dedicated serving API

Testing strategy:

- offline replay test: load stored feature snapshots from prepared data and confirm live runtime predictions match `test_predictions.csv`
- unit tests for feature ordering, scaling, calibration, and threshold application
- integration tests for long and short champion loading
- regression tests for abstain/cooldown behavior using known policy examples

Exit criteria:

- given a stored feature snapshot, the runtime can reproduce the exact prediction and signal decision path

### Phase 4: Audit Store, Retention, And Reproducibility

Objective:

- make every signal fully reproducible from stored local data

Reuse or adapt:

- metadata and lineage patterns from `features/builder.py`
- metadata and lineage patterns from `preprocessing/pipeline.py`
- `model_testing/ote_prediction_joiner.py`
- `data/prepared/*/summary.json`

New modules/files:

- `ote_live/storage/repositories.py`
- `ote_live/storage/archive.py`
- `ote_live/storage/retention.py`
- `ote_live/storage/migrations/*`

Responsibilities:

- persist all raw events, bars, feature snapshots, predictions, decisions, notifications, screenshots, and health events
- version manifests and artifact references
- archive cold data to Parquet
- support point-in-time replay of any signal

Dependencies:

- `pyarrow`
- `alembic`

Testing strategy:

- unit tests for insert/query round trips
- integration tests that reconstruct a signal from database rows only
- archive/restore test that verifies monthly Parquet snapshots can be reloaded without losing signal lineage

Exit criteria:

- every emitted signal can be regenerated from stored bars, feature JSON, model manifest, and policy config

### Phase 5: Live Dashboard, Alerts, And Chart Screenshots

Objective:

- give the live system an operator surface that is useful during continuous local operation

Reuse or adapt:

- `data/labeling/ote_label_review_app.py` for Dash/Plotly chart patterns
- `model_testing/ote_policy_metrics.py` for performance summaries
- existing `plotly` and `dash` dependencies already aligned with the repo

New modules/files:

- `ote_live/dashboard/app.py`
- `ote_live/dashboard/queries.py`
- `ote_live/dashboard/charts.py`
- `ote_live/alerts/emailer.py`
- `ote_live/alerts/sms.py`
- `ote_live/alerts/throttling.py`
- `ote_live/media/chart_capture.py`
- `ote_live/media/annotate.py`

Responsibilities:

- show a live price chart with long/short signal overlays
- show model confidence over time
- show system health status, feed freshness, and last prediction time
- show historical signal outcomes and rolling paper performance
- send alerts by email and SMS
- enforce alert thresholds and cooldowns
- capture a PNG of the current chart when a signal fires
- annotate the PNG with direction, confidence, threshold, and timestamp

Technology recommendation:

- use Dash for the first dashboard
- it matches the existing Python stack and the existing label review app
- if you later want a cleaner service boundary, put a small FastAPI layer in front of the database and keep Dash as the UI

Dependencies:

- `playwright`
- `Pillow`
- `twilio` if SMS is used
- `fastapi` and `uvicorn` only if you add an API/health layer beyond Dash

Testing strategy:

- unit tests for notification throttling and payload rendering
- snapshot tests for chart query outputs
- integration tests that generate a signal, render the dashboard state, capture a screenshot, and verify the file path is stored
- manual operator acceptance testing for readability and failure messaging

Exit criteria:

- operators can inspect live state, receive notifications, and open a screenshot for any emitted signal

### Phase 6: Reliability, Supervision, Health, And Recovery

Objective:

- make the live system suitable for a dedicated Windows PC running 24/7

Reuse or adapt:

- lifecycle ideas from the prototype `ict_app/execution/trading_orchestrator.py`, but do not use it as the base runtime

New modules/files:

- `ote_live/ops/logging.py`
- `ote_live/ops/health.py`
- `ote_live/ops/startup.py`
- `ote_live/ops/shutdown.py`
- `ote_live/ops/supervisor_notes.md`
- `scripts/run_live_signal_service.ps1`

Responsibilities:

- structured logs with rotation
- health endpoint or heartbeat file
- startup recovery from the last stored bar and open gaps
- graceful shutdown without corrupting the SQLite store
- disk-space monitoring and error alerts
- process supervision and auto-restart

Windows-specific recommendation:

- use WinSW or NSSM to run the live service and dashboard as Windows services
- use Task Scheduler only as a backup launcher, not the primary supervisor
- do not plan around `systemd` for this machine

Dependencies:

- no mandatory Python dependency beyond logging stack choice
- optional `structlog` if you want structured JSON logs more easily

Testing strategy:

- kill-and-restart integration test that proves the service resumes from the last committed bar
- synthetic disk-full and stale-feed alert tests
- long soak test on replay data for at least several market sessions

Exit criteria:

- the system recovers cleanly from disconnects, process restarts, and stale-feed conditions without losing auditability

### Phase 7: Shadow Mode On Live Paper Data

Objective:

- run the full signal system in production conditions before any real operational use

Reuse or adapt:

- `model_testing/ote_policy_metrics.py`
- `model_testing/ote_regime_slices.py`
- `model_testing/ote_policy_backtest.py`
- `model_testing/reports/ote_policy_backtests/full_run_v2/run_summary.json`
- `models/ote_model_registry_v1_v2_candidates.json`

New modules/files:

- `ote_live/shadow/reporting.py`
- `ote_live/shadow/daily_parity_job.py`
- `ote_live/shadow/comparison_report.py`

Shadow mode design:

- run active champions as the only signal-emitting models
- run challengers in parallel as shadow predictions
- record all predictions and compare them daily
- send notifications either to a test-only channel or with a clear `[SHADOW]` prefix
- compute paper outcomes using the same cost assumptions as the walk-forward backtests

Minimum shadow-mode milestones:

- 14 to 30 calendar days of uninterrupted runtime
- zero unresolved feed gaps
- feature parity pass rate above 99.9 percent on evaluated bars
- no unexplained divergence between live inference replay and offline artifact replay
- all signals reproducible from the local database
- dashboard, screenshot, and notification paths exercised end-to-end

Testing strategy:

- daily automated parity diffs between live feature snapshots and offline recompute on the same stored bars
- daily shadow summary comparing champion vs challenger hit rate, event precision, recall proxy, and paper PnL
- manual review of every service error alert during the milestone window

Exit criteria:

- you trust the live system operationally even before trusting any given model economically

### Phase 8: Quarterly Retraining, Champion-Challenger Promotion, And Rollback

Objective:

- automate the same research-grade evaluation flow used today so live deployment remains governed instead of ad hoc

Reuse or adapt:

- `data/labeling/ote_labeling_engine.py`
- `features/cli.py`
- `features/builder.py`
- `preprocessing/cli.py`
- `preprocessing/pipeline.py`
- `preprocessing/backend_attribution.py`
- `model_training/ote_training/feature_ranking.py`
- `model_training/ote_training/ote_xgboost_pipeline.py`
- `model_testing/ote_regime_slices.py`
- `model_testing/ote_threshold_policy.py`
- `model_testing/ote_abstain_policy.py`
- `model_testing/ote_policy_backtest.py`
- `models/ote_registry_loader.py`
- existing scripts in `scripts/`

New modules/files:

- `ote_live/retraining/pipeline.py`
- `ote_live/retraining/evaluator.py`
- `ote_live/retraining/promotion.py`
- `ote_live/retraining/scheduler.py`
- `scripts/run_quarterly_retrain.py`

Responsibilities:

- merge archived live data with the historical base dataset
- rerun labeling, features, prepare, attribution, training, regime slices, threshold search, and walk-forward policy backtests
- compare challenger vs incumbent using predefined criteria
- only promote if the challenger passes offline metrics, post-cost policy backtests, regime robustness checks, and the shadow confirmation requirement from the registry rules
- keep prior champion artifacts and policy manifests for rollback

Promotion recommendation:

- update the registry schema or add a parallel policy registry so promotion writes a full deployable package containing the model artifact, calibrator, feature manifest, threshold table, abstain config, and deployment notes

Dependencies:

- `apscheduler`
- any training dependencies already used by the offline OTE stack

Testing strategy:

- dry-run retrain in a staging output root
- regression test that promotion writes valid registry entries and never mutates prior artifacts in place
- rollback test that switches the live service back to the previous champion manifest cleanly

Exit criteria:

- quarterly retraining is a reproducible job, not a manual one-off research exercise

## Recommended First Production Scope

For the first live release, keep the scope narrow:

- one asset: EURUSD
- one canonical timeframe: 5-minute bars
- no auto-execution
- one active champion per direction
- one or two challengers in shadow mode
- one local operational DB
- one local dashboard
- email required, SMS optional

This gets you to a safe, auditable signal platform faster.

## Biggest Risks And How To Handle Them

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Strategy feature sprawl | Current champions depend on `strategy__...` columns from a very large catalog. | Start with a rolling batch adapter for selected strategy features, then retrain a production-focused model with a leaner feature set. |
| Policy packaging gap | Backtest-targeted abstain filters are not yet first-class deployment artifacts. | Promote a complete `live_policy.json` alongside model artifacts and extend registry validation. |
| Feed semantics mismatch | Live volume/spread semantics may differ from the historical CSV. | Normalize and document provider semantics; prefer storing raw plus canonical mapped fields. |
| Timezone/session drift | Session features and market-day labels are highly timezone-sensitive. | Reuse `features/fx_calendar.py` directly and keep absolute UTC timestamps everywhere in storage. |
| Restart gaps | A 24/7 service that misses bars can silently invalidate signals. | Persist raw events and bars immediately, detect missing bar IDs, and run automatic REST backfill on startup and during runtime. |
| Dashboard or screenshot fragility | Operator tooling often breaks after signal logic is already live. | Build end-to-end tests that generate a signal and verify UI, screenshot, and alert side effects together. |

## Recommended Next Actions

1. Build Phase 0 first and export live manifests for the current v1 champions plus the strongest v2 candidates.
2. Decide whether v1 live should serve the current strategy-heavy champions as-is or whether you want a near-term retrain on a smaller feature surface.
3. Implement Phases 1 through 3 as one vertical slice with replay-based parity tests.
4. Add Phase 4 storage and then Phase 5 operator tooling.
5. Run Phase 7 shadow mode before any real operational dependence on the signals.

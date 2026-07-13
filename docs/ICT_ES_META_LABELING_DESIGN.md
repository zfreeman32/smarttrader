# ICT / Smart Money Concepts as ML Signal Foundation

## Design Paper: ICT-Driven Meta-Labeling Pipeline for ES 5-Minute Futures

This document is a repo-audit-based design for a new ES-first ICT / Smart Money Concepts meta-labeling pipeline. It is intentionally implementation-guiding, but it does not change code, training scripts, registries, or existing production behavior.

Audit basis for this design included the current FRVP ES stack, the older EURUSD OTE stack, the generic feature/preprocessing/training/testing layers, and the current prototype ICT artifacts. Representative files reviewed include:

- `frvp/pipelines/es_primary_phase04.py`
- `data/labeling/frvp_labeling_engine.py`
- `frvp/sessions/equity.py`
- `frvp/calendars/macro.py`
- `frvp/continuity/continuous_contract.py`
- `frvp/config/instruments.py`
- `features/feature_sets/ict_context.py`
- `features/feature_sets/structure.py`
- `features/feature_sets/session.py`
- `features/feature_sets/htf_context.py`
- `preprocessing/pipeline.py`
- `preprocessing/backend_attribution.py`
- `model_training/ote_training/ote_xgboost_pipeline.py`
- `model_training/ote_training/torch_trainer.py`
- `model_testing/ote_threshold_policy.py`
- `model_testing/ote_policy_backtest.py`
- `model_testing/evaluation_costs.py`
- `ict_app/ict_detection/*.py`

## 1. Executive Summary

The proposed system is an ES futures meta-labeling pipeline where deterministic ICT / SMC rules generate candidate trade events and ML decides whether those rule-based events are worth taking. The operating question is not "where is an OTE?" and not "predict the next bar." It is:

> Given that an ICT / SMC setup fired on ES, should the system take the trade?

This differs from the current FRVP ES pipeline in one important way: FRVP uses volume-profile-driven context and FRVP-specific setup families, while the new system will use ICT structural and liquidity concepts as the primary event layer. It differs from the older EURUSD OTE pipeline because the old OTE stack is largely a direct entry-labeling stack with FX assumptions, while the new ICT design is explicitly ES-first and setup-gated. It also differs from direct OTE prediction because the model should not learn to invent entries from every bar. It should filter already-detected ICT opportunities.

The recommended design is to build a separate `ict/` research package parallel to `frvp/`, reuse the existing ES session, roll, preprocessing, attribution, training, threshold, and backtest infrastructure wherever possible, and treat current generic ICT feature logic and `ict_app/` detectors as reference material unless they pass strict causality and ES-compatibility review.

## 2. Instrument Choice: Why ES for ICT / SMC

ES is the correct primary instrument for this research because it matches both the structure of ICT concepts and the strengths of the current repo.

- Centralized venue: ES trades on a centralized futures venue, which makes session highs/lows, sweep levels, and execution assumptions cleaner than decentralized spot FX.
- Deep liquidity: ES supports realistic liquidity behavior around prior day high/low, overnight high/low, RTH open, and Initial Balance.
- Clean session structure: ES has meaningful RTH versus ETH behavior, while many ICT concepts depend on session-defined liquidity pools and open-drive behavior.
- Realistic small-size deployment: MES provides a practical path for paper trading and live small-size validation without changing the logic.
- Existing repo fit: the repo already contains ES session logic, ES cost assumptions, ES walk-forward testing, and FRVP ES pipeline scaffolding.

Comparison:

| Instrument | Strengths | Weaknesses | Role in this design |
| --- | --- | --- | --- |
| ES / MES | Centralized, deep liquidity, strong session structure, clean sweep targets, repo already ES-aware | Less FX-like than 6E for classic ICT narratives | Primary |
| EUR/USD spot | Legacy baseline in repo, deep historical research path | Decentralized, less reliable session/liquidity definitions, older pipeline assumptions are FX-specific | Legacy baseline only |
| 6E / M6E | Futures-based FX expression, still centralized, closer to old OTE lineage | Less direct fit than ES for current repo's strongest infrastructure | Future variant |
| NQ / MNQ | Rich liquidity behavior and strong reaction patterns | Higher beta, noisier moves, more fragile cost/slippage behavior | Challenger variant after ES |

Conclusion:

- ES / MES should be the primary instrument pair.
- 6E / M6E should be the first future FX-style extension.
- NQ / MNQ should be treated as a later high-beta challenger.
- EUR/USD spot should remain a historical reference only.

## 3. Existing Repo Scan

### 3.1 FRVP artifacts to reuse

The FRVP ES stack is the strongest structural template for the new ICT package.

- `frvp/pipelines/es_primary_phase04.py`
  - Current role: canonical phased ES pipeline orchestration from integrity and feature build through FRVP labeling and prepared dataset generation.
  - Reuse decision: directly reusable as the architectural template for ICT phases.
- `data/labeling/frvp_labeling_engine.py`
  - Current role: setup-driven meta-labeling engine for FRVP reversal and continuation families, with helper columns such as `sample_weight_*`, `exclude_*`, `neg_ok_*`, `concurrency_*`, and `htf_confluence_*`.
  - Reuse decision: directly reusable as the design pattern and partially reusable code path for event materialization, exclusions, concurrency, and pooled meta targets.
- `frvp/setups/detector.py`
  - Current role: deterministic FRVP setup detector with fire-rate diagnostics and setup-family logic.
  - Reuse decision: reuse the detector pattern, diagnostics style, and event abstraction; do not reuse FRVP semantics.
- `frvp/sessions/equity.py`
  - Current role: ES and 6E session framing, RTH/ETH logic, Initial Balance, half-day awareness.
  - Reuse decision: directly reusable.
- `frvp/calendars/macro.py`
  - Current role: deterministic macro-event flags for CPI, NFP, FOMC, statement, and presser windows using `data/futures_data/CPI_release_dates.txt`.
  - Reuse decision: directly reusable for ICT exclusion flags.
- `frvp/continuity/continuous_contract.py`
  - Current role: continuous futures construction with explicit handling for back-adjusted path series, absolute level series, and roll-spanning windows.
  - Reuse decision: directly reusable and mandatory for any ICT levels that reference absolute futures prices.
- `frvp/config/instruments.py`
  - Current role: ES and 6E instrument settings, including tick size, tick value, and session definitions.
  - Reuse decision: directly reusable, likely imported rather than duplicated.
- `frvp/feature_sets/frvp_context.py` and `features/feature_sets/frvp_context.py`
  - Current role: package-native FRVP feature implementation plus thin registry shim.
  - Reuse decision: directly reusable as the compatibility-shim pattern for future `ict_context`.
- `frvp/strategies/frvp_setups.py`
  - Current role: thin wrapper exposing FRVP setup detection through a stable import surface.
  - Reuse decision: directly reusable as the pattern for `features/strategies/ict_setups.py`.
- `scripts/run_frvp_training_stack.ps1`
  - Current role: FRVP training orchestration for XGBoost and TCN.
  - Reuse decision: reusable with modification after ICT prepared targets exist.
- `scripts/run_frvp_post_training_eval.ps1`
  - Current role: FRVP regime slices, threshold search, and walk-forward backtest orchestration.
  - Reuse decision: reusable with modification after ICT registry and report roots exist.
- `scripts/build_frvp_candidate_registry.py`
  - Current role: generates a candidate registry from training summaries.
  - Reuse decision: reusable with modification as the ICT registry bootstrap pattern.
- `scripts/build_frvp_shadow_live_bundle.py`
  - Current role: assembles FRVP family leaders into a shadow-live bundle with runtime manifests and packaged policies.
  - Reuse decision: reference now, direct adaptation later once ICT reaches post-training bundle stage.
- `docs/FRVP_ES_primary_audit_report.md`, `docs/FRVP_ES_primary_6E_variant_design.md`, `docs/FRVP_ML_Strategy.md`
  - Current role: design/reporting precedents for phased ES-first research work.
  - Reuse decision: reference for documentation style and governance.

### 3.2 OTE / EURUSD artifacts to reuse

The older OTE stack is still the main source of reusable label mechanics, feature alignment, attribution, and model-training infrastructure.

- `data/labeling/README.md`
  - Current role: explains causal labeling philosophy and helper-column conventions.
  - Reuse decision: directly reusable as design precedent.
- `data/labeling/labeling_engine.py`
  - Current role: unified EURUSD labeling orchestrator with multi-timeframe support and raw-data normalization patterns.
  - Reuse decision: reusable with modification; useful for orchestration structure, not for ES assumptions.
- `data/labeling/reversal_labeling_engine.py`
  - Current role: core causal label mechanics including CUSUM filtering, triple-barrier outcomes, trend scanning, exclusions, concurrency, and sample weights.
  - Reuse decision: directly reusable in parts for event validation and helper-column generation.
- `data/labeling/ote_continuation_pullback_labeling_engine.py`
  - Current role: continuation family built on top of reversal machinery.
  - Reuse decision: reusable with modification as a pattern for ICT continuation families.
- `data/labeling/ote_breakout_labeling_engine.py`
  - Current role: breakout labeling family built on the same causal toolkit.
  - Reuse decision: reusable with modification as a pattern for additional ICT families.
- `features/builder.py`, `features/config.py`, `features/registry.py`, `features/README.md`
  - Current role: canonical feature build infrastructure and configuration contract.
  - Reuse decision: directly reusable.
- `preprocessing/pipeline.py`, `preprocessing/config.py`, `preprocessing/feature_selection.py`, `preprocessing/backend_attribution.py`, `preprocessing/reporting.py`, `preprocessing/README.md`
  - Current role: prepared dataset build, target discovery, helper-column handling, feature ranking, and reporting.
  - Reuse decision: directly reusable, with new ICT targets added later.
- `model_training/ote_training/ote_xgboost_pipeline.py`
  - Current role: XGBoost and torch training pipeline with leakage filtering, Optuna, calibration, explicit purged walk-forward geometry, artifact writing, and threshold-aware metrics.
  - Reuse decision: directly reusable.
- `model_training/ote_training/torch_trainer.py`
  - Current role: TCN and LSTM training utilities.
  - Reuse decision: directly reusable.
- `model_training/ote_training/feature_ranking.py`
  - Current role: resolves backend-aware ranked feature files for training.
  - Reuse decision: directly reusable.
- `scripts/run_ote_regime_slice_report.py`, `scripts/run_ote_threshold_policy_search.py`, `scripts/run_ote_policy_backtest.py`
  - Current role: post-training evaluation entry points.
  - Reuse decision: directly reusable.
- `docs/end-to-end-README.md`
  - Current role: canonical OTE workflow runbook.
  - Reuse decision: reference for stage ordering and artifact lineage.

### 3.3 Existing ICT / SMC artifacts

The repo already contains some ICT logic, but it is split between a generic feature family and a prototype `ict_app/` track. The generic feature family is useful. The `ict_app/` track is not currently a safe foundation for the canonical ES pipeline.

| Artifact | What it currently does | Causal today? | ES-compatible today? | Used by a canonical pipeline? | Reuse decision |
| --- | --- | --- | --- | --- | --- |
| `features/feature_sets/ict_context.py` | Builds causal proximity/state features for FVGs, order blocks, equal highs/lows, BOS, CHoCH, and bars-since events; already registered as a feature family | Mostly yes; uses closed bars and confirmed swings | Partially; generic and not session/roll aware | Yes, via feature recipes and feature config defaults | Reuse with heavy modification and likely wrap around new `ict/feature_sets/ict_context.py` |
| `features/feature_sets/structure.py` | Provides causal sweep, prior high/low, breakout, and premium/discount style context | Yes | Generic, partially suitable for ES | Yes | Reuse with modification |
| `features/feature_sets/session.py` | Generic session, overlap, kill-zone, and day-part features | Yes | Mostly yes, but generic rather than ES-specific | Yes | Direct reuse with ES-specific additions |
| `features/feature_sets/htf_context.py` | Uses 30m/1h and daily/weekly context from completed HTF bars | Yes if completed-bar contract is preserved | Yes | Yes | Direct reuse |
| `ict_app/ict_detection/fvg_detector.py` | Prototype FVG detector | Mostly causal at detection time, but FX-oriented | No; pip assumptions and FX tuning | No | Reference only; rewrite for ES |
| `ict_app/ict_detection/order_block_detector.py` | Prototype order block detector with future move confirmation logic | Not cleanly causal for canonical research | No | No | Replace |
| `ict_app/ict_detection/liquidity_sweep_detector.py` | Prototype liquidity sweep detector with future reversal confirmation windows | No | No | No | Replace |
| `ict_app/ict_detection/market_structure_analyzer.py` | Prototype structure/BOS/CHoCH analysis | Partially | Not sufficiently ES-specific | No | Reference only; rewrite |
| `ict_app/ict_detection/session_filter.py` | FX-style session window filter | Yes | No; not aligned with ES RTH/ETH/IB | No | Replace |
| `ict_app/ict_detection/pattern_coordinator.py` | Standalone signal aggregator/scorer for prototype app | Mixed and not designed for current ML pipeline contracts | No | No | Replace |
| `ict_app/tests/test_ict_detectors.py` | Prototype detector tests | N/A | N/A | No | Reference only |
| `ict_app/README.md` and `ict_app/ict_detection/PHASE1_SUMMARY.md` | Prototype documentation for a separate ICT app track | N/A | N/A | No | Reference only |
| `docs/PRODUCTION_REALTIME_SIGNAL_SYSTEM_PLAN.md` | Explicitly classifies `ict_app` as prototype and non-canonical for the main pipeline | N/A | N/A | Yes, as design guidance | Directly reuse the caution |

Key conclusion from the ICT scan:

- `features/feature_sets/ict_context.py` is useful and already causal enough to keep as a seed.
- `ict_app/` should not be used as the base for the new research package.
- The original audit found no canonical ES-first ICT event layer, setup detector, or labeler. That is no longer true: the repo now has a v1 `ict/` package with core detectors, a setup detector, feature shims/recipe scaffolding, and an ICT labeling engine. What is still missing is the full prepared-dataset/training/backtest pipeline wiring.

### 3.4 Preprocessing artifacts

The existing preprocessing stack is one of the strongest directly reusable layers.

- `preprocessing/pipeline.py`
  - Handles chronological train/val/test preparation, target discovery, helper-column propagation, and report generation.
- `preprocessing/config.py`
  - Already includes FRVP and OTE targets; ICT targets can follow the same naming convention.
- `preprocessing/feature_selection.py`
  - Already supports pooled FRVP meta targets and label-family-aware target definitions. ICT can be added by analogy to `label_long_frvp_meta` and `label_short_frvp_meta`.
- `preprocessing/backend_attribution.py`
  - Already generates backend-specific ranking files for XGBoost, TCN, and LSTM.
- `preprocessing/README.md`
  - Documents prepared-root conventions, helper-column behavior, and backend attribution outputs.

Reuse decision: direct reuse once ICT labels and feature sets produce compatible target and helper columns.

### 3.5 Feature generation artifacts

The feature system is broad and already contains many pieces ICT needs.

- `features/builder.py`
  - Canonical batch feature build path. Direct reuse.
- `features/config.py`
  - Already includes `ict_context` in default feature sets and already exposes ICT parameters such as `ict_zone_max_age`, `ict_fvg_min_gap_atr`, `ict_order_block_range_atr`, `ict_liquidity_tolerance_atr`, and `ict_break_buffer_atr`. Reuse with expansion.
- `features/feature_sets/structure.py`
  - Reusable for base liquidity and structural context.
- `features/feature_sets/session.py`
  - Reusable for session and kill-zone context.
- `features/feature_sets/htf_context.py`
  - Reusable for 30m/1h context and prior-day/prior-week levels.
- `features/feature_sets/ict_context.py`
  - Reusable seed, but not enough to be the full ICT event layer.
- `features/recipes/meta_labeling.json`
  - Existing meta-label recipe that already includes `ict_context`.
- `features/recipes/frvp_meta.json`
  - Existing FRVP meta recipe that also includes `ict_context` and `frvp_context`, which is useful for optional later confluence.

Missing feature artifacts:

- `ict/feature_sets/ict_interactions.py`
- `features/feature_sets/ict_interactions.py`
- an ES-specific ICT feature recipe such as `features/recipes/ict_es_meta.json`

### 3.6 Feature importance / attribution artifacts

The repo already has the full attribution stack needed for ICT.

- `preprocessing/feature_importance.py`
- `preprocessing/backend_attribution.py`
- `model_training/ote_training/feature_ranking.py`
- `scripts/merge_ote_feature_rankings.py`

Capabilities already present:

- backend-specific merged rankings
- SHAP-compatible merged outputs for XGBoost and torch backends
- feature selection before model training
- prepared-root summaries that can be reused by ICT

Reuse decision: direct reuse. ICT should not invent a second attribution system.

### 3.7 Model training artifacts

The current training stack already matches the desired backend priority.

- `model_training/ote_training/ote_xgboost_pipeline.py`
  - Supports XGBoost first, TCN/LSTM through the torch path, explicit purged walk-forward geometry, calibration (`platt`, `isotonic`, `none`), Optuna, leakage blocking, event tolerance, and cooldown-aware evaluation.
- `model_training/ote_training/torch_trainer.py`
  - Reusable for TCN and LSTM optimization.

Recommended ICT stance:

- XGBoost first because the repo's attribution and diagnostics are already strongest for tabular models.
- TCN second once the event layer is stable and event counts justify sequence modeling.
- LSTM optional and lower priority.

### 3.8 Model testing / policy artifacts

The post-training evaluation stack is already ES-aware and should be reused rather than rewritten.

- `model_testing/ote_regime_labeler.py`
  - Deterministic regime labeling. Direct reuse.
- `model_testing/ote_threshold_policy.py`
  - Regime-aware threshold search with global fallback and policy artifacts. Direct reuse.
- `model_testing/ote_abstain_policy.py`
  - Abstain and cooldown logic. Direct reuse.
- `model_testing/ote_policy_metrics.py`
  - Policy-level performance calculations. Direct reuse.
- `model_testing/ote_policy_backtest.py`
  - Quarterly walk-forward backtesting with acceptance gates such as profitable-quarter-share and positive-composite-expectancy-share. Direct reuse.
- `model_testing/evaluation_costs.py`
  - Already contains ES and 6E cost assumptions. Current ES defaults are a useful starting point: session spread units `overlap=1.0`, `london=1.0`, `new_york=1.0`, `asia=1.5`, `off_hours=2.0`, fixed slippage `0.25` ticks, commission `0.40` ticks.
- `model_testing/ote_prediction_joiner.py`
  - Reusable for joining predictions back to source rows for diagnostics.

### 3.9 Artifact classification table

| Area | Existing artifact | Current role | Reuse decision | Required changes | Risk |
| --- | --- | --- | --- | --- | --- |
| ES pipeline structure | `frvp/pipelines/es_primary_phase04.py` | Canonical phased ES flow | Directly reusable as template | Rename phases and targets for ICT | Low |
| Event detection pattern | `frvp/setups/detector.py` | Deterministic setup detector | Reuse pattern only | Replace FRVP setup logic with ICT logic | Low |
| Meta-labeling engine pattern | `data/labeling/frvp_labeling_engine.py` | Setup-driven meta labels and helper columns | Directly reusable in structure | Add ICT event schema and ICT barriers | Low |
| Session logic | `frvp/sessions/equity.py` | ES RTH/ETH/IB framing | Directly reusable | Add any ICT-specific session annotations | Low |
| Macro exclusions | `frvp/calendars/macro.py` | Macro window flags | Directly reusable | None beyond import plumbing | Low |
| Continuity and rolls | `frvp/continuity/continuous_contract.py` | Back-adjusted path plus absolute-level continuity | Directly reusable | None beyond pipeline wiring | Low |
| Instrument config | `frvp/config/instruments.py` | ES/6E tick and session settings | Directly reusable | Possibly factor into common shared core later | Low |
| Generic ICT features | `features/feature_sets/ict_context.py` | Current ICT feature family | Reuse with heavy modification | Move core logic under `ict/`, keep shim in `features/` | Medium |
| Generic structure features | `features/feature_sets/structure.py` | Sweep and range-position context | Reuse with modification | Add ES-specific liquidity references | Medium |
| Session features | `features/feature_sets/session.py` | Generic session windows | Direct reuse plus extension | Add ES-specific flags like IB masking and RTH phases | Low |
| HTF context | `features/feature_sets/htf_context.py` | 30m/1h and prior day/week context | Direct reuse | Ensure completed-bar-only contract remains enforced | Low |
| OTE reversal machinery | `data/labeling/reversal_labeling_engine.py` | Triple barrier, trend scan, sample weights | Direct reuse in parts | Swap OTE semantics for ICT event validation | Low |
| OTE continuation/breakout | `data/labeling/ote_continuation_pullback_labeling_engine.py`, `data/labeling/ote_breakout_labeling_engine.py` | Family-specific label patterns | Reuse with modification | Adapt to ICT continuation/reversal families | Medium |
| Prepared dataset build | `preprocessing/pipeline.py` | Prepared-root generation | Directly reusable | Add ICT target discovery entries | Low |
| Backend attribution | `preprocessing/backend_attribution.py` | XGB/TCN/LSTM ranking outputs | Directly reusable | None | Low |
| Training | `model_training/ote_training/ote_xgboost_pipeline.py` | XGB and torch training with purged CV | Directly reusable | Add ICT model/target names only | Low |
| Policy testing | `model_testing/ote_threshold_policy.py`, `model_testing/ote_policy_backtest.py` | Threshold search and WFO evaluation | Directly reusable | Add ICT registry/report roots | Low |
| Registry pattern | `scripts/build_frvp_candidate_registry.py` | Candidate registry generation | Reuse with modification | Create ICT registry builder | Medium |
| Shadow bundle pattern | `scripts/build_frvp_shadow_live_bundle.py` | Post-research package assembly | Reference only for now | Use only after ICT matures | Medium |
| Prototype ICT detectors | `ict_app/ict_detection/*.py` | Prototype ICT signal logic | Obsolete/risky for direct reuse | Rewrite cleanly in `ict/` | High |
| Missing package | `ict/` | None today | Must be built | Full package skeleton and tests | Medium |
| Missing shim | `features/feature_sets/ict_interactions.py` | None today | Must be built | Register interaction features | Low |
| Missing shim | `features/strategies/ict_setups.py` | None today | Must be built | Thin wrapper over `ict.setups.detector` | Low |
| Missing label shim | `data/labeling/ict_labeling_engine.py` | None today | Must be built | Thin compatibility entrypoint | Low |

## 4. Proposed ICT Package Architecture

The recommended architecture is a dedicated `ict/` package parallel to `frvp/`, with thin compatibility shims back into the legacy `features/` and `data/labeling/` surfaces.

```text
ict/
  __init__.py
  config/
    __init__.py
    instruments.py
    setups.py
    thresholds.py
  sessions/
    __init__.py
    equity.py
  structure/
    __init__.py
    swings.py
    market_structure.py
    liquidity.py
  detectors/
    __init__.py
    fvg.py
    order_blocks.py
    displacement.py
    sweeps.py
    bos_choch.py
    premium_discount.py
  setups/
    __init__.py
    detector.py
    setup_types.py
    validators.py
  feature_sets/
    __init__.py
    ict_context.py
    ict_interactions.py
  labeling/
    __init__.py
    ict_labeling_engine.py
  preprocessing/
    __init__.py
    build_ict_dataset.py
  pipelines/
    __init__.py
    es_phase01_scan.py
    es_phase02_features.py
    es_phase03_labeling.py
    es_phase04_prepare.py
    es_phase05_train.py
    es_phase06_thresholds.py
    es_phase07_backtest.py
  reports/
    __init__.py
    diagnostics.py
  tests/
    test_causality.py
    test_setup_detection.py
    test_labeling.py
    test_es_sessions.py
```

Compatibility shims:

```text
features/feature_sets/ict_context.py
features/feature_sets/ict_interactions.py
features/strategies/ict_setups.py
data/labeling/ict_labeling_engine.py
```

Design rules for the package split:

- New logic should live under `ict/`, not directly in legacy folders.
- `features/feature_sets/ict_context.py` should become a thin registry shim similar to `features/feature_sets/frvp_context.py`.
- `features/strategies/ict_setups.py` should mirror `frvp/strategies/frvp_setups.py` and simply expose `ict.setups.detector`.
- `data/labeling/ict_labeling_engine.py` should be a compatibility entrypoint that delegates to `ict.labeling.ict_labeling_engine`.
- `ict/config/instruments.py` should import or wrap `frvp/config/instruments.py` rather than fork the ES contract.
- `ict/sessions/equity.py` should extend or wrap `frvp/sessions/equity.py` and preserve the same holiday and half-day treatment.
- `ict/preprocessing/build_ict_dataset.py` should orchestrate existing `features` and `preprocessing` utilities rather than rebuild preprocessing from scratch.

What is new:

- deterministic ES-first ICT concept detection
- deterministic ICT setup families
- ICT event materialization and barriers
- ES-specific ICT interaction features
- ICT reports and diagnostics

What should call existing repo utilities:

- raw-data normalization
- contract continuity and roll exclusions
- session calendar logic
- completed-HTF alignment
- helper-column conventions
- backend attribution
- model training
- regime labeling
- threshold search
- walk-forward testing

## 5. ICT / SMC Concept Definitions for Automation

All definitions below must be deterministic, causal, and closed-bar-only unless explicitly marked as 1-minute execution refinement.

### 5.1 Fair Value Gaps

Bullish FVG:

- Formation bar index `t` is valid only after bar `t` closes.
- Define a three-candle gap where `low[t] > high[t-2] + min_gap`.
- Zone is `[high[t-2], low[t]]`.
- `min_gap` should be ATR-normalized, with the current generic seed from `features/config.py` (`ict_fvg_min_gap_atr`) retained as the starting parameter.

Bearish FVG:

- Valid only after bar `t` closes.
- Define `high[t] < low[t-2] - min_gap`.
- Zone is `[high[t], low[t-2]]`.

Required tracked state:

- `fvg_id`
- direction
- source timeframe (`5m`, `30m`, or `1h`)
- creation time
- lower and upper bounds
- width in ticks and ATR units
- age in bars
- created_by_displacement flag
- mitigated percent
- fully invalidated flag

Feature implications:

- distance to nearest active bullish/bearish FVG
- whether price is inside the FVG
- whether price entered the FVG from premium or discount
- whether the FVG was formed immediately after a liquidity sweep
- HTF FVG alignment with 5m setup

### 5.2 Order Blocks

Bullish order block:

- Start from a local bearish candle or bearish candle cluster preceding a bullish displacement sequence.
- Confirm only after the displacement candle or displacement sequence has fully closed.
- Candidate zone should use body-first logic with optional wick extension policy configured explicitly.

Bearish order block:

- Symmetric definition using a bullish candle or bullish cluster preceding bearish displacement.

Required confirmation rules:

- displacement range >= configurable ATR multiple
- displacement close is near candle high/low
- displacement breaks a causal swing or structure level
- block is not valid until displacement is confirmed

Tracked state:

- `order_block_id`
- direction
- source candle range
- body range and full range
- age
- retest count
- mitigation state
- invalidation state
- displacement strength and volume/range intensity

### 5.3 Liquidity Sweeps

Liquidity pools should be deterministic and hierarchical.

Pool types:

- prior confirmed swing high / low
- equal highs / equal lows
- overnight high / low
- prior RTH high / low
- Initial Balance high / low
- prior week high / low

Sweep definition:

- Buy-side sweep: current high exceeds a known liquidity level by at least `sweep_buffer`, then closes back below or fails to hold above the level.
- Sell-side sweep: current low pierces a known liquidity level by at least `sweep_buffer`, then closes back above or fails to hold below the level.

Tracked state:

- sweep type
- swept level type
- penetration depth in ticks and ATR
- close-back-inside flag
- failed-sweep flag
- bars since sweep
- whether the sweep occurred at a session extreme

### 5.4 BOS / CHoCH / Market Structure Shift

Definitions should use confirmed swings only.

- BOS: price breaks the latest confirmed swing in the direction of the prevailing structure.
- CHoCH: price breaks the latest confirmed swing opposite the prior structure state.
- MSS: use as the same event family as CHoCH, but allow the implementation to distinguish "first structure flip after a sweep" from generic CHoCH if that improves diagnostics.

Anti-look-ahead rule:

- Swings must be confirmed by a fixed completed-bar window or causal swing algorithm.
- No structure break may reference a swing point that was not yet confirmed at event time.
- No "future confirmation bar" is allowed beyond the standard completed-bar swing confirmation window.

### 5.5 Displacement

Displacement should be an objective impulse measure, not a visual narrative.

Recommended definition:

- candle or short sequence range >= configurable ATR multiple
- body-to-range ratio above threshold
- close in upper quartile for bullish displacement or lower quartile for bearish displacement
- optional volume z-score confirmation if reliable ES volume is available
- bonus score if displacement immediately follows a sweep or creates an FVG

Tracked outputs:

- displacement direction
- displacement score
- body ATR multiple
- range ATR multiple
- close-location percentile
- volume z-score
- created FVG flag

### 5.6 Premium / Discount

Premium/discount must be defined from causal dealing ranges, not hindsight ranges.

Recommended range anchors:

- most recent confirmed swing low to confirmed swing high on 5m for local dealing range
- 30m confirmed swing anchors for HTF dealing range
- optional 1h anchors for trend context

Derived features:

- current price percentile within range
- discount zone flag
- premium zone flag
- equilibrium proximity
- local and HTF range alignment

### 5.7 Session Liquidity

For ES, the following session levels are meaningful and should be first-class ICT references:

- prior RTH high / low
- prior RTH close
- overnight high / low
- current RTH open
- Initial Balance high / low
- prior week high / low
- optional later FRVP confluence such as prior value area or profile anchors

These levels should be fixed only when the underlying session is complete or, for current-day levels such as IB, when the causal completion condition has been met.

## 6. ICT Setup Rule Layer

Setup detection should run on finalized 5-minute ES bars. The detector should emit candidate events and allow optional 1-minute refinement for entry confirmation only, not for defining the original 5-minute setup existence.

Shared rules for all setups:

- all setup detection is closed-bar-only on 5m
- same-side same-family events should honor a cooldown, initially 4 to 8 bars
- duplicate events anchored to the same structural level should be collapsed
- every setup must specify a family (`reversal` or `continuation`)
- every setup must specify a base holding horizon and stop anchor

### Setup 1: Liquidity Sweep + Reclaim

- Side hypothesis: reversal.
- Required conditions:
  - sell-side sweep for long or buy-side sweep for short
  - close back through the swept level on the trigger bar or within `N` bars
  - no immediate invalidation through the opposite side of the trigger structure
- Optional confluence:
  - displacement appears within `N` bars
  - nearby FVG or order block
  - premium/discount alignment
- Invalidation:
  - price closes beyond the sweep extreme by more than a configured invalidation buffer
- Minimum spacing:
  - 6 bars per side per swept-level family
- Expected move type:
  - reversal
- Label horizon:
  - 8 to 16 bars
- 1-minute refinement:
  - require a micro reclaim or micro BOS back in trade direction inside 3 to 5 minutes

Pseudocode:

```text
if sweep_detected(level_type, side_opposite_target)
and close_back_inside(level)
and not invalidated_same_bar
then emit setup_type = sweep_reclaim
```

### Setup 2: Sweep + Displacement + FVG Retrace

- Side hypothesis: reversal with confirmation.
- Required conditions:
  - initial liquidity sweep
  - displacement in opposite direction within `N` bars
  - displacement creates an FVG
  - price retraces into the FVG without invalidating the displacement leg
- Optional confluence:
  - CHoCH after sweep
  - HTF discount/premium alignment
- Invalidation:
  - FVG fully fails or price closes through the structural stop level
- Minimum spacing:
  - 6 bars per direction
- Expected move type:
  - reversal
- Label horizon:
  - 10 to 18 bars
- 1-minute refinement:
  - require 1m rejection inside the FVG or 1m micro BOS away from the zone

Pseudocode:

```text
if sweep_detected
and displacement_confirmed
and fvg_created
and retrace_into_fvg
then emit setup_type = sweep_displacement_fvg
```

### Setup 3: Order Block Retest After MSS / CHoCH

- Side hypothesis: reversal after structure shift.
- Required conditions:
  - sweep of external or internal liquidity
  - CHoCH / MSS in the new direction using confirmed swings only
  - last opposing candle or candle cluster before displacement becomes the order block
  - price retests the order block while structure remains intact
- Optional confluence:
  - overlapping FVG
  - discount/premium alignment
- Invalidation:
  - close through the order block invalidation boundary
- Minimum spacing:
  - 8 bars per direction
- Expected move type:
  - reversal
- Label horizon:
  - 12 to 20 bars
- 1-minute refinement:
  - require a 1m rejection or 1m structure continuation off the order block

Pseudocode:

```text
if sweep_detected
and choch_confirmed
and order_block_confirmed_after_displacement
and retest_of_order_block
then emit setup_type = ob_retest_after_mss
```

### Setup 4: Breaker / Failed Order Block

- Side hypothesis: reversal after failed original block.
- Current design stance:
  - Phase 2 only unless the implementation team can derive a clean causal definition from the new order-block state machine.
- Reason:
  - no strong current repo support
  - higher semantic ambiguity than FVG, sweep, or order-block retest
- Initial recommendation:
  - define but do not include in v1 labels unless event counts and diagnostics remain clean

### Setup 5: Premium / Discount Continuation Pullback

- Side hypothesis: continuation.
- Required conditions:
  - HTF trend state aligned on 30m and optionally 1h
  - pullback into discount for longs or premium for shorts
  - supportive zone such as bullish FVG / bullish order block for longs, symmetric for shorts
  - no confirmed structure failure against trend
- Optional confluence:
  - prior sweep in trend direction
  - FRVP confluence later
- Invalidation:
  - HTF trend state breaks or local structure fails
- Minimum spacing:
  - 6 bars per side
- Expected move type:
  - continuation
- Label horizon:
  - 8 to 16 bars
- 1-minute refinement:
  - require 1m pullback exhaustion or micro continuation trigger

Pseudocode:

```text
if htf_trend_up
and pullback_into_discount
and supportive_bull_zone_present
and local_structure_intact
then emit setup_type = premium_discount_continuation
```

### Setup 6: Session Open Manipulation Reversal

- Side hypothesis: ES-specific reversal.
- Required conditions:
  - RTH open sweep of overnight high/low or early false break of Initial Balance
  - reclaim back inside the level or back inside IB
  - preferably displacement or failure-follow-through confirmation
- Optional confluence:
  - nearby HTF liquidity or FVG
- Invalidation:
  - clean acceptance outside the manipulated level
- Minimum spacing:
  - once per side per opening-session pattern
- Expected move type:
  - reversal
- Label horizon:
  - 6 to 12 bars
- 1-minute refinement:
  - require a 1m rejection or micro structure shift immediately after reclaim

Pseudocode:

```text
if early_rth_sweep_of_on_or_ib_level
and reclaim_inside_reference
then emit setup_type = session_open_manipulation
```

### Setup 7: Displacement Continuation After Liquidity Raid

- Side hypothesis: continuation after initial raid and impulse.
- Required conditions:
  - liquidity raid occurs against the eventual move direction
  - strong displacement starts a directional move
  - shallow pullback holds above bullish impulse origin for longs or below bearish origin for shorts
- Optional confluence:
  - FVG or order-block pullback inside trend
- Invalidation:
  - full failure through the impulse origin
- Minimum spacing:
  - 8 bars per direction
- Expected move type:
  - continuation
- Label horizon:
  - 10 to 20 bars
- 1-minute refinement:
  - require 1m continuation alignment after the pullback

Pseudocode:

```text
if raid_detected
and displacement_starts_new_move
and pullback_holds_impulse_origin
then emit setup_type = displacement_continuation_after_raid
```

## 7. Labeling Design

The new label engine should live at `ict/labeling/ict_labeling_engine.py`, with a compatibility wrapper at `data/labeling/ict_labeling_engine.py`.

The rule layer should generate event rows with a schema like:

```text
event_time
side
setup_type
setup_family
anchor_level
entry_price
stop_reference
target_reference
htf_context
sweep_type
fvg_id
order_block_id
displacement_id
session_phase
```

Recommended label design:

- Produce separate long/short and reversal/continuation labels.
- Also produce pooled meta labels by direction.
- Match FRVP helper-column conventions exactly so existing preprocessing can discover them.

Recommended initial targets:

```text
label_long_ict_reversal
label_short_ict_reversal
label_long_ict_continuation
label_short_ict_continuation
label_long_ict_meta
label_short_ict_meta
```

Barrier design:

- Primary outcome engine should be triple barrier.
- `meta_label = 1` if target is hit before stop.
- `meta_label = 0` otherwise.
- Timeout should initially map to `0` rather than a third class unless the existing conventions prove otherwise for a specific family.

Barrier references:

- stop can be the wider of:
  - structural stop beyond swept liquidity or order block edge
  - minimum ATR stop floor
- target can be the nearer of:
  - opposing liquidity target
  - ATR-based target multiple
- per-setup max holding period should be explicit

Recommended starting horizons:

- reversal families: 8 to 18 bars
- continuation families: 8 to 20 bars
- session-open reversal: 6 to 12 bars

Event hygiene rules:

- de-duplicate events by side, setup type, and anchor zone
- enforce per-family cooldown
- compute overlap/concurrency counts
- down-weight highly concurrent events
- optionally mark `neg_ok_*` only where the existing FRVP/OTE logic would consider negatives safe

Exclusion rules should mirror FRVP where possible:

- macro event windows from `frvp/calendars/macro.py`
- half-days / thin-session rules from ES session logic
- roll-spanning windows from `frvp/continuity/continuous_contract.py`
- warmup mask
- missing-next-open or broken bar continuity
- any event whose barrier window crosses a contract-roll boundary

1-minute refinement labels:

- Keep the base 5m event labels as canonical.
- Optionally add a 1m refinement success flag or 1m execution-quality flag for later research.
- Do not let 1m refinement redefine whether the 5m setup existed in the first place.

## 8. Feature Engineering Design

The ICT feature layer should be event-aware but still compatible with the current tabular and sequence training pipelines.

### 8.1 Structural features

- distance to nearest bullish and bearish FVG
- distance to nearest bullish and bearish order block
- distance to swept liquidity level
- bars since sweep
- bars since BOS
- bars since CHoCH
- current market-structure state
- 30m and 1h trend state
- price position inside local dealing range
- premium / discount / equilibrium flags
- distance to prior RTH high / low
- distance to overnight high / low
- distance to Initial Balance high / low
- distance to prior week high / low

### 8.2 FVG features

- width in ticks
- width normalized by ATR
- age
- mitigation percent
- source timeframe
- created-by-displacement flag
- active vs invalidated flag
- overlap with order block
- overlap with recent sweep

### 8.3 Order block features

- distance to nearest active block
- age
- retest count
- invalidation status
- displacement strength from the block origin
- block range normalized by ATR
- volume percentile if reliable
- HTF/LTF alignment

### 8.4 Liquidity features

- equal-high and equal-low pool count
- nearby liquidity pool density
- sweep depth
- failed-sweep flag
- close-back-inside flag
- time since sweep
- sweep at overnight extreme flag
- sweep at prior-day extreme flag
- sweep at Initial Balance boundary flag

### 8.5 Displacement features

- body ATR multiple
- full range ATR multiple
- close location percentile
- sequence displacement score
- FVG created by displacement flag
- volume z-score
- impulse direction

### 8.6 Session features

- RTH phase
- ETH phase
- open drive flag
- open rejection flag
- Initial Balance complete flag
- Initial Balance extension flag
- lunch-lull flag
- close-ramp flag
- macro window flag
- day-of-week

### 8.7 Interaction features

Recommended explicit interactions:

```text
sweep_plus_fvg
sweep_plus_ob
sweep_plus_choch
fvg_inside_discount
ob_inside_discount
sweep_at_overnight_low
sweep_at_prior_day_low
choch_after_sweep
displacement_after_sweep
fvg_retrace_after_displacement
ict_setup_with_frvp_confluence_optional
```

FRVP confluence should remain optional. ICT must be able to run on its own.

## 9. Preprocessing and Dataset Construction

The ICT dataset should follow the same workflow pattern as the current ES FRVP stack and the current OTE prepared-data flow.

Recommended data inputs:

- raw ES 1m bars for execution refinement and intrabar diagnostics
- raw ES 5m bars as the primary modeling frame
- 30m and 1h bars derived by deterministic local resampling from canonical 5m/1m inputs

Core construction rules:

- reuse `frvp/continuity/continuous_contract.py` for contract continuity
- reuse ES session calendar logic from `frvp/sessions/equity.py`
- reuse macro exclusions from `frvp/calendars/macro.py`
- preserve absolute-level versus back-adjusted-path separation whenever liquidity levels depend on absolute price coordinates
- no forming HTF bars
- no future swing points
- no unfinished session levels before they are known
- mask Initial Balance features before IB completion
- carry warmup rows through feature build and prepared-root generation

Recommended artifact structure:

```text
artifacts/ict_es_primary_<date>/
  phase01_scan/
  phase02_features/
  phase03_labeling/
  phase04_prepared/
  phase05_training/
  phase06_thresholds/
  phase07_backtests/
```

Recommended phase responsibilities:

- `phase01_scan`
  - integrity checks
  - continuity and roll diagnostics
  - session and macro enrichment audit
  - setup fire-rate diagnostics
- `phase02_features`
  - merged 5m feature dataset
  - feature metadata
- `phase03_labeling`
  - ICT event table
  - label columns
  - helper columns
  - labeling diagnostics
- `phase04_prepared`
  - prepared target folders
  - backend attribution outputs

Prepared-root naming convention should mirror FRVP and OTE:

- `long_ict_reversal`
- `short_ict_reversal`
- `long_ict_continuation`
- `short_ict_continuation`
- `long_ict_meta`
- `short_ict_meta`

## 10. Model Architecture

Recommended initial backends:

1. XGBoost first
2. TCN second
3. LSTM optional and lower priority

Why this order:

- XGBoost is the best first backend for fast iteration, clearer attribution, simpler failure analysis, and lower event-count requirements.
- TCN is a strong second backend once event counts, setup stability, and sequence-window assumptions settle.
- LSTM is available through the current repo, but it is the least important early research path.

Recommended model groups:

```text
ict_long_reversal_xgb_v1
ict_short_reversal_xgb_v1
ict_long_continuation_xgb_v1
ict_short_continuation_xgb_v1
ict_long_meta_xgb_v1
ict_short_meta_xgb_v1
ict_long_reversal_tcn_v1
ict_short_reversal_tcn_v1
ict_long_continuation_tcn_v1
ict_short_continuation_tcn_v1
```

Training design:

- reuse `model_training/ote_training/ote_xgboost_pipeline.py`
- use backend-specific rankings from `preprocessing/backend_attribution.py`
- retain leakage blocking from the existing trainer
- retain purged walk-forward geometry from the existing trainer
- retain calibration, starting with `platt`
- pass sample weights through exactly as produced by the labeler

Class imbalance handling:

- let the existing trainer use sample weights
- only add further positive-class weighting if diagnostics show event rarity is still overwhelming
- do not oversample sequential data early unless necessary

Metrics emphasis:

- event-based precision/recall style metrics
- AP / PR emphasis
- cost-aware post-threshold evaluation
- stability across folds and time windows

Backend attribution expectations:

- XGBoost should be the first audit surface for feature sanity
- TCN should only be trained after the selected tabular feature set is already well-understood

## 11. Validation and Anti-Leakage Rules

This section is mandatory and strict. Every future implementation phase should treat these as unit-test or assertion requirements.

Required anti-leakage assertions:

- FVGs use only fully closed candles.
- Order blocks are only confirmed after the confirming displacement has closed.
- BOS / CHoCH / MSS use only causal, confirmed swings.
- HTF features use only completed 30m and 1h bars.
- Session highs/lows are only used after they are known.
- Initial Balance features are masked before IB completion.
- Prior RTH and overnight reference levels become fixed only after those sessions complete.
- No setup uses future confirmation bars beyond explicitly defined causal swing confirmation.
- No triple-barrier window may cross roll boundaries.
- No feature may encode the future outcome of the label.
- No scaler, encoder, or calibrator may fit on future data.
- No profile or ICT level may compare mixed contract coordinates.
- Purged walk-forward validation must be used.
- Embargo windows must be applied according to the trainer's purge logic.
- Macro-event exclusions must be computable from data known at the time.

Required sanity triggers:

- Sharpe above 3.0 should trigger an immediate leakage audit.
- Win rate above 65 percent should trigger a leakage audit.
- Suspiciously high setup hit rate should trigger duplicate-event and label-window audits.
- Extremely high importance for time-of-day or simple session fields should trigger an overfit review.
- Sudden large degradation between raw classifier metrics and thresholded policy metrics should trigger an event-definition review.

Required tests:

- `test_causality.py`
  - verify all detectors are closed-bar-only
- `test_setup_detection.py`
  - verify deduplication, cooldown, and session masking
- `test_labeling.py`
  - verify barriers, exclusions, and helper columns
- `test_es_sessions.py`
  - verify overnight, RTH, IB, holiday, and half-day handling

## 12. Threshold Policy and Backtesting

The ICT project should reuse the current FRVP/OTE threshold and backtest machinery rather than creating a bespoke evaluator.

Design requirements:

- ES-aware friction model
- session-specific spread and slippage assumptions
- no unvalidated feature-spread proxy
- regime-aware threshold search
- global threshold fallback
- abstain policy
- cooldown enforcement
- no concurrent positions unless explicitly modeled later
- max trades per day control
- per-session, per-setup, per-regime, per-year, and per-quarter reporting

Recommended initial cost stance:

- start from the current ES defaults in `model_testing/evaluation_costs.py`
- override later only with evidence

Recommended acceptance gates:

- OOS Sharpe target roughly 1.0 to 1.5
- DSR positive and preferably above 0.8
- profit factor above 1.2
- controlled drawdown
- positive expectancy after costs
- stability across years
- no single quarter dominates performance
- no single setup type dominates profits without explanation
- positive composite expectancy share above threshold
- realistic trade frequency around 1 to 3 high-quality trades per day

Registry and report pattern:

- follow the FRVP naming pattern for registry, reports, and candidate roots
- probable future registry path: `models/ict_es_primary_model_registry_current.json`
- probable future report roots:
  - `model_testing/reports/ict_regime_slices/ict_es_primary_<date>`
  - `model_testing/reports/ict_threshold_policies/ict_es_primary_<date>`
  - `model_testing/reports/ict_backtests/ict_es_primary_<date>`

## 13. Reporting and Diagnostics

The ICT pipeline should generate both setup-layer diagnostics and model-layer diagnostics.

Required reports:

- setup count by type
- setup count by side
- setup count by session
- setup count by regime
- setup win rate by setup type
- setup expectancy by setup type
- feature importance by model
- SHAP summary by setup family
- leakage diagnostics
- label distribution diagnostics
- triple-barrier outcome distribution
- event spacing and cooldown diagnostics
- skipped-event reasons
- macro exclusion counts
- roll exclusion counts
- session exclusion counts
- threshold policy summaries
- selected policy counts
- walk-forward equity curve
- breakdown by year / quarter / month
- breakdown by session
- breakdown by composite regime
- breakdown by setup family
- false positive analysis
- top losing setup contexts
- top winning setup contexts

Recommended output families:

- `setup_diagnostics.csv`
- `event_materialization.csv`
- `label_distribution.json`
- `ict_feature_audit.json`
- `policy_selection_summary.json`
- `run_summary.json`
- per-model `fold_summary.csv`
- per-model `selected_test_trades.csv`
- per-model breakdown tables

## 14. Implementation Roadmap

### Status snapshot (updated July 13, 2026)

The repo is no longer at pure design stage. Phases 0 through 8 have now completed a first ES-first ICT baseline cycle, Phase 10 has a first shadow-policy packaging pass, and the live dashboard integration layer is implemented for shadow operation inside `ote_live/`.

Current verification coverage:

- `tests/test_ict_phase1_scaffold.py`
- `tests/test_ict_detectors_phase2.py`
- `tests/test_ict_setup_detector_phase3.py`
- `tests/test_ict_labeling_phase4.py`
- `tests/test_features_builder.py -k "frvp_meta_recipe or ict"`

### Phase 0: Repo audit and design sign-off (`complete`)

- complete this document
- confirm reusable artifacts
- confirm architecture
- confirm ES data requirements

Completed so far:

- established the ES-first `ict/` package direction and reuse plan
- documented the FRVP/OTE infrastructure that ICT will sit on top of
- pinned the revised causal rules for sweeps, IFVG, session-open splits, and barrier geometry

### Phase 1: ICT package skeleton (`complete`)

- create `ict/`
- create configs
- create compatibility shims
- create tests

Completed so far:

- created the `ict/` package, config surfaces, report/layout scaffolding, and compatibility shims
- added package-native feature shims under `ict/feature_sets/` and registry-facing shims under `features/feature_sets/`
- added scaffold tests for package exposure, artifact layout, and recipe/build compatibility

Representative artifacts:

- `ict/config/`
- `ict/pipelines/layout.py`
- `ict/reports/diagnostics.py`
- `features/feature_sets/ict_context.py`
- `features/feature_sets/ict_interactions.py`

### Phase 2: Core detectors (`complete` for v1 detector surface)

- FVG
- order blocks
- liquidity pools
- sweeps
- BOS / CHoCH
- displacement
- premium / discount

Completed so far:

- implemented the causal detector/state surface in `ict/detectors/` and `ict/structure/`
- covered swing confirmation tax, FVG CE/IFVG behavior, sweep reclaim logic, displacement volume confirmation, order-block retests, and premium/discount + DOL behavior with tests
- aligned detector outputs to an ES-first setup engine rather than the old `ict_app/` prototype

Representative artifacts:

- `ict/detectors/fvg.py`
- `ict/detectors/order_blocks.py`
- `ict/detectors/sweeps.py`
- `ict/detectors/displacement.py`
- `ict/detectors/premium_discount.py`
- `ict/detectors/bos_choch.py`
- `ict/structure/swings.py`
- `ict/structure/liquidity.py`

### Phase 3: Setup detector (`complete` for v1 setup layer)

- setup definitions
- event generation
- deduplication
- diagnostics

Completed so far:

- implemented deterministic setup firing for the revised families
- included candidate prioritization, duplicate suppression, spacing/cooldown heuristics, per-session side limits, and fire-rate diagnostics
- implemented the revised setup split for pre-IB vs post-IB session-open reversals and IFVG-as-v1 behavior for Setup 4

Representative artifacts:

- `ict/setups/detector.py`
- `ict/setups/setup_types.py`
- `ict/setups/validators.py`

### Phase 4: Labeling engine (`complete` for v1 event-frame + triple-barrier path)

- triple barrier
- structure-based barriers
- meta-labels
- sample weights
- purge and exclusion logic

Completed so far:

- implemented `ict/labeling/ict_labeling_engine.py` plus the `data/labeling/` compatibility wrapper
- materialized ICT event rows, next-open entries, family-specific barrier geometry, pooled meta labels, helper columns, concurrency-aware sample weighting, and event diagnostics
- added mandatory 1-minute intrabar barrier ordering when 5-minute bars are ambiguous, with exclusion when 1-minute data is unavailable
- covered reversal, continuation, timeout, ambiguity-resolution, and family-vs-meta weighting behavior with tests

Representative artifacts:

- `ict/labeling/ict_labeling_engine.py`
- `data/labeling/ict_labeling_engine.py`

### Phase 5: Feature generation (`partially complete`)

- `ict_context`
- `ict_interactions`
- ICT feature recipe
- attribution compatibility

Completed so far:

- `ict/feature_sets/ict_context.py`
- `ict/feature_sets/ict_interactions.py`
- `features/feature_sets/ict_context.py`
- `features/feature_sets/ict_interactions.py`
- `features/recipes/ict_es_meta.json`

Still to finish:

- validate the event-label/feature alignment on a true prepared-root build
- decide whether to add optional 1-minute refinement/execution-quality features now or defer them
- verify downstream attribution/selection behavior once ICT targets are discoverable by the shared preprocessing stack

### Phase 6: Dataset preparation (`complete` for the first ES prepared-root build)

- build prepared training matrix
- validate class balance
- validate event counts
- generate diagnostics

Completed so far:

- wired ICT target discovery and helper-column compatibility into the shared preprocessing path
- built the first prepared-root artifact set under `artifacts/ict_es_primary/phase04_prepared/`
- validated class-balance and event-readiness outputs for the six baseline ICT targets on real ES data

### Phase 7: XGBoost training (`complete` for the July 13 baseline sweep)

- train first models
- feature importance
- threshold search
- static test diagnostics

Completed so far:

- trained the first six ICT ES baseline XGBoost candidates and registered them in `models/ict_es_primary_model_registry_current.json`
- produced training artifacts under `models/ict_es_primary_xgb_baseline_20260712_v2/<target>/`
- generated threshold-policy outputs under `model_testing/reports/ict_threshold_policies/ict_es_primary_20260713T000016Z/`

### Phase 8: Walk-forward backtest (`complete` for the first policy-evaluation pass)

- ES costs
- session and regime slicing
- acceptance gates

Completed so far:

- ran the first July 13 ICT ES walk-forward report family under `model_testing/reports/ict_backtests/ict_es_primary_20260713T000016Z*/`
- scored candidates on cost-aware post-threshold metrics including Sharpe, WFE, expectancy, and composite/session slices
- preserved the acceptance-gate outputs needed for shadow-bundle selection

### Phase 9: TCN sequence models (`not started`)

- sequence windows
- compare against XGBoost
- attribution and diagnostics

### Phase 10: Policy refinement (`complete` for the first shadow bundle pass)

- prune weak setup/session/regime pockets
- update ICT registry
- select paper-trading candidates

Completed so far:

- packaged the first ICT shadow registry at `models/ict_es_shadow_live_registry_20260713.json`
- materialized per-model live policies under `ote_live/policy_artifacts/ict_es_shadow_20260713/`
- exported unified backtest/bundle summaries under `model_testing/reports/ict_backtests/ict_es_shadow_live_bundle_20260713/`

### Phase 11: Live dashboard integration (`complete`)

- add an ICT tab to the existing Dash app
- export runtime manifests for the shadow bundle
- share the ES live collector with FRVP
- render ICT overlays and recent-setup state in `ote_live`

Completed so far:

- added `scripts/build_ict_shadow_live_bundle.py`, which packages the six July 13 ICT family leaders, writes `live_policy.json` artifacts, and exports the `ote_live/runtime_manifests/ict_es_shadow_20260713/` direction manifests used by the dashboard/runtime stack
- extended `ote_live/dashboard/view_registry.py` and `ote_live/scripts/run_live_dashboard.py` with a first-class `ICT` tab beside OTE and FRVP, including default registry/runtime-manifest paths, model ordering, and a dedicated `runtime_state` scope key of `ICT`
- wired `ote_live/dashboard/view_state.py` and `ote_live/dashboard/charts.py` to persist and render ICT levels, zones, context, and recent setup markers using `ict/setups/detector.py`
- added `ote_live/scripts/run_es_live_collector.py` so FRVP and ICT shadow models can share one IBKR ES ingestion loop and one live store instead of running duplicate ES collectors
- extended `ote_live/ingestion/signals.py` and `ote_live/features/incremental_engine.py` so ICT runtime-state persistence and dashboard-only feature surfaces work from the same shared live feature engine

### 14.1 Next phases after the first real ICT pipeline run

- Harden the shadow bundle through live observation before any promotion decision. The July 13 package is a shadow-evaluation bundle, not a production sign-off.
- Add an ICT orchestration entry point if a single-command retraining flow becomes necessary; the repo still does not have a polished one-shot ICT pipeline runner analogous to the older FRVP phase scripts.
- Compare TCN/LSTM sequence variants against the current XGBoost baseline only after the shadow XGBoost family leaders have been pressure-tested.
- Revisit optional 1-minute execution/refinement features and deeper policy pruning once the live dashboard reveals which setups and regimes are actually recurring.

### 14.2 Items skipped, simplified, or worth double-checking before running the pipeline

- The current Phase 4 labeler uses fixed integer bar horizons as the v1 baseline (`reversal_max_bars`, `session_open_reversal_max_bars`, `continuation_max_bars`). The full vol/session-aware horizon logic described in Section 7.4 is not implemented yet.
- Continuation exits are still simplified. The current engine uses family-specific stop/target geometry plus timeout handling, not a fully developed structure/trailing-exit continuation model.
- ICT label exclusions are still narrower than the full paper. Warmup, missing-next-open, incomplete horizon, invalid ATR/geometry, ambiguous 5-minute bars without resolvable 1-minute ordering, and roll-spanning `contract_id` windows are handled. Macro windows, half-days, lunch-window handling, and thin-session exclusions still need to be added.
- One-minute data is not optional in practice if you want to preserve ambiguous events. The current engine behaves correctly by excluding ambiguous 5-minute windows when aligned 1-minute data is missing, but that means event counts will shrink if the 1-minute coverage is incomplete.
- The shared preprocessing defaults are not yet ICT-aware. `preprocessing/config.py`, synthetic target plumbing, and prepared-root orchestration still need explicit ICT target registration before the first full run.
- There is no full ICT pipeline script yet. Do not assume the presence of a single-command ICT analog to the FRVP training stack just because the detector and labeler components now exist.
- Setup spacing and cooldowns are still heuristic constants. After the first full labeling pass on real data, revisit spacing using realized time-to-first-barrier and event-clustering diagnostics rather than keeping the initial guessed values.
- The current label-quality and HTF-confluence logic is intentionally simple. Before training, inspect real event samples to confirm that `htf_context`, `stop_reference`, `target_reference`, and per-family barrier geometry are behaving sensibly across all setups.
- Average-uniqueness weighting is present, but sequential bootstrap and explicit ICT fold-geometry/embargo validation still belong to the training integration phase. Do not treat Phase 4 alone as the full leakage-control story.
- Setup 4 is implemented as IFVG reversal only, which is intentional. The classic breaker remains deferred until the order-block state machine produces a clean causal definition.

## 15. Open Questions and Research Risks

- Which ICT setup family, if any, has real post-cost edge on ES?
- Are FVG retests predictive, or merely visually intuitive?
- Are order blocks stable enough for causal automation and ML?
- Does CHoCH add value beyond sweep plus displacement?
- Do session-open manipulation reversals survive realistic ES costs?
- Is 1-minute execution refinement genuinely additive, or just a source of overfit?
- Should barriers be ATR-based, structure-based, or hybrid by setup family?
- Do pooled meta models outperform family-specific models?
- Does optional FRVP confluence materially improve ICT setups?
- Does NQ transfer better than ES for some ICT families?
- Does 6E preserve FX-style ICT behavior better than spot EUR/USD?
- Are post-2022 0DTE dynamics changing session-open and sweep behavior?
- How sparse are events by family once causality and exclusion rules are enforced?
- Will the strongest-looking ICT families still deliver the target 1 to 3 trades per day after cooldown, regime filters, and costs?
# ICT Design Paper — Revised Sections 5, 6, 7

> These sections replace the originals in the ICT / SMC design paper. Corrections are confined to the ICT concepts/definitions layer (§5), the setup rule layer (§6), and the labeling rule layer (§7). Where a fix here implies a change in §8 (features), §11 (validation), or §12 (costs), the cross-reference is called out inline but those sections are **not** modified in this file.

## Revision notes (what changed and why)

- **Reversal barrier geometry flipped (§7).** Original paired a *wide* stop with a *near* target, which inverts the ICT reversal risk profile and depresses the triple-barrier base rate. Reversals now use a tight structural stop (sweep extreme + buffer, ATR-floored) and a far target at opposing liquidity.
- **Timeout labeling fixed (§7).** Blanket `timeout → 0` biases the model toward fast movers and penalizes slow-but-correct bets. Timeouts are now labeled by realized-return sign at the vertical barrier, with the blanket-0 variant retained only as an explicit A/B baseline.
- **Intrabar barrier resolution promoted to a hard requirement (§7).** 5m OHLC cannot order stop-vs-target when both sit inside one bar; 1m first-touch ordering is now mandatory for barrier resolution and fill pricing. (1m *feature* refinement stays optional.)
- **Horizons made vol/session-aware (§7).** Fixed bar-count barriers ignore ES intraday vol non-stationarity; continuation cap lifted off a hard 20 bars.
- **Cooldown reconciled with horizon (§7).** Short cooldown against long horizons guarantees overlapping, serially-correlated labels; embargo and uniqueness weighting are now pinned to the horizon.
- **Sweep rule pinned (§5.3, §6 Setup 1).** Removed the unautomatable "fails to hold" wording; one precise wick-pierce-then-close-back rule reused everywhere.
- **Displacement volume made first-class (§5.5).** ES is centralized with reliable volume; volume/relative-volume is a core displacement input, not "optional."
- **New causal primitives added (§5).** Consequent Encroachment (FVG 50%), Inversion FVG (IFVG), OTE retracement band + projection targets, Draw on Liquidity (DOL), and ES reference prices (VWAP, midnight/08:30 ET opens, RTH gap, NWOG/NDOG).
- **PDH/PDL and prior-session anchors disambiguated (§5.7).** RTH-defined vs full-Globex levels are now specified rather than left implicit.
- **Setup 6 contradiction resolved (§6).** IB-false-break requires a completed IB, which conflicts with the IB-masking rule; split into a pre-IB (ONH/ONL + PDH/PDL) variant and a post-IB (Initial Balance) variant.
- **Setup 4 given a causal path (§6).** IFVG substituted as the mechanically-definable reversal-after-failure primitive for v1; the true breaker stays Phase 2.
- **Structure-confirmation lag acknowledged (§5.4).** Causal swing confirmation enters later than hindsight-drawn structure; this tax must be measured (diagnostic belongs in §11).

---

## 5. ICT / SMC Concept Definitions for Automation

All definitions below must be deterministic, causal, and closed-bar-only unless explicitly marked as 1-minute execution refinement.

### 5.1 Fair Value Gaps

Bullish FVG:

- Formation bar index `t` is valid only after bar `t` closes.
- Define a three-candle gap where `low[t] > high[t-2] + min_gap`.
- Zone is `[high[t-2], low[t]]`.
- `min_gap` should be ATR-normalized, with the current generic seed from `features/config.py` (`ict_fvg_min_gap_atr`) retained as the starting parameter.

Bearish FVG:

- Valid only after bar `t` closes.
- Define `high[t] < low[t-2] - min_gap`.
- Zone is `[high[t], low[t-2]]`.

Consequent Encroachment (CE):

- CE is the geometric midpoint of the FVG (`(lower + upper) / 2`) and is a first-class ICT reference level, not a derived afterthought. Price frequently reacts to CE rather than to the far edge of the gap.
- CE must be computed at formation time from the fixed gap bounds, so it is causal by construction.

Inversion Fair Value Gap (IFVG):

- When an active FVG is fully violated (a closed bar trades entirely through the gap against its original direction), the gap flips polarity: a violated bullish FVG becomes a bearish IFVG and vice versa.
- The IFVG inherits the original gap bounds and CE, records the violation bar, and becomes an active opposite-direction zone.
- IFVG is the mechanically clean, causal reversal-after-failure primitive and is the recommended substitute for the harder "breaker" concept in v1 (see §6 Setup 4).

Required tracked state:

- `fvg_id`
- direction
- source timeframe (`5m`, `30m`, or `1h`)
- creation time
- lower and upper bounds
- consequent encroachment (CE) price
- width in ticks and ATR units
- age in bars
- created_by_displacement flag
- mitigated percent
- CE-tapped flag
- fully invalidated flag
- inverted flag and inversion bar (for IFVG)

Feature implications:

- distance to nearest active bullish/bearish FVG
- distance to nearest CE
- whether price is inside the FVG
- whether price has tapped CE
- whether price entered the FVG from premium or discount
- whether the FVG was formed immediately after a liquidity sweep
- whether the zone is an active IFVG (polarity-flipped)
- HTF FVG alignment with 5m setup

### 5.2 Order Blocks

Bullish order block:

- Start from a local bearish candle or bearish candle cluster preceding a bullish displacement sequence.
- Confirm only after the displacement candle or displacement sequence has fully closed.
- Candidate zone should use body-first logic with optional wick extension policy configured explicitly.

Bearish order block:

- Symmetric definition using a bullish candle or bullish cluster preceding bearish displacement.

Required confirmation rules:

- displacement range >= configurable ATR multiple
- displacement close is near candle high/low
- displacement breaks a causal swing or structure level
- block is not valid until displacement is confirmed

Tracked state:

- `order_block_id`
- direction
- source candle range
- body range and full range
- age
- retest count
- mitigation state
- invalidation state
- displacement strength and volume/range intensity

### 5.3 Liquidity Sweeps

Liquidity pools should be deterministic and hierarchical.

Pool types:

- prior confirmed swing high / low
- equal highs / equal lows
- overnight (Globex) high / low
- prior RTH high / low (see §5.7 for the RTH-vs-Globex convention)
- Initial Balance high / low
- prior week high / low

Sweep definition (single canonical rule, reused by §6):

A sweep of a level `L` in the buy-side (upside) direction is confirmed when both conditions hold on closed bars:

1. **Pierce:** some bar's high exceeds `L` by at least `sweep_buffer` (ATR- or tick-normalized), and
2. **Close-back:** within `K` bars of the pierce (`sweep_close_back_bars`, default `K = 1`, max `2`), a bar closes back on the origin side of `L` (i.e., `close < L`).

Sell-side (downside) sweeps are the mirror: a bar's low pierces `L` by at least `sweep_buffer`, then a bar closes back above `L` within `K` bars.

- The prior "closes back below **or fails to hold**" wording is removed. "Fails to hold" is not automatable without a concrete rule; the pierce + close-back-within-`K` formulation replaces it and is the only sweep definition used throughout §6.
- A pierce that is *not* reclaimed within `K` bars is recorded as an **acceptance / breakout** event, not a sweep. This distinction feeds the `failed-sweep` vs `clean-break` features.

Tracked state:

- sweep type (buy-side / sell-side)
- swept level type
- penetration depth in ticks and ATR
- close-back-inside flag
- bars-to-reclaim (0..K)
- failed-sweep flag (pierced but never reclaimed within K → acceptance)
- bars since sweep
- whether the sweep occurred at a session extreme

### 5.4 BOS / CHoCH / Market Structure Shift

Definitions should use confirmed swings only.

- BOS: price breaks the latest confirmed swing in the direction of the prevailing structure.
- CHoCH: price breaks the latest confirmed swing opposite the prior structure state.
- MSS: use as the same event family as CHoCH, but allow the implementation to distinguish "first structure flip after a sweep" from generic CHoCH if that improves diagnostics.

Anti-look-ahead rule:

- Swings must be confirmed by a fixed completed-bar window or causal swing algorithm.
- No structure break may reference a swing point that was not yet confirmed at event time.
- No "future confirmation bar" is allowed beyond the standard completed-bar swing confirmation window.

Causal-confirmation-lag rule (mandatory acknowledgment):

- A non-repainting swing requires `swing_confirm_bars` completed bars on each side before it is confirmed, so every causal BOS/CHoCH/MSS is necessarily detected *after* price has already moved off the true pivot. Discretionary ICT structure is drawn with hindsight; the automated version enters later and worse by construction.
- The implementation must record, per structure event, the price displacement between the true pivot and the confirmation bar (the "confirmation tax"). This is a first-class diagnostic that must be produced before any CHoCH-dependent setup is trusted. (The distribution/report itself lives in §11; the *requirement to capture it* is a definitional obligation here because it constrains what CHoCH can causally mean.)

### 5.5 Displacement

Displacement should be an objective impulse measure, not a visual narrative.

Recommended definition:

- candle or short sequence range >= configurable ATR multiple
- body-to-range ratio above threshold
- close in upper quartile for bullish displacement or lower quartile for bearish displacement
- **volume / relative-volume confirmation is a core component, not optional.** ES trades on a centralized venue with reliable contract volume, so displacement (the automated proxy for institutional participation) must incorporate a volume z-score or relative-volume ratio versus a rolling same-session-phase baseline. A range expansion on below-average volume is down-weighted rather than treated as equivalent to a range expansion on a volume surge.
- bonus score if displacement immediately follows a sweep or creates an FVG

Tracked outputs:

- displacement direction
- displacement score (range + body-location + volume, combined)
- body ATR multiple
- range ATR multiple
- close-location percentile
- volume z-score / relative-volume ratio (required where ES volume is present)
- created FVG flag

### 5.6 Premium / Discount and Optimal Trade Entry

Premium/discount must be defined from causal dealing ranges, not hindsight ranges.

Recommended range anchors:

- most recent confirmed swing low to confirmed swing high on 5m for local dealing range
- 30m confirmed swing anchors for HTF dealing range
- optional 1h anchors for trend context

Derived features:

- current price percentile within range
- discount zone flag (below 50%)
- premium zone flag (above 50%)
- equilibrium proximity (distance to 50%)
- local and HTF range alignment

Optimal Trade Entry (OTE) band — added because the repo has an OTE lineage but the original premium/discount definition stopped at equilibrium:

- Within a confirmed dealing range measured from the impulse leg, define the OTE retracement band as the `0.62`–`0.79` retracement, with `0.705` as the configurable sweet-spot midpoint. All levels are Fibonacci retracements of a *confirmed, causal* impulse leg — never a hindsight leg.
- Derived features: `in_ote_band` flag, distance to `0.705`, and which retracement bucket price currently occupies (`< 0.62`, `0.62–0.79`, `> 0.79`).

Projection / target levels (Draw on Liquidity, DOL):

- Project standard fib extensions of the same impulse leg for target references (configurable set, e.g. the `-0.27` / `-0.62` negative-fib extensions in ICT convention, or `1.0` / `1.5` / `2.0` symmetrical projections). These are candidate DOL targets, subordinate to actual liquidity pools when a pool is nearer.
- **Draw on Liquidity is a first-class concept**, not merely an implicit `target_reference`. Define the DOL feature as the signed distance to the nearest *untapped* HTF liquidity pool in the hypothesized trade direction (opposing-side external liquidity for reversals, continuation-direction external liquidity for continuations). This is one of the most central ICT constructs and must exist as an explicit feature and as the default target anchor in §7.

### 5.7 Session Liquidity and ES Reference Prices

For ES, the following session levels are meaningful and should be first-class ICT references. **Level-definition conventions are fixed here to prevent cross-day corruption:**

Prior-session anchors (convention pinned):

- **prior-day high / low = prior RTH session high / low** (09:30–16:00 ET), matching standard index-ICT usage. The full-Globex prior-day high/low are tracked *separately* as overnight/Globex levels and must never be conflated with PDH/PDL. Sweeps depend on which is used, so the two are distinct, explicitly-named references.
- prior RTH close
- prior week high / low (RTH-defined, with the Globex variant tracked separately if used)

Current-session and overnight anchors:

- overnight (Globex) high / low
- current RTH open (09:30 ET)
- Initial Balance high / low (first 60 minutes of RTH; not fixed until IB completes)

ES-specific reference prices (added — these are strong index-futures magnets and several are literal ICT constructs):

- **Midnight open (00:00 ET)** and **08:30 ET open** — ICT's daily-bias framework keys off these, not only the 09:30 RTH open. Track price relative to each.
- **Session and anchored VWAP** — session VWAP plus anchored VWAP from meaningful causal anchors (RTH open, prior swing, prior sweep). VWAP and its bands are reliable on centralized ES and act as reactive references.
- **RTH-close → next-open gap** — the overnight gap between prior RTH close and current RTH open, as both a level pair and a "gap open / fill" state.
- **NWOG / NDOG** — New Week Opening Gap and New Day Opening Gap (the gap between one session's close and the next session's open), tracked as ICT reference zones with their own CE.

Level-fixing rule:

- These levels are fixed only when the underlying session is complete or, for current-day levels such as IB, when the causal completion condition has been met. Overnight/Globex extremes are fixed at RTH open; IB extremes are fixed at IB completion; VWAP updates causally bar-by-bar.
- (Cross-reference to §11: prior-session and prior-week levels carried across a contract roll must be recomputed in the active contract's absolute coordinates, or they will manufacture phantom sweeps.)

---

## 6. ICT Setup Rule Layer

Setup detection should run on finalized 5-minute ES bars. The detector should emit candidate events and allow optional 1-minute refinement for entry confirmation only, not for defining the original 5-minute setup existence.

Shared rules for all setups:

- all setup detection is closed-bar-only on 5m
- all sweeps use the single canonical sweep rule from §5.3 (pierce by ≥ `sweep_buffer`, then close back on origin side within `K` bars)
- same-side same-family events should honor a cooldown (see §7 for how cooldown is reconciled with label horizon)
- duplicate events anchored to the same structural level should be collapsed
- every setup must specify a family (`reversal` or `continuation`)
- every setup must specify a base holding horizon and stop anchor consistent with the §7 barrier geometry (tight structural stop for reversals; structure/trailing exit for continuations)

### Setup 1: Liquidity Sweep + Reclaim

- Side hypothesis: reversal.
- Required conditions:
  - sell-side sweep (per §5.3) of a support-side level for long, or buy-side sweep of a resistance-side level for short
  - the sweep's own close-back condition provides the reclaim; the trigger bar is the close-back bar within `K`
  - no immediate invalidation through the opposite side of the trigger structure
- Optional confluence:
  - displacement appears within `N` bars
  - nearby FVG (or its CE) or order block
  - premium/discount / OTE-band alignment
- Invalidation:
  - price closes beyond the sweep extreme by more than a configured invalidation buffer
- Minimum spacing:
  - 6 bars per side per swept-level family (subject to §7 reconciliation)
- Expected move type:
  - reversal
- Stop / target:
  - stop just beyond the sweep extreme + buffer (ATR-floored); target = DOL / opposing liquidity (see §7)
- Label horizon:
  - reversal family horizon per §7 (vol/session-aware)
- 1-minute refinement:
  - require a micro reclaim or micro BOS back in trade direction inside 3 to 5 minutes

Pseudocode:

```text
if sweep_detected(level_type, side_opposite_target)     # §5.3 canonical rule
and close_back_inside(level) within K bars
and not invalidated_same_bar
then emit setup_type = sweep_reclaim
```

### Setup 2: Sweep + Displacement + FVG Retrace

- Side hypothesis: reversal with confirmation.
- Required conditions:
  - initial liquidity sweep (§5.3)
  - displacement in opposite direction within `N` bars (§5.5, volume-aware)
  - displacement creates an FVG
  - price retraces into the FVG (CE is the preferred retrace reference) without invalidating the displacement leg
- Optional confluence:
  - CHoCH after sweep (subject to the §5.4 confirmation-lag caveat)
  - HTF discount/premium or OTE-band alignment
- Invalidation:
  - FVG fully fails (closes through; note this creates an IFVG per §5.1) or price closes through the structural stop level
- Minimum spacing:
  - 6 bars per direction
- Expected move type:
  - reversal
- Label horizon:
  - reversal family horizon per §7
- 1-minute refinement:
  - require 1m rejection inside the FVG (near CE) or 1m micro BOS away from the zone

Pseudocode:

```text
if sweep_detected
and displacement_confirmed          # includes volume z-score
and fvg_created
and retrace_into_fvg (>= CE)
then emit setup_type = sweep_displacement_fvg
```

### Setup 3: Order Block Retest After MSS / CHoCH

- Side hypothesis: reversal after structure shift.
- Required conditions:
  - sweep of external or internal liquidity (§5.3)
  - CHoCH / MSS in the new direction using confirmed swings only (§5.4; enter only after causal confirmation, and record the confirmation tax)
  - last opposing candle or candle cluster before displacement becomes the order block
  - price retests the order block while structure remains intact
- Optional confluence:
  - overlapping FVG / CE
  - discount/premium or OTE-band alignment
- Invalidation:
  - close through the order block invalidation boundary
- Minimum spacing:
  - 8 bars per direction
- Expected move type:
  - reversal
- Label horizon:
  - reversal family horizon per §7
- 1-minute refinement:
  - require a 1m rejection or 1m structure continuation off the order block

Pseudocode:

```text
if sweep_detected
and choch_confirmed                 # causal, confirmation-lagged
and order_block_confirmed_after_displacement
and retest_of_order_block
then emit setup_type = ob_retest_after_mss
```

### Setup 4: Inversion FVG / Breaker (reversal after failed zone)

- Side hypothesis: reversal after a prior zone fails.
- **v1 stance (changed):** implement the **Inversion FVG (IFVG)** variant now, not the classic breaker. When an active FVG is fully violated it flips polarity (§5.1) and becomes a clean, causal opposite-direction zone; a retest of the resulting IFVG (CE as the reference) is the v1 signal.
- **Breaker stance:** the classic breaker (failed order block) remains Phase 2 until a clean causal definition falls out of the order-block state machine. IFVG replaces it as the v1 reversal-after-failure primitive because it is mechanically unambiguous and already tracked.
- Required conditions (IFVG variant):
  - a previously active FVG is fully violated by a closed bar → IFVG created
  - price retests the IFVG (>= CE) while broader structure supports the reversal direction
- Invalidation:
  - close fully back through the IFVG (re-inversion)
- Minimum spacing:
  - 8 bars per direction
- Expected move type:
  - reversal
- Label horizon:
  - reversal family horizon per §7

Pseudocode:

```text
if fvg_fully_violated               # produces IFVG, §5.1
and retest_of_ifvg (>= CE)
and structure_supports_new_direction
then emit setup_type = ifvg_reversal
```

### Setup 5: Premium / Discount (OTE) Continuation Pullback

- Side hypothesis: continuation.
- Required conditions:
  - HTF trend state aligned on 30m and optionally 1h
  - pullback into discount (longs) or premium (shorts), preferably into the OTE band (§5.6)
  - supportive zone such as bullish FVG / bullish order block for longs, symmetric for shorts
  - no confirmed structure failure against trend
- Optional confluence:
  - prior sweep in trend direction
  - FRVP confluence later
- Invalidation:
  - HTF trend state breaks or local structure fails
- Minimum spacing:
  - 6 bars per side
- Expected move type:
  - continuation
- Stop / target:
  - stop below the impulse origin / supportive zone; target = next DOL in trend direction with structure/trailing exit (see §7 — continuations are not hard-capped at a fixed bar count)
- Label horizon:
  - continuation family horizon per §7
- 1-minute refinement:
  - require 1m pullback exhaustion or micro continuation trigger

Pseudocode:

```text
if htf_trend_up
and pullback_into_discount (prefer OTE band)
and supportive_bull_zone_present
and local_structure_intact
then emit setup_type = premium_discount_continuation
```

### Setup 6a: Session-Open Manipulation Reversal — Pre-IB (known levels only)

- Side hypothesis: ES-specific reversal, tradable from RTH open.
- **Split rationale (changed):** the original single Setup 6 referenced an "early false break of Initial Balance," which cannot be evaluated before IB completes and therefore contradicts the IB-masking rule (§11). The pre-IB variant may reference **only levels already known at the open** — overnight (Globex) high/low, PDH/PDL, prior RTH close, midnight/08:30 opens.
- Required conditions:
  - RTH-open sweep (§5.3) of overnight high/low or PDH/PDL
  - reclaim back inside the swept level
  - preferably displacement or failure-follow-through confirmation
- Optional confluence:
  - nearby HTF liquidity, FVG/CE, or VWAP reaction
- Invalidation:
  - clean acceptance outside the manipulated level
- Minimum spacing:
  - once per side per opening-session pattern
- Expected move type:
  - reversal
- Label horizon:
  - session-open reversal horizon per §7 (short)
- 1-minute refinement:
  - require a 1m rejection or micro structure shift immediately after reclaim

Pseudocode:

```text
if early_rth_sweep_of(on_high, on_low, pdh, pdl)   # no IB reference pre-completion
and reclaim_inside_reference
then emit setup_type = session_open_manipulation_pre_ib
```

### Setup 6b: Session-Open Manipulation Reversal — Post-IB (Initial Balance)

- Side hypothesis: ES-specific reversal, evaluated only after IB completes.
- Required conditions:
  - IB is complete (IB features unmasked, §11)
  - false break of Initial Balance high/low (sweep per §5.3, then reclaim back inside IB)
  - preferably displacement or failure-follow-through confirmation
- Optional confluence:
  - nearby HTF liquidity, FVG/CE, or VWAP reaction
- Invalidation:
  - clean acceptance outside IB (IB extension)
- Minimum spacing:
  - once per side per opening-session pattern
- Expected move type:
  - reversal
- Label horizon:
  - session-open reversal horizon per §7 (short)
- 1-minute refinement:
  - require a 1m rejection or micro structure shift immediately after reclaim

Pseudocode:

```text
if ib_complete
and false_break_of(ib_high, ib_low)      # sweep + reclaim inside IB
then emit setup_type = session_open_manipulation_post_ib
```

### Setup 7: Displacement Continuation After Liquidity Raid

- Side hypothesis: continuation after initial raid and impulse.
- Required conditions:
  - liquidity raid (sweep, §5.3) occurs against the eventual move direction
  - strong displacement (§5.5, volume-aware) starts a directional move
  - shallow pullback holds above bullish impulse origin for longs or below bearish origin for shorts
- Optional confluence:
  - FVG (or CE) or order-block pullback inside trend
- Invalidation:
  - full failure through the impulse origin
- Minimum spacing:
  - 8 bars per direction
- Expected move type:
  - continuation
- Stop / target:
  - stop through the impulse origin; target = next DOL in trend direction with structure/trailing exit (§7)
- Label horizon:
  - continuation family horizon per §7
- 1-minute refinement:
  - require 1m continuation alignment after the pullback

Pseudocode:

```text
if raid_detected
and displacement_starts_new_move    # volume-confirmed
and pullback_holds_impulse_origin
then emit setup_type = displacement_continuation_after_raid
```

---

## 7. Labeling Design

The new label engine should live at `ict/labeling/ict_labeling_engine.py`, with a compatibility wrapper at `data/labeling/ict_labeling_engine.py`.

The rule layer should generate event rows with a schema like:

```text
event_time
side
setup_type
setup_family
anchor_level
entry_price
stop_reference
target_reference          # DOL / opposing liquidity by default
htf_context
sweep_type
fvg_id                    # includes IFVG lineage where applicable
ce_price                  # consequent encroachment of the operative gap
order_block_id
displacement_id
displacement_volume_z
session_phase
```

Recommended label design:

- Produce separate long/short and reversal/continuation labels.
- Also produce pooled meta labels by direction.
- Match FRVP helper-column conventions exactly so existing preprocessing can discover them.

Recommended initial targets:

```text
label_long_ict_reversal
label_short_ict_reversal
label_long_ict_continuation
label_short_ict_continuation
label_long_ict_meta
label_short_ict_meta
```

### 7.1 Barrier design

- Primary outcome engine should be triple barrier.
- `meta_label = 1` if the target barrier is touched before the stop barrier.
- `meta_label = 0` if the stop barrier is touched first.
- **Intrabar resolution is mandatory (changed to a hard requirement).** On a 5m bar whose range spans both stop and target, OHLC alone cannot order the two touches, and for tight-stop reversals this ambiguity concentrates precisely on the decisive bars. The labeling engine **must** resolve first-touch using the 1-minute series (and use 1m for realistic fill pricing). This is a barrier-resolution obligation of the labeler; it is distinct from — and does not depend on — the optional 1m refinement *features*. Where 1m data is genuinely unavailable for a window, that window is excluded rather than resolved by a guess.

### 7.2 Barrier references (geometry corrected)

The stop/target pairing is defined **per family** so that geometry matches the ICT trade logic rather than inverting it.

**Reversal families (Setups 1, 2, 3, 4, 6a, 6b):**

- **Stop = tight structural stop:** just beyond the swept liquidity extreme (or the operative zone edge — order block / FVG / IFVG boundary), plus a small buffer, **floored** by a minimum ATR stop so it is never absurdly tight. The stop is deliberately *tight*, because the setup's whole premise is that price should not return through the sweep extreme.
- **Target = Draw on Liquidity:** the opposing liquidity pool (§5.6 DOL), taken **even when far**. If no clean pool exists, fall back to a fib projection / ATR-multiple target. The target is deliberately *far*.
- This is the reverse of the original document, which paired a wide stop with a near target and thereby suppressed the target-first base rate and inverted the reversal's risk profile.

**Continuation families (Setups 5, 7):**

- **Stop = impulse origin / supportive-zone failure**, ATR-floored.
- **Target / exit = next DOL in the trend direction, with a structure-based or trailing exit** rather than a fixed near target. Continuations are allowed to run.

### 7.3 Timeout handling (corrected)

- The vertical (time) barrier is **not** blanket-mapped to `0`. A timeout is labeled by the **sign of realized return at the vertical barrier**: `meta_label = 1` if the position is in profit (beyond a configurable breakeven-plus-costs threshold) at timeout, else `0`. This is consistent with meta-labeling's question ("did the bet make money?") and avoids penalizing slow-but-correct reversals that commonly time out flat-to-green in low-liquidity windows.
- The original blanket `timeout → 0` rule is retained **only** as an explicit A/B baseline (`timeout_policy = zero` vs `timeout_policy = pnl_sign`) so its effect on base rate and expectancy can be measured rather than assumed.

### 7.4 Holding horizons (made vol/session-aware)

Fixed bar-count barriers ignore ES intraday volatility non-stationarity (an 8-bar window at the 09:30 open and an 8-bar window in the 12:00–13:30 lull are not comparable amounts of opportunity). Horizons are therefore specified as vol/session-aware rather than as raw bar counts:

- Express the max-holding vertical barrier as a **volatility-scaled** duration (e.g., a target ATR-distance budget, or a bar count scaled by a rolling same-session-phase vol estimate), with the bar-count ranges below as calm-RTH anchors, not hard caps.
- Calm-RTH anchor ranges:
  - reversal families: ~8 to 18 bars
  - session-open reversal (6a/6b): ~6 to 12 bars
  - continuation families: **no hard upper cap** — governed by the structure/trailing exit in §7.2; the ~8–20 bar range is only a floor/typical, because ES trend-day continuations routinely run well past 100 minutes and a hard 20-bar cap truncates the best trades and distorts the base rate.
- **Lunch-window handling:** for events whose vertical-barrier window would run mostly through the 12:00–13:30 ET lull, either freeze the clock through the lull or exclude late-morning entries whose window is dominated by lunch, so that a dead-liquidity cohort does not contaminate a family's base rate with forced timeouts.

### 7.5 Event hygiene and cooldown/horizon reconciliation (corrected)

- de-duplicate events by side, setup type, and anchor zone
- **Cooldown and horizon are coupled, not independent.** A 4–8 bar cooldown against 8–20+ bar horizons guarantees overlapping barrier windows → serially correlated labels and inflated effective sample size. Reconcile as follows:
  - Set the **embargo ≥ max_horizon + `swing_confirm_bars`** so late training-fold outcomes cannot leak into the test fold (pin this explicitly rather than leaving it to "trainer purge logic"; an 18-bar horizon otherwise leaks the last 18 train bars).
  - Apply **average-uniqueness (concurrency) weighting** and, for the tabular backend, **sequential bootstrap**, so overlapping labels are correctly down-weighted rather than double-counted.
  - Iteratively set the training-event minimum spacing toward the **median realized time-to-first-barrier** (computable only after a first labeling pass), instead of a guessed fixed cooldown.
- compute overlap/concurrency counts
- down-weight highly concurrent events
- optionally mark `neg_ok_*` only where the existing FRVP/OTE logic would consider negatives safe

Exclusion rules should mirror FRVP where possible:

- macro event windows from `frvp/calendars/macro.py`
- half-days / thin-session rules from ES session logic
- roll-spanning windows from `frvp/continuity/continuous_contract.py` (and, per §5.7, liquidity levels carried across a roll must be recomputed in the active contract's absolute coordinates)
- warmup mask
- missing-next-open or broken bar continuity
- any event whose barrier window crosses a contract-roll boundary
- any event whose barrier window cannot be resolved on 1m (per §7.1)

1-minute refinement labels:

- Keep the base 5m event labels as canonical.
- Optionally add a 1m refinement success flag or 1m execution-quality flag for later research.
- Do not let 1m refinement redefine whether the 5m setup existed in the first place. (Note the separation from §7.1: 1m is *mandatory* for barrier resolution but *optional* as a refinement feature.)

## 16. Final Recommendation

- Build ICT as its own ES-first package parallel to `frvp/`.
- Reuse FRVP's ES session, macro-calendar, roll, cost, threshold, registry, and backtest infrastructure wherever possible.
- Reuse OTE's triple-barrier, trend-scan, sample-weighting, model-training, and attribution infrastructure.
- Treat `features/feature_sets/ict_context.py` as a useful seed and compatibility surface, not as the full solution.
- Treat `ict_app/` as prototype/reference material only unless an artifact passes a fresh causality and ES-compatibility review.
- Start with XGBoost pooled and family-specific meta-labelers.
- Move to TCN only after the event layer, feature set, and labels are stable.
- Promote nothing until walk-forward, post-cost, regime-sliced evaluation passes realistic acceptance gates.

Bottom line: the repo already contains most of the infrastructure needed to support an ICT ES research stack, but it does not yet contain a canonical ES-first ICT event layer. The missing work is not the model stack. The missing work is the deterministic setup layer, ICT label materialization, and ES-specific structural feature package. That is exactly why the correct next step is a separate `ict/` package built on top of the existing FRVP and OTE infrastructure rather than another round of ad hoc feature additions.

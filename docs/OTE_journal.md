# OTE (EURUSD) design, plans and research history (2025-11 to 2026-07)

Consolidated, chronological record of the EURUSD OTE stack: the original system architecture vision (the root README until 2026-10), the OTE training-workflow methodology report (2026-03-21), the end-to-end runbook (2026-03-31), the production real-time signal system plan (2026-04), the 8-model live promotion plan (2026-05-27), the multi-family TCN retrain and post-training production plans (2026-05), the 1-minute retrain plan (2026-06-01) and the policy-layer/architecture research paper (2026-07-12). Later OTE work (B1 OTE contracts, B2 timestamp repair, section 3/4 and the v2 retrain) is in the B1/B2 records and `section4_progress.md`.

Contents:
- [Original system architecture vision (root README, 2025-11)](#readme)
- [Optimal Trade Entry Model Training Workflow (2026-03-21)](#ote_training_workflow_report)
- [OTE End-to-End Workflow](#end-to-end-readme)
- [Production-Ready Real-Time Trading Signal System Plan](#production_realtime_signal_system_plan)
- [8-Model OTE Live Promotion Plan](#model_promotion_and_ote_live_8_model_plan)
- [OTE multi-family TCN v1/v2 retrain plan](#ote_multi_family_tcn_v1_v2_retrain_plan)
- [Post-training production plan](#post_training_production_plan)
- [1-Minute OTE Training Plan](#model_retrain_plan_1min)
- [The Policy Layer Is the Bottleneck, Not the Backbone](#policy_layer_and_architecture_research)


<a id="readme"></a>

## Original system architecture vision (root README, 2025-11)

*Merged 2026-10-09 from `README.md`.*

I'll help you compile and organize your extensive trading system research into a coherent architecture. Let me analyze your uploaded documents and create a comprehensive system design.

### Complete EURUSD Automated Trading System Architecture

#### Executive Summary

Based on your research across ICT Smart Money concepts, deep learning pipelines, and institutional trading strategies, you need a **hybrid multi-model system** that combines:
- Rule-based algorithms for well-defined patterns (ICT concepts)
- Deep learning models for complex pattern recognition and signal generation
- Multiple specialized models for different market tasks
- Ensemble approaches for robust decision-making

---

#### System Architecture Overview

##### **Tier 1: Market Intelligence Layer** (Rule-Based + Pattern Detection)
##### **Tier 2: Signal Generation Layer** (Deep Learning Models)
##### **Tier 3: Risk & Execution Layer** (Rule-Based Logic)
##### **Tier 4: Ensemble Decision Layer** (Model Aggregation)

---

#### TIER 1: Market Intelligence Layer (Rule-Based Algorithms)

##### 🎯 **ICT Smart Money Concepts Detection** (Rule-Based)

###### **1.1 Fair Value Gap (FVG) Detector** ✅
**Type:** Pure Rule-Based Algorithm  
**Implementation:** Already available in `smartmoneyconcepts` library
```python
Purpose: Identify 3-candle inefficiency patterns
Input: OHLC data
Output: Bullish/Bearish FVG zones with top/bottom prices
Logic: 
  - Bullish FVG: previous_candle.high < next_candle.low
  - Bearish FVG: previous_candle.low > next_candle.high
  - No overlap between first and third candle wicks
Sensitivity: Configurable (Very Aggressive → Very Defensive)
```

###### **1.2 Order Block Detector** ✅
**Type:** Hybrid (Rule-Based + Volume Analysis)  
```python
Purpose: Identify institutional accumulation/distribution zones
Input: OHLC + Volume + Swing points
Output: Order block zones with volume confirmation
Logic:
  - Find last bearish candle before bullish move (bullish OB)
  - Find last bullish candle before bearish move (bearish OB)
  - Validate with volume aggregation (current + 2 previous candles)
  - Volume threshold: 2x average volume
  - Session filter: Only during high-volume sessions
Enhancement: Add ML confidence scoring layer
```

###### **1.3 Liquidity Sweep Detector** ✅
**Type:** Multi-Factor Rule-Based
```python
Purpose: Detect liquidity grabs and stop hunts
Input: OHLC + Recent highs/lows + Volume
Output: Buy-side/Sell-side sweep signals
Logic:
  - Identify clustered highs/lows (50-period lookback)
  - Detect when price takes out recent extremes
  - Distinguish: Temporary sweep (reversal) vs Run (continuation)
  - Volume confirmation required
  - Static + Dynamic sensitivity parameters
```

###### **1.4 Market Structure Break Analyzer** ✅
**Type:** Rule-Based
```python
Purpose: Identify trend continuation vs reversal
Input: OHLC + Swing highs/lows
Output: BOS (Break of Structure) or CHoCH (Change of Character)
Logic:
  - BOS: Price breaks recent swing high/low in trend direction
  - CHoCH: Price breaks counter-trend, signals reversal
  - Close-break validation option
  - Multi-timeframe confirmation
```

###### **1.5 Session-Based Context Filter** ✅
**Type:** Pure Rule-Based
```python
Purpose: Filter trades by optimal market sessions
Sessions:
  - Asian (8PM-2AM EST): Range-bound, liquidity setup
  - London Open (2AM-5AM EST): High volatility, liquidity sweeps
  - London (5AM-11AM EST): Trending moves
  - NY Open (9:30AM-11AM EST): Reversal patterns, FVG fills
  - NY (9AM-4PM EST): Order block retests
Logic: Only trade patterns that historically work in each session
```

##### 🔍 **Pattern Recognition Systems**

###### **1.6 CNN-Based Chart Pattern Detector** ⚠️ (Requires Data I Don't Have)
**Type:** Computer Vision Model (CNN)  
**Training Data Needed:** 2000+ labeled chart images
```python
Purpose: Detect complex visual patterns
Architecture: 2D CNN (ResNet-style)
Input: Candlestick chart images (grayscale, 224x224)
Output: Pattern classifications with confidence
Patterns to Detect:
  - Head & Shoulders, Inverse H&S
  - Double/Triple Tops/Bottoms
  - Ascending/Descending Triangles
  - Wedges, Flags, Pennants
  - Cup & Handle
Training: 
  - 9000+ labeled images per pattern
  - Data augmentation (rotation, zoom, noise)
  - Transfer learning from ImageNet
Performance Target: >85% precision
Note: High computational cost, use for confirmation only
```

###### **1.7 Candlestick Pattern Detector** ✅
**Type:** Rule-Based (TA-Lib) 
```python
Purpose: Identify single/multi-candle reversal patterns
Library: TA-Lib (already integrated)
Patterns (Top Priority from your analysis):
  - CDL3BLACKCROWS, CDL3INSIDE
  - CDLENGULFING, CDLHAMMER, CDLHARAMI
  - CDLMARUBOZU, CDLSHOOTINGSTAR
  - CDLMORNINGSTAR, CDLEVENINGSTAR
Note: Your analysis showed these had instability - use only as 
supplementary confirmation, not primary signals
```

---

#### TIER 2: Signal Generation Layer (Deep Learning Models)

##### 🤖 **Core Trading Signal Models** (Already Implemented)

###### **2.1 Long Signal Classification Model**
**Architecture:** Multi-Architecture Ensemble  
**Status:** ✅ Implemented (from your uploaded docs)
```python
Purpose: Predict optimal long entry points
Target: long_signal (binary)
Input Shape: [timesteps=240, features=66]
Optimized Lag Periods: [61, 93, 64, 60, 77]

Best Architectures (train all, ensemble top 3):
  1. BiLSTM + Attention (best for temporal patterns)
  2. Transformer (best for long-range dependencies)
  3. Conv1D-LSTM Hybrid (best for local + temporal)
  4. TCN (Temporal Convolutional Network)
  5. ResNet-inspired (for deep feature extraction)

Training Strategy:
  - Progressive training (20% warmup → full dataset)
  - Focal loss (gamma=2, alpha=0.25) for class imbalance
  - Time Series CV (5-fold)
  - Bayesian hyperparameter optimization
  - Early stopping with NaN detection

Top Features (from your analysis):
  1. Volume (primary)
  2. WILLR, STOCHRSI_K, CCI (momentum)
  3. ADOSC, RSI, CMO (oscillators)
  4. ADX, PLUS_DI, MINUS_DI (trend)
  5. HT_DCPHASE (cycle analysis)

Output: Probability [0-1], threshold at 0.70 for high-confidence signals
```

###### **2.2 Short Signal Classification Model**
**Architecture:** Same as Long Signal (independent pipeline)  
**Status:** ✅ Implemented
```python
Purpose: Predict optimal short entry points
Target: short_signal (binary)
Optimized Lag Periods: [70, 24, 10, 74, 39]
Note: Separate lag optimization reflects different market dynamics
Same architecture suite and training strategy as long signals
Output: Probability [0-1], threshold at 0.70
```

###### **2.3 Returns Forecasting Models** (Regression)
**Purpose:** Magnitude and timing prediction

**Model 2.3a: 1-Period Returns Forecaster**
```python
Target: returns_1 (1-period ahead return)
Architecture: LSTM, GRU, Transformer
Forecast Horizon: 1 timestep
Use Case: Short-term momentum confirmation
Output: Expected return value, used to filter signals
Threshold: Long if returns_1 > +0.001, Short if < -0.001
```

**Model 2.3b: 5-Period Returns Forecaster**
```python
Target: returns_5 (5-period ahead return)
Architecture: Conv1D-LSTM, TCN, Transformer
Forecast Horizon: 5 timesteps
Use Case: Medium-term trend validation
Output: Expected return value + confidence interval
Threshold: Long if returns_5 > +0.002, Short if < -0.002
Critical: 93.2% profit rate achieved in your analysis
```

##### 🎯 **Specialized Prediction Models** (NEW - You Need to Build These)

###### **2.4 Market Regime Classification Model**
**Type:** LSTM Classifier  
**Status:** ✅
```python
Purpose: Classify current market state
Architecture: Stacked LSTM (2-3 layers) or Transformer
Input: Multi-timeframe OHLCV (1m, 5m, 15m, 1h, 4h, daily)
Target: Market regime classification
Classes:
  0: Strong Uptrend
  1: Weak Uptrend  
  2: Ranging (consolidation)
  3: Weak Downtrend
  4: Strong Downtrend
  5: High Volatility Breakout
  6: Low Volatility Compression

Features:
  - ADX (trend strength)
  - ATR, Bollinger Band Width (volatility)
  - Rolling correlation between timeframes
  - Volume profiles
  - Rate of change across timeframes

Training Data: Label historical data using:
  - ADX thresholds (>25 = trending, <20 = ranging)
  - ATR percentiles (>75th = high vol, <25th = low vol)
  - Price action patterns

Usage: Different ICT patterns work better in different regimes
  - Trending: Order blocks, FVG fills work better
  - Ranging: Liquidity sweeps more reliable
  - Breakout: Wait for confirmation before entry
```

###### **2.5 Pattern Validation Model**
**Type:** CNN-LSTM Hybrid  
**Status:** ✅
```python
Purpose: Validate quality of detected ICT patterns
Architecture: CNN for spatial features → LSTM for temporal context
Input: 
  - Chart image (100 candles) as 2D array
  - Pattern metadata (FVG/OB/Sweep detected by rules)
  - Volume profile
Target: Pattern quality score [0-1]

Training Data (2000+ examples per pattern type):
  - Run rule-based detectors on 2+ years of data
  - Label outcomes: Did pattern lead to profitable trade?
  - Measure: Max favorable excursion, max adverse excursion
  
Classes per Pattern:
  - High Quality (>80% success rate)
  - Medium Quality (50-80% success)
  - Low Quality (<50% success)

Usage: Filter out low-quality patterns before signaling
Only trade patterns with quality score >0.75
```

###### **2.6 Multi-Timeframe Confluence Model**
**Type:** Transformer or Multi-Stream CNN  
**Status:** ✅
```python
Purpose: Analyze alignment across timeframes
Architecture: Multi-input Transformer
Inputs (parallel streams):
  - 1-minute data (last 240 bars)
  - 5-minute data (last 100 bars)
  - 15-minute data (last 50 bars)  
  - 1-hour data (last 24 bars)
  - 4-hour data (last 20 bars)

Output: Confluence score [0-1] and bias direction
Logic:
  - High confluence = all timeframes agree on direction
  - Mixed confluence = some timeframes conflicting
  
Features per timeframe:
  - Trend direction (EMA crossovers)
  - Momentum (RSI, STOCHRSI)
  - Volume confirmation
  - Key level proximity (support/resistance)

Usage: Only take trades with confluence >0.70
Helps avoid counter-trend trades and false breakouts
```

###### **2.7 Order Flow Analyzer**
**Type:** LSTM or Transformer  
**Status:** ⚠️ Need to Build (Requires Order Flow Data I Don't Have)
```python
Purpose: Analyze institutional buying/selling pressure
Architecture: Bidirectional LSTM
Input: Order flow features (if available):
  - Cumulative Volume Delta
  - Bid/Ask imbalances
  - Aggressive buy/sell ratios
  - Volume at price levels
  - Absorption events
  - Time & Sales data

Alternative (if no order flow data):
  Use volume-based proxies:
  - Volume spikes relative to average
  - Up volume vs Down volume
  - Volume-price divergences
  - Volume profile shape

Output: Pressure score [-1 to +1]
  +1 = Strong buying pressure
  -1 = Strong selling pressure
  
Training: Labeled with subsequent price movements
Target: Predict pressure that leads to continuation

Usage: Confirm ICT patterns with order flow alignment
Example: Bullish Order Block + Buying Pressure = Strong Long
```

---

#### TIER 3: Risk & Execution Layer (Rule-Based)

##### 🛡️ **Risk Management Algorithms** (Pure Rule-Based)

###### **3.1 Position Sizing Calculator**
```python
Purpose: Dynamic position sizing based on signal quality
Input: 
  - Signal confidence from models
  - Account balance
  - Current drawdown
  - Recent win rate
  - ATR (for stop distance)

Logic:
  base_risk = account_balance * 0.01  # 1% risk per trade
  
  # Confidence adjustment
  confidence_multiplier = signal_confidence  # 0.70-0.95
  
  # Drawdown adjustment
  if daily_pnl < -max_daily_loss * 0.5:
      confidence_multiplier *= 0.5  # Reduce size
  
  # Win rate adjustment
  recent_win_rate = wins / trades (last 20 trades)
  if recent_win_rate < 0.40:
      confidence_multiplier *= 0.5  # Reduce on losing streak
  
  # Calculate contracts
  risk_amount = base_risk * confidence_multiplier
  stop_distance = entry_price - stop_price
  contracts = int(risk_amount / (stop_distance * contract_multiplier))
  
  return max(1, min(contracts, max_position_size))

Max Position Size: 2-3 contracts for ES/NQ futures
Max Daily Loss: $500 (adjustable)
```

###### **3.2 Dynamic Stop Loss & Take Profit**
```python
Purpose: Adaptive stop/target based on pattern type

For ICT Patterns:
  FVG Trades:
    - Stop: Below/above FVG zone (+ 1 ATR buffer)
    - Target: Opposite FVG or next OB level
    - R:R minimum: 1:2
  
  Order Block Trades:
    - Stop: Beyond OB zone (+ 0.5 ATR)
    - Target: Previous swing high/low
    - R:R minimum: 1:3
  
  Liquidity Sweep Trades:
    - Stop: Beyond swept level (+ 0.25 ATR)
    - Target: 50% retracement or next structure
    - R:R minimum: 1:2

Trailing Stop Logic:
  - Move to break-even when profit = 1R
  - Trail by 0.5 ATR once profit = 2R
  - Lock in 50% profit at 3R target

Profit Taking:
  - Scale out 50% at 2R
  - Let 50% run to 3R+ with trailing stop
```

###### **3.3 Trade Filter System**
```python
Purpose: Multi-stage filtering to prevent bad trades

Pre-Trade Filters (Hard Stops):
  ❌ If daily loss limit reached → No trades
  ❌ If max position limit reached → No new positions
  ❌ If outside trading hours → No trades
  ❌ If major news in next 30 min → No trades
  ❌ If spread > 2 pips → No trades
  ❌ If ATR > 95th percentile → No trades (too volatile)

Quality Filters:
  ✅ Signal confidence must be >0.70
  ✅ Pattern validation score must be >0.75
  ✅ Market regime must match strategy
  ✅ Multi-timeframe confluence >0.65
  ✅ Volume above average (1.2x)
  ✅ Session alignment (pattern works in this session)

Confirmation Requirements (2 of 3 needed):
  1. ICT pattern detected + validated
  2. ML signal model fires (long/short)
  3. Returns forecast supports direction
  4. Order flow confirms (if available)
  5. Multi-timeframe alignment
```

###### **3.4 Correlation & Exposure Manager**
```python
Purpose: Prevent over-concentration in correlated positions

Track Correlations:
  - EURUSD vs other pairs (GBPUSD, AUDUSD, etc.)
  - EURUSD vs DXY (inverse correlation)
  - EURUSD vs Gold (positive correlation)
  - EURUSD vs US10Y (interest rate sensitivity)

Rules:
  - Max 2 correlated positions at once
  - If EURUSD long + GBPUSD long → reduce total size by 30%
  - If DXY showing opposite signal → reduce confidence by 20%

Macro Context:
  - Monitor Fed announcements (FOMC calendar)
  - Track ECB policy divergence
  - Consider interest rate differentials
```

---

#### TIER 4: Ensemble Decision Layer

##### 🎯 **Signal Aggregation & Voting System**

###### **4.1 Weighted Ensemble Logic**
```python
Purpose: Combine all models into final decision

Model Weights (calibrated on validation data):
  Primary Signal Generation (60% weight):
    - Deep Learning Long Classifier: 25%
    - Deep Learning Short Classifier: 25%
    - Returns Forecaster (5-period): 10%
  
  Pattern Confirmation (25% weight):
    - ICT Pattern Validation Model: 15%
    - Multi-Timeframe Confluence: 10%
  
  Context & Risk (15% weight):
    - Market Regime Classifier: 8%
    - Order Flow Analyzer: 7%

Voting Logic:
  For Long Entry:
    1. DL Long Model probability >0.70 (required)
    2. Returns_5 forecast >+0.002 (required)
    3. At least 1 ICT pattern detected AND validated >0.75
    4. Market regime = Uptrend or Ranging (not Downtrend)
    5. Confluence score >0.65
    6. Optional: Order flow positive
  
  Minimum Requirements: #1 + #2 + #3 must pass
  Optimal Trade: All 6 conditions met
  
  Final Confidence Score:
    weighted_sum = Σ(model_probability * model_weight)
    
    If weighted_sum >0.80: Execute with full size
    If 0.70-0.80: Execute with 75% size
    If <0.70: Do not trade

Similar logic for Short Entry (inverted conditions)
```

###### **4.2 Real-Time Signal Generation Pipeline**
```python
Real-Time Flow (every 5 seconds on new bar):

1. Data Ingestion:
   - Receive new OHLCV bar via WebSocket
   - Update all feature calculations (indicators, strategies)
   - Update sliding windows for all models
   
2. Rule-Based Layer Execution (parallel):
   - Run FVG detector → output zones
   - Run Order Block detector → output zones
   - Run Liquidity Sweep detector → output signals
   - Run Market Structure analyzer → output BOS/CHoCH
   - Check Session context → output optimal patterns
   
3. ML Model Inference (parallel):
   - DL Long Classifier → predict probability
   - DL Short Classifier → predict probability
   - Returns_1 Forecaster → predict next return
   - Returns_5 Forecaster → predict 5-step return
   - Market Regime Model → predict current regime
   - Pattern Validation Model → score detected patterns
   - Multi-Timeframe Confluence → compute alignment
   
4. Risk Checks (sequential):
   - Check daily loss limit
   - Check position limits
   - Check session/time filters
   - Check spread/volatility conditions
   
5. Ensemble Aggregation:
   - Compute weighted ensemble score
   - Apply voting logic
   - Determine final signal: LONG / SHORT / HOLD
   
6. If Trade Signal:
   - Calculate position size
   - Determine stop loss and take profit
   - Send order to broker API
   - Log trade details
   
7. If No Signal:
   - Monitor existing positions
   - Update trailing stops
   - Check exit conditions

Total Latency Target: <100ms from bar close to signal
```

---

#### Complete Model Training Roadmap

##### Phase 1: Core Models (Already Completed ✅)
- ✅ Long Signal Binary Classifier
- ✅ Short Signal Binary Classifier
- ✅ Returns_1 Regression Model
- ✅ Returns_5 Regression Model

##### Phase 2: ICT Rule-Based Systems (Weeks 1-2)
Build These Rule-Based Algorithms:
1. ✅ Fair Value Gap Detector (use `smartmoneyconcepts` + enhancements)
2. ✅ Order Block Detector with volume validation
3. ✅ Liquidity Sweep Detector
4. ✅ Market Structure Break Analyzer
5. ✅ Session-Based Context Filter

Implementation: Pure Python, no training needed
Estimated Time: 1-2 weeks

##### Phase 3: Validation & Context Models (Weeks 3-6)
Build These ML Models:
1. ✅ **Market Regime Classifier** (LSTM)
   - Training Data: 2+ years EURUSD, label with ADX/ATR
   - Training Time: 2-3 days
   
2. ✅ **Pattern Validation Model** (CNN-LSTM)
   - Training Data: Run rule-based detectors, label outcomes
   - Need 2000+ examples per pattern
   - Training Time: 3-5 days
   
3. ✅ **Multi-Timeframe Confluence Model** (Transformer)
   - Training Data: Multi-timeframe data with directional labels
   - Training Time: 3-4 days

Estimated Total Time: 3-4 weeks with data collection

##### Phase 4: Advanced Systems (Weeks 7-10)
Optional But Recommended:
1. ⚠️ **Chart Pattern CNN** (if you want visual pattern detection)
   - Requires 9000+ labeled images
   - Computationally expensive
   - Use only as supplementary confirmation
   
2. ⚠️ **Order Flow Analyzer** (LSTM)
   - Requires order flow data subscription ($$$)
   - Alternative: Build volume-based proxy version
   - Training Time: 2-3 days

3. ⚠️ **Adaptive Threshold Optimizer** (Reinforcement Learning)
   - Uses RL to dynamically adjust confidence thresholds
   - Learns optimal entry timing
   - Training Time: 1-2 weeks

##### Phase 5: Integration & Backtesting (Weeks 11-14)
1. ⚠️ Build ensemble aggregation system
2. ⚠️ Implement real-time data pipeline
3. ⚠️ Create risk management layer
4. ⚠️ Backtest complete system on 2+ years data
5. ⚠️ Paper trade for 2-4 weeks
6. ⚠️ Go live with small position sizes

---

#### Feature Engineering Pipeline (Enhanced)

##### Your Current Features (Keep These) ✅
From your analysis, top 15 features:
1. Volume ⭐ (most important)
2. WILLR
3. STOCHRSI_K
4. CCI
5. ADOSC
6. RSI
7. CMO
8. PLUS_DI
9. MINUS_DI
10. ADX
11. HT_DCPHASE
12. STOCHRSI_D
13. BOP
14. MACD
15. rolling_std

##### Additional Features to Add ⚠️

###### ICT-Specific Features
```python
# Add these for ICT model enhancement:
- fvg_distance (distance to nearest FVG)
- ob_distance (distance to nearest Order Block)
- liquidity_level_distance (distance to swept high/low)
- session_label (Asian/London/NY encoded)
- time_in_session (normalized 0-1)
- structure_break_count (recent BOS/CHoCH count)
```

###### Multi-Timeframe Features
```python
# For confluence model:
- ema_50_1h (1-hour EMA 50)
- rsi_4h (4-hour RSI)
- adx_15m (15-minute ADX)
- volume_ratio_5m_to_1m
- trend_alignment_score (1=all TFs bullish, -1=all bearish)
```

###### Order Flow Proxies (if no direct data)
```python
- volume_delta (up_volume - down_volume approximation)
- aggressive_buy_ratio (volume on upticks / total)
- volume_at_high_low_ratio
- volume_imbalance_score
```

---

#### Technology Stack Summary

##### Data & APIs
```python
# Market Data
- Interactive Brokers API (ib_insync) - Real-time EURUSD
- yfinance - Historical data backup
- Optional: Order flow data providers (BookMap, Sierra Chart)

# Economic Data (for institutional strategies)
- FRED API - Interest rates, CPI, GDP
- DXY (US Dollar Index) data
- Correlating asset data (Gold, US10Y, Crude Oil)
```

##### Model Training
```python
# Deep Learning
- TensorFlow 2.x with Keras
- Keras Tuner (hyperparameter optimization)
- Mixed precision training (float16)

# Traditional ML
- Scikit-learn
- XGBoost, LightGBM, CatBoost

# Computer Vision (if building CNN)
- OpenCV (chart image preprocessing)
- PIL/Pillow (image manipulation)
```

##### Infrastructure
```python
# Real-Time Processing
- asyncio (async data handling)
- websocket-client (real-time feeds)

# Data Storage
- PostgreSQL (trade history, model metrics)
- Redis (real-time caching)

# Backtesting
- Backtrader or VectorBT

# Monitoring
- MLflow (experiment tracking)
- Custom dashboard (FastAPI + React)
```

---

#### Final System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE MARKET DATA                          │
│              (EURUSD 1m OHLCV + Indicators)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 1: MARKET INTELLIGENCE LAYER               │
├─────────────────────────────────────────────────────────────┤
│  Rule-Based ICT Detection:                                   │
│  • Fair Value Gap Detector                                   │
│  • Order Block Detector                                      │
│  • Liquidity Sweep Detector                                  │
│  • Market Structure Analyzer                                 │
│  • Session Context Filter                                    │
│                                                              │
│  Pattern Recognition:                                        │
│  • CNN Chart Pattern Detector (optional)                     │
│  • Candlestick Patterns (TA-Lib)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 2: SIGNAL GENERATION LAYER                │
├─────────────────────────────────────────────────────────────┤
│  Core ML Models (Implemented ✅):                           │
│  • Long Signal Classifier (BiLSTM/Transformer)              │
│  • Short Signal Classifier (BiLSTM/Transformer)             │
│  • Returns_1 Forecaster (LSTM/GRU)                          │
│  • Returns_5 Forecaster (Conv1D-LSTM/TCN)                   │
│                                                             │
│  Specialized Models (Implemented ✅):                       │
│  • Market Regime Classifier (LSTM)                          │
│  • Pattern Validation Model (CNN-LSTM)                      │
│  • Multi-Timeframe Confluence (Transformer)                 │
│  • Order Flow Analyzer (LSTM) - Optional (Need to Build ⚠️) │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               TIER 4: ENSEMBLE DECISION LAYER                │
├─────────────────────────────────────────────────────────────┤
│  Weighted Voting System:                                     │
│  • Primary Signals: 60% (DL Long/Short + Returns)           │
│  • Pattern Confirmation: 25% (Validation + Confluence)       │
│  • Context & Risk: 15% (Regime + Order Flow)                │
│                                                              │
│  Output: LONG / SHORT / HOLD + Confidence Score              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│             TIER 3: RISK & EXECUTION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  Risk Management:                                            │
│  • Position Sizing Calculator                                │
│  • Dynamic Stop Loss & Take Profit                           │
│  • Trade Filter System (multi-stage)                         │
│  • Correlation & Exposure Manager                            │
│  • Daily Loss Limits                                         │
│                                                              │
│  Execution:                                                  │
│  • Order placement via IB API                                │
│  • Position monitoring                                       │
│  • Trailing stops                                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  BROKER (IB)   │
                  │  Live Trading  │
                  └────────────────┘
```

---

#### Immediate Next Steps (Prioritized)

##### Week 1-2: ICT Rule-Based Systems ✅
1. Install `smartmoneyconcepts` library
2. Build enhanced FVG detector with volume confirmation
3. Build Order Block detector with session filtering
4. Build Liquidity Sweep detector
5. Build Session Context Filter
6. Test all rule-based systems on historical data

##### Week 3-4: Market Regime Model ✅
1. Collect 2+ years of multi-timeframe EURUSD data
2. Label data with regime classifications (use ADX + ATR)
3. Build LSTM/Transformer classifier
4. Train with time-series CV
5. Validate on out-of-sample data
6. Integrate into pipeline

##### Week 5-6: Pattern Validation System ✅
1. Run all ICT detectors on historical data (2+ years)
2. Label pattern outcomes (profitable vs not)
3. Build CNN-LSTM validation model
4. Train with augmented data
5. Achieve >80% accuracy on pattern quality prediction
6. Integrate as filter layer

##### Week 7-8: Multi-Timeframe Confluence ✅
1. Create multi-timeframe dataset (1m/5m/15m/1h/4h)
2. Label directional alignment
3. Build Transformer or Multi-Stream model
4. Train and validate
5. Integrate as confirmation layer

##### Week 9-10: Ensemble Integration
1. Build weighted voting system
2. Implement ensemble aggregation logic
3. Create real-time inference pipeline
4. Optimize for <100ms latency

##### Week 11-12: Backtesting
1. Backtest complete system on 2+ years
2. Optimize ensemble weights
3. Tune confidence thresholds
4. Validate on holdout data (2024-2025)

##### Week 13-14: Paper Trading
1. Deploy to paper trading account
2. Monitor for 2-4 weeks
3. Track all metrics daily
4. Identify and fix issues

##### Week 15+: Live Trading
1. Start with minimum position size (1 contract)
2. Gradually scale up as confidence builds
3. Implement continuous monitoring
4. Set up model retraining pipeline

---

#### Success Metrics to Track

##### Model Performance
- Classification Models: >70% precision, >60% recall
- Regression Models: Directional accuracy >65%
- Pattern Validation: >80% accuracy
- Ensemble Confidence Calibration: Confidence ≈ Actual success rate

##### Trading Performance
- Win Rate: Target >50% (achievable with 1:2+ R:R)
- Profit Factor: >1.5
- Sharpe Ratio: >1.0
- Max Drawdown: <10%
- Recovery Factor: >3.0

##### System Performance
- Signal Generation Latency: <100ms
- False Signal Rate: <30%
- Uptime: >99.5%
- Order Execution Success: >99%

---

#### Key Insights from Your Research

1. **Multi-Model Approach**: Separate buy/sell classifiers + regression is optimal
2. **Feature Engineering**: Your top 15 features are well-identified
3. **Progressive Training**: Warmup → Fine-tuning strategy is excellent
4. **Focal Loss**: Essential for handling class imbalance in signals
5. **Time-Series CV**: Prevents data leakage, maintains temporal integrity
6. **Lag Optimization**: Different lag periods for long vs short is insightful
1. **ICT Smart Money concepts**: Powerful patterns, need rule-based detection
2. **Market Regime Classification**: Critical for knowing which strategies work now
3. **Pattern Validation**: Don't blindly trust detected patterns - validate quality
4. **Multi-Timeframe Analysis**: Essential for avoiding counter-trend trades
5. **Risk Management Layer**: All the ML in the world won't help without proper risk controls

##### Philosophical Principles
1. **Hybrid > Pure ML**: Combine rule-based (for well-defined patterns) + ML (for complex patterns)
2. **Ensemble > Single Model**: No model is perfect - aggregate multiple perspectives
3. **Context Matters**: Same pattern behaves differently in different market regimes/sessions
4. **Validation is Key**: Detect patterns cheaply (rules), validate quality expensively (ML)
5. **Risk First**: Perfect signals mean nothing if position sizing and stops are wrong

---

### EURUSD Automated Trading System - Claude Project Brief

#### 🎯 Project Goal

Build a production-ready, hybrid automated trading system for EURUSD that combines:
- **Rule-based algorithms** for ICT (Inner Circle Trading) Smart Money concepts
- **Deep learning models** for signal generation and pattern validation
- **Ensemble decision-making** for robust trade execution
- **Institutional-grade risk management** for capital preservation

**Target Performance:** 15%+ annual returns, <10% max drawdown, >50% win rate with 1:2+ risk-reward ratio.

---

#### 📋 Current Status

##### ✅ **Already Completed**
1. **Core ML Models (Trained & Production-Ready)**
   - Long Signal Binary Classifier (BiLSTM/Transformer ensemble)
   - Short Signal Binary Classifier (BiLSTM/Transformer ensemble)
   - Returns_1 Regression Model (1-period return forecasting)
   - Returns_5 Regression Model (5-period return forecasting)
   - Feature importance analysis (33 optimal features identified)
   - Progressive training pipeline with focal loss
   - Time-series cross-validation framework

2. **Data Infrastructure**
   - 2.5M+ samples of EURUSD 1-minute OHLCV data (2010-2025)
   - 66 technical indicators from TA-Lib
   - 92 strategy signal features
   - Data preprocessing pipeline
   - Feature engineering with optimized lag periods

3. **Performance Metrics Achieved**
   - Classification accuracy: >99.5%
   - Signal profit rate: 93.2%
   - Regression directional accuracy: High confidence
   - Model CV stability: Excellent (std <0.002)

4. **ICT Rule-Based Detection Layer** (Weeks 1-2)
5. **Validation & Context ML Models** (Weeks 3-6)

##### ⚠️ **Needs to Be Built**
1. **Ensemble Aggregation System** (Weeks 7-8)
2. **Risk Management & Execution Layer** (Weeks 9-10)
3. **Backtesting & Paper Trading** (Weeks 11-14)

---

#### 🏗️ System Architecture (4-Tier Approach)

```
LIVE DATA → TIER 1 (ICT Detection) → TIER 2 (ML Signals) → TIER 4 (Ensemble) → TIER 3 (Risk/Execute) → BROKER
```

##### **TIER 1: Market Intelligence Layer** (Rule-Based)
**Purpose:** Fast, algorithmic detection of high-probability patterns

1. **Fair Value Gap (FVG) Detector**
   - Use `smartmoneyconcepts` library + custom enhancements
   - Detect 3-candle inefficiency patterns
   - Add volume confirmation (>1.5x average)
   - Session filtering (London Open, NY Open optimal)

2. **Order Block (OB) Detector**
   - Identify institutional accumulation/distribution zones
   - Require 2x+ volume confirmation
   - Last bearish candle before bullish move (and vice versa)
   - Session-specific validation

3. **Liquidity Sweep Detector**
   - Detect stop hunts above/below recent highs/lows
   - 50-period lookback for clustered levels
   - Distinguish temporary sweeps (reversal) vs runs (continuation)

4. **Market Structure Break Analyzer**
   - BOS (Break of Structure) = trend continuation
   - CHoCH (Change of Character) = trend reversal
   - Multi-timeframe confirmation

5. **Session Context Filter**
   - Asian (8PM-2AM EST): Range setups
   - London Open (2AM-5AM EST): Liquidity sweeps
   - NY Open (9:30AM-11AM EST): FVG fills, reversals
   - Filter patterns by session effectiveness

**Implementation:** Pure Python, no ML training needed. Use existing libraries where possible.

---

##### **TIER 2: Signal Generation Layer** (Deep Learning)

- Long Signal Classifier → outputs probability [0-1]
- Short Signal Classifier → outputs probability [0-1]
- Returns_1 Forecaster → predicts next bar return
- Returns_5 Forecaster → predicts 5-bar ahead return

1. **Market Regime Classifier** (Priority: HIGH)
   - **Architecture:** Stacked LSTM (2-3 layers) or Transformer
   - **Input:** Multi-timeframe OHLCV (1m, 5m, 15m, 1h, 4h, daily)
   - **Output:** 7 classes (Strong Up, Weak Up, Ranging, Weak Down, Strong Down, High Vol, Low Vol)
   - **Training Data:** 2+ years EURUSD, label with:
     - ADX >25 = trending, <20 = ranging
     - ATR percentiles for volatility
     - Price action slope across timeframes
   - **Purpose:** Different ICT patterns work better in different regimes

2. **Pattern Validation Model** (Priority: HIGH)
   - **Architecture:** CNN-LSTM Hybrid
     - CNN: Extract spatial features from chart
     - LSTM: Capture temporal context
   - **Input:** 100-candle window + pattern metadata (FVG/OB/Sweep)
   - **Output:** Pattern quality score [0-1]
   - **Training Data:** 
     - Run rule-based detectors on 2+ years data
     - Label: Did pattern lead to profitable trade? (yes/no)
     - Measure: Max favorable excursion, max adverse excursion
     - Need 2000+ examples per pattern type
   - **Purpose:** Filter low-quality patterns before trading

3. **Multi-Timeframe Confluence Model** (Priority: MEDIUM)
   - **Architecture:** Multi-Stream Transformer
   - **Input:** Parallel streams of 1m/5m/15m/1h/4h data
   - **Output:** Confluence score [0-1] + directional bias
   - **Features per timeframe:**
     - Trend direction (EMA crossovers)
     - Momentum (RSI, STOCHRSI)
     - Volume confirmation
     - Distance to key levels
   - **Purpose:** Only trade when multiple timeframes align

4. **Order Flow Analyzer** (Priority: LOW - Optional)
   - **Architecture:** Bidirectional LSTM
   - **Input:** Order flow data OR volume-based proxies
     - Cumulative Volume Delta
     - Bid/Ask imbalances
     - Volume spikes and divergences
   - **Output:** Pressure score [-1 to +1]
   - **Purpose:** Confirm institutional buying/selling pressure

---

##### **TIER 3: Risk & Execution Layer** (Rule-Based)

**Risk Management Algorithms:**

1. **Dynamic Position Sizing**
```python
base_risk = account_balance × 0.01  # 1% per trade
confidence_multiplier = signal_confidence  # 0.70-0.95

# Adjustments
if daily_pnl < -max_daily_loss × 0.5:
    confidence_multiplier × 0.5  # Reduce on drawdown
    
if recent_win_rate < 0.40 (last 20 trades):
    confidence_multiplier × 0.5  # Reduce on losing streak

risk_amount = base_risk × confidence_multiplier
contracts = risk_amount / (stop_distance × contract_multiplier)
return max(1, min(contracts, 3))  # 1-3 contracts max
```

2. **Dynamic Stop Loss & Take Profit**
```python
# FVG Trades
stop = beyond FVG zone + 1 ATR
target = opposite FVG or next OB
minimum R:R = 1:2

# Order Block Trades
stop = beyond OB zone + 0.5 ATR
target = previous swing high/low
minimum R:R = 1:3

# Trailing Logic
if profit = 1R: move stop to break-even
if profit = 2R: trail by 0.5 ATR
at 2R: scale out 50%
at 3R+: let runner trail
```

3. **Multi-Stage Trade Filters**
```python
# Hard Stops (No Trade)
❌ Daily loss limit reached ($500)
❌ Max position limit reached (2 contracts)
❌ Outside trading hours (2AM-4PM EST)
❌ Major news in next 30 min
❌ Spread >2 pips
❌ ATR >95th percentile (extreme volatility)

# Quality Gates (Must Pass)
✅ Signal confidence >0.70
✅ Pattern validation >0.75
✅ Volume >1.2× average
✅ Minimum 2 of 4 confirmations
```

4. **Correlation & Exposure Management**
```python
# Track correlated positions
- Max 2 correlated pairs at once
- If EURUSD + GBPUSD long: reduce total size 30%
- If DXY opposite signal: reduce confidence 20%

# Monitor macro context
- Fed/ECB announcements (no trade 30min before/after)
- Interest rate differential changes
- Correlating assets (Gold, US10Y, DXY)
```

---

##### **TIER 4: Ensemble Decision Layer** (Aggregation Logic)

**Weighted Voting System:**

```python
# Model Contributions to Final Decision
Primary Signals (60% weight):
  - DL Long Classifier: 25%
  - DL Short Classifier: 25%
  - Returns_5 Forecaster: 10%

Pattern Confirmation (25% weight):
  - Pattern Validation Model: 15%
  - Multi-Timeframe Confluence: 10%

Context & Risk (15% weight):
  - Market Regime Classifier: 8%
  - Order Flow Analyzer: 7%

# Entry Requirements for LONG Trade
REQUIRED (must pass):
  1. DL Long probability >0.70
  2. Returns_5 forecast >+0.002
  3. At least 1 validated ICT pattern (score >0.75)

CONFIRMATION (2 of 4):
  4. Market regime = Uptrend or Ranging
  5. Multi-timeframe confluence >0.65
  6. Order flow positive (if available)
  7. Session alignment (pattern effective in current session)

# Final Decision
weighted_confidence = Σ(model_prob × model_weight)

If weighted_confidence >0.80: Execute FULL position size
If 0.70-0.80: Execute 75% position size
If <0.70: DO NOT TRADE
```

---

#### 🚀 Implementation

##### **Ensemble System**
**Goal:** Combine all models into unified decision engine

**Tasks:**
```python
1. Build Weighted Aggregation
   class EnsembleDecisionEngine:
       def __init__(self):
           # Load all models
           self.long_classifier = load_model('long_signal.keras')
           self.short_classifier = load_model('short_signal.keras')
           self.returns_5 = load_model('returns_5.keras')
           self.regime_classifier = load_model('regime.keras')
           self.pattern_validator = load_model('pattern_validator.keras')
           self.confluence_model = load_model('confluence.keras')
           
           # ICT detectors
           self.fvg_detector = FVGDetector()
           self.ob_detector = OrderBlockDetector()
           self.sweep_detector = LiquiditySweepDetector()
           
       def generate_signal(self, live_data):
           # 1. Rule-based detection
           patterns = self.detect_patterns(live_data)
           
           # 2. ML predictions
           long_prob = self.long_classifier.predict(live_data)
           short_prob = self.short_classifier.predict(live_data)
           returns = self.returns_5.predict(live_data)
           regime = self.regime_classifier.predict(live_data)
           
           # 3. Validate patterns
           validated = [p for p in patterns 
                       if self.pattern_validator.score(p) > 0.75]
           
           # 4. Check confluence
           confluence = self.confluence_model.predict(live_data)
           
           # 5. Weighted decision
           signal = self.aggregate(
               long_prob, short_prob, returns, 
               regime, validated, confluence
           )
           
           return signal

2. Implement Voting Logic
   - Define entry requirements (REQUIRED + CONFIRMATIONS)
   - Calculate weighted confidence score
   - Determine position size based on confidence
   - Output: LONG / SHORT / HOLD + size + stops

3. Build Risk Layer
   - Position sizing calculator
   - Stop loss / take profit logic
   - Trade filter system (multi-stage)
   - Correlation manager

4. Real-Time Pipeline
   - Async data ingestion (WebSocket)
   - Parallel model inference
   - <100ms latency target
   - Queue-based execution
```

**Deliverables:**
- Complete ensemble decision engine
- Risk management module
- Real-time inference pipeline
- Latency benchmarks (<100ms)

---

##### **Backtesting** 
**Goal:** Validate system on historical data

**Tasks:**
```python
1. Build Backtesting Framework
   - Use Backtrader or VectorBT
   - Implement custom strategy class
   - Simulate realistic execution:
     * Slippage (0.5-1 pip)
     * Commission ($2-5 per round turn)
     * Spread (1.5 pips average)
   
2. Run Comprehensive Backtest
   - Period: 2+ years (2022-2024)
   - Walk-forward optimization:
     * Train on 12 months
     * Test on next 3 months
     * Roll forward
   
3. Performance Analysis
   Metrics:
     - Total return
     - Sharpe ratio (target >1.0)
     - Max drawdown (target <10%)
     - Win rate (target >50%)
     - Profit factor (target >1.5)
     - Average R:R (target >1.5)
     - Recovery factor (profit / max DD)
   
   Per-Pattern Analysis:
     - Which ICT patterns perform best?
     - Which sessions are most profitable?
     - Which regime is optimal?
   
4. Optimization
   - Tune ensemble weights via grid search
   - Optimize confidence thresholds
   - Refine risk parameters
   - Test parameter sensitivity

5. Holdout Validation
   - Final test on 2024-2025 data (untouched)
   - Must achieve target metrics
   - If not: iterate on weak components
```

**Deliverables:**
- Backtesting codebase
- Performance report (charts, metrics, analysis)
- Optimized parameters
- Holdout validation results

---



#### 🛠️ Technical Stack

##### Core Libraries
```bash
# Data & APIs
pip install pandas numpy yfinance ib_insync smartmoneyconcepts

# Technical Analysis
pip install ta-lib  # or conda install -c conda-forge ta-lib

# Machine Learning
pip install tensorflow scikit-learn keras-tuner xgboost lightgbm catboost

# Backtesting
pip install backtrader vectorbt

# Real-Time
pip install asyncio websocket-client

# Visualization
pip install matplotlib seaborn plotly

# Infrastructure
pip install fastapi redis postgresql psycopg2-binary

# Experiment Tracking
pip install mlflow
```

##### Infrastructure Requirements
```
Development:
- Local machine or cloud VM
- 16GB+ RAM (for model training)
- GPU optional (speeds up training 5-10x)

Production:
- VPS with <50ms to exchange (AWS/GCP in same region as IB servers)
- 8+ core CPU
- 16GB RAM
- SSD storage
- Uptime: 99.9%

Broker:
- Interactive Brokers (paper + live account)
- TWS or IB Gateway running
- Paper trading: Free
- Live trading: $1-10/month data fees
```

---

#### 📊 Key Performance Indicators (KPIs)

##### Model Performance
| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Long Classifier Precision | >70% | >60% |
| Short Classifier Precision | >70% | >60% |
| Returns_5 Directional Acc | >65% | >55% |
| Pattern Validation Acc | >80% | >70% |
| Regime Classifier Acc | >75% | >65% |
| Ensemble Calibration | ±5% | ±10% |

##### Trading Performance
| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Annual Return | >15% | >8% |
| Sharpe Ratio | >1.0 | >0.7 |
| Max Drawdown | <10% | <15% |
| Win Rate | >50% | >45% |
| Profit Factor | >1.5 | >1.2 |
| Average R:R | >1.5:1 | >1.2:1 |

##### System Performance
| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Signal Latency | <100ms | <200ms |
| False Signal Rate | <30% | <40% |
| System Uptime | >99.5% | >99% |
| Order Fill Success | >99% | >95% |

---

#### 🎓 Guiding Principles

##### Design Philosophy
1. **Hybrid > Pure ML**: Combine rule-based (fast, interpretable) + ML (complex patterns)
2. **Ensemble > Single Model**: Aggregate multiple perspectives for robustness
3. **Context is King**: Same pattern behaves differently in different regimes/sessions
4. **Validate Everything**: Detect patterns cheaply (rules), validate quality expensively (ML)
5. **Risk First**: Perfect signals mean nothing without proper position sizing and stops

##### Development Approach
1. **Build Incrementally**: Test each component thoroughly before integration
2. **Validate Continuously**: Every model/algorithm must prove value on historical data
3. **Monitor Religiously**: Track everything in production (signals, trades, metrics)
4. **Adapt Quickly**: Market conditions change - be ready to adjust
5. **Never Stop Learning**: Implement model retraining pipeline for continuous improvement

##### Risk Management Mindset
1. **Capital Preservation**: First rule - don't lose money
2. **Position Sizing**: Most important factor in long-term success
3. **Cut Losses Fast**: No emotional attachment to losing trades
4. **Let Winners Run**: Trail stops, scale out, maximize good trades
5. **Respect Daily Limits**: Live to trade another day

---

#### 📁 Project Structure

```
eurusd-trading-system/
├── data/
│   ├── raw/                    # Raw EURUSD OHLCV data
│   ├── processed/              # Preprocessed with features
│   └── splits/                 # Train/val/test splits
│
├── models/
│   ├── trained/                # Saved model files (.keras, .h5)
│   │   ├── long_signal_classifier.keras
│   │   ├── short_signal_classifier.keras
│   │   ├── returns_5_regressor.keras
│   │   ├── regime_classifier.keras
│   │   └── pattern_validator.keras
│   └── architectures/          # Model builder functions
│
├── src/
│   ├── ict_detection/    # Rule-based ICT detectors
│   │   ├── fvg_detector.py
│   │   ├── order_block_detector.py
│   │   ├── liquidity_sweep_detector.py
│   │   └── session_filter.py
│   │
│   ├── ml_models/        # ML model training/inference
│   │   ├── regime_classifier.py
│   │   ├── pattern_validator.py
│   │   └── confluence_model.py
│   │
│   ├── risk/             # Risk management
│   │   ├── position_sizing.py
│   │   ├── stop_loss_logic.py
│   │   └── trade_filters.py
│   │
│   ├── ensemble/         # Ensemble decision engine
│   │   └── decision_engine.py
│   │
│   ├── data/                   # Data pipeline
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   │
│   ├── backtest/              # Backtesting framework
│   │   └── backtest_engine.py
│   │
│   └── execution/             # Live trading
│       ├── broker_interface.py
│       └── order_manager.py
│
├── tests/                     # Unit tests
│   ├── test_ict_detectors.py
│   ├── test_ml_models.py
│   └── test_risk_management.py
│
├── configs/                   # Configuration files
│   ├── model_config.yaml
│   ├── trading_config.yaml
│   └── risk_config.yaml
│
├── logs/                      # System logs
│   ├── training/
│   ├── backtesting/
│   └── live_trading/
│
├── requirements.txt
├── README.md
└── main.py                    # Entry point for live system
```

---

#### 🎯 Success Criteria

##### Phase Completion Criteria

**Phase 1 Complete When:**
- [x] All 5 ICT detectors functional
- [ ] Tested on 6+ months historical data
- [ ] False positive rate <40%
- [ ] Can process 1000 bars in <1 second

**Phase 2-4 Complete When:**
- [ ] Each ML model achieves target accuracy
- [ ] Models validated on holdout data (2024-2025)
- [ ] Inference time <50ms per model
- [ ] Models integrated into pipeline

**Phase 5 Complete When:**
- [ ] Ensemble generates coherent signals
- [ ] Backtests show positive returns
- [ ] Risk management prevents excessive losses
- [ ] System latency <100ms end-to-end

**Phase 6 Complete When:**
- [ ] 2+ year backtest with positive Sharpe >1.0
- [ ] Holdout validation confirms performance
- [ ] Parameter sensitivity acceptable
- [ ] All target metrics achieved

**Phase 7 Complete When:**
- [ ] 2+ weeks paper trading successful
- [ ] Live metrics match backtest (±20%)
- [ ] No critical bugs or execution failures
- [ ] Ready for live capital deployment

##### Final Go-Live Checklist
```
[ ] All 4 tiers functional and tested
[ ] Backtest: >15% annual return, <10% max DD
[ ] Paper trading: 2+ weeks successful
[ ] Risk management: hard limits working
[ ] Monitoring: dashboard operational
[ ] Alerts: critical failure notifications
[ ] Capital: start with $10k-25k
[ ] Position size: 1 contract initially
[ ] Daily limit: max $500 loss
[ ] Review: daily trade log analysis
[ ] Contingency: emergency stop plan
```

---

#### 🚨 Critical Reminders

1. **Start Small**: Begin with minimum position sizes in live trading
2. **Respect Limits**: Never override daily loss limits - psychology > strategy
3. **Monitor Daily**: Review every trade, every signal, every metric
4. **Stay Disciplined**: Trust the system, but be ready to pause if issues arise
5. **Iterate Continuously**: Market conditions change - retrain models quarterly
6. **Never Risk What You Can't Afford to Lose**: This is speculative trading
7. **Backtest ≠ Future**: Past performance doesn't guarantee future results
8. **Technology Fails**: Have contingency plans for system outages
9. **Stay Educated**: Keep learning about ICT, ML, and market structure
10. **Enjoy the Process**: Building sophisticated systems is as rewarding as profits

---

<a id="ote_training_workflow_report"></a>

## Optimal Trade Entry Model Training Workflow

*Merged 2026-10-09 from `model_training/ote_training/OTE_TRAINING_WORKFLOW_REPORT.md`.*

### An End-to-End Methodology for Causal EUR/USD Swing-Entry Detection

#### Abstract

This document describes the implemented end-to-end workflow used to train Optimal Trade Entry (OTE) models in this repository. The system is designed to detect high-quality long and short swing-entry opportunities on EUR/USD 5-minute data under severe class imbalance, strong temporal dependence, and high risk of leakage. The pipeline begins with raw OHLCV ingestion, proceeds through causal swing labeling, feature generation, target-aware preprocessing, and backend-specific model training, and terminates in calibrated probability models with event-level operating thresholds. The methodology emphasizes causal construction, volatility normalization, higher-timeframe structural context, and training procedures aligned to rare-event detection rather than generic bar classification. Because development has primarily used sampled datasets, this report intentionally documents methods, assumptions, and artifacts without presenting empirical performance results.

### 1. Research Goal and System Scope

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

#### Implemented scope versus broader research scope

The broader research notes in this repository reference 1-minute, 5-minute, 30-minute, and 1-hour data. The implemented OTE training path described here currently requires only raw 5-minute OHLCV as its base input. The code resamples that 5-minute series into 30-minute and 1-hour structure internally for labeling and feature generation. The 1-minute layer belongs to the wider research roadmap but is not required by the current training code path.

### 2. Implemented System Components

The workflow is implemented across four stages:

| Stage | Primary code | Purpose |
| --- | --- | --- |
| Raw data loading and labeling | `data/labeling/labeling_engine.py`, `data/labeling/ote_labeling_engine.py` | Convert raw EUR/USD 5-minute OHLCV into zone labels, entry labels, exclusion masks, confluence fields, and sample weights |
| Feature generation | `features/builder.py`, `features/cli.py`, `features/feature_sets/`, `features/transforms.py` | Build a modeling table from labeled bars using modular feature families plus causal transforms |
| Target-aware preprocessing | `features/preprocessing.py` | Create clean, chronological, target-specific train/validation/test datasets for `long_ote`, `short_ote`, `long_entry`, and `short_entry` |
| Model training | `model_training/ote_training/ote_xgboost_pipeline.py`, `model_training/ote_training/torch_trainer.py`, `model_training/ote_training/torch_models.py` | Train XGBoost or PyTorch OTE models with purged walk-forward validation, calibration, thresholding, and artifact export |

### 3. Raw Data Foundation and Integrity Controls

#### 3.1 Input contract

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

#### 3.2 Timestamp normalization and deduplication

The raw file is converted into a timestamp-indexed DataFrame, sorted chronologically, and deduplicated by timestamp. This creates the strict temporal ordering required by both the labeling engine and later chronological train/validation/test splits.

#### 3.3 Anomaly detection before labeling

Before any label construction occurs, the loader removes bars flagged by deterministic anomaly rules:

| Rule | Intent |
| --- | --- |
| Invalid OHLC relationships | Reject malformed bars where the high-low-open-close geometry is impossible |
| Excessive range relative to ATR | Remove bars whose range exceeds `10 x` a rolling ATR baseline |
| Non-positive volume | Exclude bars with zero or negative volume |

This is important because the OTE pipeline uses local windows extensively. A single broken bar can contaminate multiple subsequent windows, not just one sample.

#### 3.4 Multi-timeframe construction

The current code derives higher-timeframe structure directly from the cleaned 5-minute series:

- 30-minute OHLCV is built by resampling with first-open, max-high, min-low, last-close, and summed volume
- 1-hour OHLCV is produced using the same aggregation rules

These higher-timeframe frames are used later for structural ATR estimation, swing confluence, and higher-timeframe context features.

### 4. Label Construction Methodology

The implemented labeling system is not a naive pivot-labeling routine. It is a multi-stage causal labeling engine that detects swing structure, validates it with forward outcome logic, scores label quality, and creates both positive zones and safe negatives for downstream training.

#### 4.1 Labeling objectives

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

#### 4.2 Structural ATR and volatility normalization

The core labeling thresholds are scaled by structural ATR rather than raw price units. The current defaults compute:

- a standard 5-minute ATR for local context
- a structural ATR on 1-hour bars
- a causal forward-fill mapping of the 1-hour ATR back onto the 5-minute index

This design makes swing detection robust across changing volatility regimes. A reversal threshold that is meaningful in low-volatility conditions remains meaningful in high-volatility conditions because it is expressed in ATR units.

#### 4.3 Causal swing detection

The main swing detector is a zigzag-style causal algorithm with four gatekeepers:

| Gatekeeper | Default behavior |
| --- | --- |
| Confirmation retrace | Swing confirmation requires a retrace of `0.8 x` structural ATR |
| Minimum swing size | Consecutive swings must span at least `1.35 x` structural ATR |
| Minimum price separation | Consecutive swings must differ by at least `0.75 x` structural ATR |
| Minimum temporal separation | Consecutive swings must be at least 18 bars apart |

The detector operates after a 50-bar warmup period and alternates between confirmed highs and lows. This prevents consolidation noise from producing clusters of false swing events.

#### 4.4 Higher-timeframe structural confirmation

The engine also runs the same style of swing detection on:

- 30-minute bars
- 1-hour bars

Lower-timeframe swings are then annotated with:

- whether a same-direction higher-timeframe swing exists within a confluence window
- the number of bars since the previous 30-minute swing of the same type
- the number of bars since the previous 1-hour swing of the same type

In addition to per-swing matching, all 5-minute bars within a configurable window around higher-timeframe swings are marked as higher-timeframe confluence regions. This produces both event-level validation and bar-level context.

#### 4.5 Triple-barrier outcome validation

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

#### 4.6 Trend-scan cross-validation

Each swing is further evaluated by forward trend scanning. The engine computes a linear-trend t-statistic over multiple forward windows:

- 12 bars
- 24 bars
- 48 bars
- 96 bars

For a long swing, stronger positive trend t-statistics imply better follow-through. For a short swing, the sign is reversed accordingly. The best signed t-statistic is retained, and by default a minimum absolute t-statistic of 2.0 is required for the swing to pass the trend-scan filter.

#### 4.7 Swing quality scoring

Validated swings receive a continuous quality score in `[0, 1]` and a quality tier (`A`, `B`, `C`, or `reject`). The implemented scoring rule is:

| Component | Weight |
| --- | --- |
| Triple-barrier outcome score | 35% |
| Trend-scan score | 30% |
| Higher-timeframe agreement score | 20% |
| Entry-score contribution | 15% |

By default, swings below a minimum quality threshold of `0.35` are rejected from positive labeling.

#### 4.8 Zone labels

OTE zone labels are generated around validated swing extremes. The current defaults are:

| Parameter | Default |
| --- | --- |
| Pre-bars before swing extreme | 2 |
| Post-bars after swing extreme | 0 |

The result is intentionally zone-based rather than point-based. This acknowledges that a bar slightly before the exact swing extreme can still be an excellent trade entry.

#### 4.9 Precise entry labels

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

#### 4.10 Exclusion masks and safe negatives

The labeling engine explicitly separates ambiguous negatives from clean negatives.

For each direction it creates:

- `exclude_long` / `exclude_short`
- `neg_ok_long` / `neg_ok_short`

Exclusion masks are built around confirmation points and positive label zones. Only bars outside warmup and outside ambiguity zones are allowed to act as negative samples during preprocessing.

#### 4.11 Concurrency-aware sample weights

The engine computes label concurrency and inverse-concurrency sample weights. Positive samples are weighted by:

1. inverse overlap with other active label windows
2. label quality or entry quality

This produces:

- `sample_weight_long`
- `sample_weight_short`
- `sample_weight_entry_long`
- `sample_weight_entry_short`

#### 4.12 Labeling outputs

The labeling stage writes two complementary outputs:

| Output | Purpose |
| --- | --- |
| `eurusd_5min_ote_labels.csv` | Bar-level training table with labels, exclusion masks, qualities, confluence, and weights |
| `eurusd_5min_ote_swings.csv` | Swing audit trail containing confirmation metadata, outcomes, entry choice, and label tier |

Optional review layers exist through the manual labeling and review applications, allowing human overrides without destroying reproducibility of the base automated labels.

### 5. Feature Engineering Methodology

After labeling, the workflow transforms the labeled 5-minute dataset into a modeling table using a modular feature builder. The default OTE recipes are deliberately broad: they represent price behavior, volatility, trend, structure, higher-timeframe context, ICT context, exhaustion signals, microstructure proxies, and temporal/session state.

#### 5.1 Feature builder entry conditions

The builder first standardizes column names, validates OHLC consistency, and removes invalid bars if requested. It then appends feature families in the order specified by the recipe.

The default OTE recipes preserve the original market columns and label columns while adding engineered features. Warmup rows are dropped after feature generation:

- `ote_base.json` uses 200 warmup rows
- `ote_extended.json` uses 250 warmup rows

After warmup removal, numeric columns are forward-filled and then remaining missing values are replaced with zero.

#### 5.2 Default feature families

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

#### 5.3 Higher-timeframe feature construction

The higher-timeframe feature block resamples the labeled 5-minute dataset into completed 30-minute and 1-hour bars using right-closed, right-labeled windows. Those higher-timeframe features are then forward-filled back to the 5-minute index. This ensures the 5-minute model never sees unfinished higher-timeframe candles.

The higher-timeframe block includes:

- latest higher-timeframe swing highs and lows
- distance to those swings in ATR units
- higher-timeframe EMA alignment and spread
- bars since higher-timeframe swing events
- distances to previous-day, rolling-daily, and rolling-weekly highs and lows

#### 5.4 ICT-context feature construction

The ICT feature family encodes structural concepts directly from price action:

- fair value gaps defined by displacement-style three-candle geometry
- bullish and bearish order blocks defined from candle-body dominance and break conditions
- equal-high and equal-low liquidity pools using swing-level similarity tolerances
- break of structure and change of character states
- confluence counts inside a one-ATR neighborhood

These are implemented as continuous distance and age features whenever possible, rather than purely binary flags.

#### 5.5 Transform library

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

#### 5.6 Recipe configuration

Two default OTE recipes are provided:

| Recipe | Intended use |
| --- | --- |
| `features/recipes/ote_base.json` | Baseline OTE feature build with core transforms |
| `features/recipes/ote_extended.json` | Richer build with more lag columns, wider rolling windows, and additional temporal depth |

An optional `strategy_signals` family can also be added to incorporate outputs from standalone strategy modules, but it is not part of the default OTE recipe.

### 6. Target-Aware Preprocessing and Dataset Packaging

The feature dataset produced by the builder is not trained directly. It is first converted into target-specific prepared datasets for each label type. This stage is implemented by `FeaturePreprocessingPipeline`.

#### 6.1 Feature-pool resolution

The preprocessing stage first determines which columns are eligible model features:

1. If feature-builder metadata exists, it uses the metadata-defined engineered feature list.
2. Otherwise it falls back to a heuristic selector that excludes labels, helper columns, timestamps, raw OHLCV, and other non-model fields.

This protects the pipeline against accidental inclusion of target or helper columns in the feature pool.

#### 6.2 Feature encoding

Candidate features are converted to numeric form as follows:

| Input type | Preprocessing action |
| --- | --- |
| Boolean | Cast to integer |
| Numeric | Coerce to numeric |
| Categorical/string | Label-encode after null filling and trimming |
| Constant categorical | Replace with zero |

Encoder mappings are saved for auditability.

#### 6.3 Duplicate and constant feature removal

Before target-specific pruning, the pipeline removes:

- exact duplicate columns using hash-based signatures plus equality confirmation
- globally constant columns

This reduces redundant dimensionality before later importance ranking and collinearity analysis.

#### 6.4 Target discovery and usable-row construction

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

#### 6.5 Sample weight propagation

If the labeled dataset contains direction-specific sample-weight columns, those are propagated directly into the prepared splits. Weights are clipped to non-negative values and zero weights are replaced with one for valid negatives.

#### 6.6 Chronological splitting

Prepared datasets are split chronologically:

- 70% train
- 15% validation
- 15% test

The splits preserve temporal order. Date ranges are recorded in the companion target reports even though the split CSVs intentionally exclude datetime columns.

#### 6.7 Missing-value treatment

Missing-value fill rules are estimated from the training split only:

| Distribution shape | Fill strategy |
| --- | --- |
| Strong skew (`|skew| > 1`) | Median |
| Mild skew or near-symmetric | Mean |
| Entirely missing column | Zero |

These fill values are then applied consistently to train, validation, and test.

#### 6.8 Low-variance and collinearity pruning

After filling, the preprocessing stage removes:

- low-variance columns under a configurable variance threshold
- highly collinear features using absolute correlation pruning

When two features are highly correlated, the pipeline keeps the one with the stronger association score against the target and drops the weaker one. Near-duplicate feature pairs are also recorded separately.

#### 6.9 Feature ranking

For each prepared target, the pipeline computes a composite feature importance table from:

- absolute association with the target
- mutual information
- random-forest importance

These scores are rank-aggregated into a composite score, which is later used by the trainer to cap the number of loaded base features before window expansion.

#### 6.10 Prepared output contract

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

### 7. Model Training Methodology

The OTE trainer consumes one prepared target directory at a time and supports two backend families:

- XGBoost using sparse causal lag windows
- PyTorch sequence models using dense causal windows (`TCN` or `LSTM`)

#### 7.1 Training data loading and leakage blocking

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

#### 7.2 Backend-specific temporal representations

The same prepared tabular dataset can be converted into two different temporal representations.

##### XGBoost path

The XGBoost backend builds a sparse lag view:

- choose top-ranked base features
- choose a window size
- choose a number of lag anchors inside that window
- include current and historical lag snapshots
- optionally add delta features between the current bar and selected older lags

This keeps tree-based training memory-efficient while preserving causal context.

##### PyTorch path

The PyTorch backend builds dense sliding windows of shape:

`(samples, window_size, features)`

These windows feed either:

- a causal temporal convolutional network (TCN)
- an LSTM classifier

#### 7.3 Fold-safe scaling

Within each fold, the trainer fits a `RobustScaler` on the training rows only, using configurable low and high quantiles. After transformation, values are clipped to a bounded range. This creates a stable fold-local scaling regime and avoids leaking future distributional information into earlier folds.

#### 7.4 Purged walk-forward cross-validation

Model selection uses an expanding-window, purged walk-forward splitter. Each fold contains:

- a growing historical training window
- a purge gap between training and validation
- a forward-only validation segment

The purge length is chosen as the maximum of:

- sequence-context length plus an additional purge buffer
- event tolerance plus event cooldown requirements

This is meant to reduce contamination from overlapping temporal context and nearby events.

#### 7.5 Rare-event training controls

Because OTE labels are sparse, the trainer uses multiple imbalance controls simultaneously:

| Mechanism | Effect |
| --- | --- |
| Focal loss | Down-weights easy negatives and focuses on hard decision boundaries |
| Sample weights from labeling | Preserves event uniqueness and label-quality information |
| Hard-negative upweighting | Increases weight of negative bars near positives |
| Balanced tuning subsample | Downsamples easy negatives while preserving positives and hard negatives during tuning |

Hard negatives are identified by expanding positive zones with a configurable radius and emphasizing nearby negatives rather than uniformly random negatives.

#### 7.6 XGBoost objective and progressive fitting

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

#### 7.7 PyTorch architectures and training schedule

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

#### 7.8 Hyperparameter optimization

Model selection is driven by Optuna with a TPE sampler and median pruning. The objective function is not raw validation loss. Instead, each fold contributes a weighted score:

`0.70 * average_precision + 0.30 * event_fbeta_0_5`

This keeps the search aligned with both class-ranking quality and event-level detectability.

#### 7.9 Probability calibration

After cross-validation, out-of-fold probabilities are optionally calibrated using:

- Platt scaling
- isotonic regression
- or no calibration

Calibration is fit only on valid out-of-fold predictions, which preserves separation between model fitting and probability correction.

#### 7.10 Event-level threshold selection

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

#### 7.11 Final refit and held-out test scoring

After selecting the best trial:

1. the pipeline reruns cross-validation with the best parameters
2. it fits a final model on the development set
3. it applies the learned scaler and calibrator to the held-out test split
4. it exports prediction files and metadata

The held-out test stage reports metrics internally, but those numeric results are intentionally omitted from this paper.

#### 7.12 Exported training artifacts

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

### 8. Evaluation Framework Without Empirical Results

The implemented evaluation protocol spans both bar-level probability scoring and event-level trading relevance.

#### 8.1 Probability metrics

The trainer computes:

- average precision
- ROC AUC
- Brier score

These metrics characterize ranking quality, separability, and probability calibration.

#### 8.2 Event metrics

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

### 9. Reproducibility and Artifact Flow

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

### 10. Methodological Strengths

The current implementation has several strong properties:

1. It enforces causality where causality matters most: swing detection, higher-timeframe alignment, scaling, and cross-validation.
2. It treats label ambiguity explicitly through exclusion masks and safe negatives.
3. It propagates label quality and event uniqueness into training weights instead of pretending all positives are equivalent.
4. It aligns threshold selection and cross-validation scoring to rare-event detection rather than generic classification accuracy.
5. It supports both tabular and sequence backends without changing the prepared-data contract.

### 11. Current Limitations and Practical Notes

Several practical limitations should be kept in view when extending or interpreting the pipeline:

| Limitation | Practical implication |
| --- | --- |
| Prepared split CSVs do not keep timestamps | Downstream predictions are row-index keyed unless rejoined to original metadata |
| The current code computes CUSUM events but does not use them as the final sample universe | CUSUM currently acts as a diagnostic structural-break signal rather than the actual label filter |
| The broader research notes mention 1-minute execution refinement | The current training pipeline does not require 1-minute data |
| Strategy-signal features are optional and dependency-heavy | The default OTE recipes do not depend on the strategy catalog |
| This report excludes empirical results | Use the exported reports and training summaries for future full-dataset performance analysis |

### 12. Conclusion

The implemented OTE workflow in this repository is a complete rare-event financial ML pipeline, not only a model-training script. It begins with anomaly-controlled raw EUR/USD 5-minute data, constructs causal and volatility-aware swing labels, enriches the labeled bars with multi-family structural and temporal features, packages the data into target-specific chronological splits, and trains calibrated event-detection models using purged walk-forward validation. The system is therefore best understood as an integrated methodology for causal swing-entry modeling rather than a single classifier.

For full-dataset operation, the practical run instructions are provided in the companion document:

- `model_training/ote_training/README.md#full_pipeline_readme`

<a id="end-to-end-readme"></a>

## OTE End-to-End Workflow

*Merged 2026-10-09 from `docs/end-to-end-README.md`.*

This README is the operational runbook for the EURUSD 5-minute OTE stack from raw bars to model-testing evidence.

It reflects the repo state and recent runs as of March 31, 2026.

The short version is:

1. Label the raw EURUSD 5-minute bars.
2. Build the feature dataset.
3. Preprocess into model-ready target folders.
4. Build backend-aware attribution rankings.
5. Train XGBoost, TCN, and LSTM against the correct merged ranking for each backend.
6. Run regime slices.
7. Run threshold and abstain policy search.
8. Run the walk-forward backtest with frictions.
9. Promote only the models and policies that survive the full stack.

### Canonical Timezone Contract

For the current full EURUSD training flow, treat the historical raw feed as:

- raw source timezone: `GMT-6`
- canonical storage/runtime timezone: `UTC`
- feature/session clock timezone: `America/New_York`
- FX market close boundary: `17:00 America/New_York`

The current pipeline now persists that contract from:

- raw label outputs via `.metadata.json` sidecars
- feature builds via `data/features/*.metadata.json`
- prepared datasets via `summary.json` and target `report.json`
- trained model artifacts via `training_summary.json` and `model_config.json`

### Current Workflow Decision

Before the backend-aware attribution module existed, your SHAP flow was effectively XGBoost-centric:

- raw XGBoost SHAP outputs were written under `data/prepared/eurusd_5min_ote_full/shap_outputs/`
- the merged ranking was written back as the generic `feature_importance_merged.csv`
- TCN and LSTM training did not have their own backend-specific merged rankings

That means:

- XGBoost is the closest to already being aligned with the old ranking flow.
- TCN and LSTM should be retrained with the new backend-aware ranking files.
- After retraining TCN and LSTM, you should rerun regime slicing, threshold search, abstain policy evaluation, and walk-forward backtesting.
- If you want perfect apples-to-apples parity across all three backends, rerun XGBoost too.

### Recent Artifact Roots

These are the important paths from your recent runs:

- Label output: `data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv`
- Feature output: `data/features/eurusd_5min_ote_full.csv`
- Prepared root: `data/prepared/eurusd_5min_ote_full`
- Old generic merged rankings:
  - `data/prepared/eurusd_5min_ote_full/long_ote/feature_importance_merged.csv`
  - `data/prepared/eurusd_5min_ote_full/short_ote/feature_importance_merged.csv`
- Existing trained model roots:
  - `models/ote_full_xgb_v2`
  - `models/ote_full_tcn_v3_tail`
  - `models/ote_full_lstm_v1`
- Existing regime slice run:
  - `model_testing/reports/ote_regime_slices/full_registry_20260330_200iter`
- Existing threshold-policy run:
  - `model_testing/reports/ote_threshold_policies/active_champions_cost_aware_20260330`
- Existing backtest run:
  - `model_testing/reports/ote_policy_backtests/full_run_v2`

### Directory Map

- Raw bars: `data/currency_data/eurusd-5m.csv`
- Labeled bars: `data/labeling/labeled_data/`
- Feature dataset: `data/features/`
- Prepared training folders: `data/prepared/eurusd_5min_ote_full/`
- Model artifacts: `models/`
- Registry: `models/ote_model_registry.json`
- Testing reports: `model_testing/reports/`

### Stage 1: Labeling

The labeler converts raw EURUSD 5-minute OHLCV data into OTE targets plus helper columns. The current labeler uses a fixed `warmup_bars=50` internally.

This is the exact command shape already used in your batch pipeline:

```cmd
python data/labeling/labeling_engine.py ^
  --input data/currency_data/eurusd-5m.csv ^
  --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv ^
  --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv ^
  --source-timezone GMT-6 ^
  --canonical-timezone UTC ^
  --bar-timestamp-semantics as_provided
```

Primary outputs:

- `data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv`
- `data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv`

### Stage 2: Feature Generation

This expands the labeled bars into the full engineered feature set.

This is the exact command shape from your recent full-data run:

```cmd
python -m features.cli build data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv ^
  --output data/features/eurusd_5min_ote_full.csv ^
  --recipe features/recipes/ote_extended.json ^
  --all-strategies ^
  --transform-workers 4 ^
  --optimize-memory ^
  --skip-strategy-errors ^
  --progress ^
  --strategy-timeout-seconds 600 ^
  --source-timezone UTC ^
  --canonical-timezone UTC ^
  --feature-clock-timezone America/New_York ^
  --market-close-timezone America/New_York
```

Primary outputs:

- `data/features/eurusd_5min_ote_full.csv`
- `data/features/eurusd_5min_ote_full.metadata.json`

### Stage 3: Preprocessing

This step converts the full feature CSV into chronological train/val/test target folders and computes the base feature-importance table.

This is the exact command shape from your recent full-data run:

```cmd
python -m preprocessing prepare data/features/eurusd_5min_ote_full.csv ^
  --output-dir data/prepared/eurusd_5min_ote_full ^
  --scaler none ^
  --corr-threshold 0.98 ^
  --similarity-threshold 0.995 ^
  --max-analysis-rows 10000
```

Notes:

- The prepared root also includes `long_entry` and `short_entry`, but the OTE model-training flow here is for `long_ote` and `short_ote`.
- Each OTE target folder contains `train.csv`, `val.csv`, `test.csv`, `features.json`, `feature_importance.csv`, and `report.json`.
- `summary.json` and each target `report.json` now carry the timezone contract plus source-lineage metadata.

If you want one command for stages 1 to 3, use the existing batch file:

```cmd
scripts\run_eurusd_ote_pipeline.bat
```

### Stage 4: Backend-Aware Attribution Rankings

This is the new step that now belongs between preprocessing and model training.

It keeps one common prepared dataset, but writes backend-specific ranking files for:

- XGBoost
- TCN
- LSTM

The trainer now automatically prefers these files when they exist:

- `feature_importance_merged_xgboost.csv`
- `feature_importance_merged_tcn.csv`
- `feature_importance_merged_lstm.csv`

Recommended command:

```cmd
python -m preprocessing backend-attribution ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --backend xgboost ^
  --backend tcn ^
  --backend lstm ^
  --max-features 160 ^
  --base-weight 0.20 ^
  --shap-weight 0.55 ^
  --shap-positive-weight 0.25 ^
  --attribution-floor-fraction 0.15 ^
  --attribution-cumulative-importance 0.90
```

Important implementation detail:

- XGBoost uses TreeSHAP.
- TCN and LSTM use integrated gradients internally, but the module writes SHAP-compatible ranking columns so the existing merge logic still works.
- The merged backend ranking is now filtered before training with two gates:
  - floor gate: keep only features with `mean_abs_shap_all >= 15%` of the max attribution
  - cumulative gate: from those survivors, keep only the features needed to reach `90%` cumulative attribution share

Primary outputs per target:

- `shap_feature_stats_xgboost.csv`
- `shap_feature_stats_tcn.csv`
- `shap_feature_stats_lstm.csv`
- `feature_importance_merged_xgboost.csv`
- `feature_importance_merged_tcn.csv`
- `feature_importance_merged_lstm.csv`
- `backend_attribution_summary_<backend>.json`

### Stage 5: Model Training

#### General Rules

- Use the same prepared root: `data/prepared/eurusd_5min_ote_full`
- Keep new retrains in new output roots instead of overwriting the old artifacts
- Only update the registry after the new candidates clear the testing stack
- The trainer now uses the full filtered backend ranking instead of tuning a fixed top-N feature count.

#### XGBoost

Your previous XGBoost run lived under `models/ote_full_xgb_v2`. That run is probably the least urgent to redo because the old generic merged ranking came from XGBoost-based SHAP. If you want parity with the new ranking flow, rerun it too.

Command based on your past XGBoost setup:

```cmd
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --output-root models/ote_full_xgb_v3_backend_ranked ^
  --backend xgboost ^
  --targets long_ote short_ote ^
  --trials 20 ^
  --max-loaded-features 160 ^
  --window-min 8 ^
  --window-max 40 ^
  --batch-size 256 ^
  --calibration-method platt
```

Notes:

- Your prior XGBoost model root was `models/ote_full_xgb_v2`.
- XGBoost phase scheduling is handled internally through tuned `warmup_rounds`, `main_rounds`, `fine_rounds`, and `fine_lr_scale`, not via explicit CLI epoch-phase flags.
- New model artifacts now persist the prepared-dataset timezone contract and upstream source lineage in `training_summary.json` and `model_config.json`.

#### TCN

This is the backend that most clearly needs to be retrained now that backend-aware ranking files exist.

Your previous TCN run used an explicit tail schedule under `models/ote_full_tcn_v3_tail`.

Recommended rerun command:

```cmd
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --output-root models/ote_full_tcn_v4_backend_ranked ^
  --backend torch ^
  --model-type tcn ^
  --targets long_ote short_ote ^
  --trials 20 ^
  --max-loaded-features 160 ^
  --window-min 16 ^
  --window-max 32 ^
  --epochs 40 ^
  --batch-size 256 ^
  --hidden-size 64 ^
  --num-layers 3 ^
  --dropout 0.20 ^
  --learning-rate 0.001 ^
  --torch-warmup-epochs 4 ^
  --torch-main-epochs 18 ^
  --torch-fine-epochs 10 ^
  --torch-tail-epochs 8 ^
  --torch-fine-lr-scale 0.20 ^
  --torch-tail-lr-scale 0.07 ^
  --calibration-method platt
```

That matches the schedule style of your existing `ote_full_tcn_v3_tail` run:

- `epochs = 40`
- `warmup = 4`
- `main = 18`
- `fine = 10`
- `tail = 8`
- `4 + 18 + 10 + 8 = 40`

#### LSTM

Your previous LSTM benchmark lived under `models/ote_full_lstm_v1` and was long-only. Since LSTM did not have backend-specific merged rankings at the time, it should be retrained if you want a valid comparison under the new flow.

Command matching your previous long-side benchmark style:

```cmd
python -m model_training.ote_training.ote_xgboost_pipeline ^
  --prepared-root data/prepared/eurusd_5min_ote_full ^
  --output-root models/ote_full_lstm_v2_backend_ranked ^
  --backend torch ^
  --model-type lstm ^
  --targets long_ote ^
  --trials 20 ^
  --max-loaded-features 160 ^
  --window-min 8 ^
  --window-max 40 ^
  --epochs 40 ^
  --batch-size 256 ^
  --hidden-size 64 ^
  --num-layers 2 ^
  --dropout 0.20 ^
  --learning-rate 0.001 ^
  --calibration-method platt
```

If you now want a full short-side LSTM benchmark too, add `short_ote`:

```cmd
--targets long_ote short_ote
```

### Torch Warmup, Fine, And Tail Schedule Rules

These rules matter for TCN and LSTM:

- If you do not set explicit torch phase flags, the trainer auto-builds a warmup/main/fine schedule from `--epochs`.
- If you set any explicit phase count, or set `--torch-tail-epochs` above zero, then you must set all of:
  - `--torch-warmup-epochs`
  - `--torch-main-epochs`
  - `--torch-fine-epochs`
  - and the sum with `--torch-tail-epochs` must equal `--epochs`
- `--torch-fine-lr-scale` controls the lower-LR fine phase.
- `--torch-tail-lr-scale` controls the final tail phase.

Practical guidance:

- TCN: keep the explicit tail schedule you already used successfully.
- LSTM: your previous benchmark used the automatic schedule with no tail. That is still valid. If you want an explicit tail phase for LSTM now, choose a schedule that sums cleanly to `--epochs`.

Example of a valid explicit 40-epoch torch schedule:

```cmd
--epochs 40 ^
--torch-warmup-epochs 4 ^
--torch-main-epochs 20 ^
--torch-fine-epochs 10 ^
--torch-tail-epochs 6
```

### Stage 6: Registry Update

The testing scripts operate off `models/ote_model_registry.json`, so new candidate models should either:

- be added to the registry with new `model_id` values, or
- temporarily replace old artifact paths if you are intentionally superseding them

Recommended approach:

- keep the old registry entries intact
- add new candidate entries for the backend-ranked retrains
- only promote them after regime slicing, threshold policy search, and backtesting

### Stage 7: Regime Slice Report

This is your first post-training evidence layer.

Your existing full-registry run used:

- output root: `model_testing/reports/ote_regime_slices/full_registry_20260330_200iter`
- `bootstrap_iterations = 200`
- `min_positive_events = 50`

Command in that style:

```cmd
python scripts/run_ote_regime_slice_report.py ^
  --registry-path models/ote_model_registry.json ^
  --output-root model_testing/reports/ote_regime_slices/full_registry_20260330_200iter ^
  --bootstrap-iterations 200 ^
  --min-positive-events 50
```

If you only want to test fresh backend-ranked candidates first, run with repeated `--model-id` values after adding them to the registry.

Primary outputs:

- `slice_report.csv`
- `composite_bucket_winners.csv`
- per-model `oof_regime_labeled_predictions.csv`
- per-model `test_regime_labeled_predictions.csv`

### Stage 8: Threshold And Abstain Policy Search

This stage searches regime-aware thresholds and checks whether abstention improves the trading decision layer.

Your existing champion-focused run used:

- output root: `model_testing/reports/ote_threshold_policies/active_champions_cost_aware_20260330`
- `min_positive_events = 50`
- `min_events_per_month = 3.0`
- `min_trades_per_week = 3.0`
- optional: `--write-registry-policies` only after manual review of the saved threshold/backtest artifacts

Command in that style:

```cmd
python scripts/run_ote_threshold_policy_search.py ^
  --regime-report-root model_testing/reports/ote_regime_slices/full_registry_20260330_200iter ^
  --registry-path models/ote_model_registry.json ^
  --output-root model_testing/reports/ote_threshold_policies/active_champions_cost_aware_20260330 ^
  --include-role champion ^
  --min-positive-events 50 ^
  --min-events-per-month 3.0 ^
  --min-trades-per-week 3.0
```

What this does:

- searches composite-regime thresholds
- compares `global_threshold`, `regime_threshold`, and abstain-aware variants
- writes reviewable threshold/backtest artifacts first; registry write-back is now a deliberate follow-up step rather than the default research flow

Primary outputs:

- `policy_table.csv`
- `policy_evaluation.csv`
- optional per-model decision files if `--write-policy-decisions` is used

### Stage 9: Walk-Forward Policy Backtest

This is the final offline promotion gate before paper trading.

Your existing `full_run_v2` backtest used:

- regime report root: `model_testing/reports/ote_regime_slices/full_registry_20260330_200iter`
- output root: `model_testing/reports/ote_policy_backtests/full_run_v2`
- `min_train_years = 2`
- `test_window_months = 3`
- `rolling_step_months = 3`
- `min_folds = 8`
- `min_positive_events = 50`
- `min_events_per_month = 3.0`
- `min_trades_per_week = 3.0`
- `maximum_drawdown_pct = 12.0`
- `drawdown_starting_balance_units = 10000.0`
- `fixed_slippage_pips_per_trade = 0.3`
- `commission_pips_per_trade = 0.35`
- `--targeted-filter-preset full_run_v2`

Command in that style:

```cmd
python scripts/run_ote_policy_backtest.py ^
  --regime-report-root model_testing/reports/ote_regime_slices/full_registry_20260330_200iter ^
  --registry-path models/ote_model_registry.json ^
  --output-root model_testing/reports/ote_policy_backtests/full_run_v2 ^
  --include-role champion ^
  --min-train-years 2 ^
  --test-window-months 3 ^
  --rolling-step-months 3 ^
  --min-folds 8 ^
  --min-positive-events 50 ^
  --min-events-per-month 3.0 ^
  --min-trades-per-week 3.0 ^
  --maximum-drawdown-pct 12.0 ^
  --drawdown-starting-balance-units 10000 ^
  --fixed-slippage-pips-per-trade 0.3 ^
  --commission-pips-per-trade 0.35 ^
  --targeted-filter-preset full_run_v2
```

Walk-forward acceptance now uses account-equity `max_drawdown_pct_below_threshold`, computed from cumulative strategy performance plus `--drawdown-starting-balance-units`. The drawdown gate is advisory risk context rather than a hard promotion veto, so downstream registry/bundle decisions should preserve both the accepted flag and the explicit drawdown readout.

Primary outputs:

- `model_summary.csv`
- `run_summary.json`
- per-model `fold_summary.csv`
- per-model `selected_test_trades.csv`
- per-model `selected_test_equity_curve.csv`
- per-model breakdowns by year, quarter, session, composite regime, and confidence bucket

### Recommended Rerun Order Right Now

Given the new backend-aware ranking step, this is the clean rerun order I recommend:

1. Rebuild backend-aware attribution rankings on `data/prepared/eurusd_5min_ote_full`.
2. Retrain TCN with the new `feature_importance_merged_tcn.csv`.
3. Retrain LSTM with the new `feature_importance_merged_lstm.csv`.
4. Optionally rerun XGBoost so all three backends are using the same new attribution pipeline.
5. Add the new model artifacts to `models/ote_model_registry.json` as candidates.
6. Rerun regime slice reporting.
7. Rerun threshold policy search.
8. Rerun the walk-forward backtest.
9. Only then decide promotion, challenger status, or benchmark-only status.

Practical interpretation of your current belief:

- Yes, it is reasonable to treat the old XGBoost stack as mostly aligned already.
- Yes, TCN and LSTM are the backends that most clearly need retraining and retesting after the backend-aware attribution update.

### Definition Of Done

Your complete model-training workflow is only finished when all of the following exist for the candidate generation you want to promote:

- labeled dataset
- full feature dataset
- prepared OTE target folders
- backend-specific merged ranking files for the trained backends
- trained model artifacts
- registry entries pointing to the new artifacts
- regime slice report
- threshold-policy evaluation
- abstain-policy metadata where justified
- walk-forward backtest report with frictions
- final champion/challenger decision recorded in the registry

### Typical Full Run Order

```cmd
python data/labeling/labeling_engine.py ...
python -m features.cli build ...
python -m preprocessing prepare ...
python -m preprocessing backend-attribution ...
python -m model_training.ote_training.ote_xgboost_pipeline ...    rem xgboost
python -m model_training.ote_training.ote_xgboost_pipeline ...    rem tcn
python -m model_training.ote_training.ote_xgboost_pipeline ...    rem lstm
python scripts/run_ote_regime_slice_report.py ...
python scripts/run_ote_threshold_policy_search.py ...
python scripts/run_ote_policy_backtest.py ...
```

That is now the complete offline OTE workflow from labeling to model-testing evidence.


<a id="production_realtime_signal_system_plan"></a>

## Production-Ready Real-Time Trading Signal System Plan

*Merged 2026-10-09 from `docs/PRODUCTION_REALTIME_SIGNAL_SYSTEM_PLAN.md`.*

### Scope

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

### Executive Assessment

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

### Relevant File Inventory

#### Root docs, manifests, and runbooks

| Path | Role today | Production use |
| --- | --- | --- |
| `README.md` | General repo overview. | Low. Reference only. |
| `#end-to-end-readme` | Best current OTE runbook for the canonical offline pipeline. | High. Use as the canonical offline workflow reference for retraining automation. |
| `docs/papers/Optimal_Trade_Entry_Detection_Pipeline.docx.txt` | Narrative description of the OTE pipeline. | Medium. Reference for intent and assumptions. |
| `docs/papers/Quantitative_Trading_ML_Pipeline.docx.txt` | Broader project notes. | Medium. Reference only. |
| `requirements.txt` | Broad project dependency list. | Medium. Source for existing packages, but too broad for the live system. |
| `ote_requirements.txt` | Focused OTE pipeline dependency list. | High. Base requirements file to extend for the live stack. |

#### Raw data, labeling, and lineage

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

#### Feature engineering core

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

#### Feature families used by the OTE stack

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

#### Large strategy feature catalog

| Path | Role today | Production use |
| --- | --- | --- |
| `features/strategies/*.py` | Hundreds of strategy-specific feature generators. | High only because current promoted models still depend on them. Long-term goal should be to shrink this dependency surface. |
| `features/strategies/misc/*.py` | Miscellaneous strategy helpers. | Low to medium. Pull in only if selected live features actually require them. |

#### Prepared datasets and metadata

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

#### Preprocessing and backend attribution

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

#### OTE training and model artifact generation

| Path | Role today | Production use |
| --- | --- | --- |
| `model_training/ote_training/README.md` | OTE training notes. | Medium. |
| `model_training/ote_training/README.md#full_pipeline_readme` | Detailed training workflow notes. | High for retraining orchestration. |
| `model_training/ote_training/#ote_training_workflow_report` | Training report. | Medium. |
| `model_training/ote_training/feature_ranking.py` | Resolves ranked feature files and attribution gates. | High for retraining. |
| `model_training/ote_training/ote_xgboost_pipeline.py` | Core training/evaluation pipeline for XGBoost and torch backends. | Very high. Reuse directly for quarterly retraining. |
| `model_training/ote_training/torch_models.py` | TCN and LSTM classifier definitions. | Very high for live torch inference. |
| `model_training/ote_training/torch_trainer.py` | Torch training loop and evaluation helpers. | High for retraining. |
| `model_training/model_training_utils.py` | Shared training helpers. | Medium. |
| `model_training/model_layers.py` | Shared layers for older model tracks. | Low for OTE live v1. |

#### Model registry and promoted artifacts

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

#### Post-training evaluation, thresholding, abstain, and backtest

| Path | Role today | Production use |
| --- | --- | --- |
| `model_testing/#post_training_production_plan` | Internal guidance saying deterministic regime routing is the preferred next production step. | Very high. Align live plan with this guidance. |
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

#### Scripts that define the current canonical offline path

| Path | Role today | Production use |
| --- | --- | --- |
| `scripts/run_eurusd_ote_pipeline.bat` | Canonical full offline pipeline from raw CSV through prepare step. | Very high. Treat this as the current reference orchestration. |
| `scripts/run_lstm_backend_attribution_and_training.bat` | Backend attribution plus training flow. | High for retraining automation. |
| `scripts/merge_ote_feature_rankings.py` | Ranking merge utility. | Medium to high for retraining. |
| `scripts/run_ote_regime_slice_report.py` | Regime reporting entrypoint. | High for retraining/shadow reporting. |
| `scripts/run_ote_threshold_policy_search.py` | Threshold and abstain policy search entrypoint. | Very high for retraining. |
| `scripts/run_ote_policy_backtest.py` | Walk-forward policy backtest entrypoint. | Very high for retraining and promotion. |
| `scripts/run_ote_recent_shap_analysis.py` | Recent explainability utility. | Medium for diagnostics. |

#### Existing prototype live-ish app

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

#### Adjacent research modules

| Path | Role today | Production use |
| --- | --- | --- |
| `model_training/regime_classifier/*` | Separate trained regime-classifier track. | Medium. Possible future ensemble signal, but not the recommended first production route. |
| `model_training/pattern_validation/*` | Pattern validation model track. | Medium. Optional future side-model. |
| `model_training/mtf_confluence/*` | MTF confluence model track. | Medium. Optional future side-model. |
| `model_training/classification/*` | Older classification tracks. | Low for OTE live v1. |
| `model_training/regression/*` | Older regression tracks. | Low for OTE live v1. |

#### Tests already in place

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

### Current Canonical Offline Pipeline

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

### Backtest Data Sources Mapped To Real-Time Equivalents

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

### Recommended Production Architecture

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

### Recommended Production Folder Structure

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

### Recommended Runtime Database Design

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

### Recommended Signal Schema

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

### Dependencies To Add

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

### Order Of Implementation

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

### Phased Development Plan

#### Phase 0: Freeze Runtime Contracts And Live Manifests

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

#### Phase 1: Real-Time Ingestion And Canonical Bar Store

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

#### Phase 2: Streaming Feature Engine With Batch Parity

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

#### Phase 3: Model Serving, Thresholding, Abstain, And Signal Generation

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

#### Phase 4: Audit Store, Retention, And Reproducibility

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

#### Phase 5: Live Dashboard, Alerts, And Chart Screenshots

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

#### Phase 6: Reliability, Supervision, Health, And Recovery

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

#### Phase 7: Shadow Mode On Live Paper Data

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

#### Phase 8: Quarterly Retraining, Champion-Challenger Promotion, And Rollback

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

### Recommended First Production Scope

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

### Biggest Risks And How To Handle Them

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Strategy feature sprawl | Current champions depend on `strategy__...` columns from a very large catalog. | Start with a rolling batch adapter for selected strategy features, then retrain a production-focused model with a leaner feature set. |
| Policy packaging gap | Backtest-targeted abstain filters are not yet first-class deployment artifacts. | Promote a complete `live_policy.json` alongside model artifacts and extend registry validation. |
| Feed semantics mismatch | Live volume/spread semantics may differ from the historical CSV. | Normalize and document provider semantics; prefer storing raw plus canonical mapped fields. |
| Timezone/session drift | Session features and market-day labels are highly timezone-sensitive. | Reuse `features/fx_calendar.py` directly and keep absolute UTC timestamps everywhere in storage. |
| Restart gaps | A 24/7 service that misses bars can silently invalidate signals. | Persist raw events and bars immediately, detect missing bar IDs, and run automatic REST backfill on startup and during runtime. |
| Dashboard or screenshot fragility | Operator tooling often breaks after signal logic is already live. | Build end-to-end tests that generate a signal and verify UI, screenshot, and alert side effects together. |

### Recommended Next Actions

1. Build Phase 0 first and export live manifests for the current v1 champions plus the strongest v2 candidates.
2. Decide whether v1 live should serve the current strategy-heavy champions as-is or whether you want a near-term retrain on a smaller feature surface.
3. Implement Phases 1 through 3 as one vertical slice with replay-based parity tests.
4. Add Phase 4 storage and then Phase 5 operator tooling.
5. Run Phase 7 shadow mode before any real operational dependence on the signals.


<a id="model_promotion_and_ote_live_8_model_plan"></a>

## 8-Model OTE Live Promotion Plan

*Merged 2026-10-09 from `docs/MODEL_PROMOTION_AND_OTE_LIVE_8_MODEL_PLAN.md`.*

### Summary

- Final live runtime target: 8 models total, grouped as 4 long and 4 short, with `meta` as the third family and no continuation models in scope.
- Final `active` emitters: `long_ote_tcn_v2_candidate`, `short_ote_tcn_v2_candidate`, `long_reversal_tcn_champion`, `short_reversal_xgb_v1`.
- Final `candidate` shadow models: `long_breakout_tcn_champion`, `short_breakout_tcn_champion`, `long_ote_meta_tcn_champion`, `short_ote_meta_tcn_champion`.
- Research-only models kept out of the 8-model live runtime: `short_reversal_tcn_champion`, `long_breakout_xgb_v1`, legacy OTE v1/XGB/LSTM variants.

### Interface Changes

- Keep `models/ote_model_registry_live_multifamily.json` as the single live 8-model registry for reevaluation, packaging, and runtime export.
- Add `models/ote_model_registry_live_multifamily.json` as the exact 8-model live registry used by `ote_live`.
- Make live runtime semantics depend on existing registry `status`:
  - `active` = live emitter and alert-eligible
  - `candidate` = shadow-only, persisted to DB/dashboard, no operator alerts
  - `deprecated` = excluded from live export/runtime
- Keep `recommended_primary_model_id` per direction, but lock it to the OTE v2 models so the dashboard and legacy fallbacks stay stable.
- Generalize policy packaging and manifest export defaults so they no longer assume only the two OTE TCN v2 models exist.

### Implementation Steps

1. Canonicalize the model set and naming. Use `long_ote_tcn_v2_candidate` and `short_ote_tcn_v2_candidate` as the canonical live OTE IDs, stop using the older active-registry aliases in runtime config, and build the two new multifamily registries around the exact 8 live targets plus the 2 research-only comparison models.

2. Do model-side work before reevaluation. Leave `long_ote_tcn_v2_candidate`, `short_ote_tcn_v2_candidate`, `long_reversal_tcn_champion`, and `short_reversal_xgb_v1` unchanged at the artifact level. Repackage `long_ote_meta_tcn_champion` without retraining first. Retrain `short_ote_meta_tcn_champion`. Run the `long_breakout_tcn_champion` 48-epoch rescue pass. Treat `short_breakout_tcn_champion` as a pipeline repair first, then retrain it after fixing target/trade-conversion assumptions. Write all repaired artifacts to new date-stamped directories and only repoint registry `artifact_path` after reevaluation passes.

3. Rerun `model_testing` from the new multifamily candidate registry in one consistent pass. Run regime slices first, then threshold policy search, then walk-forward policy backtests. Use stable output roots such as `model_testing/reports/ote_regime_slices/multifamily_live_v1`, `model_testing/reports/ote_threshold_policies/multifamily_live_v1`, and `model_testing/reports/ote_policy_backtests/multifamily_live_v1` so live packaging can reference non-date-fragile paths.

4. Apply promotion rules after the unified rerun. Promote to `status=active` only if post-cost profitability, WFE, profitable-quarter share, composite expectancy share, and trade concentration gates pass. Keep `long_ote_tcn_v2_candidate` and `short_ote_tcn_v2_candidate` active on explicit legacy override if the rerun still shows they are profitable and the drawdown gate remains the only failing flag. Keep all other failing models at `status=candidate` so they load shadow-only.

5. Package live policy artifacts for all 8 models from the new threshold-search run, not from OTE-only defaults or direction fallbacks. Every live-target model must get a real `live_policy.json`, including threshold and abstain metadata, under `ote_live/policy_artifacts/<model_id>/`.

6. Export fresh long and short runtime manifests from `models/ote_model_registry_live_multifamily.json`. The long manifest must contain exactly `long_ote_tcn_v2_candidate`, `long_reversal_tcn_champion`, `long_breakout_tcn_champion`, and `long_ote_meta_tcn_champion`. The short manifest must contain exactly `short_ote_tcn_v2_candidate`, `short_reversal_xgb_v1`, `short_breakout_tcn_champion`, and `short_ote_meta_tcn_champion`.

7. Refactor the live collector binding logic so one run can mix active emitters and shadow models at the same time. The current primary-vs-rest logic is not enough for this mission. After the change, all `active` models emit normally, all `candidate` models run with `shadow_mode=True`, and `all_models_active` remains only as an override for special testing.

8. Refactor the dashboard from 2 primary-only inspection panels to an 8-card layout grouped Long and Short. Each card should show model id, family, status badge, latest decision, latest probability, active threshold, last signal timestamp, and a link or embedded image for the latest stored chart when available. Keep the existing all-signals price chart and recent-signal history, but drive them from the new 8-model manifests.

9. Update live-stack defaults and docs so `run_live_stack`, manifest export, and policy packaging all point at the new multifamily live registry and the stable multifamily report roots. The repo should no longer boot the live app from the old v1/v2-only candidate registry.

### Test Plan

- Extend `pytest` coverage around `ote_live` so status-driven mixed loading is verified: active models emit, candidate models shadow, deprecated models are skipped, and the exported manifests still validate.
- Keep `tests/test_ote_live_runtime_manifests.py` and `tests/test_ote_live_signal_runtime_integration.py` green, then add new cases for the 8-model long/short manifests and the mixed active-shadow collector binding path.
- Run feature parity replay on the 4 active emitters first, then on the 4 candidate shadow models after their manifests are exported.
- Run packaging and manifest export in dry-run form, then run one short live collector cycle and verify that only active models send alerts while shadow models persist `shadow` decisions for the dashboard.
- Final acceptance check: the runtime manifests contain exactly 4 long and 4 short models, the dashboard resolves all 8 cards, and the collector starts without falling back to old OTE-only defaults.

### Assumptions

- `mera` means the existing `meta` model family.
- Continuation models are fully out of scope for this mission.
- Stable model IDs are preserved; retrains change artifact directories, not model IDs.
- `short_reversal_tcn_champion` stays as a research challenger only, and `long_breakout_xgb_v1` stays as a benchmark/deprecated recovery reference only.
- The user wants 8 models visible in the live app/runtime even if some remain shadow-only after reevaluation.


<a id="ote_multi_family_tcn_v1_v2_retrain_plan"></a>

## OTE Multi-Family TCN v1/v2 Retrain, Retune, and Recalibration Plan

*Merged 2026-10-09 from `model_testing/OTE_MULTI_FAMILY_TCN_V1_V2_RETRAIN_PLAN.md`.*

### Purpose

This document turns the last two multi-family TCN training trials into a concrete next-step plan for:

- `models/ote_multi_family_tcn_v1`
- `models/ote_multi_family_tcn_v2`

The goal is to decide, for each of the six target models, whether to:

- keep the current artifact,
- recalibrate it,
- retune and retrain it,
- or retire it from the active candidate set.

This plan uses the saved `training_summary.json`, `training_history.json`, and cross-version comparisons from the two latest runs.

---

### Executive Summary

| Model | Keep Now | Why | Next Step |
|---|---|---|---|
| `short_breakout` | `v1` | Best CV and test metrics in both trials | Freeze `v1`, optional bounded challenger run |
| `long_breakout` | `v1` | `v2` collapsed to 7 test events and F0.5 `0.0513` | Retune `v2` from the `v1` neighborhood |
| `long_reversal` | `v1` active, `v2` challenger | `v2` improved AP slightly, but `v1` still wins on test event F0.5 | Recalibrate `v2` first, retrain only if needed |
| `short_reversal` | `v1` | `v2` lost too much recall and test F0.5 | Retune `v2` from the `v1` neighborhood |
| `long_continuation_pullback` | keep neither active | Both versions emit 0 test events | Full retune plus continuation calibration fix |
| `short_continuation_pullback` | keep neither active | `v2` improved AP, but thresholded behavior is still unusable | Use `v2` as the research base and retune |

---

### What The Last Two Trials Tell Us

#### 1. Most models did not run out of stability. They ran out of epoch budget.

Both `v1` and `v2` used a fixed 18-epoch PyTorch budget. The current trainer already has patience-based early stopping and best-checkpoint restore in `model_training/ote_training/torch_trainer.py`, so the issue is not "missing early stopping." The issue is that validation AUPRC kept improving through the end of training for most targets.

Observed pattern:

- In `v2`, 5 of 6 targets hit best validation AUPRC on the final epoch.
- In `v1`, 5 of 6 targets also hit best validation AUPRC on the final epoch.

Interpretation:

- The trainer is behaving as designed.
- The 18-epoch schedule is usually too short for these TCN runs.
- The current auto phase split is effectively too compressed for several targets.

#### 2. Continuation targets have a pipeline-level handicap.

In `model_training/ote_training/ote_xgboost_pipeline.py`, `resolve_calibration_method()` forces any target containing `"continuation"` to use calibration method `none`, even when `platt` was requested.

That means:

- continuation targets are not being allowed to use the same probability correction path as breakout/reversal targets,
- their thresholds are selected on uncalibrated OOF probabilities,
- and both continuation models are currently failing on held-out thresholded behavior.

#### 3. Threshold selection currently has no explicit coverage floor.

The threshold selector in `model_training/ote_training/ote_xgboost_pipeline.py` optimizes:

1. event F0.5
2. event precision
3. closeness of predicted event count to true event count

That is reasonable for strong models, but it does not explicitly require:

- a minimum number of emitted events,
- a minimum recall floor,
- or a minimum monthly trade count.

For continuation targets, where score distributions are weaker and calibration is disabled, this can still produce thresholds that are unusable on held-out data.

#### 4. `v2` improved search geometry, but not every target family improved.

`v2` brought useful changes:

- 3 CV folds instead of 2
- smaller purge-aware geometry steps
- cleaner artifact summaries

But the actual model quality was mixed:

- `short_breakout` remained excellent, but `v1` is still better.
- `long_reversal` was roughly flat to slightly mixed.
- `long_breakout` regressed sharply.
- continuation remained unusable at deployment thresholds.
- `short_reversal` regressed materially.

---

### Global Fixes Before The Next Retrain Batch

These should happen before we judge the next training run.

#### A. Increase epoch budget and make the phase schedule explicit

The stronger reference point is not the generic 18-epoch multi-family schedule. It is the proven schedule used by `models/ote_full_tcn_v2`:

```text
epochs = 40
warmup = 4
main   = 18
fine   = 10
tail   = 8
fine_lr_scale = 0.20
tail_lr_scale = 0.07
use_amp = false
```

That schedule produced strong saved results:

- `long_ote`: CV AP/F0.5 `0.6254 / 0.7738`, test AP/F0.5 `0.7625 / 0.8518`
- `short_ote`: CV AP/F0.5 `0.6239 / 0.7473`, test AP/F0.5 `0.7375 / 0.7871`

More importantly, the tail phase appears to have mattered:

- `long_ote` hit best validation AUPRC at epoch `39/40`
- `short_ote` hit best validation AUPRC at epoch `37/40`
- both improved materially during the tail phase rather than flattening early

So the next multi-family retrain batch should borrow this proven shape instead of using a shorter generic schedule.

Recommended default:

```text
--epochs 40
--torch-warmup-epochs 4
--torch-main-epochs 18
--torch-fine-epochs 10
--torch-tail-epochs 8
--torch-fine-lr-scale 0.20
--torch-tail-lr-scale 0.07
```

Recommended runtime defaults:

- `use_amp = false` for TCN runs, matching the stronger `ote_full_tcn_v2` artifacts
- keep the existing best-checkpoint restore logic
- keep the explicit phase schedule so the late tail behavior is visible in `training_history.json`

Important comparison caveat:

- `ote_full_tcn_v2` was trained on much larger OTE datasets with 8 CV folds, so it is not a perfect apples-to-apples comparison with the six multi-family targets.
- even so, it is the best on-disk evidence we have for what a successful TCN schedule looks like in this repo.

Review gate:

- If a retrained model still peaks in the final `1-2` epochs under the `40` epoch schedule, raise it to `48` for that target on the next pass.
- If a target clearly flattens before the tail, keep `40` but consider shrinking tail epochs for that specific family later.

#### B. Allow continuation calibration experiments

For the next continuation experiments, do not hard-disable calibration.

Test these variants on OOF predictions:

- `none`
- `platt`
- `isotonic`

Pick the winner by:

1. OOF average precision
2. OOF event F0.5
3. coverage sanity, meaning the threshold must emit a non-trivial number of events

#### C. Add a coverage constraint to threshold selection for weak targets

For continuation and any future weak-recall model, add a threshold acceptance floor such as:

- `predicted_events >= max(25, 0.15 * true_events)` on OOF

or

- `event_recall >= 0.10` on OOF

The exact floor can be tuned, but the next run should not accept a threshold that produces effectively zero live decisions.

#### D. Add feature-count guardrails by target family

The `long_breakout v2` collapse from `106` features to `6` is too aggressive.

Use family-specific feature guardrails:

- breakout/reversal: floor at `48` selected features
- continuation: keep in the `16-64` range until coverage is fixed

#### E. Separate "ranking quality" from "deployment quality"

A model can have good test AP and still be unusable as a deployed classifier.

For the next round, every target should be judged on both:

- ranking metrics: AP, ROC AUC, Brier
- thresholded event metrics: precision, recall, F0.5, predicted event count

---

### Model-by-Model Plan

### `short_breakout`

#### Current read

- Keep `v1`.
- `v1` is stronger than `v2` on every saved metric:
  - CV AP: `0.9628` vs `0.9250`
  - CV event F0.5: `0.9327` vs `0.8922`
  - Test AP: `0.9950` vs `0.9871`
  - Test event F0.5: `0.9736` vs `0.9457`
  - Test recall: `0.8857` vs `0.7910`

#### Recommendation

- Freeze `v1` as the active short-breakout artifact.
- Do not spend the next cycle on a full replacement attempt unless we are explicitly running a challenger check.

#### Next action

- Optional challenger run only.
- Re-run the `v1` parameter neighborhood under the `v2` 3-fold geometry to verify the advantage survives the stricter CV setup.

#### Schedule choice

- Do not make `short_breakout` the first consumer of a brand-new schedule idea.
- If a challenger run is executed, use the proven full TCN schedule from `ote_full_tcn_v2` unchanged:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This target is already excellent, so the goal is schedule verification, not aggressive exploration.

#### Next-run target settings

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: `20-24`
- selected features: `80-120`
- layers: `2`
- learning rate: `0.0008-0.0015`
- focal alpha: `0.72-0.78`
- focal gamma: `3.0-3.4`
- hard negative radius: `1`
- hard negative multiplier: `2.2-2.6`
- calibration: compare `platt` vs `isotonic`

#### Acceptance gate

Only replace `v1` if the challenger:

- matches or beats `0.9736` test event F0.5,
- keeps recall within `3` percentage points of `v1`,
- and does not degrade AP.

---

### `long_breakout`

#### Current read

- Keep `v1`.
- Retune `v2`.
- `v2` regressed badly:
  - feature count fell from `106` to `6`
  - test predicted events fell from `280` to `7`
  - test event F0.5 fell from `0.7864` to `0.0513`

#### Likely issue

`v2` over-compressed feature selection and ended up with a model that ranks okay but is far too conservative at deployment threshold.

#### Recommendation

- Do not keep `v2` as-is.
- Retrain from the `v1` neighborhood, not the `v2` neighborhood.

#### Next action

- Full retune and retrain.

#### Schedule choice

- Use the full proven TCN schedule from `ote_full_tcn_v2` as the default retrain schedule:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This target is the clearest example that the shorter multi-family run shape is not enough.
- Because `v2` collapsed to `7` test events, this retrain should keep the full tail phase intact rather than shortening it.

#### Next-run target settings

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: `20-28`
- selected features: `64-120`
- hard floor: reject trials with fewer than `32` selected features
- layers: `3`
- learning rate: `0.0035-0.0050`
- focal alpha: `0.70-0.78`
- focal gamma: `2.7-3.1`
- hard negative radius: `1-2`
- hard negative multiplier: `1.8-2.3`
- calibration: `platt` and `isotonic` comparison

#### Additional checks

- Review feature-ranking output for why the selected set collapsed to `6`.
- Compare raw OOF score percentiles and test score percentiles to see whether the issue is ranking drift, calibration drift, or both.

#### Acceptance gate

The next candidate should not be considered healthy unless it:

- clears `0.70` test event F0.5,
- emits materially more than `7` test events,
- and keeps recall above `0.20`.

---

### `long_reversal`

#### Current read

- Keep `v1` active.
- Keep `v2` as challenger only.
- `v2` slightly improved ranking metrics:
  - test AP: `0.6995` vs `0.6880`
- but `v1` still wins on deployed event behavior:
  - test event F0.5: `0.6951` vs `0.6753`
  - test recall: `0.3187` vs `0.2989`

#### Likely issue

This is close enough that the first fix should be recalibration and threshold re-search, not immediate architecture change.

#### Recommendation

- Try recalibration-first on `v2`.
- Retrain only if recalibration does not recover thresholded performance.

#### Next action

- Recalibration study on the saved `v2` OOF/test predictions.

#### Schedule choice

- First pass: no retrain, recalibration only.
- If recalibration fails, retrain with the full proven OTE schedule:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This target is close enough that we should not widen the search space until probability scaling and thresholding have been re-checked.

#### Recalibration plan

- Compare:
  - raw probabilities
  - `platt`
  - `isotonic`
- Re-search thresholds in the `0.45-0.70` range
- Prefer the version that improves event F0.5 without losing AP

#### If retraining is needed

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: keep at `24`
- selected features: `70-95`
- layers: `3`
- learning rate: `0.0040-0.0050`
- focal alpha: `0.70-0.78`
- focal gamma: `2.6-3.0`
- hard negative radius: `1`
- hard negative multiplier: `1.9-2.2`
- calibration: `platt` vs `isotonic`

#### Acceptance gate

Promote a new `long_reversal` only if it at least matches `v1` on test event F0.5, or slightly trails it while clearly improving AP and recall together.

---

### `short_reversal`

#### Current read

- Keep `v1`.
- Retune `v2`.
- `v2` became too conservative:
  - test AP: `0.5257` vs `0.6435`
  - test event F0.5: `0.5229` vs `0.7644`
  - test predicted events: `82` vs `190`
  - test recall: `0.1831` vs `0.4188`

#### Likely issue

Relative to `v1`, the `v2` neighborhood moved toward:

- shorter window
- more aggressive focal setup
- softer hard-negative multiplier

That combination appears to have hurt coverage and recall.

#### Recommendation

- Retune from the `v1` neighborhood.

#### Next action

- Full retune and retrain.

#### Schedule choice

- Use the full proven OTE schedule as the starting schedule:
  - `40` epochs
  - `4/18/10/8` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- Keep the schedule stable and move the search back toward the `v1` parameter neighborhood.
- This target does not need a continuation-specific schedule; it needs more room plus less aggressive parameter drift.

#### Next-run target settings

- schedule: full `40` epoch OTE schedule `4/18/10/8`
- window: `24-28`
- selected features: `80-100`
- layers: `3`
- learning rate: `0.0040-0.0050`
- focal alpha: `0.70-0.75`
- focal gamma: `2.8-3.2`
- hard negative radius: `1`
- hard negative multiplier: `1.9-2.2`
- calibration: `platt` vs `isotonic`
- threshold search emphasis: recover recall, not just precision

#### Acceptance gate

Do not accept a new version unless it restores:

- test event F0.5 above `0.70`
- recall above `0.35`

---

### `long_continuation_pullback`

#### Current read

- Keep neither `v1` nor `v2` active.
- Both versions emit `0` test events.
- Both versions have test event F0.5 of `0.0`.
- `v2` also expanded feature count from `19` to `142` while still failing on coverage.

#### Likely issue

This is both a modeling problem and a pipeline problem:

- continuation calibration is disabled,
- thresholding is being done on uncalibrated probabilities,
- the target is likely too precision-biased at the current focal/threshold combination,
- and the current model family is not producing stable actionable probabilities.

#### Recommendation

- Do not choose between `v1` and `v2` for deployment.
- Start a continuation recovery track.

#### Next action

1. Run a recalibration-only study on saved artifacts.
2. Then run a smaller, more disciplined retrain.

#### Schedule choice

- First pass: no retrain, recalibration only.
- If retraining is still needed after recalibration, use a continuation-specific long-tail variant instead of the raw breakout/reversal template:
  - `40` epochs
  - `4/14/12/10` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- Reason:
  - continuation needs more low-LR refinement once coverage is restored,
  - and this family should bias more of the budget toward fine and tail phases than toward the high-LR middle.

#### Recalibration plan

Using saved OOF predictions from both `v1` and `v2`, test:

- raw
- `platt`
- `isotonic`

Then re-search thresholds with a coverage floor.

If either existing artifact starts emitting sensible events after recalibration, use that as the next training baseline.

#### Next-run target settings

- schedule: continuation variant `40` epochs with `4/14/12/10`
- window: `32-48`
- selected features: `16-48`
- layers: prefer `3`; only use `4` if `3` clearly underfits
- learning rate: `0.0025-0.0040`
- focal alpha: `0.68-0.76`
- focal gamma: `1.5-2.5`
- hard negative radius: `2-4`
- hard negative multiplier: `1.0-1.4`
- calibration: test `platt` and `isotonic`

#### Why these settings

- Longer windows fit continuation better than short breakout-style windows.
- Smaller feature sets reduce noisy feature sprawl.
- Lower gamma should make the loss less precision-only and more recall-tolerant.
- Moderate hard-negative settings should avoid over-penalizing near-miss continuation setups.

#### Acceptance gate

The next continuation candidate must clear these basic gates before any champion discussion:

- non-zero held-out predicted events
- OOF threshold with coverage floor satisfied
- test event F0.5 materially above `0.0`

---

### `short_continuation_pullback`

#### Current read

- Keep neither version active.
- Use `v2` as the research base.
- `v2` improved ranking quality:
  - test AP: `0.4188` vs `0.1522`
- but thresholded behavior is still unusable:
  - test event F0.5: `0.0478`
  - test predicted events: `3`

#### Likely issue

The model is learning some ranking signal, but thresholding and probability scaling are not producing usable decision coverage.

#### Recommendation

- Use `v2` as the starting point for continuation recovery.
- Fix calibration and coverage first.

#### Next action

1. Recalibration-only study on saved `v2` predictions.
2. Then retrain with tighter feature and threshold guardrails.

#### Schedule choice

- First pass: no retrain, recalibration only.
- If retraining is still required, use the same continuation-specific long-tail variant:
  - `40` epochs
  - `4/14/12/10` phase split
  - `fine_lr_scale = 0.20`
  - `tail_lr_scale = 0.07`
  - `use_amp = false`
- This keeps the late low-LR behavior that worked in `ote_full_tcn_v2`, but pushes even more budget into fine and tail for a weaker target family.

#### Next-run target settings

- schedule: continuation variant `40` epochs with `4/14/12/10`
- window: `20-32`
- selected features: `24-64`
- layers: prefer `3`
- learning rate: `0.0015-0.0030`
- focal alpha: `0.72-0.82`
- focal gamma: `1.4-2.2`
- hard negative radius: `2-3`
- hard negative multiplier: `1.0-1.3`
- calibration: `platt` vs `isotonic`

#### Why these settings

- `v2` already improved AP, so keep its directionally better signal base.
- A smaller feature budget should make the signal less noisy.
- Lower-to-moderate gamma should help keep recall from collapsing.
- Re-enabling calibration is the most important single change.

#### Acceptance gate

Before we compare this target to any older artifact, the next version must first show:

- non-zero held-out predicted events
- event F0.5 above `0.25`
- stable OOF threshold behavior with a coverage floor

---

### Recommended Run Order

#### Wave 1: low-cost recalibration studies

Run recalibration-only comparisons on saved predictions for:

1. `long_reversal v2`
2. `long_continuation_pullback v1`
3. `long_continuation_pullback v2`
4. `short_continuation_pullback v1`
5. `short_continuation_pullback v2`

Goal:

- see whether threshold failure is mostly a calibration problem before paying for full retraining

#### Wave 2: highest-value retrains

Retrain in this order:

1. `long_breakout`
2. `short_reversal`
3. `long_reversal` only if Wave 1 recalibration does not recover `v2`

#### Wave 3: continuation recovery

Retrain:

1. `long_continuation_pullback`
2. `short_continuation_pullback`

These should be treated as research targets, not deployment targets, until they emit stable non-zero held-out events.

#### Wave 4: optional verification challenger

Run one bounded `short_breakout` challenger using the `v1` parameter neighborhood under the `v2` CV geometry.

---

### Success Criteria For The Next Round

The next round should be considered successful if it delivers all of the following:

1. At least one retrained non-continuation target that no longer peaks at the final epoch.
2. `long_breakout` recovers usable thresholded behavior and no longer collapses to single-digit event count.
3. `short_reversal` recovers recall and moves back above `0.70` test event F0.5.
4. At least one continuation target emits non-zero held-out events after recalibration or retraining.
5. The continuation path is no longer forced into `calibration = none` without an explicit comparison.

---

### Final Recommendation Snapshot

- Keep:
  - `short_breakout v1`
  - `long_breakout v1`
  - `long_reversal v1`
  - `short_reversal v1`

- Keep as challenger only:
  - `long_reversal v2`
  - `short_breakout v2`

- Retune aggressively:
  - `long_breakout v2`
  - `short_reversal v2`
  - `long_continuation_pullback v1`
  - `long_continuation_pullback v2`
  - `short_continuation_pullback v1`
  - `short_continuation_pullback v2`

- Highest-priority pipeline fix:
  - allow continuation calibration experiments and add a threshold coverage floor

<a id="post_training_production_plan"></a>

## OTE Post-Training Production Plan

*Merged 2026-10-09 from `model_testing/POST_TRAINING_PRODUCTION_PLAN.md`.*

### Purpose

This document is the implementation plan for moving the OTE stack from trained base models to a regime-aware, production-candidate trading system that can be trusted across changing market conditions.

The plan assumes the current codebase, artifacts, and directory layout in this repository as of March 29, 2026.

---

### Current Baseline Snapshot

#### Base model status

| Direction | Current leader | Why |
|---|---|---|
| Long OTE | `models/ote_full_tcn_v3_tail/long_ote` | Best overall combination of CV and held-out test performance |
| Short OTE | Mixed: `models/ote_full_xgb_v2/short_ote` and `models/ote_full_tcn_v3_tail/short_ote` | XGBoost more stable in CV, TCN better on held-out test |
| LSTM | Benchmark only | Useful diversity signal, not a primary production candidate |

#### Key numbers that drive the plan

| Model | CV mean AP | CV mean event F0.5 | Test AP | Test event F0.5 |
|---|---|---|---|---|
| `ote_full_tcn_v3_tail/long_ote` | 0.5693 | 0.7231 | 0.6987 | 0.8013 |
| `ote_full_xgb_v2/long_ote` | 0.5476 | 0.7068 | 0.6669 | 0.7041 |
| `ote_full_tcn_v3_tail/short_ote` | 0.4975 | 0.6675 | 0.6562 | 0.7878 |
| `ote_full_xgb_v2/short_ote` | 0.5479 | 0.7008 | 0.6330 | 0.7033 |

#### Working conclusions

1. TCN is the best long-side base learner.
2. Short-side production selection cannot be finalized without regime slicing.
3. Regime-aware routing is more important than building a weighted ensemble right now.
4. The current ensemble layer in `ict_app/ensemble/ensemble_engine.py` should be treated as future integration code, not as the next production step.

---

### Production Definition of Done

The OTE system is not production-ready until all of the following are true:

1. Direction-by-direction champions exist with clear challenger tracking in a machine-readable registry.
2. Performance is measured by year, session, volatility regime, and trend regime.
3. Thresholds are selected by regime, with explicit abstain rules for low-confidence and hostile conditions.
4. The walk-forward backtest includes spread, slippage, cooldowns, and position sizing with Walk-Forward Efficiency above 0.50.
5. Performance is not carried by one narrow regime or one narrow year.
6. Paper trading over at least 4 weeks confirms that live behavior matches backtest expectations.
7. Model promotion and rollback are documented and repeatable.

---

### Execution Order

1. Freeze champion and challenger baselines in a model registry.
2. Build joinable prediction outputs, deterministic regime labels, and regime-sliced reports.
3. Add regime-conditioned thresholds and a selective abstain policy.
4. Build a regime router for model selection.
5. Backtest the full policy with realistic frictions using walk-forward protocol.
6. Paper trade with champion/challenger monitoring.
7. Only then test stacking or weighted ensemble methods if routing still leaves meaningful performance on the table.

---

### Research Foundation

The design choices in this plan are grounded in published research on forex regime classification, selective prediction, and production deployment of ML trading systems. Key findings are summarized here and referenced throughout the phases.

#### Regime detection: deterministic labeling over learned classifiers

For EURUSD 5-minute intraday data, deterministic indicator-threshold regime labeling is preferred over trained classifiers (LSTM, HMM, etc.) for the following reasons:

- Published FX regime work uses ADX thresholds for trend strength and ATR percentiles for volatility as primary discriminants. These features are already computed in the OTE feature set.
- Deterministic labels produce identical outputs given identical inputs, which is critical for backtesting reproducibility and debugging. Heuristic HMM implementations in production trading tools explicitly cite this as an advantage over learned models.
- A trained regime classifier adds an entire model lifecycle (training, calibration, monitoring, retraining) to the production stack. A deterministic labeler adds a single function with no training data dependency.
- The regimes that matter for OTE routing (trend direction, volatility level, session, stress) are directly observable from features the pipeline already computes. There is no hidden state to infer.

HMMs remain a viable secondary method for cross-validation of regime assignments, particularly for volatility clustering detection. Research on EUR/USD specifically shows that 2-3 HMM states capture the dominant regime-switching dynamic, and that models with more states tend to overfit.

#### Selective prediction: separating prediction from execution

Research on confidence-threshold trading frameworks shows that separating the directional prediction from the execution decision (trade or abstain) improves risk-adjusted returns compared to traditional all-trade classification. The model produces a calibrated probability; a separate policy layer decides whether the signal is strong enough to act on, conditioned on the current regime. This is the framework used in Phase 2.

#### Walk-forward validation: the 78% false-positive rate

Research on walk-forward optimization demonstrates that up to 78% of strategies show negative out-of-sample Sharpe ratios despite positive in-sample results. Walk-Forward Efficiency (WFE), defined as the ratio of out-of-sample returns to in-sample returns, must exceed 0.50 to indicate a strategy that is not overfit. This threshold is enforced in Phase 4.

#### Transaction cost dominance in intraday forex

Published comparative analyses of ML strategies for forex directional forecasting find that incorporating institutional-level transaction costs reverses the profitability of several strategies that looked profitable on raw accuracy metrics. The Mean Absolute Directional Loss (MADL) function was proposed specifically to align model optimization with trading profitability rather than classification accuracy. Phase 4 enforces post-cost evaluation.

#### Regime overfitting from excessive slicing

Slicing performance by every combination of regime dimensions creates a combinatorial explosion where most cells have too few samples for reliable threshold estimation. The mitigation is to use composite regimes (trend × volatility only, 5-8 buckets maximum) rather than the full cross-product, and to require minimum event counts per bucket before allowing regime-specific thresholds.

#### Complexity measures as features

Research on forex volatility forecasting (2024) found that incorporating the Hurst exponent and fuzzy entropy as features significantly enhanced deep learning model accuracy. Rolling Hurst exponent is a strong regime-discriminating feature because it directly measures whether the series is trending (H > 0.5), mean-reverting (H < 0.5), or random walk (H ≈ 0.5).

---

### Regime Taxonomy

Regime labels are computed deterministically from features that already exist in the OTE feature set. No trained model is required.

#### Regime families and detection rules

##### Trend regime

| Bucket | Detection rule | Interpretation |
|---|---|---|
| strong_up | ADX > 30 AND EMA alignment > 0.7 | Directional uptrend with conviction |
| weak_up | ADX 20-30 AND close > EMA50 | Mild upward bias |
| ranging | ADX < 20 | No directional conviction |
| weak_down | ADX 20-30 AND close < EMA50 | Mild downward bias |
| strong_down | ADX > 30 AND EMA alignment < -0.7 | Directional downtrend with conviction |

Source features: `adx_14`, `ema_alignment`, `ema_spread_21_50_atr`

##### Volatility regime

| Bucket | Detection rule | Interpretation |
|---|---|---|
| low | ATR percentile rank < 25th over trailing 504 bars | Compression |
| medium | ATR percentile rank 25th-75th | Normal |
| high | ATR percentile rank > 75th | Expansion |

Source features: `atr_14`, `atr_ratio_14_50`, `rolling_vol_20`

##### Session regime

| Bucket | UTC hours | Interpretation |
|---|---|---|
| asia | 22:00-07:00 | Low liquidity, range-bound |
| london | 07:00-12:00 | Directional, liquidity sweeps |
| new_york | 12:00-17:00 | Reversals, FVG fills |
| overlap | 12:00-16:00 | Peak liquidity (subset of NY) |
| off_hours | 17:00-22:00 | Thin markets, avoid trading |

Source features: `hour_sin`, `hour_cos`, `in_london_session`, `in_london_ny_overlap`

##### Event stress regime

| Bucket | Detection rule | Interpretation |
|---|---|---|
| normal | range_shock_20 < 2.0 | Typical conditions |
| elevated | range_shock_20 2.0-3.0 | Heightened volatility |
| high | range_shock_20 > 3.0 | Extreme conditions, consider abstaining |

Source features: `range_shock_20`, `displacement_bullish`, `displacement_bearish`

#### Composite regime

The composite regime is the primary key for routing and threshold selection. It is formed by concatenating trend and volatility regimes only:

```
composite_regime = trend_regime + "_" + vol_regime
```

This produces a maximum of 15 cells (5 trend × 3 volatility). Session and stress regimes are applied as separate filters on top of the composite.

Do not use the full cross-product of all four regime families. That creates 225 cells and most will have too few samples to support reliable threshold estimation.

#### Minimum data requirements per regime bucket

- 50 positive events minimum before allowing a regime-specific threshold.
- If a bucket has fewer than 50 events, use the global threshold for that direction.
- Flag the bucket as "insufficient data for regime threshold" in reports.

---

### Phase 0: Freeze Baselines and Model Registry

#### Goal

Create a stable production starting point before adding moving parts.

#### Decisions

| Role | Model path |
|---|---|
| Long champion | `models/ote_full_tcn_v3_tail/long_ote` |
| Long challenger | `models/ote_full_xgb_v2/long_ote` |
| Short champion candidate A | `models/ote_full_xgb_v2/short_ote` |
| Short champion candidate B | `models/ote_full_tcn_v3_tail/short_ote` |
| Benchmark only | `models/ote_full_lstm_v1/long_ote` |

Short champion selection is deferred until Phase 1 regime slicing provides evidence.

#### Implementation

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

#### Proposed files

- New: `models/ote_model_registry.json`
- New: `models/ote_model_registry.schema.json`
- New: `models/ote_registry_loader.py`
- Update or deprecate: `models/best_models.py`

#### Exit criteria

Every production-facing model path comes from the registry, not from ad hoc hard-coded file picks.

---

### Phase 1: Joinable Predictions, Regime Labels, and Regime Attribution

#### Goal

Make existing prediction artifacts analyzable by regime without re-running full training, and resolve the short-side champion question with evidence.

#### Why this comes next

Current prepared split CSVs intentionally exclude timestamps, which makes downstream prediction analysis row-index keyed. That is workable for training, but awkward for regime analysis, backtests, and policy debugging.

#### Implementation: joinable predictions

Export enough row identity metadata to join `oof_predictions.csv` and `test_predictions.csv` back to source bars:

- Add a `source_row_idx` column to OOF and test prediction exports that maps each prediction back to the row index in the original feature dataset.
- Build a joiner utility that takes a prediction CSV and the original timestamped feature dataset and produces a timestamped prediction table.

#### Implementation: deterministic regime labeler

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

#### Implementation: regime-sliced report

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

#### Feature addition: rolling Hurst exponent

Add rolling Hurst exponent as a feature for future regime discrimination. Compute over 30-bar and 100-bar windows on the close series. This feature directly measures trending (H > 0.5) versus mean-reverting (H < 0.5) behavior and has been shown to improve forex volatility forecasting accuracy.

#### Proposed files

- Update: `features/preprocessing.py` (add source_row_idx export)
- Update: `model_training/ote_training/ote_xgboost_pipeline.py` (propagate source_row_idx to predictions)
- New: `model_testing/ote_prediction_joiner.py`
- New: `model_testing/ote_regime_labeler.py`
- New: `model_testing/ote_regime_slices.py`
- New: `scripts/run_ote_regime_slice_report.py`

#### Deliverables

- Joined prediction table with timestamps and source row keys.
- Regime-labeled OOF and test prediction tables.
- Report by model, direction, year, session, trend regime, and volatility regime with bootstrap CIs.

#### Exit criteria

Can answer: "Which model wins for long and short inside each major composite regime bucket?" with statistical confidence.

---

### Phase 2: Threshold by Regime and Selective Abstain Policy

#### Goal

Improve trustworthiness by changing when we trade, not only which model we use. Separate directional prediction from execution decisions.

#### Design principle

The model produces a calibrated probability. A separate policy layer decides whether the signal is strong enough to act on. This separation follows the selective classification framework: the model predicts direction; the policy chooses coverage (trade) or abstention (skip) based on confidence and regime context.

#### Hard abstain rules

These conditions produce no signal regardless of model output:

| Condition | Rule |
|---|---|
| Event stress | stress_regime = "high" (range_shock > 3.0) |
| Off-hours session | session_regime = "off_hours" (17:00-22:00 UTC) |
| Cooldown | Fewer than 4 bars (20 minutes) since last emitted signal in same direction |
| Insufficient data | Regime bucket flagged as "insufficient data" and no global fallback is above threshold |

#### Soft abstain rules: regime-conditioned thresholds

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

#### Threshold search procedure

For each composite regime bucket with at least 50 positive events:

1. Grid search threshold from 0.40 to 0.90 in steps of 0.05.
2. At each threshold, compute event F0.5 and post-cost expectancy on OOF predictions.
3. Select threshold maximizing `0.6 * event_f05 + 0.4 * normalized_expectancy`.
4. Reject any threshold that produces fewer than 3 events per month on average.
5. Use leave-one-year-out cross-validation for the threshold search: optimize on all years except one, evaluate on the held-out year, repeat for each year, average.

For buckets with fewer than 50 positive events:

- Fall back to the global threshold for that direction.
- Log the bucket as data-insufficient.

#### Minimum pip distance filter

Reject any signal where the expected move (calibrated probability × average favorable excursion for that regime) is less than 2× the spread for the current session.

#### Proposed files

- Update: `model_training/ote_training/ote_xgboost_pipeline.py` (propagate regime labels)
- New: `model_testing/ote_threshold_policy.py`
- New: `model_testing/ote_abstain_policy.py`
- New: `scripts/run_ote_threshold_policy_search.py`

#### Deliverables

- Policy table: direction × composite regime → threshold, cooldown, abstain flag, minimum event count satisfied.
- Report comparing three policies: global threshold, regime-conditioned threshold, regime-conditioned threshold plus abstain rules.
- Coverage analysis: what percentage of signals does each policy act on versus abstain.

#### Exit criteria

The regime-aware threshold policy beats the global threshold policy on event F0.5 and trading expectancy after costs, without collapsing trade counts below 3 per week per direction.

---

### Phase 3: Regime Router for Model Selection

#### Goal

Select the best base learner by market state before considering averaging or stacking.

#### Design principle

Use deterministic routing rules populated from Phase 1 evidence. Do not average model probabilities. Research shows that simple model selection by regime outperforms probability averaging when the models have different strengths across regimes. Averaging dilutes the specialist advantage.

#### Routing logic

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

#### Regime transition handling

When regime probabilities are ambiguous (the composite regime label changed within the last 6 bars), apply a conservative override:

- Use the more conservative of the two candidate models.
- Widen the threshold by 0.05 above the regime-specific threshold.
- Reduce position size to 50% of normal.

This addresses the documented pitfall that walk-forward optimization responds to regime changes with a lag.

#### Router evaluation protocol

- Compare router policy against best single-model policy and against global threshold.
- Use walk-forward evaluation: 2-year minimum training window, 3-month test window, rolling quarterly.
- Primary metric: post-cost expectancy.
- Secondary metric: coefficient of variation of monthly returns across composite regime buckets (lower is better).

#### Proposed files

- New: `model_training/ensemble/ote_regime_router.py`
- New: `model_training/ensemble/ote_router_config.json`
- Update: `model_testing/prediction.py`

#### Deliverables

- Router config file with direction × composite regime → model_id mapping.
- Router evaluation report against single-model baselines.

#### Exit criteria

Router policy beats or matches the best single-model policy in both walk-forward stability and post-cost expectancy.

---

### Phase 4: Trading Policy Backtest with Frictions

#### Goal

Promote from prediction metrics to actual trading-system validation.

#### Walk-forward protocol

| Parameter | Value | Rationale |
|---|---|---|
| Training window | Expanding, minimum 2 years | Sufficient history for regime coverage |
| Out-of-sample window | 3 months (1 quarter) | Long enough for statistical relevance |
| Purge gap | max(window_size, 40 bars) | Prevent contamination from temporal context |
| Number of folds | Determined by data length, minimum 8 | Sufficient for reliable WFE estimation |
| Re-optimization | Thresholds re-searched each fold; model weights fixed | Thresholds adapt; base models do not refit per fold |

#### Walk-Forward Efficiency requirement

WFE = out-of-sample annualized return / in-sample annualized return

**Require WFE > 0.50.** If WFE is below 0.50, the policy is likely overfit and should not proceed to paper trading.

#### Friction model

| Cost component | Specification |
|---|---|
| Spread: London/NY overlap | 1.0 pip |
| Spread: London or NY alone | 1.5 pip |
| Spread: Asia session | 2.5 pip |
| Spread: Off-hours | 3.0 pip |
| Slippage | 0.3 pip per trade |
| Commission | Per broker configuration; default $3.50 per round turn per 100k |
| Cooldown | 4 bars (20 minutes) minimum between signals in same direction |

#### Required backtest outputs

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

#### Proposed files

- New: `model_testing/ote_policy_backtest.py`
- New: `model_testing/ote_policy_metrics.py`
- New: `scripts/run_ote_policy_backtest.py`

#### Deliverables

- Full backtest report with all metrics above.
- Regime-level P&L attribution table.
- WFE calculation with per-fold breakdown.
- Equity curve with regime annotations.

#### Exit criteria

- Policy remains profitable after costs.
- Does not rely on one single year or regime.
- WFE > 0.50.
- No single trade accounts for more than 10% of total P&L.
- Positive expectancy in at least 60% of composite regime buckets.
- Maximum drawdown < 2× average monthly profit.

---

### Phase 5: Paper Trading Integration

#### Goal

Validate real-time behavior of the policy before live capital is at risk.

#### Implementation

Integrate the selected OTE policy into the execution stack in simulation mode. Use the existing `ict_app/execution/trading_orchestrator.py` and `ict_app/ensemble/ensemble_engine.py` only after policy selection is settled.

Run both champion and challenger models simultaneously during paper trading. For each signal, record:

- Timestamp, direction, all regime labels
- Champion model prediction, challenger model prediction
- Champion threshold decision, challenger threshold decision
- Regime router selection
- Actual market outcome

#### Daily monitoring dashboard

| Metric | Check |
|---|---|
| Signal count by direction and regime | Compare to historical daily average ± 2σ |
| Fill rate | Percentage of signals that would have been executed |
| Skipped trade count and reasons | Categorize by abstain rule that triggered |
| Divergence from offline expectations | Flag if signal distribution deviates from backtest |
| Champion vs. challenger hit rate | Running comparison over trailing 2-week window |
| Calibration quality (ECE) | Binned predicted probability vs. actual hit rate |

#### Alert thresholds

| Condition | Action |
|---|---|
| Signal count < 50% of historical daily average for 3 consecutive days | Investigate feature pipeline or regime distribution shift |
| Expected Calibration Error (ECE) > 0.10 | Flag for calibrator refit |
| Champion underperforms challenger by > 5% on event F0.5 over rolling 2-week window | Escalate for champion/challenger swap review |
| Any single-week drawdown exceeding worst backtest quarter | Halt and investigate |

#### Proposed files

- Update: `ict_app/execution/trading_orchestrator.py`
- Update: `ict_app/ensemble/ensemble_engine.py`
- New: `ict_app/ensemble/ote_policy_adapter.py`
- New: `scripts/run_ote_paper_trading.py`
- New: `scripts/run_ote_paper_trading_monitor.py`

#### Deliverables

- Paper trading mode using selected long and short OTE policies.
- Daily summary of signal count, fills, skipped trades, and divergence from offline expectations.
- Champion vs. challenger comparison log.

#### Exit criteria

- Paper trading for minimum 4 weeks.
- Stable signal behavior: signal count within 2σ of backtest expectations.
- No major mismatch with backtest assumptions.
- Calibration ECE < 0.10 throughout paper trading period.
- No single-week drawdown exceeding worst backtest quarter.

---

### Phase 6: Ensemble Only If Needed

#### Goal

Test stacking only if regime routing leaves meaningful performance on the table.

#### Decision criterion for proceeding

Compute the oracle router improvement: the performance gain if the router always selected the model that turned out to be correct ex-post. If the actual router captures within 90% of the oracle router's performance, do not proceed to ensemble. The router is already extracting most of the available signal, and ensembling will not justify its complexity cost.

Only proceed to ensemble if the gap between actual router and oracle router exceeds 10% relative.

#### Important constraint

Do not use the current `model_training/ensemble/stacking_model_train.py` as the direct path for OTE ensembling. It is built around an older workflow and objective.

#### Implementation if warranted

- Build OTE-specific ensembling from OOF predictions only.
- Treat regime labels as meta-features.
- Use a simple logistic regression as the meta-learner (not a complex stacking architecture). Regularize with L2, C chosen by cross-validation.
- Compare: best single model, regime router, stacked ensemble.

#### Proposed files

- New: `model_training/ensemble/ote_stacking.py`
- New: `model_training/ensemble/ote_meta_features.py`
- New: `scripts/run_ote_stacking_experiment.py`

#### Deliverables

- OTE-specific stacking experiment report.
- Promotion rule that requires stacked policy to beat router policy on stability, not just headline AP.

#### Exit criteria

Stacked model earns promotion only if it improves regime robustness and trading outcomes, not merely offline classification metrics.

---

### Known Pitfalls and Built-In Mitigations

These pitfalls are documented in published research on ML trading systems. The mitigations are embedded in the phases above, but collected here for reference.

#### Regime overfitting

**Pitfall:** Optimizing thresholds per regime bucket when many buckets have too few samples creates false confidence. The full cross-product of regime families (trend × vol × session × stress) produces 225 cells, most with insufficient data.

**Mitigation:** Use composite regimes (trend × vol only, max 15 cells). Require 50 positive events per bucket. Fall back to global threshold when data is insufficient. Use leave-one-year-out CV for threshold search.

#### Walk-forward lag on regime transitions

**Pitfall:** Walk-forward optimization adapts to regime changes reactively, not predictively. Performance deteriorates during transitions before parameters can adjust.

**Mitigation:** Phase 3 includes regime transition handling: conservative model selection, widened thresholds, and reduced position size during the 6-bar window after a regime label change.

#### Transaction costs swamping signal

**Pitfall:** Models that appear profitable before costs become unprofitable after realistic spreads and slippage. This is the most common failure mode for intraday forex systems.

**Mitigation:** Phase 4 uses session-aware spread modeling and requires post-cost expectancy. Phase 2 includes a minimum pip distance filter that rejects signals where expected move < 2× spread.

#### Calibration drift

**Pitfall:** Platt scaling or isotonic calibration fitted on development data degrades over time as market conditions shift.

**Mitigation:** Phase 5 monitors ECE weekly. If ECE > 0.10, the calibrator is refit on recent data without retraining the base model. Retraining cadence (see below) provides a structured schedule.

#### Single-regime profitability carrying the result

**Pitfall:** A system that appears profitable overall but derives all profits from one regime or one year. When that condition doesn't appear, the system bleeds money.

**Mitigation:** Phase 4 exit criteria require positive expectancy in at least 60% of composite regime buckets and profitable in at least 60% of quarterly periods.

#### Over-abstention collapsing trade count

**Pitfall:** Aggressive abstention from combined regime thresholds, confidence thresholds, and session filters reduces trade count to near zero, making performance statistically unreliable.

**Mitigation:** Phase 2 threshold search requires minimum 3 signals per week per direction. Phase 5 monitors signal count against historical baselines.

#### Single-trade concentration

**Pitfall:** A small number of trades (less than 5%) accounting for more than 30% of total profit. The system appears profitable but depends on outlier events that may not recur.

**Mitigation:** Phase 4 requires no single trade to account for more than 10% of total P&L, and computes the Herfindahl index of trade-level P&L.

---

### Retraining and Monitoring Cadence

#### Monthly monitoring

- Track calibration quality (ECE, reliability diagram).
- Track signal count distribution against historical baseline.
- Track regime distribution for structural shifts.
- Track champion vs. challenger performance.

#### Quarterly retraining trigger

Retrain if any of:

- Calibration ECE exceeds 0.10 for a full quarter.
- Signal quality (event F0.5) drops below backtest expectations for a full quarter.
- Regime distribution shifts significantly (new regime appears that was absent in training data).

#### Emergency retraining trigger

Retrain immediately if:

- Out-of-sample event F0.5 drops below 0.60 for two consecutive weeks during live trading.
- Maximum drawdown during live trading exceeds the worst backtest quarter by more than 50%.

#### Retraining procedure

1. Retrain base models on expanded dataset (include recent data).
2. Refit calibrator.
3. Re-run regime slice report (Phase 1).
4. Re-run threshold search (Phase 2).
5. Re-evaluate router rules (Phase 3).
6. Re-run backtest (Phase 4).
7. Promote new models only if they pass all promotion rules.

---

### Promotion Rules

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

### Immediate Ticket Backlog

These are the next concrete tickets to implement in order.

#### Ticket 1: OTE model registry

- Add `models/ote_model_registry.json` with the schema specified in Phase 0.
- Add `models/ote_model_registry.schema.json` for validation.
- Add `models/ote_registry_loader.py` that loads registry entries and replaces `best_models.py` lookups.
- Record long champion and short champion candidates.
- **Acceptance criteria:** All model loading in training and inference code reads from registry. No hard-coded model paths remain outside the registry.

#### Ticket 2: Joinable predictions

- Add `source_row_idx` column to OOF and test prediction exports in the training pipeline.
- Add `model_testing/ote_prediction_joiner.py` that joins predictions back to timestamped source data.
- **Acceptance criteria:** Can reconstruct timestamped predictions for any trained model without re-running training.

#### Ticket 3: Deterministic regime labeler and slice report

- Implement `model_testing/ote_regime_labeler.py` with the indicator-threshold rules from the Regime Taxonomy section.
- Implement `model_testing/ote_regime_slices.py` that generates the per-model performance report.
- Add `scripts/run_ote_regime_slice_report.py` as the entry point.
- Optionally add rolling Hurst exponent (30-bar and 100-bar windows) to the feature set.
- **Acceptance criteria:** Can answer "which model wins for long and short inside each composite regime bucket" with 95% bootstrap confidence intervals.

#### Ticket 4: Threshold policy search with abstain rules

- Implement `model_testing/ote_threshold_policy.py` with regime-conditioned threshold search.
- Implement `model_testing/ote_abstain_policy.py` with hard and soft abstain rules.
- Add `scripts/run_ote_threshold_policy_search.py` as the entry point.
- **Acceptance criteria:** Regime-aware policy beats global threshold on event F0.5 AND post-cost expectancy. Trade count remains at or above 3 per week per direction.

#### Ticket 5: Router prototype

- Implement `model_training/ensemble/ote_regime_router.py` with deterministic routing rules.
- Populate routing rules from Ticket 3 evidence.
- Evaluate router policy against single-model baselines using walk-forward protocol.
- **Acceptance criteria:** Router WFE > 0.50. Router beats or matches best single-model policy on post-cost expectancy.

---

### Recommended First Sprint

If the goal is the highest-value next sprint, it should be:

1. Add OTE model registry (Ticket 1).
2. Add joinable prediction exports (Ticket 2).
3. Build deterministic regime labeler and slice report (Ticket 3).

That sprint answers the most important unresolved question in the current repo:

> Is the short-side production path best served by XGBoost, TCN, or a regime router between them?

---

### Explicit Non-Goals for Right Now

1. Building a large multi-model weighted ensemble before regime analysis is complete.
2. Training a separate ML regime classifier model (use deterministic labeling instead).
3. Wiring all ICT, confluence, and OTE components together before the OTE policy itself is stable.
4. Optimizing for maximum trade count at the expense of regime robustness.
5. Moving to live capital before paper-trading evidence is collected.
6. Averaging model probabilities across models (use model selection by regime instead).

---

### Appendix: Feature Mapping for Regime Labels

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

<a id="model_retrain_plan_1min"></a>

## 1-Minute OTE Training Plan

*Merged 2026-10-09 from `docs/Model_Retrain_Plan_1min.md`.*

### Goal

Train a new 1-minute OTE model stack on `data/currency_data/eurusd-1m.csv` using the same offline sequence we trust on 5-minute data:

1. labeling
2. feature generation
3. preprocessing
4. backend attribution
5. focused model training
6. post-training thresholding and backtests

The plan below is intentionally anchored to what already worked in this repo rather than reopening the full search space.

---

### Repo Reality

The current maintained OTE path is still 5-minute-first:

- `data/labeling/labeling_engine.py` loads a 5-minute base series and resamples to `30m` and `1h`
- `data/labeling/reversal_labeling_engine.py` contains several explicit 5-minute assumptions in bar-based settings
- `scripts/run_ote_training.ps1` defaults to `data/prepared/eurusd_5min_ote_full`
- `model_training/ote_training/#ote_training_workflow_report` explicitly says the implemented OTE path currently requires 5-minute OHLCV as its base input

So this is not a simple dataset swap. We need a controlled 1-minute migration.

---

### What Past Runs Say To Copy

Use the post-training winners, not raw CV rank, as the guide rail.

#### Highest-value families

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

#### Families to de-prioritize

- Breakout models had excellent training leaderboard scores but poor post-training conversion.
- Do not spend the first 1-minute cycle on breakout retrains.
- Reversal and union OTE should be the first 1-minute production candidates.

---

### Dataset Scope

#### Initial comparison window

For the first 1-minute migration, keep the historical window aligned to the current validated 5-minute workflow so results are comparable.

- Recommended initial window: `2019-01-01` through `2026-03-24`
- Do not use the full `2002-2026` 1-minute file for the first wave
- Expand history only after the 1-minute pipeline is stable and directly comparable to the 5-minute baseline

#### Current 5-minute baseline we are trying to beat

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

### Phase 0: Control Baseline And Code Audit

#### Objective

Create a safe comparison point before true 1-minute labeling.

#### Tasks

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

#### Exit criteria

- We can reproduce the existing 5-minute behavior from the 1-minute source after resampling.
- We have a file-by-file list of what must become timeframe-neutral.

---

### Phase 1: 1-Minute Labeling Migration

#### Objective

Make the labeling stack treat 1-minute bars as the base series without destroying the time meaning of the existing thresholds.

#### Design rule

Keep the first 1-minute label pass time-equivalent to the 5-minute defaults. Do not leave bar-count settings unchanged.

#### Starting parameter conversion

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

#### Structural context

Keep higher-timeframe structure in the first 1-minute version:

- base series: `1m`
- structural ATR: still `1h`
- HTF confluence: still `30m` and `1h`

This preserves the repo's current idea that local execution timing can get finer while structural thresholds stay higher-timeframe.

#### Required code changes

1. Generalize `data/labeling/labeling_engine.py` naming and defaults from `5m` to base-timeframe-neutral.
2. Generalize `data/labeling/reversal_labeling_engine.py` so printed messages, defaults, and source tags are not hardcoded to `5m`.
3. Do the same for continuation and breakout labeling engines.
4. Add 1-minute output names:
   - `data/labeling/labeled_data/eurusd_1min_ote_labels_full.csv`
   - `data/labeling/labeled_data/eurusd_1min_ote_swings_full.csv`

#### Labeling acceptance checks

Do not move forward if any of these fail:

- Entry labels become extremely dense relative to the 5-minute baseline
- Ambiguous rows collapse toward zero
- Swing counts explode without a corresponding quality increase
- Triple-barrier timeouts dominate because horizons were not rescaled correctly

---

### Phase 2: 1-Minute Feature Generation

#### Objective

Build a narrower first-pass 1-minute feature table instead of repeating the full widest 5-minute feature explosion.

#### First-pass recipe strategy

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

#### What not to do in wave 1

- Do not start with `--all-strategies`
- Do not start with every historical strategy script mixed in
- Do not optimize for maximum feature count

#### Output targets

- `data/features/eurusd_1min_ote_reversal_core.csv`
- `data/features/eurusd_1min_ote_union_core.csv`

#### Feature-build checks

- Validate timezone lineage and metadata sidecars
- Check higher-timeframe features are mapped from completed bars only
- Compare feature counts against the 5-minute baseline and trim before training if the 1-minute set becomes much wider without clear value

---

### Phase 3: Preprocessing

#### Objective

Create target-specific prepared roots for 1-minute reversal and union families.

#### First-pass settings

- `scaler none`
- time-based splits only
- `corr-threshold 0.98`
- `similarity-threshold 0.995`
- `max-analysis-rows 100000`

#### Prepared roots

- `data/prepared/eurusd_1min_ote_reversal_core`
- `data/prepared/eurusd_1min_ote_union_core`

#### Targets to prepare first

1. `long_reversal`
2. `short_reversal`
3. `long_ote`
4. `short_ote`

#### Preprocessing stop rules

Stop and retune labels or features before training if:

- reversal positive counts are too low for stable CV
- OTE positives jump mainly because negatives became ambiguous or duplicated
- prepared reports show weak readiness or severe class skew changes

---

### Phase 4: Backend Attribution

#### Objective

Use backend-specific ranking early so 1-minute training is feature-disciplined from the start.

#### Backends for wave 1

- `xgboost`
- `tcn`

Skip `lstm` in the first 1-minute wave.

#### Attribution settings

##### Reversal prepared root

- `max-features 96`
- `attribution-max-rows 50000`
- `attribution-floor-fraction 0.08`
- `attribution-cumulative-importance 0.95`

##### Union OTE prepared root

- `max-features 128`
- `attribution-max-rows 75000`
- `attribution-floor-fraction 0.08`
- `attribution-cumulative-importance 0.95`

#### Rule

Use the backend-aware merged rankings in training. Do not fall back to generic `feature_importance.csv` unless attribution fails.

---

### Phase 5: Focused Training Waves

#### Time-scaling rule for trainer geometry

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

#### Wave 1: Reversal family first

Reversal is the best 1-minute candidate because it already converts well post-training and runs at lower turnover than the OTE union models.

##### 1. `short_reversal_xgb_1m_v1`

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

##### 2. `long_reversal_tcn_1m_v1`

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

#### Wave 2: Union OTE family

Only move here after reversal labeling and training look stable.

##### 3. `short_ote_union_tcn_1m_v1`

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

##### 4. `long_ote_union_tcn_1m_v1`

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

#### Wave 3: Only after wave 1 and 2 succeed

- continuation-entry specialists
- breakout specialists
- meta-label family refreshes

Breakout should stay out of the first 1-minute production cycle unless it proves it can convert post-training, not just in CV.

---

### Phase 6: Post-Training Evaluation

Use the exact same downstream stack already trusted in this repo:

1. regime slice report
2. threshold policy search
3. walk-forward backtest
4. acceptance-gate review

#### Promotion rule

Do not promote a 1-minute model based on CV AP alone.

Promote only when it wins on the same post-training evidence surface used by the current saved leaderboards:

- profitable after costs
- acceptable WFE
- profitable quarter share
- positive composite expectancy share
- largest-trade concentration limits
- drawdown constraints

---

### Command Skeletons

These are the intended artifact paths after the 1-minute refactor is in place.

#### Labeling

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

#### Feature generation

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

#### Preprocessing

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

#### Backend attribution

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

#### Training

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

### Recommended Execution Order

1. Build the 1m-to-5m control baseline.
2. Refactor labeling to support true 1-minute base bars.
3. Train `short_reversal_xgb_1m_v1`.
4. Train `long_reversal_tcn_1m_v1`.
5. If reversal wave passes downstream checks, train `short_ote_union_tcn_1m_v1`.
6. Then train `long_ote_union_tcn_1m_v1`.
7. Only after that, revisit continuation specialists and breakout.

---

### Bottom Line

The first 1-minute cycle should be a disciplined migration, not a broad search. The best prior evidence says:

- start with reversal and union OTE families
- copy the winning turnover-aware objective geometry
- scale all time-based bars from 5-minute meaning into 1-minute meaning
- keep backend attribution in the loop
- judge success on post-training robustness, not raw CV AP


<a id="policy_layer_and_architecture_research"></a>

## The Policy Layer Is the Bottleneck, Not the Backbone

*Merged 2026-10-09 from `docs/POLICY_LAYER_AND_ARCHITECTURE_RESEARCH.md`.*

### A Diagnostic of the Abstain Policy, and a Prioritized Architecture Research Program

**Author:** Quantitative Research & ML Systems
**Date:** 2026-07-12
**Status:** Research design paper — diagnostic + experiment program. No implementation yet.
**Repo target:** `model_testing/ote_abstain_policy.py`, `model_testing/ote_threshold_policy.py`, `model_training/ote_training/*`
**Companion papers:** `EURUSD_OTE_Framework` (2026), `FRVP_ES_primary_6E_variant_design` (2026-06-21), `ICT_ES_META_LABELING_DESIGN`

---

### Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Part I — The Abstain Policy Diagnostic](#2-part-i--the-abstain-policy-diagnostic)
   - 2.1 [The Smoking Gun](#21-the-smoking-gun)
   - 2.2 [Root Cause: The Objective Function Is Wrong](#22-root-cause-the-objective-function-is-wrong)
   - 2.3 [Rule-by-Rule Failure Analysis](#23-rule-by-rule-failure-analysis)
   - 2.4 [Structural Defects in the Policy Layer](#24-structural-defects-in-the-policy-layer)
   - 2.5 [The Diagnostic Protocol](#25-the-diagnostic-protocol)
   - 2.6 [The Redesign: From Veto to Sizing](#26-the-redesign-from-veto-to-sizing)
   - 2.7 [Acceptance Criteria and the Null Result](#27-acceptance-criteria-and-the-null-result)
3. [Part II — Model Architecture Research Program](#3-part-ii--model-architecture-research-program)
   - 3.1 [What the Literature Says at Our Scale](#31-what-the-literature-says-at-our-scale)
   - 3.2 [The Meta-Labeling Architecture Literature](#32-the-meta-labeling-architecture-literature)
   - 3.3 [Tier 1 — Cheap, High-Probability Wins](#33-tier-1--cheap-high-probability-wins)
   - 3.4 [Tier 2 — Sequence Architectures With Structural Fit](#34-tier-2--sequence-architectures-with-structural-fit)
   - 3.5 [Tier 3 — Pretraining, the Real Lever on ~2k Labels](#35-tier-3--pretraining-the-real-lever-on-2k-labels)
   - 3.6 [Deprioritized and Rejected](#36-deprioritized-and-rejected)
4. [Part III — Formulation Changes That Outrank Backbone Changes](#4-part-iii--formulation-changes-that-outrank-backbone-changes)
5. [Part IV — Experiment Program](#5-part-iv--experiment-program)
6. [Part V — Evaluation Discipline](#6-part-v--evaluation-discipline)
7. [Open Questions and Risks](#7-open-questions-and-risks)
8. [References](#8-references)

---

### 1. Executive Summary

This paper does two things.

**Part I** diagnoses the hard-abstain policy, which the `EURUSD_OTE_Framework` paper already flags as "mis-tuned" and names as "the largest available lever" for future work, but does not dissect. The diagnosis: the abstain layer is not merely mis-tuned, it is **structurally mis-specified**. It optimizes precision when the economics require expectancy; it applies binary vetoes where it should modulate size; its individual rules are unvalidated magic numbers; and — critically — the aggregate evidence shows it is removing *above-average-expectancy* trades. Under hard abstain, throughput fell 65% while net PnL fell 69%. That asymmetry is a mathematical proof that the vetoed trades were, on average, better than the retained ones. The layer is anti-selective.

**Part II** surveys the model-architecture landscape and proposes a prioritized experiment program. The conclusion there is deflationary in a useful way: at our event counts (~1,500–5,000 per family per side, ~5% positive rate), the published evidence does not support large gains from swapping backbones. The literature and our own leaderboard both show gradient boosting and small causal-convolutional models sitting near the ceiling of what the data supports. The architectures worth testing are those that either (a) attack the **sample-size constraint** (pretraining, tabular foundation models), (b) exploit **structure we currently discard** (the volume profile's shape), or (c) buy **ensemble diversity**.

**The load-bearing claim of this paper:** fixing the policy layer is a higher-expected-value use of engineering time than any architecture on the list, and it is roughly a week of work versus months. Part I should be executed before Part II is started.

---

## 2. Part I — The Abstain Policy Diagnostic

### 2.1 The Smoking Gun

From the TCN focus cohort in the OTE framework evaluation, comparing the global-threshold policy against its hard-abstain-augmented counterpart on `long_ote_tcn_v2`:

| Metric | Threshold only | + Hard abstain | Change |
|---|---|---|---|
| Event F0.5 | 0.382 | 0.846 | **+121%** |
| Trades / week | 11.8 | 4.1 | **−65%** |
| Net PnL (pips) | 14,573 | 4,469 | **−69%** |

The precision statistic more than doubled. The profit collapsed by more than two thirds. That much is already recorded in the framework paper, and the conclusion drawn there was that abstain "acts as a blunt trade suppressor."

But the paper stops one arithmetic step short of the real finding. Compare the two percentage drops:

- Trades removed: **65%**
- PnL removed: **69%**

If the abstain layer were removing *random* trades, PnL would fall by the same 65%. If it were removing *bad* trades — which is its entire purpose — PnL would fall by **less** than 65%, and could even rise. Instead PnL falls by **more** than trade count.

> **The abstained trades had a higher mean expectancy than the trades that survived. The abstain policy is not a filter. It is an anti-filter.**

Quantitatively: mean expectancy of retained trades ≈ 4,469 / 4.1 ≈ **1,090 pips per trade-week-unit**, versus a pre-abstain blend of 14,573 / 11.8 ≈ **1,235**. The removed cohort therefore carried a mean expectancy of roughly (14,573 − 4,469) / (11.8 − 4.1) ≈ **1,312** — about **20% higher** than the trades the policy chose to keep.

This is not a tuning problem. A tuning problem would show precision up and PnL down *proportionally less*, and would be fixed by loosening thresholds. This shows precision up and PnL down *disproportionately more*, which means the selection axis itself is pointing the wrong way.

**Why this happens is not mysterious.** The label geometry is triple-barrier with a target reward:risk of roughly 2:1. At 2:1, break-even hit rate is 33%. A candidate with a 45% hit rate is *comfortably profitable* and *precision-destructive*. Any objective that rewards precision will throw it away. The abstain policy is doing exactly what it was asked to do; it was asked for the wrong thing.

### 2.2 Root Cause: The Objective Function Is Wrong

The threshold search selects on:

```
selection_score = 0.60 · event_F0.5 + 0.40 · normalized_post_cost_expectancy
```

with the F-beta parameter set to **β = 0.55**, which the design explicitly describes as weighting precision over recall.

The abstain layer then sits **downstream** of that already precision-weighted threshold and applies *further* precision-improving vetoes.

So the architecture is:

```
model probability
      ↓
threshold selected on a 60%-precision-weighted objective
      ↓
abstain rules that improve precision
      ↓
trade
```

**At no point in this chain does anything maximize post-cost profit, Sharpe, or expectancy as a primary objective.** Expectancy appears once, at 40% weight, inside a score that is dominated by an F-measure. The abstain layer has no objective at all — it is a hand-written rule stack with no fitted parameters and no selection criterion.

The framework paper's own discussion section reaches the right verdict without following it to its conclusion: *"This is a cautionary tale about optimizing the wrong objective — precision is not P&L."* Correct. The fix is not to re-tune the rules. The fix is to delete the objective.

#### 2.2.1 Why precision was chosen, and why the reasoning fails

The motivation for a precision-weighted objective is intuitive and, in a different setting, sound: a signal system that fires constantly and is right slightly more than half the time is operationally miserable and psychologically unusable. Event F0.5 was chosen to reflect "how a trader actually consumes signals."

But this conflates two distinct desiderata:

1. **Signal parsimony** — we want few, meaningful signals rather than a firehose.
2. **Signal accuracy** — we want the signals we emit to be right.

The labeling engine and event de-duplication already deliver (1). Structural-ATR-scaled swing detection, quality scoring, and event-level evaluation *are* the parsimony mechanism. Layering a precision objective on top to deliver parsimony a second time is double-counting — and it pays for that redundant parsimony in expectancy.

### 2.3 Rule-by-Rule Failure Analysis

The `HardAbstainConfig` / `apply_abstain_policy()` layer applies five veto families in first-match precedence order. Each is analyzed below.

#### Rule 1 — `abstain_high_stress` (blocks all bars where `stress_regime == 'high'`)

**Stated rationale:** during abnormal volatility spikes, spread costs and slippage are likely to exceed the expected move.

**Why it is probably the single most destructive rule:** it contradicts the regime-slice evidence in the same framework. The per-regime analysis found that model skill is **highest in trend-aligned, high-volatility, London/overlap conditions** and **collapses in low-volatility ranges where structure is faint**. The framework's own summary of the regime table: *"the models are skilful precisely where the labeling philosophy predicts they should be — in volatile, trend-aligned, liquid-session conditions."*

Stress regime and volatility regime are not identical, but they are strongly correlated by construction. Blocking high-stress bars therefore blocks a large fraction of exactly the regime where the edge is concentrated. The rule is not filtering noise; it is filtering signal.

**The rationale is also an assumption, not a measurement.** "Spread costs are likely to exceed the expected move" is a testable proposition, and the cost model already has session-dependent spreads to test it with. It has never been tested. It should be replaced by an explicit EV computation (Rule 2, done correctly), which subsumes it: if high-stress spreads really do eat the move, the EV filter will abstain automatically, on evidence rather than on prior.

**Recommendation:** delete. Let the EV filter decide.

#### Rule 2 — Expected-move filter

**Current implementation:** abstain if `probability × regime_expected_move < 2.0 × session_spread`.

Three independent defects:

**(a) It is a gross-move filter, not an expected-value filter.** It never subtracts the loss branch. It asks "is the upside big enough?" and never asks "is the upside big enough *net of the downside*?" The economically correct test, using quantities all known at signal time from the triple-barrier configuration:

```
EV = p_cal · TP_distance
   − (1 − p_cal) · SL_distance
   − entry_spread − exit_spread
   − slippage − commission

abstain iff EV ≤ margin
```

This is the *only* principled veto in the entire system, and it is not the one being applied.

**(b) `regime_expected_move` is the wrong conditioning set.** It is an unconditional average move for the regime bucket. But we are not asking "how far does price typically travel in a strong_up_high regime?" — we are asking "how far does price travel *given that FRVP Setup 3 fired at the VAH of the prior-RTH profile with this displacement magnitude*?" Those are different distributions. The regime mean is a crude prior that ignores every setup-specific feature we spent the entire feature-engineering effort constructing.

**(c) Double-counting of probability, and an unvalidated constant.** `probability × regime_expected_move` multiplies by a probability that has *already* been used once, at the threshold gate. And the `2.0×` multiplier on session spread is a magic number. No sweep of it appears anywhere in the artifacts. It could be 1.2 or 3.5 and nobody would know.

**Recommendation:** replace entirely with the EV expression above. Estimate `E[TP | hit]` and `E[SL | stop]` from the realized barrier outcomes in the training fold, conditioned on setup family and volatility regime, not from a regime-level mean move.

#### Rule 3 — Cooldown (`cooldown_bars = 4`, unconditional)

**Stated rationale:** prevent burst trading around a single structural event.

**Why it is solving the wrong problem:** the backtest assumes a **fixed 120-bar exit**. The real exposure risk is therefore *concurrent overlapping positions* held across a 10-hour window — not two signals firing 20 minutes apart. A 4-bar cooldown is far too short to control concurrency (it blocks 3% of the holding period) and just deletes the second-best signal from a legitimate cluster.

Worse, signal clusters around a structural event are not necessarily redundant. In the FRVP/ICT formulation, a sweep followed by a displacement followed by a pullback into the resulting FVG is a *sequence of distinct, individually-labeled setups* around one structural event, and the later ones are often the higher-quality entries. A cooldown keyed to the first accepted signal systematically prefers the earliest, least-confirmed member of each cluster.

**Recommendation:** delete the cooldown. Replace with (i) an explicit **concurrency cap** (max N simultaneous open positions per side), and (ii) **overlap-aware position sizing** — if a new signal arrives while a correlated position is open, scale the new position down rather than vetoing it. If de-duplication within a structural event is genuinely wanted, do it at the *event* level in the labeler (where the machinery already exists), not with a blind bar counter at the policy layer.

#### Rule 4 — Probability-quantile filter

**Current implementation:** abstain when probability falls below a configurable quantile of the candidate distribution.

**Why it is not a decision rule:** it is a *quota*. It removes a fixed fraction of candidates regardless of whether those candidates are good. In a month where every signal is excellent, it deletes the bottom decile of excellent signals. In a month where every signal is marginal, it retains 90% of marginal signals. It has the sign of a filter and the behavior of a rationing scheme.

It is also **redundant** with the calibrated absolute threshold (which is the correct way to express "only take signals above quality X"), and — depending on the window over which the quantile is computed — potentially **leaky**. If the quantile is computed over the full candidate distribution including the test period, the policy has seen the test set's difficulty level. This needs an audit of `apply_abstain_policy`'s quantile window.

**Recommendation:** delete. The absolute calibrated threshold already does this job, correctly, and without look-ahead risk.

#### Rule 5 — Off-hours / explicit session and composite-regime blocks

**Assessment: probably the only defensible rule in the stack**, and even it should be earned rather than assumed. Off-hours spreads (3.0 pips default) are genuinely punitive and the block is cheap. But note that even here, the EV filter subsumes it: if a 3.0-pip spread kills the trade's EV, the EV filter abstains without needing a hard-coded session block. Keep it as a cheap short-circuit; do not rely on it as the mechanism.

The explicit `abstain_composite_regimes` / `abstain_composite_session_pairs` tuples are a different matter — these are hand-curated block lists. Each entry needs a counterfactual PnL justification (Section 2.5) or it should be removed.

### 2.4 Structural Defects in the Policy Layer

Beyond the individual rules, four defects are architectural.

**(a) Binary veto instead of size modulation.** A hard veto throws away all option value in a marginal trade. Given that the FRVP design already specifies fractional-Kelly sizing scaled between `P_min` and `P_max`, and the framework already computes calibrated probabilities via Platt scaling on out-of-fold predictions, we have every ingredient needed to convert vetoes into size reductions and do not use them. A trade with EV slightly above zero should be taken at 0.2× size, not skipped.

**(b) First-match precedence with no attribution.** `apply_abstain_policy()` records only the *first* matching reason in `policy_abstain_reason`. Rules later in the precedence order never get credit or blame for anything a prior rule already caught. With five compounding vetoes, we have no idea which one is doing the damage — only that collectively they remove 65% of throughput.

**(c) The rules are unfitted.** Every threshold and constant in the abstain layer (`2.0×` spread, `4` cooldown bars, the block lists) is hand-set. None is the output of a search. None has an in-fold/out-of-fold split. The rest of the pipeline is rigorous about purged walk-forward validation, embargo windows, and in-fold scaler fitting — and then the final gate before a trade is a set of hand-typed constants.

**(d) Sequential compounding against a threshold that already priced in regime quality.** The regime-conditional threshold map already raises the bar in adverse regimes (e.g. `strong_down_high` tightened to 0.80; `ranging_low` at 0.70 versus `strong_up_low` at 0.55 for longs). The abstain layer then blocks many of those same adverse-regime signals *again*. The same information is being used to reject the same trades twice. That is precisely the mechanism by which a mildly conservative system becomes a crippled one.

### 2.5 The Diagnostic Protocol

Before any redesign, run this. It is roughly one day of work, and it converts the entire preceding argument from inference into measurement.

**The counterfactual PnL attribution table.**

`policy_abstain_reason` is already recorded per candidate. For every abstained candidate in the walk-forward backtest, compute the realized post-cost PnL **as if the trade had been taken** — the outcome is already known from the triple-barrier evaluation, so no new simulation is required. Then attribute by reason.

| Abstain reason | Signals blocked | Counterfactual net PnL (pips) | Mean expectancy | Hit rate | Verdict |
|---|---|---|---|---|---|
| `high_stress` | ? | ? | ? | ? | ? |
| `off_hours` | ? | ? | ? | ? | ? |
| `blocked_composite` | ? | ? | ? | ? | ? |
| `expected_move` | ? | ? | ? | ? | ? |
| `prob_quantile` | ? | ? | ? | ? | ? |
| `cooldown` | ? | ? | ? | ? | ? |

**Decision rule:** any reason with **positive** counterfactual net PnL is destroying money and is deleted immediately, without further debate.

**Predicted result, stated in advance so it can be falsified:** `high_stress` and `cooldown` both come back strongly positive. `prob_quantile` comes back mildly positive. `expected_move` is ambiguous — it will block some genuinely uneconomic trades and some good ones, because it is directionally reasonable and quantitatively wrong. `off_hours` comes back near zero or mildly negative (i.e. correctly blocking).

**Second diagnostic — greedy ablation.** Having deleted the clearly-positive rules, evaluate each remaining rule's *incremental* contribution to net Sharpe across the 49 walk-forward folds, adding one rule at a time in order of measured value. Any rule that does not improve out-of-fold Sharpe by a margin exceeding fold-to-fold noise does not ship. Note that this must be done **out-of-fold** — the abstain rules were never subjected to purged validation, and a rule tuned on the same data it is evaluated on will always look good.

**Third diagnostic — the precision/expectancy decoupling plot.** For the retained cohort, plot realized expectancy against calibrated probability decile. If the relationship is monotone (as Platt calibration and the existing confidence-quintile analysis suggest it should be), then a *threshold* is the correct and sufficient control, and the entire hard-abstain rule stack is redundant machinery. If it is non-monotone — if, say, the top decile has *lower* expectancy because it corresponds to trades entered late in an already-extended move — that is a genuinely important finding and the abstain layer should be redesigned to target the non-monotone region specifically.

### 2.6 The Redesign: From Veto to Sizing

The replacement architecture:

```
model probability p
      ↓
Platt/isotonic calibration (already exists)
      ↓
EV = p·TP − (1−p)·SL − spread − slippage − commission
      ↓
abstain iff EV ≤ margin          ← the ONLY veto
      ↓
size = f(EV, variance)           ← fractional Kelly, already specced
      ↓
concurrency cap / exposure limit  ← replaces cooldown
      ↓
trade
```

**Design principles:**

1. **One veto, and it is economic.** `EV ≤ 0` is the only condition under which not trading is provably correct. Everything else is a sizing question. The `margin` term absorbs estimation error in `p` and should itself be fitted, not assumed.

2. **Size is continuous.** Marginal trades get small size. This preserves the right tail of the payoff distribution — which is precisely what the current precision-optimizing veto destroys.

3. **The policy objective becomes post-cost Sharpe (or expectancy with a drawdown penalty), not F0.5.** This is the single most important change in the paper. The threshold search's `0.60 · event_F0.5 + 0.40 · expectancy` should become something closer to a pure risk-adjusted return objective. Event F0.5 remains valuable as a **diagnostic** — it tells us about signal quality — but it must stop being an **objective**.

4. **Every constant is fitted, in-fold, under the existing purged walk-forward geometry.** The abstain/sizing layer gets the same validation discipline as the model.

5. **Concurrency, not cooldown.** Exposure is managed by capping simultaneous positions and scaling overlapping entries, not by blinding the system for 20 minutes.

#### 2.6.1 The learned abstain (the better version of Rule 2)

The EV expression above requires `E[TP | hit]` and `E[SL | stop]`. Rather than estimating those from regime averages, **learn them**. Train a second-stage regressor on the **realized R-multiple** of each event — not the binary meta-label — and abstain when predicted R falls below the cost hurdle.

This is not an extra model. It is an **auxiliary head on the model we already train** (see Section 4.1). The MFE/MAE head *is* the abstain model. This unifies Parts I and II of this paper: the highest-value architectural change and the highest-value policy change are the same change.

### 2.7 Acceptance Criteria and the Null Result

The redesigned policy layer ships only if, across the full purged walk-forward backtest (49 folds), it beats the **no-abstain, threshold-only, Kelly-sized** baseline on:

- Net post-cost PnL, **and**
- Monthly Sharpe, **and**
- Maximum drawdown (not worse by more than a stated tolerance), **and**
- Fold-level stability (positive in ≥ 70% of folds)

**And it must be prepared to fail.** It is entirely possible — arguably the modal outcome — that the correct answer is:

> *Take every signal above the calibrated threshold. Size it by fractional Kelly. Cap concurrency. Abstain only when EV ≤ 0. There is no additional rule that beats this.*

That is a legitimate, valuable, publishable finding, and it should be accepted rather than tuned around. The failure mode to guard against is the one the current abstain policy already fell into: continuing to add rules until the metric you happen to be looking at improves.

**Present live status, for the record:** no abstain policy is written into any entry in the active production registry. The mis-tuned layer is therefore not currently costing money in live — it is an *unrealized lever*, not an active wound. This lowers the urgency but not the value.

---

## 3. Part II — Model Architecture Research Program

### 3.1 What the Literature Says at Our Scale

The operating regime is: **~1,500–5,000 events per family per side, ~5% positive rate, 5-minute bars, engineered tabular-plus-sequence features, purged walk-forward validation.** This is the exact regime in which the published evidence is most hostile to architectural ambition.

**The tabular/GBDT ceiling is real and well-documented:**

- A study of deep incremental learning on **financial temporal tabular datasets with distribution shifts** found that increasing MLP complexity does not improve performance, that the best result came from a *two-layer* MLP providing the minimum non-linearity needed to avoid degenerating into ridge regression, that **XGBoost beat MLPs across a wide hyperparameter range**, and that modern deep tabular architectures such as TabNet do not reliably beat plain MLPs.
- A 2025 comparative evaluation across XGBoost, LightGBM, N-BEATS, N-HiTS, and the Temporal Fusion Transformer on sparse, intermittent data concluded that **localized tree-based methods achieved superior performance** and that model selection should prioritize *alignment with problem characteristics over architectural sophistication*.
- A practitioner study running **13,500 model fits** across XGBoost, LSTM, self-attention variants, and LSTM→XGBoost hybrids on crypto found XGBoost won **consistently across every asset and every fold**, with the decisive factor being small walk-forward training windows (~2,000 candles). Their observation is directly applicable: at ~2,000 samples, even a 35,000-parameter transformer is already prone to overfitting.
- A 2026 study on **MNQ futures** — same instrument class, same 5-minute bar frequency, ~944 trading days — explicitly compared gradient boosting on engineered features against an LSTM on raw 5-minute return sequences, motivated by the Kronos foundation-model results, and asked whether sequential structure even exists at single-instrument scale over four years. This is the closest published analogue to our problem and should be read before any sequence-model work is authorized.

**Our own leaderboard agrees.** The training-phase leaderboard shows TCN dominating on CV AP, but the post-training, cost-aware, walk-forward Sharpe leaderboard shows **`short_reversal_xgb_v2` at monthly Sharpe 3.61 / profit factor 2.74**, ahead of `long_reversal_tcn_v3` — and shows Breakout TCN with a spectacular CV AP of 0.969 and a post-training Sharpe of 0.81. Also on record: version-to-version retraining is non-monotonic and occasionally destructive (long LSTM v2 regressed −0.311 test AP), and headline ROC-AUC of ~0.96 badly overstates usefulness at a 5.6% base rate.

**The honest prior, therefore:**

> Architecture is not currently the binding constraint. Sample size, label quality, and the policy layer are. Any architecture proposal must justify itself against that prior, not against a leaderboard of CV AP.

The architectures below are shortlisted precisely because each attacks the actual constraint — sample size, discarded structure, or ensemble diversity — rather than simply being a fancier function approximator.

### 3.2 The Meta-Labeling Architecture Literature

There is a body of work that is *specifically about the architecture of the secondary model in a meta-labeling system*, and this repo — which is a meta-labeling system — does not currently draw on it.

**Meta-Labeling: Theory and Framework** (Joubert, *Journal of Financial Data Science*, 2022). Consolidates the technique into a single framework, explains the relationship between binary-classification metrics and strategy performance, and — critically for Part I of this paper — deconstructs meta-labeling into three components, using a controlled experiment to show how each contributes to strategy metrics and what feature types belong in the model specification. The metrics-to-performance mapping is the exact material needed to justify replacing event F0.5 as an objective.

**Meta-Labeling: Architecture** (Meyer, Joubert & Alfeus, *JFDS*, 2022). The central reference for this section. It establishes several *heterogeneous architectures* for the secondary model, organized along two axes:
- **Feature-driven architectures** — exploiting how information in the data is structured and how the selected models consume that information.
- **Strategy-driven architectures** — explicitly incorporating characteristics unique to the underlying primary strategy.

It also introduces **inverse meta-labeling**, a technique for improving the *quantity and quality of the side forecasts* produced by the primary layer.

The strategy-driven framing is directly load-bearing for us. Our `setup_type`, `setup_family`, `anchor_level`, `sweep_type`, `session_phase`, and `htf_context` are **strategy artifacts**, not generic market features. The architecture paper's argument is that these deserve *dedicated input pathways* rather than being flattened into a single concatenated feature vector alongside 128 lagged technical indicators. Our current pipeline does the flattening.

**Ensemble Meta-Labeling** (Thumm, Barucca & Joubert, *JFDS*, 2023). Systematically investigates ensemble methods for the meta-labeling secondary model and finds that **ensembles are especially beneficial when the underlying data consist of multiple regimes and are nonlinear in nature.** That sentence is a description of our ES setup with its 15-cell composite regime cross-product. We currently train multiple backends and then *discard all but one* via champion selection. This paper says that is the wrong move.

Reference implementation: `github.com/hudson-and-thames/meta-labeling`.

### 3.3 Tier 1 — Cheap, High-Probability Wins

#### 3.3.1 Heterogeneous ensembling instead of champion selection

**The change:** stop picking a champion. Fuse the calibrated probabilities of XGBoost + TCN (+ a third diverse learner) via rank-averaging and via a stacked meta-learner trained out-of-fold under the existing purged geometry.

**Why:** we already train all these models. We already calibrate them. We then throw away the ensemble and keep one. The ensemble meta-labeling literature says this is exactly backwards in multi-regime, nonlinear data. Our own results reinforce it: TCN and XGBoost have *different* strengths (TCN wins CV AP; reversal XGB wins post-training Sharpe), which is the textbook signature of decorrelated errors — the precondition for ensemble gain.

**Cost:** near zero. This is the first thing to run.

**Risk:** the stacker is another model with another opportunity to overfit. It must be trained strictly out-of-fold, and its parameters counted against the deflated-Sharpe correction (Section 6).

#### 3.3.2 TabPFN-v2 as a tabular backend

**The change:** feed the existing XGBoost feature matrix (top-N features × window × lags, plus delta features) to TabPFN-v2 as an in-context classifier.

**Why this is the best architecture/scale fit currently available:** TabPFN-v2 is a transformer pretrained on synthetic priors that performs in-context classification with no gradient training, and it is **state-of-the-art specifically on small datasets** — the regime where classical tabular regressors start to run out. Its backbone alternates **attention across features (columns)** with **attention across samples (rows)**, meaning it reasons over the ~2,000 events as a *set* rather than fitting a fixed function to them. That is a genuinely different inductive bias from gradient boosting, and it is aimed at our actual constraint.

The TabPFN-TS work further shows the same backbone handles *temporal* structure when supplied with lightweight temporal features, at only **11M parameters**, achieving state-of-the-art covariate-informed forecasting on GIFT-Eval and fev-bench. Since we already build a temporally-featurized matrix for XGBoost, the integration is close to a drop-in.

**Critical implementation constraint:** TabPFN's "context set" *is* training data. It must be constructed strictly from the in-fold training rows, respecting the purge and embargo. A naive implementation that passes the full dataset as context is a catastrophic leak. This must be an explicit assertion in the trainer, in the same family as the existing anti-leakage assertions.

**Secondary constraints:** column and row limits on the model; inference cost scales with context size. Check both against our matrix dimensions before committing.

#### 3.3.3 CatBoost / LightGBM for diversity

**The change:** add CatBoost and LightGBM as backends alongside XGBoost.

**Why:** not as replacements — as *decorrelation*. Different split-finding algorithms produce genuinely different error surfaces, which is what the ensemble consumes. CatBoost's ordered boosting and native categorical handling is a particularly good fit for the high-cardinality interactions we care about: `setup_type × session_phase × composite_regime`, which XGBoost currently sees only as one-hot noise.

**Cost:** low. Both slot into the existing Optuna + purged-CV trainer with minimal change.

### 3.4 Tier 2 — Sequence Architectures With Structural Fit

#### 3.4.1 The FRVP profile-shape encoder (the highest-novelty item on this list)

**The problem with the current representation.** The Fixed Range Volume Profile is a **distribution over price** — a histogram. We currently compress it into a handful of scalar features: distance to POC, distance to VAH/VAL, HVN/LVN flags, value-area width. This discards **shape**.

A trader looking at a profile does not read "distance to POC = 1.3 ATR." They read *morphology*: is this a b-shape (long liquidation tail), a p-shape (short covering), a double distribution (two accepted regions with a thin middle), a balanced bell, or a thin LVN shelf sitting directly above price with nothing to slow a breakout? None of those are recoverable from our scalar features, and every one of them is the actual content of the auction-theoretic argument that motivated the whole FRVP program.

**The change.** Represent the profile as a fixed-length vector of volume-per-bin over price bins, **expressed relative to current price and normalized by structural ATR** (so the representation is translation- and scale-invariant, which also solves the contract-coordinate problem flagged in the ES roll-handling section). Then run a small 1D CNN — or a DeepSets / attention encoder over the set of significant levels — over that vector.

Architecture becomes two-branch:

```
[ profile-shape branch ]         [ sequence branch ]
  volume-per-relative-bin          window of bar features
  → 1D CNN / set encoder           → TCN (existing)
            \                       /
             \                     /
              → concat → head → P(profitable)
```

**Why this is the item most worth doing:** it is the only architecture on this list that is *specific to our signal* and not available off the shelf. Every other proposal here is a better general-purpose function approximator. This one adds *information the model currently cannot see*. In our experience that is where the untapped edge usually is.

**It also applies directly to ICT.** The same relative-price-bin encoding trick works for the FVG/order-block/liquidity-level landscape: encode the map of untapped levels above and below price as a vector over relative-price bins, and let a CNN learn the geometry of the liquidity landscape rather than hand-crafting distance-to-nearest-FVG scalars.

#### 3.4.2 Dual-attention transformer (TLOB-style), adapted to bars

TLOB proposes a transformer whose blocks contain **both** self-attention over the temporal axis **and** attention over the feature axis, on the argument that market data requires learning temporal *and* spatial/feature dependencies simultaneously. It also replaces conventional z-score normalization with a **bilinear normalization layer**, explicitly because standard normalization fails under distribution shift at inference time.

**Two things to steal:**

1. **Feature-axis attention.** Our TCN reads out only the final timestep of the last dilated conv layer (`x[:, :, -1]`). It learns temporal patterns within each feature channel, and cross-channel mixing only implicitly through the pointwise convolutions. It structurally *cannot* learn an explicit, input-dependent interaction between, say, "distance to LVN" and "displacement magnitude" at a given bar. Feature-axis attention can.

2. **The normalization critique.** We use RobustScaler fit strictly in-fold — which is correct, and prevents leakage — but it is *static*. It is fit on the training fold and applied unchanged to a test fold that may live in a different volatility world. An adaptive normalization layer is worth an ablation on its own, independent of the attention architecture.

**Expected outcome:** modest. This is still a transformer at 2,000 events. Test it, do not bet on it.

#### 3.4.3 Temporal Fusion Transformer

**Recommended not for accuracy but for architecture.** TFT has first-class, distinct slots for:

- **Static covariates** → `setup_type`, `setup_family`, `side`, `htf_context`, `anchor_level` type
- **Time-varying known** inputs → session phase, time-of-day, calendar
- **Time-varying observed** inputs → the price/feature sequence

plus a **variable selection network** yielding per-event feature attribution natively, rather than post-hoc via SHAP.

This is the closest off-the-shelf match to the **strategy-driven architecture** the Meyer/Joubert paper argues for: the strategy artifacts get their own pathway. It also gives an interpretable *gate* rather than a post-hoc explanation, which matters for a system whose entire design philosophy is "interpretable rule layer + probabilistic filter."

Expect it to be data-hungry and to underperform on raw metrics at our event counts. Run it as a candidate and an interpretability instrument, not a favorite.

#### 3.4.4 Mamba / structured state-space (S4/S6) as a TCN swap

Linear-time sequence modeling with a long effective receptive field and fewer parameters than attention. Practically relevant for us: our TCN training had to **disable AMP entirely** (`resolve_amp_usage()`) after observing non-finite logits during full-data refits — a known pathology of deep dilated convolutions with float16 gradient accumulators. A state-space encoder is a drop-in replacement for the TCN encoder in the existing trainer, may not exhibit that pathology, and buys ensemble diversity even if it only matches the TCN.

**Cost:** low (encoder swap). **Expected gain:** small but positive, mostly via the ensemble.

### 3.5 Tier 3 — Pretraining, the Real Lever on ~2k Labels

**The core insight of this section:** our binding constraint is *labeled event count*, not model capacity. The literature's answer to a labeled-data constraint is not a bigger supervised model. It is **self-supervised pretraining on the unlabeled data we already have** — millions of 5-minute bars — followed by fine-tuning a small head on the sparse labeled events.

We have ~2,000 labeled events per family/side and roughly **500,000+ unlabeled ES 5-minute bars** (2015–2024). We currently use approximately 0.4% of our data.

#### 3.5.1 Self-supervised pretraining of the encoder

**Options:**

- **Masked reconstruction** (MAE-style): mask random spans of the bar sequence, reconstruct them. Simple, robust, no negative-sampling machinery.
- **TF-C (Time-Frequency Consistency)**: pretrains by enforcing that time-domain and frequency-domain views of the same window embed close together. Reported to outperform eight SOTA baselines by ~15% F1 on average when fine-tuned on downstream classification. The frequency view is attractive for us because volatility-regime structure is naturally spectral.
- **TS2Vec-style hierarchical contrastive learning**: contrastive at multiple temporal resolutions, which matches our multi-timeframe (5m / 30m / 1h) design.

**The protocol:**

1. Pretrain the TCN (or Mamba, or dual-attention) encoder on **all** ES 5-minute bars 2015–2024, self-supervised, no labels. This is a one-time cost.
2. Freeze or lightly fine-tune the encoder; train a **small classification head** on the ~2,000 FRVP/ICT meta-label events.
3. Compare OOF AP and post-cost walk-forward Sharpe against the from-scratch TCN baseline.

**Leakage discipline:** the pretraining corpus must respect the walk-forward geometry. An encoder pretrained on 2015–2024 and then evaluated on a 2019 test fold has seen the future. Either (a) pretrain separately per fold on the fold's training window only (expensive but correct), or (b) pretrain once on a strictly pre-backtest window and hold everything after it as a genuine out-of-sample era. Option (b) is cleaner and cheaper; it costs some data.

**This is the highest-ceiling item in Part II.** It directly attacks the ~1,500-event floor that currently forces XGBoost as a fallback in both the FRVP and ICT design papers.

#### 3.5.2 Kronos as a frozen feature extractor

Kronos is the first open-source foundation model for financial candlesticks (K-lines), a family of **decoder-only** models using a two-stage framework: a specialized tokenizer that quantizes continuous OHLCV into hierarchical discrete tokens, then an autoregressive transformer pretrained on **over 12 billion K-line records from 45 global exchanges**. Reported zero-shot results: **+93% RankIC over the leading general time-series foundation model** and **+87% over the best non-pretrained baseline** on price-series forecasting, plus 9% lower MAE on volatility forecasting. Accepted to AAAI 2026; weights and fine-tuning scripts are public.

Its motivating premise is directly relevant to us: generic TSFMs, despite scale, **underperform dedicated architectures on financial data**, because financial series have extreme noise, non-stationarity, heavy tails and regime changes that generic pretraining corpora do not contain.

**The cheap experiment (one week):**

1. At each event bar, run the preceding N-bar window through **frozen** Kronos.
2. Take the final hidden state (or a pooled representation) as an embedding vector.
3. Append the embedding to the existing XGBoost feature matrix.
4. Measure the change in OOF AP and walk-forward Sharpe.

**Reading the result:**
- Embedding adds nothing → strong evidence that our engineered features already saturate the available signal, and that the whole Tier-3 pretraining program is not worth pursuing. **This is a valuable negative result and it is cheap to obtain.**
- Embedding adds 2–3 points of AP → fine-tune Kronos on our corpus and promote this to a primary track.

**Run this before the expensive SSL program in 3.5.1.** It is a cheap read on whether pretraining helps *at all* on our instrument and timeframe, and it costs a week instead of a quarter.

### 3.6 Deprioritized and Rejected

| Candidate | Verdict | Reason |
|---|---|---|
| **LSTM** | Retain as diversity benchmark only | v2 regressed **−0.311 test AP** from v1. Documented as fragile and non-monotonic under retraining. Not a contender. |
| **Transformer trained from scratch** | Reject | 2,000 events. Overfits before it learns. The 13,500-fit crypto study found even 35k-parameter transformers overfit at ~2,000 samples. |
| **Reinforcement learning / policy-gradient sizing** | Reject for now | Sample-inefficient by orders of magnitude, and we have a well-specified supervised formulation with a clean EV expression. Revisit only after Part I is done and the sizing layer is a proven bottleneck. |
| **Limit-order-book models (TLOB, LiT, DeepLOB, HLOB)** | Reject as models; mine for ideas | We do not have LOB data. Their *architectural* ideas (dual attention, bilinear normalization) transfer; their models do not. |
| **Graph neural networks / cross-asset** | Defer | Interesting for an ES/NQ/6E cross-asset extension. Out of scope until single-instrument performance is settled. |

---

## 4. Part III — Formulation Changes That Outrank Backbone Changes

These are cheaper than any architecture swap and are likely to move the needle more. They are listed separately because they are changes to *what we predict*, not *what predicts it*.

### 4.1 Multi-task auxiliary heads (highest value/cost ratio in the paper)

**The change:** instead of a single binary logit, give the existing TCN/XGB multiple heads trained jointly:

- `P(TP before SL)` — the current meta-label (primary)
- `E[MFE / MAE ratio]` — regression (auxiliary)
- `E[bars to barrier]` — regression (auxiliary)
- `E[realized R-multiple]` — regression (auxiliary)

**Why this matters at 2,000 events.** A binary label extracts **one bit** of information per event. An event where price ran 3.4R before reversing and an event where it scraped 1.01R past the target are labeled identically. Auxiliary regression targets extract far more signal from each event and act as a powerful regularizer — the shared encoder must learn representations that support *all* the heads, which is much harder to overfit than a single bit.

**And it unifies with Part I.** The `E[realized R]` head **is the abstain model** described in Section 2.6.1. One architectural change delivers the auxiliary-task regularization *and* the economically-correct veto. This is the single highest-priority item across both parts of this paper.

### 4.2 Predict R, threshold afterward

The current labeling collapses **timeout → 0**, which is a lossy and slightly dishonest encoding: a trade that drifted sideways for 20 bars and one that got stopped out immediately are treated as identical failures. Predicting expected R directly and binarizing at inference preserves the distinction and gives the policy layer a continuous quantity to size on.

The ICT design already flags this as an open question ("timeout should initially map to 0 unless existing conventions prove otherwise"). This is the answer: don't map it to anything. Regress on R.

### 4.3 Regime-gated mixture of experts

**The evidence:** the framework found that regime-conditional thresholds "help only marginally over a well-chosen global threshold, because the global threshold already sits near the regime-blended optimum."

**The correct inference from that finding is not "regime doesn't matter."** It is that **regime information does not belong in the threshold — it belongs inside the model.** A threshold is a scalar; it can only shift a decision boundary up or down. It cannot change *which features matter* in a given regime, which is what actually differs between a strong_up_high and a ranging_low market.

**The change:** a mixture-of-experts head gated on the composite regime (or on a learned soft regime embedding), so different expert sub-networks specialize in different market states, with the gate learned end-to-end. This is the architecturally-correct expression of what the regime-threshold map was trying and failing to do, and it is exactly the setting where the ensemble meta-labeling paper says ensembles pay: multiple regimes, nonlinear data.

### 4.4 Inverse meta-labeling

The meta-model's ceiling is bounded by the primary rule layer — a limitation both the FRVP and ICT design papers explicitly flag ("quality bounded by the rule-based signal layer"). Inverse meta-labeling, introduced in the Meyer/Joubert architecture paper, is the published technique for improving the **quantity and quality of the side forecasts** coming out of that primary layer. If the FRVP/ICT setup definitions are the ceiling, this is the tool for raising the ceiling.

### 4.5 Sample weighting and the CUSUM sampler

Noted for completeness: the framework records that **CUSUM events are computed but unused as a sampler**. That is free, already-built machinery for improving event-sampling quality, sitting idle. It should be either used or removed.

---

## 5. Part IV — Experiment Program

Ordered by expected value per unit of engineering time. **Do not start Part II before Phase A is complete.**

#### Phase A — Policy layer (≈ 1–2 weeks). *Blocking.*

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| A1 | Counterfactual PnL attribution table by `policy_abstain_reason` | 1 day | Identifies which rules destroy money. Predicted: `high_stress`, `cooldown` positive. |
| A2 | Delete every rule with positive counterfactual PnL | 1 day | Immediate PnL recovery. |
| A3 | Implement the correct EV filter: `EV = p·TP − (1−p)·SL − costs` | 2 days | Replaces expected-move filter with an economically sound veto. |
| A4 | Replace cooldown with concurrency cap + overlap-aware sizing | 2 days | Recovers the suppressed cluster trades. |
| A5 | Re-tune the policy objective to post-cost Sharpe (drop F0.5 from the objective; retain as diagnostic) | 3 days | The root-cause fix. |
| A6 | Fractional-Kelly sizing on calibrated `p` | 2 days | Converts remaining vetoes into size reductions. |
| A7 | Full purged walk-forward re-backtest against the no-abstain baseline | 2 days | Ship/no-ship gate (Section 2.7). |

#### Phase B — Cheap architecture wins (≈ 2 weeks)

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| B1 | Ensemble existing XGB + TCN calibrated probabilities (rank-avg + OOF stack) | 1 day | Likely positive. Free. |
| B2 | TabPFN-v2 on the existing XGB feature matrix | 3 days | Biggest cheap upside. Watch context-set leakage. |
| B3 | CatBoost + LightGBM backends for ensemble diversity | 3 days | Small direct gain, real ensemble gain. |
| B4 | Multi-task auxiliary heads on the incumbent TCN (**§4.1**) | 5 days | **Highest value/cost item in the paper.** Also produces the abstain model. |

#### Phase C — Structural architecture (≈ 3–4 weeks)

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| C1 | FRVP profile-shape CNN branch (**§3.4.1**) | 2 weeks | Most novel. Adds information the model currently cannot see. |
| C2 | Same encoding applied to the ICT liquidity landscape | 1 week | Extends C1 to the ICT package. |
| C3 | Regime-gated mixture of experts (**§4.3**) | 1 week | The correct version of regime conditioning. |

#### Phase D — Pretraining (≈ 1 week gate, then a quarter if it passes)

| # | Experiment | Effort | Expected outcome |
|---|---|---|---|
| D1 | Kronos frozen embeddings → XGBoost feature matrix | 1 week | **Cheap gate.** If AP doesn't move, kill the whole Tier-3 program. |
| D2 | *(gated on D1)* SSL pretrain encoder on all unlabeled ES bars, fine-tune head on events | 4–6 weeks | Highest ceiling. Attacks the sample-size floor directly. |
| D3 | *(gated on D1)* Mamba / dual-attention encoder swaps | 1 week each | Diversity + AMP-stability. |

---

## 6. Part V — Evaluation Discipline

Every experiment in Part IV is judged on the same gate, and the gate is not CV AP.

**Primary metrics:**
1. **Cost-aware, purged walk-forward, post-cost net PnL and monthly Sharpe.** This is the only metric that has ever agreed with reality in this repo.
2. **Event F0.5** — retained as a *diagnostic*, never as an *objective*. It goes on the report; it does not go in the selection score.
3. **Fold stability** — positive in ≥ 70% of walk-forward folds.
4. **Maximum drawdown and underwater duration.**

**Why CV AP is disqualified as the primary gate:** the record is unambiguous. Breakout TCN achieved **CV AP 0.969** and **post-training Sharpe 0.81**. Reversal XGB achieved **CV AP 0.694** and **post-training Sharpe 3.61**. The correlation between the training leaderboard and the deployment leaderboard is, on our own data, close to zero or negative. AP tells us the model ranks events well. Sharpe tells us whether we make money. They are different questions and we have been optimizing the first one.

**Deflated Sharpe is mandatory.** This paper proposes roughly a dozen architectures across four model families, two instruments, and multiple policy configurations. That is a very large multiple-testing surface. Every reported Sharpe must be **deflated** for the number of configurations tried (Bailey, Borwein, López de Prado & Zhu, 2014). Without this, we will find something that looks good, and it will be noise. The existing walk-forward machinery must be extended to track the trial count.

**Leakage assertions extend to every new component.** The existing anti-leakage assertion list (closed-bar-only FVGs, causal swings, completed HTF bars, in-fold scaler fitting, purged folds, embargo windows) must be extended with:
- TabPFN context set contains **only** in-fold training rows.
- SSL pretraining corpus respects the walk-forward boundary.
- Kronos embeddings are computed from a **causal** window only.
- The abstain/sizing layer's fitted constants are fit **in-fold**.
- Profile-shape bin vectors use only completed prior-session/prior-swing profiles, in single-contract coordinates.

---

## 7. Open Questions and Risks

**Q1. Is the abstain layer's damage confined to the OTE/EUR-USD cohort, or does it carry into ES?**
The measured numbers are from the EUR/USD OTE framework. The ES FRVP pipeline inherits the same policy code. The counterfactual attribution (A1) must be run separately on ES; the roll-boundary and 0DTE-era effects could change the answer.

**Q2. Is `stress_regime` actually correlated with `vol_regime`?**
The high-stress abstain critique in §2.3 depends on this. If stress is measured by something orthogonal to volatility (e.g. a spread-widening or gap-frequency proxy), the rule may be more defensible than argued. Measure it before deleting the rule.

**Q3. Is the probability-quantile filter leaky?**
The quantile window is not documented in the artifacts available. If it is computed over the full candidate distribution including test rows, this is a live look-ahead bug affecting every reported abstain-variant number. **Audit this first — it is a correctness issue, not a tuning issue.**

**Q4. Does the confidence-expectancy relationship remain monotone at the top?**
The confidence-quintile analysis reports monotone expectancy, which — if it holds at the decile level — implies the threshold is a sufficient control and most of the abstain stack is redundant. If it *breaks* at the top decile, that is a genuinely important non-monotonicity and the redesign in §2.6 needs an extra term.

**Q5. Will the ~2,000-event floor survive the split into four FRVP families and six-plus ICT setup types?**
Both design papers already flag small-sample risk. Adding architectures does not add events. If per-family/per-side counts fall below ~1,000, most of Part II becomes moot and the correct move is Tier-3 pretraining or family pooling with a `setup_type` feature — not a better backbone.

**Q6. Is there any abstain rule that beats "take everything above threshold, size by Kelly"?**
Stated as a risk because the answer may be no, and the project must be prepared to accept that. See §2.7.

**Q7. Multiple-testing burden.**
Executing this entire program without a deflated-Sharpe discipline will manufacture a false champion with near-certainty. The trial-count tracking (Section 6) is not optional.

---

## 8. References

**Meta-labeling**
- Joubert, J. F. (2022). *Meta-Labeling: Theory and Framework.* Journal of Financial Data Science, 4(3), 31–44.
- Meyer, M., Joubert, J. F., & Alfeus, M. (2022). *Meta-Labeling: Architecture.* Journal of Financial Data Science, 4(4), 10–24.
- Thumm, D., Barucca, P., & Joubert, J. F. (2023). *Ensemble Meta-Labeling.* Journal of Financial Data Science.
- Hudson & Thames. *meta-labeling* code base. `github.com/hudson-and-thames/meta-labeling`
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (Ch. 3 triple-barrier; Ch. 7 purged CV; Ch. 10 meta-labeling.)

**Architecture — tabular / foundation models**
- Hollmann, N., et al. (2025). *TabPFN-v2* — tabular foundation model; row-attention + column-attention backbone; SOTA on small datasets.
- Hoo, S. B., Müller, S., Salinas, D., & Hutter, F. (2025–2026). *From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting.* arXiv:2501.02945. 11M parameters; SOTA on covariate-informed forecasting (GIFT-Eval, fev-bench). Package: `tabpfn-time-series`.
- Shi, Y., Fu, Z., Chen, S., Zhao, B., Xu, W., Zhang, C., & Li, J. (2025). *Kronos: A Foundation Model for the Language of Financial Markets.* arXiv:2508.02739. AAAI 2026. 12B K-lines, 45 exchanges; +93% RankIC vs. leading TSFM. `github.com/shiyu-coder/Kronos`

**Architecture — sequence models**
- Berti, L., & Kasneci, G. (2025). *TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data.* arXiv:2502.15757. Dual (temporal + feature) attention; bilinear normalization for distribution shift.
- Bai, S., Kolter, J. Z., & Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271. (The TCN reference.)
- Lim, B., et al. *Temporal Fusion Transformer* — static covariates, variable selection networks.

**Architecture — pretraining**
- Zhang, X., Zhao, Z., Tsiligkaridis, T., & Zitnik, M. (2022). *Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency (TF-C).* NeurIPS 2022. arXiv:2206.08496. +15.4% F1 average over eight baselines.

**Evidence that deep learning does not automatically win**
- *Deep incremental learning models for financial temporal tabular datasets with distribution shifts.* arXiv:2303.07925. XGBoost > MLP across a wide hyperparameter range; TabNet does not reliably beat plain MLP.
- *Comparative Analysis of Modern Machine Learning Models for Retail Sales Forecasting.* arXiv:2506.05941. Tree-based methods beat N-BEATS / N-HiTS / TFT on sparse data; "prioritize alignment with problem characteristics over architectural sophistication."
- Mesfin, M. (2026). *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ.* arXiv:2605.17724. Closest published analogue to our problem: 5-minute futures bars, single instrument, ~944 days.
- *Why We Chose XGBoost Over LSTM for Crypto Prediction.* 13,500 model fits; XGBoost beat LSTM, attention variants, and LSTM→XGBoost hybrids consistently at ~2,000-sample training windows.

**Evaluation**
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management, 40(5), 94–107.

---

### Appendix A — The Single-Paragraph Version

The abstain policy removes 65% of trades and 69% of profit, which means the trades it removes are *better* than the trades it keeps; this happens because the entire policy stack — a `0.60·event_F0.5 + 0.40·expectancy` threshold objective followed by five hand-written precision-improving vetoes — optimizes precision in a system with a 2:1 payoff ratio, where the break-even hit rate is 33% and precision-destroying trades are routinely profitable. The fix is to replace the veto stack with a single economic condition (`EV ≤ 0`), size everything else by fractional Kelly, replace the 4-bar cooldown with a concurrency cap, and re-tune the whole layer against post-cost Sharpe. On the model side, the literature and our own leaderboard both say architecture is not the binding constraint at ~2,000 events — sample size is — so the shortlist is: ensemble what we already train (free), TabPFN-v2 (built for exactly this data scale), multi-task auxiliary heads (which double as the abstain model), a CNN over the volume-profile *shape* we currently throw away (the one genuinely novel idea here), and a one-week Kronos-embedding probe to decide whether the expensive pretraining program is worth starting at all. Judge everything on deflated, cost-aware, purged walk-forward Sharpe, because CV average precision has been actively misleading us: Breakout TCN scored 0.969 AP and a Sharpe of 0.81; Reversal XGB scored 0.694 AP and a Sharpe of 3.61.

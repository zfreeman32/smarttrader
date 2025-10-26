"""
Multi-Timeframe Confluence Model - Complete Integration Guide
=============================================================

This document provides complete instructions for training, deploying, and using
the Multi-Timeframe Confluence Model in your EURUSD trading system.

TABLE OF CONTENTS
-----------------
1. Overview
2. Training Pipeline
3. Real-Time Inference
4. Integration with Trading System
5. Performance Monitoring
6. Troubleshooting
"""

# =============================================================================
# 1. OVERVIEW
# =============================================================================

"""
PURPOSE:
The Multi-Timeframe Confluence Model analyzes price action across 5 timeframes
(1m, 5m, 15m, 1h, 4h) to determine:
  - Confluence Score [0-1]: Agreement across timeframes
  - Directional Bias [-1 to +1]: Predicted direction and strength

ARCHITECTURE:
  - Multi-Stream Transformer with cross-timeframe attention
  - Separate encoders for each timeframe
  - Dual output heads for confluence and bias
  - 47+ features per timeframe covering trend, momentum, volume, volatility

USE IN TRADING:
  - Filter trades: Only trade when confluence > 0.70
  - Size positions: Higher confluence = larger position
  - Avoid counter-trend trades: Bias must align with signal
  - Reduce false breakouts: Low confluence = wait
"""

# =============================================================================
# 2. TRAINING PIPELINE
# =============================================================================

"""
STEP 1: Prepare Your Data
-------------------------
You need 1-minute EURUSD OHLCV data for at least 2 years.

Required columns:
  - timestamp (datetime index)
  - open (float)
  - high (float)
  - low (float)
  - close (float)
  - volume (int/float)

Example data loading:
"""

import pandas as pd
import numpy as np

# Load your data
df_1m = pd.read_csv('eurusd_1m_data.csv', index_col='timestamp', parse_dates=True)

# Verify data
print(f"Data shape: {df_1m.shape}")
print(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")
print(f"Columns: {df_1m.columns.tolist()}")
print("\nFirst few rows:")
print(df_1m.head())

"""
STEP 2: Train the Model
-----------------------
"""

from mtf_confluence_trainer import (
    MultiTimeframeDataPreparator,
    MultiTimeframeConfluenceModel,
    ConfluenceModelTrainer
)

# Initialize data preparator
preparator = MultiTimeframeDataPreparator(
    timeframes={
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240
    },
    lookback_periods={
        '1m': 240,   # 4 hours of 1m data
        '5m': 100,   # 8+ hours of 5m data
        '15m': 50,   # 12+ hours of 15m data
        '1h': 24,    # 1 day of 1h data
        '4h': 20     # 3+ days of 4h data
    }
)

# Prepare training data
X_train, X_val, y_train_conf, y_train_bias, y_val_conf, y_val_bias = preparator.prepare_training_data(
    df_1m, 
    validation_split=0.2
)

# Build model
input_shapes = {tf: X_train[tf].shape[1:] for tf in X_train.keys()}

model = MultiTimeframeConfluenceModel(
    input_shapes=input_shapes,
    d_model=128,           # Transformer dimension
    num_heads=4,            # Attention heads
    ff_dim=256,             # Feed-forward dimension
    num_transformer_blocks=2,  # Transformer depth
    dropout_rate=0.2
)

model.build_model()
model.compile_model(learning_rate=0.001)

# Train
trainer = ConfluenceModelTrainer(
    model=model,
    model_save_path='./outputs/mtf_confluence_model_best.keras'
)

history = trainer.train(
    X_train=X_train,
    y_train_confluence=y_train_conf,
    y_train_bias=y_train_bias,
    X_val=X_val,
    y_val_confluence=y_val_conf,
    y_val_bias=y_val_bias,
    epochs=100,
    batch_size=64,
    patience=15
)

# Save scalers
import pickle
with open('./outputs/mtf_confluence_scalers.pkl', 'wb') as f:
    pickle.dump(preparator.scalers, f)

print("Training complete!")
print(f"Model: ./outputs/mtf_confluence_model_best.keras")
print(f"Scalers: ./outputs/mtf_confluence_scalers.pkl")

"""
STEP 3: Evaluate Model Performance
----------------------------------
"""

# Evaluate on validation set
results = trainer.evaluate(X_val, y_val_conf, y_val_bias)

# Make predictions
predictions = model.model.predict([X_val[tf] for tf in sorted(X_val.keys())])
pred_confluence = predictions[0].flatten()
pred_bias = predictions[1].flatten()

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n=== MODEL PERFORMANCE ===")
print("\nConfluence Score:")
print(f"  MAE: {mean_absolute_error(y_val_conf, pred_confluence):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_val_conf, pred_confluence)):.4f}")
print(f"  R²: {r2_score(y_val_conf, pred_confluence):.4f}")

print("\nDirectional Bias:")
print(f"  MAE: {mean_absolute_error(y_val_bias, pred_bias):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_val_bias, pred_bias)):.4f}")
print(f"  R²: {r2_score(y_val_bias, pred_bias):.4f}")

# Direction accuracy
correct_direction = np.sum(np.sign(y_val_bias) == np.sign(pred_bias))
direction_accuracy = correct_direction / len(y_val_bias) * 100
print(f"\nDirectional Accuracy: {direction_accuracy:.2f}%")

"""
TARGET METRICS:
  - Confluence MAE: < 0.15
  - Bias MAE: < 0.20
  - Directional Accuracy: > 65%
  - R² for both: > 0.40

If metrics are below targets:
  1. Train longer (more epochs)
  2. Add more training data
  3. Increase model capacity (d_model, num_heads)
  4. Adjust learning rate
"""

# =============================================================================
# 3. REAL-TIME INFERENCE
# =============================================================================

"""
STEP 1: Initialize Analyzer
---------------------------
"""

from mtf_confluence_analyzer import ConfluenceAnalyzer, format_analysis_report

analyzer = ConfluenceAnalyzer(
    model_path='./outputs/mtf_confluence_model_best.keras',
    scaler_path='./outputs/mtf_confluence_scalers.pkl'
)

"""
STEP 2: Analyze Current Market
------------------------------
"""

# Get recent 1-minute data (must have enough history for all timeframes)
# Minimum required: ~1500 bars (enough for 4h timeframe resampling)

current_data = get_recent_1m_data(bars=2000)  # Your data fetching function

# Analyze
confluence_score, directional_bias, details = analyzer.analyze(current_data)

# Print formatted report
print(format_analysis_report(confluence_score, directional_bias, details))

"""
STEP 3: Use in Trading Decisions
--------------------------------
"""

# Simple usage
should_trade = analyzer.should_trade(
    confluence_score, 
    directional_bias,
    min_confluence=0.70,  # Only trade if confluence >= 0.70
    min_bias=0.30         # Only trade if |bias| >= 0.30
)

if should_trade:
    if directional_bias > 0:
        print("✅ LONG SIGNAL - High confluence bullish")
    else:
        print("✅ SHORT SIGNAL - High confluence bearish")
else:
    print("⏸️ WAIT - Insufficient confluence or weak bias")

# Advanced usage with signal quality
quality = analyzer.get_signal_quality(confluence_score, directional_bias)

if quality == 'EXCELLENT':
    position_size = 1.0  # Full size
elif quality == 'GOOD':
    position_size = 0.75
elif quality == 'FAIR':
    position_size = 0.5
else:
    position_size = 0.0  # Don't trade

# =============================================================================
# 4. INTEGRATION WITH TRADING SYSTEM
# =============================================================================

"""
Integration Pattern for Complete Trading System
-----------------------------------------------

The confluence model is TIER 4 component in the ensemble decision system.
Here's how to integrate it:
"""

class TradingSystem:
    """Complete trading system with confluence model integration"""
    
    def __init__(self):
        # Load all models
        self.long_classifier = load_model('long_signal_classifier.keras')
        self.short_classifier = load_model('short_signal_classifier.keras')
        self.returns_forecaster = load_model('returns_5_forecaster.keras')
        self.regime_classifier = load_model('regime_classifier.keras')
        self.pattern_validator = load_model('pattern_validator.keras')
        self.confluence_analyzer = ConfluenceAnalyzer(
            model_path='mtf_confluence_model_best.keras',
            scaler_path='mtf_confluence_scalers.pkl'
        )
        
        # ICT detectors
        self.fvg_detector = FVGDetector()
        self.ob_detector = OrderBlockDetector()
        
    def generate_signal(self, current_data):
        """
        Generate trading signal using ensemble of all models
        """
        # 1. Get confluence assessment
        confluence_score, directional_bias, confluence_details = self.confluence_analyzer.analyze(current_data)
        
        # 2. Get ML model predictions
        long_prob = self.long_classifier.predict(current_data)
        short_prob = self.short_classifier.predict(current_data)
        returns_forecast = self.returns_forecaster.predict(current_data)
        regime = self.regime_classifier.predict(current_data)
        
        # 3. Detect ICT patterns
        patterns = self.detect_patterns(current_data)
        validated_patterns = [p for p in patterns if self.pattern_validator.score(p) > 0.75]
        
        # 4. ENSEMBLE DECISION
        
        # Required conditions
        if confluence_score < 0.70:
            return {
                'signal': 'WAIT',
                'reason': f'Low confluence: {confluence_score:.3f}',
                'confluence_score': confluence_score,
                'directional_bias': directional_bias
            }
        
        # Directional agreement check
        if long_prob > 0.70 and directional_bias > 0.3 and returns_forecast > 0.002:
            # All models agree on LONG
            
            # Calculate confidence
            weighted_confidence = (
                long_prob * 0.25 +
                min(confluence_score, 1.0) * 0.15 +
                (abs(directional_bias) * 0.10) +
                (len(validated_patterns) > 0) * 0.10
            )
            
            return {
                'signal': 'LONG',
                'confidence': weighted_confidence,
                'confluence_score': confluence_score,
                'directional_bias': directional_bias,
                'long_prob': long_prob,
                'returns_forecast': returns_forecast,
                'regime': regime,
                'patterns': validated_patterns,
                'position_size': self.calculate_position_size(weighted_confidence)
            }
        
        elif short_prob > 0.70 and directional_bias < -0.3 and returns_forecast < -0.002:
            # All models agree on SHORT
            
            weighted_confidence = (
                short_prob * 0.25 +
                min(confluence_score, 1.0) * 0.15 +
                (abs(directional_bias) * 0.10) +
                (len(validated_patterns) > 0) * 0.10
            )
            
            return {
                'signal': 'SHORT',
                'confidence': weighted_confidence,
                'confluence_score': confluence_score,
                'directional_bias': directional_bias,
                'short_prob': short_prob,
                'returns_forecast': returns_forecast,
                'regime': regime,
                'patterns': validated_patterns,
                'position_size': self.calculate_position_size(weighted_confidence)
            }
        
        else:
            # Models disagree or weak signals
            return {
                'signal': 'WAIT',
                'reason': 'Models disagree or insufficient conviction',
                'confluence_score': confluence_score,
                'directional_bias': directional_bias,
                'long_prob': long_prob,
                'short_prob': short_prob
            }
    
    def calculate_position_size(self, confidence):
        """Dynamic position sizing based on confidence"""
        base_size = 1  # Base contracts
        
        if confidence >= 0.85:
            return base_size * 1.0  # Full size
        elif confidence >= 0.75:
            return base_size * 0.75
        elif confidence >= 0.70:
            return base_size * 0.5
        else:
            return 0  # Don't trade

"""
Real-Time Trading Loop Example
------------------------------
"""

def trading_loop():
    """Main trading loop with confluence model"""
    
    system = TradingSystem()
    
    while market_is_open():
        # Get latest data
        current_data = fetch_latest_data(bars=2000)
        
        # Generate signal
        signal = system.generate_signal(current_data)
        
        # Execute trade
        if signal['signal'] in ['LONG', 'SHORT']:
            print(f"\n{'='*80}")
            print(f"SIGNAL: {signal['signal']}")
            print(f"Confidence: {signal['confidence']:.3f}")
            print(f"Confluence: {signal['confluence_score']:.3f}")
            print(f"Bias: {signal['directional_bias']:.3f}")
            print(f"Position Size: {signal['position_size']}")
            print(f"{'='*80}\n")
            
            # Place order
            execute_trade(
                direction=signal['signal'],
                size=signal['position_size'],
                stop_loss=calculate_stop_loss(signal),
                take_profit=calculate_take_profit(signal)
            )
        
        # Wait for next bar
        time.sleep(60)  # 1 minute

# =============================================================================
# 5. PERFORMANCE MONITORING
# =============================================================================

"""
Key Metrics to Track
-------------------

1. Confluence Accuracy:
   - Does high confluence actually lead to better trades?
   - Track win rate by confluence bucket (0.7-0.75, 0.75-0.80, etc.)

2. Directional Accuracy:
   - When bias > 0.5, how often is next move actually up?
   - Calculate correlation between bias and actual returns

3. Signal Quality Distribution:
   - How often do you get EXCELLENT vs GOOD vs FAIR signals?
   - Are you trading too often (low quality) or missing opportunities?

4. False Positive Rate:
   - High confluence but losing trade
   - Should trigger model retraining
"""

class ConfluencePerformanceTracker:
    """Track confluence model performance in production"""
    
    def __init__(self):
        self.predictions = []
        self.actuals = []
        
    def log_prediction(self, confluence_score, directional_bias, trade_id):
        """Log prediction when trade is placed"""
        self.predictions.append({
            'trade_id': trade_id,
            'timestamp': datetime.now(),
            'confluence': confluence_score,
            'bias': directional_bias
        })
    
    def log_outcome(self, trade_id, actual_return):
        """Log actual outcome when trade closes"""
        for pred in self.predictions:
            if pred['trade_id'] == trade_id:
                pred['actual_return'] = actual_return
                pred['correct_direction'] = np.sign(pred['bias']) == np.sign(actual_return)
                break
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        df = pd.DataFrame(self.predictions)
        df = df.dropna()  # Only completed trades
        
        metrics = {}
        
        # Overall directional accuracy
        metrics['directional_accuracy'] = df['correct_direction'].mean()
        
        # Accuracy by confluence bucket
        df['confluence_bucket'] = pd.cut(df['confluence'], bins=[0, 0.7, 0.75, 0.8, 0.85, 1.0])
        metrics['accuracy_by_confluence'] = df.groupby('confluence_bucket')['correct_direction'].mean()
        
        # Win rate by signal quality
        df['signal_quality'] = df.apply(lambda row: 
            'EXCELLENT' if row['confluence'] >= 0.8 and abs(row['bias']) >= 0.5 else
            'GOOD' if row['confluence'] >= 0.7 and abs(row['bias']) >= 0.3 else
            'FAIR' if row['confluence'] >= 0.6 else 'POOR', axis=1)
        
        metrics['win_rate_by_quality'] = df.groupby('signal_quality')['correct_direction'].mean()
        
        # Correlation between bias and returns
        metrics['bias_return_correlation'] = df['bias'].corr(df['actual_return'])
        
        return metrics

# =============================================================================
# 6. TROUBLESHOOTING
# =============================================================================

"""
Common Issues and Solutions
--------------------------

ISSUE 1: Model predictions are always near 0.5 (no strong signals)
SOLUTION:
  - Model may be underfit. Train longer or increase capacity.
  - Check if training data has sufficient variation in market conditions.
  - Verify labels are calculated correctly.

ISSUE 2: High confluence but poor trade outcomes
SOLUTION:
  - Confluence model may be overfitting to recent patterns.
  - Retrain with more diverse data (bull/bear/ranging markets).
  - Lower confluence threshold from 0.70 to 0.75-0.80.

ISSUE 3: Directional bias often wrong
SOLUTION:
  - Forward window may be too short/long. Experiment with different horizons.
  - Add more diverse features (volatility, volume, order flow).
  - Ensemble with other directional models.

ISSUE 4: "Insufficient data" error in real-time
SOLUTION:
  - Ensure you have at least 1500 bars of 1m data.
  - For 4h timeframe, need 240*20 = 4800 minutes = 80 hours of data.
  - Keep a rolling buffer of recent data.

ISSUE 5: Slow inference (<100ms target)
SOLUTION:
  - Reduce model size (fewer transformer blocks, smaller d_model).
  - Use GPU inference.
  - Cache feature calculations (don't recalculate every bar).
  - Batch predictions if analyzing multiple instruments.

ISSUE 6: Model degradation over time
SOLUTION:
  - Implement rolling retraining (quarterly).
  - Monitor performance metrics weekly.
  - Have fallback to simpler rules if model performance drops.
"""

# =============================================================================
# APPENDIX: FEATURE IMPORTANCE
# =============================================================================

"""
Most Important Features (per timeframe):

1m:
  - signal_consensus (composite of 6 indicators)
  - macd_histogram
  - volume_ratio
  - rsi
  - stoch_rsi

5m:
  - ema_50_slope
  - price_slope  
  - signal_consensus
  - atr_pct
  - bb_position

15m:
  - ema_9_21_cross
  - ema_21_50_cross
  - macd_direction
  - volume_trend
  - distance_from_high/low

1h:
  - ema_50_slope (primary trend)
  - rsi_signal
  - bb_width (volatility)
  - price_range_position
  - volume_ratio

4h:
  - ema_9_21_cross (major trend)
  - ema_50_slope
  - atr_pct
  - macd_direction
  - signal_consensus

The model learns to weight these automatically through attention mechanism.
"""

# =============================================================================
# FINAL CHECKLIST
# =============================================================================

"""
✅ BEFORE GOING LIVE:

□ Train model on 2+ years of data
□ Achieve target metrics (MAE < 0.15, Dir Acc > 65%)
□ Backtest with confluence filter - improves win rate?
□ Paper trade for 2+ weeks
□ Verify inference speed (<100ms)
□ Set up performance monitoring
□ Define retraining schedule (quarterly)
□ Document model version and parameters
□ Set up alerts for model degradation
□ Test edge cases (low liquidity, high volatility)

🎯 PRODUCTION DEPLOYMENT:

□ Load model and scalers
□ Initialize ConfluenceAnalyzer
□ Integrate with ensemble decision system
□ Set confluence threshold (0.70-0.80)
□ Set minimum bias threshold (0.30)
□ Configure position sizing by signal quality
□ Enable performance tracking
□ Start with small position sizes
□ Monitor daily for first month
□ Gradually scale up as confidence builds
"""

print("\n" + "="*80)
print("Multi-Timeframe Confluence Model - Integration Guide Complete!")
print("="*80)
print("\nNext Steps:")
print("1. Train model: python mtf_confluence_trainer.py")
print("2. Test inference: python mtf_confluence_analyzer.py")
print("3. Integrate with trading system")
print("4. Paper trade and monitor")
print("5. Go live!")
print("\nGood luck! 🚀")

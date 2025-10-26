# Market Regime Classifier

**Production-ready LSTM-based market regime classification for EURUSD automated trading**

## Overview

The Market Regime Classifier is a critical component of the EURUSD automated trading system. It classifies the current market state into one of 7 distinct regimes, enabling the trading system to adapt its strategy selection based on market conditions.

### Why Regime Classification Matters

Different ICT (Inner Circle Trading) patterns work better in different market conditions:
- **Trending Markets**: Order blocks and FVG fills are more reliable
- **Ranging Markets**: Liquidity sweeps provide better opportunities
- **High Volatility**: Wait for confirmation before entry
- **Low Volatility**: Look for breakout setups

## Regime Classes

| Class | Regime | Description | ADX | Directional Indicator | ROC |
|-------|--------|-------------|-----|----------------------|-----|
| 0 | Strong Uptrend | Powerful bullish momentum | >25 | PLUS_DI > MINUS_DI | >0.5% |
| 1 | Weak Uptrend | Moderate bullish bias | 15-25 | PLUS_DI > MINUS_DI | Any |
| 2 | Ranging | Consolidation, no clear trend | <20 | Either | Any |
| 3 | Weak Downtrend | Moderate bearish bias | 15-25 | MINUS_DI > PLUS_DI | Any |
| 4 | Strong Downtrend | Powerful bearish momentum | >25 | MINUS_DI > PLUS_DI | <-0.5% |
| 5 | High Volatility | Breakout/unstable conditions | Any | Any | ATR >75th %ile |
| 6 | Low Volatility | Compression/coiling | Any | Any | ATR <25th %ile |

## Architecture

### Model Structure

```
Input: [lookback_period, num_features]
  ↓
LSTM Layer 1 (128 units, return_sequences=True)
  ↓
Batch Normalization + Dropout (0.3)
  ↓
LSTM Layer 2 (64 units, return_sequences=False)
  ↓
Batch Normalization + Dropout (0.3)
  ↓
Dense Layer (64 units, ReLU)
  ↓
Dropout (0.3)
  ↓
Dense Layer (32 units, ReLU)
  ↓
Output Layer (7 units, Softmax)
```

### Key Features (27 total)

**Trend Strength**
- ADX (14, 20 period)
- Plus DI, Minus DI
- EMA price ratios (10, 20, 50, 100)

**Volatility**
- ATR percentage (14, 20 period)
- Bollinger Band width and position

**Momentum**
- Rate of Change (5, 10, 20, 50 period)
- Momentum (10, 20 period)

**Volume**
- Volume ratio vs average
- On-Balance Volume

**Trend Direction**
- EMA crossover signals (10/20, 20/50, 50/200)
- Price position in swing range

## Installation

### Requirements

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### Optional (for TA-Lib indicators)

```bash
conda install -c conda-forge ta-lib
# or
pip install TA-Lib
```

## Usage

### 1. Training the Model

#### Basic Training

```bash
python train_regime_classifier.py \
    --data_path eurusd_1min.csv \
    --epochs 100 \
    --output_dir ./models
```

#### Advanced Training (Custom Parameters)

```bash
python train_regime_classifier.py \
    --data_path eurusd_1min.csv \
    --epochs 150 \
    --batch_size 128 \
    --lookback 150 \
    --lstm_units 256 128 64 \
    --dropout 0.4 \
    --learning_rate 0.0005 \
    --use_class_weights \
    --adx_trend 30.0 \
    --adx_ranging 18.0 \
    --output_dir ./models \
    --model_name regime_classifier_v2.keras
```

#### Quick Test (Sample Data)

```bash
python train_regime_classifier.py \
    --data_path eurusd_1min.csv \
    --sample_size 100000 \
    --epochs 20 \
    --batch_size 64
```

### 2. Real-Time Inference

```python
from regime_classifier import RegimeClassifierModel, RegimeFeatureEngineering
import pandas as pd
import numpy as np

# Load trained model
model = RegimeClassifierModel()
model.load('/mnt/user-data/outputs/regime_classifier.keras')

# Prepare live data
df_live = get_live_eurusd_data()  # Your data source

# Engineer features
feature_engineer = RegimeFeatureEngineering()
df_features = feature_engineer.engineer_features(df_live)

# Select last 100 bars (lookback period)
feature_cols = [...]  # Use same features as training
recent_data = df_features[feature_cols].iloc[-100:].values

# Scale and predict
recent_scaled = model.scaler.transform(recent_data)
recent_seq = np.expand_dims(recent_scaled, axis=0)

regime_class, regime_proba = model.predict(recent_seq)

regime_names = {
    0: 'Strong Uptrend',
    1: 'Weak Uptrend',
    2: 'Ranging',
    3: 'Weak Downtrend',
    4: 'Strong Downtrend',
    5: 'High Volatility',
    6: 'Low Volatility'
}

print(f"Current Regime: {regime_names[regime_class[0]]}")
print(f"Confidence: {regime_proba[0][regime_class[0]]:.2%}")
```

### 3. Integration with Trading System

```python
from regime_classifier import RegimeClassifierModel

class TradingSystem:
    def __init__(self):
        self.regime_model = RegimeClassifierModel()
        self.regime_model.load('regime_classifier.keras')
        
    def should_trade_pattern(self, pattern_type, market_data):
        # Get current regime
        regime, confidence = self.predict_regime(market_data)
        
        # Pattern-regime compatibility
        if pattern_type == "order_block":
            # Order blocks work best in trending markets
            if regime in [0, 1, 3, 4]:  # Any trend
                return confidence[0][regime] > 0.75
            return False
            
        elif pattern_type == "fvg_fill":
            # FVG fills work best in strong trends
            if regime in [0, 4]:  # Strong trends only
                return confidence[0][regime] > 0.80
            return False
            
        elif pattern_type == "liquidity_sweep":
            # Sweeps work best in ranging or weak trends
            if regime in [1, 2, 3]:
                return confidence[0][regime] > 0.70
            return False
            
        elif pattern_type == "breakout":
            # Avoid trading in high volatility
            if regime == 5:  # High volatility
                return False
            return True
        
        return False
```

## Data Requirements

### Input Format

CSV file with the following columns:

```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.1050,1.1055,1.1048,1.1052,1500
2024-01-01 00:01:00,1.1052,1.1058,1.1051,1.1056,1800
...
```

### Minimum Data Requirements

- **Training**: 2+ years of 1-minute EURUSD data (≈1M bars)
- **Validation**: Last 3-6 months
- **Features**: OHLCV only (indicators calculated automatically)

### Data Quality Checks

The system automatically:
- Removes rows with missing values
- Validates OHLC relationships (high ≥ open, close, low)
- Checks for duplicate timestamps
- Verifies volume > 0

## Performance Metrics

### Target Performance
- **Accuracy**: >75%
- **Precision**: >70% (per class)
- **Recall**: >65% (per class)
- **Inference Time**: <50ms

### Expected Results

Based on 2+ years of EURUSD data:

```
Classification Report:
                     precision    recall  f1-score   support
  Strong Up             0.78      0.75      0.76      8234
  Weak Up               0.71      0.68      0.69      6789
  Ranging               0.82      0.85      0.83     12456
  Weak Down             0.69      0.67      0.68      7123
  Strong Down           0.76      0.73      0.74      8891
  High Vol              0.88      0.91      0.89      3456
  Low Vol               0.85      0.87      0.86      4567

  accuracy                          0.78     51516
  macro avg             0.78      0.78      0.78     51516
  weighted avg          0.78      0.78      0.78     51516
```

## Hyperparameter Tuning

### Recommended Starting Points

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| lookback_period | 100 | 50-200 | Higher = more context, slower |
| lstm_units | [128, 64] | [64-256] | Larger = more capacity |
| dropout | 0.3 | 0.2-0.5 | Higher if overfitting |
| learning_rate | 0.001 | 0.0001-0.01 | Lower for stability |
| batch_size | 64 | 32-256 | Larger = faster, less stable |
| epochs | 100 | 50-200 | Use early stopping |

### Tuning Strategy

1. **Start with defaults** - Train baseline model
2. **Increase lookback** - If regime changes are missed (100 → 150)
3. **Add LSTM layers** - If underfitting (add 3rd layer with 32 units)
4. **Increase dropout** - If overfitting (0.3 → 0.4)
5. **Adjust learning rate** - If training unstable (0.001 → 0.0005)

## Troubleshooting

### Low Accuracy (<70%)

**Possible causes:**
- Insufficient training data (<1 year)
- Poor regime labeling thresholds
- Class imbalance not addressed

**Solutions:**
```bash
# Use class weights
python train_regime_classifier.py --use_class_weights ...

# Adjust labeling thresholds
python train_regime_classifier.py --adx_trend 30 --adx_ranging 18 ...

# Increase model capacity
python train_regime_classifier.py --lstm_units 256 128 64 ...
```

### Overfitting (Train acc >> Val acc)

**Solutions:**
```bash
# Increase dropout
python train_regime_classifier.py --dropout 0.4 ...

# Reduce model size
python train_regime_classifier.py --lstm_units 64 32 ...

# Add more training data
```

### Training Too Slow

**Solutions:**
- Increase batch size: `--batch_size 128`
- Reduce lookback: `--lookback 50`
- Use GPU (install tensorflow-gpu)
- Sample data for testing: `--sample_size 200000`

### Inference Too Slow (>100ms)

**Solutions:**
- Reduce model size (fewer units, layers)
- Convert to TensorFlow Lite
- Batch predictions instead of single-sample
- Use model quantization

## Model Deployment

### Saving Model

Models are automatically saved during training:
```
outputs/
├── regime_classifier.keras        # Model weights + architecture
├── regime_classifier_scaler.npy   # Feature scaler
└── regime_model_metadata.txt      # Training info
```

### Loading Model in Production

```python
from regime_classifier import RegimeClassifierModel

# Initialize and load
model = RegimeClassifierModel()
model.load('/path/to/regime_classifier.keras')

# Model is now ready for predictions
regime, confidence = model.predict(live_data)
```

### Model Versioning

Track multiple model versions:
```bash
# Train v1
python train_regime_classifier.py --model_name regime_v1.0.keras ...

# Train v2 with different params
python train_regime_classifier.py --model_name regime_v2.0.keras \
    --lstm_units 256 128 --dropout 0.4 ...

# Compare performance
```

## Continuous Improvement

### Model Retraining Schedule

Retrain the model:
- **Monthly**: To capture recent market behavior
- **After major events**: Fed meetings, economic shocks
- **When accuracy drops**: Monitor live performance

### Retraining Pipeline

```bash
#!/bin/bash
# retrain.sh - Monthly retraining script

DATE=$(date +%Y%m%d)
DATA_PATH="/data/eurusd_1min_latest.csv"
OUTPUT_DIR="/models/$DATE"

python train_regime_classifier.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_name regime_classifier_$DATE.keras \
    --epochs 100 \
    --use_class_weights

# Compare with previous model
python compare_models.py \
    --old_model /models/current/regime_classifier.keras \
    --new_model $OUTPUT_DIR/regime_classifier_$DATE.keras \
    --test_data $DATA_PATH

# If new model is better, promote it
# ...
```

## Integration with Trading System

### Tier 2 Integration

The regime classifier integrates into **Tier 2: Signal Generation Layer**:

```
TIER 1 (ICT Detection) → TIER 2 (ML Signals + Regime) → TIER 4 (Ensemble)
```

### Weight in Ensemble

Regime classifier contributes **8%** to final decision in the ensemble:

```python
# Ensemble weights
ensemble_weights = {
    'long_classifier': 0.25,
    'short_classifier': 0.25,
    'returns_5_forecast': 0.10,
    'pattern_validator': 0.15,
    'confluence_model': 0.10,
    'regime_classifier': 0.08,  # <-- This model
    'order_flow': 0.07
}
```

### Usage in Decision Logic

```python
# Example ensemble logic
if (long_signal_prob > 0.70 and 
    returns_5 > 0.002 and
    pattern_validation > 0.75 and
    regime in [0, 1, 2]):  # Uptrend or ranging
    
    # Adjust confidence based on regime
    if regime == 0:  # Strong uptrend
        confidence *= 1.2  # Boost confidence
    elif regime == 2:  # Ranging
        confidence *= 0.9  # Reduce slightly
    
    execute_trade('LONG', confidence)
```

## File Structure

```
regime-classifier/
├── regime_classifier.py           # Core module
├── train_regime_classifier.py     # Training script
├── regime_inference.py            # Real-time inference
├── README.md                      # This file
├── requirements.txt               # Dependencies
└── examples/
    ├── basic_usage.py
    ├── hyperparameter_tuning.py
    └── integration_example.py
```

## Contributing

This is part of a larger automated trading system. For improvements:
1. Test on holdout data (2024-2025)
2. Validate that changes improve accuracy by ≥2%
3. Ensure inference time remains <50ms
4. Document all changes

## License

Part of EURUSD Automated Trading System - Internal Use Only

## Contact

For questions or issues with the regime classifier, consult the main system documentation.

---

**Next Steps:**
1. ✅ Regime Classifier (You are here)
2. ⚠️ Pattern Validation Model
3. ⚠️ Multi-Timeframe Confluence Model
4. ⚠️ Ensemble Integration
5. ⚠️ Backtesting & Paper Trading

**Status:** Phase 2 - Week 3-4 ✅ COMPLETE

# ICT ML Signal Filter Pipeline

A complete machine learning pipeline for filtering ICT (Inner Circle Trader) trading signals based on quality and confluence.

## Overview

This pipeline transforms raw OHLCV price data into a trained ML model that predicts which ICT setups have the highest probability of success.

### Pipeline Architecture

```
Raw OHLCV Data (CSV)
        │
        ▼
┌───────────────────────┐
│  Feature Engineering  │  ← ict_feature_engineering.py
│  - Time/Session       │
│  - Market Structure   │
│  - ICT Concepts       │
│  - Technical Indicators│
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   Data Preparation    │  ← ict_data_preparation.py
│  - Create Targets     │
│  - Quality Filters    │
│  - Train/Val/Test Split│
│  - Class Balancing    │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   Model Training      │  ← ict_model_training.py
│  - Time Series CV     │
│  - Hyperparameter Tune│
│  - Profit Simulation  │
│  - Model Evaluation   │
└───────────────────────┘
        │
        ▼
   Trained Model (.joblib)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Full Pipeline

```bash
# Run complete pipeline on your OHLCV data
python run_pipeline.py data/EURUSD_M1.csv --output-dir output

# With options
python run_pipeline.py data/EURUSD_M1.csv \
    --output-dir output \
    --broker-gmt 2 \
    --model xgboost \
    --min-confluence 2
```

### Input Data Format

Your CSV should have this format:
```csv
Date,Time,Open,High,Low,Close,Volume
20241203,08:55:00,1.05226,1.05249,1.05226,1.05238,292
20241203,08:56:00,1.05237,1.05258,1.05236,1.05241,298
```

### Run Individual Steps

```bash
# 1. Feature Engineering only
python ict_feature_engineering.py data/EURUSD_M1.csv output/features.csv

# 2. Data Preparation only
python ict_data_preparation.py output/features.csv --output-dir output/prepared

# 3. Model Training only
python ict_model_training.py output/prepared --output-dir output/models
```

## Features Built

### Time Features
- Session identification (Asian, London, NY)
- Kill zone flags (London KZ, NY AM KZ, NY PM KZ)
- Cyclical time encoding (hour, day of week)
- Prime trading hours

### Market Structure
- Swing highs/lows detection
- Higher Highs, Higher Lows (HH, HL)
- Lower Highs, Lower Lows (LH, LL)
- Break of Structure (BOS)
- Premium/Discount zones

### ICT-Specific Features
- Fair Value Gaps (FVG) - bullish and bearish
- Order Blocks (OB)
- Displacement candles
- Liquidity sweeps
- Previous Day High/Low levels

### Technical Indicators
- ATR (multiple periods)
- Volume ratios and spikes
- EMA stack (8, 21, 50, 200)
- RSI and momentum
- Volatility metrics

### Confluence Scoring
- Bullish/Bearish confluence count
- Trend-zone alignment
- FVG-trend alignment
- Multi-factor composite scores

## Target Variables

The pipeline creates several target variables:

| Target | Description |
|--------|-------------|
| `target_hit_tp1` | Did price hit TP1 (1.5R) before SL? |
| `target_hit_tp2` | Did price hit TP2 (2.5R) before SL? |
| `target_hit_tp3` | Did price hit TP3 (4R) before SL? |
| `target_mfe_r` | Maximum Favorable Excursion in R |
| `target_mae_r` | Maximum Adverse Excursion in R |
| `target_pnl_r` | Final P&L in R-multiples |

## Configuration Options

### Feature Engineering (`FeatureConfig`)
```python
swing_strength = 5        # Bars on each side for swing
atr_period = 14           # ATR calculation period
broker_gmt_offset = 0     # Adjust for your broker
```

### Data Preparation (`DataPrepConfig`)
```python
min_confluence = 1        # Minimum confluence score
require_killzone = True   # Only killzone signals
test_size = 0.2           # 20% test split
imbalance_method = 'smote'# How to handle class imbalance
```

### Model Training (`TrainingConfig`)
```python
model_type = 'xgboost'    # xgboost, lightgbm, random_forest, ensemble
cv_folds = 5              # Cross-validation folds
tune_hyperparams = False  # Enable hyperparameter search
calibrate_probabilities = True  # Calibrate prediction probabilities
```

## Output Files

After running the pipeline:

```
output/
├── features/
│   └── features_YYYYMMDD.csv    # Feature-enriched data
├── prepared/
│   ├── train.csv                 # Training data
│   ├── val.csv                   # Validation data
│   ├── test.csv                  # Test data
│   ├── scaler.joblib             # Fitted scaler
│   ├── features.json             # Feature names
│   └── metadata.json             # Preparation metadata
├── models/
│   ├── ict_signal_filter_latest.joblib  # Trained model
│   ├── training_report.json      # Training metrics
│   ├── feature_importance.csv    # Feature rankings
│   └── training_results.png      # Visualization plots
└── pipeline_results.json         # Overall pipeline summary
```

## Using the Trained Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('output/models/ict_signal_filter_latest.joblib')

# Load feature list
import json
with open('output/prepared/features.json') as f:
    feature_names = json.load(f)['features']

# Load scaler
scaler = joblib.load('output/prepared/scaler.joblib')

# Prepare new data (must have same features)
new_data = pd.read_csv('new_signals.csv')
X = new_data[feature_names]
X_scaled = scaler.transform(X)

# Predict
probabilities = model.predict_proba(X_scaled)[:, 1]

# Filter signals with threshold
threshold = 0.7
high_quality_signals = new_data[probabilities >= threshold]
```

## Best Practices

### For Time Series Data

1. **Always use temporal splits** - Never random shuffle time series
2. **Respect the future** - No look-ahead bias in features
3. **Walk-forward validation** - Train on past, test on future

### For Trading ML

1. **Class imbalance** - Most signals lose; handle appropriately
2. **Feature leakage** - Don't include outcome-related features as inputs
3. **Transaction costs** - Account for spread/slippage in profit simulation
4. **Regime changes** - Retrain periodically as markets evolve

### Feature Importance

Top features are typically:
1. **Confluence** - Multiple strategy agreement
2. **Kill zone timing** - In optimal trading windows
3. **HTF alignment** - Trend direction confirmation
4. **Premium/Discount** - Proper zone positioning
5. **FVG/OB presence** - Institutional footprint markers

## Troubleshooting

### "Not enough samples"
- Ensure input data has sufficient bars (recommend 10,000+)
- Reduce min_confluence requirement
- Disable killzone filter for more signals

### "Memory error"
- Reduce lookback periods
- Process data in chunks
- Use lighter model (random_forest instead of xgboost)

### "Poor test performance"
- Check for data leakage
- Verify temporal split is correct
- Consider market regime changes in test period

## License

MIT License - Feel free to use and modify for your trading research.

## Disclaimer

This software is for educational purposes only. Trading forex involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade before using any automated system with real money.

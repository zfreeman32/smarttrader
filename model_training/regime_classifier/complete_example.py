"""
Complete Regime Classifier - Using Pre-Calculated Features
===========================================================

This version uses the features ALREADY in your CSV file.
No feature recalculation needed!
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regime_classifier import (
    RegimeLabeler,
    RegimeClassifierModel,
    plot_training_history,
    plot_confusion_matrix,
    plot_regime_distribution,
    calculate_class_weights
)


def step1_load_data_with_features(filepath: str):
    """Load data that ALREADY has features calculated."""
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA WITH PRE-CALCULATED FEATURES")
    print("="*80)
    
    print(f"\nLoading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Parse datetime
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    df = df.set_index('timestamp').sort_index()
    df.columns = df.columns.str.lower()
    
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def step2_map_features(df: pd.DataFrame):
    """Map your existing features to what the regime classifier expects."""
    print("\n" + "="*80)
    print("STEP 2: MAPPING EXISTING FEATURES")
    print("="*80)
    
    # Feature mapping: your_column_name -> expected_name
    feature_mapping = {
        # ADX and Directional Indicators
        'adx': 'adx_14',
        'plus_di': 'plus_di_14',
        'minus_di': 'minus_di_14',
        
        # ATR
        'atr': 'atr_14',
        'natr': 'atr_pct_14',
        
        # EMAs
        'ema_10': 'ema_10',
        'ema_20': 'ema_20',
        'ema_50': 'ema_50',
        
        # Basic OHLCV
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        
        # Momentum
        'mom': 'momentum_10',
        
        # Volume
        'ad': 'obv',
    }
    
    # Check which features exist
    print("\nChecking available features...")
    available_features = []
    missing_features = []
    
    for your_col, expected_col in feature_mapping.items():
        if your_col in df.columns:
            available_features.append((your_col, expected_col))
            print(f"  ✅ Found: {your_col} -> {expected_col}")
        else:
            missing_features.append(your_col)
    
    if missing_features:
        print(f"\n⚠️  Missing: {missing_features}")
    
    # Create a new dataframe with mapped features
    df_mapped = pd.DataFrame(index=df.index)
    
    for your_col, expected_col in available_features:
        df_mapped[expected_col] = df[your_col]
    
    # Calculate derived features from what we have
    print("\nCalculating derived features...")
    
    # ROC if we have close prices
    if 'close' in df_mapped.columns:
        df_mapped['roc_5'] = df_mapped['close'].pct_change(5) * 100
        df_mapped['roc_10'] = df_mapped['close'].pct_change(10) * 100
        df_mapped['roc_20'] = df_mapped['close'].pct_change(20) * 100
        df_mapped['roc_50'] = df_mapped['close'].pct_change(50) * 100
        print("  ✅ Calculated ROC features")
    
    # EMA ratios
    if 'close' in df_mapped.columns:
        if 'ema_10' in df_mapped.columns:
            df_mapped['price_to_ema_10'] = (df_mapped['close'] - df_mapped['ema_10']) / df_mapped['ema_10'] * 100
        if 'ema_20' in df_mapped.columns:
            df_mapped['price_to_ema_20'] = (df_mapped['close'] - df_mapped['ema_20']) / df_mapped['ema_20'] * 100
        if 'ema_50' in df_mapped.columns:
            df_mapped['price_to_ema_50'] = (df_mapped['close'] - df_mapped['ema_50']) / df_mapped['ema_50'] * 100
        print("  ✅ Calculated EMA ratios")
    
    # EMA crossovers
    if 'ema_10' in df_mapped.columns and 'ema_20' in df_mapped.columns:
        df_mapped['ema_10_20_cross'] = np.where(df_mapped['ema_10'] > df_mapped['ema_20'], 1, -1)
    if 'ema_20' in df_mapped.columns and 'ema_50' in df_mapped.columns:
        df_mapped['ema_20_50_cross'] = np.where(df_mapped['ema_20'] > df_mapped['ema_50'], 1, -1)
        print("  ✅ Calculated EMA crossovers")
    
    # Volume ratio
    if 'volume' in df_mapped.columns:
        df_mapped['volume_ma_20'] = df_mapped['volume'].rolling(20).mean()
        df_mapped['volume_ratio'] = df_mapped['volume'] / df_mapped['volume_ma_20']
        print("  ✅ Calculated volume ratio")
    
    # Bollinger Bands if we have close
    if 'close' in df_mapped.columns:
        bb_mid = df_mapped['close'].rolling(20).mean()
        bb_std = df_mapped['close'].rolling(20).std()
        df_mapped['bb_upper_20'] = bb_mid + 2 * bb_std
        df_mapped['bb_lower_20'] = bb_mid - 2 * bb_std
        df_mapped['bb_width_20'] = (df_mapped['bb_upper_20'] - df_mapped['bb_lower_20']) / bb_mid * 100
        df_mapped['bb_position_20'] = (df_mapped['close'] - df_mapped['bb_lower_20']) / (df_mapped['bb_upper_20'] - df_mapped['bb_lower_20'])
        print("  ✅ Calculated Bollinger Bands")
    
    # Swing range
    if 'high' in df_mapped.columns and 'low' in df_mapped.columns:
        df_mapped['swing_high'] = df_mapped['high'].rolling(20).max()
        df_mapped['swing_low'] = df_mapped['low'].rolling(20).min()
        df_mapped['swing_range'] = df_mapped['swing_high'] - df_mapped['swing_low']
        df_mapped['price_in_range'] = (df_mapped['close'] - df_mapped['swing_low']) / df_mapped['swing_range']
        print("  ✅ Calculated swing range")
    
    # Additional ADX periods if base ADX exists
    if 'adx_14' in df_mapped.columns:
        df_mapped['adx_20'] = df_mapped['adx_14']  # Use same value if 20-period not available
        df_mapped['plus_di_20'] = df_mapped.get('plus_di_14', 0)
        df_mapped['minus_di_20'] = df_mapped.get('minus_di_14', 0)
    
    # Additional ATR periods
    if 'atr_14' in df_mapped.columns:
        df_mapped['atr_20'] = df_mapped['atr_14']
        df_mapped['atr_pct_20'] = df_mapped.get('atr_pct_14', 0)
    
    # Additional momentum
    if 'close' in df_mapped.columns:
        df_mapped['momentum_20'] = df_mapped['close'].diff(20)
    
    print(f"\n✅ Created feature dataframe: {len(df_mapped.columns)} features")
    
    return df_mapped


def step3_label_regimes(df: pd.DataFrame):
    """Label regimes using ADX/ATR rules."""
    print("\n" + "="*80)
    print("STEP 3: LABELING REGIMES")
    print("="*80)
    
    labeler = RegimeLabeler()
    df = labeler.label_regime(df)
    
    return df


def step4_train_model(df: pd.DataFrame, quick_test: bool = False):
    """Train the regime classifier."""
    print("\n" + "="*80)
    print("STEP 4: TRAINING REGIME CLASSIFIER")
    print("="*80)
    
    # Define features to use (only those that should exist)
    feature_cols = [
        'adx_14', 'adx_20',
        'plus_di_14', 'minus_di_14',
        'atr_pct_14', 'atr_pct_20',
        'bb_width_20', 'bb_position_20',
        'roc_5', 'roc_10', 'roc_20', 'roc_50',
        'momentum_10', 'momentum_20',
        'volume_ratio',
        'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50',
        'ema_10_20_cross', 'ema_20_50_cross',
        'price_in_range'
    ]
    
    # Check which features actually exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Missing features (will be excluded): {missing_features}")
    
    print(f"\nUsing {len(available_features)} features for training")
    
    # Check data before NaN removal
    print(f"\nData before NaN removal: {len(df):,} rows")
    
    # Show NaN counts for each feature
    print("\nNaN counts per feature:")
    nan_counts = df[available_features + ['regime']].isna().sum()
    for feat, count in nan_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {feat:30s}: {count:,} ({pct:.1f}%)")
    
    # Remove NaN
    df_clean = df[available_features + ['regime']].dropna()
    print(f"\nData after NaN removal: {len(df_clean):,} rows")
    
    if len(df_clean) == 0:
        print("\n❌ ERROR: No valid data after NaN removal")
        print("\nShowing first few rows of data:")
        print(df[available_features + ['regime']].head(20))
        raise ValueError("No valid training data")
    
    if len(df_clean) < 1000:
        print(f"\n⚠️  WARNING: Only {len(df_clean):,} samples. May not be enough.")
    
    # Adjust lookback based on data size
    lookback = 50 if quick_test else 100
    if len(df_clean) < lookback * 10:
        lookback = max(20, len(df_clean) // 20)
        print(f"⚠️  Adjusted lookback to {lookback} due to limited data")
    
    print(f"\nUsing lookback period: {lookback}")
    
    # Create model
    model = RegimeClassifierModel(
        lookback_period=lookback,
        lstm_units=[64, 32] if quick_test else [128, 64],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X, y = model.prepare_sequences(df_clean, available_features, target_col='regime')
    
    print(f"✅ Created {len(X):,} sequences")
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"\nTraining set:   {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set:       {len(X_test):,} samples")
    
    # Calculate class weights
    y_train_classes = np.argmax(y_train, axis=1)
    class_weights = calculate_class_weights(y_train_classes)
    
    # Train
    print("\nTraining model...")
    epochs = 10 if quick_test else 50
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=64,
        class_weights=class_weights,
        verbose=1
    )
    
    # Plot training history
    try:
        plot_training_history(history)
        print("✅ Training history plot saved")
    except Exception as e:
        print(f"⚠️  Could not save training plot: {e}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(results['confusion_matrix'])
        print("✅ Confusion matrix plot saved")
    except Exception as e:
        print(f"⚠️  Could not save confusion matrix: {e}")
    
    # Save model
    model_path = './outputs/regime_classifier_eurusd.keras'
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    return model, results, model_path


def main():
    """Run complete training pipeline."""
    print("="*80)
    print("REGIME CLASSIFIER - USING YOUR PRE-CALCULATED FEATURES")
    print("="*80)
    print("\nThis version uses features ALREADY in your CSV")
    print("No recalculation needed!")
    print("="*80)
    
    try:
        # Configuration
        DATA_PATH = r'C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\data\currency_data\sampled\EURUSD_1min_sampled_features.csv'
        QUICK_TEST = True  # Set to False for full training
        
        # Step 1: Load data
        df = step1_load_data_with_features(DATA_PATH)
        
        # Step 2: Map features
        df_mapped = step2_map_features(df)
        
        # Step 3: Label regimes
        df_labeled = step3_label_regimes(df_mapped)
        
        # Step 4: Train model
        model, results, model_path = step4_train_model(df_labeled, quick_test=QUICK_TEST)
        
        # Final summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE! ✅")
        print("="*80)
        print("\nModel Performance:")
        print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
        print(f"  Test Precision: {results['test_precision']:.4f}")
        print(f"  Test Recall:    {results['test_recall']:.4f}")
        
        print(f"\nModel saved to: {model_path}")
        
        if results['test_accuracy'] >= 0.75:
            print("\n✅ Excellent performance!")
        elif results['test_accuracy'] >= 0.70:
            print("\n✅ Good performance!")
        else:
            print("\n⚠️  Performance below target.")
            print("   Consider training longer or checking data quality.")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
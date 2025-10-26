"""
Regime Classifier Training Script
==================================

Complete end-to-end training pipeline for the Market Regime Classifier.
This script demonstrates how to:
1. Load EURUSD data
2. Engineer features
3. Auto-label regimes
4. Train the LSTM model
5. Evaluate performance
6. Save the trained model

Usage:
    python train_regime_classifier.py --data_path eurusd_1min.csv --epochs 100
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Import regime classifier modules
from regime_classifier import (
    RegimeFeatureEngineering,
    RegimeLabeler,
    RegimeClassifierModel,
    plot_training_history,
    plot_confusion_matrix,
    plot_regime_distribution,
    calculate_class_weights
)


def load_eurusd_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load EURUSD data from CSV file.
    
    Args:
        filepath: Path to CSV file
        sample_size: Optional number of rows to sample (for testing)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading data from {filepath}...")
    
    # Try different timestamp column names
    timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
    
    df = pd.read_csv(filepath)
    
    # Find timestamp column
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
        df = df.sort_index()
    
    # Standardize column names (lowercase)
    df.columns = df.columns.str.lower()
    
    # Verify required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sample data if requested (for testing)
    if sample_size:
        if len(df) > sample_size:
            print(f"Sampling {sample_size:,} rows from {len(df):,} total rows")
            df = df.iloc[-sample_size:]
    
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def select_feature_columns(df: pd.DataFrame) -> list:
    """
    Select optimal features for regime classification.
    
    Args:
        df: DataFrame with all engineered features
        
    Returns:
        List of feature column names
    """
    feature_cols = [
        # Trend strength
        'adx_14', 'adx_20',
        'plus_di_14', 'minus_di_14',
        
        # Volatility
        'atr_pct_14', 'atr_pct_20',
        'bb_width_20', 'bb_position_20',
        
        # Momentum
        'roc_5', 'roc_10', 'roc_20', 'roc_50',
        'momentum_10', 'momentum_20',
        
        # Volume
        'volume_ratio', 'obv_ma_20',
        
        # Trend position
        'price_to_ema_10', 'price_to_ema_20', 
        'price_to_ema_50', 'price_to_ema_100',
        
        # Crossovers
        'ema_10_20_cross', 'ema_20_50_cross', 'ema_50_200_cross',
        
        # Range position
        'price_in_range'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    
    print(f"\nSelected {len(available_features)} features for training:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i:2d}. {feat}")
    
    return available_features


def prepare_train_val_test_split(X: np.ndarray, y: np.ndarray,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15):
    """
    Split data into train/validation/test sets for time series.
    Maintains temporal ordering (no shuffling).
    
    Args:
        X: Feature array
        y: Target array
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print("\nData Split:")
    print(f"  Training:   {len(X_train):,} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")
    print(f"  Total:      {n_samples:,} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main(args):
    """Main training pipeline."""
    
    print("="*80)
    print("MARKET REGIME CLASSIFIER - TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    df = load_eurusd_data(args.data_path, sample_size=args.sample_size)
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    feature_engineer = RegimeFeatureEngineering()
    df = feature_engineer.engineer_features(df)
    
    print(f"DataFrame shape after feature engineering: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    # =========================================================================
    # STEP 3: Regime Labeling
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: REGIME LABELING")
    print("="*80)
    
    labeler = RegimeLabeler(
        adx_trend_threshold=args.adx_trend,
        adx_ranging_threshold=args.adx_ranging,
        atr_high_vol_percentile=args.atr_high_pct,
        atr_low_vol_percentile=args.atr_low_pct
    )
    
    df = labeler.label_regime(df)
    
    # Plot regime distribution
    if args.plot:
        plot_regime_distribution(df)
    
    # =========================================================================
    # STEP 4: Feature Selection
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: FEATURE SELECTION")
    print("="*80)
    
    feature_cols = select_feature_columns(df)
    
    # Remove rows with NaN values
    df_clean = df.dropna()
    print(f"\nRows after removing NaN: {len(df_clean):,} (removed {len(df) - len(df_clean):,})")
    
    # =========================================================================
    # STEP 5: Sequence Preparation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: PREPARING SEQUENCES")
    print("="*80)
    
    model = RegimeClassifierModel(
        lookback_period=args.lookback,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
    )
    
    print(f"Creating sequences with lookback period: {args.lookback}")
    X, y = model.prepare_sequences(df_clean, feature_cols, target_col='regime')
    
    # =========================================================================
    # STEP 6: Train/Val/Test Split
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SPLITTING DATA")
    print("="*80)
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_train_val_test_split(
        X, y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # =========================================================================
    # STEP 7: Calculate Class Weights
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: CALCULATING CLASS WEIGHTS")
    print("="*80)
    
    y_train_classes = np.argmax(y_train, axis=1)
    class_weights = calculate_class_weights(y_train_classes) if args.use_class_weights else None
    
    # =========================================================================
    # STEP 8: Model Training
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: TRAINING MODEL")
    print("="*80)
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights,
        verbose=1
    )
    
    # Plot training history
    if args.plot:
        plot_training_history(history)
    
    # =========================================================================
    # STEP 9: Model Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 9: EVALUATING MODEL")
    print("="*80)
    
    results = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    if args.plot:
        plot_confusion_matrix(results['confusion_matrix'], normalize=True)
    
    # =========================================================================
    # STEP 10: Save Model
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 10: SAVING MODEL")
    print("="*80)
    
    model_path = output_dir / args.model_name
    model.save(str(model_path))
    
    # Save training metadata
    metadata = {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_path': args.data_path,
        'total_samples': len(df_clean),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'lookback_period': args.lookback,
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout,
        'learning_rate': args.learning_rate,
        'test_accuracy': float(results['test_accuracy']),
        'test_precision': float(results['test_precision']),
        'test_recall': float(results['test_recall']),
        'feature_columns': feature_cols
    }
    
    metadata_path = output_dir / 'regime_model_metadata.txt'
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metadata saved to {metadata_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  Test Precision: {results['test_precision']:.4f}")
    print(f"  Test Recall:    {results['test_recall']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"Outputs saved to: {output_dir}")
    
    if results['test_accuracy'] >= 0.75:
        print("\n✅ TARGET ACCURACY ACHIEVED (≥75%)")
    else:
        print(f"\n⚠️  Accuracy below target. Consider:")
        print("   - Increasing training data")
        print("   - Tuning hyperparameters")
        print("   - Adding more features")
        print("   - Adjusting regime labeling thresholds")
    
    print("="*80)
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Market Regime Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to EURUSD CSV file')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of rows to sample (for testing)')
    
    # Regime labeling parameters
    parser.add_argument('--adx_trend', type=float, default=25.0,
                       help='ADX threshold for trending market')
    parser.add_argument('--adx_ranging', type=float, default=20.0,
                       help='ADX threshold for ranging market')
    parser.add_argument('--atr_high_pct', type=float, default=75.0,
                       help='ATR percentile for high volatility')
    parser.add_argument('--atr_low_pct', type=float, default=25.0,
                       help='ATR percentile for low volatility')
    
    # Model architecture
    parser.add_argument('--lookback', type=int, default=100,
                       help='Lookback period for sequences')
    parser.add_argument('--lstm_units', type=int, nargs='+', default=[128, 64],
                       help='LSTM layer sizes (e.g., 128 64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='/mnt/user-data/outputs',
                       help='Output directory')
    parser.add_argument('--model_name', type=str, default='regime_classifier.keras',
                       help='Model filename')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Run training
    try:
        model, results = main(args)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

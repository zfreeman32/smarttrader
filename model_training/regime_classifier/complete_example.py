"""
Complete Regime Classifier Example
===================================

End-to-end demonstration of the Market Regime Classifier:
1. Generate synthetic test data
2. Train the regime classifier
3. Evaluate performance
4. Make real-time predictions
5. Integrate with trading logic

This script serves as both a tutorial and a testing pipeline.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_test_data import SyntheticEURUSDGenerator
from regime_classifier import (
    RegimeFeatureEngineering,
    RegimeLabeler,
    RegimeClassifierModel,
    plot_training_history,
    plot_confusion_matrix,
    plot_regime_distribution,
    calculate_class_weights
)
from regime_inference import RealtimeRegimePredictor


def step1_generate_data(n_samples: int = 100000):
    """
    Step 1: Generate synthetic EURUSD data for testing.
    """
    print("\n" + "="*80)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("="*80)
    
    generator = SyntheticEURUSDGenerator(initial_price=1.1000, seed=42)
    df = generator.generate_dataset(n_samples=n_samples)
    
    # Save to CSV
    output_path = '/home/claude/eurusd_test_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Data saved to: {output_path}")
    
    return df, output_path


def step2_prepare_and_label(df: pd.DataFrame):
    """
    Step 2: Engineer features and label regimes.
    """
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING & REGIME LABELING")
    print("="*80)
    
    # Feature engineering
    print("\nEngineering features...")
    feature_engineer = RegimeFeatureEngineering()
    df = feature_engineer.engineer_features(df)
    print(f"✅ Features engineered: {len(df.columns)} total columns")
    
    # Label regimes
    print("\nLabeling regimes...")
    labeler = RegimeLabeler()
    df = labeler.label_regime(df)
    
    # Compare predicted labels with true labels (if available)
    if 'true_regime' in df.columns:
        # Calculate accuracy of rule-based labeling
        df_clean = df.dropna()
        agreement = (df_clean['regime'] == df_clean['true_regime']).mean()
        print(f"\n📊 Rule-based labeling agreement with synthetic truth: {agreement:.2%}")
        print("   (This shows how well ADX/ATR rules capture regime characteristics)")
    
    return df


def step3_train_model(df: pd.DataFrame, quick_test: bool = False):
    """
    Step 3: Train the regime classifier model.
    """
    print("\n" + "="*80)
    print("STEP 3: TRAINING REGIME CLASSIFIER")
    print("="*80)
    
    # Select features
    feature_cols = [
        'adx_14', 'adx_20',
        'plus_di_14', 'minus_di_14',
        'atr_pct_14', 'atr_pct_20',
        'bb_width_20', 'bb_position_20',
        'roc_5', 'roc_10', 'roc_20', 'roc_50',
        'momentum_10', 'momentum_20',
        'volume_ratio',
        'price_to_ema_10', 'price_to_ema_20', 
        'price_to_ema_50', 'price_to_ema_100',
        'ema_10_20_cross', 'ema_20_50_cross', 'ema_50_200_cross',
        'price_in_range'
    ]
    
    # Remove NaN
    df_clean = df.dropna()
    print(f"Data rows after NaN removal: {len(df_clean):,}")
    
    # Create sequences
    print("\nPreparing sequences...")
    lookback = 50 if quick_test else 100
    model = RegimeClassifierModel(
        lookback_period=lookback,
        lstm_units=[64, 32] if quick_test else [128, 64],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    X, y = model.prepare_sequences(df_clean, feature_cols, target_col='regime')
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Training set:   {len(X_train):,} samples")
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
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(X_test, y_test)
    
    # Save model
    model_path = '/mnt/user-data/outputs/regime_classifier_test.keras'
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    return model, results


def step4_real_time_inference(model_path: str, df: pd.DataFrame):
    """
    Step 4: Demonstrate real-time inference.
    """
    print("\n" + "="*80)
    print("STEP 4: REAL-TIME INFERENCE")
    print("="*80)
    
    # Initialize predictor
    predictor = RealtimeRegimePredictor(
        model_path=model_path,
        lookback_period=100,
        min_confidence=0.70
    )
    
    # Make predictions on recent data
    print("\nMaking predictions on recent data...")
    recent_data = df.iloc[-500:]  # Last 500 bars
    
    result = predictor.predict(recent_data, return_probabilities=True)
    
    print(f"\n✅ Prediction complete!")
    print(f"   Regime: {result['regime_name']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Prediction time: {result['prediction_time_ms']:.1f}ms")
    
    if result.get('probabilities'):
        print("\n   Probability distribution:")
        for regime, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"      {regime:20s}: {bar:50s} {prob:.2%}")
    
    return predictor, result


def step5_trading_integration(predictor: RealtimeRegimePredictor):
    """
    Step 5: Demonstrate integration with trading logic.
    """
    print("\n" + "="*80)
    print("STEP 5: TRADING SYSTEM INTEGRATION")
    print("="*80)
    
    # Get current regime status
    print("\n1. Current Market Regime:")
    predictor.print_status()
    
    # Check pattern suitability
    print("\n2. Pattern Suitability Analysis:")
    patterns = {
        'order_block': 'Order Block (institutional zones)',
        'fvg': 'Fair Value Gap',
        'liquidity_sweep': 'Liquidity Sweep',
        'breakout': 'Breakout Pattern'
    }
    
    for pattern_key, pattern_name in patterns.items():
        suitable = predictor.is_regime_suitable_for_pattern(pattern_key)
        emoji = "✅" if suitable else "❌"
        print(f"   {emoji} {pattern_name:35s}: {'SUITABLE' if suitable else 'NOT RECOMMENDED'}")
    
    # Get trading bias
    print("\n3. Trading Bias:")
    bias = predictor.get_trading_bias()
    print(f"   Direction: {bias['bias'].upper()}")
    print(f"   Strength:  {bias['strength']:.2f} (0=neutral, 1=strong)")
    print(f"   Note:      {bias['description']}")
    
    # Regime trend analysis
    print("\n4. Regime Stability Analysis:")
    trend = predictor.get_regime_trend(lookback=10)
    if trend['status'] == 'ok':
        print(f"   Stability Score: {trend['stability_score']:.2%}")
        print(f"   Regime Changes:  {trend['regime_changes']} in last 10 predictions")
        print(f"   Status:          {'STABLE ✅' if trend['is_stable'] else 'UNSTABLE ⚠️'}")
        print(f"   Dominant Regime: {trend['dominant_regime']}")
    
    # Performance metrics
    print("\n5. Predictor Performance:")
    perf = predictor.get_performance_stats()
    if perf['status'] == 'ok':
        print(f"   Total Predictions:  {perf['total_predictions']:,}")
        print(f"   Avg Latency:        {perf['avg_prediction_time_ms']:.1f}ms")
        print(f"   Max Latency:        {perf['max_prediction_time_ms']:.1f}ms")
        print(f"   P95 Latency:        {perf['p95_prediction_time_ms']:.1f}ms")
        print(f"   Within 50ms target: {perf['within_50ms_target']:.1f}%")
        
        if perf['avg_prediction_time_ms'] < 50:
            print("   ✅ MEETS LATENCY TARGET (<50ms)")
        else:
            print("   ⚠️  ABOVE LATENCY TARGET")


def step6_example_trading_decision():
    """
    Step 6: Example of using regime in trading decision.
    """
    print("\n" + "="*80)
    print("STEP 6: EXAMPLE TRADING DECISION LOGIC")
    print("="*80)
    
    print("\nExample: How regime affects trading decisions\n")
    
    # Pseudo-code example
    example_code = '''
def should_take_trade(signal_type, regime, confidence):
    """
    Example trading decision logic incorporating regime.
    """
    # Base requirements
    if confidence < 0.70:
        return False, "Insufficient confidence"
    
    # Pattern-specific regime requirements
    if signal_type == "LONG":
        # Long trades
        if regime in [0, 1]:  # Uptrend
            position_size = 1.0
            return True, f"Full size long in {regime_name[regime]}"
        
        elif regime == 2:  # Ranging
            position_size = 0.5
            return True, "Half size long in ranging market"
        
        else:  # Downtrend or high vol
            return False, "Regime not suitable for long"
    
    elif signal_type == "SHORT":
        # Short trades
        if regime in [3, 4]:  # Downtrend
            position_size = 1.0
            return True, f"Full size short in {regime_name[regime]}"
        
        elif regime == 2:  # Ranging
            position_size = 0.5
            return True, "Half size short in ranging market"
        
        else:  # Uptrend or high vol
            return False, "Regime not suitable for short"
    
    return False, "Unknown signal type"


# Example usage:
current_regime = predict_regime(live_data)
signal = detect_ict_pattern(live_data)

if signal['type'] == 'order_block_long':
    can_trade, reason = should_take_trade(
        "LONG", 
        current_regime['regime'],
        current_regime['confidence']
    )
    
    if can_trade:
        execute_trade('LONG', position_size)
    else:
        log_rejection(signal, reason)
'''
    
    print(example_code)


def main():
    """
    Run complete end-to-end example.
    """
    print("="*80)
    print("MARKET REGIME CLASSIFIER - COMPLETE EXAMPLE")
    print("="*80)
    print("\nThis example demonstrates the entire regime classifier workflow:")
    print("  1. Generate synthetic test data")
    print("  2. Engineer features and label regimes")
    print("  3. Train the LSTM classifier")
    print("  4. Make real-time predictions")
    print("  5. Integrate with trading logic")
    print("  6. Show example trading decisions")
    print("\n" + "="*80)
    
    # Configuration
    QUICK_TEST = True  # Set to False for full training
    N_SAMPLES = 50000 if QUICK_TEST else 100000
    
    if QUICK_TEST:
        print("\n⚡ QUICK TEST MODE (fast training, less data)")
        print("   Set QUICK_TEST=False for full training")
    
    try:
        # Step 1: Generate data
        df, data_path = step1_generate_data(n_samples=N_SAMPLES)
        
        # Step 2: Prepare and label
        df = step2_prepare_and_label(df)
        
        # Step 3: Train model
        model, results = step3_train_model(df, quick_test=QUICK_TEST)
        
        # Step 4: Real-time inference
        model_path = '/mnt/user-data/outputs/regime_classifier_test.keras'
        predictor, result = step4_real_time_inference(model_path, df)
        
        # Step 5: Trading integration
        step5_trading_integration(predictor)
        
        # Step 6: Example trading decision
        step6_example_trading_decision()
        
        # Final summary
        print("\n" + "="*80)
        print("EXAMPLE COMPLETE! ✅")
        print("="*80)
        print("\nKey Files Generated:")
        print(f"  - Test data:  {data_path}")
        print(f"  - Model:      {model_path}")
        print(f"  - Plots:      /mnt/user-data/outputs/*.png")
        
        print("\nNext Steps:")
        print("  1. Replace synthetic data with real EURUSD data")
        print("  2. Train on 2+ years of historical data")
        print("  3. Integrate into your trading system")
        print("  4. Monitor performance in paper trading")
        
        print("\nModel Performance:")
        print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
        print(f"  Test Precision: {results['test_precision']:.4f}")
        print(f"  Test Recall:    {results['test_recall']:.4f}")
        
        if results['test_accuracy'] >= 0.75:
            print("\n  ✅ Meets target accuracy (≥75%)")
        else:
            print(f"\n  ⚠️  Below target. Train on more data for better results.")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

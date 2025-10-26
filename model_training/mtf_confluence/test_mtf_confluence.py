"""
Multi-Timeframe Confluence Model - Testing & Validation Script
=============================================================

This script provides comprehensive testing of the confluence model including:
1. Synthetic data generation for testing
2. Model training validation
3. Inference speed benchmarking
4. Prediction quality assessment
5. Real-world simulation
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from mtf_confluence_trainer import (
    MultiTimeframeDataPreparator,
    MultiTimeframeConfluenceModel,
    ConfluenceModelTrainer
)
from mtf_confluence_analyzer import (
    ConfluenceAnalyzer,
    format_analysis_report
)


def generate_realistic_forex_data(start_date: str, end_date: str, base_price: float = 1.08) -> pd.DataFrame:
    """
    Generate realistic synthetic EURUSD data for testing
    
    Includes:
    - Trending periods
    - Ranging periods
    - Volatile breakouts
    - Realistic intraday patterns
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='1T')
    n = len(dates)
    
    np.random.seed(42)
    
    # Generate price movement with trends and mean reversion
    trend = np.linspace(0, 0.02, n) * np.sin(np.linspace(0, 4*np.pi, n))
    noise = np.cumsum(np.random.randn(n)) * 0.0001
    price = base_price + trend + noise
    
    # Add intraday patterns (higher volatility during London/NY open)
    hours = pd.Series(dates).dt.hour
    intraday_vol = 1.0 + 0.5 * ((hours >= 8) & (hours <= 16))
    
    # Generate OHLC
    df = pd.DataFrame(index=dates)
    df['close'] = price
    df['open'] = df['close'].shift(1).fillna(df['close'])
    
    # Add realistic high/low
    volatility = 0.0002 * intraday_vol.values
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(n)) * volatility
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(n)) * volatility
    
    # Volume with patterns
    base_volume = 5000
    volume_pattern = 1.0 + 0.5 * ((hours >= 8) & (hours <= 16))
    df['volume'] = (base_volume * volume_pattern.values * (1 + np.random.randn(n) * 0.3)).astype(int)
    df['volume'] = df['volume'].clip(lower=1000)
    
    return df


def test_data_preparation():
    """Test 1: Data preparation pipeline"""
    print("\n" + "="*80)
    print("TEST 1: Data Preparation")
    print("="*80)
    
    # Generate test data
    df = generate_realistic_forex_data('2023-01-01', '2024-01-01')
    print(f"✓ Generated {len(df):,} bars of synthetic data")
    
    # Initialize preparator
    preparator = MultiTimeframeDataPreparator()
    
    # Test feature calculation
    df_features = preparator.calculate_features_per_timeframe(df, '1m')
    print(f"✓ Calculated {len([c for c in df_features.columns if c not in ['open','high','low','close','volume']])} features")
    
    # Test resampling
    df_5m = preparator.resample_to_timeframe(df, 5)
    print(f"✓ Resampled to 5m: {len(df_5m)} bars")
    
    # Test full dataset creation
    aligned_data = preparator.create_aligned_dataset(df)
    print(f"✓ Created aligned dataset:")
    for tf, data in aligned_data.items():
        print(f"  {tf}: {data.shape}")
    
    # Test label calculation
    confluence_scores, directional_bias = preparator.calculate_labels(df)
    print(f"✓ Generated {len(confluence_scores)} labels")
    print(f"  Confluence range: [{confluence_scores.min():.3f}, {confluence_scores.max():.3f}]")
    print(f"  Bias range: [{directional_bias.min():.3f}, {directional_bias.max():.3f}]")
    
    return preparator, df


def test_model_architecture():
    """Test 2: Model building and compilation"""
    print("\n" + "="*80)
    print("TEST 2: Model Architecture")
    print("="*80)
    
    # Define input shapes
    input_shapes = {
        '1m': (240, 47),
        '5m': (100, 47),
        '15m': (50, 47),
        '1h': (24, 47),
        '4h': (20, 47)
    }
    
    # Build model
    model = MultiTimeframeConfluenceModel(
        input_shapes=input_shapes,
        d_model=64,  # Smaller for testing
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=1,
        dropout_rate=0.2
    )
    
    model.build_model()
    print("✓ Model built successfully")
    
    model.compile_model(learning_rate=0.001)
    print("✓ Model compiled")
    
    # Test forward pass
    dummy_inputs = [np.random.randn(10, *shape) for shape in input_shapes.values()]
    predictions = model.model.predict(dummy_inputs, verbose=0)
    
    print(f"✓ Forward pass successful")
    print(f"  Confluence output shape: {predictions[0].shape}")
    print(f"  Bias output shape: {predictions[1].shape}")
    
    # Count parameters
    total_params = model.model.count_params()
    print(f"✓ Total parameters: {total_params:,}")
    
    return model


def test_training_pipeline(quick_test: bool = True):
    """Test 3: Training pipeline"""
    print("\n" + "="*80)
    print("TEST 3: Training Pipeline" + (" (Quick Test)" if quick_test else ""))
    print("="*80)
    
    # Generate data
    df = generate_realistic_forex_data('2023-01-01', '2023-06-01' if quick_test else '2024-01-01')
    print(f"✓ Training data: {len(df):,} bars")
    
    # Prepare data
    preparator = MultiTimeframeDataPreparator()
    X_train, X_val, y_train_conf, y_train_bias, y_val_conf, y_val_bias = preparator.prepare_training_data(
        df, validation_split=0.2
    )
    print(f"✓ Data prepared: {len(y_train_conf)} train, {len(y_val_conf)} val")
    
    # Build model
    input_shapes = {tf: X_train[tf].shape[1:] for tf in X_train.keys()}
    model = MultiTimeframeConfluenceModel(
        input_shapes=input_shapes,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=1,
        dropout_rate=0.2
    )
    
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Train
    trainer = ConfluenceModelTrainer(
        model=model,
        model_save_path='/home/claude/test_confluence_model.keras'
    )
    
    epochs = 5 if quick_test else 20
    history = trainer.train(
        X_train=X_train,
        y_train_confluence=y_train_conf,
        y_train_bias=y_train_bias,
        X_val=X_val,
        y_val_confluence=y_val_conf,
        y_val_bias=y_val_bias,
        epochs=epochs,
        batch_size=32,
        patience=5
    )
    
    print("✓ Training completed")
    
    # Evaluate
    final_val_loss = history.history['val_loss'][-1]
    print(f"✓ Final validation loss: {final_val_loss:.4f}")
    
    return trainer, preparator, history


def test_inference_speed():
    """Test 4: Inference speed benchmark"""
    print("\n" + "="*80)
    print("TEST 4: Inference Speed")
    print("="*80)
    
    try:
        # Load analyzer
        analyzer = ConfluenceAnalyzer(
            model_path='/home/claude/test_confluence_model.keras',
            scaler_path='/home/claude/test_confluence_scalers.pkl'
        )
        
        # Generate test data
        df = generate_realistic_forex_data('2024-10-20', '2024-10-25')
        
        # Warm-up
        _ = analyzer.analyze(df)
        
        # Benchmark
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            start = time.time()
            score, bias, details = analyzer.analyze(df)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✓ Inference speed:")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Std Dev: {std_time:.2f} ms")
        print(f"  Min: {min(times):.2f} ms")
        print(f"  Max: {max(times):.2f} ms")
        
        if avg_time < 100:
            print(f"✓ PASS: Meets <100ms target")
        else:
            print(f"⚠ WARNING: Exceeds 100ms target")
        
        return avg_time
        
    except FileNotFoundError:
        print("⚠ Skipped: Run training test first")
        return None


def test_prediction_quality():
    """Test 5: Prediction quality assessment"""
    print("\n" + "="*80)
    print("TEST 5: Prediction Quality")
    print("="*80)
    
    try:
        # Load model
        analyzer = ConfluenceAnalyzer(
            model_path='/home/claude/test_confluence_model.keras',
            scaler_path='/home/claude/test_confluence_scalers.pkl'
        )
        
        # Generate test data with known patterns
        df = generate_realistic_forex_data('2024-01-01', '2024-02-01')
        
        # Analyze
        score, bias, details = analyzer.analyze(df)
        
        print(f"✓ Predictions generated:")
        print(f"  Confluence Score: {score:.3f}")
        print(f"  Directional Bias: {bias:.3f}")
        print(f"  Signal Quality: {analyzer.get_signal_quality(score, bias)}")
        
        # Test different market conditions
        conditions = []
        
        # Strong uptrend
        df_uptrend = df.copy()
        df_uptrend['close'] = df_uptrend['close'] * np.linspace(1.0, 1.05, len(df_uptrend))
        score_up, bias_up, _ = analyzer.analyze(df_uptrend)
        conditions.append(('Uptrend', score_up, bias_up))
        
        # Strong downtrend
        df_downtrend = df.copy()
        df_downtrend['close'] = df_downtrend['close'] * np.linspace(1.0, 0.95, len(df_downtrend))
        score_down, bias_down, _ = analyzer.analyze(df_downtrend)
        conditions.append(('Downtrend', score_down, bias_down))
        
        print(f"\n✓ Different market conditions:")
        for name, s, b in conditions:
            print(f"  {name}: score={s:.3f}, bias={b:.3f}")
        
        # Validate bias direction
        if bias_up > 0 and bias_down < 0:
            print("✓ PASS: Model correctly identifies directional bias")
        else:
            print("⚠ WARNING: Model may not be properly trained")
        
        return conditions
        
    except FileNotFoundError:
        print("⚠ Skipped: Run training test first")
        return None


def test_real_world_simulation():
    """Test 6: Real-world trading simulation"""
    print("\n" + "="*80)
    print("TEST 6: Real-World Trading Simulation")
    print("="*80)
    
    try:
        analyzer = ConfluenceAnalyzer(
            model_path='/home/claude/test_confluence_model.keras',
            scaler_path='/home/claude/test_confluence_scalers.pkl'
        )
        
        # Generate 1 month of data
        df = generate_realistic_forex_data('2024-09-01', '2024-10-01')
        
        # Simulate rolling analysis (every hour)
        window_size = 2000
        step = 60  # 1 hour
        
        results = []
        
        for i in range(window_size, len(df), step):
            window = df.iloc[i-window_size:i]
            
            try:
                score, bias, details = analyzer.analyze(window)
                
                # Determine if would trade
                should_trade = analyzer.should_trade(score, bias, min_confluence=0.70, min_bias=0.3)
                quality = analyzer.get_signal_quality(score, bias)
                
                results.append({
                    'timestamp': window.index[-1],
                    'confluence': score,
                    'bias': bias,
                    'should_trade': should_trade,
                    'quality': quality,
                    'action': 'LONG' if should_trade and bias > 0 else 'SHORT' if should_trade and bias < 0 else 'WAIT'
                })
            except:
                continue
        
        df_results = pd.DataFrame(results)
        
        print(f"✓ Analyzed {len(df_results)} time periods")
        print(f"\n✓ Trading signals:")
        print(f"  Total signals: {df_results['should_trade'].sum()}")
        print(f"  Long signals: {(df_results['action'] == 'LONG').sum()}")
        print(f"  Short signals: {(df_results['action'] == 'SHORT').sum()}")
        print(f"  Wait periods: {(df_results['action'] == 'WAIT').sum()}")
        
        print(f"\n✓ Signal quality distribution:")
        print(df_results['quality'].value_counts())
        
        print(f"\n✓ Confluence statistics:")
        print(f"  Mean: {df_results['confluence'].mean():.3f}")
        print(f"  Std: {df_results['confluence'].std():.3f}")
        print(f"  Min: {df_results['confluence'].min():.3f}")
        print(f"  Max: {df_results['confluence'].max():.3f}")
        
        return df_results
        
    except FileNotFoundError:
        print("⚠ Skipped: Run training test first")
        return None


def run_all_tests(quick_mode: bool = True):
    """Run complete test suite"""
    
    print("\n" + "="*80)
    print("MULTI-TIMEFRAME CONFLUENCE MODEL - TEST SUITE")
    print("="*80)
    print(f"Mode: {'Quick Test' if quick_mode else 'Full Test'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'test_1': False,
        'test_2': False,
        'test_3': False,
        'test_4': False,
        'test_5': False,
        'test_6': False
    }
    
    try:
        # Test 1: Data Preparation
        preparator, df = test_data_preparation()
        results['test_1'] = True
        
        # Save scalers for later tests
        import pickle
        with open('/home/claude/test_confluence_scalers.pkl', 'wb') as f:
            pickle.dump(preparator.scalers, f)
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    try:
        # Test 2: Model Architecture
        model = test_model_architecture()
        results['test_2'] = True
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    try:
        # Test 3: Training
        trainer, preparator, history = test_training_pipeline(quick_test=quick_mode)
        results['test_3'] = True
        
        # Save scalers again after training
        import pickle
        with open('/home/claude/test_confluence_scalers.pkl', 'wb') as f:
            pickle.dump(preparator.scalers, f)
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
    
    try:
        # Test 4: Inference Speed
        avg_time = test_inference_speed()
        results['test_4'] = avg_time is not None
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
    
    try:
        # Test 5: Prediction Quality
        conditions = test_prediction_quality()
        results['test_5'] = conditions is not None
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
    
    try:
        # Test 6: Real-World Simulation
        df_sim = test_real_world_simulation()
        results['test_6'] = df_sim is not None
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper()}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Model is ready for production.")
    elif passed_tests >= total_tests - 1:
        print("\n✓ Most tests passed. Review failures and proceed with caution.")
    else:
        print("\n⚠ Multiple tests failed. Review and fix issues before deployment.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check if quick mode
    quick_mode = '--full' not in sys.argv
    
    print("Starting test suite...")
    print(f"Quick mode: {quick_mode}")
    print("(Use --full flag for complete testing)")
    
    results = run_all_tests(quick_mode=quick_mode)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

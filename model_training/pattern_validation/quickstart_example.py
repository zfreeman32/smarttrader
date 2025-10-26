"""
Pattern Validation System - Quick Start Example
===============================================
This script demonstrates the complete workflow using dummy data.
Use this to verify your installation and understand the workflow.

Run this first before using with real data!
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from data_generation import (
    DetectedPattern,
    PatternOutcomeLabeler,
    PatternDatasetGenerator
)
from model_architecture import PatternValidationModel
from training_pipeline import run_full_training_pipeline
from inference_pipeline import PatternValidator, TradingSystemIntegration


def generate_dummy_price_data(n_bars: int = 10000) -> pd.DataFrame:
    """Generate synthetic EURUSD price data"""
    print("Generating dummy price data...")
    
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1min')
    
    # Generate realistic price movement
    returns = np.random.randn(n_bars) * 0.0001  # ~10 pip moves
    price = 1.1000 + np.cumsum(returns)
    
    # OHLC
    high = price + np.random.uniform(0, 0.0005, n_bars)
    low = price - np.random.uniform(0, 0.0005, n_bars)
    open_price = np.roll(price, 1)
    open_price[0] = price[0]
    close = price
    
    # Volume
    volume = np.random.uniform(100, 1000, n_bars)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def generate_dummy_detected_patterns(n_patterns: int = 500) -> list:
    """Generate synthetic detected patterns"""
    print(f"Generating {n_patterns} dummy detected patterns...")
    
    patterns = []
    base_date = pd.Timestamp('2023-01-01')
    
    for i in range(n_patterns):
        # Random pattern
        pattern_type = np.random.choice(['FVG', 'OB', 'Sweep'])
        direction = np.random.choice(['bullish', 'bearish'])
        
        # Generate chart window
        chart_window = np.random.randn(100, 5)
        chart_window = np.abs(chart_window)  # Ensure positive prices
        chart_window[:, :4] += 1.1000  # Add base price
        chart_window[:, 4] *= 100  # Scale volume
        
        # Pattern metadata
        pattern = DetectedPattern(
            pattern_id=f'{pattern_type}_{i:04d}',
            timestamp=base_date + timedelta(minutes=i*20),
            pattern_type=pattern_type,
            direction=direction,
            entry_price=1.1000 + np.random.uniform(-0.01, 0.01),
            zone_top=1.1000 + np.random.uniform(0, 0.002),
            zone_bottom=1.1000 - np.random.uniform(0, 0.002),
            pattern_size_pips=np.random.uniform(5, 30),
            volume_ratio=np.random.uniform(1.0, 3.0),
            volume_confirmed=np.random.choice([True, False]),
            session=np.random.choice(['Asian', 'London', 'NY']),
            atr_at_detection=np.random.uniform(0.0005, 0.002),
            rsi=np.random.uniform(30, 70),
            adx=np.random.uniform(10, 50),
            trend_alignment=np.random.choice([True, False]),
            distance_to_structure=np.random.uniform(0, 1),
            candle_window=chart_window,
            volume_profile=chart_window[:, 4],
            indicator_values=np.random.randn(15)
        )
        
        patterns.append(pattern)
    
    return patterns


def quick_start_example():
    """Run complete workflow with dummy data"""
    
    print("\n" + "="*70)
    print("PATTERN VALIDATION SYSTEM - QUICK START")
    print("="*70)
    print("This example uses dummy data to demonstrate the complete workflow.\n")
    
    # Create output directory
    output_dir = './quickstart_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================================================================
    # STEP 1: GENERATE TRAINING DATA
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 1: GENERATING TRAINING DATA")
    print("-"*70)
    
    # Generate dummy data
    price_data = generate_dummy_price_data(n_bars=10000)
    detected_patterns = generate_dummy_detected_patterns(n_patterns=500)
    
    print(f"Created {len(price_data)} price bars")
    print(f"Created {len(detected_patterns)} detected patterns")
    
    # Generate training dataset
    generator = PatternDatasetGenerator(lookback_candles=100)
    
    training_data, metadata = generator.generate_training_data(
        detected_patterns=detected_patterns,
        price_data=price_data,
        save_path=os.path.join(output_dir, 'training_data')
    )
    
    print(f"\n✓ Training data generated: {len(metadata)} examples")
    
    # =====================================================================
    # STEP 2: TRAIN MODEL
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 2: TRAINING MODEL")
    print("-"*70)
    
    # Initialize model
    model_builder = PatternValidationModel(
        chart_shape=(100, 5),
        volume_shape=(100, 1),
        pattern_features_dim=13,
        indicator_features_dim=15,
        cnn_filters=(32, 64, 128),  # Smaller model for quick training
        lstm_units=(64, 32),
        dropout_rate=0.4
    )
    
    model_builder.build_model()
    model_builder.compile_model(learning_rate=0.001)
    
    print("\nModel architecture:")
    model_builder.summary()
    
    # Quick training (fewer epochs for demo)
    print("\nTraining model (quick demo mode)...")
    
    trainer = run_full_training_pipeline(
        data_dir=os.path.join(output_dir, 'training_data'),
        output_dir=os.path.join(output_dir, 'model_outputs'),
        use_progressive=False,  # Skip progressive for speed
        use_cv=False
    )
    
    # Override with quick training
    from training_pipeline import PatternValidationTrainer
    
    trainer = PatternValidationTrainer(
        model_builder=model_builder,
        data_dir=os.path.join(output_dir, 'training_data'),
        output_dir=os.path.join(output_dir, 'model_outputs')
    )
    
    data = trainer.load_data()
    trainer.split_data(data)
    
    # Quick training (10 epochs)
    history = model_builder.model.fit(
        [trainer.X_train['chart'], trainer.X_train['volume'],
         trainer.X_train['pattern'], trainer.X_train['indicators']],
        trainer.y_train,
        validation_data=(
            [trainer.X_val['chart'], trainer.X_val['volume'],
             trainer.X_val['pattern'], trainer.X_val['indicators']],
            trainer.y_val
        ),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(output_dir, 'pattern_validator_demo.keras')
    model_builder.model.save(model_path)
    
    print(f"\n✓ Model trained and saved to {model_path}")
    
    # Evaluate
    trainer.training_history = history.history
    results = trainer.evaluate()
    
    print(f"\nTest Set Performance:")
    print(f"  MSE: {results['test']['mse']:.4f}")
    print(f"  MAE: {results['test']['mae']:.4f}")
    print(f"  R²: {results['test']['r2']:.4f}")
    
    # =====================================================================
    # STEP 3: INFERENCE
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 3: REAL-TIME INFERENCE")
    print("-"*70)
    
    # Load validator
    validator = PatternValidator(
        model_path=model_path,
        quality_threshold=0.75
    )
    
    print("Validator loaded and ready!")
    
    # Test single pattern validation
    print("\nTesting single pattern validation...")
    
    test_pattern = detected_patterns[0]
    
    score = validator.validate_pattern(
        chart_window=test_pattern.candle_window,
        volume_profile=test_pattern.volume_profile,
        pattern_features={
            'pattern_size_normalized': test_pattern.pattern_size_pips / test_pattern.atr_at_detection,
            'volume_ratio': test_pattern.volume_ratio,
            'volume_confirmed': test_pattern.volume_confirmed,
            'session': test_pattern.session,
            'rsi': test_pattern.rsi,
            'adx': test_pattern.adx,
            'trend_alignment': test_pattern.trend_alignment,
            'distance_to_structure': test_pattern.distance_to_structure
        },
        indicator_values=test_pattern.indicator_values,
        pattern_id=test_pattern.pattern_id,
        pattern_type=test_pattern.pattern_type,
        direction=test_pattern.direction
    )
    
    print(f"\nPattern: {score.pattern_id}")
    print(f"  Type: {score.pattern_type}")
    print(f"  Direction: {score.direction}")
    print(f"  Quality Score: {score.quality_score:.3f}")
    print(f"  Quality Class: {score.quality_class}")
    print(f"  Tradeable: {score.tradeable}")
    print(f"  Confidence: {score.confidence_level}")
    
    # Test batch validation
    print("\nTesting batch validation (10 patterns)...")
    
    batch_patterns = [
        {
            'pattern_id': p.pattern_id,
            'pattern_type': p.pattern_type,
            'direction': p.direction,
            'chart_window': p.candle_window,
            'volume_profile': p.volume_profile,
            'pattern_features': {
                'pattern_size_normalized': p.pattern_size_pips / p.atr_at_detection,
                'volume_ratio': p.volume_ratio,
                'volume_confirmed': p.volume_confirmed,
                'session': p.session,
                'rsi': p.rsi,
                'adx': p.adx,
                'trend_alignment': p.trend_alignment,
                'distance_to_structure': p.distance_to_structure
            },
            'indicator_values': p.indicator_values
        }
        for p in detected_patterns[:10]
    ]
    
    scores = validator.validate_batch(batch_patterns)
    
    print(f"\nBatch validation results:")
    print(f"  Total patterns: {len(scores)}")
    print(f"  Tradeable: {sum(s.tradeable for s in scores)}")
    print(f"  Average score: {np.mean([s.quality_score for s in scores]):.3f}")
    
    # Performance stats
    stats = validator.get_performance_stats()
    print(f"\nInference Performance:")
    print(f"  Avg time: {stats['avg_inference_time_ms']:.2f} ms/pattern")
    print(f"  P95 latency: {stats['p95_inference_time_ms']:.2f} ms")
    
    # Test trading system integration
    print("\nTesting trading system integration...")
    
    integration = TradingSystemIntegration(
        validator=validator,
        min_quality_score=0.75,
        log_filtered_patterns=True
    )
    
    # Filter patterns
    validated_patterns, all_scores = integration.filter_patterns(batch_patterns[:10])
    
    filter_stats = integration.get_filter_stats()
    
    print(f"\nIntegration Results:")
    print(f"  Detected: {filter_stats['total_patterns']}")
    print(f"  Passed: {filter_stats['passed_patterns']}")
    print(f"  Filtered: {filter_stats['filtered_patterns']}")
    print(f"  Pass rate: {filter_stats['pass_rate']*100:.1f}%")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("✓ QUICK START COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nOutput files:")
    print(f"  Training data: {output_dir}/training_data/")
    print(f"  Trained model: {model_path}")
    print(f"  Model outputs: {output_dir}/model_outputs/")
    
    print("\nNext steps:")
    print("  1. Replace dummy data with your real ICT detector outputs")
    print("  2. Train on 10,000+ real patterns from 2+ years of data")
    print("  3. Achieve >80% validation accuracy")
    print("  4. Integrate with your trading system")
    print("  5. Backtest with pattern validation layer")
    print("  6. Paper trade for 2+ weeks")
    print("  7. Deploy to live trading with small positions")
    
    print("\n" + "="*70)
    print("Review README.md for detailed usage instructions.")
    print("="*70 + "\n")
    
    return validator, integration, results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        validator, integration, results = quick_start_example()
        
        print("\n✓ Quick start completed! The system is working correctly.")
        print("  Now you can use it with your real trading data.")
        
    except Exception as e:
        print(f"\n❌ Quick start failed: {e}")
        print("  Check your installation and dependencies.")
        import traceback
        traceback.print_exc()

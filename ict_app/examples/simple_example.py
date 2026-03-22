"""
Simple Example: Using ICT Pattern Detectors
Basic usage example for the EURUSD trading system
"""
import sys
sys.path.append('./outputs/eurusd_trading_system/src')

import pandas as pd
from data.data_utils import generate_sample_eurusd_data
from ict_detection import ICTPatternCoordinator


def main():
    print("="*60)
    print("  ICT Pattern Detection Example")
    print("="*60 + "\n")
    
    # Step 1: Generate or load data
    print("Step 1: Loading data...")
    df = generate_sample_eurusd_data(n_bars=500)
    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}\n")
    
    # Step 2: Initialize coordinator
    print("Step 2: Initializing pattern coordinator...")
    
    config = {
        'fvg': {
            'sensitivity': 'moderate',
            'volume_confirmation': True
        },
        'order_block': {
            'volume_threshold': 2.0
        },
        'liquidity_sweep': {
            'lookback_period': 50
        }
    }
    
    coordinator = ICTPatternCoordinator(config=config)
    print("  ✓ Coordinator initialized\n")
    
    # Step 3: Detect patterns
    print("Step 3: Detecting patterns...")
    results = coordinator.detect_all_patterns(df, use_session_filter=True)
    
    print("\nResults:")
    print("-" * 60)
    
    for pattern_type, detections in results.items():
        if detections:
            bullish = sum(1 for d in detections if d.direction == 'bullish')
            bearish = sum(1 for d in detections if d.direction == 'bearish')
            
            print(f"\n{pattern_type}:")
            print(f"  Total: {len(detections)}")
            print(f"  Bullish: {bullish}")
            print(f"  Bearish: {bearish}")
            
            # Show first detection as example
            if detections:
                first = detections[0]
                print(f"\n  Example detection:")
                print(f"    Timestamp: {first.timestamp}")
                print(f"    Direction: {first.direction}")
                print(f"    Entry: {first.entry_price:.5f}")
                print(f"    Confidence: {first.confidence:.2%}")
                print(f"    Zone: {first.bottom_price:.5f} - {first.top_price:.5f}")
    
    # Step 4: Generate trading signals
    print("\n" + "="*60)
    print("Step 4: Generating trading signals...")
    
    # Check signal at current bar
    current_bar = len(df) - 1
    signal = coordinator.generate_trading_signals(
        df, 
        current_bar, 
        min_confluence=0.7
    )
    
    print(f"\nSignal at bar {current_bar}:")
    print(f"  Price: {df.iloc[current_bar]['close']:.5f}")
    print(f"  Signal: {signal['signal']}")
    print(f"  Confidence: {signal['confidence']:.2%}")
    print(f"  Session: {signal['session']}")
    print(f"  Active patterns: {signal['active_patterns']}")
    
    # Step 5: Get statistics
    print("\n" + "="*60)
    print("Step 5: Summary statistics...")
    
    stats = coordinator.get_summary_statistics()
    
    print(f"\nTotal detections: {stats['total_detections']}")
    print("\nBy detector:")
    for detector in ['fvg', 'order_block', 'liquidity_sweep', 'market_structure']:
        s = stats[detector]
        print(f"  {detector}: {s['total_detections']} detections, "
              f"{s['avg_confidence']:.2%} avg confidence")
    
    print("\n" + "="*60)
    print("  Example completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

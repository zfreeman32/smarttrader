"""
Load Real EURUSD Data and Run ICT Detectors
Example for using your own CSV data with the trading system
"""
import sys
sys.path.append('./outputs/eurusd_trading_system/src')

import pandas as pd
from data.data_utils import load_csv_data, validate_ohlcv_data
from ict_detection import ICTPatternCoordinator


def load_your_eurusd_data(filepath: str, max_rows: int = None):
    """
    Load your EURUSD data from CSV
    
    Args:
        filepath: Path to your EURUSD_1min_features.csv file
        max_rows: Limit number of rows (useful for testing)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading data from: {filepath}")
    
    # Load the CSV file
    # Your file has: Date, Time, Open, High, Low, Close, Volume, datetime
    df = load_csv_data(
        filepath, 
        parse_dates=True,
        keep_extra_features=False  # Only keep OHLCV for ICT detectors
    )
    
    # Limit rows if specified (useful for testing)
    if max_rows:
        df = df.head(max_rows)
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Validate data integrity
    try:
        validate_ohlcv_data(df)
        print("✓ Data validation passed")
    except ValueError as e:
        print(f"✗ Data validation failed: {e}")
        return None
    
    # Show some statistics
    print(f"\nData Statistics:")
    print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    print(f"  Average volume: {df['volume'].mean():.0f}")
    print(f"  Total bars: {len(df)}")
    
    return df


def run_detectors_on_real_data(df: pd.DataFrame, 
                               use_session_filter: bool = True,
                               min_confluence: float = 0.7):
    """
    Run all ICT detectors on your real data
    
    Args:
        df: DataFrame with OHLCV data
        use_session_filter: Apply session filtering
        min_confluence: Minimum confluence for signals
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("Running ICT Pattern Detectors on Real Data")
    print("="*80 + "\n")
    
    # Initialize coordinator with custom configuration
    config = {
        'fvg': {
            'sensitivity': 'moderate',
            'min_gap_pips': 5,
            'volume_confirmation': True,
            'volume_threshold': 1.5
        },
        'order_block': {
            'swing_window': 20,
            'volume_threshold': 2.0,
            'min_move_pips': 10
        },
        'liquidity_sweep': {
            'lookback_period': 50,
            'cluster_tolerance_pips': 3,
            'volume_threshold': 1.5
        },
        'market_structure': {
            'swing_window': 10,
            'min_break_pips': 3
        }
    }
    
    coordinator = ICTPatternCoordinator(config=config)
    
    # Detect all patterns
    print("Detecting patterns...")
    results = coordinator.detect_all_patterns(df, use_session_filter=use_session_filter)
    
    # Display results
    print("\nPattern Detection Results:")
    print("-" * 80)
    
    total_detections = 0
    for pattern_type, detections in results.items():
        if detections:
            bullish = sum(1 for d in detections if d.direction == 'bullish')
            bearish = sum(1 for d in detections if d.direction == 'bearish')
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            
            print(f"\n{pattern_type}:")
            print(f"  Total: {len(detections)}")
            print(f"  Bullish: {bullish} | Bearish: {bearish}")
            print(f"  Average confidence: {avg_conf:.2%}")
            
            total_detections += len(detections)
            
            # Show first few examples
            if len(detections) > 0:
                print(f"\n  Recent detections:")
                for i, det in enumerate(detections[-3:]):  # Last 3
                    print(f"    [{i+1}] {det.timestamp} - {det.direction.upper()}")
                    print(f"        Entry: {det.entry_price:.5f}, Confidence: {det.confidence:.2%}")
    
    print(f"\n{'='*80}")
    print(f"Total Detections: {total_detections}")
    print(f"Detection Rate: {total_detections/len(df):.2%} of bars")
    print(f"{'='*80}\n")
    
    # Generate signals for recent bars
    print("\nGenerating Trading Signals for Recent Bars:")
    print("-" * 80)
    
    # Check last 10 bars for signals
    signal_bars = min(10, len(df))
    signals_found = []
    
    for i in range(len(df) - signal_bars, len(df)):
        signal = coordinator.generate_trading_signals(
            df, 
            current_bar=i,
            min_confluence=min_confluence
        )
        
        if signal['signal'] != 'HOLD':
            signals_found.append({
                'bar': i,
                'timestamp': df.index[i],
                'price': df.iloc[i]['close'],
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'session': signal['session']
            })
    
    if signals_found:
        print(f"\nFound {len(signals_found)} trading signals:\n")
        for sig in signals_found:
            print(f"  {sig['timestamp']} | Price: {sig['price']:.5f}")
            print(f"    Signal: {sig['signal']} (Confidence: {sig['confidence']:.2%})")
            print(f"    Session: {sig['session']}")
            print()
    else:
        print("\n  No trading signals found (confluence below threshold)")
        print(f"  Try lowering min_confluence (currently {min_confluence})")
    
    # Get summary statistics
    print("\nDetector Statistics:")
    print("-" * 80)
    stats = coordinator.get_summary_statistics()
    
    for detector in ['fvg', 'order_block', 'liquidity_sweep', 'market_structure']:
        s = stats[detector]
        if s['total_detections'] > 0:
            print(f"\n{detector.upper()}:")
            print(f"  Total detections: {s['total_detections']}")
            print(f"  Bullish: {s['bullish_count']}")
            print(f"  Bearish: {s['bearish_count']}")
            print(f"  Average confidence: {s['avg_confidence']:.2%}")
    
    return results, signals_found


def analyze_specific_timeframe(df: pd.DataFrame, 
                               start_date: str, 
                               end_date: str):
    """
    Analyze patterns in a specific date range
    
    Args:
        df: Full DataFrame
        start_date: Start date (e.g., '2018-04-26 00:00:00')
        end_date: End date
        
    Returns:
        Filtered results
    """
    print(f"\nAnalyzing timeframe: {start_date} to {end_date}")
    
    # Filter data by date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]
    
    print(f"Filtered to {len(df_filtered)} bars")
    
    if len(df_filtered) < 100:
        print("Warning: Less than 100 bars - may not have enough data for patterns")
    
    # Run detectors on filtered data
    results, signals = run_detectors_on_real_data(df_filtered)
    
    return results, signals


def main():
    """
    Main execution - Load your data and run detectors
    """
    print("="*80)
    print("EURUSD Real Data Analysis - ICT Pattern Detection")
    print("="*80 + "\n")
    
    # STEP 1: UPDATE THIS PATH TO YOUR CSV FILE
    # Change this to the actual path of your EURUSD_1min_features.csv file
    filepath = "data/currency_data/EURUSD_1min_features.csv"
    
    # For testing, you can limit rows (remove max_rows parameter to load all data)
    df = load_your_eurusd_data(filepath, max_rows=2000)  # Start with 2000 bars
    
    if df is None:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # STEP 2: Run detectors on your data
    results, signals = run_detectors_on_real_data(
        df, 
        use_session_filter=True,
        min_confluence=0.7
    )
    
    # STEP 3: (Optional) Analyze specific timeframe
    # Uncomment the lines below to analyze a specific date range
    # start_date = '2018-04-26 00:00:00'
    # end_date = '2018-04-26 23:59:00'
    # results_filtered, signals_filtered = analyze_specific_timeframe(df, start_date, end_date)
    
    # STEP 4: Save results (optional)
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    
    print("\nNext steps:")
    print("1. Adjust detector configurations in the config dictionary")
    print("2. Modify min_confluence threshold for more/fewer signals")
    print("3. Analyze specific timeframes or trading sessions")
    print("4. Export detection results for further analysis")
    
    return results, signals


if __name__ == "__main__":
    results, signals = main()

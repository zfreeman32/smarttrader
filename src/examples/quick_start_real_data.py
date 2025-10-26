"""
QUICK START: Run ICT Detectors on Your EURUSD Data

Simply update the FILE_PATH below and run this script!
"""
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot')

import pandas as pd
from src.data.data_utils import load_csv_data
from src.ict_detection import ICTPatternCoordinator


# ========================================================================
# STEP 1: UPDATE THIS PATH TO YOUR CSV FILE
# ========================================================================
FILE_PATH = r"C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\data\currency_data\sampled\EURUSD_1min_sampled_features.csv"


# ========================================================================
# STEP 2: CONFIGURATION (OPTIONAL - defaults work well)
# ========================================================================
# Number of bars to load (use None for all data)
MAX_BARS = 9000  # Start small for testing, then increase or set to None

# Session filtering (True = only trade patterns in their optimal sessions)
USE_SESSION_FILTER = True

# Minimum confluence for signals (0.7 = 70% confidence)
MIN_CONFLUENCE = 0.7


# ========================================================================
# MAIN SCRIPT - NO NEED TO MODIFY BELOW
# ========================================================================

def main():
    print("="*80)
    print("  EURUSD ICT PATTERN DETECTION - REAL DATA")
    print("="*80 + "\n")
    
    # Load your data
    print(f"Loading data from: {FILE_PATH}")
    try:
        df = load_csv_data(FILE_PATH, parse_dates=True)
        
        # Limit bars if specified
        if MAX_BARS and MAX_BARS < len(df):
            df = df.head(MAX_BARS)
            print(f"Limited to first {MAX_BARS} bars for testing")
        
        print(f"✓ Loaded {len(df)} bars")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}\n")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nTroubleshooting:")
        print("  1. Check the FILE_PATH is correct")
        print("  2. Ensure CSV has: Open, High, Low, Close, Volume columns")
        print("  3. Ensure datetime column or Date+Time columns exist")
        return
    
    # Initialize ICT detector system
    print("Initializing ICT Pattern Detectors...")
    coordinator = ICTPatternCoordinator()
    print("✓ Detectors ready\n")
    
    # Run pattern detection
    print("Running pattern detection...")
    print("(This may take a few seconds...)\n")
    
    results = coordinator.detect_all_patterns(df, use_session_filter=USE_SESSION_FILTER)
    
    # Display results
    print("="*80)
    print("  DETECTION RESULTS")
    print("="*80 + "\n")
    
    total_patterns = 0
    
    for pattern_type, detections in results.items():
        if not detections:
            continue
            
        total_patterns += len(detections)
        
        bullish = sum(1 for d in detections if d.direction == 'bullish')
        bearish = sum(1 for d in detections if d.direction == 'bearish')
        avg_conf = sum(d.confidence for d in detections) / len(detections)
        
        print(f"{pattern_type}:")
        print(f"  ├─ Total: {len(detections)}")
        print(f"  ├─ Bullish: {bullish} | Bearish: {bearish}")
        print(f"  └─ Avg Confidence: {avg_conf:.1%}\n")
    
    print(f"Total Patterns Detected: {total_patterns}")
    print(f"Detection Rate: {total_patterns/len(df)*100:.1f}% of bars\n")
    
    # Generate trading signals for most recent bars
    print("="*80)
    print("  TRADING SIGNALS (Last 20 bars)")
    print("="*80 + "\n")
    
    signals_found = 0
    check_bars = min(20, len(df))
    
    for i in range(len(df) - check_bars, len(df)):
        signal = coordinator.generate_trading_signals(df, i, min_confluence=MIN_CONFLUENCE)
        
        if signal['signal'] != 'HOLD':
            signals_found += 1
            bar_time = df.index[i]
            bar_price = df.iloc[i]['close']
            
            print(f"[{signal['signal']}] {bar_time}")
            print(f"  Price: {bar_price:.5f}")
            print(f"  Confidence: {signal['confidence']:.1%}")
            print(f"  Session: {signal['session']}")
            print()
    
    if signals_found == 0:
        print(f"No signals found with {MIN_CONFLUENCE:.0%}+ confluence")
        print("Try lowering MIN_CONFLUENCE in the configuration\n")
    
    # Show most recent detections by pattern type
    print("="*80)
    print("  MOST RECENT DETECTIONS (Last 3 per pattern)")
    print("="*80 + "\n")
    
    for pattern_type, detections in results.items():
        if not detections:
            continue
        
        print(f"\n{pattern_type}:")
        
        # Show last 3 detections
        for detection in detections[-3:]:
            direction_emoji = "📈" if detection.direction == 'bullish' else "📉"
            print(f"  {direction_emoji} {detection.timestamp}")
            print(f"     Entry: {detection.entry_price:.5f} | Confidence: {detection.confidence:.1%}")
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    print("Next Steps:")
    print("  1. ✓ Patterns detected on your real data")
    print("  2. → Adjust configuration if needed (see top of script)")
    print("  3. → Increase MAX_BARS to analyze more data")
    print("  4. → Use results for backtesting your strategy")
    print("  5. → Export results for further analysis\n")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

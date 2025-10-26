"""
Synthetic EURUSD Data Generator
================================

Generates realistic synthetic EURUSD 1-minute OHLCV data for testing
the regime classifier. Simulates different market regimes including
trends, ranging, and volatility patterns.

Usage:
    python generate_test_data.py --samples 100000 --output eurusd_synthetic.csv
"""

import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta
from typing import Tuple


class SyntheticEURUSDGenerator:
    """
    Generate realistic synthetic EURUSD price data with various regime characteristics.
    """
    
    def __init__(self, 
                 initial_price: float = 1.1000,
                 seed: int = 42):
        """
        Initialize synthetic data generator.
        
        Args:
            initial_price: Starting EURUSD price
            seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.seed = seed
        np.random.seed(seed)
        
    def generate_regime_sequence(self, n_samples: int) -> np.ndarray:
        """
        Generate a sequence of market regimes with realistic transitions.
        
        Regimes:
        0: Strong Uptrend
        1: Weak Uptrend
        2: Ranging
        3: Weak Downtrend
        4: Strong Downtrend
        5: High Volatility
        6: Low Volatility
        """
        regimes = []
        current_regime = 2  # Start with ranging
        
        # Regime transition probabilities
        # Each row is: [Strong Up, Weak Up, Ranging, Weak Down, Strong Down, High Vol, Low Vol]
        transition_matrix = {
            0: [0.60, 0.15, 0.10, 0.05, 0.02, 0.05, 0.03],  # From Strong Up
            1: [0.20, 0.40, 0.25, 0.10, 0.02, 0.02, 0.01],  # From Weak Up
            2: [0.10, 0.20, 0.40, 0.20, 0.05, 0.03, 0.02],  # From Ranging
            3: [0.02, 0.10, 0.25, 0.40, 0.20, 0.02, 0.01],  # From Weak Down
            4: [0.02, 0.05, 0.10, 0.15, 0.60, 0.05, 0.03],  # From Strong Down
            5: [0.15, 0.15, 0.20, 0.15, 0.15, 0.10, 0.10],  # From High Vol
            6: [0.15, 0.15, 0.30, 0.15, 0.15, 0.05, 0.05],  # From Low Vol
        }
        
        # Regime persistence (how long each regime lasts)
        regime_durations = {
            0: (100, 500),   # Strong uptrend: 100-500 bars
            1: (50, 200),    # Weak uptrend: 50-200 bars
            2: (100, 300),   # Ranging: 100-300 bars
            3: (50, 200),    # Weak downtrend: 50-200 bars
            4: (100, 500),   # Strong downtrend: 100-500 bars
            5: (20, 100),    # High volatility: 20-100 bars
            6: (50, 150),    # Low volatility: 50-150 bars
        }
        
        total_bars = 0
        while total_bars < n_samples:
            # Determine regime duration
            min_dur, max_dur = regime_durations[current_regime]
            duration = np.random.randint(min_dur, max_dur + 1)
            duration = min(duration, n_samples - total_bars)
            
            # Add regime for this duration
            regimes.extend([current_regime] * duration)
            total_bars += duration
            
            # Transition to next regime
            probs = transition_matrix[current_regime]
            current_regime = np.random.choice(len(probs), p=probs)
        
        return np.array(regimes[:n_samples])
    
    def generate_price_series(self, 
                             regimes: np.ndarray,
                             n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate OHLC price series based on regime sequence.
        
        Returns:
            Tuple of (open, high, low, close) arrays
        """
        # Regime characteristics
        regime_params = {
            # (drift, volatility, range_factor)
            0: (0.0002, 0.00015, 1.5),    # Strong uptrend: high drift, moderate vol
            1: (0.0001, 0.00012, 1.2),    # Weak uptrend: low drift, low vol
            2: (0.0000, 0.00008, 0.8),    # Ranging: no drift, low vol
            3: (-0.0001, 0.00012, 1.2),   # Weak downtrend: low drift, low vol
            4: (-0.0002, 0.00015, 1.5),   # Strong downtrend: high drift, moderate vol
            5: (0.0000, 0.00040, 2.5),    # High volatility: no drift, very high vol
            6: (0.0000, 0.00005, 0.5),    # Low volatility: no drift, very low vol
        }
        
        # Initialize arrays
        close = np.zeros(n_samples)
        open_price = np.zeros(n_samples)
        high = np.zeros(n_samples)
        low = np.zeros(n_samples)
        
        # Starting values
        close[0] = self.initial_price
        open_price[0] = self.initial_price
        
        for i in range(1, n_samples):
            regime = regimes[i]
            drift, volatility, range_factor = regime_params[regime]
            
            # Generate return
            ret = drift + volatility * np.random.randn()
            
            # Calculate close price
            close[i] = close[i-1] * (1 + ret)
            
            # Generate OHLC
            # Open is previous close with small gap
            gap = volatility * 0.5 * np.random.randn()
            open_price[i] = close[i-1] * (1 + gap)
            
            # High and low based on volatility and range
            intrabar_range = volatility * range_factor * abs(np.random.randn())
            
            if close[i] > open_price[i]:
                # Bullish bar
                high[i] = max(open_price[i], close[i]) + intrabar_range * close[i]
                low[i] = min(open_price[i], close[i]) - intrabar_range * 0.5 * close[i]
            else:
                # Bearish bar
                high[i] = max(open_price[i], close[i]) + intrabar_range * 0.5 * close[i]
                low[i] = min(open_price[i], close[i]) - intrabar_range * close[i]
            
            # Ensure OHLC relationship is valid
            high[i] = max(high[i], open_price[i], close[i])
            low[i] = min(low[i], open_price[i], close[i])
        
        return open_price, high, low, close
    
    def generate_volume(self, 
                       regimes: np.ndarray,
                       n_samples: int) -> np.ndarray:
        """
        Generate volume series based on regime.
        Higher volatility regimes have higher volume.
        """
        base_volume = 2000
        
        regime_volume_multipliers = {
            0: 1.8,  # Strong uptrend: high volume
            1: 1.2,  # Weak uptrend: moderate volume
            2: 1.0,  # Ranging: normal volume
            3: 1.2,  # Weak downtrend: moderate volume
            4: 1.8,  # Strong downtrend: high volume
            5: 2.5,  # High volatility: very high volume
            6: 0.6,  # Low volatility: low volume
        }
        
        volume = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            regime = regimes[i]
            multiplier = regime_volume_multipliers[regime]
            
            # Random volume with regime-dependent mean
            vol = base_volume * multiplier * (0.7 + 0.6 * np.random.random())
            volume[i] = int(vol)
        
        return volume
    
    def generate_dataset(self, 
                        n_samples: int,
                        start_date: str = None) -> pd.DataFrame:
        """
        Generate complete synthetic EURUSD dataset.
        
        Args:
            n_samples: Number of 1-minute bars to generate
            start_date: Starting date (default: 2 years ago)
            
        Returns:
            DataFrame with OHLCV data and regime labels
        """
        print(f"Generating {n_samples:,} synthetic EURUSD bars...")
        
        # Generate regime sequence
        print("  1. Generating regime sequence...")
        regimes = self.generate_regime_sequence(n_samples)
        
        # Generate prices
        print("  2. Generating OHLC prices...")
        open_price, high, low, close = self.generate_price_series(regimes, n_samples)
        
        # Generate volume
        print("  3. Generating volume...")
        volume = self.generate_volume(regimes, n_samples)
        
        # Generate timestamps
        print("  4. Generating timestamps...")
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 years ago
        else:
            start_date = pd.to_datetime(start_date)
        
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1min')
        
        # Create DataFrame
        print("  5. Creating DataFrame...")
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'true_regime': regimes  # Include true regime for validation
        })
        
        # Validate OHLC relationships
        assert (df['high'] >= df['open']).all(), "High must be >= Open"
        assert (df['high'] >= df['close']).all(), "High must be >= Close"
        assert (df['low'] <= df['open']).all(), "Low must be <= Open"
        assert (df['low'] <= df['close']).all(), "Low must be <= Close"
        assert (df['volume'] > 0).all(), "Volume must be positive"
        
        print(f"✅ Generated {len(df):,} bars successfully")
        print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"   Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
        
        # Print regime distribution
        regime_names = {
            0: 'Strong Uptrend',
            1: 'Weak Uptrend',
            2: 'Ranging',
            3: 'Weak Downtrend',
            4: 'Strong Downtrend',
            5: 'High Volatility',
            6: 'Low Volatility'
        }
        
        print("\n   Regime Distribution:")
        for regime_id in sorted(regime_names.keys()):
            count = (df['true_regime'] == regime_id).sum()
            pct = (count / len(df)) * 100
            print(f"      {regime_names[regime_id]:20s}: {count:6,} bars ({pct:5.2f}%)")
        
        return df


def main():
    """Main function for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic EURUSD data for testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--samples', type=int, default=100000,
                       help='Number of 1-minute bars to generate')
    parser.add_argument('--output', type=str, default='eurusd_synthetic.csv',
                       help='Output CSV filename')
    parser.add_argument('--initial_price', type=float, default=1.1000,
                       help='Starting EURUSD price')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_regime_label', action='store_true',
                       help='Exclude true_regime column from output')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SYNTHETIC EURUSD DATA GENERATOR")
    print("="*80)
    print(f"Samples:       {args.samples:,}")
    print(f"Output:        {args.output}")
    print(f"Initial price: ${args.initial_price:.4f}")
    print(f"Seed:          {args.seed}")
    print("="*80 + "\n")
    
    # Generate data
    generator = SyntheticEURUSDGenerator(
        initial_price=args.initial_price,
        seed=args.seed
    )
    
    df = generator.generate_dataset(
        n_samples=args.samples,
        start_date=args.start_date
    )
    
    # Remove true regime if requested
    if args.no_regime_label:
        df = df.drop('true_regime', axis=1)
    
    # Save to CSV
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    
    # File size
    import os
    file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
    print(f"✅ Saved successfully ({file_size:.2f} MB)")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"\nYou can now train the regime classifier using:")
    print(f"  python train_regime_classifier.py --data_path {args.output}")
    print("\nFor quick testing with less data:")
    print(f"  python train_regime_classifier.py --data_path {args.output} \\")
    print(f"      --sample_size 50000 --epochs 20")
    print("="*80)


if __name__ == "__main__":
    main()

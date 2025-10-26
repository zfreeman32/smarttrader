"""
Pattern Validation Training Data Generation Pipeline
=====================================================
Processes ICT detector outputs and labels pattern quality based on forward-looking performance.

Requirements:
- ICT detectors already implemented (FVG, Order Block, Liquidity Sweep)
- Historical OHLCV data with features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm


@dataclass
class DetectedPattern:
    """Structure for a detected ICT pattern"""
    pattern_id: str
    timestamp: pd.Timestamp
    pattern_type: str  # 'FVG', 'OB', 'Sweep'
    direction: str  # 'bullish', 'bearish'
    entry_price: float
    zone_top: float
    zone_bottom: float
    
    # Pattern metadata
    pattern_size_pips: float
    volume_ratio: float  # actual volume / avg volume
    volume_confirmed: bool
    session: str  # 'Asian', 'London', 'NY'
    atr_at_detection: float
    
    # Context features
    rsi: float
    adx: float
    trend_alignment: bool
    distance_to_structure: float  # distance to nearest S/R level
    
    # 100-candle window before detection
    candle_window: np.ndarray  # shape: [100, 5] OHLCV
    volume_profile: np.ndarray  # shape: [100]
    indicator_values: np.ndarray  # shape: [n_indicators]


@dataclass
class PatternOutcome:
    """Performance metrics for a detected pattern"""
    pattern_id: str
    
    # Performance metrics
    max_favorable_excursion: float  # Best price reached (pips)
    max_adverse_excursion: float  # Worst price reached (pips)
    mfe_bars: int  # Time to reach MFE
    mae_bars: int  # Time to reach MAE
    
    # Target achievement
    reached_1R: bool
    reached_2R: bool
    reached_3R: bool
    hit_stop: bool
    
    # Timing
    bars_to_1R: Optional[int]
    bars_to_2R: Optional[int]
    bars_to_stop: Optional[int]
    
    # Quality classification
    quality_class: str  # 'high', 'medium', 'low'
    quality_score: float  # [0-1]


class PatternOutcomeLabeler:
    """Labels pattern quality based on forward-looking performance"""
    
    def __init__(
        self,
        stop_buffer_atr: float = 0.5,
        target_rr: float = 2.0,
        max_bars_forward: int = 200
    ):
        self.stop_buffer_atr = stop_buffer_atr
        self.target_rr = target_rr
        self.max_bars_forward = max_bars_forward
    
    def calculate_stop_and_target(
        self,
        pattern: DetectedPattern
    ) -> Tuple[float, float]:
        """Calculate stop loss and target based on pattern type"""
        
        if pattern.pattern_type == 'FVG':
            # Stop beyond FVG zone + buffer
            if pattern.direction == 'bullish':
                stop = pattern.zone_bottom - (pattern.atr_at_detection * self.stop_buffer_atr)
                risk = pattern.entry_price - stop
                target = pattern.entry_price + (risk * self.target_rr)
            else:  # bearish
                stop = pattern.zone_top + (pattern.atr_at_detection * self.stop_buffer_atr)
                risk = stop - pattern.entry_price
                target = pattern.entry_price - (risk * self.target_rr)
        
        elif pattern.pattern_type == 'OB':
            # Stop beyond order block + buffer
            if pattern.direction == 'bullish':
                stop = pattern.zone_bottom - (pattern.atr_at_detection * self.stop_buffer_atr)
                risk = pattern.entry_price - stop
                target = pattern.entry_price + (risk * self.target_rr)
            else:
                stop = pattern.zone_top + (pattern.atr_at_detection * self.stop_buffer_atr)
                risk = stop - pattern.entry_price
                target = pattern.entry_price - (risk * self.target_rr)
        
        elif pattern.pattern_type == 'Sweep':
            # Tighter stop for sweeps
            if pattern.direction == 'bullish':
                stop = pattern.entry_price - (pattern.atr_at_detection * 0.25)
                risk = pattern.entry_price - stop
                target = pattern.entry_price + (risk * self.target_rr)
            else:
                stop = pattern.entry_price + (pattern.atr_at_detection * 0.25)
                risk = stop - pattern.entry_price
                target = pattern.entry_price - (risk * self.target_rr)
        
        return stop, target
    
    def measure_pattern_performance(
        self,
        pattern: DetectedPattern,
        future_prices: pd.DataFrame
    ) -> PatternOutcome:
        """
        Measure pattern performance on forward-looking data
        
        Args:
            pattern: Detected pattern
            future_prices: DataFrame with columns ['high', 'low', 'close'] 
                          for bars after pattern detection
        
        Returns:
            PatternOutcome with all performance metrics
        """
        stop, target = self.calculate_stop_and_target(pattern)
        risk = abs(pattern.entry_price - stop)
        
        # Initialize metrics
        mfe = 0.0
        mae = 0.0
        mfe_bars = 0
        mae_bars = 0
        reached_1R = False
        reached_2R = False
        reached_3R = False
        hit_stop = False
        bars_to_1R = None
        bars_to_2R = None
        bars_to_stop = None
        
        # Calculate targets
        target_1R = pattern.entry_price + (risk if pattern.direction == 'bullish' else -risk)
        target_2R = pattern.entry_price + (2 * risk if pattern.direction == 'bullish' else -2 * risk)
        target_3R = pattern.entry_price + (3 * risk if pattern.direction == 'bullish' else -3 * risk)
        
        # Analyze forward price action
        for i, (idx, row) in enumerate(future_prices.iterrows()):
            if i >= self.max_bars_forward:
                break
            
            if pattern.direction == 'bullish':
                # Check favorable excursion (upside)
                if row['high'] > pattern.entry_price:
                    current_fe = row['high'] - pattern.entry_price
                    if current_fe > mfe:
                        mfe = current_fe
                        mfe_bars = i
                
                # Check adverse excursion (downside)
                if row['low'] < pattern.entry_price:
                    current_ae = pattern.entry_price - row['low']
                    if current_ae > mae:
                        mae = current_ae
                        mae_bars = i
                
                # Check targets
                if not reached_1R and row['high'] >= target_1R:
                    reached_1R = True
                    bars_to_1R = i
                if not reached_2R and row['high'] >= target_2R:
                    reached_2R = True
                    bars_to_2R = i
                if not reached_3R and row['high'] >= target_3R:
                    reached_3R = True
                
                # Check stop
                if not hit_stop and row['low'] <= stop:
                    hit_stop = True
                    bars_to_stop = i
                    break  # Exit early if stop hit
                    
            else:  # bearish
                # Check favorable excursion (downside)
                if row['low'] < pattern.entry_price:
                    current_fe = pattern.entry_price - row['low']
                    if current_fe > mfe:
                        mfe = current_fe
                        mfe_bars = i
                
                # Check adverse excursion (upside)
                if row['high'] > pattern.entry_price:
                    current_ae = row['high'] - pattern.entry_price
                    if current_ae > mae:
                        mae = current_ae
                        mae_bars = i
                
                # Check targets
                if not reached_1R and row['low'] <= target_1R:
                    reached_1R = True
                    bars_to_1R = i
                if not reached_2R and row['low'] <= target_2R:
                    reached_2R = True
                    bars_to_2R = i
                if not reached_3R and row['low'] <= target_3R:
                    reached_3R = True
                
                # Check stop
                if not hit_stop and row['high'] >= stop:
                    hit_stop = True
                    bars_to_stop = i
                    break
        
        # Convert to pips (assuming EURUSD, 1 pip = 0.0001)
        mfe_pips = mfe * 10000
        mae_pips = mae * 10000
        
        # Classify quality
        quality_class, quality_score = self._classify_quality(
            reached_1R, reached_2R, reached_3R, hit_stop, mfe_pips, mae_pips, risk * 10000
        )
        
        return PatternOutcome(
            pattern_id=pattern.pattern_id,
            max_favorable_excursion=mfe_pips,
            max_adverse_excursion=mae_pips,
            mfe_bars=mfe_bars,
            mae_bars=mae_bars,
            reached_1R=reached_1R,
            reached_2R=reached_2R,
            reached_3R=reached_3R,
            hit_stop=hit_stop,
            bars_to_1R=bars_to_1R,
            bars_to_2R=bars_to_2R,
            bars_to_stop=bars_to_stop,
            quality_class=quality_class,
            quality_score=quality_score
        )
    
    def _classify_quality(
        self,
        reached_1R: bool,
        reached_2R: bool,
        reached_3R: bool,
        hit_stop: bool,
        mfe_pips: float,
        mae_pips: float,
        risk_pips: float
    ) -> Tuple[str, float]:
        """Classify pattern quality and assign score"""
        
        # High quality: Reached 2R+ without hitting stop
        if reached_2R and not hit_stop:
            if reached_3R:
                return 'high', 0.95
            return 'high', 0.85
        
        # Medium quality: Reached 1R or had good MFE but hit stop
        if reached_1R and not hit_stop:
            return 'medium', 0.65
        
        if mfe_pips >= risk_pips * 1.5 and hit_stop:
            return 'medium', 0.55
        
        # Low quality: Hit stop quickly or minimal movement
        if hit_stop and mae_pips > mfe_pips:
            return 'low', 0.25
        
        # Default: marginal pattern
        if mfe_pips > mae_pips:
            return 'medium', 0.50
        else:
            return 'low', 0.30


class PatternDatasetGenerator:
    """Generates training dataset from ICT detector outputs"""
    
    def __init__(
        self,
        lookback_candles: int = 100,
        outcome_labeler: Optional[PatternOutcomeLabeler] = None
    ):
        self.lookback_candles = lookback_candles
        self.outcome_labeler = outcome_labeler or PatternOutcomeLabeler()
    
    def generate_training_data(
        self,
        detected_patterns: List[DetectedPattern],
        price_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Generate complete training dataset
        
        Args:
            detected_patterns: List of patterns from ICT detectors
            price_data: Full historical OHLCV DataFrame with DatetimeIndex
            save_path: Optional path to save dataset
        
        Returns:
            training_data: Dict with X and y arrays
            metadata: DataFrame with pattern details and outcomes
        """
        print(f"Generating training data for {len(detected_patterns)} patterns...")
        
        training_examples = []
        metadata_records = []
        
        for pattern in tqdm(detected_patterns):
            try:
                # Get future prices for outcome measurement
                pattern_idx = price_data.index.get_loc(pattern.timestamp)
                future_prices = price_data.iloc[pattern_idx+1:pattern_idx+201]
                
                if len(future_prices) < 50:
                    continue  # Skip if not enough forward data
                
                # Label pattern outcome
                outcome = self.outcome_labeler.measure_pattern_performance(
                    pattern, future_prices
                )
                
                # Create training example
                example = self._create_training_example(pattern, outcome)
                training_examples.append(example)
                
                # Store metadata
                metadata_records.append({
                    'pattern_id': pattern.pattern_id,
                    'timestamp': pattern.timestamp,
                    'pattern_type': pattern.pattern_type,
                    'direction': pattern.direction,
                    'quality_class': outcome.quality_class,
                    'quality_score': outcome.quality_score,
                    'reached_2R': outcome.reached_2R,
                    'hit_stop': outcome.hit_stop,
                    'mfe_pips': outcome.max_favorable_excursion,
                    'mae_pips': outcome.max_adverse_excursion
                })
                
            except Exception as e:
                print(f"Error processing pattern {pattern.pattern_id}: {e}")
                continue
        
        print(f"Successfully generated {len(training_examples)} training examples")
        
        # Convert to arrays
        training_data = self._compile_dataset(training_examples)
        metadata_df = pd.DataFrame(metadata_records)
        
        # Print statistics
        self._print_dataset_stats(metadata_df)
        
        # Save if path provided
        if save_path:
            self._save_dataset(training_data, metadata_df, save_path)
        
        return training_data, metadata_df
    
    def _create_training_example(
        self,
        pattern: DetectedPattern,
        outcome: PatternOutcome
    ) -> Dict:
        """Create a single training example"""
        
        # Normalize candle data (min-max per window)
        candles_normalized = self._normalize_candles(pattern.candle_window)
        
        # Pattern metadata features
        pattern_features = np.array([
            1.0 if pattern.pattern_type == 'FVG' else 0.0,
            1.0 if pattern.pattern_type == 'OB' else 0.0,
            1.0 if pattern.pattern_type == 'Sweep' else 0.0,
            1.0 if pattern.direction == 'bullish' else -1.0,
            pattern.pattern_size_pips / pattern.atr_at_detection,  # Normalized size
            pattern.volume_ratio,
            1.0 if pattern.volume_confirmed else 0.0,
            1.0 if pattern.session == 'London' else 0.0,
            1.0 if pattern.session == 'NY' else 0.0,
            pattern.rsi / 100.0,  # Normalize to [0, 1]
            pattern.adx / 100.0,
            1.0 if pattern.trend_alignment else 0.0,
            pattern.distance_to_structure / pattern.atr_at_detection
        ])
        
        return {
            'chart_array': candles_normalized,
            'volume_profile': pattern.volume_profile.reshape(-1, 1),
            'pattern_features': pattern_features,
            'indicator_values': pattern.indicator_values,
            'quality_score': outcome.quality_score
        }
    
    def _normalize_candles(self, candles: np.ndarray) -> np.ndarray:
        """Min-max normalize OHLCV data"""
        normalized = candles.copy()
        
        # Normalize OHLC together (maintain relative relationships)
        price_min = candles[:, :4].min()
        price_max = candles[:, :4].max()
        if price_max > price_min:
            normalized[:, :4] = (candles[:, :4] - price_min) / (price_max - price_min)
        
        # Normalize volume separately
        vol_max = candles[:, 4].max()
        if vol_max > 0:
            normalized[:, 4] = candles[:, 4] / vol_max
        
        return normalized
    
    def _compile_dataset(self, examples: List[Dict]) -> Dict:
        """Compile list of examples into numpy arrays"""
        
        X_chart = np.array([ex['chart_array'] for ex in examples])
        X_volume = np.array([ex['volume_profile'] for ex in examples])
        X_pattern = np.array([ex['pattern_features'] for ex in examples])
        X_indicators = np.array([ex['indicator_values'] for ex in examples])
        y = np.array([ex['quality_score'] for ex in examples])
        
        return {
            'X_chart': X_chart,
            'X_volume': X_volume,
            'X_pattern': X_pattern,
            'X_indicators': X_indicators,
            'y': y
        }
    
    def _print_dataset_stats(self, metadata_df: pd.DataFrame):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"\nTotal patterns: {len(metadata_df)}")
        
        print("\nBy Pattern Type:")
        print(metadata_df['pattern_type'].value_counts())
        
        print("\nBy Quality Class:")
        print(metadata_df['quality_class'].value_counts())
        
        print("\nBy Direction:")
        print(metadata_df['direction'].value_counts())
        
        print(f"\nSuccess Metrics:")
        print(f"  Reached 2R: {metadata_df['reached_2R'].sum()} ({metadata_df['reached_2R'].mean()*100:.1f}%)")
        print(f"  Hit Stop: {metadata_df['hit_stop'].sum()} ({metadata_df['hit_stop'].mean()*100:.1f}%)")
        
        print(f"\nAverage MFE: {metadata_df['mfe_pips'].mean():.2f} pips")
        print(f"Average MAE: {metadata_df['mae_pips'].mean():.2f} pips")
        
        print("\nQuality Score Distribution:")
        print(metadata_df['quality_score'].describe())
        
        print("="*50 + "\n")
    
    def _save_dataset(self, training_data: Dict, metadata_df: pd.DataFrame, save_path: str):
        """Save dataset to disk"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(save_path, 'X_chart.npy'), training_data['X_chart'])
        np.save(os.path.join(save_path, 'X_volume.npy'), training_data['X_volume'])
        np.save(os.path.join(save_path, 'X_pattern.npy'), training_data['X_pattern'])
        np.save(os.path.join(save_path, 'X_indicators.npy'), training_data['X_indicators'])
        np.save(os.path.join(save_path, 'y.npy'), training_data['y'])
        
        # Save metadata
        metadata_df.to_csv(os.path.join(save_path, 'metadata.csv'), index=False)
        
        print(f"Dataset saved to {save_path}")


# Example usage function
def example_usage():
    """Example of how to use the data generation pipeline"""
    
    # 1. Load your historical data
    price_data = pd.read_csv('eurusd_1m.csv', index_col=0, parse_dates=True)
    
    # 2. Run your ICT detectors to get detected patterns
    # This assumes you have detectors that return DetectedPattern objects
    from your_ict_detectors import FVGDetector, OrderBlockDetector, LiquiditySweepDetector
    
    fvg_detector = FVGDetector()
    ob_detector = OrderBlockDetector()
    sweep_detector = LiquiditySweepDetector()
    
    detected_patterns = []
    detected_patterns.extend(fvg_detector.detect(price_data))
    detected_patterns.extend(ob_detector.detect(price_data))
    detected_patterns.extend(sweep_detector.detect(price_data))
    
    # 3. Generate training dataset
    generator = PatternDatasetGenerator(lookback_candles=100)
    training_data, metadata = generator.generate_training_data(
        detected_patterns=detected_patterns,
        price_data=price_data,
        save_path='./pattern_validation_data'
    )
    
    print(f"\nDataset shapes:")
    print(f"  X_chart: {training_data['X_chart'].shape}")
    print(f"  X_volume: {training_data['X_volume'].shape}")
    print(f"  X_pattern: {training_data['X_pattern'].shape}")
    print(f"  y: {training_data['y'].shape}")
    
    return training_data, metadata


if __name__ == "__main__":
    # Run example
    training_data, metadata = example_usage()
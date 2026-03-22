"""
Liquidity Sweep Detector
Detects liquidity grabs and stop hunts at key levels
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .base_detector import BaseDetector, DetectionResult


class LiquiditySweepDetector(BaseDetector):
    """
    Detects Liquidity Sweeps - when price takes out recent highs/lows to grab stops
    then reverses (sweep) or continues (run)
    
    Buy-side sweep: Takes out recent highs then may reverse down
    Sell-side sweep: Takes out recent lows then may reverse up
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Liquidity Sweep Detector
        
        Args:
            config: Configuration dictionary
                - lookback_period: Bars to look back for highs/lows (default: 50)
                - cluster_tolerance_pips: Pips tolerance for level clustering (default: 3)
                - volume_threshold: Volume multiplier for confirmation (default: 1.5)
                - reversal_bars: Bars to check for reversal (default: 5)
                - reversal_threshold_pips: Pips for reversal confirmation (default: 5)
                - min_sweep_pips: Minimum sweep distance (default: 2)
        """
        default_config = {
            'lookback_period': 50,
            'cluster_tolerance_pips': 3,
            'volume_threshold': 1.5,
            'reversal_bars': 5,
            'reversal_threshold_pips': 5,
            'min_sweep_pips': 2,
            'pip_value': 0.0001
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name="LiquiditySweep_Detector", config=default_config)
    
    def detect(self, df: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect Liquidity Sweeps in the price data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, timestamp
            
        Returns:
            List of DetectionResult objects for detected sweeps
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        self.validate_data(df, required_cols)
        
        if len(df) < self.config['lookback_period'] + 10:
            return []
        
        results = []
        
        # Calculate volume average
        avg_volume = self.calculate_volume_average(df, period=20)
        
        # Detect buy-side sweeps (taking out highs)
        buy_side_sweeps = self._detect_buy_side_sweeps(df, avg_volume)
        results.extend(buy_side_sweeps)
        
        # Detect sell-side sweeps (taking out lows)
        sell_side_sweeps = self._detect_sell_side_sweeps(df, avg_volume)
        results.extend(sell_side_sweeps)
        
        return results
    
    def _find_clustered_levels(self, prices: List[float], 
                              tolerance: float) -> List[float]:
        """
        Find clustered price levels (multiple highs/lows at similar prices)
        
        Args:
            prices: List of price levels
            tolerance: Price tolerance for clustering
            
        Returns:
            List of clustered levels (average of each cluster)
        """
        if not prices:
            return []
        
        sorted_prices = sorted(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if price - current_cluster[-1] <= tolerance:
                current_cluster.append(price)
            else:
                # Average the cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _detect_buy_side_sweeps(self, df: pd.DataFrame,
                               avg_volume: pd.Series) -> List[DetectionResult]:
        """
        Detect buy-side liquidity sweeps (taking out recent highs)
        
        Args:
            df: Price data
            avg_volume: Average volume series
            
        Returns:
            List of buy-side sweep detections
        """
        results = []
        lookback = self.config['lookback_period']
        cluster_tolerance = self.config['cluster_tolerance_pips'] * self.config['pip_value']
        min_sweep = self.config['min_sweep_pips'] * self.config['pip_value']
        
        # Start from lookback position
        for i in range(lookback, len(df) - self.config['reversal_bars']):
            current_bar = df.iloc[i]
            current_high = current_bar['high']
            
            # Get recent highs in lookback window
            lookback_window = df.iloc[max(0, i - lookback):i]
            recent_highs = lookback_window['high'].tolist()
            
            # Find clustered high levels
            clustered_highs = self._find_clustered_levels(recent_highs, cluster_tolerance)
            
            if not clustered_highs:
                continue
            
            # Check if current bar swept any clustered high
            highest_cluster = max(clustered_highs)
            sweep_distance = current_high - highest_cluster
            
            if sweep_distance >= min_sweep:
                # Volume confirmation
                volume_ratio = current_bar['volume'] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 0
                
                if volume_ratio >= self.config['volume_threshold']:
                    # Check for reversal or continuation
                    future_bars = df.iloc[i + 1:min(i + 1 + self.config['reversal_bars'], len(df))]
                    
                    if len(future_bars) > 0:
                        future_low = future_bars['low'].min()
                        reversal_distance = current_high - future_low
                        reversal_threshold = self.config['reversal_threshold_pips'] * self.config['pip_value']
                        
                        is_reversal = reversal_distance >= reversal_threshold
                        sweep_type = 'reversal' if is_reversal else 'continuation'
                        
                        # Direction: bearish if reversal (price came back down), bullish if continuation
                        direction = 'bearish' if is_reversal else 'bullish'
                        
                        confidence = self._calculate_confidence(
                            sweep_distance=sweep_distance,
                            volume_ratio=volume_ratio,
                            is_reversal=is_reversal,
                            reversal_distance=reversal_distance if is_reversal else 0
                        )
                        
                        result = DetectionResult(
                            pattern_type='LiquiditySweep',
                            timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else i,
                            direction=direction,
                            entry_price=current_bar['close'],
                            confidence=confidence,
                            top_price=current_high,
                            bottom_price=highest_cluster,
                            volume_confirmation=True,
                            metadata={
                                'sweep_type': sweep_type,
                                'side': 'buy_side',
                                'swept_level': highest_cluster,
                                'sweep_distance_pips': sweep_distance / self.config['pip_value'],
                                'volume_ratio': volume_ratio,
                                'cluster_count': len([h for h in recent_highs if abs(h - highest_cluster) <= cluster_tolerance]),
                                'is_reversal': is_reversal,
                                'reversal_distance_pips': reversal_distance / self.config['pip_value'] if is_reversal else 0
                            }
                        )
                        
                        results.append(result)
                        self.store_detection(result)
        
        return results
    
    def _detect_sell_side_sweeps(self, df: pd.DataFrame,
                                avg_volume: pd.Series) -> List[DetectionResult]:
        """
        Detect sell-side liquidity sweeps (taking out recent lows)
        
        Args:
            df: Price data
            avg_volume: Average volume series
            
        Returns:
            List of sell-side sweep detections
        """
        results = []
        lookback = self.config['lookback_period']
        cluster_tolerance = self.config['cluster_tolerance_pips'] * self.config['pip_value']
        min_sweep = self.config['min_sweep_pips'] * self.config['pip_value']
        
        # Start from lookback position
        for i in range(lookback, len(df) - self.config['reversal_bars']):
            current_bar = df.iloc[i]
            current_low = current_bar['low']
            
            # Get recent lows in lookback window
            lookback_window = df.iloc[max(0, i - lookback):i]
            recent_lows = lookback_window['low'].tolist()
            
            # Find clustered low levels
            clustered_lows = self._find_clustered_levels(recent_lows, cluster_tolerance)
            
            if not clustered_lows:
                continue
            
            # Check if current bar swept any clustered low
            lowest_cluster = min(clustered_lows)
            sweep_distance = lowest_cluster - current_low
            
            if sweep_distance >= min_sweep:
                # Volume confirmation
                volume_ratio = current_bar['volume'] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 0
                
                if volume_ratio >= self.config['volume_threshold']:
                    # Check for reversal or continuation
                    future_bars = df.iloc[i + 1:min(i + 1 + self.config['reversal_bars'], len(df))]
                    
                    if len(future_bars) > 0:
                        future_high = future_bars['high'].max()
                        reversal_distance = future_high - current_low
                        reversal_threshold = self.config['reversal_threshold_pips'] * self.config['pip_value']
                        
                        is_reversal = reversal_distance >= reversal_threshold
                        sweep_type = 'reversal' if is_reversal else 'continuation'
                        
                        # Direction: bullish if reversal (price came back up), bearish if continuation
                        direction = 'bullish' if is_reversal else 'bearish'
                        
                        confidence = self._calculate_confidence(
                            sweep_distance=sweep_distance,
                            volume_ratio=volume_ratio,
                            is_reversal=is_reversal,
                            reversal_distance=reversal_distance if is_reversal else 0
                        )
                        
                        result = DetectionResult(
                            pattern_type='LiquiditySweep',
                            timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else i,
                            direction=direction,
                            entry_price=current_bar['close'],
                            confidence=confidence,
                            top_price=lowest_cluster,
                            bottom_price=current_low,
                            volume_confirmation=True,
                            metadata={
                                'sweep_type': sweep_type,
                                'side': 'sell_side',
                                'swept_level': lowest_cluster,
                                'sweep_distance_pips': sweep_distance / self.config['pip_value'],
                                'volume_ratio': volume_ratio,
                                'cluster_count': len([l for l in recent_lows if abs(l - lowest_cluster) <= cluster_tolerance]),
                                'is_reversal': is_reversal,
                                'reversal_distance_pips': reversal_distance / self.config['pip_value'] if is_reversal else 0
                            }
                        )
                        
                        results.append(result)
                        self.store_detection(result)
        
        return results
    
    def _calculate_confidence(self, sweep_distance: float, volume_ratio: float,
                             is_reversal: bool, reversal_distance: float) -> float:
        """
        Calculate confidence score for liquidity sweep detection
        
        Args:
            sweep_distance: Distance price swept beyond level
            volume_ratio: Volume compared to average
            is_reversal: Whether it's a reversal sweep
            reversal_distance: Distance of reversal if applicable
            
        Returns:
            Confidence score between 0 and 1
        """
        # Sweep distance score
        sweep_pips = sweep_distance / self.config['pip_value']
        sweep_score = min(sweep_pips / 5.0, 1.0)  # Max out at 5 pips
        
        # Volume score
        volume_score = min(volume_ratio / 3.0, 1.0)  # Max out at 3x volume
        
        # Reversal quality score (higher for stronger reversals)
        if is_reversal:
            reversal_pips = reversal_distance / self.config['pip_value']
            reversal_score = min(reversal_pips / 10.0, 1.0)  # Max out at 10 pips
        else:
            reversal_score = 0.5  # Moderate confidence for continuation
        
        # Weighted combination
        confidence = (
            sweep_score * 0.3 +
            volume_score * 0.3 +
            reversal_score * 0.4
        )
        
        return min(max(confidence, 0.0), 1.0)

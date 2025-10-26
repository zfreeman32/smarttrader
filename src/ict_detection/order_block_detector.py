"""
Order Block (OB) Detector
Identifies institutional accumulation/distribution zones
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from .base_detector import BaseDetector, DetectionResult


class OrderBlockDetector(BaseDetector):
    """
    Detects Order Blocks (OBs) - institutional accumulation/distribution zones
    
    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Order Block Detector
        
        Args:
            config: Configuration dictionary
                - swing_window: Window for swing detection (default: 20)
                - volume_threshold: Volume multiplier for confirmation (default: 2.0)
                - volume_lookback: Bars for volume average (default: 20)
                - min_move_pips: Minimum move size in pips (default: 10)
                - session_filter: Only detect during high-volume sessions (default: False)
        """
        default_config = {
            'swing_window': 20,
            'volume_threshold': 2.0,
            'volume_lookback': 20,
            'min_move_pips': 10,
            'session_filter': False,
            'pip_value': 0.0001,
            'aggregate_volume_bars': 3  # Current + 2 previous for volume check
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name="OrderBlock_Detector", config=default_config)
    
    def detect(self, df: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect Order Blocks in the price data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, timestamp
            
        Returns:
            List of DetectionResult objects for detected Order Blocks
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        self.validate_data(df, required_cols)
        
        if len(df) < self.config['swing_window'] + 5:
            return []
        
        results = []
        
        # Calculate volume average
        avg_volume = self.calculate_volume_average(df, period=self.config['volume_lookback'])
        
        # Find swing points
        swing_highs = self.find_swing_highs(df, window=self.config['swing_window'])
        swing_lows = self.find_swing_lows(df, window=self.config['swing_window'])
        
        # Detect bullish order blocks (last down candle before up move)
        bullish_obs = self._detect_bullish_order_blocks(
            df, swing_lows, avg_volume
        )
        results.extend(bullish_obs)
        
        # Detect bearish order blocks (last up candle before down move)
        bearish_obs = self._detect_bearish_order_blocks(
            df, swing_highs, avg_volume
        )
        results.extend(bearish_obs)
        
        return results
    
    def _detect_bullish_order_blocks(self, df: pd.DataFrame, 
                                    swing_lows: pd.Series,
                                    avg_volume: pd.Series) -> List[DetectionResult]:
        """
        Detect bullish order blocks
        
        Args:
            df: Price data
            swing_lows: Boolean series marking swing lows
            avg_volume: Average volume series
            
        Returns:
            List of bullish order block detections
        """
        results = []
        min_move = self.config['min_move_pips'] * self.config['pip_value']
        
        # Find swing lows
        swing_low_indices = swing_lows[swing_lows].index
        
        for swing_idx in swing_low_indices:
            swing_pos = df.index.get_loc(swing_idx)
            
            # Need at least a few bars after swing low
            if swing_pos >= len(df) - 3:
                continue
            
            swing_low_price = df.iloc[swing_pos]['low']
            
            # Look back to find the last bearish candle before the swing low
            last_bearish_idx = None
            for i in range(swing_pos, max(0, swing_pos - 10), -1):
                candle = df.iloc[i]
                if candle['close'] < candle['open']:  # Bearish candle
                    last_bearish_idx = i
                    break
            
            if last_bearish_idx is None:
                continue
            
            # Check if price moved significantly up after swing low
            future_high = df.iloc[swing_pos:min(swing_pos + 20, len(df))]['high'].max()
            move_size = future_high - swing_low_price
            
            if move_size < min_move:
                continue
            
            # Volume confirmation: aggregate volume of last bearish + 2 previous bars
            volume_bars = min(self.config['aggregate_volume_bars'], last_bearish_idx + 1)
            start_idx = max(0, last_bearish_idx - volume_bars + 1)
            aggregated_volume = df.iloc[start_idx:last_bearish_idx + 1]['volume'].sum()
            avg_vol_at_time = avg_volume.iloc[last_bearish_idx]
            
            volume_multiplier = aggregated_volume / (avg_vol_at_time * volume_bars) if avg_vol_at_time > 0 else 0
            
            if volume_multiplier >= self.config['volume_threshold']:
                ob_candle = df.iloc[last_bearish_idx]
                
                confidence = self._calculate_confidence(
                    move_size=move_size,
                    volume_multiplier=volume_multiplier,
                    candle=ob_candle
                )
                
                result = DetectionResult(
                    pattern_type='OrderBlock',
                    timestamp=df.index[swing_pos] if hasattr(df.index, 'to_pydatetime') else swing_pos,
                    direction='bullish',
                    entry_price=ob_candle['close'],
                    confidence=confidence,
                    top_price=ob_candle['high'],  # Top of OB zone
                    bottom_price=ob_candle['low'],  # Bottom of OB zone
                    volume_confirmation=True,
                    metadata={
                        'ob_candle_idx': last_bearish_idx,
                        'swing_low': swing_low_price,
                        'future_high': future_high,
                        'move_pips': move_size / self.config['pip_value'],
                        'volume_multiplier': volume_multiplier,
                        'aggregated_volume': aggregated_volume
                    }
                )
                
                results.append(result)
                self.store_detection(result)
        
        return results
    
    def _detect_bearish_order_blocks(self, df: pd.DataFrame,
                                    swing_highs: pd.Series,
                                    avg_volume: pd.Series) -> List[DetectionResult]:
        """
        Detect bearish order blocks
        
        Args:
            df: Price data
            swing_highs: Boolean series marking swing highs
            avg_volume: Average volume series
            
        Returns:
            List of bearish order block detections
        """
        results = []
        min_move = self.config['min_move_pips'] * self.config['pip_value']
        
        # Find swing highs
        swing_high_indices = swing_highs[swing_highs].index
        
        for swing_idx in swing_high_indices:
            swing_pos = df.index.get_loc(swing_idx)
            
            # Need at least a few bars after swing high
            if swing_pos >= len(df) - 3:
                continue
            
            swing_high_price = df.iloc[swing_pos]['high']
            
            # Look back to find the last bullish candle before the swing high
            last_bullish_idx = None
            for i in range(swing_pos, max(0, swing_pos - 10), -1):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:  # Bullish candle
                    last_bullish_idx = i
                    break
            
            if last_bullish_idx is None:
                continue
            
            # Check if price moved significantly down after swing high
            future_low = df.iloc[swing_pos:min(swing_pos + 20, len(df))]['low'].min()
            move_size = swing_high_price - future_low
            
            if move_size < min_move:
                continue
            
            # Volume confirmation: aggregate volume of last bullish + 2 previous bars
            volume_bars = min(self.config['aggregate_volume_bars'], last_bullish_idx + 1)
            start_idx = max(0, last_bullish_idx - volume_bars + 1)
            aggregated_volume = df.iloc[start_idx:last_bullish_idx + 1]['volume'].sum()
            avg_vol_at_time = avg_volume.iloc[last_bullish_idx]
            
            volume_multiplier = aggregated_volume / (avg_vol_at_time * volume_bars) if avg_vol_at_time > 0 else 0
            
            if volume_multiplier >= self.config['volume_threshold']:
                ob_candle = df.iloc[last_bullish_idx]
                
                confidence = self._calculate_confidence(
                    move_size=move_size,
                    volume_multiplier=volume_multiplier,
                    candle=ob_candle
                )
                
                result = DetectionResult(
                    pattern_type='OrderBlock',
                    timestamp=df.index[swing_pos] if hasattr(df.index, 'to_pydatetime') else swing_pos,
                    direction='bearish',
                    entry_price=ob_candle['close'],
                    confidence=confidence,
                    top_price=ob_candle['high'],  # Top of OB zone
                    bottom_price=ob_candle['low'],  # Bottom of OB zone
                    volume_confirmation=True,
                    metadata={
                        'ob_candle_idx': last_bullish_idx,
                        'swing_high': swing_high_price,
                        'future_low': future_low,
                        'move_pips': move_size / self.config['pip_value'],
                        'volume_multiplier': volume_multiplier,
                        'aggregated_volume': aggregated_volume
                    }
                )
                
                results.append(result)
                self.store_detection(result)
        
        return results
    
    def _calculate_confidence(self, move_size: float, volume_multiplier: float,
                            candle: pd.Series) -> float:
        """
        Calculate confidence score for Order Block detection
        
        Args:
            move_size: Size of the subsequent move
            volume_multiplier: Volume ratio compared to average
            candle: The order block candle
            
        Returns:
            Confidence score between 0 and 1
        """
        # Move size score (larger moves = higher confidence)
        move_pips = move_size / self.config['pip_value']
        move_score = min(move_pips / 30.0, 1.0)  # Max out at 30 pips
        
        # Volume score (higher volume = higher confidence)
        volume_score = min(volume_multiplier / 4.0, 1.0)  # Max out at 4x volume
        
        # Candle quality score (strong directional candle)
        candle_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        body_ratio = candle_size / candle_range if candle_range > 0 else 0
        
        # Weighted combination
        confidence = (
            move_score * 0.4 +
            volume_score * 0.4 +
            body_ratio * 0.2
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def is_price_near_ob(self, price: float, ob: DetectionResult, 
                        tolerance_atr: float = 0.5) -> bool:
        """
        Check if price is near an order block zone
        
        Args:
            price: Current price
            ob: Order block DetectionResult
            tolerance_atr: ATR multiplier for "near" threshold
            
        Returns:
            True if price is near the OB
        """
        ob_range = ob.top_price - ob.bottom_price
        tolerance = ob_range * tolerance_atr
        
        return (ob.bottom_price - tolerance) <= price <= (ob.top_price + tolerance)

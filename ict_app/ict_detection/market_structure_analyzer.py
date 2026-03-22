"""
Market Structure Break Analyzer
Identifies Break of Structure (BOS) and Change of Character (CHoCH)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base_detector import BaseDetector, DetectionResult


class MarketStructureAnalyzer(BaseDetector):
    """
    Analyzes market structure breaks
    
    BOS (Break of Structure): Price breaks recent swing in trend direction
    CHoCH (Change of Character): Price breaks counter-trend, signals reversal
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Market Structure Analyzer
        
        Args:
            config: Configuration dictionary
                - swing_window: Window for swing detection (default: 10)
                - close_break_only: Require close above/below (default: False)
                - min_break_pips: Minimum break distance (default: 3)
                - confirmation_bars: Bars to confirm break (default: 2)
        """
        default_config = {
            'swing_window': 10,
            'close_break_only': False,
            'min_break_pips': 3,
            'confirmation_bars': 2,
            'pip_value': 0.0001
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name="MarketStructure_Analyzer", config=default_config)
        
        # Track market structure state
        self.current_trend = 'neutral'  # 'uptrend', 'downtrend', 'neutral'
        self.last_swing_high = None
        self.last_swing_low = None
    
    def detect(self, df: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect Market Structure Breaks (BOS and CHoCH)
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, timestamp
            
        Returns:
            List of DetectionResult objects for detected structure breaks
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        self.validate_data(df, required_cols)
        
        if len(df) < self.config['swing_window'] * 2:
            return []
        
        results = []
        
        # Find swing points
        swing_highs = self.find_swing_highs(df, window=self.config['swing_window'])
        swing_lows = self.find_swing_lows(df, window=self.config['swing_window'])
        
        # Analyze structure breaks
        structure_breaks = self._analyze_structure_breaks(
            df, swing_highs, swing_lows
        )
        results.extend(structure_breaks)
        
        return results
    
    def _analyze_structure_breaks(self, df: pd.DataFrame,
                                 swing_highs: pd.Series,
                                 swing_lows: pd.Series) -> List[DetectionResult]:
        """
        Analyze price action for structure breaks
        
        Args:
            df: Price data
            swing_highs: Boolean series marking swing highs
            swing_lows: Boolean series marking swing lows
            
        Returns:
            List of structure break detections
        """
        results = []
        min_break = self.config['min_break_pips'] * self.config['pip_value']
        
        # Get swing high and low indices
        high_indices = swing_highs[swing_highs].index.tolist()
        low_indices = swing_lows[swing_lows].index.tolist()
        
        # Analyze each bar for potential breaks
        for i in range(self.config['swing_window'] * 2, len(df)):
            current_bar = df.iloc[i]
            
            # Find most recent swing high and low before current bar
            recent_highs = [idx for idx in high_indices if df.index.get_loc(idx) < i]
            recent_lows = [idx for idx in low_indices if df.index.get_loc(idx) < i]
            
            if not recent_highs or not recent_lows:
                continue
            
            # Get the most recent swing points
            last_high_idx = recent_highs[-1]
            last_low_idx = recent_lows[-1]
            
            last_high_price = df.loc[last_high_idx, 'high']
            last_low_price = df.loc[last_low_idx, 'low']
            
            # Determine current trend based on swing sequence
            if df.index.get_loc(last_high_idx) > df.index.get_loc(last_low_idx):
                # Most recent swing is high, potential downtrend
                potential_trend = 'downtrend'
            else:
                # Most recent swing is low, potential uptrend
                potential_trend = 'uptrend'
            
            # Check for break of high (bullish break)
            if self._is_break_confirmed(current_bar, last_high_price, 'high', df, i):
                break_distance = current_bar['high'] - last_high_price
                
                if break_distance >= min_break:
                    # Determine if BOS or CHoCH
                    if potential_trend == 'uptrend':
                        pattern = 'BOS'  # Breaking higher in uptrend = continuation
                        direction = 'bullish'
                    else:
                        pattern = 'CHoCH'  # Breaking higher in downtrend = reversal
                        direction = 'bullish'
                    
                    confidence = self._calculate_confidence(
                        break_distance=break_distance,
                        candle=current_bar,
                        pattern=pattern
                    )
                    
                    result = DetectionResult(
                        pattern_type=pattern,
                        timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else i,
                        direction=direction,
                        entry_price=current_bar['close'],
                        confidence=confidence,
                        top_price=current_bar['high'],
                        bottom_price=last_high_price,
                        volume_confirmation=False,
                        metadata={
                            'broken_level': last_high_price,
                            'break_distance_pips': break_distance / self.config['pip_value'],
                            'previous_trend': potential_trend,
                            'structure_type': 'swing_high',
                            'last_swing_idx': df.index.get_loc(last_high_idx)
                        }
                    )
                    
                    results.append(result)
                    self.store_detection(result)
                    
                    # Update trend state
                    self.current_trend = 'uptrend'
                    self.last_swing_high = last_high_price
            
            # Check for break of low (bearish break)
            elif self._is_break_confirmed(current_bar, last_low_price, 'low', df, i):
                break_distance = last_low_price - current_bar['low']
                
                if break_distance >= min_break:
                    # Determine if BOS or CHoCH
                    if potential_trend == 'downtrend':
                        pattern = 'BOS'  # Breaking lower in downtrend = continuation
                        direction = 'bearish'
                    else:
                        pattern = 'CHoCH'  # Breaking lower in uptrend = reversal
                        direction = 'bearish'
                    
                    confidence = self._calculate_confidence(
                        break_distance=break_distance,
                        candle=current_bar,
                        pattern=pattern
                    )
                    
                    result = DetectionResult(
                        pattern_type=pattern,
                        timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else i,
                        direction=direction,
                        entry_price=current_bar['close'],
                        confidence=confidence,
                        top_price=last_low_price,
                        bottom_price=current_bar['low'],
                        volume_confirmation=False,
                        metadata={
                            'broken_level': last_low_price,
                            'break_distance_pips': break_distance / self.config['pip_value'],
                            'previous_trend': potential_trend,
                            'structure_type': 'swing_low',
                            'last_swing_idx': df.index.get_loc(last_low_idx)
                        }
                    )
                    
                    results.append(result)
                    self.store_detection(result)
                    
                    # Update trend state
                    self.current_trend = 'downtrend'
                    self.last_swing_low = last_low_price
        
        return results
    
    def _is_break_confirmed(self, current_bar: pd.Series, level: float,
                          break_type: str, df: pd.DataFrame, 
                          current_idx: int) -> bool:
        """
        Check if a level break is confirmed
        
        Args:
            current_bar: Current bar data
            level: Price level to break
            break_type: 'high' or 'low'
            df: Full DataFrame
            current_idx: Current bar index
            
        Returns:
            True if break is confirmed
        """
        if self.config['close_break_only']:
            # Require close above/below level
            if break_type == 'high':
                return current_bar['close'] > level
            else:
                return current_bar['close'] < level
        else:
            # Wick break is sufficient
            if break_type == 'high':
                return current_bar['high'] > level
            else:
                return current_bar['low'] < level
    
    def _calculate_confidence(self, break_distance: float, 
                             candle: pd.Series, pattern: str) -> float:
        """
        Calculate confidence score for structure break
        
        Args:
            break_distance: Distance price broke beyond level
            candle: The breaking candle
            pattern: 'BOS' or 'CHoCH'
            
        Returns:
            Confidence score between 0 and 1
        """
        # Break distance score
        break_pips = break_distance / self.config['pip_value']
        distance_score = min(break_pips / 10.0, 1.0)  # Max out at 10 pips
        
        # Candle strength score
        candle_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        strength_score = candle_size / candle_range if candle_range > 0 else 0
        
        # Pattern score (CHoCH is more significant than BOS)
        pattern_score = 0.9 if pattern == 'CHoCH' else 0.7
        
        # Weighted combination
        confidence = (
            distance_score * 0.4 +
            strength_score * 0.3 +
            pattern_score * 0.3
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def get_current_trend(self) -> str:
        """Get current market trend state"""
        return self.current_trend
    
    def get_last_structure_levels(self) -> Dict[str, Optional[float]]:
        """Get last swing high and low levels"""
        return {
            'last_swing_high': self.last_swing_high,
            'last_swing_low': self.last_swing_low
        }

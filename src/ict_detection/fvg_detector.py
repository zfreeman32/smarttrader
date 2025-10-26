"""
Fair Value Gap (FVG) Detector
Identifies 3-candle inefficiency patterns where price leaves gaps
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from .base_detector import BaseDetector, DetectionResult


class FVGDetector(BaseDetector):
    """
    Detects Fair Value Gaps (FVGs) - 3-candle patterns with price inefficiencies
    
    Bullish FVG: Gap between candle 1 high and candle 3 low (price moves up fast)
    Bearish FVG: Gap between candle 1 low and candle 3 high (price moves down fast)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize FVG Detector
        
        Args:
            config: Configuration dictionary
                - sensitivity: 'aggressive', 'moderate', 'defensive' (default: 'moderate')
                - min_gap_pips: Minimum gap size in pips (default: 5)
                - volume_confirmation: Require volume confirmation (default: True)
                - volume_threshold: Volume multiplier threshold (default: 1.5)
        """
        default_config = {
            'sensitivity': 'moderate',
            'min_gap_pips': 5,
            'volume_confirmation': True,
            'volume_threshold': 1.5,
            'pip_value': 0.0001  # For EURUSD
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name="FVG_Detector", config=default_config)
        
        # Sensitivity settings affect minimum gap requirements
        self.sensitivity_multipliers = {
            'aggressive': 0.5,    # Detect smaller gaps
            'moderate': 1.0,      # Standard detection
            'defensive': 2.0      # Only large gaps
        }
    
    def detect(self, df: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect Fair Value Gaps in the price data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, timestamp
            
        Returns:
            List of DetectionResult objects for detected FVGs
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        self.validate_data(df, required_cols)
        
        if len(df) < 3:
            return []
        
        results = []
        
        # Calculate volume average if volume confirmation required
        if self.config['volume_confirmation']:
            avg_volume = self.calculate_volume_average(df, period=20)
        
        # Get sensitivity multiplier
        sensitivity = self.config['sensitivity']
        multiplier = self.sensitivity_multipliers.get(sensitivity, 1.0)
        min_gap = self.config['min_gap_pips'] * self.config['pip_value'] * multiplier
        
        # Iterate through data starting from index 2 (need 3 candles)
        for i in range(2, len(df)):
            candle_1 = df.iloc[i-2]
            candle_2 = df.iloc[i-1]
            candle_3 = df.iloc[i]
            
            # Check for Bullish FVG
            # Condition: candle_1.high < candle_3.low (gap between them)
            bullish_gap = candle_3['low'] - candle_1['high']
            
            if bullish_gap > min_gap:
                # No overlap between first and third candle wicks
                if candle_1['high'] < candle_3['low']:
                    # Volume confirmation
                    volume_ok = True
                    if self.config['volume_confirmation']:
                        vol_threshold = avg_volume.iloc[i] * self.config['volume_threshold']
                        volume_ok = candle_2['volume'] >= vol_threshold
                    
                    if volume_ok:
                        confidence = self._calculate_confidence(
                            gap_size=bullish_gap,
                            volume_ratio=candle_2['volume'] / avg_volume.iloc[i] if self.config['volume_confirmation'] else 1.5,
                            candle_2=candle_2
                        )
                        
                        result = DetectionResult(
                            pattern_type='FVG',
                            timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else i,
                            direction='bullish',
                            entry_price=candle_3['close'],
                            confidence=confidence,
                            top_price=candle_3['low'],  # Top of FVG zone
                            bottom_price=candle_1['high'],  # Bottom of FVG zone
                            volume_confirmation=volume_ok,
                            metadata={
                                'gap_size_pips': bullish_gap / self.config['pip_value'],
                                'candle_1_high': candle_1['high'],
                                'candle_3_low': candle_3['low'],
                                'middle_candle_idx': i-1
                            }
                        )
                        
                        results.append(result)
                        self.store_detection(result)
            
            # Check for Bearish FVG
            # Condition: candle_1.low > candle_3.high (gap between them)
            bearish_gap = candle_1['low'] - candle_3['high']
            
            if bearish_gap > min_gap:
                # No overlap between first and third candle wicks
                if candle_1['low'] > candle_3['high']:
                    # Volume confirmation
                    volume_ok = True
                    if self.config['volume_confirmation']:
                        vol_threshold = avg_volume.iloc[i] * self.config['volume_threshold']
                        volume_ok = candle_2['volume'] >= vol_threshold
                    
                    if volume_ok:
                        confidence = self._calculate_confidence(
                            gap_size=bearish_gap,
                            volume_ratio=candle_2['volume'] / avg_volume.iloc[i] if self.config['volume_confirmation'] else 1.5,
                            candle_2=candle_2
                        )
                        
                        result = DetectionResult(
                            pattern_type='FVG',
                            timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else i,
                            direction='bearish',
                            entry_price=candle_3['close'],
                            confidence=confidence,
                            top_price=candle_1['low'],  # Top of FVG zone
                            bottom_price=candle_3['high'],  # Bottom of FVG zone
                            volume_confirmation=volume_ok,
                            metadata={
                                'gap_size_pips': bearish_gap / self.config['pip_value'],
                                'candle_1_low': candle_1['low'],
                                'candle_3_high': candle_3['high'],
                                'middle_candle_idx': i-1
                            }
                        )
                        
                        results.append(result)
                        self.store_detection(result)
        
        return results
    
    def _calculate_confidence(self, gap_size: float, volume_ratio: float, 
                            candle_2: pd.Series) -> float:
        """
        Calculate confidence score for FVG detection
        
        Args:
            gap_size: Size of the gap in price units
            volume_ratio: Volume ratio compared to average
            candle_2: Middle candle (the explosive move)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from gap size (larger gaps = higher confidence)
        gap_pips = gap_size / self.config['pip_value']
        gap_score = min(gap_pips / 20.0, 1.0)  # Max out at 20 pips
        
        # Volume confirmation score
        volume_score = min(volume_ratio / 3.0, 1.0)  # Max out at 3x volume
        
        # Middle candle strength (larger candle = higher confidence)
        candle_size = abs(candle_2['close'] - candle_2['open'])
        candle_range = candle_2['high'] - candle_2['low']
        body_ratio = candle_size / candle_range if candle_range > 0 else 0
        strength_score = body_ratio  # Strong directional candle
        
        # Weighted combination
        confidence = (
            gap_score * 0.4 +
            volume_score * 0.3 +
            strength_score * 0.3
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def get_active_fvgs(self, df: pd.DataFrame, current_price: float,
                       lookback_bars: int = 50) -> List[DetectionResult]:
        """
        Get FVGs that haven't been filled yet and are still relevant
        
        Args:
            df: Current price data
            current_price: Current market price
            lookback_bars: How many bars to look back
            
        Returns:
            List of active (unfilled) FVGs
        """
        recent_detections = [d for d in self.detection_history[-lookback_bars:]
                           if d.pattern_type == 'FVG']
        
        active_fvgs = []
        
        for fvg in recent_detections:
            # Check if FVG has been filled
            if fvg.direction == 'bullish':
                # Bullish FVG is filled if price revisits the gap
                if current_price > fvg.top_price:
                    active_fvgs.append(fvg)
            else:
                # Bearish FVG is filled if price revisits the gap
                if current_price < fvg.bottom_price:
                    active_fvgs.append(fvg)
        
        return active_fvgs
    
    def is_price_in_fvg_zone(self, price: float, fvg: DetectionResult) -> bool:
        """
        Check if price is currently in an FVG zone
        
        Args:
            price: Current price
            fvg: FVG DetectionResult
            
        Returns:
            True if price is in the FVG zone
        """
        return fvg.bottom_price <= price <= fvg.top_price

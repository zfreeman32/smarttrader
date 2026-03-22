"""
Candlestick Pattern Detector
Uses TA-Lib to identify single and multi-candle reversal patterns
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from .base_detector import BaseDetector, DetectionResult


class CandlestickPatternDetector(BaseDetector):
    """
    Detects candlestick patterns using TA-Lib
    Priority patterns based on analysis showing stability concerns
    Use only as supplementary confirmation, not primary signals
    """
    
    # Priority patterns from the analysis
    PRIORITY_PATTERNS = [
        'CDL3BLACKCROWS',
        'CDL3INSIDE',
        'CDLENGULFING',
        'CDLHAMMER',
        'CDLHARAMI',
        'CDLMARUBOZU',
        'CDLSHOOTINGSTAR',
        'CDLMORNINGSTAR',
        'CDLEVENINGSTAR'
    ]
    
    # Pattern interpretations
    PATTERN_DIRECTIONS = {
        'CDL3BLACKCROWS': 'bearish',
        'CDL3INSIDE': 'neutral',
        'CDLENGULFING': 'both',  # Can be bullish or bearish
        'CDLHAMMER': 'bullish',
        'CDLHARAMI': 'both',
        'CDLMARUBOZU': 'both',
        'CDLSHOOTINGSTAR': 'bearish',
        'CDLMORNINGSTAR': 'bullish',
        'CDLEVENINGSTAR': 'bearish'
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize Candlestick Pattern Detector
        
        Args:
            config: Configuration dictionary
                - use_priority_only: Only detect priority patterns (default: True)
                - min_pattern_strength: Minimum pattern strength (default: 50)
                - confirmation_required: Require next bar confirmation (default: True)
        """
        default_config = {
            'use_priority_only': True,
            'min_pattern_strength': 50,
            'confirmation_required': True
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name="Candlestick_Detector", config=default_config)
    
    def detect(self, df: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect candlestick patterns in the price data
        
        Args:
            df: DataFrame with columns: open, high, low, close
            
        Returns:
            List of DetectionResult objects for detected patterns
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        self.validate_data(df, required_cols)
        
        if len(df) < 10:
            return []
        
        try:
            import talib
        except ImportError:
            print("Warning: TA-Lib not available. Candlestick pattern detection will be skipped.")
            print("To enable: Install TA-Lib from https://github.com/mrjbq7/ta-lib")
            return []
        
        results = []
        
        # Get pattern list to detect
        patterns_to_detect = (self.PRIORITY_PATTERNS if self.config['use_priority_only'] 
                            else self._get_all_talib_patterns())
        
        # Detect each pattern
        for pattern_name in patterns_to_detect:
            try:
                # Get TA-Lib pattern function
                pattern_func = getattr(talib, pattern_name)
                
                # Detect pattern
                pattern_result = pattern_func(
                    df['open'].values,
                    df['high'].values,
                    df['low'].values,
                    df['close'].values
                )
                
                # Find non-zero values (pattern detections)
                detections = np.where(pattern_result != 0)[0]
                
                for idx in detections:
                    # Get pattern strength (TA-Lib returns +/-100 for strength)
                    strength = abs(pattern_result[idx])
                    
                    if strength < self.config['min_pattern_strength']:
                        continue
                    
                    # Determine direction
                    direction = self._get_pattern_direction(
                        pattern_name, pattern_result[idx]
                    )
                    
                    # Check confirmation if required
                    if self.config['confirmation_required'] and idx < len(df) - 1:
                        confirmed = self._check_confirmation(df, idx, direction)
                        if not confirmed:
                            continue
                    
                    # Calculate confidence based on pattern strength
                    confidence = strength / 100.0
                    
                    result = DetectionResult(
                        pattern_type=pattern_name,
                        timestamp=df.index[idx] if hasattr(df.index, 'to_pydatetime') else idx,
                        direction=direction,
                        entry_price=df.iloc[idx]['close'],
                        confidence=confidence,
                        top_price=df.iloc[idx]['high'],
                        bottom_price=df.iloc[idx]['low'],
                        volume_confirmation=False,
                        metadata={
                            'pattern_strength': int(strength),
                            'raw_signal': int(pattern_result[idx]),
                            'candle_idx': idx
                        }
                    )
                    
                    results.append(result)
                    self.store_detection(result)
                    
            except AttributeError:
                # Pattern function not available in this TA-Lib version
                continue
        
        return results
    
    def _get_pattern_direction(self, pattern_name: str, signal: int) -> str:
        """
        Determine pattern direction based on pattern type and signal
        
        Args:
            pattern_name: Name of the pattern
            signal: TA-Lib signal value (+100 bullish, -100 bearish)
            
        Returns:
            Direction: 'bullish', 'bearish', or 'neutral'
        """
        base_direction = self.PATTERN_DIRECTIONS.get(pattern_name, 'neutral')
        
        if base_direction == 'both':
            # Direction determined by signal
            return 'bullish' if signal > 0 else 'bearish'
        elif base_direction == 'neutral':
            # Some patterns are neutral, check signal
            return 'bullish' if signal > 0 else 'bearish' if signal < 0 else 'neutral'
        else:
            return base_direction
    
    def _check_confirmation(self, df: pd.DataFrame, idx: int, 
                          expected_direction: str) -> bool:
        """
        Check if next bar confirms the pattern
        
        Args:
            df: Price data
            idx: Current bar index
            expected_direction: Expected direction from pattern
            
        Returns:
            True if confirmed
        """
        if idx >= len(df) - 1:
            return False
        
        current_close = df.iloc[idx]['close']
        next_close = df.iloc[idx + 1]['close']
        
        if expected_direction == 'bullish':
            # Next bar should close higher
            return next_close > current_close
        elif expected_direction == 'bearish':
            # Next bar should close lower
            return next_close < current_close
        else:
            # Neutral patterns don't need confirmation
            return True
    
    def _get_all_talib_patterns(self) -> List[str]:
        """
        Get list of all available TA-Lib candlestick patterns
        
        Returns:
            List of pattern function names
        """
        try:
            import talib
            # Get all pattern functions (start with CDL)
            all_functions = [func for func in dir(talib) 
                           if func.startswith('CDL') and func.isupper()]
            return all_functions
        except ImportError:
            return self.PRIORITY_PATTERNS
    
    def get_pattern_description(self, pattern_name: str) -> Dict[str, str]:
        """
        Get description and interpretation of a pattern
        
        Args:
            pattern_name: Pattern name
            
        Returns:
            Dictionary with pattern information
        """
        descriptions = {
            'CDL3BLACKCROWS': {
                'name': 'Three Black Crows',
                'type': 'Reversal',
                'direction': 'Bearish',
                'description': 'Three consecutive long bearish candles',
                'reliability': 'High',
                'note': 'Strong bearish reversal signal'
            },
            'CDL3INSIDE': {
                'name': 'Three Inside Up/Down',
                'type': 'Reversal',
                'direction': 'Both',
                'description': 'Three-candle pattern with inside bars',
                'reliability': 'Medium',
                'note': 'Wait for direction confirmation'
            },
            'CDLENGULFING': {
                'name': 'Engulfing Pattern',
                'type': 'Reversal',
                'direction': 'Both',
                'description': 'Large candle engulfs previous candle',
                'reliability': 'High',
                'note': 'Strong reversal signal when at extremes'
            },
            'CDLHAMMER': {
                'name': 'Hammer',
                'type': 'Reversal',
                'direction': 'Bullish',
                'description': 'Small body, long lower shadow',
                'reliability': 'Medium-High',
                'note': 'Best at support levels'
            },
            'CDLHARAMI': {
                'name': 'Harami Pattern',
                'type': 'Reversal',
                'direction': 'Both',
                'description': 'Small candle inside previous large candle',
                'reliability': 'Medium',
                'note': 'Indecision pattern, wait for confirmation'
            },
            'CDLMARUBOZU': {
                'name': 'Marubozu',
                'type': 'Continuation',
                'direction': 'Both',
                'description': 'No shadows, strong directional candle',
                'reliability': 'Medium',
                'note': 'Shows strong momentum'
            },
            'CDLSHOOTINGSTAR': {
                'name': 'Shooting Star',
                'type': 'Reversal',
                'direction': 'Bearish',
                'description': 'Small body, long upper shadow',
                'reliability': 'Medium-High',
                'note': 'Best at resistance levels'
            },
            'CDLMORNINGSTAR': {
                'name': 'Morning Star',
                'type': 'Reversal',
                'direction': 'Bullish',
                'description': 'Three-candle bullish reversal pattern',
                'reliability': 'High',
                'note': 'Strong bullish reversal signal'
            },
            'CDLEVENINGSTAR': {
                'name': 'Evening Star',
                'type': 'Reversal',
                'direction': 'Bearish',
                'description': 'Three-candle bearish reversal pattern',
                'reliability': 'High',
                'note': 'Strong bearish reversal signal'
            }
        }
        
        return descriptions.get(pattern_name, {
            'name': pattern_name,
            'type': 'Unknown',
            'direction': 'Unknown',
            'description': 'Pattern description not available',
            'reliability': 'Unknown',
            'note': 'Use with caution'
        })

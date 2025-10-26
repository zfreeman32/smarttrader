"""
Base Detector Class for ICT Pattern Detection
Provides common functionality for all pattern detectors
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionResult:
    """Standard format for detection results"""
    pattern_type: str
    timestamp: datetime
    direction: str  # 'bullish' or 'bearish'
    entry_price: float
    confidence: float
    top_price: Optional[float] = None
    bottom_price: Optional[float] = None
    volume_confirmation: bool = False
    metadata: Optional[Dict] = None


class BaseDetector(ABC):
    """
    Abstract base class for all ICT pattern detectors
    Provides common functionality and enforces interface
    """
    
    def __init__(self, name: str, config: Dict = None):
        """
        Initialize detector
        
        Args:
            name: Detector name
            config: Configuration dictionary with detector-specific parameters
        """
        self.name = name
        self.config = config or {}
        self.detection_history = []
        
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> List[DetectionResult]:
        """
        Main detection method - must be implemented by subclasses
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            List of DetectionResult objects
        """
        pass
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame has required columns
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{self.name}: Missing required columns: {missing}")
        return True
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_average(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate rolling average volume
        
        Args:
            df: DataFrame with volume data
            period: Rolling window period
            
        Returns:
            Series with average volume
        """
        return df['volume'].rolling(window=period).mean()
    
    def is_volume_spike(self, current_volume: float, avg_volume: float, 
                       threshold: float = 2.0) -> bool:
        """
        Check if current volume is a spike compared to average
        
        Args:
            current_volume: Current bar volume
            avg_volume: Average volume
            threshold: Multiplier threshold (default 2x)
            
        Returns:
            True if volume spike detected
        """
        return current_volume > (avg_volume * threshold)
    
    def find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Find swing high points using rolling window
        
        Args:
            df: DataFrame with price data
            window: Window size for swing detection
            
        Returns:
            Boolean series marking swing highs
        """
        highs = df['high']
        swing_highs = pd.Series(False, index=df.index)
        
        for i in range(window, len(df) - window):
            left_max = highs.iloc[i-window:i].max()
            right_max = highs.iloc[i+1:i+window+1].max()
            current = highs.iloc[i]
            
            if current > left_max and current > right_max:
                swing_highs.iloc[i] = True
                
        return swing_highs
    
    def find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Find swing low points using rolling window
        
        Args:
            df: DataFrame with price data
            window: Window size for swing detection
            
        Returns:
            Boolean series marking swing lows
        """
        lows = df['low']
        swing_lows = pd.Series(False, index=df.index)
        
        for i in range(window, len(df) - window):
            left_min = lows.iloc[i-window:i].min()
            right_min = lows.iloc[i+1:i+window+1].min()
            current = lows.iloc[i]
            
            if current < left_min and current < right_min:
                swing_lows.iloc[i] = True
                
        return swing_lows
    
    def get_recent_extremes(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
        """
        Get recent high and low extremes
        
        Args:
            df: DataFrame with price data
            lookback: Number of bars to look back
            
        Returns:
            Tuple of (recent_high, recent_low)
        """
        recent_data = df.tail(lookback)
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        
        return recent_high, recent_low
    
    def store_detection(self, result: DetectionResult):
        """
        Store detection result in history
        
        Args:
            result: DetectionResult object
        """
        self.detection_history.append(result)
        
        # Keep only last 1000 detections to prevent memory issues
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with detection statistics
        """
        if not self.detection_history:
            return {
                'total_detections': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_confidence': 0.0
            }
        
        bullish = sum(1 for d in self.detection_history if d.direction == 'bullish')
        bearish = sum(1 for d in self.detection_history if d.direction == 'bearish')
        avg_conf = np.mean([d.confidence for d in self.detection_history])
        
        return {
            'total_detections': len(self.detection_history),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'avg_confidence': float(avg_conf)
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"{self.name}(detections={stats['total_detections']}, avg_conf={stats['avg_confidence']:.3f})"

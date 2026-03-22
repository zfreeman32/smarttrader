"""
ICT Pattern Coordinator
Manages all ICT detectors and coordinates pattern detection
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from .base_detector import DetectionResult
from .fvg_detector import FVGDetector
from .order_block_detector import OrderBlockDetector
from .liquidity_sweep_detector import LiquiditySweepDetector
from .market_structure_analyzer import MarketStructureAnalyzer
from .session_filter import SessionFilter, TradingSession
from .candlestick_detector import CandlestickPatternDetector


class ICTPatternCoordinator:
    """
    Coordinates all ICT pattern detectors and provides unified interface
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Pattern Coordinator
        
        Args:
            config: Configuration dictionary for all detectors
        """
        self.config = config or {}
        
        # Initialize all detectors
        self.fvg_detector = FVGDetector(
            self.config.get('fvg', {})
        )
        
        self.ob_detector = OrderBlockDetector(
            self.config.get('order_block', {})
        )
        
        self.sweep_detector = LiquiditySweepDetector(
            self.config.get('liquidity_sweep', {})
        )
        
        self.structure_analyzer = MarketStructureAnalyzer(
            self.config.get('market_structure', {})
        )
        
        self.session_filter = SessionFilter(
            self.config.get('session_filter', {}).get('timezone', 'US/Eastern')
        )
        
        self.candlestick_detector = CandlestickPatternDetector(
            self.config.get('candlestick', {})
        )
        
        # All detections storage
        self.all_detections = []
        
    def detect_all_patterns(self, df: pd.DataFrame, 
                          use_session_filter: bool = True) -> Dict[str, List[DetectionResult]]:
        """
        Run all detectors on the data
        
        Args:
            df: DataFrame with OHLCV data
            use_session_filter: Apply session filtering
            
        Returns:
            Dictionary with pattern type as key and list of detections as value
        """
        results = {
            'FVG': [],
            'OrderBlock': [],
            'LiquiditySweep': [],
            'BOS': [],
            'CHoCH': [],
            'Candlestick': []
        }
        
        # Detect FVGs
        fvg_detections = self.fvg_detector.detect(df)
        if use_session_filter:
            fvg_detections = self._filter_by_session(fvg_detections, 'FVG')
        results['FVG'] = fvg_detections
        
        # Detect Order Blocks
        ob_detections = self.ob_detector.detect(df)
        if use_session_filter:
            ob_detections = self._filter_by_session(ob_detections, 'OrderBlock')
        results['OrderBlock'] = ob_detections
        
        # Detect Liquidity Sweeps
        sweep_detections = self.sweep_detector.detect(df)
        if use_session_filter:
            sweep_detections = self._filter_by_session(sweep_detections, 'LiquiditySweep')
        results['LiquiditySweep'] = sweep_detections
        
        # Analyze Market Structure
        structure_detections = self.structure_analyzer.detect(df)
        for detection in structure_detections:
            if detection.pattern_type == 'BOS':
                results['BOS'].append(detection)
            elif detection.pattern_type == 'CHoCH':
                results['CHoCH'].append(detection)
        
        # Detect Candlestick Patterns
        candlestick_detections = self.candlestick_detector.detect(df)
        results['Candlestick'] = candlestick_detections
        
        # Store all detections
        for pattern_list in results.values():
            self.all_detections.extend(pattern_list)
        
        return results
    
    def _filter_by_session(self, detections: List[DetectionResult],
                          pattern_type: str) -> List[DetectionResult]:
        """
        Filter detections based on session effectiveness
        
        Args:
            detections: List of detections
            pattern_type: Type of pattern
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for detection in detections:
            if self.session_filter.should_trade_pattern(
                pattern_type, 
                detection.timestamp,
                min_score=0.7
            ):
                filtered.append(detection)
        
        return filtered
    
    def get_confluence_score(self, df: pd.DataFrame, bar_idx: int,
                           window: int = 10) -> Dict[str, float]:
        """
        Calculate confluence score for a specific bar
        Multiple patterns occurring together increase confidence
        
        Args:
            df: Price data
            bar_idx: Bar index to check
            window: Window around bar to check for patterns
            
        Returns:
            Dictionary with confluence information
        """
        # Get timestamp for the bar
        timestamp = df.index[bar_idx] if hasattr(df.index, 'to_pydatetime') else bar_idx
        current_price = df.iloc[bar_idx]['close']
        
        # Count patterns near this bar
        bullish_count = 0
        bearish_count = 0
        
        # Check recent detections
        recent_detections = [d for d in self.all_detections
                           if abs(self._get_bar_distance(d.timestamp, timestamp)) <= window]
        
        for detection in recent_detections:
            if detection.direction == 'bullish':
                bullish_count += detection.confidence
            elif detection.direction == 'bearish':
                bearish_count += detection.confidence
        
        # Calculate confluence
        total_patterns = bullish_count + bearish_count
        
        if total_patterns == 0:
            return {
                'confluence_score': 0.0,
                'direction': 'neutral',
                'bullish_weight': 0.0,
                'bearish_weight': 0.0,
                'pattern_count': 0
            }
        
        bullish_ratio = bullish_count / total_patterns
        bearish_ratio = bearish_count / total_patterns
        
        # Determine dominant direction
        if bullish_ratio > 0.6:
            direction = 'bullish'
            score = bullish_ratio
        elif bearish_ratio > 0.6:
            direction = 'bearish'
            score = bearish_ratio
        else:
            direction = 'mixed'
            score = 0.5
        
        return {
            'confluence_score': score,
            'direction': direction,
            'bullish_weight': bullish_ratio,
            'bearish_weight': bearish_ratio,
            'pattern_count': len(recent_detections)
        }
    
    def _get_bar_distance(self, timestamp1, timestamp2) -> int:
        """Calculate distance in bars between two timestamps"""
        if isinstance(timestamp1, (int, np.integer)):
            return abs(timestamp1 - timestamp2)
        # For datetime objects
        return 0  # Simplified - would need actual bar counting
    
    def get_active_patterns(self, df: pd.DataFrame, current_bar: int,
                          lookback_bars: int = 50) -> Dict[str, List[DetectionResult]]:
        """
        Get patterns that are still active/relevant
        
        Args:
            df: Price data
            current_bar: Current bar index
            lookback_bars: How many bars to look back
            
        Returns:
            Dictionary of active patterns by type
        """
        current_price = df.iloc[current_bar]['close']
        
        active = {
            'FVG': [],
            'OrderBlock': [],
            'LiquiditySweep': [],
            'BOS': [],
            'CHoCH': []
        }
        
        # Check FVGs - are they still unfilled?
        active['FVG'] = self.fvg_detector.get_active_fvgs(
            df, current_price, lookback_bars
        )
        
        # Get other recent patterns
        start_bar = max(0, current_bar - lookback_bars)
        
        for detection in self.all_detections:
            # Skip if too old
            if isinstance(detection.timestamp, int):
                if detection.timestamp < start_bar:
                    continue
            
            pattern_type = detection.pattern_type
            if pattern_type in active:
                active[pattern_type].append(detection)
        
        return active
    
    def generate_trading_signals(self, df: pd.DataFrame, current_bar: int,
                                min_confluence: float = 0.7) -> Dict:
        """
        Generate trading signals based on pattern confluence
        
        Args:
            df: Price data
            current_bar: Current bar index
            min_confluence: Minimum confluence score required
            
        Returns:
            Dictionary with signal information
        """
        # Get confluence at current bar
        confluence = self.get_confluence_score(df, current_bar)
        
        # Get active patterns
        active_patterns = self.get_active_patterns(df, current_bar)
        
        # Get current session
        timestamp = df.index[current_bar] if hasattr(df.index, 'to_pydatetime') else datetime.now()
        session_info = self.session_filter.get_current_session_info(timestamp)
        
        # Generate signal
        signal = 'HOLD'
        confidence = 0.0
        
        if confluence['confluence_score'] >= min_confluence:
            if confluence['direction'] == 'bullish':
                signal = 'LONG'
                confidence = confluence['confluence_score']
            elif confluence['direction'] == 'bearish':
                signal = 'SHORT'
                confidence = confluence['confluence_score']
        
        return {
            'signal': signal,
            'confidence': confidence,
            'confluence': confluence,
            'active_patterns': {k: len(v) for k, v in active_patterns.items()},
            'session': session_info['session'],
            'session_optimal_patterns': session_info['optimal_patterns'],
            'timestamp': timestamp
        }
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for all detectors
        
        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            'fvg': self.fvg_detector.get_statistics(),
            'order_block': self.ob_detector.get_statistics(),
            'liquidity_sweep': self.sweep_detector.get_statistics(),
            'market_structure': self.structure_analyzer.get_statistics(),
            'candlestick': self.candlestick_detector.get_statistics(),
            'total_detections': len(self.all_detections),
            'session_stats': self.session_filter.get_all_session_stats()
        }
    
    def reset(self):
        """Reset all detectors and clear detection history"""
        self.all_detections = []
        self.fvg_detector.detection_history = []
        self.ob_detector.detection_history = []
        self.sweep_detector.detection_history = []
        self.structure_analyzer.detection_history = []
        self.candlestick_detector.detection_history = []

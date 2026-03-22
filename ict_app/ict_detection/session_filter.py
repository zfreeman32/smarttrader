"""
Session-Based Context Filter
Filters patterns and trades based on optimal market sessions
"""
import pandas as pd
from datetime import time
from typing import List, Dict, Optional
from enum import Enum


class TradingSession(Enum):
    """Trading session definitions"""
    ASIAN = "asian"
    LONDON_OPEN = "london_open"
    LONDON = "london"
    NY_OPEN = "ny_open"
    NY = "ny"
    UNKNOWN = "unknown"


class SessionFilter:
    """
    Filter trades based on market sessions and their characteristics
    
    Sessions (EST timezone):
    - Asian (8PM-2AM): Range-bound, liquidity setup
    - London Open (2AM-5AM): High volatility, liquidity sweeps
    - London (5AM-11AM): Trending moves
    - NY Open (9:30AM-11AM): Reversal patterns, FVG fills
    - NY (9AM-4PM): Order block retests
    """
    
    # Session time ranges (EST)
    SESSION_TIMES = {
        TradingSession.ASIAN: (time(20, 0), time(2, 0)),  # 8PM-2AM
        TradingSession.LONDON_OPEN: (time(2, 0), time(5, 0)),  # 2AM-5AM
        TradingSession.LONDON: (time(5, 0), time(11, 0)),  # 5AM-11AM
        TradingSession.NY_OPEN: (time(9, 30), time(11, 0)),  # 9:30AM-11AM
        TradingSession.NY: (time(9, 0), time(16, 0))  # 9AM-4PM
    }
    
    # Pattern effectiveness by session
    PATTERN_SESSION_MATRIX = {
        'FVG': {
            TradingSession.ASIAN: 0.6,  # Moderate in ranging markets
            TradingSession.LONDON_OPEN: 0.8,  # Good during volatility
            TradingSession.LONDON: 0.9,  # Excellent during trends
            TradingSession.NY_OPEN: 0.95,  # Best - FVG fills common
            TradingSession.NY: 0.85  # Very good
        },
        'OrderBlock': {
            TradingSession.ASIAN: 0.5,  # Less effective in range
            TradingSession.LONDON_OPEN: 0.7,  # Moderate
            TradingSession.LONDON: 0.85,  # Good trending moves
            TradingSession.NY_OPEN: 0.8,  # Good
            TradingSession.NY: 0.9  # Excellent for retests
        },
        'LiquiditySweep': {
            TradingSession.ASIAN: 0.7,  # Setup for sweeps
            TradingSession.LONDON_OPEN: 0.95,  # Excellent for sweeps
            TradingSession.LONDON: 0.8,  # Good
            TradingSession.NY_OPEN: 0.85,  # Very good
            TradingSession.NY: 0.75  # Moderate
        },
        'BOS': {
            TradingSession.ASIAN: 0.5,  # Range-bound
            TradingSession.LONDON_OPEN: 0.8,  # Good
            TradingSession.LONDON: 0.9,  # Excellent trending
            TradingSession.NY_OPEN: 0.85,  # Very good
            TradingSession.NY: 0.8  # Good
        },
        'CHoCH': {
            TradingSession.ASIAN: 0.6,  # Moderate
            TradingSession.LONDON_OPEN: 0.85,  # Very good for reversals
            TradingSession.LONDON: 0.8,  # Good
            TradingSession.NY_OPEN: 0.9,  # Excellent for reversals
            TradingSession.NY: 0.75  # Moderate
        }
    }
    
    def __init__(self, timezone: str = 'US/Eastern'):
        """
        Initialize Session Filter
        
        Args:
            timezone: Timezone for session times (default: US/Eastern)
        """
        self.timezone = timezone
        self.session_stats = {session: {'trades': 0, 'wins': 0} 
                            for session in TradingSession}
    
    def get_session(self, timestamp: pd.Timestamp) -> TradingSession:
        """
        Determine which trading session a timestamp belongs to
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            TradingSession enum
        """
        # Convert to EST if needed
        if timestamp.tz is not None:
            timestamp = timestamp.tz_convert('US/Eastern')
        
        current_time = timestamp.time()
        
        # Check each session (order matters for overlaps)
        # NY_OPEN is subset of NY, check it first
        ny_open_start, ny_open_end = self.SESSION_TIMES[TradingSession.NY_OPEN]
        if ny_open_start <= current_time <= ny_open_end:
            return TradingSession.NY_OPEN
        
        # Check other sessions
        for session, (start_time, end_time) in self.SESSION_TIMES.items():
            if session == TradingSession.NY_OPEN:
                continue
            
            # Handle sessions crossing midnight (like Asian)
            if start_time > end_time:
                # Crosses midnight
                if current_time >= start_time or current_time <= end_time:
                    return session
            else:
                if start_time <= current_time <= end_time:
                    return session
        
        return TradingSession.UNKNOWN
    
    def get_pattern_session_score(self, pattern_type: str, 
                                  session: TradingSession) -> float:
        """
        Get effectiveness score for a pattern in a specific session
        
        Args:
            pattern_type: Pattern type (FVG, OrderBlock, etc.)
            session: Trading session
            
        Returns:
            Effectiveness score (0-1)
        """
        if pattern_type not in self.PATTERN_SESSION_MATRIX:
            return 0.5  # Default moderate score
        
        return self.PATTERN_SESSION_MATRIX[pattern_type].get(session, 0.5)
    
    def should_trade_pattern(self, pattern_type: str, timestamp: pd.Timestamp,
                            min_score: float = 0.7) -> bool:
        """
        Determine if a pattern should be traded based on session
        
        Args:
            pattern_type: Pattern type
            timestamp: When pattern occurred
            min_score: Minimum effectiveness score required
            
        Returns:
            True if pattern should be traded
        """
        session = self.get_session(timestamp)
        score = self.get_pattern_session_score(pattern_type, session)
        
        return score >= min_score
    
    def get_session_characteristics(self, session: TradingSession) -> Dict[str, str]:
        """
        Get characteristics of a trading session
        
        Args:
            session: Trading session
            
        Returns:
            Dictionary with session characteristics
        """
        characteristics = {
            TradingSession.ASIAN: {
                'volatility': 'low',
                'trend': 'range-bound',
                'best_for': 'liquidity setup, identifying ranges',
                'avoid': 'breakout trades',
                'typical_behavior': 'Consolidation, setting up levels for London'
            },
            TradingSession.LONDON_OPEN: {
                'volatility': 'high',
                'trend': 'volatile, directional',
                'best_for': 'liquidity sweeps, breakouts',
                'avoid': 'range-bound strategies',
                'typical_behavior': 'High impact moves, stop hunts, strong directional bias'
            },
            TradingSession.LONDON: {
                'volatility': 'medium-high',
                'trend': 'trending',
                'best_for': 'trend following, FVGs, order blocks',
                'avoid': 'counter-trend trades',
                'typical_behavior': 'Sustained directional moves, respect of structure'
            },
            TradingSession.NY_OPEN: {
                'volatility': 'high',
                'trend': 'volatile, reversal potential',
                'best_for': 'reversals, FVG fills, CHoCH',
                'avoid': 'premature entries',
                'typical_behavior': 'High volume, potential reversals, FVG revisits'
            },
            TradingSession.NY: {
                'volatility': 'medium',
                'trend': 'continuation or reversal',
                'best_for': 'order block retests, trend continuation',
                'avoid': 'late session trades (after 2PM)',
                'typical_behavior': 'Follow-through on morning moves, respect of key levels'
            }
        }
        
        return characteristics.get(session, {
            'volatility': 'unknown',
            'trend': 'unknown',
            'best_for': 'unknown',
            'avoid': 'unknown',
            'typical_behavior': 'Unknown session'
        })
    
    def get_optimal_patterns_for_session(self, session: TradingSession,
                                        min_score: float = 0.8) -> List[str]:
        """
        Get list of optimal patterns for a session
        
        Args:
            session: Trading session
            min_score: Minimum effectiveness score
            
        Returns:
            List of pattern types optimal for this session
        """
        optimal_patterns = []
        
        for pattern_type, session_scores in self.PATTERN_SESSION_MATRIX.items():
            if session_scores.get(session, 0) >= min_score:
                optimal_patterns.append(pattern_type)
        
        return optimal_patterns
    
    def update_session_stats(self, session: TradingSession, won: bool):
        """
        Update session statistics with trade outcome
        
        Args:
            session: Trading session
            won: Whether trade was won
        """
        self.session_stats[session]['trades'] += 1
        if won:
            self.session_stats[session]['wins'] += 1
    
    def get_session_win_rate(self, session: TradingSession) -> Optional[float]:
        """
        Get win rate for a specific session
        
        Args:
            session: Trading session
            
        Returns:
            Win rate (0-1) or None if no trades
        """
        stats = self.session_stats[session]
        if stats['trades'] == 0:
            return None
        
        return stats['wins'] / stats['trades']
    
    def get_all_session_stats(self) -> Dict:
        """
        Get statistics for all sessions
        
        Returns:
            Dictionary with session statistics
        """
        stats = {}
        
        for session in TradingSession:
            win_rate = self.get_session_win_rate(session)
            stats[session.value] = {
                'trades': self.session_stats[session]['trades'],
                'wins': self.session_stats[session]['wins'],
                'win_rate': win_rate if win_rate is not None else 0.0
            }
        
        return stats
    
    def get_current_session_info(self, timestamp: pd.Timestamp) -> Dict:
        """
        Get comprehensive information about current session
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary with session information
        """
        session = self.get_session(timestamp)
        characteristics = self.get_session_characteristics(session)
        optimal_patterns = self.get_optimal_patterns_for_session(session)
        win_rate = self.get_session_win_rate(session)
        
        return {
            'session': session.value,
            'characteristics': characteristics,
            'optimal_patterns': optimal_patterns,
            'win_rate': win_rate,
            'timestamp': timestamp
        }

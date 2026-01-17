"""
ICT Feature Engineering Pipeline - FIXED VERSION
=================================================
Transforms raw OHLCV data into ICT-specific features for ML training.

FIXES APPLIED:
1. Lookahead leakage eliminated in swing detection (shifted by strength bars)
2. Order Block detection fixed - uses past-only confirmation
3. Liquidity sweep detection O(N) using incremental last_swing_high/low
4. body_pct clipped to [0, 1] for sanity
5. Volatility scaling auto-detects bar frequency
6. Session flags use minute-of-day for precision
7. Column naming normalized before reference
8. Cleanup uses warmup trimming instead of blanket bfill
9. FVG detection enhanced with displacement direction check
10. Swing confirmation properly delayed (no lookahead)

Features Built:
- Market Structure (swings, trends, BOS)
- Time/Session Features (kill zones, sessions)
- Price Action (FVG, displacement, OB)
- Technical Indicators (ATR, volume ratios)
- Premium/Discount Zones
- Key Level Distances

Input: Raw OHLCV CSV
Output: Feature-enriched CSV

Author: ICT ML System
Version: 2.0 (Fixed)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    
    # Swing Detection
    swing_strength: int = 5          # Bars on each side for swing confirmation
    swing_lookback: int = 50         # Bars to look back for swings
    
    # ATR Settings
    atr_period: int = 14
    atr_period_long: int = 50
    
    # FVG Settings
    min_fvg_atr: float = 0.3         # Minimum FVG size in ATR
    max_fvg_lookback: int = 20       # Bars to look back for FVG
    
    # Displacement
    displacement_atr_min: float = 1.2
    displacement_body_ratio: float = 0.6
    
    # Volume
    volume_lookback: int = 20
    volume_spike_threshold: float = 1.5
    
    # Session Times (ET hours:minutes) - More precise than hour-only
    # Asian Session: 7:00 PM - 12:00 AM ET (crosses midnight)
    asian_start_hour: int = 19
    asian_start_min: int = 0
    asian_end_hour: int = 0
    asian_end_min: int = 0
    
    # London Kill Zone: 2:00 AM - 5:00 AM ET
    london_kz_start_hour: int = 2
    london_kz_start_min: int = 0
    london_kz_end_hour: int = 5
    london_kz_end_min: int = 0
    
    # NY AM Kill Zone: 9:30 AM - 12:00 PM ET (precise open at 9:30)
    nyam_kz_start_hour: int = 9
    nyam_kz_start_min: int = 30
    nyam_kz_end_hour: int = 12
    nyam_kz_end_min: int = 0
    
    # NY PM Kill Zone: 2:00 PM - 4:00 PM ET
    nypm_kz_start_hour: int = 14
    nypm_kz_start_min: int = 0
    nypm_kz_end_hour: int = 16
    nypm_kz_end_min: int = 0
    
    # Silver Bullet Windows (precise to minute)
    # London SB: 3:00 AM - 4:00 AM ET (also known as 10-11 AM London)
    london_sb_start_hour: int = 3
    london_sb_start_min: int = 0
    london_sb_end_hour: int = 4
    london_sb_end_min: int = 0
    
    # NY AM SB: 10:00 AM - 11:00 AM ET
    nyam_sb_start_hour: int = 10
    nyam_sb_start_min: int = 0
    nyam_sb_end_hour: int = 11
    nyam_sb_end_min: int = 0
    
    # NY PM SB: 2:00 PM - 3:00 PM ET
    nypm_sb_start_hour: int = 14
    nypm_sb_start_min: int = 0
    nypm_sb_end_hour: int = 15
    nypm_sb_end_min: int = 0
    
    # Premium/Discount
    pd_lookback: int = 50
    discount_threshold: float = 40.0  # Below 40% = discount
    premium_threshold: float = 60.0   # Above 60% = premium
    
    # Key Levels
    pdhl_enabled: bool = True        # Previous Day High/Low
    pwhl_enabled: bool = True        # Previous Week High/Low
    
    # Order Block confirmation
    ob_confirmation_bars: int = 3    # Bars to look back for OB confirmation
    ob_min_displacement_atr: float = 1.5  # Min move after OB for validation
    
    # Warmup period (bars to drop at start due to indicator warmup)
    warmup_period: int = 200


# =============================================================================
# DATA LOADING
# =============================================================================

class OHLCVLoader:
    """Load and parse OHLCV data from CSV"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV.
        
        Expected format:
        Date,Time,Open,High,Low,Close,Volume
        20241203,08:55:00,1.05226,1.05249,1.05226,1.05238,292
        
        Also handles already-lowercase columns.
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # FIX: Normalize columns BEFORE referencing them
        df.columns = df.columns.str.lower().str.strip()
        
        # Parse datetime - handle both Date/Time and datetime columns
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['date'].astype(str) + ' ' + df['time'].astype(str),
                format='%Y%m%d %H:%M:%S'
            )
        else:
            raise ValueError("CSV must have either 'datetime' column or 'date' and 'time' columns")
        
        # Set datetime as index but keep as column too
        df = df.set_index('datetime', drop=False)
        df = df.sort_index()
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN in OHLCV
        initial_len = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with missing OHLCV data")
        
        # Calculate basic price info
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        
        # FIX: Clip body_pct to [0, 1] to prevent explosion
        body_pct_raw = df['body'] / df['range'].replace(0, np.nan)
        df['body_pct'] = body_pct_raw.clip(0, 1).fillna(0)
        
        # Direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        
        # Detect bar frequency for volatility scaling
        df = self._detect_bar_frequency(df)
        
        logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        logger.info(f"Detected bar frequency: {df['bar_frequency_minutes'].iloc[0]} minutes")
        
        return df
    
    def _detect_bar_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: Detect bar frequency from timestamps for proper volatility scaling.
        """
        time_deltas = df['datetime'].diff().dt.total_seconds() / 60  # in minutes
        # Use median to handle gaps (weekends, holidays)
        bar_freq_minutes = time_deltas.median()
        
        # Round to nearest standard timeframe
        standard_tfs = [1, 5, 15, 30, 60, 240, 1440]  # M1, M5, M15, M30, H1, H4, D1
        bar_freq_minutes = min(standard_tfs, key=lambda x: abs(x - bar_freq_minutes))
        
        df['bar_frequency_minutes'] = bar_freq_minutes
        
        # Calculate annualization factor based on frequency
        # Assuming 252 trading days, 24 hours (forex)
        bars_per_year = (252 * 24 * 60) / bar_freq_minutes
        df['annualization_factor'] = np.sqrt(bars_per_year)
        
        return df


# =============================================================================
# TIME/SESSION FEATURES
# =============================================================================

class TimeFeatureBuilder:
    """Build time and session-related features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all time-based features"""
        logger.info("Building time features...")
        
        df = df.copy()
        
        # Basic time components
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        # FIX: Use minute-of-day for precise session tracking
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        
        # Eastern Time hour (already in ET per user request)
        df['et_hour'] = df['hour']
        df['et_minute'] = df['minute']
        df['et_minute_of_day'] = df['minute_of_day']
        
        # Cyclical encoding for hour (captures circular nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['et_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['et_hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)  # 5 trading days
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        
        # Session identification with minute precision
        df = self._add_session_features(df)
        
        # Trading day features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
        
        # Prime trading hours
        df['is_prime_hour'] = df['et_hour'].isin([9, 10, 11, 14, 15]).astype(int)
        
        # Session transitions (often volatile) - using minute ranges
        df['london_open_window'] = self._in_time_range(
            df, 3, 0, 3, 30  # 3:00-3:30 AM ET
        ).astype(int)
        
        df['ny_open_window'] = self._in_time_range(
            df, 9, 30, 10, 0  # 9:30-10:00 AM ET
        ).astype(int)
        
        return df
    
    def _in_time_range(self, df: pd.DataFrame, 
                       start_hour: int, start_min: int,
                       end_hour: int, end_min: int) -> pd.Series:
        """
        FIX: Check if time is in range using minute_of_day for precision.
        Handles overnight ranges (end < start).
        """
        start_mod = start_hour * 60 + start_min
        end_mod = end_hour * 60 + end_min
        mod = df['et_minute_of_day']
        
        if end_mod > start_mod:
            # Normal range (e.g., 9:30 to 12:00)
            return (mod >= start_mod) & (mod < end_mod)
        else:
            # Overnight range (e.g., 19:00 to 00:00)
            return (mod >= start_mod) | (mod < end_mod)
    
    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add kill zone and session flags with minute precision"""
        
        cfg = self.config
        
        # Asian Session (7 PM - 12 AM ET, crosses midnight)
        df['in_asian'] = self._in_time_range(
            df, cfg.asian_start_hour, cfg.asian_start_min,
            cfg.asian_end_hour, cfg.asian_end_min
        ).astype(int)
        
        # London Kill Zone (2:00 AM - 5:00 AM ET)
        df['in_london_kz'] = self._in_time_range(
            df, cfg.london_kz_start_hour, cfg.london_kz_start_min,
            cfg.london_kz_end_hour, cfg.london_kz_end_min
        ).astype(int)
        
        # NY AM Kill Zone (9:30 AM - 12:00 PM ET)
        df['in_nyam_kz'] = self._in_time_range(
            df, cfg.nyam_kz_start_hour, cfg.nyam_kz_start_min,
            cfg.nyam_kz_end_hour, cfg.nyam_kz_end_min
        ).astype(int)
        
        # NY PM Kill Zone (2:00 PM - 4:00 PM ET)
        df['in_nypm_kz'] = self._in_time_range(
            df, cfg.nypm_kz_start_hour, cfg.nypm_kz_start_min,
            cfg.nypm_kz_end_hour, cfg.nypm_kz_end_min
        ).astype(int)
        
        # Combined kill zone flag
        df['in_any_kz'] = (df['in_london_kz'] | df['in_nyam_kz'] | df['in_nypm_kz']).astype(int)
        
        # FIX: Silver Bullet windows with minute precision
        # London Silver Bullet (3:00 AM - 4:00 AM ET)
        df['in_london_sb'] = self._in_time_range(
            df, cfg.london_sb_start_hour, cfg.london_sb_start_min,
            cfg.london_sb_end_hour, cfg.london_sb_end_min
        ).astype(int)
        
        # NY AM Silver Bullet (10:00 AM - 11:00 AM ET)
        df['in_nyam_sb'] = self._in_time_range(
            df, cfg.nyam_sb_start_hour, cfg.nyam_sb_start_min,
            cfg.nyam_sb_end_hour, cfg.nyam_sb_end_min
        ).astype(int)
        
        # NY PM Silver Bullet (2:00 PM - 3:00 PM ET)
        df['in_nypm_sb'] = self._in_time_range(
            df, cfg.nypm_sb_start_hour, cfg.nypm_sb_start_min,
            cfg.nypm_sb_end_hour, cfg.nypm_sb_end_min
        ).astype(int)
        
        # Combined Silver Bullet flag
        df['in_sb_window'] = (df['in_london_sb'] | df['in_nyam_sb'] | df['in_nypm_sb']).astype(int)
        
        # London/NY overlap (8:00 AM - 12:00 PM ET) - highest liquidity
        df['in_overlap'] = self._in_time_range(df, 8, 0, 12, 0).astype(int)
        
        return df


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicatorBuilder:
    """Build technical indicators"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        logger.info("Building technical indicators...")
        
        df = df.copy()
        
        # ATR (Average True Range)
        df = self._add_atr(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Moving averages for trend
        df = self._add_moving_averages(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum
        df = self._add_momentum_features(df)
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR"""
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        df['atr'] = tr.rolling(window=self.config.atr_period).mean()
        df['atr_long'] = tr.rolling(window=self.config.atr_period_long).mean()
        
        # ATR ratio (volatility expansion/contraction)
        df['atr_ratio'] = df['atr'] / df['atr_long']
        
        # Normalized ATR (ATR as % of price)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Range in ATR
        df['range_atr'] = df['range'] / df['atr']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features"""
        
        # Average volume
        df['volume_sma'] = df['volume'].rolling(window=self.config.volume_lookback).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volume spike
        df['volume_spike'] = (df['volume_ratio'] > self.config.volume_spike_threshold).astype(int)
        
        # Volume trend
        df['volume_sma_fast'] = df['volume'].rolling(window=5).mean()
        df['volume_trend'] = (df['volume_sma_fast'] > df['volume_sma']).astype(int)
        
        # Relative volume
        df['rvol'] = df['volume'] / df['volume'].rolling(window=50).mean()
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages for trend detection"""
        
        # EMAs
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Price relative to EMAs (use ATR with fillna to avoid div by zero)
        atr_safe = df['atr'].replace(0, np.nan)
        df['close_vs_ema8'] = (df['close'] - df['ema_8']) / atr_safe
        df['close_vs_ema21'] = (df['close'] - df['ema_21']) / atr_safe
        df['close_vs_ema50'] = (df['close'] - df['ema_50']) / atr_safe
        
        # EMA alignment (trend strength)
        df['ema_bullish_stack'] = ((df['ema_8'] > df['ema_21']) & 
                                    (df['ema_21'] > df['ema_50'])).astype(int)
        df['ema_bearish_stack'] = ((df['ema_8'] < df['ema_21']) & 
                                    (df['ema_21'] < df['ema_50'])).astype(int)
        
        # Trend direction
        df['trend_ema'] = np.where(df['ema_bullish_stack'], 1,
                                   np.where(df['ema_bearish_stack'], -1, 0))
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        
        # Rolling standard deviation of returns
        df['returns'] = df['close'].pct_change()
        
        # FIX: Use frequency-aware annualization factor
        annualization_factor = df['annualization_factor'].iloc[0] if 'annualization_factor' in df.columns else np.sqrt(252 * 24 * 60)
        df['volatility'] = df['returns'].rolling(window=20).std() * annualization_factor
        
        # Volatility regime
        df['vol_sma'] = df['volatility'].rolling(window=50).mean()
        df['high_volatility'] = (df['volatility'] > df['vol_sma'] * 1.5).astype(int)
        df['low_volatility'] = (df['volatility'] < df['vol_sma'] * 0.5).astype(int)
        
        # Bollinger Band width (volatility proxy)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_width'] = (4 * df['bb_std']) / df['bb_middle']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features"""
        
        # ROC (Rate of Change)
        df['roc_5'] = df['close'].pct_change(periods=5) * 100
        df['roc_10'] = df['close'].pct_change(periods=10) * 100
        df['roc_20'] = df['close'].pct_change(periods=20) * 100
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df


# =============================================================================
# MARKET STRUCTURE FEATURES - FIXED FOR LOOKAHEAD
# =============================================================================

class MarketStructureBuilder:
    """Build market structure features (swings, BOS, trends) - NO LOOKAHEAD"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        logger.info("Building market structure features...")
        
        df = df.copy()
        
        # FIX: Identify swing points with proper confirmation delay (no lookahead)
        df = self._identify_swings_no_lookahead(df)
        
        # Market structure (HH, HL, LH, LL)
        df = self._add_structure_features(df)
        
        # Break of Structure
        df = self._add_bos_features(df)
        
        # Premium/Discount zones
        df = self._add_premium_discount(df)
        
        return df
    
    def _identify_swings_no_lookahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: Identify swing highs and lows WITHOUT lookahead.
        
        A swing high/low is confirmed only after 'strength' bars have passed,
        meaning we look back 2*strength bars and confirm the middle point.
        The signal appears at the CONFIRMATION bar, not the swing bar itself.
        """
        
        strength = self.config.swing_strength
        
        # Initialize columns
        df['is_swing_high'] = 0
        df['is_swing_low'] = 0
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        df['swing_high_bar_idx'] = np.nan  # Which bar was the actual swing
        df['swing_low_bar_idx'] = np.nan
        
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        # We can only confirm a swing at bar i if we're looking at bar (i - strength)
        # meaning bar (i - strength) must have 'strength' bars on each side
        for i in range(2 * strength, n):
            # The candidate swing is at position (i - strength)
            candidate_idx = i - strength
            
            # Check swing high at candidate_idx
            is_sh = True
            for j in range(1, strength + 1):
                left_idx = candidate_idx - j
                right_idx = candidate_idx + j
                if left_idx < 0 or right_idx >= n:
                    is_sh = False
                    break
                if highs[candidate_idx] <= highs[left_idx] or highs[candidate_idx] <= highs[right_idx]:
                    is_sh = False
                    break
            
            if is_sh:
                # Signal appears at confirmation bar (i), not at swing bar
                df.iloc[i, df.columns.get_loc('is_swing_high')] = 1
                df.iloc[i, df.columns.get_loc('swing_high_price')] = highs[candidate_idx]
                df.iloc[i, df.columns.get_loc('swing_high_bar_idx')] = candidate_idx
            
            # Check swing low at candidate_idx
            is_sl = True
            for j in range(1, strength + 1):
                left_idx = candidate_idx - j
                right_idx = candidate_idx + j
                if left_idx < 0 or right_idx >= n:
                    is_sl = False
                    break
                if lows[candidate_idx] >= lows[left_idx] or lows[candidate_idx] >= lows[right_idx]:
                    is_sl = False
                    break
            
            if is_sl:
                df.iloc[i, df.columns.get_loc('is_swing_low')] = 1
                df.iloc[i, df.columns.get_loc('swing_low_price')] = lows[candidate_idx]
                df.iloc[i, df.columns.get_loc('swing_low_bar_idx')] = candidate_idx
        
        # Forward fill swing prices for reference (only use confirmed swings)
        df['last_swing_high'] = df['swing_high_price'].ffill()
        df['last_swing_low'] = df['swing_low_price'].ffill()
        
        # Distance to last swings (in ATR)
        atr_safe = df['atr'].replace(0, np.nan)
        df['dist_to_swing_high'] = (df['last_swing_high'] - df['close']) / atr_safe
        df['dist_to_swing_low'] = (df['close'] - df['last_swing_low']) / atr_safe
        
        return df
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add HH, HL, LH, LL structure features"""
        
        df['higher_high'] = 0
        df['lower_high'] = 0
        df['higher_low'] = 0
        df['lower_low'] = 0
        
        # Track structure based on confirmed swings
        prev_sh_price = None
        prev_sl_price = None
        
        for i in range(len(df)):
            # Check for new swing high confirmation
            if df.iloc[i]['is_swing_high'] == 1:
                curr_sh = df.iloc[i]['swing_high_price']
                if prev_sh_price is not None:
                    if curr_sh > prev_sh_price:
                        df.iloc[i, df.columns.get_loc('higher_high')] = 1
                    else:
                        df.iloc[i, df.columns.get_loc('lower_high')] = 1
                prev_sh_price = curr_sh
            
            # Check for new swing low confirmation
            if df.iloc[i]['is_swing_low'] == 1:
                curr_sl = df.iloc[i]['swing_low_price']
                if prev_sl_price is not None:
                    if curr_sl > prev_sl_price:
                        df.iloc[i, df.columns.get_loc('higher_low')] = 1
                    else:
                        df.iloc[i, df.columns.get_loc('lower_low')] = 1
                prev_sl_price = curr_sl
        
        # Rolling structure assessment
        lookback = self.config.swing_lookback
        
        df['hh_count'] = df['higher_high'].rolling(window=lookback, min_periods=1).sum()
        df['hl_count'] = df['higher_low'].rolling(window=lookback, min_periods=1).sum()
        df['lh_count'] = df['lower_high'].rolling(window=lookback, min_periods=1).sum()
        df['ll_count'] = df['lower_low'].rolling(window=lookback, min_periods=1).sum()
        
        # Trend determination based on structure
        bullish_structure = df['hh_count'] + df['hl_count']
        bearish_structure = df['lh_count'] + df['ll_count']
        
        df['structure_bias'] = np.where(
            bullish_structure > bearish_structure + 1, 1,
            np.where(bearish_structure > bullish_structure + 1, -1, 0)
        )
        
        return df
    
    def _add_bos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Break of Structure features - uses only past data"""
        
        df['bos_bullish'] = 0
        df['bos_bearish'] = 0
        
        # BOS occurs when price closes beyond last confirmed swing
        # Shift by 1 to ensure we're comparing to the swing that was known at prior bar
        last_sh_shifted = df['last_swing_high'].shift(1)
        last_sl_shifted = df['last_swing_low'].shift(1)
        
        df['bos_bullish'] = (
            (df['close'] > last_sh_shifted) & 
            (df['close'].shift(1) <= last_sh_shifted) &
            last_sh_shifted.notna()
        ).astype(int)
        
        df['bos_bearish'] = (
            (df['close'] < last_sl_shifted) & 
            (df['close'].shift(1) >= last_sl_shifted) &
            last_sl_shifted.notna()
        ).astype(int)
        
        # Bars since last BOS
        df['bars_since_bullish_bos'] = df.groupby(
            df['bos_bullish'].cumsum()
        ).cumcount()
        
        df['bars_since_bearish_bos'] = df.groupby(
            df['bos_bearish'].cumsum()
        ).cumcount()
        
        # Recent BOS (within last 20 bars)
        df['recent_bullish_bos'] = (df['bars_since_bullish_bos'] < 20).astype(int)
        df['recent_bearish_bos'] = (df['bars_since_bearish_bos'] < 20).astype(int)
        
        return df
    
    def _add_premium_discount(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add premium/discount zone features"""
        
        lookback = self.config.pd_lookback
        
        # Rolling high and low (past only)
        df['range_high'] = df['high'].rolling(window=lookback).max()
        df['range_low'] = df['low'].rolling(window=lookback).min()
        df['range_size'] = df['range_high'] - df['range_low']
        
        # Price position in range (0-100)
        df['price_position'] = np.where(
            df['range_size'] > 0,
            ((df['close'] - df['range_low']) / df['range_size']) * 100,
            50
        )
        
        # Zone classification
        df['in_discount'] = (df['price_position'] < self.config.discount_threshold).astype(int)
        df['in_premium'] = (df['price_position'] > self.config.premium_threshold).astype(int)
        df['in_equilibrium'] = (
            (df['price_position'] >= self.config.discount_threshold) & 
            (df['price_position'] <= self.config.premium_threshold)
        ).astype(int)
        
        # Equilibrium level
        df['equilibrium'] = (df['range_high'] + df['range_low']) / 2
        atr_safe = df['atr'].replace(0, np.nan)
        df['dist_to_eq'] = (df['close'] - df['equilibrium']) / atr_safe
        
        return df


# =============================================================================
# ICT-SPECIFIC FEATURES - FIXED FOR LOOKAHEAD
# =============================================================================

class ICTFeatureBuilder:
    """Build ICT-specific features (FVG, OB, displacement, sweeps) - NO LOOKAHEAD"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ICT-specific features"""
        logger.info("Building ICT-specific features...")
        
        df = df.copy()
        
        # Fair Value Gaps (past-only, no lookahead)
        df = self._add_fvg_features(df)
        
        # Displacement candles (past-only)
        df = self._add_displacement_features(df)
        
        # Order Blocks - FIXED: uses past-only confirmation
        df = self._add_order_block_features_fixed(df)
        
        # Liquidity sweeps - FIXED: O(N) using incremental tracking
        df = self._add_sweep_features_optimized(df)
        
        # Key levels (PDH/PDL)
        df = self._add_key_level_features(df)
        
        return df
    
    def _add_fvg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Fair Value Gaps - NO LOOKAHEAD.
        FVG is identified at bar i based on bars i-2, i-1, i (all past).
        """
        
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        df['fvg_size'] = 0.0
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan
        df['in_bullish_fvg'] = 0
        df['in_bearish_fvg'] = 0
        
        highs = df['high'].values
        lows = df['low'].values
        atr_vals = df['atr'].values
        is_bullish = df['is_bullish'].values
        is_bearish = df['is_bearish'].values
        
        # FVG detection requires looking at 3-candle patterns
        # All candles (i-2, i-1, i) are past/current - no lookahead
        for i in range(2, len(df)):
            atr = atr_vals[i]
            if atr <= 0 or np.isnan(atr):
                continue
            
            # Bullish FVG: candle[i] low > candle[i-2] high
            # Additional check: middle candle should be bullish (displacement direction)
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                if gap_size / atr >= self.config.min_fvg_atr:
                    # Check for bullish displacement (candle i-1 should be bullish)
                    if is_bullish[i-1] or is_bullish[i]:
                        df.iloc[i, df.columns.get_loc('bullish_fvg')] = 1
                        df.iloc[i, df.columns.get_loc('fvg_size')] = gap_size / atr
                        df.iloc[i, df.columns.get_loc('fvg_high')] = lows[i]
                        df.iloc[i, df.columns.get_loc('fvg_low')] = highs[i-2]
            
            # Bearish FVG: candle[i] high < candle[i-2] low
            if highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                if gap_size / atr >= self.config.min_fvg_atr:
                    # Check for bearish displacement
                    if is_bearish[i-1] or is_bearish[i]:
                        df.iloc[i, df.columns.get_loc('bearish_fvg')] = 1
                        df.iloc[i, df.columns.get_loc('fvg_size')] = gap_size / atr
                        df.iloc[i, df.columns.get_loc('fvg_high')] = lows[i-2]
                        df.iloc[i, df.columns.get_loc('fvg_low')] = highs[i]
        
        # Track if price is currently in an unfilled FVG
        df = self._track_fvg_zones(df)
        
        # Recent FVG flags
        df['recent_bullish_fvg'] = df['bullish_fvg'].rolling(
            window=self.config.max_fvg_lookback, min_periods=1
        ).sum().clip(0, 1)
        
        df['recent_bearish_fvg'] = df['bearish_fvg'].rolling(
            window=self.config.max_fvg_lookback, min_periods=1
        ).sum().clip(0, 1)
        
        return df
    
    def _track_fvg_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track if price is in an FVG zone - NO LOOKAHEAD"""
        
        # Track active FVGs as list of (type, high, low, bar_idx)
        fvg_zones = []
        max_lookback = self.config.max_fvg_lookback
        
        highs = df['high'].values
        lows = df['low'].values
        fvg_high_vals = df['fvg_high'].values
        fvg_low_vals = df['fvg_low'].values
        bullish_fvg = df['bullish_fvg'].values
        bearish_fvg = df['bearish_fvg'].values
        
        in_bull_fvg = np.zeros(len(df))
        in_bear_fvg = np.zeros(len(df))
        
        for i in range(len(df)):
            # Add new FVGs
            if bullish_fvg[i]:
                fvg_zones.append(('bull', fvg_high_vals[i], fvg_low_vals[i], i))
            
            if bearish_fvg[i]:
                fvg_zones.append(('bear', fvg_high_vals[i], fvg_low_vals[i], i))
            
            # Check if price is in any active FVG and filter filled/old ones
            active_zones = []
            for fvg_type, fvg_h, fvg_l, fvg_bar in fvg_zones:
                # Check if FVG is still valid (not too old)
                if i - fvg_bar > max_lookback:
                    continue
                
                # Check if price filled the FVG (using current bar's action)
                if fvg_type == 'bull' and lows[i] <= fvg_l:
                    continue  # Filled
                if fvg_type == 'bear' and highs[i] >= fvg_h:
                    continue  # Filled
                
                active_zones.append((fvg_type, fvg_h, fvg_l, fvg_bar))
                
                # Check if current price is in the FVG
                if lows[i] <= fvg_h and highs[i] >= fvg_l:
                    if fvg_type == 'bull':
                        in_bull_fvg[i] = 1
                    else:
                        in_bear_fvg[i] = 1
            
            fvg_zones = active_zones
        
        df['in_bullish_fvg'] = in_bull_fvg.astype(int)
        df['in_bearish_fvg'] = in_bear_fvg.astype(int)
        
        return df
    
    def _add_displacement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify displacement candles - past-only, no lookahead"""
        
        df['is_displacement'] = 0
        df['displacement_size'] = 0.0
        
        # FIX: Use clipped body_pct (already fixed in loader)
        body_pct = df['body_pct'].fillna(0)
        range_atr = df['range_atr'].fillna(0)
        
        # Displacement: large body candle with strong directional move
        is_disp = (
            (body_pct >= self.config.displacement_body_ratio) &
            (range_atr >= self.config.displacement_atr_min)
        )
        
        df.loc[is_disp, 'is_displacement'] = 1
        df.loc[is_disp, 'displacement_size'] = range_atr[is_disp]
        
        # Bullish vs Bearish displacement
        df['bullish_displacement'] = (df['is_displacement'] & df['is_bullish']).astype(int)
        df['bearish_displacement'] = (df['is_displacement'] & df['is_bearish']).astype(int)
        
        # Recent displacement
        df['recent_bullish_disp'] = df['bullish_displacement'].rolling(
            window=10, min_periods=1
        ).sum().clip(0, 1)
        
        df['recent_bearish_disp'] = df['bearish_displacement'].rolling(
            window=10, min_periods=1
        ).sum().clip(0, 1)
        
        return df
    
    def _add_order_block_features_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: Identify Order Blocks using PAST-ONLY confirmation.
        
        Instead of looking at future bars to confirm OB, we:
        1. Look BACKWARD to find the last opposite-color candle before a displacement
        2. The OB is marked at the bar where we detect the displacement (confirmation bar)
        
        This eliminates lookahead while maintaining the ICT concept.
        """
        
        df['bullish_ob'] = 0
        df['bearish_ob'] = 0
        df['ob_high'] = np.nan
        df['ob_low'] = np.nan
        
        is_bullish = df['is_bullish'].values
        is_bearish = df['is_bearish'].values
        bullish_disp = df['bullish_displacement'].values
        bearish_disp = df['bearish_displacement'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        closes = df['close'].values
        atr_vals = df['atr'].values
        
        lookback = self.config.ob_confirmation_bars
        min_disp_atr = self.config.ob_min_displacement_atr
        
        for i in range(lookback + 1, len(df)):
            atr = atr_vals[i]
            if atr <= 0 or np.isnan(atr):
                continue
            
            # Check for Bullish Order Block:
            # Current bar is a bullish displacement -> look back for last bearish candle
            if bullish_disp[i]:
                # Find last bearish candle in lookback window
                for j in range(i - 1, max(i - lookback - 1, 0), -1):
                    if is_bearish[j]:
                        # This is a potential bullish OB
                        # Verify the move from OB to current bar is significant
                        move = closes[i] - lows[j]
                        if move / atr >= min_disp_atr:
                            df.iloc[i, df.columns.get_loc('bullish_ob')] = 1
                            df.iloc[i, df.columns.get_loc('ob_high')] = highs[j]
                            df.iloc[i, df.columns.get_loc('ob_low')] = lows[j]
                        break
            
            # Check for Bearish Order Block:
            # Current bar is a bearish displacement -> look back for last bullish candle
            if bearish_disp[i]:
                for j in range(i - 1, max(i - lookback - 1, 0), -1):
                    if is_bullish[j]:
                        # This is a potential bearish OB
                        move = highs[j] - closes[i]
                        if move / atr >= min_disp_atr:
                            df.iloc[i, df.columns.get_loc('bearish_ob')] = 1
                            df.iloc[i, df.columns.get_loc('ob_high')] = highs[j]
                            df.iloc[i, df.columns.get_loc('ob_low')] = lows[j]
                        break
        
        # Recent OB
        df['recent_bullish_ob'] = df['bullish_ob'].rolling(
            window=20, min_periods=1
        ).sum().clip(0, 1)
        
        df['recent_bearish_ob'] = df['bearish_ob'].rolling(
            window=20, min_periods=1
        ).sum().clip(0, 1)
        
        return df
    
    def _add_sweep_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: Identify liquidity sweeps - O(N) using incremental tracking.
        
        Instead of slicing dataframe on every bar (O(N²)), we:
        1. Track last confirmed swing high/low incrementally
        2. Check sweep condition: wick beyond swing, close back inside
        """
        
        df['sweep_high'] = 0
        df['sweep_low'] = 0
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Use the already-computed last_swing_high/low (confirmed, no lookahead)
        last_sh = df['last_swing_high'].values
        last_sl = df['last_swing_low'].values
        
        n = len(df)
        
        for i in range(1, n):
            # Use swing from previous bar to avoid any confusion
            prev_sh = last_sh[i - 1] if i > 0 else np.nan
            prev_sl = last_sl[i - 1] if i > 0 else np.nan
            
            # Sweep high: wick above swing high, close below it (rejection)
            if not np.isnan(prev_sh):
                if highs[i] > prev_sh and closes[i] < prev_sh:
                    df.iloc[i, df.columns.get_loc('sweep_high')] = 1
            
            # Sweep low: wick below swing low, close above it (rejection)
            if not np.isnan(prev_sl):
                if lows[i] < prev_sl and closes[i] > prev_sl:
                    df.iloc[i, df.columns.get_loc('sweep_low')] = 1
        
        # Recent sweeps
        df['recent_sweep_high'] = df['sweep_high'].rolling(
            window=10, min_periods=1
        ).sum().clip(0, 1)
        
        df['recent_sweep_low'] = df['sweep_low'].rolling(
            window=10, min_periods=1
        ).sum().clip(0, 1)
        
        return df
    
    def _add_key_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Previous Day High/Low features"""
        
        if not self.config.pdhl_enabled:
            return df
        
        # Group by date
        df['date'] = df['datetime'].dt.date
        
        # Calculate daily high/low
        daily_hl = df.groupby('date').agg({
            'high': 'max',
            'low': 'min',
            'open': 'first',
            'close': 'last'
        }).reset_index()
        
        daily_hl.columns = ['date', 'day_high', 'day_low', 'day_open', 'day_close']
        
        # Shift to get previous day (no lookahead - only past days)
        daily_hl['pdh'] = daily_hl['day_high'].shift(1)
        daily_hl['pdl'] = daily_hl['day_low'].shift(1)
        
        # Merge back
        df = df.merge(daily_hl[['date', 'pdh', 'pdl']], on='date', how='left')
        
        # Distance to PDH/PDL
        atr_safe = df['atr'].replace(0, np.nan)
        df['dist_to_pdh'] = (df['pdh'] - df['close']) / atr_safe
        df['dist_to_pdl'] = (df['close'] - df['pdl']) / atr_safe
        
        # Near PDH/PDL
        df['near_pdh'] = (abs(df['dist_to_pdh']) < 1.5).astype(int)
        df['near_pdl'] = (abs(df['dist_to_pdl']) < 1.5).astype(int)
        
        # Above/Below PDH/PDL
        df['above_pdh'] = (df['close'] > df['pdh']).astype(int)
        df['below_pdl'] = (df['close'] < df['pdl']).astype(int)
        
        return df


# =============================================================================
# CANDLE PATTERN FEATURES
# =============================================================================

class CandlePatternBuilder:
    """Build candle pattern features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candle pattern features"""
        logger.info("Building candle pattern features...")
        
        df = df.copy()
        
        # Wick ratios
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        range_safe = df['range'].replace(0, np.nan)
        df['upper_wick_ratio'] = (df['upper_wick'] / range_safe).clip(0, 1).fillna(0)
        df['lower_wick_ratio'] = (df['lower_wick'] / range_safe).clip(0, 1).fillna(0)
        
        # Candle types
        df['is_doji'] = (df['body_pct'] < 0.1).astype(int)
        df['is_hammer'] = (
            (df['lower_wick_ratio'] > 0.6) & 
            (df['upper_wick_ratio'] < 0.1) &
            (df['is_bullish'] == 1)
        ).astype(int)
        df['is_shooting_star'] = (
            (df['upper_wick_ratio'] > 0.6) & 
            (df['lower_wick_ratio'] < 0.1) &
            (df['is_bearish'] == 1)
        ).astype(int)
        
        df['is_marubozu'] = (df['body_pct'] > 0.9).astype(int)
        
        # Engulfing patterns
        df['bullish_engulf'] = (
            (df['is_bullish'] == 1) &
            (df['is_bearish'].shift(1) == 1) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        df['bearish_engulf'] = (
            (df['is_bearish'] == 1) &
            (df['is_bullish'].shift(1) == 1) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        # Rejection candles
        df['bullish_rejection'] = (
            (df['lower_wick_ratio'] > 0.5) &
            (df['is_bullish'] == 1)
        ).astype(int)
        
        df['bearish_rejection'] = (
            (df['upper_wick_ratio'] > 0.5) &
            (df['is_bearish'] == 1)
        ).astype(int)
        
        return df


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

class FeatureEngineeringPipeline:
    """Main pipeline that orchestrates all feature builders"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Initialize builders
        self.loader = OHLCVLoader(self.config)
        self.time_builder = TimeFeatureBuilder(self.config)
        self.tech_builder = TechnicalIndicatorBuilder(self.config)
        self.structure_builder = MarketStructureBuilder(self.config)
        self.ict_builder = ICTFeatureBuilder(self.config)
        self.candle_builder = CandlePatternBuilder(self.config)
    
    def run(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            input_path: Path to input OHLCV CSV
            output_path: Path to save feature-enriched CSV
        
        Returns:
            DataFrame with all features
        """
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING PIPELINE (FIXED - NO LOOKAHEAD)")
        logger.info("="*60)
        
        # Load data
        df = self.loader.load(input_path)
        initial_cols = len(df.columns)
        
        # Build features in order (some depend on others)
        df = self.time_builder.build(df)
        df = self.tech_builder.build(df)
        df = self.structure_builder.build(df)
        df = self.ict_builder.build(df)
        df = self.candle_builder.build(df)
        
        # Add final composite features
        df = self._add_composite_features(df)
        
        # FIX: Clean up with warmup trimming instead of blanket bfill
        df = self._cleanup_fixed(df)
        
        final_cols = len(df.columns)
        logger.info(f"Added {final_cols - initial_cols} features")
        logger.info(f"Total columns: {final_cols}")
        
        # Save
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        
        return df
    
    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite/interaction features"""
        
        # Alignment scores
        df['trend_zone_aligned'] = (
            ((df['trend_ema'] > 0) & (df['in_discount'] == 1)) |
            ((df['trend_ema'] < 0) & (df['in_premium'] == 1))
        ).astype(int)
        
        # FVG alignment with trend
        df['fvg_trend_aligned'] = (
            ((df['recent_bullish_fvg'] == 1) & (df['trend_ema'] > 0)) |
            ((df['recent_bearish_fvg'] == 1) & (df['trend_ema'] < 0))
        ).astype(int)
        
        # Multi-factor alignment
        df['bullish_confluence'] = (
            df['in_discount'].astype(int) +
            (df['trend_ema'] > 0).astype(int) +
            df['recent_bullish_fvg'].astype(int) +
            df['recent_sweep_low'].astype(int) +
            df['in_any_kz'].astype(int)
        )
        
        df['bearish_confluence'] = (
            df['in_premium'].astype(int) +
            (df['trend_ema'] < 0).astype(int) +
            df['recent_bearish_fvg'].astype(int) +
            df['recent_sweep_high'].astype(int) +
            df['in_any_kz'].astype(int)
        )
        
        # Max confluence
        df['max_confluence'] = df[['bullish_confluence', 'bearish_confluence']].max(axis=1)
        
        # Net direction
        df['net_confluence'] = df['bullish_confluence'] - df['bearish_confluence']
        
        return df
    
    def _cleanup_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: Clean up dataframe with warmup trimming.
        
        Instead of blanket bfill (which can leak info), we:
        1. Drop the warmup period at the start
        2. Only use forward fill for NaN values
        3. Fill remaining NaN with 0 for numeric columns
        """
        
        warmup = self.config.warmup_period
        
        # Drop warmup rows
        if len(df) > warmup:
            df = df.iloc[warmup:].reset_index(drop=True)
            logger.info(f"Dropped {warmup} warmup bars")
        
        # Create a mask of rows where critical indicators are valid
        critical_cols = ['atr', 'close', 'volume']
        valid_mask = df[critical_cols].notna().all(axis=1)
        
        # Only keep valid rows
        df = df[valid_mask].reset_index(drop=True)
        
        # For remaining NaN values, use forward fill ONLY (no bfill)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                # Forward fill only
                df[col] = df[col].ffill()
                # Fill remaining NaN with 0 (will only affect start of series)
                df[col] = df[col].fillna(0)
        
        logger.info(f"Final dataset: {len(df)} bars after cleanup")
        
        return df


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Feature Engineering Pipeline (Fixed)')
    parser.add_argument('input', help='Input OHLCV CSV file')
    parser.add_argument('output', help='Output features CSV file')
    parser.add_argument('--swing-strength', type=int, default=5,
                        help='Swing detection strength (default: 5)')
    parser.add_argument('--warmup', type=int, default=200,
                        help='Warmup period to drop (default: 200)')
    
    args = parser.parse_args()
    
    # Create config
    config = FeatureConfig(
        swing_strength=args.swing_strength,
        warmup_period=args.warmup
    )
    
    # Run pipeline
    pipeline = FeatureEngineeringPipeline(config)
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
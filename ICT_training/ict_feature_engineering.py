"""
ICT Feature Engineering Pipeline
================================
Transforms raw OHLCV data into ICT-specific features for ML training.

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
Version: 1.0
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
    
    # Time Settings (Eastern Time)
    broker_gmt_offset: int = 0       # Adjust based on your broker
    
    # Session Times (ET hours)
    asian_start: int = 19            # 7 PM ET
    asian_end: int = 0               # Midnight ET
    london_kz_start: int = 2         # 2 AM ET
    london_kz_end: int = 5           # 5 AM ET
    nyam_kz_start: int = 9           # 9 AM ET
    nyam_kz_end: int = 12            # 12 PM ET
    nypm_kz_start: int = 14          # 2 PM ET
    nypm_kz_end: int = 16            # 4 PM ET
    
    # Premium/Discount
    pd_lookback: int = 50
    discount_threshold: float = 40.0  # Below 40% = discount
    premium_threshold: float = 60.0   # Above 60% = premium
    
    # Key Levels
    pdhl_enabled: bool = True        # Previous Day High/Low
    pwhl_enabled: bool = True        # Previous Week High/Low


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
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y%m%d %H:%M:%S'
        )
        
        # Set datetime as index but keep as column too
        df = df.set_index('datetime', drop=False)
        df = df.sort_index()
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
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
        df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)
        
        # Direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        
        logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        
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
        
        # Convert to Eastern Time (approximate - adjust broker_gmt_offset as needed)
        # ET is UTC-5 (EST) or UTC-4 (EDT)
        df['et_hour'] = (df['hour'] - self.config.broker_gmt_offset - 5) % 24
        
        # Cyclical encoding for hour (captures circular nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['et_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['et_hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)  # 5 trading days
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        
        # Session identification
        df = self._add_session_features(df)
        
        # Trading day features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
        
        # Prime trading hours
        df['is_prime_hour'] = df['et_hour'].isin([9, 10, 11, 14, 15]).astype(int)
        
        # Session transitions (often volatile)
        df['london_open_hour'] = (df['et_hour'] == 3).astype(int)  # ~3 AM ET
        df['ny_open_hour'] = (df['et_hour'] == 9).astype(int)      # 9:30 AM ET approx
        
        return df
    
    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add kill zone and session flags"""
        
        et_hour = df['et_hour']
        
        # Asian Session (7 PM - 12 AM ET, crosses midnight)
        df['in_asian'] = ((et_hour >= self.config.asian_start) | 
                          (et_hour < self.config.asian_end)).astype(int)
        
        # London Kill Zone (2 AM - 5 AM ET)
        df['in_london_kz'] = ((et_hour >= self.config.london_kz_start) & 
                              (et_hour < self.config.london_kz_end)).astype(int)
        
        # NY AM Kill Zone (9 AM - 12 PM ET)
        df['in_nyam_kz'] = ((et_hour >= self.config.nyam_kz_start) & 
                            (et_hour < self.config.nyam_kz_end)).astype(int)
        
        # NY PM Kill Zone (2 PM - 4 PM ET)
        df['in_nypm_kz'] = ((et_hour >= self.config.nypm_kz_start) & 
                            (et_hour < self.config.nypm_kz_end)).astype(int)
        
        # Combined kill zone flag
        df['in_any_kz'] = (df['in_london_kz'] | df['in_nyam_kz'] | df['in_nypm_kz']).astype(int)
        
        # Silver Bullet windows (10-11 AM, 2-3 PM ET)
        df['in_sb_window'] = (((et_hour >= 10) & (et_hour < 11)) |
                              ((et_hour >= 14) & (et_hour < 15))).astype(int)
        
        # London/NY overlap (8 AM - 12 PM ET) - highest liquidity
        df['in_overlap'] = ((et_hour >= 8) & (et_hour < 12)).astype(int)
        
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
        
        # Price relative to EMAs
        df['close_vs_ema8'] = (df['close'] - df['ema_8']) / df['atr']
        df['close_vs_ema21'] = (df['close'] - df['ema_21']) / df['atr']
        df['close_vs_ema50'] = (df['close'] - df['ema_50']) / df['atr']
        
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
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252 * 24 * 60)  # Annualized for minute data
        
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
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df


# =============================================================================
# MARKET STRUCTURE FEATURES
# =============================================================================

class MarketStructureBuilder:
    """Build market structure features (swings, BOS, trends)"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        logger.info("Building market structure features...")
        
        df = df.copy()
        
        # Identify swing points
        df = self._identify_swings(df)
        
        # Market structure (HH, HL, LH, LL)
        df = self._add_structure_features(df)
        
        # Break of Structure
        df = self._add_bos_features(df)
        
        # Premium/Discount zones
        df = self._add_premium_discount(df)
        
        return df
    
    def _identify_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify swing highs and lows"""
        
        strength = self.config.swing_strength
        
        df['is_swing_high'] = 0
        df['is_swing_low'] = 0
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(strength, len(df) - strength):
            # Check swing high
            is_sh = True
            for j in range(1, strength + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_sh = False
                    break
            
            if is_sh:
                df.iloc[i, df.columns.get_loc('is_swing_high')] = 1
                df.iloc[i, df.columns.get_loc('swing_high_price')] = highs[i]
            
            # Check swing low
            is_sl = True
            for j in range(1, strength + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_sl = False
                    break
            
            if is_sl:
                df.iloc[i, df.columns.get_loc('is_swing_low')] = 1
                df.iloc[i, df.columns.get_loc('swing_low_price')] = lows[i]
        
        # Forward fill swing prices for reference
        df['last_swing_high'] = df['swing_high_price'].ffill()
        df['last_swing_low'] = df['swing_low_price'].ffill()
        
        # Distance to last swings (in ATR)
        df['dist_to_swing_high'] = (df['last_swing_high'] - df['close']) / df['atr']
        df['dist_to_swing_low'] = (df['close'] - df['last_swing_low']) / df['atr']
        
        return df
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add HH, HL, LH, LL structure features"""
        
        # Get swing high/low sequences
        swing_highs = df[df['is_swing_high'] == 1]['high'].values
        swing_lows = df[df['is_swing_low'] == 1]['low'].values
        
        df['higher_high'] = 0
        df['lower_high'] = 0
        df['higher_low'] = 0
        df['lower_low'] = 0
        
        # Track structure
        prev_sh = None
        prev_sl = None
        
        sh_indices = df[df['is_swing_high'] == 1].index.tolist()
        sl_indices = df[df['is_swing_low'] == 1].index.tolist()
        
        for idx in sh_indices:
            curr_sh = df.loc[idx, 'high']
            if prev_sh is not None:
                if curr_sh > prev_sh:
                    df.loc[idx, 'higher_high'] = 1
                else:
                    df.loc[idx, 'lower_high'] = 1
            prev_sh = curr_sh
        
        for idx in sl_indices:
            curr_sl = df.loc[idx, 'low']
            if prev_sl is not None:
                if curr_sl > prev_sl:
                    df.loc[idx, 'higher_low'] = 1
                else:
                    df.loc[idx, 'lower_low'] = 1
            prev_sl = curr_sl
        
        # Rolling structure assessment
        lookback = self.config.swing_lookback
        
        df['hh_count'] = df['higher_high'].rolling(window=lookback, min_periods=1).sum()
        df['hl_count'] = df['higher_low'].rolling(window=lookback, min_periods=1).sum()
        df['lh_count'] = df['lower_high'].rolling(window=lookback, min_periods=1).sum()
        df['ll_count'] = df['lower_low'].rolling(window=lookback, min_periods=1).sum()
        
        # Trend determination based on structure
        bullish_structure = (df['hh_count'] + df['hl_count']) 
        bearish_structure = (df['lh_count'] + df['ll_count'])
        
        df['structure_bias'] = np.where(
            bullish_structure > bearish_structure + 1, 1,
            np.where(bearish_structure > bullish_structure + 1, -1, 0)
        )
        
        return df
    
    def _add_bos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Break of Structure features"""
        
        df['bos_bullish'] = 0
        df['bos_bearish'] = 0
        
        # BOS occurs when price closes beyond last swing
        df['bos_bullish'] = ((df['close'] > df['last_swing_high'].shift(1)) & 
                             (df['close'].shift(1) <= df['last_swing_high'].shift(1))).astype(int)
        
        df['bos_bearish'] = ((df['close'] < df['last_swing_low'].shift(1)) & 
                             (df['close'].shift(1) >= df['last_swing_low'].shift(1))).astype(int)
        
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
        
        # Rolling high and low
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
        df['in_equilibrium'] = ((df['price_position'] >= self.config.discount_threshold) & 
                                 (df['price_position'] <= self.config.premium_threshold)).astype(int)
        
        # Equilibrium level
        df['equilibrium'] = (df['range_high'] + df['range_low']) / 2
        df['dist_to_eq'] = (df['close'] - df['equilibrium']) / df['atr']
        
        return df


# =============================================================================
# ICT-SPECIFIC FEATURES
# =============================================================================

class ICTFeatureBuilder:
    """Build ICT-specific features (FVG, OB, displacement, sweeps)"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ICT-specific features"""
        logger.info("Building ICT-specific features...")
        
        df = df.copy()
        
        # Fair Value Gaps
        df = self._add_fvg_features(df)
        
        # Displacement candles
        df = self._add_displacement_features(df)
        
        # Order Blocks
        df = self._add_order_block_features(df)
        
        # Liquidity sweeps
        df = self._add_sweep_features(df)
        
        # Key levels (PDH/PDL)
        df = self._add_key_level_features(df)
        
        return df
    
    def _add_fvg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify Fair Value Gaps"""
        
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        df['fvg_size'] = 0.0
        df['in_bullish_fvg'] = 0
        df['in_bearish_fvg'] = 0
        
        # FVG detection requires looking at 3-candle patterns
        for i in range(2, len(df)):
            # Bullish FVG: candle[i] low > candle[i-2] high
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                gap_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
                atr = df.iloc[i]['atr']
                
                if atr > 0 and gap_size / atr >= self.config.min_fvg_atr:
                    df.iloc[i, df.columns.get_loc('bullish_fvg')] = 1
                    df.iloc[i, df.columns.get_loc('fvg_size')] = gap_size / atr
            
            # Bearish FVG: candle[i] high < candle[i-2] low
            if df.iloc[i]['high'] < df.iloc[i-2]['low']:
                gap_size = df.iloc[i-2]['low'] - df.iloc[i]['high']
                atr = df.iloc[i]['atr']
                
                if atr > 0 and gap_size / atr >= self.config.min_fvg_atr:
                    df.iloc[i, df.columns.get_loc('bearish_fvg')] = 1
                    df.iloc[i, df.columns.get_loc('fvg_size')] = gap_size / atr
        
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
        """Track if price is in an FVG zone"""
        
        fvg_zones = []  # List of (type, high, low, bar_idx)
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Add new FVGs
            if row['bullish_fvg']:
                fvg_high = df.iloc[i]['low']
                fvg_low = df.iloc[i-2]['high']
                fvg_zones.append(('bull', fvg_high, fvg_low, i))
            
            if row['bearish_fvg']:
                fvg_high = df.iloc[i-2]['low']
                fvg_low = df.iloc[i]['high']
                fvg_zones.append(('bear', fvg_high, fvg_low, i))
            
            # Check if price is in any active FVG
            in_bull_fvg = 0
            in_bear_fvg = 0
            
            # Remove filled FVGs and check current position
            active_zones = []
            for fvg_type, fvg_h, fvg_l, fvg_bar in fvg_zones:
                # Check if FVG is still valid (not too old, not filled)
                if i - fvg_bar > self.config.max_fvg_lookback:
                    continue
                
                # Check if price filled the FVG
                if fvg_type == 'bull' and row['low'] <= fvg_l:
                    continue  # Filled
                if fvg_type == 'bear' and row['high'] >= fvg_h:
                    continue  # Filled
                
                active_zones.append((fvg_type, fvg_h, fvg_l, fvg_bar))
                
                # Check if current price is in the FVG
                if row['low'] <= fvg_h and row['high'] >= fvg_l:
                    if fvg_type == 'bull':
                        in_bull_fvg = 1
                    else:
                        in_bear_fvg = 1
            
            fvg_zones = active_zones
            df.iloc[i, df.columns.get_loc('in_bullish_fvg')] = in_bull_fvg
            df.iloc[i, df.columns.get_loc('in_bearish_fvg')] = in_bear_fvg
        
        return df
    
    def _add_displacement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify displacement candles"""
        
        df['is_displacement'] = 0
        df['displacement_size'] = 0.0
        
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
    
    def _add_order_block_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify Order Blocks"""
        
        df['bullish_ob'] = 0
        df['bearish_ob'] = 0
        
        for i in range(1, len(df) - 1):
            # Bullish OB: bearish candle followed by strong bullish move
            if df.iloc[i]['is_bearish']:
                # Check if next candles show strong bullish move
                if i + 3 < len(df):
                    future_high = df.iloc[i+1:i+4]['high'].max()
                    ob_high = df.iloc[i]['high']
                    atr = df.iloc[i]['atr']
                    
                    if atr > 0 and (future_high - ob_high) / atr > 1.5:
                        df.iloc[i, df.columns.get_loc('bullish_ob')] = 1
            
            # Bearish OB: bullish candle followed by strong bearish move
            if df.iloc[i]['is_bullish']:
                if i + 3 < len(df):
                    future_low = df.iloc[i+1:i+4]['low'].min()
                    ob_low = df.iloc[i]['low']
                    atr = df.iloc[i]['atr']
                    
                    if atr > 0 and (ob_low - future_low) / atr > 1.5:
                        df.iloc[i, df.columns.get_loc('bearish_ob')] = 1
        
        # Recent OB
        df['recent_bullish_ob'] = df['bullish_ob'].rolling(
            window=20, min_periods=1
        ).sum().clip(0, 1)
        
        df['recent_bearish_ob'] = df['bearish_ob'].rolling(
            window=20, min_periods=1
        ).sum().clip(0, 1)
        
        return df
    
    def _add_sweep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify liquidity sweeps"""
        
        df['sweep_high'] = 0
        df['sweep_low'] = 0
        
        for i in range(self.config.swing_strength + 1, len(df)):
            # Get last swing high and low
            sh = df.iloc[:i][df.iloc[:i]['is_swing_high'] == 1]['high']
            sl = df.iloc[:i][df.iloc[:i]['is_swing_low'] == 1]['low']
            
            if len(sh) > 0:
                last_sh = sh.iloc[-1]
                # Sweep high: wick above swing high, close below
                if df.iloc[i]['high'] > last_sh and df.iloc[i]['close'] < last_sh:
                    df.iloc[i, df.columns.get_loc('sweep_high')] = 1
            
            if len(sl) > 0:
                last_sl = sl.iloc[-1]
                # Sweep low: wick below swing low, close above
                if df.iloc[i]['low'] < last_sl and df.iloc[i]['close'] > last_sl:
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
        
        # Shift to get previous day
        daily_hl['pdh'] = daily_hl['day_high'].shift(1)
        daily_hl['pdl'] = daily_hl['day_low'].shift(1)
        
        # Merge back
        df = df.merge(daily_hl[['date', 'pdh', 'pdl']], on='date', how='left')
        
        # Distance to PDH/PDL
        df['dist_to_pdh'] = (df['pdh'] - df['close']) / df['atr']
        df['dist_to_pdl'] = (df['close'] - df['pdl']) / df['atr']
        
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
        
        df['upper_wick_ratio'] = df['upper_wick'] / df['range'].replace(0, np.nan)
        df['lower_wick_ratio'] = df['lower_wick'] / df['range'].replace(0, np.nan)
        
        # Candle types
        df['is_doji'] = (df['body_pct'] < 0.1).astype(int)
        df['is_hammer'] = ((df['lower_wick_ratio'] > 0.6) & 
                           (df['upper_wick_ratio'] < 0.1) &
                           (df['is_bullish'] == 1)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick_ratio'] > 0.6) & 
                                   (df['lower_wick_ratio'] < 0.1) &
                                   (df['is_bearish'] == 1)).astype(int)
        
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
        logger.info("FEATURE ENGINEERING PIPELINE")
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
        
        # Clean up
        df = self._cleanup(df)
        
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
    
    def _cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up dataframe"""
        
        # Remove rows with NaN in critical columns
        critical_cols = ['atr', 'close', 'volume']
        df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
        
        # Fill remaining NaN with appropriate values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                # Use forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN, fill with 0
                df[col] = df[col].fillna(0)
        
        return df


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Feature Engineering Pipeline')
    parser.add_argument('input', help='Input OHLCV CSV file')
    parser.add_argument('output', help='Output features CSV file')
    parser.add_argument('--broker-gmt', type=int, default=0, 
                        help='Broker GMT offset (default: 0)')
    parser.add_argument('--swing-strength', type=int, default=5,
                        help='Swing detection strength (default: 5)')
    
    args = parser.parse_args()
    
    # Create config
    config = FeatureConfig(
        broker_gmt_offset=args.broker_gmt,
        swing_strength=args.swing_strength
    )
    
    # Run pipeline
    pipeline = FeatureEngineeringPipeline(config)
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()

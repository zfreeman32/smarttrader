"""
Data Utilities
Functions for loading, generating, and processing market data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_sample_eurusd_data(
    n_bars: int = 1000,
    start_price: float = 1.1000,
    volatility: float = 0.0003,
    trend: float = 0.0,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate sample EURUSD OHLCV data for testing
    
    Args:
        n_bars: Number of bars to generate
        start_price: Starting price
        volatility: Price volatility
        trend: Trend strength (positive=up, negative=down)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV data and timestamp index
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps (1-minute bars)
    start_time = datetime(2024, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]
    
    # Generate price movements
    returns = np.random.normal(trend, volatility, n_bars)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    data = []
    for i, price in enumerate(prices):
        # Add intrabar volatility
        high_offset = abs(np.random.normal(0, volatility * 0.5))
        low_offset = abs(np.random.normal(0, volatility * 0.5))
        
        open_price = price + np.random.normal(0, volatility * 0.3)
        high_price = price + high_offset
        low_price = price - low_offset
        close_price = price
        
        # Ensure OHLC relationships are valid
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (with some patterns)
        base_volume = 1000
        volume = base_volume * (1 + abs(np.random.normal(0, 0.5)))
        
        # Add volume spikes occasionally
        if np.random.random() < 0.1:
            volume *= np.random.uniform(2, 4)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'timestamp'
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    df = df.copy()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df


def generate_trending_data(n_bars: int = 500, direction: str = 'up') -> pd.DataFrame:
    """
    Generate data with a clear trend
    
    Args:
        n_bars: Number of bars
        direction: 'up' or 'down'
        
    Returns:
        DataFrame with trending data
    """
    trend = 0.00005 if direction == 'up' else -0.00005
    return generate_sample_eurusd_data(
        n_bars=n_bars,
        volatility=0.0002,
        trend=trend
    )


def generate_ranging_data(n_bars: int = 500) -> pd.DataFrame:
    """
    Generate ranging/sideways market data
    
    Args:
        n_bars: Number of bars
        
    Returns:
        DataFrame with ranging data
    """
    return generate_sample_eurusd_data(
        n_bars=n_bars,
        volatility=0.0001,
        trend=0.0
    )


def generate_volatile_data(n_bars: int = 500) -> pd.DataFrame:
    """
    Generate highly volatile market data
    
    Args:
        n_bars: Number of bars
        
    Returns:
        DataFrame with volatile data
    """
    return generate_sample_eurusd_data(
        n_bars=n_bars,
        volatility=0.0008,
        trend=0.0
    )


def load_csv_data(filepath: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file
    
    Args:
        filepath: Path to CSV file
        parse_dates: Whether to parse date column
        
    Returns:
        DataFrame with OHLCV data
    """
    df = pd.read_csv(filepath)
    
    # Try to find timestamp/datetime column
    timestamp_cols = ['timestamp', 'datetime', 'date', 'time']
    timestamp_col = None
    
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col and parse_dates:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.set_index(timestamp_col, inplace=True)
    
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    
    # Handle different column name formats
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            column_mapping[col] = 'open'
        elif 'high' in col_lower:
            column_mapping[col] = 'high'
        elif 'low' in col_lower:
            column_mapping[col] = 'low'
        elif 'close' in col_lower:
            column_mapping[col] = 'close'
        elif 'volume' in col_lower or 'vol' in col_lower:
            column_mapping[col] = 'volume'
    
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
    
    # Verify all required columns exist
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df[required]


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample data to different timeframe
    
    Args:
        df: DataFrame with OHLCV data (must have datetime index)
        timeframe: Target timeframe ('5T', '15T', '1H', '4H', '1D')
        
    Returns:
        Resampled DataFrame
    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    resampled = df.resample(timeframe).agg(ohlc_dict).dropna()
    
    return resampled


def add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trading session labels to data
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with session column added
    """
    from .tier1_ict_detection.session_filter import SessionFilter
    
    df = df.copy()
    session_filter = SessionFilter()
    
    df['session'] = df.index.map(
        lambda x: session_filter.get_session(x).value
    )
    
    return df


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data integrity
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError if not
    """
    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check for NaN values
    if df[required].isnull().any().any():
        raise ValueError("Data contains NaN values")
    
    # Check OHLC relationships
    invalid_high = (df['high'] < df['low']).any()
    if invalid_high:
        raise ValueError("Found bars where high < low")
    
    invalid_high_open = (df['high'] < df['open']).any()
    if invalid_high_open:
        raise ValueError("Found bars where high < open")
    
    invalid_high_close = (df['high'] < df['close']).any()
    if invalid_high_close:
        raise ValueError("Found bars where high < close")
    
    invalid_low_open = (df['low'] > df['open']).any()
    if invalid_low_open:
        raise ValueError("Found bars where low > open")
    
    invalid_low_close = (df['low'] > df['close']).any()
    if invalid_low_close:
        raise ValueError("Found bars where low > close")
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError("Found negative or zero prices")
    
    # Check for negative volume
    if (df['volume'] < 0).any():
        raise ValueError("Found negative volume")
    
    return True


def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Split data into train and test sets (time-series split)
    
    Args:
        df: DataFrame to split
        test_size: Proportion for test set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    return train, test

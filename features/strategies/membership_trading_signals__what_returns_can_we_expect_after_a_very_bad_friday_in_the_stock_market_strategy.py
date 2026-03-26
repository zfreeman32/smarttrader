import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

def membership_trading_signals__what_returns_can_we_expect_after_a_very_bad_friday_in_the_stock_market_strategy(stock_df, low_window=20, high_window=50):
    """
    Generates trading signals based on the membership plans trading strategy.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with 'Close' column.
    low_window (int): Short moving average window size.
    high_window (int): Long moving average window size.
    
    Returns:
    DataFrame: Signals indicating 'long', 'short', or 'neutral' positions.
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_ma'] = stock_df['Close'].rolling(window=low_window).mean()
    signals['long_ma'] = stock_df['Close'].rolling(window=high_window).mean()
    signals['signal'] = 'neutral'
    signals.loc[(signals['short_ma'] > signals['long_ma']) & (signals['short_ma'].shift(1) <= signals['long_ma'].shift(1)), 'signal'] = 'long'
    signals.loc[(signals['short_ma'] < signals['long_ma']) & (signals['short_ma'].shift(1) >= signals['long_ma'].shift(1)), 'signal'] = 'short'
    signals.drop(['short_ma', 'long_ma'], axis=1, inplace=True)
    return signals

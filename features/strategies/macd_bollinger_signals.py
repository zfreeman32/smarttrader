
import pandas as pd
import numpy as np
from ta import trend

# MACD and Bollinger Bands Strategy
def macd_bollinger_signals(stock_df, bb_window=20, bb_std=2, macd_slow=26, macd_fast=12, macd_signal=9):
    signals = pd.DataFrame(index=stock_df.index)
    close = stock_df['Close']

    # Calculate Bollinger Bands
    moving_average = close.rolling(window=bb_window).mean()
    std_dev = close.rolling(window=bb_window).std()
    upper_band = moving_average + (std_dev * bb_std)
    lower_band = moving_average - (std_dev * bb_std)

    # Calculate MACD
    macd_indicator = trend.MACD(
        close,
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()

    # Generate signals
    signals['macd_signal'] = 'neutral'
    signals.loc[(close > upper_band) & (macd_line > signal_line), 'macd_signal'] = 'long'
    signals.loc[(close < lower_band) & (macd_line < signal_line), 'macd_signal'] = 'short'
    
    return signals

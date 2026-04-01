
import pandas as pd
import numpy as np

# Relative Vigor Index (RVI) Trading Strategy
def rvi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)

    numerator = (
        (stock_df['Close'] - stock_df['Open'])
        + 2 * (stock_df['Close'].shift(1) - stock_df['Open'].shift(1))
        + 2 * (stock_df['Close'].shift(2) - stock_df['Open'].shift(2))
        + (stock_df['Close'].shift(3) - stock_df['Open'].shift(3))
    ) / 6.0
    denominator = (
        (stock_df['High'] - stock_df['Low'])
        + 2 * (stock_df['High'].shift(1) - stock_df['Low'].shift(1))
        + 2 * (stock_df['High'].shift(2) - stock_df['Low'].shift(2))
        + (stock_df['High'].shift(3) - stock_df['Low'].shift(3))
    ) / 6.0
    raw_rvi = numerator.rolling(window=window).mean() / denominator.replace(0, np.nan).rolling(window=window).mean()
    signal_line = raw_rvi.rolling(window=4).mean()

    signals['RVI'] = raw_rvi
    signals['RVI_Signal'] = 'neutral'
    
    # Generating buy and sell signals based on RVI line crossing
    signals.loc[(signals['RVI'] > signal_line) & (signals['RVI'].shift(1) <= signal_line.shift(1)), 'RVI_Signal'] = 'long'
    signals.loc[(signals['RVI'] < signal_line) & (signals['RVI'].shift(1) >= signal_line.shift(1)), 'RVI_Signal'] = 'short'
    
    return signals[['RVI_Signal']]

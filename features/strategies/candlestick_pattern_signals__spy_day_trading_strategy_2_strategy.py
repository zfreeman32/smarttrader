import pandas as pd
import numpy as np
from ta import momentum

def candlestick_pattern_signals__spy_day_trading_strategy_2_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    for i in range(1, len(stock_df)):
        if stock_df['Close'].iloc[i] > stock_df['Open'].iloc[i] and stock_df['Close'].iloc[i - 1] < stock_df['Open'].iloc[i - 1] and (stock_df['Open'].iloc[i] < stock_df['Close'].iloc[i - 1]) and (stock_df['Close'].iloc[i] > stock_df['Open'].iloc[i - 1]):
            signals['signal'].iloc[i] = 'long'
        elif stock_df['Close'].iloc[i] < stock_df['Open'].iloc[i] and stock_df['Close'].iloc[i - 1] > stock_df['Open'].iloc[i - 1] and (stock_df['Open'].iloc[i] > stock_df['Close'].iloc[i - 1]) and (stock_df['Close'].iloc[i] < stock_df['Open'].iloc[i - 1]):
            signals['signal'].iloc[i] = 'short'
    return signals

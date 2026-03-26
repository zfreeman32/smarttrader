import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

def moving_average_crossover_signals__monday_tuesday_trade_in_nasdaq_update_strategy(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = -1
    signals['crossover_signal'] = 'neutral'
    signals.loc[(signals['signal'] == 1) & (signals['signal'].shift(1) != 1), 'crossover_signal'] = 'long'
    signals.loc[(signals['signal'] == -1) & (signals['signal'].shift(1) != -1), 'crossover_signal'] = 'short'
    signals.drop(['short_mavg', 'long_mavg', 'signal'], axis=1, inplace=True)
    return signals

import pandas as pd
import numpy as np
from ta import trend

def moving_average_crossover_signals__sp_500_and_gold_rotation_strategy_spy_and_gld_rotation_strategy(stock_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'positions'], axis=1, inplace=True)
    return signals

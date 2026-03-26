import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

def moving_average_crossover_signals__which_oscillating_indicator_is_the_best_for_trading_strategies_strategy(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'
    signals.drop(['Short_MA', 'Long_MA', 'signal', 'positions'], axis=1, inplace=True)
    return signals

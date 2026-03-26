import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

def moving_average_crossover_signals__moving_average_crossover_strategy(stock_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['position'] = signals['signal'].diff()
    signals['market_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'market_signal'] = 'long'
    signals.loc[signals['position'] == -1, 'market_signal'] = 'short'
    signals = signals[['market_signal']]
    return signals

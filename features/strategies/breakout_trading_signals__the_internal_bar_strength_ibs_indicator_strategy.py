import pandas as pd
import numpy as np
from ta import momentum, trend

def breakout_trading_signals__the_internal_bar_strength_ibs_indicator_strategy(stock_df, breakout_window=20, volatility_window=10):
    signals = pd.DataFrame(index=stock_df.index)
    signals['high'] = stock_df['High'].rolling(window=breakout_window).max()
    signals['low'] = stock_df['Low'].rolling(window=breakout_window).min()
    signals['atr'] = stock_df['Close'].rolling(window=volatility_window).apply(lambda x: np.max(x) - np.min(x))
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['high'].shift(1), 'signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['low'].shift(1), 'signal'] = 'short'
    signals.drop(['high', 'low', 'atr'], axis=1, inplace=True)
    return signals

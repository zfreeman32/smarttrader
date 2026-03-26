import pandas as pd
import numpy as np
from ta import volatility

def bollinger_band_signals__python_bollinger_band_trading_strategy_strategy(stock_df, window=20, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    sma = stock_df['Close'].rolling(window=window).mean()
    rolling_std = stock_df['Close'].rolling(window=window).std()
    signals['Upper Band'] = sma + rolling_std * num_std_dev
    signals['Lower Band'] = sma - rolling_std * num_std_dev
    signals['Signal'] = 'neutral'
    signals.loc[stock_df['Close'] < signals['Lower Band'], 'Signal'] = 'long'
    signals.loc[stock_df['Close'] > signals['Upper Band'], 'Signal'] = 'short'
    return signals[['Signal']]

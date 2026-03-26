import pandas as pd
import numpy as np

def monthly_trading_strategy_signals__four_up_days_in_row_sp_500_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    short_window = 10
    long_window = 50
    signals['SMA_Short'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['SMA_Long'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0
    signals.loc[signals['SMA_Short'] > signals['SMA_Long'], 'signal'] = 1
    signals.loc[signals['SMA_Short'] < signals['SMA_Long'], 'signal'] = -1
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'trading_signal'] = 'short'
    signals.drop(['SMA_Short', 'SMA_Long', 'signal'], axis=1, inplace=True)
    return signals

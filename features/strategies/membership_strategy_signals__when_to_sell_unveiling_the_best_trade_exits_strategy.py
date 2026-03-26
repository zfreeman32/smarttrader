import pandas as pd
import numpy as np

def membership_strategy_signals__when_to_sell_unveiling_the_best_trade_exits_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    short_window = 10
    long_window = 30
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'membership_signal'] = 'long'
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'membership_signal'] = 'short'
    signals.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    return signals

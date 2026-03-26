import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

def quantified_strategies_signals__the_overnight_edge_is_it_still_working_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['average_price'] = stock_df['Close'].rolling(window=20).mean()
    signals['std_dev'] = stock_df['Close'].rolling(window=20).std()
    signals['long_entry'] = signals['average_price'] + signals['std_dev']
    signals['short_entry'] = signals['average_price'] - signals['std_dev']
    signals['strategy_signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['long_entry'], 'strategy_signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['short_entry'], 'strategy_signal'] = 'short'
    signals.drop(['average_price', 'std_dev', 'long_entry', 'short_entry'], axis=1, inplace=True)
    return signals

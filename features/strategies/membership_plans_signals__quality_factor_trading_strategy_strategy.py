import pandas as pd
import numpy as np

def membership_plans_signals__quality_factor_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_strategy'] = 'neutral'
    signals['short_mavg'] = stock_df['Close'].rolling(window=10).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=30).mean()
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'membership_strategy'] = 'long'
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'membership_strategy'] = 'short'
    signals.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    return signals

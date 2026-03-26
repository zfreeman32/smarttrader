import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

def membership_strategy_signals__monday_tuesday_in_nasdaq_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].pct_change()
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['price_change'] > 0.02, 'membership_signal'] = 'long'
    signals.loc[signals['price_change'] < -0.02, 'membership_signal'] = 'short'
    signals.drop(['price_change'], axis=1, inplace=True)
    return signals

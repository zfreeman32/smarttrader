import pandas as pd
import numpy as np

def membership_strategy_signals__crude_oil_and_us_dollar_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    signals['price_change'] = stock_df['Close'].pct_change(periods=5)
    signals.loc[signals['price_change'] > 0.02, 'membership_signal'] = 'long'
    signals.loc[signals['price_change'] < -0.02, 'membership_signal'] = 'short'
    signals.drop(['price_change'], axis=1, inplace=True)
    return signals

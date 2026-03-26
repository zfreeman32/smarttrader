import pandas as pd
import numpy as np

def membership_strategy_signals__country_etfs_performance_per_quarter_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    signals['membership_signal'] = 'neutral'
    entry_threshold = stock_df['Close'].shift(1) * 1.01
    exit_threshold = stock_df['Close'].shift(1) * 0.99
    signals.loc[signals['Close'] > entry_threshold, 'membership_signal'] = 'long'
    signals.loc[signals['Close'] < exit_threshold, 'membership_signal'] = 'short'
    return signals[['membership_signal']]

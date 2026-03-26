import pandas as pd
import numpy as np

def quantified_strategies_signals__etf_sector_rotation_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    signals.loc[stock_df['Close'].rolling(window=10).mean() > stock_df['Close'].rolling(window=30).mean(), 'membership_signal'] = 'long'
    signals.loc[stock_df['Close'].rolling(window=10).mean() < stock_df['Close'].rolling(window=30).mean(), 'membership_signal'] = 'short'
    return signals

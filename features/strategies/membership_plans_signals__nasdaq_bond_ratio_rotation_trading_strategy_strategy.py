import pandas as pd
import numpy as np

def membership_plans_signals__nasdaq_bond_ratio_rotation_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    stock_df['Monthly Return'] = stock_df['Close'].pct_change()
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Monthly Return'] > 0) & (stock_df['Monthly Return'].shift(1) <= 0), 'signal'] = 'long'
    signals.loc[(stock_df['Monthly Return'] < 0) & (stock_df['Monthly Return'].shift(1) >= 0), 'signal'] = 'short'
    return signals[['signal']]

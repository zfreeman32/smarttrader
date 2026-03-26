import pandas as pd
import numpy as np

def membership_plans_signals__big_moves_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    current_month = stock_df.index[0].month
    if current_month in [7, 12]:
        signals['membership_signal'] = 'neutral'
    else:
        stock_returns = stock_df['Close'].pct_change()
        mean_return = stock_returns.mean()
        signals['membership_signal'] = np.where(stock_returns > mean_return, 'long', 'short')
    return signals

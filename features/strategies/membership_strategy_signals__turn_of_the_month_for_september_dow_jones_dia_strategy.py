import pandas as pd
import numpy as np

def membership_strategy_signals__turn_of_the_month_for_september_dow_jones_dia_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    stock_df['Monthly_Return'] = stock_df['Close'].pct_change(periods=30)
    stock_df['Signal'] = np.where(stock_df['Monthly_Return'] > 0.05, 'long', np.where(stock_df['Monthly_Return'] < -0.05, 'short', 'neutral'))
    signals['membership_signal'] = stock_df['Signal']
    return signals

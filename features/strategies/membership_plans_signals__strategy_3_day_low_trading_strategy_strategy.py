import pandas as pd
import numpy as np

def membership_plans_signals__strategy_3_day_low_trading_strategy_strategy(stock_df, threshold=0.05):
    signals = pd.DataFrame(index=stock_df.index)
    stock_df['Monthly_Return'] = stock_df['Close'].pct_change(periods=21)
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Monthly_Return'] > threshold, 'signal'] = 'long'
    signals.loc[stock_df['Monthly_Return'] < -threshold, 'signal'] = 'short'
    return signals

import pandas as pd
import numpy as np

def membership_plans_signals__a_profitable_mean_reversion_strategy_using_week_days_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['monthly_returns'] = stock_df['Close'].pct_change(periods=30)
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['monthly_returns'] > 0.03, 'membership_signal'] = 'long'
    signals.loc[signals['monthly_returns'] < -0.03, 'membership_signal'] = 'short'
    return signals[['membership_signal']]

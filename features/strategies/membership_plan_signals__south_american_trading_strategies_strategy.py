import pandas as pd
import numpy as np

def membership_plan_signals__south_american_trading_strategies_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['returns'] = stock_df['Close'].pct_change()
    signals['signal'] = 'neutral'
    signals.loc[signals['returns'] > 0.02, 'signal'] = 'long'
    signals.loc[signals['returns'] < -0.02, 'signal'] = 'short'
    return signals[['signal']]

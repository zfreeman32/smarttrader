import pandas as pd
import numpy as np

def membership_plan_signals__monthly_momentum_in_spy_and_eem_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['last_close'] = stock_df['Close'].shift(1)
    signals['current_close'] = stock_df['Close']
    signals['signal'] = 'neutral'
    signals.loc[signals['current_close'] > signals['last_close'], 'signal'] = 'long'
    signals.loc[signals['current_close'] < signals['last_close'], 'signal'] = 'short'
    return signals[['signal']]

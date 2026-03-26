import pandas as pd
import numpy as np

def membership_plans_signals__performance_of_the_opex_week_of_september_strategy(stock_df, membership_type='Gold'):
    signals = pd.DataFrame(index=stock_df.index)
    if membership_type == 'Platinum':
        signals['signal'] = np.where(stock_df['Close'] > stock_df['Open'], 'long', 'short')
    elif membership_type == 'Gold':
        signals['signal'] = np.where(stock_df['Close'] < stock_df['Open'], 'short', 'long')
    else:
        signals['signal'] = 'neutral'
    return signals

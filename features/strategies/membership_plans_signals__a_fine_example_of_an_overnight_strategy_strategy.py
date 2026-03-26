import pandas as pd
import numpy as np

def membership_plans_signals__a_fine_example_of_an_overnight_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > stock_df['Close'].rolling(window=20).mean(), 'signal'] = 'long'
    signals.loc[stock_df['Close'] < stock_df['Close'].rolling(window=20).mean(), 'signal'] = 'short'
    return signals

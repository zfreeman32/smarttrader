import pandas as pd
import numpy as np
import datetime as dt

def membership_plans_signals__trade_during_a_recession_strategy(stock_df):
    """
    Generates trading signals based on the membership plans trading strategy.
    This strategy evaluates quantifiable market patterns to generate long, short, or neutral signals.
    Monthly strategies are published, except in July and December.
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    if not pd.api.types.is_datetime64_any_dtype(stock_df.index):
        stock_df.index = pd.to_datetime(stock_df.index)
    moving_average = stock_df['Close'].rolling(window=20).mean()
    signals['moving_average'] = moving_average
    signals.loc[(stock_df['Close'] > moving_average) & (stock_df['Close'].shift(1) <= moving_average.shift(1)), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < moving_average) & (stock_df['Close'].shift(1) >= moving_average.shift(1)), 'signal'] = 'short'
    signals.drop(['moving_average'], axis=1, inplace=True)
    return signals

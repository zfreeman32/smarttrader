import pandas as pd
import numpy as np
from ta import trend

def stc_signals__schaff_trend_cycle_indicator_strategy(stock_df, cycle_length=10, long_cycle=23, short_cycle=50):
    signals = pd.DataFrame(index=stock_df.index)
    window_slow = max(long_cycle, short_cycle)
    window_fast = min(long_cycle, short_cycle)
    stc = trend.STCIndicator(
        stock_df['Close'],
        window_slow=window_slow,
        window_fast=window_fast,
        cycle=cycle_length,
    )
    signals['STC'] = stc.stc()
    signals['stc_signal'] = 'neutral'
    signals.loc[(signals['STC'] < 25) & (signals['STC'].shift(1) >= 25), 'stc_signal'] = 'long'
    signals.loc[(signals['STC'] > 75) & (signals['STC'].shift(1) <= 75), 'stc_signal'] = 'short'
    signals.drop(['STC'], axis=1, inplace=True)
    return signals

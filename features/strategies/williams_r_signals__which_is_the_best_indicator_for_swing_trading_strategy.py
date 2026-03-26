import pandas as pd
import numpy as np
from ta.momentum import WilliamsRIndicator

def williams_r_signals__which_is_the_best_indicator_for_swing_trading_strategy(stock_df, window=14, overbought_level=-20, oversold_level=-80):
    signals = pd.DataFrame(index=stock_df.index)
    williams_r = WilliamsRIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window)
    signals['Williams %R'] = williams_r.williams_r()
    signals['williams_r_signal'] = 'neutral'
    signals.loc[signals['Williams %R'] < oversold_level, 'williams_r_signal'] = 'long'
    signals.loc[signals['Williams %R'] > overbought_level, 'williams_r_signal'] = 'short'
    return signals[['williams_r_signal']]

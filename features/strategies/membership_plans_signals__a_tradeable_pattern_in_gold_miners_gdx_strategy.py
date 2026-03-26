import pandas as pd
import numpy as np

def membership_plans_signals__a_tradeable_pattern_in_gold_miners_gdx_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = np.where(np.random.rand(len(stock_df)) > 0.5, 'long', 'short')
    signals['signal'] = signals['signal'].where(np.random.rand(len(stock_df)) > 0.1, 'neutral')
    return signals

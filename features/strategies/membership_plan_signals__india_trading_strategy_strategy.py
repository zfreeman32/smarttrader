import pandas as pd
import numpy as np

def membership_plan_signals__india_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    platinum_threshold = 1990
    gold_threshold = 990
    membership_fee = np.random.choice([platinum_threshold, gold_threshold], len(stock_df))
    signals.loc[membership_fee == platinum_threshold, 'membership_signal'] = 'long'
    signals.loc[(membership_fee == gold_threshold) & (stock_df.index.month != 7) & (stock_df.index.month != 12), 'membership_signal'] = 'short'
    return signals

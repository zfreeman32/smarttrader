import pandas as pd
import numpy as np

def membership_plans_signals__different_trade_entries_comparisons_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    platinum_price = 1990
    gold_price = 990
    strategies_count = 20
    bonus_strategies_count = 10
    if platinum_price <= 1990:
        strategies_count += 15
    entry_condition = stock_df['Close'] > stock_df['Open']
    exit_condition = stock_df['Close'] < stock_df['Open']
    signals.loc[entry_condition, 'signal'] = 'long'
    signals.loc[exit_condition, 'signal'] = 'short'
    return signals

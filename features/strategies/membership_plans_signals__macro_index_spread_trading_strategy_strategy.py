import pandas as pd
import numpy as np

def membership_plans_signals__macro_index_spread_trading_strategy_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    price_levels = {'platinum': 1990, 'gold': 990, 'basic': 199}
    membership_type = 'platinum'
    strategies = {'platinum': 15, 'gold': 10, 'basic': 1}
    if membership_type in strategies:
        num_strategies = strategies[membership_type]
        signals['signal'] = 'neutral'
        for i in range(num_strategies):
            if np.random.rand() > 0.5:
                signals['signal'].iloc[i] = 'long' if i % 2 == 0 else 'short'
    return signals

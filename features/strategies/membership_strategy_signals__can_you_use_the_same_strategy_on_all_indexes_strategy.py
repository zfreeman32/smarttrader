import pandas as pd
import numpy as np
from ta import momentum

def membership_strategy_signals__can_you_use_the_same_strategy_on_all_indexes_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['ROC'] = stock_df['Close'].pct_change(periods=5)
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['ROC'] > 0.02, 'membership_signal'] = 'long'
    signals.loc[signals['ROC'] < -0.02, 'membership_signal'] = 'short'
    return signals[['membership_signal']]

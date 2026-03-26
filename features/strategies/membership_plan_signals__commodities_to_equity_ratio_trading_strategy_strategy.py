import pandas as pd
import numpy as np

def membership_plan_signals__commodities_to_equity_ratio_trading_strategy_strategy(stock_df, strategy_list, strategy_bonus, months=12):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    active_strategies = strategy_list.count(True)
    if active_strategies >= 15:
        signals['membership_signal'] = 'long'
    elif active_strategies >= 10:
        signals['membership_signal'] = 'short'
    if months > 0:
        signals['bonus_signal'] = 'strategies available'
    signals['final_signal'] = np.where(signals['membership_signal'] == 'neutral', 'neutral', signals['membership_signal'])
    return signals[['membership_signal', 'bonus_signal', 'final_signal']]

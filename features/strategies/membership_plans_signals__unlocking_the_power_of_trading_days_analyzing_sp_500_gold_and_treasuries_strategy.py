import pandas as pd

def membership_plans_signals__unlocking_the_power_of_trading_days_analyzing_sp_500_gold_and_treasuries_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    monthly_groups = stock_df.resample('ME')
    for month, group in monthly_groups:
        first_day = group.index[0]
        last_day = group.index[-1]
        signals.loc[first_day, 'signal'] = 'long'
        signals.loc[last_day, 'signal'] = 'short'
    signals['signal'] = signals['signal'].ffill()
    signals.loc[signals.index.month.isin([7, 12]), 'signal'] = 'neutral'
    return signals

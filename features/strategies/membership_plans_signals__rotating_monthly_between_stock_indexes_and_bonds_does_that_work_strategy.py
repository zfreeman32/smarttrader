import pandas as pd

def membership_plans_signals__rotating_monthly_between_stock_indexes_and_bonds_does_that_work_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    signals.loc[stock_df['Close'].pct_change() > 0.01, 'membership_signal'] = 'long'
    signals.loc[stock_df['Close'].pct_change() < -0.01, 'membership_signal'] = 'short'
    return signals

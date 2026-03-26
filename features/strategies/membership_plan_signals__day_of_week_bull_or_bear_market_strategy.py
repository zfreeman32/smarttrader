import pandas as pd

def membership_plan_signals__day_of_week_bull_or_bear_market_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['strategies_available'] = (stock_df['Close'] > stock_df['Close'].rolling(window=20).mean()).astype(int)
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['strategies_available'] == 1, 'membership_signal'] = 'long'
    signals.loc[signals['strategies_available'] == 0, 'membership_signal'] = 'short'
    signals.drop(['strategies_available'], axis=1, inplace=True)
    return signals

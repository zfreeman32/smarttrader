import pandas as pd

def membership_based_trading_signals__trend_following_treasuries_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    moving_average = stock_df['Close'].rolling(window=10).mean()
    signals.loc[stock_df['Close'] > moving_average, 'membership_signal'] = 'long'
    signals.loc[stock_df['Close'] < moving_average, 'membership_signal'] = 'short'
    return signals

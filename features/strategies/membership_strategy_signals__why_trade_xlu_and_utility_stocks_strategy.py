import pandas as pd

def membership_strategy_signals__why_trade_xlu_and_utility_stocks_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    signals.loc[stock_df['Close'] < stock_df['Close'].rolling(window=20).mean(), 'membership_signal'] = 'long'
    signals.loc[stock_df['Close'] > stock_df['Close'].rolling(window=20).mean(), 'membership_signal'] = 'short'
    return signals

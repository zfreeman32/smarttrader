import pandas as pd

def membership_strategy_signals__spy_strategies_using_vix_indicator_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > stock_df['Close'].rolling(window=5).mean()) & (stock_df['Volume'] > stock_df['Volume'].rolling(window=5).mean()), 'membership_signal'] = 'long'
    signals.loc[(stock_df['Close'] < stock_df['Close'].rolling(window=5).mean()) & (stock_df['Volume'] < stock_df['Volume'].rolling(window=5).mean()), 'membership_signal'] = 'short'
    return signals

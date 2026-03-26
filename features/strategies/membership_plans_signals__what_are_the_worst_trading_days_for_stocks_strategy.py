import pandas as pd
import numpy as np

def membership_plans_signals__what_are_the_worst_trading_days_for_stocks_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_buy_signal'] = np.where((stock_df['Close'].pct_change().rolling(window=10).mean() > 0) & (stock_df['Close'] > stock_df['Close'].rolling(window=20).mean()), 'long', 'neutral')
    signals['membership_sell_signal'] = np.where((stock_df['Close'].pct_change().rolling(window=10).mean() < 0) & (stock_df['Close'] < stock_df['Close'].rolling(window=20).mean()), 'short', 'neutral')
    signals['final_signal'] = np.where(signals['membership_buy_signal'] == 'long', 'long', np.where(signals['membership_sell_signal'] == 'short', 'short', 'neutral'))
    signals.drop(['membership_buy_signal', 'membership_sell_signal'], axis=1, inplace=True)
    return signals

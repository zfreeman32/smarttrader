import pandas as pd
import numpy as np

def membership_plans_signals__should_you_trade_or_invest_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=5).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=20).mean()
    signals['signal'] = 0
    signals['signal'][5:] = np.where(signals['Short_MA'][5:] > signals['Long_MA'][5:], 1, 0)
    signals['signal'][5:] = np.where(signals['Short_MA'][5:] < signals['Long_MA'][5:], -1, signals['signal'][5:])
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'membership_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'membership_signal'] = 'short'
    signals = signals[['membership_signal']]
    return signals

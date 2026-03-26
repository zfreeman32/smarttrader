import pandas as pd
import numpy as np
from ta import momentum

def rsi_signals__new_5_day_low_in_spy_and_closes_in_the_low_of_the_range_strategy(stock_df, window=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = momentum.RSIIndicator(stock_df['Close'], window)
    signals['RSI'] = rsi.rsi()
    signals['rsi_signal'] = 'neutral'
    signals.loc[signals['RSI'] < oversold, 'rsi_signal'] = 'long'
    signals.loc[signals['RSI'] > overbought, 'rsi_signal'] = 'short'
    signals.loc[(signals['RSI'] >= oversold) & (signals['RSI'] <= overbought), 'rsi_signal'] = 'neutral'
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals

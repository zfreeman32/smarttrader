import pandas as pd
import numpy as np
from ta import momentum

def momentum_trading_signals__the_stochastic_indicator_does_it_work_strategy(stock_df, window=14, threshold=0.02):
    signals = pd.DataFrame(index=stock_df.index)
    momentum_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['RSI'] = momentum_indicator.rsi()
    signals['signal'] = 'neutral'
    signals.loc[signals['RSI'] > 70, 'signal'] = 'short'
    signals.loc[signals['RSI'] < 30, 'signal'] = 'long'
    signals['signal'] = signals['signal'].shift(1)
    return signals[['signal']]

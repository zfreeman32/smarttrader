import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

def rsi_signals__platinum_futures_strategy_strategy(stock_df, rsi_period=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    signals['RSI'] = momentum.RSIIndicator(stock_df['Close'], window=rsi_period).rsi()
    signals['rsi_signal'] = 'neutral'
    signals.loc[signals['RSI'] > overbought, 'rsi_signal'] = 'short'
    signals.loc[signals['RSI'] < oversold, 'rsi_signal'] = 'long'
    return signals[['rsi_signal']]

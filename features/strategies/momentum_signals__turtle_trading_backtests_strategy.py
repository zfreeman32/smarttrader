import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

def momentum_signals__turtle_trading_backtests_strategy(stock_df, period=14, threshold=0.02):
    signals = pd.DataFrame(index=stock_df.index)
    momentum_indicator = momentum.MomentumIndicator(stock_df['Close'], window=period)
    signals['momentum'] = momentum_indicator.momentum()
    signals['momentum_signal'] = 'neutral'
    signals.loc[signals['momentum'] > threshold, 'momentum_signal'] = 'long'
    signals.loc[signals['momentum'] < -threshold, 'momentum_signal'] = 'short'
    return signals[['momentum_signal']]

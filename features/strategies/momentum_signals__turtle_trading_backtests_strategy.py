import pandas as pd
import numpy as np

def momentum_signals__turtle_trading_backtests_strategy(stock_df, period=14, threshold=0.02):
    signals = pd.DataFrame(index=stock_df.index)
    signals['momentum'] = stock_df['Close'].pct_change(periods=period)
    signals['momentum_signal'] = 'neutral'
    signals.loc[signals['momentum'] > threshold, 'momentum_signal'] = 'long'
    signals.loc[signals['momentum'] < -threshold, 'momentum_signal'] = 'short'
    return signals[['momentum_signal']]

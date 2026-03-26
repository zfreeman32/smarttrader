import pandas as pd
import numpy as np

def roc_signals__rate_of_change_trading_strategy_strategy(stock_df, period=14, overbought=20, oversold=-20):
    signals = pd.DataFrame(index=stock_df.index)
    roc = stock_df['Close'].pct_change(periods=period) * 100
    signals['ROC'] = roc
    signals['roc_signal'] = 'neutral'
    signals.loc[(signals['ROC'] > overbought) & (signals['ROC'].shift(1) <= overbought), 'roc_signal'] = 'short'
    signals.loc[(signals['ROC'] < oversold) & (signals['ROC'].shift(1) >= oversold), 'roc_signal'] = 'long'
    return signals.drop(['ROC'], axis=1)

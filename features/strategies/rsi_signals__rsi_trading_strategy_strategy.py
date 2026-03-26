import pandas as pd
from ta.momentum import RSIIndicator

def rsi_signals__rsi_trading_strategy_strategy(stock_df, rsi_window=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = RSIIndicator(stock_df['Close'], window=rsi_window)
    signals['RSI'] = rsi.rsi()
    signals['rsi_signal'] = 'neutral'
    signals.loc[(signals['RSI'] < oversold) & (signals['RSI'].shift(1) >= oversold), 'rsi_signal'] = 'long'
    signals.loc[(signals['RSI'] > overbought) & (signals['RSI'].shift(1) <= overbought), 'rsi_signal'] = 'short'
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals

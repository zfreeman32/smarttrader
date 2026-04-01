import pandas as pd
import numpy as np
from ta import momentum

# Functions from volswitch.py
# VolSwitch Strategy
def vols_switch_signals(stock_df, length=14, rsi_length=14, avg_length=50, rsi_overbought_level=70, rsi_oversold_level=30):
    close = stock_df['Close']
    volatility = close.rolling(window=length).std()
    sma = close.rolling(window=avg_length).mean()
    rsi = momentum.RSIIndicator(close, window=rsi_length).rsi()

    # Buy signal based on VolSwitch strategy
    buy_signal = (
        (volatility.shift(1) > volatility)
        & (close > sma)
        & (rsi > rsi_oversold_level)
    )

    # Sell signal based on VolSwitch strategy
    sell_signal = (
        (volatility.shift(1) < volatility)
        & (close < sma)
        & (rsi < rsi_overbought_level)
    )
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['vols_switch_buy_signal'] = 0
    signals['vols_switch_sell_signal'] = 0
    
    # Create signals
    signals.loc[buy_signal, 'vols_switch_buy_signal'] = 1
    signals.loc[sell_signal, 'vols_switch_sell_signal'] = 1
 
    return signals

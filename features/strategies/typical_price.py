import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def sve_ha_typ_cross_signals(df, typical_length=14, ha_length=14):
    def typical_price(frame):
        return (frame['High'] + frame['Low'] + frame['Close']) / 3

    def heikin_ashi(frame):
        ha_close = (frame['Open'] + frame['High'] + frame['Low'] + frame['Close']) / 4
        ha_open = (frame['Open'].shift(1) + frame['Close'].shift(1)) / 2
        ha_high = frame[['High', 'Open', 'Close']].max(axis=1)
        ha_low = frame[['Low', 'Open', 'Close']].min(axis=1)
        return ha_open, ha_high, ha_low, ha_close

    signals = pd.DataFrame(index=df.index)
    
    tp = typical_price(df)
    tp_ema = talib.EMA(tp, timeperiod=typical_length)
    
    ha_open, ha_high, ha_low, ha_close = heikin_ashi(df)
    ha_avg = (ha_open + ha_high + ha_low + ha_close) / 4
    ha_ema = talib.EMA(ha_avg, timeperiod=ha_length)

    # Initialize signal column with 'neutral'
    signals['sve_ha_typ_cross_buy_signal'] = 0
    signals['sve_ha_typ_cross_sell_signal'] = 0

    # Assign 'buy' and 'sell' signals
    signals.loc[(tp_ema > ha_ema) & (tp_ema.shift() < ha_ema.shift()), 'sve_ha_typ_cross_buy_signal'] = 1
    signals.loc[(tp_ema < ha_ema) & (tp_ema.shift() > ha_ema.shift()), 'sve_ha_typ_cross_sell_signal'] = 1

    return signals[['sve_ha_typ_cross_buy_signal', 'sve_ha_typ_cross_sell_signal']]

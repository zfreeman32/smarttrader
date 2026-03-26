import pandas as pd
import numpy as np
from ta.trend import MACD

def macd_signals__python_and_macd_trading_strategy_strategy(stock_df):
    macd = MACD(stock_df['Close'])
    stock_df['MACD'] = macd.macd()
    stock_df['Signal'] = macd.macd_signal()
    stock_df['Histogram'] = macd.macd_diff()
    signals = pd.DataFrame(index=stock_df.index)
    signals['macd_signal'] = 'neutral'
    signals.loc[stock_df['Histogram'] > 0, 'macd_signal'] = 'long'
    signals.loc[stock_df['Histogram'] < 0, 'macd_signal'] = 'short'
    return signals

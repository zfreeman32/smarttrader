import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def cons_bars_down_se(data, consecutive_bars_down=4, price='Close'):
    """Generates a signal when a certain number of consecutive down bars are found.

    Args:
    data (pd.DataFrame): DataFrame containing OHLCV data.
    consecutive_bars_down (int, optional): Number of consecutive down bars to trigger the signal. Defaults to 4.
    price (str, optional): The price column to use in the analysis. Defaults to "Close".

    Returns:
    pd.DataFrame: DataFrame with signals.
    """
    signals = pd.DataFrame(index=data.index)
    bars_down = data[price] < data[price].shift(1)
    consecutive_sum = bars_down.rolling(window=consecutive_bars_down).sum()
    signals['ConsBarsDownSE_sell_signal'] = np.where(consecutive_sum >= consecutive_bars_down, 1, 0)
    return signals
